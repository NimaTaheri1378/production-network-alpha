from __future__ import annotations

import argparse, datetime as dt, html, json, math, os, platform, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

WORKSPACE = Path("<workspace>")
PRED_COLS = {"raw": "pred_raw", "ridge": "pred_ridge", "lightgbm": "pred_lightgbm"}

def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_ints(x: str) -> list[int]:
    out = []
    for p in str(x).split(","):
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return sorted(set(out))

def parse_floats(x: str) -> list[float]:
    return [float(p.strip()) for p in str(x).split(",") if p.strip()]

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def run_cmd(cmd: list[str]) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}

def ensure_workspace(project_root: Path) -> None:
    expected = (WORKSPACE / "production-network-alpha").resolve()
    if not WORKSPACE.exists():
        raise RuntimeError(f"Missing workspace: {WORKSPACE}")
    if project_root.resolve() != expected:
        raise RuntimeError(f"Wrong repo: {project_root}; expected {expected}")

def pred_path(root: Path, h: int) -> Path:
    return root / "data" / "processed" / "model_runs" / "phase5_1_modeling_full" / f"phase5_1_predictions_h{h}d.parquet"

def returns_path(root: Path) -> Path:
    return root / "data" / "processed" / "model_matrix_full" / "full_crsp_return_panel_features.parquet"

def read_predictions(root: Path, h: int, model: str, years: list[int], split: str) -> pd.DataFrame:
    path = pred_path(root, h)
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    pred_col = PRED_COLS[model]
    cols = ["signal_date", "target_permno", "target_label", "split", "signal_year", pred_col]
    df = pd.read_parquet(path, columns=cols)
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df["target_permno"] = pd.to_numeric(df["target_permno"], errors="coerce").astype("Int64")
    df["signal_year"] = pd.to_numeric(df["signal_year"], errors="coerce").astype("Int64")
    df["target_label"] = safe_num(df["target_label"])
    df[pred_col] = safe_num(df[pred_col])
    df = df[df["split"].eq(split) & df["signal_year"].isin(set(years))].copy()
    df = df.dropna(subset=["signal_date", "target_permno", pred_col])
    if df.empty:
        raise RuntimeError(f"Empty predictions after filter: h={h}, model={model}, years={years}")
    return df

def read_returns(root: Path, start_year: int, end_year: int) -> pd.DataFrame:
    path = returns_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Missing returns: {path}")
    cols = ["permno", "date", "ret_adj", "abret_mkt", "vwretd", "dollar_vol_21d"]
    r = pd.read_parquet(path, columns=cols)
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r["permno"] = pd.to_numeric(r["permno"], errors="coerce").astype("Int64")
    for c in ["ret_adj", "abret_mkt", "vwretd", "dollar_vol_21d"]:
        r[c] = safe_num(r[c])
    lo = pd.Timestamp(year=start_year, month=1, day=1) - pd.Timedelta(days=10)
    hi = pd.Timestamp(year=end_year + 1, month=3, day=31)
    r = r[(r["date"] >= lo) & (r["date"] <= hi)].dropna(subset=["permno", "date", "ret_adj", "abret_mkt"]).copy()
    if r.empty:
        raise RuntimeError("Return panel empty after filter")
    return r

def map_base_dates(signal_dates: pd.Series, calendar: np.ndarray) -> pd.Series:
    sig = pd.to_datetime(signal_dates, errors="coerce").to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(calendar, sig, side="left")
    mapped = np.full(len(sig), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = idx < len(calendar)
    mapped[ok] = calendar[idx[ok]]
    return pd.Series(pd.to_datetime(mapped), index=signal_dates.index)

def select_positions(pred: pd.DataFrame, pred_col: str, top_frac: float, min_names: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, diag = [], []
    for d, g in pred.groupby("signal_date", sort=True):
        g = g.dropna(subset=[pred_col, "target_permno"]).copy()
        n = len(g)
        if n < min_names or g[pred_col].nunique() < 10:
            diag.append({"signal_date": d, "n_names": n, "selected_long": 0, "selected_short": 0, "skipped": True})
            continue
        k = max(1, int(math.floor(n * top_frac)))
        g = g.sort_values(pred_col, ascending=True, kind="mergesort")
        short = g.head(k).copy()
        long = g.tail(k).copy()
        short["side"] = "short"; long["side"] = "long"
        short["base_weight"] = -0.5 / len(short)
        long["base_weight"] = 0.5 / len(long)
        rows.append(pd.concat([long, short], ignore_index=True)[["signal_date", "base_date", "target_permno", pred_col, "target_label", "side", "base_weight"]])
        diag.append({"signal_date": d, "n_names": n, "selected_long": len(long), "selected_short": len(short), "skipped": False})
    if not rows:
        raise RuntimeError("No eligible signal dates for portfolio construction")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diag)

def expand_holdings(selected: pd.DataFrame, calendar: np.ndarray, h: int) -> pd.DataFrame:
    cal_index = {np.datetime64(d): i for i, d in enumerate(calendar)}
    rows = []
    for base_date, g in selected.groupby("base_date", sort=True):
        if pd.isna(base_date):
            continue
        idx0 = cal_index.get(np.datetime64(pd.Timestamp(base_date)))
        if idx0 is None:
            continue
        for j in range(idx0 + 1, min(idx0 + h + 1, len(calendar))):
            p = g[["signal_date", "target_permno", "side", "base_weight"]].copy()
            p["date"] = pd.Timestamp(calendar[j])
            rows.append(p)
    if not rows:
        raise RuntimeError("No holdings generated")
    out = pd.concat(rows, ignore_index=True).rename(columns={"target_permno": "permno", "base_weight": "raw_weight"})
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    return out.dropna(subset=["permno", "date", "raw_weight"])

def normalize_weights(pos: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    p = pos.groupby(["date", "permno"], as_index=False).agg(raw_weight=("raw_weight", "sum"))
    m = p.merge(returns[["date", "permno", "ret_adj", "abret_mkt", "dollar_vol_21d"]], on=["date", "permno"], how="inner")
    if m.empty:
        raise RuntimeError("No holdings matched return panel")
    gross = m.groupby("date")["raw_weight"].transform(lambda x: float(np.abs(x).sum()))
    m = m[gross > 0].copy()
    gross = m.groupby("date")["raw_weight"].transform(lambda x: float(np.abs(x).sum()))
    m["weight"] = m["raw_weight"] / gross
    dg = m.groupby("date")["weight"].apply(lambda x: np.abs(x).sum()).rename("gross").reset_index()
    return m.merge(dg, on="date", how="left")

def turnover_by_date(w: pd.DataFrame) -> pd.DataFrame:
    out, prev = [], {}
    for date, g in w.sort_values("date").groupby("date", sort=True):
        curr = {int(r.permno): float(r.weight) for r in g[["permno", "weight"]].itertuples(index=False)}
        keys = set(prev) | set(curr)
        to = 0.5 * sum(abs(curr.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        out.append({"date": pd.Timestamp(date), "one_way_turnover": float(to), "n_positions": int(len(curr)), "gross_after_normalization": float(np.abs(g["weight"]).sum())})
        prev = curr
    return pd.DataFrame(out)

def portfolio_returns(w: pd.DataFrame, costs: list[float]) -> pd.DataFrame:
    d = (
        w.groupby("date", as_index=False)
        .apply(lambda g: pd.Series({
            "portfolio_raw_return": float((g["weight"] * g["ret_adj"]).sum()),
            "portfolio_abnormal_return": float((g["weight"] * g["abret_mkt"]).sum()),
            "long_gross": float(g.loc[g["weight"] > 0, "weight"].sum()),
            "short_gross_abs": float(-g.loc[g["weight"] < 0, "weight"].sum()),
            "n_positions": int(len(g)),
            "gross": float(np.abs(g["weight"]).sum()),
            "median_dollar_vol_21d": float(pd.to_numeric(g["dollar_vol_21d"], errors="coerce").median()),
        }))
        .reset_index(drop=True)
    )
    d = d.merge(turnover_by_date(w)[["date", "one_way_turnover"]], on="date", how="left")
    d["one_way_turnover"] = d["one_way_turnover"].fillna(0.0)
    for bps in costs:
        s = str(bps).replace(".", "p")
        d[f"net_abnormal_return_cost_{s}bps"] = d["portfolio_abnormal_return"] - d["one_way_turnover"] * (bps / 10000.0)
        d[f"net_raw_return_cost_{s}bps"] = d["portfolio_raw_return"] - d["one_way_turnover"] * (bps / 10000.0)
    return d.sort_values("date")

def max_drawdown(r: pd.Series) -> float:
    x = pd.to_numeric(r, errors="coerce").fillna(0.0)
    wealth = (1.0 + x).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min()) if len(dd) else np.nan

def perf_stats(daily: pd.DataFrame, col: str) -> dict[str, Any]:
    r = pd.to_numeric(daily[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"return_col": col, "n_days": 0}
    mu = float(r.mean())
    sd = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    out = {
        "return_col": col,
        "n_days": int(len(r)),
        "mean_daily": mu,
        "vol_daily": sd,
        "tstat_naive_daily": float(mu / (sd / math.sqrt(len(r)))) if sd and np.isfinite(sd) and sd > 0 else np.nan,
        "ann_sharpe_naive": float(mu / sd * math.sqrt(252)) if sd and np.isfinite(sd) and sd > 0 else np.nan,
        "cumulative_return": float((1.0 + r).prod() - 1.0),
        "max_drawdown": max_drawdown(r),
        "win_rate": float((r > 0).mean()),
    }
    try:
        import statsmodels.api as sm
        y = r.to_numpy(dtype=float)
        ols = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": min(20, max(1, int(len(y) ** 0.25)))})
        out["tstat_hac_daily_mean"] = float(ols.tvalues[0])
    except Exception:
        out["tstat_hac_daily_mean"] = np.nan
    return out

def run_strategy(root: Path, returns: pd.DataFrame, calendar: np.ndarray, h: int, model: str, years: list[int], split: str, top_frac: float, min_names: int, costs: list[float], protected: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pred_col = PRED_COLS[model]
    pred = read_predictions(root, h, model, years, split)
    pred["base_date"] = map_base_dates(pred["signal_date"], calendar)
    pred = pred.dropna(subset=["base_date"])
    selected, signal_diag = select_positions(pred, pred_col, top_frac, min_names)
    positions = expand_holdings(selected, calendar, h)
    weights = normalize_weights(positions, returns)
    daily = portfolio_returns(weights, costs)
    daily["horizon_days"] = h
    daily["model"] = model
    daily["calendar_year"] = pd.to_datetime(daily["date"]).dt.year
    signal_diag["horizon_days"] = h
    signal_diag["model"] = model
    out = protected / f"h{h}d_{model}_{min(years)}_{max(years)}"
    out.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out / "selected_signal_positions.parquet", index=False)
    daily.to_parquet(out / "daily_portfolio_returns.parquet", index=False)
    weights[["date", "permno", "weight", "ret_adj", "abret_mkt"]].to_parquet(out / "daily_security_weights.parquet", index=False)
    val = {
        "horizon_days": h, "model": model,
        "prediction_rows": int(len(pred)),
        "selected_signal_position_rows": int(len(selected)),
        "expanded_position_rows": int(len(positions)),
        "weighted_security_day_rows": int(len(weights)),
        "daily_return_rows": int(len(daily)),
        "signal_dates_total": int(signal_diag["signal_date"].nunique()),
        "signal_dates_skipped": int(signal_diag["skipped"].sum()),
        "avg_daily_positions": float(daily["n_positions"].mean()),
        "avg_daily_turnover": float(daily["one_way_turnover"].mean()),
        "avg_daily_gross": float(daily["gross"].mean()),
        "max_abs_gross_error": float((daily["gross"] - 1.0).abs().max()),
    }
    return daily, signal_diag, val

def performance_tables(daily_all: pd.DataFrame, costs: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, yr = [], []
    for (h, model), g in daily_all.groupby(["horizon_days", "model"], sort=True):
        for col in ["portfolio_abnormal_return", "portfolio_raw_return"]:
            r = perf_stats(g, col); r.update({"horizon_days": h, "model": model, "cost_bps": 0.0, "net_or_gross": "gross", "period": "full"}); rows.append(r)
        for bps in costs:
            s = str(bps).replace(".", "p")
            for col in [f"net_abnormal_return_cost_{s}bps", f"net_raw_return_cost_{s}bps"]:
                if col in g:
                    r = perf_stats(g, col); r.update({"horizon_days": h, "model": model, "cost_bps": bps, "net_or_gross": "net", "period": "full"}); rows.append(r)
        for year, gy in g.groupby("calendar_year"):
            for bps in costs:
                s = str(bps).replace(".", "p")
                col = f"net_abnormal_return_cost_{s}bps"
                if col in gy:
                    r = perf_stats(gy, col); r.update({"horizon_days": h, "model": model, "cost_bps": bps, "calendar_year": int(year)}); yr.append(r)
    return pd.DataFrame(rows), pd.DataFrame(yr)

def choose_candidate(perf: pd.DataFrame, yearly: pd.DataFrame) -> dict[str, Any]:
    c = perf[perf["return_col"].astype(str).str.contains("net_abnormal_return") & perf["cost_bps"].eq(10.0) & perf["period"].eq("full")].copy()
    if c.empty:
        return {"status": "no_candidate"}
    pos_years = []
    for r in c.itertuples(index=False):
        y = yearly[(yearly["horizon_days"].eq(r.horizon_days)) & (yearly["model"].eq(r.model)) & (yearly["cost_bps"].eq(r.cost_bps))]
        pos_years.append(int((pd.to_numeric(y["ann_sharpe_naive"], errors="coerce") > 0).sum()) if not y.empty else 0)
    c["n_positive_sharpe_years"] = pos_years
    c["selection_score"] = pd.to_numeric(c["ann_sharpe_naive"], errors="coerce").fillna(-999) + 0.05 * c["n_positive_sharpe_years"]
    b = c.sort_values(["selection_score", "ann_sharpe_naive", "cumulative_return"], ascending=False).iloc[0]
    return {
        "status": "selected",
        "model": str(b["model"]),
        "horizon_days": int(b["horizon_days"]),
        "cost_bps": float(b["cost_bps"]),
        "ann_sharpe_naive": float(b["ann_sharpe_naive"]) if pd.notna(b["ann_sharpe_naive"]) else None,
        "tstat_naive_daily": float(b["tstat_naive_daily"]) if pd.notna(b["tstat_naive_daily"]) else None,
        "tstat_hac_daily_mean": float(b.get("tstat_hac_daily_mean")) if pd.notna(b.get("tstat_hac_daily_mean", np.nan)) else None,
        "cumulative_return": float(b["cumulative_return"]) if pd.notna(b["cumulative_return"]) else None,
        "max_drawdown": float(b["max_drawdown"]) if pd.notna(b["max_drawdown"]) else None,
        "n_positive_sharpe_years": int(b["n_positive_sharpe_years"]),
    }

def make_figures(out_dir: Path, daily_all: pd.DataFrame, perf: pd.DataFrame, yearly: pd.DataFrame) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    primary = "net_abnormal_return_cost_10p0bps"
    for model, gm in daily_all.groupby("model"):
        plt.figure(figsize=(12, 6))
        for h, g in gm.groupby("horizon_days"):
            if primary in g:
                wealth = (1.0 + pd.to_numeric(g[primary], errors="coerce").fillna(0.0)).cumprod() - 1.0
                plt.plot(g["date"], wealth, label=f"{model} {h}d")
        plt.axhline(0, linewidth=1); plt.xlabel("Date"); plt.ylabel("Cumulative net abnormal return")
        plt.title(f"Phase 6.1 full backtest, 10 bps cost — {model}"); plt.legend(ncol=2); plt.tight_layout()
        p = fig_dir / f"phase6_1_cumulative_10bps_{model}.png"; plt.savefig(p, dpi=180); plt.close(); made.append({"figure": f"cumulative_10bps_{model}", "path": str(p)})
    show = perf[perf["return_col"].astype(str).str.contains("net_abnormal_return") & perf["period"].eq("full")]
    for bps, gb in show.groupby("cost_bps"):
        plt.figure(figsize=(11, 6))
        for model, gm in gb.groupby("model"):
            gm = gm.sort_values("horizon_days")
            plt.plot(gm["horizon_days"], gm["ann_sharpe_naive"], marker="o", label=model)
        plt.axhline(0, linewidth=1); plt.xlabel("Horizon"); plt.ylabel("Annualized Sharpe")
        plt.title(f"Cost frontier: net abnormal Sharpe at {bps:g} bps"); plt.legend(); plt.tight_layout()
        p = fig_dir / f"phase6_1_net_sharpe_cost_{str(bps).replace('.','p')}bps.png"; plt.savefig(p, dpi=180); plt.close(); made.append({"figure": f"net_sharpe_cost_{bps:g}bps", "path": str(p)})
    yy = yearly[yearly["cost_bps"].eq(10.0)]
    for model, gm in yy.groupby("model"):
        pivot = gm.pivot_table(index="calendar_year", columns="horizon_days", values="ann_sharpe_naive", aggfunc="first").sort_index()
        plt.figure(figsize=(11, 6)); width = 0.8 / max(len(pivot.columns), 1); x = np.arange(len(pivot.index))
        for i, h in enumerate(pivot.columns):
            plt.bar(x + (i - len(pivot.columns)/2) * width + width/2, pivot[h], width=width, label=f"{h}d")
        plt.axhline(0, linewidth=1); plt.xticks(x, [str(y) for y in pivot.index]); plt.xlabel("Year"); plt.ylabel("Annualized Sharpe")
        plt.title(f"Yearly net abnormal Sharpe, 10 bps — {model}"); plt.legend(); plt.tight_layout()
        p = fig_dir / f"phase6_1_yearly_sharpe_10bps_{model}.png"; plt.savefig(p, dpi=180); plt.close(); made.append({"figure": f"yearly_sharpe_10bps_{model}", "path": str(p)})
    turnover = daily_all.groupby(["model", "horizon_days"], as_index=False)["one_way_turnover"].mean()
    plt.figure(figsize=(10.5, 6))
    for model, gm in turnover.groupby("model"):
        gm = gm.sort_values("horizon_days")
        plt.plot(gm["horizon_days"], gm["one_way_turnover"], marker="o", label=model)
    plt.xlabel("Horizon"); plt.ylabel("Average one-way daily turnover"); plt.title("Average turnover by horizon"); plt.legend(); plt.tight_layout()
    p = fig_dir / "phase6_1_turnover_by_horizon.png"; plt.savefig(p, dpi=180); plt.close(); made.append({"figure": "turnover_by_horizon", "path": str(p)})
    return made

def render_report(path: Path, summary: dict[str, Any], candidate: dict[str, Any], perf: pd.DataFrame, yearly: pd.DataFrame, validation: pd.DataFrame, figures: list[dict[str, str]], env: dict[str, Any]) -> None:
    def table(df: pd.DataFrame) -> str:
        return df.to_html(index=False, escape=True, classes="data") if df is not None and not df.empty else "<p>No rows.</p>"
    cards = []
    for label, val in [
        ("Validation", "PASS" if summary.get("validation_passed") else "FAIL"),
        ("Strategies", summary.get("n_strategies")),
        ("Daily rows", summary.get("daily_rows")),
        ("Production model", candidate.get("model")),
        ("Production horizon", candidate.get("horizon_days")),
        ("Net Sharpe @10bps", candidate.get("ann_sharpe_naive")),
    ]:
        txt = f"{val:,.3f}" if isinstance(val, float) else f"{val:,}" if isinstance(val, int) else html.escape(str(val))
        cards.append(f"<div class='card'><div class='kicker'>{html.escape(label)}</div><h3>{txt}</h3></div>")
    fig_html = "".join(f"<div class='figure'><h3>{html.escape(f['figure'].replace('_',' ').title())}</h3><img src='figures/{html.escape(Path(f['path']).name)}'></div>" for f in figures)
    top = perf[perf["return_col"].astype(str).str.contains("net_abnormal_return") & perf["cost_bps"].eq(10.0)].sort_values("ann_sharpe_naive", ascending=False).head(30)
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Phase 6.1 Full Backtest</title>
<style>:root{{--bg:#07111f;--text:#eef6ff;--muted:#9fb7ce;--line:rgba(255,255,255,.14)}}*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif;background:radial-gradient(circle at top left,#183b66,var(--bg) 42%);color:var(--text)}}header{{padding:46px 56px 28px;border-bottom:1px solid var(--line)}}h1{{margin:0;font-size:42px;letter-spacing:-.04em}}.subtitle{{color:var(--muted);font-size:17px;max-width:1100px;line-height:1.55}}main{{padding:28px 56px 60px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px;margin:22px 0 36px}}.card{{background:linear-gradient(180deg,rgba(255,255,255,.075),rgba(255,255,255,.035));border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 18px 40px rgba(0,0,0,.18)}}.card h3{{margin:7px 0 0;font-size:28px}}.kicker{{text-transform:uppercase;font-size:11px;letter-spacing:.16em;color:var(--muted)}}section{{background:rgba(15,30,51,.78);border:1px solid var(--line);border-radius:22px;padding:24px;margin:22px 0;overflow:auto}}table.data{{width:100%;border-collapse:collapse;font-size:13px}}table.data th{{text-align:left;color:#d8eaff;background:rgba(255,255,255,.08)}}table.data th,table.data td{{padding:9px 10px;border-bottom:1px solid rgba(255,255,255,.09);vertical-align:top}}.figure img{{width:100%;max-width:1120px;border-radius:16px;border:1px solid var(--line);background:white}}pre{{white-space:pre-wrap;background:rgba(0,0,0,.28);border:1px solid var(--line);border-radius:14px;padding:16px;color:#dbecff}}</style></head>
<body><header><h1>Phase 6.1 Full Portfolio Backtest</h1><p class='subtitle'>Full local-only portfolio diagnostics on the 2022–2024 test sample across raw, ridge, and LightGBM signals, all horizons, and transaction-cost assumptions. Protected position-level Parquet files remain local.</p></header><main>
<div class='grid'>{''.join(cards)}</div>
<section><h2>Production candidate</h2>{table(pd.DataFrame([candidate]))}</section>
<section><h2>Top 10 bps net abnormal strategies</h2>{table(top)}</section>
<section><h2>Full performance summary</h2>{table(perf)}</section>
<section><h2>Yearly performance summary</h2>{table(yearly)}</section>
<section><h2>Validation diagnostics</h2>{table(validation)}</section>
<section><h2>Figures</h2>{fig_html}</section>
<section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main></body></html>"""
    path.write_text(doc)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--test-years", default="2022-2024")
    ap.add_argument("--horizons", default="1,2,5,10,20")
    ap.add_argument("--models", default="raw,ridge,lightgbm")
    ap.add_argument("--split", default="test")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--min-names-per-signal", type=int, default=50)
    ap.add_argument("--cost-bps", default="0,5,10,25,50")
    args = ap.parse_args()

    root = args.project_root.resolve(); ensure_workspace(root)
    out_dir = args.out_dir.resolve(); log_dir = args.log_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    protected = root / "data" / "processed" / "backtests" / "phase6_1_backtest_full"
    protected.mkdir(parents=True, exist_ok=True)

    years = parse_ints(args.test_years); horizons = parse_ints(args.horizons)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    costs = parse_floats(args.cost_bps)
    bad = sorted(set(models) - set(PRED_COLS))
    if bad:
        raise ValueError(f"Unknown models: {bad}")

    print("=" * 80)
    print("Phase 6.1 full portfolio/backtest")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {root}")
    print(f"Output dir: {out_dir}")
    print(f"Protected backtest dir: {protected}")
    print(f"Years={years}; horizons={horizons}; models={models}; costs={costs}")
    print("=" * 80)

    env = {
        "utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_count": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in ["PNA_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]},
        "git": run_cmd(["bash", "-lc", "command -v git || true"]),
    }
    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    returns = read_returns(root, min(years), max(years))
    calendar = np.array(sorted(pd.to_datetime(returns["date"].dropna().drop_duplicates()).to_numpy(dtype="datetime64[ns]")))
    print(f"[RETURNS] rows={len(returns):,}, calendar_days={len(calendar):,}, first={pd.Timestamp(calendar[0]).date()}, last={pd.Timestamp(calendar[-1]).date()}")

    daily_frames, signal_frames, validation_rows = [], [], []
    for h in horizons:
        for model in models:
            print(f"[BACKTEST] model={model}, horizon={h}d")
            daily, sig, val = run_strategy(root, returns, calendar, h, model, years, args.split, args.top_frac, args.min_names_per_signal, costs, protected)
            daily_frames.append(daily); signal_frames.append(sig); validation_rows.append(val)
            print(f"[BACKTEST] {model} h={h}: daily_rows={len(daily):,}, avg_turnover={val['avg_daily_turnover']:.3f}, avg_positions={val['avg_daily_positions']:.1f}")

    daily_all = pd.concat(daily_frames, ignore_index=True)
    signal_diag = pd.concat(signal_frames, ignore_index=True)
    validation = pd.DataFrame(validation_rows)
    perf, yearly = performance_tables(daily_all, costs)
    candidate = choose_candidate(perf, yearly)

    daily_all.to_csv(out_dir / "phase6_1_daily_portfolio_returns.csv", index=False)
    signal_diag.to_csv(out_dir / "phase6_1_signal_date_diagnostics.csv", index=False)
    validation.to_csv(out_dir / "phase6_1_validation_diagnostics.csv", index=False)
    perf.to_csv(out_dir / "phase6_1_performance_summary.csv", index=False)
    yearly.to_csv(out_dir / "phase6_1_yearly_performance_summary.csv", index=False)

    checks = {
        "daily_rows_positive": len(daily_all) > 100,
        "all_strategies_completed": len(validation) == len(horizons) * len(models),
        "gross_normalized_close_to_one": bool(validation["max_abs_gross_error"].fillna(999).lt(1e-8).all()),
        "turnover_finite": bool(np.isfinite(validation["avg_daily_turnover"]).all()),
        "performance_finite": bool(np.isfinite(pd.to_numeric(perf["ann_sharpe_naive"], errors="coerce").dropna()).all()),
        "candidate_selected": candidate.get("status") == "selected",
        "full_test_span_has_2022_2024": set([2022, 2023, 2024]).issubset(set(pd.to_datetime(daily_all["date"]).dt.year.unique())),
    }
    summary = {
        "generated_at_utc": utc_now(),
        "workspace": str(WORKSPACE),
        "project_root": str(root),
        "test_years": years,
        "horizons": horizons,
        "models": models,
        "top_frac": args.top_frac,
        "cost_bps": costs,
        "n_strategies": int(len(validation)),
        "daily_rows": int(len(daily_all)),
        "production_candidate": candidate,
        "checks": checks,
        "validation_passed": bool(all(checks.values())),
        "protected_backtest_dir": str(protected),
        "note": "Local-only full backtest. Uses Phase 5.1 predictions and Phase 4.1 CRSP return panel; no WRDS query.",
    }
    with (out_dir / "phase6_1_quality_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    inventory = [{"file": str(p), "size_bytes": p.stat().st_size, "protected_local_only": True} for p in sorted(protected.rglob("*.parquet"))]
    pd.DataFrame(inventory).to_csv(out_dir / "protected_local_phase6_1_inventory.csv", index=False)
    figures = make_figures(out_dir, daily_all, perf, yearly)
    render_report(out_dir / "phase6_1_full_backtest_report.html", summary, candidate, perf, yearly, validation, figures, env)

    lines = [
        "# Phase 6.1 full portfolio/backtest summary", "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Validation passed: {summary['validation_passed']}",
        f"- Test years: {years}",
        f"- Horizons: {horizons}",
        f"- Models: {models}",
        f"- Strategies completed: {summary['n_strategies']}",
        f"- Daily return rows: {summary['daily_rows']:,}",
        f"- Production candidate: {candidate}",
        f"- Report: {out_dir / 'phase6_1_full_backtest_report.html'}", "",
        "Data policy: protected position/weight Parquet files remain local and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE6_1_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if summary["validation_passed"] else 4

if __name__ == "__main__":
    raise SystemExit(main())
