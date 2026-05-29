from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKSPACE = Path("<workspace>")
PRED_COL = "pred_lightgbm"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ints(x: str) -> list[int]:
    out: list[int] = []
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


def read_predictions(root: Path, h: int, years: list[int], split: str) -> pd.DataFrame:
    path = pred_path(root, h)
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    cols = ["signal_date", "target_permno", "target_label", "split", "signal_year", PRED_COL]
    df = pd.read_parquet(path, columns=cols)
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df["target_permno"] = pd.to_numeric(df["target_permno"], errors="coerce").astype("Int64")
    df["signal_year"] = pd.to_numeric(df["signal_year"], errors="coerce").astype("Int64")
    df["target_label"] = safe_num(df["target_label"])
    df[PRED_COL] = safe_num(df[PRED_COL])
    df = df[df["split"].eq(split) & df["signal_year"].isin(set(years))].copy()
    df = df.dropna(subset=["signal_date", "target_permno", PRED_COL])
    if df.empty:
        raise RuntimeError(f"Empty predictions after filter: h={h}, years={years}")
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


def allowed_base_dates(calendar_dates: pd.Series, variant: str) -> set[pd.Timestamp]:
    cal = pd.Series(pd.to_datetime(calendar_dates.dropna().unique())).sort_values().reset_index(drop=True)
    if variant == "daily":
        return set(pd.Timestamp(x) for x in cal)
    if variant == "weekly_first":
        tmp = pd.DataFrame({"date": cal})
        tmp["week"] = tmp["date"].dt.to_period("W-FRI")
        chosen = tmp.groupby("week")["date"].min()
        return set(pd.Timestamp(x) for x in chosen)
    if variant == "weekly_last":
        tmp = pd.DataFrame({"date": cal})
        tmp["week"] = tmp["date"].dt.to_period("W-FRI")
        chosen = tmp.groupby("week")["date"].max()
        return set(pd.Timestamp(x) for x in chosen)
    if variant == "twice_weekly_mon_thu":
        return set(pd.Timestamp(x) for x in cal[cal.dt.weekday.isin([0, 3])])
    if variant == "every_5th_trading_day":
        return set(pd.Timestamp(x) for x in cal.iloc[::5])
    raise ValueError(f"Unknown variant: {variant}")


def select_positions(pred: pd.DataFrame, top_frac: float, min_names: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, diag = [], []
    for d, g in pred.groupby("base_date", sort=True):
        g = g.dropna(subset=[PRED_COL, "target_permno"]).copy()
        n = len(g)
        if n < min_names or g[PRED_COL].nunique() < 10:
            diag.append({"base_date": d, "n_names": n, "selected_long": 0, "selected_short": 0, "skipped": True})
            continue
        k = max(1, int(math.floor(n * top_frac)))
        g = g.sort_values(PRED_COL, ascending=True, kind="mergesort")
        short = g.head(k).copy()
        long = g.tail(k).copy()
        short["side"] = "short"; long["side"] = "long"
        short["base_weight"] = -0.5 / len(short)
        long["base_weight"] = 0.5 / len(long)
        rows.append(pd.concat([long, short], ignore_index=True)[["signal_date", "base_date", "target_permno", PRED_COL, "target_label", "side", "base_weight"]])
        diag.append({"base_date": d, "n_names": n, "selected_long": len(long), "selected_short": len(short), "skipped": False})
    if not rows:
        raise RuntimeError("No eligible base dates for portfolio construction")
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
            p = g[["signal_date", "base_date", "target_permno", "side", "base_weight"]].copy()
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


def run_strategy(root: Path, returns: pd.DataFrame, calendar: np.ndarray, h: int, years: list[int], split: str, variant: str, top_frac: float, min_names: int, costs: list[float], protected: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pred = read_predictions(root, h, years, split)
    pred["base_date"] = map_base_dates(pred["signal_date"], calendar)
    pred = pred.dropna(subset=["base_date"])
    allowed = allowed_base_dates(pred["base_date"], variant)
    pred = pred[pred["base_date"].isin(allowed)].copy()
    if pred.empty:
        raise RuntimeError(f"No predictions after variant filter: {variant}")
    selected, signal_diag = select_positions(pred, top_frac, min_names)
    positions = expand_holdings(selected, calendar, h)
    weights = normalize_weights(positions, returns)
    daily = portfolio_returns(weights, costs)
    daily["horizon_days"] = h
    daily["model"] = "lightgbm"
    daily["rebalance_variant"] = variant
    daily["calendar_year"] = pd.to_datetime(daily["date"]).dt.year
    signal_diag["horizon_days"] = h
    signal_diag["model"] = "lightgbm"
    signal_diag["rebalance_variant"] = variant
    out = protected / f"h{h}d_lightgbm_{variant}_{min(years)}_{max(years)}"
    out.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out / "selected_signal_positions.parquet", index=False)
    daily.to_parquet(out / "daily_portfolio_returns.parquet", index=False)
    weights[["date", "permno", "weight", "ret_adj", "abret_mkt"]].to_parquet(out / "daily_security_weights.parquet", index=False)
    val = {
        "horizon_days": h,
        "model": "lightgbm",
        "rebalance_variant": variant,
        "prediction_rows": int(len(pred)),
        "selected_signal_position_rows": int(len(selected)),
        "expanded_position_rows": int(len(positions)),
        "weighted_security_day_rows": int(len(weights)),
        "daily_return_rows": int(len(daily)),
        "signal_dates_total": int(signal_diag["base_date"].nunique()),
        "signal_dates_skipped": int(signal_diag["skipped"].sum()),
        "avg_daily_positions": float(daily["n_positions"].mean()),
        "avg_daily_turnover": float(daily["one_way_turnover"].mean()),
        "avg_daily_gross": float(daily["gross"].mean()),
        "max_abs_gross_error": float((daily["gross"] - 1.0).abs().max()),
    }
    return daily, signal_diag, val


def performance_tables(daily_all: pd.DataFrame, costs: list[float]) -> pd.DataFrame:
    rows = []
    for (h, variant), g in daily_all.groupby(["horizon_days", "rebalance_variant"], sort=True):
        r = perf_stats(g, "portfolio_abnormal_return")
        r.update({"horizon_days": h, "model": "lightgbm", "rebalance_variant": variant, "cost_bps": 0.0, "net_or_gross": "gross", "period": "full"})
        rows.append(r)
        for bps in costs:
            s = str(bps).replace(".", "p")
            col = f"net_abnormal_return_cost_{s}bps"
            if col in g:
                r = perf_stats(g, col)
                r.update({"horizon_days": h, "model": "lightgbm", "rebalance_variant": variant, "cost_bps": bps, "net_or_gross": "net", "period": "full"})
                rows.append(r)
    return pd.DataFrame(rows)


def make_figures(out_dir: Path, perf: pd.DataFrame, validation: pd.DataFrame, daily_all: pd.DataFrame) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, str]] = []

    p10 = perf[perf["return_col"].eq("net_abnormal_return_cost_10p0bps")].copy()
    if not p10.empty:
        p10["label"] = p10["rebalance_variant"] + " h" + p10["horizon_days"].astype(str)
        p10 = p10.sort_values("ann_sharpe_naive", ascending=True)
        plt.figure(figsize=(11, 6.5))
        plt.barh(p10["label"], p10["ann_sharpe_naive"])
        plt.xlabel("Annualized Sharpe, 10 bps one-way cost")
        plt.title("Phase 6.2 low-turnover pilot: net abnormal Sharpe")
        plt.tight_layout()
        path = fig_dir / "phase6_2_net_sharpe_10bps.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "net_sharpe_10bps", "path": str(path)})

    if not validation.empty:
        v = validation.copy()
        v["label"] = v["rebalance_variant"] + " h" + v["horizon_days"].astype(str)
        v = v.sort_values("avg_daily_turnover", ascending=True)
        plt.figure(figsize=(11, 6.5))
        plt.barh(v["label"], v["avg_daily_turnover"])
        plt.xlabel("Average daily one-way turnover")
        plt.title("Turnover by rebalance variant")
        plt.tight_layout()
        path = fig_dir / "phase6_2_turnover_by_variant.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "turnover_by_variant", "path": str(path)})

    if not daily_all.empty:
        p = daily_all.copy()
        p["date"] = pd.to_datetime(p["date"])
        plt.figure(figsize=(12, 6.5))
        for (h, variant), g in p.groupby(["horizon_days", "rebalance_variant"]):
            if h not in [5, 10]:
                continue
            col = "net_abnormal_return_cost_10p0bps"
            g = g.sort_values("date")
            wealth = (1.0 + pd.to_numeric(g[col], errors="coerce").fillna(0)).cumprod() - 1.0
            plt.plot(g["date"], wealth, label=f"{variant} h{h}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative net abnormal return")
        plt.title("10 bps net cumulative abnormal return, pilot variants")
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = fig_dir / "phase6_2_cumulative_10bps.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "cumulative_10bps", "path": str(path)})

    return made


def render_report(out_path: Path, quality: dict[str, Any], perf: pd.DataFrame, validation: pd.DataFrame, figures: list[dict[str, str]], env: dict[str, Any]) -> None:
    cards = []
    candidate = quality.get("best_cost10_candidate", {})
    metrics = [
        ("Validation", "PASS" if quality.get("validation_passed") else "FAIL"),
        ("Best variant", candidate.get("rebalance_variant")),
        ("Best horizon", candidate.get("horizon_days")),
        ("Best 10bps Sharpe", candidate.get("ann_sharpe_naive")),
        ("Best 10bps cumulative", candidate.get("cumulative_return")),
        ("Best avg turnover", candidate.get("avg_daily_turnover")),
    ]
    for label, val in metrics:
        if isinstance(val, float):
            txt = f"{val:,.4f}"
        elif val is None:
            txt = "n/a"
        else:
            txt = html.escape(str(val))
        cards.append(f"<div class='card'><div class='kicker'>{html.escape(label)}</div><h3>{txt}</h3></div>")

    fig_html = []
    for fig in figures:
        rel = "figures/" + Path(fig["path"]).name
        fig_html.append(f"<div class='figure'><h3>{html.escape(fig['figure'].replace('_',' ').title())}</h3><img src='{html.escape(rel)}'></div>")

    doc = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>Phase 6.2 Low-Turnover Pilot</title>
<style>
:root {{ --bg:#07111f; --text:#eef6ff; --muted:#9fb7ce; --line:rgba(255,255,255,.14); }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#183b66,var(--bg) 42%); color:var(--text); }}
header {{ padding:46px 56px 28px; border-bottom:1px solid var(--line); }} h1 {{ margin:0; font-size:42px; letter-spacing:-.04em; }} .subtitle {{ color:var(--muted); font-size:17px; max-width:1100px; line-height:1.55; }}
main {{ padding:28px 56px 60px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:18px; margin:22px 0 36px; }} .card {{ background:linear-gradient(180deg,rgba(255,255,255,.075),rgba(255,255,255,.035)); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 18px 40px rgba(0,0,0,.18); }} .card h3 {{ margin:7px 0 0; font-size:26px; }} .kicker {{ text-transform:uppercase; font-size:11px; letter-spacing:.16em; color:var(--muted); }}
section {{ background:rgba(15,30,51,.78); border:1px solid var(--line); border-radius:22px; padding:24px; margin:22px 0; overflow:auto; }} table.data {{ width:100%; border-collapse:collapse; font-size:13px; }} table.data th {{ text-align:left; color:#d8eaff; background:rgba(255,255,255,.08); }} table.data th, table.data td {{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.09); vertical-align:top; }} .figure img {{ width:100%; max-width:1120px; border-radius:16px; border:1px solid var(--line); background:white; }} pre {{ white-space:pre-wrap; background:rgba(0,0,0,.28); border:1px solid var(--line); border-radius:14px; padding:16px; color:#dbecff; }}
</style></head><body><header><h1>Phase 6.2 Low-Turnover Pilot</h1><p class='subtitle'>Local-only pilot that tests whether lower-frequency LightGBM rebalancing improves net abnormal-return economics after the Phase 6.1 cost drag. No WRDS queries; protected positions remain local.</p></header><main>
<div class='grid'>{''.join(cards)}</div>
<section><h2>Performance summary</h2>{perf.to_html(index=False, escape=True, classes='data') if not perf.empty else '<p>No performance rows.</p>'}</section>
<section><h2>Validation diagnostics</h2>{validation.to_html(index=False, escape=True, classes='data') if not validation.empty else '<p>No validation rows.</p>'}</section>
<section><h2>Figures</h2>{''.join(fig_html)}</section>
<section><h2>Quality summary</h2><pre>{html.escape(json.dumps(quality, indent=2, default=str))}</pre></section>
<section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main></body></html>"""
    out_path.write_text(doc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--years", default="2024")
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--variants", default="daily,twice_weekly_mon_thu,weekly_first,weekly_last,every_5th_trading_day")
    ap.add_argument("--cost-bps", default="0,5,10,25,50")
    ap.add_argument("--split", default="test")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--min-names", type=int, default=40)
    args = ap.parse_args()

    root = args.project_root.resolve()
    ensure_workspace(root)
    out_dir = args.out_dir.resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir.resolve(); log_dir.mkdir(parents=True, exist_ok=True)
    protected = root / "data" / "processed" / "backtests" / "phase6_2_turnover_pilot"
    protected.mkdir(parents=True, exist_ok=True)

    years = parse_ints(args.years)
    horizons = parse_ints(args.horizons)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    costs = parse_floats(args.cost_bps)

    print("=" * 80)
    print("Phase 6.2 low-turnover portfolio pilot")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {root}")
    print(f"Output dir: {out_dir}")
    print(f"Protected backtest dir: {protected}")
    print(f"Years={years}; horizons={horizons}; variants={variants}; costs={costs}")
    print("=" * 80)

    env = {
        "utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cpu_count": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in ["PNA_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]},
        "git": run_cmd(["bash", "-lc", "command -v git || true"]),
    }

    returns = read_returns(root, min(years), max(years))
    calendar = np.array(sorted(returns["date"].dropna().unique()), dtype="datetime64[ns]")
    print(f"[RETURNS] rows={len(returns):,}, calendar_days={len(calendar):,}, first={pd.Timestamp(calendar.min()).date()}, last={pd.Timestamp(calendar.max()).date()}")

    daily_frames, diag_frames, validation_rows = [], [], []
    errors: list[dict[str, Any]] = []
    for h in horizons:
        for variant in variants:
            try:
                print(f"[PILOT] variant={variant}, horizon={h}d")
                daily, diag, val = run_strategy(root, returns, calendar, h, years, args.split, variant, args.top_frac, args.min_names, costs, protected)
                daily_frames.append(daily)
                diag_frames.append(diag)
                validation_rows.append(val)
                print(f"[PILOT] {variant} h={h}: daily_rows={len(daily):,}, avg_turnover={val['avg_daily_turnover']:.4f}, avg_positions={val['avg_daily_positions']:.1f}")
            except Exception as exc:
                print(f"[WARN] failed variant={variant}, horizon={h}: {exc}")
                errors.append({"horizon_days": h, "rebalance_variant": variant, "error": repr(exc)})

    if not daily_frames:
        raise RuntimeError("No Phase 6.2 strategies completed")

    daily_all = pd.concat(daily_frames, ignore_index=True)
    signal_diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    validation = pd.DataFrame(validation_rows)
    perf = performance_tables(daily_all, costs)

    daily_all.to_csv(out_dir / "phase6_2_daily_portfolio_returns.csv", index=False)
    signal_diag.to_csv(out_dir / "phase6_2_signal_date_diagnostics.csv", index=False)
    validation.to_csv(out_dir / "phase6_2_validation_diagnostics.csv", index=False)
    perf.to_csv(out_dir / "phase6_2_performance_summary.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "phase6_2_errors.csv", index=False)

    p10 = perf[perf["return_col"].eq("net_abnormal_return_cost_10p0bps")].copy()
    if p10.empty:
        candidate = {"status": "not_selected", "reason": "missing_10bps_rows"}
    else:
        joined = p10.merge(validation[["horizon_days", "rebalance_variant", "avg_daily_turnover", "avg_daily_positions"]], on=["horizon_days", "rebalance_variant"], how="left")
        best = joined.sort_values(["ann_sharpe_naive", "cumulative_return"], ascending=[False, False]).iloc[0]
        candidate = best.to_dict()
        candidate["status"] = "selected"

    daily_done = int(len(validation))
    expected = len(horizons) * len(variants)
    checks = {
        "any_strategy_completed": daily_done > 0,
        "at_least_half_strategies_completed": daily_done >= max(1, expected // 2),
        "daily_rows_positive": int(len(daily_all)) > 0,
        "turnover_finite": bool(np.isfinite(validation["avg_daily_turnover"]).all()) if not validation.empty else False,
        "performance_finite": bool(np.isfinite(perf["ann_sharpe_naive"].replace([np.inf, -np.inf], np.nan).dropna()).all()) if not perf.empty else False,
        "candidate_selected": candidate.get("status") == "selected",
    }
    quality = {
        "generated_at_utc": utc_now(),
        "workspace": str(WORKSPACE),
        "project_root": str(root),
        "years": years,
        "horizons": horizons,
        "variants": variants,
        "top_frac": args.top_frac,
        "cost_bps": costs,
        "n_strategies_completed": daily_done,
        "n_strategies_expected": expected,
        "daily_rows": int(len(daily_all)),
        "best_cost10_candidate": candidate,
        "checks": checks,
        "validation_passed": bool(all(checks.values())),
        "protected_backtest_dir": str(protected),
        "note": "Local-only turnover pilot using Phase 5.1 predictions and Phase 4.1 return panel; no WRDS query.",
    }

    with (out_dir / "phase6_2_quality_summary.json").open("w") as f:
        json.dump(quality, f, indent=2, default=str)
    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    figures = make_figures(out_dir, perf, validation, daily_all)
    render_report(out_dir / "phase6_2_turnover_pilot_report.html", quality, perf, validation, figures, env)

    summary_lines = [
        "# Phase 6.2 low-turnover portfolio pilot summary",
        "",
        f"- Generated at UTC: {quality['generated_at_utc']}",
        f"- Validation passed: {quality['validation_passed']}",
        f"- Years: {years}",
        f"- Horizons: {horizons}",
        f"- Variants completed: {daily_done} / {expected}",
        f"- Daily return rows: {len(daily_all):,}",
        f"- Best 10bps candidate: {candidate}",
        f"- Report: {out_dir / 'phase6_2_turnover_pilot_report.html'}",
        "",
        "Data policy: protected position/weight Parquet files remain local and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE6_2_SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))

    return 0 if quality["validation_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
