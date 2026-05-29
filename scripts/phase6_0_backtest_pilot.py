from __future__ import annotations

import argparse
import datetime as dt
import html
import importlib.metadata as md
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

WORKSPACE_REQUIRED = Path("<workspace>")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_cmd(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}


def package_status() -> list[dict[str, str]]:
    pkgs = ["pandas", "numpy", "pyarrow", "matplotlib", "scipy", "statsmodels"]
    rows = []
    for pkg in pkgs:
        try:
            rows.append({"package": pkg, "version": md.version(pkg), "status": "ok"})
        except md.PackageNotFoundError:
            rows.append({"package": pkg, "version": "", "status": "missing"})
    return rows


def parse_ints(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_floats(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw).split(','):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def ensure_workspace(project_root: Path) -> None:
    ws = Path("<workspace>")
    if not ws.exists():
        raise RuntimeError(f"Workspace does not exist: {ws}")
    if project_root.resolve() != (ws / "production-network-alpha").resolve():
        raise RuntimeError(f"Wrong project_root={project_root}; expected {ws / 'production-network-alpha'}")


def prediction_path(project_root: Path, horizon: int) -> Path:
    return project_root / "data" / "processed" / "model_runs" / "phase5_1_modeling_full" / f"phase5_1_predictions_h{horizon}d.parquet"


def returns_path(project_root: Path) -> Path:
    return project_root / "data" / "processed" / "model_matrix_full" / "full_crsp_return_panel_features.parquet"


def read_predictions(path: Path, pred_col: str, pilot_year: int, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 5.1 prediction cache: {path}")
    needed = ["signal_date", "target_permno", "target_label", "split", "signal_year", pred_col]
    df = pd.read_parquet(path, columns=needed)
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df["target_permno"] = pd.to_numeric(df["target_permno"], errors="coerce").astype("Int64")
    df["signal_year"] = pd.to_numeric(df["signal_year"], errors="coerce").astype("Int64")
    df["target_label"] = pd.to_numeric(df["target_label"], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df[df["split"].eq(split) & df["signal_year"].eq(pilot_year)].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["signal_date", "target_permno", "target_label", pred_col])
    if df.empty:
        raise RuntimeError(f"No predictions after filter path={path}, split={split}, year={pilot_year}, pred_col={pred_col}")
    return df


def read_returns(project_root: Path, start_year: int, end_year: int) -> pd.DataFrame:
    path = returns_path(project_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing CRSP return panel from Phase 4.1: {path}")
    cols = ["permno", "date", "ret_adj", "abret_mkt", "vwretd", "dollar_vol_21d"]
    ret = pd.read_parquet(path, columns=cols)
    ret["date"] = pd.to_datetime(ret["date"], errors="coerce")
    ret["permno"] = pd.to_numeric(ret["permno"], errors="coerce").astype("Int64")
    for c in ["ret_adj", "abret_mkt", "vwretd", "dollar_vol_21d"]:
        ret[c] = pd.to_numeric(ret[c], errors="coerce")
    # Keep enough buffer for holding periods that roll into next year.
    lo = pd.Timestamp(year=start_year, month=1, day=1) - pd.Timedelta(days=10)
    hi = pd.Timestamp(year=end_year + 1, month=3, day=31)
    ret = ret[(ret["date"] >= lo) & (ret["date"] <= hi)].copy()
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna(subset=["permno", "date", "ret_adj", "abret_mkt"])
    if ret.empty:
        raise RuntimeError("Return panel is empty after pilot date filter")
    return ret


def map_base_dates(signal_dates: pd.Series, calendar: np.ndarray) -> pd.Series:
    sig = pd.to_datetime(signal_dates, errors="coerce").to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(calendar, sig, side="left")
    mapped = np.full(len(sig), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = idx < len(calendar)
    mapped[ok] = calendar[idx[ok]]
    return pd.Series(pd.to_datetime(mapped), index=signal_dates.index)


def select_signal_positions(pred: pd.DataFrame, pred_col: str, top_frac: float, min_names: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diag = []
    for d, g in pred.groupby("signal_date", sort=True):
        g = g.dropna(subset=[pred_col, "target_permno"]).copy()
        n = len(g)
        if n < min_names or g[pred_col].nunique() < 10:
            diag.append({"signal_date": d, "n_names": n, "selected_long": 0, "selected_short": 0, "skipped": True})
            continue
        k = max(1, int(math.floor(n * top_frac)))
        # Stable first ranking prevents qcut ties from breaking the pipeline.
        g = g.sort_values(pred_col, ascending=True)
        short = g.head(k).copy()
        long = g.tail(k).copy()
        short["side"] = "short"
        long["side"] = "long"
        short["base_weight"] = -0.5 / len(short)
        long["base_weight"] = 0.5 / len(long)
        rows.append(pd.concat([long, short], ignore_index=True)[["signal_date", "base_date", "target_permno", pred_col, "target_label", "side", "base_weight"]])
        diag.append({"signal_date": d, "n_names": n, "selected_long": len(long), "selected_short": len(short), "skipped": False})
    if not rows:
        raise RuntimeError("No signal dates had enough names for portfolio construction")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diag)


def expand_holdings(selected: pd.DataFrame, calendar: np.ndarray, horizon: int) -> pd.DataFrame:
    cal = pd.to_datetime(calendar)
    cal_series = pd.Series(cal)
    rows = []
    # Grouping by base_date is efficient and keeps the explicit holding-period rule auditable.
    for base_date, g in selected.groupby("base_date", sort=True):
        if pd.isna(base_date):
            continue
        idx_arr = np.where(calendar == np.datetime64(pd.Timestamp(base_date)))[0]
        if len(idx_arr) == 0:
            continue
        idx0 = int(idx_arr[0])
        hold_idx = range(idx0 + 1, min(idx0 + horizon + 1, len(calendar)))
        for j in hold_idx:
            hdate = pd.Timestamp(calendar[j])
            part = g[["signal_date", "target_permno", "side", "base_weight"]].copy()
            part["date"] = hdate
            rows.append(part)
    if not rows:
        raise RuntimeError("No holdings were generated; check calendar and base-date mapping")
    pos = pd.concat(rows, ignore_index=True).rename(columns={"target_permno": "permno", "base_weight": "raw_weight"})
    pos["permno"] = pd.to_numeric(pos["permno"], errors="coerce").astype("Int64")
    return pos.dropna(subset=["permno", "date", "raw_weight"])


def normalize_weights_after_returns(positions: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    p = positions.groupby(["date", "permno"], as_index=False).agg(raw_weight=("raw_weight", "sum"))
    merged = p.merge(returns[["date", "permno", "ret_adj", "abret_mkt", "dollar_vol_21d"]], on=["date", "permno"], how="inner")
    if merged.empty:
        raise RuntimeError("No portfolio holdings matched CRSP returns")
    gross = merged.groupby("date")["raw_weight"].transform(lambda x: float(np.abs(x).sum()))
    merged = merged[gross > 0].copy()
    merged["weight"] = merged["raw_weight"] / gross[gross > 0]
    # Final hard check: gross should be 1.0 per active date after return availability.
    daily_gross = merged.groupby("date")["weight"].apply(lambda x: np.abs(x).sum()).rename("gross")
    merged = merged.merge(daily_gross.reset_index(), on="date", how="left")
    return merged


def turnover_by_date(weights: pd.DataFrame) -> pd.DataFrame:
    out = []
    prev: dict[int, float] = {}
    for date, g in weights.sort_values("date").groupby("date", sort=True):
        curr = {int(r.permno): float(r.weight) for r in g[["permno", "weight"]].itertuples(index=False)}
        keys = set(prev) | set(curr)
        turnover = 0.5 * sum(abs(curr.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        out.append({"date": pd.Timestamp(date), "one_way_turnover": float(turnover), "n_positions": int(len(curr)), "gross_after_normalization": float(np.abs(g["weight"]).sum())})
        prev = curr
    return pd.DataFrame(out)


def portfolio_daily_returns(weights: pd.DataFrame, cost_bps: list[float]) -> pd.DataFrame:
    daily = (
        weights.groupby("date", as_index=False)
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
    to = turnover_by_date(weights)
    daily = daily.merge(to[["date", "one_way_turnover"]], on="date", how="left")
    daily["one_way_turnover"] = daily["one_way_turnover"].fillna(0.0)
    for bps in cost_bps:
        suffix = str(bps).replace('.', 'p')
        daily[f"net_abnormal_return_cost_{suffix}bps"] = daily["portfolio_abnormal_return"] - daily["one_way_turnover"] * (bps / 10000.0)
        daily[f"net_raw_return_cost_{suffix}bps"] = daily["portfolio_raw_return"] - daily["one_way_turnover"] * (bps / 10000.0)
    return daily.sort_values("date")


def max_drawdown(ret: pd.Series) -> float:
    r = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan


def perf_stats(daily: pd.DataFrame, return_col: str) -> dict[str, Any]:
    r = pd.to_numeric(daily[return_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"return_col": return_col, "n_days": 0}
    mu = float(r.mean())
    sd = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    tstat = float(mu / (sd / math.sqrt(len(r)))) if sd and np.isfinite(sd) and sd > 0 else np.nan
    sharpe = float(mu / sd * math.sqrt(252)) if sd and np.isfinite(sd) and sd > 0 else np.nan
    cum = float((1.0 + r).prod() - 1.0)
    return {
        "return_col": return_col,
        "n_days": int(len(r)),
        "mean_daily": mu,
        "vol_daily": sd,
        "tstat_naive_daily": tstat,
        "ann_sharpe_naive": sharpe,
        "cumulative_return": cum,
        "max_drawdown": max_drawdown(r),
        "win_rate": float((r > 0).mean()),
    }


def run_one_strategy(
    project_root: Path,
    horizon: int,
    model: str,
    pilot_year: int,
    split: str,
    top_frac: float,
    min_names: int,
    cost_bps: list[float],
    protected_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pred_col_map = {"lightgbm": "pred_lightgbm", "ridge": "pred_ridge", "raw": "pred_raw"}
    pred_col = pred_col_map[model]
    pred = read_predictions(prediction_path(project_root, horizon), pred_col, pilot_year, split)
    returns = read_returns(project_root, pilot_year, pilot_year + 1)
    calendar = np.array(sorted(pd.to_datetime(returns["date"].dropna().drop_duplicates()).to_numpy(dtype="datetime64[ns]")))
    pred["base_date"] = map_base_dates(pred["signal_date"], calendar)
    pred = pred.dropna(subset=["base_date"])
    selected, signal_diag = select_signal_positions(pred, pred_col, top_frac, min_names)
    positions = expand_holdings(selected, calendar, horizon)
    weights = normalize_weights_after_returns(positions, returns)
    daily = portfolio_daily_returns(weights, cost_bps)
    daily["horizon_days"] = horizon
    daily["model"] = model
    daily["pilot_year"] = pilot_year
    signal_diag["horizon_days"] = horizon
    signal_diag["model"] = model
    signal_diag["pilot_year"] = pilot_year
    # Protected local-only details for audit; not included in upload bundle.
    out_sub = protected_dir / f"h{horizon}d_{model}_{pilot_year}"
    out_sub.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out_sub / "selected_signal_positions.parquet", index=False)
    daily.to_parquet(out_sub / "daily_portfolio_returns.parquet", index=False)
    weights[["date", "permno", "weight", "ret_adj", "abret_mkt"]].to_parquet(out_sub / "daily_security_weights.parquet", index=False)
    validation = {
        "horizon_days": horizon,
        "model": model,
        "pilot_year": pilot_year,
        "prediction_rows": int(len(pred)),
        "selected_signal_position_rows": int(len(selected)),
        "expanded_position_rows": int(len(positions)),
        "weighted_security_day_rows": int(len(weights)),
        "daily_return_rows": int(len(daily)),
        "signal_dates_total": int(signal_diag["signal_date"].nunique()),
        "signal_dates_skipped": int(signal_diag["skipped"].sum()),
        "avg_daily_positions": float(daily["n_positions"].mean()) if len(daily) else np.nan,
        "avg_daily_turnover": float(daily["one_way_turnover"].mean()) if len(daily) else np.nan,
        "avg_daily_gross": float(daily["gross"].mean()) if len(daily) else np.nan,
        "max_abs_gross_error": float((daily["gross"] - 1.0).abs().max()) if len(daily) else np.nan,
    }
    return daily, selected, signal_diag, validation


def make_figures(out_dir: Path, daily_all: pd.DataFrame, perf: pd.DataFrame) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, str]] = []
    if daily_all.empty:
        return made

    for (h, m), g in daily_all.groupby(["horizon_days", "model"]):
        plt.figure(figsize=(11, 5.8))
        for col in [c for c in g.columns if c.startswith("net_abnormal_return_cost_")]:
            label = col.replace("net_abnormal_return_cost_", "").replace("p", ".")
            wealth = (1.0 + pd.to_numeric(g[col], errors="coerce").fillna(0.0)).cumprod() - 1.0
            plt.plot(g["date"], wealth, label=label)
        plt.axhline(0, linewidth=1)
        plt.xlabel("Date")
        plt.ylabel("Cumulative abnormal return")
        plt.title(f"Phase 6.0 pilot cumulative net abnormal return — {m}, {h}d")
        plt.legend(title="Cost")
        plt.tight_layout()
        path = fig_dir / f"phase6_0_cumulative_{m}_h{h}d.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": f"cumulative_{m}_h{h}d", "path": str(path)})

        plt.figure(figsize=(10.5, 5.8))
        plt.hist(pd.to_numeric(g["one_way_turnover"], errors="coerce").dropna(), bins=50)
        plt.xlabel("One-way turnover")
        plt.ylabel("Days")
        plt.title(f"Phase 6.0 pilot turnover — {m}, {h}d")
        plt.tight_layout()
        path = fig_dir / f"phase6_0_turnover_{m}_h{h}d.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": f"turnover_{m}_h{h}d", "path": str(path)})

    if not perf.empty:
        show = perf[perf["return_col"].str.contains("net_abnormal_return")].copy()
        show = show.sort_values(["horizon_days", "model", "cost_bps"])
        plt.figure(figsize=(10.5, 5.8))
        for model, gm in show.groupby("model"):
            # Plot the primary 10 bps cost if present, else first cost.
            pick = gm[gm["cost_bps"].eq(10.0)] if (gm["cost_bps"].eq(10.0)).any() else gm
            pick = pick.drop_duplicates(["horizon_days"])
            plt.plot(pick["horizon_days"], pick["ann_sharpe_naive"], marker="o", label=model)
        plt.axhline(0, linewidth=1)
        plt.xlabel("Horizon, trading days")
        plt.ylabel("Naive annualized Sharpe, net abnormal return")
        plt.title("Phase 6.0 pilot net abnormal Sharpe by horizon")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "phase6_0_net_sharpe_by_horizon.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "net_sharpe_by_horizon", "path": str(path)})

    return made


def render_report(out_path: Path, summary: dict[str, Any], perf: pd.DataFrame, validation: pd.DataFrame, daily_preview: pd.DataFrame, figures: list[dict[str, str]], env: dict[str, Any]) -> None:
    cards = []
    metrics = [
        ("Validation", "PASS" if summary.get("validation_passed") else "FAIL"),
        ("Strategies", summary.get("n_strategies")),
        ("Daily rows", summary.get("daily_rows")),
        ("Best net Sharpe", summary.get("best_net_sharpe_10bps")),
        ("Best strategy", summary.get("best_strategy_10bps")),
        ("Pilot year", summary.get("pilot_year")),
    ]
    for label, val in metrics:
        if isinstance(val, float):
            txt = f"{val:,.3f}"
        elif isinstance(val, int):
            txt = f"{val:,}"
        else:
            txt = html.escape(str(val))
        cards.append(f"<div class='card'><div class='kicker'>{html.escape(label)}</div><h3>{txt}</h3></div>")

    def table(df: pd.DataFrame) -> str:
        return df.to_html(index=False, escape=True, classes="data") if df is not None and not df.empty else "<p>No rows.</p>"

    fig_html = []
    for fig in figures:
        rel = "figures/" + Path(fig["path"]).name
        fig_html.append(f"<div class='figure'><h3>{html.escape(fig['figure'].replace('_', ' ').title())}</h3><img src='{html.escape(rel)}'></div>")

    doc = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>Phase 6.0 Portfolio Backtest Pilot</title>
<style>
:root {{ --bg:#07111f; --text:#eef6ff; --muted:#9fb7ce; --line:rgba(255,255,255,.14); }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#183b66,var(--bg) 42%); color:var(--text); }}
header {{ padding:46px 56px 28px; border-bottom:1px solid var(--line); }} h1 {{ margin:0; font-size:42px; letter-spacing:-.04em; }} .subtitle {{ color:var(--muted); font-size:17px; max-width:1080px; line-height:1.55; }}
main {{ padding:28px 56px 60px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:18px; margin:22px 0 36px; }} .card {{ background:linear-gradient(180deg,rgba(255,255,255,.075),rgba(255,255,255,.035)); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 18px 40px rgba(0,0,0,.18); }} .card h3 {{ margin:7px 0 0; font-size:28px; }} .kicker {{ text-transform:uppercase; font-size:11px; letter-spacing:.16em; color:var(--muted); }}
section {{ background:rgba(15,30,51,.78); border:1px solid var(--line); border-radius:22px; padding:24px; margin:22px 0; overflow:auto; }} table.data {{ width:100%; border-collapse:collapse; font-size:13px; }} table.data th {{ text-align:left; color:#d8eaff; background:rgba(255,255,255,.08); }} table.data th, table.data td {{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.09); vertical-align:top; }} .figure img {{ width:100%; max-width:1100px; border-radius:16px; border:1px solid var(--line); background:white; }} pre {{ white-space:pre-wrap; background:rgba(0,0,0,.28); border:1px solid var(--line); border-radius:14px; padding:16px; color:#dbecff; }}
</style></head><body><header><h1>Phase 6.0 Portfolio Backtest Pilot</h1><p class='subtitle'>Local-only pilot portfolio diagnostics from Phase 5.1 predictions and Phase 4.1 CRSP return panel. This tests the trading engine before the single full-scale backtest.</p></header><main>
<div class='grid'>{''.join(cards)}</div>
<section><h2>Performance summary</h2>{table(perf)}</section>
<section><h2>Validation diagnostics</h2>{table(validation)}</section>
<section><h2>Daily return preview</h2>{table(daily_preview)}</section>
<section><h2>Figures</h2>{''.join(fig_html)}</section>
<section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main></body></html>"""
    out_path.write_text(doc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--pilot-year", type=int, default=2024)
    parser.add_argument("--horizons", default="5,10")
    parser.add_argument("--models", default="lightgbm")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--min-names-per-signal", type=int, default=50)
    parser.add_argument("--cost-bps", default="0,5,10,25")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    ensure_workspace(project_root)
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    protected_dir = project_root / "data" / "processed" / "backtests" / "phase6_0_backtest_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    protected_dir.mkdir(parents=True, exist_ok=True)

    horizons = parse_ints(args.horizons)
    models = [m.strip().lower() for m in args.models.split(',') if m.strip()]
    allowed = {"lightgbm", "ridge", "raw"}
    if not set(models).issubset(allowed):
        raise ValueError(f"models must be subset of {allowed}; got {models}")
    cost_bps = parse_floats(args.cost_bps)

    print("================================================================================")
    print("Phase 6.0 portfolio/backtest pilot")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {project_root}")
    print(f"Output dir: {out_dir}")
    print(f"Protected backtest dir: {protected_dir}")
    print(f"Pilot year: {args.pilot_year}; horizons={horizons}; models={models}")
    print("================================================================================")

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
        "packages": package_status(),
        "git": run_cmd(["bash", "-lc", "command -v git || true"]),
    }
    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    daily_frames = []
    signal_diags = []
    validations = []
    for h in horizons:
        for model in models:
            print(f"[BACKTEST] pilot strategy model={model}, horizon={h}d")
            daily, selected, signal_diag, val = run_one_strategy(
                project_root=project_root,
                horizon=h,
                model=model,
                pilot_year=args.pilot_year,
                split=args.split,
                top_frac=args.top_frac,
                min_names=args.min_names_per_signal,
                cost_bps=cost_bps,
                protected_dir=protected_dir,
            )
            daily_frames.append(daily)
            signal_diags.append(signal_diag)
            validations.append(val)
            print(f"[BACKTEST] {model} h={h}: daily_rows={len(daily):,}, avg_turnover={val['avg_daily_turnover']:.3f}")

    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    signal_diag_all = pd.concat(signal_diags, ignore_index=True) if signal_diags else pd.DataFrame()
    validation_df = pd.DataFrame(validations)

    perf_rows = []
    if not daily_all.empty:
        for (h, model), g in daily_all.groupby(["horizon_days", "model"], sort=True):
            for col in ["portfolio_abnormal_return", "portfolio_raw_return"]:
                r = perf_stats(g, col)
                r.update({"horizon_days": h, "model": model, "cost_bps": 0.0, "net_or_gross": "gross"})
                perf_rows.append(r)
            for bps in cost_bps:
                suffix = str(bps).replace('.', 'p')
                for col in [f"net_abnormal_return_cost_{suffix}bps", f"net_raw_return_cost_{suffix}bps"]:
                    if col in g.columns:
                        r = perf_stats(g, col)
                        r.update({"horizon_days": h, "model": model, "cost_bps": bps, "net_or_gross": "net"})
                        perf_rows.append(r)
    perf = pd.DataFrame(perf_rows)

    daily_all.to_csv(out_dir / "phase6_0_daily_portfolio_returns.csv", index=False)
    perf.to_csv(out_dir / "phase6_0_performance_summary.csv", index=False)
    validation_df.to_csv(out_dir / "phase6_0_validation_diagnostics.csv", index=False)
    signal_diag_all.to_csv(out_dir / "phase6_0_signal_date_diagnostics.csv", index=False)

    # Select primary economic summary at 10 bps net abnormal return.
    primary_col = "net_abnormal_return_cost_10p0bps"
    if primary_col not in daily_all.columns:
        primary_col = "net_abnormal_return_cost_10bps" if "net_abnormal_return_cost_10bps" in daily_all.columns else None
    primary_perf = perf[perf["return_col"].astype(str).str.contains("net_abnormal_return") & perf["cost_bps"].eq(10.0)].copy()
    if primary_perf.empty:
        primary_perf = perf[perf["return_col"].eq("portfolio_abnormal_return")].copy()
    best = primary_perf.sort_values("ann_sharpe_naive", ascending=False).head(1)
    best_strategy = None
    best_sharpe = None
    if not best.empty:
        br = best.iloc[0]
        best_strategy = f"{br['model']}_h{int(br['horizon_days'])}d_cost{br['cost_bps']}bps"
        best_sharpe = None if pd.isna(br["ann_sharpe_naive"]) else float(br["ann_sharpe_naive"])

    checks = {
        "daily_rows_positive": len(daily_all) > 20,
        "all_strategies_completed": len(validation_df) == len(horizons) * len(models),
        "gross_normalized_close_to_one": bool(validation_df["max_abs_gross_error"].fillna(999).lt(1e-8).all()) if not validation_df.empty else False,
        "turnover_finite": bool(np.isfinite(validation_df["avg_daily_turnover"]).all()) if not validation_df.empty else False,
        "performance_finite": bool(np.isfinite(pd.to_numeric(perf["ann_sharpe_naive"], errors="coerce").dropna()).all()) if not perf.empty else False,
        "best_strategy_available": best_strategy is not None,
    }

    summary = {
        "generated_at_utc": utc_now(),
        "workspace": str(WORKSPACE_REQUIRED),
        "project_root": str(project_root),
        "pilot_year": args.pilot_year,
        "horizons": horizons,
        "models": models,
        "top_frac": args.top_frac,
        "cost_bps": cost_bps,
        "n_strategies": int(len(validation_df)),
        "daily_rows": int(len(daily_all)),
        "best_strategy_10bps": best_strategy,
        "best_net_sharpe_10bps": best_sharpe,
        "checks": checks,
        "validation_passed": bool(all(checks.values())),
        "protected_backtest_dir": str(protected_dir),
        "note": "Local-only pilot. Uses Phase 5.1 predictions and Phase 4.1 CRSP return panel; no WRDS query.",
    }
    with (out_dir / "phase6_0_quality_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    protected_inventory = []
    for p in sorted(protected_dir.rglob("*.parquet")):
        protected_inventory.append({"file": str(p), "size_bytes": p.stat().st_size, "protected_local_only": True})
    pd.DataFrame(protected_inventory).to_csv(out_dir / "protected_local_phase6_0_inventory.csv", index=False)

    figures = make_figures(out_dir, daily_all, perf)
    render_report(
        out_path=out_dir / "phase6_0_backtest_pilot_report.html",
        summary=summary,
        perf=perf,
        validation=validation_df,
        daily_preview=daily_all.head(50),
        figures=figures,
        env=env,
    )

    lines = [
        "# Phase 6.0 portfolio/backtest pilot summary",
        "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Validation passed: {summary['validation_passed']}",
        f"- Pilot year: {args.pilot_year}",
        f"- Horizons: {horizons}",
        f"- Models: {models}",
        f"- Strategies completed: {summary['n_strategies']}",
        f"- Daily return rows: {summary['daily_rows']:,}",
        f"- Best 10 bps net abnormal strategy: {best_strategy}",
        f"- Best 10 bps net abnormal Sharpe: {best_sharpe}",
        f"- Report: {out_dir / 'phase6_0_backtest_pilot_report.html'}",
        "",
        "Data policy: protected position/weight Parquet files remain local and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE6_0_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if summary["validation_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
