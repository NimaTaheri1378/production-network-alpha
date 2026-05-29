from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    print(f"[READ] {path}: rows={len(df):,}, cols={len(df.columns):,}")
    return df


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def max_drawdown(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def perf_stats(returns: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {
            "n_days": 0,
            "mean_daily": np.nan,
            "vol_daily": np.nan,
            "ann_sharpe": np.nan,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }
    mean = float(r.mean())
    vol = float(r.std(ddof=1))
    return {
        "n_days": int(len(r)),
        "mean_daily": mean,
        "vol_daily": vol,
        "ann_sharpe": float(np.sqrt(252.0) * mean / vol) if vol > 0 else np.nan,
        "cumulative_return": float((1.0 + r).prod() - 1.0),
        "max_drawdown": max_drawdown(r),
        "win_rate": float((r > 0).mean()),
    }


def moving_block_bootstrap(
    returns: np.ndarray,
    n_boot: int = 1000,
    block_len: int = 20,
    seed: int = 20260529,
) -> dict[str, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < max(30, block_len):
        return {
            "bootstrap_n": int(n_boot),
            "block_len": int(block_len),
            "boot_mean_p05": np.nan,
            "boot_mean_p50": np.nan,
            "boot_mean_p95": np.nan,
            "boot_sharpe_p05": np.nan,
            "boot_sharpe_p50": np.nan,
            "boot_sharpe_p95": np.nan,
            "prob_mean_positive": np.nan,
            "prob_sharpe_positive": np.nan,
        }
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    starts = np.arange(0, n - block_len + 1)
    boot_means = np.empty(n_boot, dtype=float)
    boot_sharpes = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = []
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        for s in chosen:
            sampled.append(r[s : s + block_len])
        x = np.concatenate(sampled)[:n]
        mu = x.mean()
        sd = x.std(ddof=1)
        boot_means[b] = mu
        boot_sharpes[b] = np.sqrt(252.0) * mu / sd if sd > 0 else np.nan
    boot_sharpes = boot_sharpes[np.isfinite(boot_sharpes)]
    return {
        "bootstrap_n": int(n_boot),
        "block_len": int(block_len),
        "boot_mean_p05": float(np.quantile(boot_means, 0.05)),
        "boot_mean_p50": float(np.quantile(boot_means, 0.50)),
        "boot_mean_p95": float(np.quantile(boot_means, 0.95)),
        "boot_sharpe_p05": float(np.quantile(boot_sharpes, 0.05)) if len(boot_sharpes) else np.nan,
        "boot_sharpe_p50": float(np.quantile(boot_sharpes, 0.50)) if len(boot_sharpes) else np.nan,
        "boot_sharpe_p95": float(np.quantile(boot_sharpes, 0.95)) if len(boot_sharpes) else np.nan,
        "prob_mean_positive": float((boot_means > 0).mean()),
        "prob_sharpe_positive": float((boot_sharpes > 0).mean()) if len(boot_sharpes) else np.nan,
    }


def prepare_phase6_3(base: Path) -> dict[str, pd.DataFrame | dict[str, Any]]:
    art = base / "artifacts" / "backtest_turnover_full"
    perf = read_csv_required(art / "phase6_3_performance_summary.csv")
    yearly = read_csv_required(art / "phase6_3_yearly_performance_summary.csv")
    daily = read_csv_required(art / "phase6_3_daily_portfolio_returns.csv")
    validation = read_csv_required(art / "phase6_3_validation_diagnostics.csv")
    quality = read_json_optional(art / "phase6_3_quality_summary.json")
    return {"perf": perf, "yearly": yearly, "daily": daily, "validation": validation, "quality": quality}


def prepare_phase6_1(base: Path) -> dict[str, pd.DataFrame | dict[str, Any]]:
    art = base / "artifacts" / "backtest_full"
    out: dict[str, pd.DataFrame | dict[str, Any]] = {}
    if (art / "phase6_1_performance_summary.csv").exists():
        out["perf"] = pd.read_csv(art / "phase6_1_performance_summary.csv")
    else:
        out["perf"] = pd.DataFrame()
    if (art / "phase6_1_quality_summary.json").exists():
        out["quality"] = read_json_optional(art / "phase6_1_quality_summary.json")
    else:
        out["quality"] = {}
    return out


def prepare_phase5_1(base: Path) -> dict[str, pd.DataFrame | dict[str, Any]]:
    art = base / "artifacts" / "modeling_full"
    out: dict[str, pd.DataFrame | dict[str, Any]] = {}
    for key, fname in [
        ("model_summary", "phase5_1_model_summary.csv"),
        ("feature_importances", "phase5_1_lgbm_feature_importances.csv"),
        ("rank_ic", "phase5_1_daily_rank_ic.csv"),
        ("decile", "phase5_1_decile_monotonicity.csv"),
    ]:
        p = art / fname
        out[key] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    out["quality"] = read_json_optional(art / "phase5_1_quality_summary.json")
    return out


def attach_year_robustness(perf: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    p = perf.copy()
    y = yearly[(yearly["cost_bps"].eq(10.0)) & (yearly["net_or_gross"].eq("net"))].copy()
    if y.empty:
        return p
    y["positive_sharpe"] = y["ann_sharpe_naive"] > 0
    y["positive_return"] = y["cumulative_return"] > 0
    agg = (
        y.groupby(["horizon_days", "model", "rebalance_variant"], dropna=False)
        .agg(
            positive_sharpe_years=("positive_sharpe", "sum"),
            min_year_sharpe=("ann_sharpe_naive", "min"),
            positive_return_years=("positive_return", "sum"),
            min_year_cumulative=("cumulative_return", "min"),
            worst_year_drawdown=("max_drawdown", "min"),
        )
        .reset_index()
    )
    return p.merge(agg, on=["horizon_days", "model", "rebalance_variant"], how="left")


def select_candidates(perf: pd.DataFrame, yearly: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = attach_year_robustness(perf, yearly)
    p = p.merge(
        validation[["horizon_days", "model", "rebalance_variant", "avg_daily_turnover", "avg_daily_positions"]],
        on=["horizon_days", "model", "rebalance_variant"],
        how="left",
    )
    cost10 = p[(p["cost_bps"].eq(10.0)) & (p["net_or_gross"].eq("net"))].copy()
    # Penalise candidates that fail a year, but keep numerical ranking visible.
    cost10["robust_score"] = (
        cost10["ann_sharpe_naive"].fillna(-999)
        + 0.25 * cost10["positive_sharpe_years"].fillna(0)
        + 0.25 * cost10["positive_return_years"].fillna(0)
        - 0.50 * (cost10["max_drawdown"].abs().fillna(0))
        - 0.50 * cost10["avg_daily_turnover"].fillna(0)
    )
    cost10 = cost10.sort_values(["positive_sharpe_years", "positive_return_years", "ann_sharpe_naive"], ascending=[False, False, False])
    top = cost10.head(10).copy()
    return cost10, top


def bootstrap_candidates(daily: pd.DataFrame, candidates: pd.DataFrame, n_boot: int, block_len: int) -> pd.DataFrame:
    rows = []
    candidate_keys = candidates[["horizon_days", "model", "rebalance_variant"]].drop_duplicates().head(10)
    for r in candidate_keys.itertuples(index=False):
        g = daily[
            daily["horizon_days"].eq(r.horizon_days)
            & daily["model"].eq(r.model)
            & daily["rebalance_variant"].eq(r.rebalance_variant)
        ].copy()
        if g.empty:
            continue
        ret_col = "net_abnormal_return_cost_10p0bps"
        stats = perf_stats(g[ret_col])
        boot = moving_block_bootstrap(g[ret_col].to_numpy(), n_boot=n_boot, block_len=block_len)
        rows.append(
            {
                "horizon_days": r.horizon_days,
                "model": r.model,
                "rebalance_variant": r.rebalance_variant,
                **stats,
                **boot,
            }
        )
    return pd.DataFrame(rows)


def compare_phase6_1_to_6_3(phase6_1_perf: pd.DataFrame, phase6_3_perf: pd.DataFrame) -> pd.DataFrame:
    if phase6_1_perf.empty:
        return pd.DataFrame()
    p61 = phase6_1_perf[
        phase6_1_perf["cost_bps"].eq(10.0)
        & phase6_1_perf["net_or_gross"].eq("net")
        & phase6_1_perf["model"].eq("lightgbm")
    ].copy()
    p63 = phase6_3_perf[
        phase6_3_perf["cost_bps"].eq(10.0)
        & phase6_3_perf["net_or_gross"].eq("net")
        & phase6_3_perf["model"].eq("lightgbm")
    ].copy()
    if p61.empty or p63.empty:
        return pd.DataFrame()
    # Phase 6.1 usually has simple daily strategy by horizon; map to daily variant.
    if "rebalance_variant" not in p61.columns:
        p61["rebalance_variant"] = "daily"
    keep = ["horizon_days", "model", "rebalance_variant", "ann_sharpe_naive", "cumulative_return", "max_drawdown", "tstat_hac_daily_mean"]
    old = p61[[c for c in keep if c in p61.columns]].rename(
        columns={
            "ann_sharpe_naive": "phase6_1_sharpe",
            "cumulative_return": "phase6_1_cumret",
            "max_drawdown": "phase6_1_mdd",
            "tstat_hac_daily_mean": "phase6_1_hac_t",
        }
    )
    new = p63[keep].rename(
        columns={
            "ann_sharpe_naive": "phase6_3_sharpe",
            "cumulative_return": "phase6_3_cumret",
            "max_drawdown": "phase6_3_mdd",
            "tstat_hac_daily_mean": "phase6_3_hac_t",
        }
    )
    cmp = old.merge(new, on=["horizon_days", "model", "rebalance_variant"], how="inner")
    if not cmp.empty:
        cmp["sharpe_delta"] = cmp["phase6_3_sharpe"] - cmp["phase6_1_sharpe"]
        cmp["cumret_delta"] = cmp["phase6_3_cumret"] - cmp["phase6_1_cumret"]
    return cmp


def copy_phase_figures(project_root: Path, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "source"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        project_root / "artifacts" / "backtest_turnover_full" / "figures",
        project_root / "artifacts" / "backtest_full" / "figures",
        project_root / "artifacts" / "modeling_full" / "figures",
    ]
    for src in sources:
        if not src.exists():
            continue
        for p in src.glob("*.png"):
            target = fig_dir / p.name
            try:
                target.write_bytes(p.read_bytes())
            except Exception as exc:
                print(f"[WARN] could not copy {p}: {exc}")


def make_figures(out_dir: Path, perf_rank: pd.DataFrame, yearly: pd.DataFrame, daily: pd.DataFrame, boot: pd.DataFrame, fi: pd.DataFrame) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, str]] = []

    # 10bps Sharpe ranking
    if not perf_rank.empty:
        plot = perf_rank.copy().sort_values("ann_sharpe_naive")
        labels = plot["horizon_days"].astype(str) + "d / " + plot["rebalance_variant"].astype(str)
        plt.figure(figsize=(11, 6.5))
        plt.barh(labels, plot["ann_sharpe_naive"])
        plt.xlabel("10 bps net abnormal Sharpe")
        plt.title("Phase 6.3 full backtest: low-turnover strategy ranking")
        plt.tight_layout()
        path = fig_dir / "phase7_0_phase6_3_10bps_net_sharpe_ranking.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "10bps_net_sharpe_ranking", "path": str(path)})

    # yearly heatmap-like grouped bars for top 5
    if not perf_rank.empty and not yearly.empty:
        top_keys = perf_rank[["horizon_days", "model", "rebalance_variant"]].head(5)
        parts = []
        for r in top_keys.itertuples(index=False):
            g = yearly[
                yearly["horizon_days"].eq(r.horizon_days)
                & yearly["model"].eq(r.model)
                & yearly["rebalance_variant"].eq(r.rebalance_variant)
                & yearly["cost_bps"].eq(10.0)
                & yearly["net_or_gross"].eq("net")
            ].copy()
            g["strategy"] = str(r.horizon_days) + "d / " + str(r.rebalance_variant)
            parts.append(g)
        yplot = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not yplot.empty:
            strategies = yplot["strategy"].drop_duplicates().tolist()
            years = sorted(yplot["calendar_year"].unique().tolist())
            x = np.arange(len(strategies))
            width = 0.25
            plt.figure(figsize=(12, 6))
            for i, year in enumerate(years):
                vals = []
                for strategy in strategies:
                    v = yplot[(yplot["strategy"].eq(strategy)) & (yplot["calendar_year"].eq(year))]["ann_sharpe_naive"]
                    vals.append(float(v.iloc[0]) if len(v) else np.nan)
                plt.bar(x + (i - (len(years)-1)/2)*width, vals, width=width, label=str(year))
            plt.axhline(0, linewidth=0.8)
            plt.xticks(x, strategies, rotation=35, ha="right")
            plt.ylabel("Annual Sharpe by calendar year")
            plt.title("Year-by-year robustness of top low-turnover candidates")
            plt.legend()
            plt.tight_layout()
            path = fig_dir / "phase7_0_yearly_sharpe_top_candidates.png"
            plt.savefig(path, dpi=180)
            plt.close()
            made.append({"figure": "yearly_sharpe_top_candidates", "path": str(path)})

    # cumulative 10bps top 4 candidates
    if not perf_rank.empty and not daily.empty:
        plt.figure(figsize=(12, 6.5))
        for r in perf_rank.head(4).itertuples(index=False):
            g = daily[
                daily["horizon_days"].eq(r.horizon_days)
                & daily["model"].eq(r.model)
                & daily["rebalance_variant"].eq(r.rebalance_variant)
            ].copy()
            if g.empty:
                continue
            g["date"] = pd.to_datetime(g["date"], errors="coerce")
            g = g.sort_values("date")
            curve = (1.0 + pd.to_numeric(g["net_abnormal_return_cost_10p0bps"], errors="coerce").fillna(0.0)).cumprod() - 1.0
            plt.plot(g["date"], curve, label=f"{int(r.horizon_days)}d / {r.rebalance_variant}")
        plt.axhline(0, linewidth=0.8)
        plt.ylabel("Cumulative 10 bps net abnormal return")
        plt.title("Cumulative returns of top low-turnover candidates")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "phase7_0_cumulative_top_candidates_10bps.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "cumulative_top_candidates_10bps", "path": str(path)})

    # bootstrap CI chart
    if not boot.empty:
        plot = boot.copy().sort_values("ann_sharpe")
        labels = plot["horizon_days"].astype(str) + "d / " + plot["rebalance_variant"].astype(str)
        lower = plot["ann_sharpe"] - plot["boot_sharpe_p05"]
        upper = plot["boot_sharpe_p95"] - plot["ann_sharpe"]
        plt.figure(figsize=(11, 6.5))
        plt.errorbar(plot["ann_sharpe"], labels, xerr=[lower, upper], fmt="o", capsize=4)
        plt.axvline(0, linewidth=0.8)
        plt.xlabel("Annualized Sharpe with moving-block bootstrap 90% interval")
        plt.title("Bootstrap uncertainty: 10 bps net abnormal returns")
        plt.tight_layout()
        path = fig_dir / "phase7_0_bootstrap_sharpe_ci.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "bootstrap_sharpe_ci", "path": str(path)})

    # Feature importance if present
    if fi is not None and not fi.empty:
        cols = fi.columns
        # Try common naming patterns.
        feature_col = "feature" if "feature" in cols else cols[0]
        imp_col = None
        for c in ["importance", "gain", "mean_importance", "split", "importance_mean"]:
            if c in cols:
                imp_col = c
                break
        if imp_col is None and len(cols) >= 2:
            numeric = [c for c in cols if pd.api.types.is_numeric_dtype(fi[c])]
            imp_col = numeric[0] if numeric else cols[1]
        if imp_col:
            fip = fi.copy()
            fip[imp_col] = pd.to_numeric(fip[imp_col], errors="coerce")
            fip = fip.dropna(subset=[imp_col]).sort_values(imp_col, ascending=False).head(20).sort_values(imp_col)
            if not fip.empty:
                plt.figure(figsize=(11, 7))
                plt.barh(fip[feature_col].astype(str), fip[imp_col])
                plt.xlabel(str(imp_col))
                plt.title("Top LightGBM features from Phase 5.1")
                plt.tight_layout()
                path = fig_dir / "phase7_0_top_lgbm_features.png"
                plt.savefig(path, dpi=180)
                plt.close()
                made.append({"figure": "top_lgbm_features", "path": str(path)})

    return made


def generate_decision_summary(best: pd.Series, bootstrap_best: pd.Series | None) -> dict[str, Any]:
    sharpe = float(best.get("ann_sharpe_naive", np.nan))
    hac_t = float(best.get("tstat_hac_daily_mean", np.nan))
    pos_years = int(best.get("positive_sharpe_years", 0)) if pd.notna(best.get("positive_sharpe_years", np.nan)) else 0
    max_dd = float(best.get("max_drawdown", np.nan))
    min_year = float(best.get("min_year_sharpe", np.nan))
    prob = None
    if bootstrap_best is not None and "prob_sharpe_positive" in bootstrap_best:
        prob = float(bootstrap_best.get("prob_sharpe_positive", np.nan))

    if sharpe >= 1.0 and hac_t >= 2.0 and pos_years >= 3 and min_year > 0:
        status = "production_ready_candidate"
    elif sharpe >= 0.5 and pos_years >= 3 and max_dd > -0.20:
        status = "research_candidate_not_yet_production"
    else:
        status = "weak_candidate_needs_more_robustness"

    return {
        "status": status,
        "selected_model": str(best.get("model")),
        "selected_horizon_days": int(best.get("horizon_days")),
        "selected_rebalance_variant": str(best.get("rebalance_variant")),
        "cost_bps": float(best.get("cost_bps")),
        "annualized_sharpe": sharpe,
        "cumulative_return": float(best.get("cumulative_return", np.nan)),
        "max_drawdown": max_dd,
        "hac_tstat": hac_t,
        "positive_sharpe_years": pos_years,
        "min_year_sharpe": min_year,
        "bootstrap_prob_sharpe_positive": prob,
        "recommendation": "Use as research/README headline, not live production, until bootstrap/placebo/sector-neutral and capacity diagnostics pass." if status != "production_ready_candidate" else "Candidate is strong enough for paper-style production specification and additional live-sim validation.",
    }


def html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "<p>No rows.</p>"
    show = df.head(max_rows) if max_rows else df
    return show.to_html(index=False, escape=True, classes="data")


def render_html(out: Path, summary: dict[str, Any], tables: dict[str, pd.DataFrame], figures: list[dict[str, str]], env: dict[str, Any]) -> None:
    decision = summary.get("decision", {})
    cards = []
    for label, val in [
        ("Decision", decision.get("status")),
        ("Selected model", decision.get("selected_model")),
        ("Horizon", str(decision.get("selected_horizon_days")) + "d"),
        ("Rebalance", decision.get("selected_rebalance_variant")),
        ("10 bps net Sharpe", decision.get("annualized_sharpe")),
        ("HAC t-stat", decision.get("hac_tstat")),
        ("Max drawdown", decision.get("max_drawdown")),
        ("Positive Sharpe years", decision.get("positive_sharpe_years")),
    ]:
        if isinstance(val, float):
            txt = f"{val:,.3f}"
        else:
            txt = html.escape(str(val))
        cards.append(f"<div class='card'><div class='kicker'>{html.escape(label)}</div><h3>{txt}</h3></div>")

    fig_html = []
    for fig in figures:
        rel = "figures/" + Path(fig["path"]).name
        fig_html.append(f"<div class='figure'><h3>{html.escape(fig['figure'].replace('_',' ').title())}</h3><img src='{html.escape(rel)}'></div>")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 7.0 Robustness and Production Decision</title>
<style>
:root {{ --bg:#07111f; --text:#eef6ff; --muted:#9fb7ce; --line:rgba(255,255,255,.14); }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#183b66,var(--bg) 42%); color:var(--text); }}
header {{ padding:46px 56px 28px; border-bottom:1px solid var(--line); }} h1 {{ margin:0; font-size:42px; letter-spacing:-.04em; }} .subtitle {{ color:var(--muted); font-size:17px; max-width:1100px; line-height:1.55; }}
main {{ padding:28px 56px 60px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:18px; margin:22px 0 36px; }} .card {{ background:linear-gradient(180deg,rgba(255,255,255,.075),rgba(255,255,255,.035)); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 18px 40px rgba(0,0,0,.18); }} .card h3 {{ margin:7px 0 0; font-size:25px; }} .kicker {{ text-transform:uppercase; font-size:11px; letter-spacing:.16em; color:var(--muted); }}
section {{ background:rgba(15,30,51,.78); border:1px solid var(--line); border-radius:22px; padding:24px; margin:22px 0; overflow:auto; }} table.data {{ width:100%; border-collapse:collapse; font-size:13px; }} table.data th {{ text-align:left; color:#d8eaff; background:rgba(255,255,255,.08); }} table.data th, table.data td {{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.09); vertical-align:top; }} .figure img {{ width:100%; max-width:1150px; border-radius:16px; border:1px solid var(--line); background:white; }} pre {{ white-space:pre-wrap; background:rgba(0,0,0,.28); border:1px solid var(--line); border-radius:14px; padding:16px; color:#dbecff; }}
</style></head><body><header><h1>Phase 7.0 Robustness and Production Decision</h1><p class="subtitle">Aggregate-only decision report combining Phase 5 forecast diagnostics, Phase 6.1 baseline backtests, and Phase 6.3 low-turnover portfolio results. No WRDS query and no protected vendor records are included.</p></header><main>
<div class="grid">{''.join(cards)}</div>
<section><h2>Decision summary</h2><pre>{html.escape(json.dumps(decision, indent=2, default=str))}</pre></section>
<section><h2>10 bps net Phase 6.3 ranking</h2>{html_table(tables.get('cost10_rank'), 20)}</section>
<section><h2>Moving-block bootstrap robustness</h2>{html_table(tables.get('bootstrap'), 20)}</section>
<section><h2>Yearly performance of top candidates</h2>{html_table(tables.get('yearly_top'), 50)}</section>
<section><h2>Phase 6.1 vs Phase 6.3 comparison</h2>{html_table(tables.get('phase6_comparison'), 20)}</section>
<section><h2>Validation diagnostics</h2>{html_table(tables.get('validation'), 20)}</section>
<section><h2>Figures</h2>{''.join(fig_html)}</section>
<section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main></body></html>"""
    out.write_text(doc)


def write_markdown_summary(path: Path, decision: dict[str, Any], best_rank: pd.DataFrame) -> None:
    lines = [
        "# Phase 7.0 robustness and production decision",
        "",
        f"Generated at UTC: {utc_now()}",
        "",
        "## Decision",
        "",
        f"- Status: **{decision.get('status')}**",
        f"- Selected candidate: **{decision.get('selected_model')} / {decision.get('selected_horizon_days')}d / {decision.get('selected_rebalance_variant')} / {decision.get('cost_bps')} bps**",
        f"- Annualized Sharpe: **{decision.get('annualized_sharpe'):.3f}**",
        f"- Cumulative return: **{decision.get('cumulative_return'):.2%}**",
        f"- Max drawdown: **{decision.get('max_drawdown'):.2%}**",
        f"- HAC t-stat: **{decision.get('hac_tstat'):.3f}**",
        f"- Positive Sharpe years: **{decision.get('positive_sharpe_years')}**",
        "",
        "## Interpretation",
        "",
        str(decision.get("recommendation")),
        "",
        "## Top 10 bps net candidates",
        "",
    ]
    if not best_rank.empty:
        cols = [
            "horizon_days", "model", "rebalance_variant", "ann_sharpe_naive", "cumulative_return",
            "max_drawdown", "tstat_hac_daily_mean", "avg_daily_turnover", "positive_sharpe_years", "min_year_sharpe",
        ]
        lines.append(best_rank[[c for c in cols if c in best_rank.columns]].head(10).to_markdown(index=False))
    lines.append("")
    lines.append("Data policy: this report is aggregate-only and includes no raw WRDS/vendor records.")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("<workspace>"))
    parser.add_argument("--project-root", type=Path, default=Path("<repo-root>"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--block-len", type=int, default=20)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    project_root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 7.0 robustness and production decision")
    print(f"UTC: {utc_now()}")
    print(f"Workspace: {workspace}")
    print(f"Project root: {project_root}")
    print(f"Output dir: {out_dir}")
    print("=" * 80)

    env = {
        "utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "workspace": str(workspace),
        "project_root": str(project_root),
        "thread_env": {k: os.environ.get(k) for k in ["PNA_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "CUDA_VISIBLE_DEVICES"]},
    }

    p63 = prepare_phase6_3(project_root)
    p61 = prepare_phase6_1(project_root)
    p51 = prepare_phase5_1(project_root)

    perf63 = p63["perf"]  # type: ignore[index]
    yearly63 = p63["yearly"]  # type: ignore[index]
    daily63 = p63["daily"]  # type: ignore[index]
    validation63 = p63["validation"]  # type: ignore[index]

    cost10_rank, top = select_candidates(perf63, yearly63, validation63)  # type: ignore[arg-type]
    if cost10_rank.empty:
        raise RuntimeError("No 10 bps net candidates found in Phase 6.3 performance summary.")

    boot = bootstrap_candidates(daily63, cost10_rank, args.bootstrap_samples, args.block_len)  # type: ignore[arg-type]
    cmp = compare_phase6_1_to_6_3(p61.get("perf", pd.DataFrame()), perf63)  # type: ignore[arg-type]

    best = cost10_rank.iloc[0]
    bbest = None
    if not boot.empty:
        bb = boot[
            boot["horizon_days"].eq(best["horizon_days"])
            & boot["model"].eq(best["model"])
            & boot["rebalance_variant"].eq(best["rebalance_variant"])
        ]
        if len(bb):
            bbest = bb.iloc[0]
    decision = generate_decision_summary(best, bbest)

    # Yearly top table
    top_keys = cost10_rank[["horizon_days", "model", "rebalance_variant"]].head(5)
    yearly_parts = []
    for r in top_keys.itertuples(index=False):
        g = yearly63[
            yearly63["horizon_days"].eq(r.horizon_days)
            & yearly63["model"].eq(r.model)
            & yearly63["rebalance_variant"].eq(r.rebalance_variant)
            & yearly63["cost_bps"].eq(10.0)
            & yearly63["net_or_gross"].eq("net")
        ].copy()
        yearly_parts.append(g)
    yearly_top = pd.concat(yearly_parts, ignore_index=True) if yearly_parts else pd.DataFrame()

    # Save tables.
    cost10_rank.to_csv(out_dir / "phase7_0_10bps_candidate_ranking.csv", index=False)
    boot.to_csv(out_dir / "phase7_0_bootstrap_robustness.csv", index=False)
    yearly_top.to_csv(out_dir / "phase7_0_yearly_top_candidates.csv", index=False)
    cmp.to_csv(out_dir / "phase7_0_phase6_1_vs_6_3_comparison.csv", index=False)
    validation63.to_csv(out_dir / "phase7_0_validation_diagnostics.csv", index=False)

    fi = p51.get("feature_importances", pd.DataFrame())  # type: ignore[assignment]
    figures = make_figures(out_dir, cost10_rank, yearly63, daily63, boot, fi)  # type: ignore[arg-type]
    copy_phase_figures(project_root, out_dir)

    summary = {
        "generated_at_utc": utc_now(),
        "validation_passed": True,
        "decision": decision,
        "n_phase6_3_candidates": int(len(cost10_rank)),
        "n_bootstrap_rows": int(len(boot)),
        "n_figures_created": int(len(figures)),
        "source_artifacts": {
            "phase5_1": str(project_root / "artifacts" / "modeling_full"),
            "phase6_1": str(project_root / "artifacts" / "backtest_full"),
            "phase6_3": str(project_root / "artifacts" / "backtest_turnover_full"),
        },
        "data_policy": "Aggregate-only report; protected Parquet and raw vendor records are not included.",
    }
    with (out_dir / "phase7_0_quality_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    tables = {
        "cost10_rank": cost10_rank[[c for c in ["horizon_days", "model", "rebalance_variant", "ann_sharpe_naive", "cumulative_return", "max_drawdown", "tstat_hac_daily_mean", "avg_daily_turnover", "positive_sharpe_years", "min_year_sharpe", "positive_return_years", "min_year_cumulative"] if c in cost10_rank.columns]],
        "bootstrap": boot,
        "yearly_top": yearly_top[[c for c in ["calendar_year", "horizon_days", "model", "rebalance_variant", "ann_sharpe_naive", "cumulative_return", "max_drawdown", "tstat_hac_daily_mean", "win_rate"] if c in yearly_top.columns]],
        "phase6_comparison": cmp,
        "validation": validation63,
    }
    render_html(out_dir / "phase7_0_robustness_decision_report.html", summary, tables, figures, env)
    write_markdown_summary(out_dir / "PHASE7_0_DECISION_MEMO.md", decision, cost10_rank)

    print("# Phase 7.0 robustness and production decision summary")
    print()
    print(f"- Generated at UTC: {summary['generated_at_utc']}")
    print(f"- Validation passed: {summary['validation_passed']}")
    print(f"- Decision status: {decision['status']}")
    print(f"- Selected candidate: {decision['selected_model']} / {decision['selected_horizon_days']}d / {decision['selected_rebalance_variant']} / {decision['cost_bps']}bps")
    print(f"- Annualized Sharpe: {decision['annualized_sharpe']:.3f}")
    print(f"- Cumulative return: {decision['cumulative_return']:.2%}")
    print(f"- Max drawdown: {decision['max_drawdown']:.2%}")
    print(f"- HAC t-stat: {decision['hac_tstat']:.3f}")
    print(f"- Report: {out_dir / 'phase7_0_robustness_decision_report.html'}")
    print()
    print("Data policy: aggregate-only report; protected Parquet files stay local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
