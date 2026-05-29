from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import importlib.metadata as md
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PNA_WORKSPACE_FIXED = Path("<workspace>")
PNA_REPO_FIXED = PNA_WORKSPACE_FIXED / "production-network-alpha"

DATE_COLS = {
    "signal_date",
    "label_base_date",
    "label_end_date_1d",
    "label_end_date_2d",
    "label_end_date_5d",
    "label_end_date_10d",
    "label_end_date_20d",
}
ID_COLS = {"target_gvkey", "target_permno"}
EXCLUDE_FROM_FEATURES_PREFIXES = ("fwd_ret_", "fwd_mkt_", "fwd_abret_", "label_end_date_", "no_lookahead_ok_")
EXCLUDE_FROM_FEATURES = {
    "target_gvkey",
    "target_permno",
    "signal_date",
    "signal_year",
    "label_base_date",
    "ret_adj",
    "vwretd",
    "abret_mkt",
    "signal_to_base_days",
}
CORE_SIGNAL_FEATURES = [
    "spillover_signed_shock",
    "spillover_abs_shock",
    "spillover_positive_shock",
    "spillover_negative_shock",
    "spillover_source_events",
    "spillover_source_stories",
    "n_shocked_neighbor_firms",
    "n_active_shock_edges",
    "mean_edge_weight",
    "max_edge_weight",
    "mean_relationship_age_days",
    "mean_supplier_customer_hhi",
    "dir_abs_customer_news_to_supplier",
    "dir_abs_supplier_news_to_customer",
    "dir_n_edges_customer_news_to_supplier",
    "dir_n_edges_supplier_news_to_customer",
    "dir_n_neighbors_customer_news_to_supplier",
    "dir_n_neighbors_supplier_news_to_customer",
    "dir_signed_customer_news_to_supplier",
    "dir_signed_supplier_news_to_customer",
    "own_news_events",
    "own_news_stories",
    "own_signed_news_shock",
    "own_abs_news_shock",
]
LAGGED_MARKET_FEATURES = [
    "ret_mom_21d",
    "ret_mom_126d",
    "idio_vol_63d",
    "dollar_vol_21d",
    "log_mktcap",
]
OPTIONAL_LIQUIDITY_SCALE_FEATURES = ["prc", "vol", "shrout", "mktcap"]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_fixed_workspace(project_root: Path) -> None:
    expected = PNA_REPO_FIXED.resolve()
    actual = project_root.resolve()
    if actual != expected:
        raise SystemExit(
            "Wrong project root.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}\n"
            "This guard prevents mixing outputs across allocations."
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_status() -> list[dict[str, str]]:
    pkgs = ["pandas", "numpy", "pyarrow", "duckdb", "polars", "lightgbm", "sklearn", "scipy", "matplotlib", "plotly"]
    rows: list[dict[str, str]] = []
    for pkg in pkgs:
        try:
            rows.append({"package": pkg, "version": md.version(pkg), "status": "ok"})
        except md.PackageNotFoundError:
            rows.append({"package": pkg, "version": "", "status": "missing"})
    return rows


def parse_horizons(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def list_partitions(matrix_dir: Path) -> list[Path]:
    parts = sorted(matrix_dir.glob("model_matrix_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No model matrix partitions found in {matrix_dir}")
    return parts


def year_from_path(path: Path) -> int:
    stem = path.stem
    return int(stem.split("_")[-1])


def finite_mask(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        mask &= ~np.isinf(values.to_numpy(dtype="float64", na_value=np.nan))
    return mask


def discover_feature_columns(columns: list[str]) -> dict[str, list[str]]:
    numeric_candidates = []
    for col in columns:
        if col in EXCLUDE_FROM_FEATURES:
            continue
        if col in DATE_COLS or col in ID_COLS:
            continue
        if any(col.startswith(prefix) for prefix in EXCLUDE_FROM_FEATURES_PREFIXES):
            continue
        if col == "pure_spillover_no_own_news":
            continue
        numeric_candidates.append(col)

    core = [c for c in CORE_SIGNAL_FEATURES if c in columns]
    market = [c for c in LAGGED_MARKET_FEATURES if c in columns]
    optional = [c for c in OPTIONAL_LIQUIDITY_SCALE_FEATURES if c in columns]
    default = []
    seen = set()
    for group in [core, market]:
        for c in group:
            if c not in seen:
                default.append(c)
                seen.add(c)

    return {
        "default_model_features": default,
        "core_signal_features": core,
        "lagged_market_features": market,
        "optional_liquidity_scale_features": optional,
        "all_numeric_candidate_features": numeric_candidates,
        "excluded_same_day_return_features": [c for c in ["ret_adj", "vwretd", "abret_mkt"] if c in columns],
        "identifier_columns": [c for c in ["target_gvkey", "target_permno"] if c in columns],
        "date_columns": [c for c in sorted(DATE_COLS) if c in columns],
    }


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def summarize_partition(path: Path, horizons: list[int], feature_cols: list[str], min_cov: float) -> tuple[list[dict[str, Any]], dict[int, pd.DataFrame]]:
    year = year_from_path(path)
    df = pd.read_parquet(path)
    df = coerce_dates(df)

    rows: list[dict[str, Any]] = []
    ready_by_horizon: dict[int, pd.DataFrame] = {}

    base_present = df["label_base_date"].notna() if "label_base_date" in df else pd.Series(False, index=df.index)
    pure = df["pure_spillover_no_own_news"].fillna(False).astype(bool) if "pure_spillover_no_own_news" in df else pd.Series(False, index=df.index)
    signal_to_base = pd.to_numeric(df.get("signal_to_base_days", pd.Series(np.nan, index=df.index)), errors="coerce")
    finite_features = finite_mask(df, feature_cols)

    for h in horizons:
        label_col = f"fwd_abret_{h}d"
        look_col = f"no_lookahead_ok_{h}d"
        end_col = f"label_end_date_{h}d"

        label_present = df[label_col].notna() if label_col in df else pd.Series(False, index=df.index)
        no_lookahead = df[look_col].fillna(False).astype(bool) if look_col in df else pd.Series(False, index=df.index)
        finite_label = finite_mask(df, [label_col]) if label_col in df else pd.Series(False, index=df.index)
        valid = base_present & label_present & no_lookahead & finite_label & finite_features

        coverage = float(label_present.mean()) if len(df) else 0.0
        valid_coverage = float(valid.mean()) if len(df) else 0.0
        pure_valid = valid & pure

        if end_col in df.columns and df[end_col].notna().any():
            max_end = pd.to_datetime(df[end_col], errors="coerce").max()
            min_end = pd.to_datetime(df[end_col], errors="coerce").min()
        else:
            min_end = pd.NaT
            max_end = pd.NaT

        if "label_base_date" in df.columns and df["label_base_date"].notna().any():
            min_base = df["label_base_date"].min()
            max_base = df["label_base_date"].max()
        else:
            min_base = pd.NaT
            max_base = pd.NaT

        rows.append(
            {
                "signal_year": year,
                "horizon_days": h,
                "rows": int(len(df)),
                "label_present_rows": int(label_present.sum()),
                "ready_rows": int(valid.sum()),
                "pure_ready_rows": int(pure_valid.sum()),
                "label_coverage": coverage,
                "ready_coverage": valid_coverage,
                "year_is_model_eligible": bool(valid_coverage >= min_cov),
                "base_date_present_rows": int(base_present.sum()),
                "min_label_base_date": None if pd.isna(min_base) else str(pd.Timestamp(min_base).date()),
                "max_label_base_date": None if pd.isna(max_base) else str(pd.Timestamp(max_base).date()),
                "min_label_end_date": None if pd.isna(min_end) else str(pd.Timestamp(min_end).date()),
                "max_label_end_date": None if pd.isna(max_end) else str(pd.Timestamp(max_end).date()),
                "signal_to_base_days_min": None if signal_to_base.dropna().empty else float(signal_to_base.min()),
                "signal_to_base_days_max": None if signal_to_base.dropna().empty else float(signal_to_base.max()),
                "infinite_feature_or_label_rows": int((~finite_features | ~finite_label).sum()),
                "no_lookahead_fail_rows": int((label_present & ~no_lookahead).sum()),
            }
        )

        if valid.any():
            ready = df.loc[valid].copy()
            ready["model_horizon_days"] = h
            ready["target_label"] = pd.to_numeric(ready[label_col], errors="coerce")
            ready["model_split_signal_year"] = year
            ready_by_horizon[h] = ready

    return rows, ready_by_horizon


def split_policy(coverage_df: pd.DataFrame, min_cov: float) -> dict[str, Any]:
    eligible_by_year = (
        coverage_df.groupby("signal_year")
        .agg(min_ready_coverage=("ready_coverage", "min"), min_ready_rows=("ready_rows", "min"))
        .reset_index()
    )
    eligible_years = sorted(eligible_by_year.loc[eligible_by_year["min_ready_coverage"] >= min_cov, "signal_year"].astype(int).tolist())
    dropped_years = sorted(set(coverage_df["signal_year"].astype(int)) - set(eligible_years))

    # Keep the originally intended train/validation windows, but automatically stop test at the last eligible year.
    last_eligible = max(eligible_years) if eligible_years else None
    train_years = [y for y in eligible_years if 2015 <= y <= 2019]
    validation_years = [y for y in eligible_years if 2020 <= y <= 2021]
    test_years = [y for y in eligible_years if 2022 <= y <= min(2025, last_eligible or 2021)]

    return {
        "min_year_label_coverage": min_cov,
        "eligible_years_all_horizons": eligible_years,
        "dropped_years": dropped_years,
        "recommended_train_years": train_years,
        "recommended_validation_years": validation_years,
        "recommended_test_years": test_years,
        "last_eligible_year": last_eligible,
        "note": "Years are eligible only if every configured horizon has ready_coverage >= threshold. This automatically drops 2025 when CRSP labels are unavailable.",
    }


def make_figures(out_dir: Path, coverage: pd.DataFrame, split: dict[str, Any]) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, str]] = []

    pivot = coverage.pivot(index="signal_year", columns="horizon_days", values="ready_coverage").sort_index()
    if not pivot.empty:
        plt.figure(figsize=(11, 5.8))
        for col in pivot.columns:
            plt.plot(pivot.index, pivot[col], marker="o", label=f"{col}d")
        plt.axhline(split["min_year_label_coverage"], linestyle="--", linewidth=1, label="eligibility threshold")
        plt.xlabel("Signal year")
        plt.ylabel("Ready label coverage")
        plt.title("Model-ready label coverage by year and horizon")
        plt.legend(ncol=3)
        plt.tight_layout()
        path = fig_dir / "phase4_2_ready_coverage_by_year_horizon.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "ready_coverage_by_year_horizon", "path": str(path)})

    rows = coverage.groupby("signal_year").agg(ready_rows=("ready_rows", "min"), pure_ready_rows=("pure_ready_rows", "min")).reset_index()
    if not rows.empty:
        plt.figure(figsize=(11, 5.8))
        plt.bar(rows["signal_year"], rows["ready_rows"], label="Ready rows, min across horizons")
        plt.bar(rows["signal_year"], rows["pure_ready_rows"], label="Pure ready rows, min across horizons")
        plt.xlabel("Signal year")
        plt.ylabel("Rows")
        plt.title("Conservative model-ready row counts by year")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "phase4_2_ready_rows_by_year.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "ready_rows_by_year", "path": str(path)})

    dropped = set(split.get("dropped_years", []))
    if dropped:
        plt.figure(figsize=(8, 4.8))
        vals = [1 if y in dropped else 0 for y in sorted(coverage["signal_year"].unique())]
        plt.bar(sorted(coverage["signal_year"].unique()), vals)
        plt.xlabel("Signal year")
        plt.ylabel("Dropped from modeling")
        plt.title("Years dropped because labels are unavailable or sparse")
        plt.tight_layout()
        path = fig_dir / "phase4_2_dropped_years.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "dropped_years", "path": str(path)})

    return made


def render_html(out_path: Path, summary: dict[str, Any], coverage: pd.DataFrame, split_df: pd.DataFrame, feature_df: pd.DataFrame, figures: list[dict[str, str]], env: dict[str, Any]) -> None:
    def table(df: pd.DataFrame) -> str:
        return df.to_html(index=False, escape=True, classes="data") if not df.empty else "<p>No rows.</p>"

    cards = []
    metrics = [
        ("Validation", "PASS" if summary["validation_passed"] else "FAIL"),
        ("Eligible years", len(summary["split_policy"]["eligible_years_all_horizons"])),
        ("Dropped years", ", ".join(map(str, summary["split_policy"]["dropped_years"])) or "none"),
        ("Train years", f"{min(summary['split_policy']['recommended_train_years'] or [0])}–{max(summary['split_policy']['recommended_train_years'] or [0])}"),
        ("Validation years", f"{min(summary['split_policy']['recommended_validation_years'] or [0])}–{max(summary['split_policy']['recommended_validation_years'] or [0])}"),
        ("Test years", f"{min(summary['split_policy']['recommended_test_years'] or [0])}–{max(summary['split_policy']['recommended_test_years'] or [0])}"),
    ]
    for label, val in metrics:
        cards.append(f"<div class='card'><div class='kicker'>{html.escape(str(label))}</div><h3>{html.escape(str(val))}</h3></div>")

    fig_html = []
    for fig in figures:
        rel = "figures/" + Path(fig["path"]).name
        fig_html.append(f"<div class='figure'><h3>{html.escape(fig['figure'].replace('_', ' ').title())}</h3><img src='{html.escape(rel)}'></div>")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 4.2 Model-Ready QA</title>
<style>
:root {{ --bg:#07111f; --text:#eef6ff; --muted:#9fb7ce; --line:rgba(255,255,255,.14); }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#183b66,var(--bg) 42%); color:var(--text); }}
header {{ padding:46px 56px 28px; border-bottom:1px solid var(--line); }} h1 {{ margin:0; font-size:42px; letter-spacing:-.04em; }} .subtitle {{ color:var(--muted); font-size:17px; max-width:1050px; line-height:1.55; }}
main {{ padding:28px 56px 60px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:18px; margin:22px 0 36px; }} .card {{ background:linear-gradient(180deg,rgba(255,255,255,.075),rgba(255,255,255,.035)); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 18px 40px rgba(0,0,0,.18); }} .card h3 {{ margin:7px 0 0; font-size:26px; }} .kicker {{ text-transform:uppercase; font-size:11px; letter-spacing:.16em; color:var(--muted); }}
section {{ background:rgba(15,30,51,.78); border:1px solid var(--line); border-radius:22px; padding:24px; margin:22px 0; overflow:auto; }} table.data {{ width:100%; border-collapse:collapse; font-size:13px; }} table.data th {{ text-align:left; color:#d8eaff; background:rgba(255,255,255,.08); }} table.data th, table.data td {{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.09); vertical-align:top; }} .figure img {{ width:100%; max-width:1100px; border-radius:16px; border:1px solid var(--line); background:white; }} pre {{ white-space:pre-wrap; background:rgba(0,0,0,.28); border:1px solid var(--line); border-radius:14px; padding:16px; color:#dbecff; }}
</style></head><body><header><h1>Phase 4.2 Model-Ready QA</h1><p class="subtitle">Converts the full Phase 4.1 matrix into protected model-ready partitions, automatically dropping signal years with insufficient forward-label coverage. This protects Phase 5 from accidentally training or testing on unlabeled 2025 rows.</p></header><main>
<div class="grid">{''.join(cards)}</div>
<section><h2>Coverage by year and horizon</h2>{table(coverage)}</section>
<section><h2>Recommended split policy</h2>{table(split_df)}</section>
<section><h2>Feature manifest summary</h2>{table(feature_df)}</section>
<section><h2>Figures</h2>{''.join(fig_html)}</section>
<section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main></body></html>"""
    out_path.write_text(doc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--horizons", default="1,2,5,10,20")
    parser.add_argument("--min-year-label-coverage", type=float, default=0.50)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    ensure_fixed_workspace(project_root)
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    horizons = parse_horizons(args.horizons)

    matrix_dir = project_root / "data" / "processed" / "model_matrix_full" / "model_matrix_by_year"
    ready_root = project_root / "data" / "processed" / "model_matrix_ready"
    ready_by_horizon_root = ready_root / "by_horizon"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ready_root.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("Phase 4.2 model-ready QA patch")
    print(f"UTC: {utc_now()}")
    print(f"Workspace: {PNA_WORKSPACE_FIXED}")
    print(f"Project root: {project_root}")
    print(f"Input matrix dir: {matrix_dir}")
    print(f"Protected model-ready dir: {ready_root}")
    print(f"Output dir: {out_dir}")
    print(f"Horizons: {horizons}")
    print("================================================================================")

    env = {
        "utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "thread_env": {k: os.environ.get(k) for k in ["PNA_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]},
        "packages": package_status(),
    }
    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    partitions = list_partitions(matrix_dir)
    print(f"[INFO] Found {len(partitions)} partitions")
    first_cols = list(pd.read_parquet(partitions[0], columns=None).columns)
    feature_manifest = discover_feature_columns(first_cols)
    feature_cols = feature_manifest["default_model_features"]
    if not feature_cols:
        raise RuntimeError("Default model feature list is empty; cannot validate finite features.")

    all_rows: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []

    for path in partitions:
        year = year_from_path(path)
        print(f"[READ] {path.name}")
        rows, ready_by_horizon = summarize_partition(path, horizons, feature_cols, args.min_year_label_coverage)
        all_rows.extend(rows)
        for h, ready in ready_by_horizon.items():
            hdir = ready_by_horizon_root / f"horizon_{h}d"
            pure = ready[ready["pure_spillover_no_own_news"].fillna(False).astype(bool)].copy() if "pure_spillover_no_own_news" in ready else ready.iloc[0:0].copy()
            out_path = hdir / f"model_ready_{year}_h{h}d.parquet"
            pure_path = hdir / f"model_ready_pure_{year}_h{h}d.parquet"
            write_parquet(ready, out_path)
            write_parquet(pure, pure_path)
            written.append({
                "signal_year": year,
                "horizon_days": h,
                "file": str(out_path),
                "rows": int(len(ready)),
                "pure_file": str(pure_path),
                "pure_rows": int(len(pure)),
            })
            print(f"[WRITE] year={year} h={h}d rows={len(ready):,} pure={len(pure):,}")

    coverage = pd.DataFrame(all_rows).sort_values(["signal_year", "horizon_days"])
    coverage.to_csv(out_dir / "phase4_2_label_coverage_by_year_horizon.csv", index=False)

    split = split_policy(coverage, args.min_year_label_coverage)
    split_df = pd.DataFrame([
        {"role": "train", "years": ",".join(map(str, split["recommended_train_years"]))},
        {"role": "validation", "years": ",".join(map(str, split["recommended_validation_years"]))},
        {"role": "test", "years": ",".join(map(str, split["recommended_test_years"]))},
        {"role": "dropped", "years": ",".join(map(str, split["dropped_years"]))},
    ])
    split_df.to_csv(out_dir / "phase4_2_recommended_split_policy.csv", index=False)

    written_df = pd.DataFrame(written).sort_values(["horizon_days", "signal_year"])
    written_df.to_csv(out_dir / "phase4_2_model_ready_inventory.csv", index=False)

    feature_manifest.update({
        "target_columns_by_horizon": {f"{h}d": f"fwd_abret_{h}d" for h in horizons},
        "canonical_target_column_in_ready_files": "target_label",
        "default_training_filter": "Use files under data/processed/model_matrix_ready/by_horizon/horizon_{h}d/ and exclude years listed in dropped_years.",
        "recommended_split_policy": split,
    })
    with (out_dir / "phase4_2_feature_manifest.json").open("w") as f:
        json.dump(feature_manifest, f, indent=2)

    feature_df = pd.DataFrame([
        {"feature_group": k, "n_columns": len(v), "columns_preview": ", ".join(map(str, v[:12]))}
        for k, v in feature_manifest.items()
        if isinstance(v, list)
    ])
    feature_df.to_csv(out_dir / "phase4_2_feature_manifest_summary.csv", index=False)

    # Protected inventory with hashes, local only.
    protected_inventory = []
    for p in sorted(ready_root.glob("**/*.parquet")):
        protected_inventory.append({"file": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p), "protected_local_only": True})
    pd.DataFrame(protected_inventory).to_csv(out_dir / "protected_local_phase4_2_inventory.csv", index=False)

    validation_checks = {
        "input_partitions_positive": len(partitions) > 0,
        "ready_files_written": len(written_df) > 0,
        "all_configured_horizons_written": set(written_df["horizon_days"].unique()) == set(horizons),
        "dropped_2025_if_unlabeled": (2025 not in split["eligible_years_all_horizons"]) if 2025 in set(coverage["signal_year"]) else True,
        "train_validation_test_nonempty": bool(split["recommended_train_years"] and split["recommended_validation_years"] and split["recommended_test_years"]),
        "no_ready_inf_rows": int(coverage["infinite_feature_or_label_rows"].sum()) == 0,
        "no_lookahead_fail_rows": int(coverage["no_lookahead_fail_rows"].sum()) == 0,
    }

    summary = {
        "generated_at_utc": utc_now(),
        "workspace": str(PNA_WORKSPACE_FIXED),
        "project_root": str(project_root),
        "input_matrix_dir": str(matrix_dir),
        "protected_model_ready_dir": str(ready_root),
        "horizons": horizons,
        "input_partitions": len(partitions),
        "written_ready_files": int(len(written_df)),
        "total_ready_rows": int(written_df["rows"].sum()) if len(written_df) else 0,
        "total_pure_ready_rows": int(written_df["pure_rows"].sum()) if len(written_df) else 0,
        "split_policy": split,
        "validation_checks": validation_checks,
        "validation_passed": bool(all(validation_checks.values())),
    }
    with (out_dir / "phase4_2_quality_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    figures = make_figures(out_dir, coverage, split)
    render_html(out_dir / "phase4_2_model_ready_report.html", summary, coverage, split_df, feature_df, figures, env)

    lines = [
        "# Phase 4.2 model-ready QA summary",
        "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Validation passed: {summary['validation_passed']}",
        f"- Input partitions: {summary['input_partitions']}",
        f"- Ready files written: {summary['written_ready_files']}",
        f"- Total ready rows across horizons: {summary['total_ready_rows']:,}",
        f"- Total pure ready rows across horizons: {summary['total_pure_ready_rows']:,}",
        f"- Eligible years: {split['eligible_years_all_horizons']}",
        f"- Dropped years: {split['dropped_years']}",
        f"- Recommended train years: {split['recommended_train_years']}",
        f"- Recommended validation years: {split['recommended_validation_years']}",
        f"- Recommended test years: {split['recommended_test_years']}",
        f"- Report: {out_dir / 'phase4_2_model_ready_report.html'}",
        "",
        "Data policy: protected model-ready Parquet files remain local and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE4_2_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    return 0 if summary["validation_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
