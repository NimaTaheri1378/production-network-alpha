
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def as_int64(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required local protected file not found: {path}")
    return pd.read_parquet(path)


def prepare_stocknames(stocknames: pd.DataFrame) -> pd.DataFrame:
    s = stocknames.copy()
    s.columns = [str(c).lower() for c in s.columns]

    required = ["permno", "namedt", "nameenddt", "ticker", "comnam", "shrcd", "exchcd", "siccd"]
    missing = [c for c in required if c not in s.columns]
    if missing:
        raise ValueError(f"crsp_stocknames cache missing columns: {missing}")

    s["permno"] = as_int64(s["permno"])
    s["namedt"] = pd.to_datetime(s["namedt"], errors="coerce")
    s["nameenddt"] = pd.to_datetime(s["nameenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))

    for c in ["shrcd", "exchcd", "siccd"]:
        s[c] = as_int64(s[c])

    for c in ["ticker", "comnam"]:
        s[c] = s[c].astype("string")

    s = s.dropna(subset=["permno", "namedt"]).sort_values(["permno", "namedt", "nameenddt"])
    s = s.drop_duplicates(["permno", "namedt", "nameenddt", "ticker", "shrcd", "exchcd"], keep="last")
    return s


def assign_stocknames_asof(
    edges: pd.DataFrame,
    stocknames: pd.DataFrame,
    side: str,
    max_stockname_stale_days: int,
) -> pd.DataFrame:
    perm_col = f"{side}_permno"
    if perm_col not in edges.columns:
        raise ValueError(f"Missing edge column: {perm_col}")

    drop_cols = [
        f"{side}_ticker",
        f"{side}_comnam",
        f"{side}_shrcd",
        f"{side}_exchcd",
        f"{side}_siccd",
        f"{side}_namedt",
        f"{side}_nameenddt",
        f"{side}_stockname_window_active",
        f"{side}_stockname_stale_days",
        f"{side}_stockname_usable",
    ]
    edges = edges.drop(columns=[c for c in drop_cols if c in edges.columns], errors="ignore").copy()

    left = edges[["edge_id", perm_col, "map_date"]].copy()
    left["permno"] = as_int64(left[perm_col])
    left["map_date"] = pd.to_datetime(left["map_date"], errors="coerce")
    left = left.dropna(subset=["permno", "map_date"]).sort_values(["permno", "map_date"])

    if left.empty:
        for c in drop_cols:
            if c not in edges.columns:
                edges[c] = pd.NA
        return edges

    cols = ["ticker", "comnam", "shrcd", "exchcd", "siccd", "namedt", "nameenddt"]
    pieces = []

    stock_groups = {
        int(k): g.reset_index(drop=True)
        for k, g in stocknames.groupby("permno", sort=False)
        if pd.notna(k)
    }

    for permno_value, group in left.groupby("permno", sort=False):
        permno_int = int(permno_value)
        names = stock_groups.get(permno_int)
        if names is None or names.empty:
            continue

        dates = names["namedt"].to_numpy(dtype="datetime64[ns]")
        query_dates = group["map_date"].to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(dates, query_dates, side="right") - 1
        valid = idx >= 0

        if not valid.any():
            continue

        group_valid = group.loc[valid, ["edge_id", "map_date"]].reset_index(drop=True)
        selected = names.iloc[idx[valid]][cols].reset_index(drop=True)

        out = group_valid[["edge_id"]].copy()
        for c in cols:
            out[f"{side}_{c}"] = selected[c].to_numpy()
        pieces.append(out)

    if not pieces:
        assigned = pd.DataFrame(columns=["edge_id"] + [f"{side}_{c}" for c in cols])
    else:
        assigned = pd.concat(pieces, ignore_index=True)

    edges = edges.merge(assigned, on="edge_id", how="left")

    edges[f"{side}_namedt"] = pd.to_datetime(edges[f"{side}_namedt"], errors="coerce")
    edges[f"{side}_nameenddt"] = pd.to_datetime(edges[f"{side}_nameenddt"], errors="coerce")
    edges["map_date"] = pd.to_datetime(edges["map_date"], errors="coerce")

    edges[f"{side}_stockname_window_active"] = (
        edges[f"{side}_namedt"].notna()
        & edges[f"{side}_nameenddt"].notna()
        & (edges[f"{side}_namedt"] <= edges["map_date"])
        & (edges[f"{side}_nameenddt"] >= edges["map_date"])
    )

    stale = (edges["map_date"] - edges[f"{side}_nameenddt"]).dt.days
    stale = stale.where(stale > 0, 0)
    edges[f"{side}_stockname_stale_days"] = stale

    edges[f"{side}_stockname_usable"] = (
        edges[f"{side}_namedt"].notna()
        & (
            edges[f"{side}_stockname_window_active"]
            | (edges[f"{side}_stockname_stale_days"].fillna(10**9) <= max_stockname_stale_days)
        )
    )

    for c in [f"{side}_shrcd", f"{side}_exchcd", f"{side}_siccd"]:
        edges[c] = as_int64(edges[c])

    return edges


def recompute_common_flags(edges: pd.DataFrame, max_stockname_stale_days: int) -> pd.DataFrame:
    e = edges.copy()

    if "common_share_major_exchange_pair" in e.columns and "common_share_major_exchange_pair_phase2_original" not in e.columns:
        e["common_share_major_exchange_pair_phase2_original"] = e["common_share_major_exchange_pair"].fillna(False).astype(bool)

    for c in ["supplier_permno", "customer_permno"]:
        if c in e.columns:
            e[c] = as_int64(e[c])

    both_permno = e["supplier_permno"].notna() & e["customer_permno"].notna()
    common = e["supplier_shrcd"].isin([10, 11]) & e["customer_shrcd"].isin([10, 11])
    major_exchange = e["supplier_exchcd"].isin([1, 2, 3]) & e["customer_exchcd"].isin([1, 2, 3])

    strict_stockname = e["supplier_stockname_window_active"].fillna(False) & e["customer_stockname_window_active"].fillna(False)
    usable_stockname = e["supplier_stockname_usable"].fillna(False) & e["customer_stockname_usable"].fillna(False)

    e["common_share_major_exchange_pair_strict_stockname_window"] = both_permno & common & major_exchange & strict_stockname
    e["common_share_major_exchange_pair"] = both_permno & common & major_exchange & usable_stockname

    e["stockname_patch_max_stale_days"] = max_stockname_stale_days
    e["stockname_patch_generated_at_utc"] = utc_now()
    return e


def make_waterfall(edges: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"step": "active_edges_from_phase2", "rows": int(len(edges))},
        {
            "step": "both_supplier_customer_permno_mapped",
            "rows": int((edges["supplier_permno"].notna() & edges["customer_permno"].notna()).sum()),
        },
    ]

    if "common_share_major_exchange_pair_phase2_original" in edges.columns:
        rows.append({
            "step": "phase2_original_common_major_exchange",
            "rows": int(edges["common_share_major_exchange_pair_phase2_original"].fillna(False).sum()),
        })

    rows.extend([
        {
            "step": "strict_stockname_window_common_major_exchange",
            "rows": int(edges["common_share_major_exchange_pair_strict_stockname_window"].fillna(False).sum()),
        },
        {
            "step": "asof_stockname_common_major_exchange",
            "rows": int(edges["common_share_major_exchange_pair"].fillna(False).sum()),
        },
    ])
    return pd.DataFrame(rows)


def edge_start_year_summary(edges: pd.DataFrame) -> pd.DataFrame:
    e = edges.copy()
    e["edge_year"] = pd.to_numeric(e["edge_year"], errors="coerce").astype("Int64")

    rows = []
    for year, g in e.dropna(subset=["edge_year"]).groupby("edge_year"):
        row = {
            "edge_start_year": int(year),
            "edges": int(len(g)),
            "both_permno_mapped": int((g["supplier_permno"].notna() & g["customer_permno"].notna()).sum()),
            "common_major_asof": int(g["common_share_major_exchange_pair"].fillna(False).sum()),
            "common_major_strict": int(g["common_share_major_exchange_pair_strict_stockname_window"].fillna(False).sum()),
            "supplier_stockname_usable": int(g["supplier_stockname_usable"].fillna(False).sum()),
            "customer_stockname_usable": int(g["customer_stockname_usable"].fillna(False).sum()),
            "supplier_stockname_window_active": int(g["supplier_stockname_window_active"].fillna(False).sum()),
            "customer_stockname_window_active": int(g["customer_stockname_window_active"].fillna(False).sum()),
        }
        if "common_share_major_exchange_pair_phase2_original" in g.columns:
            row["common_major_phase2_original"] = int(g["common_share_major_exchange_pair_phase2_original"].fillna(False).sum())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("edge_start_year")


def active_snapshot_summary(edges: pd.DataFrame, sample_start: str, sample_end: str) -> pd.DataFrame:
    start = pd.Timestamp(sample_start)
    end = pd.Timestamp(sample_end)

    e = edges.copy()
    e["edge_start_date"] = pd.to_datetime(e["edge_start_date"], errors="coerce")
    e["edge_end_date"] = pd.to_datetime(e["edge_end_date"], errors="coerce")

    rows = []
    for year in range(start.year, end.year + 1):
        for label, raw_date in [
            ("midyear", pd.Timestamp(year=year, month=6, day=30)),
            ("yearend", pd.Timestamp(year=year, month=12, day=31)),
        ]:
            snap = min(max(raw_date, start), end)
            active = e[(e["edge_start_date"] <= snap) & (e["edge_end_date"] >= snap)].copy()

            row = {
                "year": year,
                "snapshot": label,
                "snapshot_date": snap.date().isoformat(),
                "active_edges": int(len(active)),
                "both_permno_mapped": int((active["supplier_permno"].notna() & active["customer_permno"].notna()).sum()),
                "common_major_asof": int(active["common_share_major_exchange_pair"].fillna(False).sum()),
                "common_major_strict": int(active["common_share_major_exchange_pair_strict_stockname_window"].fillna(False).sum()),
                "unique_supplier_gvkeys": int(active["supplier_gvkey"].nunique(dropna=True)),
                "unique_customer_gvkeys": int(active["customer_gvkey"].nunique(dropna=True)),
                "unique_supplier_permnos": int(active["supplier_permno"].nunique(dropna=True)),
                "unique_customer_permnos": int(active["customer_permno"].nunique(dropna=True)),
            }
            if "common_share_major_exchange_pair_phase2_original" in active.columns:
                row["common_major_phase2_original"] = int(active["common_share_major_exchange_pair_phase2_original"].fillna(False).sum())
            rows.append(row)

    return pd.DataFrame(rows)


def stockname_diagnostics(edges: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for side in ["supplier", "customer"]:
        stale = pd.to_numeric(edges[f"{side}_stockname_stale_days"], errors="coerce")
        out[f"{side}_stockname_rows_assigned"] = int(edges[f"{side}_namedt"].notna().sum())
        out[f"{side}_stockname_window_active"] = int(edges[f"{side}_stockname_window_active"].fillna(False).sum())
        out[f"{side}_stockname_usable"] = int(edges[f"{side}_stockname_usable"].fillna(False).sum())
        out[f"{side}_stockname_stale_positive"] = int((stale.fillna(0) > 0).sum())
        out[f"{side}_stockname_stale_p50"] = None if stale.dropna().empty else float(stale.quantile(0.50))
        out[f"{side}_stockname_stale_p95"] = None if stale.dropna().empty else float(stale.quantile(0.95))
        out[f"{side}_stockname_stale_max"] = None if stale.dropna().empty else float(stale.max())

    out["common_major_asof"] = int(edges["common_share_major_exchange_pair"].fillna(False).sum())
    out["common_major_strict"] = int(edges["common_share_major_exchange_pair_strict_stockname_window"].fillna(False).sum())
    if "common_share_major_exchange_pair_phase2_original" in edges.columns:
        out["common_major_phase2_original"] = int(edges["common_share_major_exchange_pair_phase2_original"].fillna(False).sum())
    return out


def make_figures(
    out_dir: Path,
    waterfall: pd.DataFrame,
    start_year: pd.DataFrame,
    snapshots: pd.DataFrame,
    edges: pd.DataFrame,
) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, str]] = []

    plt.figure(figsize=(10.5, 5.8))
    plt.barh(waterfall["step"], waterfall["rows"])
    plt.gca().invert_yaxis()
    plt.xlabel("Edges")
    plt.title("Phase 2.1 stockname QA waterfall")
    plt.tight_layout()
    path = fig_dir / "phase2_1_stockname_waterfall.png"
    plt.savefig(path, dpi=180)
    plt.close()
    made.append({"figure": "stockname_waterfall", "path": str(path)})

    if not start_year.empty:
        plt.figure(figsize=(11.5, 6.0))
        x = start_year["edge_start_year"]
        if "common_major_phase2_original" in start_year.columns:
            plt.plot(x, start_year["common_major_phase2_original"], marker="o", label="Phase 2 original")
        plt.plot(x, start_year["common_major_strict"], marker="o", label="Strict active stockname window")
        plt.plot(x, start_year["common_major_asof"], marker="o", label="As-of stockname, stale-capped")
        plt.xlabel("Edge start year")
        plt.ylabel("Common-share major-exchange edges")
        plt.title("Common-share filter before and after as-of stockname patch")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "phase2_1_common_filter_by_edge_start_year.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "common_filter_by_edge_start_year", "path": str(path)})

    if not snapshots.empty:
        yearend = snapshots[snapshots["snapshot"].eq("yearend")].copy()
        plt.figure(figsize=(11.5, 6.0))
        plt.plot(yearend["year"], yearend["active_edges"], marker="o", label="All active edges")
        plt.plot(yearend["year"], yearend["both_permno_mapped"], marker="o", label="Both sides PERMNO-mapped")
        plt.plot(yearend["year"], yearend["common_major_asof"], marker="o", label="Common major, as-of")
        plt.xlabel("Snapshot year")
        plt.ylabel("Active edges at year-end")
        plt.title("Point-in-time production-network size by active year-end snapshot")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "phase2_1_yearend_active_graph_size.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "yearend_active_graph_size", "path": str(path)})

    stale_cols = ["supplier_stockname_stale_days", "customer_stockname_stale_days"]
    vals = []
    for c in stale_cols:
        if c in edges.columns:
            vals.append(pd.to_numeric(edges[c], errors="coerce"))
    if vals:
        stale = pd.concat(vals, ignore_index=True).dropna()
        stale = stale[stale > 0]
        if not stale.empty:
            plt.figure(figsize=(10.5, 5.8))
            plt.hist(stale.clip(upper=stale.quantile(0.99)), bins=50)
            plt.xlabel("Positive stockname stale days, clipped at p99")
            plt.ylabel("Supplier/customer edge-side observations")
            plt.title("Stockname as-of fallback age distribution")
            plt.tight_layout()
            path = fig_dir / "phase2_1_stockname_stale_days.png"
            plt.savefig(path, dpi=180)
            plt.close()
            made.append({"figure": "stockname_stale_days", "path": str(path)})

    return made


def render_report(
    out_path: Path,
    summary: dict[str, Any],
    waterfall: pd.DataFrame,
    start_year: pd.DataFrame,
    snapshots: pd.DataFrame,
    diagnostics: dict[str, Any],
    figures: list[dict[str, str]],
    env: dict[str, Any],
) -> None:
    cards = []
    metrics = [
        ("Phase 2 original common-major", diagnostics.get("common_major_phase2_original")),
        ("Strict common-major", diagnostics.get("common_major_strict")),
        ("As-of common-major", diagnostics.get("common_major_asof")),
        ("Supplier stockname usable", diagnostics.get("supplier_stockname_usable")),
        ("Customer stockname usable", diagnostics.get("customer_stockname_usable")),
        ("Max stale cap, days", summary.get("max_stockname_stale_days")),
    ]

    for label, value in metrics:
        if value is None:
            val = "n/a"
        elif isinstance(value, float):
            val = f"{value:,.2f}"
        elif isinstance(value, int):
            val = f"{value:,}"
        else:
            val = html.escape(str(value))
        cards.append(f"""
        <div class="card">
          <div class="kicker">{html.escape(label)}</div>
          <h3>{val}</h3>
        </div>
        """)

    fig_html = []
    for fig in figures:
        rel = "figures/" + Path(fig["path"]).name
        fig_html.append(f"<div class='figure'><h3>{html.escape(fig['figure'].replace('_', ' ').title())}</h3><img src='{html.escape(rel)}'></div>")

    waterfall_html = waterfall.to_html(index=False, escape=True, classes="data")
    start_html = start_year.to_html(index=False, escape=True, classes="data") if not start_year.empty else "<p>No start-year summary.</p>"
    snapshot_html = snapshots.to_html(index=False, escape=True, classes="data") if not snapshots.empty else "<p>No snapshot summary.</p>"
    diag_html = pd.DataFrame([diagnostics]).to_html(index=False, escape=True, classes="data")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Production Network Alpha — Phase 2.1 Graph QA Patch</title>
<style>
:root {{
  --bg: #07111f;
  --panel: #0f1e33;
  --text: #eef6ff;
  --muted: #9fb7ce;
  --line: rgba(255,255,255,.14);
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif; background: radial-gradient(circle at top left, #183b66, var(--bg) 42%); color: var(--text); }}
header {{ padding: 46px 56px 28px; border-bottom: 1px solid var(--line); }}
h1 {{ margin: 0; font-size: 42px; letter-spacing: -.04em; }}
.subtitle {{ color: var(--muted); font-size: 17px; max-width: 1080px; line-height: 1.55; }}
main {{ padding: 28px 56px 60px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; margin: 22px 0 36px; }}
.card {{ background: linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.035)); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 18px 40px rgba(0,0,0,.18); }}
.card h3 {{ margin: 7px 0 0; font-size: 28px; letter-spacing: -.03em; }}
.kicker {{ text-transform: uppercase; font-size: 11px; letter-spacing: .16em; color: var(--muted); }}
section {{ background: rgba(15,30,51,.78); border: 1px solid var(--line); border-radius: 22px; padding: 24px; margin: 22px 0; overflow: auto; }}
h2 {{ margin-top: 0; letter-spacing: -.02em; }}
table.data {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
table.data th {{ text-align: left; color: #d8eaff; background: rgba(255,255,255,.08); }}
table.data th, table.data td {{ padding: 9px 10px; border-bottom: 1px solid rgba(255,255,255,.09); vertical-align: top; }}
.figure {{ margin: 18px 0 30px; }}
.figure img {{ width: 100%; max-width: 1100px; border-radius: 16px; border: 1px solid var(--line); background: white; }}
pre {{ white-space: pre-wrap; background: rgba(0,0,0,.28); border: 1px solid var(--line); border-radius: 14px; padding: 16px; color: #dbecff; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.pill {{ border: 1px solid var(--line); background: rgba(255,255,255,.07); border-radius: 999px; padding: 8px 12px; color: var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>Phase 2.1 Graph QA Patch</h1>
  <p class="subtitle">Safer CRSP stock-name as-of assignment, strict/as-of common-share filters, and corrected active-year graph visuals. This report is aggregate-only and does not contain raw WRDS/vendor records.</p>
  <div class="meta">
    <span class="pill">Generated: {html.escape(summary.get("generated_at_utc", ""))}</span>
    <span class="pill">Max stockname stale cap: {html.escape(str(summary.get("max_stockname_stale_days")))} days</span>
    <span class="pill">Output: corrected graph backbone</span>
  </div>
</header>
<main>
  <div class="grid">{''.join(cards)}</div>
  <section><h2>Patch waterfall</h2>{waterfall_html}</section>
  <section><h2>Stockname diagnostics</h2>{diag_html}</section>
  <section><h2>Edge-start-year common-share filter comparison</h2>{start_html}</section>
  <section><h2>Active graph snapshot summary</h2>{snapshot_html}</section>
  <section><h2>Figures</h2>{''.join(fig_html)}</section>
  <section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main>
</body>
</html>
"""
    out_path.write_text(doc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--sample-start", default="2015-01-01")
    parser.add_argument("--sample-end", default="2025-12-31")
    parser.add_argument("--max-stockname-stale-days", type=int, default=550)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()

    processed_dir = project_root / "data" / "processed" / "graph_backbone"
    raw_dir = project_root / "data" / "raw" / "wrds" / "phase2_graph_backbone"

    edges_path = processed_dir / "edges_supplier_customer_all.parquet"
    common_path = processed_dir / "edges_supplier_customer_common_us.parquet"
    stocknames_path = raw_dir / "crsp_stocknames.parquet"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("Phase 2.1 graph QA patch")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {project_root}")
    print(f"Output dir: {out_dir}")
    print(f"Log dir: {log_dir}")
    print(f"Input edges: {edges_path}")
    print(f"Input stocknames cache: {stocknames_path}")
    print("================================================================================")

    env = {
        "utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cpu_count": os.cpu_count(),
        "thread_env": {
            k: os.environ.get(k)
            for k in ["PNA_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]
        },
    }

    edges = read_required(edges_path)
    stocknames = prepare_stocknames(read_required(stocknames_path))

    print(f"[INFO] Loaded edges: {len(edges):,} rows, {len(edges.columns):,} columns")
    print(f"[INFO] Loaded CRSP stocknames: {len(stocknames):,} rows")

    for date_col in ["map_date", "edge_start_date", "edge_end_date", "known_date", "disclosure_date"]:
        if date_col in edges.columns:
            edges[date_col] = pd.to_datetime(edges[date_col], errors="coerce")

    edges = assign_stocknames_asof(edges, stocknames, "supplier", args.max_stockname_stale_days)
    edges = assign_stocknames_asof(edges, stocknames, "customer", args.max_stockname_stale_days)
    edges = recompute_common_flags(edges, args.max_stockname_stale_days)

    waterfall = make_waterfall(edges)
    start_year = edge_start_year_summary(edges)
    snapshots = active_snapshot_summary(edges, args.sample_start, args.sample_end)
    diagnostics = stockname_diagnostics(edges)

    v2_all = processed_dir / "edges_supplier_customer_all_v2.parquet"
    v2_common = processed_dir / "edges_supplier_customer_common_us_v2.parquet"

    write_parquet(edges, v2_all)
    write_parquet(edges[edges["common_share_major_exchange_pair"].fillna(False)].copy(), v2_common)

    backup_all = processed_dir / "edges_supplier_customer_all_phase2_original.parquet"
    backup_common = processed_dir / "edges_supplier_customer_common_us_phase2_original.parquet"

    if edges_path.exists() and not backup_all.exists():
        shutil.copy2(edges_path, backup_all)
    if common_path.exists() and not backup_common.exists():
        shutil.copy2(common_path, backup_common)

    shutil.copy2(v2_all, edges_path)
    shutil.copy2(v2_common, common_path)

    waterfall.to_csv(out_dir / "phase2_1_patch_waterfall.csv", index=False)
    start_year.to_csv(out_dir / "phase2_1_edge_start_year_summary.csv", index=False)
    snapshots.to_csv(out_dir / "phase2_1_active_snapshot_summary.csv", index=False)
    pd.DataFrame([diagnostics]).to_csv(out_dir / "phase2_1_stockname_diagnostics.csv", index=False)

    summary = {
        "generated_at_utc": utc_now(),
        "sample_start": args.sample_start,
        "sample_end": args.sample_end,
        "max_stockname_stale_days": args.max_stockname_stale_days,
        "input_edges": str(edges_path),
        "input_stocknames": str(stocknames_path),
        "v2_all_edges": str(v2_all),
        "v2_common_edges": str(v2_common),
        "canonical_all_edges_overwritten": str(edges_path),
        "canonical_common_edges_overwritten": str(common_path),
        "diagnostics": diagnostics,
    }

    with (out_dir / "phase2_1_quality_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    figures = make_figures(out_dir, waterfall, start_year, snapshots, edges)
    render_report(
        out_path=out_dir / "phase2_1_graph_qa_report.html",
        summary=summary,
        waterfall=waterfall,
        start_year=start_year,
        snapshots=snapshots,
        diagnostics=diagnostics,
        figures=figures,
        env=env,
    )

    processed_inventory = []
    for path in sorted(processed_dir.glob("*.parquet")):
        processed_inventory.append({
            "file": str(path),
            "size_bytes": path.stat().st_size,
        })
    pd.DataFrame(processed_inventory).to_csv(out_dir / "protected_processed_inventory_after_patch.csv", index=False)

    summary_lines = [
        "# Phase 2.1 graph QA patch summary",
        "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Phase 2 original common-major edges: {diagnostics.get('common_major_phase2_original')}",
        f"- Strict stockname-window common-major edges: {diagnostics.get('common_major_strict')}",
        f"- As-of stale-capped common-major edges: {diagnostics.get('common_major_asof')}",
        f"- Max stockname stale cap: {args.max_stockname_stale_days} days",
        f"- Corrected all-edge file: {v2_all}",
        f"- Corrected common-edge file: {v2_common}",
        f"- Canonical common-edge file overwritten for next phase: {common_path}",
        f"- Report: {out_dir / 'phase2_1_graph_qa_report.html'}",
        "",
        "Data policy: protected Parquet files remain local and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE2_1_SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
