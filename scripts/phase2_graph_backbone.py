
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SAMPLE_START_DEFAULT = "2015-01-01"
SAMPLE_END_DEFAULT = "2025-12-31"
DEFAULT_REPORTING_LAG_DAYS = 90
DEFAULT_MAX_EDGE_DAYS = 550


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def mask_user(username: str | None) -> str | None:
    if not username:
        return None
    if len(username) <= 3:
        return username[0] + "***"
    return username[:2] + "***" + username[-1]


def detect_wrds_username() -> tuple[str | None, str]:
    env_user = os.environ.get("WRDS_USERNAME")
    if env_user:
        return env_user, "WRDS_USERNAME"

    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        try:
            for line in pgpass.read_text(errors="ignore").splitlines():
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":")
                if len(parts) >= 5 and "wrds" in line.lower():
                    return parts[-2], "~/.pgpass"
            for line in pgpass.read_text(errors="ignore").splitlines():
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":")
                if len(parts) >= 5:
                    return parts[-2], "~/.pgpass"
        except Exception:
            return None, "~/.pgpass unreadable"
    return None, "not found"


def run_cmd(cmd: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}


def package_status() -> list[dict[str, str]]:
    wanted = [
        "pandas", "numpy", "pyarrow", "duckdb", "polars", "wrds", "sqlalchemy",
        "psycopg2", "matplotlib", "plotly", "pytest", "dash", "dash-cytoscape",
    ]
    rows = []
    for pkg in wanted:
        try:
            rows.append({"package": pkg, "version": md.version(pkg), "status": "ok"})
        except md.PackageNotFoundError:
            rows.append({"package": pkg, "version": "", "status": "missing"})
    return rows


def clean_gvkey(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.mask(out.isin(["", "nan", "NaN", "None", "<NA>"]))
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.zfill(6)
    return out


def safe_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_sql_cached(db, name: str, sql: str, cache_path: Path, force_refresh: bool) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force_refresh:
        print(f"[CACHE] {name}: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[WRDS] {name}: querying...")
    t0 = time.time()
    df = db.raw_sql(sql)
    df.columns = [str(c).lower() for c in df.columns]
    write_parquet(df, cache_path)
    print(f"[WRDS] {name}: rows={len(df):,}, cols={len(df.columns):,}, elapsed={time.time() - t0:.1f}s")
    return df


def connect_wrds():
    import wrds

    user, source = detect_wrds_username()
    print(f"[INFO] WRDS username source: {source}; user={mask_user(user)}")
    try:
        db = wrds.Connection(wrds_username=user, verbose=False) if user else wrds.Connection(verbose=False)
    except TypeError:
        db = wrds.Connection()

    try:
        db.raw_sql("set statement_timeout to '1800000ms'")
    except Exception:
        pass

    return db, user, source


def extract_inputs(db, raw_dir: Path, force_refresh: bool) -> dict[str, pd.DataFrame]:
    queries = {
        "supply_seglink": """
            select
                gvkey, cid, cnms, ctype, salecs, sid, srcdate,
                cgvkey, scusip, stic, conm, ccusip, ctic, cconm
            from wrdsapps_link_supplychain.seglink
            where srcdate is not null
        """,
        "segments_customer": """
            select
                gvkey, cid, cnms, ctype, gareac, gareat, salecs, sid, stype, srcdate
            from comp_segments_hist_daily.wrds_seg_customer
            where srcdate is not null
        """,
        "ccm_linktable": """
            select
                gvkey, linkprim, liid, linktype, lpermno, lpermco, usedflag, linkdt, linkenddt
            from crsp_a_ccm.ccmxpf_linktable
            where lpermno is not null
        """,
        "comp_company": """
            select gvkey, conm, cik, sic, naics, gsector, gind, gsubind
            from comp_na_daily_all.company
        """,
        "crsp_stocknames": """
            select permno, permco, namedt, nameenddt, ticker, comnam, shrcd, exchcd, siccd
            from crsp_a_stock.stocknames
            where permno is not null
        """,
    }

    out = {}
    for name, sql in queries.items():
        out[name] = read_sql_cached(db, name, sql, raw_dir / f"{name}.parquet", force_refresh)

    return out


def prepare_ccm(ccm: pd.DataFrame) -> pd.DataFrame:
    c = ccm.copy()
    c["gvkey"] = clean_gvkey(c["gvkey"])
    c["lpermno"] = safe_int(c["lpermno"])
    c["lpermco"] = safe_int(c["lpermco"])
    c["linkdt"] = pd.to_datetime(c["linkdt"], errors="coerce")
    c["linkenddt"] = pd.to_datetime(c["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    c["usedflag"] = pd.to_numeric(c.get("usedflag", 0), errors="coerce").fillna(0).astype(int)
    c["linktype"] = c["linktype"].astype("string")
    c["linkprim"] = c["linkprim"].astype("string")

    linktype_rank = {"LC": 0, "LU": 1, "LS": 2, "LN": 3, "LX": 4, "LD": 5}
    linkprim_rank = {"P": 0, "C": 1, "J": 2, "N": 3}

    c["linktype_rank"] = c["linktype"].map(linktype_rank).fillna(9).astype(int)
    c["linkprim_rank"] = c["linkprim"].map(linkprim_rank).fillna(9).astype(int)
    c = c.dropna(subset=["gvkey", "lpermno", "linkdt"])
    return c


def prepare_stocknames(stocknames: pd.DataFrame) -> pd.DataFrame:
    s = stocknames.copy()
    s["permno"] = safe_int(s["permno"])
    s["permco"] = safe_int(s["permco"])
    s["namedt"] = pd.to_datetime(s["namedt"], errors="coerce")
    s["nameenddt"] = pd.to_datetime(s["nameenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))

    for col in ["ticker", "comnam"]:
        if col in s.columns:
            s[col] = s[col].astype("string")

    for col in ["shrcd", "exchcd", "siccd"]:
        if col in s.columns:
            s[col] = safe_int(s[col])

    return s.dropna(subset=["permno", "namedt"])


def assign_ccm(edges: pd.DataFrame, ccm: pd.DataFrame, side: str) -> pd.DataFrame:
    gvkey_col = f"{side}_gvkey"
    date_col = "map_date"

    tmp = edges[["edge_id", gvkey_col, date_col]].rename(columns={gvkey_col: "gvkey"})
    tmp["gvkey"] = clean_gvkey(tmp["gvkey"])

    merged = tmp.merge(ccm, on="gvkey", how="left")
    mask = (
        merged["lpermno"].notna()
        & (merged["linkdt"] <= merged[date_col])
        & (merged["linkenddt"] >= merged[date_col])
    )

    best = merged.loc[mask].copy()
    if best.empty:
        return edges

    best = best.sort_values(
        ["edge_id", "usedflag", "linktype_rank", "linkprim_rank", "linkdt"],
        ascending=[True, False, True, True, False],
    ).drop_duplicates("edge_id", keep="first")

    keep = best[["edge_id", "lpermno", "lpermco", "linktype", "linkprim", "linkdt", "linkenddt"]].rename(
        columns={
            "lpermno": f"{side}_permno",
            "lpermco": f"{side}_permco",
            "linktype": f"{side}_linktype",
            "linkprim": f"{side}_linkprim",
            "linkdt": f"{side}_linkdt",
            "linkenddt": f"{side}_linkenddt",
        }
    )

    return edges.merge(keep, on="edge_id", how="left")


def assign_stocknames(edges: pd.DataFrame, stocknames: pd.DataFrame, side: str) -> pd.DataFrame:
    permno_col = f"{side}_permno"
    if permno_col not in edges.columns:
        return edges

    tmp = edges[["edge_id", permno_col, "map_date"]].rename(columns={permno_col: "permno"})
    merged = tmp.merge(stocknames, on="permno", how="left")
    mask = (
        merged["permno"].notna()
        & (merged["namedt"] <= merged["map_date"])
        & (merged["nameenddt"] >= merged["map_date"])
    )

    best = merged.loc[mask].copy()
    if best.empty:
        return edges

    best = best.sort_values(["edge_id", "namedt"], ascending=[True, False]).drop_duplicates("edge_id", keep="first")
    keep_cols = ["edge_id", "ticker", "comnam", "shrcd", "exchcd", "siccd", "namedt", "nameenddt"]
    keep = best[keep_cols].rename(columns={c: f"{side}_{c}" for c in keep_cols if c != "edge_id"})

    return edges.merge(keep, on="edge_id", how="left")


def build_edges(
    seglink: pd.DataFrame,
    ccm: pd.DataFrame,
    stocknames: pd.DataFrame,
    company: pd.DataFrame,
    sample_start: str,
    sample_end: str,
    reporting_lag_days: int,
    max_edge_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = pd.Timestamp(sample_start)
    end = pd.Timestamp(sample_end)

    raw = seglink.copy()
    raw["supplier_gvkey"] = clean_gvkey(raw["gvkey"])
    raw["customer_gvkey"] = clean_gvkey(raw["cgvkey"])
    raw["disclosure_date"] = pd.to_datetime(raw["srcdate"], errors="coerce")
    raw["revenue_to_customer"] = pd.to_numeric(raw["salecs"], errors="coerce")

    waterfall: list[dict[str, Any]] = []
    waterfall.append({"step": "raw_seglink_rows", "rows": int(len(raw))})

    work = raw.dropna(subset=["supplier_gvkey", "customer_gvkey", "disclosure_date"]).copy()
    waterfall.append({"step": "non_null_supplier_customer_date", "rows": int(len(work))})

    work = work[work["supplier_gvkey"].ne(work["customer_gvkey"])].copy()
    waterfall.append({"step": "remove_self_links", "rows": int(len(work))})

    work["revenue_to_customer"] = work["revenue_to_customer"].fillna(0.0)
    work = work[work["revenue_to_customer"] >= 0].copy()
    waterfall.append({"step": "non_negative_revenue", "rows": int(len(work))})

    work["known_date"] = work["disclosure_date"] + pd.to_timedelta(reporting_lag_days, unit="D")
    work["map_date"] = work["known_date"].where(work["known_date"] >= start, start)

    agg = (
        work.groupby(["supplier_gvkey", "customer_gvkey", "disclosure_date", "known_date", "map_date"], dropna=False)
        .agg(
            revenue_to_customer=("revenue_to_customer", "sum"),
            source_rows=("gvkey", "size"),
            first_cid=("cid", "first"),
            customer_name_raw=("cconm", "first"),
            supplier_name_raw=("conm", "first"),
            customer_name_segment=("cnms", "first"),
            customer_ticker_raw=("ctic", "first"),
            supplier_ticker_raw=("stic", "first"),
            customer_cusip_raw=("ccusip", "first"),
            supplier_cusip_raw=("scusip", "first"),
        )
        .reset_index()
    )

    agg = agg.sort_values(["supplier_gvkey", "customer_gvkey", "known_date", "disclosure_date"]).reset_index(drop=True)
    agg["edge_id"] = np.arange(len(agg), dtype=np.int64)
    waterfall.append({"step": "deduplicated_supplier_customer_date", "rows": int(len(agg))})

    supplier_total = agg.groupby(["supplier_gvkey", "disclosure_date"], dropna=False)["revenue_to_customer"].transform("sum")
    agg["supplier_reported_customer_sales_total"] = supplier_total
    agg["sales_share_supplier_reported"] = np.where(supplier_total > 0, agg["revenue_to_customer"] / supplier_total, np.nan)
    agg["supplier_customer_hhi_reported"] = (
        agg["sales_share_supplier_reported"]
        .fillna(0)
        .pow(2)
        .groupby([agg["supplier_gvkey"], agg["disclosure_date"]])
        .transform("sum")
    )

    agg["next_known_date_pair"] = agg.groupby(["supplier_gvkey", "customer_gvkey"])["known_date"].shift(-1)
    max_end = agg["known_date"] + pd.to_timedelta(max_edge_days, unit="D")
    next_end = agg["next_known_date_pair"] - pd.Timedelta(days=1)
    end_candidates = pd.concat([max_end.rename("max_end"), next_end.rename("next_end")], axis=1)
    agg["edge_end_date"] = end_candidates.min(axis=1, skipna=True)
    agg["edge_end_date"] = agg["edge_end_date"].fillna(max_end)
    agg["edge_end_date"] = agg["edge_end_date"].clip(upper=end)
    agg["edge_start_date"] = agg["known_date"]

    bad_end = agg["edge_end_date"] < agg["edge_start_date"]
    agg.loc[bad_end, "edge_end_date"] = agg.loc[bad_end, "edge_start_date"]

    first_known = agg.groupby(["supplier_gvkey", "customer_gvkey"])["known_date"].transform("min")
    agg["relationship_age_days"] = (agg["known_date"] - first_known).dt.days
    agg["relationship_sequence"] = agg.groupby(["supplier_gvkey", "customer_gvkey"]).cumcount() + 1
    agg["edge_year"] = agg["edge_start_date"].dt.year.astype("Int64")

    active = agg[(agg["edge_end_date"] >= start) & (agg["edge_start_date"] <= end)].copy()
    waterfall.append({"step": "active_in_sample_after_lag", "rows": int(len(active))})

    ccm_prepped = prepare_ccm(ccm)
    stock_prepped = prepare_stocknames(stocknames)

    active = assign_ccm(active, ccm_prepped, "supplier")
    active = assign_ccm(active, ccm_prepped, "customer")
    active["both_gvkeys_mapped_to_permno"] = active["supplier_permno"].notna() & active["customer_permno"].notna()
    waterfall.append({"step": "both_supplier_customer_permno_mapped", "rows": int(active["both_gvkeys_mapped_to_permno"].sum())})

    active = assign_stocknames(active, stock_prepped, "supplier")
    active = assign_stocknames(active, stock_prepped, "customer")

    common_supplier = active["supplier_shrcd"].isin([10, 11])
    common_customer = active["customer_shrcd"].isin([10, 11])
    major_exch_supplier = active["supplier_exchcd"].isin([1, 2, 3])
    major_exch_customer = active["customer_exchcd"].isin([1, 2, 3])

    active["common_share_major_exchange_pair"] = (
        active["both_gvkeys_mapped_to_permno"]
        & common_supplier
        & common_customer
        & major_exch_supplier
        & major_exch_customer
    )
    waterfall.append({"step": "common_shares_major_exchanges_pair", "rows": int(active["common_share_major_exchange_pair"].sum())})

    company_clean = company.copy()
    company_clean["gvkey"] = clean_gvkey(company_clean["gvkey"])
    company_cols = [c for c in ["gvkey", "sic", "naics", "gsector", "gind", "gsubind"] if c in company_clean.columns]
    company_clean = company_clean[company_cols].drop_duplicates("gvkey")

    active = active.merge(
        company_clean.add_prefix("supplier_company_"),
        left_on="supplier_gvkey",
        right_on="supplier_company_gvkey",
        how="left",
    ).drop(columns=["supplier_company_gvkey"], errors="ignore")

    active = active.merge(
        company_clean.add_prefix("customer_company_"),
        left_on="customer_gvkey",
        right_on="customer_company_gvkey",
        how="left",
    ).drop(columns=["customer_company_gvkey"], errors="ignore")

    qc = {
        "sample_start": sample_start,
        "sample_end": sample_end,
        "reporting_lag_days": reporting_lag_days,
        "max_edge_days": max_edge_days,
        "waterfall": waterfall,
        "active_edges": int(len(active)),
        "active_edges_both_permno_mapped": int(active["both_gvkeys_mapped_to_permno"].sum()),
        "active_edges_common_major_exchange_pair": int(active["common_share_major_exchange_pair"].sum()),
        "unique_supplier_gvkeys": int(active["supplier_gvkey"].nunique(dropna=True)),
        "unique_customer_gvkeys": int(active["customer_gvkey"].nunique(dropna=True)),
        "unique_supplier_permnos": int(active["supplier_permno"].nunique(dropna=True)) if "supplier_permno" in active else 0,
        "unique_customer_permnos": int(active["customer_permno"].nunique(dropna=True)) if "customer_permno" in active else 0,
        "median_sales_share_supplier_reported": None if active["sales_share_supplier_reported"].dropna().empty else float(active["sales_share_supplier_reported"].median()),
        "p95_sales_share_supplier_reported": None if active["sales_share_supplier_reported"].dropna().empty else float(active["sales_share_supplier_reported"].quantile(0.95)),
    }

    return active, qc


def build_nodes(edges: pd.DataFrame) -> pd.DataFrame:
    supplier_cols = {
        "supplier_gvkey": "gvkey",
        "supplier_permno": "permno",
        "supplier_permco": "permco",
        "supplier_ticker": "ticker",
        "supplier_comnam": "comnam",
        "supplier_shrcd": "shrcd",
        "supplier_exchcd": "exchcd",
        "supplier_siccd": "siccd",
        "supplier_company_sic": "company_sic",
        "supplier_company_naics": "company_naics",
        "supplier_company_gsector": "gsector",
        "supplier_company_gind": "gind",
        "supplier_company_gsubind": "gsubind",
    }

    customer_cols = {k.replace("supplier_", "customer_"): v for k, v in supplier_cols.items()}
    frames = []

    for side, mapping in [("supplier", supplier_cols), ("customer", customer_cols)]:
        existing = {k: v for k, v in mapping.items() if k in edges.columns}
        if not existing:
            continue
        part = edges[list(existing)].rename(columns=existing).copy()
        part["side_seen"] = side
        frames.append(part)

    if not frames:
        return pd.DataFrame()

    nodes = pd.concat(frames, ignore_index=True).dropna(subset=["gvkey"]).drop_duplicates()
    agg = (
        nodes.groupby("gvkey", dropna=False)
        .agg({c: "first" for c in nodes.columns if c not in ["gvkey", "side_seen"]} | {"side_seen": lambda x: ",".join(sorted(set(map(str, x))))})
        .reset_index()
    )
    return agg


def annual_summary(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()

    rows = []
    for year, g in edges.groupby("edge_year"):
        rows.append({
            "year": int(year),
            "edges": int(len(g)),
            "edges_both_permno_mapped": int(g["both_gvkeys_mapped_to_permno"].sum()),
            "edges_common_major_exchange_pair": int(g["common_share_major_exchange_pair"].sum()),
            "unique_supplier_gvkeys": int(g["supplier_gvkey"].nunique()),
            "unique_customer_gvkeys": int(g["customer_gvkey"].nunique()),
            "total_reported_customer_sales": float(g["revenue_to_customer"].sum()),
            "median_sales_share_supplier_reported": float(g["sales_share_supplier_reported"].median()) if g["sales_share_supplier_reported"].notna().any() else np.nan,
            "mean_supplier_customer_hhi_reported": float(g["supplier_customer_hhi_reported"].mean()) if g["supplier_customer_hhi_reported"].notna().any() else np.nan,
        })

    return pd.DataFrame(rows).sort_values("year")


def degree_summary(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()

    out_degree = edges.groupby("supplier_gvkey").size().rename("out_degree").reset_index().rename(columns={"supplier_gvkey": "gvkey"})
    in_degree = edges.groupby("customer_gvkey").size().rename("in_degree").reset_index().rename(columns={"customer_gvkey": "gvkey"})
    deg = out_degree.merge(in_degree, on="gvkey", how="outer").fillna(0)
    deg["total_degree"] = deg["out_degree"] + deg["in_degree"]
    return deg.sort_values("total_degree", ascending=False)


def make_figures(edges: pd.DataFrame, annual: pd.DataFrame, degree: pd.DataFrame, waterfall: pd.DataFrame, fig_dir: Path) -> list[dict[str, str]]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, str]] = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping static figures: {exc}")
        return made

    plt.figure(figsize=(10, 5.5))
    wf = waterfall.copy()
    plt.barh(wf["step"], wf["rows"])
    plt.gca().invert_yaxis()
    plt.xlabel("Rows")
    plt.title("Phase 2 supply-chain graph construction waterfall")
    plt.tight_layout()
    path = fig_dir / "phase2_sample_waterfall.png"
    plt.savefig(path, dpi=180)
    plt.close()
    made.append({"figure": "sample_waterfall", "path": str(path)})

    if not annual.empty:
        plt.figure(figsize=(10, 5.5))
        plt.plot(annual["year"], annual["edges"], marker="o", label="All active edges")
        plt.plot(annual["year"], annual["edges_common_major_exchange_pair"], marker="o", label="Common-share major-exchange pairs")
        plt.xlabel("Edge start year")
        plt.ylabel("Edges")
        plt.title("Production-network edges by year")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "phase2_edges_by_year.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "edges_by_year", "path": str(path)})

    if not degree.empty:
        plt.figure(figsize=(10, 5.5))
        vals = degree["total_degree"].clip(upper=degree["total_degree"].quantile(0.99))
        plt.hist(vals, bins=50)
        plt.xlabel("Total degree, clipped at p99")
        plt.ylabel("Nodes")
        plt.title("Degree distribution of GVKEY production-network nodes")
        plt.tight_layout()
        path = fig_dir / "phase2_degree_distribution.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "degree_distribution", "path": str(path)})

    if "sales_share_supplier_reported" in edges.columns and edges["sales_share_supplier_reported"].notna().any():
        plt.figure(figsize=(10, 5.5))
        vals = edges["sales_share_supplier_reported"].dropna().clip(upper=1.0)
        plt.hist(vals, bins=50)
        plt.xlabel("Customer sales share within supplier disclosed customer set")
        plt.ylabel("Edges")
        plt.title("Reported edge-weight distribution")
        plt.tight_layout()
        path = fig_dir / "phase2_edge_weight_distribution.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "edge_weight_distribution", "path": str(path)})

    return made


def make_redacted_network_html(edges: pd.DataFrame, out_path: Path, max_edges: int = 160) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        out_path.write_text(f"<html><body><p>Plotly unavailable: {html.escape(str(exc))}</p></body></html>")
        return

    if edges.empty:
        out_path.write_text("<html><body><p>No edges available.</p></body></html>")
        return

    latest_year = int(edges["edge_year"].dropna().max())
    g = edges[edges["edge_year"].eq(latest_year)].copy()
    g = g.sort_values("revenue_to_customer", ascending=False).head(max_edges)

    nodes = sorted(set(g["supplier_gvkey"].dropna().astype(str)) | set(g["customer_gvkey"].dropna().astype(str)))
    node_map = {node: f"N{i + 1:04d}" for i, node in enumerate(nodes)}
    n = max(len(nodes), 1)
    coords = {node: (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i, node in enumerate(nodes)}

    edge_x = []
    edge_y = []
    for r in g.itertuples(index=False):
        s = str(r.supplier_gvkey)
        c = str(r.customer_gvkey)
        x0, y0 = coords[s]
        x1, y1 = coords[c]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [coords[node][0] for node in nodes]
    node_y = [coords[node][1] for node in nodes]
    labels = [node_map[node] for node in nodes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.8), hoverinfo="skip", name="edges"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center", marker=dict(size=8), name="redacted nodes"))
    fig.update_layout(
        title=f"Redacted production-network preview, edge_start_year={latest_year}",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def render_report(out_path: Path, qc: dict[str, Any], annual: pd.DataFrame, degree: pd.DataFrame, figures: list[dict[str, str]], env: dict[str, Any]) -> None:
    cards = []
    metrics = [
        ("Active edges", qc.get("active_edges")),
        ("Both sides mapped to PERMNO", qc.get("active_edges_both_permno_mapped")),
        ("Common-share major-exchange edges", qc.get("active_edges_common_major_exchange_pair")),
        ("Supplier GVKEYs", qc.get("unique_supplier_gvkeys")),
        ("Customer GVKEYs", qc.get("unique_customer_gvkeys")),
        ("Median reported sales share", qc.get("median_sales_share_supplier_reported")),
    ]

    for label, value in metrics:
        if isinstance(value, float):
            val = f"{value:,.4f}"
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

    waterfall_html = pd.DataFrame(qc.get("waterfall", [])).to_html(index=False, escape=True, classes="data")
    annual_html = annual.to_html(index=False, escape=True, classes="data") if not annual.empty else "<p>No annual summary available.</p>"
    degree_html = degree.head(30).drop(columns=["gvkey"], errors="ignore").to_html(index=False, escape=True, classes="data") if not degree.empty else "<p>No degree summary available.</p>"

    figure_html = []
    for fig in figures:
        rel = Path(fig["path"]).name
        figure_html.append(f"<div class='figure'><h3>{html.escape(fig['figure'].replace('_', ' ').title())}</h3><img src='figures/{html.escape(rel)}' /></div>")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Production Network Alpha — Phase 2 Graph Backbone</title>
<style>
:root {{
  --bg: #07111f;
  --panel: #0f1e33;
  --text: #eef6ff;
  --muted: #9fb7ce;
  --line: rgba(255,255,255,.14);
  --accent: #7aa2ff;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif; background: radial-gradient(circle at top left, #183b66, var(--bg) 42%); color: var(--text); }}
header {{ padding: 46px 56px 28px; border-bottom: 1px solid var(--line); }}
h1 {{ margin: 0; font-size: 42px; letter-spacing: -.04em; }}
.subtitle {{ color: var(--muted); font-size: 17px; max-width: 1050px; line-height: 1.55; }}
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
.figure img {{ width: 100%; max-width: 1050px; border-radius: 16px; border: 1px solid var(--line); background: white; }}
pre {{ white-space: pre-wrap; background: rgba(0,0,0,.28); border: 1px solid var(--line); border-radius: 14px; padding: 16px; color: #dbecff; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.pill {{ border: 1px solid var(--line); background: rgba(255,255,255,.07); border-radius: 999px; padding: 8px 12px; color: var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>Phase 2 Graph Backbone</h1>
  <p class="subtitle">Point-in-time supplier–customer graph construction from WRDS supply-chain links, with conservative reporting lag, CRSP/CCM mapping, common-share filters, and aggregate-only QA visuals.</p>
  <div class="meta">
    <span class="pill">Generated: {html.escape(utc_now())}</span>
    <span class="pill">Sample: {html.escape(str(qc.get('sample_start')))} to {html.escape(str(qc.get('sample_end')))}</span>
    <span class="pill">Reporting lag: {html.escape(str(qc.get('reporting_lag_days')))} days</span>
    <span class="pill">Max edge life: {html.escape(str(qc.get('max_edge_days')))} days</span>
  </div>
</header>
<main>
  <div class="grid">{''.join(cards)}</div>
  <section><h2>Construction waterfall</h2>{waterfall_html}</section>
  <section><h2>Annual graph summary</h2>{annual_html}</section>
  <section><h2>Degree summary, redacted</h2>{degree_html}</section>
  <section><h2>Figures</h2>{''.join(figure_html)}</section>
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
    parser.add_argument("--sample-start", default=SAMPLE_START_DEFAULT)
    parser.add_argument("--sample-end", default=SAMPLE_END_DEFAULT)
    parser.add_argument("--reporting-lag-days", type=int, default=DEFAULT_REPORTING_LAG_DAYS)
    parser.add_argument("--max-edge-days", type=int, default=DEFAULT_MAX_EDGE_DAYS)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    raw_dir = project_root / "data" / "raw" / "wrds" / "phase2_graph_backbone"
    processed_dir = project_root / "data" / "processed" / "graph_backbone"
    fig_dir = out_dir / "figures"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("Phase 2 WRDS graph backbone")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {project_root}")
    print(f"Output dir: {out_dir}")
    print(f"Protected raw cache dir: {raw_dir}")
    print(f"Protected processed dir: {processed_dir}")
    print(f"Log dir: {log_dir}")
    print("================================================================================")

    env = {
        "utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cpu_count": os.cpu_count(),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_env": {k: os.environ.get(k) for k in ["PNA_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]},
        "packages": package_status(),
        "git": run_cmd(["bash", "-lc", "command -v git || true"]),
    }

    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    db = None
    try:
        db, user, user_source = connect_wrds()
        inputs = extract_inputs(db, raw_dir, args.force_refresh)
    except Exception:
        print("[ERROR] Phase 2 WRDS extraction failed.")
        print(traceback.format_exc())
        return 2
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    try:
        edges, qc = build_edges(
            seglink=inputs["supply_seglink"],
            ccm=inputs["ccm_linktable"],
            stocknames=inputs["crsp_stocknames"],
            company=inputs["comp_company"],
            sample_start=args.sample_start,
            sample_end=args.sample_end,
            reporting_lag_days=args.reporting_lag_days,
            max_edge_days=args.max_edge_days,
        )
        nodes = build_nodes(edges)
        annual = annual_summary(edges)
        degree = degree_summary(edges)
        waterfall = pd.DataFrame(qc["waterfall"])
    except Exception:
        print("[ERROR] Phase 2 graph construction failed.")
        print(traceback.format_exc())
        return 3

    write_parquet(edges, processed_dir / "edges_supplier_customer_all.parquet")
    write_parquet(edges[edges["common_share_major_exchange_pair"]].copy(), processed_dir / "edges_supplier_customer_common_us.parquet")
    write_parquet(nodes, processed_dir / "nodes_gvkey_permno.parquet")

    annual.to_csv(out_dir / "annual_graph_summary.csv", index=False)
    waterfall.to_csv(out_dir / "construction_waterfall.csv", index=False)
    degree.drop(columns=["gvkey"], errors="ignore").head(200).to_csv(out_dir / "degree_summary_redacted_top200.csv", index=False)
    pd.DataFrame({"column": edges.columns}).to_csv(out_dir / "edge_column_manifest.csv", index=False)

    raw_inventory = []
    for path in sorted(raw_dir.glob("*.parquet")):
        raw_inventory.append({"file": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(raw_inventory).to_csv(out_dir / "protected_raw_cache_inventory.csv", index=False)

    processed_inventory = []
    for path in sorted(processed_dir.glob("*.parquet")):
        processed_inventory.append({"file": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(processed_inventory).to_csv(out_dir / "protected_processed_inventory.csv", index=False)

    with (out_dir / "phase2_quality_summary.json").open("w") as f:
        json.dump(qc, f, indent=2, default=str)

    figures = make_figures(edges, annual, degree, waterfall, fig_dir)
    make_redacted_network_html(edges[edges["common_share_major_exchange_pair"]].copy(), out_dir / "phase2_network_preview_redacted.html")
    render_report(out_dir / "phase2_graph_backbone_report.html", qc, annual, degree, figures, env)

    summary = [
        "# Phase 2 graph backbone summary",
        "",
        f"- Generated at UTC: {utc_now()}",
        f"- Active supplier-customer edge intervals: {qc['active_edges']:,}",
        f"- Both sides mapped to CRSP PERMNO: {qc['active_edges_both_permno_mapped']:,}",
        f"- Common-share major-exchange pairs: {qc['active_edges_common_major_exchange_pair']:,}",
        f"- Unique supplier GVKEYs: {qc['unique_supplier_gvkeys']:,}",
        f"- Unique customer GVKEYs: {qc['unique_customer_gvkeys']:,}",
        f"- Report: {out_dir / 'phase2_graph_backbone_report.html'}",
        f"- Protected processed edge file: {processed_dir / 'edges_supplier_customer_common_us.parquet'}",
        "",
        "Data policy: raw/protected Parquet files are local only and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE2_SUMMARY.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
