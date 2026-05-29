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

HORIZONS = [1, 2, 5, 10, 20]
DEFAULT_SAMPLE_START = "2015-01-01"
DEFAULT_SAMPLE_END = "2025-12-31"
DEFAULT_CRSP_START = "2014-07-01"
DEFAULT_CRSP_END = "2026-03-31"
DEFAULT_SQL_CHUNK = 1000
DEFAULT_MAX_SPILLOVER_ROWS_PER_YEAR = 0  # 0 means no cap for full scale


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def mask_user(username: str | None) -> str | None:
    if not username:
        return None
    return username[:2] + "***" + username[-1] if len(username) > 3 else username[0] + "***"


def detect_wrds_username() -> tuple[str | None, str]:
    env_user = os.environ.get("WRDS_USERNAME")
    if env_user:
        return env_user, "WRDS_USERNAME"
    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        try:
            lines = pgpass.read_text(errors="ignore").splitlines()
            for line in lines:
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":")
                if len(parts) >= 5 and "wrds" in line.lower():
                    return parts[-2], "~/.pgpass"
            for line in lines:
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":")
                if len(parts) >= 5:
                    return parts[-2], "~/.pgpass"
        except Exception:
            return None, "~/.pgpass unreadable"
    return None, "not found"


def connect_wrds():
    import wrds

    user, source = detect_wrds_username()
    try:
        db = wrds.Connection(wrds_username=user, verbose=False) if user else wrds.Connection(verbose=False)
    except TypeError:
        db = wrds.Connection()
    try:
        db.raw_sql("set statement_timeout to '1800000ms'")
    except Exception:
        pass
    return db, user, source


def run_cmd(cmd: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}


def package_status() -> list[dict[str, str]]:
    pkgs = ["pandas", "numpy", "pyarrow", "wrds", "sqlalchemy", "psycopg2", "matplotlib", "plotly", "duckdb", "polars", "lightgbm", "sklearn"]
    rows = []
    for pkg in pkgs:
        try:
            rows.append({"package": pkg, "version": md.version(pkg), "status": "ok"})
        except md.PackageNotFoundError:
            rows.append({"package": pkg, "version": "", "status": "missing"})
    return rows


def q(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_in(values: list[int | str]) -> str:
    return "(" + ",".join(q(v) for v in values) + ")"


def chunks(vals: list[int | str], n: int) -> list[list[int | str]]:
    return [vals[i : i + n] for i in range(0, len(vals), n)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_token(value: Any, prefix: str = "id") -> str | None:
    if value is None or pd.isna(value):
        return None
    raw = str(value).encode("utf-8")
    return f"{prefix}_" + hashlib.sha256(b"pna_phase4_1_" + raw).hexdigest()[:16]


def redact_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        lc = col.lower()
        if lc.endswith("gvkey") or lc in {"gvkey", "source_gvkey", "target_gvkey"}:
            out[col] = out[col].map(lambda x: hash_token(x, "g"))
        elif lc.endswith("permno") or lc in {"permno", "source_permno", "target_permno"}:
            out[col] = out[col].map(lambda x: hash_token(x, "p"))
        elif lc.endswith("permco") or lc in {"permco"}:
            out[col] = out[col].map(lambda x: hash_token(x, "c"))
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def as_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def clean_gvkey(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)


def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required protected local file not found: {path}")
    return pd.read_parquet(path)


def build_target_universe(edges: pd.DataFrame, news: pd.DataFrame, sample_start: str, sample_end: str, out_dir: Path) -> pd.DataFrame:
    edges = edges.copy()
    news = news.copy()
    for col in ["edge_start_date", "edge_end_date"]:
        edges[col] = to_dt(edges[col])
    news["news_signal_date"] = to_dt(news["news_signal_date"])
    start = pd.Timestamp(sample_start)
    end = pd.Timestamp(sample_end)
    news = news[(news["news_signal_date"] >= start) & (news["news_signal_date"] <= end)].copy()

    node_frames = []
    for side in ["supplier", "customer"]:
        part = pd.DataFrame({
            "gvkey": clean_gvkey(edges[f"{side}_gvkey"]),
            "permno": as_int(edges[f"{side}_permno"]),
            "side": side,
        })
        node_frames.append(part)
    node_rows = pd.concat(node_frames, ignore_index=True).dropna(subset=["gvkey", "permno"])

    degree = node_rows.groupby(["gvkey", "permno"], dropna=False).size().rename("graph_degree").reset_index()
    news_counts = (
        news.assign(gvkey=clean_gvkey(news["gvkey"]))
        .groupby("gvkey", dropna=False)
        .agg(own_news_node_days=("news_signal_date", "nunique"), own_news_events=("n_events", "sum"))
        .reset_index()
    )
    target = degree.merge(news_counts, on="gvkey", how="left")
    target["own_news_node_days"] = target["own_news_node_days"].fillna(0)
    target["own_news_events"] = target["own_news_events"].fillna(0)
    target = target.sort_values(["graph_degree", "own_news_node_days"], ascending=[False, False]).drop_duplicates("gvkey")
    target["permno"] = as_int(target["permno"])
    redact_identifiers(target).to_csv(out_dir / "phase4_1_target_universe_redacted.csv", index=False)
    return target


def prepare_edges(edges: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    e = edges.copy()
    target_gvkeys = set(targets["gvkey"].astype(str))
    for col in ["supplier_gvkey", "customer_gvkey"]:
        e[col] = clean_gvkey(e[col])
    for col in ["edge_start_date", "edge_end_date"]:
        e[col] = to_dt(e[col])
    for col in ["supplier_permno", "customer_permno"]:
        e[col] = as_int(e[col])
    e["edge_weight"] = pd.to_numeric(e.get("sales_share_supplier_reported", np.nan), errors="coerce")
    fallback_weight = float(e["edge_weight"].dropna().median()) if e["edge_weight"].notna().any() else 1.0
    e["edge_weight"] = e["edge_weight"].fillna(fallback_weight).clip(lower=0.0, upper=1.0)
    e["relationship_age_days"] = pd.to_numeric(e.get("relationship_age_days", np.nan), errors="coerce")
    e["supplier_customer_hhi_reported"] = pd.to_numeric(e.get("supplier_customer_hhi_reported", np.nan), errors="coerce")
    return e, target_gvkeys


def prepare_news_year(news: pd.DataFrame, year: int, sample_start: str, sample_end: str) -> pd.DataFrame:
    n = news.copy()
    n["gvkey"] = clean_gvkey(n["gvkey"])
    n["permno"] = as_int(n["permno"])
    n["news_signal_date"] = to_dt(n["news_signal_date"])
    n = n[(n["news_signal_date"].dt.year.eq(year)) & (n["news_signal_date"] >= pd.Timestamp(sample_start)) & (n["news_signal_date"] <= pd.Timestamp(sample_end))].copy()
    for col in ["signed_news_shock", "positive_news_shock", "negative_news_shock", "abs_news_shock", "n_events", "n_stories"]:
        if col not in n.columns:
            n[col] = 0.0
        n[col] = pd.to_numeric(n[col], errors="coerce").fillna(0.0)
    return n


def build_spillover_features_for_year(
    e: pd.DataFrame,
    n: pd.DataFrame,
    target_gvkeys: set[str],
    year: int,
    max_spillover_rows: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if n.empty:
        return pd.DataFrame(), pd.DataFrame(), {"year": year, "news_node_days": 0, "raw_spillover_rows": 0, "feature_rows": 0}

    supplier_edges = e[e["customer_gvkey"].isin(target_gvkeys)].copy()
    supplier_join = n.merge(
        supplier_edges,
        left_on="gvkey",
        right_on="supplier_gvkey",
        how="inner",
        suffixes=("_news", "_edge"),
    )
    supplier_join = supplier_join[(supplier_join["edge_start_date"] <= supplier_join["news_signal_date"]) & (supplier_join["edge_end_date"] >= supplier_join["news_signal_date"])].copy()
    supplier_join["direction"] = "supplier_news_to_customer"
    supplier_join["source_gvkey"] = supplier_join["supplier_gvkey"]
    supplier_join["source_permno"] = supplier_join["supplier_permno"]
    supplier_join["target_gvkey"] = supplier_join["customer_gvkey"]
    supplier_join["target_permno"] = supplier_join["customer_permno"]

    customer_edges = e[e["supplier_gvkey"].isin(target_gvkeys)].copy()
    customer_join = n.merge(
        customer_edges,
        left_on="gvkey",
        right_on="customer_gvkey",
        how="inner",
        suffixes=("_news", "_edge"),
    )
    customer_join = customer_join[(customer_join["edge_start_date"] <= customer_join["news_signal_date"]) & (customer_join["edge_end_date"] >= customer_join["news_signal_date"])].copy()
    customer_join["direction"] = "customer_news_to_supplier"
    customer_join["source_gvkey"] = customer_join["customer_gvkey"]
    customer_join["source_permno"] = customer_join["customer_permno"]
    customer_join["target_gvkey"] = customer_join["supplier_gvkey"]
    customer_join["target_permno"] = customer_join["supplier_permno"]

    rows = pd.concat([supplier_join, customer_join], ignore_index=True)
    raw_rows = int(len(rows))
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame(), {"year": year, "news_node_days": int(len(n)), "raw_spillover_rows": 0, "feature_rows": 0}

    rows["weighted_signed_shock"] = rows["signed_news_shock"] * rows["edge_weight"]
    rows["weighted_abs_shock"] = rows["abs_news_shock"] * rows["edge_weight"]
    rows["weighted_positive_shock"] = rows["positive_news_shock"] * rows["edge_weight"]
    rows["weighted_negative_shock"] = rows["negative_news_shock"] * rows["edge_weight"]
    rows["target_permno"] = as_int(rows["target_permno"])
    rows["source_permno"] = as_int(rows["source_permno"])

    if max_spillover_rows and max_spillover_rows > 0 and len(rows) > max_spillover_rows:
        rows = rows.assign(abs_rank_value=rows["weighted_abs_shock"].abs()).sort_values("abs_rank_value", ascending=False).head(max_spillover_rows).drop(columns=["abs_rank_value"])

    base_keys = ["target_gvkey", "target_permno", "news_signal_date"]
    agg_all = (
        rows.groupby(base_keys, dropna=False)
        .agg(
            spillover_signed_shock=("weighted_signed_shock", "sum"),
            spillover_abs_shock=("weighted_abs_shock", "sum"),
            spillover_positive_shock=("weighted_positive_shock", "sum"),
            spillover_negative_shock=("weighted_negative_shock", "sum"),
            spillover_source_events=("n_events", "sum"),
            spillover_source_stories=("n_stories", "sum"),
            n_shocked_neighbor_firms=("source_gvkey", "nunique"),
            n_active_shock_edges=("source_gvkey", "size"),
            mean_edge_weight=("edge_weight", "mean"),
            max_edge_weight=("edge_weight", "max"),
            mean_relationship_age_days=("relationship_age_days", "mean"),
            mean_supplier_customer_hhi=("supplier_customer_hhi_reported", "mean"),
        )
        .reset_index()
    )

    dir_agg = (
        rows.groupby(base_keys + ["direction"], dropna=False)
        .agg(
            dir_signed=("weighted_signed_shock", "sum"),
            dir_abs=("weighted_abs_shock", "sum"),
            dir_n_edges=("source_gvkey", "size"),
            dir_n_neighbors=("source_gvkey", "nunique"),
        )
        .reset_index()
    )
    wide = dir_agg.pivot_table(index=base_keys, columns="direction", values=["dir_signed", "dir_abs", "dir_n_edges", "dir_n_neighbors"], aggfunc="sum", fill_value=0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    features = agg_all.merge(wide, on=base_keys, how="left")
    for col in features.columns:
        if col.startswith("dir_"):
            features[col] = features[col].fillna(0.0)

    own = n.rename(columns={
        "gvkey": "target_gvkey",
        "permno": "target_permno",
        "n_events": "own_news_events",
        "n_stories": "own_news_stories",
        "signed_news_shock": "own_signed_news_shock",
        "abs_news_shock": "own_abs_news_shock",
    })[["target_gvkey", "target_permno", "news_signal_date", "own_news_events", "own_news_stories", "own_signed_news_shock", "own_abs_news_shock"]]
    features = features.merge(own, on=base_keys, how="left")
    for col in ["own_news_events", "own_news_stories", "own_signed_news_shock", "own_abs_news_shock"]:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)
    features["pure_spillover_no_own_news"] = features["own_abs_news_shock"].eq(0)
    features = features.rename(columns={"news_signal_date": "signal_date"})
    features["signal_year"] = year

    event_redacted_cols = [
        "news_signal_date", "direction", "source_gvkey", "target_gvkey", "edge_weight", "weighted_signed_shock", "weighted_abs_shock",
        "relationship_age_days", "supplier_customer_hhi_reported"
    ]
    rows_redacted = rows[event_redacted_cols].copy().rename(columns={"news_signal_date": "signal_date"})
    stats = {"year": year, "news_node_days": int(len(n)), "raw_spillover_rows": raw_rows, "kept_spillover_rows": int(len(rows)), "feature_rows": int(len(features)), "pure_feature_rows": int(features["pure_spillover_no_own_news"].sum())}
    return features, rows_redacted, stats


def build_all_spillover_features(
    edges: pd.DataFrame,
    news: pd.DataFrame,
    targets: pd.DataFrame,
    sample_start: str,
    sample_end: str,
    features_dir: Path,
    out_dir: Path,
    max_spillover_rows_per_year: int,
) -> tuple[list[Path], pd.DataFrame]:
    features_dir.mkdir(parents=True, exist_ok=True)
    e, target_gvkeys = prepare_edges(edges, targets)
    years = list(range(pd.Timestamp(sample_start).year, pd.Timestamp(sample_end).year + 1))
    feature_paths: list[Path] = []
    stats_rows = []
    redacted_samples = []
    for year in years:
        print(f"[FEATURES] Building spillover features year={year} ...")
        n = prepare_news_year(news, year, sample_start, sample_end)
        features, rows_redacted, stats = build_spillover_features_for_year(e, n, target_gvkeys, year, max_spillover_rows_per_year, out_dir)
        stats_rows.append(stats)
        if not features.empty:
            path = features_dir / f"spillover_features_{year}.parquet"
            write_parquet(features, path)
            feature_paths.append(path)
            if len(redacted_samples) < 5:
                redacted_samples.append(rows_redacted.head(2000))
        print(f"[FEATURES] year={year}: {stats}")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "phase4_1_spillover_feature_year_stats.csv", index=False)
    if redacted_samples:
        redact_identifiers(pd.concat(redacted_samples, ignore_index=True)).to_csv(out_dir / "phase4_1_spillover_events_redacted_sample.csv", index=False)
    return feature_paths, stats_df


def safe_sql(db, sql: str, label: str) -> pd.DataFrame:
    t0 = time.time()
    print(f"[WRDS] {label} ...")
    df = db.raw_sql(sql)
    df.columns = [str(c).lower() for c in df.columns]
    print(f"[WRDS] {label}: rows={len(df):,}, elapsed={time.time() - t0:.1f}s")
    return df


def query_crsp_returns(db, permnos: list[int], start_date: str, end_date: str, raw_dir: Path, chunk_size: int, force_refresh: bool) -> dict[str, pd.DataFrame]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dsf_path = raw_dir / f"crsp_dsf_full_{start_date}_{end_date}.parquet"
    dsi_path = raw_dir / f"crsp_dsi_full_{start_date}_{end_date}.parquet"
    dl_path = raw_dir / f"crsp_dsedelist_full_{start_date}_{end_date}.parquet"

    if dsf_path.exists() and dsi_path.exists() and dl_path.exists() and not force_refresh:
        print("[CACHE] Using cached full CRSP extracts.")
        return {"dsf": pd.read_parquet(dsf_path), "dsi": pd.read_parquet(dsi_path), "dl": pd.read_parquet(dl_path)}

    all_chunks = chunks(permnos, chunk_size)
    frames = []
    for i, vals in enumerate(all_chunks, start=1):
        sql = f"""
            select permno, date, ret, retx, prc, shrout, vol
            from crsp_a_stock.dsf
            where permno in {sql_in(vals)}
              and date between date {q(start_date)} and date {q(end_date)}
        """
        frames.append(safe_sql(db, sql, f"CRSP dsf full chunk {i}/{len(all_chunks)}"))
    dsf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    dsi_sql = f"""
        select date, vwretd, ewretd
        from crsp_a_stock.dsi
        where date between date {q(start_date)} and date {q(end_date)}
    """
    dsi = safe_sql(db, dsi_sql, "CRSP dsi full market returns")

    dl_frames = []
    for i, vals in enumerate(all_chunks, start=1):
        sql = f"""
            select permno, dlstdt, dlret, dlstcd
            from crsp_a_stock.dsedelist
            where permno in {sql_in(vals)}
              and dlstdt between date {q(start_date)} and date {q(end_date)}
        """
        dl_frames.append(safe_sql(db, sql, f"CRSP dsedelist full chunk {i}/{len(all_chunks)}"))
    dl = pd.concat(dl_frames, ignore_index=True) if dl_frames else pd.DataFrame()

    write_parquet(dsf, dsf_path)
    write_parquet(dsi, dsi_path)
    write_parquet(dl, dl_path)
    return {"dsf": dsf, "dsi": dsi, "dl": dl}


def safe_log_positive(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    mask = s.gt(0) & np.isfinite(s)
    out.loc[mask] = np.log(s.loc[mask].astype(float))
    return out


def prepare_returns(crsp: dict[str, pd.DataFrame], processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dsf = crsp["dsf"].copy()
    dsi = crsp["dsi"].copy()
    dl = crsp["dl"].copy()

    if dsf.empty or dsi.empty:
        raise RuntimeError("CRSP dsf/dsi returned no rows.")

    dsf["permno"] = as_int(dsf["permno"])
    dsf["date"] = to_dt(dsf["date"])
    for col in ["ret", "retx", "prc", "shrout", "vol"]:
        dsf[col] = pd.to_numeric(dsf[col], errors="coerce")

    dsi["date"] = to_dt(dsi["date"])
    for col in ["vwretd", "ewretd"]:
        dsi[col] = pd.to_numeric(dsi[col], errors="coerce")

    if not dl.empty:
        dl["permno"] = as_int(dl["permno"])
        dl["dlstdt"] = to_dt(dl["dlstdt"])
        dl["dlret"] = pd.to_numeric(dl["dlret"], errors="coerce")
        dl = dl.rename(columns={"dlstdt": "date"})
        dsf = dsf.merge(dl[["permno", "date", "dlret", "dlstcd"]], on=["permno", "date"], how="left")
    else:
        dsf["dlret"] = np.nan
        dsf["dlstcd"] = np.nan

    out = dsf.merge(dsi[["date", "vwretd", "ewretd"]], on="date", how="left")
    base_ret = out["ret"].copy()
    out["ret_adj"] = np.where(
        out["dlret"].notna(),
        (1.0 + base_ret.fillna(0.0)) * (1.0 + out["dlret"].fillna(0.0)) - 1.0,
        base_ret,
    )
    out["abret_mkt"] = out["ret_adj"] - out["vwretd"]
    out["mktcap"] = out["prc"].abs() * out["shrout"]
    out["dollar_vol"] = out["prc"].abs() * out["vol"]
    out = out.sort_values(["permno", "date"]).reset_index(drop=True)

    feature_frames = []
    for permno, g in out.groupby("permno", sort=False):
        x = g.copy().sort_values("date")
        gross = 1.0 + x["ret_adj"]
        x["ret_mom_21d"] = gross.shift(1).rolling(21, min_periods=10).apply(np.prod, raw=True) - 1.0
        x["ret_mom_126d"] = gross.shift(1).rolling(126, min_periods=60).apply(np.prod, raw=True) - 1.0
        x["idio_vol_63d"] = x["abret_mkt"].shift(1).rolling(63, min_periods=20).std()
        x["dollar_vol_21d"] = x["dollar_vol"].shift(1).rolling(21, min_periods=10).mean()
        x["log_mktcap"] = safe_log_positive(x["mktcap"])
        feature_frames.append(x)
    ret_panel = pd.concat(feature_frames, ignore_index=True)

    labels = []
    keep_cols = ["permno", "date", "ret_adj", "vwretd", "abret_mkt", "ret_mom_21d", "ret_mom_126d", "idio_vol_63d", "dollar_vol_21d", "log_mktcap", "prc", "vol", "shrout", "mktcap"]
    for permno, g in ret_panel.groupby("permno", sort=False):
        x = g.sort_values("date").reset_index(drop=True).copy()
        ret_arr = x["ret_adj"].to_numpy(dtype=float)
        mkt_arr = x["vwretd"].to_numpy(dtype=float)
        date_arr = x["date"].to_numpy()
        n = len(x)
        for h in HORIZONS:
            f_ret = np.full(n, np.nan)
            f_mkt = np.full(n, np.nan)
            f_ab = np.full(n, np.nan)
            end_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
            # Explicit loop is stable and transparent. CRSP rows are modest enough for this stage.
            for i in range(n - h):
                rr = ret_arr[i + 1 : i + h + 1]
                mm = mkt_arr[i + 1 : i + h + 1]
                if np.isfinite(rr).all() and np.isfinite(mm).all():
                    cr = float(np.prod(1.0 + rr) - 1.0)
                    cm = float(np.prod(1.0 + mm) - 1.0)
                    f_ret[i] = cr
                    f_mkt[i] = cm
                    f_ab[i] = cr - cm
                    end_dates[i] = date_arr[i + h]
            x[f"fwd_ret_{h}d"] = f_ret
            x[f"fwd_mkt_{h}d"] = f_mkt
            x[f"fwd_abret_{h}d"] = f_ab
            x[f"label_end_date_{h}d"] = end_dates
        labels.append(x[keep_cols + [c for c in x.columns if c.startswith("fwd_") or c.startswith("label_end_date_")]])
    label_panel = pd.concat(labels, ignore_index=True)

    diagnostics = {
        "ret_panel_rows": int(len(ret_panel)),
        "label_panel_rows": int(len(label_panel)),
        "unique_permnos": int(ret_panel["permno"].nunique()),
        "zero_or_negative_mktcap_rows": int((pd.to_numeric(ret_panel["mktcap"], errors="coerce") <= 0).sum()),
        "nonfinite_log_mktcap_rows": int((~np.isfinite(pd.to_numeric(ret_panel["log_mktcap"], errors="coerce"))).sum()),
        "infinite_log_mktcap_rows": int(np.isinf(pd.to_numeric(ret_panel["log_mktcap"], errors="coerce")).sum()),
    }
    write_parquet(ret_panel, processed_dir / "full_crsp_return_panel_features.parquet")
    write_parquet(label_panel, processed_dir / "full_crsp_forward_labels.parquet")
    return ret_panel, label_panel, diagnostics


def map_signal_to_trading_day(features: pd.DataFrame, market_dates: pd.Series) -> pd.DataFrame:
    f = features.copy()
    f["signal_date"] = to_dt(f["signal_date"])
    cal = pd.DataFrame({"label_base_date": pd.to_datetime(market_dates.dropna().drop_duplicates()).sort_values()})
    left = f.sort_values("signal_date")
    mapped = pd.merge_asof(left, cal, left_on="signal_date", right_on="label_base_date", direction="forward")
    return mapped


def build_model_matrix(features: pd.DataFrame, label_panel: pd.DataFrame) -> pd.DataFrame:
    labels = label_panel.rename(columns={"date": "label_base_date", "permno": "target_permno"}).copy()
    labels["target_permno"] = as_int(labels["target_permno"])
    labels["label_base_date"] = to_dt(labels["label_base_date"])
    f = features.copy()
    f["target_permno"] = as_int(f["target_permno"])
    f["label_base_date"] = to_dt(f["label_base_date"])
    matrix = f.merge(labels, on=["target_permno", "label_base_date"], how="left")
    for h in HORIZONS:
        matrix[f"no_lookahead_ok_{h}d"] = to_dt(matrix[f"label_end_date_{h}d"]) > matrix["label_base_date"]
    matrix["signal_to_base_days"] = (to_dt(matrix["label_base_date"]) - to_dt(matrix["signal_date"])).dt.days
    return matrix


def summarize_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_date = (
        matrix.groupby("label_base_date", dropna=False)
        .agg(
            rows=("target_gvkey", "size"),
            target_gvkeys=("target_gvkey", "nunique"),
            mean_abs_spillover=("spillover_abs_shock", "mean"),
            pure_rows=("pure_spillover_no_own_news", "sum"),
            fwd5_mean=("fwd_abret_5d", "mean"),
            fwd20_mean=("fwd_abret_20d", "mean"),
        )
        .reset_index()
    )
    by_date["label_base_date"] = by_date["label_base_date"].astype(str)

    stats_rows = []
    for h in HORIZONS:
        y = pd.to_numeric(matrix[f"fwd_abret_{h}d"], errors="coerce")
        x = pd.to_numeric(matrix["spillover_signed_shock"], errors="coerce")
        mask = y.notna() & x.notna()
        rank_ic = float(x[mask].rank().corr(y[mask].rank())) if mask.sum() >= 50 else np.nan
        pure_mask = mask & matrix["pure_spillover_no_own_news"].fillna(False)
        pure_rank_ic = float(x[pure_mask].rank().corr(y[pure_mask].rank())) if pure_mask.sum() >= 50 else np.nan
        stats_rows.append({
            "horizon_days": h,
            "n_labeled": int(mask.sum()),
            "mean_fwd_abret": float(y[mask].mean()) if mask.any() else np.nan,
            "std_fwd_abret": float(y[mask].std()) if mask.any() else np.nan,
            "rank_ic_spillover_signed": rank_ic,
            "pure_n_labeled": int(pure_mask.sum()),
            "pure_rank_ic_spillover_signed": pure_rank_ic,
        })
    horizon_stats = pd.DataFrame(stats_rows)

    work = matrix[matrix["fwd_abret_5d"].notna() & matrix["spillover_signed_shock"].notna()].copy()
    if len(work) >= 100:
        work["signal_decile"] = pd.qcut(work["spillover_signed_shock"].rank(method="first"), 10, labels=False) + 1
        deciles = work.groupby("signal_decile").agg(
            rows=("target_gvkey", "size"),
            mean_fwd_abret_1d=("fwd_abret_1d", "mean"),
            mean_fwd_abret_5d=("fwd_abret_5d", "mean"),
            mean_fwd_abret_20d=("fwd_abret_20d", "mean"),
            mean_signal=("spillover_signed_shock", "mean"),
        ).reset_index()
    else:
        deciles = pd.DataFrame()
    return by_date, horizon_stats, deciles


def validation_report(matrix: pd.DataFrame, crsp_rows: int, spillover_rows: int, target_rows: int, return_diagnostics: dict[str, Any]) -> dict[str, Any]:
    label_cov = {f"label_cov_{h}d": float(matrix[f"fwd_abret_{h}d"].notna().mean()) for h in HORIZONS}
    no_look = {f"no_lookahead_{h}d": bool(matrix.loc[matrix[f"fwd_abret_{h}d"].notna(), f"no_lookahead_ok_{h}d"].all()) for h in HORIZONS}
    pure_rows = int(matrix["pure_spillover_no_own_news"].fillna(False).sum())
    numeric_cols = matrix.select_dtypes(include=["number"]).columns.tolist()
    # Only true +/-inf is forbidden. NaNs are expected in rolling controls and near sample boundaries.
    inf_count = 0
    for col in numeric_cols:
        arr = pd.to_numeric(matrix[col], errors="coerce").to_numpy(dtype=float)
        inf_count += int(np.isinf(arr).sum())
    signal_to_base = pd.to_numeric(matrix["signal_to_base_days"], errors="coerce")
    checks = {
        "target_universe_positive": target_rows > 0,
        "spillover_rows_positive": spillover_rows > 0,
        "crsp_rows_positive": crsp_rows > 0,
        "model_matrix_rows_ge_100000": len(matrix) >= 100000,
        "label_5d_coverage_ge_50pct": label_cov["label_cov_5d"] >= 0.50,
        "label_20d_coverage_ge_50pct": label_cov["label_cov_20d"] >= 0.50,
        "pure_spillover_rows_positive": pure_rows > 0,
        "no_lookahead_all_horizons": all(no_look.values()),
        "no_infinite_numeric_values": inf_count == 0,
        "no_negative_signal_to_base_days": bool((signal_to_base.dropna() >= 0).all()),
        "log_mktcap_no_inf": int(return_diagnostics.get("infinite_log_mktcap_rows", 1)) == 0,
    }
    return {
        "generated_at_utc": utc_now(),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "rows": int(len(matrix)),
        "target_rows": int(target_rows),
        "spillover_feature_rows": int(spillover_rows),
        "crsp_return_rows": int(crsp_rows),
        "pure_spillover_rows": pure_rows,
        "label_coverage": label_cov,
        "no_lookahead": no_look,
        "infinite_numeric_values": int(inf_count),
        "signal_to_base_days_min": None if signal_to_base.dropna().empty else int(signal_to_base.min()),
        "signal_to_base_days_max": None if signal_to_base.dropna().empty else int(signal_to_base.max()),
        "return_diagnostics": return_diagnostics,
    }


def make_figures(out_dir: Path, validation: dict[str, Any], horizon_stats: pd.DataFrame, deciles: pd.DataFrame, by_year: pd.DataFrame, matrix_sample: pd.DataFrame) -> list[dict[str, str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable: {exc}")
        return []

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    checks = validation.get("checks", {})
    plt.figure(figsize=(10.5, 5.8))
    labels = list(checks.keys())
    vals = [1 if checks[k] else 0 for k in labels]
    plt.barh(labels, vals)
    plt.xlim(0, 1.1)
    plt.xlabel("Pass = 1")
    plt.title("Phase 4.1 full-scale validation checks")
    plt.tight_layout()
    p = fig_dir / "phase4_1_validation_checks.png"
    plt.savefig(p, dpi=180)
    plt.close()
    made.append({"figure": "validation_checks", "path": str(p)})

    if not horizon_stats.empty:
        plt.figure(figsize=(9.5, 5.5))
        plt.plot(horizon_stats["horizon_days"], horizon_stats["rank_ic_spillover_signed"], marker="o", label="All rows")
        plt.plot(horizon_stats["horizon_days"], horizon_stats["pure_rank_ic_spillover_signed"], marker="o", label="No own-news rows")
        plt.axhline(0, linewidth=0.8)
        plt.xlabel("Forward horizon, trading days")
        plt.ylabel("Rank IC")
        plt.title("Full-scale spillover signal rank IC by horizon")
        plt.legend()
        plt.tight_layout()
        p = fig_dir / "phase4_1_rank_ic_by_horizon.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append({"figure": "rank_ic_by_horizon", "path": str(p)})

    if not deciles.empty:
        plt.figure(figsize=(9.5, 5.5))
        plt.bar(deciles["signal_decile"].astype(int), deciles["mean_fwd_abret_5d"])
        plt.axhline(0, linewidth=0.8)
        plt.xlabel("Spillover signed-shock decile")
        plt.ylabel("Mean 5-day forward market-adjusted return")
        plt.title("Full-scale decile monotonicity check")
        plt.tight_layout()
        p = fig_dir / "phase4_1_decile_5d.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append({"figure": "decile_5d", "path": str(p)})

    if not by_year.empty:
        plt.figure(figsize=(10.5, 5.8))
        plt.plot(by_year["signal_year"], by_year["matrix_rows"], marker="o", label="All matrix rows")
        plt.plot(by_year["signal_year"], by_year["pure_spillover_rows"], marker="o", label="Pure spillover rows")
        plt.xlabel("Signal year")
        plt.ylabel("Rows")
        plt.title("Full-scale model matrix rows by signal year")
        plt.legend()
        plt.tight_layout()
        p = fig_dir / "phase4_1_matrix_rows_by_year.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append({"figure": "matrix_rows_by_year", "path": str(p)})

    vals = pd.to_numeric(matrix_sample.get("spillover_signed_shock", pd.Series(dtype=float)), errors="coerce").dropna()
    if not vals.empty:
        cap = vals.abs().quantile(0.99)
        plt.figure(figsize=(9.5, 5.5))
        plt.hist(vals.clip(-cap, cap), bins=60)
        plt.xlabel("Spillover signed shock, sample clipped at p99 abs")
        plt.ylabel("Target-days")
        plt.title("Full-scale spillover signal distribution sample")
        plt.tight_layout()
        p = fig_dir / "phase4_1_spillover_distribution_sample.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append({"figure": "spillover_distribution_sample", "path": str(p)})

    return made


def render_report(out_path: Path, summary: dict[str, Any], validation: dict[str, Any], by_date: pd.DataFrame, horizon_stats: pd.DataFrame, deciles: pd.DataFrame, by_year: pd.DataFrame, figures: list[dict[str, str]], env: dict[str, Any]) -> None:
    def tbl(df: pd.DataFrame) -> str:
        return df.to_html(index=False, escape=True, classes="data") if df is not None and not df.empty else "<p>No rows.</p>"

    cards = []
    for label, value in [
        ("Validation", "PASS" if validation.get("passed") else "FAIL"),
        ("Model matrix rows", summary.get("model_matrix_rows")),
        ("Target GVKEYs", summary.get("target_gvkeys")),
        ("Pure spillover rows", validation.get("pure_spillover_rows")),
        ("CRSP return rows", validation.get("crsp_return_rows")),
        ("5d label coverage", validation.get("label_coverage", {}).get("label_cov_5d")),
    ]:
        if isinstance(value, float):
            txt = f"{value:.2%}" if 0 <= value <= 1 else f"{value:,.4f}"
        elif isinstance(value, int):
            txt = f"{value:,}"
        else:
            txt = html.escape(str(value))
        cards.append(f"<div class='card'><div class='kicker'>{html.escape(label)}</div><h3>{txt}</h3></div>")

    fig_html = []
    for fig in figures:
        rel = "figures/" + Path(fig["path"]).name
        fig_html.append(f"<div class='figure'><h3>{html.escape(fig['figure'].replace('_',' ').title())}</h3><img src='{html.escape(rel)}'></div>")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 4.1 Full Labels</title>
<style>
:root {{ --bg:#07111f; --text:#eef6ff; --muted:#9fb7ce; --line:rgba(255,255,255,.14); }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#183b66,var(--bg) 42%); color:var(--text); }}
header {{ padding:46px 56px 28px; border-bottom:1px solid var(--line); }} h1 {{ margin:0; font-size:42px; letter-spacing:-.04em; }} .subtitle {{ color:var(--muted); font-size:17px; max-width:1050px; line-height:1.55; }}
main {{ padding:28px 56px 60px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:18px; margin:22px 0 36px; }} .card {{ background:linear-gradient(180deg,rgba(255,255,255,.075),rgba(255,255,255,.035)); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 18px 40px rgba(0,0,0,.18); }} .card h3 {{ margin:7px 0 0; font-size:28px; }} .kicker {{ text-transform:uppercase; font-size:11px; letter-spacing:.16em; color:var(--muted); }}
section {{ background:rgba(15,30,51,.78); border:1px solid var(--line); border-radius:22px; padding:24px; margin:22px 0; overflow:auto; }} table.data {{ width:100%; border-collapse:collapse; font-size:13px; }} table.data th {{ text-align:left; color:#d8eaff; background:rgba(255,255,255,.08); }} table.data th, table.data td {{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.09); vertical-align:top; }} .figure img {{ width:100%; max-width:1100px; border-radius:16px; border:1px solid var(--line); background:white; }} pre {{ white-space:pre-wrap; background:rgba(0,0,0,.28); border:1px solid var(--line); border-radius:14px; padding:16px; color:#dbecff; }}
</style></head><body><header><h1>Phase 4.1 Full CRSP Labels</h1><p class="subtitle">Full-scale production-network spillover features, CRSP market-adjusted forward labels, and no-lookahead validations. Protected CRSP and model-matrix Parquet files remain local and outside the upload bundle.</p></header><main>
<div class="grid">{''.join(cards)}</div>
<section><h2>Validation</h2>{tbl(pd.DataFrame([validation]))}</section>
<section><h2>Year summary</h2>{tbl(by_year)}</section>
<section><h2>Horizon statistics</h2>{tbl(horizon_stats)}</section>
<section><h2>5-day decile check</h2>{tbl(deciles)}</section>
<section><h2>Daily aggregate summary sample</h2>{tbl(by_date.head(120))}</section>
<section><h2>Figures</h2>{''.join(fig_html)}</section>
<section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main></body></html>"""
    out_path.write_text(doc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--sample-start", default=DEFAULT_SAMPLE_START)
    parser.add_argument("--sample-end", default=DEFAULT_SAMPLE_END)
    parser.add_argument("--crsp-start", default=DEFAULT_CRSP_START)
    parser.add_argument("--crsp-end", default=DEFAULT_CRSP_END)
    parser.add_argument("--sql-chunk-size", type=int, default=DEFAULT_SQL_CHUNK)
    parser.add_argument("--max-spillover-rows-per-year", type=int, default=DEFAULT_MAX_SPILLOVER_ROWS_PER_YEAR)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    raw_dir = root / "data" / "raw" / "wrds" / "phase4_returns_full"
    processed_dir = root / "data" / "processed" / "model_matrix_full"
    features_dir = processed_dir / "spillover_features_by_year"
    matrix_dir = processed_dir / "model_matrix_by_year"
    edges_path = root / "data" / "processed" / "graph_backbone" / "edges_supplier_customer_common_us.parquet"
    news_path = root / "data" / "processed" / "news_shocks_full" / "node_day_news_shocks.parquet"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 4.1 CRSP labels/model matrix FULL SCALE")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {root}")
    print(f"Graph edges: {edges_path}")
    print(f"News shocks: {news_path}")
    print(f"Output dir: {out_dir}")
    print(f"Protected raw CRSP dir: {raw_dir}")
    print(f"Sample: {args.sample_start} to {args.sample_end}")
    print(f"CRSP query: {args.crsp_start} to {args.crsp_end}")
    print("=" * 80)

    env = {
        "utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "thread_env": {k: os.environ.get(k) for k in ["PNA_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]},
        "packages": package_status(),
        "git": run_cmd(["bash", "-lc", "command -v git || true"]),
    }
    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    db = None
    try:
        edges = read_required(edges_path)
        news = read_required(news_path)
        targets = build_target_universe(edges, news, args.sample_start, args.sample_end, out_dir)
        feature_paths, feature_year_stats = build_all_spillover_features(
            edges, news, targets, args.sample_start, args.sample_end,
            features_dir, out_dir, args.max_spillover_rows_per_year,
        )
        if not feature_paths:
            raise RuntimeError("No full-scale spillover feature partitions were created.")
        permnos = set(pd.to_numeric(targets["permno"], errors="coerce").dropna().astype(int).tolist())
        for path in feature_paths:
            # Read just the target permno column. Parquet column projection keeps this cheap.
            f_perm = pd.read_parquet(path, columns=["target_permno"])
            permnos |= set(pd.to_numeric(f_perm["target_permno"], errors="coerce").dropna().astype(int).tolist())
        permnos = sorted(permnos)
        print(f"[INFO] Full-scale CRSP permno universe: {len(permnos):,}")
        db, user, source = connect_wrds()
        print(f"[INFO] WRDS connected user={mask_user(user)} source={source}")
        crsp = query_crsp_returns(db, permnos, args.crsp_start, args.crsp_end, raw_dir, args.sql_chunk_size, args.force_refresh)
    except Exception:
        print("[ERROR] Phase 4.1 full-scale extraction/features failed.")
        print(traceback.format_exc())
        return 2
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    try:
        ret_panel, label_panel, return_diagnostics = prepare_returns(crsp, processed_dir)
        market_dates = crsp["dsi"]["date"]
        matrix_paths = []
        matrix_year_stats = []
        by_date_parts = []
        matrix_samples = []
        for path in feature_paths:
            year = int(path.stem.split("_")[-1])
            print(f"[MATRIX] Building model matrix year={year} from {path.name} ...")
            features = pd.read_parquet(path)
            features_mapped = map_signal_to_trading_day(features, market_dates)
            matrix_y = build_model_matrix(features_mapped, label_panel)
            matrix_y["signal_year"] = year
            out_path = matrix_dir / f"model_matrix_{year}.parquet"
            write_parquet(matrix_y, out_path)
            matrix_paths.append(out_path)
            matrix_year_stats.append({
                "signal_year": year,
                "matrix_rows": int(len(matrix_y)),
                "pure_spillover_rows": int(matrix_y["pure_spillover_no_own_news"].fillna(False).sum()),
                "target_gvkeys": int(matrix_y["target_gvkey"].nunique()),
                "label_cov_5d": float(matrix_y["fwd_abret_5d"].notna().mean()),
                "label_cov_20d": float(matrix_y["fwd_abret_20d"].notna().mean()),
            })
            by_date_part, _, _ = summarize_matrix(matrix_y)
            by_date_part["signal_year"] = year
            by_date_parts.append(by_date_part)
            if len(matrix_samples) < 5:
                matrix_samples.append(matrix_y.head(20000))
            print(f"[MATRIX] year={year}: rows={len(matrix_y):,}")

        matrix = pd.concat([pd.read_parquet(p) for p in matrix_paths], ignore_index=True)
        by_date, horizon_stats, deciles = summarize_matrix(matrix)
        by_year = pd.DataFrame(matrix_year_stats).sort_values("signal_year")
        validation = validation_report(matrix, len(ret_panel), int(feature_year_stats["feature_rows"].sum()), len(targets), return_diagnostics)
        matrix_sample = pd.concat(matrix_samples, ignore_index=True) if matrix_samples else matrix.head(50000)
    except Exception:
        print("[ERROR] Phase 4.1 full-scale labels/model matrix construction failed.")
        print(traceback.format_exc())
        return 3

    # Store aggregate artifacts only in upload bundle. Protected matrices remain local under data/processed.
    by_date.to_csv(out_dir / "phase4_1_daily_aggregate_summary.csv", index=False)
    horizon_stats.to_csv(out_dir / "phase4_1_horizon_stats.csv", index=False)
    deciles.to_csv(out_dir / "phase4_1_decile_check_5d.csv", index=False)
    by_year.to_csv(out_dir / "phase4_1_matrix_year_summary.csv", index=False)
    pd.DataFrame([return_diagnostics]).to_csv(out_dir / "phase4_1_return_diagnostics.csv", index=False)
    redact_identifiers(matrix_sample.head(10000)).to_csv(out_dir / "phase4_1_model_matrix_redacted_sample.csv", index=False)
    pd.DataFrame([validation]).to_csv(out_dir / "phase4_1_validation_report_flat.csv", index=False)
    with (out_dir / "phase4_1_validation_report.json").open("w") as f:
        json.dump(validation, f, indent=2, default=str)

    protected_inventory = []
    for base in [raw_dir, processed_dir]:
        for p in sorted(base.rglob("*.parquet")):
            protected_inventory.append({"file": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p), "protected_local_only": True})
    pd.DataFrame(protected_inventory).to_csv(out_dir / "protected_local_phase4_1_inventory.csv", index=False)

    summary = {
        "generated_at_utc": utc_now(),
        "sample_start": args.sample_start,
        "sample_end": args.sample_end,
        "validation_passed": validation["passed"],
        "target_gvkeys": int(targets["gvkey"].nunique()),
        "model_matrix_rows": int(len(matrix)),
        "pure_spillover_rows": int(validation["pure_spillover_rows"]),
        "crsp_return_rows": int(validation["crsp_return_rows"]),
        "feature_partitions": len(feature_paths),
        "matrix_partitions": len(matrix_paths),
        "protected_model_matrix_dir": str(matrix_dir),
        "protected_return_panel": str(processed_dir / "full_crsp_return_panel_features.parquet"),
        "return_diagnostics": return_diagnostics,
    }
    with (out_dir / "phase4_1_quality_summary.json").open("w") as f:
        json.dump({"summary": summary, "validation": validation, "feature_year_stats": feature_year_stats.to_dict(orient="records")}, f, indent=2, default=str)

    figures = make_figures(out_dir, validation, horizon_stats, deciles, by_year, matrix_sample)
    render_report(out_dir / "phase4_1_full_labels_report.html", summary, validation, by_date, horizon_stats, deciles, by_year, figures, env)

    lines = [
        "# Phase 4.1 CRSP labels/model-matrix full-scale summary", "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Validation passed: {summary['validation_passed']}",
        f"- Sample: {args.sample_start} to {args.sample_end}",
        f"- Target GVKEYs: {summary['target_gvkeys']:,}",
        f"- Model matrix rows: {summary['model_matrix_rows']:,}",
        f"- Pure spillover rows: {summary['pure_spillover_rows']:,}",
        f"- CRSP return rows: {summary['crsp_return_rows']:,}",
        f"- Feature partitions: {summary['feature_partitions']:,}",
        f"- Matrix partitions: {summary['matrix_partitions']:,}",
        f"- Infinite log_mktcap rows: {return_diagnostics.get('infinite_log_mktcap_rows')}",
        f"- Report: {out_dir / 'phase4_1_full_labels_report.html'}", "",
        "Data policy: protected CRSP and model-matrix Parquet files remain local and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE4_1_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if validation["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
