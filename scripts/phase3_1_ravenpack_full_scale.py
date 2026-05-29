from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SAMPLE_START_DEFAULT = "2015-01-01"
SAMPLE_END_DEFAULT = "2025-12-31"
NEWS_COLS = [
    "timestamp_utc",
    "rp_story_id",
    "rp_entity_id",
    "entity_type",
    "entity_name",
    "country_code",
    "relevance",
    "event_sentiment_score",
    "event_relevance",
    "event_similarity_key",
    "event_similarity_days",
    "topic",
    "group",
    "type",
    "sub_type",
    "fact_level",
    "event_start_date_utc",
    "event_end_date_utc",
    "related_entity",
    "relationship",
    "category",
    "news_type",
    "rp_source_id",
    "source_name",
    "rp_story_event_index",
    "rp_story_event_count",
]
FORBIDDEN_TEXT_COLS = {"headline", "event_text", "provider_story_id", "story_text", "body"}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def mask_user(username: str | None) -> str | None:
    if not username:
        return None
    return username[:2] + "***" + username[-1] if len(username) > 3 else username[0] + "***"


def norm(x: Any) -> str | None:
    if x is None or pd.isna(x):
        return None
    s = re.sub(r"[^A-Za-z0-9]", "", str(x).upper().strip())
    if not s or s in {"NA", "NAN", "NONE", "NULL", "<NA>"}:
        return None
    return s


def clean_gvkey(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)


def as_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def q(x: Any) -> str:
    return "'" + str(x).replace("'", "''") + "'"


def sql_in(values: list[str]) -> str:
    if not values:
        return "('')"
    return "(" + ",".join(q(v) for v in values) + ")"


def chunked(values: list[str], n: int) -> list[list[str]]:
    return [values[i : i + n] for i in range(0, len(values), n)]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        print(f"[CACHE] {path}")
        return pd.read_parquet(path)
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}


def package_status() -> list[dict[str, str]]:
    pkgs = [
        "pandas",
        "numpy",
        "pyarrow",
        "fastparquet",
        "duckdb",
        "polars",
        "wrds",
        "sqlalchemy",
        "psycopg2",
        "matplotlib",
        "plotly",
    ]
    rows = []
    for pkg in pkgs:
        try:
            rows.append({"package": pkg, "version": md.version(pkg), "status": "ok"})
        except md.PackageNotFoundError:
            rows.append({"package": pkg, "version": "", "status": "missing"})
    return rows


def detect_wrds_username() -> tuple[str | None, str]:
    env_user = os.environ.get("WRDS_USERNAME")
    if env_user:
        return env_user, "WRDS_USERNAME"
    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        try:
            for line in pgpass.read_text(errors="ignore").splitlines():
                if line and not line.startswith("#") and "wrds" in line.lower():
                    parts = line.split(":")
                    if len(parts) >= 5:
                        return parts[-2], "~/.pgpass"
            for line in pgpass.read_text(errors="ignore").splitlines():
                if line and not line.startswith("#"):
                    parts = line.split(":")
                    if len(parts) >= 5:
                        return parts[-2], "~/.pgpass"
        except Exception:
            return None, "~/.pgpass unreadable"
    return None, "not found"


def connect_wrds():
    import wrds

    user, source = detect_wrds_username()
    print(f"[INFO] WRDS user source={source}, user={mask_user(user)}")
    db = wrds.Connection(wrds_username=user, verbose=False) if user else wrds.Connection(verbose=False)
    try:
        db.raw_sql("set statement_timeout to '1800000ms'")
    except Exception:
        pass
    return db, user, source


def raw_sql(db, sql: str, label: str) -> pd.DataFrame:
    t0 = time.time()
    print(f"[WRDS] {label} ...")
    df = db.raw_sql(sql)
    df.columns = [str(c).lower() for c in df.columns]
    print(f"[WRDS] {label}: rows={len(df):,}, elapsed={time.time() - t0:.1f}s")
    return df


def build_identifier_universe(edge_path: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing corrected graph edge file: {edge_path}")
    edges = pd.read_parquet(edge_path)
    print(f"[INFO] Loaded corrected common-edge graph: rows={len(edges):,}, cols={len(edges.columns):,}")

    frames = []
    for side in ["supplier", "customer"]:
        gv_col = f"{side}_gvkey"
        if gv_col not in edges.columns:
            continue
        frame = pd.DataFrame(
            {
                "side": side,
                "gvkey": clean_gvkey(edges[gv_col]),
                "permno": as_int(edges.get(f"{side}_permno", pd.Series(pd.NA, index=edges.index))),
                "permco": as_int(edges.get(f"{side}_permco", pd.Series(pd.NA, index=edges.index))),
                "ticker": edges.get(f"{side}_ticker", pd.Series(pd.NA, index=edges.index)).astype("string"),
                "cusip_raw": edges.get(f"{side}_cusip_raw", pd.Series(pd.NA, index=edges.index)).astype("string"),
            }
        )
        degree = edges.groupby(gv_col).size().rename("network_degree").reset_index()
        degree["gvkey"] = clean_gvkey(degree[gv_col])
        frame = frame.merge(degree[["gvkey", "network_degree"]], on="gvkey", how="left")
        frames.append(frame)

    if not frames:
        raise RuntimeError("No supplier/customer node columns found in graph edge file.")

    nodes = pd.concat(frames, ignore_index=True).dropna(subset=["gvkey"]).drop_duplicates()
    nodes["network_degree"] = pd.to_numeric(nodes["network_degree"], errors="coerce").fillna(1)
    nodes = nodes.sort_values("network_degree", ascending=False)
    nodes = nodes.drop_duplicates("gvkey", keep="first").copy()
    nodes["ticker_norm"] = nodes["ticker"].map(norm)
    nodes["cusip_norm"] = nodes["cusip_raw"].map(norm)
    nodes["cusip9"] = nodes["cusip_norm"].str[:9]
    nodes["cusip8"] = nodes["cusip_norm"].str[:8]
    nodes["cusip6"] = nodes["cusip_norm"].str[:6]

    id_rows: list[dict[str, Any]] = []

    def add(row: Any, kind: str, value: Any, priority: int) -> None:
        nv = norm(value)
        if nv:
            id_rows.append(
                {
                    "gvkey": row.gvkey,
                    "permno": None if pd.isna(row.permno) else int(row.permno),
                    "id_kind": kind,
                    "id_value_norm": nv,
                    "priority": priority,
                    "network_degree": float(row.network_degree),
                }
            )

    for row in nodes.itertuples(index=False):
        add(row, "gvkey", row.gvkey, 1)
        add(row, "permno", None if pd.isna(row.permno) else str(int(row.permno)), 2)
        add(row, "cusip9", row.cusip9, 3)
        add(row, "cusip8", row.cusip8, 4)
        add(row, "cusip6", row.cusip6, 5)
        add(row, "ticker", row.ticker_norm, 6)

    ids = pd.DataFrame(id_rows).drop_duplicates()
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes.drop(columns=["cusip_norm"], errors="ignore").to_csv(out_dir / "full_nodes_redacted.csv", index=False)
    ids.drop(columns=["id_value_norm"], errors="ignore").to_csv(out_dir / "full_identifier_rows_redacted.csv", index=False)
    ids.groupby("id_kind").size().rename("n_rows").reset_index().to_csv(out_dir / "full_identifier_counts.csv", index=False)
    print(f"[INFO] Node universe: gvkeys={nodes['gvkey'].nunique():,}, identifier rows={len(ids):,}")
    return nodes, ids


def classify_dtype(x: Any) -> str | None:
    z = norm(x)
    if not z:
        return None
    if "GVKEY" in z:
        return "gvkey"
    if "PERMNO" in z:
        return "permno"
    if "CUSIP" in z:
        return "cusip"
    if "TICKER" in z or "SYMBOL" in z:
        return "ticker"
    return None


def query_mappings(db, ids: pd.DataFrame, raw_dir: Path, chunk_size: int, force_refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    type_path = raw_dir / "mapping_data_type_counts.parquet"
    long_path = raw_dir / "mapping_long_filtered.parquet"
    fallback_path = raw_dir / "mapping_fallback_filtered.parquet"

    if not force_refresh and type_path.exists() and long_path.exists() and fallback_path.exists():
        print("[CACHE] Using cached RavenPack mapping extracts.")
        return pd.read_parquet(type_path), pd.read_parquet(long_path), pd.read_parquet(fallback_path)

    type_counts = raw_sql(
        db,
        """
        select data_type, count(*) as n_rows
        from ravenpack_common.rpa_company_mappings
        group by data_type
        order by n_rows desc, data_type
        """,
        "RavenPack mapping data_type counts",
    )
    type_counts["match_kind"] = type_counts["data_type"].map(classify_dtype)
    data_types = sorted(type_counts.loc[type_counts["match_kind"].notna(), "data_type"].dropna().astype(str).unique())
    if not data_types:
        raise RuntimeError("No useful data_type values found in rpa_company_mappings.")

    values = sorted(ids["id_value_norm"].dropna().astype(str).unique())
    frames = []
    type_in = sql_in(data_types)
    value_chunks = chunked(values, chunk_size)
    for idx, vals in enumerate(value_chunks, start=1):
        sql = f"""
            select rp_entity_id, entity_type, data_type, data_value, range_start, range_end
            from ravenpack_common.rpa_company_mappings
            where data_type in {type_in}
              and upper(regexp_replace(coalesce(data_value,''), '[^A-Za-z0-9]', '', 'g')) in {sql_in(vals)}
        """
        frames.append(raw_sql(db, sql, f"rpa_company_mappings filtered chunk {idx}/{len(value_chunks)}"))
    long_map = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fb_frames = []
    for idx, vals in enumerate(value_chunks, start=1):
        in_list = sql_in(vals)
        sql = f"""
            select rp_entity_id, entity_type, entity_name, ticker, cusip, isin
            from ravenpack_common.wrds_rpa_company_mappings
            where upper(regexp_replace(coalesce(ticker,''), '[^A-Za-z0-9]', '', 'g')) in {in_list}
               or upper(regexp_replace(coalesce(cusip,''), '[^A-Za-z0-9]', '', 'g')) in {in_list}
        """
        try:
            fb_frames.append(raw_sql(db, sql, f"wrds_rpa_company_mappings fallback chunk {idx}/{len(value_chunks)}"))
        except Exception as exc:
            print(f"[WARN] Fallback mapping chunk {idx} failed: {exc}")
    fallback = pd.concat(fb_frames, ignore_index=True) if fb_frames else pd.DataFrame()

    write_parquet(type_counts, type_path)
    write_parquet(long_map, long_path)
    write_parquet(fallback, fallback_path)
    return type_counts, long_map, fallback


def build_entity_map(ids: pd.DataFrame, long_map: pd.DataFrame, fallback: pd.DataFrame, processed_dir: Path, out_dir: Path) -> pd.DataFrame:
    processed_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    ids = ids.copy()
    ids["id_value_norm"] = ids["id_value_norm"].astype(str)

    if not long_map.empty:
        lm = long_map.copy()
        lm["data_value_norm"] = lm["data_value"].map(norm)
        lm["map_kind"] = lm["data_type"].map(classify_dtype)
        lm["range_start"] = pd.to_datetime(lm["range_start"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
        lm["range_end"] = pd.to_datetime(lm["range_end"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
        specs = [("gvkey", ["gvkey"]), ("permno", ["permno"]), ("cusip", ["cusip9", "cusip8", "cusip6"]), ("ticker", ["ticker"])]
        for map_kind, id_kinds in specs:
            left = lm[lm["map_kind"].eq(map_kind)]
            right = ids[ids["id_kind"].isin(id_kinds)]
            if not left.empty and not right.empty:
                joined = left.merge(right, left_on="data_value_norm", right_on="id_value_norm", how="inner")
                if not joined.empty:
                    joined["mapping_source"] = "rpa_company_mappings"
                    pieces.append(joined)

    if not fallback.empty:
        fb = fallback.copy()
        fb["range_start"] = pd.Timestamp("1900-01-01")
        fb["range_end"] = pd.Timestamp("2099-12-31")
        for col, id_kinds, bump in [("cusip", ["cusip9", "cusip8", "cusip6"], 20), ("ticker", ["ticker"], 30)]:
            if col not in fb.columns:
                continue
            left = fb.copy()
            left["data_type"] = col.upper()
            left["data_value"] = left[col]
            left["data_value_norm"] = left[col].map(norm)
            right = ids[ids["id_kind"].isin(id_kinds)]
            joined = left.merge(right, left_on="data_value_norm", right_on="id_value_norm", how="inner")
            if not joined.empty:
                joined["priority"] = pd.to_numeric(joined["priority"], errors="coerce").fillna(999).astype(int) + bump
                joined["mapping_source"] = "wrds_rpa_company_mappings"
                pieces.append(joined)

    if not pieces:
        raise RuntimeError("No RavenPack entity-node mappings found for the graph universe.")

    em = pd.concat(pieces, ignore_index=True)
    keep = [
        "rp_entity_id",
        "entity_type",
        "gvkey",
        "permno",
        "id_kind",
        "id_value_norm",
        "data_type",
        "range_start",
        "range_end",
        "priority",
        "network_degree",
        "mapping_source",
    ]
    for col in keep:
        if col not in em.columns:
            em[col] = pd.NA
    em = em[keep].dropna(subset=["rp_entity_id", "gvkey"]).copy()
    em["gvkey"] = clean_gvkey(em["gvkey"])
    em["permno"] = as_int(em["permno"])
    em["rp_entity_id"] = em["rp_entity_id"].astype("string")
    em["priority"] = pd.to_numeric(em["priority"], errors="coerce").fillna(999).astype(int)
    em["network_degree"] = pd.to_numeric(em["network_degree"], errors="coerce").fillna(0)
    em["range_start"] = pd.to_datetime(em["range_start"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    em["range_end"] = pd.to_datetime(em["range_end"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    em = em.sort_values(["gvkey", "rp_entity_id", "priority", "network_degree"], ascending=[True, True, True, False])
    em = em.drop_duplicates(["gvkey", "rp_entity_id"], keep="first")

    write_parquet(em, processed_dir / "ravenpack_entity_node_map.parquet")

    summary = (
        em.groupby(["mapping_source", "id_kind", "data_type"], dropna=False)
        .agg(rows=("rp_entity_id", "size"), gvkeys=("gvkey", "nunique"), rp_entities=("rp_entity_id", "nunique"))
        .reset_index()
        .sort_values(["rows", "gvkeys"], ascending=False)
    )
    summary.to_csv(out_dir / "full_mapping_summary_by_source.csv", index=False)
    print(f"[INFO] Entity map: rows={len(em):,}, gvkeys={em['gvkey'].nunique():,}, rp_entities={em['rp_entity_id'].nunique():,}")
    return em


def add_market_time_columns(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return news
    n = news.copy()
    ts = pd.to_datetime(n["timestamp_utc"], errors="coerce", utc=True)
    n = n.loc[ts.notna()].copy()
    ts = ts.loc[ts.notna()]
    n["timestamp_utc"] = ts.dt.tz_localize(None)
    ts_et = ts.dt.tz_convert(ZoneInfo("America/New_York"))
    n["timestamp_et"] = ts_et.dt.tz_localize(None)
    n["market_date"] = n["timestamp_et"].dt.date.astype(str)
    minutes = n["timestamp_et"].dt.hour * 60 + n["timestamp_et"].dt.minute
    n["market_time_bucket"] = np.select([minutes < 570, minutes <= 960], ["pre_open", "regular_session"], default="post_close")
    signal_dt = pd.to_datetime(n["market_date"], errors="coerce")
    signal_dt = signal_dt.where(~n["market_time_bucket"].eq("post_close"), signal_dt + pd.Timedelta(days=1))
    n["news_signal_date"] = signal_dt.dt.date.astype(str)
    return n


def event_key_series(df: pd.DataFrame) -> pd.Series:
    for col in ["rp_story_id", "rp_entity_id", "event_similarity_key", "topic", "group", "type", "sub_type"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("string").fillna("")
    return np.where(
        df["event_similarity_key"].str.len() > 0,
        df["event_similarity_key"],
        df["topic"]
        + "|"
        + df["group"]
        + "|"
        + df["type"]
        + "|"
        + df["sub_type"]
        + "|"
        + df["timestamp_utc"].astype(str),
    )


def process_raw_chunk(raw_news: pd.DataFrame, entity_map: pd.DataFrame, mapped_path: Path) -> dict[str, Any]:
    if raw_news.empty:
        write_parquet(pd.DataFrame(), mapped_path)
        return {"raw_rows": 0, "entity_dedup_rows": 0, "mapped_rows": 0, "forbidden_text_columns_absent": True}

    forbidden_absent = not bool(set(raw_news.columns) & FORBIDDEN_TEXT_COLS)
    n = add_market_time_columns(raw_news)
    for col in ["relevance", "event_relevance", "event_sentiment_score", "event_similarity_days"]:
        if col not in n.columns:
            n[col] = np.nan
        n[col] = pd.to_numeric(n[col], errors="coerce")

    n["event_key"] = event_key_series(n)
    n["dedup_entity_key"] = n["rp_entity_id"].astype("string") + "|" + n["rp_story_id"].astype("string") + "|" + pd.Series(n["event_key"], index=n.index).astype("string")
    n = n.sort_values(["dedup_entity_key", "relevance", "event_relevance", "timestamp_utc"], ascending=[True, False, False, True])
    n = n.drop_duplicates("dedup_entity_key", keep="first")

    em = entity_map[["rp_entity_id", "gvkey", "permno", "range_start", "range_end", "priority", "mapping_source"]].copy()
    em["rp_entity_id"] = em["rp_entity_id"].astype("string")
    mapped = n.merge(em, on="rp_entity_id", how="inner")
    if mapped.empty:
        write_parquet(mapped, mapped_path)
        return {"raw_rows": int(len(raw_news)), "entity_dedup_rows": int(len(n)), "mapped_rows": 0, "forbidden_text_columns_absent": forbidden_absent}

    event_day = pd.to_datetime(mapped["timestamp_utc"], errors="coerce").dt.floor("D")
    mapped["range_start"] = pd.to_datetime(mapped["range_start"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    mapped["range_end"] = pd.to_datetime(mapped["range_end"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    mapped = mapped[(event_day >= mapped["range_start"]) & (event_day <= mapped["range_end"])].copy()

    if mapped.empty:
        write_parquet(mapped, mapped_path)
        return {"raw_rows": int(len(raw_news)), "entity_dedup_rows": int(len(n)), "mapped_rows": 0, "forbidden_text_columns_absent": forbidden_absent}

    mapped["sentiment_scaled"] = mapped["event_sentiment_score"] / 100.0
    novelty = 1.0 / (1.0 + mapped["event_similarity_days"].fillna(0).clip(lower=0))
    mapped["attention_weight"] = (mapped["relevance"].fillna(0) / 100.0) * (mapped["event_relevance"].fillna(0) / 100.0)
    mapped["signed_news_shock"] = mapped["sentiment_scaled"] * mapped["attention_weight"] * novelty
    mapped["positive_news_shock"] = mapped["signed_news_shock"].clip(lower=0)
    mapped["negative_news_shock"] = mapped["signed_news_shock"].clip(upper=0)
    mapped["abs_news_shock"] = mapped["signed_news_shock"].abs()
    mapped["is_negative_news"] = mapped["signed_news_shock"] < 0
    mapped["is_positive_news"] = mapped["signed_news_shock"] > 0

    keep_cols = [
        "timestamp_utc",
        "timestamp_et",
        "market_date",
        "market_time_bucket",
        "news_signal_date",
        "rp_story_id",
        "rp_entity_id",
        "gvkey",
        "permno",
        "event_key",
        "relevance",
        "event_relevance",
        "event_sentiment_score",
        "event_similarity_days",
        "topic",
        "group",
        "type",
        "sub_type",
        "category",
        "news_type",
        "source_name",
        "priority",
        "mapping_source",
        "signed_news_shock",
        "positive_news_shock",
        "negative_news_shock",
        "abs_news_shock",
        "is_negative_news",
        "is_positive_news",
    ]
    for col in keep_cols:
        if col not in mapped.columns:
            mapped[col] = pd.NA
    mapped = mapped[keep_cols]
    write_parquet(mapped, mapped_path)
    return {
        "raw_rows": int(len(raw_news)),
        "entity_dedup_rows": int(len(n)),
        "mapped_rows": int(len(mapped)),
        "forbidden_text_columns_absent": forbidden_absent,
    }


def extract_and_process_news(
    db,
    entity_map: pd.DataFrame,
    raw_dir: Path,
    processed_dir: Path,
    start_year: int,
    end_year: int,
    entity_chunk_size: int,
    min_relevance: float,
    min_event_relevance: float,
    force_refresh: bool,
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = raw_dir / "ravenpack_equities_chunks"
    mapped_chunk_dir = processed_dir / "mapped_event_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    mapped_chunk_dir.mkdir(parents=True, exist_ok=True)

    selected = ",\n                ".join(f'"{col}"' for col in NEWS_COLS)
    entity_ids = sorted(entity_map["rp_entity_id"].dropna().astype(str).unique().tolist())
    entity_chunks = chunked(entity_ids, entity_chunk_size)
    if not entity_chunks:
        raise RuntimeError("No RavenPack entity IDs available for full extraction.")

    manifest_rows: list[dict[str, Any]] = []
    total_queries = (end_year - start_year + 1) * len(entity_chunks)
    qnum = 0

    for year in range(start_year, end_year + 1):
        table = f"ravenpack_dj.rpa_djpr_equities_{year}"
        for chunk_idx, ids in enumerate(entity_chunks, start=1):
            qnum += 1
            raw_path = chunk_dir / f"ravenpack_equities_{year}_chunk{chunk_idx:04d}.parquet"
            mapped_path = mapped_chunk_dir / f"mapped_events_{year}_chunk{chunk_idx:04d}.parquet"
            label = f"RavenPack {year} chunk {chunk_idx}/{len(entity_chunks)} query {qnum}/{total_queries}"

            if raw_path.exists() and mapped_path.exists() and not force_refresh:
                print(f"[CACHE] {label}: raw and mapped chunks exist")
                raw_rows = int(pd.read_parquet(raw_path, columns=["rp_entity_id"]).shape[0]) if raw_path.stat().st_size > 0 else 0
                mapped_rows = int(pd.read_parquet(mapped_path, columns=["gvkey"]).shape[0]) if mapped_path.stat().st_size > 0 else 0
                manifest_rows.append(
                    {
                        "year": year,
                        "chunk": chunk_idx,
                        "entity_ids": len(ids),
                        "raw_rows": raw_rows,
                        "entity_dedup_rows": None,
                        "mapped_rows": mapped_rows,
                        "raw_path": str(raw_path),
                        "mapped_path": str(mapped_path),
                        "used_cache": True,
                        "forbidden_text_columns_absent": True,
                    }
                )
                continue

            sql = f"""
                select
                    {selected}
                from {table}
                where rp_entity_id in {sql_in(ids)}
                  and timestamp_utc >= timestamp '{year}-01-01 00:00:00'
                  and timestamp_utc < timestamp '{year + 1}-01-01 00:00:00'
                  and coalesce(relevance, 0) >= {float(min_relevance)}
                  and coalesce(event_relevance, 0) >= {float(min_event_relevance)}
                  and event_sentiment_score is not null
            """
            try:
                raw_news = raw_sql(db, sql, label)
            except Exception as exc:
                print(f"[ERROR] Query failed for {year} chunk {chunk_idx}: {exc}")
                raise

            write_parquet(raw_news, raw_path)
            metrics = process_raw_chunk(raw_news, entity_map, mapped_path)
            manifest_rows.append(
                {
                    "year": year,
                    "chunk": chunk_idx,
                    "entity_ids": len(ids),
                    "raw_rows": metrics["raw_rows"],
                    "entity_dedup_rows": metrics["entity_dedup_rows"],
                    "mapped_rows": metrics["mapped_rows"],
                    "raw_path": str(raw_path),
                    "mapped_path": str(mapped_path),
                    "used_cache": False,
                    "forbidden_text_columns_absent": metrics["forbidden_text_columns_absent"],
                }
            )
            del raw_news

    manifest = pd.DataFrame(manifest_rows)
    return manifest


def load_all_mapped_chunks(processed_dir: Path) -> pd.DataFrame:
    mapped_files = sorted((processed_dir / "mapped_event_chunks").glob("mapped_events_*_chunk*.parquet"))
    if not mapped_files:
        return pd.DataFrame()
    frames = []
    for idx, path in enumerate(mapped_files, start=1):
        df = pd.read_parquet(path)
        if not df.empty:
            frames.append(df)
        if idx % 25 == 0:
            print(f"[INFO] Loaded {idx:,}/{len(mapped_files):,} mapped chunks")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def finalize_dedup_and_node_day(mapped: pd.DataFrame, processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mapped.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"[INFO] Final global mapped-event dedup input rows: {len(mapped):,}")
    mapped["gvkey"] = clean_gvkey(mapped["gvkey"])
    mapped["permno"] = as_int(mapped["permno"])
    mapped["timestamp_utc"] = pd.to_datetime(mapped["timestamp_utc"], errors="coerce")
    mapped["relevance"] = pd.to_numeric(mapped["relevance"], errors="coerce")
    mapped["event_relevance"] = pd.to_numeric(mapped["event_relevance"], errors="coerce")
    mapped["priority"] = pd.to_numeric(mapped["priority"], errors="coerce").fillna(999).astype(int)
    mapped["gvkey_story_event_key"] = mapped["gvkey"].astype("string") + "|" + mapped["rp_story_id"].astype("string") + "|" + mapped["event_key"].astype("string")
    mapped = mapped.sort_values(
        ["gvkey_story_event_key", "priority", "relevance", "event_relevance", "timestamp_utc"],
        ascending=[True, True, False, False, True],
    )
    dedup = mapped.drop_duplicates("gvkey_story_event_key", keep="first").copy()
    print(f"[INFO] Final global mapped-event rows after gvkey/story/event dedup: {len(dedup):,}")

    for col in ["signed_news_shock", "positive_news_shock", "negative_news_shock", "abs_news_shock"]:
        dedup[col] = pd.to_numeric(dedup[col], errors="coerce").fillna(0.0)
    dedup["is_negative_news"] = dedup["signed_news_shock"] < 0
    dedup["is_positive_news"] = dedup["signed_news_shock"] > 0
    dedup["news_signal_date"] = pd.to_datetime(dedup["news_signal_date"], errors="coerce").dt.date.astype(str)
    dedup["signal_year"] = pd.to_datetime(dedup["news_signal_date"], errors="coerce").dt.year.astype("Int64")

    node_day = (
        dedup.groupby(["gvkey", "permno", "news_signal_date"], dropna=False)
        .agg(
            n_events=("rp_story_id", "size"),
            n_stories=("rp_story_id", "nunique"),
            n_rp_entities=("rp_entity_id", "nunique"),
            signed_news_shock=("signed_news_shock", "sum"),
            positive_news_shock=("positive_news_shock", "sum"),
            negative_news_shock=("negative_news_shock", "sum"),
            abs_news_shock=("abs_news_shock", "sum"),
            n_negative_events=("is_negative_news", "sum"),
            n_positive_events=("is_positive_news", "sum"),
            mean_relevance=("relevance", "mean"),
            max_relevance=("relevance", "max"),
            n_pre_open=("market_time_bucket", lambda x: int((x == "pre_open").sum())),
            n_regular_session=("market_time_bucket", lambda x: int((x == "regular_session").sum())),
            n_post_close=("market_time_bucket", lambda x: int((x == "post_close").sum())),
        )
        .reset_index()
    )
    node_day["signal_year"] = pd.to_datetime(node_day["news_signal_date"], errors="coerce").dt.year.astype("Int64")

    event_class = (
        dedup.groupby(["group", "type", "sub_type"], dropna=False)
        .agg(
            n_events=("rp_story_id", "size"),
            n_stories=("rp_story_id", "nunique"),
            n_gvkeys=("gvkey", "nunique"),
            abs_news_shock=("abs_news_shock", "sum"),
            signed_news_shock=("signed_news_shock", "sum"),
        )
        .reset_index()
        .sort_values("n_events", ascending=False)
    )

    by_year = (
        dedup.groupby("signal_year", dropna=False)
        .agg(
            mapped_events_dedup=("rp_story_id", "size"),
            stories=("rp_story_id", "nunique"),
            gvkeys=("gvkey", "nunique"),
            abs_news_shock=("abs_news_shock", "sum"),
            signed_news_shock=("signed_news_shock", "sum"),
        )
        .reset_index()
        .sort_values("signal_year")
    )

    write_parquet(dedup, processed_dir / "mapped_events_deduplicated.parquet")
    write_parquet(node_day, processed_dir / "node_day_news_shocks.parquet")
    write_parquet(event_class, processed_dir / "event_class_summary.parquet")
    write_parquet(by_year, processed_dir / "mapped_event_year_summary.parquet")
    return dedup, node_day, event_class, by_year


def make_figures(out_dir: Path, manifest: pd.DataFrame, by_year: pd.DataFrame, node_day: pd.DataFrame, event_class: pd.DataFrame, validation: dict[str, Any]) -> list[dict[str, str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping figures: {exc}")
        return []

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, str]] = []

    if validation.get("checks"):
        checks = validation["checks"]
        plt.figure(figsize=(10.5, 5.8))
        labels = list(checks.keys())
        vals = [1 if checks[k] else 0 for k in labels]
        plt.barh(labels, vals)
        plt.xlim(0, 1.1)
        plt.xlabel("Pass = 1")
        plt.title("Phase 3.1 full-scale validation checks")
        plt.tight_layout()
        path = fig_dir / "phase3_1_validation_checks.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "validation_checks", "path": str(path)})

    if not manifest.empty:
        yearly_raw = manifest.groupby("year", dropna=False).agg(raw_rows=("raw_rows", "sum"), mapped_rows=("mapped_rows", "sum")).reset_index()
        plt.figure(figsize=(11, 6))
        plt.plot(yearly_raw["year"], yearly_raw["raw_rows"], marker="o", label="Raw filtered RavenPack rows")
        plt.plot(yearly_raw["year"], yearly_raw["mapped_rows"], marker="o", label="Mapped rows before global dedup")
        plt.xlabel("RavenPack partition year")
        plt.ylabel("Rows")
        plt.title("Full-scale RavenPack extraction rows by year")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "phase3_1_rows_by_year.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "rows_by_year", "path": str(path)})

    if not node_day.empty:
        nd_year = node_day.groupby("signal_year", dropna=False).agg(node_day_rows=("gvkey", "size"), gvkeys=("gvkey", "nunique"), abs_news_shock=("abs_news_shock", "sum")).reset_index()
        plt.figure(figsize=(11, 6))
        plt.plot(nd_year["signal_year"], nd_year["node_day_rows"], marker="o", label="Node-day rows")
        plt.plot(nd_year["signal_year"], nd_year["gvkeys"], marker="o", label="GVKEYs with shocks")
        plt.xlabel("Signal year")
        plt.ylabel("Count")
        plt.title("Full-scale node-day news shock coverage")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "phase3_1_node_day_coverage_by_year.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "node_day_coverage_by_year", "path": str(path)})

        vals = pd.to_numeric(node_day["signed_news_shock"], errors="coerce").dropna()
        if not vals.empty:
            cap = vals.abs().quantile(0.995)
            plt.figure(figsize=(10.5, 5.8))
            plt.hist(vals.clip(-cap, cap), bins=80)
            plt.xlabel("Node-day signed news shock, clipped at p99.5 absolute value")
            plt.ylabel("Node-days")
            plt.title("Full-scale node-day signed news shock distribution")
            plt.tight_layout()
            path = fig_dir / "phase3_1_node_day_shock_distribution.png"
            plt.savefig(path, dpi=180)
            plt.close()
            made.append({"figure": "node_day_shock_distribution", "path": str(path)})

    if not event_class.empty:
        top = event_class.head(25).copy()
        labels = top[["group", "type", "sub_type"]].fillna("NA").astype(str).agg(" / ".join, axis=1)
        plt.figure(figsize=(12, 8))
        plt.barh(labels, top["n_events"])
        plt.gca().invert_yaxis()
        plt.xlabel("Deduplicated mapped events")
        plt.title("Top full-scale RavenPack event classes")
        plt.tight_layout()
        path = fig_dir / "phase3_1_top_event_classes.png"
        plt.savefig(path, dpi=180)
        plt.close()
        made.append({"figure": "top_event_classes", "path": str(path)})

    return made


def render_report(
    out_path: Path,
    summary: dict[str, Any],
    validation: dict[str, Any],
    mapping_summary: pd.DataFrame,
    manifest_year: pd.DataFrame,
    node_day_summary: pd.DataFrame,
    event_class: pd.DataFrame,
    figures: list[dict[str, str]],
    env: dict[str, Any],
) -> None:
    def table(df: pd.DataFrame) -> str:
        return df.to_html(index=False, escape=True, classes="data") if df is not None and not df.empty else "<p>No rows.</p>"

    cards = []
    for label, value in [
        ("Validation", "PASS" if validation.get("passed") else "FAIL"),
        ("Target GVKEYs", summary.get("target_gvkeys")),
        ("Mapped GVKEYs", summary.get("mapped_gvkeys")),
        ("Mapped RP entities", summary.get("mapped_rp_entities")),
        ("Raw filtered rows", summary.get("raw_rows")),
        ("Dedup mapped events", summary.get("mapped_events_dedup")),
        ("Node-day rows", summary.get("node_day_rows")),
        ("Node-day GVKEYs", summary.get("node_day_gvkeys")),
    ]:
        txt = f"{value:,}" if isinstance(value, int) else html.escape(str(value))
        cards.append(f"<div class='card'><div class='kicker'>{html.escape(label)}</div><h3>{txt}</h3></div>")

    fig_html = []
    for fig in figures:
        rel = "figures/" + Path(fig["path"]).name
        fig_html.append(f"<div class='figure'><h3>{html.escape(fig['figure'].replace('_', ' ').title())}</h3><img src='{html.escape(rel)}'></div>")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 3.1 RavenPack Full Scale</title>
<style>
:root {{ --bg:#07111f; --text:#eef6ff; --muted:#9fb7ce; --line:rgba(255,255,255,.14); }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#183b66,var(--bg) 42%); color:var(--text); }}
header {{ padding:46px 56px 28px; border-bottom:1px solid var(--line); }} h1 {{ margin:0; font-size:42px; letter-spacing:-.04em; }} .subtitle {{ color:var(--muted); font-size:17px; max-width:1100px; line-height:1.55; }}
main {{ padding:28px 56px 60px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:18px; margin:22px 0 36px; }} .card {{ background:linear-gradient(180deg,rgba(255,255,255,.075),rgba(255,255,255,.035)); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 18px 40px rgba(0,0,0,.18); }} .card h3 {{ margin:7px 0 0; font-size:28px; }} .kicker {{ text-transform:uppercase; font-size:11px; letter-spacing:.16em; color:var(--muted); }}
section {{ background:rgba(15,30,51,.78); border:1px solid var(--line); border-radius:22px; padding:24px; margin:22px 0; overflow:auto; }} table.data {{ width:100%; border-collapse:collapse; font-size:13px; }} table.data th {{ text-align:left; color:#d8eaff; background:rgba(255,255,255,.08); }} table.data th, table.data td {{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.09); vertical-align:top; }} .figure img {{ width:100%; max-width:1120px; border-radius:16px; border:1px solid var(--line); background:white; }} pre {{ white-space:pre-wrap; background:rgba(0,0,0,.28); border:1px solid var(--line); border-radius:14px; padding:16px; color:#dbecff; }}
</style></head><body><header><h1>Phase 3.1 RavenPack Full Scale</h1><p class="subtitle">One full-scale RavenPack extraction after the Phase 3.0 pilot passed. The pipeline uses server-side filtering, chunked annual partitions, entity-level and gvkey/story/event deduplication, and aggregate-only reporting. Raw vendor Parquet files stay local.</p></header><main>
<div class="grid">{''.join(cards)}</div>
<section><h2>Validation</h2>{table(pd.DataFrame([validation]))}</section>
<section><h2>Mapping summary</h2>{table(mapping_summary)}</section>
<section><h2>Extraction summary by year</h2>{table(manifest_year)}</section>
<section><h2>Node-day summary by year</h2>{table(node_day_summary)}</section>
<section><h2>Top event classes</h2>{table(event_class.head(40))}</section>
<section><h2>Figures</h2>{''.join(fig_html)}</section>
<section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main></body></html>"""
    out_path.write_text(doc)


def protected_inventory(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    rows = []
    for base in [raw_dir, processed_dir]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.parquet")):
            rows.append(
                {
                    "file": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "protected_local_only": True,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--sample-start", default=SAMPLE_START_DEFAULT)
    parser.add_argument("--sample-end", default=SAMPLE_END_DEFAULT)
    parser.add_argument("--entity-chunk-size", type=int, default=1500)
    parser.add_argument("--mapping-chunk-size", type=int, default=3000)
    parser.add_argument("--min-relevance", type=float, default=90.0)
    parser.add_argument("--min-event-relevance", type=float, default=90.0)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    raw_dir = root / "data" / "raw" / "wrds" / "phase3_ravenpack_full"
    processed_dir = root / "data" / "processed" / "news_shocks_full"
    graph_path = root / "data" / "processed" / "graph_backbone" / "edges_supplier_customer_common_us.parquet"
    start_year = pd.Timestamp(args.sample_start).year
    end_year = pd.Timestamp(args.sample_end).year

    for p in [out_dir, log_dir, raw_dir, processed_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 3.1 RavenPack full-scale extraction")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {root}")
    print(f"Graph: {graph_path}")
    print(f"Output dir: {out_dir}")
    print(f"Protected raw dir: {raw_dir}")
    print(f"Protected processed dir: {processed_dir}")
    print(f"Years: {start_year}-{end_year}")
    print(f"Entity chunk size: {args.entity_chunk_size}")
    print("=" * 80)

    env = {
        "utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "thread_env": {k: os.environ.get(k) for k in ["PNA_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "PYARROW_NUM_THREADS"]},
        "packages": package_status(),
        "git": run_cmd(["bash", "-lc", "command -v git || true"]),
    }
    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    db = None
    try:
        nodes, ids = build_identifier_universe(graph_path, out_dir)
        db, user, source = connect_wrds()
        type_counts, long_map, fallback = query_mappings(db, ids, raw_dir, args.mapping_chunk_size, args.force_refresh)
        entity_map = build_entity_map(ids, long_map, fallback, processed_dir, out_dir)

        mapping_rate = entity_map["gvkey"].nunique() / max(ids["gvkey"].nunique(), 1)
        print(f"[VALIDATE] Mapping rate: {mapping_rate:.3%}")
        if mapping_rate < 0.50:
            raise RuntimeError(f"Mapping rate too low for full-scale extraction: {mapping_rate:.3%}")

        manifest = extract_and_process_news(
            db=db,
            entity_map=entity_map,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            start_year=start_year,
            end_year=end_year,
            entity_chunk_size=args.entity_chunk_size,
            min_relevance=args.min_relevance,
            min_event_relevance=args.min_event_relevance,
            force_refresh=args.force_refresh,
        )
    except Exception:
        print("[ERROR] Phase 3.1 full-scale extraction failed.")
        print(traceback.format_exc())
        return 2
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    manifest.to_csv(out_dir / "full_extraction_manifest_redacted.csv", index=False)
    manifest_year = manifest.groupby("year", dropna=False).agg(
        chunks=("chunk", "nunique"),
        raw_rows=("raw_rows", "sum"),
        entity_dedup_rows=("entity_dedup_rows", "sum"),
        mapped_rows=("mapped_rows", "sum"),
        entity_ids=("entity_ids", "sum"),
    ).reset_index().sort_values("year")
    manifest_year.to_csv(out_dir / "full_extraction_summary_by_year.csv", index=False)

    try:
        mapped = load_all_mapped_chunks(processed_dir)
        dedup, node_day, event_class, by_year = finalize_dedup_and_node_day(mapped, processed_dir)
    except Exception:
        print("[ERROR] Phase 3.1 final aggregation failed.")
        print(traceback.format_exc())
        return 3

    node_day_summary = pd.DataFrame()
    if not node_day.empty:
        node_day_summary = node_day.groupby("signal_year", dropna=False).agg(
            node_day_rows=("gvkey", "size"),
            gvkeys=("gvkey", "nunique"),
            stories_proxy=("n_stories", "sum"),
            n_events=("n_events", "sum"),
            abs_news_shock=("abs_news_shock", "sum"),
            signed_news_shock=("signed_news_shock", "sum"),
            mean_abs_news_shock=("abs_news_shock", "mean"),
        ).reset_index().sort_values("signal_year")
        node_day_summary.to_csv(out_dir / "full_node_day_summary_by_year.csv", index=False)

    event_class.to_csv(out_dir / "full_top_event_classes_redacted.csv", index=False)
    by_year.to_csv(out_dir / "full_mapped_event_summary_by_signal_year.csv", index=False)

    mapping_summary = pd.DataFrame(
        [
            {
                "target_gvkeys": int(ids["gvkey"].nunique()),
                "target_identifier_rows": int(len(ids)),
                "mapped_gvkeys": int(entity_map["gvkey"].nunique()),
                "mapped_rp_entities": int(entity_map["rp_entity_id"].nunique()),
                "entity_map_rows": int(len(entity_map)),
                "mapping_rate_gvkey": float(entity_map["gvkey"].nunique() / max(ids["gvkey"].nunique(), 1)),
            }
        ]
    )
    mapping_summary.to_csv(out_dir / "full_mapping_summary.csv", index=False)

    years = list(range(start_year, end_year + 1))
    raw_rows = int(manifest["raw_rows"].fillna(0).sum()) if not manifest.empty else 0
    mapped_rows_pre = int(manifest["mapped_rows"].fillna(0).sum()) if not manifest.empty else 0
    years_with_raw = sorted(manifest.loc[manifest["raw_rows"].fillna(0) > 0, "year"].dropna().astype(int).unique().tolist()) if not manifest.empty else []
    checks = {
        "mapping_rate_ge_50pct": float(mapping_summary.loc[0, "mapping_rate_gvkey"]) >= 0.50,
        "mapped_rp_entities_positive": int(mapping_summary.loc[0, "mapped_rp_entities"]) > 0,
        "raw_rows_positive": raw_rows > 0,
        "mapped_rows_positive": mapped_rows_pre > 0,
        "global_dedup_not_larger_than_mapped": len(dedup) <= mapped_rows_pre,
        "node_day_rows_positive": len(node_day) > 0,
        "node_day_gvkeys_positive": (node_day["gvkey"].nunique() if not node_day.empty else 0) > 0,
        "all_requested_years_have_raw_rows": set(years).issubset(set(years_with_raw)),
        "forbidden_text_columns_absent": bool(manifest["forbidden_text_columns_absent"].fillna(True).all()) if not manifest.empty else False,
    }
    validation = {
        "generated_at_utc": utc_now(),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "sample_start": args.sample_start,
        "sample_end": args.sample_end,
        "min_relevance": args.min_relevance,
        "min_event_relevance": args.min_event_relevance,
        "entity_chunk_size": args.entity_chunk_size,
        "mapping_chunk_size": args.mapping_chunk_size,
        "requested_years": years,
        "years_with_raw_rows": years_with_raw,
    }
    with (out_dir / "full_validation_report.json").open("w") as f:
        json.dump(validation, f, indent=2, default=str)

    inv = protected_inventory(raw_dir, processed_dir)
    inv.to_csv(out_dir / "protected_local_full_inventory.csv", index=False)

    summary = {
        "generated_at_utc": utc_now(),
        "sample_start": args.sample_start,
        "sample_end": args.sample_end,
        "target_gvkeys": int(mapping_summary.loc[0, "target_gvkeys"]),
        "mapped_gvkeys": int(mapping_summary.loc[0, "mapped_gvkeys"]),
        "mapping_rate_gvkey": float(mapping_summary.loc[0, "mapping_rate_gvkey"]),
        "mapped_rp_entities": int(mapping_summary.loc[0, "mapped_rp_entities"]),
        "raw_rows": raw_rows,
        "mapped_rows_pre_global_dedup": mapped_rows_pre,
        "mapped_events_dedup": int(len(dedup)),
        "node_day_rows": int(len(node_day)),
        "node_day_gvkeys": int(node_day["gvkey"].nunique()) if not node_day.empty else 0,
        "validation_passed": bool(validation["passed"]),
        "protected_node_day_path": str(processed_dir / "node_day_news_shocks.parquet"),
        "protected_entity_map_path": str(processed_dir / "ravenpack_entity_node_map.parquet"),
        "protected_mapped_events_deduped_path": str(processed_dir / "mapped_events_deduplicated.parquet"),
    }
    with (out_dir / "phase3_1_quality_summary.json").open("w") as f:
        json.dump({"summary": summary, "validation": validation}, f, indent=2, default=str)

    figures = make_figures(out_dir, manifest, by_year, node_day, event_class, validation)
    render_report(
        out_path=out_dir / "phase3_1_ravenpack_full_scale_report.html",
        summary=summary,
        validation=validation,
        mapping_summary=mapping_summary,
        manifest_year=manifest_year,
        node_day_summary=node_day_summary,
        event_class=event_class,
        figures=figures,
        env=env,
    )

    lines = [
        "# Phase 3.1 RavenPack full-scale summary",
        "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Validation passed: {summary['validation_passed']}",
        f"- Target GVKEYs: {summary['target_gvkeys']:,}",
        f"- Mapped GVKEYs: {summary['mapped_gvkeys']:,} ({summary['mapping_rate_gvkey']:.2%})",
        f"- Mapped RP entities: {summary['mapped_rp_entities']:,}",
        f"- Raw filtered RavenPack rows: {summary['raw_rows']:,}",
        f"- Pre-global-dedup mapped rows: {summary['mapped_rows_pre_global_dedup']:,}",
        f"- Deduplicated mapped events: {summary['mapped_events_dedup']:,}",
        f"- Node-day rows: {summary['node_day_rows']:,}",
        f"- Node-day GVKEYs: {summary['node_day_gvkeys']:,}",
        f"- Report: {out_dir / 'phase3_1_ravenpack_full_scale_report.html'}",
        f"- Protected node-day output: {processed_dir / 'node_day_news_shocks.parquet'}",
        "",
        "Data policy: raw RavenPack and full mapped-event Parquet files remain local and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE3_1_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if validation["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
