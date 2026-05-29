from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NEWS_COLS = [
    "timestamp_utc", "rp_story_id", "rp_entity_id", "entity_type", "entity_name",
    "country_code", "relevance", "event_sentiment_score", "event_relevance",
    "event_similarity_key", "event_similarity_days", "topic", "group", "type", "sub_type",
    "fact_level", "event_start_date_utc", "event_end_date_utc", "related_entity",
    "relationship", "category", "news_type", "rp_source_id", "source_name",
    "rp_story_event_index", "rp_story_event_count",
]
FORBIDDEN_TEXT_COLS = {"headline", "event_text", "provider_story_id"}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    return "(" + ",".join(q(v) for v in values) + ")"


def chunked(values: list[str], n: int) -> list[list[str]]:
    return [values[i:i + n] for i in range(0, len(values), n)]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


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
        except Exception:
            return None, "~/.pgpass unreadable"
    return None, "not found"


def connect_wrds():
    import wrds
    user, source = detect_wrds_username()
    print(f"[INFO] WRDS user source={source}, user={(user[:2] + '***' + user[-1]) if user and len(user) > 3 else user}")
    db = wrds.Connection(wrds_username=user, verbose=False) if user else wrds.Connection(verbose=False)
    try:
        db.raw_sql("set statement_timeout to '900000ms'")
    except Exception:
        pass
    return db


def raw_sql(db, sql: str, label: str) -> pd.DataFrame:
    t0 = time.time()
    print(f"[WRDS] {label} ...")
    df = db.raw_sql(sql)
    df.columns = [str(c).lower() for c in df.columns]
    print(f"[WRDS] {label}: rows={len(df):,}, elapsed={time.time() - t0:.1f}s")
    return df


def build_identifiers(edge_path: Path, out_dir: Path, pilot_node_limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing graph file: {edge_path}")
    edges = pd.read_parquet(edge_path)
    print(f"[INFO] Loaded corrected graph: {len(edges):,} rows")

    frames = []
    for side in ["supplier", "customer"]:
        gv = f"{side}_gvkey"
        if gv not in edges.columns:
            continue
        frame = pd.DataFrame({
            "side": side,
            "gvkey": clean_gvkey(edges[gv]),
            "permno": as_int(edges.get(f"{side}_permno", pd.Series(pd.NA, index=edges.index))),
            "ticker": edges.get(f"{side}_ticker", pd.Series(pd.NA, index=edges.index)).astype("string"),
            "cusip_raw": edges.get(f"{side}_cusip_raw", pd.Series(pd.NA, index=edges.index)).astype("string"),
        })
        deg = edges.groupby(gv).size().rename("degree").reset_index()
        deg["gvkey"] = clean_gvkey(deg[gv])
        frame = frame.merge(deg[["gvkey", "degree"]], on="gvkey", how="left")
        frames.append(frame)

    nodes = pd.concat(frames, ignore_index=True).dropna(subset=["gvkey"]).drop_duplicates()
    nodes["degree"] = pd.to_numeric(nodes["degree"], errors="coerce").fillna(1)
    nodes = nodes.sort_values("degree", ascending=False).drop_duplicates("gvkey").head(pilot_node_limit).copy()
    nodes["ticker_norm"] = nodes["ticker"].map(norm)
    nodes["cusip_norm"] = nodes["cusip_raw"].map(norm)
    nodes["cusip9"] = nodes["cusip_norm"].str[:9]
    nodes["cusip8"] = nodes["cusip_norm"].str[:8]
    nodes["cusip6"] = nodes["cusip_norm"].str[:6]

    id_rows = []
    for r in nodes.itertuples(index=False):
        for kind, val, priority in [
            ("gvkey", r.gvkey, 1),
            ("permno", None if pd.isna(r.permno) else str(int(r.permno)), 2),
            ("cusip9", r.cusip9, 3),
            ("cusip8", r.cusip8, 4),
            ("cusip6", r.cusip6, 5),
            ("ticker", r.ticker_norm, 6),
        ]:
            nv = norm(val)
            if nv:
                id_rows.append({"gvkey": r.gvkey, "permno": None if pd.isna(r.permno) else int(r.permno), "id_kind": kind, "id_value_norm": nv, "priority": priority, "degree": float(r.degree)})
    ids = pd.DataFrame(id_rows).drop_duplicates()
    nodes.drop(columns=["cusip_norm"], errors="ignore").to_csv(out_dir / "pilot_nodes_redacted.csv", index=False)
    ids.drop(columns=["id_value_norm"], errors="ignore").to_csv(out_dir / "pilot_identifier_rows_redacted.csv", index=False)
    ids.groupby("id_kind").size().rename("n_rows").reset_index().to_csv(out_dir / "pilot_identifier_counts.csv", index=False)
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


def query_mappings(db, ids: pd.DataFrame, raw_dir: Path, chunk_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    type_counts = raw_sql(db, """
        select data_type, count(*) as n_rows
        from ravenpack_common.rpa_company_mappings
        group by data_type
        order by n_rows desc, data_type
    """, "mapping data_type counts")
    type_counts["match_kind"] = type_counts["data_type"].map(classify_dtype)
    data_types = sorted(type_counts.loc[type_counts["match_kind"].notna(), "data_type"].dropna().astype(str).unique())
    if not data_types:
        raise RuntimeError("No useful data_type values found in rpa_company_mappings")

    values = sorted(ids["id_value_norm"].dropna().astype(str).unique())
    frames = []
    for i, vals in enumerate(chunked(values, chunk_size), start=1):
        sql = f"""
            select rp_entity_id, entity_type, data_type, data_value, range_start, range_end
            from ravenpack_common.rpa_company_mappings
            where data_type in {sql_in(data_types)}
              and upper(regexp_replace(coalesce(data_value,''), '[^A-Za-z0-9]', '', 'g')) in {sql_in(vals)}
        """
        frames.append(raw_sql(db, sql, f"rpa_company_mappings chunk {i}"))
    long_map = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fb_frames = []
    for i, vals in enumerate(chunked(values, chunk_size), start=1):
        sql = f"""
            select rp_entity_id, entity_type, entity_name, ticker, cusip, isin
            from ravenpack_common.wrds_rpa_company_mappings
            where upper(regexp_replace(coalesce(ticker,''), '[^A-Za-z0-9]', '', 'g')) in {sql_in(vals)}
               or upper(regexp_replace(coalesce(cusip,''), '[^A-Za-z0-9]', '', 'g')) in {sql_in(vals)}
        """
        try:
            fb_frames.append(raw_sql(db, sql, f"wrds_rpa_company_mappings fallback chunk {i}"))
        except Exception as exc:
            print(f"[WARN] fallback mapping failed on chunk {i}: {exc}")
    fallback = pd.concat(fb_frames, ignore_index=True) if fb_frames else pd.DataFrame()

    write_parquet(type_counts, raw_dir / "pilot_mapping_data_type_counts.parquet")
    write_parquet(long_map, raw_dir / "pilot_mapping_long_filtered.parquet")
    write_parquet(fallback, raw_dir / "pilot_mapping_fallback_filtered.parquet")
    return type_counts, long_map, fallback


def build_entity_map(ids: pd.DataFrame, long_map: pd.DataFrame, fallback: pd.DataFrame, processed_dir: Path, out_dir: Path) -> pd.DataFrame:
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
        for mkind, ikinds in specs:
            j = lm[lm["map_kind"].eq(mkind)].merge(ids[ids["id_kind"].isin(ikinds)], left_on="data_value_norm", right_on="id_value_norm", how="inner")
            if not j.empty:
                j["mapping_source"] = "rpa_company_mappings"
                pieces.append(j)

    if not fallback.empty:
        fb = fallback.copy()
        fb["range_start"] = pd.Timestamp("1900-01-01")
        fb["range_end"] = pd.Timestamp("2099-12-31")
        for col, ikinds, bump in [("cusip", ["cusip9", "cusip8", "cusip6"], 20), ("ticker", ["ticker"], 30)]:
            if col in fb.columns:
                left = fb.copy()
                left["data_type"] = col.upper()
                left["data_value"] = left[col]
                left["data_value_norm"] = left[col].map(norm)
                j = left.merge(ids[ids["id_kind"].isin(ikinds)], left_on="data_value_norm", right_on="id_value_norm", how="inner")
                if not j.empty:
                    j["priority"] = pd.to_numeric(j["priority"], errors="coerce").fillna(999).astype(int) + bump
                    j["mapping_source"] = "wrds_rpa_company_mappings"
                    pieces.append(j)

    if not pieces:
        raise RuntimeError("No RavenPack-to-graph mappings were found")
    em = pd.concat(pieces, ignore_index=True)
    keep = ["rp_entity_id", "entity_type", "gvkey", "permno", "id_kind", "data_type", "range_start", "range_end", "priority", "degree", "mapping_source"]
    for c in keep:
        if c not in em.columns:
            em[c] = pd.NA
    em = em[keep].dropna(subset=["rp_entity_id", "gvkey"]).copy()
    em["gvkey"] = clean_gvkey(em["gvkey"])
    em["permno"] = as_int(em["permno"])
    em["rp_entity_id"] = em["rp_entity_id"].astype("string")
    em["priority"] = pd.to_numeric(em["priority"], errors="coerce").fillna(999).astype(int)
    em["degree"] = pd.to_numeric(em["degree"], errors="coerce").fillna(0)
    em = em.sort_values(["gvkey", "rp_entity_id", "priority", "degree"], ascending=[True, True, True, False]).drop_duplicates(["gvkey", "rp_entity_id"])
    write_parquet(em, processed_dir / "pilot_ravenpack_entity_node_map.parquet")
    em.drop(columns=[], errors="ignore").to_csv(out_dir / "pilot_entity_node_map_redacted.csv", index=False)
    return em


def query_news(db, em: pd.DataFrame, raw_dir: Path, year: int, entity_limit: int) -> tuple[pd.DataFrame, float, float]:
    ids = em.sort_values("degree", ascending=False)["rp_entity_id"].dropna().astype(str).drop_duplicates().head(entity_limit).tolist()
    if not ids:
        raise RuntimeError("No RP entity IDs to query")
    selected = ",\n                ".join(f'"{c}"' for c in NEWS_COLS)
    table = f"ravenpack_dj.rpa_djpr_equities_{year}"
    attempts = [(90.0, 90.0), (75.0, 75.0), (50.0, 0.0)]
    last = pd.DataFrame()
    used = attempts[-1]
    for rel, erel in attempts:
        sql = f"""
            select {selected}
            from {table}
            where rp_entity_id in {sql_in(ids)}
              and timestamp_utc >= timestamp '{year}-01-01 00:00:00'
              and timestamp_utc < timestamp '{year + 1}-01-01 00:00:00'
              and coalesce(relevance, 0) >= {rel}
              and coalesce(event_relevance, 0) >= {erel}
              and event_sentiment_score is not null
            order by timestamp_utc, rp_story_id, rp_entity_id
        """
        last = raw_sql(db, sql, f"RavenPack pilot events rel>={rel} event_rel>={erel}")
        used = (rel, erel)
        if len(last) >= 100:
            break
    write_parquet(last, raw_dir / f"pilot_ravenpack_events_{year}.parquet")
    return last, used[0], used[1]


def add_time(news: pd.DataFrame) -> pd.DataFrame:
    n = news.copy()
    n["timestamp_utc"] = pd.to_datetime(n["timestamp_utc"], errors="coerce")
    n = n.dropna(subset=["timestamp_utc"])
    ts = n["timestamp_utc"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT").dt.tz_convert(ZoneInfo("America/New_York"))
    n["timestamp_et"] = ts.dt.tz_localize(None)
    n["market_date"] = n["timestamp_et"].dt.date.astype(str)
    mins = n["timestamp_et"].dt.hour * 60 + n["timestamp_et"].dt.minute
    n["market_time_bucket"] = np.select([mins < 570, mins <= 960], ["pre_open", "regular_session"], default="post_close")
    sig = pd.to_datetime(n["market_date"], errors="coerce")
    sig = sig.where(~n["market_time_bucket"].eq("post_close"), sig + pd.Timedelta(days=1))
    n["news_signal_date"] = sig.dt.date.astype(str)
    return n


def aggregate_news(raw_news: pd.DataFrame, em: pd.DataFrame, processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if raw_news.empty:
        return raw_news, pd.DataFrame(), pd.DataFrame()
    n = add_time(raw_news)
    for c in ["relevance", "event_relevance", "event_sentiment_score", "event_similarity_days"]:
        n[c] = pd.to_numeric(n[c], errors="coerce")
    for c in ["rp_story_id", "rp_entity_id", "event_similarity_key", "topic", "group", "type", "sub_type"]:
        n[c] = n.get(c, pd.Series("", index=n.index)).astype("string").fillna("")
    n["event_key"] = np.where(n["event_similarity_key"].str.len() > 0, n["event_similarity_key"], n["topic"] + "|" + n["group"] + "|" + n["type"] + "|" + n["sub_type"] + "|" + n["timestamp_utc"].astype(str))
    n["dedup_key"] = n["rp_entity_id"] + "|" + n["rp_story_id"] + "|" + n["event_key"]
    dedup = n.sort_values(["dedup_key", "relevance", "event_relevance", "timestamp_utc"], ascending=[True, False, False, True]).drop_duplicates("dedup_key")

    em2 = em.copy()
    em2["rp_entity_id"] = em2["rp_entity_id"].astype("string")
    mapped = dedup.merge(em2[["rp_entity_id", "gvkey", "permno", "range_start", "range_end", "priority"]], on="rp_entity_id", how="inner")
    if mapped.empty:
        return dedup, mapped, pd.DataFrame()
    mapped["event_day"] = pd.to_datetime(mapped["timestamp_utc"], errors="coerce").dt.floor("D")
    mapped["range_start"] = pd.to_datetime(mapped["range_start"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    mapped["range_end"] = pd.to_datetime(mapped["range_end"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    mapped = mapped[(mapped["event_day"] >= mapped["range_start"]) & (mapped["event_day"] <= mapped["range_end"])].copy()
    mapped = mapped.sort_values(["dedup_key", "gvkey", "priority"]).drop_duplicates(["dedup_key", "gvkey"])
    scale = 100.0 if mapped["event_sentiment_score"].abs().max(skipna=True) > 1.5 else 1.0
    novelty = 1.0 / (1.0 + mapped["event_similarity_days"].fillna(0).clip(lower=0))
    mapped["sentiment_scaled"] = mapped["event_sentiment_score"] / scale
    mapped["attention_weight"] = (mapped["relevance"].fillna(0) / 100.0) * (mapped["event_relevance"].fillna(0) / 100.0)
    mapped["signed_news_shock"] = mapped["sentiment_scaled"] * mapped["attention_weight"] * novelty
    mapped["positive_news_shock"] = mapped["signed_news_shock"].clip(lower=0)
    mapped["negative_news_shock"] = mapped["signed_news_shock"].clip(upper=0)
    mapped["abs_news_shock"] = mapped["signed_news_shock"].abs()

    node_day = mapped.groupby(["gvkey", "permno", "news_signal_date"], dropna=False).agg(
        n_events=("rp_story_id", "size"),
        n_stories=("rp_story_id", "nunique"),
        signed_news_shock=("signed_news_shock", "sum"),
        positive_news_shock=("positive_news_shock", "sum"),
        negative_news_shock=("negative_news_shock", "sum"),
        abs_news_shock=("abs_news_shock", "sum"),
        mean_relevance=("relevance", "mean"),
        max_relevance=("relevance", "max"),
    ).reset_index()
    write_parquet(dedup, processed_dir / "pilot_events_deduplicated.parquet")
    write_parquet(mapped, processed_dir / "pilot_events_mapped.parquet")
    write_parquet(node_day, processed_dir / "pilot_node_day_news_shocks.parquet")
    return dedup, mapped, node_day


def make_outputs(out_dir: Path, summary: dict[str, Any], mapped: pd.DataFrame, node_day: pd.DataFrame) -> None:
    if not mapped.empty:
        mapped.groupby(["group", "type", "sub_type"], dropna=False).agg(n_events=("rp_story_id", "size"), n_gvkeys=("gvkey", "nunique"), abs_shock=("abs_news_shock", "sum")).reset_index().sort_values("n_events", ascending=False).head(40).to_csv(out_dir / "pilot_event_classes_redacted.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "pilot_summary.csv", index=False)
    with (out_dir / "pilot_validation_report.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    html_doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Phase 3.0 RavenPack Pilot</title>
<style>body{{font-family:Arial,sans-serif;background:#07111f;color:#eef6ff;margin:40px}}table{{border-collapse:collapse;background:#0f1e33}}td,th{{border:1px solid #334;padding:8px}}.card{{display:inline-block;margin:8px;padding:14px;border:1px solid #334;border-radius:12px;background:#0f1e33}}</style></head><body>
<h1>Phase 3.0 RavenPack Pilot</h1><p>Small pilot only: mapping, event extraction, deduplication, and node-day shock aggregation. Raw RavenPack files remain local.</p>
<div class='card'><b>Validation</b><br>{html.escape('PASS' if summary['pilot_passed'] else 'FAIL')}</div>
<div class='card'><b>Mapped RP entities</b><br>{summary['mapped_rp_entities']:,}</div>
<div class='card'><b>Raw events</b><br>{summary['raw_rows']:,}</div>
<div class='card'><b>Mapped events</b><br>{summary['mapped_rows']:,}</div>
<div class='card'><b>Node-days</b><br>{summary['node_day_rows']:,}</div>
<h2>Summary</h2>{pd.DataFrame([summary]).to_html(index=False, escape=True)}
</body></html>"""
    (out_dir / "phase3_0_ravenpack_pilot_report.html").write_text(html_doc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--pilot-year", type=int, default=2024)
    ap.add_argument("--pilot-node-limit", type=int, default=500)
    ap.add_argument("--pilot-entity-limit", type=int, default=500)
    ap.add_argument("--sql-chunk-size", type=int, default=1000)
    args = ap.parse_args()

    root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    raw_dir = root / "data" / "raw" / "wrds" / "phase3_ravenpack_pilot"
    processed_dir = root / "data" / "processed" / "news_shocks_pilot"
    graph_path = root / "data" / "processed" / "graph_backbone" / "edges_supplier_customer_common_us.parquet"
    for p in [out_dir, log_dir, raw_dir, processed_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 3.0 RavenPack pilot, pilot-first validation only")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {root}")
    print(f"Output dir: {out_dir}")
    print("=" * 80)

    db = None
    try:
        nodes, ids = build_identifiers(graph_path, out_dir, args.pilot_node_limit)
        db = connect_wrds()
        type_counts, long_map, fallback = query_mappings(db, ids, raw_dir, args.sql_chunk_size)
        em = build_entity_map(ids, long_map, fallback, processed_dir, out_dir)
        raw_news, used_rel, used_erel = query_news(db, em, raw_dir, args.pilot_year, args.pilot_entity_limit)
    except Exception as exc:
        print("[ERROR] Phase 3.0 pilot failed")
        import traceback
        print(traceback.format_exc())
        return 2
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    dedup, mapped, node_day = aggregate_news(raw_news, em, processed_dir)
    checks = {
        "mapped_rp_entities_ge_20": int(em["rp_entity_id"].nunique()) >= 20,
        "raw_rows_positive": len(raw_news) > 0,
        "mapped_rows_positive": len(mapped) > 0,
        "node_day_rows_positive": len(node_day) > 0,
        "dedup_not_larger_than_raw": len(dedup) <= len(raw_news),
        "forbidden_text_columns_absent": not bool(set(raw_news.columns) & FORBIDDEN_TEXT_COLS),
    }
    summary = {
        "generated_at_utc": utc_now(),
        "pilot_year": args.pilot_year,
        "pilot_node_limit": args.pilot_node_limit,
        "pilot_entity_limit": args.pilot_entity_limit,
        "used_min_relevance": used_rel,
        "used_min_event_relevance": used_erel,
        "pilot_passed": bool(all(checks.values())),
        "checks": checks,
        "target_gvkeys": int(ids["gvkey"].nunique()),
        "mapped_gvkeys": int(em["gvkey"].nunique()),
        "mapped_rp_entities": int(em["rp_entity_id"].nunique()),
        "raw_rows": int(len(raw_news)),
        "dedup_rows": int(len(dedup)),
        "mapped_rows": int(len(mapped)),
        "node_day_rows": int(len(node_day)),
        "node_day_gvkeys": int(node_day["gvkey"].nunique()) if not node_day.empty else 0,
        "protected_node_day_path": str(processed_dir / "pilot_node_day_news_shocks.parquet"),
    }
    make_outputs(out_dir, summary, mapped, node_day)

    inventory = []
    for base in [raw_dir, processed_dir]:
        for p in sorted(base.glob("*.parquet")):
            inventory.append({"file": str(p), "size_bytes": p.stat().st_size, "protected_local_only": True})
    pd.DataFrame(inventory).to_csv(out_dir / "protected_local_pilot_inventory.csv", index=False)

    lines = [
        "# Phase 3.0 RavenPack pilot summary", "",
        f"- Pilot validation passed: {summary['pilot_passed']}",
        f"- Mapped RP entities: {summary['mapped_rp_entities']:,}",
        f"- Raw pilot rows: {summary['raw_rows']:,}",
        f"- Mapped event rows: {summary['mapped_rows']:,}",
        f"- Node-day rows: {summary['node_day_rows']:,}",
        f"- Report: {out_dir / 'phase3_0_ravenpack_pilot_report.html'}", "",
        "Data policy: raw RavenPack pilot Parquet files remain local and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE3_0_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if summary["pilot_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
