
from __future__ import annotations

import argparse
import datetime as dt
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

import pandas as pd


SAMPLE_START = "2015-01-01"
SAMPLE_END = "2025-12-31"

CORE_TARGETS: dict[str, list[tuple[str, str]]] = {
    "crsp_stock": [
        ("crsp_a_stock", "dsf"),
        ("crsp_a_stock", "dsf_v2"),
        ("crsp_a_stock", "dsenames"),
        ("crsp_a_stock", "stocknames"),
        ("crsp_a_stock", "dsedelist"),
        ("crsp_a_stock", "dsi"),
    ],
    "crsp_ccm": [
        ("crsp_a_ccm", "ccmxpf_linktable"),
        ("crsp_a_ccm", "ccmxpf_lnkhist"),
        ("crsp_a_ccm", "ccmxpf_lnkused"),
    ],
    "compustat": [
        ("comp_na_daily_all", "funda"),
        ("comp_na_daily_all", "fundq"),
        ("comp_na_daily_all", "company"),
        ("comp_na_daily_all", "security"),
        ("comp_na_daily_all", "names"),
        ("comp_na_daily_all", "co_hgic"),
    ],
    "supply_chain": [
        ("wrdsapps_link_supplychain", "seglink"),
        ("wrdsapps_link_supplychain", "seed_20180418"),
        ("comp_segments_hist_daily", "seg_customer"),
        ("comp_segments_hist_daily", "wrds_seg_customer"),
        ("comp_segments_hist_daily", "seg_ann"),
        ("comp_segments_hist_daily", "seg_annfund"),
        ("comp_segments_hist_daily", "seg_product"),
        ("comp_segments_hist_daily", "wrds_seg_product"),
        ("comp_segments_hist_daily", "wrds_segmerged"),
    ],
    "ravenpack_common": [
        ("ravenpack_common", "rpa_entity_mappings"),
        ("ravenpack_common", "rpa_company_mappings"),
        ("ravenpack_common", "wrds_rpa_entity_mappings"),
        ("ravenpack_common", "wrds_rpa_company_mappings"),
        ("ravenpack_common", "wrds_rpa_all_mappings"),
        ("ravenpack_common", "rpa_taxonomy"),
        ("ravenpack_common", "rpa_source_list"),
    ],
    "ravenpack_dj": [
        *[("ravenpack_dj", f"rpa_djpr_equities_{year}") for year in range(2015, 2026)],
        *[("ravenpack_dj", f"rpa_djpr_global_macro_{year}") for year in range(2015, 2026)],
    ],
    "ibes": [
        ("tr_ibes", "id"),
        ("tr_ibes", "det_epsus"),
        ("tr_ibes", "statsum_epsus"),
        ("tr_ibes", "det_xepsus"),
        ("tr_ibes", "statsum_xepsus"),
        ("tr_ibes", "recddet"),
        ("tr_ibes", "recdsum"),
        ("wrdsapps_link_crsp_ibes", "ibcrsphist"),
    ],
    "liquidity": [
        ("contrib_liquidity_taq", "bbd"),
        ("contrib_liquidity_taq", "ilc"),
        ("wrdsapps_link_crsp_taqm", "taqmsec"),
        ("wrdsapps_link_crsp_taq", "taqmsf"),
    ],
}


REQUIRED_COLUMNS: dict[str, list[str]] = {
    "crsp_a_stock.dsf": ["permno", "date", "ret", "retx", "prc", "vol", "shrout", "cfacpr", "cfacshr"],
    "crsp_a_stock.dsf_v2": ["permno", "permco", "mthcaldt", "dlycaldt", "dlyret", "dlyretx", "dlyprc", "dlyvol", "shrout"],
    "crsp_a_stock.stocknames": ["permno", "permco", "namedt", "nameenddt", "ticker", "comnam", "shrcd", "exchcd", "siccd"],
    "crsp_a_stock.dsenames": ["permno", "permco", "namedt", "nameendt", "ticker", "comnam", "shrcd", "exchcd", "siccd"],
    "crsp_a_stock.dsedelist": ["permno", "dlstdt", "dlret", "dlstcd"],
    "crsp_a_ccm.ccmxpf_linktable": ["gvkey", "lpermno", "lpermco", "linkdt", "linkenddt", "linkprim", "linktype", "liid"],
    "crsp_a_ccm.ccmxpf_lnkhist": ["gvkey", "lpermno", "lpermco", "linkdt", "linkenddt", "linkprim", "linktype", "liid"],
    "comp_na_daily_all.funda": ["gvkey", "datadate", "fyear", "indfmt", "consol", "popsrc", "datafmt", "tic", "cusip", "conm", "at", "sale", "ceq", "seq", "lt", "ni", "prcc_f", "csho"],
    "comp_na_daily_all.company": ["gvkey", "conm", "tic", "cusip", "cik", "sic", "naics", "gsector", "gind", "gsubind"],
    "comp_segments_hist_daily.seg_customer": ["gvkey", "datadate", "cid", "cnms", "ctype", "stype", "salecs"],
    "comp_segments_hist_daily.wrds_seg_customer": ["gvkey", "datadate", "srcdate", "cid", "cnms", "ctype", "stype", "salecs"],
    "wrdsapps_link_supplychain.seglink": ["gvkey", "cgvkey", "srcdate", "datadate", "cid", "stype", "salecs"],
    "ravenpack_common.rpa_entity_mappings": ["rp_entity_id", "entity_id", "entity_name"],
    "ravenpack_common.rpa_company_mappings": ["rp_entity_id", "company_id", "isin", "cusip", "sedol", "ticker"],
    "ravenpack_dj.rpa_djpr_equities_2024": ["timestamp_utc", "rp_entity_id", "entity_id", "relevance", "event_relevance", "ess", "aes", "ens", "ens_similarity_key", "event_similarity_key"],
    "tr_ibes.id": ["ticker", "cusip", "cname", "oftic", "sdates", "edates"],
    "tr_ibes.det_epsus": ["ticker", "cusip", "fpedats", "anntims", "analys", "value", "measure", "fpi"],
    "tr_ibes.statsum_epsus": ["ticker", "cusip", "fpedats", "statpers", "numest", "meanest", "medest", "stdev"],
    "wrdsapps_link_crsp_ibes.ibcrsphist": ["ticker", "permno", "sdate", "edate"],
    "contrib_liquidity_taq.bbd": ["permno", "date", "ticker", "spread"],
    "contrib_liquidity_taq.ilc": ["permno", "date", "ticker"],
}


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


def q(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def run_cmd(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
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


def pkg_status() -> pd.DataFrame:
    wanted = [
        "pandas",
        "numpy",
        "scipy",
        "statsmodels",
        "linearmodels",
        "wrds",
        "psycopg2",
        "sqlalchemy",
        "pyarrow",
        "duckdb",
        "polars",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "torch",
        "plotly",
        "dash",
        "dash-cytoscape",
        "pytest",
    ]
    rows = []
    for pkg in wanted:
        try:
            version = md.version(pkg)
            status = "ok"
        except md.PackageNotFoundError:
            version = ""
            status = "missing"
        rows.append({"package": pkg, "version": version, "status": status})
    return pd.DataFrame(rows)


def target_frame() -> pd.DataFrame:
    rows = []
    for family, pairs in CORE_TARGETS.items():
        for schema, table in pairs:
            rows.append({"family": family, "table_schema": schema, "table_name": table, "fqtn": f"{schema}.{table}"})
    return pd.DataFrame(rows).drop_duplicates()


def connect_wrds():
    import wrds

    user, source = detect_wrds_username()
    print(f"[INFO] WRDS username source: {source}; user={mask_user(user)}")

    try:
        db = wrds.Connection(wrds_username=user, verbose=False) if user else wrds.Connection(verbose=False)
    except TypeError:
        db = wrds.Connection()

    try:
        db.raw_sql("set statement_timeout to '180000ms'")
    except Exception:
        pass

    return db, user, source


def get_tables(db, targets: pd.DataFrame) -> pd.DataFrame:
    """Return target table metadata while skipping schemas without USAGE privilege.

    Some WRDS catalogs expose metadata names for schemas where the current user lacks
    schema usage. Calling has_table_privilege on those schemas can raise a hard
    permission error. This function first probes schema USAGE, filters inaccessible
    schemas, then performs table-level checks only where safe.
    """
    targets = targets.copy()
    schema_names = sorted(set(targets["table_schema"].astype(str)))

    if not schema_names:
        out = targets.copy()
        out["table_type"] = pd.NA
        out["approx_rows"] = pd.NA
        out["has_select"] = False
        out["schema_has_usage"] = False
        return out

    schema_literals = ",".join(q(s) for s in schema_names)
    schema_sql = f"""
    select
        schema_name,
        has_schema_privilege(current_user, schema_name, 'USAGE') as has_usage
    from information_schema.schemata
    where schema_name in ({schema_literals})
    order by schema_name
    """

    try:
        schema_status = db.raw_sql(schema_sql)
        schema_status.columns = [c.lower() for c in schema_status.columns]
        accessible = set(
            schema_status.loc[
                schema_status["has_usage"].fillna(False).astype(bool),
                "schema_name",
            ].astype(str)
        )
    except Exception as exc:
        print(f"[WARN] Schema privilege probe failed: {exc}")
        print("[WARN] Falling back to all target schemas except known restricted schemas.")
        accessible = set(schema_names) - restricted

    inaccessible = sorted(set(schema_names) - accessible)
    if inaccessible:
        print("[WARN] Skipping schemas without USAGE privilege:")
        for schema in inaccessible:
            print(f"  - {schema}")

    work_targets = targets[targets["table_schema"].isin(accessible)].copy()

    if work_targets.empty:
        got = pd.DataFrame(
            columns=["table_schema", "table_name", "table_type", "approx_rows", "has_select"]
        )
    else:
        pairs = work_targets[["table_schema", "table_name"]].drop_duplicates()
        clauses = [
            f"(t.table_schema={q(r.table_schema)} and t.table_name={q(r.table_name)})"
            for r in pairs.itertuples(index=False)
        ]
        accessible_literals = ",".join(q(s) for s in sorted(accessible))

        sql = f"""
        with approx as (
            select
                n.nspname as table_schema,
                c.relname as table_name,
                c.reltuples::bigint as approx_rows
            from pg_class c
            join pg_namespace n on n.oid = c.relnamespace
            where c.relkind in ('r','p','v','m')
              and n.nspname in ({accessible_literals})
        )
        select
            t.table_schema,
            t.table_name,
            t.table_type,
            a.approx_rows,
            has_table_privilege(
                current_user,
                quote_ident(t.table_schema)||'.'||quote_ident(t.table_name),
                'SELECT'
            ) as has_select
        from information_schema.tables t
        left join approx a
          on a.table_schema = t.table_schema
         and a.table_name = t.table_name
        where {" or ".join(clauses)}
        order by t.table_schema, t.table_name
        """
        got = db.raw_sql(sql)
        got.columns = [c.lower() for c in got.columns]

    merged = targets.merge(got, on=["table_schema", "table_name"], how="left")
    merged["schema_has_usage"] = merged["table_schema"].isin(accessible)

    if "table_type" not in merged.columns:
        merged["table_type"] = pd.NA
    if "approx_rows" not in merged.columns:
        merged["approx_rows"] = pd.NA
    if "has_select" not in merged.columns:
        merged["has_select"] = False

    merged["has_select"] = merged["has_select"].fillna(False).astype(bool)
    return merged

def get_columns(db, tables: pd.DataFrame) -> pd.DataFrame:
    """Read column metadata only for existing selectable tables in usable schemas."""
    if tables.empty:
        return pd.DataFrame(
            columns=["table_schema", "table_name", "column_name", "data_type", "ordinal_position"]
        )

    mask = tables["table_type"].notna()

    if "schema_has_usage" in tables.columns:
        mask = mask & tables["schema_has_usage"].fillna(False).astype(bool)

    if "has_select" in tables.columns:
        mask = mask & tables["has_select"].fillna(False).astype(bool)

    existing = tables.loc[mask, ["table_schema", "table_name"]].drop_duplicates()

    if existing.empty:
        return pd.DataFrame(
            columns=["table_schema", "table_name", "column_name", "data_type", "ordinal_position"]
        )

    clauses = [
        f"(table_schema={q(r.table_schema)} and table_name={q(r.table_name)})"
        for r in existing.itertuples(index=False)
    ]

    sql = f"""
    select
        table_schema,
        table_name,
        column_name,
        data_type,
        ordinal_position
    from information_schema.columns
    where {" or ".join(clauses)}
    order by table_schema, table_name, ordinal_position
    """

    cols = db.raw_sql(sql)
    cols.columns = [c.lower() for c in cols.columns]
    return cols

def enrich_status(tables: pd.DataFrame, cols: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for (schema, table), group in cols.groupby(["table_schema", "table_name"]):
        col_map[f"{schema}.{table}"] = set(group["column_name"].astype(str).str.lower())

    rows = []
    for r in tables.itertuples(index=False):
        fqtn = f"{r.table_schema}.{r.table_name}"
        present = col_map.get(fqtn, set())
        wanted = [c.lower() for c in REQUIRED_COLUMNS.get(fqtn, [])]
        matched = [c for c in wanted if c in present]
        missing = [c for c in wanted if c not in present]
        n_cols = len(present)

        if pd.isna(getattr(r, "table_type")):
            readiness = "missing_table"
        elif not bool(getattr(r, "has_select")):
            readiness = "no_select_privilege"
        elif wanted and len(matched) == 0:
            readiness = "exists_but_column_names_need_mapping"
        elif wanted and missing:
            readiness = "partial_column_match"
        else:
            readiness = "ready_metadata"

        rows.append({
            "family": r.family,
            "table_schema": r.table_schema,
            "table_name": r.table_name,
            "fqtn": fqtn,
            "table_type": getattr(r, "table_type"),
            "approx_rows": getattr(r, "approx_rows"),
            "has_select": bool(getattr(r, "has_select")) if not pd.isna(getattr(r, "has_select")) else False,
            "n_columns": n_cols,
            "required_columns": ",".join(wanted),
            "matched_required_columns": ",".join(matched),
            "missing_required_columns": ",".join(missing),
            "readiness": readiness,
        })
    return pd.DataFrame(rows)


def choose_first(status: pd.DataFrame, fqtns: list[str]) -> dict[str, Any]:
    for fqtn in fqtns:
        row = status[status["fqtn"].eq(fqtn)]
        if len(row) and bool(row.iloc[0]["has_select"]):
            return {
                "schema": str(row.iloc[0]["table_schema"]),
                "table": str(row.iloc[0]["table_name"]),
                "fqtn": fqtn,
                "readiness": str(row.iloc[0]["readiness"]),
                "approx_rows": None if pd.isna(row.iloc[0]["approx_rows"]) else int(float(row.iloc[0]["approx_rows"])),
            }
    return {"schema": None, "table": None, "fqtn": None, "readiness": "not_available", "approx_rows": None}


def build_schema_map(status: pd.DataFrame, cols: pd.DataFrame, user: str | None) -> dict[str, Any]:
    rp_equities = []
    for year in range(2015, 2026):
        fqtn = f"ravenpack_dj.rpa_djpr_equities_{year}"
        row = status[status["fqtn"].eq(fqtn)]
        if len(row) and bool(row.iloc[0]["has_select"]):
            rp_equities.append({
                "year": year,
                "schema": "ravenpack_dj",
                "table": f"rpa_djpr_equities_{year}",
                "fqtn": fqtn,
                "approx_rows": None if pd.isna(row.iloc[0]["approx_rows"]) else int(float(row.iloc[0]["approx_rows"])),
            })

    source_choices = {
        "crsp_daily_returns": choose_first(status, ["crsp_a_stock.dsf", "crsp_a_stock.dsf_v2"]),
        "crsp_names": choose_first(status, ["crsp_a_stock.stocknames", "crsp_a_stock.dsenames"]),
        "crsp_delisting": choose_first(status, ["crsp_a_stock.dsedelist"]),
        "crsp_compustat_link": choose_first(status, ["crsp_a_ccm.ccmxpf_linktable", "crsp_a_ccm.ccmxpf_lnkhist"]),
        "compustat_fundamentals": choose_first(status, ["comp_na_daily_all.funda"]),
        "compustat_company": choose_first(status, ["comp_na_daily_all.company"]),
        "supply_chain_link": choose_first(status, [
            "wrdsapps_link_supplychain.seglink",
            "comp_segments_hist_daily.wrds_seg_customer",
            "comp_segments_hist_daily.seg_customer",
        ]),
        "supply_chain_segments_fallback": choose_first(status, [
            "comp_segments_hist_daily.wrds_seg_customer",
            "comp_segments_hist_daily.seg_customer",
        ]),
        "ravenpack_entity_mapping": choose_first(status, [
            "ravenpack_common.rpa_entity_mappings",
            "ravenpack_common.wrds_rpa_entity_mappings",
        ]),
        "ravenpack_company_mapping": choose_first(status, [
            "ravenpack_common.rpa_company_mappings",
            "ravenpack_common.wrds_rpa_company_mappings",
        ]),
        "ibes_detail": choose_first(status, ["tr_ibes.det_epsus"]),
        "ibes_summary": choose_first(status, ["tr_ibes.statsum_epsus"]),
        "ibes_id": choose_first(status, ["tr_ibes.id"]),
        "crsp_ibes_link": choose_first(status, ["wrdsapps_link_crsp_ibes.ibcrsphist"]),
        "liquidity_bbd": choose_first(status, ["contrib_liquidity_taq.bbd"]),
        "liquidity_ilc": choose_first(status, ["contrib_liquidity_taq.ilc"]),
    }

    return {
        "generated_at_utc": utc_now(),
        "wrds_connected": True,
        "wrds_user_masked_for_log": mask_user(user),
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "source_choices": source_choices,
        "ravenpack_equities_partitions": rp_equities,
        "notes": [
            "This file is metadata-only and contains no raw vendor records.",
            "Actual extraction scripts should use only rows whose has_select is true.",
            "RavenPack global_macro tables are available but equities partitions are preferred for firm-level shocks.",
        ],
    }


def write_yaml(path: Path, obj: dict[str, Any]) -> None:
    def emit(value: Any, indent: int = 0) -> list[str]:
        sp = " " * indent
        lines: list[str] = []
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{sp}{k}:")
                    lines.extend(emit(v, indent + 2))
                elif v is None:
                    lines.append(f"{sp}{k}: null")
                elif isinstance(v, bool):
                    lines.append(f"{sp}{k}: {str(v).lower()}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{sp}{k}: {v}")
                else:
                    safe = str(v).replace('"', '\\"')
                    lines.append(f'{sp}{k}: "{safe}"')
        elif isinstance(value, list):
            if not value:
                lines.append(f"{sp}[]")
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{sp}-")
                    lines.extend(emit(item, indent + 2))
                else:
                    safe = str(item).replace('"', '\\"')
                    lines.append(f'{sp}- "{safe}"')
        else:
            lines.append(f"{sp}{value}")
        return lines

    path.write_text("# Auto-generated by phase1_schema_drilldown.py\n" + "\n".join(emit(obj)) + "\n")


def write_sql_templates(repo: Path, schema_map: dict[str, Any]) -> None:
    extract_dir = repo / "sql" / "wrds_extract"
    validate_dir = repo / "sql" / "wrds_validate"
    extract_dir.mkdir(parents=True, exist_ok=True)
    validate_dir.mkdir(parents=True, exist_ok=True)

    source_choices = schema_map["source_choices"]

    validation_sql = []
    validation_sql.append("-- Phase 1 metadata validation. Safe: row counts only.")
    validation_sql.append("-- Generated at " + utc_now())
    for alias, spec in source_choices.items():
        fqtn = spec.get("fqtn")
        if fqtn:
            validation_sql.append(f"select '{alias}' as alias, '{fqtn}' as table_name, count(*) as n_rows from {fqtn} limit 1;")
    (validate_dir / "phase1_target_row_counts.sql").write_text("\n".join(validation_sql) + "\n")

    crsp_fqtn = source_choices["crsp_daily_returns"].get("fqtn") or "crsp_a_stock.dsf"
    ccm_fqtn = source_choices["crsp_compustat_link"].get("fqtn") or "crsp_a_ccm.ccmxpf_linktable"
    comp_fqtn = source_choices["compustat_fundamentals"].get("fqtn") or "comp_na_daily_all.funda"
    supply_fqtn = source_choices["supply_chain_link"].get("fqtn") or "wrdsapps_link_supplychain.seglink"
    rp_map_fqtn = source_choices["ravenpack_entity_mapping"].get("fqtn") or "ravenpack_common.rpa_entity_mappings"

    (extract_dir / "01_crsp_daily_returns_template.sql").write_text(f"""-- Template only. Do not run until column names are confirmed from Phase 1.
select *
from {crsp_fqtn}
where date between '{SAMPLE_START}' and '{SAMPLE_END}';
""")

    (extract_dir / "02_ccm_link_template.sql").write_text(f"""-- Template only. Point-in-time CRSP/Compustat link extraction.
select *
from {ccm_fqtn}
where coalesce(linkenddt, date '2099-12-31') >= date '{SAMPLE_START}'
  and linkdt <= date '{SAMPLE_END}';
""")

    (extract_dir / "03_compustat_funda_template.sql").write_text(f"""-- Template only. Annual fundamentals extraction.
select *
from {comp_fqtn}
where datadate between '{SAMPLE_START}' and '{SAMPLE_END}'
  and indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C';
""")

    (extract_dir / "04_supply_chain_template.sql").write_text(f"""-- Template only. Supply-chain links / segment customer records.
select *
from {supply_fqtn};
""")

    rp_parts = schema_map.get("ravenpack_equities_partitions", [])
    rp_union = "\nunion all\n".join(
        [f"select * from {part['fqtn']}" for part in rp_parts]
    )
    if not rp_union:
        rp_union = "-- No RavenPack equities partitions found with select privilege."

    (extract_dir / "05_ravenpack_equities_union_template.sql").write_text(f"""-- Template only. RavenPack equities annual partition union.
-- Add column projection and date filters after Phase 1 confirms exact timestamp/date columns.
{rp_union};
""")

    (extract_dir / "06_ravenpack_mapping_template.sql").write_text(f"""-- Template only. RavenPack entity mapping.
select *
from {rp_map_fqtn};
""")


def render_html(out: Path, status: pd.DataFrame, cols: pd.DataFrame, packages: pd.DataFrame, schema_map: dict[str, Any], env: dict[str, Any]) -> None:
    cards = []
    family_order = ["crsp_stock", "crsp_ccm", "compustat", "supply_chain", "ravenpack_common", "ravenpack_dj", "ibes", "liquidity"]
    for family in family_order:
        g = status[status["family"].eq(family)]
        n_ready = int((g["has_select"] == True).sum()) if len(g) else 0
        n_total = int(len(g))
        cls = "good" if n_ready else "bad"
        pct = 0 if n_total == 0 else round(100 * n_ready / n_total)
        cards.append(f"""
        <div class="card {cls}">
          <div class="kicker">{html.escape(family.replace("_", " ").title())}</div>
          <h3>{n_ready} / {n_total} target tables selectable</h3>
          <div class="bar"><span style="width:{pct}%"></span></div>
          <p>{pct}% metadata readiness</p>
        </div>
        """)

    show_cols = [
        "family",
        "fqtn",
        "approx_rows",
        "has_select",
        "n_columns",
        "readiness",
        "matched_required_columns",
        "missing_required_columns",
    ]
    status_html = status[show_cols].sort_values(["family", "fqtn"]).to_html(index=False, escape=True, classes="data")

    source_choice_rows = []
    for alias, spec in schema_map.get("source_choices", {}).items():
        source_choice_rows.append({
            "alias": alias,
            "fqtn": spec.get("fqtn"),
            "readiness": spec.get("readiness"),
            "approx_rows": spec.get("approx_rows"),
        })
    source_choice_html = pd.DataFrame(source_choice_rows).to_html(index=False, escape=True, classes="data")

    rp_parts = pd.DataFrame(schema_map.get("ravenpack_equities_partitions", []))
    rp_html = rp_parts.to_html(index=False, escape=True, classes="data") if len(rp_parts) else "<p>No RavenPack equities partitions with select privilege were found.</p>"

    missing_pkg_html = packages.to_html(index=False, escape=True, classes="data")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Production Network Alpha — Phase 1 Schema Drilldown</title>
<style>
:root {{
  --bg: #07111f;
  --panel: #0f1e33;
  --text: #eef6ff;
  --muted: #9fb7ce;
  --line: rgba(255,255,255,.14);
  --good: #3ddc97;
  --bad: #ff6b6b;
  --accent: #7aa2ff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
  background: radial-gradient(circle at top left, #183b66, var(--bg) 42%);
  color: var(--text);
}}
header {{ padding: 46px 56px 28px; border-bottom: 1px solid var(--line); }}
h1 {{ margin: 0; font-size: 42px; letter-spacing: -.04em; }}
.subtitle {{ color: var(--muted); font-size: 17px; max-width: 1050px; line-height: 1.55; }}
main {{ padding: 28px 56px 60px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; margin: 22px 0 36px; }}
.card {{
  background: linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.035));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 18px 40px rgba(0,0,0,.18);
}}
.card h3 {{ margin: 7px 0 8px; font-size: 18px; }}
.card p {{ margin: 8px 0 0; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
.kicker {{ text-transform: uppercase; font-size: 11px; letter-spacing: .16em; color: var(--muted); }}
.good .kicker {{ color: var(--good); }}
.bad .kicker {{ color: var(--bad); }}
.bar {{ height: 8px; background: rgba(255,255,255,.11); border-radius: 999px; margin-top: 14px; overflow: hidden; }}
.bar span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--good)); }}
section {{ background: rgba(15,30,51,.78); border: 1px solid var(--line); border-radius: 22px; padding: 24px; margin: 22px 0; overflow: auto; }}
h2 {{ margin-top: 0; letter-spacing: -.02em; }}
table.data {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
table.data th {{ text-align: left; color: #d8eaff; background: rgba(255,255,255,.08); }}
table.data th, table.data td {{ padding: 9px 10px; border-bottom: 1px solid rgba(255,255,255,.09); vertical-align: top; }}
pre {{ white-space: pre-wrap; background: rgba(0,0,0,.28); border: 1px solid var(--line); border-radius: 14px; padding: 16px; color: #dbecff; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.pill {{ border: 1px solid var(--line); background: rgba(255,255,255,.07); border-radius: 999px; padding: 8px 12px; color: var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>Phase 1 Schema Drilldown</h1>
  <p class="subtitle">Targeted WRDS metadata, table privileges, and exact column inventory for the production-network alpha pipeline. This report is metadata-only and contains no raw vendor records.</p>
  <div class="meta">
    <span class="pill">Generated: {html.escape(utc_now())}</span>
    <span class="pill">Sample: {SAMPLE_START} to {SAMPLE_END}</span>
    <span class="pill">Python: {html.escape(sys.version.split()[0])}</span>
    <span class="pill">Selectable target tables: {int((status["has_select"] == True).sum())} / {len(status)}</span>
  </div>
</header>
<main>
  <div class="grid">{''.join(cards)}</div>
  <section><h2>Chosen source map</h2>{source_choice_html}</section>
  <section><h2>RavenPack equities partitions</h2>{rp_html}</section>
  <section><h2>Target table readiness</h2>{status_html}</section>
  <section><h2>Package status</h2>{missing_pkg_html}</section>
  <section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main>
</body>
</html>
"""
    out.write_text(doc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("Phase 1 WRDS targeted schema drilldown")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {project_root}")
    print(f"Output dir: {out_dir}")
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
        "git": run_cmd(["bash", "-lc", "command -v git || true"]),
        "module_git_probe": run_cmd(["bash", "-lc", "module avail git 2>&1 | head -80"], timeout=20),
    }

    packages = pkg_status()
    packages.to_csv(out_dir / "package_status.csv", index=False)
    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    targets = target_frame()
    targets.to_csv(out_dir / "target_table_manifest.csv", index=False)
    print(f"[INFO] Target table count: {len(targets)}")

    user = None
    try:
        db, user, user_source = connect_wrds()
        tables = get_tables(db, targets)
        cols = get_columns(db, tables)
        try:
            db.close()
        except Exception:
            pass
    except Exception:
        print("[ERROR] Phase 1 WRDS drilldown failed.")
        print(traceback.format_exc())
        return 2

    tables.to_csv(out_dir / "target_tables_raw.csv", index=False)
    cols.to_csv(out_dir / "target_columns.csv", index=False)

    status = enrich_status(tables, cols)
    status.to_csv(out_dir / "target_table_status.csv", index=False)

    schema_map = build_schema_map(status, cols, user)
    with (out_dir / "schema_map_phase1.json").open("w") as f:
        json.dump(schema_map, f, indent=2)

    write_yaml(project_root / "configs" / "schema_map.yml", schema_map)
    write_sql_templates(project_root, schema_map)

    render_html(
        out=out_dir / "phase1_schema_drilldown_report.html",
        status=status,
        cols=cols,
        packages=packages,
        schema_map=schema_map,
        env=env,
    )

    summary = [
        "# Phase 1 schema drilldown summary",
        "",
        f"- Generated at UTC: {utc_now()}",
        f"- Target tables checked: {len(status)}",
        f"- Tables with SELECT privilege: {int((status['has_select'] == True).sum())}",
        f"- Column rows discovered: {len(cols)}",
        f"- Report: {out_dir / 'phase1_schema_drilldown_report.html'}",
        f"- Target status: {out_dir / 'target_table_status.csv'}",
        f"- Target columns: {out_dir / 'target_columns.csv'}",
        f"- Updated schema map: {project_root / 'configs' / 'schema_map.yml'}",
    ]
    (out_dir / "PHASE1_SUMMARY.md").write_text("\n".join(summary) + "\n")

    print("\n".join(summary))
    print("[INFO] Missing packages:")
    print(packages[packages["status"].eq("missing")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
