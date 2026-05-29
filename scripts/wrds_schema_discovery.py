from __future__ import annotations

import argparse
import datetime as dt
import html
import importlib.metadata as importlib_metadata
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

import pandas as pd


ALIAS_SPECS: dict[str, dict[str, list[str]]] = {
    "crsp_stock_returns": {
        "schema": ["crsp", "crsp_a_stock", "crspq", "crspa"],
        "table": ["dsf", "msf", "stock", "daily", "monthly", "returns", "dsenames", "stocknames"],
        "columns": ["permno", "permco", "date", "ret", "retx", "prc", "shrout", "vol", "cfacpr", "cfacshr"],
    },
    "crsp_compustat_link": {
        "schema": ["crsp", "ccm", "crsp_a_ccm", "comp"],
        "table": ["ccm", "ccmxpf", "link", "lnkhist", "linktable"],
        "columns": ["gvkey", "permno", "permco", "linkdt", "linkenddt", "linkprim", "linktype", "liid"],
    },
    "compustat_fundamentals": {
        "schema": ["comp", "compa", "comp_na_daily_all", "comp_na"],
        "table": ["funda", "fundq", "company", "security", "names", "co_hgic"],
        "columns": ["gvkey", "datadate", "fyear", "at", "sale", "ceq", "seq", "lt", "ni", "prcc_f", "csho"],
    },
    "compustat_segments_supply_chain": {
        "schema": ["comp_segments_hist_daily", "compseg", "segments", "comp", "wrdsapps"],
        "table": ["segment", "segments", "customer", "customers", "supplier", "busseg", "geoseg", "seg_customer"],
        "columns": ["gvkey", "srcdate", "datadate", "conm", "cnms", "sales", "revt", "stype", "cid", "cgvkey"],
    },
    "wrdsapps_supply_chain_link": {
        "schema": ["wrdsapps", "wrdsapps_link_supplychain", "supplychain"],
        "table": ["supply", "supplychain", "customer", "supplier", "link", "relationship", "relations"],
        "columns": ["gvkey", "supplier_gvkey", "customer_gvkey", "permno", "ticker", "startdate", "enddate"],
    },
    "ravenpack_dow_jones_news": {
        "schema": ["ravenpack_dj", "ravenpack_common", "raven", "ravenpack", "rp", "dj"],
        "table": ["raven", "rp", "dj", "news", "story", "event", "entity", "ess", "rpa"],
        "columns": ["rp_entity_id", "entity_id", "timestamp_utc", "event_timestamp", "relevance", "sentiment", "novelty", "event"],
    },
    "ibes_attention_expectations": {
        "schema": ["tr_ibes", "ibes"],
        "table": ["det", "statsum", "names", "id", "link", "act", "rec", "epsus", "guidance"],
        "columns": ["ticker", "cusip", "oftic", "fpedats", "analys", "numest", "meanest", "medest", "stdev"],
    },
    "taq_daily_liquidity": {
        "schema": ["taq", "taqms", "nyse_taq", "wrds_taq", "contrib_liquidity_taq"],
        "table": ["taq", "daily", "cq", "ct", "nbbo", "quote", "trade", "mast", "liquidity", "spread"],
        "columns": ["date", "time_m", "sym_root", "ticker", "bid", "ask", "price", "size", "spread"],
    },
}

PACKAGE_DISTS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "linearmodels": "linearmodels",
    "wrds": "wrds",
    "psycopg2": "psycopg2",
    "sqlalchemy": "sqlalchemy",
    "pyarrow": "pyarrow",
    "fastparquet": "fastparquet",
    "duckdb": "duckdb",
    "polars": "polars",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "torch": "torch",
    "plotly": "plotly",
    "dash": "dash",
    "dash_cytoscape": "dash-cytoscape",
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def mask_user(username: str | None) -> str | None:
    if not username:
        return None
    if len(username) <= 3:
        return username[0] + "***"
    return username[:2] + "***" + username[-1]


def run_cmd(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}


def package_versions() -> list[dict[str, str]]:
    rows = []
    for import_name, dist_name in PACKAGE_DISTS.items():
        try:
            version = importlib_metadata.version(dist_name)
            status = "ok"
        except importlib_metadata.PackageNotFoundError:
            version = ""
            status = "missing"
        rows.append({"package": import_name, "distribution": dist_name, "version": version, "status": status})
    return rows


def read_meminfo() -> dict[str, Any]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        parts = val.strip().split()
        if not parts:
            continue
        try:
            kb = int(parts[0])
        except ValueError:
            continue
        values[key] = kb
    return {
        "MemTotal_GB": round(values.get("MemTotal", 0) / 1024 / 1024, 2),
        "MemAvailable_GB": round(values.get("MemAvailable", 0) / 1024 / 1024, 2),
        "SwapTotal_GB": round(values.get("SwapTotal", 0) / 1024 / 1024, 2),
    }


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


def safe_sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def score_name(schema: str, table: str, spec: dict[str, list[str]]) -> int:
    schema_l = schema.lower()
    table_l = table.lower()
    full = f"{schema_l}.{table_l}"
    score = 0

    for kw in spec.get("schema", []):
        kw_l = kw.lower()
        if schema_l == kw_l:
            score += 10
        elif kw_l in schema_l:
            score += 5
        elif kw_l in full:
            score += 2

    for kw in spec.get("table", []):
        kw_l = kw.lower()
        if table_l == kw_l:
            score += 14
        elif re.search(rf"(^|[_\W]){re.escape(kw_l)}($|[_\W])", table_l):
            score += 8
        elif kw_l in table_l:
            score += 4
        elif kw_l in full:
            score += 2

    return score


def write_schema_map(path: Path, schema_map: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Auto-generated by scripts/wrds_schema_discovery.py")
    lines.append(f"generated_at_utc: {schema_map.get('generated_at_utc')}")
    lines.append(f"wrds_connected: {str(schema_map.get('wrds_connected')).lower()}")
    lines.append(f"wrds_user_masked_for_log: {schema_map.get('wrds_user_masked_for_log')}")
    lines.append("aliases:")
    aliases = schema_map.get("aliases", {})
    for alias, block in aliases.items():
        lines.append(f"  {alias}:")
        lines.append(f"    status: {block.get('status')}")
        lines.append("    candidates:")
        for cand in block.get("candidates", []):
            lines.append(f"      - schema: {cand.get('schema')}")
            lines.append(f"        table: {cand.get('table')}")
            lines.append(f"        score: {cand.get('score')}")
            approx = cand.get("approx_rows")
            lines.append(f"        approx_rows: {approx if approx is not None else 'null'}")
            cols = cand.get("matched_columns", [])
            if cols:
                lines.append("        matched_columns:")
                for col in cols:
                    lines.append(f"          - {col}")
            else:
                lines.append("        matched_columns: []")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def fetch_wrds_schema(out_dir: Path) -> dict[str, Any]:
    started = time.time()
    wrds_user, user_source = detect_wrds_username()
    result: dict[str, Any] = {
        "connected": False,
        "username_masked": mask_user(wrds_user),
        "username_source": user_source,
        "error": None,
        "elapsed_seconds": None,
        "tables_rows": 0,
        "candidate_rows": 0,
    }

    try:
        import wrds  # type: ignore

        print(f"[INFO] Connecting to WRDS as {mask_user(wrds_user) or 'default user'}...")
        try:
            db = wrds.Connection(wrds_username=wrds_user, verbose=False) if wrds_user else wrds.Connection(verbose=False)
        except TypeError:
            db = wrds.Connection()

        try:
            db.raw_sql("set statement_timeout to '120000ms'")
        except Exception:
            pass

        tables_sql = """
        select
            t.table_schema,
            t.table_name,
            t.table_type
        from information_schema.tables t
        where t.table_schema not in ('information_schema', 'pg_catalog')
        order by t.table_schema, t.table_name
        """
        print("[INFO] Reading information_schema.tables...")
        tables = db.raw_sql(tables_sql)
        tables.columns = [c.lower() for c in tables.columns]

        rows_sql = """
        select
            n.nspname as table_schema,
            c.relname as table_name,
            c.reltuples::bigint as approx_rows
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname not in ('information_schema', 'pg_catalog')
          and c.relkind in ('r', 'p', 'v', 'm')
        """
        try:
            print("[INFO] Reading approximate row counts from pg_class...")
            row_counts = db.raw_sql(rows_sql)
            row_counts.columns = [c.lower() for c in row_counts.columns]
            tables = tables.merge(row_counts, on=["table_schema", "table_name"], how="left")
        except Exception as exc:
            print(f"[WARN] Approximate row counts unavailable: {exc}")
            tables["approx_rows"] = pd.NA

        tables.to_csv(out_dir / "schema_tables_all.csv", index=False)
        print(f"[INFO] Wrote {len(tables):,} table metadata rows.")

        candidate_records = []
        for alias, spec in ALIAS_SPECS.items():
            for row in tables.itertuples(index=False):
                schema = str(getattr(row, "table_schema"))
                table = str(getattr(row, "table_name"))
                s = score_name(schema, table, spec)
                if s > 0:
                    candidate_records.append(
                        {
                            "alias": alias,
                            "table_schema": schema,
                            "table_name": table,
                            "table_type": str(getattr(row, "table_type", "")),
                            "approx_rows": getattr(row, "approx_rows", pd.NA),
                            "name_score": s,
                        }
                    )

        candidates = pd.DataFrame(candidate_records)
        if candidates.empty:
            candidates = pd.DataFrame(
                columns=[
                    "alias",
                    "table_schema",
                    "table_name",
                    "table_type",
                    "approx_rows",
                    "name_score",
                    "matched_columns",
                    "column_score",
                    "total_score",
                ]
            )
        else:
            candidates = candidates.sort_values(
                ["alias", "name_score", "approx_rows"],
                ascending=[True, False, False],
            )
            pair_list = list(
                candidates[["table_schema", "table_name"]]
                .drop_duplicates()
                .head(500)
                .itertuples(index=False, name=None)
            )
            columns = pd.DataFrame(
                columns=["table_schema", "table_name", "column_name", "data_type", "ordinal_position"]
            )

            if pair_list:
                conditions = [
                    f"(table_schema={safe_sql_literal(schema)} and table_name={safe_sql_literal(table)})"
                    for schema, table in pair_list
                ]
                col_sql = f"""
                select
                    table_schema,
                    table_name,
                    column_name,
                    data_type,
                    ordinal_position
                from information_schema.columns
                where {" or ".join(conditions)}
                order by table_schema, table_name, ordinal_position
                """
                try:
                    print(f"[INFO] Reading columns for {len(pair_list):,} candidate tables...")
                    columns = db.raw_sql(col_sql)
                    columns.columns = [c.lower() for c in columns.columns]
                    columns.to_csv(out_dir / "schema_columns_candidates.csv", index=False)
                except Exception as exc:
                    print(f"[WARN] Candidate column metadata unavailable: {exc}")

            col_map: dict[tuple[str, str], set[str]] = {}
            if not columns.empty:
                for (schema, table), group in columns.groupby(["table_schema", "table_name"]):
                    col_map[(str(schema), str(table))] = set(group["column_name"].str.lower().tolist())

            enriched = []
            for row in candidates.itertuples(index=False):
                alias = str(row.alias)
                schema = str(row.table_schema)
                table = str(row.table_name)
                present = col_map.get((schema, table), set())
                wanted = [c.lower() for c in ALIAS_SPECS[alias].get("columns", [])]
                matched = [c for c in wanted if c in present]
                rec = row._asdict()
                rec["matched_columns"] = ",".join(matched)
                rec["column_score"] = 3 * len(matched)
                rec["total_score"] = int(row.name_score) + int(rec["column_score"])
                enriched.append(rec)

            candidates = pd.DataFrame(enriched).sort_values(
                ["alias", "total_score", "name_score", "approx_rows"],
                ascending=[True, False, False, False],
            )

        candidates.to_csv(out_dir / "schema_candidates.csv", index=False)

        schema_map = {
            "generated_at_utc": utc_now(),
            "wrds_connected": True,
            "wrds_user_masked_for_log": mask_user(wrds_user),
            "aliases": {},
        }

        for alias in ALIAS_SPECS:
            top = candidates[candidates["alias"] == alias].head(10) if not candidates.empty else pd.DataFrame()
            schema_map["aliases"][alias] = {
                "status": "candidate_found" if len(top) else "not_found",
                "candidates": [
                    {
                        "schema": str(r.table_schema),
                        "table": str(r.table_name),
                        "score": int(r.total_score),
                        "approx_rows": None if pd.isna(r.approx_rows) else int(float(r.approx_rows)),
                        "matched_columns": [] if not str(r.matched_columns) else [x for x in str(r.matched_columns).split(",") if x],
                    }
                    for r in top.itertuples(index=False)
                ],
            }

        write_schema_map(out_dir.parent.parent / "configs" / "schema_map.yml", schema_map)
        with (out_dir / "schema_map.json").open("w") as f:
            json.dump(schema_map, f, indent=2)

        try:
            db.close()
        except Exception:
            pass

        result.update(
            {
                "connected": True,
                "tables_rows": int(len(tables)),
                "candidate_rows": int(len(candidates)),
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
        return result

    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        result["elapsed_seconds"] = round(time.time() - started, 2)
        print("[ERROR] WRDS schema discovery failed.")
        print(result["traceback"])
        return result


def render_html_report(out_path: Path, env: dict[str, Any], packages: list[dict[str, str]], wrds_result: dict[str, Any], out_dir: Path) -> None:
    candidates_path = out_dir / "schema_candidates.csv"
    tables_path = out_dir / "schema_tables_all.csv"

    candidates = pd.read_csv(candidates_path) if candidates_path.exists() else pd.DataFrame()
    tables = pd.read_csv(tables_path) if tables_path.exists() else pd.DataFrame()
    package_df = pd.DataFrame(packages)

    cards = []
    for alias in ALIAS_SPECS:
        cls = "bad"
        status = "Not found"
        label = "No candidate"
        score = 0
        if not candidates.empty and alias in set(candidates["alias"]):
            top = candidates[candidates["alias"] == alias].sort_values("total_score", ascending=False).head(1)
            if len(top):
                r = top.iloc[0]
                cls = "good"
                status = "Candidate found"
                label = f"{r['table_schema']}.{r['table_name']}"
                score = int(r["total_score"])
        cards.append(
            f"""
            <div class="card {cls}">
              <div class="kicker">{html.escape(status)}</div>
              <h3>{html.escape(alias.replace('_', ' ').title())}</h3>
              <p>{html.escape(label)}</p>
              <div class="bar"><span style="width:{min(score, 60) / 60 * 100:.0f}%"></span></div>
              <div class="score">score {score}</div>
            </div>
            """
        )

    if not candidates.empty:
        show_cols = [c for c in ["alias", "table_schema", "table_name", "approx_rows", "name_score", "column_score", "total_score", "matched_columns"] if c in candidates.columns]
        top_candidates_html = (
            candidates.sort_values(["alias", "total_score"], ascending=[True, False])
            .groupby("alias")
            .head(8)[show_cols]
            .to_html(index=False, escape=True, classes="data")
        )
    else:
        top_candidates_html = "<p>No candidate tables were produced.</p>"

    if not tables.empty:
        schema_summary_html = (
            tables.groupby("table_schema")
            .size()
            .sort_values(ascending=False)
            .head(40)
            .reset_index(name="n_tables")
            .to_html(index=False, escape=True, classes="data")
        )
    else:
        schema_summary_html = "<p>No table summary available.</p>"

    gpu_text = env.get("gpu", {}).get("stdout") or env.get("gpu", {}).get("stderr") or "nvidia-smi unavailable"

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Production Network Alpha Phase 0</title>
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
.subtitle {{ color: var(--muted); font-size: 17px; max-width: 1000px; line-height: 1.55; }}
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
.card p {{ margin: 0; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
.kicker {{ text-transform: uppercase; font-size: 11px; letter-spacing: .16em; color: var(--muted); }}
.good .kicker {{ color: var(--good); }}
.bad .kicker {{ color: var(--bad); }}
.bar {{ height: 8px; background: rgba(255,255,255,.11); border-radius: 999px; margin-top: 14px; overflow: hidden; }}
.bar span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--good)); }}
.score {{ margin-top: 8px; color: var(--muted); font-size: 12px; }}
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
  <h1>Production Network Alpha</h1>
  <p class="subtitle">Phase 0 WRDS schema discovery and environment audit. This confirms live table access before heavy ETL.</p>
  <div class="meta">
    <span class="pill">Generated: {html.escape(utc_now())}</span>
    <span class="pill">WRDS connected: {html.escape(str(wrds_result.get("connected")))}</span>
    <span class="pill">WRDS user: {html.escape(str(wrds_result.get("username_masked")))}</span>
    <span class="pill">Tables scanned: {html.escape(str(wrds_result.get("tables_rows")))}</span>
    <span class="pill">Candidate rows: {html.escape(str(wrds_result.get("candidate_rows")))}</span>
  </div>
</header>
<main>
  <div class="grid">{''.join(cards)}</div>
  <section><h2>Top candidate tables</h2>{top_candidates_html}</section>
  <section><h2>Accessible schema summary</h2>{schema_summary_html}</section>
  <section><h2>Python package audit</h2>{package_df.to_html(index=False, escape=True, classes="data")}</section>
  <section><h2>Runtime environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
  <section><h2>GPU snapshot</h2><pre>{html.escape(gpu_text)}</pre></section>
  <section><h2>WRDS diagnostic</h2><pre>{html.escape(json.dumps(wrds_result, indent=2, default=str))}</pre></section>
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
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("Phase 0 WRDS schema discovery")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {project_root}")
    print(f"Output dir: {out_dir}")
    print(f"Log dir: {log_dir}")
    print(f"Threads requested: {args.threads}")
    print("================================================================================")

    env = {
        "utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_env": {
            "PNA_THREADS": os.environ.get("PNA_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS"),
        },
        "memory": read_meminfo(),
        "gpu": run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        ),
    }

    packages = package_versions()
    pd.DataFrame(packages).to_csv(out_dir / "package_audit.csv", index=False)

    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    wrds_result = fetch_wrds_schema(out_dir=out_dir)

    with (out_dir / "wrds_discovery_status.json").open("w") as f:
        json.dump(wrds_result, f, indent=2, default=str)

    render_html_report(
        out_path=out_dir / "schema_discovery_report.html",
        env=env,
        packages=packages,
        wrds_result=wrds_result,
        out_dir=out_dir,
    )

    summary_lines = [
        "# Phase 0 summary",
        "",
        f"- Generated at UTC: {utc_now()}",
        f"- WRDS connected: {wrds_result.get('connected')}",
        f"- WRDS user: {wrds_result.get('username_masked')}",
        f"- Tables scanned: {wrds_result.get('tables_rows')}",
        f"- Candidate rows: {wrds_result.get('candidate_rows')}",
        f"- HTML report: {out_dir / 'schema_discovery_report.html'}",
        f"- Schema map: {project_root / 'configs' / 'schema_map.yml'}",
    ]
    (out_dir / "PHASE0_SUMMARY.md").write_text("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    return 0 if wrds_result.get("connected") else 2


if __name__ == "__main__":
    raise SystemExit(main())
