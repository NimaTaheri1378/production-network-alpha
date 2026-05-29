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
PILOT_YEAR_DEFAULT = 2024
PILOT_TARGET_LIMIT_DEFAULT = 500
SQL_CHUNK_DEFAULT = 1000


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


def connect_wrds():
    import wrds

    user, source = detect_wrds_username()
    try:
        db = wrds.Connection(wrds_username=user, verbose=False) if user else wrds.Connection(verbose=False)
    except TypeError:
        db = wrds.Connection()
    try:
        db.raw_sql("set statement_timeout to '900000ms'")
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
    pkgs = ["pandas", "numpy", "pyarrow", "wrds", "sqlalchemy", "psycopg2", "matplotlib", "plotly", "duckdb", "polars"]
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
    return f"{prefix}_" + hashlib.sha256(b"pna_phase4_0_" + raw).hexdigest()[:16]


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


def build_target_universe(edges: pd.DataFrame, news: pd.DataFrame, pilot_year: int, target_limit: int, out_dir: Path) -> pd.DataFrame:
    edges = edges.copy()
    news = news.copy()
    for col in ["edge_start_date", "edge_end_date"]:
        edges[col] = to_dt(edges[col])
    news["news_signal_date"] = to_dt(news["news_signal_date"])
    news_yr = news[news["news_signal_date"].dt.year.eq(pilot_year)].copy()

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
        news_yr.assign(gvkey=clean_gvkey(news_yr["gvkey"]))
        .groupby("gvkey", dropna=False)
        .agg(own_news_node_days=("news_signal_date", "nunique"), own_news_events=("n_events", "sum"))
        .reset_index()
    )
    target = degree.merge(news_counts, on="gvkey", how="left")
    target["own_news_node_days"] = target["own_news_node_days"].fillna(0)
    target["own_news_events"] = target["own_news_events"].fillna(0)
    target = target.sort_values(["graph_degree", "own_news_node_days"], ascending=[False, False]).drop_duplicates("gvkey")
    target = target.head(target_limit).copy()
    target["permno"] = as_int(target["permno"])
    redact_identifiers(target).to_csv(out_dir / "phase4_0_pilot_target_universe_redacted.csv", index=False)
    return target


def build_spillover_features(edges: pd.DataFrame, news: pd.DataFrame, targets: pd.DataFrame, pilot_year: int, max_spillover_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = edges.copy()
    n = news.copy()
    target_gvkeys = set(targets["gvkey"].astype(str))

    for col in ["supplier_gvkey", "customer_gvkey"]:
        e[col] = clean_gvkey(e[col])
    for col in ["edge_start_date", "edge_end_date"]:
        e[col] = to_dt(e[col])
    for col in ["supplier_permno", "customer_permno"]:
        e[col] = as_int(e[col])

    n["gvkey"] = clean_gvkey(n["gvkey"])
    n["permno"] = as_int(n["permno"])
    n["news_signal_date"] = to_dt(n["news_signal_date"])
    n = n[n["news_signal_date"].dt.year.eq(pilot_year)].copy()

    for col in ["signed_news_shock", "positive_news_shock", "negative_news_shock", "abs_news_shock", "n_events", "n_stories"]:
        if col not in n.columns:
            n[col] = 0.0
        n[col] = pd.to_numeric(n[col], errors="coerce").fillna(0.0)

    e["edge_weight"] = pd.to_numeric(e.get("sales_share_supplier_reported", np.nan), errors="coerce")
    fallback_weight = float(e["edge_weight"].dropna().median()) if e["edge_weight"].notna().any() else 1.0
    e["edge_weight"] = e["edge_weight"].fillna(fallback_weight).clip(lower=0.0, upper=1.0)
    e["relationship_age_days"] = pd.to_numeric(e.get("relationship_age_days", np.nan), errors="coerce")
    e["supplier_customer_hhi_reported"] = pd.to_numeric(e.get("supplier_customer_hhi_reported", np.nan), errors="coerce")

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
    if rows.empty:
        raise RuntimeError("No active spillover rows in pilot. Check graph/news dates and target universe.")

    rows["weighted_signed_shock"] = rows["signed_news_shock"] * rows["edge_weight"]
    rows["weighted_abs_shock"] = rows["abs_news_shock"] * rows["edge_weight"]
    rows["weighted_positive_shock"] = rows["positive_news_shock"] * rows["edge_weight"]
    rows["weighted_negative_shock"] = rows["negative_news_shock"] * rows["edge_weight"]
    rows["target_permno"] = as_int(rows["target_permno"])
    rows["source_permno"] = as_int(rows["source_permno"])

    if len(rows) > max_spillover_rows:
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

    event_redacted_cols = [
        "signal_date", "direction", "source_gvkey", "target_gvkey", "edge_weight", "weighted_signed_shock", "weighted_abs_shock",
        "relationship_age_days", "supplier_customer_hhi_reported"
    ]
    rows_redacted = rows.rename(columns={"news_signal_date": "signal_date"})[event_redacted_cols].copy()
    return features, rows_redacted


def safe_sql(db, sql: str, label: str) -> pd.DataFrame:
    t0 = time.time()
    print(f"[WRDS] {label} ...")
    df = db.raw_sql(sql)
    df.columns = [str(c).lower() for c in df.columns]
    print(f"[WRDS] {label}: rows={len(df):,}, elapsed={time.time() - t0:.1f}s")
    return df


def query_crsp_returns(db, permnos: list[int], start_date: str, end_date: str, raw_dir: Path, chunk_size: int, force_refresh: bool) -> dict[str, pd.DataFrame]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dsf_path = raw_dir / f"crsp_dsf_pilot_{start_date}_{end_date}.parquet"
    dsi_path = raw_dir / f"crsp_dsi_pilot_{start_date}_{end_date}.parquet"
    dl_path = raw_dir / f"crsp_dsedelist_pilot_{start_date}_{end_date}.parquet"

    if dsf_path.exists() and dsi_path.exists() and dl_path.exists() and not force_refresh:
        return {"dsf": pd.read_parquet(dsf_path), "dsi": pd.read_parquet(dsi_path), "dl": pd.read_parquet(dl_path)}

    frames = []
    for i, vals in enumerate(chunks(permnos, chunk_size), start=1):
        sql = f"""
            select permno, date, ret, retx, prc, shrout, vol
            from crsp_a_stock.dsf
            where permno in {sql_in(vals)}
              and date between date {q(start_date)} and date {q(end_date)}
        """
        frames.append(safe_sql(db, sql, f"CRSP dsf pilot chunk {i}/{len(chunks(permnos, chunk_size))}"))
    dsf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    dsi_sql = f"""
        select date, vwretd, ewretd
        from crsp_a_stock.dsi
        where date between date {q(start_date)} and date {q(end_date)}
    """
    dsi = safe_sql(db, dsi_sql, "CRSP dsi pilot market returns")

    dl_frames = []
    for i, vals in enumerate(chunks(permnos, chunk_size), start=1):
        sql = f"""
            select permno, dlstdt, dlret, dlstcd
            from crsp_a_stock.dsedelist
            where permno in {sql_in(vals)}
              and dlstdt between date {q(start_date)} and date {q(end_date)}
        """
        dl_frames.append(safe_sql(db, sql, f"CRSP dsedelist pilot chunk {i}/{len(chunks(permnos, chunk_size))}"))
    dl = pd.concat(dl_frames, ignore_index=True) if dl_frames else pd.DataFrame()

    write_parquet(dsf, dsf_path)
    write_parquet(dsi, dsi_path)
    write_parquet(dl, dl_path)
    return {"dsf": dsf, "dsi": dsi, "dl": dl}


def prepare_returns(crsp: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        x["ret_mom_21d"] = (1.0 + x["ret_adj"]).shift(1).rolling(21, min_periods=10).apply(np.prod, raw=True) - 1.0
        x["ret_mom_126d"] = (1.0 + x["ret_adj"]).shift(1).rolling(126, min_periods=60).apply(np.prod, raw=True) - 1.0
        x["idio_vol_63d"] = x["abret_mkt"].shift(1).rolling(63, min_periods=20).std()
        x["dollar_vol_21d"] = x["dollar_vol"].shift(1).rolling(21, min_periods=10).mean()
        x["log_mktcap"] = np.log(x["mktcap"].where(x["mktcap"] > 0))
        feature_frames.append(x)
    ret_panel = pd.concat(feature_frames, ignore_index=True)

    labels = []
    keep_cols = ["permno", "date", "ret_adj", "vwretd", "abret_mkt", "ret_mom_21d", "ret_mom_126d", "idio_vol_63d", "dollar_vol_21d", "log_mktcap", "prc", "vol", "shrout"]
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
    return ret_panel, label_panel


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
    return matrix


def validation_report(matrix: pd.DataFrame, crsp_rows: int, spillover_rows: int, target_rows: int) -> dict[str, Any]:
    label_cov = {f"label_cov_{h}d": float(matrix[f"fwd_abret_{h}d"].notna().mean()) for h in HORIZONS}
    no_look = {f"no_lookahead_{h}d": bool(matrix.loc[matrix[f"fwd_abret_{h}d"].notna(), f"no_lookahead_ok_{h}d"].all()) for h in HORIZONS}
    pure_rows = int(matrix["pure_spillover_no_own_news"].fillna(False).sum())
    checks = {
        "target_universe_positive": target_rows > 0,
        "spillover_rows_positive": spillover_rows > 0,
        "crsp_rows_positive": crsp_rows > 0,
        "model_matrix_rows_ge_1000": len(matrix) >= 1000,
        "label_5d_coverage_ge_50pct": label_cov["label_cov_5d"] >= 0.50,
        "label_20d_coverage_ge_50pct": label_cov["label_cov_20d"] >= 0.50,
        "pure_spillover_rows_positive": pure_rows > 0,
        "no_lookahead_all_horizons": all(no_look.values()),
    }
    return {
        "generated_at_utc": utc_now(),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "rows": int(len(matrix)),
        "target_rows": int(target_rows),
        "spillover_event_rows_redacted": int(spillover_rows),
        "crsp_return_rows": int(crsp_rows),
        "pure_spillover_rows": pure_rows,
        "label_coverage": label_cov,
        "no_lookahead": no_look,
    }


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

    decile_rows = []
    work = matrix.copy()
    work = work[work["fwd_abret_5d"].notna() & work["spillover_signed_shock"].notna()].copy()
    if len(work) >= 100:
        work["signal_decile"] = pd.qcut(work["spillover_signed_shock"].rank(method="first"), 10, labels=False) + 1
        dec = work.groupby("signal_decile").agg(rows=("target_gvkey", "size"), mean_fwd_abret_5d=("fwd_abret_5d", "mean"), mean_fwd_abret_20d=("fwd_abret_20d", "mean"), mean_signal=("spillover_signed_shock", "mean")).reset_index()
        decile_rows.append(dec)
    deciles = pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()
    return by_date, horizon_stats, deciles


def make_figures(out_dir: Path, validation: dict[str, Any], horizon_stats: pd.DataFrame, deciles: pd.DataFrame, matrix: pd.DataFrame) -> list[dict[str, str]]:
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
    plt.title("Phase 4.0 pilot validation checks")
    plt.tight_layout()
    p = fig_dir / "phase4_0_validation_checks.png"
    plt.savefig(p, dpi=180)
    plt.close()
    made.append({"figure": "validation_checks", "path": str(p)})

    if not horizon_stats.empty:
        plt.figure(figsize=(9.5, 5.5))
        plt.plot(horizon_stats["horizon_days"], horizon_stats["rank_ic_spillover_signed"], marker="o", label="All rows")
        plt.plot(horizon_stats["horizon_days"], horizon_stats["pure_rank_ic_spillover_signed"], marker="o", label="No own-news rows")
        plt.axhline(0, linewidth=0.8)
        plt.xlabel("Forward horizon, trading days")
        plt.ylabel("Pilot rank IC")
        plt.title("Pilot spillover signal rank IC by horizon")
        plt.legend()
        plt.tight_layout()
        p = fig_dir / "phase4_0_rank_ic_by_horizon.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append({"figure": "rank_ic_by_horizon", "path": str(p)})

    if not deciles.empty:
        plt.figure(figsize=(9.5, 5.5))
        plt.bar(deciles["signal_decile"].astype(int), deciles["mean_fwd_abret_5d"])
        plt.axhline(0, linewidth=0.8)
        plt.xlabel("Spillover signed-shock decile")
        plt.ylabel("Mean 5-day forward market-adjusted return")
        plt.title("Pilot decile monotonicity check")
        plt.tight_layout()
        p = fig_dir / "phase4_0_decile_5d.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append({"figure": "decile_5d", "path": str(p)})

    vals = pd.to_numeric(matrix["spillover_signed_shock"], errors="coerce").dropna()
    if not vals.empty:
        cap = vals.abs().quantile(0.99)
        plt.figure(figsize=(9.5, 5.5))
        plt.hist(vals.clip(-cap, cap), bins=60)
        plt.xlabel("Spillover signed shock, clipped at p99 abs")
        plt.ylabel("Target-days")
        plt.title("Pilot spillover signal distribution")
        plt.tight_layout()
        p = fig_dir / "phase4_0_spillover_distribution.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append({"figure": "spillover_distribution", "path": str(p)})

    return made


def render_report(out_path: Path, summary: dict[str, Any], validation: dict[str, Any], by_date: pd.DataFrame, horizon_stats: pd.DataFrame, deciles: pd.DataFrame, figures: list[dict[str, str]], env: dict[str, Any]) -> None:
    def tbl(df: pd.DataFrame) -> str:
        return df.to_html(index=False, escape=True, classes="data") if df is not None and not df.empty else "<p>No rows.</p>"

    cards = []
    for label, value in [
        ("Pilot validation", "PASS" if validation.get("passed") else "FAIL"),
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
<html lang="en"><head><meta charset="utf-8"><title>Phase 4.0 Labels Pilot</title>
<style>
:root {{ --bg:#07111f; --text:#eef6ff; --muted:#9fb7ce; --line:rgba(255,255,255,.14); }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#183b66,var(--bg) 42%); color:var(--text); }}
header {{ padding:46px 56px 28px; border-bottom:1px solid var(--line); }} h1 {{ margin:0; font-size:42px; letter-spacing:-.04em; }} .subtitle {{ color:var(--muted); font-size:17px; max-width:1050px; line-height:1.55; }}
main {{ padding:28px 56px 60px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:18px; margin:22px 0 36px; }} .card {{ background:linear-gradient(180deg,rgba(255,255,255,.075),rgba(255,255,255,.035)); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 18px 40px rgba(0,0,0,.18); }} .card h3 {{ margin:7px 0 0; font-size:28px; }} .kicker {{ text-transform:uppercase; font-size:11px; letter-spacing:.16em; color:var(--muted); }}
section {{ background:rgba(15,30,51,.78); border:1px solid var(--line); border-radius:22px; padding:24px; margin:22px 0; overflow:auto; }} table.data {{ width:100%; border-collapse:collapse; font-size:13px; }} table.data th {{ text-align:left; color:#d8eaff; background:rgba(255,255,255,.08); }} table.data th, table.data td {{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.09); vertical-align:top; }} .figure img {{ width:100%; max-width:1100px; border-radius:16px; border:1px solid var(--line); background:white; }} pre {{ white-space:pre-wrap; background:rgba(0,0,0,.28); border:1px solid var(--line); border-radius:14px; padding:16px; color:#dbecff; }}
</style></head><body><header><h1>Phase 4.0 CRSP Labels Pilot</h1><p class="subtitle">Pilot-only construction of one-hop production-network spillover features, CRSP market-adjusted forward labels, and no-lookahead validations. Protected CRSP and model-matrix Parquet files remain local and outside the upload bundle.</p></header><main>
<div class="grid">{''.join(cards)}</div>
<section><h2>Validation</h2>{tbl(pd.DataFrame([validation]))}</section>
<section><h2>Horizon statistics</h2>{tbl(horizon_stats)}</section>
<section><h2>5-day decile check</h2>{tbl(deciles)}</section>
<section><h2>Daily aggregate summary</h2>{tbl(by_date.head(60))}</section>
<section><h2>Figures</h2>{''.join(fig_html)}</section>
<section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main></body></html>"""
    out_path.write_text(doc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--pilot-year", type=int, default=PILOT_YEAR_DEFAULT)
    parser.add_argument("--pilot-target-limit", type=int, default=PILOT_TARGET_LIMIT_DEFAULT)
    parser.add_argument("--sql-chunk-size", type=int, default=SQL_CHUNK_DEFAULT)
    parser.add_argument("--max-spillover-rows", type=int, default=250000)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    raw_dir = root / "data" / "raw" / "wrds" / "phase4_returns_pilot"
    processed_dir = root / "data" / "processed" / "model_matrix_pilot"
    edges_path = root / "data" / "processed" / "graph_backbone" / "edges_supplier_customer_common_us.parquet"
    news_path = root / "data" / "processed" / "news_shocks_full" / "node_day_news_shocks.parquet"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 4.0 CRSP labels/model matrix pilot")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {root}")
    print(f"Graph edges: {edges_path}")
    print(f"News shocks: {news_path}")
    print(f"Output dir: {out_dir}")
    print(f"Protected raw CRSP dir: {raw_dir}")
    print(f"Pilot year: {args.pilot_year}")
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
        targets = build_target_universe(edges, news, args.pilot_year, args.pilot_target_limit, out_dir)
        features, rows_redacted = build_spillover_features(edges, news, targets, args.pilot_year, args.max_spillover_rows)
        redact_identifiers(rows_redacted.head(10000)).to_csv(out_dir / "phase4_0_spillover_events_redacted_sample.csv", index=False)
        redact_identifiers(features.head(10000)).to_csv(out_dir / "phase4_0_spillover_features_redacted_head.csv", index=False)

        permnos = sorted(set(pd.to_numeric(features["target_permno"], errors="coerce").dropna().astype(int).tolist()))
        query_start = f"{args.pilot_year - 1}-07-01"
        query_end = f"{args.pilot_year + 1}-03-31"
        db, user, source = connect_wrds()
        print(f"[INFO] WRDS connected user={mask_user(user)} source={source}")
        crsp = query_crsp_returns(db, permnos, query_start, query_end, raw_dir, args.sql_chunk_size, args.force_refresh)
    except Exception:
        print("[ERROR] Phase 4.0 pilot extraction/build failed.")
        print(traceback.format_exc())
        return 2
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    try:
        ret_panel, label_panel = prepare_returns(crsp)
        features_mapped = map_signal_to_trading_day(features, crsp["dsi"]["date"])
        matrix = build_model_matrix(features_mapped, label_panel)
        matrix["signal_to_base_days"] = (to_dt(matrix["label_base_date"]) - to_dt(matrix["signal_date"])).dt.days
        by_date, horizon_stats, deciles = summarize_matrix(matrix)
        validation = validation_report(matrix, len(ret_panel), len(rows_redacted), len(targets))
    except Exception:
        print("[ERROR] Phase 4.0 pilot labels/model matrix construction failed.")
        print(traceback.format_exc())
        return 3

    write_parquet(ret_panel, processed_dir / "pilot_crsp_return_panel_features.parquet")
    write_parquet(label_panel, processed_dir / "pilot_crsp_forward_labels.parquet")
    write_parquet(matrix, processed_dir / "pilot_spillover_model_matrix.parquet")

    by_date.to_csv(out_dir / "phase4_0_daily_aggregate_summary.csv", index=False)
    horizon_stats.to_csv(out_dir / "phase4_0_horizon_stats.csv", index=False)
    deciles.to_csv(out_dir / "phase4_0_decile_check_5d.csv", index=False)
    pd.DataFrame([validation]).to_csv(out_dir / "phase4_0_validation_report_flat.csv", index=False)
    with (out_dir / "phase4_0_validation_report.json").open("w") as f:
        json.dump(validation, f, indent=2, default=str)

    protected_inventory = []
    for base in [raw_dir, processed_dir]:
        for p in sorted(base.glob("*.parquet")):
            protected_inventory.append({"file": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p), "protected_local_only": True})
    pd.DataFrame(protected_inventory).to_csv(out_dir / "protected_local_phase4_0_inventory.csv", index=False)

    summary = {
        "generated_at_utc": utc_now(),
        "pilot_year": args.pilot_year,
        "pilot_passed": validation["passed"],
        "target_gvkeys": int(targets["gvkey"].nunique()),
        "model_matrix_rows": int(len(matrix)),
        "pure_spillover_rows": int(validation["pure_spillover_rows"]),
        "crsp_return_rows": int(validation["crsp_return_rows"]),
        "protected_model_matrix": str(processed_dir / "pilot_spillover_model_matrix.parquet"),
        "protected_return_panel": str(processed_dir / "pilot_crsp_return_panel_features.parquet"),
    }
    with (out_dir / "phase4_0_quality_summary.json").open("w") as f:
        json.dump({"summary": summary, "validation": validation}, f, indent=2, default=str)

    figures = make_figures(out_dir, validation, horizon_stats, deciles, matrix)
    render_report(out_dir / "phase4_0_labels_pilot_report.html", summary, validation, by_date, horizon_stats, deciles, figures, env)

    lines = [
        "# Phase 4.0 CRSP labels/model-matrix pilot summary", "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Pilot validation passed: {summary['pilot_passed']}",
        f"- Pilot year: {args.pilot_year}",
        f"- Target GVKEYs: {summary['target_gvkeys']:,}",
        f"- Model matrix rows: {summary['model_matrix_rows']:,}",
        f"- Pure spillover rows: {summary['pure_spillover_rows']:,}",
        f"- CRSP return rows: {summary['crsp_return_rows']:,}",
        f"- Report: {out_dir / 'phase4_0_labels_pilot_report.html'}", "",
        "Data policy: protected CRSP and model-matrix Parquet files remain local and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE4_0_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if validation["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
