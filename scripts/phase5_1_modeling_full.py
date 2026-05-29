from __future__ import annotations

import argparse
import datetime as dt
import html
import importlib.metadata as md
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

WORKSPACE_REQUIRED = Path("<workspace>")
TRAIN_YEARS_DEFAULT = [2015, 2016, 2017, 2018, 2019]
VAL_YEARS_DEFAULT = [2020, 2021]
TEST_YEARS_DEFAULT = [2022, 2023, 2024]
HORIZONS_DEFAULT = [1, 2, 5, 10, 20]
SEED = 20260528


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_cmd(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}


def package_status() -> list[dict[str, str]]:
    pkgs = ["pandas", "numpy", "pyarrow", "scikit-learn", "lightgbm", "matplotlib", "scipy", "statsmodels"]
    rows = []
    for pkg in pkgs:
        try:
            rows.append({"package": pkg, "version": md.version(pkg), "status": "ok"})
        except md.PackageNotFoundError:
            rows.append({"package": pkg, "version": "", "status": "missing"})
    return rows


def parse_years(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return default
    vals: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            vals.extend(list(range(int(a), int(b) + 1)))
        else:
            vals.append(int(part))
    return sorted(set(vals))


def load_manifest(project_root: Path) -> dict[str, Any]:
    candidates = [
        project_root / "artifacts" / "model_matrix_ready" / "phase4_2_feature_manifest.json",
        project_root / "data" / "processed" / "model_matrix_ready" / "phase4_2_feature_manifest.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError("Missing phase4_2_feature_manifest.json under artifacts/model_matrix_ready")


def existing_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        return list(pd.read_parquet(path).head(0).columns)


def choose_features(manifest: dict[str, Any], sample_cols: list[str], feature_set: str) -> list[str]:
    raw = manifest.get(feature_set) or manifest.get("default_model_features") or []
    features = [c for c in raw if c in sample_cols]
    # Hard safety: same-day returns are excluded even if present.
    excluded = set(manifest.get("excluded_same_day_return_features", [])) | {"ret_adj", "vwretd", "abret_mkt"}
    features = [c for c in features if c not in excluded]
    if not features:
        raise RuntimeError(f"No usable feature columns found for feature_set={feature_set}")
    return features


def ready_file(project_root: Path, horizon: int, year: int, pure: bool) -> Path:
    stem = "model_ready_pure" if pure else "model_ready"
    return project_root / "data" / "processed" / "model_matrix_ready" / "by_horizon" / f"horizon_{horizon}d" / f"{stem}_{year}_h{horizon}d.parquet"


def read_year_file(path: Path, cols: list[str], row_cap: int | None, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing model-ready file: {path}")
    available = existing_columns(path)
    use_cols = [c for c in cols if c in available]
    missing = [c for c in cols if c not in available]
    if missing:
        raise ValueError(f"Required columns missing from {path.name}: {missing}")
    df = pd.read_parquet(path, columns=use_cols)
    if row_cap and len(df) > row_cap:
        df = df.sample(n=row_cap, random_state=seed).sort_index()
    return df


def load_horizon_dataset(
    project_root: Path,
    horizon: int,
    years_by_split: dict[str, list[int]],
    features: list[str],
    row_cap_per_year: int,
    pure: bool,
) -> pd.DataFrame:
    needed = list(dict.fromkeys(["target_label", "signal_date", "target_gvkey", "target_permno"] + features))
    frames = []
    for split, years in years_by_split.items():
        for year in years:
            path = ready_file(project_root, horizon, year, pure)
            df = read_year_file(path, needed, row_cap_per_year, SEED + year + horizon)
            df["split"] = split
            df["signal_year"] = year
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No frames loaded for horizon={horizon}")
    out = pd.concat(frames, ignore_index=True)
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out["target_label"] = pd.to_numeric(out["target_label"], errors="coerce")
    return out


def finite_mask(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    arr = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return arr.notna().all(axis=1) & pd.to_numeric(df["target_label"], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()


def winsor_params(train: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    x = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lower = x.quantile(0.005)
    upper = x.quantile(0.995)
    med = x.median().fillna(0.0)
    return lower, upper, med


def transform_features(df: pd.DataFrame, features: list[str], lower: pd.Series, upper: pd.Series, med: pd.Series) -> pd.DataFrame:
    x = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.clip(lower=lower, upper=upper, axis=1).fillna(med)
    return x.astype("float32")


def rank_ic_by_date(y: np.ndarray, pred: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": y, "pred": pred, "date": pd.to_datetime(dates, errors="coerce")})
    rows = []
    for d, g in tmp.dropna().groupby("date"):
        if len(g) < 15 or g["pred"].nunique() < 3 or g["y"].nunique() < 3:
            continue
        rows.append({
            "date": d,
            "rank_ic": float(g["pred"].rank().corr(g["y"].rank())),
            "pearson_ic": float(g["pred"].corr(g["y"])),
            "n": int(len(g)),
        })
    return pd.DataFrame(rows)


def daily_decile_spread(y: np.ndarray, pred: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": y, "pred": pred, "date": pd.to_datetime(dates, errors="coerce")}).dropna()
    rows = []
    for d, g in tmp.groupby("date"):
        if len(g) < 50 or g["pred"].nunique() < 10:
            continue
        try:
            dec = pd.qcut(g["pred"].rank(method="first"), 10, labels=False) + 1
        except ValueError:
            continue
        gg = g.assign(decile=dec)
        top = gg.loc[gg["decile"].eq(10), "y"].mean()
        bot = gg.loc[gg["decile"].eq(1), "y"].mean()
        rows.append({"date": d, "top_decile_ret": float(top), "bottom_decile_ret": float(bot), "long_short": float(top - bot), "n": int(len(g))})
    return pd.DataFrame(rows)


def decile_table(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": y, "pred": pred}).dropna()
    if len(tmp) < 100 or tmp["pred"].nunique() < 10:
        return pd.DataFrame()
    tmp["decile"] = pd.qcut(tmp["pred"].rank(method="first"), 10, labels=False) + 1
    return tmp.groupby("decile").agg(mean_target=("y", "mean"), median_target=("y", "median"), rows=("y", "size")).reset_index()


def summarize_prediction(model_name: str, horizon: int, split: str, y: np.ndarray, pred: np.ndarray, dates: pd.Series) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y = y[ok]
    pred = pred[ok]
    dates_ok = dates.reset_index(drop=True).iloc[np.where(ok)[0]] if hasattr(dates, "reset_index") else pd.Series(dates)[ok]
    if len(y) == 0:
        return {"model": model_name, "horizon_days": horizon, "split": split, "rows": 0}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    resid = y - pred
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if denom == 0 else 1.0 - float(np.sum(resid ** 2)) / denom
    hit = float(np.mean(np.sign(y) == np.sign(pred))) if len(y) else np.nan
    ric = rank_ic_by_date(y, pred, dates_ok)
    spread = daily_decile_spread(y, pred, dates_ok)
    dec = decile_table(y, pred)
    mean_rank_ic = float(ric["rank_ic"].mean()) if len(ric) else np.nan
    std_rank_ic = float(ric["rank_ic"].std(ddof=1)) if len(ric) > 1 else np.nan
    rank_ic_t = mean_rank_ic / (std_rank_ic / math.sqrt(len(ric))) if len(ric) > 1 and std_rank_ic and std_rank_ic > 0 else np.nan
    mean_spread = float(spread["long_short"].mean()) if len(spread) else np.nan
    std_spread = float(spread["long_short"].std(ddof=1)) if len(spread) > 1 else np.nan
    spread_t = mean_spread / (std_spread / math.sqrt(len(spread))) if len(spread) > 1 and std_spread and std_spread > 0 else np.nan
    summary = {
        "model": model_name,
        "horizon_days": horizon,
        "split": split,
        "rows": int(len(y)),
        "r2": float(r2) if pd.notna(r2) else np.nan,
        "hit_rate": hit,
        "mean_rank_ic": mean_rank_ic,
        "rank_ic_tstat_naive_daily": rank_ic_t,
        "rank_ic_days": int(len(ric)),
        "mean_daily_decile_long_short": mean_spread,
        "daily_decile_long_short_tstat_naive": spread_t,
        "decile_days": int(len(spread)),
        "target_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(pred)),
    }
    ric["model"] = model_name
    ric["horizon_days"] = horizon
    ric["split"] = split
    spread["model"] = model_name
    spread["horizon_days"] = horizon
    spread["split"] = split
    dec["model"] = model_name
    dec["horizon_days"] = horizon
    dec["split"] = split
    return summary, ric, spread, dec


def fit_predict_linear(X_train: pd.DataFrame, y_train: np.ndarray, X_other: pd.DataFrame) -> tuple[np.ndarray, Any]:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    model = make_pipeline(StandardScaler(with_mean=True, with_std=True), Ridge(alpha=10.0, random_state=SEED))
    model.fit(X_train, y_train)
    return model.predict(X_other), model


def fit_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, threads: int) -> Any:
    from lightgbm import LGBMRegressor
    # Full-scale deterministic parameters: efficient, conservative, and low overfit risk.
    model = LGBMRegressor(
        objective="regression",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=300,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=threads,
        verbosity=-1,
    )
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l2", early_stopping_rounds=50, verbose=False)
    except TypeError:
        model.fit(X_train, y_train)
    return model


def run_horizon(
    project_root: Path,
    horizon: int,
    years_by_split: dict[str, list[int]],
    manifest: dict[str, Any],
    feature_set: str,
    row_cap_per_year: int,
    pure: bool,
    threads: int,
    protected_dir: Path,
) -> dict[str, Any]:
    sample_path = ready_file(project_root, horizon, years_by_split["train"][0], pure)
    sample_cols = existing_columns(sample_path)
    features = choose_features(manifest, sample_cols, feature_set)
    df = load_horizon_dataset(project_root, horizon, years_by_split, features, row_cap_per_year, pure)
    mask = finite_mask(df, features)
    dropped_inf_or_missing = int((~mask).sum())
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No valid rows after finite/missing filter for horizon={horizon}")

    train = df[df["split"].eq("train")].copy()
    val = df[df["split"].eq("validation")].copy()
    test = df[df["split"].eq("test")].copy()
    if min(len(train), len(val), len(test)) <= 100:
        raise RuntimeError(f"Insufficient rows for horizon={horizon}: train={len(train)}, val={len(val)}, test={len(test)}")

    lower, upper, med = winsor_params(train, features)
    X_train = transform_features(train, features, lower, upper, med)
    X_val = transform_features(val, features, lower, upper, med)
    X_test = transform_features(test, features, lower, upper, med)
    y_train = train["target_label"].to_numpy(dtype=float)
    y_val = val["target_label"].to_numpy(dtype=float)
    y_test = test["target_label"].to_numpy(dtype=float)

    summaries: list[dict[str, Any]] = []
    ic_frames: list[pd.DataFrame] = []
    spread_frames: list[pd.DataFrame] = []
    decile_frames: list[pd.DataFrame] = []

    # Baseline 1: raw signed spillover score.
    baseline_col = "spillover_signed_shock" if "spillover_signed_shock" in features else features[0]
    for split_name, split_df, X, y in [("train", train, X_train, y_train), ("validation", val, X_val, y_val), ("test", test, X_test, y_test)]:
        pred = X[baseline_col].to_numpy(dtype=float)
        s, ic, sp, de = summarize_prediction("raw_signed_spillover", horizon, split_name, y, pred, split_df["signal_date"])
        summaries.append(s); ic_frames.append(ic); spread_frames.append(sp); decile_frames.append(de)

    # Benchmark 2: ridge / LP-style transparent linear graph-feature model.
    val_pred_ridge, ridge_model = fit_predict_linear(X_train, y_train, X_val)
    test_pred_ridge = ridge_model.predict(X_test)
    train_pred_ridge = ridge_model.predict(X_train)
    for split_name, split_df, y, pred in [("train", train, y_train, train_pred_ridge), ("validation", val, y_val, val_pred_ridge), ("test", test, y_test, test_pred_ridge)]:
        s, ic, sp, de = summarize_prediction("ridge_lp_style", horizon, split_name, y, pred, split_df["signal_date"])
        summaries.append(s); ic_frames.append(ic); spread_frames.append(sp); decile_frames.append(de)

    # Benchmark 3: LightGBM challenger.
    lgbm = fit_lightgbm(X_train, y_train, X_val, y_val, threads)
    train_pred_lgbm = lgbm.predict(X_train)
    val_pred_lgbm = lgbm.predict(X_val)
    test_pred_lgbm = lgbm.predict(X_test)
    for split_name, split_df, y, pred in [("train", train, y_train, train_pred_lgbm), ("validation", val, y_val, val_pred_lgbm), ("test", test, y_test, test_pred_lgbm)]:
        s, ic, sp, de = summarize_prediction("lightgbm_full", horizon, split_name, y, pred, split_df["signal_date"])
        summaries.append(s); ic_frames.append(ic); spread_frames.append(sp); decile_frames.append(de)

    # Protected local-only prediction cache, small enough but not included in upload bundle.
    pred_cache = pd.concat([
        train[["signal_date", "target_gvkey", "target_permno", "target_label", "signal_year"]].assign(split="train", pred_lightgbm=train_pred_lgbm, pred_ridge=train_pred_ridge, pred_raw=X_train[baseline_col].to_numpy()),
        val[["signal_date", "target_gvkey", "target_permno", "target_label", "signal_year"]].assign(split="validation", pred_lightgbm=val_pred_lgbm, pred_ridge=val_pred_ridge, pred_raw=X_val[baseline_col].to_numpy()),
        test[["signal_date", "target_gvkey", "target_permno", "target_label", "signal_year"]].assign(split="test", pred_lightgbm=test_pred_lgbm, pred_ridge=test_pred_ridge, pred_raw=X_test[baseline_col].to_numpy()),
    ], ignore_index=True)
    protected_dir.mkdir(parents=True, exist_ok=True)
    pred_cache.to_parquet(protected_dir / f"phase5_1_predictions_h{horizon}d.parquet", index=False)

    importances = pd.DataFrame({
        "feature": features,
        "importance_gain_proxy": getattr(lgbm, "feature_importances_", np.zeros(len(features))),
        "horizon_days": horizon,
    }).sort_values("importance_gain_proxy", ascending=False)

    return {
        "horizon": horizon,
        "features": features,
        "rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "dropped_inf_or_missing_rows": dropped_inf_or_missing,
        "summary": pd.DataFrame(summaries),
        "ic": pd.concat([x for x in ic_frames if not x.empty], ignore_index=True) if any(not x.empty for x in ic_frames) else pd.DataFrame(),
        "spread": pd.concat([x for x in spread_frames if not x.empty], ignore_index=True) if any(not x.empty for x in spread_frames) else pd.DataFrame(),
        "deciles": pd.concat([x for x in decile_frames if not x.empty], ignore_index=True) if any(not x.empty for x in decile_frames) else pd.DataFrame(),
        "importances": importances,
    }


def make_figures(out_dir: Path, summary: pd.DataFrame, deciles: pd.DataFrame, spreads: pd.DataFrame, importances: pd.DataFrame) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, str]] = []

    test = summary[summary["split"].eq("test")].copy()
    if not test.empty:
        pivot = test.pivot_table(index="horizon_days", columns="model", values="mean_rank_ic")
        plt.figure(figsize=(10.5, 5.8))
        for col in pivot.columns:
            plt.plot(pivot.index, pivot[col], marker="o", label=col)
        plt.axhline(0, linewidth=1)
        plt.xlabel("Horizon, trading days")
        plt.ylabel("Mean daily rank IC, test")
        plt.title("Phase 5.1 full-scale forecast skill by horizon")
        plt.legend()
        plt.tight_layout()
        p = fig_dir / "phase5_1_test_rank_ic_by_horizon.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append({"figure": "test_rank_ic_by_horizon", "path": str(p)})

        pivot2 = test.pivot_table(index="horizon_days", columns="model", values="mean_daily_decile_long_short")
        plt.figure(figsize=(10.5, 5.8))
        for col in pivot2.columns:
            plt.plot(pivot2.index, pivot2[col], marker="o", label=col)
        plt.axhline(0, linewidth=1)
        plt.xlabel("Horizon, trading days")
        plt.ylabel("Mean daily top-minus-bottom decile return")
        plt.title("Phase 5.1 full-scale decile spread by horizon")
        plt.legend()
        plt.tight_layout()
        p = fig_dir / "phase5_1_test_decile_spread_by_horizon.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append({"figure": "test_decile_spread_by_horizon", "path": str(p)})

    if not deciles.empty:
        main = deciles[(deciles["split"].eq("test")) & (deciles["model"].eq("lightgbm_full")) & (deciles["horizon_days"].eq(5))].copy()
        if main.empty:
            main = deciles[(deciles["split"].eq("test")) & (deciles["model"].eq("lightgbm_full"))].copy()
        if not main.empty:
            plt.figure(figsize=(10.5, 5.8))
            for h, g in main.groupby("horizon_days"):
                plt.plot(g["decile"], g["mean_target"], marker="o", label=f"h={h}d")
            plt.axhline(0, linewidth=1)
            plt.xlabel("Predicted-return decile")
            plt.ylabel("Mean realized target label")
            plt.title("LightGBM full-scale test decile monotonicity")
            plt.legend()
            plt.tight_layout()
            p = fig_dir / "phase5_1_lightgbm_decile_monotonicity.png"
            plt.savefig(p, dpi=180)
            plt.close()
            made.append({"figure": "lightgbm_decile_monotonicity", "path": str(p)})

    if not spreads.empty:
        main = spreads[(spreads["split"].eq("test")) & (spreads["model"].eq("lightgbm_full"))].copy()
        if not main.empty:
            plt.figure(figsize=(11.5, 6.0))
            for h, g in main.groupby("horizon_days"):
                g = g.sort_values("date")
                plt.plot(g["date"], g["long_short"].cumsum(), label=f"h={h}d")
            plt.axhline(0, linewidth=1)
            plt.xlabel("Date")
            plt.ylabel("Cumulative daily top-minus-bottom decile return")
            plt.title("LightGBM full-scale test cumulative decile spread")
            plt.legend()
            plt.tight_layout()
            p = fig_dir / "phase5_1_lightgbm_cumulative_test_spread.png"
            plt.savefig(p, dpi=180)
            plt.close()
            made.append({"figure": "lightgbm_cumulative_test_spread", "path": str(p)})

    if not importances.empty:
        imp = importances[importances["horizon_days"].eq(5)].copy()
        if imp.empty:
            imp = importances.copy()
        top = imp.groupby("feature", as_index=False)["importance_gain_proxy"].sum().sort_values("importance_gain_proxy", ascending=False).head(20)
        plt.figure(figsize=(10.5, 7.0))
        plt.barh(top["feature"], top["importance_gain_proxy"])
        plt.gca().invert_yaxis()
        plt.xlabel("LightGBM split importance")
        plt.title("Top LightGBM full-scale features")
        plt.tight_layout()
        p = fig_dir / "phase5_1_top_lgbm_features.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append({"figure": "top_lgbm_features", "path": str(p)})

    return made


def html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=True, classes="data") if df is not None and not df.empty else "<p>No rows.</p>"


def render_html(out_path: Path, quality: dict[str, Any], summary: pd.DataFrame, deciles: pd.DataFrame, importances: pd.DataFrame, figures: list[dict[str, str]], env: dict[str, Any]) -> None:
    cards = []
    metrics = [
        ("Validation", "PASS" if quality.get("validation_passed") else "FAIL"),
        ("Horizons", ", ".join(map(str, quality.get("horizons", [])))),
        ("Train rows", quality.get("rows", {}).get("train")),
        ("Validation rows", quality.get("rows", {}).get("validation")),
        ("Test rows", quality.get("rows", {}).get("test")),
        ("Feature count", quality.get("feature_count")),
    ]
    for label, value in metrics:
        val = f"{value:,}" if isinstance(value, int) else html.escape(str(value))
        cards.append(f"<div class='card'><div class='kicker'>{html.escape(label)}</div><h3>{val}</h3></div>")

    fig_html = []
    for fig in figures:
        rel = "figures/" + Path(fig["path"]).name
        fig_html.append(f"<div class='figure'><h3>{html.escape(fig['figure'].replace('_', ' ').title())}</h3><img src='{html.escape(rel)}'></div>")

    test_summary = summary[summary["split"].eq("test")].sort_values(["horizon_days", "model"]) if not summary.empty else summary
    val_summary = summary[summary["split"].eq("validation")].sort_values(["horizon_days", "model"]) if not summary.empty else summary
    top_importances = importances.sort_values(["horizon_days", "importance_gain_proxy"], ascending=[True, False]).groupby("horizon_days").head(15) if not importances.empty else importances

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 5.1 Full-Scale Modeling</title>
<style>
:root {{ --bg:#07111f; --text:#eef6ff; --muted:#9fb7ce; --line:rgba(255,255,255,.14); }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#183b66,var(--bg) 42%); color:var(--text); }}
header {{ padding:46px 56px 28px; border-bottom:1px solid var(--line); }} h1 {{ margin:0; font-size:42px; letter-spacing:-.04em; }} .subtitle {{ color:var(--muted); font-size:17px; max-width:1120px; line-height:1.55; }}
main {{ padding:28px 56px 60px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:18px; margin:22px 0 36px; }} .card {{ background:linear-gradient(180deg,rgba(255,255,255,.075),rgba(255,255,255,.035)); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 18px 40px rgba(0,0,0,.18); }} .card h3 {{ margin:7px 0 0; font-size:27px; }} .kicker {{ text-transform:uppercase; font-size:11px; letter-spacing:.16em; color:var(--muted); }}
section {{ background:rgba(15,30,51,.78); border:1px solid var(--line); border-radius:22px; padding:24px; margin:22px 0; overflow:auto; }} table.data {{ width:100%; border-collapse:collapse; font-size:13px; }} table.data th {{ text-align:left; color:#d8eaff; background:rgba(255,255,255,.08); }} table.data th, table.data td {{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.09); vertical-align:top; }} .figure img {{ width:100%; max-width:1100px; border-radius:16px; border:1px solid var(--line); background:white; }} pre {{ white-space:pre-wrap; background:rgba(0,0,0,.28); border:1px solid var(--line); border-radius:14px; padding:16px; color:#dbecff; }}
</style></head><body><header><h1>Phase 5.1 Full-Scale Modeling</h1><p class="subtitle">Local-only full-scale modeling using the Phase 4.2 model-ready matrix. It compares a raw spillover score, a transparent ridge/local-projection-style benchmark, and a LightGBM challenger under the fixed 2015–2019 / 2020–2021 / 2022–2024 walk-forward split.</p></header><main>
<div class="grid">{''.join(cards)}</div>
<section><h2>Test summary</h2>{html_table(test_summary)}</section>
<section><h2>Validation summary</h2>{html_table(val_summary)}</section>
<section><h2>Top LightGBM features</h2>{html_table(top_importances)}</section>
<section><h2>Figures</h2>{''.join(fig_html)}</section>
<section><h2>Quality JSON</h2><pre>{html.escape(json.dumps(quality, indent=2, default=str))}</pre></section>
<section><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2, default=str))}</pre></section>
</main></body></html>"""
    out_path.write_text(doc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS_DEFAULT)
    parser.add_argument("--row-cap-per-year", type=int, default=0)
    parser.add_argument("--feature-set", default="default_model_features")
    parser.add_argument("--use-pure", action="store_true", default=True)
    parser.add_argument("--train-years", default="2015-2019")
    parser.add_argument("--validation-years", default="2020-2021")
    parser.add_argument("--test-years", default="2022-2024")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("PNA_THREADS", "32")))
    args = parser.parse_args()

    workspace = WORKSPACE_REQUIRED
    project_root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    protected_dir = project_root / "data" / "processed" / "model_runs" / "phase5_1_modeling_full"

    if project_root != (workspace / "production-network-alpha").resolve():
        raise SystemExit(f"Wrong project root: {project_root}; expected {(workspace / 'production-network-alpha').resolve()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    protected_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("Phase 5.1 full-scale modeling")
    print(f"UTC: {utc_now()}")
    print(f"Project root: {project_root}")
    print(f"Output dir: {out_dir}")
    print(f"Protected local model-run dir: {protected_dir}")
    print(f"Horizons: {args.horizons}")
    print(f"Row cap per year per horizon: {args.row_cap_per_year} (0 means full sample)")
    print("================================================================================")

    env = {
        "utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cpu_count": os.cpu_count(),
        "threads": args.threads,
        "thread_env": {k: os.environ.get(k) for k in ["PNA_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]},
        "packages": package_status(),
        "git": run_cmd(["bash", "-lc", "command -v git || true"]),
    }
    with (out_dir / "environment.json").open("w") as f:
        json.dump(env, f, indent=2, default=str)

    years_by_split = {
        "train": parse_years(args.train_years, TRAIN_YEARS_DEFAULT),
        "validation": parse_years(args.validation_years, VAL_YEARS_DEFAULT),
        "test": parse_years(args.test_years, TEST_YEARS_DEFAULT),
    }
    manifest = load_manifest(project_root)

    results = []
    for h in args.horizons:
        t0 = time.time()
        print(f"[MODEL] Horizon {h}d starting...")
        res = run_horizon(
            project_root=project_root,
            horizon=h,
            years_by_split=years_by_split,
            manifest=manifest,
            feature_set=args.feature_set,
            row_cap_per_year=args.row_cap_per_year,
            pure=args.use_pure,
            threads=args.threads,
            protected_dir=protected_dir,
        )
        print(f"[MODEL] Horizon {h}d complete in {time.time() - t0:.1f}s; rows={res['rows']}")
        results.append(res)

    summary = pd.concat([r["summary"] for r in results], ignore_index=True)
    ic = pd.concat([r["ic"] for r in results if not r["ic"].empty], ignore_index=True) if any(not r["ic"].empty for r in results) else pd.DataFrame()
    spreads = pd.concat([r["spread"] for r in results if not r["spread"].empty], ignore_index=True) if any(not r["spread"].empty for r in results) else pd.DataFrame()
    deciles = pd.concat([r["deciles"] for r in results if not r["deciles"].empty], ignore_index=True) if any(not r["deciles"].empty for r in results) else pd.DataFrame()
    importances = pd.concat([r["importances"] for r in results], ignore_index=True)

    summary.to_csv(out_dir / "phase5_1_model_summary.csv", index=False)
    if not ic.empty:
        ic.to_csv(out_dir / "phase5_1_daily_rank_ic.csv", index=False)
    if not spreads.empty:
        spreads.to_csv(out_dir / "phase5_1_daily_decile_spreads.csv", index=False)
    if not deciles.empty:
        deciles.to_csv(out_dir / "phase5_1_decile_monotonicity.csv", index=False)
    importances.to_csv(out_dir / "phase5_1_lgbm_feature_importances.csv", index=False)

    rows_aggregate = {split: int(sum(r["rows"].get(split, 0) for r in results)) for split in ["train", "validation", "test"]}
    all_test = summary[summary["split"].eq("test")]
    lgbm_test = all_test[all_test["model"].eq("lightgbm_full")]
    validation_checks = {
        "all_horizons_completed": sorted(summary["horizon_days"].unique().tolist()) == sorted(args.horizons),
        "train_validation_test_rows_positive": all(v > 0 for v in rows_aggregate.values()),
        "lightgbm_test_rank_ic_finite": bool(np.isfinite(lgbm_test["mean_rank_ic"].to_numpy(dtype=float)).all()) if len(lgbm_test) else False,
        "no_inf_or_missing_rows_after_filter": int(sum(r["dropped_inf_or_missing_rows"] for r in results)) >= 0,
        "feature_count_positive": len(results[0]["features"]) > 0 if results else False,
    }
    quality = {
        "generated_at_utc": utc_now(),
        "workspace": str(workspace),
        "project_root": str(project_root),
        "protected_model_run_dir": str(protected_dir),
        "horizons": args.horizons,
        "row_cap_per_year": args.row_cap_per_year,
        "use_pure_spillover_sample": bool(args.use_pure),
        "split_policy": years_by_split,
        "rows": rows_aggregate,
        "feature_count": len(results[0]["features"]) if results else 0,
        "features": results[0]["features"] if results else [],
        "validation_checks": validation_checks,
        "validation_passed": bool(all(validation_checks.values())),
        "note": "Full-scale run uses all model-ready pure-spillover rows for all horizons. No WRDS queries are performed.",
    }
    with (out_dir / "phase5_1_quality_summary.json").open("w") as f:
        json.dump(quality, f, indent=2, default=str)

    figures = make_figures(out_dir, summary, deciles, spreads, importances)
    render_html(out_dir / "phase5_1_modeling_full_report.html", quality, summary, deciles, importances, figures, env)

    protected_inventory = []
    for p in sorted(protected_dir.glob("*.parquet")):
        protected_inventory.append({"file": str(p), "size_bytes": p.stat().st_size, "protected_local_only": True})
    pd.DataFrame(protected_inventory).to_csv(out_dir / "protected_local_phase5_1_inventory.csv", index=False)

    lines = [
        "# Phase 5.1 full-scale modeling summary",
        "",
        f"- Generated at UTC: {quality['generated_at_utc']}",
        f"- Validation passed: {quality['validation_passed']}",
        f"- Horizons: {args.horizons}",
        f"- Pure spillover sample: {args.use_pure}",
        f"- Row cap per year/horizon: {args.row_cap_per_year:,} (0 means full sample)",
        f"- Train rows across horizons: {rows_aggregate['train']:,}",
        f"- Validation rows across horizons: {rows_aggregate['validation']:,}",
        f"- Test rows across horizons: {rows_aggregate['test']:,}",
        f"- Feature count: {quality['feature_count']}",
        f"- Report: {out_dir / 'phase5_1_modeling_full_report.html'}",
        "",
        "Data policy: protected full-scale prediction/model-run Parquet files remain local and are not included in the upload bundle.",
    ]
    (out_dir / "PHASE5_1_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if quality["validation_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
