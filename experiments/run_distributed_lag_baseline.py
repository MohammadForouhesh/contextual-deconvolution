#!/usr/bin/env python3
"""
Run Distributed Lag Baseline (XGBoost with 14 lagged promotional features)
on M5 (500 SKUs) and Favorita (97 items).

Usage:
    source /opt/homebrew/Caskroom/miniconda/base/bin/activate goi_clean
    KMP_DUPLICATE_LIB_OK=TRUE python run_distributed_lag_baseline.py

Features per day t:
- promo_lag0 ... promo_lag14 (15 features)
- snap_lag0 ... snap_lag14 (15 features)
- demand_lag1, demand_lag7, demand_lag14 (3 features)
Total: 33 features + time features

Train on first T-28 days, test on last 28 days.
"""

import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List

sys.path.insert(0, _os.path.join(_ROOT, "code"))

from contextual_deconvolution.loaders import load_m5_data, load_favorita_data
from contextual_deconvolution.metrics import wmape, variance_ratio, rmse
from contextual_deconvolution.experiments import CONFIG_DEFAULT, evaluate_sku, run_evaluation


def build_distributed_lag_features(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    n_lags: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix with distributed lag features.

    For each day t >= n_lags:
    - promo_lag0, ..., promo_lag_n (f_promo[t], ..., f_promo[t-n_lags])
    - snap_lag0, ..., snap_lag_n (f_snap[t], ..., f_snap[t-n_lags])
    - demand_lag1, demand_lag7, demand_lag14
    - calendar features: day_of_week, month, time_index
    """
    T = len(b)
    features = []
    for t in range(n_lags, T):
        feat = []
        # Promotional lags
        for lag in range(n_lags + 1):
            feat.append(f_promo[t - lag])
        # SNAP lags
        for lag in range(n_lags + 1):
            feat.append(f_snap[t - lag])
        # Demand lags
        feat.append(b[t - 1])
        feat.append(b[t - 7])
        feat.append(b[t - 14])
        # Calendar features
        feat.append(t % 7)           # day of week
        feat.append((t // 30) % 12)  # month proxy
        feat.append(t)               # time trend
        features.append(feat)
    return np.array(features), np.arange(n_lags, T)


def run_xgboost_distributed_lag(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
    n_lags: int = 14,
    n_estimators: int = 200,
    max_depth: int = 5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """XGBoost with distributed lag features."""
    try:
        import xgboost as xgb
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor as xgb_fallback
        xgb = type('xgb', (), {'XGBRegressor': xgb_fallback})()

    X, idx = build_distributed_lag_features(b, f_promo, f_snap, n_lags)
    y = b[idx]
    split = len(y) - forecast_horizon
    if split <= 50:
        return None, None, None, None

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, random_state=42, verbosity=0,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fitted_train = model.predict(X_train)
    return y_pred, y_test, fitted_train, model


def evaluate_distributed_lag_sku(sku_data: Dict, config: Dict) -> Dict:
    """Evaluate distributed lag baseline on a single SKU."""
    b = sku_data["demand"]
    f_promo = sku_data["f_promo"]
    f_snap = sku_data["f_snap"]
    T = len(b)
    horizon = config["forecast_horizon"]
    split = T - horizon
    b_test = b[split:]

    t0 = time.time()
    pred, actual, fitted_train, model = run_xgboost_distributed_lag(
        b, f_promo, f_snap, horizon
    )
    elapsed = time.time() - t0

    if pred is None:
        return {
            "sku_id": sku_data.get("sku_id", 0),
            "category": sku_data.get("category", "unknown"),
            "wmape": np.nan,
            "variance_ratio": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "runtime": elapsed,
        }

    return {
        "sku_id": sku_data.get("sku_id", 0),
        "category": sku_data.get("category", "unknown"),
        "wmape": wmape(actual, pred),
        "variance_ratio": variance_ratio(pred, actual),
        "rmse": rmse(actual, pred),
        "mae": np.mean(np.abs(actual - pred)),
        "runtime": elapsed,
    }


def run_dataset(dataset: List[Dict], dataset_name: str, out_dir: str) -> pd.DataFrame:
    """Run distributed lag baseline on a dataset."""
    print(f"\n{'='*70}")
    print(f"Running Distributed Lag Baseline on {dataset_name} ({len(dataset)} items)")
    print(f"{'='*70}")

    config = CONFIG_DEFAULT.copy()
    config["forecast_horizon"] = 28

    results = []
    for i, sku_data in enumerate(dataset):
        try:
            r = evaluate_distributed_lag_sku(sku_data, config)
            results.append(r)
            if (i + 1) % 50 == 0 or i == len(dataset) - 1:
                print(f"  Processed {i+1}/{len(dataset)} items")
        except Exception as e:
            warnings.warn(f"Failed for SKU {sku_data.get('sku_id', i)}: {e}")
            results.append({
                "sku_id": sku_data.get("sku_id", i),
                "category": sku_data.get("category", "unknown"),
                "wmape": np.nan, "variance_ratio": np.nan,
                "rmse": np.nan, "mae": np.nan, "runtime": np.nan,
            })

    df = pd.DataFrame(results)
    df = df.dropna()

    print(f"\n--- {dataset_name} Distributed Lag Results ---")
    print(f"  n = {len(df)}")
    print(f"  wMAPE:     {df['wmape'].mean():.4f} ± {df['wmape'].std():.4f}")
    print(f"  RMSE:      {df['rmse'].mean():.4f} ± {df['rmse'].std():.4f}")
    print(f"  VarRatio:  {df['variance_ratio'].mean():.4f} ± {df['variance_ratio'].std():.4f}")
    print(f"  Median VR: {df['variance_ratio'].median():.4f}")
    print(f"  Runtime:   {df['runtime'].mean():.3f}s ± {df['runtime'].std():.3f}s per SKU")

    out_path = os.path.join(out_dir, f"distributed_lag_{dataset_name.lower().replace(' ', '_')}.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return df


def run_cd_comparison(dataset: List[Dict], dataset_name: str, out_dir: str) -> pd.DataFrame:
    """Run CD method on the same dataset for comparison."""
    print(f"\n{'='*70}")
    print(f"Running CD baseline on {dataset_name} ({len(dataset)} items)")
    print(f"{'='*70}")

    config = CONFIG_DEFAULT.copy()
    config["run_baselines"] = False
    config["run_ablations"] = False
    config["use_ista"] = True

    results, summary = run_evaluation(dataset, config=config, n_jobs=-1)

    if len(summary) == 0:
        print("No CD results available")
        return pd.DataFrame()

    cd_summary = summary[summary["method"] == "CD"]
    if len(cd_summary) == 0:
        print("No CD results in summary")
        return pd.DataFrame()

    print(f"\n--- {dataset_name} CD Results ---")
    print(f"  n = {len(cd_summary)}")
    print(f"  wMAPE:     {cd_summary['wmape'].mean():.4f} ± {cd_summary['wmape'].std():.4f}")
    print(f"  RMSE:      {cd_summary['rmse'].mean():.4f} ± {cd_summary['rmse'].std():.4f}")
    print(f"  VarRatio:  {cd_summary['variance_ratio'].mean():.4f} ± {cd_summary['variance_ratio'].std():.4f}")
    print(f"  Median VR: {cd_summary['variance_ratio'].median():.4f}")

    out_path = os.path.join(out_dir, f"cd_{dataset_name.lower().replace(' ', '_')}.csv")
    cd_summary.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return cd_summary


if __name__ == "__main__":
    m5_dir = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")
    favorita_dir = _os.path.join(_ROOT, "data/favorita-grocery-sales-forecasting")
    out_dir = _os.path.join(_ROOT, "paper/results")
    os.makedirs(out_dir, exist_ok=True)

    total_start = time.time()

    # --- M5: 500 SKUs ---
    print("\nLoading M5 data (500 SKUs)...")
    m5_data = load_m5_data(m5_dir, n_skus=500, random_state=42)
    print(f"Loaded {len(m5_data)} SKUs")

    m5_dl = run_dataset(m5_data, "M5_500", out_dir)
    m5_cd = run_cd_comparison(m5_data, "M5_500", out_dir)

    # --- Favorita: 97 items ---
    print("\nLoading Favorita data (97 items)...")
    favorita_data = load_favorita_data(favorita_dir, n_items=97, random_state=42)
    print(f"Loaded {len(favorita_data)} items")

    if len(favorita_data) > 0:
        fav_dl = run_dataset(favorita_data, "Favorita_97", out_dir)
        fav_cd = run_cd_comparison(favorita_data, "Favorita_97", out_dir)
    else:
        print("WARNING: No Favorita data loaded, skipping")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"TOTAL RUNTIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")
