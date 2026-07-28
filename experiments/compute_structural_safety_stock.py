#!/usr/bin/env python3
"""
Compute safety stock for structural baselines on both M5 (200 SKUs) and Favorita (97 items).

This script re-runs the four structural baselines from run_structural_baselines.py,
computes safety stock via inventory_cost(), and saves updated CSVs with a safety_stock column.

Usage:
    python3 code/compute_structural_safety_stock.py
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

warnings.filterwarnings('ignore')

sys.path.insert(0, _os.path.join(_ROOT, "code"))

from contextual_deconvolution.loaders import load_m5_data, load_favorita_data
from contextual_deconvolution.metrics import wmape, variance_ratio, rmse, inventory_cost

# Import baseline functions from the existing script
from run_structural_baselines import (
    run_arimax,
    run_ets_covariates,
    run_prophet_or_prophet_like,
    run_distributed_lag_ridge,
)


def evaluate_method_with_safety_stock(
    sku_data: Dict,
    method_name: str,
    forecast_horizon: int = 28,
) -> Dict:
    """
    Run a single structural baseline on a single SKU and return metrics + safety stock.
    """
    b = sku_data["demand"]
    f_promo = sku_data["f_promo"]
    f_snap = sku_data["f_snap"]
    T = len(b)
    if T < forecast_horizon + 50:
        return {
            "sku_id": sku_data.get("sku_id", 0),
            "category": sku_data.get("category", "unknown"),
            "wmape": np.nan,
            "variance_ratio": np.nan,
            "rmse": np.nan,
            "safety_stock": np.nan,
            "method": method_name,
            "runtime": np.nan,
        }

    method_map = {
        "ARIMAX": run_arimax,
        "ETS_covariates": run_ets_covariates,
        "Prophet-like": run_prophet_or_prophet_like,
        "DistributedLag_Ridge": run_distributed_lag_ridge,
    }
    func = method_map[method_name]

    t0 = time.time()
    try:
        pred, actual, fitted_train, model = func(b, f_promo, f_snap, forecast_horizon)
    except Exception as e:
        warnings.warn(f"{method_name} failed for SKU {sku_data.get('sku_id', 0)}: {e}")
        pred = None
    elapsed = time.time() - t0

    if pred is None or actual is None or len(pred) != len(actual):
        return {
            "sku_id": sku_data.get("sku_id", 0),
            "category": sku_data.get("category", "unknown"),
            "wmape": np.nan,
            "variance_ratio": np.nan,
            "rmse": np.nan,
            "safety_stock": np.nan,
            "method": method_name,
            "runtime": elapsed,
        }

    # Compute safety stock using inventory_cost
    try:
        avg_daily_cost, safety_stock = inventory_cost(
            actual=actual,
            forecast=pred,
            fitted_train=fitted_train,
        )
    except Exception as e:
        warnings.warn(f"inventory_cost failed for SKU {sku_data.get('sku_id', 0)}: {e}")
        safety_stock = np.nan

    return {
        "sku_id": sku_data.get("sku_id", 0),
        "category": sku_data.get("category", "unknown"),
        "wmape": wmape(actual, pred),
        "variance_ratio": variance_ratio(pred, actual),
        "rmse": rmse(actual, pred),
        "safety_stock": safety_stock,
        "method": method_name,
        "runtime": elapsed,
    }


def run_dataset_baselines_with_safety_stock(dataset: List[Dict], dataset_name: str) -> pd.DataFrame:
    """Run all four structural baselines on a dataset, computing safety stock."""
    print(f"\n{'='*70}")
    print(f"Running structural baselines with safety stock on {dataset_name} ({len(dataset)} items)")
    print(f"{'='*70}")

    methods = ["ARIMAX", "ETS_covariates", "Prophet-like", "DistributedLag_Ridge"]
    all_results = []

    for method in methods:
        print(f"\n  --- {method} ---")
        method_results = []
        for i, sku_data in enumerate(dataset):
            try:
                r = evaluate_method_with_safety_stock(sku_data, method, forecast_horizon=28)
                method_results.append(r)
                if (i + 1) % 50 == 0 or i == len(dataset) - 1:
                    print(f"    Processed {i+1}/{len(dataset)} items")
            except Exception as e:
                warnings.warn(f"Unhandled error for {method} SKU {sku_data.get('sku_id', i)}: {e}")
                method_results.append({
                    "sku_id": sku_data.get("sku_id", i),
                    "category": sku_data.get("category", "unknown"),
                    "wmape": np.nan,
                    "variance_ratio": np.nan,
                    "rmse": np.nan,
                    "safety_stock": np.nan,
                    "method": method,
                    "runtime": np.nan,
                })
        all_results.extend(method_results)

    df = pd.DataFrame(all_results)
    return df


def print_summary(df: pd.DataFrame, dataset_name: str):
    """Print mean ± std summary table for each method including safety stock."""
    print(f"\n{'='*70}")
    print(f"Summary: {dataset_name}")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'n':>6} {'wMAPE':>12} {'RMSE':>12} {'VarRatio':>12} {'SafetyStock':>12} {'Runtime':>10}")
    print("-" * 90)
    for method in df["method"].unique():
        sub = df[df["method"] == method].dropna(subset=["wmape"])
        if len(sub) == 0:
            print(f"{method:<25} {'--':>6} {'--':>12} {'--':>12} {'--':>12} {'--':>12} {'--':>10}")
            continue
        wmape_m = sub["wmape"].mean()
        wmape_s = sub["wmape"].std()
        rmse_m = sub["rmse"].mean()
        rmse_s = sub["rmse"].std()
        vr_m = sub["variance_ratio"].mean()
        vr_s = sub["variance_ratio"].std()
        ss_m = sub["safety_stock"].mean()
        ss_s = sub["safety_stock"].std()
        rt_m = sub["runtime"].mean()
        print(f"{method:<25} {len(sub):>6} {wmape_m:.4f}±{wmape_s:.4f} {rmse_m:.2f}±{rmse_s:.2f} {vr_m:.4f}±{vr_s:.4f} {ss_m:.2f}±{ss_s:.2f} {rt_m:.3f}s")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    m5_dir = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")
    favorita_dir = _os.path.join(_ROOT, "data/favorita-grocery-sales-forecasting")
    out_dir = _os.path.join(_ROOT, "code/results")
    os.makedirs(out_dir, exist_ok=True)

    total_start = time.time()

    # --- M5: 200 SKUs ---
    print("\nLoading M5 data (200 SKUs)...")
    m5_data = load_m5_data(m5_dir, n_skus=200, random_state=42)
    print(f"Loaded {len(m5_data)} SKUs")

    m5_results = run_dataset_baselines_with_safety_stock(m5_data, "M5_200")
    m5_out = os.path.join(out_dir, "m5_200_structural_baselines.csv")
    m5_results.to_csv(m5_out, index=False)
    print(f"\nSaved M5 results to {m5_out}")
    print_summary(m5_results, "M5_200")

    # --- Favorita: 97 items ---
    print("\nLoading Favorita data (97 items)...")
    favorita_data = load_favorita_data(favorita_dir, n_items=97, random_state=42)
    print(f"Loaded {len(favorita_data)} items")

    if len(favorita_data) > 0:
        fav_results = run_dataset_baselines_with_safety_stock(favorita_data, "Favorita_97")
        fav_out = os.path.join(out_dir, "favorita_structural_baselines.csv")
        fav_results.to_csv(fav_out, index=False)
        print(f"\nSaved Favorita results to {fav_out}")
        print_summary(fav_results, "Favorita_97")
    else:
        print("WARNING: No Favorita data loaded, skipping")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"TOTAL RUNTIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")
