#!/usr/bin/env python3
"""
Run M5 evaluation with pooled kernels + optional Tuned XGBoost baseline.

Usage:
    # Full M5 with global pooling, skip prices for speed:
    python scripts/run_m5_full_suite.py --pooling global --skip_prices

    # Full M5 with category pooling + Tuned XGBoost baseline:
    python scripts/run_m5_full_suite.py --pooling category --run_baselines --skip_prices

    # Per-SKU (no pooling) with baselines on first 1000 SKUs:
    python scripts/run_m5_full_suite.py --pooling per_sku --n_skus 1000 --run_baselines --skip_prices
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import (
    CONFIG_DEFAULT,
    estimate_all_kernels,
    pool_kernels,
    evaluate_sku_pooled,
)
from contextual_deconvolution.baselines import run_tuned_xgboost
from contextual_deconvolution.metrics import wmape, variance_ratio, inventory_cost, rmse, mae, spectral_flatness, max_drawdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/m5-data/m5-forecasting-accuracy")
    parser.add_argument("--n_skus", type=int, default=None, help="Number of SKUs (default: all)")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--pooling", type=str, default="global", choices=["per_sku", "category", "global"])
    parser.add_argument("--skip_prices", action="store_true", help="Skip sell price computation for fast loading")
    parser.add_argument("--run_baselines", action="store_true", help="Include Tuned XGBoost baseline")
    parser.add_argument("--kernel_subset", type=int, default=500, help="SKUs to estimate kernel from (for global/category)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = CONFIG_DEFAULT.copy()
    config["run_ablations"] = False
    config["run_baselines"] = False  # We handle baselines manually

    print("=" * 70)
    print(f"M5 EVALUATION | Pooling={args.pooling} | Baselines={args.run_baselines}")
    print("=" * 70)

    print(f"\nLoading M5 data (skip_prices={args.skip_prices})...")
    dataset = load_m5_data(args.data_dir, n_skus=args.n_skus, random_state=42, skip_prices=args.skip_prices)
    print(f"Loaded {len(dataset)} SKUs")
    
    cats = pd.Series([d["category"] for d in dataset]).value_counts()
    print(f"\nCategories:\n{cats.to_string()}")

    # --- Kernel estimation ---
    print(f"\n--- Stage 1: Estimating per-SKU kernels from {min(args.kernel_subset, len(dataset))} SKUs ---")
    t0 = time.time()
    subset = dataset[:min(args.kernel_subset, len(dataset))]
    kernels = estimate_all_kernels(subset, config)
    print(f"\n--- Stage 2: Pooling {len(kernels)} per-SKU kernels ({args.pooling}) ---")
    pooled = pool_kernels(kernels, strategy=args.pooling)
    t1 = time.time()
    print(f"Pooled to {len(set(k['category'] for k in pooled.values()))} categories + 1 global in {t1 - t0:.1f}s")
    print(f"\n--- Estimating kernels from {min(args.kernel_subset, len(dataset))} SKUs ---")
    t0 = time.time()
    subset = dataset[:min(args.kernel_subset, len(dataset))]
    kernels = estimate_all_kernels(subset, config)
    pooled = pool_kernels(kernels, strategy=args.pooling)
    t1 = time.time()
    print(f"Done in {t1 - t0:.1f}s")

    # --- Build kernel lookup based on pooling strategy ---
    print(f"\n--- Building kernel lookup ({args.pooling}) ---")
    
    if args.pooling == "global":
        # All SKUs share the same global kernel
        global_kernel = list(pooled.values())[0]
        def get_kernel(sku):
            return global_kernel
    elif args.pooling == "category":
        # Build category -> kernel mapping
        cat_to_kernel = {}
        for k in pooled.values():
            cat_to_kernel[k["category"]] = k
        def get_kernel(sku):
            cat = sku.get("category", "unknown")
            return cat_to_kernel.get(cat, list(pooled.values())[0])
    else:  # per_sku
        # Per-SKU kernels only exist for the subset; fall back to category median for others
        def get_kernel(sku):
            sku_id = sku.get("sku_id", 0)
            if sku_id in pooled:
                return pooled[sku_id]
            # Fallback: use category median or global
            cat = sku.get("category", "unknown")
            for k in pooled.values():
                if k["category"] == cat:
                    return k
            return list(pooled.values())[0]

    # --- Evaluation ---
    print(f"\n--- Evaluating {len(dataset)} SKUs ---")
    t2 = time.time()
    all_rows = []
    
    for i, sku in enumerate(dataset):
        sku_id = sku.get("sku_id", 0)
        k = get_kernel(sku)
        
        # CD with pooled kernel
        r = evaluate_sku_pooled(sku, k["phi_promo"], k["phi_snap"], config)
        for method, metrics in r["forecasts"].items():
            all_rows.append({
                "sku_id": sku_id,
                "category": sku.get("category", "unknown"),
                "method": method,
                **{k: v for k, v in metrics.items() if k not in ("pred", "actual")},
            })
        
        # Tuned XGBoost baseline
        if args.run_baselines:
            b = sku["demand"]
            f_promo = sku["f_promo"]
            f_snap = sku["f_snap"]
            horizon = config["forecast_horizon"]
            
            xgb_pred, xgb_actual, xgb_fitted, _ = run_tuned_xgboost(b, f_promo, f_snap, horizon)
            if xgb_pred is not None:
                inv_cost, ss = inventory_cost(
                    xgb_actual, xgb_pred, xgb_fitted,
                    lead_time=config["lead_time"],
                    h=config["holding_cost"],
                    p=config["stockout_cost"],
                    service_level=config["service_level"],
                )
                all_rows.append({
                    "sku_id": sku_id,
                    "category": sku.get("category", "unknown"),
                    "method": "Tuned XGBoost",
                    "wmape": wmape(xgb_actual, xgb_pred),
                    "variance_ratio": variance_ratio(xgb_pred, xgb_actual),
                    "rmse": rmse(xgb_actual, xgb_pred),
                    "mae": mae(xgb_actual, xgb_pred),
                    "inventory_cost": inv_cost,
                    "safety_stock": ss,
                    "spectral_flatness": spectral_flatness(xgb_pred),
                    "max_drawdown": max_drawdown(xgb_pred - xgb_actual),
                })
        
        if (i + 1) % 50 == 0 or i == len(dataset) - 1:
            elapsed = time.time() - t2
            per_sku = elapsed / (i + 1)
            remaining = per_sku * (len(dataset) - i - 1)
            print(f"  {i+1}/{len(dataset)} | elapsed: {elapsed:.0f}s | per_sku: {per_sku:.2f}s | remaining: {remaining:.0f}s")

    t3 = time.time()
    print(f"\nTotal evaluation: {t3 - t2:.1f}s ({(t3-t2)/len(dataset):.3f}s per SKU)")

    # --- Aggregate results ---
    summary = pd.DataFrame(all_rows)
    
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    agg = summary.groupby("method").agg({
        "wmape": ["mean", "std", "median"],
        "variance_ratio": ["mean", "std", "median"],
        "rmse": ["mean", "std", "median"],
        "safety_stock": ["mean", "std", "median"],
    }).reset_index()
    agg.columns = [' '.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    print(agg.to_string(index=False))
    
    # Per-category breakdown for CD
    print("\n--- CD by Category ---")
    cd = summary[summary["method"] == "CD"]
    if len(cd) > 0:
        cat_agg = cd.groupby("category").agg({
            "wmape": ["mean", "std", "median"],
            "variance_ratio": ["mean", "std", "median"],
            "safety_stock": ["mean", "std", "median"],
        }).reset_index()
        cat_agg.columns = [' '.join(col).strip() if col[1] else col[0] for col in cat_agg.columns.values]
        print(cat_agg.to_string(index=False))

    # Save
    suffix = f"m5_{args.pooling}"
    if args.run_baselines:
        suffix += "_with_xgb"
    if args.skip_prices:
        suffix += "_fast"
    
    out_json = os.path.join(args.output_dir, f"{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(agg.to_dict(), f, indent=2, default=str)
    summary.to_csv(os.path.join(args.output_dir, f"{suffix}.csv"), index=False)
    print(f"\nSaved to {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
