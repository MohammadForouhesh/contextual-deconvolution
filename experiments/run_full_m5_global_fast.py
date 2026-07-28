#!/usr/bin/env python3
"""
Run full M5 evaluation with global pooled kernels - fast version.

Usage:
    python scripts/run_full_m5_global_fast.py --data_dir ../data/m5-data/m5-forecasting-accuracy --output_dir results
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

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


def _eval_worker(args):
    """Worker function for parallel evaluation."""
    sku_data, phi_promo, phi_snap, config = args
    try:
        return evaluate_sku_pooled(sku_data, phi_promo, phi_snap, config)
    except Exception as e:
        print(f"Error evaluating SKU {sku_data.get('sku_id', 0)}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/m5-data/m5-forecasting-accuracy")
    parser.add_argument("--n_skus", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--kernel_subset", type=int, default=1000, help="Number of SKUs to estimate kernels from")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = CONFIG_DEFAULT.copy()
    config["run_baselines"] = False
    config["baseline_methods"] = []
    config["run_ablations"] = False

    print(f"Loading M5 data...")
    dataset = load_m5_data(args.data_dir, n_skus=args.n_skus, random_state=42)
    print(f"Loaded {len(dataset)} SKUs")

    # Step 1: Estimate kernels on a subset
    print(f"Estimating kernels on {args.kernel_subset} SKUs...")
    t0 = time.time()
    subset = dataset[:args.kernel_subset] if len(dataset) > args.kernel_subset else dataset
    kernels = estimate_all_kernels(subset, config)
    pooled = pool_kernels(kernels, strategy="global")
    global_kernel = list(pooled.values())[0]
    t1 = time.time()
    print(f"Kernel estimation: {t1 - t0:.1f}s")

    # Step 2: Evaluate all SKUs with the global kernel
    print(f"Evaluating {len(dataset)} SKUs with global kernel...")
    t2 = time.time()
    
    worker_args = [
        (sku, global_kernel["phi_promo"], global_kernel["phi_snap"], config)
        for sku in dataset
    ]
    
    with Pool(args.n_workers) as pool:
        results = pool.map(_eval_worker, worker_args)
    
    t3 = time.time()
    print(f"Evaluation: {t3 - t2:.1f}s ({(t3-t2)/len(dataset):.3f}s per SKU)")
    print(f"Total: {t3 - t0:.1f}s")

    # Build summary
    all_rows = []
    for result in results:
        if result is None:
            continue
        for method, metrics in result["forecasts"].items():
            all_rows.append({
                "sku_id": result["sku_id"],
                "category": result.get("category", "unknown"),
                "method": method,
                **{k: v for k, v in metrics.items() if k not in ("pred", "actual")},
            })

    summary = pd.DataFrame(all_rows)
    
    print("\nAggregate Results:")
    agg = summary.groupby("method").agg({
        "wmape": ["mean", "std", "median"],
        "variance_ratio": ["mean", "std", "median"],
        "rmse": ["mean", "std", "median"],
        "safety_stock": ["mean", "std", "median"],
    }).reset_index()
    agg.columns = [' '.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    print(agg.to_string(index=False))

    # Per-category
    print("\nCD Results by Category:")
    cd_summary = summary[summary["method"] == "CD"]
    cat_agg = cd_summary.groupby("category").agg({
        "wmape": ["mean", "std", "median"],
        "variance_ratio": ["mean", "std", "median"],
        "safety_stock": ["mean", "std", "median"],
    }).reset_index()
    cat_agg.columns = [' '.join(col).strip() if col[1] else col[0] for col in cat_agg.columns.values]
    print(cat_agg.to_string(index=False))

    # Save
    out_json = os.path.join(args.output_dir, "m5_full_global_pooled.json")
    with open(out_json, "w") as f:
        json.dump(agg.to_dict(), f, indent=2, default=str)
    summary.to_csv(os.path.join(args.output_dir, "m5_full_global_pooled.csv"), index=False)
    print(f"\nSaved to {out_json}")


if __name__ == "__main__":
    main()
