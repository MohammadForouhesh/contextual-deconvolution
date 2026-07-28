#!/usr/bin/env python3
"""
Run full M5 evaluation with pooled kernels.

Usage:
    python scripts/run_full_m5_pooled.py --data_dir ../data/m5-data/m5-forecasting-accuracy --output_dir results
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
from contextual_deconvolution.metrics import wmape, variance_ratio, inventory_cost, rmse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/m5-data/m5-forecasting-accuracy")
    parser.add_argument("--n_skus", type=int, default=None, help="Number of SKUs to evaluate (default: all)")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--pooling", type=str, default="category", choices=["per_sku", "category", "global"])
    parser.add_argument("--kernel_subset", type=int, default=500,
                        help="SKUs to estimate kernels from before pooling (paper uses 500). "
                             "Ignored when --pooling per_sku, which needs a kernel per SKU.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = CONFIG_DEFAULT.copy()
    config["baseline_methods"] = []  # Skip baselines for speed
    config["run_baselines"] = False
    config["run_ablations"] = False

    print("=" * 70)
    print(f"FULL M5 EVALUATION WITH {args.pooling.upper()} POOLED KERNELS")
    print("=" * 70)

    print(f"\nLoading M5 data from {args.data_dir}...")
    if args.n_skus is None:
        print("Loading ALL SKUs...")
    dataset = load_m5_data(args.data_dir, n_skus=args.n_skus, random_state=42)
    print(f"Loaded {len(dataset)} SKUs")
    
    # Show category distribution
    cats = pd.Series([d["category"] for d in dataset]).value_counts()
    print(f"\nCategory distribution:")
    print(cats.to_string())

    # --- Step 1: Estimate kernels on a representative subset (paper's method) ---
    if args.pooling == "per_sku":
        est_data = dataset  # per-SKU pooling needs a kernel for every SKU
        print(f"\n--- Step 1: Estimating per-SKU kernels (all {len(dataset)} SKUs) ---")
    else:
        rng = np.random.RandomState(42)
        n_sub = min(args.kernel_subset, len(dataset))
        sub_idx = rng.choice(len(dataset), size=n_sub, replace=False)
        est_data = [dataset[i] for i in sub_idx]
        print(f"\n--- Step 1: Estimating kernels on a {n_sub}-SKU subset ---")
    t0 = time.time()
    kernels = estimate_all_kernels(est_data, config)
    t1 = time.time()
    print(f"Kernel estimation time: {t1 - t0:.1f}s "
          f"({(t1 - t0) / max(len(est_data), 1):.2f}s per SKU over {len(est_data)} SKUs)")

    # --- Step 2: Pool to category/global kernels, applied across the full catalog ---
    print(f"\n--- Step 2: Pooling kernels ({args.pooling}) ---")
    n_lags = config["n_lags"]
    phi_snap_shared = np.zeros(n_lags + 1)
    phi_snap_shared[0] = 1.0  # SNAP kernel is an impulse (matches estimate_all_kernels)

    global_kernel = np.median(np.stack([k["phi_promo"] for k in kernels.values()]), axis=0)
    cat_lists = {}
    for k in kernels.values():
        cat_lists.setdefault(k["category"], []).append(k["phi_promo"])
    category_kernel = {c: np.median(np.stack(v), axis=0) for c, v in cat_lists.items()}
    if args.pooling == "category":
        print(f"Number of categories: {len(category_kernel)} "
              f"({', '.join(sorted(category_kernel))})")
    elif args.pooling == "global":
        print("Using single global kernel for all SKUs")

    def _kernel_for(sku_data):
        if args.pooling == "per_sku":
            k = kernels[sku_data.get("sku_id", 0)]
            return k["phi_promo"], k["phi_snap"]
        if args.pooling == "global":
            return global_kernel, phi_snap_shared
        cat = sku_data.get("category", "unknown")
        return category_kernel.get(cat, global_kernel), phi_snap_shared

    # --- Step 3: Evaluate the full catalog in parallel ---
    print(f"\n--- Step 3: Evaluating all {len(dataset)} SKUs ---")
    t2 = time.time()

    def _eval_one(sku_data):
        phi_promo, phi_snap = _kernel_for(sku_data)
        return evaluate_sku_pooled(sku_data, phi_promo, phi_snap, config)

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(_eval_one)(sku_data) for sku_data in dataset
        )
    except Exception:
        import traceback
        print("\n" + "!" * 70)
        print("PARALLEL EVALUATION FAILED -- falling back to SERIAL (this will be slow).")
        print("Investigate the traceback below instead of accepting the serial path:")
        traceback.print_exc()
        print("!" * 70 + "\n")
        results = [_eval_one(sku_data) for sku_data in dataset]
    
    all_rows = []
    for result in results:
        for method, metrics in result["forecasts"].items():
            all_rows.append({
                "sku_id": result["sku_id"],
                "category": result.get("category", "unknown"),
                "method": method,
                **{k: v for k, v in metrics.items() if k not in ("pred", "actual")},
            })
    
    t3 = time.time()
    print(f"Evaluation time: {t3 - t2:.1f}s ({(t3-t2)/len(dataset):.2f}s per SKU)")
    print(f"Total time: {t3 - t0:.1f}s")

    summary = pd.DataFrame(all_rows)
    
    # Aggregate by method
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    agg = summary.groupby("method").agg({
        "wmape": ["mean", "std", "median"],
        "variance_ratio": ["mean", "std", "median"],
        "rmse": ["mean", "std", "median"],
        "inventory_cost": ["mean", "std", "median"],
        "safety_stock": ["mean", "std", "median"],
    }).reset_index()
    
    # Flatten columns
    agg.columns = [' '.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    print(agg.to_string(index=False))
    
    # Per-category results for CD
    print("\n--- CD Results by Category ---")
    cd_summary = summary[summary["method"] == "CD"]
    cat_agg = cd_summary.groupby("category").agg({
        "wmape": ["mean", "std", "median"],
        "variance_ratio": ["mean", "std", "median"],
        "safety_stock": ["mean", "std", "median"],
    }).reset_index()
    cat_agg.columns = [' '.join(col).strip() if col[1] else col[0] for col in cat_agg.columns.values]
    print(cat_agg.to_string(index=False))

    # Save results
    out_path = os.path.join(args.output_dir, f"m5_{args.pooling}_pooled.json")
    with open(out_path, "w") as f:
        json.dump(agg.to_dict(), f, indent=2, default=str)
    
    summary.to_csv(os.path.join(args.output_dir, f"m5_{args.pooling}_pooled.csv"), index=False)
    
    print(f"\nResults saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
