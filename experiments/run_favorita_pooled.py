#!/usr/bin/env python3
"""
Run Favorita evaluation with global pooled kernels.

Usage:
    python scripts/run_favorita_pooled.py --data_dir ../data/favorita --n_items 100 --output_dir results
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextual_deconvolution.loaders import load_favorita_data
from contextual_deconvolution.experiments import (
    CONFIG_DEFAULT,
    estimate_all_kernels,
    pool_kernels,
    evaluate_sku_pooled,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/favorita-grocery-sales-forecasting")
    parser.add_argument("--n_items", type=int, default=3000,
                        help="Cap on items (default 3000; ~2863 pass the min-length filter). Pass 0 for the full catalog.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--pooling", type=str, default="global", choices=["per_sku", "category", "global"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = CONFIG_DEFAULT.copy()
    config["run_baselines"] = False
    config["baseline_methods"] = []
    config["run_ablations"] = False

    print(f"Loading Favorita data...")
    dataset = load_favorita_data(
        args.data_dir,
        n_items=(args.n_items if args.n_items and args.n_items > 0 else None),
        random_state=42,
    )
    print(f"Loaded {len(dataset)} items")
    
    cats = pd.Series([d["category"] for d in dataset]).value_counts()
    print(f"Category distribution:\n{cats.to_string()}")

    # Estimate kernels
    print(f"Estimating kernels...")
    t0 = time.time()
    kernels = estimate_all_kernels(dataset, config)
    pooled = pool_kernels(kernels, strategy=args.pooling)
    t1 = time.time()
    print(f"Kernel estimation: {t1 - t0:.1f}s")

    # Evaluate
    print(f"Evaluating {len(dataset)} items...")
    t2 = time.time()
    all_rows = []
    for i, sku in enumerate(dataset):
        sku_id = sku.get("sku_id", 0)
        k = pooled[sku_id]
        r = evaluate_sku_pooled(sku, k["phi_promo"], k["phi_snap"], config)
        for method, metrics in r["forecasts"].items():
            all_rows.append({
                "sku_id": r["sku_id"],
                "category": r.get("category", "unknown"),
                "method": method,
                **{k: v for k, v in metrics.items() if k not in ("pred", "actual")},
            })
        if (i+1) % 50 == 0:
            print(f"  {i+1} done, elapsed: {time.time()-t2:.1f}s")

    t3 = time.time()
    print(f"Evaluation: {t3 - t2:.1f}s")

    summary = pd.DataFrame(all_rows)
    agg = summary.groupby("method").agg({
        "wmape": ["mean", "std", "median"],
        "variance_ratio": ["mean", "std", "median"],
        "rmse": ["mean", "std", "median"],
        "safety_stock": ["mean", "std", "median"],
    }).reset_index()
    agg.columns = [' '.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    print("\nAggregate Results:")
    print(agg.to_string(index=False))

    out_path = os.path.join(args.output_dir, f"favorita_{args.pooling}_pooled.json")
    with open(out_path, "w") as f:
        json.dump(agg.to_dict(), f, indent=2, default=str)
    summary.to_csv(os.path.join(args.output_dir, f"favorita_{args.pooling}_pooled.csv"), index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
