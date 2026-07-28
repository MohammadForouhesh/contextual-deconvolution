#!/usr/bin/env python3
"""
Run comparison: Contextual Deconvolution vs Tuned XGBoost on M5.

Usage:
    python scripts/run_tuned_xgboost_comparison.py --data_dir ../data/m5-data/m5-forecasting-accuracy --n_skus 50 --output_dir results
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import run_evaluation, CONFIG_DEFAULT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/m5-data/m5-forecasting-accuracy")
    parser.add_argument("--n_skus", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = CONFIG_DEFAULT.copy()
    # Only run CD and Tuned XGBoost for speed
    config["baseline_methods"] = ["Tuned XGBoost"]
    config["run_ablations"] = False

    print("=" * 70)
    print("CD vs TUNED XGBOOST COMPARISON")
    print("=" * 70)

    print(f"\nLoading M5 data from {args.data_dir}...")
    dataset = load_m5_data(args.data_dir, n_skus=args.n_skus, random_state=42)
    print(f"Loaded {len(dataset)} SKUs")

    print("\nRunning evaluation...")
    results, summary = run_evaluation(dataset, config, n_jobs=-1)

    # Compute aggregate stats
    agg = summary.groupby("method").agg({
        "wmape": ["mean", "std", "median"],
        "variance_ratio": ["mean", "std", "median"],
        "rmse": ["mean", "std", "median"],
        "inventory_cost": ["mean", "std", "median"],
        "safety_stock": ["mean", "std", "median"],
    }).reset_index()
    
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(agg.to_string(index=False))

    # Save results
    out_path = os.path.join(args.output_dir, "tuned_xgboost_comparison.json")
    with open(out_path, "w") as f:
        # Flatten MultiIndex columns for JSON
        agg.columns = [' '.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
        json.dump(agg.to_dict(), f, indent=2, default=str)
    
    # Also save per-SKU metrics for CI computation
    summary.to_csv(os.path.join(args.output_dir, "tuned_xgboost_comparison.csv"), index=False)
    
    print(f"\nResults saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
