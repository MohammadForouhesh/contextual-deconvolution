#!/usr/bin/env python3
"""
Run one batch of M5 evaluation with random sampling.
Call with --batch N to run batch N (0-4), each with 100 random SKUs.
Results are appended to a shared CSV file.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import os
import sys
import argparse
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import run_evaluation, CONFIG_DEFAULT

RESULTS_DIR = _os.path.join(_ROOT, "paper/results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0, help="Batch number (0-4)")
    parser.add_argument("--n_per_batch", type=int, default=100)
    args = parser.parse_args()

    batch = args.batch
    n_per = args.n_per_batch
    seed = 42 + batch * 1000

    print(f"=" * 60)
    print(f"Batch {batch}: {n_per} random SKUs (seed={seed})")
    print(f"=" * 60)

    config = CONFIG_DEFAULT.copy()
    config["use_ista"] = True
    config["run_baselines"] = True
    config["baseline_methods"] = ["XGBoost"]
    config["dl_baseline_methods"] = []
    config["run_ablations"] = False

    # Load random sample
    print("Loading M5 data...")
    start = time.time()
    dataset = load_m5_data(
        _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy"),
        n_skus=n_per,
        random_state=seed,
    )
    load_time = time.time() - start
    print(f"Loaded {len(dataset)} SKUs in {load_time:.1f}s")

    # Run evaluation
    print(f"Running evaluation...")
    start = time.time()
    results, summary = run_evaluation(dataset, config=config, n_jobs=-1)
    eval_time = time.time() - start
    print(f"Completed in {eval_time:.1f}s ({eval_time/len(results):.2f}s per SKU)")

    # Add batch identifier and unique SKU ID
    summary["batch"] = batch
    summary["unique_id"] = summary["batch"].astype(str) + "_" + summary["sku_id"].astype(str)

    # Append to shared file
    output_path = os.path.join(RESULTS_DIR, "m5_500sku_batches.csv")
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, summary], ignore_index=True)
    else:
        combined = summary
    combined.to_csv(output_path, index=False)
    print(f"Appended results to {output_path} (total rows: {len(combined)})")

    # Print batch summary
    print("\nBatch aggregate (mean ± std):")
    agg = summary.groupby("method").agg({
        "wmape": ["mean", "std"],
        "variance_ratio": ["mean", "std"],
        "rmse": ["mean", "std"],
        "inventory_cost": ["mean", "std"],
    }).round(4)
    print(agg)

    print(f"\nBatch {batch} complete.")


if __name__ == "__main__":
    main()
