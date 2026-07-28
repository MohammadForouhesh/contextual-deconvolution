#!/usr/bin/env python3
"""
Run Favorita evaluation at scale: 100 random items with XGBoost baseline.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import os
import sys
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextual_deconvolution.loaders import load_favorita_data
from contextual_deconvolution.experiments import run_evaluation, CONFIG_DEFAULT

RESULTS_DIR = _os.path.join(_ROOT, "paper/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

config = CONFIG_DEFAULT.copy()
config["use_ista"] = True
config["run_baselines"] = True
config["baseline_methods"] = ["XGBoost"]
config["dl_baseline_methods"] = []
config["run_ablations"] = False

print("=" * 60)
print("Favorita Large-Scale Evaluation: 100 random items")
print("=" * 60)

print("Loading Favorita data...")
start = time.time()
dataset = load_favorita_data(
    _os.path.join(_ROOT, "data/favorita-grocery-sales-forecasting"),
    n_items=100,
    random_state=12345,
)
load_time = time.time() - start
print(f"Loaded {len(dataset)} items in {load_time:.1f}s")

print("Running evaluation...")
start = time.time()
results, summary = run_evaluation(dataset, config=config, n_jobs=-1)
eval_time = time.time() - start
print(f"Completed in {eval_time:.1f}s ({eval_time/len(results):.2f}s per item)")

summary["dataset"] = "favorita"
output_path = os.path.join(RESULTS_DIR, "favorita_100item_results.csv")
summary.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

agg = summary.groupby("method").agg({
    "wmape": ["mean", "std"],
    "variance_ratio": ["mean", "std"],
    "rmse": ["mean", "std"],
    "inventory_cost": ["mean", "std"],
    "safety_stock": ["mean", "std"],
}).round(4)

print("\nAggregate results (mean ± std):")
print(agg)

# Save aggregate
agg_path = os.path.join(RESULTS_DIR, "favorita_100item_aggregate.csv")
agg.to_csv(agg_path)
print(f"\nSaved aggregate to {agg_path}")

# Statistical tests (paired by sku_id)
from scipy import stats

for metric in ["wmape", "variance_ratio", "rmse", "safety_stock", "inventory_cost"]:
    pivot = summary.pivot_table(index="sku_id", columns="method", values=metric, aggfunc="first")
    if "CD" not in pivot.columns or "XGBoost" not in pivot.columns:
        continue
    common = pivot.index[pivot["CD"].notna() & pivot["XGBoost"].notna()]
    if len(common) < 10:
        continue
    our_m = pivot.loc[common, "CD"].values
    xgb_m = pivot.loc[common, "XGBoost"].values
    w_stat, w_p = stats.wilcoxon(our_m, xgb_m)
    d = np.mean(our_m - xgb_m) / (np.std(our_m - xgb_m, ddof=1) + 1e-10)
    print(f"{metric:20s}: Ours {np.mean(our_m):.4f} vs XGB {np.mean(xgb_m):.4f}, p={w_p:.2e}, d={d:.3f}")

print("\nFavorita evaluation complete.")
