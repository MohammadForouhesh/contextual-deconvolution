#!/usr/bin/env python3
"""
Large-scale M5 evaluation: 500 SKUs with baselines + timing + per-SKU results.
Outputs comprehensive results for statistical testing and scalability analysis.
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
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import run_evaluation, CONFIG_DEFAULT
from contextual_deconvolution.metrics import wmape, variance_ratio, rmse, mae, inventory_cost

# Set seed for reproducibility
np.random.seed(42)

RESULTS_DIR = _os.path.join(_ROOT, "paper/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================================================================
# Configuration
# ========================================================================
N_SKUS = int(os.environ.get("N_SKUS", 100))  # default 100; override e.g. N_SKUS=2000 for a scale test
N_JOBS = -1  # Use all CPUs

config = CONFIG_DEFAULT.copy()
config["use_ista"] = True
config["run_baselines"] = True
config["baseline_methods"] = ["XGBoost"]  # Only fast baseline for large-scale
config["dl_baseline_methods"] = []
config["run_ablations"] = False

print(f"=" * 60)
print(f"Large-scale M5 Evaluation: {N_SKUS} SKUs")
print(f"Parallel jobs: {N_JOBS}")
print(f"=" * 60)

# ========================================================================
# Load data
# ========================================================================
print("Loading M5 data...")
start = time.time()
dataset = load_m5_data(
    _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy"),
    n_skus=N_SKUS
)
load_time = time.time() - start
print(f"Loaded {len(dataset)} SKUs in {load_time:.1f}s")

# ========================================================================
# Run evaluation with timing
# ========================================================================
print(f"\nRunning evaluation on {len(dataset)} SKUs...")
start = time.time()
results, summary = run_evaluation(dataset, config=config, n_jobs=N_JOBS)
eval_time = time.time() - start
print(f"Evaluation completed in {eval_time:.1f}s ({eval_time/len(results):.2f}s per SKU)")

# ========================================================================
# Save raw summary
# ========================================================================
summary_path = os.path.join(RESULTS_DIR, "m5_500sku_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\nSaved summary to {summary_path}")

# ========================================================================
# Compute aggregate statistics per method
# ========================================================================
print("\n" + "=" * 60)
print("AGGREGATE RESULTS (mean ± std)")
print("=" * 60)

agg = summary.groupby("method").agg({
    "wmape": ["mean", "std", "median"],
    "variance_ratio": ["mean", "std", "median"],
    "rmse": ["mean", "std", "median"],
    "mae": ["mean", "std", "median"],
    "inventory_cost": ["mean", "std", "median"],
    "safety_stock": ["mean", "std", "median"],
}).round(4)

print(agg)

# Save aggregate
agg_path = os.path.join(RESULTS_DIR, "m5_500sku_aggregate.csv")
agg.to_csv(agg_path)
print(f"\nSaved aggregate to {agg_path}")

# ========================================================================
# Per-SKU paired results for statistical testing
# ========================================================================
print("\n" + "=" * 60)
print("BUILDING PAIRED DATASET FOR STATISTICAL TESTS")
print("=" * 60)

# Pivot to wide format: one row per SKU, columns for each method
methods = summary["method"].unique()
metrics = ["wmape", "variance_ratio", "rmse", "inventory_cost", "safety_stock"]

paired = {}
for metric in metrics:
    pivot = summary.pivot(index="sku_id", columns="method", values=metric)
    paired[metric] = pivot

# Save paired data
for metric, df in paired.items():
    path = os.path.join(RESULTS_DIR, f"m5_500sku_paired_{metric}.csv")
    df.to_csv(path)
    print(f"Saved paired {metric} data: {path}")

# ========================================================================
# Statistical significance tests
# ========================================================================
from scipy import stats

print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 60)

our_methods = [m for m in methods if "Ours" in m or m == "CD"]
baseline_methods = [m for m in methods if m not in our_methods and m != "Prophet" and m != "STL+ARIMA"]

sig_results = []

for metric in metrics:
    pivot = paired[metric]
    for our in our_methods:
        if our not in pivot.columns:
            continue
        for baseline in baseline_methods:
            if baseline not in pivot.columns:
                continue
            # Get paired samples (same SKUs)
            common_skus = pivot.index[pivot[our].notna() & pivot[baseline].notna()]
            if len(common_skus) < 10:
                continue
            ours_vals = pivot.loc[common_skus, our].values
            base_vals = pivot.loc[common_skus, baseline].values
            
            # Wilcoxon signed-rank test (paired, non-parametric)
            try:
                w_stat, w_p = stats.wilcoxon(ours_vals, base_vals, alternative="two-sided")
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            
            # Paired t-test
            t_stat, t_p = stats.ttest_rel(ours_vals, base_vals)
            
            # Effect size: Cohen's d for paired samples
            diff = ours_vals - base_vals
            cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
            
            sig_results.append({
                "metric": metric,
                "our_method": our,
                "baseline": baseline,
                "n_skus": len(common_skus),
                "our_mean": np.mean(ours_vals),
                "base_mean": np.mean(base_vals),
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "ttest_stat": t_stat,
                "ttest_p": t_p,
                "cohens_d": cohens_d,
            })

sig_df = pd.DataFrame(sig_results)
sig_path = os.path.join(RESULTS_DIR, "m5_500sku_significance.csv")
sig_df.to_csv(sig_path, index=False)
print(f"\nSaved significance tests to {sig_path}")

# Print key results
print("\nKey significance results (Wilcoxon p-value, Bonferroni-corrected for multiple comparisons):")
n_comparisons = len(sig_df) // len(metrics) * len(metrics)  # rough estimate
for _, row in sig_df.iterrows():
    if row["metric"] == "variance_ratio":
        sig = "***" if row["wilcoxon_p"] < 0.001 / len(sig_df) else "**" if row["wilcoxon_p"] < 0.01 / len(sig_df) else "*" if row["wilcoxon_p"] < 0.05 / len(sig_df) else "ns"
        print(f"  {row['our_method']} vs {row['baseline']}: VR mean {row['our_mean']:.4f} vs {row['base_mean']:.4f}, "
              f"p={row['wilcoxon_p']:.2e} {sig}, d={row['cohens_d']:.3f}")

# ========================================================================
# Scalability data (timing from this run + per-SKU decomposition timing)
# ========================================================================
print("\n" + "=" * 60)
print("SCALABILITY DATA")
print("=" * 60)

scalability = {
    "n_skus": len(results),
    "total_time_seconds": eval_time,
    "time_per_sku_seconds": eval_time / len(results) if results else 0,
    "load_time_seconds": load_time,
    "timestamp": pd.Timestamp.now().isoformat(),
}

scal_path = os.path.join(RESULTS_DIR, "m5_500ku_scalability.json")
with open(scal_path, "w") as f:
    json.dump(scalability, f, indent=2)
print(f"Saved scalability data to {scal_path}")
print(f"  Total time: {eval_time:.1f}s for {len(results)} SKUs")
print(f"  Per SKU: {eval_time/len(results):.2f}s")

# ========================================================================
# Sensitivity: performance vs promotion density
# ========================================================================
print("\n" + "=" * 60)
print("SENSITIVITY: PERFORMANCE VS PROMOTION DENSITY")
print("=" * 60)

# Compute promotion density per SKU from per-SKU price-discount depth.
# NOTE: f_promo is the global event calendar (identical for every SKU), so binning
# on it collapses all SKUs into a single bin; promo_depth varies per SKU.
promo_density = []
for r in results:
    promo_depth = np.asarray(r.get("promo_depth", np.zeros(1)))
    density = np.mean(promo_depth > 0) if len(promo_depth) > 0 else 0
    promo_density.append({"sku_id": r["sku_id"], "promo_density": density})

df_density = pd.DataFrame(promo_density)

# Merge with summary
merged = summary.merge(df_density, on="sku_id", how="left")

# Bin by promotion density
merged["promo_bin"] = pd.cut(merged["promo_density"], bins=[0, 0.02, 0.05, 0.1, 1.0],
                               labels=["Very Low (<2%)", "Low (2-5%)", "Medium (5-10%)", "High (>10%)"],
                               include_lowest=True)

sens = merged.groupby(["promo_bin", "method"], observed=True).agg(
    n=("variance_ratio", "count"),
    vr_mean=("variance_ratio", "mean"),
    vr_median=("variance_ratio", "median"),
    wmape_mean=("wmape", "mean"),
    wmape_median=("wmape", "median"),
    safety_stock_mean=("safety_stock", "mean"),
    safety_stock_median=("safety_stock", "median"),
    inv_cost_mean=("inventory_cost", "mean"),
    inv_cost_median=("inventory_cost", "median"),
).round(4)

print(sens.to_string())
sens_path = os.path.join(RESULTS_DIR, "m5_500sku_sensitivity_promo.csv")
sens.to_csv(sens_path)
print(f"\nSaved sensitivity analysis to {sens_path}")

# ========================================================================
# Failure case analysis
# ========================================================================
print("\n" + "=" * 60)
print("FAILURE CASE ANALYSIS")
print("=" * 60)

# Find SKUs where our method does worse than baselines
our_vr = summary[summary["method"].str.contains("Ours|CD")][["sku_id", "variance_ratio"]].rename(columns={"variance_ratio": "ours_vr"})

failure_cases = []
for _, row in our_vr.iterrows():
    sku = row["sku_id"]
    sku_data = summary[summary["sku_id"] == sku]
    base_vr = sku_data[sku_data["method"] != "CD"]["variance_ratio"]
    if len(base_vr) > 0:
        min_base_vr = base_vr.min()
        if row["ours_vr"] > min_base_vr * 1.5:  # Our VR is 50% worse than best baseline
            failure_cases.append({
                "sku_id": sku,
                "ours_vr": row["ours_vr"],
                "best_baseline_vr": min_base_vr,
                "ratio": row["ours_vr"] / (min_base_vr + 1e-10),
            })

fail_df = pd.DataFrame(failure_cases)
if len(fail_df) > 0:
    fail_df = fail_df.sort_values("ratio", ascending=False)
    fail_path = os.path.join(RESULTS_DIR, "m5_500sku_failure_cases.csv")
    fail_df.to_csv(fail_path, index=False)
    print(f"Found {len(fail_df)} failure cases ({len(fail_df)/len(our_vr)*100:.1f}% of SKUs)")
    print(f"Saved to {fail_path}")
    print(fail_df.head(10).to_string(index=False))
else:
    print("No clear failure cases found (our method competitive on all SKUs)")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
