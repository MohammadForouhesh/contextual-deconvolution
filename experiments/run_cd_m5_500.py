#!/usr/bin/env python3
"""Run CD on 500 M5 SKUs and save results."""
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import os, sys, time, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, _os.path.join(_ROOT, "code"))
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import CONFIG_DEFAULT, run_evaluation

m5_dir = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")
out_dir = _os.path.join(_ROOT, "paper/results")
os.makedirs(out_dir, exist_ok=True)

print("Loading M5 data (500 SKUs)...")
data = load_m5_data(m5_dir, n_skus=500, random_state=42)
print(f"Loaded {len(data)} SKUs")

config = CONFIG_DEFAULT.copy()
config["run_baselines"] = False
config["run_ablations"] = False
config["use_ista"] = True

print("Running CD (parallel, n_jobs=-1)...")
t0 = time.time()
results, summary = run_evaluation(data, config=config, n_jobs=-1)
elapsed = time.time() - t0
print(f"CD completed in {elapsed:.1f}s")

cd_summary = summary[summary["method"] == "CD"]
print(f"\n--- CD Results (n={len(cd_summary)}) ---")
print(f"  wMAPE:     {cd_summary['wmape'].mean():.4f} ± {cd_summary['wmape'].std():.4f}")
print(f"  RMSE:      {cd_summary['rmse'].mean():.4f} ± {cd_summary['rmse'].std():.4f}")
print(f"  VarRatio:  {cd_summary['variance_ratio'].mean():.4f} ± {cd_summary['variance_ratio'].std():.4f}")
print(f"  Median VR: {cd_summary['variance_ratio'].median():.4f}")

cd_summary.to_csv(os.path.join(out_dir, "cd_m5_500.csv"), index=False)
print(f"Saved to {out_dir}/cd_m5_500.csv")
