import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

base = _os.path.join(_ROOT, "code/results")
df = pd.read_csv(os.path.join(base, "tuned_xgboost_comparison.csv"))

# Pivot to get per-SKU VR for CD and Tuned XGBoost
cd = df[df['method'] == 'CD'][['sku_id', 'variance_ratio', 'wmape', 'safety_stock']].rename(
    columns={'variance_ratio': 'vr_cd', 'wmape': 'wmape_cd', 'safety_stock': 'ss_cd'})
xgb = df[df['method'] == 'Tuned XGBoost'][['sku_id', 'variance_ratio', 'wmape', 'safety_stock']].rename(
    columns={'variance_ratio': 'vr_xgb', 'wmape': 'wmape_xgb', 'safety_stock': 'ss_xgb'})

paired = cd.merge(xgb, on='sku_id')
paired['vr_diff'] = paired['vr_cd'] - paired['vr_xgb']  # CD - Tuned XGBoost
paired['wmape_diff'] = paired['wmape_cd'] - paired['wmape_xgb']
paired['ss_diff'] = paired['ss_cd'] - paired['ss_xgb']

print(f"Paired SKUs: {len(paired)}")
print(f"\nVR Difference (CD - Tuned XGBoost):")
print(f"  Mean:   {paired['vr_diff'].mean():.4f}")
print(f"  Median: {paired['vr_diff'].median():.4f}")
print(f"  Std:    {paired['vr_diff'].std():.4f}")
print(f"  CD wins (diff < 0): {(paired['vr_diff'] < 0).sum()} / {len(paired)} ({(paired['vr_diff'] < 0).mean()*100:.1f}%)")
print(f"  CD loses (diff > 0): {(paired['vr_diff'] > 0).sum()} / {len(paired)} ({(paired['vr_diff'] > 0).mean()*100:.1f}%)")
print(f"  Tie (diff = 0): {(paired['vr_diff'] == 0).sum()}")

print(f"\nwMAPE Difference (CD - Tuned XGBoost):")
print(f"  Mean:   {paired['wmape_diff'].mean():.4f}")
print(f"  Median: {paired['wmape_diff'].median():.4f}")
print(f"  CD wins (diff < 0): {(paired['wmape_diff'] < 0).sum()} / {len(paired)} ({(paired['wmape_diff'] < 0).mean()*100:.1f}%)")

print(f"\nSafety Stock Difference (CD - Tuned XGBoost):")
print(f"  Mean:   {paired['ss_diff'].mean():.4f}")
print(f"  Median: {paired['ss_diff'].median():.4f}")
print(f"  CD wins (carries less, diff < 0): {(paired['ss_diff'] < 0).sum()} / {len(paired)} ({(paired['ss_diff'] < 0).mean()*100:.1f}%)")

# Save paired data
paired.to_csv(os.path.join(base, "paired_comparison.csv"), index=False)
print(f"\nSaved paired data to paired_comparison.csv")
