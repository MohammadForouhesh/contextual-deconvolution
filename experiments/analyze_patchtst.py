import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import pandas as pd
import numpy as np

df = pd.read_csv(_os.path.join(_ROOT, "code/results/m5_200 PatchTST_global.csv"))

print("=== PatchTST vs CD on 200 M5 SKUs (Global Pooled) ===")
print()

# Aggregate
agg = df.groupby("method").agg({
    "wmape": ["mean", "std", "median"],
    "variance_ratio": ["mean", "std", "median"],
    "safety_stock": ["mean", "std", "median"],
    "rmse": ["mean", "std", "median"],
})
print(agg.to_string())
print()

# Paired comparison
cd = df[df["method"] == "CD"].set_index("sku_id")
pt = df[df["method"] == "PatchTST"].set_index("sku_id")
paired = cd.join(pt, lsuffix="_cd", rsuffix="_pt")

paired["vr_diff"] = paired["variance_ratio_cd"] - paired["variance_ratio_pt"]
paired["wmape_diff"] = paired["wmape_cd"] - paired["wmape_pt"]
paired["ss_diff"] = paired["safety_stock_cd"] - paired["safety_stock_pt"]

print("=== Paired (CD - PatchTST) ===")
print(f"CD lower VR: {(paired['vr_diff'] < 0).sum()}/{len(paired)} = {(paired['vr_diff'] < 0).mean()*100:.1f}%")
print(f"CD lower wMAPE: {(paired['wmape_diff'] < 0).sum()}/{len(paired)} = {(paired['wmape_diff'] < 0).mean()*100:.1f}%")
print(f"CD lower safety stock: {(paired['ss_diff'] < 0).sum()}/{len(paired)} = {(paired['ss_diff'] < 0).mean()*100:.1f}%")
print()
print(f"VR diff median: {paired['vr_diff'].median():.4f}")
print(f"wMAPE diff median: {paired['wmape_diff'].median():.4f}")
print(f"Safety stock diff median: {paired['ss_diff'].median():.4f}")
print()

# Per-category
print("=== Per Category (CD) ===")
cd_only = df[df["method"] == "CD"]
print(cd_only.groupby("category")[["wmape", "variance_ratio", "safety_stock"]].median().to_string())
print()
print("=== Per Category (PatchTST) ===")
pt_only = df[df["method"] == "PatchTST"]
print(pt_only.groupby("category")[["wmape", "variance_ratio", "safety_stock"]].median().to_string())
