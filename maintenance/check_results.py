import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import pandas as pd
import os

base = _os.path.join(_ROOT, "code/results")

# 1. Check tuned_xgboost_comparison.csv
df_tuned = pd.read_csv(os.path.join(base, "tuned_xgboost_comparison.csv"))
print("=== tuned_xgboost_comparison.csv ===")
print(f"Shape: {df_tuned.shape}")
print(f"Methods: {df_tuned['method'].unique().tolist()}")
print(df_tuned.groupby("method").agg({
    "wmape": ["mean", "std", "median"],
    "variance_ratio": ["mean", "std", "median"],
    "safety_stock": ["mean", "std", "median"],
}).to_string())
print()

# 2. Check m5_1500_global_pooled.csv
df_m5_1500 = pd.read_csv(os.path.join(base, "m5_1500_global_pooled.csv"))
print("=== m5_1500_global_pooled.csv ===")
print(f"Shape: {df_m5_1500.shape}")
print(f"Methods: {df_m5_1500['method'].unique().tolist()}")
print(df_m5_1500.groupby("method").agg({
    "wmape": ["mean", "std", "median"],
    "variance_ratio": ["mean", "std", "median"],
}).to_string())
print()

# 3. Check favorita_global_pooled.csv
df_fav = pd.read_csv(os.path.join(base, "favorita_global_pooled.csv"))
print("=== favorita_global_pooled.csv ===")
print(f"Shape: {df_fav.shape}")
print(f"Methods: {df_fav['method'].unique().tolist()}")
print(df_fav.groupby("method").agg({
    "wmape": ["mean", "std", "median"],
    "variance_ratio": ["mean", "std", "median"],
}).to_string())
