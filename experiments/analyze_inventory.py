import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import pandas as pd

df = pd.read_csv(_os.path.join(_ROOT, "code/results/inventory_simulation_100skus.csv"))

print("=== Aggregate ===")
agg = df.groupby("method").agg({
    "wmape": ["mean", "median"],
    "variance_ratio": ["mean", "median"],
    "safety_stock": ["mean", "median"],
    "avg_daily_cost": ["mean", "median"],
    "holding_cost": ["mean", "median"],
    "stockout_cost": ["mean", "median"],
    "event_fill_rate": ["mean", "median"],
    "event_stockout_rate": ["mean", "median"],
})
print(agg.to_string())

print("\n=== Paired (CD - XGB) ===")
cd = df[df["method"] == "CD"].set_index("sku_id")
xgb = df[df["method"] == "Tuned XGBoost"].set_index("sku_id")
paired = cd.join(xgb, lsuffix="_cd", rsuffix="_xgb")

# Compute differences
paired["ss_diff"] = paired["safety_stock_cd"] - paired["safety_stock_xgb"]
paired["hold_diff"] = paired["holding_cost_cd"] - paired["holding_cost_xgb"]
paired["stock_diff"] = paired["stockout_cost_cd"] - paired["stockout_cost_xgb"]
paired["fill_diff"] = paired["event_fill_rate_cd"] - paired["event_fill_rate_xgb"]
paired["total_cost_cd"] = paired["holding_cost_cd"] + paired["stockout_cost_cd"]
paired["total_cost_xgb"] = paired["holding_cost_xgb"] + paired["stockout_cost_xgb"]
paired["total_diff"] = paired["total_cost_cd"] - paired["total_cost_xgb"]

print(f"CD lower safety stock: {(paired['ss_diff'] < 0).sum()}/{len(paired)} = {(paired['ss_diff'] < 0).mean()*100:.1f}%")
print(f"CD lower holding cost: {(paired['hold_diff'] < 0).sum()}/{len(paired)} = {(paired['hold_diff'] < 0).mean()*100:.1f}%")
print(f"CD lower stockout cost: {(paired['stock_diff'] < 0).sum()}/{len(paired)} = {(paired['stock_diff'] < 0).mean()*100:.1f}%")
print(f"CD lower total cost: {(paired['total_diff'] < 0).sum()}/{len(paired)} = {(paired['total_diff'] < 0).mean()*100:.1f}%")
print(f"CD higher event fill rate: {(paired['fill_diff'] > 0).sum()}/{len(paired)} = {(paired['fill_diff'] > 0).mean()*100:.1f}%")
print(f"CD lower event stockout rate: {(paired['event_stockout_rate_cd'] < paired['event_stockout_rate_xgb']).sum()}/{len(paired)}")

print("\n=== Median Differences ===")
print(f"Safety stock diff median: {paired['ss_diff'].median():.4f}")
print(f"Holding cost diff median: {paired['hold_diff'].median():.4f}")
print(f"Stockout cost diff median: {paired['stock_diff'].median():.4f}")
print(f"Total cost diff median: {paired['total_diff'].median():.4f}")
print(f"Event fill rate diff median: {paired['fill_diff'].median():.4f}")

# Per-category
print("\n=== Per Category (CD) ===")
cd_only = df[df["method"] == "CD"]
print(cd_only.groupby("category")[["safety_stock", "event_fill_rate", "event_stockout_rate"]].median().to_string())

print("\n=== Per Category (Tuned XGBoost) ===")
xgb_only = df[df["method"] == "Tuned XGBoost"]
print(xgb_only.groupby("category")[["safety_stock", "event_fill_rate", "event_stockout_rate"]].median().to_string())
