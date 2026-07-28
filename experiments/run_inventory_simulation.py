#!/usr/bin/env python3
"""
Run detailed inventory simulation on 100 M5 SKUs with event-window metrics.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import CONFIG_DEFAULT, estimate_all_kernels, pool_kernels, evaluate_sku_pooled
from contextual_deconvolution.baselines import run_tuned_xgboost
from contextual_deconvolution.metrics import wmape, variance_ratio, inventory_cost_with_events


def main():
    config = CONFIG_DEFAULT.copy()
    config["run_ablations"] = False
    config["run_baselines"] = False

    print("Loading M5 data (100 SKUs)...")
    dataset = load_m5_data(
        "../data/m5-data/m5-forecasting-accuracy",
        n_skus=100, random_state=42, skip_prices=True
    )
    print(f"Loaded {len(dataset)} SKUs")

    # Estimate global pooled kernel from first 50
    print("Estimating global pooled kernel...")
    kernels = estimate_all_kernels(dataset[:50], config)
    pooled = pool_kernels(kernels, strategy="global")
    global_kernel = list(pooled.values())[0]

    print("Evaluating with detailed inventory metrics...")
    results = []

    for i, sku in enumerate(dataset):
        sku_id = sku.get("sku_id", 0)
        b = sku["demand"]
        f_promo = sku["f_promo"]
        f_snap = sku["f_snap"]
        T = len(b)
        split = T - config["forecast_horizon"]
        b_test = b[split:]
        f_promo_test = f_promo[split:]
        f_snap_test = f_snap[split:]
        event_mask = np.maximum(f_promo_test, f_snap_test)

        # CD forecast
        r = evaluate_sku_pooled(sku, global_kernel["phi_promo"], global_kernel["phi_snap"], config)
        cd_pred = r["forecasts"]["CD"]["pred"]
        cd_fitted = r["forecasts"]["CD"].get("fitted", None)
        # Reconstruct fitted from x_base + A_spec @ x_spec if not stored
        if cd_fitted is None:
            from contextual_deconvolution.core import build_Aspec
            split = len(r["x_base"])
            A_spec = build_Aspec(global_kernel["phi_promo"], global_kernel["phi_snap"], f_promo[:split], f_snap[:split], split, causal=True)
            cd_fitted = r["x_base"] + A_spec @ r["x_spec"]

        # Tuned XGBoost forecast
        xgb_pred, xgb_actual, xgb_fitted, _ = run_tuned_xgboost(b, f_promo, f_snap, config["forecast_horizon"])

        # Compute inventory metrics
        for method, pred, actual, fitted in [
            ("CD", cd_pred, b_test, cd_fitted),
            ("Tuned XGBoost", xgb_pred, xgb_actual, xgb_fitted),
        ]:
            if pred is None or actual is None:
                continue
            cost, ss, hold, stock, fill_rate, stockout_rate = inventory_cost_with_events(
                actual, pred, event_mask,
                fitted_train=fitted,
                lead_time=config["lead_time"],
                h=config["holding_cost"],
                p=config["stockout_cost"],
                service_level=config["service_level"],
            )
            results.append({
                "sku_id": sku_id,
                "category": sku.get("category", "unknown"),
                "method": method,
                "wmape": wmape(actual, pred),
                "variance_ratio": variance_ratio(pred, actual),
                "safety_stock": ss,
                "avg_daily_cost": cost,
                "holding_cost": hold,
                "stockout_cost": stock,
                "event_fill_rate": fill_rate,
                "event_stockout_rate": stockout_rate,
                "event_days": int(np.sum(event_mask)),
            })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(dataset)} done")

    df = pd.DataFrame(results)
    out_csv = "results/inventory_simulation_100skus.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("INVENTORY SIMULATION RESULTS (100 SKUs)")
    print("=" * 70)

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

    # Paired comparison
    cd_df = df[df["method"] == "CD"].set_index("sku_id")
    xgb_df = df[df["method"] == "Tuned XGBoost"].set_index("sku_id")
    paired = cd_df.join(xgb_df, lsuffix="_cd", rsuffix="_xgb")

    print("\n--- Paired Comparison (CD vs Tuned XGBoost) ---")
    print(f"CD carries less safety stock: {(paired['safety_stock_cd'] < paired['safety_stock_xgb']).sum()}/{len(paired)}")
    print(f"CD has lower holding cost: {(paired['holding_cost_cd'] < paired['holding_cost_xgb']).sum()}/{len(paired)}")
    print(f"CD has lower stockout cost: {(paired['stockout_cost_cd'] < paired['stockout_cost_xgb']).sum()}/{len(paired)}")
    print(f"CD has higher event fill rate: {(paired['event_fill_rate_cd'] > paired['event_fill_rate_xgb']).sum()}/{len(paired)}")
    print(f"CD has lower event stockout rate: {(paired['event_stockout_rate_cd'] < paired['event_stockout_rate_xgb']).sum()}/{len(paired)}")

    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
