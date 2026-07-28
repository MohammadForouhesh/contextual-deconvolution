#!/usr/bin/env python3
"""
Run PatchTST neural baseline on stratified M5 subset alongside CD.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import CONFIG_DEFAULT, estimate_all_kernels, pool_kernels, evaluate_sku_pooled
from contextual_deconvolution.baselines_dl import run_patchtst
from contextual_deconvolution.metrics import (
    wmape, variance_ratio, inventory_cost, rmse, mae,
    spectral_flatness, max_drawdown,
    mase, rmsse, order_variance_under_replenishment,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/m5-data/m5-forecasting-accuracy")
    parser.add_argument("--n_skus", type=int, default=200, help="Number of SKUs (stratified by category)")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--pooling", type=str, default="global", choices=["per_sku", "category", "global"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = CONFIG_DEFAULT.copy()
    config["run_ablations"] = False
    config["run_baselines"] = False

    print("=" * 70)
    print(f"PATCHTST BASELINE | n_skus={args.n_skus} | Pooling={args.pooling}")
    print("=" * 70)

    # Load all M5 data
    print("\nLoading M5 data...")
    full_dataset = load_m5_data(args.data_dir, n_skus=None, skip_prices=True, random_state=42)
    print(f"Total SKUs available: {len(full_dataset)}")

    # Stratified sampling: equal from each category
    cats = pd.Series([d["category"] for d in full_dataset]).value_counts()
    print(f"\nCategories: {cats.to_dict()}")

    by_cat = {cat: [] for cat in cats.index}
    for d in full_dataset:
        by_cat[d["category"]].append(d)

    n_per_cat = args.n_skus // len(cats)
    dataset = []
    for cat, items in by_cat.items():
        np.random.seed(42)
        selected = np.random.choice(len(items), min(n_per_cat, len(items)), replace=False)
        for idx in selected:
            dataset.append(items[idx])

    # If we need more to reach n_skus, sample randomly from remainder
    while len(dataset) < args.n_skus:
        remaining = [d for d in full_dataset if d not in dataset]
        if not remaining:
            break
        dataset.append(remaining[np.random.randint(len(remaining))])

    print(f"\nStratified sample: {len(dataset)} SKUs")
    cats_sample = pd.Series([d["category"] for d in dataset]).value_counts()
    print(cats_sample.to_string())

    # Kernel estimation
    print(f"\n--- Estimating kernels ---")
    t0 = time.time()
    kernels = estimate_all_kernels(dataset, config)
    pooled = pool_kernels(kernels, strategy=args.pooling)
    t1 = time.time()
    print(f"Done in {t1 - t0:.1f}s")

    # Build kernel lookup
    if args.pooling == "global":
        global_kernel = list(pooled.values())[0]
        def get_kernel(sku): return global_kernel
    elif args.pooling == "category":
        cat_to_kernel = {}
        for k in pooled.values():
            cat_to_kernel[k["category"]] = k
        def get_kernel(sku):
            return cat_to_kernel.get(sku.get("category", "unknown"), list(pooled.values())[0])
    else:
        def get_kernel(sku):
            sku_id = sku.get("sku_id", 0)
            if sku_id in pooled:
                return pooled[sku_id]
            cat = sku.get("category", "unknown")
            for k in pooled.values():
                if k["category"] == cat:
                    return k
            return list(pooled.values())[0]

    # Evaluation
    print(f"\n--- Evaluating {len(dataset)} SKUs ---")
    t2 = time.time()
    all_rows = []

    for i, sku in enumerate(dataset):
        sku_id = sku.get("sku_id", 0)
        k = get_kernel(sku)

        # CD with pooled kernel
        r = evaluate_sku_pooled(sku, k["phi_promo"], k["phi_snap"], config)
        for method, metrics in r["forecasts"].items():
            all_rows.append({
                "sku_id": sku_id,
                "category": sku.get("category", "unknown"),
                "method": method,
                **{k: v for k, v in metrics.items() if k not in ("pred", "actual")},
            })

        # PatchTST baseline
        b = sku["demand"]
        f_promo = sku["f_promo"]
        f_snap = sku["f_snap"]
        horizon = config["forecast_horizon"]

        pt_pred, pt_actual, pt_fitted, _ = run_patchtst(b, f_promo, f_snap, horizon, device=args.device)
        if pt_pred is not None:
            inv_cost, ss = inventory_cost(
                pt_actual, pt_pred, pt_fitted,
                lead_time=config["lead_time"],
                h=config["holding_cost"],
                p=config["stockout_cost"],
                service_level=config["service_level"],
            )
            all_rows.append({
                "sku_id": sku_id,
                "category": sku.get("category", "unknown"),
                "method": "PatchTST",
                "wmape": wmape(pt_actual, pt_pred),
                "variance_ratio": variance_ratio(pt_pred, pt_actual),
                "rmse": rmse(pt_actual, pt_pred),
                "mae": mae(pt_actual, pt_pred),
                "mase": mase(pt_actual, pt_pred),
                "rmsse": rmsse(pt_actual, pt_pred),
                "order_variance": order_variance_under_replenishment(pt_pred, pt_actual, fitted_train=pt_fitted),
                "inventory_cost": inv_cost,
                "safety_stock": ss,
                "spectral_flatness": spectral_flatness(pt_pred),
                "max_drawdown": max_drawdown(pt_pred - pt_actual),
            })

        if (i + 1) % 10 == 0 or i == len(dataset) - 1:
            elapsed = time.time() - t2
            per_item = elapsed / (i + 1)
            remaining = per_item * (len(dataset) - i - 1)
            print(f"  {i+1}/{len(dataset)} | elapsed: {elapsed:.0f}s | per_item: {per_item:.1f}s | remaining: {remaining:.0f}s")

    t3 = time.time()
    print(f"\nTotal evaluation: {t3 - t2:.1f}s ({(t3-t2)/len(dataset):.1f}s per SKU)")

    # Aggregate
    summary = pd.DataFrame(all_rows)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    agg = summary.groupby("method").agg({
        "wmape": ["mean", "std", "median"],
        "variance_ratio": ["mean", "std", "median"],
        "rmse": ["mean", "std", "median"],
        "safety_stock": ["mean", "std", "median"],
    }).reset_index()
    agg.columns = [' '.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    print(agg.to_string(index=False))

    # Save
    suffix = f"m5_{args.n_skus} PatchTST_{args.pooling}"
    out_csv = os.path.join(args.output_dir, f"{suffix}.csv")
    summary.to_csv(out_csv, index=False)
    out_json = os.path.join(args.output_dir, f"{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(agg.to_dict(), f, indent=2, default=str)
    print(f"\nSaved to {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
