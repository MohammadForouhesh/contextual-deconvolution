#!/usr/bin/env python3
"""
Run Contextual Deconvolution (CD) with global pooled kernel on the exact 90
Favorita items used in the structural baselines, so metrics can be compared
fairly.
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
import warnings

# Add project code to path
sys.path.insert(0, _os.path.join(_ROOT, "code"))

from contextual_deconvolution.experiments import (
    CONFIG_DEFAULT,
    estimate_all_kernels,
    pool_kernels,
    evaluate_sku_pooled,
)
from contextual_deconvolution.metrics import wmape, variance_ratio, rmse, inventory_cost
from tqdm import tqdm


def load_favorita_data_filtered(data_dir, target_sku_ids):
    """
    Load Favorita data for specific item-store combinations.
    target_sku_ids: set of strings like '1463610_8'
    """
    train_path = os.path.join(data_dir, "train.csv")
    items = pd.read_csv(os.path.join(data_dir, "items.csv"))
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    holidays = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"))
    oil = pd.read_csv(os.path.join(data_dir, "oil.csv"))

    holidays["date"] = pd.to_datetime(holidays["date"])
    oil["date"] = pd.to_datetime(oil["date"])

    # Parse target item_nbrs from sku_ids
    target_items = set()
    target_pairs = set()
    for sid in target_sku_ids:
        parts = sid.split("_")
        if len(parts) == 2:
            try:
                item_nbr = int(parts[0])
                store_nbr = int(parts[1])
                target_items.add(item_nbr)
                target_pairs.add((item_nbr, store_nbr))
            except ValueError:
                pass

    print(f"[Favorita] Loading data for {len(target_pairs)} target item-store pairs...")

    # Read chunks and filter for target item_nbrs
    chunk_iter = pd.read_csv(
        train_path,
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        dtype={"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32", "onpromotion": "object"},
        chunksize=1_000_000,
    )

    filtered_chunks = []
    for chunk in chunk_iter:
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk = chunk[chunk["item_nbr"].isin(target_items)]
        if len(chunk) > 0:
            filtered_chunks.append(chunk)

    if not filtered_chunks:
        return []

    train = pd.concat(filtered_chunks, ignore_index=True)
    del filtered_chunks

    # Merge metadata
    train = train.merge(items, on="item_nbr", how="left")
    train = train.merge(stores, on="store_nbr", how="left")

    # Get unique item-store combinations, filtered to exact target pairs
    item_store_pairs = train[["item_nbr", "store_nbr"]].drop_duplicates()
    item_store_pairs = item_store_pairs[
        item_store_pairs.apply(lambda r: (r["item_nbr"], r["store_nbr"]) in target_pairs, axis=1)
    ]

    print(f"[Favorita] Found {len(item_store_pairs)} matching item-store pairs in data.")

    data = []
    for _, pair in item_store_pairs.iterrows():
        item_nbr = pair["item_nbr"]
        store_nbr = pair["store_nbr"]

        sub = train[
            (train["item_nbr"] == item_nbr) & (train["store_nbr"] == store_nbr)
        ].sort_values("date")

        if len(sub) < 30:
            continue

        dates = pd.date_range(sub["date"].min(), sub["date"].max())
        demand = sub.set_index("date")["unit_sales"].reindex(dates, fill_value=0).values
        T = len(dates)

        # Holiday features
        holiday_dates = set(holidays["date"].values)
        f_promo = np.array([1.0 if d in holiday_dates else 0.0 for d in dates])

        # On-promotion flag
        onpromo_series = sub.set_index("date")["onpromotion"].reindex(dates, fill_value=0)
        if onpromo_series.dtype == object:
            onpromo_series = onpromo_series.map(lambda x: 1.0 if str(x).lower() in ('true', '1.0', '1') else 0.0)
        promo_depth = onpromo_series.astype(float).values

        # Oil price as macro feature
        oil_aligned = oil.set_index("date")["dcoilwtico"].reindex(dates).ffill().bfill().values

        data.append({
            "sku_id": f"{item_nbr}_{store_nbr}",
            "item_nbr": item_nbr,
            "store_nbr": store_nbr,
            "category": sub["family"].iloc[0] if "family" in sub.columns else "unknown",
            "volume_stratum": "medium",
            "promo_freq": "medium",
            "demand": demand,
            "f_promo": f_promo,
            "f_snap": np.zeros(T),
            "promo_depth": promo_depth,
            "oil_price": oil_aligned,
        })

    return data


def main():
    # Paths
    favorita_dir = _os.path.join(_ROOT, "data/favorita-grocery-sales-forecasting")
    structural_csv = _os.path.join(_ROOT, "code/results/favorita_structural_baselines.csv")
    output_csv = _os.path.join(_ROOT, "code/results/favorita_cd_structural_items.csv")

    # 1. Read structural baseline sku_ids
    struct_df = pd.read_csv(structural_csv)
    target_ids = set(struct_df["sku_id"].unique())
    print(f"Target IDs from structural baselines: {len(target_ids)}")

    # 2. Load Favorita data filtered to target items
    start = time.time()
    dataset = load_favorita_data_filtered(favorita_dir, target_ids)
    load_time = time.time() - start
    print(f"Loaded {len(dataset)} items in {load_time:.1f}s")

    if len(dataset) == 0:
        print("ERROR: No items matched. Exiting.")
        return

    # Verify coverage
    loaded_ids = {d["sku_id"] for d in dataset}
    missing = target_ids - loaded_ids
    if missing:
        print(f"WARNING: Missing {len(missing)} items from target set: {sorted(missing)[:5]}...")

    # 3. Config - only CD, no baselines
    config = CONFIG_DEFAULT.copy()
    config["use_ista"] = True
    config["run_baselines"] = False
    config["run_ablations"] = False
    config["forecast_horizon"] = 28
    print(f"Using config: {config}")

    # 4. Estimate per-SKU kernels
    print("Estimating per-SKU kernels...")
    start = time.time()
    kernels = estimate_all_kernels(dataset, config)
    print(f"Kernel estimation done in {time.time() - start:.1f}s")

    # 5. Pool to global kernel
    print("Pooling kernels globally...")
    pooled_kernels = pool_kernels(kernels, strategy="global")
    print(f"Pooled {len(pooled_kernels)} kernels.")

    # 6. Evaluate each SKU with pooled kernel
    print("Evaluating each SKU with pooled kernel...")
    results = []
    for sku_data in tqdm(dataset, desc="Evaluating"):
        sku_id = sku_data["sku_id"]
        if sku_id not in pooled_kernels:
            warnings.warn(f"No pooled kernel for {sku_id}, skipping.")
            continue

        pk = pooled_kernels[sku_id]
        try:
            r = evaluate_sku_pooled(sku_data, pk["phi_promo"], pk["phi_snap"], config)
            if r is None or "forecasts" not in r or "CD" not in r["forecasts"]:
                warnings.warn(f"No CD forecast for {sku_id}, skipping.")
                continue

            cd_metrics = r["forecasts"]["CD"]
            results.append({
                "sku_id": sku_id,
                "category": sku_data.get("category", "unknown"),
                "wmape": cd_metrics["wmape"],
                "variance_ratio": cd_metrics["variance_ratio"],
                "rmse": cd_metrics["rmse"],
                "safety_stock": cd_metrics["safety_stock"],
            })
        except Exception as e:
            warnings.warn(f"Evaluation failed for SKU {sku_id}: {e}")
            continue

    print(f"Successfully evaluated {len(results)} items.")

    # 7. Save results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df = results_df[["sku_id", "category", "wmape", "variance_ratio", "rmse", "safety_stock"]]
    results_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

    # 8. Print summary
    if len(results_df) > 0:
        wmape_mean = results_df["wmape"].mean()
        wmape_std = results_df["wmape"].std()
        vr_mean = results_df["variance_ratio"].mean()
        vr_std = results_df["variance_ratio"].std()
        vr_median = results_df["variance_ratio"].median()
        rmse_mean = results_df["rmse"].mean()
        rmse_std = results_df["rmse"].std()
        ss_mean = results_df["safety_stock"].mean()
        ss_std = results_df["safety_stock"].std()

        print("\n" + "=" * 60)
        print(f"Favorita CD ({len(results_df)} items, same as structural baselines):")
        print(f"  wMAPE:        {wmape_mean:.6f} ± {wmape_std:.6f}")
        print(f"  VR:           {vr_mean:.6f} ± {vr_std:.6f}, median {vr_median:.6f}")
        print(f"  RMSE:         {rmse_mean:.6f} ± {rmse_std:.6f}")
        print(f"  Safety Stock: {ss_mean:.6f} ± {ss_std:.6f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
