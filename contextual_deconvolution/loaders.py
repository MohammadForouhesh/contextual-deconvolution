"""
Data loaders for M5 and Favorita datasets.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def _assert_channels_live(data, channels=("f_promo", "f_snap"), name=""):
    """Fail loudly if an exogenous event channel is all-zero across the whole dataset.
    Two silent dead-channel bugs (M5 SNAP via wrong state parse, Favorita holidays via a
    Timestamp-vs-datetime64 membership mismatch) both failed closed to zeros with no error;
    this guard makes that failure mode surface immediately instead of corrupting results."""
    import warnings as _w
    if not data:
        return data
    for ch in channels:
        total = sum(float(np.asarray(d.get(ch, 0.0)).sum()) for d in data)
        active = sum(1 for d in data if float(np.asarray(d.get(ch, 0.0)).sum()) > 0)
        if total <= 0:
            _w.warn(f"[{name}] event channel '{ch}' is ALL ZERO across {len(data)} series "
                    f"-- likely a dead-channel bug (check the loader).", RuntimeWarning, stacklevel=2)
        elif active < max(1, len(data) // 20):
            _w.warn(f"[{name}] event channel '{ch}' fires for only {active}/{len(data)} series "
                    f"-- verify this is intended.", RuntimeWarning, stacklevel=2)
    return data


def load_m5_data(
    data_dir: str,
    n_skus: Optional[int] = None,
    random_state: Optional[int] = None,
    skip_prices: bool = False,
) -> List[Dict]:
    """
    Load M5 dataset from Kaggle files.

    Parameters
    ----------
    data_dir : str
        Directory containing calendar.csv, sales_train_evaluation.csv, sell_prices.csv
    n_skus : int or None
        If provided, limit to N SKUs.
    random_state : int or None
        Random seed for sampling.
    skip_prices : bool
        If True, skip sell price computation (promo_depth will be zeros).
    """
    calendar = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    sales = pd.read_csv(os.path.join(data_dir, "sales_train_evaluation.csv"))

    demand_cols = [c for c in sales.columns if c.startswith("d_")]
    T_full = len(demand_cols)
    calendar_subset = calendar.iloc[:T_full].copy()

    # --- Defense-in-depth split sourcing -------------------------------------
    # The held-out test window is the last `forecast_horizon` days (the M5 validation
    # window, d_1914..d_1941). To make it structurally impossible for the test targets to
    # enter any training computation, we source the TRAINING days from
    # sales_train_validation.csv (which physically ends at d_1913 and cannot contain the
    # answer) and take ONLY the test days from sales_train_evaluation.csv. The two files
    # are identical over d_1..d_1913, so this changes provenance, not the numbers; we assert
    # the overlap matches to catch a wrong or corrupted file pair. Falls back to the
    # evaluation file alone if the validation file is absent.
    HORIZON = 28
    _val_path = os.path.join(data_dir, "sales_train_validation.csv")
    demand_source = None
    if os.path.exists(_val_path) and len(demand_cols) > HORIZON:
        val = pd.read_csv(_val_path)
        val_cols = [c for c in val.columns if c.startswith("d_")]
        train_cols = demand_cols[:len(demand_cols) - HORIZON]   # d_1..d_1913
        if set(train_cols).issubset(val_cols):
            key = ["item_id", "store_id"]
            val_idx = val.set_index(key)
            eval_idx = sales.set_index(key)
            common = eval_idx.index.intersection(val_idx.index)
            overlap = np.array_equal(
                val_idx.loc[common, train_cols].to_numpy(float),
                eval_idx.loc[common, train_cols].to_numpy(float))
            if overlap:
                # training days from validation.csv, test days from evaluation.csv
                merged = eval_idx.copy()
                merged.loc[common, train_cols] = val_idx.loc[common, train_cols]
                demand_source = merged.reset_index()
    if demand_source is None:
        demand_source = sales
        print("[M5] sales_train_validation.csv not found or mismatched; "
              "using sales_train_evaluation.csv alone (fitting still stops at the split).")
    _demand_by_key = {(r["item_id"], r["store_id"]): r[demand_cols].to_numpy(float)
                      for _, r in demand_source.iterrows()}

    f_promo_global = (
        (calendar_subset["event_type_1"].notna())
        | (calendar_subset["event_type_2"].notna())
    ).astype(float).values

    # Random sampling if requested
    if n_skus is not None and random_state is not None:
        rng = np.random.RandomState(random_state)
        all_indices = np.arange(len(sales))
        selected = rng.choice(all_indices, size=min(n_skus, len(sales)), replace=False)
        sales = sales.iloc[selected].reset_index(drop=True)
    elif n_skus is not None:
        sales = sales.iloc[:n_skus].reset_index(drop=True)

    if skip_prices:
        data = []
        for idx, row in sales.iterrows():
            item_id = row["item_id"]
            store_id = row["store_id"]
            state = store_id.split("_")[0]   # "CA_1" -> "CA"; [1] took the store number, so snap_{state} silently fell back to zeros
            demand = _demand_by_key[(item_id, store_id)]   # train days from validation.csv, test days from evaluation.csv
            f_snap = (
                calendar_subset.get(f"snap_{state}", pd.Series(0, index=calendar_subset.index))
                .fillna(0)
                .values
            )
            data.append({
                "sku_id": idx,
                "item_id": item_id,
                "store_id": store_id,
                "state": state,
                "category": row.get("cat_id", "food").lower(),
                "dept_id": row.get("dept_id", ""),
                "volume_stratum": "medium",
                "promo_freq": "medium",
                "demand": demand,
                "f_promo": f_promo_global,
                "f_snap": f_snap,
                "promo_depth": np.zeros(T_full),
                "sell_price": np.ones(T_full),
                "split": T_full - HORIZON,
            })
        return _assert_channels_live(data, name="M5")

    prices = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))
    
    # Pre-compute all sell prices and promo_depth in one batch for efficiency
    prices_merged = prices.merge(
        calendar_subset[["wm_yr_wk", "d"]], on="wm_yr_wk", how="right"
    )
    prices_merged = prices_merged.sort_values(["item_id", "store_id", "d"])
    prices_merged["sell_price"] = prices_merged.groupby(["item_id", "store_id"])["sell_price"].transform(lambda x: x.ffill().bfill())
    
    price_pivot = prices_merged.pivot_table(
        index=["item_id", "store_id"], columns="d", values="sell_price"
    ).fillna(1.0)
    
    data = []
    for idx, row in sales.iterrows():
        item_id = row["item_id"]
        store_id = row["store_id"]
        state = store_id.split("_")[0]   # "CA_1" -> "CA"; [1] took the store number, so snap_{state} silently fell back to zeros
        demand = _demand_by_key[(row['item_id'], row['store_id'])]
        f_snap = (
            calendar_subset.get(f"snap_{state}", pd.Series(0, index=calendar_subset.index))
            .fillna(0)
            .values
        )
        key = (item_id, store_id)
        if key in price_pivot.index:
            sell_price = price_pivot.loc[key].values
            if len(sell_price) < T_full:
                sell_price = np.pad(sell_price, (0, T_full - len(sell_price)), mode="edge")
            else:
                sell_price = sell_price[:T_full]
        else:
            sell_price = np.ones(T_full)
        baseline_price = pd.Series(sell_price).rolling(window=30, min_periods=1).median().values
        promo_depth = np.clip(1.0 - sell_price / np.maximum(baseline_price, 0.01), 0, 1)
        data.append({
            "sku_id": idx,
            "item_id": item_id,
            "store_id": store_id,
            "state": state,
            "category": row.get("cat_id", "food").lower(),
            "dept_id": row.get("dept_id", ""),
            "volume_stratum": "medium",
            "promo_freq": "medium",
            "demand": demand,
            "f_promo": f_promo_global,
            "f_snap": f_snap,
            "promo_depth": promo_depth,
            "sell_price": sell_price,
            "split": T_full - HORIZON,
        })
    return _assert_channels_live(data, name="M5")


def load_favorita_data(
    data_dir: str,
    n_items: Optional[int] = None,
    random_state: int = 42,
) -> List[Dict]:
    """
    Load Favorita grocery sales forecasting dataset.

    Uses chunked reading to avoid loading the full 1.25GB train.csv into memory.

    Parameters
    ----------
    data_dir : str
        Directory containing train.csv, items.csv, stores.csv, holidays_events.csv, oil.csv
    n_items : int or None
        If provided, limit to N random item-store combinations.
    random_state : int
        Random seed for item sampling.

    Returns
    -------
    data : list of dicts
    """
    train_path = os.path.join(data_dir, "train.csv")
    items = pd.read_csv(os.path.join(data_dir, "items.csv"))
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    holidays = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"))
    oil = pd.read_csv(os.path.join(data_dir, "oil.csv"))

    holidays["date"] = pd.to_datetime(holidays["date"])
    oil["date"] = pd.to_datetime(oil["date"])

    # --- Step 1: Determine target item_nbr values via chunked scan ---
    if n_items is not None:
        rng = np.random.RandomState(random_state)
        all_items = set()
        chunk_iter = pd.read_csv(
            train_path,
            usecols=["item_nbr"],
            dtype={"item_nbr": "int32"},
            chunksize=1_000_000,
        )
        for chunk in chunk_iter:
            all_items.update(chunk["item_nbr"].unique().tolist())
        target_items = set(rng.choice(list(all_items), size=min(n_items, len(all_items)), replace=False))
        print(f"[Favorita] Sampling {len(target_items)} items from {len(all_items)} total.")
    else:
        target_items = None
        print("[Favorita] Loading ALL items (this may take a while).")

    # --- Step 2: Read chunks and filter for target items ---
    chunk_iter = pd.read_csv(
        train_path,
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        dtype={"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32", "onpromotion": "object"},
        chunksize=1_000_000,
    )

    filtered_chunks = []
    for chunk in chunk_iter:
        chunk["date"] = pd.to_datetime(chunk["date"])
        if target_items is not None:
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

    # Get unique item-store combinations
    item_store_pairs = train[["item_nbr", "store_nbr"]].drop_duplicates()
    if n_items is not None and len(item_store_pairs) > n_items:
        item_store_pairs = item_store_pairs.sample(n=n_items, random_state=random_state)

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

        # On-promotion flag = the actual promotional event channel (this is what CD deconvolves).
        onpromo_series = sub.set_index("date")["onpromotion"].reindex(dates, fill_value=0)
        if onpromo_series.dtype == object:
            onpromo_series = onpromo_series.map(lambda x: 1.0 if str(x).lower() in ('true', '1.0', '1') else 0.0)
        onpromo = onpromo_series.astype(float).values

        # Holiday calendar (secondary channel). Match on normalized dates -- comparing
        # pd.Timestamp against a set of numpy datetime64 is always False, which previously
        # left this channel all-zero. Transferred holidays are excluded (their listed date is
        # not the day off); the corresponding "Transfer" row marks the actual day off.
        _real_off = holidays[((holidays["type"].isin(["Holiday", "Additional", "Bridge", "Event"]))
                              & (~holidays.get("transferred", False)))
                             | (holidays["type"] == "Transfer")]
        holiday_dates = set(pd.to_datetime(_real_off["date"]).dt.normalize())
        norm_dates = pd.DatetimeIndex(dates).normalize()
        f_holiday = np.array([1.0 if d in holiday_dates else 0.0 for d in norm_dates])

        f_promo = onpromo            # promotional events = the deconvolution target
        promo_depth = onpromo        # depth proxy (binary on-promotion)

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
            "f_promo": f_promo,          # on-promotion events (live; has real carryover)
            "f_snap": f_holiday,         # corrected holiday calendar (second channel; was dead)
            "promo_depth": promo_depth,
            "oil_price": oil_aligned,
        })

    return _assert_channels_live(data, name="Favorita")
