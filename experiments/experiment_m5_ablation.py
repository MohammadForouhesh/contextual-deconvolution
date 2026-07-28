#!/usr/bin/env python3
"""
Component-by-Component Ablation on M5 Dataset

Runs 5 ablation configurations on a small subset of M5 SKUs (20):
  1. Baseline only           — Huber detrending, no operator
  2. Trend + calendar        — Fixed calendar regressors (Ridge), no kernel
  3. Trend + kernel (id)     — Identity operator for events
  4. Trend + kernel (DoE)    — Full method with estimated DoE kernel
  5. Full CD                 — Two-stream forecasting with baseline + operator

Reports wMAPE, Variance Ratio, and RMSE for each. Saves results to CSV.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.core import (
    robust_detrend,
    estimate_kernel,
    fit_parametric_kernel,
    kernel_from_theta,
    build_Aspec,
    decompose_ista,
)
from contextual_deconvolution.metrics import wmape, variance_ratio, rmse
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SKUS = 20
FORECAST_HORIZON = 28
N_LAGS = 14
ISTA_ITER = 100
DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "m5-data", "m5-forecasting-accuracy"
)

print(f"Loading M5 data from: {DATA_DIR}")
start = time.time()
dataset = load_m5_data(DATA_DIR, n_skus=N_SKUS, random_state=42)
print(f"Loaded {len(dataset)} SKUs in {time.time()-start:.1f}s.")

# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------


def ablation_baseline_only(b_train, b_test, f_promo_train, f_promo_test,
                           f_snap_train, f_snap_test, depth_train, depth_test):
    """1. Baseline only: Huber detrending, no operator."""
    b_full = np.concatenate([b_train, b_test])
    mu = robust_detrend(b_full, lambda_smooth=10.0)
    split = len(b_train)
    recent = mu[split - 30 : split]
    trend = (recent[-1] - recent[0]) / 30.0 if len(recent) > 1 else 0.0
    pred = mu[split - 1] + trend * np.arange(1, len(b_test) + 1)
    return pred


def ablation_trend_calendar(b_train, b_test, f_promo_train, f_promo_test,
                            f_snap_train, f_snap_test, depth_train, depth_test):
    """2. Trend + calendar: Ridge on Fourier seasonality + events."""
    T = len(b_train)
    t = np.arange(T)
    X = [t]
    for period in [7]:
        for i in range(1, 4):
            X.append(np.sin(2 * np.pi * i * t / period))
            X.append(np.cos(2 * np.pi * i * t / period))
    X = np.column_stack(X + [f_promo_train, f_snap_train, depth_train])

    model = Ridge(alpha=1.0)
    model.fit(X, b_train)

    t_fut = np.arange(T, T + len(b_test))
    X_fut = [t_fut]
    for period in [7]:
        for i in range(1, 4):
            X_fut.append(np.sin(2 * np.pi * i * t_fut / period))
            X_fut.append(np.cos(2 * np.pi * i * t_fut / period))
    X_fut = np.column_stack(X_fut + [f_promo_test, f_snap_test, depth_test])
    return model.predict(X_fut)


def ablation_trend_kernel_id(b_train, b_test, f_promo_train, f_promo_test,
                               f_snap_train, f_snap_test, depth_train, depth_test):
    """3. Trend + kernel (identity): Use identity operator for events."""
    T_full = len(b_train) + len(b_test)
    split = len(b_train)
    b_full = np.concatenate([b_train, b_test])
    f_promo_full = np.concatenate([f_promo_train, f_promo_test])
    f_snap_full = np.concatenate([f_snap_train, f_snap_test])

    A_id = np.eye(T_full)
    x_base, x_spec = decompose_ista(
        b_full, A_id, lambda_smooth=10.0, lambda_sparse=8.0, n_iter=ISTA_ITER
    )

    recent = x_base[split - 30 : split]
    trend = (recent[-1] - recent[0]) / 30.0 if len(recent) > 1 else 0.0
    base_forecast = x_base[split - 1] + trend * np.arange(1, len(b_test) + 1)

    shocks = x_spec[x_spec > 1e-3]
    avg_shock = np.median(shocks) if len(shocks) > 0 else np.std(x_base) * 0.3
    future_shocks = np.where((f_promo_test > 0) | (f_snap_test > 0), avg_shock, 0.0)
    return base_forecast + future_shocks


def ablation_trend_kernel_doe(b_train, b_test, f_promo_train, f_promo_test,
                                f_snap_train, f_snap_test, depth_train, depth_test):
    """4. Trend + kernel (DoE): Full method with estimated DoE kernel."""
    T_full = len(b_train) + len(b_test)
    split = len(b_train)
    b_full = np.concatenate([b_train, b_test])
    f_promo_full = np.concatenate([f_promo_train, f_promo_test])
    f_snap_full = np.concatenate([f_snap_train, f_snap_test])

    phi_promo_emp, _ = estimate_kernel(b_train, f_promo_train, n_lags=N_LAGS, pre_window=30)
    phi_snap_emp, _ = estimate_kernel(b_train, f_snap_train, n_lags=N_LAGS, pre_window=30)

    theta_promo = fit_parametric_kernel(phi_promo_emp, family="surge_vacuum")
    if theta_promo is not None:
        phi_promo = kernel_from_theta(theta_promo, N_LAGS)
    else:
        phi_promo = np.zeros(N_LAGS + 1)
        phi_promo[0] = 1.0

    phi_snap = np.zeros(N_LAGS + 1)
    phi_snap[0] = 1.0

    A_spec = build_Aspec(phi_promo, phi_snap, f_promo_full, f_snap_full, T_full, causal=True)
    x_base, x_spec = decompose_ista(
        b_full, A_spec, lambda_smooth=10.0, lambda_sparse=8.0, n_iter=ISTA_ITER
    )

    recent = x_base[split - 30 : split]
    trend = (recent[-1] - recent[0]) / 30.0 if len(recent) > 1 else 0.0
    base_forecast = x_base[split - 1] + trend * np.arange(1, len(b_test) + 1)

    shocks = x_spec[x_spec > 1e-3]
    avg_shock = np.median(shocks) if len(shocks) > 0 else np.std(x_base) * 0.3
    future_shocks = np.zeros(len(b_test))
    future_shocks[f_promo_test > 0] = avg_shock
    future_shocks[f_snap_test > 0] = avg_shock * 0.5

    A_spec_fut = build_Aspec(phi_promo, phi_snap, f_promo_test, f_snap_test, len(b_test), causal=True)
    return base_forecast + A_spec_fut @ future_shocks


def ablation_full_cd(b_train, b_test, f_promo_train, f_promo_test,
                     f_snap_train, f_snap_test, depth_train, depth_test):
    """5. Full CD: Two-stream forecasting with depth-aware shock model."""
    T_full = len(b_train) + len(b_test)
    split = len(b_train)
    b_full = np.concatenate([b_train, b_test])
    f_promo_full = np.concatenate([f_promo_train, f_promo_test])
    f_snap_full = np.concatenate([f_snap_train, f_snap_test])
    depth_full = np.concatenate([depth_train, depth_test])

    phi_promo_emp, _ = estimate_kernel(b_train, f_promo_train, n_lags=N_LAGS, pre_window=30)
    phi_snap_emp, _ = estimate_kernel(b_train, f_snap_train, n_lags=N_LAGS, pre_window=30)

    theta_promo = fit_parametric_kernel(phi_promo_emp, family="surge_vacuum")
    if theta_promo is not None:
        phi_promo = kernel_from_theta(theta_promo, N_LAGS)
    else:
        phi_promo = np.zeros(N_LAGS + 1)
        phi_promo[0] = 1.0

    phi_snap = np.zeros(N_LAGS + 1)
    phi_snap[0] = 1.0

    A_spec = build_Aspec(phi_promo, phi_snap, f_promo_full, f_snap_full, T_full, causal=True)
    x_base, x_spec = decompose_ista(
        b_full, A_spec, lambda_smooth=10.0, lambda_sparse=8.0, n_iter=ISTA_ITER
    )

    # Build a simple depth-shock model from training data
    event_starts = np.where((f_promo_train[1:] > 0) & (f_promo_train[:-1] == 0))[0] + 1
    depth_shock_pairs = []
    for start in event_starts:
        if start >= len(x_spec):
            continue
        shock = x_spec[start]
        if shock < 1e-3:
            continue
        d = depth_train[start] if start < len(depth_train) else 0.0
        depth_shock_pairs.append((d, shock))

    avg_shock = np.median(x_spec[x_spec > 1e-3]) if np.any(x_spec > 1e-3) else np.std(x_base) * 0.3

    recent = x_base[split - 30 : split]
    trend = (recent[-1] - recent[0]) / 30.0 if len(recent) > 1 else 0.0
    base_forecast = x_base[split - 1] + trend * np.arange(1, len(b_test) + 1)

    future_shocks = np.zeros(len(b_test))
    for t in range(len(b_test)):
        if f_promo_test[t] > 0:
            depth = depth_test[t] if t < len(depth_test) else 0.0
            if depth > 0 and len(depth_shock_pairs) > 3:
                # Simple scaling by average depth-shock ratio
                depths_arr = np.array([p[0] for p in depth_shock_pairs])
                shocks_arr = np.array([p[1] for p in depth_shock_pairs])
                valid = depths_arr > 0
                if np.any(valid):
                    avg_ratio = np.median(shocks_arr[valid] / depths_arr[valid])
                    future_shocks[t] = avg_ratio * depth
                else:
                    future_shocks[t] = avg_shock
            else:
                future_shocks[t] = avg_shock
        elif f_snap_test[t] > 0:
            future_shocks[t] = avg_shock * 0.5

    A_spec_fut = build_Aspec(phi_promo, phi_snap, f_promo_test, f_snap_test, len(b_test), causal=True)
    return base_forecast + A_spec_fut @ future_shocks


ABLATIONS = [
    ("Baseline only", ablation_baseline_only),
    ("Trend + calendar", ablation_trend_calendar),
    ("Trend + kernel (identity)", ablation_trend_kernel_id),
    ("Trend + kernel (DoE)", ablation_trend_kernel_doe),
    ("Full CD", ablation_full_cd),
]

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

all_results = []
successful_skus = 0

for i, sku_data in enumerate(dataset):
    b = sku_data["demand"]
    T = len(b)
    split = T - FORECAST_HORIZON
    if split <= 50:
        print(f"  SKU {i}: too short (T={T}), skipping")
        continue

    b_train, b_test = b[:split], b[split:]
    f_promo = sku_data["f_promo"]
    f_snap = sku_data["f_snap"]
    depth = sku_data.get("promo_depth", np.zeros(T))
    f_promo_train, f_promo_test = f_promo[:split], f_promo[split:]
    f_snap_train, f_snap_test = f_snap[:split], f_snap[split:]
    depth_train, depth_test = depth[:split], depth[split:]

    sku_id = sku_data.get("sku_id", 0)
    item_id = sku_data.get("item_id", "unknown")
    successful_skus += 1

    for name, func in ABLATIONS:
        t0 = time.time()
        try:
            pred = func(b_train, b_test, f_promo_train, f_promo_test,
                        f_snap_train, f_snap_test, depth_train, depth_test)
            if pred is None or len(pred) != len(b_test):
                continue
            w = wmape(b_test, pred)
            vr = variance_ratio(pred, b_test)
            r = rmse(b_test, pred)
            all_results.append({
                "sku_id": sku_id,
                "item_id": item_id,
                "method": name,
                "wmape": w,
                "variance_ratio": vr,
                "rmse": r,
            })
        except Exception as e:
            print(f"  SKU {sku_id} / {name} failed: {e}")
            continue
        print(f"  SKU {sku_id:3d} | {name:25s} | {time.time()-t0:.2f}s | wMAPE={w:.4f} RMSE={r:.2f}")

print(f"\nCompleted {successful_skus} SKUs, {len(all_results)} total runs.")

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------

df = pd.DataFrame(all_results)
if len(df) == 0:
    print("No results produced. Check data loading.")
    sys.exit(1)

summary = df.groupby("method").agg({
    "wmape": ["mean", "std", "median"],
    "variance_ratio": ["mean", "std", "median"],
    "rmse": ["mean", "std", "median"],
}).reset_index()

# Flatten multi-index columns
summary.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in summary.columns.values]

# Enforce method order
method_order = [a[0] for a in ABLATIONS]
summary["method"] = pd.Categorical(summary["method"], categories=method_order, ordered=True)
summary = summary.sort_values("method").reset_index(drop=True)

print("\n" + "=" * 70)
print(f"ABLATION STUDY SUMMARY (M5, {successful_skus} SKUs)")
print("=" * 70)
for _, row in summary.iterrows():
    print(f"{row['method']:30s} | wMAPE={row['wmape_mean']:.4f} ± {row['wmape_std']:.4f} "
          f"| VR={row['variance_ratio_mean']:.3f} ± {row['variance_ratio_std']:.3f} "
          f"| RMSE={row['rmse_mean']:.2f} ± {row['rmse_std']:.2f}")
print("=" * 70)

# Save CSV
out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "m5_ablation_results.csv")
summary.to_csv(csv_path, index=False)
print(f"\nSaved summary CSV to: {csv_path}")

# Also save per-SKU results
csv_per_sku_path = os.path.join(out_dir, "m5_ablation_per_sku.csv")
df.to_csv(csv_per_sku_path, index=False)
print(f"Saved per-SKU CSV to: {csv_per_sku_path}")
