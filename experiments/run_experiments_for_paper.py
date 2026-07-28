#!/usr/bin/env python3
"""
Run two experiments on M5 real data for the research paper.

Experiment 1: Two-Stage vs Joint Optimization (50 SKUs)
Experiment 2: Calendar Degradation (50 SKUs)

Usage:
    source /opt/homebrew/Caskroom/miniconda/base/bin/activate goi_clean
    KMP_DUPLICATE_LIB_OK=TRUE python run_experiments_for_paper.py
"""

import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed

# Add package path
sys.path.insert(0, _os.path.join(_ROOT, "code"))

from contextual_deconvolution.core import (
    build_Aspec, build_difference_operator, robust_detrend,
    estimate_kernel, fit_parametric_kernel, kernel_from_theta,
    decompose_ista, compute_decomposition_quality,
)
from contextual_deconvolution.metrics import wmape, variance_ratio, rmse
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import CONFIG_DEFAULT, evaluate_sku, forecast_two_stream


# ---------------------------------------------------------------------------
# Joint Optimization (block-coordinate descent)
# ---------------------------------------------------------------------------

def _build_phi_design(b, x_base, x_spec, f_promo, f_snap, phi_snap, n_lags):
    """Build design matrix for phi update."""
    T = len(b)
    r = b - x_base
    
    # Subtract snap contribution
    for k in range(len(phi_snap)):
        if k < T:
            idx = np.arange(T - k)
            r[idx + k] -= phi_snap[k] * f_snap[idx + k] * x_spec[idx]
    
    valid = f_promo > 0
    n_valid = np.sum(valid)
    if n_valid < n_lags * 2:
        return None, None
    
    X = np.zeros((n_valid, n_lags + 1))
    for k in range(n_lags + 1):
        if k < T:
            shift = np.zeros(T)
            shift[k:] = x_spec[:T-k]
            X[:, k] = shift[valid]
    
    y = r[valid] / np.maximum(f_promo[valid], 1e-6)
    return X, y


def joint_optimize_sku(sku_data: Dict, config: Dict, max_iter: int = 3) -> Dict:
    """Joint optimization via block-coordinate descent."""
    b = sku_data["demand"]
    f_promo = sku_data["f_promo"]
    f_snap = sku_data["f_snap"]
    T = len(b)
    n_lags = config["n_lags"]
    lambda_smooth = config["lambda_smooth"]
    lambda_sparse = config["lambda_sparse"]
    
    # Initialize with two-stage estimate
    phi_promo_emp, n_promo = estimate_kernel(b, f_promo, n_lags, config["pre_window"])
    theta_promo = fit_parametric_kernel(phi_promo_emp, family="surge_vacuum")
    if theta_promo is not None:
        phi_promo = kernel_from_theta(theta_promo, n_lags)
    else:
        phi_promo = np.zeros(n_lags + 1)
        phi_promo[0] = 1.0
    
    phi_snap = np.zeros(n_lags + 1)
    phi_snap[0] = 1.0
    
    # Block-coordinate descent (max 3 iterations for speed)
    for _ in range(max_iter):
        A_spec = build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=True)
        x_base, x_spec = decompose_ista(b, A_spec, lambda_smooth, lambda_sparse, n_iter=50)
        
        X, y = _build_phi_design(b, x_base, x_spec, f_promo, f_snap, phi_snap, n_lags)
        if X is None:
            break
        
        # Fast least squares with bounds
        phi_init = phi_promo.copy()
        phi_init[phi_init < 0] = 0.1
        bounds = [(1.0, 1.0)] + [(0.0, 2.0)] * n_lags
        
        result = optimize.minimize(
            lambda p: np.sum((y - X @ p) ** 2) + 0.1 * np.sum(np.diff(p) ** 2),
            phi_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50}
        )
        phi_new = result.x
        phi_new = np.maximum(phi_new, 0)
        if phi_new[0] > 0:
            phi_new = phi_new / phi_new[0]
        phi_new[0] = 1.0
        
        if np.max(np.abs(phi_new - phi_promo)) < 1e-4:
            break
        phi_promo = phi_new
    
    # Final decomposition
    A_spec = build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=True)
    x_base, x_spec = decompose_ista(b, A_spec, lambda_smooth, lambda_sparse, n_iter=100)
    decomp = compute_decomposition_quality(b, x_base, x_spec, A_spec)
    
    return {
        "x_base": x_base, "x_spec": x_spec,
        "phi_promo": phi_promo, "phi_snap": phi_snap,
        "decomp": decomp, "n_promo": n_promo,
    }


# ---------------------------------------------------------------------------
# Experiment 1: Two-Stage vs Joint Optimization
# ---------------------------------------------------------------------------

def _process_sku_exp1(args):
    """Process one SKU for Experiment 1."""
    sku_data, config = args
    sku_id = sku_data.get("item_id", sku_data.get("sku_id", "unknown"))
    
    result = {"sku_id": sku_id, "category": sku_data.get("category", "unknown")}
    
    # Two-Stage (using existing evaluate_sku for consistency)
    try:
        t0 = time.time()
        ts_eval = evaluate_sku(sku_data, config)
        ts_time = time.time() - t0
        
        if "CD" in ts_eval.get("forecasts", {}):
            fc = ts_eval["forecasts"]["CD"]
            result.update({
                "ts_wmape": fc["wmape"],
                "ts_variance_ratio": fc["variance_ratio"],
                "ts_rmse": fc["rmse"],
                "ts_sparsity": ts_eval["decomp"]["sparsity"],
                "ts_smoothness": ts_eval["decomp"]["smoothness"],
                "ts_in_sample_rmse": ts_eval["decomp"]["rmse"],
                "ts_n_promo": ts_eval["n_promo"],
                "ts_time": ts_time,
            })
        else:
            for k in ["ts_wmape", "ts_variance_ratio", "ts_rmse", "ts_sparsity", "ts_smoothness",
                      "ts_in_sample_rmse", "ts_n_promo", "ts_time"]:
                result[k] = np.nan
    except Exception as e:
        warnings.warn(f"Two-stage failed for {sku_id}: {e}")
        for k in ["ts_wmape", "ts_variance_ratio", "ts_rmse", "ts_sparsity", "ts_smoothness",
                  "ts_in_sample_rmse", "ts_n_promo", "ts_time"]:
            result[k] = np.nan
    
    # Joint Optimization
    try:
        t0 = time.time()
        jo = joint_optimize_sku(sku_data, config, max_iter=3)
        jo_time = time.time() - t0
        
        T = len(sku_data["demand"])
        split = T - config["forecast_horizon"]
        b_test = sku_data["demand"][split:]
        f_promo_test = sku_data["f_promo"][split:]
        f_snap_test = sku_data["f_snap"][split:]
        
        pred, _, _ = forecast_two_stream(
            jo["x_base"], jo["x_spec"], f_promo_test, f_snap_test,
            jo["phi_promo"], jo["phi_snap"]
        )
        
        A_full = build_Aspec(jo["phi_promo"], jo["phi_snap"],
                             sku_data["f_promo"], sku_data["f_snap"], T, causal=True)
        recon = jo["x_base"] + A_full @ jo["x_spec"]
        
        result.update({
            "jo_wmape": wmape(b_test, pred),
            "jo_variance_ratio": variance_ratio(pred, b_test),
            "jo_rmse": rmse(b_test, pred),
            "jo_sparsity": jo["decomp"]["sparsity"],
            "jo_smoothness": jo["decomp"]["smoothness"],
            "jo_in_sample_rmse": jo["decomp"]["rmse"],
            "jo_n_promo": jo["n_promo"],
            "jo_time": jo_time,
        })
    except Exception as e:
        warnings.warn(f"Joint optimization failed for {sku_id}: {e}")
        for k in ["jo_wmape", "jo_variance_ratio", "jo_rmse", "jo_sparsity", "jo_smoothness",
                  "jo_in_sample_rmse", "jo_n_promo", "jo_time"]:
            result[k] = np.nan
    
    return result


def run_experiment1(data_dir: str, n_skus: int = 50, seed: int = 42, n_jobs: int = -1) -> pd.DataFrame:
    """Run two-stage vs joint optimization on M5 SKUs."""
    print("=" * 70)
    print("EXPERIMENT 1: Two-Stage vs Joint Optimization on Real M5 Data")
    print("=" * 70)
    
    print(f"\nLoading M5 data ({n_skus} SKUs, seed={seed})...")
    dataset = load_m5_data(data_dir, n_skus=n_skus, random_state=seed)
    print(f"Loaded {len(dataset)} SKUs")
    
    config = CONFIG_DEFAULT.copy()
    config["run_baselines"] = False
    config["run_ablations"] = False
    config["use_ista"] = True
    
    print(f"\nRunning evaluation (serial, per-SKU ~0.2s)...")
    args = [(sku, config) for sku in dataset]
    results = []
    for arg in args:
        r = _process_sku_exp1(arg)
        results.append(r)
        if len(results) % 10 == 0:
            print(f"  Completed {len(results)}/{len(args)} SKUs")
    
    df = pd.DataFrame([r for r in results if r is not None])
    return df


# ---------------------------------------------------------------------------
# Experiment 2: Calendar Degradation
# ---------------------------------------------------------------------------

def _process_sku_exp2(args):
    """Process one SKU for Experiment 2."""
    sku_data, config, level, seed = args
    sku_id = sku_data.get("item_id", sku_data.get("sku_id", "unknown"))
    
    f_promo_orig = sku_data["f_promo"].copy()
    f_snap_orig = sku_data["f_snap"].copy()
    promo_days = np.where(f_promo_orig > 0)[0]
    snap_days = np.where(f_snap_orig > 0)[0]
    
    rng = np.random.RandomState(seed + hash(sku_id) % 1000 + int(level * 100))
    
    f_promo = f_promo_orig.copy()
    f_snap = f_snap_orig.copy()
    
    if level > 0:
        if len(promo_days) > 0:
            n_remove = int(len(promo_days) * level)
            remove_idx = rng.choice(len(promo_days), size=n_remove, replace=False)
            f_promo[promo_days[remove_idx]] = 0
        if len(snap_days) > 0:
            n_remove = int(len(snap_days) * level)
            remove_idx = rng.choice(len(snap_days), size=n_remove, replace=False)
            f_snap[snap_days[remove_idx]] = 0
    
    test_data = sku_data.copy()
    test_data["f_promo"] = f_promo
    test_data["f_snap"] = f_snap
    
    try:
        t0 = time.time()
        result = evaluate_sku(test_data, config)
        elapsed = time.time() - t0
        
        if "CD" in result.get("forecasts", {}):
            fc = result["forecasts"]["CD"]
            return {
                "sku_id": sku_id,
                "category": sku_data.get("category", "unknown"),
                "degradation_level": level,
                "wmape": fc["wmape"],
                "variance_ratio": fc["variance_ratio"],
                "rmse": fc["rmse"],
                "sparsity": result["decomp"]["sparsity"],
                "smoothness": result["decomp"]["smoothness"],
                "n_promo": result["n_promo"],
                "runtime": elapsed,
                "n_promo_days": int(np.sum(f_promo > 0)),
                "n_snap_days": int(np.sum(f_snap > 0)),
            }
    except Exception as e:
        warnings.warn(f"Failed for {sku_id} at level {level}: {e}")
    
    return {
        "sku_id": sku_id, "category": sku_data.get("category", "unknown"),
        "degradation_level": level,
        "wmape": np.nan, "variance_ratio": np.nan, "rmse": np.nan,
        "sparsity": np.nan, "smoothness": np.nan, "n_promo": np.nan,
        "runtime": np.nan, "n_promo_days": np.nan, "n_snap_days": np.nan,
    }


def run_experiment2(data_dir: str, n_skus: int = 50, seed: int = 42, n_jobs: int = -1) -> pd.DataFrame:
    """Test calendar degradation: 0%, 25%, 50%, 75%, 100% events removed."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Calendar Degradation on Real M5 Data")
    print("=" * 70)
    
    print(f"\nLoading M5 data ({n_skus} SKUs, seed={seed})...")
    dataset = load_m5_data(data_dir, n_skus=n_skus, random_state=seed)
    print(f"Loaded {len(dataset)} SKUs")
    
    config = CONFIG_DEFAULT.copy()
    config["run_baselines"] = False
    config["run_ablations"] = False
    config["use_ista"] = True
    
    degradation_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    
    args = []
    for sku in dataset:
        for level in degradation_levels:
            args.append((sku, config, level, seed))
    
    print(f"\nRunning {len(args)} total evaluations (serial, per-eval ~0.2s)...")
    results = []
    for arg in args:
        r = _process_sku_exp2(arg)
        results.append(r)
        if len(results) % 50 == 0:
            print(f"  Completed {len(results)}/{len(args)} evaluations")
    
    df = pd.DataFrame([r for r in results if r is not None])
    return df


# ---------------------------------------------------------------------------
# Summary & Save
# ---------------------------------------------------------------------------

def print_summary(df1: pd.DataFrame, df2: pd.DataFrame, out_dir: str):
    """Print compact summary and save results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Experiment 1
    print("\n--- Experiment 1: Two-Stage vs Joint Optimization ---")
    
    metrics = [
        ("wmape", "ts_wmape", "jo_wmape"),
        ("variance_ratio", "ts_variance_ratio", "jo_variance_ratio"),
        ("rmse", "ts_rmse", "jo_rmse"),
        ("in_sample_rmse", "ts_in_sample_rmse", "jo_in_sample_rmse"),
        ("sparsity", "ts_sparsity", "jo_sparsity"),
        ("smoothness", "ts_smoothness", "jo_smoothness"),
    ]
    
    for label, col_ts, col_jo in metrics:
        if col_ts in df1.columns and col_jo in df1.columns:
            m_ts = df1[col_ts].mean()
            m_jo = df1[col_jo].mean()
            diff = m_jo - m_ts
            diff_pct = (diff / abs(m_ts) * 100) if m_ts != 0 else 0
            print(f"  {label:20s}: Two-Stage={m_ts:.4f}, Joint={m_jo:.4f} ({diff:+.4f}, {diff_pct:+.1f}%)")
    
    if "ts_time" in df1.columns and "jo_time" in df1.columns:
        print(f"\n  Runtime:")
        print(f"    Two-Stage avg: {df1['ts_time'].mean():.2f}s ± {df1['ts_time'].std():.2f}s")
        print(f"    Joint avg:     {df1['jo_time'].mean():.2f}s ± {df1['jo_time'].std():.2f}s")
        print(f"    Slowdown:      {df1['jo_time'].mean() / df1['ts_time'].mean():.2f}x")
    
    # Experiment 2
    print("\n--- Experiment 2: Calendar Degradation ---")
    for level in sorted(df2["degradation_level"].unique()):
        subset = df2[df2["degradation_level"] == level]
        print(f"\n  Degradation {level:>4.0%} (n={len(subset)}):")
        for metric in ["wmape", "variance_ratio", "rmse", "sparsity"]:
            if metric in subset.columns:
                m = subset[metric].mean()
                s = subset[metric].std()
                print(f"    {metric:20s}: {m:.4f} ± {s:.4f}")
    
    # Save
    df1.to_csv(os.path.join(out_dir, "exp1_two_stage_vs_joint.csv"), index=False)
    df2.to_csv(os.path.join(out_dir, "exp2_calendar_degradation.csv"), index=False)
    
    summary1 = df1.groupby("category")[[c for c in df1.columns if c.startswith(("ts_", "jo_")) and c not in ("ts_time", "jo_time")]].mean()
    summary1.to_csv(os.path.join(out_dir, "exp1_summary_by_category.csv"))
    
    summary2 = df2.groupby("degradation_level")[["wmape", "variance_ratio", "rmse", "sparsity", "smoothness"]].mean()
    summary2.to_csv(os.path.join(out_dir, "exp2_summary_by_level.csv"))
    
    print(f"\nResults saved to {out_dir}:")
    print(f"  exp1_two_stage_vs_joint.csv")
    print(f"  exp2_calendar_degradation.csv")
    print(f"  exp1_summary_by_category.csv")
    print(f"  exp2_summary_by_level.csv")


if __name__ == "__main__":
    data_dir = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")
    out_dir = _os.path.join(_ROOT, "paper/figures")
    
    os.makedirs(out_dir, exist_ok=True)
    total_start = time.time()
    
    n_jobs = -1  # Use all cores
    
    df1 = run_experiment1(data_dir, n_skus=50, seed=42, n_jobs=n_jobs)
    df2 = run_experiment2(data_dir, n_skus=50, seed=42, n_jobs=n_jobs)
    
    print_summary(df1, df2, out_dir)
    
    total_time = time.time() - total_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
