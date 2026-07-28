#!/usr/bin/env python3
"""
Run comprehensive evaluation including new baselines (Croston, SBA, LightGBM)
and new metrics (MASE, RMSSE, Order Variance) on M5 data.

Usage:
    python3 code/run_comprehensive_baselines.py
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
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

sys.path.insert(0, _os.path.join(_ROOT, "code"))

from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.metrics import (
    wmape, variance_ratio, rmse, mase, rmsse, order_variance_under_replenishment
)
from contextual_deconvolution.baselines import (
    run_tuned_xgboost, run_xgboost, run_prophet_like, run_stl_arima,
    run_croston, run_sba, run_lightgbm
)
from contextual_deconvolution.core import (
    build_Aspec, estimate_kernel, fit_parametric_kernel, kernel_from_theta,
    decompose_ista, compute_decomposition_quality
)
from contextual_deconvolution.experiments import CONFIG_DEFAULT, evaluate_sku


def evaluate_cd_on_sku(sku_data: Dict, config: Dict) -> Dict:
    """Evaluate CD on a single SKU and return all metrics."""
    sku_id = sku_data.get("sku_id", 0)
    category = sku_data.get("category", "unknown")
    b = sku_data["demand"]
    f_promo = sku_data["f_promo"]
    f_snap = sku_data["f_snap"]
    T = len(b)
    split = T - config["forecast_horizon"]
    b_test = b[split:]
    f_promo_test = f_promo[split:T]
    f_snap_test = f_snap[split:T]
    
    t0 = time.time()
    try:
        result = evaluate_sku(sku_data, config)
        elapsed = time.time() - t0
        
        if "CD" in result.get("forecasts", {}):
            fc = result["forecasts"]["CD"]
            pred = fc["pred"]  # Key is 'pred', not 'forecast'
            
            # Reconstruct fitted values from decomposition result
            x_base = result.get("x_base")
            x_spec = result.get("x_spec")
            phi_promo = result.get("phi_promo")
            phi_snap = result.get("phi_snap")
            if x_base is not None and x_spec is not None and phi_promo is not None:
                A_spec = build_Aspec(phi_promo, phi_snap, f_promo[:split], f_snap[:split], split, causal=True)
                fitted_train = x_base + A_spec @ x_spec
            else:
                fitted_train = None
            
            # Compute all metrics
            wmape_val = fc["wmape"]
            vr_val = fc["variance_ratio"]
            rmse_val = fc["rmse"]
            mase_val = mase(b_test, pred)
            rmsse_val = rmsse(b_test, pred)
            order_var = order_variance_under_replenishment(
                pred, b_test, fitted_train=fitted_train
            )
            
            return {
                "sku_id": sku_id,
                "category": category,
                "method": "CD",
                "wmape": wmape_val,
                "variance_ratio": vr_val,
                "rmse": rmse_val,
                "mase": mase_val,
                "rmsse": rmsse_val,
                "order_variance": order_var,
                "runtime": elapsed,
            }
    except Exception as e:
        warnings.warn(f"CD failed for SKU {sku_id}: {e}")
    
    return {
        "sku_id": sku_id, "category": category, "method": "CD",
        "wmape": np.nan, "variance_ratio": np.nan, "rmse": np.nan,
        "mase": np.nan, "rmsse": np.nan, "order_variance": np.nan,
        "runtime": np.nan,
    }


def evaluate_baseline_on_sku(sku_data: Dict, method_name: str, config: Dict) -> Dict:
    """Evaluate a single baseline on a single SKU."""
    sku_id = sku_data.get("sku_id", 0)
    category = sku_data.get("category", "unknown")
    b = sku_data["demand"]
    f_promo = sku_data["f_promo"]
    f_snap = sku_data["f_snap"]
    T = len(b)
    split = T - config["forecast_horizon"]
    b_test = b[split:]
    
    method_map = {
        "Tuned_XGBoost": run_tuned_xgboost,
        "Vanilla_XGBoost": run_xgboost,
        "Prophet-like": run_prophet_like,
        "STL+ARIMA": run_stl_arima,
        "Croston": run_croston,
        "SBA": run_sba,
        "LightGBM": run_lightgbm,
    }
    
    func = method_map.get(method_name)
    if func is None:
        return {
            "sku_id": sku_id, "category": category, "method": method_name,
            "wmape": np.nan, "variance_ratio": np.nan, "rmse": np.nan,
            "mase": np.nan, "rmsse": np.nan, "order_variance": np.nan,
            "runtime": np.nan,
        }
    
    t0 = time.time()
    try:
        pred, actual, fitted_train, model = func(b, f_promo, f_snap, forecast_horizon=config["forecast_horizon"])
        elapsed = time.time() - t0
        
        if pred is None or actual is None or len(pred) != len(actual):
            return {
                "sku_id": sku_id, "category": category, "method": method_name,
                "wmape": np.nan, "variance_ratio": np.nan, "rmse": np.nan,
                "mase": np.nan, "rmsse": np.nan, "order_variance": np.nan,
                "runtime": elapsed,
            }
        
        wmape_val = wmape(actual, pred)
        vr_val = variance_ratio(pred, actual)
        rmse_val = rmse(actual, pred)
        mase_val = mase(actual, pred)
        rmsse_val = rmsse(actual, pred)
        order_var = order_variance_under_replenishment(pred, actual, fitted_train=fitted_train)
        
        return {
            "sku_id": sku_id,
            "category": category,
            "method": method_name,
            "wmape": wmape_val,
            "variance_ratio": vr_val,
            "rmse": rmse_val,
            "mase": mase_val,
            "rmsse": rmsse_val,
            "order_variance": order_var,
            "runtime": elapsed,
        }
    except Exception as e:
        warnings.warn(f"{method_name} failed for SKU {sku_id}: {e}")
        return {
            "sku_id": sku_id, "category": category, "method": method_name,
            "wmape": np.nan, "variance_ratio": np.nan, "rmse": np.nan,
            "mase": np.nan, "rmsse": np.nan, "order_variance": np.nan,
            "runtime": np.nan,
        }


def run_comprehensive_evaluation(dataset: List[Dict], dataset_name: str, n_skus: int = None) -> pd.DataFrame:
    """Run all methods including new baselines and metrics."""
    print(f"\n{'='*70}")
    print(f"Comprehensive Evaluation: {dataset_name}")
    print(f"{'='*70}")
    
    if n_skus:
        dataset = dataset[:n_skus]
    
    config = CONFIG_DEFAULT.copy()
    config["run_baselines"] = False
    config["run_ablations"] = False
    config["use_ista"] = True
    
    methods = [
        "CD",
        "Tuned_XGBoost",
        "Vanilla_XGBoost",
        "Prophet-like",
        "STL+ARIMA",
        "Croston",
        "SBA",
        "LightGBM",
    ]
    
    all_results = []
    
    for method in methods:
        print(f"\n  --- {method} ---")
        method_results = []
        for i, sku_data in enumerate(dataset):
            if method == "CD":
                r = evaluate_cd_on_sku(sku_data, config)
            else:
                r = evaluate_baseline_on_sku(sku_data, method, config)
            method_results.append(r)
            if (i + 1) % 50 == 0 or i == len(dataset) - 1:
                print(f"    Processed {i+1}/{len(dataset)} items")
        all_results.extend(method_results)
    
    df = pd.DataFrame(all_results)
    return df


def print_comprehensive_summary(df: pd.DataFrame, dataset_name: str):
    """Print summary with all metrics including new ones."""
    print(f"\n{'='*70}")
    print(f"Summary: {dataset_name}")
    print(f"{'='*70}")
    
    print(f"\n{'Method':<18} {'n':>6} {'wMAPE':>10} {'MASE':>10} {'RMSSE':>10} {'VR':>10} {'OrdVar':>10} {'RMSE':>10}")
    print("-" * 90)
    
    for method in df["method"].unique():
        sub = df[df["method"] == method].dropna(subset=["wmape"])
        if len(sub) == 0:
            print(f"{method:<18} {'--':>6} {'--':>10} {'--':>10} {'--':>10} {'--':>10} {'--':>10} {'--':>10}")
            continue
        
        def fmt_mean_std(col):
            vals = sub[col].dropna()
            if len(vals) == 0:
                return "--"
            return f"{vals.mean():.4f}±{vals.std():.4f}"
        
        def fmt_median(col):
            vals = sub[col].dropna()
            if len(vals) == 0:
                return "--"
            return f"{vals.median():.4f}"
        
        wmape_str = fmt_mean_std("wmape")
        mase_str = fmt_mean_std("mase")
        rmsse_str = fmt_mean_std("rmsse")
        vr_str = fmt_mean_std("variance_ratio")
        ordvar_str = fmt_mean_std("order_variance")
        rmse_str = fmt_mean_std("rmse")
        
        print(f"{method:<18} {len(sub):>6} {wmape_str:>10} {mase_str:>10} {rmsse_str:>10} {vr_str:>10} {ordvar_str:>10} {rmse_str:>10}")
    
    # Also print median table for metrics where median is more appropriate
    print(f"\n{'Method':<18} {'n':>6} {'wMAPE_med':>10} {'MASE_med':>10} {'RMSSE_med':>10} {'VR_med':>10} {'OrdVar_med':>10}")
    print("-" * 80)
    for method in df["method"].unique():
        sub = df[df["method"] == method].dropna(subset=["wmape"])
        if len(sub) == 0:
            continue
        print(f"{method:<18} {len(sub):>6} {sub['wmape'].median():>10.4f} {sub['mase'].median():>10.4f} {sub['rmsse'].median():>10.4f} {sub['variance_ratio'].median():>10.4f} {sub['order_variance'].median():>10.4f}")


if __name__ == "__main__":
    m5_dir = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")
    out_dir = _os.path.join(_ROOT, "code/results")
    os.makedirs(out_dir, exist_ok=True)
    
    total_start = time.time()
    
    # --- M5: 200 SKUs ---
    print("\nLoading M5 data (200 SKUs)...")
    m5_data = load_m5_data(m5_dir, n_skus=200, random_state=42)
    print(f"Loaded {len(m5_data)} SKUs")
    
    m5_results = run_comprehensive_evaluation(m5_data, "M5_200", n_skus=200)
    m5_out = os.path.join(out_dir, "m5_200_comprehensive_baselines.csv")
    m5_results.to_csv(m5_out, index=False)
    print(f"\nSaved M5 results to {m5_out}")
    print_comprehensive_summary(m5_results, "M5_200")
    
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"TOTAL RUNTIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")
