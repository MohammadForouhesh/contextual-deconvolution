#!/usr/bin/env python3
"""
EDMD operational metrics evaluation for Favorita data.

Fits category-level and global EDMD operators, extracts impulse responses,
 evaluates via the same Stage-3 pipeline as M5, and computes:
  - wMAPE (forecast accuracy)
  - Variance Ratio (Bullwhip inflation)
  - Safety Stock (inventory cost)

Reuses analysis functions from analyze_koopman_spectral_favorita.py and
 evaluate_sku from run_edmd_ablation.py (adapted for Favorita features).
"""

import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, _os.path.join(_ROOT, "code"))
from contextual_deconvolution.loaders import load_favorita_data
from contextual_deconvolution.core import (
    build_Aspec, decompose_ista, compute_decomposition_quality,
    robust_detrend,
)
from contextual_deconvolution.metrics import variance_ratio, wmape, inventory_cost
from contextual_deconvolution.experiments import CONFIG_DEFAULT

from run_edmd_ablation import build_snapshots, polynomial_features, edmd_with_inputs, extract_impulse_response
from analyze_koopman_spectral_favorita import run_edmd_on_favorita_pooled

warnings.filterwarnings('ignore')

N_LAGS = 14


def evaluate_sku_favorita(sku_data, phi, config):
    """
    Evaluate a single Favorita item using EDMD impulse response as the kernel.
    
    Favorita uses 'promo_depth' as the only event feature (no SNAP).
    """
    b = sku_data["demand"]
    f_promo = sku_data.get("promo_depth", np.zeros(len(b)))
    # No SNAP in Favorita; pass zeros
    f_snap = np.zeros(len(b))
    
    T = len(b)
    split = T - config["forecast_horizon"]
    b_test = b[split:]
    f_promo_test = f_promo[split:]
    f_snap_test = f_snap[split:]
    
    # Build operator using EDMD impulse response as the kernel
    # phi_snap is identity (instantaneous) since no SNAP
    phi_snap = np.zeros(N_LAGS + 1)
    phi_snap[0] = 1.0
    
    A_spec = build_Aspec(phi, phi_snap, f_promo, f_snap, T, causal=True)
    x_base, x_spec = decompose_ista(b, A_spec, config["lambda_smooth"], config["lambda_sparse"])
    
    T_fut = len(f_promo_test)
    recent = x_base[-30:]
    trend = (recent[-1] - recent[0]) / 30.0 if len(recent) > 1 else 0.0
    base_forecast = x_base[-1] + trend * np.arange(1, T_fut + 1)
    
    hist_shocks = x_spec[x_spec > 1e-3]
    avg_shock = np.median(hist_shocks) if len(hist_shocks) > 0 else np.std(x_base) * 0.5
    
    future_shocks = np.zeros(T_fut)
    for t in range(T_fut):
        if f_promo_test[t] > 0 or f_snap_test[t] > 0:
            future_shocks[t] = avg_shock
    
    A_spec_fut = build_Aspec(phi, phi_snap, f_promo_test, f_snap_test, T_fut)
    forecast = base_forecast + A_spec_fut @ future_shocks
    
    # Compute metrics
    vr = variance_ratio(forecast, b_test)
    wm = wmape(b_test, forecast)
    decomp = compute_decomposition_quality(b, x_base, x_spec, A_spec)
    
    # Safety stock from inventory simulation
    _, safety_stock = inventory_cost(
        actual=b_test,
        forecast=forecast,
        fitted_train=x_base,
        lead_time=config.get("lead_time", 7),
        h=config.get("holding_cost", 1.0),
        p=config.get("stockout_cost", 10.0),
    )
    
    return {
        "vr": vr,
        "wmape": wm,
        "sparsity": decomp["sparsity"],
        "rmse": decomp["rmse"],
        "safety_stock": safety_stock,
    }


def run_parametric_kernel_favorita(sku_data):
    """Fit per-SKU parametric surge-and-vacuum kernel for Favorita."""
    from contextual_deconvolution.core import estimate_kernel, fit_parametric_kernel, kernel_from_theta
    b = sku_data["demand"]
    f_promo = sku_data.get("promo_depth", np.zeros(len(b)))
    T = len(b)
    split = T - 28
    b_train = b[:split]
    f_promo_train = f_promo[:split]
    phi_promo_emp, _ = estimate_kernel(b_train, f_promo_train, N_LAGS, 30)
    theta_promo = fit_parametric_kernel(phi_promo_emp, family="surge_vacuum")
    if theta_promo is not None:
        phi_promo = kernel_from_theta(theta_promo, N_LAGS)
    else:
        phi_promo = np.zeros(N_LAGS + 1)
        phi_promo[0] = 1.0
    return phi_promo


def main():
    favorita_dir = _os.path.join(_ROOT, "data/favorita-grocery-sales-forecasting")
    out_dir = _os.path.join(_ROOT, "paper/results")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading Favorita data (500 items)...")
    data = load_favorita_data(favorita_dir, n_items=500, random_state=42)
    print(f"Loaded {len(data)} items")
    
    categories = {}
    for sku in data:
        cat = sku.get("category", "unknown")
        categories.setdefault(cat, []).append(sku)
    print(f"Categories: { {k: len(v) for k, v in categories.items()} }")
    
    config = CONFIG_DEFAULT.copy()
    
    # --- Fit per-SKU parametric kernels (for comparison) ---
    print("\nFitting per-SKU parametric kernels...")
    parametric_kernels = []
    for sku in data:
        parametric_kernels.append(run_parametric_kernel_favorita(sku))
    
    # --- Fit EDMD at category level ---
    print("\nFitting category-level EDMD on Favorita...")
    edmd_category_results = {}
    for cat, sku_list in categories.items():
        print(f"  Category '{cat}': {len(sku_list)} items")
        res = run_edmd_on_favorita_pooled(sku_list, normalize=True)
        edmd_category_results[cat] = res
        if res and "error" not in res:
            print(f"    Stable: {res['stable']}, rho={res['spectral_radius']:.4f}, "
                  f"samples={res['n_samples']}, SKUs={res['n_skus']}")
    
    # --- Fit EDMD at global level ---
    print("\nFitting global-level EDMD on Favorita...")
    edmd_global_result = run_edmd_on_favorita_pooled(data, normalize=True)
    if edmd_global_result and "error" not in edmd_global_result:
        print(f"  Global: Stable={edmd_global_result['stable']}, "
              f"rho={edmd_global_result['spectral_radius']:.4f}, "
              f"samples={edmd_global_result['n_samples']}")
    
    # --- Extract valid IRs ---
    edmd_cat_ir = {}
    for cat, res in edmd_category_results.items():
        if res and "error" not in res and res.get("stable", False):
            edmd_cat_ir[cat] = res["h"]
    edmd_global_ir = edmd_global_result["h"] if (edmd_global_result and "error" not in edmd_global_result and edmd_global_result.get("stable", False)) else None
    
    # --- Stage 3 evaluation on all items ---
    print(f"\nRunning Stage 3 evaluation on {len(data)} items...")
    metrics = {"parametric": [], "edmd_cat": [], "edmd_global": []}
    t0 = time.time()
    
    for i, sku in enumerate(data):
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Item {i}/{len(data)}  (elapsed: {elapsed:.1f}s)")
        
        metrics["parametric"].append(evaluate_sku_favorita(sku, parametric_kernels[i], config))
        
        cat = sku.get("category", "unknown")
        if cat in edmd_cat_ir:
            metrics["edmd_cat"].append(evaluate_sku_favorita(sku, edmd_cat_ir[cat], config))
        else:
            metrics["edmd_cat"].append({"vr": np.nan, "wmape": np.nan, "sparsity": np.nan, "rmse": np.nan, "safety_stock": np.nan})
        
        if edmd_global_ir is not None:
            metrics["edmd_global"].append(evaluate_sku_favorita(sku, edmd_global_ir, config))
        else:
            metrics["edmd_global"].append({"vr": np.nan, "wmape": np.nan, "sparsity": np.nan, "rmse": np.nan, "safety_stock": np.nan})
    
    total_time = time.time() - t0
    print(f"  Total evaluation time: {total_time:.1f}s ({total_time/len(data):.3f}s per item)")
    
    # --- Summarize ---
    print("\n" + "="*60)
    print("FAVORITA EDMD OPERATIONAL METRICS SUMMARY")
    print("="*60)
    
    for method in ["parametric", "edmd_cat", "edmd_global"]:
        vals = [v for v in metrics[method] if not np.isnan(v.get("vr", np.nan))]
        if not vals:
            continue
        vr_vals = [v['vr'] for v in vals]
        wm_vals = [v['wmape'] for v in vals]
        sp_vals = [v['sparsity'] for v in vals]
        ss_vals = [v['safety_stock'] for v in vals if not np.isnan(v.get('safety_stock', np.nan))]
        
        print(f"\n{method:12s}:")
        print(f"  VR    median={np.median(vr_vals):.4f},  mean={np.mean(vr_vals):.4f},  std={np.std(vr_vals):.4f}")
        print(f"  wMAPE median={np.median(wm_vals):.3f},  mean={np.mean(wm_vals):.3f},  std={np.std(wm_vals):.3f}")
        print(f"  Spars median={np.median(sp_vals):.1f},   mean={np.mean(sp_vals):.1f},   std={np.std(sp_vals):.1f}")
        print(f"  Safety median={np.median(ss_vals):.3f},  mean={np.mean(ss_vals):.3f},  std={np.std(ss_vals):.3f}")
    
    # Wilcoxon tests
    from scipy.stats import wilcoxon
    param_vr = [v["vr"] for v in metrics["parametric"]]
    edmd_cat_vr = [v["vr"] for v in metrics["edmd_cat"] if not np.isnan(v["vr"])]
    edmd_glob_vr = [v["vr"] for v in metrics["edmd_global"] if not np.isnan(v["vr"])]
    
    p_cat = 1.0
    p_glob = 1.0
    if len(edmd_cat_vr) == len(param_vr):
        try:
            _, p_cat = wilcoxon(param_vr, edmd_cat_vr)
        except:
            pass
    if len(edmd_glob_vr) == len(param_vr):
        try:
            _, p_glob = wilcoxon(param_vr, edmd_glob_vr)
        except:
            pass
    print(f"\nWilcoxon p (parametric vs cat-EDMD):  {p_cat:.3f}")
    print(f"Wilcoxon p (parametric vs glob-EDMD): {p_glob:.3f}")
    
    # Save results
    results_df = pd.DataFrame({
        "sku_id": [d.get("sku_id", i) for i, d in enumerate(data)],
        "category": [d.get("category", "unknown") for d in data],
        "param_vr": [m["vr"] for m in metrics["parametric"]],
        "param_wmape": [m["wmape"] for m in metrics["parametric"]],
        "param_sparsity": [m["sparsity"] for m in metrics["parametric"]],
        "param_safety_stock": [m["safety_stock"] for m in metrics["parametric"]],
        "edmd_cat_vr": [m["vr"] for m in metrics["edmd_cat"]],
        "edmd_cat_wmape": [m["wmape"] for m in metrics["edmd_cat"]],
        "edmd_cat_sparsity": [m["sparsity"] for m in metrics["edmd_cat"]],
        "edmd_cat_safety_stock": [m["safety_stock"] for m in metrics["edmd_cat"]],
        "edmd_global_vr": [m["vr"] for m in metrics["edmd_global"]],
        "edmd_global_wmape": [m["wmape"] for m in metrics["edmd_global"]],
        "edmd_global_sparsity": [m["sparsity"] for m in metrics["edmd_global"]],
        "edmd_global_safety_stock": [m["safety_stock"] for m in metrics["edmd_global"]],
    })
    out_csv = os.path.join(out_dir, "edmd_favorita_operational_metrics.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")
    
    # Print final paper summary
    print("\n" + "="*60)
    print("PAPER SUMMARY (Favorita EDMD)")
    print("="*60)
    param_vr_med = np.median([v["vr"] for v in metrics["parametric"]])
    param_wm_med = np.median([v["wmape"] for v in metrics["parametric"]])
    param_ss_med = np.median([v["safety_stock"] for v in metrics["parametric"] if not np.isnan(v.get("safety_stock", np.nan))])
    edmd_cat_vr_med = np.median([v["vr"] for v in metrics["edmd_cat"] if not np.isnan(v["vr"])])
    edmd_cat_wm_med = np.median([v["wmape"] for v in metrics["edmd_cat"] if not np.isnan(v["wmape"])])
    edmd_cat_ss_med = np.median([v["safety_stock"] for v in metrics["edmd_cat"] if not np.isnan(v.get("safety_stock", np.nan))])
    edmd_glob_vr_med = np.median([v["vr"] for v in metrics["edmd_global"] if not np.isnan(v["vr"])])
    edmd_glob_wm_med = np.median([v["wmape"] for v in metrics["edmd_global"] if not np.isnan(v["wmape"])])
    edmd_glob_ss_med = np.median([v["safety_stock"] for v in metrics["edmd_global"] if not np.isnan(v.get("safety_stock", np.nan))])
    
    print(f"\nParametric:      VR={param_vr_med:.4f},  wMAPE={param_wm_med:.3f},  SafetyStock={param_ss_med:.3f}")
    print(f"Category-EDMD:   VR={edmd_cat_vr_med:.4f},  wMAPE={edmd_cat_wm_med:.3f},  SafetyStock={edmd_cat_ss_med:.3f}")
    print(f"Global-EDMD:     VR={edmd_glob_vr_med:.4f},  wMAPE={edmd_glob_wm_med:.3f},  SafetyStock={edmd_glob_ss_med:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
