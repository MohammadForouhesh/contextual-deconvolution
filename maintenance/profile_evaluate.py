#!/usr/bin/env python3
"""Profile evaluate_sku to find the bottleneck."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import CONFIG_DEFAULT
from contextual_deconvolution.core import (
    estimate_kernel, fit_parametric_kernel, kernel_from_theta,
    build_Aspec, decompose_ista, decompose_cvxpy, compute_decomposition_quality,
)
import numpy as np

print("Loading M5 data (1 SKU)...", flush=True)
dataset = load_m5_data("data/m5-data/m5-forecasting-accuracy", n_skus=1)
d = dataset[0]
print(f"SKU: {d['item_id']}, T={len(d['demand'])}", flush=True)

b = d["demand"]
f_promo = d["f_promo"]
f_snap = d["f_snap"]
T = len(b)

horizon = 28
split = T - horizon
b_train = b[:split]
f_promo_train = f_promo[:split]
f_snap_train = f_snap[:split]

config = CONFIG_DEFAULT.copy()
config["use_ista"] = True  # CRITICAL: use ISTA for speed

t0 = time.time()
print("\n--- Stage 1: Kernel estimation ---", flush=True)
t1 = time.time()
phi_promo_emp, n_promo = estimate_kernel(b_train, f_promo_train, config["n_lags"], config["pre_window"])
print(f"  estimate_kernel(promo): {time.time()-t1:.2f}s, n_promo={n_promo}", flush=True)

t1 = time.time()
phi_snap_emp, n_snap = estimate_kernel(b_train, f_snap_train, config["n_lags"], config["pre_window"])
print(f"  estimate_kernel(snap): {time.time()-t1:.2f}s, n_snap={n_snap}", flush=True)

t1 = time.time()
theta_promo = fit_parametric_kernel(phi_promo_emp, family="surge_vacuum")
print(f"  fit_parametric_kernel: {time.time()-t1:.2f}s", flush=True)

t1 = time.time()
if theta_promo is not None:
    phi_promo = kernel_from_theta(theta_promo, config["n_lags"])
else:
    phi_promo = np.zeros(config["n_lags"] + 1)
    phi_promo[0] = 1.0
print(f"  kernel_from_theta: {time.time()-t1:.2f}s", flush=True)

phi_snap = np.zeros(config["n_lags"] + 1)
phi_snap[0] = 1.0

print("\n--- Stage 2: Decomposition ---", flush=True)
t1 = time.time()
A_spec = build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=True)
print(f"  build_Aspec: {time.time()-t1:.2f}s, type={type(A_spec)}, nnz={A_spec.nnz}", flush=True)

t1 = time.time()
if config.get("use_ista", False):
    x_base, x_spec = decompose_ista(b, A_spec, config["lambda_smooth"], config["lambda_sparse"])
    print(f"  decompose_ista: {time.time()-t1:.2f}s", flush=True)
else:
    x_base, x_spec, obj_val = decompose_cvxpy(b, A_spec, config["lambda_smooth"], config["lambda_sparse"])
    print(f"  decompose_cvxpy: {time.time()-t1:.2f}s", flush=True)

t1 = time.time()
decomp = compute_decomposition_quality(b, x_base, x_spec, A_spec)
print(f"  compute_decomposition_quality: {time.time()-t1:.2f}s", flush=True)

print(f"\nTotal so far: {time.time()-t0:.2f}s", flush=True)
print(f"Decomp: {decomp}", flush=True)
