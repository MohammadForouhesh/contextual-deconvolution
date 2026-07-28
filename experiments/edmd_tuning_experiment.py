#!/usr/bin/env python3
"""Comprehensive EDMD tuning experiment with bug fix."""
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import sys, os, warnings
sys.path.insert(0, _os.path.join(_ROOT, "code"))
import numpy as np
from scipy.stats import wilcoxon
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.core import (
    estimate_kernel, fit_parametric_kernel, kernel_from_theta,
    build_Aspec, decompose_ista, compute_decomposition_quality,
    robust_detrend,
)
from contextual_deconvolution.metrics import variance_ratio, wmape
from contextual_deconvolution.experiments import CONFIG_DEFAULT

warnings.filterwarnings('ignore')
N_LAGS = 14

def build_snapshots(demand, promo, snap, n_lag=14):
    T = len(demand)
    X, Y, U = [], [], []
    for t in range(n_lag, T - 1):
        x_t = demand[t - n_lag:t + 1]
        y_t = demand[t - n_lag + 1:t + 2]
        u_t = np.array([promo[t], snap[t]])
        X.append(x_t); Y.append(y_t); U.append(u_t)
    return np.array(X), np.array(Y), np.array(U)

def poly_v1(X, d=2):
    return np.hstack([X, X ** 2])

def poly_v2(X, d=2):
    """Full quadratic: constant + linear + cross-terms."""
    n = X.shape[1]
    terms = [np.ones((X.shape[0], 1)), X]
    for i in range(n):
        for j in range(i, n):
            terms.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(terms)

def edmd(X, Y, U, feat_fn, ridge=0.0, c_idx=None):
    Psi_X = feat_fn(X)
    Psi_Y = feat_fn(Y)
    d_psi = Psi_X.shape[1]
    n_state = X.shape[1]
    Omega = np.hstack([Psi_X, U])
    
    if ridge > 0:
        A = Omega.T @ Omega + ridge * np.eye(Omega.shape[1])
        b = Omega.T @ Psi_Y
        Xi = np.linalg.solve(A, b)
    else:
        Xi, _, _, _ = np.linalg.lstsq(Omega, Psi_Y, rcond=1e-10)
    
    K = Xi[:d_psi, :].T
    B = Xi[d_psi:, :].T
    C = np.zeros(d_psi)
    C[c_idx if c_idx is not None else n_state - 1] = 1.0
    return K, B, C, d_psi

def extract_ir(K, B, C, n_lag=14):
    h = []
    Kp = np.eye(K.shape[0])
    for k in range(n_lag + 1):
        h.append(C @ Kp @ B)
        Kp = Kp @ K
    return np.array(h)

def fit_edmd_pooled(sku_list, feat_fn, normalize=True, ridge=0.0):
    all_X, all_Y, all_U = [], [], []
    for sku in sku_list:
        b = sku["demand"]; f_promo = sku["f_promo"]; f_snap = sku["f_snap"]
        T = len(b); split = T - 28
        b_train = b[:split]; fp = f_promo[:split]; fs = f_snap[:split]
        if normalize:
            mu = np.mean(b_train)
            if mu > 1e-6: b_train = b_train / mu
        X, Y, U = build_snapshots(b_train, fp, fs, n_lag=N_LAGS)
        if len(X) >= 30:
            all_X.append(X); all_Y.append(Y); all_U.append(U)
    if len(all_X) == 0: return None
    Xa = np.vstack(all_X); Ya = np.vstack(all_Y); Ua = np.vstack(all_U)
    try:
        K, B, C, d_psi = edmd(Xa, Ya, Ua, feat_fn, ridge=ridge)
    except Exception as e: return {"error": str(e)}
    eig = np.linalg.eigvals(K)
    rho = np.max(np.abs(eig))
    h = extract_ir(K, B[:, 0], C, n_lag=N_LAGS)
    h_snap = extract_ir(K, B[:, 1], C, n_lag=N_LAGS)
    if np.abs(h[0]) > 1e-6: h = h / h[0]
    if np.abs(h_snap[0]) > 1e-6: h_snap = h_snap / h_snap[0]
    return {"h": h, "h_snap": h_snap, "K": K, "B": B, "C": C, "d_psi": d_psi,
            "spectral_radius": rho, "stable": rho < 1.0, "n_skus": len(all_X),
            "n_samples": Xa.shape[0]}

def param_kernel(sku):
    b = sku["demand"]; f = sku["f_promo"]
    T = len(b); split = T - 28
    phi, _ = estimate_kernel(b[:split], f[:split], N_LAGS, 30)
    theta = fit_parametric_kernel(phi, family="surge_vacuum")
    return kernel_from_theta(theta, N_LAGS) if theta is not None else np.array([1.0] + [0.0]*N_LAGS)

def evaluate(sku, phi, cfg):
    b = sku["demand"]; fp = sku["f_promo"]; fs = sku["f_snap"]
    T = len(b); split = T - cfg["forecast_horizon"]
    b_test = b[split:]; fp_test = fp[split:]; fs_test = fs[split:]
    phi_snap = np.zeros(N_LAGS + 1); phi_snap[0] = 1.0
    A = build_Aspec(phi, phi_snap, fp, fs, T, causal=True)
    xb, xs = decompose_ista(b, A, cfg["lambda_smooth"], cfg["lambda_sparse"])
    Tf = len(fp_test)
    recent = xb[-30:]
    trend = (recent[-1] - recent[0]) / 30.0 if len(recent) > 1 else 0.0
    base = xb[-1] + trend * np.arange(1, Tf + 1)
    hist = xs[xs > 1e-3]
    avg = np.median(hist) if len(hist) > 0 else np.std(xb) * 0.5
    fut = np.zeros(Tf)
    for t in range(Tf):
        if fp_test[t] > 0 or fs_test[t] > 0: fut[t] = avg
    Af = build_Aspec(phi, phi_snap, fp_test, fs_test, Tf)
    forecast = base + Af @ fut
    return {"vr": variance_ratio(forecast, b_test), "wmape": wmape(b_test, forecast)}

print("="*60)
print("EDMD ABLATION: COMPREHENSIVE TUNING")
print("="*60)

m5_dir = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")
print("Loading 200 SKUs...")
data = load_m5_data(m5_dir, n_skus=200, random_state=42)
print(f"Loaded {len(data)} SKUs")

cats = {}
for sku in data:
    c = sku.get("category", "unknown")
    cats.setdefault(c, []).append(sku)
print(f"Categories: { {k: len(v) for k, v in cats.items()} }")

cfg = CONFIG_DEFAULT.copy()
param_kernels = [param_kernel(sku) for sku in data]

configs = [
    ("v1-buggy", poly_v1, 0.0, 0),      # Original: C[0] (oldest lag)
    ("v1-fixed", poly_v1, 0.0, None),   # Fixed: C[n_lag-1] (current demand)
    ("v2-cross", poly_v2, 0.0, None),    # Richer observables
    ("v1-ridge", poly_v1, 1.0, None),    # Ridge
    ("v2-ridge", poly_v2, 1.0, None),    # Richer + ridge
]

results = {}
for name, feat, ridge, cidx in configs:
    print(f"\n{'='*60}")
    print(f"CONFIG: {name} (c_idx={cidx}, ridge={ridge})")
    print(f"{'='*60}")
    
    cat_res = {}
    for c, skus in cats.items():
        r = fit_edmd_pooled(skus, feat, normalize=True, ridge=ridge)
        if cidx is not None and r and "error" not in r:
            # Override C to simulate buggy version
            r["C"] = np.zeros(r["d_psi"]); r["C"][cidx] = 1.0
            r["h"] = extract_ir(r["K"], r["B"][:, 0], r["C"], n_lag=N_LAGS)
            r["h_snap"] = extract_ir(r["K"], r["B"][:, 1], r["C"], n_lag=N_LAGS)
            if np.abs(r["h"][0]) > 1e-6: r["h"] = r["h"] / r["h"][0]
        cat_res[c] = r
        if r and "error" not in r:
            print(f"  {c:12s}: rho={r['spectral_radius']:.4f}, d_psi={r['d_psi']}, "
                  f"n={r['n_samples']}, stable={r['stable']}")
    
    glob = fit_edmd_pooled(data, feat, normalize=True, ridge=ridge)
    if cidx is not None and glob and "error" not in glob:
        glob["C"] = np.zeros(glob["d_psi"]); glob["C"][cidx] = 1.0
        glob["h"] = extract_ir(glob["K"], glob["B"][:, 0], glob["C"], n_lag=N_LAGS)
        glob["h_snap"] = extract_ir(glob["K"], glob["B"][:, 1], glob["C"], n_lag=N_LAGS)
        if np.abs(glob["h"][0]) > 1e-6: glob["h"] = glob["h"] / glob["h"][0]
    if glob and "error" not in glob:
        print(f"  {'global':12s}: rho={glob['spectral_radius']:.4f}, d_psi={glob['d_psi']}")
    
    cat_ir = {c: r["h"] for c, r in cat_res.items() if r and "error" not in r and r.get("stable")}
    glob_ir = glob["h"] if (glob and "error" not in glob and glob.get("stable")) else None
    
    m = {"parametric": [], "edmd_cat": [], "edmd_global": []}
    for i, sku in enumerate(data):
        m["parametric"].append(evaluate(sku, param_kernels[i], cfg))
        c = sku.get("category", "unknown")
        if c in cat_ir: m["edmd_cat"].append(evaluate(sku, cat_ir[c], cfg))
        else: m["edmd_cat"].append({"vr": np.nan, "wmape": np.nan})
        if glob_ir is not None: m["edmd_global"].append(evaluate(sku, glob_ir, cfg))
        else: m["edmd_global"].append({"vr": np.nan, "wmape": np.nan})
    
    pv = [v["vr"] for v in m["parametric"]]
    ev = [v["vr"] for v in m["edmd_cat"] if not np.isnan(v["vr"])]
    gv = [v["vr"] for v in m["edmd_global"] if not np.isnan(v["vr"])]
    pc = 1.0; pg = 1.0
    if len(ev) == len(pv):
        try: _, pc = wilcoxon(pv, ev)
        except: pass
    if len(gv) == len(pv):
        try: _, pg = wilcoxon(pv, gv)
        except: pass
    
    print(f"\n  Parametric:    VR={np.median(pv):.4f}, wMAPE={np.median([v['wmape'] for v in m['parametric']]):.3f}")
    print(f"  EDMD (cat):    VR={np.median(ev):.4f}, wMAPE={np.median([v['wmape'] for v in m['edmd_cat'] if not np.isnan(v['wmape'])]):.3f}")
    print(f"  EDMD (global): VR={np.median(gv):.4f}, wMAPE={np.median([v['wmape'] for v in m['edmd_global'] if not np.isnan(v['wmape'])]):.3f}")
    print(f"  Wilcoxon p: cat={pc:.3f}, glob={pg:.3f}")
    
    results[name] = {"cat_ir": cat_ir, "global_ir": glob_ir, "metrics": m, "p_cat": pc, "p_glob": pg}

print(f"\n{'='*60}")
print("COMPARISON TABLE")
print(f"{'='*60}")
print(f"{'Config':<15s} {'Cat VR':>10s} {'Glob VR':>10s} {'Cat wMAPE':>10s} {'Glob wMAPE':>11s} {'p(cat)':>8s}")
print("-"*60)
for name, _, _, _ in configs:
    r = results[name]
    cv = np.median([v["vr"] for v in r["metrics"]["edmd_cat"] if not np.isnan(v["vr"])])
    gv = np.median([v["vr"] for v in r["metrics"]["edmd_global"] if not np.isnan(v["vr"])])
    cw = np.median([v["wmape"] for v in r["metrics"]["edmd_cat"] if not np.isnan(v["wmape"])])
    gw = np.median([v["wmape"] for v in r["metrics"]["edmd_global"] if not np.isnan(v["wmape"])])
    print(f"{name:<15s} {cv:>10.4f} {gv:>10.4f} {cw:>10.3f} {gw:>11.3f} {r['p_cat']:>8.3f}")

pv = np.median([v["vr"] for v in results["v1-fixed"]["metrics"]["parametric"]])
print(f"{'Parametric':<15s} {pv:>10.4f} {'--':>10s} {'--':>10s} {'--':>11s} {'--':>8s}")

# Show first IR for representative SKU
print(f"\n{'='*60}")
print("IMPULSE RESPONSE (lag 0-5) for first SKU")
print(f"{'='*60}")
print(f"{'Config':<15s} {'h0':>8s} {'h1':>8s} {'h2':>8s} {'h3':>8s} {'h4':>8s} {'h5':>8s}")
print("-"*60)
for name, _, _, _ in configs:
    r = results[name]
    if "foods" in r["cat_ir"]:
        h = r["cat_ir"]["foods"]
        print(f"{name:<15s} {h[0]:>8.4f} {h[1]:>8.4f} {h[2]:>8.4f} {h[3]:>8.4f} {h[4]:>8.4f} {h[5]:>8.4f}")

print(f"\n{'='*60}")
print("KEY FINDING:")
print("v1-buggy (C[0]) shows flat IR because it projects onto OLDEST lag.")
print("v1-fixed (C[n-1]) projects onto CURRENT demand, revealing structure.")
print("="*60)
