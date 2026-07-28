import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import numpy as np
import time
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, _os.path.join(_ROOT, "code"))

from contextual_deconvolution.synthetic import generate_synthetic_sku
from contextual_deconvolution.core import (
    estimate_kernel, fit_parametric_kernel, build_Aspec, decompose_ista,
    kernel_from_theta,
)
from contextual_deconvolution.experiments import evaluate_sku, CONFIG_DEFAULT


def build_Aspec_from_theta(theta, f_promo, f_snap, T, n_lags=14, causal=True):
    """Build A_spec from kernel parameters (alpha, beta, tau1, tau2)."""
    alpha, beta, tau1, tau2 = theta
    k = np.arange(n_lags + 1)
    phi_promo = alpha * np.exp(-k / tau1) - beta * np.exp(-k / tau2)
    phi_snap = np.zeros(n_lags + 1)
    phi_snap[0] = 1.0
    return build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=causal)


def objective_full(theta, b, f_promo, f_snap, lambda_smooth, lambda_sparse, n_lags=14):
    """Full objective: reconstruction + regularization for fixed theta."""
    T = len(b)
    A_spec = build_Aspec_from_theta(theta, f_promo, f_snap, T, n_lags)
    x_base, x_spec = decompose_ista(b, A_spec, lambda_smooth, lambda_sparse)
    recon = np.linalg.norm(b - (x_base + A_spec @ x_spec))**2
    smooth = lambda_smooth * np.linalg.norm(np.diff(x_base, n=2))**2
    sparse = lambda_sparse * np.linalg.norm(x_spec, 1)
    return recon + smooth + sparse


def two_stage_pipeline(sku, config):
    """Standard two-stage pipeline."""
    b = sku["demand"]
    f_promo = sku["f_promo"]
    f_snap = sku["f_snap"]
    T = len(b)
    
    start = time.time()
    phi_promo_emp, _ = estimate_kernel(b, f_promo, config["n_lags"], config["pre_window"])
    theta_promo = fit_parametric_kernel(phi_promo_emp, family="surge_vacuum")
    if theta_promo is not None:
        phi_promo = kernel_from_theta(theta_promo, config["n_lags"])
    else:
        phi_promo = np.zeros(config["n_lags"] + 1)
        phi_promo[0] = 1.0
    
    phi_snap = np.zeros(config["n_lags"] + 1)
    phi_snap[0] = 1.0
    
    A_spec = build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=True)
    x_base, x_spec = decompose_ista(b, A_spec, config["lambda_smooth"], config["lambda_sparse"])
    elapsed = time.time() - start
    
    recon = np.linalg.norm(b - (x_base + A_spec @ x_spec))**2
    smooth = config["lambda_smooth"] * np.linalg.norm(np.diff(x_base, n=2))**2
    sparse = config["lambda_sparse"] * np.linalg.norm(x_spec, 1)
    obj = recon + smooth + sparse
    
    return {
        "theta": theta_promo if theta_promo is not None else (1.0, 0.6, 2.0, 5.0),
        "x_base": x_base, "x_spec": x_spec, "A_spec": A_spec,
        "obj": obj, "recon": recon, "time": elapsed,
    }


def joint_alternating_pipeline(sku, config, max_iter=10, tol=1e-4):
    """Joint optimization via alternating minimization."""
    b = sku["demand"]
    f_promo = sku["f_promo"]
    f_snap = sku["f_snap"]
    T = len(b)
    
    # Initialize with two-stage estimate
    phi_promo_emp, _ = estimate_kernel(b, f_promo, config["n_lags"], config["pre_window"])
    theta_init = fit_parametric_kernel(phi_promo_emp, family="surge_vacuum")
    if theta_init is None:
        theta_init = np.array([1.0, 0.6, 2.0, 5.0])
    else:
        theta_init = np.array(theta_init)
    
    theta = theta_init.copy()
    bounds = [(0.1, 5.0), (0.01, 3.0), (0.5, 10.0), (1.0, 20.0)]
    
    start = time.time()
    
    # Precompute smoothness operator
    L2 = np.diff(np.eye(T), n=2, axis=0)
    
    for it in range(max_iter):
        # Stage A: Fix theta, solve for x_base, x_spec
        A_spec = build_Aspec_from_theta(theta, f_promo, f_snap, T, config["n_lags"])
        x_base, x_spec = decompose_ista(b, A_spec, config["lambda_smooth"], config["lambda_sparse"])
        
        # Stage B: Fix x_base, x_spec, optimize theta
        def theta_obj(th):
            A = build_Aspec_from_theta(th, f_promo, f_snap, T, config["n_lags"])
            r = b - (x_base + A @ x_spec)
            return np.dot(r, r)
        
        res = minimize(theta_obj, theta, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 20, 'disp': False})
        theta_new = res.x
        
        # Check convergence
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    
    elapsed = time.time() - start
    
    A_spec = build_Aspec_from_theta(theta, f_promo, f_snap, T, config["n_lags"])
    x_base, x_spec = decompose_ista(b, A_spec, config["lambda_smooth"], config["lambda_sparse"])
    
    recon = np.linalg.norm(b - (x_base + A_spec @ x_spec))**2
    smooth = config["lambda_smooth"] * np.linalg.norm(np.diff(x_base, n=2))**2
    sparse = config["lambda_sparse"] * np.linalg.norm(x_spec, 1)
    obj = recon + smooth + sparse
    
    return {
        "theta": tuple(theta), "x_base": x_base, "x_spec": x_spec, "A_spec": A_spec,
        "obj": obj, "recon": recon, "time": elapsed, "iterations": it + 1,
    }


def main():
    config = CONFIG_DEFAULT.copy()
    
    print("=" * 70)
    print("TWO-STAGE vs JOINT OPTIMIZATION ABLATION")
    print("=" * 70)
    
    results = []
    for seed in range(10):
        sku = generate_synthetic_sku(T=500, n_promos=20, seed=seed)
        true_theta = (
            sku["kernel_params"]["alpha"],
            sku["kernel_params"]["beta"],
            sku["kernel_params"]["tau1"],
            sku["kernel_params"]["tau2"],
        )
        
        ts = two_stage_pipeline(sku, config)
        joint = joint_alternating_pipeline(sku, config, max_iter=10, tol=1e-4)
        
        # Kernel recovery error
        ts_err = np.linalg.norm(np.array(ts["theta"]) - np.array(true_theta)) / np.linalg.norm(np.array(true_theta))
        joint_err = np.linalg.norm(np.array(joint["theta"]) - np.array(true_theta)) / np.linalg.norm(np.array(true_theta))
        
        # Objective improvement
        obj_improvement = (ts["obj"] - joint["obj"]) / ts["obj"] * 100
        
        results.append({
            "seed": seed,
            "ts_obj": ts["obj"], "joint_obj": joint["obj"],
            "ts_recon": ts["recon"], "joint_recon": joint["recon"],
            "ts_time": ts["time"], "joint_time": joint["time"],
            "ts_theta_err": ts_err, "joint_theta_err": joint_err,
            "obj_improvement": obj_improvement,
            "joint_iters": joint["iterations"],
        })
        
        print(f"Seed {seed}: TS obj={ts['obj']:.2f}, Joint obj={joint['obj']:.2f} ({obj_improvement:+.2f}%), "
              f"TS θ_err={ts_err:.3f}, Joint θ_err={joint_err:.3f}, "
              f"TS time={ts['time']:.3f}s, Joint time={joint['time']:.3f}s ({joint['iterations']} iters)")
    
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    
    df = {k: [r[k] for r in results] for k in results[0].keys()}
    
    for metric in ["ts_obj", "joint_obj", "ts_recon", "joint_recon",
                     "ts_theta_err", "joint_theta_err", "obj_improvement",
                     "ts_time", "joint_time"]:
        vals = np.array(df[metric])
        print(f"{metric:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    # Save results
    out = _os.path.join(_ROOT, "paper/results/two_stage_vs_joint.csv")
    import pandas as pd
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
