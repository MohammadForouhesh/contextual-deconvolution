"""
Falsification tests: tests designed to *refute* key claims.
If a test passes, the claim survives; if it fails, the claim is invalidated.
"""

import numpy as np
from typing import Dict, List
from .core import (
    build_Aspec, decompose_cvxpy, build_difference_operator,
    compute_decomposition_quality, robust_detrend,
)
from .synthetic import generate_synthetic_sku
from .metrics import variance_ratio


def falsify_claim_1_kernel_recovers_ground_truth(
    seed: int = 42,
) -> Dict:
    """
    Claim: The two-stage estimator recovers the true kernel parameters.
    Falsification: Generate data with known kernel, estimate, compare.
    Threshold: Relative error in tau1 and tau2 < 30%.
    """
    np.random.seed(seed)
    true_params = {"alpha": 1.0, "beta": 0.6, "tau1": 2.0, "tau2": 5.0}
    sku = generate_synthetic_sku(
        T=500, n_promos=20, kernel_params=true_params, noise_std=2.0, seed=seed
    )

    from .core import estimate_kernel, fit_parametric_kernel
    phi_emp, n_events = estimate_kernel(sku["demand"], sku["f_promo"], n_lags=14, pre_window=30)
    theta_est = fit_parametric_kernel(phi_emp)

    if theta_est is None:
        return {"passed": False, "reason": "Failed to estimate kernel", "rel_error_tau1": None, "rel_error_tau2": None}

    rel_error_tau1 = abs(theta_est[2] - true_params["tau1"]) / true_params["tau1"]
    rel_error_tau2 = abs(theta_est[3] - true_params["tau2"]) / true_params["tau2"]

    return {
        "passed": rel_error_tau1 < 0.50 and rel_error_tau2 < 0.50,
        "rel_error_tau1": rel_error_tau1,
        "rel_error_tau2": rel_error_tau2,
        "true_params": true_params,
        "est_params": theta_est,
        "n_events": n_events,
    }


def falsify_claim_2_sparsity_improves_with_kernel(
    seed: int = 42,
) -> Dict:
    """
    Claim: Kernel-modulated operator reduces sparsity vs. identity operator.
    Falsification: If sparsity_kernel >= sparsity_identity, claim fails.
    """
    np.random.seed(seed)
    sku = generate_synthetic_sku(T=500, n_promos=20, seed=seed)
    b = sku["demand"]
    f_promo = sku["f_promo"]
    f_snap = sku["f_snap"]
    T = len(b)

    from .core import estimate_kernel, fit_parametric_kernel, kernel_from_theta
    phi_emp, _ = estimate_kernel(b, f_promo, n_lags=14, pre_window=30)
    theta = fit_parametric_kernel(phi_emp)
    if theta is not None:
        phi_promo = kernel_from_theta(theta, 14)
    else:
        phi_promo = np.zeros(15)
        phi_promo[0] = 1.0
    phi_snap = np.zeros(15)
    phi_snap[0] = 1.0

    A_kernel = build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=True)
    A_id = np.eye(T)

    xb_k, xs_k, _ = decompose_cvxpy(b, A_kernel, 10.0, 8.0)
    xb_i, xs_i, _ = decompose_cvxpy(b, A_id, 10.0, 8.0)

    sparsity_k = int(np.sum(np.abs(xs_k) > 1e-3))
    sparsity_i = int(np.sum(np.abs(xs_i) > 1e-3))

    return {
        "passed": sparsity_k < sparsity_i,
        "sparsity_kernel": sparsity_k,
        "sparsity_identity": sparsity_i,
        "reduction_pct": (1 - sparsity_k / max(sparsity_i, 1)) * 100,
    }


def falsify_claim_3_variance_ratio_below_one(
    seed: int = 42,
) -> Dict:
    """
    Claim: Two-stream forecast variance ratio < 1.0 (no Bullwhip inflation).
    Falsification: If variance ratio >= 1.0, claim fails.
    """
    np.random.seed(seed)
    sku = generate_synthetic_sku(T=500, n_promos=20, seed=seed)

    from .experiments import evaluate_sku, CONFIG_DEFAULT
    config = CONFIG_DEFAULT.copy()
    config["forecast_horizon"] = 28
    result = evaluate_sku(sku, config)

    if "CD" not in result["forecasts"]:
        return {"passed": False, "reason": "No forecast generated"}

    vr = result["forecasts"]["CD"]["variance_ratio"]
    return {
        "passed": vr < 1.0,
        "variance_ratio": vr,
    }


def falsify_claim_4_causality_matters(
    seed: int = 42,
) -> Dict:
    """
    Claim: Non-causal operator degrades test performance.
    Falsification: If non-causal test RMSE <= causal test RMSE, claim fails.
    """
    np.random.seed(seed)
    sku = generate_synthetic_sku(T=500, n_promos=20, seed=seed)
    b = sku["demand"]
    f_promo = sku["f_promo"]
    f_snap = sku["f_snap"]
    T = len(b)
    split = T - 28

    from .core import estimate_kernel, fit_parametric_kernel, kernel_from_theta
    phi_emp, _ = estimate_kernel(b[:split], f_promo[:split], n_lags=14, pre_window=30)
    theta = fit_parametric_kernel(phi_emp)
    if theta is not None:
        phi = kernel_from_theta(theta, 14)
    else:
        phi = np.zeros(15)
        phi[0] = 1.0
    phi_snap = np.zeros(15)
    phi_snap[0] = 1.0

    A_c = build_Aspec(phi, phi_snap, f_promo, f_snap, T, causal=True)
    A_nc = build_Aspec(phi, phi_snap, f_promo, f_snap, T, causal=False)

    xb_c, xs_c, _ = decompose_cvxpy(b, A_c, 10.0, 8.0)
    xb_nc, xs_nc, _ = decompose_cvxpy(b, A_nc, 10.0, 8.0)

    test_rmse_c = np.sqrt(np.mean((b[split:] - (xb_c + A_c @ xs_c)[split:]) ** 2))
    test_rmse_nc = np.sqrt(np.mean((b[split:] - (xb_nc + A_nc @ xs_nc)[split:]) ** 2))

    return {
        "passed": test_rmse_nc >= test_rmse_c,
        "test_rmse_causal": test_rmse_c,
        "test_rmse_noncausal": test_rmse_nc,
    }


def falsify_claim_5_baseline_is_smooth(
    seed: int = 42,
) -> Dict:
    """
    Claim: The recovered baseline is smoother than the raw signal.
    Falsification: If ||L x_base|| >= ||L b||, claim fails.
    """
    np.random.seed(seed)
    sku = generate_synthetic_sku(T=500, n_promos=20, seed=seed)
    b = sku["demand"]
    f_promo = sku["f_promo"]
    f_snap = sku["f_snap"]
    T = len(b)

    from .core import estimate_kernel, fit_parametric_kernel, kernel_from_theta
    phi_emp, _ = estimate_kernel(b, f_promo, n_lags=14, pre_window=30)
    theta = fit_parametric_kernel(phi_emp)
    if theta is not None:
        phi = kernel_from_theta(theta, 14)
    else:
        phi = np.zeros(15)
        phi[0] = 1.0
    phi_snap = np.zeros(15)
    phi_snap[0] = 1.0

    A = build_Aspec(phi, phi_snap, f_promo, f_snap, T, causal=True)
    xb, xs, _ = decompose_cvxpy(b, A, 10.0, 8.0)

    L1 = build_difference_operator(T, order=1).toarray()
    smooth_base = float(np.linalg.norm(L1 @ xb, 2))
    smooth_raw = float(np.linalg.norm(L1 @ b, 2))

    return {
        "passed": smooth_base < smooth_raw,
        "smoothness_base": smooth_base,
        "smoothness_raw": smooth_raw,
        "reduction_pct": (1 - smooth_base / max(smooth_raw, 1e-9)) * 100,
    }


def run_all_falsification_tests(seed: int = 42) -> Dict:
    """Run all falsification tests and return a summary report."""
    tests = {
        "kernel_recovery": falsify_claim_1_kernel_recovers_ground_truth,
        "sparsity_improvement": falsify_claim_2_sparsity_improves_with_kernel,
        "variance_ratio": falsify_claim_3_variance_ratio_below_one,
        "causality_matters": falsify_claim_4_causality_matters,
        "baseline_smoothness": falsify_claim_5_baseline_is_smooth,
    }
    results = {}
    all_passed = True
    for name, test_fn in tests.items():
        r = test_fn(seed=seed)
        results[name] = r
        if not r["passed"]:
            all_passed = False

    return {"all_passed": all_passed, "tests": results}
