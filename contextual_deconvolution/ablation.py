"""
Ablation studies to isolate the contribution of each component.
"""

import numpy as np
from typing import List, Dict, Tuple
import pandas as pd

from .core import (
    build_Aspec, decompose_cvxpy, decompose_ista,
    compute_decomposition_quality, estimate_kernel, fit_parametric_kernel,
    kernel_from_theta,
)
from .metrics import wmape, variance_ratio, inventory_cost


def ablation_identity_vs_kernel(
    dataset: List[Dict],
    config: Dict,
) -> pd.DataFrame:
    """
    Ablation 1: Compare kernel-modulated operator vs. identity operator (A_spec = I).
    Measures: reconstruction RMSE, sparsity, baseline smoothness, forecast variance.
    """
    rows = []
    for sku in dataset:
        b = sku["demand"]
        f_promo = sku["f_promo"]
        f_snap = sku["f_snap"]
        T = len(b)

        # Estimate kernel
        phi_promo_emp, _ = estimate_kernel(b, f_promo, config["n_lags"], config["pre_window"])
        theta = fit_parametric_kernel(phi_promo_emp)
        if theta is not None:
            phi_promo = kernel_from_theta(theta, config["n_lags"])
        else:
            phi_promo = np.zeros(config["n_lags"] + 1)
            phi_promo[0] = 1.0
        phi_snap = np.zeros(config["n_lags"] + 1)
        phi_snap[0] = 1.0

        # Kernel-modulated
        A_kernel = build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=True)
        xb_k, xs_k, _ = decompose_cvxpy(b, A_kernel, config["lambda_smooth"], config["lambda_sparse"])
        dq_k = compute_decomposition_quality(b, xb_k, xs_k, A_kernel)

        # Identity
        A_id = np.eye(T)
        xb_i, xs_i, _ = decompose_cvxpy(b, A_id, config["lambda_smooth"], config["lambda_sparse"])
        dq_i = compute_decomposition_quality(b, xb_i, xs_i, A_id)

        rows.append({
            "sku_id": sku.get("sku_id", 0),
            "rmse_kernel": dq_k["rmse"], "sparsity_kernel": dq_k["sparsity"], "smoothness_kernel": dq_k["smoothness"],
            "rmse_identity": dq_i["rmse"], "sparsity_identity": dq_i["sparsity"], "smoothness_identity": dq_i["smoothness"],
        })

    return pd.DataFrame(rows)


def ablation_causality(
    dataset: List[Dict],
    config: Dict,
) -> pd.DataFrame:
    """
    Ablation 2: Compare causal vs. non-causal operator (allows super-diagonals).
    Measures: test RMSE, variance ratio. Non-causal should overfit training.
    """
    rows = []
    for sku in dataset:
        b = sku["demand"]
        f_promo = sku["f_promo"]
        f_snap = sku["f_snap"]
        T = len(b)
        split = T - config["forecast_horizon"]

        phi_promo_emp, _ = estimate_kernel(b[:split], f_promo[:split], config["n_lags"], config["pre_window"])
        theta = fit_parametric_kernel(phi_promo_emp)
        if theta is not None:
            phi_promo = kernel_from_theta(theta, config["n_lags"])
        else:
            phi_promo = np.zeros(config["n_lags"] + 1)
            phi_promo[0] = 1.0
        phi_snap = np.zeros(config["n_lags"] + 1)
        phi_snap[0] = 1.0

        A_causal = build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=True)
        A_nc = build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=False)

        xb_c, xs_c, _ = decompose_cvxpy(b, A_causal, config["lambda_smooth"], config["lambda_sparse"])
        xb_nc, xs_nc, _ = decompose_cvxpy(b, A_nc, config["lambda_smooth"], config["lambda_sparse"])

        # Train/test split for fairness
        train_rmse_c = np.sqrt(np.mean((b[:split] - (xb_c + A_causal @ xs_c)[:split]) ** 2))
        train_rmse_nc = np.sqrt(np.mean((b[:split] - (xb_nc + A_nc @ xs_nc)[:split]) ** 2))
        test_rmse_c = np.sqrt(np.mean((b[split:] - (xb_c + A_causal @ xs_c)[split:]) ** 2))
        test_rmse_nc = np.sqrt(np.mean((b[split:] - (xb_nc + A_nc @ xs_nc)[split:]) ** 2))

        rows.append({
            "sku_id": sku.get("sku_id", 0),
            "train_rmse_causal": train_rmse_c, "test_rmse_causal": test_rmse_c,
            "train_rmse_noncausal": train_rmse_nc, "test_rmse_noncausal": test_rmse_nc,
        })

    return pd.DataFrame(rows)


def ablation_kernel_misspecification(
    dataset: List[Dict],
    config: Dict,
) -> pd.DataFrame:
    """
    Ablation 3: Test sensitivity to kernel misspecification.
    Fit surge-and-vacuum, then force exponential decay (no vacuum tail) and impulse.
    """
    rows = []
    for sku in dataset:
        b = sku["demand"]
        f_promo = sku["f_promo"]
        f_snap = sku["f_snap"]
        T = len(b)

        phi_promo_emp, _ = estimate_kernel(b, f_promo, config["n_lags"], config["pre_window"])

        families = ["surge_vacuum", "exp_decay", "impulse"]
        for family in families:
            theta = fit_parametric_kernel(phi_promo_emp, family=family)
            if theta is not None:
                phi = kernel_from_theta(theta, config["n_lags"])
            else:
                phi = np.zeros(config["n_lags"] + 1)
                phi[0] = 1.0
            phi_snap = np.zeros(config["n_lags"] + 1)
            phi_snap[0] = 1.0

            A = build_Aspec(phi, phi_snap, f_promo, f_snap, T, causal=True)
            xb, xs, _ = decompose_cvxpy(b, A, config["lambda_smooth"], config["lambda_sparse"])
            dq = compute_decomposition_quality(b, xb, xs, A)
            rows.append({
                "sku_id": sku.get("sku_id", 0),
                "family": family,
                "rmse": dq["rmse"], "sparsity": dq["sparsity"], "smoothness": dq["smoothness"],
            })

    return pd.DataFrame(rows)


def ablation_regularization_sensitivity(
    dataset: List[Dict],
    config: Dict,
    lambda_smooth_values: np.ndarray,
    lambda_sparse_values: np.ndarray,
) -> pd.DataFrame:
    """
    Ablation 4: Grid search over regularization parameters.
    """
    rows = []
    for sku in dataset:
        b = sku["demand"]
        f_promo = sku["f_promo"]
        f_snap = sku["f_snap"]
        T = len(b)

        phi_promo_emp, _ = estimate_kernel(b, f_promo, config["n_lags"], config["pre_window"])
        theta = fit_parametric_kernel(phi_promo_emp)
        if theta is not None:
            phi_promo = kernel_from_theta(theta, config["n_lags"])
        else:
            phi_promo = np.zeros(config["n_lags"] + 1)
            phi_promo[0] = 1.0
        phi_snap = np.zeros(config["n_lags"] + 1)
        phi_snap[0] = 1.0

        A = build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=True)

        for l_s in lambda_smooth_values:
            for l_1 in lambda_sparse_values:
                xb, xs, _ = decompose_cvxpy(b, A, l_s, l_1)
                dq = compute_decomposition_quality(b, xb, xs, A)
                rows.append({
                    "sku_id": sku.get("sku_id", 0),
                    "lambda_smooth": l_s, "lambda_sparse": l_1,
                    "rmse": dq["rmse"], "sparsity": dq["sparsity"], "smoothness": dq["smoothness"],
                })

    return pd.DataFrame(rows)
