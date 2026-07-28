"""
Core module: operators, convex decomposition, and kernel estimation.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple, Dict, List


def build_difference_operator(T: int, order: int = 2) -> sparse.csr_matrix:
    """Build sparse finite-difference operator L of given order."""
    if order == 1:
        i = np.arange(T - 1)
        L = sparse.csr_matrix(
            (-np.ones(T - 1), (i, i)), shape=(T - 1, T)
        ) + sparse.csr_matrix(
            (np.ones(T - 1), (i, i + 1)), shape=(T - 1, T)
        )
        return L
    elif order == 2:
        i = np.arange(T - 2)
        L = sparse.csr_matrix(
            (np.ones(T - 2), (i, i)), shape=(T - 2, T)
        ) + sparse.csr_matrix(
            (-2 * np.ones(T - 2), (i, i + 1)), shape=(T - 2, T)
        ) + sparse.csr_matrix(
            (np.ones(T - 2), (i, i + 2)), shape=(T - 2, T)
        )
        return L
    else:
        raise ValueError("order must be 1 or 2")


def build_stacked_difference_operator(T: int) -> sparse.csr_matrix:
    """Build stacked L = [L1; L2] for first- and second-order differences."""
    L1 = build_difference_operator(T, order=1)
    L2 = build_difference_operator(T, order=2)
    return sparse.vstack([L1, L2]).tocsr()


def build_difference_operator(T: int, order: int = 2) -> sparse.csr_matrix:
    """Build sparse finite-difference operator L of given order."""
    if order == 1:
        i = np.arange(T - 1)
        L = sparse.csr_matrix(
            (-np.ones(T - 1), (i, i)), shape=(T - 1, T)
        ) + sparse.csr_matrix(
            (np.ones(T - 1), (i, i + 1)), shape=(T - 1, T)
        )
        return L
    elif order == 2:
        i = np.arange(T - 2)
        L = sparse.csr_matrix(
            (np.ones(T - 2), (i, i)), shape=(T - 2, T)
        ) + sparse.csr_matrix(
            (-2 * np.ones(T - 2), (i, i + 1)), shape=(T - 2, T)
        ) + sparse.csr_matrix(
            (np.ones(T - 2), (i, i + 2)), shape=(T - 2, T)
        )
        return L
    else:
        raise ValueError("order must be 1 or 2")


def build_stacked_difference_operator(T: int) -> sparse.csr_matrix:
    """Build stacked L = [L1; L2] for first- and second-order differences."""
    L1 = build_difference_operator(T, order=1)
    L2 = build_difference_operator(T, order=2)
    return sparse.vstack([L1, L2]).tocsr()


def build_Aspec(
    phi_promo: np.ndarray,
    phi_snap: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    T: int,
    causal: bool = True,
) -> sparse.csr_matrix:
    """
    Build context-modulated banded operator A_spec from kernels and features.

    Parameters
    ----------
    phi_promo, phi_snap : arrays of shape (n_lags + 1,)
        Kernel weights for each lag.
    f_promo, f_snap : arrays of shape (T,)
        Binary event indicators.
    T : int
        Time series length.
    causal : bool
        If True, only sub-diagonals (k >= 0). If False, allows super-diagonals.

    Returns
    -------
    A : sparse.csr_matrix of shape (T, T)
        The context-modulated operator.
    """
    n = len(phi_promo) - 1
    A = sparse.lil_matrix((T, T))
    k_range = range(0, n + 1) if causal else range(-n, n + 1)
    for k in k_range:
        k_idx = abs(k)
        if k_idx > n:
            continue
        diag_vals = phi_promo[k_idx] * f_promo + phi_snap[k_idx] * f_snap
        if k >= 0:
            if k < T:
                idx = np.arange(T - k)
                A[idx + k, idx] += diag_vals[k:]
        else:
            k_abs = abs(k)
            if k_abs < T:
                idx = np.arange(T - k_abs)
                A[idx, idx + k_abs] += diag_vals[k_abs:]
    return A.tocsr()


def robust_detrend(
    b: np.ndarray,
    lambda_smooth: float = 10.0,
    max_iter: int = 30,
    delta: Optional[float] = None,
) -> np.ndarray:
    """
    Iteratively reweighted least squares (IRLS) with Huber-like behavior.
    Returns a robust trend estimate mu.
    Uses sparse normal equations for O(T) per-iteration cost.
    """
    T = len(b)
    diffs = np.diff(b)
    if delta is None:
        delta = 2.0 * np.median(np.abs(diffs - np.median(diffs))) / 0.6745
        delta = max(delta, 1e-6)

    # Precompute L1.T @ L1 structure (tridiagonal)
    # L1 is (T-1, T) with -1 on diagonal and +1 on super-diagonal
    # L1.T @ L1 has: 1 on corners, 2 on interior diagonal, -1 on off-diagonals
    main_diag = np.full(T, 2.0)
    main_diag[0] = 1.0
    main_diag[-1] = 1.0
    off_diag = np.full(T - 1, -1.0)
    LtL = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="csr")

    mu = b.copy()
    for _ in range(max_iter):
        r = b - mu
        weights = np.where(np.abs(r) <= delta, 1.0, delta / (np.abs(r) + 1e-12))
        # Normal equations: (W + lambda * LtL) @ mu = W @ b
        M = sparse.diags([weights], [0], format="csr") + lambda_smooth * LtL
        rhs = weights * b
        mu_new = spsolve(M, rhs)
        if np.max(np.abs(mu_new - mu)) < 1e-6:
            break
        mu = mu_new
    return mu


def estimate_kernel(
    b: np.ndarray,
    event_indicator: np.ndarray,
    n_lags: int = 14,
    pre_window: int = 30,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Estimate kernel by averaging detrended responses around event starts.

    Returns
    -------
    kernel : ndarray of shape (n_lags + 1,) or None
        Normalized kernel shape.
    n_events : int
        Number of valid events used.
    """
    T = len(b)
    event_starts = np.where(
        (event_indicator[1:] > 0) & (event_indicator[:-1] == 0)
    )[0] + 1
    if len(event_starts) < 2:
        return None, 0

    mu = robust_detrend(b, lambda_smooth=10.0)
    r = b - mu
    responses = []
    valid_events = 0
    for d in event_starts:
        if d + n_lags < T and d - pre_window >= 0:
            pre_level = np.median(r[d - pre_window : d])
            response = r[d : d + n_lags + 1] - pre_level
            if response[0] > 0:
                responses.append(response)
                valid_events += 1

    if valid_events < 2:
        return None, 0

    kernel = np.median(responses, axis=0)
    if kernel[0] > 0:
        kernel = kernel / kernel[0]
    return kernel, valid_events


def fit_parametric_kernel(
    empirical_kernel: Optional[np.ndarray],
    family: str = "surge_vacuum",
) -> Optional[np.ndarray]:
    """
    Fit a parametric kernel family to empirical shape via nonlinear least squares.

    Parameters
    ----------
    empirical_kernel : ndarray or None
        Empirical kernel weights.
    family : str
        "surge_vacuum" (Difference of Exponentials) or "impulse" or "exp_decay".

    Returns
    -------
    theta : ndarray or None
        Parameters [alpha, beta, tau1, tau2] for surge_vacuum, or None.
    """
    if empirical_kernel is None:
        return None
    k = np.arange(len(empirical_kernel))

    if family == "impulse":
        return np.array([1.0, 0.0, 0.0, 0.0])

    if family == "exp_decay":
        def loss(theta):
            tau = theta[0]
            if tau <= 0.1:
                return 1e10
            phi = np.exp(-k / tau)
            return np.sum((empirical_kernel[:8] - phi[:8]) ** 2)
        result = minimize(loss, [3.0], method="L-BFGS-B", bounds=[(0.5, 20.0)])
        if not result.success or np.isnan(result.fun):
            return None
        return np.array([1.0, 0.0, result.x[0], 0.0]) if result.success else None

    # Default: surge-and-vacuum (Difference of Exponentials)
    def loss(theta):
        alpha, beta, tau1, tau2 = theta
        if alpha <= beta or beta <= 0 or tau1 <= 0.1 or tau2 <= tau1:
            return 1e10
        phi = alpha * np.exp(-k / tau1) - beta * np.exp(-k / tau2)
        if phi[0] != 0:
            phi = phi / phi[0]
        return np.sum((empirical_kernel[: min(8, len(empirical_kernel))] - phi[: min(8, len(empirical_kernel))]) ** 2)

    best_loss = 1e10
    best_theta = None
    inits = [(1.0, 0.5, 2.0, 5.0), (1.5, 0.8, 1.5, 4.0), (0.8, 0.3, 3.0, 7.0)]
    for init in inits:
        result = minimize(
            loss, init, method="L-BFGS-B",
            bounds=[(0.5, 3.0), (0.1, 2.0), (0.5, 10.0), (1.0, 20.0)],
        )
        if result.fun < best_loss:
            best_loss = result.fun
            best_theta = result.x
    return best_theta


def decompose_cvxpy(
    b: np.ndarray,
    A_spec,
    lambda_smooth: float = 10.0,
    lambda_sparse: float = 8.0,
    use_second_order: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Solve convex decomposition problem via CVXPY.

    min ||b - (x_base + A_spec @ x_spec)||^2 + lambda_smooth^2 ||L x_base||^2 + lambda_sparse ||x_spec||_1

    Returns
    -------
    x_base, x_spec, objective_value
    """
    import cvxpy as cp

    T = len(b)
    # Guard against NaN/Inf in inputs
    if not np.all(np.isfinite(b)):
        import warnings
        warnings.warn("Input b contains NaN or Inf. Returning zeros.")
        return np.zeros(T), np.zeros(T), None
    if sparse.issparse(A_spec):
        if not np.all(np.isfinite(A_spec.data)):
            import warnings
            warnings.warn("Input A_spec contains NaN or Inf. Returning zeros.")
            return np.zeros(T), np.zeros(T), None
    else:
        if not np.all(np.isfinite(A_spec)):
            import warnings
            warnings.warn("Input A_spec contains NaN or Inf. Returning zeros.")
            return np.zeros(T), np.zeros(T), None

    x_base = cp.Variable(T)
    x_spec = cp.Variable(T)

    if use_second_order:
        L = build_stacked_difference_operator(T).toarray()
    else:
        L = build_difference_operator(T, order=1).toarray()

    objective = cp.Minimize(
        cp.sum_squares(b - (x_base + A_spec @ x_spec))
        + lambda_smooth * cp.sum_squares(L @ x_base)
        + lambda_sparse * cp.norm(x_spec, 1)
    )
    prob = cp.Problem(objective)
    try:
        prob.solve(solver=cp.OSQP, max_iter=10000, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, max_iters=5000, verbose=False)
        except Exception:
            try:
                prob.solve(verbose=False)
            except Exception:
                return np.zeros(T), np.zeros(T), None

    if x_base.value is None or x_spec.value is None:
        x_base.value = np.zeros(T)
        x_spec.value = np.zeros(T)

    return x_base.value, x_spec.value, prob.value


def decompose_ista(
    b: np.ndarray,
    A_spec,
    lambda_smooth: float = 10.0,
    lambda_sparse: float = 8.0,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast decomposition via iterative soft-thresholding (ISTA-like).
    Much faster than CVXPY for large-scale problems.

    Solves: min ||b - (x_base + A_spec @ x_spec)||^2 + lambda_smooth*||L2 x_base||^2 + lambda_sparse*||x_spec||_1
    """
    T = len(b)
    L2 = build_difference_operator(T, order=2).toarray()
    LtL = L2.T @ L2

    # Efficient Lipschitz constant estimation for sparse or dense A_spec
    if sparse.issparse(A_spec):
        # Use Frobenius norm as upper bound on spectral norm (cheaper than power iteration)
        A_norm_sq = np.sum(A_spec.data ** 2)
    else:
        A_norm_sq = np.linalg.norm(A_spec, 2) ** 2
    # Fast upper bound for ||L.T @ L||_2 via max row sum (Gershgorin)
    LtL_norm = float(np.max(np.abs(LtL).sum(axis=1)))
    lipschitz = 2.0 * (1 + A_norm_sq) + lambda_smooth * LtL_norm
    step = 1.0 / lipschitz

    x_base = np.zeros(T)
    x_spec = np.zeros(T)

    for _ in range(n_iter):
        recon = x_base + A_spec @ x_spec
        residual = recon - b

        grad_base = residual + lambda_smooth * (LtL @ x_base)
        grad_spec = A_spec.T @ residual

        x_base_new = x_base - step * grad_base
        x_spec_new = x_spec - step * grad_spec

        # Soft-thresholding for sparsity
        x_spec_new = np.sign(x_spec_new) * np.maximum(np.abs(x_spec_new) - step * lambda_sparse, 0)

        if np.max(np.abs(x_base_new - x_base)) < tol and np.max(np.abs(x_spec_new - x_spec)) < tol:
            break
        x_base = x_base_new
        x_spec = x_spec_new

    return x_base, x_spec


def compute_decomposition_quality(
    b: np.ndarray,
    x_base: np.ndarray,
    x_spec: np.ndarray,
    A_spec,
) -> Dict[str, float]:
    """Compute reconstruction RMSE, sparsity, and baseline smoothness."""
    recon = x_base + A_spec @ x_spec
    rmse = float(np.sqrt(np.mean((b - recon) ** 2)))
    sparsity = int(np.sum(np.abs(x_spec) > 1e-3))
    T = len(b)
    L1 = build_difference_operator(T, order=1).toarray()
    smoothness = float(np.linalg.norm(L1 @ x_base, 2))
    return {"rmse": rmse, "sparsity": sparsity, "smoothness": smoothness}


def kernel_from_theta(
    theta: np.ndarray, n_lags: int
) -> np.ndarray:
    """Build kernel array from parametric parameters [alpha, beta, tau1, tau2]."""
    alpha, beta, tau1, tau2 = theta
    k = np.arange(n_lags + 1)
    # Guard against very small tau values causing overflow
    tau1 = max(tau1, 0.1)
    tau2 = max(tau2, 0.1)
    phi = alpha * np.exp(-k / tau1) - beta * np.exp(-k / tau2)
    if phi[0] != 0 and not np.isnan(phi[0]) and not np.isinf(phi[0]):
        phi = phi / phi[0]
    else:
        phi = np.zeros_like(phi)
        phi[0] = 1.0
    return phi
