#!/usr/bin/env python3
"""Dictionary-robustness of Koopman non-normality.

Reviewer concern: is the observed non-normality of EDMD-learned Koopman operators
an artifact of the (degree-2 polynomial) observable dictionary, or intrinsic to the
promotional-demand dynamics?

We fit the pooled Koopman operator under a range of observable dictionaries
(linear, quadratic, full-quadratic-with-cross-terms, cubic) and, optionally,
different delay-embedding dimensions, and measure non-normality for each:
  - spectral radius rho, spectral abscissa alpha, numerical abscissa omega
  - departure from normality (omega - alpha)
  - Henrici number ||K*K - KK*||_F / ||K||_F^2
  - max transient-growth factor  max_k ||K^k||_2 / rho^k
A normal operator has departure ~ 0, Henrici ~ 0, transient factor ~ 1.
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from contextual_deconvolution.loaders import load_m5_data

warnings.filterwarnings("ignore")
N_LAGS = 14
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "m5-data", "m5-forecasting-accuracy")


# ---------- snapshots + EDMD ----------
def build_snapshots(demand, promo, snap, n_lag=N_LAGS):
    T = len(demand)
    X, Y, U = [], [], []
    for t in range(n_lag, T - 1):
        X.append(demand[t - n_lag:t + 1])
        Y.append(demand[t - n_lag + 1:t + 2])
        U.append([promo[t], snap[t]])
    return np.array(X), np.array(Y), np.array(U)


def f_linear(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def f_quad_v1(X):
    return np.hstack([np.ones((X.shape[0], 1)), X, X ** 2])


def f_quad_full(X):
    n = X.shape[1]
    terms = [np.ones((X.shape[0], 1)), X]
    for i in range(n):
        for j in range(i, n):
            terms.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(terms)


def f_cubic(X):
    return np.hstack([np.ones((X.shape[0], 1)), X, X ** 2, X ** 3])


def edmd_K(X, Y, U, feat, ridge=1.0):
    Psi_X, Psi_Y = feat(X), feat(Y)
    Omega = np.hstack([Psi_X, U])
    d = Psi_X.shape[1]
    A = Omega.T @ Omega + ridge * np.eye(Omega.shape[1])
    Xi = np.linalg.solve(A, Omega.T @ Psi_Y)
    return Xi[:d, :].T  # Koopman matrix K on observables


def fit_pooled_K(skus, feat, n_lag=N_LAGS, ridge=1.0):
    Xs, Ys, Us = [], [], []
    for s in skus:
        b = s["demand"]; split = len(b) - 28
        bt = b[:split].astype(float)
        mu = bt.mean()
        if mu > 1e-6:
            bt = bt / mu
        X, Y, U = build_snapshots(bt, s["f_promo"][:split], s["f_snap"][:split], n_lag)
        if len(X) >= 30:
            Xs.append(X); Ys.append(Y); Us.append(U)
    if not Xs:
        return None
    return edmd_K(np.vstack(Xs), np.vstack(Ys), np.vstack(Us), feat, ridge)


# ---------- non-normality metrics ----------
def spectral_abscissa(A):
    return float(np.max(np.real(np.linalg.eigvals(A))))


def numerical_abscissa(A):
    H = 0.5 * (A + A.conj().T)
    return float(np.max(np.real(np.linalg.eigvals(H))))


def henrici(A):
    comm = A.conj().T @ A - A @ A.conj().T
    return float(np.linalg.norm(comm, "fro") / (np.linalg.norm(A, "fro") ** 2 + 1e-12))


def max_transient_factor(K, k_max=20):
    rho = np.max(np.abs(np.linalg.eigvals(K)))
    Kp = np.eye(K.shape[0])
    best = 0.0
    for k in range(1, k_max + 1):
        Kp = Kp @ K
        fac = np.linalg.norm(Kp, 2) / (rho ** k + 1e-12)
        best = max(best, fac)
    return float(best), float(rho)


def analyze(K):
    alpha = spectral_abscissa(K)
    omega = numerical_abscissa(K)
    tf, rho = max_transient_factor(K)
    return {
        "dim": int(K.shape[0]),
        "rho": round(rho, 4),
        "alpha": round(alpha, 4),
        "omega": round(omega, 4),
        "departure_omega_minus_alpha": round(omega - alpha, 4),
        "henrici": round(henrici(K), 4),
        "max_transient_factor": round(tf, 2),
    }


def main():
    print("Loading M5 (120 SKUs, skip prices)...")
    data = load_m5_data(DATA, n_skus=120, random_state=42, skip_prices=True)
    cats = {}
    for s in data:
        cats.setdefault(s.get("category", "unknown"), []).append(s)

    dicts = [
        ("linear", f_linear),
        ("quadratic (deg<=2)", f_quad_v1),
        ("full-quadratic+cross", f_quad_full),
        ("cubic (deg<=3)", f_cubic),
    ]

    out = {"global": {}, "categories": {}}
    print("\n=== GLOBAL pooled operator ===")
    print(f"{'dictionary':24s} {'dim':>4} {'rho':>7} {'alpha':>7} {'omega':>7} "
          f"{'omega-alpha':>11} {'henrici':>8} {'transient':>9}")
    for name, feat in dicts:
        K = fit_pooled_K(data, feat)
        m = analyze(K)
        out["global"][name] = m
        print(f"{name:24s} {m['dim']:>4} {m['rho']:>7.3f} {m['alpha']:>7.3f} "
              f"{m['omega']:>7.3f} {m['departure_omega_minus_alpha']:>11.3f} "
              f"{m['henrici']:>8.3f} {m['max_transient_factor']:>9.2f}")

    print("\n=== per-CATEGORY, quadratic dictionary (transient factor range) ===")
    for c, skus in cats.items():
        K = fit_pooled_K(skus, f_quad_v1)
        if K is None:
            continue
        m = analyze(K)
        out["categories"][c] = m
        print(f"  {c:12s}: rho={m['rho']:.3f}  transient={m['max_transient_factor']:.2f}  "
              f"henrici={m['henrici']:.3f}  departure={m['departure_omega_minus_alpha']:.3f}")

    with open(os.path.join(os.path.dirname(__file__), "..", "paper", "results",
                           "edmd_dictionary_robustness.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> paper/results/edmd_dictionary_robustness.json")


if __name__ == "__main__":
    main()
