"""
Unit tests for core operators and solvers.
"""

import numpy as np
import unittest

from contextual_deconvolution.core import (
    build_difference_operator, build_stacked_difference_operator,
    build_Aspec, robust_detrend, estimate_kernel, fit_parametric_kernel,
    decompose_cvxpy, decompose_ista, compute_decomposition_quality,
    kernel_from_theta,
)


class TestOperators(unittest.TestCase):
    def test_difference_operator_shape(self):
        L1 = build_difference_operator(10, order=1)
        self.assertEqual(L1.shape, (9, 10))
        L2 = build_difference_operator(10, order=2)
        self.assertEqual(L2.shape, (8, 10))

    def test_stacked_difference_operator(self):
        L = build_stacked_difference_operator(10)
        self.assertEqual(L.shape, (17, 10))

    def test_Aspec_causal(self):
        phi = np.array([1.0, 0.5, 0.25])
        f = np.zeros(10)
        f[3] = 1.0
        A = build_Aspec(phi, np.zeros(3), f, np.zeros(10), 10, causal=True)
        # build_Aspec returns a scipy sparse matrix; densify for the triangular check
        A_d = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
        # A should be lower triangular
        self.assertTrue(np.all(np.triu(A_d, k=1) == 0))
        # Main diagonal at t=3 should be 1.0
        self.assertAlmostEqual(A[3, 3], 1.0)
        # Sub-diagonal: diag_vals[3] goes to A[3,2]
        self.assertAlmostEqual(A[3, 2], 0.5)
        # Second sub-diagonal: diag_vals[3] goes to A[3,1] for k=2
        self.assertAlmostEqual(A[3, 1], 0.25)
        # No entry at A[4,2] because f[4]=0
        self.assertAlmostEqual(A[4, 2], 0.0)

    def test_Aspec_noncausal(self):
        phi = np.array([1.0, 0.5])
        f = np.zeros(10)
        f[5] = 1.0
        A = build_Aspec(phi, np.zeros(2), f, np.zeros(10), 10, causal=False)
        A_d = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
        # Should have super-diagonals
        self.assertTrue(np.any(np.triu(A_d, k=1) != 0))


class TestRobustDetrend(unittest.TestCase):
    def test_smooth_output(self):
        t = np.arange(100)
        b = 10 + 0.05 * t + np.random.normal(0, 1, 100)
        mu = robust_detrend(b, lambda_smooth=10.0)
        self.assertEqual(len(mu), 100)
        # Trend should be monotonic-ish
        self.assertGreater(np.mean(np.diff(mu)), 0)

    def test_downweights_outliers(self):
        b = np.ones(50) * 10
        b[25] = 100  # outlier
        mu = robust_detrend(b, lambda_smooth=10.0)
        # mu[25] should not be pulled all the way to 100
        self.assertLess(mu[25], 50)


class TestKernelEstimation(unittest.TestCase):
    def test_estimate_kernel_on_synthetic(self):
        from contextual_deconvolution.synthetic import generate_synthetic_sku
        sku = generate_synthetic_sku(T=300, n_promos=10, seed=42)
        phi, n_events = estimate_kernel(sku["demand"], sku["f_promo"], 14, 30)
        self.assertIsNotNone(phi)
        self.assertGreater(n_events, 0)
        self.assertEqual(len(phi), 15)
        self.assertAlmostEqual(phi[0], 1.0, places=3)

    def test_fit_parametric_kernel(self):
        k = np.arange(15)
        phi_true = np.exp(-k / 2.0) - 0.6 * np.exp(-k / 5.0)
        phi_true = phi_true / phi_true[0]
        theta = fit_parametric_kernel(phi_true)
        self.assertIsNotNone(theta)
        self.assertEqual(len(theta), 4)

    def test_kernel_from_theta(self):
        theta = np.array([1.0, 0.6, 2.0, 5.0])
        phi = kernel_from_theta(theta, 14)
        self.assertEqual(len(phi), 15)
        self.assertAlmostEqual(phi[0], 1.0)


class TestDecomposition(unittest.TestCase):
    def test_cvxpy_reconstruction(self):
        from contextual_deconvolution.synthetic import generate_synthetic_sku
        sku = generate_synthetic_sku(T=100, n_promos=3, seed=42)
        b = sku["demand"]
        T = len(b)
        A = build_Aspec(sku["phi_promo"], sku["phi_snap"],
                        sku["f_promo"], sku["f_snap"], T, causal=True)
        xb, xs, obj = decompose_cvxpy(b, A, 10.0, 8.0)
        self.assertIsNotNone(xb)
        self.assertIsNotNone(xs)
        recon = xb + A @ xs
        rmse = np.sqrt(np.mean((b - recon) ** 2))
        self.assertLess(rmse, 10.0)  # Should reconstruct reasonably well

    def test_ista_matches_cvxpy_approx(self):
        from contextual_deconvolution.synthetic import generate_synthetic_sku
        sku = generate_synthetic_sku(T=100, n_promos=3, seed=42)
        b = sku["demand"]
        T = len(b)
        A = build_Aspec(sku["phi_promo"], sku["phi_snap"],
                        sku["f_promo"], sku["f_snap"], T, causal=True)
        xb_c, xs_c, _ = decompose_cvxpy(b, A, 10.0, 8.0)
        xb_i, xs_i = decompose_ista(b, A, 10.0, 8.0, n_iter=200)
        rmse_base = np.sqrt(np.mean((xb_c - xb_i) ** 2))
        rmse_spec = np.sqrt(np.mean((xs_c - xs_i) ** 2))
        # ISTA should be reasonably close to CVXPY (not exact, but same order of magnitude)
        self.assertLess(rmse_base, 50.0)
        self.assertLess(rmse_spec, 50.0)

    def test_decomposition_quality(self):
        from contextual_deconvolution.synthetic import generate_synthetic_sku
        sku = generate_synthetic_sku(T=100, n_promos=3, seed=42)
        b = sku["demand"]
        T = len(b)
        A = build_Aspec(sku["phi_promo"], sku["phi_snap"],
                        sku["f_promo"], sku["f_snap"], T, causal=True)
        xb, xs, _ = decompose_cvxpy(b, A, 10.0, 8.0)
        dq = compute_decomposition_quality(b, xb, xs, A)
        self.assertIn("rmse", dq)
        self.assertIn("sparsity", dq)
        self.assertIn("smoothness", dq)
        self.assertGreaterEqual(dq["sparsity"], 0)


if __name__ == "__main__":
    unittest.main()
