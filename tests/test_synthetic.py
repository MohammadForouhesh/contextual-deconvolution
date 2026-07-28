"""
Unit tests for synthetic data generation.
"""

import numpy as np
import unittest

from contextual_deconvolution.synthetic import (
    generate_synthetic_sku, generate_stratified_dataset,
)


class TestSyntheticSKU(unittest.TestCase):
    def test_basic_shape(self):
        sku = generate_synthetic_sku(T=100, n_promos=3, seed=42)
        self.assertEqual(len(sku["demand"]), 100)
        self.assertEqual(len(sku["baseline_true"]), 100)
        self.assertEqual(len(sku["x_spec_true"]), 100)
        self.assertEqual(len(sku["f_promo"]), 100)
        self.assertEqual(len(sku["f_snap"]), 100)
        self.assertEqual(len(sku["promo_depth"]), 100)

    def test_demand_nonnegative(self):
        sku = generate_synthetic_sku(T=100, n_promos=3, seed=42)
        self.assertTrue(np.all(sku["demand"] >= 0))

    def test_promo_events_present(self):
        sku = generate_synthetic_sku(T=200, n_promos=5, seed=42)
        self.assertGreater(np.sum(sku["f_promo"] > 0), 0)
        self.assertEqual(len(sku["promo_starts"]), 5)

    def test_ground_truth_reconstructs(self):
        sku = generate_synthetic_sku(T=200, n_promos=5, seed=42, noise_std=0.0)
        recon = sku["baseline_true"] + sku["A_spec_true"] @ sku["x_spec_true"]
        np.testing.assert_allclose(sku["demand"], recon, atol=1e-6)

    def test_snap_events_present(self):
        sku = generate_synthetic_sku(T=200, n_snaps=4, seed=42)
        self.assertGreater(np.sum(sku["f_snap"] > 0), 0)

    def test_kernel_properties(self):
        sku = generate_synthetic_sku(T=100, n_promos=3, seed=42)
        phi = sku["phi_promo"]
        # Surge at k=0
        self.assertGreater(phi[0], 0)
        # Vacuum tail eventually negative
        self.assertTrue(np.any(phi < 0))


class TestStratifiedDataset(unittest.TestCase):
    def test_dataset_size(self):
        data = generate_stratified_dataset(n_skus=30, T=100, seed=42)
        self.assertEqual(len(data), 30)

    def test_stratification(self):
        data = generate_stratified_dataset(n_skus=30, T=100, seed=42)
        categories = [d["category"] for d in data]
        self.assertIn("food", categories)
        self.assertIn("household", categories)
        volumes = [d["volume_stratum"] for d in data]
        self.assertIn("high", volumes)
        self.assertIn("low", volumes)

    def test_different_kernels(self):
        data = generate_stratified_dataset(n_skus=10, T=100, seed=42)
        # Food and household should have different kernel parameters
        food_params = [d["kernel_params"] for d in data if d["category"] == "food"]
        hh_params = [d["kernel_params"] for d in data if d["category"] == "household"]
        if len(food_params) > 0 and len(hh_params) > 0:
            self.assertNotEqual(food_params[0]["tau1"], hh_params[0]["tau1"])


if __name__ == "__main__":
    unittest.main()
