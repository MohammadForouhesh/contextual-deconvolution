"""
Synthetic data generation with known ground truth for falsification and evaluation.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple


def generate_synthetic_sku(
    T: int = 730,
    item_level: float = 100.0,
    trend_slope: float = 0.02,
    n_promos: int = 12,
    promo_duration_range: Tuple[int, int] = (3, 8),
    promo_depth_range: Tuple[float, float] = (0.05, 0.50),
    n_snaps: int = 24,
    kernel_params: Dict[str, float] = None,
    noise_std: float = 4.0,
    seed: Optional[int] = None,
) -> Dict:
    """
    Generate a single synthetic SKU with known ground truth.

    Parameters
    ----------
    T : int
        Number of time steps.
    item_level : float
        Base demand level.
    trend_slope : float
        Linear trend per day.
    n_promos : int
        Number of promotional events.
    promo_duration_range : tuple
        (min, max) duration of each promo in days.
    promo_depth_range : tuple
        (min, max) discount fraction.
    n_snaps : int
        Number of SNAP events (uniformly spaced if possible).
    kernel_params : dict
        Keys: alpha, beta, tau1, tau2 for surge-and-vacuum kernel.
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int or None
        Random seed.

    Returns
    -------
    data : dict
        Contains demand, baseline, x_spec, f_promo, f_snap, promo_depth, kernel, etc.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(T)
    dow = t % 7
    month = (t // 30) % 12

    # Structural baseline
    trend = 5 + trend_slope * t
    weekly = 8 * np.sin(2 * np.pi * dow / 7 - np.pi / 2)
    annual = 12 * np.sin(2 * np.pi * month / 12)
    baseline = item_level + trend + weekly + annual

    # Promotions
    f_promo = np.zeros(T)
    promo_depth = np.zeros(T)
    promo_starts = []
    promo_durations = []
    promo_depths_list = []

    for _ in range(n_promos):
        start = np.random.randint(20, T - 30)
        duration = np.random.randint(*promo_duration_range)
        depth = np.random.uniform(*promo_depth_range)
        f_promo[start : start + duration] = 1
        promo_depth[start : start + duration] = depth
        promo_starts.append(start)
        promo_durations.append(duration)
        promo_depths_list.append(depth)

    # SNAP days
    f_snap = np.zeros(T)
    snap_days_list = []
    snap_spacing = max(T // max(n_snaps, 1), 14)
    for snap_day in range(snap_spacing, T, snap_spacing):
        if snap_day < T:
            f_snap[snap_day] = 1
            snap_days_list.append(snap_day)

    # Default kernel parameters
    if kernel_params is None:
        kernel_params = {"alpha": 1.0, "beta": 0.6, "tau1": 2.0, "tau2": 5.0}
    alpha = kernel_params["alpha"]
    beta = kernel_params["beta"]
    tau1 = kernel_params["tau1"]
    tau2 = kernel_params["tau2"]

    n_lags = 14
    k = np.arange(n_lags + 1)
    phi_promo = alpha * np.exp(-k / tau1) - beta * np.exp(-k / tau2)
    phi_snap = np.zeros(n_lags + 1)
    phi_snap[0] = 1.0

    # Build ground-truth A_spec
    A_spec = np.zeros((T, T))
    for lag in range(n_lags + 1):
        diag_vals = phi_promo[lag] * f_promo + phi_snap[lag] * f_snap
        if lag < T:
            idx = np.arange(T - lag)
            A_spec[idx + lag, idx] += diag_vals[lag:]

    # Generate latent shocks with physical scaling
    x_spec_true = np.zeros(T)
    for i, start in enumerate(promo_starts):
        depth = promo_depths_list[i]
        depth_mult = 3.2 * (depth / 0.5) ** 0.6
        dow_start = start % 7
        dow_mult = 1.4 if dow_start in [4, 5] else 1.0
        shock = 30.0 * depth_mult * dow_mult
        x_spec_true[start] = shock

    for snap_day in snap_days_list:
        x_spec_true[snap_day] = np.random.uniform(15, 25)

    # Observed demand
    noise = np.random.normal(0, noise_std, T)
    demand = baseline + A_spec @ x_spec_true + noise
    demand = np.maximum(demand, 0)

    return {
        "demand": demand,
        "baseline_true": baseline,
        "x_spec_true": x_spec_true,
        "f_promo": f_promo,
        "f_snap": f_snap,
        "promo_depth": promo_depth,
        "promo_starts": promo_starts,
        "promo_durations": promo_durations,
        "promo_depths_list": promo_depths_list,
        "snap_days": snap_days_list,
        "phi_promo": phi_promo,
        "phi_snap": phi_snap,
        "kernel_params": kernel_params,
        "A_spec_true": A_spec,
        "noise_std": noise_std,
    }


def generate_stratified_dataset(
    n_skus: int = 30,
    T: int = 730,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate a stratified synthetic dataset with multiple SKU types.

    Strata:
    - Volume: high / medium / low
    - Promo frequency: high / low
    - Category: food / household (different kernel parameters)
    """
    np.random.seed(seed)
    volumes = ["high", "medium", "low"] * (n_skus // 3 + 3)
    promo_freqs = ["high", "low"] * (n_skus // 2 + 3)
    categories = ["food", "household"] * (n_skus // 2 + 3)

    data = []
    for sku_id in range(n_skus):
        vol = volumes[sku_id]
        pfreq = promo_freqs[sku_id]
        category = categories[sku_id]

        if vol == "high":
            item_level = np.random.uniform(80, 150)
            trend_slope = 0.03
        elif vol == "medium":
            item_level = np.random.uniform(40, 80)
            trend_slope = 0.02
        else:
            item_level = np.random.uniform(10, 40)
            trend_slope = 0.01

        if pfreq == "high":
            n_promos = np.random.poisson(24)
        else:
            n_promos = np.random.poisson(6)

        # Category-dependent kernel physics
        if category == "food":
            tau1 = np.random.uniform(1.5, 2.5)
            tau2 = np.random.uniform(3.5, 5.5)
            cat_mult = 1.2
        else:
            tau1 = np.random.uniform(2.0, 3.5)
            tau2 = np.random.uniform(5.0, 8.0)
            cat_mult = 0.8

        kernel_params = {"alpha": 1.0, "beta": 0.6, "tau1": tau1, "tau2": tau2}

        sku_data = generate_synthetic_sku(
            T=T,
            item_level=item_level,
            trend_slope=trend_slope,
            n_promos=n_promos,
            kernel_params=kernel_params,
            seed=seed + sku_id,
        )
        sku_data["sku_id"] = sku_id
        sku_data["category"] = category
        sku_data["volume_stratum"] = vol
        sku_data["promo_freq"] = pfreq
        sku_data["cat_mult"] = cat_mult

        # Apply category multiplier to shocks
        for i in range(len(sku_data["promo_starts"])):
            sku_data["x_spec_true"][sku_data["promo_starts"][i]] *= cat_mult

        data.append(sku_data)

    return data
