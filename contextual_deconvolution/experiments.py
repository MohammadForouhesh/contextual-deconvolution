"""
Evaluation framework: run full pipeline on synthetic or real data.
Optimized version with parallelization, optional baselines, and new metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import Ridge

from .core import (
    estimate_kernel, fit_parametric_kernel, build_Aspec,
    decompose_cvxpy, decompose_ista, compute_decomposition_quality,
    kernel_from_theta,
)
from .metrics import wmape, variance_ratio, inventory_cost, rmse, mae, spectral_flatness, max_drawdown

# Lazy import baselines to avoid loading heavy dependencies (e.g., xgboost) when not needed
_baselines_loaded = False

def _load_baselines():
    global _baselines_loaded, run_xgboost, run_tuned_xgboost, run_prophet_like, run_stl_arima
    if not _baselines_loaded:
        from .baselines import run_xgboost, run_tuned_xgboost, run_prophet_like, run_stl_arima
        _baselines_loaded = True


def estimate_all_kernels(dataset: List[Dict], config: Dict) -> Dict:
    """
    Estimate per-SKU kernels for all SKUs in the dataset.
    Returns a dict mapping sku_id -> {'phi_promo': ..., 'phi_snap': ..., 'theta_promo': ...}
    """
    from tqdm import tqdm
    kernels = {}
    for sku_data in tqdm(dataset, desc="Estimating per-SKU kernels"):
        sku_id = sku_data.get("sku_id", 0)
        b = sku_data["demand"]
        f_promo = sku_data["f_promo"]
        f_snap = sku_data["f_snap"]
        T = len(b)
        split = T - config["forecast_horizon"]
        b_train = b[:split]
        f_promo_train = f_promo[:split]
        f_snap_train = f_snap[:split]
        
        phi_promo_emp, n_promo = estimate_kernel(
            b_train, f_promo_train, config["n_lags"], config["pre_window"]
        )
        phi_snap_emp, n_snap = estimate_kernel(
            b_train, f_snap_train, config["n_lags"], config["pre_window"]
        )
        
        theta_promo = fit_parametric_kernel(phi_promo_emp, family="surge_vacuum")
        if theta_promo is not None:
            phi_promo = kernel_from_theta(theta_promo, config["n_lags"])
        else:
            phi_promo = np.zeros(config["n_lags"] + 1)
            phi_promo[0] = 1.0
        
        phi_snap = np.zeros(config["n_lags"] + 1)
        phi_snap[0] = 1.0
        
        kernels[sku_id] = {
            "phi_promo": phi_promo,
            "phi_snap": phi_snap,
            "theta_promo": theta_promo,
            "n_promo": n_promo,
            "n_snap": n_snap,
            "category": sku_data.get("category", "unknown"),
        }
    return kernels


def pool_kernels(kernels: Dict, strategy: str = "category") -> Dict:
    """
    Pool per-SKU kernels by category or globally.
    
    Parameters
    ----------
    kernels : dict
        Output from estimate_all_kernels.
    strategy : str
        "per_sku", "category", or "global"
    
    Returns
    -------
    dict mapping sku_id -> pooled kernel dict
    """
    if strategy == "per_sku":
        return kernels
    
    # Collect all phi_promo arrays
    all_phi_promo = np.stack([k["phi_promo"] for k in kernels.values()])
    
    # Global median kernel
    global_phi_promo = np.median(all_phi_promo, axis=0)
    
    # Category-level median kernels
    category_kernels = {}
    for sku_id, k in kernels.items():
        cat = k["category"]
        if cat not in category_kernels:
            category_kernels[cat] = []
        category_kernels[cat].append(k["phi_promo"])
    
    category_pooled = {}
    for cat, phi_list in category_kernels.items():
        category_pooled[cat] = np.median(np.stack(phi_list), axis=0)
    
    # Assign pooled kernels to each SKU
    pooled = {}
    for sku_id, k in kernels.items():
        cat = k["category"]
        if strategy == "global":
            phi_promo = global_phi_promo
        elif strategy == "category":
            phi_promo = category_pooled.get(cat, global_phi_promo)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        pooled[sku_id] = {
            "phi_promo": phi_promo,
            "phi_snap": k["phi_snap"],  # SNAP kernel stays as-is (usually impulse)
            "theta_promo": k["theta_promo"],
            "n_promo": k["n_promo"],
            "n_snap": k["n_snap"],
            "category": cat,
        }
    
    return pooled


def evaluate_sku_pooled(
    sku_data: Dict,
    phi_promo: np.ndarray,
    phi_snap: np.ndarray,
    config: Dict,
) -> Dict:
    """
    Evaluate a single SKU with pre-computed (possibly pooled) kernels.
    """
    b = sku_data["demand"]
    f_promo = sku_data["f_promo"]
    f_snap = sku_data["f_snap"]
    promo_depth = sku_data.get("promo_depth", np.zeros(len(b)))
    category = sku_data.get("category", "unknown")
    T = len(b)
    
    horizon = config["forecast_horizon"]
    split = T - horizon
    b_train, b_test = b[:split], b[split:]
    f_promo_test, f_snap_test = f_promo[split:], f_snap[split:]
    
    # Stage 2: Convex decomposition on the TRAINING window only (strictly out-of-sample:
    # the held-out test demand is never seen by the decomposition or the forecast).
    A_spec = build_Aspec(phi_promo, phi_snap, f_promo[:split], f_snap[:split], split, causal=True)
    if config.get("use_ista", False):
        x_base, x_spec = decompose_ista(
            b_train, A_spec, config["lambda_smooth"], config["lambda_sparse"]
        )
        obj_val = None
    else:
        x_base, x_spec, obj_val = decompose_cvxpy(
            b_train, A_spec, config["lambda_smooth"], config["lambda_sparse"],
            use_second_order=config.get("use_second_order", True),
        )

    decomp = compute_decomposition_quality(b_train, x_base, x_spec, A_spec)
    
    results = {
        "sku_id": sku_data.get("sku_id", 0),
        "category": category,
        "decomp": decomp,
        "x_base": x_base,
        "x_spec": x_spec,
        "phi_promo": phi_promo,
        "phi_snap": phi_snap,
        "f_promo": f_promo,
        "f_snap": f_snap,
        "promo_depth": promo_depth,
        "promo_starts": sku_data.get("promo_starts", []),
        "b_test": b_test,
    }
    
    # Two-stream forecasts
    ours_naive, _, _ = forecast_two_stream(
        x_base, x_spec, f_promo_test, f_snap_test, phi_promo, phi_snap
    )
    
    methods = {
        "CD": (ours_naive, b_test, x_base + A_spec @ x_spec),
    }
    
    # Baseline forecasts (optional)
    if config.get("run_baselines", True):
        _load_baselines()
        baseline_methods = config.get("baseline_methods", ["XGBoost", "Prophet", "STL+ARIMA"])
        if "Tuned XGBoost" in baseline_methods:
            xgb_pred, xgb_actual, xgb_fitted, _ = run_tuned_xgboost(b, f_promo, f_snap, horizon)
            if xgb_pred is not None:
                methods["Tuned XGBoost"] = (xgb_pred, xgb_actual, xgb_fitted)
        if "XGBoost" in baseline_methods:
            xgb_pred, xgb_actual, xgb_fitted, _ = run_xgboost(b, f_promo, f_snap, horizon)
            if xgb_pred is not None:
                methods["XGBoost"] = (xgb_pred, xgb_actual, xgb_fitted)
        if "Prophet" in baseline_methods:
            prop_pred, prop_actual, prop_fitted, _ = run_prophet_like(b, f_promo, f_snap, horizon)
            if prop_pred is not None:
                methods["Prophet"] = (prop_pred, prop_actual, prop_fitted)
        if "STL+ARIMA" in baseline_methods:
            stl_pred, stl_actual, stl_fitted, _ = run_stl_arima(b, f_promo, f_snap, horizon)
            if stl_pred is not None:
                methods["STL+ARIMA"] = (stl_pred, stl_actual, stl_fitted)
    
    # Compute metrics
    forecasts = {}
    for name, (pred, actual, fitted) in methods.items():
        if pred is None or actual is None:
            continue
        
        inv_cost, ss = inventory_cost(
            actual, pred, fitted,
            lead_time=config["lead_time"],
            h=config["holding_cost"],
            p=config["stockout_cost"],
            service_level=config["service_level"],
        )
        
        forecasts[name] = {
            "wmape": wmape(actual, pred),
            "variance_ratio": variance_ratio(pred, actual),
            "rmse": rmse(actual, pred),
            "mae": mae(actual, pred),
            "inventory_cost": inv_cost,
            "safety_stock": ss,
            "spectral_flatness": spectral_flatness(pred),
            "max_drawdown": max_drawdown(pred - actual),
            "pred": pred,
            "actual": actual,
        }
    
    results["forecasts"] = forecasts
    return results


CONFIG_DEFAULT = {
    "n_lags": 14,
    "pre_window": 30,
    "lambda_smooth": 10.0,
    "lambda_sparse": 8.0,
    "forecast_horizon": 28,
    "lead_time": 7,
    "holding_cost": 1.0,
    "stockout_cost": 10.0,
    "service_level": 0.95,
    "use_ista": True,
    "use_second_order": True,
    "run_ablations": False,      # Skip ablations for speed on large datasets
    "run_baselines": True,       # Can disable to skip XGBoost/Prophet/STL
    "baseline_methods": ["XGBoost", "Prophet", "STL+ARIMA"],
    "dl_baseline_methods": [],   # ["N-BEATS", "PatchTST", "DeepAR"]
    "device": "cpu",
}


def prescriptive_shock_model(decomposed_results: List[Dict]) -> Tuple[Optional[Ridge], Optional[List[str]]]:
    """Train Ridge regression on historically decomposed shocks."""
    rows = []
    for r in decomposed_results:
        sku_id = r.get("sku_id", 0)
        category = r.get("category", "unknown")
        x_spec = r["x_spec"]
        x_base = r["x_base"]
        f_promo = r["f_promo"]
        promo_depth = r.get("promo_depth", np.zeros(len(f_promo)))
        promo_starts = r.get("promo_starts", [])

        for start in promo_starts:
            if start >= len(x_spec):
                continue
            shock = x_spec[start]
            if shock < 1e-3:
                continue
            depth = promo_depth[start] if start < len(promo_depth) else 0.0
            dow = start % 7
            pre_base = np.median(x_base[max(0, start - 7):start]) if start > 0 else x_base[0]
            rows.append({
                "shock": shock, "depth": depth, "dow": dow,
                "category": category, "pre_base": pre_base, "sku_id": sku_id,
            })

    df = pd.DataFrame(rows)
    if len(df) < 10:
        return None, None

    df["dow_weekend"] = ((df["dow"] == 4) | (df["dow"] == 5)).astype(float)
    df["is_food"] = (df["category"] == "food").astype(float)
    feature_cols = ["depth", "dow_weekend", "is_food", "pre_base"]
    X = df[feature_cols].values
    y = df["shock"].values

    model = Ridge(alpha=10.0)
    model.fit(X, y)
    return model, feature_cols


def forecast_two_stream(
    x_base_hist: np.ndarray,
    x_spec_hist: np.ndarray,
    future_promo: np.ndarray,
    future_snap: np.ndarray,
    phi_promo: np.ndarray,
    phi_snap: np.ndarray,
    future_depth: Optional[np.ndarray] = None,
    shock_model: Optional[Ridge] = None,
    feature_cols: Optional[List[str]] = None,
    category: str = "food",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-Stream forecasting: baseline extrapolation + deterministic transient injection."""
    T_fut = len(future_promo)
    recent = x_base_hist[-30:]
    trend = (recent[-1] - recent[0]) / 30.0 if len(recent) > 1 else 0.0
    base_forecast = x_base_hist[-1] + trend * np.arange(1, T_fut + 1)

    future_shocks = np.zeros(T_fut)
    is_food = 1.0 if category == "food" else 0.0
    pre_base = np.median(x_base_hist[-7:]) if len(x_base_hist) > 0 else 0.0

    hist_shocks = x_spec_hist[x_spec_hist > 1e-3]
    avg_shock = np.median(hist_shocks) if len(hist_shocks) > 0 else np.std(x_base_hist) * 0.5

    for t in range(T_fut):
        if future_promo[t] > 0:
            if shock_model is not None and future_depth is not None:
                depth = future_depth[t] if t < len(future_depth) else 0.0
                dow = t % 7
                dow_weekend = 1.0 if dow in [4, 5] else 0.0
                x_vec = np.array([[depth, dow_weekend, is_food, pre_base]])
                pred_shock = shock_model.predict(x_vec)[0]
                future_shocks[t] = max(pred_shock, 0.0)
            else:
                future_shocks[t] = avg_shock
        elif future_snap[t] > 0:
            future_shocks[t] = avg_shock * 0.5

    A_spec_fut = build_Aspec(phi_promo, phi_snap, future_promo, future_snap, T_fut)
    forecast = base_forecast + A_spec_fut @ future_shocks
    return forecast, base_forecast, future_shocks


def evaluate_sku(
    sku_data: Dict,
    config: Dict,
) -> Dict:
    """
    Run full evaluation pipeline on a single SKU.
    Returns a dict with decomposition results, forecasts, and metrics.
    """
    b = sku_data["demand"]
    f_promo = sku_data["f_promo"]
    f_snap = sku_data["f_snap"]
    promo_depth = sku_data.get("promo_depth", np.zeros(len(b)))
    category = sku_data.get("category", "unknown")
    T = len(b)

    horizon = config["forecast_horizon"]
    split = T - horizon
    b_train, b_test = b[:split], b[split:]
    f_promo_train, f_promo_test = f_promo[:split], f_promo[split:]
    f_snap_train, f_snap_test = f_snap[:split], f_snap[split:]
    depth_train, depth_test = promo_depth[:split], promo_depth[split:]

    # Stage 1: Kernel estimation (on training data only)
    phi_promo_emp, n_promo = estimate_kernel(
        b_train, f_promo_train, config["n_lags"], config["pre_window"]
    )
    phi_snap_emp, n_snap = estimate_kernel(
        b_train, f_snap_train, config["n_lags"], config["pre_window"]
    )

    theta_promo = fit_parametric_kernel(phi_promo_emp, family="surge_vacuum")
    if theta_promo is not None:
        phi_promo = kernel_from_theta(theta_promo, config["n_lags"])
    else:
        phi_promo = np.zeros(config["n_lags"] + 1)
        phi_promo[0] = 1.0

    phi_snap = np.zeros(config["n_lags"] + 1)
    phi_snap[0] = 1.0

    # Stage 2: Convex decomposition on the TRAINING window only (strictly out-of-sample)
    A_spec = build_Aspec(phi_promo, phi_snap, f_promo[:split], f_snap[:split], split, causal=True)
    if config.get("use_ista", False):
        x_base, x_spec = decompose_ista(
            b_train, A_spec, config["lambda_smooth"], config["lambda_sparse"]
        )
        obj_val = None
    else:
        x_base, x_spec, obj_val = decompose_cvxpy(
            b_train, A_spec, config["lambda_smooth"], config["lambda_sparse"],
            use_second_order=config.get("use_second_order", True),
        )

    decomp = compute_decomposition_quality(b_train, x_base, x_spec, A_spec)

    # Ablations (optional, skip for large datasets)
    if config.get("run_ablations", False):
        A_id = np.eye(T)
        x_base_id, x_spec_id, _ = decompose_cvxpy(
            b, A_id, config["lambda_smooth"], config["lambda_sparse"],
            use_second_order=config.get("use_second_order", True),
        )
        decomp_id = compute_decomposition_quality(b, x_base_id, x_spec_id, A_id)

        A_nc = build_Aspec(phi_promo, phi_snap, f_promo, f_snap, T, causal=False)
        x_base_nc, x_spec_nc, _ = decompose_cvxpy(
            b, A_nc, config["lambda_smooth"], config["lambda_sparse"],
            use_second_order=config.get("use_second_order", True),
        )
        decomp_nc = compute_decomposition_quality(b, x_base_nc, x_spec_nc, A_nc)
    else:
        decomp_id = decomp_nc = None
        x_base_id = x_spec_id = x_base_nc = x_spec_nc = None

    results = {
        "sku_id": sku_data.get("sku_id", 0),
        "category": category,
        "decomp": decomp,
        "decomp_id": decomp_id,
        "decomp_nc": decomp_nc,
        "x_base": x_base,
        "x_spec": x_spec,
        "phi_promo": phi_promo,
        "phi_snap": phi_snap,
        "theta_promo": theta_promo,
        "f_promo": f_promo,
        "f_snap": f_snap,
        "promo_depth": promo_depth,
        "promo_starts": sku_data.get("promo_starts", []),
        "b_test": b_test,
        "n_promo": n_promo,
        "n_snap": n_snap,
    }

    # Two-stream forecasts (naive) - always compute
    ours_naive, _, _ = forecast_two_stream(
        x_base, x_spec, f_promo_test, f_snap_test, phi_promo, phi_snap
    )

    # Build methods dict
    methods = {
        "CD": (ours_naive, b_test, x_base + A_spec @ x_spec),
    }

    if config.get("run_ablations", False):
        ours_id, _, _ = forecast_two_stream(
            x_base_id, x_spec_id, f_promo_test, f_snap_test, phi_promo, phi_snap
        )
        ours_nc, _, _ = forecast_two_stream(
            x_base_nc, x_spec_nc, f_promo_test, f_snap_test, phi_promo, phi_snap
        )
        methods["CD (Identity)"] = (ours_id, b_test, x_base_id + np.eye(T) @ x_spec_id)
        methods["CD (Non-Causal)"] = (ours_nc, b_test, x_base_nc + A_nc @ x_spec_nc)

    # Baseline forecasts (optional)
    if config.get("run_baselines", True):
        _load_baselines()
        baseline_methods = config.get("baseline_methods", ["XGBoost", "Prophet", "STL+ARIMA"])
        if "Tuned XGBoost" in baseline_methods:
            xgb_pred, xgb_actual, xgb_fitted, _ = run_tuned_xgboost(b, f_promo, f_snap, horizon)
            if xgb_pred is not None:
                methods["Tuned XGBoost"] = (xgb_pred, xgb_actual, xgb_fitted)
        if "XGBoost" in baseline_methods:
            xgb_pred, xgb_actual, xgb_fitted, _ = run_xgboost(b, f_promo, f_snap, horizon)
            if xgb_pred is not None:
                methods["XGBoost"] = (xgb_pred, xgb_actual, xgb_fitted)
        if "Prophet" in baseline_methods:
            prop_pred, prop_actual, prop_fitted, _ = run_prophet_like(b, f_promo, f_snap, horizon)
            if prop_pred is not None:
                methods["Prophet"] = (prop_pred, prop_actual, prop_fitted)
        if "STL+ARIMA" in baseline_methods:
            stl_pred, stl_actual, stl_fitted, _ = run_stl_arima(b, f_promo, f_snap, horizon)
            if stl_pred is not None:
                methods["STL+ARIMA"] = (stl_pred, stl_actual, stl_fitted)
        # Deep learning baselines (PyTorch)
        dl_methods = config.get("dl_baseline_methods", [])
        if dl_methods:
            from .baselines_dl import run_nbeats, run_patchtst, run_deepar
            device = config.get("device", "cpu")
            if "N-BEATS" in dl_methods:
                nb_pred, nb_actual, nb_fitted, _ = run_nbeats(b, f_promo, f_snap, horizon, device=device)
                if nb_pred is not None:
                    methods["N-BEATS"] = (nb_pred, nb_actual, nb_fitted)
            if "PatchTST" in dl_methods:
                pt_pred, pt_actual, pt_fitted, _ = run_patchtst(b, f_promo, f_snap, horizon, device=device)
                if pt_pred is not None:
                    methods["PatchTST"] = (pt_pred, pt_actual, pt_fitted)
            if "DeepAR" in dl_methods:
                da_pred, da_actual, da_fitted, _ = run_deepar(b, f_promo, f_snap, horizon, device=device)
                if da_pred is not None:
                    methods["DeepAR"] = (da_pred, da_actual, da_fitted)

    # Compute metrics for all methods
    forecasts = {}
    for name, (pred, actual, fitted) in methods.items():
        if pred is None or actual is None:
            continue
        
        inv_cost, ss = inventory_cost(
            actual, pred, fitted,
            lead_time=config["lead_time"],
            h=config["holding_cost"],
            p=config["stockout_cost"],
            service_level=config["service_level"],
        )
        
        forecasts[name] = {
            "wmape": wmape(actual, pred),
            "variance_ratio": variance_ratio(pred, actual),
            "rmse": rmse(actual, pred),
            "mae": mae(actual, pred),
            "inventory_cost": inv_cost,
            "safety_stock": ss,
            "spectral_flatness": spectral_flatness(pred),
            "max_drawdown": max_drawdown(pred - actual),
            "pred": pred,
            "actual": actual,
        }

    results["forecasts"] = forecasts
    return results


def _evaluate_one(args):
    """Wrapper for parallel evaluation."""
    sku_data, config = args
    try:
        return evaluate_sku(sku_data, config)
    except Exception as e:
        import warnings
        warnings.warn(f"Evaluation failed for SKU {sku_data.get('sku_id', 'unknown')}: {e}")
        return None


def run_evaluation(
    dataset: List[Dict],
    config: Optional[Dict] = None,
    n_jobs: int = -1,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Run full evaluation on a dataset with optional parallelization.

    Parameters
    ----------
    dataset : list of dicts
    config : dict or None
    n_jobs : int
        Number of parallel jobs. -1 uses all CPUs. 1 = serial.

    Returns
    -------
    results : list of dicts
    summary : DataFrame
    """
    if config is None:
        config = CONFIG_DEFAULT.copy()

    if n_jobs == 1 or len(dataset) == 1:
        # Serial execution
        results = []
        for sku_data in dataset:
            r = evaluate_sku(sku_data, config)
            results.append(r)
    else:
        # Parallel execution
        try:
            from joblib import Parallel, delayed
            args = [(sku, config) for sku in dataset]
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_evaluate_one)(arg) for arg in args
            )
            results = [r for r in results if r is not None]
        except ImportError:
            import warnings
            warnings.warn("joblib not available, falling back to serial execution")
            results = []
            for sku_data in dataset:
                r = evaluate_sku(sku_data, config)
                results.append(r)

    # Build summary DataFrame
    rows = []
    for r in results:
        if r is None:
            continue
        sku_id = r["sku_id"]
        category = r["category"]
        for method, metrics in r["forecasts"].items():
            rows.append({
                "sku_id": sku_id,
                "category": category,
                "method": method,
                **{k: v for k, v in metrics.items() if k not in ("pred", "actual")},
            })

    summary = pd.DataFrame(rows)
    return results, summary
