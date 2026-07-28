#!/usr/bin/env python3
"""
Run four structural baseline forecasting methods requested by the reviewer:
1. ARIMAX with promotional dummies
2. ETS with covariates
3. Prophet with event regressors (or Prophet-like if fbprophet not installed)
4. Simple distributed-lag OLS (Ridge regression on lagged promotional features)

Usage:
    python3 code/run_structural_baselines.py
"""

import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List

warnings.filterwarnings('ignore')

sys.path.insert(0, _os.path.join(_ROOT, "code"))

from contextual_deconvolution.loaders import load_m5_data, load_favorita_data
from contextual_deconvolution.metrics import wmape, variance_ratio, rmse

# statsmodels is required
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------------------
# Feature builders
# ------------------------------------------------------------------------------

def build_distributed_lag_features(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    n_lags: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix with distributed lag features.
    For each day t >= n_lags:
    - promo_lag0, ..., promo_lag_n (f_promo[t], ..., f_promo[t-n_lags])
    - snap_lag0, ..., snap_lag_n (f_snap[t], ..., f_snap[t-n_lags])
    - demand_lag1, demand_lag7, demand_lag14
    - calendar features: day_of_week, month, time_index
    """
    T = len(b)
    features = []
    for t in range(n_lags, T):
        feat = []
        for lag in range(n_lags + 1):
            feat.append(f_promo[t - lag])
        for lag in range(n_lags + 1):
            feat.append(f_snap[t - lag])
        feat.append(b[t - 1])
        feat.append(b[t - 7])
        feat.append(b[t - 14])
        feat.append(t % 7)
        feat.append((t // 30) % 12)
        feat.append(t)
        features.append(feat)
    return np.array(features), np.arange(n_lags, T)


def _dl_feature_row(dem, f_promo, f_snap, t, n_lags):
    """One distributed-lag feature row for absolute day t (demand lags from `dem`;
    promo/snap lags and calendar from the known exogenous inputs)."""
    feat = [f_promo[t - lag] for lag in range(n_lags + 1)]
    feat += [f_snap[t - lag] for lag in range(n_lags + 1)]
    feat += [dem[t - 1], dem[t - 7], dem[t - 14]]
    feat += [t % 7, (t // 30) % 12, t]
    return feat


def build_prophet_features(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
) -> np.ndarray:
    """Build Prophet-like Fourier + event regressors."""
    T = len(b)
    t = np.arange(T)
    X = [t]
    for period in [7]:
        for i in range(1, 4):
            X.append(np.sin(2 * np.pi * i * t / period))
            X.append(np.cos(2 * np.pi * i * t / period))
    for period in [365]:
        for i in range(1, 3):
            X.append(np.sin(2 * np.pi * i * t / period))
            X.append(np.cos(2 * np.pi * i * t / period))
    X = np.column_stack(X + [f_promo, f_snap])
    return X


# ------------------------------------------------------------------------------
# Baseline 1: ARIMAX with promotional dummies
# ------------------------------------------------------------------------------

def run_arimax(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """
    ARIMA(1,1,1) with exogenous regressors [f_promo, f_snap].
    Falls back to simpler orders if fitting fails.
    """
    T = len(b)
    split = T - forecast_horizon
    b_train = b[:split]
    X_train = np.column_stack([f_promo[:split], f_snap[:split]])
    X_test  = np.column_stack([f_promo[split:T], f_snap[split:T]])

    orders_to_try = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 1, 0)]
    for order in orders_to_try:
        try:
            model = ARIMA(b_train, exog=X_train, order=order)
            fit = model.fit(method_kwargs={'warn_convergence': False})
            pred = fit.forecast(steps=forecast_horizon, exog=X_test)
            fitted_train = fit.fittedvalues
            return pred, b[split:], fitted_train, fit
        except Exception:
            continue
    return None, None, None, None


# ------------------------------------------------------------------------------
# Baseline 2: ETS with covariates (two-step approach)
# ------------------------------------------------------------------------------

def run_ets_covariates(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """
    Two-step ETS with covariates:
    1. Fit Holt-Winters (or Holt) on demand series.
    2. Regress residuals on promo/snap to get event effects.
    3. Add event effects to the ETS forecast.
    """
    T = len(b)
    split = T - forecast_horizon
    train = b[:split]

    # Step 1: Fit ETS
    try:
        hw = ExponentialSmoothing(
            train, seasonal_periods=7, trend='add', seasonal='add', damped_trend=False
        )
        hw_fit = hw.fit(optimized=True, remove_bias=True)
    except Exception:
        try:
            hw = ExponentialSmoothing(
                train, seasonal_periods=7, trend='add', seasonal='add'
            )
            hw_fit = hw.fit(optimized=True)
        except Exception:
            try:
                hw = ExponentialSmoothing(train, trend='add')
                hw_fit = hw.fit(optimized=True)
            except Exception:
                return None, None, None, None

    # Forecast from ETS
    try:
        hw_fc = hw_fit.forecast(steps=forecast_horizon)
    except Exception:
        return None, None, None, None

    # Step 2: Regress residuals on promo/snap in training
    residuals = train - hw_fit.fittedvalues
    X_train_resid = np.column_stack([f_promo[:split], f_snap[:split]])
    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_resid, residuals)
        event_effect_train = ridge.predict(X_train_resid)
        event_effect_test = ridge.predict(np.column_stack([f_promo[split:T], f_snap[split:T]]))
    except Exception:
        event_effect_train = np.zeros(split)
        event_effect_test = np.zeros(forecast_horizon)

    pred = hw_fc + event_effect_test
    fitted_train = hw_fit.fittedvalues + event_effect_train
    return pred, b[split:], fitted_train, hw_fit


# ------------------------------------------------------------------------------
# Baseline 3: Prophet with event regressors (or Prophet-like fallback)
# ------------------------------------------------------------------------------

def run_prophet_or_prophet_like(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """
    Try to use actual Prophet library; if not available, fall back to the
    existing Prophet-like Fourier+Ridge model from baselines.py.
    """
    try:
        from prophet import Prophet
        HAS_PROPHET = True
    except ImportError:
        try:
            from fbprophet import Prophet
            HAS_PROPHET = True
        except ImportError:
            HAS_PROPHET = False

    T = len(b)
    split = T - forecast_horizon
    dates = pd.date_range(start='2011-01-29', periods=T, freq='D')  # M5 start date proxy

    if HAS_PROPHET:
        try:
            df = pd.DataFrame({
                'ds': dates[:split],
                'y': b[:split],
                'promo': f_promo[:split],
                'snap': f_snap[:split],
            })
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
            )
            m.add_regressor('promo', prior_scale=10.0, mode='additive')
            m.add_regressor('snap', prior_scale=10.0, mode='additive')
            m.fit(df)

            future = pd.DataFrame({'ds': dates[split:T]})
            future['promo'] = f_promo[split:T]
            future['snap'] = f_snap[split:T]
            forecast = m.predict(future)
            pred = forecast['yhat'].values

            # fitted values on training
            train_future = pd.DataFrame({'ds': dates[:split]})
            train_future['promo'] = f_promo[:split]
            train_future['snap'] = f_snap[:split]
            fitted_train = m.predict(train_future)['yhat'].values
            return pred, b[split:], fitted_train, m
        except Exception as e:
            print(f"  Prophet failed: {e}, falling back to Prophet-like")

    # Fallback: Prophet-like Fourier + Ridge
    X = build_prophet_features(b, f_promo, f_snap)
    model = Ridge(alpha=1.0)
    model.fit(X[:split], b[:split])
    t_fut = np.arange(T, T + forecast_horizon)
    X_fut = [t_fut]
    for period in [7]:
        for i in range(1, 4):
            X_fut.append(np.sin(2 * np.pi * i * t_fut / period))
            X_fut.append(np.cos(2 * np.pi * i * t_fut / period))
    for period in [365]:
        for i in range(1, 3):
            X_fut.append(np.sin(2 * np.pi * i * t_fut / period))
            X_fut.append(np.cos(2 * np.pi * i * t_fut / period))
    X_fut = np.column_stack(X_fut + [f_promo[-forecast_horizon:], f_snap[-forecast_horizon:]])
    y_pred = model.predict(X_fut)
    fitted_train = model.predict(X[:split])
    return y_pred, b[split:], fitted_train, model


# ------------------------------------------------------------------------------
# Baseline 4: Simple distributed-lag OLS (Ridge)
# ------------------------------------------------------------------------------

def run_distributed_lag_ridge(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
    n_lags: int = 14,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """
    Ridge regression on distributed lag features (promo, snap, demand lags, calendar).
    """
    T = len(b)
    split = T - forecast_horizon
    if split <= n_lags + 50:
        return None, None, None, None
    X, idx = build_distributed_lag_features(b[:split], f_promo[:split], f_snap[:split], n_lags)
    if len(X) == 0:
        return None, None, None, None
    y = b[:split][idx]

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    fitted_train = model.predict(X)

    # Recursive multi-step forecast (demand lags from own predictions; calendar known)
    dem = list(b[:split])
    y_pred = []
    for h in range(forecast_horizon):
        t = split + h
        row = _dl_feature_row(np.asarray(dem), f_promo, f_snap, t, n_lags)
        p = float(model.predict(np.array([row]))[0])
        y_pred.append(p)
        dem.append(p)
    return np.asarray(y_pred), b[split:], fitted_train, model


# ------------------------------------------------------------------------------
# Evaluation harness
# ------------------------------------------------------------------------------

def evaluate_method_on_sku(
    sku_data: Dict,
    method_name: str,
    forecast_horizon: int = 28,
) -> Dict:
    """Run a single structural baseline on a single SKU and return metrics."""
    b = sku_data["demand"]
    f_promo = sku_data["f_promo"]
    f_snap = sku_data["f_snap"]
    T = len(b)
    if T < forecast_horizon + 50:
        return {
            "sku_id": sku_data.get("sku_id", 0),
            "category": sku_data.get("category", "unknown"),
            "wmape": np.nan,
            "variance_ratio": np.nan,
            "rmse": np.nan,
            "method": method_name,
            "runtime": np.nan,
        }

    method_map = {
        "ARIMAX": run_arimax,
        "ETS_covariates": run_ets_covariates,
        "Prophet-like": run_prophet_or_prophet_like,
        "DistributedLag_Ridge": run_distributed_lag_ridge,
    }
    func = method_map[method_name]

    t0 = time.time()
    try:
        pred, actual, fitted_train, model = func(b, f_promo, f_snap, forecast_horizon)
    except Exception as e:
        warnings.warn(f"{method_name} failed for SKU {sku_data.get('sku_id', 0)}: {e}")
        pred = None
    elapsed = time.time() - t0

    if pred is None or actual is None or len(pred) != len(actual):
        return {
            "sku_id": sku_data.get("sku_id", 0),
            "category": sku_data.get("category", "unknown"),
            "wmape": np.nan,
            "variance_ratio": np.nan,
            "rmse": np.nan,
            "method": method_name,
            "runtime": elapsed,
        }

    return {
        "sku_id": sku_data.get("sku_id", 0),
        "category": sku_data.get("category", "unknown"),
        "wmape": wmape(actual, pred),
        "variance_ratio": variance_ratio(pred, actual),
        "rmse": rmse(actual, pred),
        "method": method_name,
        "runtime": elapsed,
    }


def run_dataset_baselines(dataset: List[Dict], dataset_name: str) -> pd.DataFrame:
    """Run all four structural baselines on a dataset."""
    print(f"\n{'='*70}")
    print(f"Running structural baselines on {dataset_name} ({len(dataset)} items)")
    print(f"{'='*70}")

    methods = ["ARIMAX", "ETS_covariates", "Prophet-like", "DistributedLag_Ridge"]
    all_results = []

    for method in methods:
        print(f"\n  --- {method} ---")
        method_results = []
        for i, sku_data in enumerate(dataset):
            try:
                r = evaluate_method_on_sku(sku_data, method, forecast_horizon=28)
                method_results.append(r)
                if (i + 1) % 50 == 0 or i == len(dataset) - 1:
                    print(f"    Processed {i+1}/{len(dataset)} items")
            except Exception as e:
                warnings.warn(f"Unhandled error for {method} SKU {sku_data.get('sku_id', i)}: {e}")
                method_results.append({
                    "sku_id": sku_data.get("sku_id", i),
                    "category": sku_data.get("category", "unknown"),
                    "wmape": np.nan,
                    "variance_ratio": np.nan,
                    "rmse": np.nan,
                    "method": method,
                    "runtime": np.nan,
                })
        all_results.extend(method_results)

    df = pd.DataFrame(all_results)
    return df


def print_summary(df: pd.DataFrame, dataset_name: str):
    """Print mean ± std summary table for each method."""
    print(f"\n{'='*70}")
    print(f"Summary: {dataset_name}")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'n':>6} {'wMAPE':>12} {'RMSE':>12} {'VarRatio':>12} {'Runtime':>10}")
    print("-" * 70)
    for method in df["method"].unique():
        sub = df[df["method"] == method].dropna(subset=["wmape"])
        if len(sub) == 0:
            print(f"{method:<25} {'--':>6} {'--':>12} {'--':>12} {'--':>12} {'--':>10}")
            continue
        wmape_m = sub["wmape"].mean()
        wmape_s = sub["wmape"].std()
        rmse_m = sub["rmse"].mean()
        rmse_s = sub["rmse"].std()
        vr_m = sub["variance_ratio"].mean()
        vr_s = sub["variance_ratio"].std()
        rt_m = sub["runtime"].mean()
        print(f"{method:<25} {len(sub):>6} {wmape_m:.4f}±{wmape_s:.4f} {rmse_m:.2f}±{rmse_s:.2f} {vr_m:.4f}±{vr_s:.4f} {rt_m:.3f}s")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    m5_dir = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")
    favorita_dir = _os.path.join(_ROOT, "data/favorita-grocery-sales-forecasting")
    out_dir = _os.path.join(_ROOT, "code/results")
    os.makedirs(out_dir, exist_ok=True)

    total_start = time.time()

    # --- M5: 200 SKUs ---
    print("\nLoading M5 data (200 SKUs)...")
    m5_data = load_m5_data(m5_dir, n_skus=200, random_state=42)
    print(f"Loaded {len(m5_data)} SKUs")

    m5_results = run_dataset_baselines(m5_data, "M5_200")
    # NOTE: write to a distinct *_base name. The canonical *_structural_baselines.csv
    # (which additionally carries a safety_stock column) is produced by
    # compute_structural_safety_stock.py; writing the same name here would clobber that
    # richer file, since this script runs later in run_paper.py's alphabetical order.
    m5_out = os.path.join(out_dir, "m5_200_structural_baselines_base.csv")
    m5_results.to_csv(m5_out, index=False)
    print(f"\nSaved M5 results to {m5_out}")
    print_summary(m5_results, "M5_200")

    # --- Favorita: 97 items ---
    print("\nLoading Favorita data (97 items)...")
    favorita_data = load_favorita_data(favorita_dir, n_items=97, random_state=42)
    print(f"Loaded {len(favorita_data)} items")

    if len(favorita_data) > 0:
        fav_results = run_dataset_baselines(favorita_data, "Favorita_97")
        # distinct *_base name; see the M5 note above (compute_structural_safety_stock.py
        # owns the canonical safety-stock-bearing favorita_structural_baselines.csv).
        fav_out = os.path.join(out_dir, "favorita_structural_baselines_base.csv")
        fav_results.to_csv(fav_out, index=False)
        print(f"\nSaved Favorita results to {fav_out}")
        print_summary(fav_results, "Favorita_97")
    else:
        print("WARNING: No Favorita data loaded, skipping")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"TOTAL RUNTIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")
