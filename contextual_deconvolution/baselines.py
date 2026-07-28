"""
Baseline forecasting methods for comparison.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def build_tuned_xgboost_features(
    b: np.ndarray, f_promo: np.ndarray, f_snap: np.ndarray, n_lags: int = 28
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Build a rich feature matrix for tuned gradient boosting baseline.
    
    Features include:
      - Lagged demand: 1,2,3,4,5,6,7,14,21,28 days
      - Rolling statistics (mean, std, min, max) over 7,14,28 days
      - Event features: promo, snap, plus rolling sums over 7,14,28 days
      - Calendar features: day_of_week, month, day_of_month, is_weekend,
        week_of_year, month_progress, days_in_month
    """
    T = len(b)
    if T < n_lags + 1:
        return np.array([]), np.array([]), []
    
    lag_days = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
    lag_days = [d for d in lag_days if d <= n_lags]
    roll_windows = [7, 14, 28]
    roll_windows = [w for w in roll_windows if w <= n_lags]
    
    features = []
    for t in range(n_lags, T):
        feat = []
        # Lagged demand
        for d in lag_days:
            feat.append(b[t - d])
        
        # Rolling statistics on demand
        for w in roll_windows:
            window = b[t - w:t]
            feat.append(np.mean(window))
            feat.append(np.std(window) + 1e-8)  # small epsilon to avoid zero std
            feat.append(np.min(window))
            feat.append(np.max(window))
        
        # Event features
        feat.append(f_promo[t])
        feat.append(f_snap[t])
        for w in roll_windows:
            feat.append(np.sum(f_promo[t - w:t]))
            feat.append(np.sum(f_snap[t - w:t]))
        
        # Calendar features
        feat.append(t % 7)                    # day_of_week
        feat.append((t // 30) % 12)           # month
        feat.append(t % 30)                   # day_of_month
        feat.append(1 if t % 7 >= 5 else 0)   # is_weekend
        feat.append((t // 7) % 52)            # week_of_year
        feat.append((t % 30) / 30.0)          # month_progress
        feat.append(30)                       # days_in_month (approx)
        feat.append(t)                        # time index
        
        features.append(feat)
    
    # Build feature names
    names = []
    for d in lag_days:
        names.append(f'demand_lag_{d}')
    for w in roll_windows:
        names.append(f'demand_roll_mean_{w}')
        names.append(f'demand_roll_std_{w}')
        names.append(f'demand_roll_min_{w}')
        names.append(f'demand_roll_max_{w}')
    names.append('promo')
    names.append('snap')
    for w in roll_windows:
        names.append(f'promo_sum_{w}')
        names.append(f'snap_sum_{w}')
    names += ['day_of_week', 'month', 'day_of_month', 'is_weekend',
              'week_of_year', 'month_progress', 'days_in_month', 'time_index']
    
    return np.array(features), np.arange(n_lags, T), names


def _tuned_feature_row(dem, f_promo, f_snap, t, lag_days, roll_windows):
    """One feature row for absolute day t, matching build_tuned_xgboost_features's layout.
    Demand lag/rolling features are read from `dem` (training history + prior predictions);
    event and calendar features use the known values at day t (legitimate exogenous inputs)."""
    feat = [dem[t - d] for d in lag_days]
    for w in roll_windows:
        win = dem[t - w:t]
        feat += [np.mean(win), np.std(win) + 1e-8, np.min(win), np.max(win)]
    feat += [f_promo[t], f_snap[t]]
    for w in roll_windows:
        feat += [np.sum(f_promo[t - w:t]), np.sum(f_snap[t - w:t])]
    feat += [t % 7, (t // 30) % 12, t % 30, 1 if t % 7 >= 5 else 0,
             (t // 7) % 52, (t % 30) / 30.0, 30, t]
    return feat


def run_tuned_xgboost(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """Tuned XGBoost with rich time-series features.
    
    Uses lagged demand (1-28 days), rolling statistics (7/14/28 windows),
    event features, and calendar features. Hyperparameters are tuned for
    retail demand forecasting with early stopping on validation set.
    """
    n_lags = 28
    T = len(b)
    split = T - forecast_horizon
    if split <= n_lags + 50:
        return None, None, None, None

    # Train on the TRAINING window only (no held-out demand enters feature construction).
    X, idx, feature_names = build_tuned_xgboost_features(
        b[:split], f_promo[:split], f_snap[:split], n_lags=n_lags
    )
    if len(X) == 0:
        return None, None, None, None
    y = b[:split][idx]
    val_split = int(0.8 * len(y))
    if val_split <= 30:
        return None, None, None, None

    if HAS_XGBOOST:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20,
        )
        model.fit(
            X[:val_split], y[:val_split],
            eval_set=[(X[val_split:], y[val_split:])],
            verbose=False,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
        )
        model.fit(X[:val_split], y[:val_split])

    fitted_train = model.predict(X)

    # Recursive multi-step forecast: lag/rolling features for test days come from the
    # model's OWN predictions (plus the known future calendar), never actual test demand.
    lag_days = [d for d in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28] if d <= n_lags]
    roll_windows = [w for w in [7, 14, 28] if w <= n_lags]
    dem = list(b[:split])
    y_pred = []
    for h in range(forecast_horizon):
        t = split + h
        row = _tuned_feature_row(np.asarray(dem), f_promo, f_snap, t, lag_days, roll_windows)
        p = float(model.predict(np.array([row]))[0])
        y_pred.append(p)
        dem.append(p)
    return np.asarray(y_pred), b[split:], fitted_train, model


# Keep old function for backward compatibility
def build_xgboost_features(
    b: np.ndarray, f_promo: np.ndarray, f_snap: np.ndarray, n_lags: int = 14
) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix for gradient boosting baseline (legacy)."""
    T = len(b)
    features = []
    for t in range(n_lags, T):
        feat = [
            b[t - 1], b[t - 7], b[t - 14],
            f_promo[t], f_snap[t],
            np.sum(f_promo[t - 7:t]),
            np.sum(f_snap[t - 30:t]),
            t % 7,
            (t // 30) % 12,
            t,
        ]
        features.append(feat)
    return np.array(features), np.arange(n_lags, T)


def _xgb_feature_row(dem, f_promo, f_snap, t):
    """One feature row for absolute day t, matching build_xgboost_features (legacy layout)."""
    return [dem[t - 1], dem[t - 7], dem[t - 14], f_promo[t], f_snap[t],
            np.sum(f_promo[t - 7:t]), np.sum(f_snap[t - 30:t]),
            t % 7, (t // 30) % 12, t]


def run_xgboost(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
    n_estimators: int = 200,
    max_depth: int = 5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """XGBoost or GradientBoostingRegressor baseline (legacy, simple features)."""
    n_lags = 14
    T = len(b)
    split = T - forecast_horizon
    if split <= n_lags + 50:
        return None, None, None, None
    X, idx = build_xgboost_features(b[:split], f_promo[:split], f_snap[:split])
    if len(X) == 0:
        return None, None, None, None
    y = b[:split][idx]

    if HAS_XGBOOST:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.1, random_state=42, verbosity=0
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
    model.fit(X, y)
    fitted_train = model.predict(X)

    # Recursive multi-step forecast (no held-out demand)
    dem = list(b[:split])
    y_pred = []
    for h in range(forecast_horizon):
        t = split + h
        row = _xgb_feature_row(np.asarray(dem), f_promo, f_snap, t)
        p = float(model.predict(np.array([row]))[0])
        y_pred.append(p)
        dem.append(p)
    return np.asarray(y_pred), b[split:], fitted_train, model


def run_prophet_like(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """Prophet-like additive baseline using Fourier seasonality + event regressors."""
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

    split = T - forecast_horizon
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



# -----------------------------------------------------------------------------
# Intermittent demand baselines: Croston and SBA
# -----------------------------------------------------------------------------

def _croston_split(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split demand into sizes and intervals.
    
    Returns:
        sizes: non-zero demand values
        intervals: number of periods between non-zero demands
        demand_times: indices where demand > 0
    """
    demand_times = np.where(b > 0)[0]
    if len(demand_times) < 2:
        return np.array([np.mean(b)]), np.array([1.0]), demand_times
    
    sizes = b[demand_times]
    intervals = np.diff(demand_times)
    return sizes, intervals, demand_times


def run_croston(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
    alpha: float = 0.1,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """Croston's method for intermittent demand forecasting.
    
    Uses exponential smoothing separately on demand sizes and inter-demand
    intervals. Forecast = size / interval.
    
    Args:
        alpha: Smoothing parameter for both size and interval (default 0.1)
    """
    T = len(b)
    split = T - forecast_horizon
    
    sizes, intervals, demand_times = _croston_split(b[:split])
    if len(sizes) < 2:
        # Not enough demand occurrences, fall back to mean
        forecast = np.full(forecast_horizon, np.mean(b[:split]))
        return forecast, b[split:], None, None
    
    # Exponential smoothing on sizes and intervals
    size_smooth = sizes[0]
    interval_smooth = float(intervals[0])
    
    for i in range(1, len(sizes)):
        size_smooth = alpha * sizes[i] + (1 - alpha) * size_smooth
        if i < len(intervals):
            interval_smooth = alpha * intervals[i] + (1 - alpha) * interval_smooth
    
    # Forecast is the ratio
    if interval_smooth <= 0:
        forecast_value = 0.0
    else:
        forecast_value = size_smooth / interval_smooth
    
    forecast = np.full(forecast_horizon, forecast_value)
    
    # Fitted values on training (in-sample forecast)
    fitted = np.full(split, np.nan)
    # In-sample Croston: at each demand occurrence, update and predict
    size_s = sizes[0] if len(sizes) > 0 else np.mean(b[:split])
    int_s = float(intervals[0]) if len(intervals) > 0 else 1.0
    last_demand_idx = demand_times[0] if len(demand_times) > 0 else 0
    
    for t in range(split):
        if t in demand_times[:split]:
            idx = np.where(demand_times[:split] == t)[0]
            if len(idx) > 0:
                size_s = alpha * b[t] + (1 - alpha) * size_s
                if idx[0] > 0 and idx[0] - 1 < len(intervals):
                    int_s = alpha * intervals[idx[0] - 1] + (1 - alpha) * int_s
                last_demand_idx = t
        if int_s > 0:
            fitted[t] = size_s / int_s
        else:
            fitted[t] = 0.0
    
    return forecast, b[split:], fitted, None


def run_sba(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
    alpha: float = 0.1,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """Syntetos-Boylan Approximation (SBA) for intermittent demand.
    
    An improved Croston method that debiases the forecast by multiplying
    by a correction factor: (1 - alpha/2).
    
    Args:
        alpha: Smoothing parameter for both size and interval (default 0.1)
    """
    T = len(b)
    split = T - forecast_horizon
    
    sizes, intervals, demand_times = _croston_split(b[:split])
    if len(sizes) < 2:
        forecast = np.full(forecast_horizon, np.mean(b[:split]) * (1 - alpha / 2))
        return forecast, b[split:], None, None
    
    # Exponential smoothing
    size_smooth = sizes[0]
    interval_smooth = float(intervals[0])
    
    for i in range(1, len(sizes)):
        size_smooth = alpha * sizes[i] + (1 - alpha) * size_smooth
        if i < len(intervals):
            interval_smooth = alpha * intervals[i] + (1 - alpha) * interval_smooth
    
    # SBA correction factor
    if interval_smooth <= 0:
        forecast_value = 0.0
    else:
        forecast_value = (1 - alpha / 2) * size_smooth / interval_smooth
    
    forecast = np.full(forecast_horizon, forecast_value)
    
    # Fitted values
    fitted = np.full(split, np.nan)
    size_s = sizes[0] if len(sizes) > 0 else np.mean(b[:split])
    int_s = float(intervals[0]) if len(intervals) > 0 else 1.0
    
    for t in range(split):
        if t in demand_times[:split]:
            idx = np.where(demand_times[:split] == t)[0]
            if len(idx) > 0:
                size_s = alpha * b[t] + (1 - alpha) * size_s
                if idx[0] > 0 and idx[0] - 1 < len(intervals):
                    int_s = alpha * intervals[idx[0] - 1] + (1 - alpha) * int_s
        if int_s > 0:
            fitted[t] = (1 - alpha / 2) * size_s / int_s
        else:
            fitted[t] = 0.0
    
    return forecast, b[split:], fitted, None


# -----------------------------------------------------------------------------
# LightGBM baseline (M5 competition winner style)
# -----------------------------------------------------------------------------

def run_lightgbm(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
    n_estimators: int = 500,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """LightGBM baseline with the same rich features as tuned XGBoost.
    
    LightGBM was the core of the M5 winning solution. We use the same
    feature engineering as Tuned XGBoost for a fair comparison.
    """
    try:
        import lightgbm as lgb
        HAS_LIGHTGBM = True
    except ImportError:
        HAS_LIGHTGBM = False
    
    if not HAS_LIGHTGBM:
        # Fall back to GradientBoostingRegressor with similar features
        return run_tuned_xgboost(b, f_promo, f_snap, forecast_horizon,
                                  n_estimators=n_estimators, max_depth=max_depth,
                                  learning_rate=learning_rate)
    
    n_lags = 28
    T = len(b)
    split = T - forecast_horizon
    if split <= n_lags + 50:
        return None, None, None, None
    X, idx, feature_names = build_tuned_xgboost_features(
        b[:split], f_promo[:split], f_snap[:split], n_lags=n_lags
    )
    if len(X) == 0:
        return None, None, None, None
    y = b[:split][idx]
    val_split = int(0.8 * len(y))
    if val_split <= 30:
        return None, None, None, None

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X[:val_split], y[:val_split],
        eval_set=[(X[val_split:], y[val_split:])],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    fitted_train = model.predict(X)

    # Recursive multi-step forecast (no held-out demand)
    lag_days = [d for d in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28] if d <= n_lags]
    roll_windows = [w for w in [7, 14, 28] if w <= n_lags]
    dem = list(b[:split])
    y_pred = []
    for h in range(forecast_horizon):
        t = split + h
        row = _tuned_feature_row(np.asarray(dem), f_promo, f_snap, t, lag_days, roll_windows)
        p = float(model.predict(np.array([row]))[0])
        y_pred.append(p)
        dem.append(p)
    return np.asarray(y_pred), b[split:], fitted_train, model


def run_stl_arima(
    b: np.ndarray,
    f_promo: np.ndarray,
    f_snap: np.ndarray,
    forecast_horizon: int = 28,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
    """STL decomposition + seasonal ARIMA on deseasonalized series."""
    try:
        from statsmodels.tsa.seasonal import STL
        from statsmodels.tsa.arima.model import ARIMA
        HAS_STATSMODELS = True
    except ImportError:
        HAS_STATSMODELS = False
    
    if not HAS_STATSMODELS:
        return None, None, None, None

    T = len(b)
    split = T - forecast_horizon
    train = b[:split]

    try:
        stl = STL(train, period=7, robust=True)
        res = stl.fit()
        trend = res.trend
        seasonal = res.seasonal
        resid = train - trend - seasonal

        deseasonalized = trend + resid
        arima_model = ARIMA(deseasonalized, order=(1, 1, 1))
        arima_fitted = arima_model.fit()
        fc_deseas = arima_fitted.forecast(steps=forecast_horizon)

        last_seasonal = seasonal[-7:]
        seasonal_idx = np.arange(forecast_horizon) % 7
        fc_seas = last_seasonal[seasonal_idx]
        forecast = fc_deseas + fc_seas
        fitted_train = trend + seasonal + arima_fitted.fittedvalues
        return forecast, b[split:], fitted_train, None
    except Exception as e:
        print(f"  STL+ARIMA failed: {e}")
        return None, None, None, None
