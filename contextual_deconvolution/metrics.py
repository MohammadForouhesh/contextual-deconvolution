"""
Evaluation metrics: statistical accuracy, operational stability, and inventory cost.
"""

import numpy as np
from typing import Optional, Tuple


def wmape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted MAPE."""
    denom = np.sum(actual)
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - forecast)) / denom)


def variance_ratio(forecast: np.ndarray, actual: np.ndarray, min_var: float = 0.0) -> float:
    """Forecast variance / Actual variance (Bullwhip metric).
    
    Args:
        min_var: Minimum actual variance threshold. Items with var(actual) < min_var
        are returned as NaN to signal they should be excluded from aggregate stats.
    """
    var_actual = np.var(actual)
    if var_actual < min_var:
        return np.nan
    if var_actual == 0:
        return 1.0
    return float(np.var(forecast) / var_actual)


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(actual - forecast)))


def inventory_cost(
    actual: np.ndarray,
    forecast: np.ndarray,
    fitted_train: Optional[np.ndarray] = None,
    lead_time: int = 7,
    h: float = 1.0,
    p: float = 10.0,
    service_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Simulate daily base-stock review with given lead time.
    Returns (avg_daily_cost, safety_stock).
    """
    T = len(actual)
    if T == 0:
        return 0.0, 0.0

    if fitted_train is not None and len(fitted_train) > 1:
        train_residuals = fitted_train
        train_residuals_std = np.std(train_residuals) * 0.3
    else:
        train_residuals_std = np.std(forecast) * 0.3

    safety_stock = train_residuals_std * 1.645
    if safety_stock < 0 or np.isnan(safety_stock):
        safety_stock = np.std(forecast) * 1.645

    mean_d = np.mean(actual)
    inventory = mean_d * lead_time + safety_stock
    pipeline = [mean_d] * lead_time

    total_holding = 0.0
    total_stockout = 0.0

    padded_forecast = np.concatenate([forecast, [np.mean(forecast)] * lead_time])

    for t in range(T):
        arrival = pipeline.pop(0)
        inventory += arrival

        demand = actual[t]
        if inventory >= demand:
            total_holding += h * (inventory - demand)
            inventory -= demand
        else:
            total_stockout += p * (demand - inventory)
            inventory = 0

        future_start = t + 1
        future_end = future_start + lead_time
        target = np.sum(padded_forecast[future_start:future_end]) + safety_stock
        inv_pos = inventory + sum(pipeline)
        order = max(0.0, target - inv_pos)
        pipeline.append(order)

    avg_daily_cost = (total_holding + total_stockout) / T
    return avg_daily_cost, safety_stock


def inventory_cost_with_events(
    actual: np.ndarray,
    forecast: np.ndarray,
    event_mask: np.ndarray,
    fitted_train: Optional[np.ndarray] = None,
    lead_time: int = 7,
    h: float = 1.0,
    p: float = 10.0,
    service_level: float = 0.95,
) -> Tuple[float, float, float, float, float, float]:
    """
    Simulate daily base-stock review with event-window metrics.

    event_mask: binary array indicating event days (promo, snap, etc.)
    Returns (avg_daily_cost, safety_stock, total_holding, total_stockout,
             event_fill_rate, event_stockout_rate).
    """
    T = len(actual)
    if T == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if fitted_train is not None and len(fitted_train) > 1:
        train_residuals = fitted_train
        train_residuals_std = np.std(train_residuals) * 0.3
    else:
        train_residuals_std = np.std(forecast) * 0.3

    safety_stock = train_residuals_std * 1.645
    if safety_stock < 0 or np.isnan(safety_stock):
        safety_stock = np.std(forecast) * 1.645

    mean_d = np.mean(actual)
    inventory = mean_d * lead_time + safety_stock
    pipeline = [mean_d] * lead_time

    total_holding = 0.0
    total_stockout = 0.0
    event_demand_met = 0.0
    event_demand_total = 0.0
    event_stockout_days = 0
    event_days = 0

    padded_forecast = np.concatenate([forecast, [np.mean(forecast)] * lead_time])

    for t in range(T):
        arrival = pipeline.pop(0)
        inventory += arrival

        demand = actual[t]
        if inventory >= demand:
            total_holding += h * (inventory - demand)
            inventory -= demand
            met = demand
        else:
            total_stockout += p * (demand - inventory)
            met = inventory
            inventory = 0

        # Track event-window metrics
        if event_mask[t] > 0:
            event_days += 1
            event_demand_total += demand
            event_demand_met += met
            if met < demand:
                event_stockout_days += 1

        future_start = t + 1
        future_end = future_start + lead_time
        target = np.sum(padded_forecast[future_start:future_end]) + safety_stock
        inv_pos = inventory + sum(pipeline)
        order = max(0.0, target - inv_pos)
        pipeline.append(order)

    avg_daily_cost = (total_holding + total_stockout) / T
    event_fill_rate = event_demand_met / event_demand_total if event_demand_total > 0 else 1.0
    event_stockout_rate = event_stockout_days / event_days if event_days > 0 else 0.0

    return avg_daily_cost, safety_stock, total_holding, total_stockout, event_fill_rate, event_stockout_rate


def mase(actual: np.ndarray, forecast: np.ndarray, seasonal_period: int = 7) -> float:
    """Mean Absolute Scaled Error (MASE).
    
    Scales forecast MAE by the MAE of a naive seasonal forecast
    (same day in previous season). Values < 1 indicate the forecast
    is better than the naive benchmark; values > 1 indicate it is worse.
    """
    if len(actual) < seasonal_period + 1:
        return np.nan
    
    mae_forecast = np.mean(np.abs(actual - forecast))
    
    # Naive seasonal forecast: same day in previous season
    naive_forecast = actual[:-seasonal_period]
    naive_actual = actual[seasonal_period:]
    mae_naive = np.mean(np.abs(naive_actual - naive_forecast))
    
    if mae_naive == 0:
        return np.nan
    
    return float(mae_forecast / mae_naive)


def rmsse(actual: np.ndarray, forecast: np.ndarray, seasonal_period: int = 7) -> float:
    """Root Mean Squared Scaled Error (RMSSE) -- single-series M5 metric.
    
    Scales forecast RMSE by the RMSE of a naive seasonal forecast.
    """
    if len(actual) < seasonal_period + 1:
        return np.nan
    
    mse_forecast = np.mean((actual - forecast) ** 2)
    
    # Naive seasonal forecast
    naive_forecast = actual[:-seasonal_period]
    naive_actual = actual[seasonal_period:]
    mse_naive = np.mean((naive_actual - naive_forecast) ** 2)
    
    if mse_naive == 0:
        return np.nan
    
    return float(np.sqrt(mse_forecast / mse_naive))


def order_variance_under_replenishment(
    forecast: np.ndarray,
    actual: np.ndarray,
    lead_time: int = 7,
    service_level: float = 0.95,
    fitted_train: Optional[np.ndarray] = None,
) -> float:
    """Compute variance of orders placed by a base-stock replenishment policy.
    
    This measures the Bullwhip effect at the order level: how much variance
    is amplified in the order signal sent to suppliers compared to the
    actual demand variance. This is the correct operational metric for
    supply chain stability, as it captures the variance of the orders
    that actually flow upstream.
    
    Returns: Var(orders) / Var(actual) - 1, analogous to VR but for orders.
    """
    T = len(actual)
    if T == 0:
        return np.nan
    
    var_actual = np.var(actual)
    if var_actual == 0:
        return np.nan
    
    if fitted_train is not None and len(fitted_train) > 1:
        train_residuals_std = np.std(fitted_train) * 0.3
    else:
        train_residuals_std = np.std(forecast) * 0.3
    
    safety_stock = train_residuals_std * 1.645
    if safety_stock < 0 or np.isnan(safety_stock):
        safety_stock = np.std(forecast) * 1.645
    
    mean_d = np.mean(actual)
    inventory = mean_d * lead_time + safety_stock
    pipeline = [mean_d] * lead_time
    orders = []
    
    padded_forecast = np.concatenate([forecast, [np.mean(forecast)] * lead_time])
    
    for t in range(T):
        arrival = pipeline.pop(0)
        inventory += arrival
        
        demand = actual[t]
        if inventory >= demand:
            inventory -= demand
        else:
            inventory = 0
        
        future_start = t + 1
        future_end = future_start + lead_time
        target = np.sum(padded_forecast[future_start:future_end]) + safety_stock
        inv_pos = inventory + sum(pipeline)
        order = max(0.0, target - inv_pos)
        orders.append(order)
        pipeline.append(order)
    
    var_orders = np.var(orders)
    return float(var_orders / var_actual - 1.0)


def spectral_flatness(x: np.ndarray) -> float:
    """Spectral flatness as a proxy for volatility."""
    ps = np.abs(np.fft.rfft(x)) ** 2
    ps = ps[ps > 0]
    if len(ps) == 0:
        return 0.0
    geometric_mean = np.exp(np.mean(np.log(ps)))
    arithmetic_mean = np.mean(ps)
    return float(geometric_mean / arithmetic_mean)


def max_drawdown(x: np.ndarray) -> float:
    """Maximum drawdown of a time series."""
    cumulative = np.cumsum(x)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
