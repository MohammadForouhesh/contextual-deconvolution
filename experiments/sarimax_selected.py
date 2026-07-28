#!/usr/bin/env python3
"""Properly specified SARIMAX baseline: stationarity-tested differencing, AIC-based order
selection, and weekly seasonality.

The `run_arimax` baseline in run_structural_baselines.py returns the FIRST order that
converges from a fixed list, performs no stationarity test, and has no seasonal component --
a serious handicap on daily retail data with a dominant weekly cycle. This module supplies a
fair-fight alternative:

  d      chosen by ADF (H0: unit root) + KPSS (H0: stationary), max_d=2
  (p,q)  chosen by AIC over a small grid with the seasonal part switched off
  (P,Q)  chosen by AIC at the selected (p,q), with period m=7
  exog   the same future promo/SNAP calendar every other calendar-aware baseline receives

Selection uses the training window only. Stepwise (13 fits) rather than a full 36-fit grid,
which is the standard auto.arima approximation and keeps catalog-scale evaluation feasible.
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def select_d(y, max_d=2, alpha=0.05):
    """Smallest d for which ADF rejects a unit root AND KPSS fails to reject stationarity."""
    from statsmodels.tsa.stattools import adfuller, kpss
    for d in range(max_d + 1):
        x = np.diff(y, n=d) if d > 0 else np.asarray(y, float)
        if len(x) < 30 or np.std(x) < 1e-9:
            return d
        try:
            adf_p = adfuller(x, autolag="AIC")[1]
        except Exception:
            adf_p = 1.0
        try:
            kpss_p = kpss(x, regression="c", nlags="auto")[1]
        except Exception:
            kpss_p = 0.0
        if adf_p < alpha and kpss_p > alpha:
            return d
    return 1


def run_sarimax_selected(b, f_promo, f_snap, forecast_horizon=28, m=7,
                         p_max=2, q_max=2, return_spec=False):
    """SARIMAX with tested d, AIC-selected (p,q) then (P,Q), weekly seasonality, exog calendar.

    Returns (pred, actual, fitted_train, spec) matching the other baselines' signature.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    b = np.asarray(b, float)
    T = len(b)
    split = T - forecast_horizon
    y = b[:split]
    X = np.column_stack([np.asarray(f_promo, float)[:split], np.asarray(f_snap, float)[:split]])
    Xf = np.column_stack([np.asarray(f_promo, float)[split:T], np.asarray(f_snap, float)[split:T]])
    if len(y) < 3 * m or np.std(y) < 1e-9:
        return (None, None, None, None) if not return_spec else (None, None, None, None)

    d = select_d(y)

    def _aic(order, seasonal):
        try:
            fit = SARIMAX(y, exog=X, order=order, seasonal_order=seasonal,
                          enforce_stationarity=False, enforce_invertibility=False
                          ).fit(disp=False, maxiter=50)
            a = fit.aic
            return (a if np.isfinite(a) else np.inf), fit
        except Exception:
            return np.inf, None

    # stage 1: non-seasonal (p,q) by AIC
    best = (np.inf, None, (0, d, 0))
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            a, fit = _aic((p, d, q), (0, 0, 0, 0))
            if a < best[0]:
                best = (a, fit, (p, d, q))
    order = best[2]

    # stage 2: seasonal (P,Q) at the selected (p,q)
    best_s = (best[0], best[1], (0, 0, 0, 0))
    for P in range(2):
        for Q in range(2):
            if P == 0 and Q == 0:
                continue
            a, fit = _aic(order, (P, 0, Q, m))
            if a < best_s[0]:
                best_s = (a, fit, (P, 0, Q, m))
    aic, fit, seasonal = best_s
    if fit is None:
        return (None, None, None, None)

    try:
        pred = np.asarray(fit.forecast(steps=forecast_horizon, exog=Xf), float)
        fitted = np.asarray(fit.fittedvalues, float)
    except Exception:
        return (None, None, None, None)
    if not np.all(np.isfinite(pred)):
        return (None, None, None, None)
    pred = np.maximum(pred, 0.0)
    spec = {"order": order, "seasonal_order": seasonal, "d": d, "aic": float(aic)}
    return (pred, b[split:], fitted, spec)


if __name__ == "__main__":
    import os, sys, time
    _H = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_H, ".."))
    from contextual_deconvolution.loaders import load_m5_data
    from contextual_deconvolution.metrics import wmape
    from run_structural_baselines import run_arimax

    root = os.path.abspath(os.path.join(_H, "..", ".."))
    ds = load_m5_data(os.path.join(root, "data", "m5-data", "m5-forecasting-accuracy"),
                      n_skus=6, random_state=42, skip_prices=True)
    print(f"{'sku':>4} {'d':>2} {'order':>10} {'seasonal':>14} {'SARIMAX':>9} {'old ARIMAX':>11} {'sec':>6}")
    for i, s in enumerate(ds):
        b, fp, fs = s["demand"], s["f_promo"], s["f_snap"]
        t0 = time.time()
        pred, act, _, spec = run_sarimax_selected(b, fp, fs, 28)
        dt = time.time() - t0
        old = run_arimax(b, fp, fs, 28)[0]
        w_new = wmape(act, pred) if pred is not None else float("nan")
        w_old = wmape(b[-28:], old) if old is not None else float("nan")
        print(f"{i:>4} {spec['d']:>2} {str(spec['order']):>10} {str(spec['seasonal_order']):>14} "
              f"{w_new:9.4f} {w_old:11.4f} {dt:6.1f}")
