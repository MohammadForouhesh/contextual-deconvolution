#!/usr/bin/env python3
"""Cross-sectional reliability of per-SKU forecast error on M5 (paper Table `tab:reliability`).

Regenerates the dispersion result behind Finding 2: median accuracy is similar across
methods, but dispersion and catastrophic-failure rate are not. Repeats the evaluation over
several independent SKU draws so the ranking is shown to be a property of the methods
rather than of one sample.

Protocol (identical to the rest of the paper):
  * strictly out-of-sample: the convex decomposition is fit on the training window only and
    every baseline forecasts recursively (no oracle lags)
  * information parity: each feature-based baseline receives the same future promo/SNAP
    calendar CD uses
  * the seed selects WHICH SKUs are drawn from the 30,490-SKU catalog; CD and the baselines
    are otherwise deterministic

Outputs (written to results/):
  reliability_per_sku.csv    one row per (seed, method, SKU)
  reliability_by_seed.csv    per-seed summary statistics
  reliability_summary.csv    mean +/- SD across seeds  <-- Table `tab:reliability`

Usage:
  python experiments/run_reliability_dispersion.py                     # paper settings
  python experiments/run_reliability_dispersion.py --seeds 42 --n_skus 500   # quick check
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")     # avoid OpenMP oversubscription:
os.environ.setdefault("MKL_NUM_THREADS", "1")     # xgboost/lightgbm/torch deadlock otherwise
import argparse, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from joblib import Parallel, delayed
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.core import estimate_kernel, fit_parametric_kernel, kernel_from_theta
from contextual_deconvolution.experiments import evaluate_sku_pooled, CONFIG_DEFAULT
from contextual_deconvolution.metrics import wmape

DATA_DIR = os.path.join(_HERE, "..", "data", "m5-data", "m5-forecasting-accuracy")
OUT_DIR = os.path.join(_HERE, "..", "results")
BLOWUP = 2.0          # wMAPE above this counts as a catastrophic failure (>200% error)

FAMILY = {"CD": "---", "TunedXGB": "boosted", "VanillaXGB": "boosted", "LightGBM": "boosted",
          "ARIMAX": "structural", "SARIMAX": "structural", "ETS+cov": "structural", "DL-Ridge": "structural",
          "Prophet-like": "structural", "STL+ARIMA": "structural",
          "Croston": "intermittent", "SBA": "intermittent"}


def _eval_sku(sku, gk, cfg, horizon, with_sarimax=False):
    """All methods on one SKU. Imports inside the worker to keep each process single-threaded."""
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings as _w
    _w.filterwarnings("ignore")
    from contextual_deconvolution.baselines import (run_tuned_xgboost, run_xgboost, run_lightgbm,
                                                    run_prophet_like, run_stl_arima,
                                                    run_croston, run_sba)
    import run_structural_baselines as SB

    b = np.asarray(sku["demand"], float)
    fp = np.asarray(sku["f_promo"], float)
    fs = np.asarray(sku["f_snap"], float)
    split = len(b) - horizon
    b_test = b[split:]
    out = {}
    if np.all(b_test == 0) or np.std(b[:split]) < 1e-9:
        return out
    try:
        r = evaluate_sku_pooled(sku, gk["phi_promo"], gk["phi_snap"], cfg)["forecasts"]["CD"]
        out["CD"] = r["wmape"]
    except Exception:
        pass
    for name, fn in (("TunedXGB", run_tuned_xgboost), ("VanillaXGB", run_xgboost),
                     ("LightGBM", run_lightgbm), ("Prophet-like", run_prophet_like),
                     ("STL+ARIMA", run_stl_arima), ("ARIMAX", SB.run_arimax),
                     ("ETS+cov", SB.run_ets_covariates),
                     ("DL-Ridge", SB.run_distributed_lag_ridge),
                     ("Croston", run_croston), ("SBA", run_sba)):
        try:
            o = fn(b, fp, fs, horizon)
            pred = o[0] if isinstance(o, tuple) else o
            if pred is not None and len(pred) == horizon and np.all(np.isfinite(pred)):
                out[name] = wmape(b_test, pred)
        except Exception:
            pass
    if with_sarimax:
        # ~40x slower than the rest at catalog scale (pathological SARIMAX convergence on
        # intermittent series); off by default and reported separately at smaller N.
        try:
            from sarimax_selected import run_sarimax_selected
            pred = run_sarimax_selected(b, fp, fs, horizon)[0]
            if pred is not None and len(pred) == horizon and np.all(np.isfinite(pred)):
                out["SARIMAX"] = wmape(b_test, pred)
        except Exception:
            pass
    return out


def _global_kernel(dataset, cfg, horizon, n_fit=500):
    """Median-pooled promotional kernel, estimated on the training window only."""
    kernels = []
    for s in dataset[:n_fit]:
        b = np.asarray(s["demand"], float)
        f = np.asarray(s["f_promo"], float)
        split = len(b) - horizon
        phi, n_ev = estimate_kernel(b[:split], f[:split], cfg["n_lags"], cfg["pre_window"])
        theta = fit_parametric_kernel(phi, family="surge_vacuum") if phi is not None else None
        if theta is not None:
            kernels.append(kernel_from_theta(theta, cfg["n_lags"]))
    phi_snap = np.zeros(cfg["n_lags"] + 1)
    phi_snap[0] = 1.0
    phi_promo = np.median(np.stack(kernels), axis=0) if kernels else phi_snap.copy()
    return {"phi_promo": phi_promo, "phi_snap": phi_snap}


def run_seed(seed, n_skus, cfg, horizon, n_jobs, with_sarimax=False):
    dataset = load_m5_data(DATA_DIR, n_skus=n_skus, random_state=seed, skip_prices=True)
    gk = _global_kernel(dataset, cfg, horizon)
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_eval_sku)(s, gk, cfg, horizon, with_sarimax) for s in dataset)
    df = pd.DataFrame([{"seed": seed, "method": m, "wMAPE": w}
                       for r in res for m, w in r.items()])
    df.to_csv(os.path.join(OUT_DIR, f"reliability_seed{seed}.csv"), index=False)  # checkpoint
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 42],
                    help="SKU-draw seeds (default: the four used in the paper)")
    ap.add_argument("--n_skus", type=int, default=2000, help="SKUs per draw (default 2000)")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--with-sarimax", action="store_true",
                    help="include the AIC-selected seasonal SARIMAX (~40x slower; off by default)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = CONFIG_DEFAULT.copy()
    cfg.update(use_ista=True, run_baselines=False)
    horizon = cfg["forecast_horizon"]

    frames = []
    for sd in args.seeds:
        print(f"\n=== seed {sd}: {args.n_skus} SKUs ===", flush=True)
        frames.append(run_seed(sd, args.n_skus, cfg, horizon, args.n_jobs, args.with_sarimax))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, "reliability_per_sku.csv"), index=False)

    by_seed = (df.groupby(["seed", "method"])["wMAPE"]
                 .agg(n="size", median="median", mean="mean", sd="std",
                      p99=lambda s: s.quantile(0.99), max="max",
                      blowup_pct=lambda s: 100.0 * (s > BLOWUP).mean())
                 .reset_index())
    by_seed.to_csv(os.path.join(OUT_DIR, "reliability_by_seed.csv"), index=False)

    summary = (by_seed.groupby("method")
               .agg(median_mean=("median", "mean"), median_sd=("median", "std"),
                    sd_mean=("sd", "mean"), sd_sd=("sd", "std"),
                    p99_mean=("p99", "mean"), p99_sd=("p99", "std"),
                    blowup_mean=("blowup_pct", "mean"), blowup_sd=("blowup_pct", "std"))
               .sort_values("sd_mean"))
    summary.insert(0, "family", [FAMILY.get(m, "?") for m in summary.index])
    summary.to_csv(os.path.join(OUT_DIR, "reliability_summary.csv"))

    print("\n" + "=" * 104)
    print(f"TABLE tab:reliability — mean +/- SD over {len(args.seeds)} draws of {args.n_skus} SKUs")
    print("=" * 104)
    for m, r in summary.iterrows():
        print(f"{m:14s} {r.family:12s} median {r.median_mean:5.3f}+/-{r.median_sd:4.3f} | "
              f"SD {r.sd_mean:6.3f}+/-{r.sd_sd:5.3f} | p99 {r.p99_mean:6.2f}+/-{r.p99_sd:5.2f} | "
              f"blowup {r.blowup_mean:5.2f}%+/-{r.blowup_sd:4.2f}")
    print("\nrank of CD (1 = lowest) in each draw:")
    for sd in sorted(by_seed.seed.unique()):
        g = by_seed[by_seed.seed == sd]
        r_sd = g.sort_values("sd")["method"].tolist().index("CD") + 1
        r_bl = g.sort_values("blowup_pct")["method"].tolist().index("CD") + 1
        print(f"  seed {sd:>3}: SD rank {r_sd}/{len(g)} | blowup rank {r_bl}/{len(g)}")
    print(f"\nwrote reliability_{{per_sku,by_seed,summary}}.csv to {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
