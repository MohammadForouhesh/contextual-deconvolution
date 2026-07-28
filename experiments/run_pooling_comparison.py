#!/usr/bin/env python3
"""Regenerate tab:pooling_comparison (M5, 200 SKUs): CD forecast quality under per-SKU,
category-pooled, and global-pooled kernels (mean +/- std of wMAPE, RMSE, VR, safety stock).
Shows hierarchical pooling does not degrade quality. Live-SNAP loader.

Writes results/pooling_comparison.csv.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, warnings
from collections import defaultdict
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from joblib import Parallel, delayed
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.core import estimate_kernel, fit_parametric_kernel, kernel_from_theta
from contextual_deconvolution.experiments import evaluate_sku_pooled, CONFIG_DEFAULT

DATA = os.path.join(_HERE, "..", "data", "m5-data", "m5-forecasting-accuracy")
OUT = os.path.join(_HERE, "..", "results")
cfg = CONFIG_DEFAULT.copy(); cfg.update(use_ista=True, run_baselines=False)
H = cfg["forecast_horizon"]; NL = cfg["n_lags"]
IMP = np.zeros(NL + 1); IMP[0] = 1.0  # impulse SNAP kernel (snap handled by baseline here)


def est_kernel(sku):
    b = np.asarray(sku["demand"], float); f = np.asarray(sku["f_promo"], float); sp = len(b) - H
    phi, ne = estimate_kernel(b[:sp], f[:sp], NL, cfg["pre_window"])
    th = fit_parametric_kernel(phi, family="surge_vacuum") if phi is not None else None
    return kernel_from_theta(th, NL) if th is not None else None


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_m5_data(DATA, n_skus=200, random_state=42, skip_prices=True)
    print(f"[pooling] {len(ds)} SKUs; estimating per-SKU kernels...", flush=True)

    # Pass 1: per-SKU kernels (parallel)
    kernels = Parallel(n_jobs=-1, backend="loky")(delayed(est_kernel)(s) for s in ds)
    # Pool: global = median of all valid; category = median within category
    valid = [k for k in kernels if k is not None]
    gphi = np.median(np.stack(valid), 0) if valid else IMP.copy()
    bycat = defaultdict(list)
    for s, k in zip(ds, kernels):
        if k is not None:
            bycat[s["category"]].append(k)
    cat_kernel = {c: np.median(np.stack(ks), 0) for c, ks in bycat.items()}

    # Pass 2: evaluate CD under each pooling strategy (parallel)
    def one(i):
        s = ds[i]
        b = np.asarray(s["demand"], float); sp = len(b) - H; bt = b[sp:]
        if np.all(bt == 0) or np.std(b[:sp]) < 1e-9:
            return None
        kps = kernels[i] if kernels[i] is not None else gphi
        strat = {"Per-SKU": kps,
                 "Category-Pooled": cat_kernel.get(s["category"], gphi),
                 "Global-Pooled": gphi}
        o = {}
        for nm, kp in strat.items():
            try:
                r = evaluate_sku_pooled(s, kp, IMP, cfg)["forecasts"]["CD"]
                o[nm] = (r["wmape"], r["rmse"], r["variance_ratio"], r["safety_stock"])
            except Exception:
                pass
        return o if len(o) == 3 else None

    res = [r for r in Parallel(n_jobs=-1, backend="loky", verbose=1)(delayed(one)(i) for i in range(len(ds))) if r]
    print(f"[pooling] {len(res)} SKUs with all three strategies", flush=True)

    rows = []
    for nm in ("Per-SKU", "Category-Pooled", "Global-Pooled"):
        A = np.array([r[nm] for r in res])  # cols: wmape, rmse, vr, ss
        rows.append(dict(strategy=nm, n=len(A),
                         wmape_mean=A[:, 0].mean(), wmape_std=A[:, 0].std(),
                         rmse_mean=A[:, 1].mean(), rmse_std=A[:, 1].std(),
                         vr_mean=A[:, 2].mean(), vr_std=A[:, 2].std(),
                         ss_mean=A[:, 3].mean(), ss_std=A[:, 3].std()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "pooling_comparison.csv"), index=False)
    print("\n=== POOLING COMPARISON (M5, 200 SKUs) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("wrote pooling_comparison.csv", flush=True)


if __name__ == "__main__":
    main()
