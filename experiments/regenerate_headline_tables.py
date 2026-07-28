#!/usr/bin/env python3
"""Regenerate the paper's headline M5 results: Table 1 (scale validation), Finding 1, Table 2.

  Table 1  CD on the full 30,490-SKU catalog under category- and global-pooled kernels
  Finding 1 CD vs Vanilla/Tuned XGBoost on a 2,000-SKU clean subset (VR, safety-stock win-rates)
  Table 2  all methods on the same stratified 200-SKU subset (wMAPE, MASE, RMSSE, VR, Ord.Var)

Protocol: strictly out-of-sample (train-only decomposition, recursive baselines), information
parity on the exogenous calendar, seeded.

Outputs -> results/{table1_scale,finding1_xgb,table2_baselines}.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..")); sys.path.insert(0, _HERE)
from joblib import Parallel, delayed
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.core import estimate_kernel, fit_parametric_kernel, kernel_from_theta
from contextual_deconvolution.experiments import (evaluate_sku_pooled, estimate_all_kernels,
                                                  pool_kernels, CONFIG_DEFAULT)
from contextual_deconvolution.metrics import (wmape, variance_ratio, mase, rmsse,
                                              order_variance_under_replenishment as order_var)

DATA = os.path.join(_HERE, "..", "data", "m5-data", "m5-forecasting-accuracy")
OUT = os.path.join(_HERE, "..", "results")
cfg = CONFIG_DEFAULT.copy(); cfg.update(use_ista=True, run_baselines=False)
H = cfg["forecast_horizon"]; NL = cfg["n_lags"]


def global_kernel(ds, n_fit=500):
    ks = []
    for s in ds[:n_fit]:
        b = np.asarray(s["demand"], float); f = np.asarray(s["f_promo"], float)
        sp = len(b) - H
        phi, ne = estimate_kernel(b[:sp], f[:sp], NL, cfg["pre_window"])
        th = fit_parametric_kernel(phi, family="surge_vacuum") if phi is not None else None
        if th is not None: ks.append(kernel_from_theta(th, NL))
    z = np.zeros(NL + 1); z[0] = 1.0
    return {"phi_promo": np.median(np.stack(ks), axis=0) if ks else z.copy(), "phi_snap": z}


def cd_metrics(sku, kern):
    b = np.asarray(sku["demand"], float); sp = len(b) - H; bt = b[sp:]
    if np.all(bt == 0) or np.std(b[:sp]) < 1e-9: return None
    try:
        r = evaluate_sku_pooled(sku, kern["phi_promo"], kern["phi_snap"], cfg)["forecasts"]["CD"]
        return dict(wmape=r["wmape"], vr=r["variance_ratio"], ss=r["safety_stock"])
    except Exception:
        return None


# ---------------------------------------------------------------- Table 1
def table1(n_skus, n_jobs):
    ds = load_m5_data(DATA, n_skus=(n_skus or None), random_state=42, skip_prices=True)
    print(f"[Table 1] {len(ds)} SKUs", flush=True)
    kern_all = estimate_all_kernels(ds, cfg)
    rows = []
    for strategy in ("category", "global"):
        pooled = pool_kernels(kern_all, strategy=strategy)
        res = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
            delayed(cd_metrics)(s, pooled[s.get("sku_id", 0)]) for s in ds)
        res = [r for r in res if r]
        d = pd.DataFrame(res)
        rows.append(dict(strategy=strategy, n=len(d),
                         wmape_mean=d.wmape.mean(), wmape_sd=d.wmape.std(), wmape_med=d.wmape.median(),
                         vr_mean=d.vr.mean(), vr_sd=d.vr.std(), vr_med=d.vr.median(),
                         ss_mean=d.ss.mean(), ss_sd=d.ss.std(), ss_med=d.ss.median(),
                         vr_gt_0p1=100 * (d.vr > 0.1).mean(), vr_p95=d.vr.quantile(.95)))
        print(f"  {strategy:9s} wMAPE {d.wmape.mean():.2f}+/-{d.wmape.std():.2f} (med {d.wmape.median():.2f}) | "
              f"VR {d.vr.mean():.3f}+/-{d.vr.std():.2f} (med {d.vr.median():.4f}) | "
              f"SS {d.ss.mean():.3f}+/-{d.ss.std():.2f} (med {d.ss.median():.3f})", flush=True)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "table1_scale.csv"), index=False)
    return t


# ---------------------------------------------------------------- Finding 1
def finding1(n_jobs):
    ds = load_m5_data(DATA, n_skus=2000, random_state=42, skip_prices=True)
    kern = global_kernel(ds)
    def one(sku):
        from contextual_deconvolution.baselines import run_tuned_xgboost, run_xgboost
        b = np.asarray(sku["demand"], float); fp = np.asarray(sku["f_promo"], float)
        fs = np.asarray(sku["f_snap"], float); sp = len(b) - H; bt = b[sp:]
        if np.all(bt == 0) or np.std(b[:sp]) < 1e-9: return None
        o = {}
        c = cd_metrics(sku, kern)
        if c: o["cd"] = c
        for nm, fn in (("txgb", run_tuned_xgboost), ("vxgb", run_xgboost)):
            try:
                p, a, ft, _ = fn(b, fp, fs, H)
                if p is not None:
                    from contextual_deconvolution.metrics import inventory_cost
                    _, ss = inventory_cost(a, p, ft, lead_time=cfg["lead_time"], h=cfg["holding_cost"],
                                           p=cfg["stockout_cost"], service_level=cfg["service_level"])
                    o[nm] = dict(wmape=wmape(a, p), vr=variance_ratio(p, a), ss=ss)
            except Exception: pass
        return o
    res = [r for r in Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(delayed(one)(s) for s in ds) if r]
    def med(k, f):
        v = [r[k][f] for r in res if k in r and np.isfinite(r[k][f])]
        return float(np.median(v)) if v else np.nan
    pair = [(r["cd"], r["txgb"]) for r in res if "cd" in r and "txgb" in r]
    vr_win = float(np.mean([c["vr"] < x["vr"] for c, x in pair]))
    ss_win = float(np.mean([c["ss"] < x["ss"] for c, x in pair]))
    out = dict(n=len(res), n_paired=len(pair),
               cd_vr=med("cd","vr"), cd_wmape=med("cd","wmape"),
               txgb_vr=med("txgb","vr"), txgb_wmape=med("txgb","wmape"),
               vxgb_vr=med("vxgb","vr"), ratio=med("txgb","vr")/max(med("cd","vr"),1e-12),
               cd_wins_vr=100*vr_win, cd_wins_ss=100*ss_win)
    print(f"[Finding 1] CD VR {out['cd_vr']:.4f} | TunedXGB VR {out['txgb_vr']:.4f} "
          f"({out['ratio']:.1f}x) wMAPE {out['txgb_wmape']:.2f} | VanillaXGB VR {out['vxgb_vr']:.2f} | "
          f"CD wins VR {out['cd_wins_vr']:.1f}% SS {out['cd_wins_ss']:.1f}% (n={out['n_paired']})", flush=True)
    pd.DataFrame([out]).to_csv(os.path.join(OUT, "finding1_xgb.csv"), index=False)
    return out


# ---------------------------------------------------------------- Table 2
def table2(n_jobs):
    ds = load_m5_data(DATA, n_skus=200, random_state=42, skip_prices=True)
    kern = global_kernel(ds, n_fit=200)
    def one(sku):
        from contextual_deconvolution.baselines import (run_tuned_xgboost, run_xgboost, run_lightgbm,
                                                        run_prophet_like, run_stl_arima)
        from contextual_deconvolution.baselines_dl import run_patchtst
        import run_structural_baselines as SB
        b = np.asarray(sku["demand"], float); fp = np.asarray(sku["f_promo"], float)
        fs = np.asarray(sku["f_snap"], float); sp = len(b) - H; bt = b[sp:]; tr = b[:sp]
        if np.all(bt == 0) or np.std(tr) < 1e-9: return None
        o = {}
        def put(nm, p):
            if p is None or len(p) != H or not np.all(np.isfinite(p)): return
            o[nm] = dict(wmape=wmape(bt, p), mase=mase(bt, p), rmsse=rmsse(bt, p),
                         vr=variance_ratio(p, bt),
                         ov=order_var(p, bt, lead_time=cfg["lead_time"]))
        c = cd_metrics(sku, kern)
        if c:
            try:
                r = evaluate_sku_pooled(sku, kern["phi_promo"], kern["phi_snap"], cfg)["forecasts"]["CD"]
                put("CD", np.asarray(r["pred"], float))
            except Exception: pass
        for nm, fn in (("Tuned XGBoost", run_tuned_xgboost), ("LightGBM", run_lightgbm),
                       ("Prophet-like", run_prophet_like), ("STL+ARIMA", run_stl_arima),
                       ("ARIMAX", SB.run_arimax), ("ETS", SB.run_ets_covariates),
                       ("DL-Ridge", SB.run_distributed_lag_ridge)):
            try:
                out_ = fn(b, fp, fs, H); put(nm, out_[0] if isinstance(out_, tuple) else out_)
            except Exception: pass
        try: put("PatchTST", run_patchtst(b, fp, fs, H, device="cpu")[0])
        except Exception: pass
        return o
    res = [r for r in Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(delayed(one)(s) for s in ds) if r]
    methods = ["ARIMAX","ETS","Prophet-like","DL-Ridge","LightGBM","Tuned XGBoost","PatchTST","STL+ARIMA","CD"]
    rows = []
    for m in methods:
        v = [r[m] for r in res if m in r]
        if not v: continue
        d = pd.DataFrame(v)
        rows.append(dict(method=m, n=len(d), wmape=d.wmape.median(), mase=d.mase.median(),
                         rmsse=d.rmsse.median(), vr=d.vr.median(), ord_var=d.ov.median()))
    t = pd.DataFrame(rows)
    print("\n[Table 2] median over 200 SKUs")
    print(t.to_string(index=False, float_format=lambda x: f"{x:8.4f}"), flush=True)
    t.to_csv(os.path.join(OUT, "table2_baselines.csv"), index=False)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=int, default=0, help="SKUs for Table 1 (0 = entire catalog)")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--only", default="all", choices=["all", "t1", "f1", "t2"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.only in ("all", "t2"): table2(a.n_jobs)
    if a.only in ("all", "f1"): finding1(a.n_jobs)
    if a.only in ("all", "t1"): table1(a.full, a.n_jobs)
    print("\nwrote table1_scale.csv, finding1_xgb.csv, table2_baselines.csv to results/")
