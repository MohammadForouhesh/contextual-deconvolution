#!/usr/bin/env python3
"""Re-run (live SNAP) two M5 tables that used CD:
  (A) kernel-ablation A=I row  : learned global kernel vs impulse operator, 2,000 SKUs, + hobbies
  (B) promotion-density table   : CD vs Tuned XGBoost stratified by per-SKU promo density

Outputs -> results/{kernel_ablation_m5,promo_density}.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..")); sys.path.insert(0, _HERE)
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.core import estimate_kernel, fit_parametric_kernel, kernel_from_theta
from contextual_deconvolution.experiments import evaluate_sku_pooled, CONFIG_DEFAULT
from contextual_deconvolution.baselines import run_tuned_xgboost
from contextual_deconvolution.metrics import wmape, variance_ratio

DATA = os.path.join(_HERE, "..", "data", "m5-data", "m5-forecasting-accuracy")
OUT = os.path.join(_HERE, "..", "results")
cfg = CONFIG_DEFAULT.copy(); cfg.update(use_ista=True, run_baselines=False)
H = cfg["forecast_horizon"]; NL = cfg["n_lags"]

def gkernel(ds, n_fit=500):
    ks = []
    for s in ds[:n_fit]:
        b = np.asarray(s["demand"], float); f = np.asarray(s["f_promo"], float); sp = len(b)-H
        phi, ne = estimate_kernel(b[:sp], f[:sp], NL, cfg["pre_window"])
        th = fit_parametric_kernel(phi, family="surge_vacuum") if phi is not None else None
        if th is not None: ks.append(kernel_from_theta(th, NL))
    z = np.zeros(NL+1); z[0] = 1.0
    return np.median(np.stack(ks), 0) if ks else z.copy(), z

def ablation(n_jobs, n_skus=2000):
    ds = load_m5_data(DATA, n_skus=n_skus, random_state=42, skip_prices=True)
    gphi, psnap = gkernel(ds)
    imp = np.zeros(NL+1); imp[0] = 1.0
    def one(sku):
        b = np.asarray(sku["demand"], float); sp = len(b)-H; bt = b[sp:]
        if np.all(bt == 0) or np.std(b[:sp]) < 1e-9: return None
        o = {}
        for nm, kp in (("learned", gphi), ("impulse", imp)):
            try:
                r = evaluate_sku_pooled(sku, kp, imp if nm == "impulse" else psnap, cfg)["forecasts"]["CD"]
                o[nm] = (r["variance_ratio"], r["wmape"], r["safety_stock"])
            except Exception: pass
        return o
    res = [r for r in Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(delayed(one)(s) for s in ds) if r]
    def med(k, i): v = [r[k][i] for r in res if k in r and np.isfinite(r[k][i])]; return float(np.median(v))
    def win(i): p = [(r["learned"][i], r["impulse"][i]) for r in res if "learned" in r and "impulse" in r]; return 100*np.mean([x < y for x, y in p])
    row = dict(dataset="M5 full (2000)", n=len(res),
               learned_vr=med("learned",0), learned_wm=med("learned",1), learned_ss=med("learned",2),
               impulse_vr=med("impulse",0), impulse_wm=med("impulse",1), impulse_ss=med("impulse",2),
               learned_wins_vr=win(0))
    print(f"[ablation] learned VR {row['learned_vr']:.4f} wMAPE {row['learned_wm']:.3f} | "
          f"impulse VR {row['impulse_vr']:.4f} wMAPE {row['impulse_wm']:.3f} | learned wins VR {row['learned_wins_vr']:.0f}%", flush=True)
    pd.DataFrame([row]).to_csv(os.path.join(OUT, "kernel_ablation_m5.csv"), index=False)

def promo_density(n_jobs, n_skus=2000):
    ds = load_m5_data(DATA, n_skus=n_skus, random_state=42, skip_prices=False)  # need per-SKU promo_depth
    gphi, psnap = gkernel(ds)
    def one(sku):
        b = np.asarray(sku["demand"], float); fp = np.asarray(sku["f_promo"], float); fs = np.asarray(sku["f_snap"], float)
        sp = len(b)-H; bt = b[sp:]
        if np.all(bt == 0) or np.std(b[:sp]) < 1e-9: return None
        pd_ = np.asarray(sku.get('promo_depth', np.zeros(len(b))), float)
        dens = float((pd_[:sp] > 0.01).mean())   # per-SKU promotion density (price-discount days)
        o = {"dens": dens}
        try:
            r = evaluate_sku_pooled(sku, gphi, psnap, cfg)["forecasts"]["CD"]
            o["cd"] = (r["variance_ratio"], r["wmape"], r["safety_stock"])
        except Exception: pass
        try:
            p, a, ft, _ = run_tuned_xgboost(b, fp, fs, H)
            if p is not None:
                from contextual_deconvolution.metrics import inventory_cost
                _, xss = inventory_cost(a, p, ft, lead_time=7, h=1.0, p=10.0, service_level=0.95)
                o["xgb"] = (variance_ratio(p, a), wmape(a, p), xss)
        except Exception: pass
        return o
    res = [r for r in Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(delayed(one)(s) for s in ds) if r and "cd" in r and "xgb" in r]
    dens = np.array([r["dens"] for r in res]); qs = np.quantile(dens, [0, .33, .66, 1.0])
    rows = []
    for lo, hi, lab in ((qs[0], qs[1], "low"), (qs[1], qs[2], "medium"), (qs[2], qs[3]+1e-9, "high")):
        sub = [r for r in res if lo <= r["dens"] < hi]
        if not sub: continue
        vr_win = 100*np.mean([r["cd"][0] < r["xgb"][0] for r in sub])
        ss_win = 100*np.mean([r["cd"][2] < r["xgb"][2] for r in sub])
        wm_win = 100*np.mean([r["cd"][1] < r["xgb"][1] for r in sub])
        rows.append(dict(stratum=lab, n=len(sub),
                         cd_vr=np.median([r["cd"][0] for r in sub]), xgb_vr=np.median([r["xgb"][0] for r in sub]),
                         cd_wm=np.median([r["cd"][1] for r in sub]), xgb_wm=np.median([r["xgb"][1] for r in sub]),
                         vr_win=vr_win, ss_win=ss_win, wm_win=wm_win))
    t = pd.DataFrame(rows)
    print("\n[promo-density] CD vs Tuned XGBoost by promo density (live SNAP):")
    print(t.to_string(index=False, float_format=lambda x: f"{x:7.3f}"), flush=True)
    t.to_csv(os.path.join(OUT, "promo_density.csv"), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="all", choices=["all", "ablation", "density"]); ap.add_argument("--n_jobs", type=int, default=-1); ap.add_argument("--n_skus", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.only in ("all", "ablation"): ablation(a.n_jobs, a.n_skus)
    if a.only in ("all", "density"): promo_density(a.n_jobs, a.n_skus)
    print("\nwrote kernel_ablation_m5.csv, promo_density.csv")
