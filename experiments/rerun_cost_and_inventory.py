#!/usr/bin/env python3
"""Re-run Finding 4b: the h/p cost crossover (bootstrap CI) and the inventory-simulation
decomposition, with the corrected loader (live SNAP channel).

Strictly out-of-sample: CD fit on the training window, Tuned XGBoost forecasts recursively,
both given the same calendar. Cost model = inventory_cost_with_events (base-stock policy,
lead time 7, stockout cost fixed at $10/unit, holding cost varied to sweep h/p).

Outputs -> results/{cost_crossover,inventory_decomposition}.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..")); sys.path.insert(0, _HERE)
from joblib import Parallel, delayed
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.core import estimate_kernel, fit_parametric_kernel, kernel_from_theta, build_Aspec
from contextual_deconvolution.experiments import evaluate_sku_pooled, forecast_two_stream, CONFIG_DEFAULT
from contextual_deconvolution.baselines import run_tuned_xgboost
from contextual_deconvolution.metrics import inventory_cost_with_events

DATA = os.path.join(_HERE, "..", "data", "m5-data", "m5-forecasting-accuracy")
OUT = os.path.join(_HERE, "..", "results")
cfg = CONFIG_DEFAULT.copy(); cfg.update(use_ista=True, run_baselines=False)
H = cfg["forecast_horizon"]; NL = cfg["n_lags"]; P = 10.0
# Fine, uniform grid over the operationally meaningful range around the crossover (h/p in [0, 0.4]).
RATIOS = np.round(np.arange(0.0, 0.4001, 0.02), 3)

def gkernel(ds, n_fit=500):
    ks = []
    for s in ds[:n_fit]:
        b = np.asarray(s["demand"], float); f = np.asarray(s["f_promo"], float); sp = len(b)-H
        phi, ne = estimate_kernel(b[:sp], f[:sp], NL, cfg["pre_window"])
        th = fit_parametric_kernel(phi, family="surge_vacuum") if phi is not None else None
        if th is not None: ks.append(kernel_from_theta(th, NL))
    z = np.zeros(NL+1); z[0] = 1.0
    return {"phi_promo": np.median(np.stack(ks), 0) if ks else z.copy(), "phi_snap": z}

def per_sku(sku, gk):
    b = np.asarray(sku["demand"], float); fp = np.asarray(sku["f_promo"], float); fs = np.asarray(sku["f_snap"], float)
    sp = len(b)-H; bt = b[sp:]; emask = fp[sp:sp+H]
    if np.all(bt == 0) or np.std(b[:sp]) < 1e-9: return None
    try:
        r = evaluate_sku_pooled(sku, gk["phi_promo"], gk["phi_snap"], cfg)
        cd_pred, _, _ = forecast_two_stream(r["x_base"], r["x_spec"], fp[sp:], fs[sp:], r["phi_promo"], r["phi_snap"])
        A = build_Aspec(r["phi_promo"], r["phi_snap"], fp[:sp], fs[:sp], sp, causal=True)
        cd_fit = r["x_base"] + A @ r["x_spec"]
        xg_pred, xa, xg_fit, _ = run_tuned_xgboost(b, fp, fs, H)
        if xg_pred is None: return None
        cd_c, xg_c = [], []
        cd_ss = xg_ss = cd_hold = xg_hold = cd_stk = xg_stk = None
        for rr in RATIOS:
            h = rr*P
            ac, ss_a, th_a, ts_a, _, _ = inventory_cost_with_events(bt, cd_pred, emask, fitted_train=cd_fit, lead_time=7, h=h, p=P, service_level=0.95)
            bc, ss_b, th_b, ts_b, _, _ = inventory_cost_with_events(bt, xg_pred, emask, fitted_train=xg_fit, lead_time=7, h=h, p=P, service_level=0.95)
            cd_c.append(ac); xg_c.append(bc)
            if abs(rr-0.10) < 1e-9:   # decomposition reported at h/p = 0.1
                cd_ss, xg_ss = ss_a, ss_b; cd_hold, xg_hold = th_a, th_b; cd_stk, xg_stk = ts_a, ts_b
        return dict(cd_c=np.array(cd_c), xg_c=np.array(xg_c),
                    cd_ss=cd_ss, xg_ss=xg_ss, cd_hold=cd_hold, xg_hold=xg_hold, cd_stk=cd_stk, xg_stk=xg_stk)
    except Exception:
        return None

def crossover(cd_med, xg_med):
    d = cd_med - xg_med
    for i in range(len(d)-1):
        if d[i] > 0 and d[i+1] <= 0:
            return RATIOS[i] + (RATIOS[i+1]-RATIOS[i])*d[i]/(d[i]-d[i+1])
    return RATIOS[0] if d[0] <= 0 else np.nan

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_skus", type=int, default=1000); ap.add_argument("--n_jobs", type=int, default=-1)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    ds = load_m5_data(DATA, n_skus=a.n_skus, random_state=42, skip_prices=True)
    gk = gkernel(ds)
    res = [r for r in Parallel(n_jobs=a.n_jobs, backend="loky", verbose=2)(delayed(per_sku)(s, gk) for s in ds) if r]
    n = len(res); print(f"\nvalid SKUs: {n}")
    cd = np.stack([r["cd_c"] for r in res]); xg = np.stack([r["xg_c"] for r in res])
    pt = crossover(np.median(cd, 0), np.median(xg, 0))
    rng = np.random.default_rng(42); bs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n); c = crossover(np.median(cd[idx], 0), np.median(xg[idx], 0))
        if np.isfinite(c): bs.append(c)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    print("\n=== FINDING 4b: cost crossover (live SNAP) ===")
    print(f"  crossover h/p = {pt:.3f} | 95% CI [{lo:.3f}, {hi:.3f}] (B=2000, {100*len(bs)/2000:.0f}% crossed)")
    cm, xm = np.median(cd, 0), np.median(xg, 0)
    print("   h/p    CD$    XGB$   CD-XGB")
    for i, rr in enumerate(RATIOS):
        print(f"  {rr:5.2f} {cm[i]:7.2f} {xm[i]:7.2f} {cm[i]-xm[i]:+7.3f}")
    pd.DataFrame({"h_p": RATIOS, "cd_cost": cm, "xgb_cost": xm}).to_csv(os.path.join(OUT, "cost_crossover.csv"), index=False)

    def med(k): v = [r[k] for r in res if np.isfinite(r[k])]; return float(np.median(v))
    def win(a_, b_): p = [(r[a_], r[b_]) for r in res if np.isfinite(r[a_]) and np.isfinite(r[b_])]; return 100*np.mean([x < y for x, y in p])
    print("\n=== INVENTORY DECOMPOSITION at h/p=0.1 (median per SKU) ===")
    print(f"  safety stock : CD {med('cd_ss'):.3f} vs XGB {med('xg_ss'):.3f}  (CD lower on {win('cd_ss','xg_ss'):.0f}%)")
    print(f"  holding cost : CD {med('cd_hold'):.2f} vs XGB {med('xg_hold'):.2f}  (CD lower on {win('cd_hold','xg_hold'):.0f}%)")
    print(f"  stockout cost: CD {med('cd_stk'):.2f} vs XGB {med('xg_stk'):.2f}  (CD lower on {win('cd_stk','xg_stk'):.0f}%)")
    pd.DataFrame([dict(cd_ss=med('cd_ss'), xg_ss=med('xg_ss'), ss_win=win('cd_ss','xg_ss'),
                       cd_hold=med('cd_hold'), xg_hold=med('xg_hold'),
                       cd_stk=med('cd_stk'), xg_stk=med('xg_stk'), stk_win=win('cd_stk','xg_stk'),
                       crossover=pt, ci_lo=lo, ci_hi=hi, n=n)]).to_csv(os.path.join(OUT, "inventory_decomposition.csv"), index=False)
    print(f"\nwrote cost_crossover.csv, inventory_decomposition.csv")

if __name__ == "__main__":
    main()
