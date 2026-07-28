#!/usr/bin/env python3
"""Favorita headline (Table `tab:favorita`, Finding 5) with the corrected loader: live
on-promotion + holiday channels (both were dead before the July-16 fix). CD (global-pooled,
2-channel) vs Tuned XGBoost, both receiving the same known calendar. Paper filter: exclude
items whose test-window std < 0.5 (VR undefined/degenerate there).

Writes
------
  results/favorita_vr_peritem.csv     per-item paired VRs  (consumed by figures/generate_favorita_vr_distribution.py)
  results/favorita_headline.csv       the summary row behind tab:favorita
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from contextual_deconvolution.loaders import load_favorita_data
from contextual_deconvolution.core import (build_Aspec, decompose_ista, estimate_kernel,
                                           fit_parametric_kernel, kernel_from_theta)
from contextual_deconvolution.baselines import run_tuned_xgboost
from contextual_deconvolution.metrics import inventory_cost, wmape, variance_ratio, rmse

FAV = os.path.join(_HERE, "..", "data", "favorita-grocery-sales-forecasting")
OUT = os.path.join(_HERE, "..", "results")
H = 28; NL = 14; LS, LP = 10.0, 8.0
_KERNS = None


def est(b, f, split):
    if f[:split].sum() < 3:
        return None
    phi, ne = estimate_kernel(b[:split], f[:split], NL, 30)
    if phi is None or ne < 2:
        return None
    th = fit_parametric_kernel(phi, family="surge_vacuum")
    return kernel_from_theta(th, NL) if th is not None else phi


def xb_fit(b, split, fp, fs):
    """CD fitted values on the training window = x_base + A x_spec."""
    A = None; z = np.zeros(NL + 1); zf = np.zeros(split)
    ch = {"f_promo": fp, "f_snap": fs}
    for nm in ("f_promo", "f_snap"):
        Am = build_Aspec(_KERNS[nm], z, ch[nm][:split], zf, split, causal=True)
        A = Am if A is None else A + Am
    xb, xs = decompose_ista(b[:split], A, LS, LP)
    return xb + A @ xs


def cd_forecast(b, chans, kerns, split):
    A = None; z = np.zeros(NL + 1); zf = np.zeros(split)
    for nm in ("f_promo", "f_snap"):
        Am = build_Aspec(kerns[nm], z, chans[nm][:split], zf, split, causal=True)
        A = Am if A is None else A + Am
    xb, xs = decompose_ista(b[:split], A, LS, LP)
    rec = xb[-30:]; tr = (rec[-1] - rec[0]) / 30.0 if len(rec) > 1 else 0.0
    base = xb[-1] + tr * np.arange(1, H + 1)
    pos = xs[xs > 1e-9]; fb = np.median(pos) if pos.size else float(np.std(xb) * 0.5)
    fut = np.zeros(H)
    for nm in ("f_promo", "f_snap"):
        h_ = xs[:split][chans[nm][:split] > 0]; h_ = h_[h_ > 1e-9]
        fut += (np.median(h_) if h_.size >= 3 else fb) * chans[nm][split:split + H]
    Af = None
    for nm in ("f_promo", "f_snap"):
        Am = build_Aspec(kerns[nm], z, chans[nm][split:split + H], np.zeros(H), H, causal=True)
        Af = Am if Af is None else Af + Am
    return np.maximum(base + Af @ fut, 0.0)


def main():
    global _KERNS
    os.makedirs(OUT, exist_ok=True)
    print("loading Favorita (corrected channels; n=3000 -> ~2863 pass min length) ...", flush=True)
    ds = load_favorita_data(FAV, n_items=3000, random_state=42)
    ds = [s for s in ds if len(s["demand"]) >= NL + 2 * H + 10]
    print(f"kept {len(ds)} items", flush=True)
    ksp, ksh = [], []
    for s in ds:
        b = np.asarray(s["demand"], float); sp = len(b) - H
        kp_ = est(b, np.asarray(s["f_promo"], float), sp)
        kh_ = est(b, np.asarray(s["f_snap"], float), sp)
        if kp_ is not None: ksp.append(kp_)
        if kh_ is not None: ksh.append(kh_)
    z = np.zeros(NL + 1); z[0] = 1.0
    kerns = {"f_promo": np.median(np.stack(ksp), 0) if ksp else z.copy(),
             "f_snap":  np.median(np.stack(ksh), 0) if ksh else z.copy()}
    _KERNS = kerns
    carry = lambda p: 1 - p[0] ** 2 / np.sum(np.asarray(p) ** 2)
    print(f"onpromotion kernel {np.round(kerns['f_promo'][:6],3)} carryover={carry(kerns['f_promo']):.3f}")
    print(f"holiday     kernel {np.round(kerns['f_snap'][:6],3)} carryover={carry(kerns['f_snap']):.3f}")

    def one(s):
        b = np.asarray(s["demand"], float); fp = np.asarray(s["f_promo"], float)
        fs = np.asarray(s["f_snap"], float); sp = len(b) - H; bt = b[sp:]
        if np.std(bt) < 0.5 or np.std(b[:sp]) < 1e-9:
            return None
        o = {}
        try:
            p = cd_forecast(b, {"f_promo": fp, "f_snap": fs}, kerns, sp)
            _, ss = inventory_cost(bt, p, xb_fit(b, sp, fp, fs), lead_time=7, h=1.0, p=10.0, service_level=0.95)
            o["cd"] = dict(vr=variance_ratio(p, bt), wm=wmape(bt, p), rmse=rmse(bt, p), ss=ss)
        except Exception:
            pass
        try:
            xp, xa, xf, _ = run_tuned_xgboost(b, fp, fs, H)
            if xp is not None:
                _, xss = inventory_cost(xa, xp, xf, lead_time=7, h=1.0, p=10.0, service_level=0.95)
                o["xgb"] = dict(vr=variance_ratio(xp, xa), wm=wmape(xa, xp), rmse=rmse(xa, xp), ss=xss)
        except Exception:
            pass
        return o if o else None

    res = [r for r in Parallel(n_jobs=-1, backend="loky", verbose=2)(delayed(one)(s) for s in ds) if r]

    def med(k, f):
        v = [r[k][f] for r in res if k in r and np.isfinite(r[k][f])]; return float(np.median(v)) if v else np.nan
    def ms(k, f):
        v = np.array([r[k][f] for r in res if k in r and np.isfinite(r[k][f])]); return float(v.mean()), float(v.std())

    pair = [(r["cd"], r["xgb"]) for r in res if "cd" in r and "xgb" in r]
    vr_win = float(np.mean([c["vr"] < x["vr"] for c, x in pair]))
    pv = wilcoxon([c["vr"] for c, x in pair], [x["vr"] for c, x in pair]).pvalue if len(pair) > 10 else np.nan
    ratio = med("xgb", "vr") / max(med("cd", "vr"), 1e-12)
    n_cd = sum("cd" in r for r in res); n_xgb = sum("xgb" in r for r in res)

    # per-item paired VRs -> feeds the VR-distribution figure (no second expensive eval)
    pd.DataFrame({"cd_vr": [c["vr"] for c, x in pair], "xgb_vr": [x["vr"] for c, x in pair]}).to_csv(
        os.path.join(OUT, "favorita_vr_peritem.csv"), index=False)

    rows = []
    for k, lab in (("cd", "CD"), ("xgb", "Tuned XGBoost")):
        wm_m, wm_s = ms(k, "wm"); rm_m, rm_s = ms(k, "rmse")
        rows.append(dict(method=lab, n=(n_cd if k == "cd" else n_xgb),
                         wmape_mean=wm_m, wmape_std=wm_s, wmape_med=med(k, "wm"),
                         rmse_mean=rm_m, rmse_std=rm_s, vr_med=med(k, "vr"), ss_med=med(k, "ss")))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "favorita_headline.csv"), index=False)

    print("\n=== FAVORITA HEADLINE (corrected channels) — tab:favorita ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  VR ratio XGB/CD: {ratio:.1f}x | CD wins VR {100*vr_win:.1f}% (paired n={len(pair)}, p={pv:.2e})")
    print(f"  wrote favorita_headline.csv and favorita_vr_peritem.csv", flush=True)


if __name__ == "__main__":
    main()
