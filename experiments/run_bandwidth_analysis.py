#!/usr/bin/env python3
"""
Data-driven bandwidth analysis for the kernel lag n.

Estimates the empirical promotional impulse response on 500 M5 SKUs out to a LARGE
lag (28, i.e. well beyond the default n=14) and, per category, measures how much of
the response the default 14-lag window captures. This gives a non-circular, Koopman-free
justification for n=14: the window is not tuned to a metric, it is set so that it captures
essentially all of the impulse-response energy / actionable carryover.

Reported per category (foods / household / hobbies / global):
  - energy@14 : fraction of the median IR's L2 energy contained in lags 0..14
  - n_e95     : smallest lag whose window captures >= 95% of the IR energy
  - n_e99     : smallest lag whose window captures >= 99% of the IR energy
  - L10       : largest lag at which the median IR still exceeds 10% of the event-day peak

Writes results/m5_bandwidth_analysis.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.core import estimate_kernel

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "m5-data", "m5-forecasting-accuracy")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
N_LAGS = 28          # estimate well beyond the default n=14
PRE_WINDOW = 30
FORECAST_HORIZON = 28
N_SKUS = 500
MIN_EVENTS = 3


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Loading {N_SKUS} M5 SKUs...", flush=True)
    data = load_m5_data(DATA, n_skus=N_SKUS, random_state=42, skip_prices=True)

    cats = {}
    for s in data:
        b = np.asarray(s["demand"], float)
        f = np.asarray(s["f_promo"], float)
        split = len(b) - FORECAST_HORIZON
        phi, n_events = estimate_kernel(b[:split], f[:split], n_lags=N_LAGS, pre_window=PRE_WINDOW)
        if phi is None or n_events < MIN_EVENTS:
            continue
        cats.setdefault(s.get("category", "unknown"), []).append(phi)
    cats["global"] = [p for v in list(cats.values()) for p in v]

    rows = []
    for name in ["foods", "household", "hobbies", "global"]:
        if name not in cats:
            continue
        A = np.array(cats[name])
        med = np.median(A, axis=0)                       # robust category IR
        rel = np.abs(med) / (abs(med[0]) if abs(med[0]) > 1e-9 else 1.0)
        energy = med ** 2
        cum = np.cumsum(energy) / np.sum(energy)         # energy captured by lag k
        e14 = float(cum[14])
        n_e95 = int(np.argmax(cum >= 0.95))
        n_e99 = int(np.argmax(cum >= 0.99))
        # largest lag (contiguous from lag 0) whose IR still exceeds 10% of peak
        L10 = 0
        for k in range(len(rel)):
            if rel[k] >= 0.10:
                L10 = k
            elif k > 0:
                break
        rows.append(dict(category=name, n_skus=len(A), energy_at_14=round(e14, 3),
                         n_e95=n_e95, n_e99=n_e99, L10=L10))

    df = pd.DataFrame(rows)
    print("\n=== Data-driven bandwidth (M5, empirical IR estimated to lag 28) ===")
    print(df.to_string(index=False))
    out = os.path.join(OUT, "m5_bandwidth_analysis.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
