# Contextual Deconvolution for Variance-Stable Demand Sensing

Reference implementation and full reproduction pipeline for the paper:

> **Contextual Deconvolution for Variance-Stable Demand Sensing: Kernel-Modulated Operators in Promotional Retail**

Contextual Deconvolution (CD) reframes SKU-level demand sensing as a **convex inverse problem**: observed demand is decomposed into a smooth structural baseline plus a sparse stream of promotion-driven shocks propagated through a kernel-modulated banded operator. The target is *operational* stability — Bullwhip variance, safety stock, catastrophic-failure rate, and inventory cost — rather than point accuracy alone.

Everything runs **CPU-only**, is **seeded and deterministic**, and reproduces from a single command.

---

## Table of contents

- [Method in brief](#method-in-brief)
- [What the paper claims (and does not)](#what-the-paper-claims-and-does-not)
- [Headline results](#headline-results)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data setup](#data-setup)
- [Reproducing the paper](#reproducing-the-paper)
- [Evaluation protocol](#evaluation-protocol)
- [Metrics](#metrics)
- [Testing](#testing)
- [Runtime and hardware](#runtime-and-hardware)
- [Citation](#citation)

---

## Method in brief

Demand `b` is modelled as

```
b  =  x_base  +  A_spec(θ) · x_spec  +  ε
```

and recovered by solving the convex program

```
min   ‖b − (x_base + A_spec·x_spec)‖²  +  λ_smooth‖L·x_base‖²  +  λ_sparse‖x_spec‖₁
```

- **`x_base`** — smooth structural baseline (the capacity-planning signal)
- **`x_spec`** — sparse transient shocks (the event-staging signal)
- **`A_spec(θ)`** — lower-triangular banded operator whose sub-diagonals are modulated by the exogenous event calendar, built from a causal kernel `φ(k)` (impulse, exponential decay, or a signed surge-and-vacuum Difference-of-Exponentials)

The operator is **data-derived, not imposed**: the kernel is estimated per SKU and **reduces to the identity wherever the promotional response is impulsive** (most of M5, all of Favorita), contributing measurable accuracy and stability only where genuine multi-day carryover exists (e.g. the M5 *Hobbies* category). The operational gains therefore rest on the structural decomposition itself, with the kernel as an adaptive refinement.

Estimation is a **two-stage pipeline** — robust Huber detrending, then distributed-lag ridge regression for the kernel — followed by the convex decomposition with the kernel held fixed. **Hierarchical partial pooling** shrinks per-SKU kernels toward category and global medians, so SKUs with sparse event histories remain deployable without per-SKU tuning.

Forecasting uses **two streams**: the baseline is extrapolated, and known future promotional/SNAP calendars are injected deterministically through the fixed kernel.

## What the paper claims (and does not)

The paper is deliberate about the scope of its evidence:

- **The Variance Ratio and standard-deviation-based safety stock are treated as diagnostics, not objectives.** Both are minimized by *any* sufficiently smooth forecast (a constant forecast attains `VR → 0`), so a low VR alone is not evidence of a good forecast. They are always reported alongside wMAPE, and inventory cost is used as the arbiter.
- **The load-bearing accuracy result is *reliability*, not central tendency.** Across eleven methods, CD attains the lowest cross-sectional dispersion of per-SKU error and an order-of-magnitude lower catastrophic-failure rate; its median-accuracy advantage is real but small.
- **The cost accounting is honest and directional.** CD's smoothing lowers safety stock, holding cost, and order variance but under-provisions event spikes, so it reduces *total* inventory cost only when holding costs exceed ~20% of stockout costs. Outside that regime CD is an operational-stability and inventory-capital layer, not an expected-cost minimizer.

## Headline results

Medians over strictly out-of-sample 28-day horizons, with the corrected exogenous-channel loaders.

| Result                                                                    | Value                                                                                                   |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Cross-sectional error dispersion (SD of per-SKU wMAPE), M5                | **0.23 ± 0.02** vs 1.11–4.51 for the other ten baselines — lowest, and first in all four draws |
| Catastrophic-failure rate (wMAPE > 200%), M5                              | **0.8% ± 0.1** vs 9.9–20.6% for every baseline                                                  |
| Inventory-cost crossover (strictly out-of-sample, 1,000 SKUs)             | **h/p ≈ 0.20** (95% bootstrap CI [0.17, 0.25])                                                   |
| CD median Variance Ratio, full M5 catalog (30,490 SKUs; 29,678 evaluated) | 0.0007                                                                                                  |
| CD median Variance Ratio, Favorita (2,845 items)                          | 0.0072                                                                                                  |
| Tuned XGBoost VR on a matched 2,000-SKU M5 subset                         | 0.0135 (≈21× higher than CD)                                                                          |
| CD wins the paired VR comparison                                          | 86% of SKUs on M5, 74.5% on Favorita                                                                    |
| Safety stock vs Tuned XGBoost, matched service level                      | 0.16× on M5, 0.38× on Favorita                                                                        |
| Full-catalog runtime (30,490 SKUs, 16-core CPU)                           | ~40 minutes                                                                                             |

Robustness is further validated via calendar corruption, kernel misspecification, and timing-shift tests, and a supporting operator-theoretic (Koopman/EDMD) analysis shows the learned demand operators are non-normal while CD's compact parametric kernel matches their operational performance.

## Repository layout

```
.
├── code/
│   ├── run_paper.py                  # single-command orchestrator (see below)
│   ├── requirements.txt
│   ├── contextual_deconvolution/     # the library
│   │   ├── core.py                   # operators, convex solvers (ISTA/CVXPY), kernel estimation
│   │   ├── experiments.py            # evaluation framework, pooling, two-stream forecasting
│   │   ├── baselines.py              # XGBoost, LightGBM, Prophet-like, STL+ARIMA, Croston/SBA
│   │   ├── baselines_dl.py           # PatchTST, N-BEATS, DeepAR (PyTorch, optional)
│   │   ├── loaders.py                # M5, Favorita, and Pandora dataset loaders
│   │   ├── metrics.py                # wMAPE, MASE, RMSSE, VR, order variance, inventory cost
│   │   ├── ablation.py               # identity / non-causal / misspecified-kernel ablations
│   │   ├── falsification.py          # falsification tests on synthetic ground truth
│   │   └── synthetic.py              # synthetic generators with known kernels
│   ├── experiments/                  # 37 runnable experiment scripts
│   ├── figures/                      # 8 figure-generation scripts
│   ├── tests/                        # pytest suite (core operators, synthetic recovery)
│   └── results/                      # numerical outputs (CSV/JSON)
├── data/                             # datasets (not tracked; see Data setup)
├── followup/                         # exploratory work kept for follow-up papers
└── paper/                            # LaTeX source, figures, compiled PDF
```

`run_paper.py` auto-creates two compatibility symlinks (`code/data → ../data`, `code/paper → ../paper`) so scripts that resolve paths relative to their own location find the repository-root folders. No script is modified.

The scripts behind the paper's main tables and figures are:

| Paper artifact                                                 | Script                                                                                     |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Table 1 (scale validation), Finding 1, Table 2 (all baselines) | `experiments/regenerate_headline_tables.py`                                              |
| Reliability / dispersion table                                 | `experiments/run_reliability_dispersion.py`                                              |
| Favorita headline (Finding 5)                                  | `experiments/run_favorita_headline.py`                                                   |
| Cost crossover + inventory decomposition                       | `experiments/rerun_cost_and_inventory.py`                                                |
| Kernel ablation + promotion-density strata                     | `experiments/rerun_ablation_promodensity.py`                                             |
| Pooling-strategy comparison                                    | `experiments/run_pooling_comparison.py`                                                  |
| Cost-ratio figure / Favorita VR figure                         | `figures/generate_cost_ratio_sweep.py`, `figures/generate_favorita_vr_distribution.py` |

## Installation

Requires **Python 3.9+**.

```bash
git clone <repository-url>
cd contexual-deconvolution
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r code/requirements.txt
```

Core dependencies: `numpy`, `scipy`, `pandas`, `cvxpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `statsmodels`, `tqdm`.

Optional extras:

- `lightgbm` — enables the LightGBM baseline (falls back to gradient boosting if absent)
- `torch` — enables the deep-learning baselines in `baselines_dl.py` (PatchTST, N-BEATS, DeepAR)

Every dependency is CPU-only; no GPU is required at any point.

## Data setup

The datasets are public but redistribution-restricted, so they are **not** included. Download them and place them at these exact paths:

**M5 Forecasting — Accuracy**

```
data/m5-data/m5-forecasting-accuracy/
├── calendar.csv
├── sales_train_evaluation.csv
├── sales_train_validation.csv
└── sell_prices.csv
```

**Corporación Favorita Grocery Sales Forecasting**

```
data/favorita-grocery-sales-forecasting/
├── train.csv
├── items.csv
├── stores.csv
├── holidays_events.csv
└── oil.csv
```

Loaders read these lazily and in chunks — Favorita's `train.csv` is never loaded into memory in full. Synthetic experiments and the unit tests require no downloads.

## Reproducing the paper

A single command regenerates every table and figure:

```bash
cd code
python run_paper.py
```

Stages run in order and can be invoked individually:

| Command                             | Action                         |
| ----------------------------------- | ------------------------------ |
| `python run_paper.py experiments` | fit models, write`results/*` |
| `python run_paper.py figures`     | render`paper/figures/*.png`  |
| `python run_paper.py reproduce`   | run the pytest suite           |
| `python run_paper.py all`         | all three (default)            |

Useful flags:

```bash
python run_paper.py all --list            # dry run: print the ordered plan, execute nothing
python run_paper.py all --stop-on-error   # halt at the first failing script
```

Scripts are discovered automatically and sorted, so the plan is deterministic. Progress and per-script timings are printed, and a summary table is emitted at the end. By default a failing script is reported but does not abort the run.

### Running individual experiments

Scripts in `code/experiments/` are standalone and expose `argparse` interfaces. Representative examples:

```bash
cd code

# M5 headline: Table 1 (scale), Finding 1, Table 2 (all baselines)
python experiments/regenerate_headline_tables.py

# Reliability / dispersion across four independent 2,000-SKU draws
python experiments/run_reliability_dispersion.py

# Favorita headline (CD vs Tuned XGBoost, live promotion + holiday channels)
python experiments/run_favorita_headline.py

# Inventory-cost crossover (h/p sweep) + inventory decomposition
python experiments/rerun_cost_and_inventory.py
python figures/generate_cost_ratio_sweep.py

# Data-driven kernel bandwidth analysis (justifies n = 14)
python experiments/run_bandwidth_analysis.py
```

Pass `--help` to any script for its full option list.

## Evaluation protocol

The reproduction pipeline enforces the protocol described in the paper. These points are load-bearing for the reported numbers:

**Strictly out-of-sample.** The final 28 days of every series are held out. The convex decomposition is fit on the training window *alone*, and every baseline produces a genuine multi-step forecast — gradient-boosted and distributed-lag models forecast **recursively**, feeding their own predictions back as lags. No method, CD included, observes any test-window demand. On M5 the held-out window is the `d_1914`–`d_1941` validation phase, sourced from the evaluation file while the training phase is sourced from the validation file, so the split cannot silently leak.

**Information parity.** The only future information used at forecast time is the exogenous promotional/SNAP calendar, which is published in advance. Every feature-based baseline — Tuned and Vanilla XGBoost, LightGBM, Prophet-like, distributed-lag ridge, ARIMAX, ETS — receives the *identical* future calendar that CD injects into its structural stream. The deep sequence baselines (PatchTST, N-BEATS, DeepAR) are autoregressive by construction and consume only past demand; they are therefore reported as **accuracy baselines only**, and no operational-variance conclusions are drawn from them.

**Live exogenous channels.** The loaders assert that promotional and calendar channels are non-degenerate before evaluation (`_assert_channels_live`), guarding against the type-mismatch bugs that can silently zero an exogenous channel and inflate a method's apparent variance advantage.

**Determinism.** All splits, subsamples, and model seeds are fixed (`random_state=42`). Re-running yields identical numbers.

**Statistics.** Metrics are reported as medians across SKUs unless stated otherwise; the mean is inflated by a tail of high-variance SKUs. Paired comparisons use Wilcoxon signed-rank tests with a matched-pairs rank-biserial effect size reported alongside, since at catalog scale the p-value certifies direction rather than magnitude.

## Metrics

Implemented in `contextual_deconvolution/metrics.py`:

| Metric                        | Definition                                                            |
| ----------------------------- | --------------------------------------------------------------------- |
| **wMAPE**               | weighted MAPE;`1.0` = 100% error                                    |
| **MASE**                | scaled vs. seasonal naive;`< 1` beats naive                         |
| **RMSSE**               | M5-style root relative squared error                                  |
| **Variance Ratio (VR)** | `Var(forecast) / Var(actual)` — Bullwhip inflation                 |
| **Order Variance**      | `Var(orders)/Var(actual) − 1` under a base-stock policy            |
| **Safety Stock**        | service-level buffer from forecast-error dispersion                   |
| **Inventory cost**      | holding + stockout under a simulated base-stock policy with lead time |

A caution carried through the paper: **VR is degenerate at the limit** — a constant forecast attains `VR → 0` while being useless. It is therefore always reported alongside wMAPE, near-zero-variance items are excluded, and inventory cost is the arbiter.

## Testing

```bash
cd code
python -m pytest tests/ -q
```

Covers the core operators and solvers (`test_core.py`) and synthetic recovery with known ground-truth kernels. The suite also runs as the `reproduce` stage of `run_paper.py`.

## Runtime and hardware

CPU-only throughout; no GPU, no cluster, no per-SKU neural training.

| Workload                                       | Approximate time  |
| ---------------------------------------------- | ----------------- |
| Full M5 catalog (30,490 SKUs), pooled kernel   | ~40 min, 16 cores |
| Per-SKU decomposition with pre-computed kernel | ~0.13 s           |
| Favorita (2,845 items)                         | ~10 min           |
| Unit tests                                     | < 1 min           |
