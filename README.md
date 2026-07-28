# Contextual Deconvolution for Variance-Stable Demand Sensing

Reference implementation and full reproduction pipeline for the paper:

> **Contextual Deconvolution for Variance-Stable Demand Sensing: Kernel-Modulated Operators in Promotional Retail**
> Mohammad Forouhesh, Amirkabir University of Technology

Contextual Deconvolution (CD) reframes SKU-level demand sensing as a **convex inverse problem**: observed demand is decomposed into a smooth structural baseline plus a sparse stream of promotion-driven shocks propagated through a kernel-modulated banded operator. The objective is *operational* stability — Bullwhip variance, safety stock, and inventory cost — rather than point accuracy alone.

Everything runs **CPU-only**, is **seeded and deterministic**, and reproduces from a single command.

---

## Table of contents

- [Method in brief](#method-in-brief)
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
b  =  A_base · x_base  +  A_spec(θ) · x_spec  +  ε
```

and recovered by solving the convex program

```
min   ‖b − (x_base + A_spec·x_spec)‖²  +  λ_smooth‖L·x_base‖²  +  λ_sparse‖x_spec‖₁
```

- **`x_base`** — smooth structural baseline (capacity planning signal)
- **`x_spec`** — sparse transient shocks (event staging signal)
- **`A_spec(θ)`** — lower-triangular banded operator whose sub-diagonals are modulated by the exogenous event calendar, built from a causal kernel `φ(k)` (impulse, exponential decay, or signed surge-and-vacuum Difference-of-Exponentials)

Estimation is a **two-stage pipeline** — robust Huber detrending, then distributed-lag ridge regression for the kernel — followed by the convex decomposition with the kernel held fixed. **Hierarchical partial pooling** shrinks per-SKU kernels toward category and global medians, so SKUs with sparse event histories remain deployable without per-SKU tuning.

Forecasting uses **two streams**: the baseline is extrapolated, and known future promotional/SNAP calendars are injected deterministically through the fixed kernel.

## Headline results

All figures are medians over strictly out-of-sample 28-day horizons.

| Result                                           | Value                                                 |
| ------------------------------------------------ | ----------------------------------------------------- |
| CD Variance Ratio, full M5 catalog (30,490 SKUs) | **0.0024**                                      |
| CD Variance Ratio, Favorita (2,863 items)        | **0.0052**                                      |
| Tuned XGBoost VR, matched M5 subset              | 0.013 (≈5.8× higher)                                |
| CD wins paired VR vs Tuned XGBoost               | 73.8% of SKUs                                         |
| Inventory-cost crossover                         | **h/p ≈ 0.18** (95% bootstrap CI [0.13, 0.23]) |
| Full-catalog runtime (30k SKUs, 16-core CPU)     | ~40 minutes                                           |

**Interpretation matters here.** The Variance Ratio and standard-deviation-based safety stock are both minimized by *any* sufficiently smooth forecast, so the paper treats them as **diagnostics**, not objectives. The load-bearing claim is the inventory-cost accounting: CD's smoothing lowers safety stock, holding cost, and order variance but under-provisions event spikes, so it reduces *total* cost only when holding costs exceed ~18% of stockout costs. Outside that regime CD is an operational-stability and inventory-capital layer, not an expected-cost minimizer.

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
│   │   ├── loaders.py                # M5 and Favorita dataset loaders
│   │   ├── metrics.py                # wMAPE, MASE, RMSSE, VR, order variance, inventory cost
│   │   ├── ablation.py               # identity / non-causal / misspecified-kernel ablations
│   │   ├── falsification.py          # falsification tests on synthetic ground truth
│   │   └── synthetic.py              # synthetic generators with known kernels
│   ├── experiments/                  # 36 runnable experiment scripts
│   ├── figures/                      # 19 figure-generation scripts
│   ├── tests/                        # pytest suite (core operators, synthetic recovery)
│   └── results/                      # numerical outputs (CSV/JSON)
├── data/                             # datasets (not tracked; see Data setup)
└── paper/                            # LaTeX source, figures, compiled PDF
```

`run_paper.py` auto-creates two compatibility symlinks (`code/data → ../data`, `code/paper → ../paper`) so scripts that resolve paths relative to their own location continue to find the repository-root folders. No script is modified.

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

The datasets are public but redistribution-restricted, so they are **not** included. Download them from Kaggle and place them at these exact paths:

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

# Full-catalog M5 evaluation with category-pooled kernels
python experiments/run_full_m5_pooled.py

# Favorita with a global pooled kernel (default cap ~3000 items)
python experiments/run_favorita_pooled.py --pooling global --n_items 3000

# Head-to-head against all baselines on an M5 subset
python experiments/run_comprehensive_baselines.py

# Inventory simulation and the holding/stockout cost sweep
python experiments/run_inventory_simulation.py
python figures/generate_cost_ratio_sweep.py

# Data-driven kernel bandwidth analysis (justifies n = 14)
python experiments/run_bandwidth_analysis.py
```

Pass `--help` to any script for its full option list.

## Evaluation protocol

The reproduction pipeline enforces the protocol described in the paper. These points are load-bearing for the reported numbers:

**Strictly out-of-sample.** The final 28 days of every series are held out. The convex decomposition is fit on the training window *alone*, and every baseline produces a genuine multi-step forecast — gradient-boosted and distributed-lag models forecast **recursively**, feeding their own predictions back as lags. No method, CD included, observes any test-window demand.

**Information parity.** The only future information used at forecast time is the exogenous promotional/SNAP calendar, which is published in advance. Every feature-based baseline — Tuned and Vanilla XGBoost, LightGBM, Prophet-like, distributed-lag ridge, ARIMAX — receives the *identical* future calendar that CD injects into its structural stream. The deep sequence baselines (PatchTST, N-BEATS, DeepAR) are autoregressive by construction and consume only past demand; they are therefore reported as **accuracy baselines only**, and no operational-variance conclusions are drawn from them.

**Determinism.** All splits, subsamples, and model seeds are fixed (`random_state=42`). Re-running yields identical numbers.

**Statistics.** Metrics are reported as medians across SKUs unless stated otherwise; the mean is inflated by a tail of high-variance SKUs. Paired comparisons use Wilcoxon signed-rank tests with Bonferroni correction.

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

A caution carried through the paper: **VR is degenerate at the limit** — a constant forecast attains `VR → 0` while being useless. It is therefore always reported alongside wMAPE, near-zero-variance items are excluded, and cost is used as the arbiter.

## Testing

```bash
cd code
python -m pytest tests/ -q
```

Covers the core operators and solvers (`test_core.py`) and synthetic recovery with known ground-truth kernels (`test_synthetic.py`). The suite also runs as the `reproduce` stage of `run_paper.py`.

## Runtime and hardware

CPU-only throughout; no GPU, no cluster, no per-SKU neural training.

| Workload                                       | Approximate time  |
| ---------------------------------------------- | ----------------- |
| Full M5 catalog (30,490 SKUs), pooled kernel   | ~40 min, 16 cores |
| Per-SKU decomposition with pre-computed kernel | ~0.13 s           |
| Favorita (2,863 items)                         | ~10 min           |
| Unit tests                                     | < 1 min           |

Catalog-scale runs parallelize across SKUs with `joblib`. The dominant cost in head-to-head comparisons is training the per-SKU baselines, not CD itself.
