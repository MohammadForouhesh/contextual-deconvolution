"""
Contextual Deconvolution for Demand Sensing
============================================

Reproducible code for the paper "Contextual Deconvolution: Resolving the
RMSE Trap in Enterprise Demand Sensing via Kernel-Modulated Operators".

The package reframes demand forecasting as a convex inverse problem,
decomposing observed demand into:
- a smooth structural baseline (trend + seasonality)
- a sparse transient signal (promotions, SNAP, stockouts)

Key modules:
- core.py          : operators, convex decomposition, kernel estimation
- synthetic.py     : synthetic data generators with known ground truth
- loaders.py       : M5 and Favorita dataset loaders
- baselines.py     : XGBoost, Prophet-like, STL+ARIMA baselines
- metrics.py       : wMAPE, variance ratio, inventory cost simulation
- experiments.py   : evaluation framework and reporting
- ablation.py      : ablation studies (identity, non-causal, misspecified)
- falsification.py : falsification tests for key claims

Scripts:
- scripts/run_experiments.py   : full synthetic evaluation pipeline
- scripts/run_m5.py           : evaluation on M5 real data
- scripts/run_favorita.py     : evaluation on Favorita real data
- scripts/make_figures.py     : regenerate all paper figures

Tests:
- tests/test_core.py          : unit tests for core operators and solvers
- tests/test_synthetic.py     : unit tests for synthetic data generation
"""

__version__ = "0.1.0"
