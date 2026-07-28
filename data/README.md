# Datasets

The datasets used in this paper are public but **redistribution-restricted**, so they are not
included in this repository. Download them from Kaggle and unpack them to the exact paths
below — the loaders in `code/contextual_deconvolution/loaders.py` resolve these paths relative
to the repository root, so no configuration is needed if the layout matches.

Nothing here needs to be downloaded to run the unit tests or the synthetic experiments.

---

## M5 Forecasting — Accuracy

Walmart daily SKU-level sales, 2011–2016. Used for the main evaluation (30,490 SKUs), the
comprehensive baseline comparison, and the reliability/dispersion analysis.

**Source:** [https://www.kaggle.com/competitions/m5-forecasting-accuracy/data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

```
data/m5-data/m5-forecasting-accuracy/
├── calendar.csv                 date ↔ event/SNAP calendar
├── sales_train_evaluation.csv   daily unit sales per SKU
├── sales_train_validation.csv
└── sell_prices.csv              weekly prices (optional; loaders accept skip_prices=True)
```

## Corporación Favorita Grocery Sales Forecasting

Ecuadorian grocery chain daily sales. Used for the cross-dataset generalization study
(2,863 items).

**Source:** [https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

```
data/favorita-grocery-sales-forecasting/
├── train.csv              daily unit sales (large; read in chunks, never loaded whole)
├── items.csv              item metadata (family used as the pooling category)
├── stores.csv
├── holidays_events.csv    the event calendar CD deconvolves
└── oil.csv
```

> Kaggle ships several of these as `.7z`. Decompress them so the `.csv` files sit directly in
> the directory shown above.

---

## Disk and runtime

| dataset  | on disk | full-catalog runtime                  |
| -------- | ------- | ------------------------------------- |
| M5       | ~0.9 GB | ~40 min for 30,490 SKUs (16-core CPU) |
| Favorita | ~5 GB   | ~10 min for 2,863 items               |

Everything is CPU-only; no GPU is required at any point.

## Verifying the layout

```bash
python3 -c "
from pathlib import Path
ok = True
for p in ['data/m5-data/m5-forecasting-accuracy/calendar.csv',
          'data/m5-data/m5-forecasting-accuracy/sales_train_evaluation.csv',
          'data/favorita-grocery-sales-forecasting/train.csv',
          'data/favorita-grocery-sales-forecasting/holidays_events.csv']:
    e = Path(p).exists(); ok &= e
    print(('  OK  ' if e else '  MISSING '), p)
print('layout OK' if ok else 'fix the paths above before running experiments')
"
```
