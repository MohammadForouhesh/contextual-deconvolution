import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import numpy as np
import pandas as pd
import time
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import evaluate_sku, CONFIG_DEFAULT

DATA_DIR = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")
OUT_PATH = _os.path.join(_ROOT, "paper/results/calendar_noise_robustness.csv")


def corrupt_features(f_promo, f_snap, noise_level, seed):
    """Randomly flip binary feature values with probability noise_level."""
    rng = np.random.RandomState(seed)
    f_promo_c = f_promo.copy()
    f_snap_c = f_snap.copy()
    
    # Flip promo days
    promo_mask = f_promo_c > 0
    flip_promo = rng.random(len(f_promo_c)) < noise_level
    f_promo_c[flip_promo & promo_mask] = 0
    f_promo_c[flip_promo & ~promo_mask] = 1
    
    # Flip SNAP days
    snap_mask = f_snap_c > 0
    flip_snap = rng.random(len(f_snap_c)) < noise_level
    f_snap_c[flip_snap & snap_mask] = 0
    f_snap_c[flip_snap & ~snap_mask] = 1
    
    return f_promo_c, f_snap_c


def main():
    print("Loading M5 data (random 50 SKUs)...")
    dataset = load_m5_data(DATA_DIR, n_skus=25, random_state=42)
    print(f"Loaded {len(dataset)} SKUs")
    
    config = CONFIG_DEFAULT.copy()
    config["use_ista"] = True
    config["run_baselines"] = False
    config["run_ablations"] = False
    config["dl_baseline_methods"] = []
    
    results = []
    
    for i, sku in enumerate(dataset):
        sku_id = sku['sku_id']
        print(f"[{i+1}/{len(dataset)}] SKU {sku_id}...")
        
        for noise_level in [0.0, 0.10, 0.25, 0.50]:
            # Create a copy of the SKU data with corrupted features
            sku_copy = dict(sku)
            if noise_level > 0:
                f_promo_c, f_snap_c = corrupt_features(
                    sku['f_promo'], sku['f_snap'], noise_level, seed=42
                )
                sku_copy['f_promo'] = f_promo_c
                sku_copy['f_snap'] = f_snap_c
            
            start = time.time()
            r = evaluate_sku(sku_copy, config)
            elapsed = time.time() - start
            
            # Extract metrics from CD forecast
            fc = r['forecasts']['CD']
            results.append({
                'sku_id': sku_id,
                'noise_level': noise_level,
                'wmape': fc['wmape'],
                'variance_ratio': fc['variance_ratio'],
                'rmse': fc['rmse'],
                'safety_stock': fc['safety_stock'],
                'inventory_cost': fc['inventory_cost'],
                'runtime': elapsed,
            })
            print(f"  Noise {noise_level:.0%}: wMAPE={fc['wmape']:.4f}, VR={fc['variance_ratio']:.4f}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False)
    
    print("\n" + "="*60)
    print("CALENDAR NOISE ROBUSTNESS SUMMARY")
    print("="*60)
    for noise_level in [0.0, 0.10, 0.25, 0.50]:
        subset = df[df['noise_level'] == noise_level]
        print(f"\nNoise {noise_level:.0%} (n={len(subset)}):")
        for col in ['wmape', 'variance_ratio', 'safety_stock']:
            print(f"  {col:20s}: {subset[col].mean():.4f} ± {subset[col].std():.4f}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == '__main__':
    main()
