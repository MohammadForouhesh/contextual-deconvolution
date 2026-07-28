import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import numpy as np
import pandas as pd
from scipy import stats
import sys
sys.path.insert(0, _os.path.join(_ROOT, "code/contextual_deconvolution"))

from experiments import evaluate_sku
from core import build_Aspec
from loaders import load_m5_data
import time

def corrupt_calendar(calendar_df, snap_col, event_cols, noise_level, seed):
    """Randomly flip event days with probability noise_level."""
    rng = np.random.RandomState(seed)
    corrupted = calendar_df.copy()
    
    # Flip SNAP days
    snap_mask = corrupted[snap_col].values.astype(bool)
    flip_snap = rng.random(len(snap_mask)) < noise_level
    corrupted.loc[flip_snap, snap_col] = 1 - corrupted.loc[flip_snap, snap_col]
    
    # Flip event columns
    for col in event_cols:
        if col in corrupted.columns:
            flip = rng.random(len(corrupted)) < noise_level
            corrupted.loc[flip, col] = 1 - corrupted.loc[flip, col]
    
    return corrupted


def run_calendar_noise_test(data_dir, n_skus=50, noise_levels=[0.0, 0.10, 0.25, 0.50], seed=42):
    """Run calendar noise robustness test."""
    print(f"Loading M5 data (random {n_skus} SKUs)...")
    data = load_m5_data(data_dir, n_skus=n_skus, random_state=seed)
    
    # Get event columns
    calendar = data['calendar']
    event_cols = [c for c in calendar.columns if c not in 
                  ['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'd', 
                   'snap_CA', 'snap_TX', 'snap_WI']]
    
    # Determine which SNAP column to use (take first available)
    snap_cols = [c for c in calendar.columns if 'snap_' in c]
    snap_col = snap_cols[0] if snap_cols else None
    
    print(f"Event columns: {event_cols}")
    print(f"SNAP column: {snap_col}")
    
    results = []
    
    for sku in data['skus']:
        sku_id = sku['id']
        print(f"\nProcessing SKU {sku_id}...")
        
        for noise_level in noise_levels:
            if noise_level == 0.0:
                # Use clean calendar
                test_data = data
            else:
                # Corrupt calendar
                corrupted_calendar = corrupt_calendar(
                    data['calendar'].copy(), snap_col, event_cols, noise_level, seed=seed
                )
                test_data = {k: v for k, v in data.items()}
                test_data['calendar'] = corrupted_calendar
            
            # Run evaluation
            start = time.time()
            metrics = evaluate_sku(sku, test_data)
            elapsed = time.time() - start
            
            results.append({
                'sku_id': sku_id,
                'noise_level': noise_level,
                'wmape': metrics['wmape'],
                'variance_ratio': metrics['variance_ratio'],
                'rmse': metrics['rmse'],
                'safety_stock': metrics['safety_stock'],
                'inventory_cost': metrics['inventory_cost'],
                'runtime': elapsed,
            })
            print(f"  Noise {noise_level:.0%}: wMAPE={metrics['wmape']:.4f}, VR={metrics['variance_ratio']:.4f}")
    
    df = pd.DataFrame(results)
    df.to_csv(_os.path.join(_ROOT, "paper/results/calendar_noise_robustness.csv"), index=False)
    
    # Print aggregate summary
    print("\n" + "="*60)
    print("CALENDAR NOISE ROBUSTNESS SUMMARY")
    print("="*60)
    for noise_level in noise_levels:
        subset = df[df['noise_level'] == noise_level]
        print(f"\nNoise {noise_level:.0%} (n={len(subset)}):")
        for col in ['wmape', 'variance_ratio', 'safety_stock']:
            print(f"  {col:20s}: {subset[col].mean():.4f} ± {subset[col].std():.4f}")
    
    return df


if __name__ == '__main__':
    data_dir = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")
    df = run_calendar_noise_test(data_dir, n_skus=50, noise_levels=[0.0, 0.10, 0.25, 0.50])
    print("\nDone. Results saved to calendar_noise_robustness.csv")
