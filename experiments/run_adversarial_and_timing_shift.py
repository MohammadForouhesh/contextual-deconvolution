import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import numpy as np
import pandas as pd
import sys, os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, _os.path.join(_ROOT, "code"))
from contextual_deconvolution.loaders import load_m5_data
from contextual_deconvolution.experiments import evaluate_sku, CONFIG_DEFAULT
from contextual_deconvolution.core import build_Aspec, decompose_ista, estimate_kernel, fit_parametric_kernel, kernel_from_theta

sns.set_style("white")
sns.set_context("paper", font_scale=1.2)
FIG_DIR = _os.path.join(_ROOT, "paper/figures")
os.makedirs(FIG_DIR, exist_ok=True)

DATA_DIR = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")


def corrupt_features(f_promo, f_snap, noise_level, seed):
    """Randomly flip binary feature values with probability noise_level."""
    rng = np.random.RandomState(seed)
    f_promo_c = f_promo.copy()
    f_snap_c = f_snap.copy()
    
    promo_mask = f_promo_c > 0
    flip_promo = rng.random(len(f_promo_c)) < noise_level
    f_promo_c[flip_promo & promo_mask] = 0
    f_promo_c[flip_promo & ~promo_mask] = 1
    
    snap_mask = f_snap_c > 0
    flip_snap = rng.random(len(f_snap_c)) < noise_level
    f_snap_c[flip_snap & snap_mask] = 0
    f_snap_c[flip_snap & ~snap_mask] = 1
    
    return f_promo_c, f_snap_c


def run_adversarial_with_ci(n_skus=50, noise_levels=[0.0, 0.30, 0.50, 0.70], seed=42):
    """Run adversarial test with 50 SKUs and save per-SKU results for CI computation."""
    print(f"Loading M5 data (random {n_skus} SKUs)...")
    dataset = load_m5_data(DATA_DIR, n_skus=n_skus, random_state=seed)
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
        
        for noise_level in noise_levels:
            sku_copy = dict(sku)
            if noise_level > 0:
                f_promo_c, f_snap_c = corrupt_features(
                    sku['f_promo'], sku['f_snap'], noise_level, seed=seed + i
                )
                sku_copy['f_promo'] = f_promo_c
                sku_copy['f_snap'] = f_snap_c
            
            start = time.time()
            r = evaluate_sku(sku_copy, config)
            elapsed = time.time() - start
            
            fc = r['forecasts']['CD']
            # Also compute sparsity and baseline smoothness
            x_base = r['x_base']
            x_spec = r['x_spec']
            sparsity = np.sum(np.abs(x_spec) > 1e-3)
            baseline_smoothness = np.sum(np.diff(x_base, 2) ** 2)
            
            results.append({
                'sku_id': sku_id,
                'noise_level': noise_level,
                'wmape': fc['wmape'],
                'variance_ratio': fc['variance_ratio'],
                'rmse': fc['rmse'],
                'safety_stock': fc['safety_stock'],
                'inventory_cost': fc['inventory_cost'],
                'sparsity': sparsity,
                'baseline_smoothness': baseline_smoothness,
                'runtime': elapsed,
            })
            print(f"  Noise {noise_level:.0%}: wMAPE={fc['wmape']:.4f}, VR={fc['variance_ratio']:.4f}, sparsity={sparsity}")
    
    df = pd.DataFrame(results)
    out_path = _os.path.join(_ROOT, "paper/results/calendar_noise_robustness_50sku.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return df


def generate_figure7_with_ci(df):
    """Generate Figure 7 with 95% confidence intervals."""
    metrics = ['wmape', 'variance_ratio', 'sparsity', 'baseline_smoothness']
    labels = ['wMAPE', 'Variance Ratio', 'Sparsity ($\\|x_{\\text{spec}}\\|_0$)', 'Baseline Smoothness']
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    noise_labels = {0.0: '0% (Clean)', 0.30: '30%', 0.50: '50%', 0.70: '70%'}
    
    for ax, metric, label in zip(axes, metrics, labels):
        # Compute mean and 95% CI per noise level
        stats = []
        for nl in sorted(df['noise_level'].unique()):
            subset = df[df['noise_level'] == nl][metric]
            mean = subset.mean()
            std = subset.std()
            n = len(subset)
            ci = 1.96 * std / np.sqrt(n) if n > 0 else 0
            stats.append({'noise_level': nl, 'mean': mean, 'ci': ci, 'n': n})
        
        stats_df = pd.DataFrame(stats)
        x = np.arange(len(stats_df))
        
        ax.plot(x, stats_df['mean'], 'o-', color='#1F77B4', lw=2, markersize=8, label='Mean')
        ax.fill_between(x, stats_df['mean'] - stats_df['ci'], stats_df['mean'] + stats_df['ci'],
                        alpha=0.3, color='#1F77B4', label='95% CI')
        
        ax.set_xticks(x)
        ax.set_xticklabels([noise_labels.get(nl, f'{nl:.0%}') for nl in stats_df['noise_level']], rotation=15, ha='right')
        ax.set_ylabel(label, fontsize=11)
        ax.grid(axis='y', ls=':', alpha=0.4)
        sns.despine(ax=ax, top=True, right=True)
        
        # Add value labels
        for i, (m, c) in enumerate(zip(stats_df['mean'], stats_df['ci'])):
            ax.text(i, m + c + (stats_df['mean'].max() - stats_df['mean'].min()) * 0.05,
                    f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    axes[0].legend(loc='upper right', frameon=False, fontsize=9)
    fig.suptitle('Real-Data Adversarial Robustness (50 M5 SKUs) with 95% Confidence Intervals', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, 'figure_adversarial_realdata_with_ci.png')
    fig.savefig(out_path, dpi=400, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_timing_shift_detection(n_skus=50, shifts=[0, 1, 2, 3], seed=42):
    """Run timing shift evaluation and track event detection at shifted locations."""
    print(f"\n{'='*60}")
    print("TIMING SHIFT DETECTION")
    print(f"{'='*60}")
    print(f"Loading M5 data (random {n_skus} SKUs)...")
    dataset = load_m5_data(DATA_DIR, n_skus=n_skus, random_state=seed)
    print(f"Loaded {len(dataset)} SKUs")
    
    config = CONFIG_DEFAULT.copy()
    config["use_ista"] = True
    config["run_baselines"] = False
    config["run_ablations"] = False
    
    results = []
    
    for i, sku in enumerate(dataset):
        sku_id = sku['sku_id']
        f_promo_orig = sku['f_promo'].copy()
        promo_starts = np.where(np.diff(np.concatenate([[0], f_promo_orig, [0]])) == 1)[0]
        
        if len(promo_starts) == 0:
            continue
        
        for shift in shifts:
            sku_copy = dict(sku)
            if shift > 0:
                f_promo_shifted = np.zeros_like(f_promo_orig)
                for start in promo_starts:
                    new_start = min(start + shift, len(f_promo_orig) - 1)
                    f_promo_shifted[new_start] = 1
                sku_copy['f_promo'] = f_promo_shifted
            else:
                sku_copy['f_promo'] = f_promo_orig
            
            r = evaluate_sku(sku_copy, config)
            x_spec = r['x_spec']
            
            # Check if events are detected at original vs shifted locations
            sparsity = np.sum(np.abs(x_spec) > 1e-3)
            
            # Detected shocks at original promo locations
            detected_at_original = sum(1 for s in promo_starts if s < len(x_spec) and np.abs(x_spec[s]) > 1e-3)
            
            # Detected shocks at shifted locations
            shifted_starts = [min(s + shift, len(x_spec) - 1) for s in promo_starts]
            detected_at_shifted = sum(1 for s in shifted_starts if s < len(x_spec) and np.abs(x_spec[s]) > 1e-3)
            
            fc = r['forecasts']['CD']
            results.append({
                'sku_id': sku_id,
                'shift': shift,
                'n_promos': len(promo_starts),
                'wmape': fc['wmape'],
                'variance_ratio': fc['variance_ratio'],
                'sparsity': sparsity,
                'detected_at_original': detected_at_original,
                'detected_at_shifted': detected_at_shifted,
                'detection_rate_original': detected_at_original / len(promo_starts) if len(promo_starts) > 0 else 0,
                'detection_rate_shifted': detected_at_shifted / len(promo_starts) if len(promo_starts) > 0 else 0,
            })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(dataset)}] processed")
    
    df = pd.DataFrame(results)
    out_path = _os.path.join(_ROOT, "paper/results/timing_shift_detection.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TIMING SHIFT DETECTION SUMMARY")
    print("=" * 60)
    for shift in sorted(df['shift'].unique()):
        subset = df[df['shift'] == shift]
        print(f"\nShift ±{shift} days (n={len(subset)} SKUs):")
        print(f"  wMAPE: {subset['wmape'].mean():.4f} ± {subset['wmape'].std():.4f}")
        print(f"  VR: {subset['variance_ratio'].mean():.4f} ± {subset['variance_ratio'].std():.4f}")
        print(f"  Sparsity: {subset['sparsity'].mean():.1f} ± {subset['sparsity'].std():.1f}")
        print(f"  Detection at original: {subset['detection_rate_original'].mean():.2%}")
        print(f"  Detection at shifted:  {subset['detection_rate_shifted'].mean():.2%}")
    
    return df


if __name__ == '__main__':
    # Task 1: Re-run adversarial test with 50 SKUs and generate CI figure
    print("=" * 60)
    print("TASK 1: Adversarial Test with 50 SKUs + CI Figure")
    print("=" * 60)
    df_adv = run_adversarial_with_ci(n_skus=50, noise_levels=[0.0, 0.30, 0.50, 0.70])
    generate_figure7_with_ci(df_adv)
    
    # Task 2: Timing shift detection
    print("\n" + "=" * 60)
    print("TASK 2: Timing Shift Detection")
    print("=" * 60)
    df_shift = run_timing_shift_detection(n_skus=50, shifts=[0, 1, 2, 3])
    
    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETE")
    print("=" * 60)
