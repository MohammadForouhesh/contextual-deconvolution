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


def run_both_methods_adversarial(n_skus=50, noise_levels=[0.0, 0.30, 0.50, 0.70], seed=42):
    """Run adversarial test for both CD and XGBoost on same SKUs with same corruption."""
    print(f"Loading M5 data (random {n_skus} SKUs)...")
    dataset = load_m5_data(DATA_DIR, n_skus=n_skus, random_state=seed)
    print(f"Loaded {len(dataset)} SKUs")
    
    config = CONFIG_DEFAULT.copy()
    config["use_ista"] = True
    config["run_baselines"] = True  # Enable XGBoost
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
            
            # Extract metrics for both CD and XGBoost
            for method_name in ['CD', 'XGBoost']:
                if method_name == 'CD':
                    key = 'CD'
                else:
                    key = 'XGBoost'
                
                if key not in r['forecasts']:
                    continue
                
                fc = r['forecasts'][key]
                
                # Compute sparsity for CD only
                if method_name == 'CD':
                    x_base = r['x_base']
                    x_spec = r['x_spec']
                    sparsity = np.sum(np.abs(x_spec) > 1e-3)
                    baseline_smoothness = np.sum(np.diff(x_base, 2) ** 2)
                else:
                    sparsity = np.nan
                    baseline_smoothness = np.nan
                
                results.append({
                    'sku_id': sku_id,
                    'noise_level': noise_level,
                    'method': method_name,
                    'wmape': fc['wmape'],
                    'variance_ratio': fc['variance_ratio'],
                    'rmse': fc['rmse'],
                    'safety_stock': fc['safety_stock'],
                    'inventory_cost': fc['inventory_cost'],
                    'sparsity': sparsity,
                    'baseline_smoothness': baseline_smoothness,
                    'runtime': elapsed,
                })
            
            cd_fc = r['forecasts'].get('CD', {})
            xgb_fc = r['forecasts'].get('XGBoost', {})
            print(f"  Noise {noise_level:.0%}: CD(wMAPE={cd_fc.get('wmape', np.nan):.4f}, VR={cd_fc.get('variance_ratio', np.nan):.4f}) "
                  f"XGB(wMAPE={xgb_fc.get('wmape', np.nan):.4f}, VR={xgb_fc.get('variance_ratio', np.nan):.4f})")
    
    df = pd.DataFrame(results)
    out_path = _os.path.join(_ROOT, "paper/results/adversarial_cd_vs_xgb.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return df


def generate_comparison_figure(df):
    """Generate four-panel figure comparing CD vs XGBoost under adversarial corruption."""
    metrics = ['wmape', 'variance_ratio', 'sparsity', 'baseline_smoothness']
    labels = ['wMAPE', 'Variance Ratio', 'Sparsity ($\\|x_{\\text{spec}}\\|_0$)', 'Baseline Smoothness']
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    noise_labels = {0.0: '0% (Clean)', 0.30: '30%', 0.50: '50%', 0.70: '70%'}
    colors = {'CD': '#1F77B4', 'XGBoost': '#D62728'}
    
    for ax, metric, label in zip(axes, metrics, labels):
        for method in ['CD', 'XGBoost']:
            method_df = df[df['method'] == method]
            stats = []
            for nl in sorted(method_df['noise_level'].unique()):
                subset = method_df[method_df['noise_level'] == nl][metric].dropna()
                if len(subset) == 0:
                    continue
                mean = subset.mean()
                std = subset.std()
                n = len(subset)
                ci = 1.96 * std / np.sqrt(n) if n > 0 else 0
                stats.append({'noise_level': nl, 'mean': mean, 'ci': ci, 'n': n})
            
            if not stats:
                continue
            
            stats_df = pd.DataFrame(stats)
            x = np.arange(len(stats_df))
            
            ax.plot(x, stats_df['mean'], 'o-', color=colors[method], lw=2, markersize=7, label=method)
            ax.fill_between(x, stats_df['mean'] - stats_df['ci'], stats_df['mean'] + stats_df['ci'],
                            alpha=0.2, color=colors[method])
            
            # Add value labels for CD
            if method == 'CD':
                for i, m in enumerate(stats_df['mean']):
                    ax.text(i, m + stats_df['ci'].iloc[i] + (stats_df['mean'].max() - stats_df['mean'].min()) * 0.05,
                            f'{m:.3f}', ha='center', fontsize=8, fontweight='bold', color=colors[method])
        
        ax.set_xticks(np.arange(len(noise_labels)))
        ax.set_xticklabels([noise_labels.get(nl, f'{nl:.0%}') for nl in sorted(df['noise_level'].unique())],
                          rotation=15, ha='right')
        ax.set_ylabel(label, fontsize=11)
        ax.grid(axis='y', ls=':', alpha=0.4)
        sns.despine(ax=ax, top=True, right=True)
        ax.legend(loc='best', frameon=False, fontsize=9)
    
    fig.suptitle('Adversarial Robustness: CD vs. XGBoost (50 M5 SKUs, 95% CI)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, 'figure_adversarial_cd_vs_xgb.png')
    fig.savefig(out_path, dpi=400, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("ADVERSARIAL TEST: CD vs. XGBoost")
    print("=" * 60)
    df = run_both_methods_adversarial(n_skus=50, noise_levels=[0.0, 0.30, 0.50, 0.70])
    generate_comparison_figure(df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method in ['CD', 'XGBoost']:
        print(f"\n{method}:")
        for nl in [0.0, 0.30, 0.50, 0.70]:
            subset = df[(df['method'] == method) & (df['noise_level'] == nl)]
            print(f"  Noise {nl:.0%}: wMAPE={subset['wmape'].mean():.4f}±{subset['wmape'].std():.4f}, "
                  f"VR={subset['variance_ratio'].mean():.4f}±{subset['variance_ratio'].std():.4f}")
