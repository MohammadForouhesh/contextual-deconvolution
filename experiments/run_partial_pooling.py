import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, "code")):
    _ROOT = _os.path.dirname(_ROOT)   # repo root = first ancestor containing code/
import os
import sys
import numpy as np
import pandas as pd
import time
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
DATA_DIR = _os.path.join(_ROOT, "data/m5-data/m5-forecasting-accuracy")


def extract_dept_categories(data_dir, n_skus=50, seed=42):
    """Load M5 and extract real department categories."""
    sales = pd.read_csv(os.path.join(data_dir, "sales_train_evaluation.csv"))
    rng = np.random.RandomState(seed)
    selected = rng.choice(len(sales), size=min(n_skus, len(sales)), replace=False)
    sales = sales.iloc[selected].reset_index(drop=True)
    
    # Extract category from dept_id
    categories = []
    for dept in sales['dept_id']:
        if dept.startswith('FOODS'):
            categories.append('food')
        elif dept.startswith('HOBBIES'):
            categories.append('hobbies')
        elif dept.startswith('HOUSEHOLD'):
            categories.append('household')
        else:
            categories.append('other')
    
    sales['category'] = categories
    return sales


def estimate_kernel_per_sku(sku_data, config):
    """Estimate per-SKU kernel parameters."""
    b = sku_data["demand"]
    f_promo = sku_data["f_promo"]
    f_snap = sku_data["f_snap"]
    n_lags = config["n_lags"]
    pre_window = config["pre_window"]
    
    phi_promo_emp, n_promo = estimate_kernel(b, f_promo, n_lags, pre_window)
    theta_promo = fit_parametric_kernel(phi_promo_emp, family="surge_vacuum")
    if theta_promo is not None:
        phi_promo = kernel_from_theta(theta_promo, n_lags)
    else:
        phi_promo = np.zeros(n_lags + 1)
        phi_promo[0] = 1.0
    
    phi_snap = np.zeros(n_lags + 1)
    phi_snap[0] = 1.0
    
    return phi_promo, phi_snap, theta_promo


def run_pooled_vs_per_sku(n_skus=50, seed=42):
    """Compare per-SKU kernel estimation vs. category-pooled estimation."""
    print("=" * 60)
    print("PARTIAL POOLING: Per-SKU vs. Category-Pooled Kernels")
    print("=" * 60)
    
    # Load data with real categories
    sales_meta = extract_dept_categories(DATA_DIR, n_skus=n_skus, seed=seed)
    dataset = load_m5_data(DATA_DIR, n_skus=n_skus, random_state=seed)
    
    # Assign real categories
    for i, d in enumerate(dataset):
        d['category'] = sales_meta['category'].iloc[i]
    
    config = CONFIG_DEFAULT.copy()
    config["use_ista"] = True
    config["run_baselines"] = False
    config["run_ablations"] = False
    
    # Step 1: Estimate per-SKU kernels for all SKUs
    print("\nStep 1: Estimating per-SKU kernels...")
    per_sku_kernels = {}
    for sku in dataset:
        phi_promo, phi_snap, theta = estimate_kernel_per_sku(sku, config)
        per_sku_kernels[sku['sku_id']] = (phi_promo, phi_snap, theta)
    
    # Step 2: Pool kernels by category
    print("Step 2: Pooling kernels by category...")
    categories = set(d['category'] for d in dataset)
    print(f"Categories found: {categories}")
    
    pooled_kernels = {}
    for cat in categories:
        cat_skus = [d for d in dataset if d['category'] == cat]
        cat_kernels = [per_sku_kernels[d['sku_id']][0] for d in cat_skus]
        
        # Pool by taking median across SKUs (partial pooling)
        pooled_phi = np.median(np.stack(cat_kernels), axis=0)
        pooled_snap = np.zeros(config["n_lags"] + 1)
        pooled_snap[0] = 1.0
        
        pooled_kernels[cat] = (pooled_phi, pooled_snap)
        print(f"  {cat}: {len(cat_skus)} SKUs, pooled kernel sum={pooled_phi.sum():.4f}")
    
    # Also create a global pool (all SKUs)
    all_kernels = [per_sku_kernels[d['sku_id']][0] for d in dataset]
    global_pooled_phi = np.median(np.stack(all_kernels), axis=0)
    global_pooled_snap = np.zeros(config["n_lags"] + 1)
    global_pooled_snap[0] = 1.0
    pooled_kernels['global'] = (global_pooled_phi, global_pooled_snap)
    print(f"  global: {len(dataset)} SKUs, pooled kernel sum={global_pooled_phi.sum():.4f}")
    
    # Step 3: Evaluate each SKU with per-SKU, category-pooled, and global-pooled kernels
    print("\nStep 3: Evaluating per-SKU vs. pooled...")
    
    results = []
    
    for sku in dataset:
        sku_id = sku['sku_id']
        cat = sku['category']
        b = sku['demand']
        f_promo = sku['f_promo']
        f_snap = sku['f_snap']
        T = len(b)
        
        # Per-SKU
        phi_ps, snap_ps = per_sku_kernels[sku_id][:2]
        A_ps = build_Aspec(phi_ps, snap_ps, f_promo, f_snap, T, causal=True)
        xb_ps, xs_ps = decompose_ista(b, A_ps, config["lambda_smooth"], config["lambda_sparse"])
        
        # Category-pooled
        phi_cp, snap_cp = pooled_kernels[cat]
        A_cp = build_Aspec(phi_cp, snap_cp, f_promo, f_snap, T, causal=True)
        xb_cp, xs_cp = decompose_ista(b, A_cp, config["lambda_smooth"], config["lambda_sparse"])
        
        # Global-pooled
        phi_gp, snap_gp = pooled_kernels['global']
        A_gp = build_Aspec(phi_gp, snap_gp, f_promo, f_snap, T, causal=True)
        xb_gp, xs_gp = decompose_ista(b, A_gp, config["lambda_smooth"], config["lambda_sparse"])
        
        # Simple forecast: use baseline + known events (no future, just in-sample fit quality)
        # We measure via decomposition quality: sparsity and reconstruction RMSE
        for name, xb, xs, A in [
            ('Per-SKU', xb_ps, xs_ps, A_ps),
            ('Category-Pooled', xb_cp, xs_cp, A_cp),
            ('Global-Pooled', xb_gp, xs_gp, A_gp),
        ]:
            recon = xb + A @ xs
            rmse = np.sqrt(np.mean((b - recon) ** 2))
            sparsity = np.sum(np.abs(xs) > 1e-3)
            baseline_smoothness = np.sum(np.diff(xb, 2) ** 2)
            
            results.append({
                'sku_id': sku_id,
                'category': cat,
                'method': name,
                'rmse': rmse,
                'sparsity': sparsity,
                'baseline_smoothness': baseline_smoothness,
            })
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Per-SKU vs. Pooled Decomposition Quality")
    print("=" * 60)
    summary = df.groupby('method').agg({
        'rmse': ['mean', 'std'],
        'sparsity': ['mean', 'std', 'median'],
        'baseline_smoothness': ['mean', 'std'],
    }).round(4)
    print(summary)
    
    # Save results
    out_path = _os.path.join(_ROOT, "paper/results/pooled_vs_per_sku.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    
    return df


def generate_pooled_figure(df):
    """Generate figure comparing per-SKU vs. pooled."""
    metrics = ['rmse', 'sparsity', 'baseline_smoothness']
    labels = ['Reconstruction RMSE', 'Sparsity ($\\|x_{\\text{spec}}\\|_0$)', 'Baseline Smoothness']
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    
    methods = ['Per-SKU', 'Category-Pooled', 'Global-Pooled']
    colors = ['#1F77B4', '#2CA02C', '#FF7F0E']
    
    for ax, metric, label in zip(axes, metrics, labels):
        # Box plot
        data_to_plot = [df[df['method'] == m][metric].values for m in methods]
        bp = ax.boxplot(data_to_plot, labels=methods, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
        ax.grid(axis='y', ls=':', alpha=0.4)
        sns.despine(ax=ax, top=True, right=True)
    
    fig.suptitle('Partial Pooling: Per-SKU vs. Category-Pooled Kernel Estimation (50 M5 SKUs)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, 'figure_partial_pooling.png')
    fig.savefig(out_path, dpi=400, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    df = run_pooled_vs_per_sku(n_skus=50, seed=42)
    generate_pooled_figure(df)
    print("\nDone!")
