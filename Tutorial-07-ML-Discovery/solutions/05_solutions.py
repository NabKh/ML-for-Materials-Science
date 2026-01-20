"""
Module 5: Model Evaluation - Exercise Solutions
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor

# =============================================================================
# Exercise: Compare Different CV Strategies
# =============================================================================
def compare_cv_strategies(X, y):
    """
    Compare 3-fold, 5-fold, and 10-fold cross-validation.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values

    Returns:
    --------
    results : dict
        CV results for each k value
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    k_values = [3, 5, 10]
    results = {}

    print("Comparing Cross-Validation Strategies:")
    print("="*60)
    print(f"{'K-Folds':<10} {'Mean R²':<15} {'Std R²':<15} {'Min R²':<10} {'Max R²':<10}")
    print("-"*60)

    for k in k_values:
        scores = cross_val_score(rf, X, y, cv=k, scoring='r2')
        results[k] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        print(f"{k:<10} {scores.mean():<15.4f} {scores.std():<15.4f} "
              f"{scores.min():<10.4f} {scores.max():<10.4f}")

    # Analysis
    print("\n" + "="*60)
    print("Analysis:")
    print("-"*60)

    # Find lowest variance
    lowest_var_k = min(results.keys(), key=lambda k: results[k]['std'])
    print(f"✓ Lowest variance: {lowest_var_k}-fold (std = {results[lowest_var_k]['std']:.4f})")

    # Trade-off explanation
    print(f"\nTrade-offs:")
    print(f"  • 3-fold: Faster, but higher variance (fewer folds = less averaging)")
    print(f"  • 5-fold: Good balance between speed and reliability")
    print(f"  • 10-fold: Most stable estimates, but slower and may have small test sets")

    return results


def visualize_cv_comparison(results):
    """Visualize CV comparison results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_values = list(results.keys())

    # Plot 1: Mean scores with error bars
    ax1 = axes[0]
    means = [results[k]['mean'] for k in k_values]
    stds = [results[k]['std'] for k in k_values]

    colors = ['#22d3ee', '#6366f1', '#f472b6']
    bars = ax1.bar(k_values, means, yerr=stds, color=colors, alpha=0.7,
                   capsize=10, edgecolor='white', linewidth=2)

    ax1.set_xlabel('K (Number of Folds)', fontsize=12)
    ax1.set_ylabel('Mean R² Score', fontsize=12)
    ax1.set_title('CV Performance by K-Folds', fontsize=14, fontweight='bold')
    ax1.set_xticks(k_values)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', fontsize=10)

    # Plot 2: Distribution of scores for each k
    ax2 = axes[1]
    for i, k in enumerate(k_values):
        scores = results[k]['scores']
        x = np.random.normal(i, 0.04, len(scores))
        ax2.scatter(x, scores, c=colors[i], alpha=0.7, s=100, label=f'{k}-fold')
        ax2.hlines(results[k]['mean'], i-0.2, i+0.2, colors=colors[i], linewidth=2)

    ax2.set_xlabel('K-Folds', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Score Distribution by K-Folds', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(k_values)))
    ax2.set_xticklabels([f'{k}-fold' for k in k_values])
    ax2.legend()

    plt.tight_layout()
    plt.savefig('figures/05_cv_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Bonus: Other CV Strategies
# =============================================================================
def compare_advanced_cv_strategies(X, y):
    """Compare other CV strategies: Stratified, ShuffleSplit, LOO."""
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

    print("\nAdvanced CV Strategies Comparison:")
    print("="*60)

    # Standard K-Fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kfold = cross_val_score(rf, X, y, cv=kfold, scoring='r2')
    print(f"Standard 5-Fold:     {scores_kfold.mean():.4f} ± {scores_kfold.std():.4f}")

    # Repeated K-Fold (via multiple runs)
    all_scores = []
    for seed in range(5):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')
        all_scores.extend(scores)
    print(f"Repeated 5-Fold (5x): {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")

    # Shuffle Split (Monte Carlo CV)
    ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores_ss = cross_val_score(rf, X, y, cv=ss, scoring='r2')
    print(f"Shuffle Split (10x):  {scores_ss.mean():.4f} ± {scores_ss.std():.4f}")

    # Leave-One-Out (only for small datasets!)
    if len(X) <= 100:
        loo = LeaveOneOut()
        scores_loo = cross_val_score(rf, X, y, cv=loo, scoring='r2')
        print(f"Leave-One-Out:       {scores_loo.mean():.4f} ± {scores_loo.std():.4f}")
    else:
        print(f"Leave-One-Out:       Skipped (dataset too large: {len(X)} samples)")

    print("\nRecommendation:")
    print("  • For most cases: 5-fold or 10-fold CV")
    print("  • For small datasets (<100 samples): Repeated K-Fold or LOO")
    print("  • For large datasets: 5-fold or Shuffle Split")

# Example usage:
# results = compare_cv_strategies(X, y)
# visualize_cv_comparison(results)
# compare_advanced_cv_strategies(X, y)
