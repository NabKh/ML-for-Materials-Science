"""
Module 3: Featurization Basics - Exercise Solutions
"""

from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Exercise 1: Compare Featurizer Presets
# =============================================================================
def compare_presets():
    """Compare different ElementProperty presets."""
    presets = ['magpie', 'matminer', 'deml']

    print("Comparing ElementProperty Presets:")
    print("="*50)

    preset_info = []
    for preset in presets:
        try:
            ep = ElementProperty.from_preset(preset)
            n_features = len(ep.feature_labels())
            preset_info.append({
                'preset': preset,
                'n_features': n_features,
                'sample_features': ep.feature_labels()[:5]
            })
            print(f"\n{preset.upper()} preset:")
            print(f"  • Number of features: {n_features}")
            print(f"  • Sample features:")
            for feat in ep.feature_labels()[:5]:
                print(f"    - {feat}")
        except Exception as e:
            print(f"\n{preset.upper()} preset: Error - {e}")

    # Summary comparison
    print("\n" + "="*50)
    print("Summary:")
    print(f"{'Preset':<15} {'Features':<15}")
    print("-"*30)
    for info in preset_info:
        print(f"{info['preset']:<15} {info['n_features']:<15}")

    return preset_info

# Run comparison
# preset_info = compare_presets()


# =============================================================================
# Exercise 2: Feature Importance with Random Forest
# =============================================================================
def feature_importance_analysis(X, y, feature_names):
    """
    Train a Random Forest and extract feature importances.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
    feature_names : list
        Names of features

    Returns:
    --------
    importance_df : DataFrame
        Feature importances sorted by importance
    """
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Get feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Print top features
    print("Top 15 Features by Random Forest Importance:")
    print("="*60)
    for i, row in importance_df.head(15).iterrows():
        bar = "█" * int(row['importance'] * 100)
        print(f"{row['feature'][:40]:<40} {row['importance']:.4f} {bar}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 20
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))

    importance_plot = importance_df.head(top_n)
    ax.barh(range(top_n), importance_plot['importance'], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(importance_plot['feature'])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Random Forest Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('figures/03_rf_importance_solution.png', dpi=150, bbox_inches='tight')
    plt.show()

    return importance_df, rf

# Example usage:
# importance_df, rf = feature_importance_analysis(X, y, selected_kbest)


# =============================================================================
# Bonus: Correlation Analysis Between Features
# =============================================================================
def analyze_feature_correlations(X, feature_names, threshold=0.9):
    """Find highly correlated features."""
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pairs.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'correlation': corr_matrix[i, j]
                })

    # Sort by correlation
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)

    print(f"\nHighly Correlated Feature Pairs (|r| > {threshold}):")
    print("="*70)
    for pair in high_corr_pairs[:10]:
        print(f"{pair['feature_1'][:25]:<25} ↔ {pair['feature_2'][:25]:<25} r={pair['correlation']:.3f}")

    return high_corr_pairs
