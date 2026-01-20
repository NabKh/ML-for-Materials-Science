"""
Module 6: Explainable AI - Exercise Solutions
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not installed. Install with: pip install shap")

# =============================================================================
# Exercise: Analyze Feature Interactions
# =============================================================================
def analyze_feature_interactions(shap_values, X_test, feature_names, features_to_analyze=None):
    """
    Create dependence plots for multiple features to analyze interactions.

    Parameters:
    -----------
    shap_values : shap.Explanation
        SHAP values from explainer
    X_test : array-like
        Test data
    feature_names : list
        Feature names
    features_to_analyze : list, optional
        Specific features to analyze. If None, uses top 5 by importance.
    """
    if not HAS_SHAP:
        print("SHAP not available. Please install with: pip install shap")
        return

    # Determine features to analyze
    if features_to_analyze is None:
        # Get top 5 features by mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
        features_to_analyze = [feature_names[i] for i in top_indices]

    print("Analyzing Feature Interactions:")
    print("="*60)
    print(f"Features: {features_to_analyze}")
    print("-"*60)

    # Create dependence plots
    n_features = len(features_to_analyze)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(features_to_analyze):
        if i < len(axes) - 1:  # Leave last subplot for summary
            ax = axes[i]
            plt.sca(ax)
            shap.dependence_plot(
                feature,
                shap_values.values,
                X_test,
                feature_names=feature_names,
                show=False,
                ax=ax
            )
            ax.set_title(f'{feature}', fontsize=12, fontweight='bold')

    # Last subplot: Summary of relationships
    ax = axes[-1]
    ax.text(0.5, 0.5, 'Feature Interaction Summary\n\n'
            'Look for:\n'
            '• Linear relationships (straight lines)\n'
            '• Non-linear relationships (curves)\n'
            '• Interactions (color patterns)\n\n'
            'Colored by interacting feature',
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/06_feature_interactions.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print interpretation
    print("\n💡 Interpretation Guide:")
    print("-"*60)
    print("• If SHAP value increases with feature value → positive relationship")
    print("• If SHAP value decreases with feature value → negative relationship")
    print("• Curved patterns → non-linear relationships")
    print("• Color patterns → feature interactions")
    print("• Vertical spread at same x → other features matter too")


def analyze_interaction_effects(shap_values, X_test, feature_names):
    """Calculate and visualize SHAP interaction values."""
    if not HAS_SHAP:
        print("SHAP not available.")
        return

    # Get mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    # Create correlation matrix of SHAP values
    shap_corr = np.corrcoef(shap_values.values.T)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Feature importance
    ax1 = axes[0]
    sorted_idx = np.argsort(mean_abs_shap)
    top_10_idx = sorted_idx[-10:]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 10))
    ax1.barh([feature_names[i] for i in top_10_idx],
             mean_abs_shap[top_10_idx], color=colors)
    ax1.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax1.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')

    # Plot 2: SHAP correlation heatmap (top 10 features)
    ax2 = axes[1]
    top_10_names = [feature_names[i] for i in top_10_idx]
    top_10_corr = shap_corr[np.ix_(top_10_idx, top_10_idx)]

    im = ax2.imshow(top_10_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xticklabels(top_10_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(top_10_names, fontsize=8)
    ax2.set_title('SHAP Value Correlations', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Correlation')

    plt.tight_layout()
    plt.savefig('figures/06_shap_interactions.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Find strongest interactions
    print("\nStrongest Feature Interactions (by SHAP correlation):")
    print("="*60)

    interactions = []
    for i in range(len(top_10_idx)):
        for j in range(i+1, len(top_10_idx)):
            interactions.append({
                'feature_1': top_10_names[i],
                'feature_2': top_10_names[j],
                'correlation': top_10_corr[i, j]
            })

    interactions = sorted(interactions, key=lambda x: abs(x['correlation']), reverse=True)

    for inter in interactions[:5]:
        sign = "+" if inter['correlation'] > 0 else "-"
        print(f"  {inter['feature_1'][:20]:<20} ↔ {inter['feature_2'][:20]:<20} "
              f"r={sign}{abs(inter['correlation']):.3f}")


# =============================================================================
# Bonus: SHAP Force Plot for Multiple Samples
# =============================================================================
def explain_predictions(model, explainer, X_test, y_test, feature_names, n_samples=5):
    """Explain multiple predictions using SHAP."""
    if not HAS_SHAP:
        print("SHAP not available.")
        return

    shap_values = explainer(X_test[:n_samples])

    print(f"\nExplaining {n_samples} Predictions:")
    print("="*60)

    for i in range(n_samples):
        y_pred = model.predict(X_test[i:i+1])[0]
        y_actual = y_test[i]
        error = y_pred - y_actual

        print(f"\nSample {i}:")
        print(f"  Actual: {y_actual:.3f} eV")
        print(f"  Predicted: {y_pred:.3f} eV")
        print(f"  Error: {error:+.3f} eV")

        # Top contributing features
        feature_contributions = list(zip(feature_names, shap_values.values[i]))
        feature_contributions = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)

        print(f"  Top 3 contributing features:")
        for feat, contrib in feature_contributions[:3]:
            sign = "+" if contrib > 0 else ""
            print(f"    • {feat[:30]}: {sign}{contrib:.4f}")

# Example usage:
# analyze_feature_interactions(shap_values, X_test_s, feature_names)
# analyze_interaction_effects(shap_values, X_test_s, feature_names)
# explain_predictions(model, explainer, X_test_s, y_test, feature_names)
