"""
Module 4: Classical ML Models - Exercise Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# =============================================================================
# Exercise: Tune Random Forest Hyperparameters
# =============================================================================
def tune_random_forest(X, y):
    """
    Grid search for optimal Random Forest hyperparameters.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values

    Returns:
    --------
    best_params : dict
        Best hyperparameters found
    results_df : DataFrame
        All results from grid search
    """
    # Define parameter grid
    n_estimators_options = [50, 100, 200]
    max_depth_options = [5, 10, 15, 20, None]
    min_samples_split_options = [2, 5, 10]

    results = []

    print("Grid Search for Random Forest Hyperparameters:")
    print("="*70)
    print(f"{'n_estimators':<15} {'max_depth':<12} {'min_samples':<12} {'Mean R²':<12} {'Std':<10}")
    print("-"*70)

    best_score = -np.inf
    best_params = {}

    total = len(n_estimators_options) * len(max_depth_options) * len(min_samples_split_options)
    count = 0

    for n_est in n_estimators_options:
        for depth in max_depth_options:
            for min_samples in min_samples_split_options:
                count += 1
                rf = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_split=min_samples,
                    random_state=42,
                    n_jobs=-1
                )

                cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()

                results.append({
                    'n_estimators': n_est,
                    'max_depth': depth if depth else 'None',
                    'min_samples_split': min_samples,
                    'mean_r2': mean_score,
                    'std_r2': std_score
                })

                depth_str = str(depth) if depth else 'None'
                print(f"{n_est:<15} {depth_str:<12} {min_samples:<12} {mean_score:<12.4f} {std_score:<10.4f}")

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'min_samples_split': min_samples
                    }

    results_df = pd.DataFrame(results).sort_values('mean_r2', ascending=False)

    print("\n" + "="*70)
    print(f"Best Parameters:")
    print(f"  n_estimators: {best_params['n_estimators']}")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  min_samples_split: {best_params['min_samples_split']}")
    print(f"  Mean R²: {best_score:.4f}")

    return best_params, results_df


def visualize_grid_search(results_df):
    """Visualize grid search results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Effect of n_estimators
    ax1 = axes[0]
    for depth in results_df['max_depth'].unique():
        subset = results_df[results_df['max_depth'] == depth]
        grouped = subset.groupby('n_estimators')['mean_r2'].mean()
        ax1.plot(grouped.index, grouped.values, 'o-', label=f'depth={depth}')
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('Mean R²')
    ax1.set_title('Effect of n_estimators')
    ax1.legend(fontsize=8)

    # Effect of max_depth
    ax2 = axes[1]
    grouped = results_df.groupby('max_depth')['mean_r2'].mean()
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(grouped)))
    ax2.bar(range(len(grouped)), grouped.values, color=colors)
    ax2.set_xticks(range(len(grouped)))
    ax2.set_xticklabels(grouped.index)
    ax2.set_xlabel('max_depth')
    ax2.set_ylabel('Mean R²')
    ax2.set_title('Effect of max_depth')

    # Effect of min_samples_split
    ax3 = axes[2]
    grouped = results_df.groupby('min_samples_split')['mean_r2'].mean()
    ax3.bar(grouped.index, grouped.values, color='#6366f1', alpha=0.7)
    ax3.set_xlabel('min_samples_split')
    ax3.set_ylabel('Mean R²')
    ax3.set_title('Effect of min_samples_split')

    plt.tight_layout()
    plt.savefig('figures/04_grid_search_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Alternative: Using sklearn's GridSearchCV
# =============================================================================
def tune_with_gridsearchcv(X, y):
    """Use sklearn's GridSearchCV for hyperparameter tuning."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='r2',
        n_jobs=-1, verbose=1, return_train_score=True
    )

    grid_search.fit(X, y)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search

# Example usage:
# best_params, results_df = tune_random_forest(X_train_scaled, y_train)
# visualize_grid_search(results_df)
