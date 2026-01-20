"""
Module 1: ML Fundamentals - Exercise Solutions
"""

# =============================================================================
# Exercise 1: Identify the Learning Type
# =============================================================================
exercise_1 = {
    "Predicting melting point in Kelvin": "A",  # Supervised Regression
    "Identifying if a crystal is cubic, tetragonal, or hexagonal": "B",  # Supervised Classification
    "Finding natural groups in a dataset of 1000 alloys": "C",  # Unsupervised Clustering
    "Visualizing 50 features in 2D with t-SNE": "D",  # Unsupervised Dimensionality Reduction
}

print("Exercise 1 - Answers:")
for question, answer in exercise_1.items():
    answer_text = {
        "A": "Supervised Regression",
        "B": "Supervised Classification",
        "C": "Unsupervised Clustering",
        "D": "Unsupervised Dimensionality Reduction"
    }[answer]
    print(f"  {question}")
    print(f"  → {answer_text}\n")


# =============================================================================
# Exercise 2: Detect Overfitting
# =============================================================================
model_results = {
    "Model A": {"Train R²": 0.95, "Test R²": 0.93},
    "Model B": {"Train R²": 0.99, "Test R²": 0.45},
    "Model C": {"Train R²": 0.60, "Test R²": 0.58},
    "Model D": {"Train R²": 0.88, "Test R²": 0.70},
}

print("\nExercise 2 - Analysis:")
print("="*50)

# Analysis
print("\n1. Which model(s) show overfitting?")
print("   → Model B (Train R²=0.99, Test R²=0.45, Gap=0.54)")
print("   → Model D also shows some overfitting (Gap=0.18)")

print("\n2. Which model(s) might be underfitting?")
print("   → Model C (Both Train and Test R² are low at ~0.60)")
print("     This suggests the model is too simple.")

print("\n3. Which model would you choose and why?")
print("   → Model A is the best choice:")
print("     - High performance (R² = 0.93 on test set)")
print("     - Small train-test gap (0.02) indicates good generalization")
print("     - No signs of overfitting or underfitting")


# =============================================================================
# Exercise 3: Implement Cross-Validation with Ridge
# =============================================================================
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np

# Generate sample data (same as in notebook)
np.random.seed(42)
n_materials = 200
electronegativity = np.random.uniform(1.5, 4.0, n_materials)
atomic_radius = np.random.uniform(0.5, 2.5, n_materials)
valence_electrons = np.random.randint(1, 8, n_materials)
band_gap = (0.5 * electronegativity + 0.3 * atomic_radius -
            0.1 * valence_electrons + np.random.normal(0, 0.3, n_materials))
X = np.column_stack([electronegativity, atomic_radius, valence_electrons])
y = band_gap

# Different regularization strengths
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
results = []

print("\nExercise 3 - Ridge Cross-Validation Results:")
print("="*50)
print(f"{'Alpha':<10} {'Mean R²':<12} {'Std':<10}")
print("-"*35)

for alpha in alphas:
    model = Ridge(alpha=alpha)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    results.append({
        'alpha': alpha,
        'mean': cv_scores.mean(),
        'std': cv_scores.std()
    })
    print(f"{alpha:<10} {cv_scores.mean():<12.4f} {cv_scores.std():<10.4f}")

# Find best alpha
best_result = max(results, key=lambda x: x['mean'])
print(f"\nBest alpha: {best_result['alpha']} (Mean R² = {best_result['mean']:.4f})")
