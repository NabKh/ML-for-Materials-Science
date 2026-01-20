"""
Module 2: Data Foundation - Exercise Solutions
"""

# =============================================================================
# Exercise 1: Query Different Properties
# =============================================================================
from mp_api.client import MPRester

# Replace with your API key
MP_API_KEY = "YOUR_API_KEY"

def exercise_1_query_stable_oxides():
    """Query materials with formation energy < -1 eV/atom, containing oxygen, 2-3 elements"""
    with MPRester(MP_API_KEY) as mpr:
        stable_oxides = mpr.materials.summary.search(
            formation_energy_per_atom=(-10, -1),  # Very stable (< -1 eV/atom)
            elements=["O"],  # Contains oxygen
            nelements=(2, 3),  # Binary or ternary
            fields=[
                "material_id",
                "formula_pretty",
                "formation_energy_per_atom",
                "band_gap",
                "nelements"
            ],
            num_chunks=1,
            chunk_size=100
        )

    print(f"Found {len(stable_oxides)} stable oxides")
    print("\nTop 10 by stability (most negative formation energy):")

    # Sort by formation energy
    sorted_oxides = sorted(stable_oxides, key=lambda x: x.formation_energy_per_atom)

    for mat in sorted_oxides[:10]:
        print(f"  {mat.material_id}: {mat.formula_pretty:15} "
              f"Ef = {mat.formation_energy_per_atom:.3f} eV/atom, "
              f"Eg = {mat.band_gap:.2f} eV")

    return stable_oxides

# Uncomment to run:
# stable_oxides = exercise_1_query_stable_oxides()


# =============================================================================
# Exercise 2: Explore Data Quality
# =============================================================================
import pandas as pd
import numpy as np

def data_quality_report(df):
    """Generate a data quality report for a materials DataFrame."""
    report = {}

    # Number of samples
    report['n_samples'] = len(df)

    # Number of features (excluding non-numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report['n_features'] = len(numeric_cols)

    # Missing values
    report['missing_values'] = df.isnull().sum().sum()
    report['missing_by_column'] = df.isnull().sum().to_dict()

    # Duplicate formulas (if 'formula' column exists)
    if 'formula' in df.columns:
        report['duplicate_formulas'] = df.duplicated(subset=['formula']).sum()

    # Data types
    report['dtypes'] = df.dtypes.value_counts().to_dict()

    # Basic statistics for numeric columns
    report['numeric_stats'] = {
        'columns': list(numeric_cols),
        'means': df[numeric_cols].mean().to_dict(),
        'stds': df[numeric_cols].std().to_dict(),
        'mins': df[numeric_cols].min().to_dict(),
        'maxs': df[numeric_cols].max().to_dict(),
    }

    # Check for infinite values
    report['infinite_values'] = np.isinf(df[numeric_cols]).sum().sum()

    return report


def print_quality_report(report):
    """Pretty print the data quality report."""
    print("="*60)
    print("DATA QUALITY REPORT")
    print("="*60)

    print(f"\n📊 Dataset Overview:")
    print(f"  • Samples: {report['n_samples']}")
    print(f"  • Numeric features: {report['n_features']}")

    print(f"\n🔍 Data Quality:")
    print(f"  • Total missing values: {report['missing_values']}")
    print(f"  • Infinite values: {report['infinite_values']}")
    if 'duplicate_formulas' in report:
        print(f"  • Duplicate formulas: {report['duplicate_formulas']}")

    # Show columns with missing values
    missing_cols = {k: v for k, v in report['missing_by_column'].items() if v > 0}
    if missing_cols:
        print(f"\n  Columns with missing values:")
        for col, count in missing_cols.items():
            print(f"    - {col}: {count}")

    print(f"\n📈 Data Types:")
    for dtype, count in report['dtypes'].items():
        print(f"  • {dtype}: {count} columns")

    print("\n" + "="*60)


# Example usage:
# df = pd.read_csv('../data/sample_datasets/materials_bandgap.csv')
# report = data_quality_report(df)
# print_quality_report(report)
