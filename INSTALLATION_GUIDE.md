# Installation Guide for ML-for-Materials-Science Tutorials

This guide covers installation for all three tutorials:
- **Tutorial-07**: ML Discovery (Classical ML for Materials)
- **Tutorial-08**: Neural Network Potentials (M3GNet, CHGNet, MACE)
- **Tutorial-09**: Advanced Features (Atomic Descriptors, Active Learning)

---

## Quick Start for Google Colab

Add this cell at the **beginning** of any notebook and run it first:

```python
# Install all required packages for ML-for-Materials-Science tutorials
!pip install pymatgen matminer shap dscribe ase matgl torch -q

# Restart runtime after installation (only needed once)
# Go to: Runtime -> Restart runtime
```

After running, restart the runtime once, then you're ready to go.

---

## Package Requirements by Tutorial

### Tutorial-07: ML Discovery

| Notebook | Additional Packages Needed |
|----------|---------------------------|
| 01_ml_fundamentals | ipywidgets |
| 02_data_foundation | pymatgen, matminer |
| 03_featurization_basics | pymatgen, matminer |
| 04_classical_ml_models | - |
| 05_model_evaluation | - |
| 06_explainable_ai | shap |
| 07_project_bandgap | pymatgen, matminer, shap |

**Colab install:**
```python
!pip install pymatgen matminer shap -q
```

---

### Tutorial-08: Neural Network Potentials

| Notebook | Additional Packages Needed |
|----------|---------------------------|
| 01_why_nnps | - |
| 02_gnn_basics | torch, torch_geometric |
| 03_universal_mlips | torch, matgl, pymatgen, ase |
| 04_pretrained_models | torch, matgl, pymatgen, ase |
| 05_md_with_nnps | torch, matgl, ase |
| 06_fine_tuning | torch, matgl |
| 07_project_phonons | torch, matgl, ase, phonopy |

**Colab install:**
```python
!pip install torch matgl pymatgen ase -q
# Optional for specific models:
!pip install chgnet -q  # For CHGNet directly
!pip install mace-torch -q  # For MACE (may need GPU)
```

---

### Tutorial-09: Advanced Features

| Notebook | Additional Packages Needed |
|----------|---------------------------|
| 01_atomic_descriptors | dscribe, ase, pymatgen |
| 02_electronic_features | pymatgen |
| 03_dimensionality_reduction | umap-learn |
| 04_active_learning | modAL |
| 05_multi_objective | pymoo |
| 06_generative_models | torch |
| 07_project_alloy_design | pymatgen, matminer |

**Colab install:**
```python
!pip install dscribe ase pymatgen umap-learn modAL pymoo torch -q
```

---

## Full Installation (All Tutorials)

### Option 1: Google Colab (Recommended for Beginners)

Run this once at the start of your session:

```python
# Complete installation for all tutorials
!pip install numpy pandas matplotlib seaborn scikit-learn -q
!pip install pymatgen matminer shap -q
!pip install torch matgl ase dscribe -q
!pip install umap-learn modAL pymoo -q

print("Installation complete! Please restart runtime.")
print("Go to: Runtime -> Restart runtime")
```

**Pre-installed in Colab** (no need to install):
- numpy, pandas, matplotlib, seaborn, scikit-learn, ipywidgets

---

### Option 2: Local Installation with Conda (Recommended)

```bash
# Step 1: Create environment
conda create -n MatSci python=3.10 -y
conda activate MatSci

# Step 2: Install core packages
conda install numpy pandas matplotlib seaborn scikit-learn ipywidgets jupyter -y

# Step 3: Install PyTorch (choose one based on your system)
# For CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install materials science packages
pip install pymatgen matminer shap
pip install matgl ase dscribe
pip install umap-learn modAL pymoo

# Step 5: Launch Jupyter
jupyter notebook
```

---

### Option 3: Local Installation with pip

```bash
# Create virtual environment
python -m venv matsci_env
source matsci_env/bin/activate  # On Windows: matsci_env\Scripts\activate

# Install all packages
pip install numpy pandas matplotlib seaborn scikit-learn ipywidgets jupyter
pip install torch
pip install pymatgen matminer shap
pip install matgl ase dscribe
pip install umap-learn modAL pymoo

# Launch Jupyter
jupyter notebook
```

---

## Verification Script

Run this code to check if all packages are installed:

```python
def check_installation():
    packages = {
        # Core
        'numpy': 'Core',
        'pandas': 'Core',
        'matplotlib': 'Core',
        'seaborn': 'Core',
        'sklearn': 'Core',

        # Tutorial 07
        'pymatgen': 'Tutorial 07, 08, 09',
        'matminer': 'Tutorial 07',
        'shap': 'Tutorial 07',

        # Tutorial 08
        'torch': 'Tutorial 08, 09',
        'matgl': 'Tutorial 08',
        'ase': 'Tutorial 08, 09',

        # Tutorial 09
        'dscribe': 'Tutorial 09',
    }

    print("=" * 60)
    print("Package Installation Check")
    print("=" * 60)

    all_ok = True
    for package, used_in in packages.items():
        try:
            __import__(package)
            status = "[OK]"
        except ImportError:
            status = "[X] "
            all_ok = False
        print(f"{status} {package:<15} - {used_in}")

    print("=" * 60)
    if all_ok:
        print("All packages installed successfully!")
    else:
        print("Some packages missing. Install them before proceeding.")
    print("=" * 60)

check_installation()
```

---

## Troubleshooting

### Common Issues and Solutions

**1. "ModuleNotFoundError" after pip install in Colab**
```
Solution: Restart the runtime
Go to: Runtime -> Restart runtime
```

**2. PyTorch installation fails**
```bash
# Try installing from conda instead:
conda install pytorch -c pytorch
```

**3. matgl import error**
```bash
# Make sure PyTorch is installed first:
pip install torch
pip install matgl
```

**4. dscribe installation fails**
```bash
# Install dependencies first:
pip install ase
pip install dscribe
```

**5. pymatgen installation fails**
```bash
# Try conda:
conda install -c conda-forge pymatgen
```

**6. SHAP installation issues**
```bash
# Make sure you have a C compiler, then:
pip install shap --no-cache-dir
```

**7. Kernel dies / Out of memory**
```
Solutions:
- Close other applications
- Use smaller datasets
- Use Google Colab (free GPU available)
- Enable GPU: Runtime -> Change runtime type -> GPU
```

**8. CUDA/GPU not detected for PyTorch**
```python
import torch
print(torch.cuda.is_available())  # Should be True if GPU available

# If False, reinstall PyTorch with CUDA:
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Minimum System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 4 GB | 8+ GB |
| Storage | 5 GB | 10 GB |
| Python | 3.8+ | 3.10 |
| GPU | Not required | NVIDIA (for NNPs) |

---

## Package Versions (Tested)

These versions are known to work together:

```
numpy>=1.21
pandas>=1.3
matplotlib>=3.5
seaborn>=0.11
scikit-learn>=1.0
pymatgen>=2023.0
matminer>=0.8
shap>=0.41
torch>=2.0
matgl>=0.9
ase>=3.22
dscribe>=2.0
```

---

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Make sure you're using a compatible Python version (3.8-3.11)
3. Try running in Google Colab first to isolate the issue
4. Open an issue on GitHub: https://github.com/NabKh/ML-for-Materials-Science/issues

Include in your issue:
- Operating system
- Python version: `python --version`
- Full error message
- Which notebook you're running

---

## Quick Reference

| Tutorial | Quick Colab Install |
|----------|---------------------|
| Tutorial-07 | `!pip install pymatgen matminer shap -q` |
| Tutorial-08 | `!pip install torch matgl pymatgen ase -q` |
| Tutorial-09 | `!pip install dscribe ase pymatgen torch -q` |
| **All**     | `!pip install pymatgen matminer shap torch matgl ase dscribe -q` |

---

Happy Learning!
