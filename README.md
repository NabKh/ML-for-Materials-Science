# Machine Learning for Materials Science

A comprehensive, interactive learning path for applying machine learning to materials discovery, property prediction, and atomistic simulations.

## Learning Path Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ML FOR MATERIALS SCIENCE                                 │
│                    Complete Learning Path                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Tutorial 07          Tutorial 08              Tutorial 09                  │
│  ┌─────────────┐     ┌─────────────────┐      ┌──────────────────┐          │
│  │ ML          │     │ Neural Network  │      │ Advanced         │          │
│  │ Discovery   │ ──► │ Potentials      │ ──►  │ Features         │          │
│  │             │     │                 │      │                  │          │
│  │ • ML Basics │     │ • GNN Basics    │      │ • SOAP/MBTR      │          │
│  │ • matminer  │     │ • M3GNet/CHGNet │      │ • Active Learn   │          │
│  │ • sklearn   │     │ • MD with NNPs  │      │ • Bayesian Opt   │          │
│  │ • SHAP      │     │ • Fine-tuning   │      │ • Genertic Models│          │
│  └─────────────┘     └─────────────────┘      └──────────────────┘          │
│                                                                             │
│  Difficulty:  🟢 Beginner  →  🟡 Intermediate  →  🔴 Advanced                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Before starting, you should be comfortable with:
- **Python basics**: variables, functions, loops, classes
- **NumPy & Pandas**: array operations, DataFrames
- **Basic chemistry/materials science**: what are crystals, compositions, properties
- **Optional but helpful**: Previous tutorials (01-06) in this series

## Quick Start

### Option 1: Cloud (No Installation Required)

Each notebook includes buttons to launch in:
- **Google Colab**: Click "Open in Colab" badge
- **Binder**: Click "Launch Binder" badge

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/NabKh/ML-for-Materials-Science.git
cd ML-for-Materials-Science

# Create conda environment
conda env create -f environment.yml
conda activate ml-materials

# Verify installation
jupyter notebook setup_check.ipynb

# Start learning!
jupyter lab
```

### GPU Support (Optional)

For faster neural network training:
```bash
# After activating environment
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Tutorial Structure

### Tutorial 07: ML for Materials Discovery
*Foundation course - Start here!*

| # | Notebook | Difficulty | Key Topics |
|---|----------|------------|------------|
| 1 | ML Fundamentals | 🟢 | Supervised/unsupervised, overfitting, cross-validation |
| 2 | Data Foundation | 🟢 | Materials Project API, data cleaning, splits |
| 3 | Featurization Basics | 🟢🟡 | matminer, composition & structure features |
| 4 | Classical ML Models | 🟡 | Linear → Random Forest → XGBoost |
| 5 | Model Evaluation | 🟡 | Metrics, learning curves, hyperparameter tuning |
| 6 | Explainable AI | 🟡🔴 | SHAP values, feature importance |
| 7 | Project: Band Gap | 🔴 | End-to-end ML pipeline |

### Tutorial 08: Neural Network Potentials
*Deep learning for atomistic simulations*

| # | Notebook | Difficulty | Key Topics |
|---|----------|------------|------------|
| 1 | Why NNPs? | 🟢 | DFT limitations, accuracy vs speed |
| 2 | GNN Basics | 🟡 | Graphs, message passing, CGCNN |
| 3 | Universal MLIPs | 🟡 | M3GNet, CHGNet, MACE architectures |
| 4 | Pretrained Models | 🟡 | MatGL, loading models, predictions |
| 5 | MD with NNPs | 🟡🔴 | ASE integration, simulations |
| 6 | Fine-tuning | 🔴 | Transfer learning, avoiding forgetting |
| 7 | Project: Phonons | 🔴 | Phonon calculation with NNPs |

### Tutorial 09: Advanced Features & Discovery
*Cutting-edge ML for materials*

| # | Notebook | Difficulty | Key Topics |
|---|----------|------------|------------|
| 1 | Atomic Descriptors | 🟡 | SOAP, MBTR, ACSF with DScribe |
| 2 | Electronic Features | 🟡 | DOS fingerprints, band structure |
| 3 | Dimensionality Reduction | 🟡 | PCA, t-SNE, UMAP visualization |
| 4 | Active Learning | 🟡🔴 | Bayesian optimization, acquisition |
| 5 | Multi-objective Opt | 🔴 | Pareto fronts, trade-offs |
| 6 | Generative Models | 🔴 | VAE, diffusion intro |
| 7 | Project: Alloy Design | 🔴 | Design alloy with target properties |

## Interactive Features

### Jupyter Widgets
Every notebook includes interactive elements:
- **Sliders** to explore hyperparameters
- **Dropdowns** to select models/features
- **Checkboxes** for feature selection
- **Interactive plots** with Plotly

### Self-Check Quizzes
Test your understanding with embedded quizzes:
```python
# Example quiz widget
quiz.check_answer("What prevents overfitting?", your_answer)
```

### Visual Learning
- Animated diagrams explaining concepts
- Side-by-side model comparisons
- Interactive feature importance plots
- 3D materials space exploration

## Key Libraries Used

| Library | Purpose | Documentation |
|---------|---------|---------------|
| `scikit-learn` | Classical ML algorithms | [docs](https://scikit-learn.org/) |
| `matminer` | Materials featurization | [docs](https://hackingmaterials.lbl.gov/matminer/) |
| `pymatgen` | Materials analysis | [docs](https://pymatgen.org/) |
| `dscribe` | Atomic descriptors | [docs](https://singroup.github.io/dscribe/) |
| `matgl` | Graph neural networks | [docs](https://matgl.ai/) |
| `shap` | Model explainability | [docs](https://shap.readthedocs.io/) |
| `mp-api` | Materials Project API | [docs](https://docs.materialsproject.org/) |

## Data Sources

| Database | Content | Access |
|----------|---------|--------|
| Materials Project | ~150,000 materials | API key (free) |
| AFLOW | ~3.5M materials | Open |
| OQMD | ~1M materials | Open |
| JARVIS-DFT | ~75,000 materials | Open |

**Note**: You'll need a free Materials Project API key. Get one at:
https://materialsproject.org/api


## Contributing

Found an error? Have a suggestion? Please open an issue or pull request!

## Citation

If you use these tutorials in your research or teaching, please cite:
```
Khossossi, N. (2026). ML for Materials Science: Interactive Tutorial Series.
https://sustai-nabil.com/teaching
```

## License

This work is licensed under CC BY-NC-SA 4.0. You are free to share and adapt
for non-commercial purposes with attribution.

## Acknowledgments

- Materials Project team for pymatgen and matminer
- MatGL developers for M3GNet/CHGNet implementations
- DScribe team for atomic descriptors
- The open-source ML and materials science communities

---

**Ready to start?** Open `Tutorial-07-ML-Discovery/notebooks/01_ml_fundamentals.ipynb`

Questions? Contact: n.khossossi@differ.nl
