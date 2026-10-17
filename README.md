# Machine Learning for Materials Science

A comprehensive, interactive learning path for applying machine learning to materials discovery, property prediction, and atomistic simulations.

## Learning Path Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ML FOR MATERIALS SCIENCE                                 │
│                    Complete Learning Path                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Tutorial 07          Tutorial 08 (planned)    Tutorial 09 (planned)        │
│  ┌─────────────┐     ┌─────────────────┐      ┌──────────────────┐          │
│  │ ML          │     │ Neural Network  │      │ Advanced         │          │
│  │ Discovery   │ ──► │ Potentials      │ ──►  │ Features         │          │
│  │             │     │                 │      │                  │          │
│  │ • ML Basics │     │ • GNN Basics    │      │ • SOAP/MBTR      │          │
│  │ • matminer  │     │ • M3GNet/CHGNet │      │ • Active Learn   │          │
│  │ • sklearn   │     │ • MD with NNPs  │      │ • Bayesian Opt   │          │
│  │ • SHAP      │     │ • Fine-tuning   │      │ • Generative ML  │          │
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
- **Optional but helpful**: the earlier DFT tutorials of this series, e.g. [Tutorial-04-VASP-Catalysis](https://github.com/NabKh/Tutorial-04-VASP-Catalysis)

## Quick Start

### Option 1: Cloud (No Installation Required)

Every notebook starts with an **Open in Colab** badge. Notebooks 01–03 also have a **Launch Binder** badge.

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
*Foundation course - Start here! Available now in [`Tutorial-07-ML-Discovery/`](Tutorial-07-ML-Discovery), with worked solutions in `solutions/` and a sample band-gap dataset in `data/sample_datasets/`.*

| # | Notebook | Difficulty | Key Topics |
|---|----------|------------|------------|
| 1 | ML Fundamentals | 🟢 | Supervised/unsupervised, overfitting, cross-validation |
| 2 | Data Foundation | 🟢 | Materials Project API, data cleaning, splits |
| 3 | Featurization Basics | 🟢🟡 | matminer, composition & structure features |
| 4 | Classical ML Models | 🟡 | Linear → Random Forest → XGBoost |
| 5 | Model Evaluation | 🟡 | Metrics, learning curves, hyperparameter tuning |
| 6 | Explainable AI | 🟡🔴 | SHAP values, feature importance |
| 7 | Project: Band Gap | 🔴 | End-to-end ML pipeline |

### Tutorial 08: Neural Network Potentials (planned)
*Deep learning for atomistic simulations. Not released yet; the outline below is the plan. Watch the repository to be notified.*

| # | Notebook | Difficulty | Key Topics |
|---|----------|------------|------------|
| 1 | Why NNPs? | 🟢 | DFT limitations, accuracy vs speed |
| 2 | GNN Basics | 🟡 | Graphs, message passing, CGCNN |
| 3 | Universal MLIPs | 🟡 | M3GNet, CHGNet, MACE architectures |
| 4 | Pretrained Models | 🟡 | MatGL, loading models, predictions |
| 5 | MD with NNPs | 🟡🔴 | ASE integration, simulations |
| 6 | Fine-tuning | 🔴 | Transfer learning, avoiding forgetting |
| 7 | Project: Phonons | 🔴 | Phonon calculation with NNPs |

### Tutorial 09: Advanced Features & Discovery (planned)
*Cutting-edge ML for materials. Not released yet; the outline below is the plan.*

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

- **Learning objectives** (a project goal in notebook 07) and a **time estimate** at the top of every notebook.
- **Interactive widgets**: a click-to-answer "supervised or unsupervised?" quiz in notebook 01 (ipywidgets).
- **Self-check quizzes** in notebooks 01, 06 and 07.
- **Worked solutions** for the exercises of notebooks 01–06 in `Tutorial-07-ML-Discovery/solutions/`.
- **End-to-end project** in notebook 07: a band-gap model from data to evaluation.

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
| [Materials Project](https://materialsproject.org) | ~163,000 materials (database version 2026.04.13) | API key (free) |
| [AFLOW](https://aflow.org) | DFT-computed compounds and prototypes | Open |
| [OQMD](https://oqmd.org) | DFT formation energies and stabilities | Open |
| [JARVIS-DFT](https://jarvis.nist.gov) | ~77,000 materials | Open |

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

MIT, see [LICENSE](LICENSE).

## Acknowledgments

- Materials Project team for pymatgen and matminer
- MatGL developers for M3GNet/CHGNet implementations
- DScribe team for atomic descriptors
- The open-source ML and materials science communities

---

**Ready to start?** Open `Tutorial-07-ML-Discovery/notebooks/01_ml_fundamentals.ipynb`

Questions? Contact: n.khossossi@differ.nl
