[![Python Version](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/tatendakasirori/LSUA_PROJECT/actions/workflows/main.yml/badge.svg)](https://github.com/tatendakasirori/LSUA_PROJECT/actions)
[![GitHub last commit](https://img.shields.io/github/last-commit/tatendakasirori/LSUA_PROJECT)](https://github.com/tatendakasirori/LSUA_PROJECT/commits/main)

# Circumventing Density Functional Theory: Efficient Prediction of Molecular Properties Using Classical Machine Learning Models

This repository contains all code, data processing scripts, and models developed under the **LSUA SURE 2025** program to explore the use of classical machine learning models as computationally efficient surrogates for **Density Functional Theory (DFT)** in predicting molecular properties from the **QM9 dataset**.

> **Goal:** Replace costly DFT calculations with interpretable, lightweight ML models trained on engineered molecular descriptors to predict properties such as **dipole moment (μ), HOMO, LUMO**, and **HOMO-LUMO gap**.

---

## 📂 Project Structure

```
LSUA_PROJECT/
├── data/ # Raw and processed datasets
├── descriptors/ # Output descriptor files (2D, 3D, Mordred)
├── models/ # Saved models and configurations
├── notebooks/ # Jupyter notebooks for EDA and experiments
├── results/
│ ├── metrics/ # Evaluation metrics
│ ├── plots/ # Visualizations (PCA, SHAP, etc.)
│ └── logs/ # Training logs
├── scripts/ # Descriptor calculation & cleaning
├── src/ # Core ML training, logging, utils
├── environment.yml # Conda environment with dependencies
├── requirements.txt # Optional pip-based install
└── README.md # You are here
```

---

## Dataset: QM9

- Source: [`tensorflow_datasets.qm9`](https://www.tensorflow.org/datasets/catalog/qm9)
- ~134,000 small organic molecules (C, H, O, N, F)
- Key target properties:
  - `mu`: Dipole moment (Debye)
  - `HOMO`: Highest Occupied Molecular Orbital energy (eV)
  - `LUMO`: Lowest Unoccupied Molecular Orbital energy (eV)
  - `gap`: HOMO–LUMO energy gap (eV)

---

## Molecular Descriptors Used

- **3D Shape Descriptors:**  
  `RadiusOfGyration`, `Asphericity`, `Eccentricity`,  
  `PMI1`, `PMI2`, `SpherocityIndex`, `NPR2`

- **Spectral Descriptors (Autocorrelation-based):**  
  `SpAbs_A`, `SpMax_A`, `SpDiam_A`, `SpMAD_A`, `VE1_A`, `VE3_A`, `VR1_A`

- **Topological and Structural Counts:**  
  - Number of heteroatoms: `nHetero`  
  - Bond counts: `nBonds`, `nBondsO` (oxygen), `nBondsS` (sulfur), `nBondsM`, `nBondsKD`  
  - Carbon hybridization counts: `C1SP1`, `C2SP1`, `C1SP2`, `C2SP2`, `C1SP3`, `C2SP3`  
  - Hybridization ratio: `HybRatio`  
  - Fraction of sp3 carbons: `FCSP3`

> **Note:** Molecules that fail during descriptor generation are dropped to maintain data integrity.

---

## Machine Learning Models

| Model               | Notes                              |
|---------------------|-----------------------------------|
| Ridge Regression    | Baseline linear model             |
| SVR (RBF kernel)    | Captures nonlinear relationships |
| Random Forest       | Handles noisy, high-dimensional data |
| Gradient Boosting   | Robust and interpretable          |
| XGBoost             | Fast, regularized gradient boosting |
| LightGBM            | Leaf-wise boosting for large data |
| CatBoost            | Handles categorical data internally |
| Neural Network      | TensorFlow-based MLP baseline     |
| **Stacked Model**   | Ensemble meta-learner combining base models |

---
## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/your-username/LSUA_PROJECT.git
cd LSUA_PROJECT
```
### 2. Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate lsua_qm9
```
### 3. Generate Molecular Descriptors

```bash
python scripts/compute_rdkit_descriptors.py
python scripts/compute_full_mordred_descriptors.py
```
### 4. Train the models

```bash
python src/train_models.py
```
---
## 📈 Results
- Evaluation metrics are saved in results/metrics/
- Visualizations (e.g., PCA, SHAP) in results/plots/
- Logs and training outputs in results/logs/

Model performance is benchmarked primarily using Mean Squared Error (MSE) and R² score across all target properties.

---
## 🧱 Built With
- Python 3.10
- RDKit
- Mordred
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- TensorFlow
- Pandas, NumPy, Seaborn, Matplotlib
  ---
  
## Contributing
Feel free to submit issues and enhancement requests!
## License
MIT License - feel free to use and modify as needed. 


