# 🧠 NeuroSense AI
### Interpretable Alzheimer's Disease Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Course](https://img.shields.io/badge/CI2018-Machine%20Learning-purple?style=flat-square)

> **"ML explains risk — clinically, interpretably, accurately."**

---

## 📌 Overview

**NeuroSense AI** is a pure Machine Learning system for early detection and risk prediction of Alzheimer's disease using real clinical data from the **OASIS Longitudinal MRI Dataset**. The system achieves **96% accuracy** using a Soft Voting Ensemble of Random Forest and SVM models, with full clinical explainability through feature importance analysis.

Built as the **CI2018 Machine Learning** course project at **VIT Pune, AY 2025-26**.

---

## 🎯 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:--------:|:---------:|:------:|:--------:|
| **Voting Ensemble** | **96.00%** 🟢 | **0.9629** | **0.9600** | **0.9599** |
| Random Forest | 94.67% | 0.9479 | 0.9467 | 0.9466 |
| SVM (Linear) | 94.67% | 0.9517 | 0.9467 | 0.9465 |
| SVM (Polynomial) | 94.67% | 0.9517 | 0.9467 | 0.9465 |
| SVM (RBF) | 90.67% | 0.9092 | 0.9067 | 0.9065 |

**5-Fold Cross-Validation: 94.42% ± 3.82% (RF) | 95.40% ± 3.17% (SVM-RBF)**

---

## 📂 Project Structure

```
NeuroSense-AI/
│
├── 📁 ml/                          # Core ML module (CI2018)
│   ├── data_loader.py              # Unit I  — Data loading, preprocessing, feature engineering
│   ├── preprocessing.py            # Unit II — Normalization, PCA, LDA, K-Fold CV
│   ├── models.py                   # Unit V  — Random Forest, SVM (3 kernels), Voting Ensemble
│   ├── evaluation.py               # Unit III— Confusion Matrix, ROC/AUC, Precision, Recall, F1
│   ├── feature_importance.py       # Explainability — Feature ranking, per-patient risk
│   └── generate_dataset.py         # OASIS dataset downloader + synthetic fallback
│
├── 📁 data/
│   └── oasis_longitudinal.csv      # Real OASIS clinical dataset (auto-downloaded)
│
├── 📁 outputs/                     # Generated charts and plots
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   ├── feature_importance.png
│   ├── pca_visualization.png
│   ├── lda_visualization.png
│   └── model_comparison.png
│
├── 📁 website/                     # Full HTML/CSS/JS website
│   └── index.html
│
├── run_ml_pipeline.py              # Main pipeline — run this
├── app.py                          # Streamlit web app
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn plotly streamlit imbalanced-learn
```

### 2. Run the ML Pipeline
```bash
python run_ml_pipeline.py
```
This will:
- Auto-download the OASIS dataset
- Preprocess + engineer clinical features
- Train all models with hyperparameter tuning
- Generate all evaluation charts in `outputs/`

### 3. Launch the Web App
```bash
streamlit run app.py
```

### 4. Open the Website
```
Open website/index.html in any browser
```

---

## 🧪 Dataset — OASIS Longitudinal

| Property | Value |
|:---------|:------|
| Source | Open Access Series of Imaging Studies (OASIS) |
| Records | 373 clinical patient records |
| Features | Age, Gender, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF |
| Target | Demented / Nondemented |
| Missing Values | SES (19), MMSE (2) — filled with median |

**Clinical Feature Engineering (6 derived features):**
| Feature | Formula | Clinical Meaning |
|:--------|:--------|:----------------|
| `Cognitive_Index` | MMSE × (1 − CDR×0.5) | Combined cognitive + dementia score |
| `CDR_MMSE_product` | CDR × (30 − MMSE) | Severity × cognitive deficit |
| `MMSE_CDR_diff` | MMSE − (CDR×10) | Net cognitive score |
| `Age_MMSE_ratio` | Age / (MMSE+1) | Cognitive decline relative to age |
| `Brain_Vol_Ratio` | nWBV / (eTIV/1000) | Normalized brain volume |
| `Age_nWBV_interact` | Age × (1 − nWBV) | Age-adjusted brain atrophy |

---

## ⚙️ Pipeline Steps

```
Step 1 → Load OASIS Dataset (auto-download)
Step 2 → Preprocess + Feature Engineering (Unit I + II)
Step 3 → PCA & LDA Visualization (Unit II)
Step 4 → Train Models with Hyperparameter Tuning (Unit V)
Step 5 → 5-Fold Stratified Cross-Validation (Unit II)
Step 6 → Evaluation Metrics (Unit III)
Step 7 → Feature Importance & Clinical Explainability
```

---

## 📚 CI2018 Syllabus Coverage

| Unit | Topic | Implementation |
|:-----|:------|:--------------|
| **Unit I** | ML Basics, Data Types, Feature Engineering, Train/Test Split | `data_loader.py` |
| **Unit II** | Normalization, K-Fold CV, PCA, LDA, Dimensionality Reduction | `preprocessing.py` |
| **Unit III** | Confusion Matrix, Accuracy, Precision, Recall, F1, ROC/AUC | `evaluation.py` |
| **Unit IV** | Gradient Descent, Cost Functions, Bias-Variance, Regularization | `models.py` (tuning) |
| **Unit V** | SVM (3 kernels), Random Forest, Bagging, Feature Importance | `models.py`, `feature_importance.py` |

---

**VIT Pune · B.Tech CSE (AI) · CI2018 Machine Learning · AY 2025-26**

---

## � License

MIT License — free to use for academic purposes.
