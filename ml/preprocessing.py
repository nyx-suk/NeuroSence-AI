"""
Module 2 - Preprocessing & Normalization
Covers CI2018 Unit II: Normalization, K-Fold CV, PCA, LDA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def normalize(X_train, X_test, method='standard'):
    """
    Unit II: Normalize features.
    method='standard' → Z-score (mean=0, std=1)
    method='minmax'   → Scale to [0, 1]
    Also imputes any remaining NaNs with column median before scaling.
    """
    # Impute any residual NaNs (safety net)
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print(f"✅ Imputation + Normalization applied: {method}")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# K-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────

def get_kfold(n_splits=5):
    """
    Unit II: Stratified K-Fold Cross Validation.
    Stratified ensures class balance in each fold.
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"✅ K-Fold CV configured: {n_splits} folds (Stratified)")
    return kf


# ─────────────────────────────────────────────
# PCA — DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────

def apply_pca(X_train_scaled, X_test_scaled, n_components=2):
    """
    Unit II: Principal Component Analysis.
    Reduces high-dimensional feature space to n_components for visualization.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    explained = pca.explained_variance_ratio_ * 100
    print(f"\n📉 PCA → {n_components} components")
    for i, ev in enumerate(explained):
        print(f"  PC{i+1}: {ev:.2f}% variance explained")
    print(f"  Total: {sum(explained):.2f}% variance retained")

    return X_train_pca, X_test_pca, pca


def plot_pca(X_pca, y, class_names, save_path='outputs/pca_visualization.png'):
    """
    Unit II: Plot PCA 2D scatter — visualize class separation.
    """
    plt.figure(figsize=(9, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    unique_classes = sorted(np.unique(y))

    for i, cls in enumerate(unique_classes):
        mask = (y == cls)
        label = class_names[i] if class_names else f"Class {cls}"
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[i % len(colors)],
            label=label, alpha=0.7, edgecolors='k', linewidths=0.4, s=60
        )

    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.title('PCA — 2D Visualization of Alzheimer Classes', fontsize=14, fontweight='bold')
    plt.legend(title='Class', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ PCA plot saved → {save_path}")


# ─────────────────────────────────────────────
# LDA — LINEAR DISCRIMINANT ANALYSIS
# ─────────────────────────────────────────────

def apply_lda(X_train_scaled, X_test_scaled, y_train, n_components=2):
    """
    Unit II: Linear Discriminant Analysis.
    Supervised dimensionality reduction — maximizes class separability.
    """
    n_classes = len(np.unique(y_train))
    n_comp = min(n_components, n_classes - 1)

    lda = LDA(n_components=n_comp)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda  = lda.transform(X_test_scaled)

    explained = lda.explained_variance_ratio_ * 100
    print(f"\n📉 LDA → {n_comp} components")
    for i, ev in enumerate(explained):
        print(f"  LD{i+1}: {ev:.2f}% variance explained")

    return X_train_lda, X_test_lda, lda


def plot_lda(X_lda, y, class_names, save_path='outputs/lda_visualization.png'):
    """Plot LDA 2D scatter."""
    plt.figure(figsize=(9, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    unique_classes = sorted(np.unique(y))

    for i, cls in enumerate(unique_classes):
        mask = (y == cls)
        label = class_names[i] if class_names else f"Class {cls}"
        plt.scatter(
            X_lda[mask, 0],
            X_lda[mask, 1] if X_lda.shape[1] > 1 else np.zeros(mask.sum()),
            c=colors[i % len(colors)],
            label=label, alpha=0.7, edgecolors='k', linewidths=0.4, s=60
        )

    plt.xlabel('LD1', fontsize=12)
    plt.ylabel('LD2' if X_lda.shape[1] > 1 else '', fontsize=12)
    plt.title('LDA — Class Separation Visualization', fontsize=14, fontweight='bold')
    plt.legend(title='Class', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ LDA plot saved → {save_path}")
