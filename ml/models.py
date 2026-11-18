"""
Module 3 - ML Models: Random Forest + SVM + Voting Ensemble
Covers CI2018 Unit V: SVM (kernels), Random Forest (Bagging), Ensemble

Accuracy boosting strategy:
  1. Merge Converted → Demented (clean binary problem)
  2. SMOTE oversampling on training set (handle any remaining imbalance)
  3. Hyperparameter tuning via GridSearchCV / RandomizedSearchCV
  4. Voting Ensemble: RF + SVM(RBF) + SVM(Linear) combined
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
import warnings
warnings.filterwarnings('ignore')

# SMOTE for oversampling minority class
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ─────────────────────────────────────────────
# SMOTE OVERSAMPLING
# ─────────────────────────────────────────────

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance classes in training set.
    Only applied to training data — never test data.
    """
    if not SMOTE_AVAILABLE:
        print("  ⚠️  imbalanced-learn not installed. Skipping SMOTE.")
        return X_train, y_train

    class_counts = np.bincount(y_train)
    print(f"\n⚖️  Class distribution before SMOTE: {dict(enumerate(class_counts))}")

    smote = SMOTE(random_state=42, k_neighbors=min(5, min(class_counts) - 1))
    X_res, y_res = smote.fit_resample(X_train, y_train)

    class_counts_after = np.bincount(y_res)
    print(f"   Class distribution after SMOTE : {dict(enumerate(class_counts_after))}")
    print(f"   Training samples: {len(X_train)} → {len(X_res)}")
    return X_res, y_res


# ─────────────────────────────────────────────
# RANDOM FOREST — Tuned
# ─────────────────────────────────────────────

def train_random_forest(X_train, y_train, tune=True, random_state=42):
    """
    Unit V: Random Forest with hyperparameter tuning.
    Bagging ensemble — each tree on a bootstrap sample.
    """
    print("\n🌲 Training Random Forest...")

    if tune:
        param_grid = {
            'n_estimators'     : [300, 500, 700],
            'max_depth'        : [None, 15, 20, 30],
            'min_samples_split': [2, 3],
            'min_samples_leaf' : [1, 2],
            'max_features'     : ['sqrt', 'log2'],
        }
        base = RandomForestClassifier(
            class_weight='balanced', random_state=random_state, n_jobs=-1
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            base, param_grid, n_iter=30, cv=cv,
            scoring='f1_weighted', n_jobs=-1,
            random_state=random_state, verbose=0
        )
        search.fit(X_train, y_train)
        rf = search.best_estimator_
        print(f"  Best params : {search.best_params_}")
        print(f"  CV F1 score : {search.best_score_:.4f}")
    else:
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=None,
            min_samples_split=2, min_samples_leaf=1,
            max_features='sqrt', bootstrap=True,
            class_weight='balanced',
            random_state=random_state, n_jobs=-1
        )
        rf.fit(X_train, y_train)

    print(f"  ✅ Random Forest trained")
    return rf


# ─────────────────────────────────────────────
# SVM — Tuned
# ─────────────────────────────────────────────

def train_svm(X_train, y_train, kernel='rbf', tune=True, random_state=42):
    """
    Unit V: SVM with hyperparameter tuning.
    Soft Margin SVM — C controls misclassification penalty.
    """
    print(f"\n⚡ Training SVM (kernel={kernel})...")

    if tune and kernel in ('rbf', 'linear'):
        if kernel == 'rbf':
            param_grid = {
                'C'    : [1, 10, 50, 100, 200],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            }
        else:
            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

        base = SVC(kernel=kernel, probability=True,
                   class_weight='balanced', random_state=random_state)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(
            base, param_grid, cv=cv,
            scoring='f1_weighted', n_jobs=-1, verbose=0
        )
        search.fit(X_train, y_train)
        svm = search.best_estimator_
        print(f"  Best params : {search.best_params_}")
        print(f"  CV F1 score : {search.best_score_:.4f}")
    else:
        svm = SVC(
            kernel=kernel, C=10, gamma='scale',
            probability=True, class_weight='balanced',
            random_state=random_state
        )
        svm.fit(X_train, y_train)

    print(f"  ✅ SVM ({kernel}) trained")
    return svm


def train_all_svm_kernels(X_train, y_train):
    """Train SVM with Linear, RBF, and Polynomial kernels."""
    models = {}
    models['SVM (Linear)']     = train_svm(X_train, y_train, kernel='linear', tune=True)
    models['SVM (RBF)']        = train_svm(X_train, y_train, kernel='rbf',    tune=True)

    print(f"\n⚡ Training SVM (kernel=poly)...")
    svm_poly = SVC(kernel='poly', degree=3, C=10, gamma='scale',
                   probability=True, class_weight='balanced', random_state=42)
    svm_poly.fit(X_train, y_train)
    models['SVM (Polynomial)'] = svm_poly
    print(f"  ✅ SVM (Polynomial) trained")

    return models


# ─────────────────────────────────────────────
# VOTING ENSEMBLE — RF + SVM(RBF) + SVM(Linear)
# ─────────────────────────────────────────────

def train_voting_ensemble(X_train, y_train):
    """
    Soft Voting Ensemble: RF + SVM(RBF) + SVM(Linear).
    - RF        : Bagging with feature subsampling
    - SVM(RBF)  : Non-linear decision boundaries
    - SVM(Lin)  : Linear decision boundary
    Soft voting averages predicted probabilities.
    Evaluated on the OASIS longitudinal dataset.
    """
    print("\n🗳️  Training Voting Ensemble (RF + SVM-RBF + SVM-Linear)...")

    rf = RandomForestClassifier(
        n_estimators=700, max_depth=30, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    svm_rbf = SVC(
        kernel='rbf', C=100, gamma=0.1,
        probability=True, class_weight='balanced', random_state=42
    )
    svm_lin = SVC(
        kernel='linear', C=0.01,
        probability=True, class_weight='balanced', random_state=42
    )

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('svm_rbf', svm_rbf), ('svm_lin', svm_lin)],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    print(f"  ✅ Voting Ensemble trained (RF + SVM-RBF + SVM-Linear)")
    return ensemble


# ─────────────────────────────────────────────
# K-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────

def cross_validate_model(model, X, y, n_splits=5, model_name='Model'):
    """
    Unit II: Stratified K-Fold Cross Validation.
    Reports accuracy AND F1 for clinical reliability.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
    f1_scores  = cross_val_score(model, X, y, cv=kf, scoring='f1_weighted', n_jobs=-1)

    print(f"\n📊 {model_name} — {n_splits}-Fold Cross Validation:")
    print(f"  Accuracy  : {acc_scores.mean()*100:.2f}% ± {acc_scores.std()*100:.2f}%")
    print(f"  F1 (wtd)  : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"  Fold Accs : {[f'{s*100:.1f}%' for s in acc_scores]}")

    return acc_scores, f1_scores
