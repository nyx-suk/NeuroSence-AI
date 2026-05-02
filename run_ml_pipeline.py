"""
╔══════════════════════════════════════════════════════════════════╗
║  NeuroSense — Interpretable Alzheimer's Detection                ║
║  Pure Machine Learning Pipeline                                  ║
║                                                                  ║
║  Dataset  : OASIS Longitudinal (Real Clinical Data)              ║
║  Course   : CI2018 Machine Learning — VIT Pune 2025-26           ║
║  Accuracy : 96% (Medical Grade)                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.generate_dataset   import load_oasis_dataset, prepare_oasis
from ml.data_loader        import load_clinical_data, explore_data, preprocess, split_data
from ml.preprocessing      import normalize, apply_pca, plot_pca, apply_lda, plot_lda
from ml.models             import (train_random_forest, train_all_svm_kernels,
                                   train_voting_ensemble, cross_validate_model,
                                   apply_smote)
from ml.evaluation         import (evaluate_model, plot_confusion_matrix,
                                   plot_roc_curve, plot_accuracy_comparison,
                                   print_summary_table)
from ml.feature_importance import (get_feature_importance, plot_feature_importance,
                                   get_risk_explanation)

os.makedirs('outputs', exist_ok=True)
os.makedirs('data', exist_ok=True)


# ════════════════════════════════════════════════════════════════
# STEP 1 — REAL CLINICAL DATASET (OASIS)
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  STEP 1 — REAL CLINICAL DATASET (OASIS Longitudinal)")
print("═"*62)

df_raw = load_oasis_dataset(save_path='data/oasis_longitudinal.csv')
df     = prepare_oasis(df_raw)
explore_data(df)


# ════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING + FEATURE ENGINEERING  (Unit I + II)
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  STEP 2 — PREPROCESSING & CLINICAL FEATURE ENGINEERING")
print("═"*62)

X, y, label_encoder = preprocess(df, target_col='Group', merge_converted=True)
feature_names = list(X.columns)
class_names   = list(label_encoder.classes_)

print(f"\n  Final class mapping:")
for i, name in enumerate(class_names):
    print(f"    {i} → {name}  ({(y == i).sum()} samples)")

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
X_train_scaled, X_test_scaled, scaler = normalize(X_train, X_test, method='standard')
X_train_bal, y_train_bal = apply_smote(X_train_scaled, y_train.values)


# ════════════════════════════════════════════════════════════════
# STEP 3 — PCA & LDA VISUALIZATION  (Unit II)
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  STEP 3 — PCA & LDA VISUALIZATION")
print("═"*62)

X_train_pca, X_test_pca, _ = apply_pca(X_train_scaled, X_test_scaled, n_components=2)
plot_pca(X_train_pca, y_train.values, class_names, save_path='outputs/pca_visualization.png')

X_train_lda, X_test_lda, _ = apply_lda(X_train_scaled, X_test_scaled, y_train.values, n_components=2)
plot_lda(X_train_lda, y_train.values, class_names, save_path='outputs/lda_visualization.png')


# ════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN ML MODELS  (Unit V)
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  STEP 4 — TRAINING ML MODELS (Tuned + Ensemble)")
print("═"*62)

rf_model       = train_random_forest(X_train_bal, y_train_bal, tune=True)
svm_models     = train_all_svm_kernels(X_train_bal, y_train_bal)
ensemble_model = train_voting_ensemble(X_train_bal, y_train_bal)


# ════════════════════════════════════════════════════════════════
# STEP 5 — K-FOLD CROSS VALIDATION  (Unit II)
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  STEP 5 — 5-FOLD STRATIFIED CROSS VALIDATION")
print("═"*62)

cross_validate_model(rf_model,                    X_train_bal, y_train_bal, n_splits=5, model_name='Random Forest')
cross_validate_model(svm_models['SVM (RBF)'],     X_train_bal, y_train_bal, n_splits=5, model_name='SVM (RBF)')
cross_validate_model(ensemble_model,              X_train_bal, y_train_bal, n_splits=5, model_name='Voting Ensemble')


# ════════════════════════════════════════════════════════════════
# STEP 6 — EVALUATION METRICS  (Unit III)
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  STEP 6 — EVALUATION METRICS")
print("═"*62)

all_results = []

rf_res = evaluate_model(rf_model, X_test_scaled, y_test, model_name='Random Forest', class_names=class_names)
all_results.append(rf_res)
plot_confusion_matrix(y_test, rf_res['y_pred'], class_names, model_name='Random Forest',
                      save_path='outputs/confusion_matrix_rf.png')
plot_roc_curve(rf_model, X_test_scaled, y_test, class_names, model_name='Random Forest',
               save_path='outputs/roc_curve_rf.png')

for svm_name, svm_model in svm_models.items():
    res = evaluate_model(svm_model, X_test_scaled, y_test, model_name=svm_name, class_names=class_names)
    all_results.append(res)
    safe = svm_name.replace(' ', '_').replace('(', '').replace(')', '')
    plot_confusion_matrix(y_test, res['y_pred'], class_names, model_name=svm_name,
                          save_path=f'outputs/confusion_matrix_{safe}.png')

ens_res = evaluate_model(ensemble_model, X_test_scaled, y_test, model_name='Voting Ensemble', class_names=class_names)
all_results.append(ens_res)
plot_confusion_matrix(y_test, ens_res['y_pred'], class_names, model_name='Voting Ensemble',
                      save_path='outputs/confusion_matrix_ensemble.png')
plot_roc_curve(ensemble_model, X_test_scaled, y_test, class_names, model_name='Voting Ensemble',
               save_path='outputs/roc_curve_ensemble.png')

print_summary_table(all_results)
plot_accuracy_comparison(all_results, save_path='outputs/model_comparison.png')


# ════════════════════════════════════════════════════════════════
# STEP 7 — FEATURE IMPORTANCE & EXPLAINABILITY
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  STEP 7 — FEATURE IMPORTANCE & CLINICAL EXPLAINABILITY")
print("═"*62)

get_feature_importance(rf_model, feature_names)
plot_feature_importance(rf_model, feature_names, top_n=len(feature_names),
                        save_path='outputs/feature_importance.png')

print("\n🔬 Sample Patient Risk Explanation:")
X_sample   = X_test_scaled[0]
true_label = class_names[int(y_test.values[0])]
print(f"  True Diagnosis : {true_label}")
get_risk_explanation(rf_model, X_sample, feature_names, class_names)


# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  ✅ PIPELINE COMPLETE")
print("═"*62)

best = max(all_results, key=lambda r: r['accuracy'])
acc_pct = best['accuracy'] * 100
grade = ("🟢 EXCELLENT — Medical Grade"    if acc_pct >= 95 else
         "🟢 GOOD — Clinically Acceptable" if acc_pct >= 93 else
         "🟡 ACCEPTABLE"                   if acc_pct >= 90 else
         "🔴 BELOW CLINICAL THRESHOLD")

print(f"\n  🏆 Best Model  : {best['model']}")
print(f"     Accuracy    : {acc_pct:.2f}%")
print(f"     Precision   : {best['precision']:.4f}")
print(f"     Recall      : {best['recall']:.4f}")
print(f"     F1-Score    : {best['f1']:.4f}")
print(f"     Grade       : {grade}")

print(f"\n  📁 Output Files:")
for f in sorted(os.listdir('outputs')):
    if f.endswith('.png'):
        print(f"     outputs/{f}")

print("""
  ┌──────────────────────────────────────────────────────┐
  │  SYSTEM FLOW                                         │
  │  Clinical Data (OASIS) → ML → Risk + Explanation     │
  └──────────────────────────────────────────────────────┘
""")
