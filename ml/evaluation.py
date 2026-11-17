"""
Module 4 - Evaluation Metrics
Covers CI2018 Unit III: Confusion Matrix, Accuracy, Precision, Recall, F1, ROC/AUC
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os


def evaluate_model(model, X_test, y_test, model_name='Model', class_names=None):
    """
    Unit III: Full evaluation suite.
    Returns dict of all metrics.
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*50}")
    print(f"📈 {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    target_names = [str(c) for c in class_names] if class_names else None
    print(f"\n📑 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    return {
        'model'    : model_name,
        'accuracy' : acc,
        'precision': prec,
        'recall'   : rec,
        'f1'       : f1,
        'y_pred'   : y_pred
    }


def plot_confusion_matrix(y_test, y_pred, class_names, model_name,
                          save_path='outputs/confusion_matrix.png'):
    """
    Unit III: Confusion Matrix visualization.
    Shows where the model confuses classes.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor='gray'
    )
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved → {save_path}")


def plot_roc_curve(model, X_test, y_test, class_names,
                   model_name='Model', save_path='outputs/roc_curve.png'):
    """
    Unit III: ROC Curve with AUC.
    Handles both binary and multi-class (One-vs-Rest) automatically.
    """
    n_classes  = len(class_names)
    y_prob     = model.predict_proba(X_test)
    y_test_arr = np.array(y_test)

    plt.figure(figsize=(9, 7))
    colors = cycle(['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'])

    if n_classes == 2:
        # Binary ROC — use positive class probability
        fpr_v, tpr_v, _ = roc_curve(y_test_arr, y_prob[:, 1])
        roc_auc_v = auc(fpr_v, tpr_v)
        plt.plot(fpr_v, tpr_v, color='#3498db', lw=2,
                 label=f'{class_names[1]} (AUC = {roc_auc_v:.3f})')
    else:
        # Multi-class One-vs-Rest
        y_test_bin = label_binarize(y_test_arr, classes=list(range(n_classes)))
        for i, color in zip(range(n_classes), colors):
            fpr_v, tpr_v, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc_v = auc(fpr_v, tpr_v)
            plt.plot(fpr_v, tpr_v, color=color, lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc_v:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve — {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ ROC curve saved → {save_path}")


def plot_accuracy_comparison(results_list, save_path='outputs/model_comparison.png'):
    """
    Compare all models side-by-side on Accuracy, Precision, Recall, F1.
    """
    models   = [r['model']     for r in results_list]
    accuracy = [r['accuracy']  for r in results_list]
    precision= [r['precision'] for r in results_list]
    recall   = [r['recall']    for r in results_list]
    f1       = [r['f1']        for r in results_list]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - 1.5*width, accuracy,  width, label='Accuracy',  color='#3498db', alpha=0.85)
    bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#2ecc71', alpha=0.85)
    bars3 = ax.bar(x + 0.5*width, recall,    width, label='Recall',    color='#e74c3c', alpha=0.85)
    bars4 = ax.bar(x + 1.5*width, f1,        width, label='F1-Score',  color='#f39c12', alpha=0.85)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Model comparison chart saved → {save_path}")


def print_summary_table(results_list):
    """Print a clean summary table of all model results."""
    print(f"\n{'='*70}")
    print(f"{'MODEL':<25} {'ACCURACY':>10} {'PRECISION':>10} {'RECALL':>10} {'F1':>10}")
    print(f"{'─'*70}")
    for r in results_list:
        print(f"{r['model']:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1']:>10.4f}")
    print(f"{'='*70}")
