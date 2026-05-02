"""
Module 5 - Feature Importance & Explainability
Covers CI2018 Unit V: Random Forest feature importance
Project requirement: Interpretability — "ML explains risk"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd


def get_feature_importance(rf_model, feature_names):
    """
    Extract feature importances from trained Random Forest.
    Uses Mean Decrease in Impurity (Gini importance).
    Higher value = more important feature for prediction.
    """
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort descending

    print("\n🔍 Feature Importance Ranking:")
    print(f"  {'Rank':<5} {'Feature':<30} {'Importance':>12}")
    print(f"  {'─'*50}")
    for rank, idx in enumerate(indices):
        print(f"  {rank+1:<5} {feature_names[idx]:<30} {importances[idx]:>12.4f}")

    return importances, indices


def plot_feature_importance(rf_model, feature_names,
                            top_n=15, save_path='outputs/feature_importance.png'):
    """
    Bar chart: Feature vs Importance (top N features).
    Horizontal bar for readability with many features.
    """
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Take top N
    top_indices = indices[:top_n]
    top_names   = [feature_names[i] for i in top_indices]
    top_values  = importances[top_indices]

    # Reverse for horizontal bar (most important at top)
    top_names  = top_names[::-1]
    top_values = top_values[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_names)))
    bars = ax.barh(top_names, top_values, color=colors, edgecolor='gray',
                   linewidth=0.5, height=0.7)

    # Add value labels
    for bar, val in zip(bars, top_values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left', fontsize=9)

    ax.set_xlabel('Importance Score (Mean Decrease in Impurity)', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importances — Random Forest', fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(top_values) * 1.18)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Feature importance plot saved → {save_path}")


def get_risk_explanation(rf_model, X_sample, feature_names, class_names):
    """
    For a single patient sample, explain the prediction:
    - Show predicted class + probability
    - Show top contributing features (based on global importance)
    This is the 'Interpretability' core of the project.
    """
    proba = rf_model.predict_proba(X_sample.reshape(1, -1))[0]
    pred_class = np.argmax(proba)
    confidence = proba[pred_class]

    importances = rf_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]

    print(f"\n🧠 Risk Explanation for Sample:")
    print(f"  Predicted Class : {class_names[pred_class]}")
    print(f"  Confidence      : {confidence*100:.1f}%")
    print(f"\n  Class Probabilities:")
    for i, (cls, prob) in enumerate(zip(class_names, proba)):
        bar = '█' * int(prob * 30)
        print(f"    {cls:<25} {bar:<30} {prob*100:.1f}%")

    print(f"\n  Top Contributing Features:")
    for rank, idx in enumerate(top_idx):
        val = X_sample[idx]
        imp = importances[idx]
        print(f"    {rank+1}. {feature_names[idx]:<25} value={val:.3f}  importance={imp:.4f}")

    return pred_class, confidence, proba
