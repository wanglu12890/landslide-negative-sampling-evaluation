"""Holdout metrics and simple exportable Matplotlib figures."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, balanced_accuracy_score, roc_curve)
from .rf_model import susceptibility_scores
from .output_utils import write_csv


def evaluate(model, test, feature_names: list[str], run) -> dict:
    """Evaluate the untouched test set and save auditable prediction records."""
    x, y = test[feature_names], test.label
    predicted, scores = model.predict(x), susceptibility_scores(model, x)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = dict(ROC_AUC=roc_auc_score(y, scores), Accuracy=accuracy_score(y, predicted),
        Precision=precision_score(y, predicted, zero_division=0), Recall=recall_score(y, predicted, zero_division=0),
        F1=f1_score(y, predicted, zero_division=0), Specificity=float(tn/(tn+fp)),
        Balanced_Accuracy=balanced_accuracy_score(y, predicted))
    write_csv(pd.DataFrame(metrics.items(), columns=["metric", "value"]), run / "metrics/model_metrics.csv")
    pd.DataFrame(matrix, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]).to_csv(
        run / "metrics/confusion_matrix.csv", encoding="utf-8-sig", index_label="actual_class")
    write_csv(test[["sample_id", "label"]].assign(predicted_label=predicted, susceptibility_score=scores),
              run / "metrics/test_predictions.csv")
    fpr, tpr, thresholds = roc_curve(y, scores)
    write_csv(pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=thresholds)), run / "metrics/roc_curve_points.csv")
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.plot(fpr, tpr, label=f"RF AUC = {metrics['ROC_AUC']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1), title="Test-set ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(run / "figures/roc_curve.png", dpi=300, facecolor="white")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"],
           xlabel="Predicted class", ylabel="Actual class", title="Test-set confusion matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{[['TN', 'FP'], ['FN', 'TP']][row][col]}\n{matrix[row, col]}", ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max()/2 else "black")
    fig.tight_layout()
    fig.savefig(run / "figures/confusion_matrix.png", dpi=300, facecolor="white")
    plt.close(fig)
    importance = pd.DataFrame(dict(factor=feature_names, importance=model.feature_importances_)).sort_values(
        "importance", ascending=False, kind="stable").reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance)+1))
    write_csv(importance, run / "metrics/feature_importance.csv")
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.barh(importance.factor.iloc[::-1], importance.importance.iloc[::-1])
    ax.set(xlabel="Impurity-based importance", title="Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(run / "figures/feature_importance.png", dpi=300, facecolor="white")
    plt.close(fig)
    return {**metrics, "confusion_matrix": dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)),
            "feature_importance": importance.to_dict("records")}
