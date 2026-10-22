"""
evaluation.py — Model evaluation utilities for the cyclone prediction system.

Functions
---------
evaluate_model(y_true, y_prob, title)
    Print a comprehensive set of classification metrics.
plot_feature_importance(model_pipeline, feature_names, title)
    Horizontal bar chart of feature importances from a sklearn Pipeline.
plot_training_history(history)
    Line chart of Keras training vs. validation loss.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for evaluation utilities.") from exc


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_prob, title: str = "Model") -> None:
    """
    Print classification metrics for a binary probabilistic classifier.

    Handles the degenerate case where *y_true* contains only one class by
    skipping ROC-AUC (which is undefined) and printing a warning instead of
    raising an exception.

    Parameters
    ----------
    y_true : array-like of int (0/1)
        Ground-truth binary labels.
    y_prob : array-like of float
        Predicted probabilities for the positive class.
    title : str
        Heading printed before the metrics block.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    n_classes = len(np.unique(y_true))

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"\nClass distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))

    if n_classes < 2:
        print("\n[WARNING] Only one class present in y_true — ROC-AUC is undefined, skipping.")
    else:
        roc = roc_auc_score(y_true, y_prob)
        print(f"\nROC-AUC  : {roc:.4f}")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
        print(f"PR-AUC   : {pr_auc:.4f}")
    except Exception as exc:
        print(f"[WARNING] PR-AUC could not be computed: {exc}")

    try:
        brier = brier_score_loss(y_true, y_prob)
        print(f"Brier    : {brier:.4f}")
    except Exception as exc:
        print(f"[WARNING] Brier score could not be computed: {exc}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_feature_importance(model_pipeline, feature_names: list[str], title: str = "Feature Importance") -> None:
    """
    Plot a horizontal bar chart of feature importances from a sklearn Pipeline.

    The Pipeline is expected to have a step named ``'classifier'`` that exposes
    a ``feature_importances_`` attribute (e.g. RandomForestClassifier,
    XGBClassifier).

    Parameters
    ----------
    model_pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline containing at least a ``'classifier'`` step.
    feature_names : list of str
        Names corresponding to the input features (before imputation/scaling).
    title : str
        Plot title.
    """
    try:
        clf = model_pipeline.named_steps["classifier"]
        importances = clf.feature_importances_
    except (AttributeError, KeyError) as exc:
        warnings.warn(f"Could not extract feature importances: {exc}")
        return

    n = len(importances)
    names = list(feature_names)[:n]
    if len(names) < n:
        names += [f"feature_{i}" for i in range(len(names), n)]

    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.35)))
    ax.barh(np.array(names)[order], importances[order], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    plt.show()


def plot_training_history(history) -> None:
    """
    Plot training vs. validation loss from a Keras History object.

    Parameters
    ----------
    history : keras.callbacks.History
        Returned by ``model.fit()``.
    """
    hist = history.history if hasattr(history, "history") else history

    train_loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    if not train_loss:
        print("[WARNING] No loss data found in training history.")
        return

    epochs = range(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Training loss", marker="o", markersize=3)
    if val_loss:
        ax.plot(epochs, val_loss, label="Validation loss", marker="s", markersize=3, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    plt.tight_layout()
    plt.show()
