"""
Metrics computation for benchmark results.

CANONICAL IMPLEMENTATION extracted from 04_run_celltype_benchmark.py.

Handles:
- Unknown labels in test set (labels not seen during training)
- Binary vs multiclass classification
- Insufficient samples edge cases
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray | None,
    y_pred_class: np.ndarray,
    clf_classes: np.ndarray,
) -> dict[str, float]:
    """
    Compute classification metrics robust to class imbalance.

    Handles case where test set contains labels not seen during training.
    This can happen with StratifiedGroupKFold when rare classes have few donors.

    Args:
        y_true: True labels (original label space)
        y_pred_proba: Predicted probabilities (n_samples × n_classes), or None if unavailable
        y_pred_class: Predicted class labels (in classifier's label space)
        clf_classes: Classes known to the classifier

    Returns:
        Dictionary of metrics with NaN for cases where computation fails

    """
    logger = logging.getLogger("benchmark")
    label_map = {c: i for i, c in enumerate(clf_classes)}

    # CRITICAL FIX: Identify samples with labels unknown to the classifier
    known_mask = np.array([y in label_map for y in y_true])
    n_unknown = (~known_mask).sum()

    if n_unknown > 0:
        unknown_labels = set(y_true[~known_mask])
        logger.warning(
            f"Found {n_unknown} test samples with {len(unknown_labels)} unseen label(s): "
            f"{list(unknown_labels)[:5]}{'...' if len(unknown_labels) > 5 else ''}. "
            "These will be excluded from metrics."
        )

    # Check if we have enough valid samples
    n_valid = known_mask.sum()
    if n_valid < 2:
        return {
            "balanced_acc": np.nan,
            "MCC": np.nan,
            "F1_macro": np.nan,
            "log_loss": np.nan,
            "n_unknown_labels": int(n_unknown),
            "n_valid_test_samples": int(n_valid),
            "metric_error": "insufficient_valid_samples",
        }

    # Filter to only known labels
    y_true_filtered = y_true[known_mask]
    y_pred_class_filtered = y_pred_class[known_mask]
    if y_pred_proba is not None:
        y_pred_proba_filtered = y_pred_proba[known_mask]
    else:
        y_pred_proba_filtered = None

    y_true_idx = np.array([label_map[y] for y in y_true_filtered])

    # Check we still have multiple classes after filtering
    unique_true = np.unique(y_true_idx)
    if len(unique_true) < 2:
        return {
            "balanced_acc": np.nan,
            "MCC": np.nan,
            "F1_macro": np.nan,
            "log_loss": np.nan,
            "n_unknown_labels": int(n_unknown),
            "n_valid_test_samples": int(n_valid),
            "metric_error": "single_class_after_filtering",
        }

    # Base metrics (don't need probabilities)
    metrics = {
        "balanced_acc": float(
            balanced_accuracy_score(y_true_idx, y_pred_class_filtered)
        ),
        "MCC": float(matthews_corrcoef(y_true_idx, y_pred_class_filtered)),
        "F1_macro": float(
            f1_score(
                y_true_idx, y_pred_class_filtered, average="macro", zero_division=0
            )
        ),
        "n_unknown_labels": int(n_unknown),
        "n_valid_test_samples": int(n_valid),
    }

    # Probability-based metrics (only if probabilities available)
    if y_pred_proba_filtered is not None:
        try:
            metrics["log_loss"] = float(
                log_loss(
                    y_true_idx,
                    y_pred_proba_filtered,
                    labels=list(range(len(clf_classes))),
                )
            )
        except (ValueError, RuntimeWarning):
            metrics["log_loss"] = np.nan

        n_classes = len(clf_classes)
        if n_classes == 2:
            try:
                metrics["AUROC"] = float(
                    roc_auc_score(y_true_idx, y_pred_proba_filtered[:, 1])
                )
                metrics["AUPRC"] = float(
                    average_precision_score(y_true_idx, y_pred_proba_filtered[:, 1])
                )
            except (ValueError, RuntimeWarning):
                metrics["AUROC"] = np.nan
                metrics["AUPRC"] = np.nan
        else:
            try:
                metrics["AUROC_ovr"] = float(
                    roc_auc_score(
                        y_true_idx,
                        y_pred_proba_filtered,
                        multi_class="ovr",
                        average="macro",
                    )
                )
            except (ValueError, RuntimeWarning):
                metrics["AUROC_ovr"] = np.nan
    else:
        # No probabilities available
        metrics["log_loss"] = np.nan
        metrics["proba_unavailable"] = True

    return metrics


def compute_donor_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    donors: np.ndarray,
    clf_classes: np.ndarray,
) -> dict[str, float]:
    """
    Compute per-donor accuracy and aggregate metrics.

    Aggregates cell-level predictions to donor level using:
    - Per-donor accuracy (what % of each donor's cells are correct)
    - Mean and worst-case donor accuracy
    - Donor-level metrics using majority vote

    Args:
        y_true: True labels (cell-level)
        y_pred: Predicted class labels (cell-level)
        donors: Donor IDs for each cell
        clf_classes: Classes known to the classifier

    Returns:
        Dictionary of donor-level metrics

    """
    try:
        label_map = {c: i for i, c in enumerate(clf_classes)}

        df = pd.DataFrame(
            {
                "donor": donors,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

        # Filter to known labels
        df = df[df["y_true"].isin(label_map.keys())].copy()

        if len(df) < 2:
            return {"donor_metric_error": "insufficient_samples_after_filtering"}

        # Per-donor accuracy
        donor_acc = df.groupby("donor", observed=True).apply(
            lambda g: (g["y_true"] == g["y_pred"]).mean(), include_groups=False
        )

        # Majority vote aggregation
        donor_agg = (
            df.groupby("donor", observed=True)
            .agg(
                y_true_mode=(
                    "y_true",
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
                ),
                y_pred_mode=(
                    "y_pred",
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
                ),
            )
            .reset_index()
        )

        y_true_donor_idx = np.array(
            [label_map.get(y, -1) for y in donor_agg["y_true_mode"]]
        )
        y_pred_donor_idx = np.array(
            [label_map.get(y, -1) for y in donor_agg["y_pred_mode"]]
        )

        # Filter out any -1 (unknown) mappings
        valid_mask = (y_true_donor_idx >= 0) & (y_pred_donor_idx >= 0)
        n_valid = valid_mask.sum()

        if n_valid < 2:
            return {
                "donor_metric_error": "insufficient_valid_donors",
                "n_valid_donors": int(n_valid),
            }

        y_true_donor_idx = y_true_donor_idx[valid_mask]
        y_pred_donor_idx = y_pred_donor_idx[valid_mask]

        return {
            "balanced_acc_donor": float(
                balanced_accuracy_score(y_true_donor_idx, y_pred_donor_idx)
            ),
            "MCC_donor": float(matthews_corrcoef(y_true_donor_idx, y_pred_donor_idx)),
            "F1_macro_donor": float(
                f1_score(
                    y_true_donor_idx, y_pred_donor_idx, average="macro", zero_division=0
                )
            ),
            "mean_donor_acc": float(donor_acc.mean()),
            "worst_donor_acc": float(donor_acc.min()),
            "std_donor_acc": float(donor_acc.std()),
            "n_valid_donors": int(n_valid),
        }
    except Exception as e:
        return {"donor_metric_error": str(e)}
