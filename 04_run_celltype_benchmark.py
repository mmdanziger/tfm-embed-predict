#!/usr/bin/env python3
"""
Benchmark transcriptomic foundation models vs raw baselines for cell type prediction.

Task: dataset_id → predict cell_type (across held-out donors)
Evaluation: Donor-stratified K-fold cross-validation (stratified on disease)

Features compared:
- raw_lognorm: Library-size normalized log1p counts (sparse)
- raw_pca50: PCA reduction of lognorm data
- random_proj50: Random projection baseline (sanity check)
- obsm[*]: Foundation model embeddings (scVI, Geneformer, etc.)

This benchmark tests models on their intended task (cell type prediction),
complementing the disease benchmark which tests generalization.
"""

import argparse
import concurrent.futures
import gc
import glob
import logging
import os
import shutil
import threading
import time
import traceback
import uuid
import warnings
from pathlib import Path

import cellxgene_census
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from tqdm.auto import tqdm

# =============================================================================
# Global semaphore to limit concurrent Census operations
# TileDB creates contexts internally when fetching embeddings, causing exhaustion
# =============================================================================
CENSUS_SEMAPHORE = threading.Semaphore(2)


def setup_logging(log_file: str) -> logging.Logger:
    """Configures thread-safe logger for file output and captures warnings."""
    logging.captureWarnings(True)

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(threadName)-15s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    # Console handler for warnings/errors
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.handlers = []
    warnings_logger.propagate = False
    warnings_logger.addHandler(fh)

    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="y_pred contains classes not in y_true"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="Transforming to str index"
    )

    return logger


def log_resources(logger: logging.Logger, message: str) -> None:
    """Logs current resource usage."""
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        logger.info(
            f"{message} | RSS={mem_info.rss / 1e9:.2f}GB | CPU={cpu_percent:.1f}%"
        )
    except ImportError:
        logger.info(message)


def sparse_normalize_log1p(
    X: sparse.csr_matrix, target_sum: float = 1e4
) -> sparse.csr_matrix:
    """Efficiently normalizes library size and applies log1p to sparse CSR matrix."""
    X = X.tocsr().copy()
    counts = np.array(X.sum(axis=1)).flatten()
    counts[counts == 0] = 1
    scale_factors = target_sum / counts
    scale_mat = sparse.diags(scale_factors)
    X = scale_mat @ X
    X.data = np.log1p(X.data)
    return X


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_class: np.ndarray,
    clf_classes: np.ndarray,
) -> dict[str, float]:
    """
    Computes classification metrics robust to class imbalance.

    FIXED: Handles case where test set contains labels not seen during training.
    This can happen with StratifiedGroupKFold when rare classes have few donors.
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
            f"These will be excluded from metrics."
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
    y_pred_proba_filtered = y_pred_proba[known_mask]
    y_pred_class_filtered = y_pred_class[known_mask]

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
        "log_loss": float(
            log_loss(
                y_true_idx, y_pred_proba_filtered, labels=list(range(len(clf_classes)))
            )
        ),
        "n_unknown_labels": int(n_unknown),
        "n_valid_test_samples": int(n_valid),
    }

    n_classes = len(clf_classes)
    if n_classes == 2:
        metrics["AUROC"] = float(roc_auc_score(y_true_idx, y_pred_proba_filtered[:, 1]))
        metrics["AUPRC"] = float(
            average_precision_score(y_true_idx, y_pred_proba_filtered[:, 1])
        )
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
        except ValueError:
            metrics["AUROC_ovr"] = np.nan

    return metrics


def compute_donor_level_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    donors: np.ndarray,
    clf_classes: np.ndarray,
) -> dict[str, float]:
    """
    Computes per-donor accuracy and aggregates.

    For cell type prediction, we compute:
    - Per-donor accuracy (what % of each donor's cells are correct)
    - Mean and worst-case donor accuracy
    - Standard donor-level metrics (majority vote)

    FIXED: Handles unseen labels in test set gracefully.
    """
    try:
        label_map = {c: i for i, c in enumerate(clf_classes)}

        df = pd.DataFrame(
            {
                "donor": donors,
                "y_true": y_true,
                "y_pred": clf_classes[np.argmax(y_pred_proba, axis=1)],
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


def validate_task_feasibility(
    y: np.ndarray,
    donors: np.ndarray,
    n_splits: int,
) -> tuple[bool, str, dict]:
    """
    Validates that a task can be successfully cross-validated.

    Returns:
        (is_feasible, reason, stats)

    """
    # Count donors per class
    donor_class = {}
    for d, label in zip(donors, y):
        if d not in donor_class:
            donor_class[d] = set()
        donor_class[d].add(label)

    # For cell type prediction, donors can have multiple cell types
    # Count unique donors
    n_donors = len(donor_class)
    n_classes = len(np.unique(y))

    stats = {
        "n_classes": n_classes,
        "n_donors": n_donors,
    }

    # Check total donors
    if n_donors < 2:
        return False, "total_donors_lt2", stats

    # Check total classes
    if n_classes < 2:
        return False, "total_classes_lt2", stats

    return True, "ok", stats


def benchmark_one_task(
    task_row: pd.Series,
    census: object,
    embedding_keys: list[str],
    n_splits: int = 5,
    random_state: int = 0,
    target_sum: float = 1e4,
    pca_components: int = 50,
    clf_n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Runs cell type prediction benchmark on a single dataset.

    Task: predict cell_type across held-out donors (stratified on disease).
    Uses only cell types that participated in the disease benchmark.

    Args:
        task_row: Row from cell type manifest (one per dataset)
        census: Shared census connection
        embedding_keys: List of obsm keys to evaluate
        n_splits: Number of CV folds
        random_state: Random seed
        target_sum: Target library size for normalization
        pca_components: Number of PCA components
        clf_n_jobs: Number of threads for classifier

    Returns:
        DataFrame with results (never empty on error/skip)

    """
    dataset_id = task_row["dataset_id"]
    eligible_cell_types = task_row["cell_type"]  # List from manifest
    task_id = f"dataset::{dataset_id}"
    logger = logging.getLogger("benchmark")

    # Context for error reporting - captured early for debugging
    task_context = {
        "n_cells": None,
        "n_donors": None,
        "n_cell_types_requested": len(eligible_cell_types)
        if isinstance(eligible_cell_types, list)
        else None,
        "n_cell_types_found": None,
    }

    # Track variables for cleanup in finally block
    adata = None
    X_lognorm = None
    X_raw = None
    embeddings = None

    try:
        logger.info(
            f"[START] {task_id} ({len(eligible_cell_types)} cell types requested)"
        )
        log_resources(logger, f"[{task_id}] Pre-fetch")

        # Build cell type filter - escape single quotes in cell type names
        escaped_cell_types = [ct.replace("'", "\\'") for ct in eligible_cell_types]
        cell_type_filter = (
            "(" + " or ".join([f"cell_type=='{ct}'" for ct in escaped_cell_types]) + ")"
        )
        obs_filter = f"is_primary_data==True and dataset_id=='{dataset_id}' and {cell_type_filter}"

        # Use semaphore to limit concurrent Census operations
        # This prevents TileDB context exhaustion when fetching embeddings
        with CENSUS_SEMAPHORE:
            adata = cellxgene_census.get_anndata(
                census=census,
                organism="homo_sapiens",
                measurement_name="RNA",
                obs_value_filter=obs_filter,
                obs_column_names=["dataset_id", "cell_type", "donor_id", "disease"],
                obs_embeddings=embedding_keys,
            )

        log_resources(logger, f"[{task_id}] Post-fetch")
        logger.info(f"[DATA LOADED] {task_id} ({adata.n_obs} cells)")

        # Update context for error reporting
        task_context["n_cells"] = adata.n_obs

        if adata.n_obs == 0:
            logger.warning(f"[SKIP] {task_id}: No cells returned")
            return pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": "no_cells",
                        **task_context,
                    }
                ]
            )

        # Validate and convert data types
        adata.obs["disease"] = adata.obs["disease"].astype(str)
        adata.obs["donor_id"] = adata.obs["donor_id"].astype(str)
        adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)

        # Extract arrays for processing
        y = adata.obs["cell_type"].values
        donors = adata.obs["donor_id"].values
        diseases = adata.obs["disease"].values  # Used for stratification

        # Update context
        task_context["n_donors"] = len(np.unique(donors))
        task_context["n_cell_types_found"] = len(np.unique(y))
        task_context["n_diseases"] = len(np.unique(diseases))

        # Validate task feasibility
        is_feasible, reason, stats = validate_task_feasibility(y, donors, n_splits)
        if not is_feasible:
            logger.info(f"[SKIP] {task_id}: {reason}")
            return pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": reason,
                        **task_context,
                        **stats,
                    }
                ]
            )

        # Check minimum cell types
        n_cell_types = len(np.unique(y))
        if n_cell_types < 2:
            logger.info(f"[SKIP] {task_id}: Only {n_cell_types} cell type(s)")
            return pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": "too_few_celltypes",
                        **task_context,
                    }
                ]
            )

        # Determine number of folds
        n_donors = len(np.unique(donors))
        k_folds = min(n_splits, n_donors)

        if k_folds < 2:
            logger.info(f"[SKIP] {task_id}: Only {n_donors} donor(s)")
            return pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": "too_few_donors",
                        **task_context,
                    }
                ]
            )

        # Extract embeddings early (fail fast if missing)
        embeddings = {}
        for k in embedding_keys:
            if k not in adata.obsm:
                raise KeyError(f"Missing required embedding: {k}")
            embeddings[k] = np.asarray(adata.obsm[k])
            logger.info(
                f"[{task_id}] Loaded embedding {k}: shape {embeddings[k].shape}"
            )

        # Prepare raw expression matrix
        if not sparse.issparse(adata.X):
            adata.X = sparse.csr_matrix(adata.X)
        else:
            adata.X = adata.X.tocsr()

        # Filter genes at dataset level (remove zero-expression genes)
        gene_totals = np.array(adata.X.sum(axis=0)).flatten()
        genes_mask = gene_totals > 0
        X_raw = adata.X[:, genes_mask]
        n_genes_kept = genes_mask.sum()

        logger.info(
            f"[{task_id}] Kept {n_genes_kept}/{len(genes_mask)} genes with non-zero expression"
        )

        # Apply library size normalization and log1p
        X_lognorm = sparse_normalize_log1p(X_raw, target_sum)

        # Free raw X early to reduce memory pressure
        del X_raw
        adata.X = None

        log_resources(logger, f"[{task_id}] Pre-processing complete")

        # Cross-validation setup
        # Stratify on DISEASE to ensure diseased donors appear in each fold
        # Group by DONOR to prevent data leakage
        sgkf = StratifiedGroupKFold(
            n_splits=k_folds, shuffle=True, random_state=random_state
        )

        results = []
        skipped_folds = 0
        nan_predictions_count = 0

        for fold_i, (train_idx, test_idx) in enumerate(
            sgkf.split(X_lognorm, diseases, groups=donors)
        ):
            # Extract fold data
            y_train, y_test = y[train_idx], y[test_idx]
            donors_train, donors_test = donors[train_idx], donors[test_idx]
            diseases_test = diseases[test_idx]

            # Count classes in each split
            n_train_classes = len(np.unique(y_train))
            n_test_classes = len(np.unique(y_test))

            logger.info(
                f"[FOLD {fold_i}] {task_id} | "
                f"train={len(y_train)} cells, {len(np.unique(donors_train))} donors, {n_train_classes} classes | "
                f"test={len(y_test)} cells, {len(np.unique(donors_test))} donors, {n_test_classes} classes"
            )

            # Skip degenerate folds
            if n_train_classes < 2:
                logger.info(
                    f"[SKIP FOLD] {task_id} fold {fold_i}: only {n_train_classes} train class(es)"
                )
                skipped_folds += 1
                continue

            if n_test_classes < 2:
                logger.info(
                    f"[SKIP FOLD] {task_id} fold {fold_i}: only {n_test_classes} test class(es)"
                )
                skipped_folds += 1
                continue

            # Split expression data
            X_train_ln = X_lognorm[train_idx]
            X_test_ln = X_lognorm[test_idx]

            # =================================================================
            # Build feature sets to compare
            # =================================================================
            features_to_test = []

            # Feature Set 1: Raw Lognorm (sparse, no additional scaling)
            # This is our simplest baseline - just normalized counts
            features_to_test.append(("raw_lognorm", X_train_ln, X_test_ln))

            # Feature Set 2: PCA on Lognorm -> StandardScaled
            # Captures major axes of variation, reduces dimensionality
            n_pca = min(
                pca_components, X_train_ln.shape[1] - 1, X_train_ln.shape[0] - 1
            )
            if n_pca >= 2:
                pca = TruncatedSVD(n_components=n_pca, random_state=random_state)
                X_train_pca = pca.fit_transform(X_train_ln)
                X_test_pca = pca.transform(X_test_ln)

                scaler_pca = StandardScaler()
                X_train_pca_scaled = scaler_pca.fit_transform(X_train_pca)
                X_test_pca_scaled = scaler_pca.transform(X_test_pca)
                features_to_test.append(
                    ("raw_pca50", X_train_pca_scaled, X_test_pca_scaled)
                )
            else:
                logger.warning(f"[{task_id}] Skipping PCA: n_pca={n_pca} < 2")

            # Feature Set 3: Random Projection baseline (sanity check)
            # If this performs well, something is wrong with our setup
            rp = SparseRandomProjection(
                n_components=pca_components,
                dense_output=True,
                random_state=random_state,
            )
            # Fit on minimal data (random projection doesn't learn from data)
            rp.fit(X_train_ln[:1].astype(np.float64))
            X_train_rp = rp.transform(X_train_ln.astype(np.float64))
            X_test_rp = rp.transform(X_test_ln.astype(np.float64))

            scaler_rp = StandardScaler()
            X_train_rp_scaled = scaler_rp.fit_transform(X_train_rp)
            X_test_rp_scaled = scaler_rp.transform(X_test_rp)
            features_to_test.append(
                ("random_proj50", X_train_rp_scaled, X_test_rp_scaled)
            )

            # Feature Sets 4+: Precomputed Embeddings (scaled)
            # These are the foundation model representations we're evaluating
            for emb_name, emb_matrix in embeddings.items():
                X_train_emb = emb_matrix[train_idx]
                X_test_emb = emb_matrix[test_idx]

                # Always scale embeddings for fair comparison
                scaler_emb = StandardScaler()
                X_train_emb_scaled = scaler_emb.fit_transform(X_train_emb)
                X_test_emb_scaled = scaler_emb.transform(X_test_emb)

                features_to_test.append(
                    (f"obsm[{emb_name}]", X_train_emb_scaled, X_test_emb_scaled)
                )

            # =================================================================
            # Train and evaluate each feature set
            # =================================================================
            for feat_name, X_tr, X_te in features_to_test:
                t0 = time.perf_counter()

                # SGD Logistic Regression with balanced class weights
                clf = SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-5,
                    class_weight="balanced",  # Critical for imbalanced data
                    max_iter=2000,
                    tol=1e-3,
                    early_stopping=True,
                    n_iter_no_change=5,
                    average=True,  # Use averaged weights for stability
                    n_jobs=clf_n_jobs,
                    random_state=random_state,
                )
                clf.fit(X_tr, y_train)

                # Get probability predictions
                y_pred_proba = clf.predict_proba(X_te)

                # Handle numerical instability in predict_proba
                nan_mask = ~np.isfinite(y_pred_proba).all(axis=1)
                if nan_mask.any():
                    nan_predictions_count += nan_mask.sum()
                    # Fall back to hard predictions for unstable samples
                    y_pred_fallback = clf.predict(X_te[nan_mask])
                    for i, pred_class in enumerate(y_pred_fallback):
                        class_idx = np.where(clf.classes_ == pred_class)[0][0]
                        row_idx = np.where(nan_mask)[0][i]
                        y_pred_proba[row_idx, :] = 0.0
                        y_pred_proba[row_idx, class_idx] = 1.0

                y_pred_class = np.argmax(y_pred_proba, axis=1)

                # Compute metrics (with label mismatch handling)
                metrics = compute_metrics(
                    y_test, y_pred_proba, y_pred_class, clf.classes_
                )
                donor_metrics = compute_donor_level_metrics(
                    y_test, y_pred_proba, donors[test_idx], clf.classes_
                )

                # Record results
                results.append(
                    {
                        "dataset_id": dataset_id,
                        "fold": fold_i,
                        "feature": feat_name,
                        "n_genes": n_genes_kept,
                        "n_cell_types": n_cell_types,
                        "n_train_cells": len(y_train),
                        "n_test_cells": len(y_test),
                        "n_train_donors": len(np.unique(donors_train)),
                        "n_test_donors": len(np.unique(donors_test)),
                        "n_train_classes": n_train_classes,
                        "n_test_classes": n_test_classes,
                        "n_diseased_test_donors": len(
                            np.unique(donors_test[diseases_test != "normal"])
                        ),
                        "elapsed_s": time.perf_counter() - t0,
                        **metrics,
                        **donor_metrics,
                    }
                )

        # =================================================================
        # Post-processing and result assembly
        # =================================================================

        # Check if all folds were skipped
        if not results:
            logger.warning(f"[SKIP] {task_id}: All {k_folds} folds were degenerate")
            return pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": f"all_{k_folds}_folds_degenerate",
                        "skipped_folds": skipped_folds,
                        **task_context,
                    }
                ]
            )

        if nan_predictions_count > 0:
            logger.warning(f"[{task_id}] Fixed {nan_predictions_count} NaN predictions")

        if skipped_folds > 0:
            logger.info(
                f"[{task_id}] Completed with {skipped_folds}/{k_folds} folds skipped"
            )

        log_resources(logger, f"[{task_id}] Complete")
        logger.info(f"[SUCCESS] {task_id}: {len(results)} fold-feature combinations")

        return pd.DataFrame(results)

    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"[FAILED] {task_id}\n{error_msg}")
        return pd.DataFrame(
            [
                {
                    "dataset_id": dataset_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": error_msg,
                    **task_context,
                }
            ]
        )

    finally:
        # Explicit cleanup to prevent memory accumulation across tasks
        # ThreadPoolExecutor can hold references longer than expected
        if adata is not None:
            del adata
        if X_lognorm is not None:
            del X_lognorm
        if embeddings is not None:
            del embeddings
        gc.collect()


def get_completed_tasks(temp_dir: str, retry_errors: bool = True) -> set[str]:
    """
    Scans temp directory to identify completed tasks for checkpoint resume.

    Args:
        temp_dir: Directory containing batch shard files
        retry_errors: If True, don't count errored tasks as complete (will retry)

    Returns:
        Set of dataset_ids that are already complete

    """
    if not os.path.exists(temp_dir):
        return set()

    completed = set()
    shard_files = glob.glob(f"{temp_dir}/batch_*.parquet")

    for shard_file in shard_files:
        try:
            df = pd.read_parquet(shard_file)

            if "dataset_id" not in df.columns:
                continue

            # Check for success: has MCC metric
            has_results = (
                df["MCC"].notna()
                if "MCC" in df.columns
                else pd.Series(False, index=df.index)
            )

            # Check for permanent skip (data issues, not transient errors)
            has_skip = (
                df["skip_reason"].notna()
                if "skip_reason" in df.columns
                else pd.Series(False, index=df.index)
            )

            # Check for errors
            has_error = (
                df["error"].notna()
                if "error" in df.columns
                else pd.Series(False, index=df.index)
            )

            if retry_errors:
                # Only count successes and permanent skips (errors will be retried)
                complete_mask = has_results | has_skip
            else:
                # Count everything including errors (no retry)
                complete_mask = has_results | has_skip | has_error

            complete_df = df[complete_mask]
            # For cell type benchmark, task_id is just dataset_id
            completed.update(complete_df["dataset_id"].unique())

        except Exception:
            continue

    return completed


def inspect_progress(temp_dir: str) -> dict:
    """
    Inspects benchmark progress from temp directory.

    Can be called during a run to check status:
        python -c "from run_celltype_benchmark import inspect_progress; import json; print(json.dumps(inspect_progress('/path/to/temp'), indent=2))"
    """
    if not os.path.exists(temp_dir):
        return {"status": "temp_dir_not_found"}

    shard_files = glob.glob(f"{temp_dir}/batch_*.parquet")
    if not shard_files:
        return {"status": "no_results_yet", "temp_dir": temp_dir}

    all_dfs = []
    for f in shard_files:
        try:
            all_dfs.append(pd.read_parquet(f))
        except Exception:
            continue

    if not all_dfs:
        return {"status": "shard_read_errors", "n_shards": len(shard_files)}

    combined = pd.concat(all_dfs, ignore_index=True)

    # Count unique datasets
    n_datasets = combined["dataset_id"].nunique()

    # Count successes, failures, skips
    has_mcc = (
        combined["MCC"].notna()
        if "MCC" in combined.columns
        else pd.Series(False, index=combined.index)
    )
    has_error = (
        combined["error"].notna()
        if "error" in combined.columns
        else pd.Series(False, index=combined.index)
    )
    has_skip = (
        combined["skip_reason"].notna()
        if "skip_reason" in combined.columns
        else pd.Series(False, index=combined.index)
    )

    # Get unique features tested
    features = (
        combined["feature"].unique().tolist() if "feature" in combined.columns else []
    )

    # Performance summary
    perf_summary = {}
    if "MCC" in combined.columns and has_mcc.any():
        results_df = combined[has_mcc]
        for feat in features:
            feat_df = results_df[results_df["feature"] == feat]
            if not feat_df.empty:
                perf_summary[feat] = {
                    "mean_MCC": round(float(feat_df["MCC"].mean()), 4),
                    "median_MCC": round(float(feat_df["MCC"].median()), 4),
                    "std_MCC": round(float(feat_df["MCC"].std()), 4),
                    "n_folds": len(feat_df),
                }

    return {
        "status": "in_progress",
        "temp_dir": temp_dir,
        "n_shards": len(shard_files),
        "n_datasets_attempted": n_datasets,
        "n_success_rows": int(has_mcc.sum()),
        "n_error_rows": int(has_error.sum()),
        "n_skip_rows": int(has_skip.sum()),
        "features_tested": features,
        "performance_by_feature": perf_summary,
    }


def run_benchmarks(
    celltype_manifest_path: str,
    census_uri: str | None,
    census_version: str,
    embedding_keys: list[str],
    out_parquet: str,
    log_file: str,
    n_splits: int = 5,
    task_limit: int | None = None,
    max_workers: int = 4,
    keep_temp: bool = False,
    total_cores: int = 16,
    resume: bool = True,
    retry_errors: bool = True,
) -> None:
    """
    Runs cell type prediction benchmarks from cell type manifest.

    Supports checkpoint/resume: if temp directory exists, skips completed tasks.
    """
    logger = setup_logging(log_file)

    # Load cell type manifest
    tasks_df = pd.read_parquet(celltype_manifest_path)
    tasks_df = tasks_df.sample(frac=1, random_state=42).reset_index(
        drop=True
    )  # Shuffle with fixed seed

    if task_limit:
        tasks_df = tasks_df.head(task_limit)

    # Create temp directory for batch shards
    temp_dir = f"{os.path.dirname(out_parquet) or '.'}/_temp_shards_celltype_{Path(out_parquet).stem}"
    os.makedirs(temp_dir, exist_ok=True)

    # Calculate threads per worker
    clf_n_jobs = max(1, total_cores // max_workers)

    # Check for completed tasks if resuming
    completed_tasks = set()
    if resume:
        completed_tasks = get_completed_tasks(temp_dir, retry_errors=retry_errors)
        if completed_tasks:
            logger.info(f"RESUME: Found {len(completed_tasks)} completed datasets")
            tasks_df = tasks_df[~tasks_df["dataset_id"].isin(completed_tasks)].copy()

    logger.info("=" * 80)
    logger.info("CELL TYPE BENCHMARK RUN STARTED")
    logger.info(
        f"  Datasets remaining: {len(tasks_df)} (completed: {len(completed_tasks)})"
    )
    logger.info(
        f"  Total cell types: {tasks_df['n_cell_types'].sum() if len(tasks_df) > 0 else 0}"
    )
    logger.info(f"  Workers: {max_workers}")
    logger.info(f"  Threads per worker: {clf_n_jobs}")
    logger.info(f"  Census semaphore limit: {CENSUS_SEMAPHORE._value}")
    logger.info(f"  Census: {census_version} (uri={census_uri})")
    logger.info(f"  Embeddings: {embedding_keys}")
    logger.info(f"  Output: {out_parquet}")
    logger.info(f"  Temp Dir: {temp_dir}")
    logger.info(f"  Resume: {resume}, Retry errors: {retry_errors}")
    logger.info("=" * 80)

    if len(tasks_df) == 0:
        logger.info("No tasks to process. Merging existing results...")
    else:

        def flush_batch(buffer_list: list[pd.DataFrame]) -> None:
            """Writes buffered results to a new shard file."""
            if not buffer_list:
                return

            batch_id = str(uuid.uuid4())[:8]
            batch_path = f"{temp_dir}/batch_{batch_id}.parquet"

            combined = pd.concat(buffer_list, ignore_index=True)
            combined.to_parquet(batch_path, index=False)

            logger.info(
                f"[FLUSH] Wrote batch {batch_id}: {len(combined)} rows from {len(buffer_list)} tasks"
            )
            log_resources(logger, f"Post-flush {batch_id}")

        # Open Census connection once and share across all workers
        logger.info("Opening shared Census connection...")
        with cellxgene_census.open_soma(
            uri=census_uri, census_version=census_version
        ) as census:
            logger.info("Census connection established")

            # Execute tasks in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all tasks
                future_to_task = {}
                for idx, row in tasks_df.iterrows():
                    future = executor.submit(
                        benchmark_one_task,
                        task_row=row,
                        census=census,
                        embedding_keys=embedding_keys,
                        n_splits=n_splits,
                        clf_n_jobs=clf_n_jobs,
                    )
                    future_to_task[future] = row["dataset_id"]

                buffer = []
                n_success = 0
                n_failed = 0
                n_skipped = 0

                # Process completions with progress bar
                with tqdm(total=len(tasks_df), desc="Processing datasets") as pbar:
                    for future in concurrent.futures.as_completed(future_to_task):
                        dataset_id = future_to_task[future]

                        try:
                            result_df = future.result()
                        except Exception as e:
                            logger.error(f"[EXECUTOR ERROR] {dataset_id}: {e}")
                            result_df = pd.DataFrame(
                                [
                                    {
                                        "dataset_id": dataset_id,
                                        "error": str(e),
                                        "error_type": "executor_error",
                                    }
                                ]
                            )

                        if not result_df.empty:
                            buffer.append(result_df)

                            # Track status
                            if (
                                "error" in result_df.columns
                                and result_df["error"].notna().any()
                            ):
                                n_failed += 1
                            elif (
                                "skip_reason" in result_df.columns
                                and result_df["skip_reason"].notna().any()
                            ):
                                n_skipped += 1
                            else:
                                n_success += 1

                        # Flush every 5 completed tasks
                        if len(buffer) >= 5:
                            flush_batch(buffer)
                            buffer = []

                        pbar.update(1)
                        pbar.set_postfix(
                            success=n_success, failed=n_failed, skipped=n_skipped
                        )

                # Final flush
                if buffer:
                    flush_batch(buffer)

    # Merge all shards into final output
    logger.info("Merging shards into final output...")
    shard_files = sorted(glob.glob(f"{temp_dir}/batch_*.parquet"))

    if not shard_files:
        logger.warning("No shard files found - no results generated")
        return

    all_shards = [pd.read_parquet(f) for f in shard_files]
    final_df = pd.concat(all_shards, ignore_index=True)

    final_df.to_parquet(out_parquet, index=False)

    # Calculate final statistics
    has_mcc = (
        final_df["MCC"].notna()
        if "MCC" in final_df.columns
        else pd.Series(False, index=final_df.index)
    )
    has_error = (
        final_df["error"].notna()
        if "error" in final_df.columns
        else pd.Series(False, index=final_df.index)
    )
    has_skip = (
        final_df["skip_reason"].notna()
        if "skip_reason" in final_df.columns
        else pd.Series(False, index=final_df.index)
    )

    n_total_success = final_df[has_mcc]["dataset_id"].nunique()
    n_total_error = final_df[has_error]["dataset_id"].nunique()
    n_total_skip = final_df[has_skip]["dataset_id"].nunique()

    logger.info("=" * 80)
    logger.info("CELL TYPE BENCHMARK RUN COMPLETED")
    logger.info(f"  Success: {n_total_success} datasets")
    logger.info(f"  Failed: {n_total_error} datasets")
    logger.info(f"  Skipped: {n_total_skip} datasets")
    logger.info(f"  Final output: {out_parquet} ({final_df.shape[0]} rows)")
    logger.info("=" * 80)

    # Cleanup temp directory
    if not keep_temp:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temp directory: {temp_dir}")
    else:
        logger.info(f"Kept temp directory: {temp_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark foundation models vs baselines for cell type prediction"
    )
    ap.add_argument(
        "--celltype_manifest",
        required=True,
        help="Path to cell type task manifest parquet",
    )
    ap.add_argument("--out_parquet", required=True, help="Output parquet path")
    ap.add_argument(
        "--log_file", default="celltype_benchmark.log", help="Log file path"
    )
    ap.add_argument("--census_uri", default=None, help="Local Census URI (None for S3)")
    ap.add_argument("--census_version", default="2025-01-30", help="Census version")
    ap.add_argument(
        "--embeddings",
        nargs="+",
        default=["scvi", "geneformer", "tf-sapiens", "tf-exemplar-human"],
        help="Embedding keys in obsm",
    )
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    ap.add_argument(
        "--task_limit",
        type=int,
        default=None,
        help="Limit number of datasets (for testing)",
    )
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    ap.add_argument(
        "--cores", type=int, default=16, help="Total CPU cores (for thread budgeting)"
    )
    ap.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary shard directory after completion",
    )
    ap.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable checkpoint resume (start from scratch)",
    )
    ap.add_argument(
        "--no_retry_errors",
        action="store_true",
        help="Don't retry failed tasks on resume",
    )
    args = ap.parse_args()

    run_benchmarks(
        celltype_manifest_path=args.celltype_manifest,
        census_uri=args.census_uri,
        census_version=args.census_version,
        embedding_keys=args.embeddings,
        out_parquet=args.out_parquet,
        log_file=args.log_file,
        n_splits=args.folds,
        task_limit=args.task_limit,
        max_workers=args.workers,
        keep_temp=args.keep_temp,
        total_cores=args.cores,
        resume=not args.no_resume,
        retry_errors=not args.no_retry_errors,
    )


if __name__ == "__main__":
    main()
