#!/usr/bin/env python3
"""
Modular scRNA-seq classification benchmark.

Core architecture:
1. run_cv_benchmark() - Core CV function (folds × features)
2. run_low_data_benchmark() - Wrapper adding N × bootstrap loops

This separation allows:
- Using just CV without downsampling
- Adding downsampling when needed
- Easy extension for other sampling strategies
"""

import os
import time
import argparse
import logging
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Callable


from sklearn.preprocessing import StandardScaler, MaxAbsScaler, LabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    log_loss,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    recall_score,
)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: str) -> logging.Logger:
    """Configures logger for file output."""
    logging.captureWarnings(True)
    
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False
    
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)
    
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    return logger


# =============================================================================
# Data Preprocessing
# =============================================================================

def preprocess_expression_matrix(adata):
    """Filter genes with zero expression."""
    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    
    gene_mask = np.array(X.sum(axis=0)).flatten() > 0
    X_filtered = X[:, gene_mask]
    
    return X_filtered, gene_mask


def lognorm_cp10k(X: sparse.csr_matrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    """Normalize counts per cell and log-transform."""
    X = X.tocsr().copy()
    counts_per_cell = np.array(X.sum(axis=1)).flatten()
    counts_per_cell[counts_per_cell == 0] = 1
    
    scaling_factors = target_sum / counts_per_cell
    scale_mat = sparse.diags(scaling_factors)
    X = scale_mat @ X
    X.data = np.log1p(X.data)
    
    return X


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_raw_features(X, train_idx, test_idx):
    """Raw normalized expression with MaxAbsScaler."""
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    return X_train, X_test


def extract_pca_features(X, train_idx, test_idx, n_components=50):
    """PCA features with StandardScaler."""
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    
    n_comp = min(n_components, X_train_raw.shape[0] - 1, X_train_raw.shape[1] - 1)
    
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    X_train_pca = svd.fit_transform(X_train_raw)
    X_test_pca = svd.transform(X_test_raw)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    return X_train_scaled, X_test_scaled


def extract_random_projection_features(X, train_idx, test_idx, n_components=50):
    """Random projection features with StandardScaler."""
    rp = SparseRandomProjection(n_components=n_components, dense_output=True, random_state=0)
    rp.fit(X[train_idx][:1])
    
    X_train_proj = rp.transform(X[train_idx].astype(np.float64))
    X_test_proj = rp.transform(X[test_idx].astype(np.float64))
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proj)
    X_test_scaled = scaler.transform(X_test_proj)
    
    return X_train_scaled, X_test_scaled


def extract_embedding_features(embeddings, train_idx, test_idx):
    """Pre-computed embeddings with StandardScaler."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings[train_idx])
    X_test = scaler.transform(embeddings[test_idx])
    
    return X_train, X_test





# =============================================================================
# Model Training and Evaluation
# =============================================================================

def train_and_evaluate_classifier(
    X_train, 
    X_test, 
    y_train, 
    y_test,
    alpha: float = 1e-5,
    n_jobs: int = 1
):
    """Train SGD classifier and generate predictions."""
    
    if len(y_train) < 10:  # super tiny N
        clf = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
            n_jobs=n_jobs,
            early_stopping=False  # <--- FIX
        )
    else:
        clf = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
            n_jobs=n_jobs,
            early_stopping=True
        )
        
    
    clf.fit(X_train, y_train)
    probabilities = clf.predict_proba(X_test)
    predictions = clf.predict(X_test)
    decision_scores = clf.decision_function(X_test)
    
    return clf, probabilities, predictions, decision_scores


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """Calculate classification metrics for binary or multiclass."""
    if logger is None:
        logger = logging.getLogger("benchmark")
    
    is_binary = len(classes) == 2
    
    if len(y_true) < 2:
        logger.warning("Less than 2 test samples, returning NaN metrics")
        return {
            "AUROC": np.nan, "AUPRC": np.nan, "balanced_acc": np.nan,
            "F1": np.nan, "recall": np.nan, "MCC": np.nan, "log_loss": np.nan,
            "TP": 0, "FP": 0, "TN": 0, "FN": 0,
        }
    
    # AUROC and AUPRC
    try:
        if is_binary:
            y_true = (y_true != "Healthy").astype(int)
            y_pred = (y_pred != "Healthy").astype(int)
            classes = (classes != "Healthy").astype(int)

            y_score = y_proba[:, 1]
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
        else:
            lb = LabelBinarizer()
            y_true_bin = lb.fit_transform(y_true)
            
            if y_true_bin.shape[1] < len(classes):
                y_true_bin_full = np.zeros((len(y_true), len(classes)))
                for i, cls in enumerate(lb.classes_):
                    cls_idx = np.where(classes == cls)[0][0]
                    y_true_bin_full[:, cls_idx] = y_true_bin[:, i] if y_true_bin.ndim > 1 else y_true_bin
                y_true_bin = y_true_bin_full
            
            auroc = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
            auprc = average_precision_score(y_true_bin, y_proba, average='macro')
    except Exception as e:
        logger.warning(f"Error computing AUROC/AUPRC: {e}")
        auroc = np.nan
        auprc = np.nan
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    if is_binary:
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = np.diag(cm).sum()
        fp = cm.sum(axis=0).sum() - tp
        fn = cm.sum(axis=1).sum() - tp
        tn = cm.sum() - tp - fp - fn
    
    # F1 and Recall
    if is_binary:
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
    else:
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "F1": float(f1),
        "recall": float(recall),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
    
    return metrics


def compute_donor_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: np.ndarray
) -> Dict[str, float]:
    """Calculate per-donor accuracy metrics."""
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "sample": sample_ids
    })
    
    donor_accuracies = df.groupby("sample").apply(
        lambda x: (x.y_true == x.y_pred).mean()
    )
    
    return {
        "mean_donor_acc": float(donor_accuracies.mean()),
        "worst_donor_acc": float(donor_accuracies.min()),
        "std_donor_acc": float(donor_accuracies.std()),
    }

def fold_iterator(X, labels, sample_ids, n_folds, cv_folds=None):
    if cv_folds is not None:
        for fold_id, test_samples in enumerate(cv_folds):
            test_mask = np.isin(sample_ids, test_samples)
            train_mask = ~test_mask
            yield fold_id, np.where(train_mask)[0], np.where(test_mask)[0]
    else:
        sgkf = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=0
        )
        for fold_id, (train_idx, test_idx) in enumerate(
            sgkf.split(X, labels, groups=sample_ids)
        ):
            yield fold_id, train_idx, test_idx



# =============================================================================
# CORE CV FUNCTION (Folds × Features)
# =============================================================================

def run_cv_benchmark(
    X_normalized: sparse.csr_matrix,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    cell_names: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    feature_extractors: Dict[str, Callable],
    fold_name: str = "fold",
    alpha: float = 1e-5,
    n_jobs: int = 1,
    save_predictions: bool = True,
    logger: Optional[logging.Logger] = None,
    **metadata  # Additional metadata to include in results (e.g., n_per_class, bootstrap)
) -> Tuple[List[Dict], List[Dict]]:
    """
    Core CV benchmark function: train/test split × feature extractors.
    
    This is the CORE function that handles:
    - Single train/test split
    - Multiple feature extraction methods
    - Training, prediction, evaluation
    
    Args:
        X_normalized: Log-normalized expression matrix
        sample_ids: Donor/sample IDs for each cell
        labels: Class labels for each cell
        cell_names: Cell barcodes/names
        train_indices: Indices of training cells
        test_indices: Indices of test cells
        feature_extractors: Dict of {name: extraction_function}
        fold_name: Name of this fold/split
        alpha: L2 regularization strength
        n_jobs: Number of CPU threads
        save_predictions: Whether to save per-cell predictions
        logger: Logger instance
        **metadata: Additional fields to include in results (e.g., n_per_class, bootstrap)
        
    Returns:
        results: List of result dicts (one per feature type)
        predictions: List of prediction dicts (one per cell × feature type)
    """
    if logger is None:
        logger = logging.getLogger("benchmark")
    
    results = []
    predictions = []
    
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    classes = np.unique(labels)
    
    logger.info(f"  {fold_name}: {len(train_indices)} train, {len(test_indices)} test cells")
    print(feature_extractors.items())
    # Iterate over feature extraction methods
    for feature_name, extractor_func in feature_extractors.items():
        try:
            # Extract features
            X_train, X_test = extractor_func(X_normalized, train_indices, test_indices)
            
            # Train and predict
            clf, y_proba, y_pred, y_logits = train_and_evaluate_classifier(
                X_train, X_test, y_train, y_test, alpha=alpha, n_jobs=n_jobs
            )
            
            # Calculate metrics
            cell_metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba, classes, logger)
            donor_metrics = compute_donor_level_metrics(y_test, y_pred, sample_ids[test_indices])
            
            # Store aggregate results
            result = {
                "fold": fold_name,
                "feature_type": feature_name,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                **metadata,  # Include any additional metadata (N, bootstrap, etc.)
                **cell_metrics,
                **donor_metrics,
            }
            results.append(result)
            # Store per-cell predictions if requested
            if save_predictions:
                for i, idx in enumerate(test_indices):
                    pred_record = {
                        "fold": fold_name,
                        "feature_type": feature_name,
                        **metadata,  # Include metadata in predictions too
                        "cell_name": cell_names[idx],
                        "sample": sample_ids[idx],
                        "true_label": y_test[i],
                        "predicted_label": y_pred[i],
                    }
                    
                    # Add probabilities
                    for class_idx, class_label in enumerate(classes):
                        pred_record[f"prob_{class_label}"] = y_proba[i, class_idx]
                    
                    # Add logits
                    if y_logits.ndim == 1:
                        pred_record["logit"] = y_logits[i]
                    else:
                        for class_idx, class_label in enumerate(classes):
                            pred_record[f"logit_{class_label}"] = y_logits[i, class_idx]
                    
                    predictions.append(pred_record)
        
        except Exception as e:
            logger.error(f"  Error with {feature_name}: {e}")
            logger.error(traceback.format_exc())
            
            result = {
                "fold": fold_name,
                "feature_type": feature_name,
                **metadata,
                "error": str(e),
            }
            results.append(result)
    
    return results, predictions


# =============================================================================
# WRAPPER: Standard K-Fold CV (no downsampling)
# =============================================================================

def run_standard_cv(
    X_normalized: sparse.csr_matrix,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    cell_names: np.ndarray,
    feature_extractors: Dict[str, Callable],
    cv_folds: Optional[List[List[str]]] = None,
    n_folds: int = 5,
    alpha: float = 1e-5,
    n_jobs: int = 1,
    save_predictions: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standard K-fold CV without downsampling.
    
    Use this when you want normal cross-validation (no low-data regime).
    
    Args:
        X_normalized: Log-normalized expression matrix
        sample_ids: Donor/sample IDs
        labels: Class labels
        cell_names: Cell barcodes
        feature_extractors: Dict of feature extraction functions
        n_folds: Number of CV folds
        alpha: L2 regularization
        n_jobs: CPU threads
        save_predictions: Save per-cell predictions
        logger: Logger instance
        
    Returns:
        results_df: Aggregated metrics
        predictions_df: Per-cell predictions
    """
    if logger is None:
        logger = logging.getLogger("benchmark")
    
    logger.info("="*60)
    logger.info("STANDARD K-FOLD CV (no downsampling)")
    logger.info(f"  Folds: {n_folds}")
    logger.info(f"  Features: {list(feature_extractors.keys())}")
    logger.info("="*60)
    
    all_results = []
    all_predictions = []
    
    for fold_id, train_idx, test_idx in fold_iterator(
            X_normalized, labels, sample_ids, n_folds, cv_folds
        ):
        print(f"Processing fold {fold_id + 1}...")
        
        fold_name = f"fold_{fold_id + 1}"
        logger.info(f"\nProcessing {fold_name}...")
        
        # Run core CV benchmark for this fold
        results, predictions = run_cv_benchmark(
            X_normalized=X_normalized,
            sample_ids=sample_ids,
            labels=labels,
            cell_names=cell_names,
            train_indices=train_idx,
            test_indices=test_idx,
            feature_extractors=feature_extractors,
            fold_name=fold_name,
            alpha=alpha,
            n_jobs=n_jobs,
            save_predictions=save_predictions,
            logger=logger,
        )
        
        all_results.extend(results)
        all_predictions.extend(predictions)
        #pd.DataFrame(all_results).to_csv('./result_tmp', index=False)
        #pd.DataFrame(all_predictions).to_csv('./pred_tmp', index=False)
    
    logger.info(f"\nCompleted {n_folds} folds!")
    logger.info(f"  Total results: {len(all_results)}")
    logger.info(f"  Total predictions: {len(all_predictions)}")
    
    return pd.DataFrame(all_results), pd.DataFrame(all_predictions)


# =============================================================================
# WRAPPER: Low-Data Regime (Folds × N × Bootstrap)
# =============================================================================

def downsample_to_n_per_class(
    train_idx: np.ndarray,
    y_train: np.ndarray,
    n_per_class: int,
    random_state: int,
    logger: logging.Logger
) -> Tuple[np.ndarray, bool]:
    """Downsample training data to N samples per class."""
    rng = np.random.RandomState(random_state)
    selected_idx = []
    all_classes_met = True
    
    for class_label in np.unique(y_train):
        class_mask = y_train == class_label
        class_indices = train_idx[class_mask]
        n_available = len(class_indices)
        
        if n_available < n_per_class:
            logger.warning(
                f"    Class {class_label}: only {n_available}/{n_per_class} available"
            )
            selected_idx.extend(class_indices)
            all_classes_met = False
        else:
            selected = rng.choice(class_indices, size=n_per_class, replace=False)
            selected_idx.extend(selected)
    
    return np.array(selected_idx), all_classes_met


def run_low_data_cv(
    X_normalized: sparse.csr_matrix,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    cell_names: np.ndarray,
    feature_extractors: Dict[str, Callable],
    cv_folds: Optional[List[List[str]]] = None,
    n_samples_list: List[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    n_bootstrap: int = 10,
    n_folds: int = 5,
    alpha: float = 1e-5,
    n_jobs: int = 1,
    save_predictions: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Low-data regime: K-fold CV × N per class × Bootstrap.
    
    Use this to test the super-low data regime with downsampling.
    
    Structure:
        for fold in folds:
            for N in n_samples_list:
                for bootstrap in range(n_bootstrap):
                    downsample to N per class
                    run_cv_benchmark()  # core function
    
    Args:
        X_normalized: Log-normalized expression matrix
        sample_ids: Donor/sample IDs
        labels: Class labels
        cell_names: Cell barcodes
        feature_extractors: Dict of feature extraction functions
        n_samples_list: List of N values (samples per class)
        n_bootstrap: Bootstrap replicates per N
        n_folds: Number of CV folds
        alpha: L2 regularization
        n_jobs: CPU threads
        save_predictions: Save per-cell predictions
        logger: Logger instance
        
    Returns:
        results_df: Aggregated metrics
        predictions_df: Per-cell predictions
    """
    if logger is None:
        logger = logging.getLogger("benchmark")
    
    logger.info("="*60)
    logger.info("LOW-DATA REGIME CV (downsampling + bootstrap)")
    logger.info(f"  Folds: {n_folds}")
    logger.info(f"  N values: {n_samples_list}")
    logger.info(f"  Bootstrap: {n_bootstrap}")
    logger.info(f"  Features: {list(feature_extractors.keys())}")
    logger.info("="*60)
    
    all_results = []
    all_predictions = []

    for fold_id, train_idx, test_idx in fold_iterator(
            X_normalized, labels, sample_ids, n_folds, cv_folds
        ):
        print(f"Processing fold {fold_id + 1}...")
            
        fold_name = f"fold_{fold_id + 1}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {fold_name}")
        
        y_train_full = labels[train_idx]
        
        # Middle loop: N (samples per class)
        for n_per_class in n_samples_list:
            logger.info(f"\n  N={n_per_class} samples per class:")
            
            # Inner loop: Bootstrap replicates
            for bootstrap_idx in range(1, n_bootstrap + 1):
                # Downsample training data
                random_seed = fold_id * 1000 + bootstrap_idx
                train_idx_downsampled, all_classes_met = downsample_to_n_per_class(
                    train_idx=train_idx,
                    y_train=y_train_full,
                    n_per_class=n_per_class,
                    random_state=random_seed,
                    logger=logger
                )
                
                logger.info(
                    f"    Bootstrap {bootstrap_idx}/{n_bootstrap}: "
                    f"{len(train_idx_downsampled)} cells selected"
                )
                
                # Run core CV benchmark with downsampled training data
                results, predictions = run_cv_benchmark(
                    X_normalized=X_normalized,
                    sample_ids=sample_ids,
                    labels=labels,
                    cell_names=cell_names,
                    train_indices=train_idx_downsampled,
                    test_indices=test_idx,
                    feature_extractors=feature_extractors,
                    fold_name=fold_name,
                    alpha=alpha,
                    n_jobs=n_jobs,
                    save_predictions=save_predictions,
                    logger=logger,
                    # Additional metadata for low-data regime
                    n_per_class=n_per_class,
                    bootstrap=bootstrap_idx,
                    all_classes_met=all_classes_met,
                )
                
                all_results.extend(results)
                all_predictions.extend(predictions)
    
    logger.info(f"\n{'='*60}")
    logger.info("Benchmark complete!")
    logger.info(f"  Total results: {len(all_results)}")
    logger.info(f"  Total predictions: {len(all_predictions)}")
    logger.info("="*60)
    
    return pd.DataFrame(all_results), pd.DataFrame(all_predictions)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Command-line interface."""
    ap = argparse.ArgumentParser(description="Modular scRNA-seq classification benchmark")
    ap.add_argument("--adata_path", required=True, help="Path to AnnData (.h5ad)")
    ap.add_argument("--label_col", required=True, help="Column with class labels")
    ap.add_argument("--sample_col", default="sample", help="Column with donor/sample IDs")
    ap.add_argument("--out_results", required=True, help="Output results CSV")
    ap.add_argument("--out_predictions", required=True, help="Output predictions CSV")
    ap.add_argument("--log_file", default="benchmark.log", help="Log file")
    
    # Mode selection
    ap.add_argument(
        "--mode",
        choices=["standard", "low_data"],
        default="low_data",
        help="Benchmark mode: 'standard' CV or 'low_data' regime with downsampling"
    )
    ap.add_argument(
        "--embeddings",
        nargs="+",
        default=['tformer', 'bmfm', 'CLS', 'hvg4096_19_11'],
        help="Embedding keys in adata.obsm"
    )

    # Standard CV parameters
    ap.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    
    # Low-data regime parameters
    ap.add_argument(
        "--n_samples",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        help="N values (samples per class) for low-data mode"
    )
    ap.add_argument("--n_bootstrap", type=int, default=10, help="Bootstrap replicates (low-data mode)")
    
    # Model parameters
    ap.add_argument("--alpha", type=float, default=1e-5, help="L2 regularization")
    ap.add_argument("--n_jobs", type=int, default=1, help="CPU threads")
    
    # Feature selection
    ap.add_argument(
        "--features",
        nargs="+",
        default=["raw", "pca", "randproj"],
        choices=["raw", "pca", "randproj"],
        help="Feature extraction methods"
    )
    ap.add_argument("--no_predictions", action="store_true", help="Don't save per-cell predictions")
    
    args = ap.parse_args()
    
    # Setup
    logger = setup_logging(args.log_file)
    logger.info("="*80)
    logger.info(f"BENCHMARK STARTED - Mode: {args.mode}")
    logger.info(f"  Data: {args.adata_path}")
    logger.info(f"  Label: {args.label_col}, Sample: {args.sample_col}")
    logger.info("="*80)
    
    # Load data
    import scanpy as sc
    adata = sc.read_h5ad(args.adata_path)
    logger.info(f"Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    embeddings_dict = {}


    # Preprocess
    X_filtered, _ = preprocess_expression_matrix(adata)
    logger.info(f"Filtered to {X_filtered.shape[1]} genes")
    
    X_normalized = lognorm_cp10k(X_filtered)
    logger.info("Log-normalization complete")
    
    # Extract metadata
    sample_ids = adata.obs[args.sample_col].values
    labels = adata.obs[args.label_col].values
    cell_names = adata.obs.index.values

    samples_list1 = ['H055','H064','H071','H021','H023','H032','H020','H028','H084','H123','H087','H113']
    samples_list2 = ['H099','H080','H063','H037','H044','H024','H040','H053','H097','H107','H036','H119']
    samples_list3 = ['H122','H117','H103','H059','H047','H042','H093','H056','H029','H095','H045','H022']
    samples_list4 = ['H104','H054','H050','H026','H082','H034','H035','H052','H074','H105','H114','H118']
    samples_list5 = ['H077','H075','H038','H043','H039','H092','','','H072','H101','H110','H049']
    
    folds = []
    for a,b,c,d,e in zip(samples_list1, samples_list2, samples_list3, samples_list4, samples_list5):
        folds.append([s for s in [a,b,c,d,e] if s])


    cv_folds = folds
    
    # Setup feature extractors
    feature_extractors = {}
    if "raw" in args.features:
        feature_extractors["raw"] = extract_raw_features
    if "pca" in args.features:
        feature_extractors["pca"] = extract_pca_features
    if "randproj" in args.features:
        feature_extractors["randproj"] = extract_random_projection_features
    for emb_key in args.embeddings:
        if emb_key not in adata.obsm:
            raise KeyError(f"Embedding '{emb_key}' not found in adata.obsm")
    
        feature_extractors[emb_key] = (
            lambda X, tr, te, emb=adata.obsm[emb_key]:
                extract_embedding_features(emb, tr, te)
        )
    
    # Run benchmark based on mode
    if args.mode == "standard":
        results_df, predictions_df = run_standard_cv(
            X_normalized=X_normalized,
            sample_ids=sample_ids,
            labels=labels,
            cell_names=cell_names,
            cv_folds=cv_folds,
            feature_extractors=feature_extractors,
            n_folds=args.n_folds,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            save_predictions=not args.no_predictions,
            logger=logger,
        )
    else:  # low_data
        results_df, predictions_df = run_low_data_cv(
            X_normalized=X_normalized,
            sample_ids=sample_ids,
            labels=labels,
            cell_names=cell_names,
            cv_folds=cv_folds,
            feature_extractors=feature_extractors,
            n_samples_list=args.n_samples,
            n_bootstrap=args.n_bootstrap,
            n_folds=args.n_folds,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            save_predictions=not args.no_predictions,
            logger=logger,
        )
    
    # Save results
    results_df.to_csv(args.out_results, index=False)
    logger.info(f"Saved results: {args.out_results}")
    
    if not args.no_predictions:
        predictions_df.to_csv(args.out_predictions, index=False)
        logger.info(f"Saved predictions: {args.out_predictions}")
    
    logger.info("="*80)
    logger.info("FINISHED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    main()
