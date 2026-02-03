#!/usr/bin/env python3
"""
Pseudobulk benchmark for donor-level classification.

This module extends modular_benchmark.py with pseudobulk-specific aggregation methods.
Imports core functions from modular_benchmark and adds pseudobulk feature extractors.

Key differences from cell-level:
- Aggregates cells by donor before classification
- One prediction per donor (not per cell)
- Uses pooling strategies: mean, median, max
- Only standard CV supported (no bootstrap/downsampling at donor level)
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Callable

# Import core functions from modular_benchmark
from modular_benchmark import (
    setup_logging,
    preprocess_expression_matrix,
    lognorm_cp10k,
    extract_pca_features,
    train_and_evaluate_classifier,
    compute_comprehensive_metrics,
    fold_iterator,
    # Don't import run_standard_cv - we have our own pseudobulk version
)

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold


# =============================================================================
# Pseudobulk Aggregation Methods
# =============================================================================

def aggregate_pseudobulk_sum_raw(
    X_raw: sparse.csr_matrix,
    sample_ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_sum: float = 1e4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pseudobulk by summing raw counts per donor, then normalize.
    
    Mimics bulk RNA-seq: sum raw counts → normalize → log-transform.
    
    Args:
        X_raw: Raw count matrix (NOT log-normalized)
        sample_ids: Donor/sample ID for each cell
        train_idx: Indices of training cells
        test_idx: Indices of test cells
        target_sum: Target sum for normalization
        
    Returns:
        X_train_pb, X_test_pb: Pseudobulk features (one row per donor)
    """
    train_donors = np.unique(sample_ids[train_idx])
    test_donors = np.unique(sample_ids[test_idx])
    
    # Create pseudobulk for training donors
    X_train_pb_list = []
    for donor in train_donors:
        donor_mask = sample_ids[train_idx] == donor
        donor_cells = train_idx[donor_mask]
        pseudobulk = X_raw[donor_cells].sum(axis=0)
        
        if sparse.issparse(X_raw):
            pseudobulk = sparse.csr_matrix(pseudobulk)
        else:
            pseudobulk = np.asarray(pseudobulk).reshape(1, -1)
        
        X_train_pb_list.append(pseudobulk)
    
    X_train_pb = sparse.vstack(X_train_pb_list) if sparse.issparse(X_raw) else np.vstack(X_train_pb_list)
    
    # Create pseudobulk for test donors
    X_test_pb_list = []
    for donor in test_donors:
        donor_mask = sample_ids[test_idx] == donor
        donor_cells = test_idx[donor_mask]
        pseudobulk = X_raw[donor_cells].sum(axis=0)
        
        if sparse.issparse(X_raw):
            pseudobulk = sparse.csr_matrix(pseudobulk)
        else:
            pseudobulk = np.asarray(pseudobulk).reshape(1, -1)
        
        X_test_pb_list.append(pseudobulk)
    
    X_test_pb = sparse.vstack(X_test_pb_list) if sparse.issparse(X_raw) else np.vstack(X_test_pb_list)
    
    # Normalize pseudobulk samples (CP10K + log1p)
    X_train_pb = lognorm_cp10k(X_train_pb, target_sum)
    X_test_pb = lognorm_cp10k(X_test_pb, target_sum)
    
    # Scale
    scaler = MaxAbsScaler()
    X_train_pb = scaler.fit_transform(X_train_pb)
    X_test_pb = scaler.transform(X_test_pb)
    
    return X_train_pb, X_test_pb


def aggregate_pseudobulk_pool_features(
    features: np.ndarray,
    sample_ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    mode: str = "mean"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pseudobulk by pooling features per donor.
    
    Args:
        features: Feature matrix (cells × features)
        sample_ids: Donor/sample IDs
        train_idx: Training cell indices
        test_idx: Test cell indices
        mode: Pooling strategy - "mean", "median", or "max"
        
    Returns:
        X_train_pb, X_test_pb: Pseudobulk features (one row per donor)
    """
    assert mode in {"mean", "median", "max"}, f"Invalid mode: {mode}"
    
    train_donors = np.unique(sample_ids[train_idx])
    test_donors = np.unique(sample_ids[test_idx])
    
    def pool(mat):
        if sparse.issparse(mat):
            mat = mat.toarray()
        if mode == "mean":
            return mat.mean(axis=0)
        elif mode == "median":
            return np.median(mat, axis=0)
        elif mode == "max":
            return mat.max(axis=0)
    
    X_train_pb = np.vstack([
        pool(features[train_idx][sample_ids[train_idx] == d])
        for d in train_donors
    ])
    
    X_test_pb = np.vstack([
        pool(features[test_idx][sample_ids[test_idx] == d])
        for d in test_donors
    ])
    
    scaler = StandardScaler()
    X_train_pb = scaler.fit_transform(X_train_pb)
    X_test_pb = scaler.transform(X_test_pb)
    
    return X_train_pb, X_test_pb


# =============================================================================
# Pseudobulk Feature Extractors (with sample_ids signature)
# =============================================================================

def extract_pseudobulk_raw_sum(X_raw, sample_ids, train_idx, test_idx):
    """Pseudobulk: sum raw counts per donor."""
    return aggregate_pseudobulk_sum_raw(X_raw, sample_ids, train_idx, test_idx)


def extract_pseudobulk_pool_lognorm(X_lognorm, sample_ids, train_idx, test_idx, mode="mean"):
    """Pseudobulk: pool log-normalized counts per donor."""
    return aggregate_pseudobulk_pool_features(X_lognorm, sample_ids, train_idx, test_idx, mode=mode)


def extract_pseudobulk_pool_pca(X, sample_ids, train_idx, test_idx, n_components=50, mode="mean"):
    """Pseudobulk: PCA at cell level, then pool per donor."""
    # First do PCA at cell level
    X_train_cells, X_test_cells = extract_pca_features(X, train_idx, test_idx, n_components)
    
    # Reconstruct full PCA matrix
    full_pca = np.zeros((len(sample_ids), X_train_cells.shape[1]))
    full_pca[train_idx] = X_train_cells
    full_pca[test_idx] = X_test_cells
    
    return aggregate_pseudobulk_pool_features(full_pca, sample_ids, train_idx, test_idx, mode=mode)


def extract_pseudobulk_pool_embedding(embeddings, sample_ids, train_idx, test_idx, mode="mean"):
    """Pseudobulk: pool embeddings per donor."""
    return aggregate_pseudobulk_pool_features(embeddings, sample_ids, train_idx, test_idx, mode=mode)


def extract_pseudobulk_pool_embedding_pca(
    embeddings,
    sample_ids,
    train_idx,
    test_idx,
    n_components=50,
    mode="mean",
):
    """
    Pseudobulk: PCA on embeddings at cell level, then pool per donor.
    
    This is for: embedding → PCA → pool workflow.
    """
    from sklearn.decomposition import PCA
    
    # PCA at cell level (train only!)
    pca = PCA(n_components=n_components, random_state=0)
    X_train_cells = pca.fit_transform(embeddings[train_idx])
    X_test_cells = pca.transform(embeddings[test_idx])
    
    # Reconstruct full matrix
    full_pca = np.zeros((len(sample_ids), n_components))
    full_pca[train_idx] = X_train_cells
    full_pca[test_idx] = X_test_cells
    
    return aggregate_pseudobulk_pool_features(
        full_pca, sample_ids, train_idx, test_idx, mode=mode
    )


# =============================================================================
# Pseudobulk-Specific CV Functions
# =============================================================================

def run_cv_benchmark_pseudobulk(
    X_normalized: sparse.csr_matrix,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    feature_extractors: Dict[str, Callable],
    fold_name: str = "fold",
    alpha: float = 1e-5,
    n_jobs: int = 1,
    save_predictions: bool = True,
    logger: Optional[logging.Logger] = None,
    X_raw: Optional[sparse.csr_matrix] = None,
    **metadata
) -> Tuple[List[Dict], List[Dict]]:
    """
    Core CV benchmark for PSEUDOBULK (donor-level) methods.
    
    Key differences from cell-level:
    - Aggregates cells by donor
    - One prediction per donor (not per cell)
    - Uses donor-level labels
    
    Args:
        X_normalized: Log-normalized expression matrix
        sample_ids: Donor/sample IDs for each cell
        labels: Class labels for each cell
        train_indices: Indices of training cells
        test_indices: Indices of test cells
        feature_extractors: Dict of pseudobulk extraction functions
        fold_name: Name of this fold
        alpha: L2 regularization
        n_jobs: CPU threads
        save_predictions: Save per-donor predictions
        logger: Logger instance
        X_raw: Raw counts (for pseudobulk_raw_sum)
        **metadata: Additional metadata
        
    Returns:
        results: List of result dicts (one per feature type)
        predictions: List of prediction dicts (one per donor × feature type)
    """
    from modular_benchmark import train_and_evaluate_classifier, compute_comprehensive_metrics
    
    if logger is None:
        logger = logging.getLogger("pseudobulk")
    
    results = []
    predictions = []
    
    # Get donor-level information
    train_donors = np.unique(sample_ids[train_indices])
    test_donors = np.unique(sample_ids[test_indices])
    
    # Get donor-level labels (use first occurrence for each donor)
    y_train_donor = np.array([labels[sample_ids == d][0] for d in train_donors])
    y_test_donor = np.array([labels[sample_ids == d][0] for d in test_donors])
    
    classes = np.unique(labels)
    
    logger.info(f"  {fold_name}: {len(train_donors)} train donors, {len(test_donors)} test donors")
    
    # Iterate over feature extraction methods
    for feature_name, extractor_func in feature_extractors.items():
        try:
            # Check if needs raw counts
            import inspect
            func_params = inspect.signature(extractor_func).parameters
            
            if 'X_raw' in func_params:
                if X_raw is None:
                    raise ValueError(f"{feature_name} requires X_raw but it was not provided")
                X_train, X_test = extractor_func(X_raw, sample_ids, train_indices, test_indices)
            else:
                X_train, X_test = extractor_func(X_normalized, sample_ids, train_indices, test_indices)
            
            logger.info(f"    {feature_name}: {X_train.shape[0]} train donors, {X_test.shape[0]} test donors")
            
            # Train and predict
            clf, y_proba, y_pred, y_logits = train_and_evaluate_classifier(
                X_train, X_test, y_train_donor, y_test_donor, alpha=alpha, n_jobs=n_jobs
            )
            
            # Calculate metrics
            cell_metrics = compute_comprehensive_metrics(
                y_test_donor, y_pred, y_proba, classes, logger
            )
            
            # Store aggregate results
            result = {
                "fold": fold_name,
                "feature_type": feature_name,
                "n_train_donors": len(train_donors),
                "n_test_donors": len(test_donors),
                "is_pseudobulk": True,
                **metadata,
                **cell_metrics,
            }
            results.append(result)
            
            # Store per-donor predictions if requested
            if save_predictions:
                for i, donor in enumerate(test_donors):
                    pred_record = {
                        "fold": fold_name,
                        "feature_type": feature_name,
                        **metadata,
                        "donor_id": donor,
                        "id_type": "donor",
                        "true_label": y_test_donor[i],
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
            logger.error(f"    Error with {feature_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            result = {
                "fold": fold_name,
                "feature_type": feature_name,
                **metadata,
                "error": str(e),
            }
            results.append(result)
    
    return results, predictions


def run_standard_cv_pseudobulk(
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
    X_raw: Optional[sparse.csr_matrix] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standard K-fold CV for pseudobulk (donor-level) methods.
    
    This is the main entry point for pseudobulk benchmarks.
    
    Args:
        X_normalized: Log-normalized expression matrix
        sample_ids: Donor/sample IDs
        labels: Class labels
        cell_names: Cell barcodes (not used, for API consistency)
        feature_extractors: Dict of pseudobulk extraction functions
        n_folds: Number of CV folds
        alpha: L2 regularization
        n_jobs: CPU threads
        save_predictions: Save per-donor predictions
        logger: Logger instance
        X_raw: Raw counts (for pseudobulk_raw_sum)
        
    Returns:
        results_df: Aggregated metrics per fold
        predictions_df: Per-donor predictions
    """
    from sklearn.model_selection import StratifiedGroupKFold
    
    if logger is None:
        logger = logging.getLogger("pseudobulk")
    
    logger.info("="*60)
    logger.info("PSEUDOBULK STANDARD K-FOLD CV")
    logger.info(f"  Folds: {n_folds}")
    logger.info(f"  Features: {list(feature_extractors.keys())}")
    logger.info("="*60)
    
    all_results = []
    all_predictions = []
    
    # Iterate over folds
    for fold_id, train_idx, test_idx in fold_iterator(
            X_normalized, labels, sample_ids, n_folds, cv_folds
        ):
        fold_name = f"fold_{fold_id + 1}"
        logger.info(f"\nProcessing {fold_name}...")
        
        # Run pseudobulk CV benchmark for this fold
        results, predictions = run_cv_benchmark_pseudobulk(
            X_normalized=X_normalized,
            sample_ids=sample_ids,
            labels=labels,
            train_indices=train_idx,
            test_indices=test_idx,
            feature_extractors=feature_extractors,
            fold_name=fold_name,
            alpha=alpha,
            n_jobs=n_jobs,
            save_predictions=save_predictions,
            logger=logger,
            X_raw=X_raw,
        )
        
        all_results.extend(results)
        all_predictions.extend(predictions)
    
    logger.info(f"\nCompleted {n_folds} folds!")
    logger.info(f"  Total results: {len(all_results)}")
    logger.info(f"  Total predictions: {len(all_predictions)}")
    
    return pd.DataFrame(all_results), pd.DataFrame(all_predictions)


# =============================================================================
# Helper: Create feature extractors dict
# =============================================================================

def create_pseudobulk_feature_extractors(
    adata,
    pooling_mode: str = "median",
    embedding_keys: Optional[List[str]] = None,
    include_raw_sum: bool = True,
    include_lognorm: bool = True,
    include_pca: bool = True,
    include_embedding_pca: bool = False,  # NEW: PCA on embeddings
    pca_components: int = 50,
) -> Dict[str, Callable]:
    """
    Create dictionary of pseudobulk feature extractors.
    
    Args:
        adata: AnnData object (for accessing embeddings)
        pooling_mode: "mean", "median", or "max" for pooling strategies
        embedding_keys: List of embedding names in adata.obsm (None = all)
        include_raw_sum: Include raw count sum method
        include_lognorm: Include log-normalized pooling
        include_pca: Include PCA on lognorm + pooling
        include_embedding_pca: Include PCA on embeddings + pooling (NEW!)
        pca_components: Number of PCA components
        
    Returns:
        Dict of feature_name -> extraction_function
    """
    extractors = {}
    
    # Raw sum (special case - needs X_raw)
    if include_raw_sum:
        extractors["pseudobulk_raw_sum"] = lambda X_raw, sample_ids, tr, te: \
            extract_pseudobulk_raw_sum(X_raw, sample_ids, tr, te)
    
    # Log-normalized pooling
    if include_lognorm:
        extractors[f"pseudobulk_lognorm_{pooling_mode}"] = lambda X, sample_ids, tr, te: \
            extract_pseudobulk_pool_lognorm(X, sample_ids, tr, te, mode=pooling_mode)
    
    # PCA on lognorm + pooling
    if include_pca:
        extractors[f"pseudobulk_pca{pca_components}_{pooling_mode}"] = lambda X, sample_ids, tr, te: \
            extract_pseudobulk_pool_pca(X, sample_ids, tr, te, n_components=pca_components, mode=pooling_mode)
    
    # Embeddings
    if embedding_keys is None:
        # Auto-detect embeddings in obsm
        embedding_keys = [k for k in adata.obsm.keys() if not k.startswith('_')]
    
    for emb_key in embedding_keys:
        if emb_key not in adata.obsm:
            logging.warning(f"Embedding '{emb_key}' not found in adata.obsm, skipping")
            continue
        
        embeddings = adata.obsm[emb_key].copy()  # Copy to avoid reference issues
        
        # Direct pooling of embeddings
        # FIX: Use default argument to capture embeddings correctly
        extractors[f"pseudobulk_{emb_key}_{pooling_mode}"] = \
            (lambda X, sample_ids, tr, te, emb=embeddings, mode=pooling_mode: 
                extract_pseudobulk_pool_embedding(emb, sample_ids, tr, te, mode=mode))
        
        # NEW: PCA on embeddings + pooling
        if include_embedding_pca:
            extractors[f"pseudobulk_{emb_key}_pca{pca_components}_{pooling_mode}"] = \
                (lambda X, sample_ids, tr, te, emb=embeddings, n_comp=pca_components, mode=pooling_mode:
                    extract_pseudobulk_pool_embedding_pca(
                        emb, sample_ids, tr, te,
                        n_components=n_comp,
                        mode=mode
                    ))
    
    return extractors


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Command-line interface for pseudobulk benchmark."""
    ap = argparse.ArgumentParser(description="Pseudobulk (donor-level) benchmark")
    ap.add_argument("--adata_path", required=True, help="Path to AnnData (.h5ad)")
    ap.add_argument("--label_col", required=True, help="Column with class labels")
    ap.add_argument("--sample_col", default="sample", help="Column with donor/sample IDs")
    ap.add_argument("--out_results", required=True, help="Output results CSV")
    ap.add_argument("--out_predictions", required=True, help="Output predictions CSV")
    ap.add_argument("--log_file", default="pseudobulk_benchmark.log", help="Log file")
    
    # Pseudobulk-specific parameters
    ap.add_argument(
        "--pooling_mode",
        choices=["mean", "median", "max"],
        default="median",
        help="Pooling strategy for aggregating cells to donors"
    )
    ap.add_argument(
        "--embeddings",
        nargs="+",
        default=None,
        help="Embedding keys in obsm to test (None = auto-detect all)"
    )
    ap.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="Number of PCA components"
    )
    ap.add_argument(
        "--no_raw_sum",
        action="store_true",
        help="Don't include raw count sum method"
    )
    ap.add_argument(
        "--no_lognorm",
        action="store_true",
        help="Don't include log-normalized pooling"
    )
    ap.add_argument(
        "--no_pca",
        action="store_true",
        help="Don't include PCA + pooling"
    )
    ap.add_argument(
        "--embedding_pca",
        action="store_true",
        help="Include PCA on embeddings before pooling (embedding → PCA → pool)"
    )
    
    # Standard CV parameters
    ap.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    ap.add_argument("--alpha", type=float, default=1e-5, help="L2 regularization")
    ap.add_argument("--n_jobs", type=int, default=1, help="CPU threads")
    ap.add_argument("--no_predictions", action="store_true", help="Don't save per-donor predictions")
    
    args = ap.parse_args()
    
    # Setup
    logger = setup_logging(args.log_file)
    logger.info("="*80)
    logger.info("PSEUDOBULK BENCHMARK STARTED")
    logger.info(f"  Data: {args.adata_path}")
    logger.info(f"  Label: {args.label_col}, Sample: {args.sample_col}")
    logger.info(f"  Pooling mode: {args.pooling_mode}")
    logger.info("="*80)
    
    # Load data
    import scanpy as sc
    adata = sc.read_h5ad(args.adata_path)
    logger.info(f"Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    # Preprocess
    X_filtered, _ = preprocess_expression_matrix(adata)
    logger.info(f"Filtered to {X_filtered.shape[1]} genes")
    
    # Keep raw counts for pseudobulk sum method
    X_raw = X_filtered.copy()
    
    # Compute log-normalization
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
    
    logger.info(f"Unique donors: {len(np.unique(sample_ids))}")
    logger.info(f"Unique labels: {np.unique(labels)}")
    
    # Create feature extractors
    feature_extractors = create_pseudobulk_feature_extractors(
        adata=adata,
        pooling_mode=args.pooling_mode,
        embedding_keys=args.embeddings,
        include_raw_sum=not args.no_raw_sum,
        include_lognorm=not args.no_lognorm,
        include_pca=not args.no_pca,
        include_embedding_pca=args.embedding_pca,  # NEW!
        pca_components=args.pca_components,
    )
    
    logger.info(f"Testing {len(feature_extractors)} pseudobulk methods:")
    for name in feature_extractors.keys():
        logger.info(f"  - {name}")
    
    # Run pseudobulk-specific standard CV
    results_df, predictions_df = run_standard_cv_pseudobulk(
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
        X_raw=X_raw,  # Pass raw counts
    )
    
    # Save results
    results_df.to_csv(args.out_results, index=False)
    logger.info(f"Saved results: {args.out_results}")
    
    if not args.no_predictions:
        predictions_df.to_csv(args.out_predictions, index=False)
        logger.info(f"Saved predictions: {args.out_predictions}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Performance Summary (mean across folds):")
    logger.info("="*80)
    summary = results_df.groupby("feature_type").agg({
        "balanced_acc": ["mean", "std"],
        "F1": ["mean", "std"],
        "AUROC": ["mean", "std"],
    }).round(3)
    logger.info("\n" + str(summary))
    
    logger.info("="*80)
    logger.info("FINISHED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    main()
