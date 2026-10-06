#!/usr/bin/env python3
"""
Unified scRNA-seq classification benchmark - Single interface for all benchmark modes.

This module consolidates modular_benchmark.py and pseudobulk_benchmark.py into a
single interface with mode flags. All implementations use the unified code in
benchmark.py, preprocessing.py, and metrics.py.

Benchmark modes:
- standard: Standard cross-validation on single-cell data
- low_data: Downsampling benchmark (low-data regime)
- pseudobulk: Donor-level classification via cell aggregation

Main functions:
- run_cv_benchmark(): Standard CV on single-cell data
- run_low_data_benchmark(): Downsampling benchmark
- run_pseudobulk_benchmark(): CV on donor-aggregated data
- create_pseudobulk_adata(): Aggregate cells to donors
"""
import json

import argparse
import logging

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Callable

from benchmark import (
    build_representations,
    run_predictions,
    run_predictions_downsampled,
    setup_logging,
)
from preprocessing import preprocess_counts

# ============================================================================
# Single-cell benchmark functions (from modular_benchmark.py)
# ============================================================================


def run_cv_benchmark(
    adata,
    label_col: str,
    donor_col: str = "donor_id",
    embedding_keys: list[str] = None,
    n_folds: int = 5,
    alpha: float = 1e-5,
    random_state: int = 0,
    cv_folds: Optional[List[List[str]]] = None,
    **metadata,
) -> pd.DataFrame:
    """
    Run standard cross-validation benchmark on single-cell data.

    Thin wrapper around benchmark.run_predictions() for backward compatibility.

    Args:
        adata: AnnData object with raw counts in .X
        label_col: Column in adata.obs with class labels
        donor_col: Column in adata.obs with donor/sample IDs
        embedding_keys: List of embedding keys in adata.obsm (e.g., ["scvi", "geneformer"])
        n_folds: Number of CV folds
        alpha: L2 regularization strength
        random_state: Random seed
        **metadata: Additional metadata to include in results

    Returns:
        DataFrame with results for each fold × representation

    """
    representations = build_representations(adata, embedding_keys=embedding_keys)
    return run_predictions(
        adata,
        label_col=label_col,
        representations=representations,
        n_folds=n_folds,
        alpha=alpha,
        donor_col=donor_col,
        random_state=random_state,
        preprocess=True,
        cv_folds=cv_folds,
        **metadata,
    )


def run_low_data_benchmark(
    adata,
    label_col: str,
    n_per_class_list: list[int],
    donor_col: str = "donor_id",
    embedding_keys: list[str] = None,
    n_bootstrap: int = 10,
    n_folds: int = 5,
    alpha: float = 1e-5,
    random_state: int = 0,
    cv_folds: Optional[List[List[str]]] = None,
    **metadata,
) -> pd.DataFrame:
    """
    Run low-data regime benchmark with downsampling.

    Thin wrapper around benchmark.run_predictions_downsampled() for backward compatibility.

    Args:
        adata: AnnData object with raw counts in .X
        label_col: Column in adata.obs with class labels
        n_per_class_list: List of sample sizes to try (e.g., [10, 25, 50, 100])
        donor_col: Column in adata.obs with donor/sample IDs
        embedding_keys: List of embedding keys in adata.obsm
        n_bootstrap: Number of bootstrap iterations per sample size
        n_folds: Number of CV folds
        alpha: L2 regularization strength
        random_state: Random seed
        **metadata: Additional metadata to include in results

    Returns:
        DataFrame with results for each (n_per_class, bootstrap, fold, representation)

    """
    representations = build_representations(adata, embedding_keys=embedding_keys)
    return run_predictions_downsampled(
        adata,
        label_col=label_col,
        representations=representations,
        n_per_class_list=n_per_class_list,
        n_bootstrap=n_bootstrap,
        n_folds=n_folds,
        alpha=alpha,
        donor_col=donor_col,
        random_state=random_state,
        cv_folds=cv_folds,
        **metadata,
    )


# ============================================================================
# Pseudobulk benchmark functions (from pseudobulk_benchmark.py)
# ============================================================================


def create_pseudobulk_adata(
    adata: AnnData,
    donor_col: str = "donor_id",
    pooling: str = "median",
    embedding_keys: list[str] = None,
) -> AnnData:
    """
    Aggregate cells to donors (pseudobulk) via pooling.

    Args:
        adata: AnnData object with preprocessed data
        donor_col: Column in adata.obs with donor IDs
        pooling: Pooling strategy - "mean", "median", or "sum"
        embedding_keys: List of embedding keys in adata.obsm to pool (None = all)

    Returns:
        AnnData with one row per donor, aggregated expression and embeddings

    """
    assert pooling in {"mean", "median", "sum"}, f"Invalid pooling: {pooling}"

    donors = adata.obs[donor_col].values
    unique_donors = np.unique(donors)

    # Aggregate expression matrix
    X_pb_list = []
    obs_records = []

    for donor in unique_donors:
        donor_mask = donors == donor
        donor_cells = adata[donor_mask]

        # Pool expression
        X_donor = donor_cells.X
        if sparse.issparse(X_donor):
            X_donor = X_donor.toarray()

        if pooling == "mean":
            X_pooled = X_donor.mean(axis=0)
        elif pooling == "median":
            X_pooled = np.median(X_donor, axis=0)
        elif pooling == "sum":
            X_pooled = X_donor.sum(axis=0)

        X_pb_list.append(X_pooled)

        # Extract donor-level metadata (take first cell's metadata)
        donor_obs = donor_cells.obs.iloc[0].to_dict()
        obs_records.append(donor_obs)

    X_pb = np.vstack(X_pb_list)
    obs_pb = pd.DataFrame(obs_records, index=unique_donors)

    # Create pseudobulk adata
    adata_pb = AnnData(X=X_pb, obs=obs_pb, var=adata.var)

    # Aggregate embeddings
    if embedding_keys is None and adata.obsm:
        embedding_keys = list(adata.obsm.keys())

    if embedding_keys:
        for key in embedding_keys:
            if key not in adata.obsm:
                continue

            emb = adata.obsm[key]
            emb_pb_list = []

            for donor in unique_donors:
                donor_mask = donors == donor
                emb_donor = emb[donor_mask]

                if pooling == "mean":
                    emb_pooled = emb_donor.mean(axis=0)
                elif pooling == "median":
                    emb_pooled = np.median(emb_donor, axis=0)
                elif pooling == "sum":
                    emb_pooled = emb_donor.sum(axis=0)

                emb_pb_list.append(emb_pooled)

            adata_pb.obsm[key] = np.vstack(emb_pb_list)

    return adata_pb


def run_pseudobulk_benchmark(
    adata,
    label_col: str,
    donor_col: str = "donor_id",
    pooling: str = "median",
    embedding_keys: list[str] = None,
    n_folds: int = 5,
    alpha: float = 1e-5,
    random_state: int = 0,
    cv_folds: Optional[List[List[str]]] = None,
    **metadata,
) -> pd.DataFrame:
    """
    Run cross-validation on pseudobulk (donor-aggregated) data.

    This function:
    1. Preprocesses the data (if raw counts)
    2. Aggregates cells to donors via pooling
    3. Runs standard CV on donor-level data

    Args:
        adata: AnnData object with raw counts in .X
        label_col: Column in adata.obs with class labels
        donor_col: Column in adata.obs with donor IDs
        pooling: Pooling strategy - "mean", "median", or "sum"
        embedding_keys: List of embedding keys in adata.obsm
        n_folds: Number of CV folds
        alpha: L2 regularization strength
        random_state: Random seed
        **metadata: Additional metadata to include in results

    Returns:
        DataFrame with results for each fold × representation

    """
    # Preprocess
    X_lognorm, gene_mask = preprocess_counts(adata.X)
    adata_preprocessed = AnnData(
        X=X_lognorm, obs=adata.obs.copy(), var=adata.var[gene_mask].copy()
    )

    # Copy embeddings if present
    if adata.obsm and embedding_keys:
        for key in embedding_keys:
            if key in adata.obsm:
                adata_preprocessed.obsm[key] = adata.obsm[key]

    # Create pseudobulk
    adata_pb = create_pseudobulk_adata(
        adata_preprocessed,
        donor_col=donor_col,
        pooling=pooling,
        embedding_keys=embedding_keys,
    )

    # Build representations and run CV
    representations = build_representations(adata_pb, embedding_keys=embedding_keys)
    results = run_predictions(
        adata_pb,
        label_col=label_col,
        representations=representations,
        n_folds=n_folds,
        alpha=alpha,
        donor_col=donor_col,
        random_state=random_state,
        preprocess=False,  # Already preprocessed
        pooling=pooling,  # Add pooling to metadata
        cv_folds=cv_folds,
        **metadata,
    )

    return results


# ============================================================================
# CLI
# ============================================================================


def main():
    """Unified command-line interface for all benchmark modes."""
    ap = argparse.ArgumentParser(
        description="Unified scRNA-seq classification benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark modes:
  standard   - Standard cross-validation on single-cell data
  low_data   - Downsampling benchmark (low-data regime)
  pseudobulk - Donor-level classification via cell aggregation

Examples:
  # Standard CV
  python single_dataset_benchmark.py --mode standard --adata_path data.h5ad --label_col disease --out_results results.csv

  # Low-data regime
  python single_dataset_benchmark.py --mode low_data --adata_path data.h5ad --label_col disease --n_per_class 10 25 50 --out_results results.csv

  # Pseudobulk with median pooling
  python single_dataset_benchmark.py --mode pseudobulk --adata_path data.h5ad --label_col disease --pooling median --out_results results.csv
        """,
    )

    # Core arguments (required for all modes)
    ap.add_argument("--adata_path", required=True, help="Path to AnnData (.h5ad)")
    ap.add_argument("--label_col", required=True, help="Column with class labels")
    ap.add_argument(
        "--donor_col", default="donor_id", help="Column with donor/sample IDs"
    )
    ap.add_argument("--out_results", required=True, help="Output results CSV")
    ap.add_argument(
        "--log_file", default="benchmark.log", help="Log file (default: benchmark.log)"
    )

    # Mode selection
    ap.add_argument(
        "--mode",
        choices=["standard", "low_data", "pseudobulk"],
        default="standard",
        help="Benchmark mode (default: standard)",
    )

    # Embeddings (applies to all modes)
    ap.add_argument(
        "--embeddings",
        nargs="+",
        default=None,
        help="Embedding keys in adata.obsm (None = auto-detect all)",
    )

    # Cross-validation parameters (applies to all modes)
    ap.add_argument(
        "--n_folds", type=int, default=5, help="Number of CV folds (default: 5)"
    )

    ap.add_argument(
        "--cv_folds", type=str, default=None,
        help="JSON string: list of folds, each fold is a list of sample IDs"
    )
    
    ap.add_argument(
        "--alpha",
        type=float,
        default=1e-5,
        help="L2 regularization strength (default: 1e-5)",
    )
    ap.add_argument(
        "--random_state", type=int, default=0, help="Random seed (default: 0)"
    )

    # Low-data regime parameters (only for low_data mode)
    ap.add_argument(
        "--n_per_class",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="Sample sizes per class for low-data mode (default: [10, 25, 50, 100])",
    )
    ap.add_argument(
        "--n_bootstrap",
        type=int,
        default=10,
        help="Bootstrap replicates for low-data mode (default: 10)",
    )

    # Pseudobulk parameters (only for pseudobulk mode)
    ap.add_argument(
        "--pooling",
        choices=["mean", "median", "sum"],
        default="median",
        help="Pooling strategy for pseudobulk mode (default: median)",
    )

    args = ap.parse_args()

    # Setup logging
    logger = setup_logging(
        args.log_file,
        logger_name="benchmark",
        include_console=True,
        console_level=logging.INFO,
        include_thread_name=False,
        filter_warnings=True,
    )

    # Log header
    logger.info("=" * 80)
    logger.info(f"UNIFIED BENCHMARK - Mode: {args.mode.upper()}")
    logger.info(f"  Data: {args.adata_path}")
    logger.info(f"  Label: {args.label_col}, Donor: {args.donor_col}")
    if args.mode == "pseudobulk":
        logger.info(f"  Pooling: {args.pooling}")
    logger.info("=" * 80)

    # Load data
    adata = sc.read_h5ad(args.adata_path)
    logger.info(f"Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if args.cv_folds is not None:
        cv_folds: Optional[List[List[str]]] = json.loads(args.cv_folds)
    else:
        cv_folds = None

    
    if args.mode == "pseudobulk":
        n_donors = adata.obs[args.donor_col].nunique()
        logger.info(f"Found {n_donors} unique donors")

    # Run benchmark based on mode
    if args.mode == "standard":
        logger.info("Running standard CV benchmark...")
        results_df = run_cv_benchmark(
            adata,
            label_col=args.label_col,
            donor_col=args.donor_col,
            embedding_keys=args.embeddings,
            n_folds=args.n_folds,
            alpha=args.alpha,
            random_state=args.random_state,
            cv_folds=cv_folds
        )

    elif args.mode == "low_data":
        logger.info("Running low-data regime benchmark...")
        results_df = run_low_data_benchmark(
            adata,
            label_col=args.label_col,
            n_per_class_list=args.n_per_class,
            donor_col=args.donor_col,
            embedding_keys=args.embeddings,
            n_bootstrap=args.n_bootstrap,
            n_folds=args.n_folds,
            alpha=args.alpha,
            random_state=args.random_state,
            cv_folds=cv_folds
        )

    elif args.mode == "pseudobulk":
        logger.info("Running pseudobulk benchmark...")
        results_df = run_pseudobulk_benchmark(
            adata,
            label_col=args.label_col,
            donor_col=args.donor_col,
            pooling=args.pooling,
            embedding_keys=args.embeddings,
            n_folds=args.n_folds,
            alpha=args.alpha,
            random_state=args.random_state,
            cv_folds=cv_folds
        )

    # Save results
    results_df.to_csv(args.out_results, index=False)
    logger.info(f"Results saved to: {args.out_results}")
    logger.info(f"Total rows: {len(results_df)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
