#!/usr/bin/env python3
"""
Modular scRNA-seq classification benchmark - Thin wrapper around unified implementation.

This module provides backward-compatible APIs and CLI for running benchmarks.
The actual implementation lives in benchmark.py, preprocessing.py, and metrics.py.

Main functions:
- run_cv_benchmark(): Run cross-validation benchmark
- run_low_data_benchmark(): Run benchmark with downsampling (low-data regime)
"""

import argparse
import logging

import pandas as pd
import scanpy as sc

from benchmark import (
    build_representations,
    run_predictions,
    run_predictions_downsampled,
    setup_logging,
)


def run_cv_benchmark(
    adata,
    label_col: str,
    donor_col: str = "donor_id",
    embedding_keys: list[str] = None,
    n_folds: int = 5,
    alpha: float = 1e-5,
    random_state: int = 0,
    **metadata,
) -> pd.DataFrame:
    """
    Run standard cross-validation benchmark.

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
        **metadata,
    )


def main():
    """Command-line interface for modular benchmark."""
    ap = argparse.ArgumentParser(
        description="Modular scRNA-seq classification benchmark"
    )
    ap.add_argument("--adata_path", required=True, help="Path to AnnData (.h5ad)")
    ap.add_argument("--label_col", required=True, help="Column with class labels")
    ap.add_argument(
        "--donor_col", default="donor_id", help="Column with donor/sample IDs"
    )
    ap.add_argument("--out_results", required=True, help="Output results CSV")
    ap.add_argument("--log_file", default="benchmark.log", help="Log file")

    # Mode selection
    ap.add_argument(
        "--mode",
        choices=["standard", "low_data"],
        default="standard",
        help="Benchmark mode: 'standard' CV or 'low_data' regime with downsampling",
    )
    ap.add_argument(
        "--embeddings",
        nargs="+",
        default=None,
        help="Embedding keys in adata.obsm (None = auto-detect)",
    )

    # Standard CV parameters
    ap.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")

    # Low-data regime parameters
    ap.add_argument(
        "--n_per_class",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="Sample sizes per class for low-data mode",
    )
    ap.add_argument(
        "--n_bootstrap",
        type=int,
        default=10,
        help="Bootstrap replicates (low-data mode)",
    )

    # Model parameters
    ap.add_argument("--alpha", type=float, default=1e-5, help="L2 regularization")
    ap.add_argument("--random_state", type=int, default=0, help="Random seed")

    args = ap.parse_args()

    # Setup
    logger = setup_logging(
        args.log_file,
        logger_name="benchmark",
        include_console=True,
        console_level=logging.INFO,
        include_thread_name=False,
        filter_warnings=True,
    )
    logger.info("=" * 80)
    logger.info(f"MODULAR BENCHMARK - Mode: {args.mode}")
    logger.info(f"  Data: {args.adata_path}")
    logger.info(f"  Label: {args.label_col}, Donor: {args.donor_col}")
    logger.info("=" * 80)

    # Load data
    adata = sc.read_h5ad(args.adata_path)
    logger.info(f"Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Run benchmark
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
        )
    else:  # low_data
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
        )

    # Save results
    results_df.to_csv(args.out_results, index=False)
    logger.info(f"Results saved to: {args.out_results}")
    logger.info(f"Total rows: {len(results_df)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
