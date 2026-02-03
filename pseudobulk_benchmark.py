#!/usr/bin/env python3
"""
Pseudobulk (donor-level) classification benchmark - Thin wrapper around unified implementation.

This module provides functions to aggregate cells to donors (pseudobulk) and run benchmarks.
The actual CV implementation lives in benchmark.py.

Main functions:
- create_pseudobulk_adata(): Aggregate cells to donors
- run_pseudobulk_benchmark(): Run CV on pseudobulk data
"""

import argparse
import logging

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from benchmark import build_representations, run_predictions, setup_logging
from preprocessing import preprocess_counts


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
        **metadata,
    )

    return results


def main():
    """Command-line interface for pseudobulk benchmark."""
    ap = argparse.ArgumentParser(description="Pseudobulk (donor-level) benchmark")
    ap.add_argument("--adata_path", required=True, help="Path to AnnData (.h5ad)")
    ap.add_argument("--label_col", required=True, help="Column with class labels")
    ap.add_argument(
        "--donor_col", default="donor_id", help="Column with donor/sample IDs"
    )
    ap.add_argument("--out_results", required=True, help="Output results CSV")
    ap.add_argument("--log_file", default="pseudobulk_benchmark.log", help="Log file")

    # Pseudobulk-specific parameters
    ap.add_argument(
        "--pooling",
        choices=["mean", "median", "sum"],
        default="median",
        help="Pooling strategy for aggregating cells to donors",
    )
    ap.add_argument(
        "--embeddings",
        nargs="+",
        default=None,
        help="Embedding keys in obsm to test (None = auto-detect all)",
    )

    # Standard CV parameters
    ap.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    ap.add_argument("--alpha", type=float, default=1e-5, help="L2 regularization")
    ap.add_argument("--random_state", type=int, default=0, help="Random seed")

    args = ap.parse_args()

    # Setup
    logger = setup_logging(
        args.log_file,
        logger_name="pseudobulk",
        include_console=True,
        console_level=logging.INFO,
        include_thread_name=False,
        filter_warnings=True,
    )
    logger.info("=" * 80)
    logger.info("PSEUDOBULK BENCHMARK STARTED")
    logger.info(f"  Data: {args.adata_path}")
    logger.info(f"  Label: {args.label_col}, Donor: {args.donor_col}")
    logger.info(f"  Pooling mode: {args.pooling}")
    logger.info("=" * 80)

    # Load data
    adata = sc.read_h5ad(args.adata_path)
    logger.info(f"Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Count donors
    n_donors = adata.obs[args.donor_col].nunique()
    logger.info(f"Found {n_donors} unique donors")

    # Run benchmark
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
    )

    # Save results
    results_df.to_csv(args.out_results, index=False)
    logger.info(f"Results saved to: {args.out_results}")
    logger.info(f"Total rows: {len(results_df)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
