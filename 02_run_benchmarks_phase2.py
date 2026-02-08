#!/usr/bin/env python3
"""
Phase 2 ONLY: Process cached .h5ad files from Phase 1

This script skips Phase 1 (Census data fetching) and directly runs
benchmarks on the .h5ad files already downloaded to the cache directory.

Use this when:
- Phase 1 is still running but you want to test Phase 2 on downloaded files
- You want to re-run Phase 2 with different parameters
- You already have cached data from a previous run

The script will:
1. Find all .h5ad files in cache directory
2. Skip files already processed (checks temp shards)
3. Run CV benchmarks in parallel using ProcessPoolExecutor
4. Save results to output parquet file
"""

import argparse
import glob
import logging
import os
import shutil
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm

from benchmark import setup_logging


def get_completed_tasks(temp_dir: str, retry_errors: bool = True) -> set[str]:
    """Scan temp shards to find completed tasks."""
    if not os.path.exists(temp_dir):
        return set()

    completed = set()
    for shard_file in glob.glob(f"{temp_dir}/batch_*.parquet"):
        try:
            df = pd.read_parquet(shard_file)
            if "cell_type" not in df.columns or "dataset_id" not in df.columns:
                continue

            has_results = (
                df["MCC"].notna()
                if "MCC" in df.columns
                else pd.Series(False, index=df.index)
            )
            has_skip = (
                df["skip_reason"].notna()
                if "skip_reason" in df.columns
                else pd.Series(False, index=df.index)
            )
            has_error = (
                df["error"].notna()
                if "error" in df.columns
                else pd.Series(False, index=df.index)
            )

            if retry_errors:
                mask = has_results | has_skip
            else:
                mask = has_results | has_skip | has_error

            task_ids = df.loc[mask, "dataset_id"] + "::" + df.loc[mask, "cell_type"]
            completed.update(task_ids.unique())
        except Exception:
            continue

    return completed


def flush_results(results: list[dict], temp_dir: str, logger: logging.Logger) -> None:
    """Write results to a new shard file."""
    if not results:
        return

    batch_id = str(uuid.uuid4())[:8]
    batch_path = f"{temp_dir}/batch_{batch_id}.parquet"

    df = pd.DataFrame(results)
    df.to_parquet(batch_path, index=False)
    logger.info(f"[FLUSH] Wrote {len(df)} rows to batch {batch_id}")


def merge_shards(temp_dir: str, out_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Merge all shard files into final output."""
    shard_files = sorted(glob.glob(f"{temp_dir}/batch_*.parquet"))

    if not shard_files:
        logger.warning("No shard files found")
        return pd.DataFrame()

    logger.info(f"Merging {len(shard_files)} shards...")

    dfs = []
    for f in shard_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    final = pd.concat(dfs, ignore_index=True)
    final.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(final)} rows to {out_path}")

    return final


def run_single_task(args_tuple) -> list[dict]:
    """
    Run CV benchmark on a single cached task.
    
    This is the same function from 02_run_benchmarks.py
    """
    cache_path, embedding_keys, n_splits, alpha, pca_components, random_state = (
        args_tuple
    )

    try:
        import time
        from benchmark import build_representations, run_predictions

        # Load cached data (already preprocessed!)
        adata = sc.read_h5ad(cache_path)

        dataset_id = adata.obs["dataset_id"].iloc[0]
        cell_type = adata.obs["cell_type"].iloc[0]
        n_genes = adata.X.shape[1]

        # Check if we have enough donors for CV
        donors = adata.obs["donor_id"].astype(str).values
        n_donors = len(np.unique(donors))
        k_folds = min(n_splits, n_donors)

        if k_folds < 2:
            return [
                {
                    "dataset_id": dataset_id,
                    "cell_type": cell_type,
                    "skip_reason": "too_few_donors_for_cv",
                }
            ]

        # Build representations (raw, PCA, embeddings)
        representations = build_representations(adata, embedding_keys=embedding_keys)

        # Run benchmark using unified implementation
        t0 = time.perf_counter()
        results_df = run_predictions(
            adata,
            label_col="disease",
            representations=representations,
            n_folds=k_folds,
            alpha=alpha,
            random_state=random_state,
            preprocess=False,  # Data already preprocessed in cache!
            dataset_id=dataset_id,
            cell_type=cell_type,
            n_genes=n_genes,
        )

        # Convert DataFrame to list of dicts for compatibility
        results = results_df.to_dict(orient="records")

        # Rename 'representation' column to 'feature' for backward compatibility
        for r in results:
            r["feature"] = r.pop("representation")

        if not results:
            return [
                {
                    "dataset_id": dataset_id,
                    "cell_type": cell_type,
                    "skip_reason": "all_folds_degenerate",
                }
            ]

        return results

    except Exception as e:
        return [
            {
                "cache_path": cache_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }
        ]


def main():
    ap = argparse.ArgumentParser(
        description="Phase 2 ONLY: Process cached .h5ad files"
    )
    ap.add_argument(
        "--cache_dir",
        required=True,
        help="Directory with cached .h5ad files from Phase 1 (e.g., ./_adata_cache)",
    )
    ap.add_argument(
        "--out_parquet",
        required=True,
        help="Output results parquet file",
    )
    ap.add_argument(
        "--log_file",
        default="benchmark_phase2_only.log",
        help="Log file",
    )
    ap.add_argument(
        "--embeddings",
        nargs="+",
        default=["scvi", "geneformer", "tf-sapiens", "tf-exemplar-human"],
        help="Embedding keys to use from obsm",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1e-5,
        help="L2 regularization parameter",
    )
    ap.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="Number of PCA components",
    )
    ap.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed",
    )
    ap.add_argument(
        "--compute_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    ap.add_argument(
        "--file_limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    ap.add_argument(
        "--flush_every",
        type=int,
        default=50,
        help="Flush results every N tasks",
    )
    ap.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temp shards after merging",
    )
    ap.add_argument(
        "--no_retry_errors",
        action="store_true",
        help="Don't retry tasks that previously errored",
    )
    args = ap.parse_args()

    logger = setup_logging(args.log_file)

    # Setup paths
    temp_dir = f"{os.path.dirname(args.out_parquet) or '.'}/_temp_shards_{Path(args.out_parquet).stem}"
    os.makedirs(temp_dir, exist_ok=True)

    # Find all cached .h5ad files
    cache_files = sorted(glob.glob(f"{args.cache_dir}/*.h5ad"))

    if not cache_files:
        logger.error(f"ERROR: No .h5ad files found in {args.cache_dir}")
        logger.error("Make sure Phase 1 has downloaded some files first!")
        logger.error(f"Looking for files matching: {args.cache_dir}/*.h5ad")
        return

    logger.info(f"{'=' * 70}")
    logger.info("PHASE 2 ONLY - Processing Cached .h5ad Files")
    logger.info(f"  Cache directory: {args.cache_dir}")
    logger.info(f"  Found cached files: {len(cache_files)}")
    logger.info(f"  Output: {args.out_parquet}")
    logger.info(f"  Temp shards: {temp_dir}")
    logger.info(f"  Workers: {args.compute_workers or os.cpu_count()}")
    logger.info(f"{'=' * 70}")

    # Check which tasks are already completed
    completed_tasks = get_completed_tasks(temp_dir, retry_errors=not args.no_retry_errors)
    logger.info(f"  Already completed (in temp shards): {len(completed_tasks)}")

    # Filter to tasks not yet completed
    remaining_files = []
    for cache_path in cache_files:
        try:
            # Quick check: load just obs to get task ID
            adata = sc.read_h5ad(cache_path, backed='r')
            dataset_id = str(adata.obs["dataset_id"].iloc[0])
            cell_type = str(adata.obs["cell_type"].iloc[0])
            task_id = f"{dataset_id}::{cell_type}"
            
            if task_id not in completed_tasks:
                remaining_files.append(cache_path)
        except Exception as e:
            logger.warning(f"Could not read metadata from {cache_path}: {e}")
            # Include it anyway - will error in processing if truly broken
            remaining_files.append(cache_path)

    logger.info(f"  Files to process: {len(remaining_files)}")

    if args.file_limit:
        remaining_files = remaining_files[:args.file_limit]
        logger.info(f"  Limited to first {args.file_limit} files")

    if len(remaining_files) == 0:
        logger.info("All cached files already processed!")
        logger.info("Merging existing shards...")
        merge_shards(temp_dir, args.out_parquet, logger)
        return

    # =========================================================================
    # PHASE 2: COMPUTE
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("Starting Phase 2: CV Computation")
    logger.info(f"{'=' * 70}")

    compute_workers = args.compute_workers or os.cpu_count()
    logger.info(f"Using {compute_workers} parallel workers")

    # Prepare task arguments
    task_args = [
        (
            cache_path,
            args.embeddings,
            args.folds,
            args.alpha,
            args.pca_components,
            args.random_state,
        )
        for cache_path in remaining_files
    ]

    all_results = []

    with ProcessPoolExecutor(max_workers=compute_workers) as executor:
        futures = {
            executor.submit(run_single_task, task_arg): task_arg[0]
            for task_arg in task_args
        }

        with tqdm(total=len(futures), desc="Computing CV", unit="file") as pbar:
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    results = future.result()
                    all_results.extend(results)

                    # Periodic flush
                    if idx % args.flush_every == 0:
                        flush_results(all_results, temp_dir, logger)
                        all_results = []

                    pbar.update(1)

                except Exception as e:
                    cache_path = futures[future]
                    logger.error(f"Failed to process {cache_path}: {e}")
                    pbar.update(1)

    # Final flush
    if all_results:
        flush_results(all_results, temp_dir, logger)

    # =========================================================================
    # MERGE RESULTS
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("Merging all shard files...")
    logger.info(f"{'=' * 70}")

    final_df = merge_shards(temp_dir, args.out_parquet, logger)

    # Cleanup
    if not args.keep_temp:
        logger.info(f"Cleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("PHASE 2 COMPLETE!")
    logger.info(f"  Output file: {args.out_parquet}")
    logger.info(f"  Total rows: {len(final_df)}")
    
    if len(final_df) > 0:
        # Summary stats
        if "MCC" in final_df.columns:
            n_success = final_df["MCC"].notna().sum()
            logger.info(f"  Successful runs: {n_success}")
        if "skip_reason" in final_df.columns:
            n_skipped = final_df["skip_reason"].notna().sum()
            logger.info(f"  Skipped: {n_skipped}")
        if "error" in final_df.columns:
            n_errors = final_df["error"].notna().sum()
            logger.info(f"  Errors: {n_errors}")
    
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
