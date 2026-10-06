#!/usr/bin/env python3
"""
Phase 2 ONLY: Process cached .h5ad files from Phase 1 with MULTIPLE MODES

This script skips Phase 1 (Census data fetching) and directly runs
benchmarks on the .h5ad files already downloaded to the cache directory.

Benchmark modes:
- standard: Standard cross-validation on single-cell data
- low_data: Downsampling benchmark (low-data regime)
- pseudobulk: Donor-level classification via cell aggregation

Use this when:
- Phase 1 is still running but you want to test Phase 2 on downloaded files
- You want to re-run Phase 2 with different parameters or modes
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


# =============================================================================
# COMPUTE FUNCTIONS - STANDARD MODE
# =============================================================================


def run_single_task_standard(args_tuple) -> list[dict]:
    """
    Run STANDARD CV benchmark on a single cached task.
    """
    (
        cache_path,
        embedding_keys,
        n_splits,
        alpha,
        pca_components,
        random_state,
        min_donors_per_stratum,
        baselines
    ) = args_tuple

    try:
        import time
        from benchmark import build_representations, run_predictions

        # Load cached data (already preprocessed!)
        adata = sc.read_h5ad(cache_path)

        dataset_id = adata.obs["dataset_id"].iloc[0]
        cell_type = adata.obs["cell_type"].iloc[0]
        n_genes = adata.X.shape[1]

        # Manifest guarantees min_donors_per_stratum >= 2; cap folds by it so
        # StratifiedGroupKFold never sees a stratum with fewer donors than splits.
        k_folds = min(n_splits, int(min_donors_per_stratum))

        # Build representations (raw, PCA, embeddings)
        representations = build_representations(adata, embedding_keys=embedding_keys, baselines=baselines)

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


# =============================================================================
# COMPUTE FUNCTIONS - LOW DATA MODE
# =============================================================================


def run_single_task_lowdata(args_tuple) -> list[dict]:
    """
    Run LOW-DATA CV benchmark on a single cached task.
    """
    (
        cache_path,
        embedding_keys,
        n_splits,
        alpha,
        pca_components,
        random_state,
        n_per_class_list,
        n_bootstrap,
        min_donors_per_stratum,
        baselines
    ) = args_tuple

    try:
        import time
        from benchmark import build_representations, run_predictions_downsampled

        # Load cached data (already preprocessed!)
        adata = sc.read_h5ad(cache_path)

        dataset_id = adata.obs["dataset_id"].iloc[0]
        cell_type = adata.obs["cell_type"].iloc[0]
        n_genes = adata.X.shape[1]

        k_folds = min(n_splits, int(min_donors_per_stratum))

        # Build representations (raw, PCA, embeddings)
        representations = build_representations(adata, embedding_keys=embedding_keys, baselines=baselines)

        # Run low-data benchmark
        t0 = time.perf_counter()
        results_df = run_predictions_downsampled(
            adata,
            label_col="disease",
            representations=representations,
            n_per_class_list=n_per_class_list,
            n_bootstrap=n_bootstrap,
            n_folds=k_folds,
            alpha=alpha,
            random_state=random_state,
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


# =============================================================================
# COMPUTE FUNCTIONS - PSEUDOBULK MODE
# =============================================================================


def run_single_task_pseudobulk(args_tuple) -> list[dict]:
    """
    Run PSEUDOBULK CV benchmark on a single cached task.
    """
    (
        cache_path,
        embedding_keys,
        n_splits,
        alpha,
        pca_components,
        random_state,
        pooling,
        min_donors_per_stratum,
        baselines
    ) = args_tuple

    try:
        import time
        from benchmark import build_representations, run_predictions
        from single_dataset_benchmark import create_pseudobulk_adata

        # Load cached data (already preprocessed!)
        adata = sc.read_h5ad(cache_path)

        dataset_id = adata.obs["dataset_id"].iloc[0]
        cell_type = adata.obs["cell_type"].iloc[0]
        n_genes = adata.X.shape[1]

        # Check if we have enough donors for CV
        donors = adata.obs["donor_id"].astype(str).values
        n_donors = len(np.unique(donors))
        k_folds = min(n_splits, int(min_donors_per_stratum))

        if k_folds < 2:
            return [
                {
                    "dataset_id": dataset_id,
                    "cell_type": cell_type,
                    "skip_reason": "too_few_donors_for_cv",
                }
            ]

        # Create pseudobulk data
        adata_pb = create_pseudobulk_adata(
            adata, donor_col="donor_id", pooling=pooling, embedding_keys=embedding_keys
        )
        del adata
        import gc; gc.collect()

        # Build representations (raw, PCA, embeddings)
        representations = build_representations(adata_pb, embedding_keys=embedding_keys, baselines=baselines)

        # Run benchmark on pseudobulk data
        t0 = time.perf_counter()
        results_df = run_predictions(
            adata_pb,
            label_col="disease",
            representations=representations,
            n_folds=k_folds,
            alpha=alpha,
            random_state=random_state,
            preprocess=False,  # Data already preprocessed!
            dataset_id=dataset_id,
            cell_type=cell_type,
            n_genes=n_genes,
            pooling=pooling,  # Add pooling to metadata
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


# =============================================================================
# MAIN
# =============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Phase 2 ONLY: Process cached .h5ad files with multiple modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark modes:
  standard   - Standard cross-validation on single-cell data
  low_data   - Downsampling benchmark (low-data regime)
  pseudobulk - Donor-level classification via cell aggregation

Examples:
  # Standard mode
  python phase2_only.py --mode standard --cache_dir ./_adata_cache --out_parquet results.parquet

  # Low-data mode with custom parameters
  python phase2_only.py --mode low_data --cache_dir ./_adata_cache --out_parquet results.parquet --n_per_class 5 10 25 --n_bootstrap 5

  # Pseudobulk mode
  python phase2_only.py --mode pseudobulk --cache_dir ./_adata_cache --out_parquet results.parquet --pooling median
        """,
    )

    # Core arguments
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
        help="Log file (default: benchmark_phase2_only.log)",
    )
    ap.add_argument(
        "--tasks_manifest",
        default=None,
        help="Task manifest parquet (optional, speeds up - use same file from Phase 1)",
    )

    # Mode selection
    ap.add_argument(
        "--mode",
        choices=["standard", "low_data", "pseudobulk"],
        default="standard",
        help="Benchmark mode (default: standard)",
    )
    
    ap.add_argument(
       "--baselines",
        nargs="+",
        default=None,
        help="Baseline representations to include: raw_lognorm, raw_pca50, random_proj50. Default: all. Pass none to skip all.",
    )
    # Embeddings
    ap.add_argument(
        "--embeddings",
        nargs="+",
        default=None,
        help="Embedding keys to use from obsm",
    )

    # CV parameters
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1e-5,
        help="L2 regularization parameter (default: 1e-5)",
    )
    ap.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="Number of PCA components (default: 50)",
    )
    ap.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    # Low-data mode parameters
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

    # Pseudobulk mode parameters
    ap.add_argument(
        "--pooling",
        choices=["mean", "median", "sum"],
        default="median",
        help="Pooling strategy for pseudobulk mode (default: median)",
    )

    # Parallelization
    ap.add_argument(
        "--compute_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )

    # Other options
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
        help="Flush results every N tasks (default: 50)",
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

    if args.baselines and len(args.baselines) == 1 and args.baselines[0].lower() == "none":
    	args.baselines = []

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
    logger.info(f"PHASE 2 ONLY - Mode: {args.mode.upper()}")
    logger.info(f"  Cache directory: {args.cache_dir}")
    logger.info(f"  Found cached files: {len(cache_files)}")
    logger.info(f"  Output: {args.out_parquet}")
    logger.info(f"  Temp shards: {temp_dir}")
    logger.info(f"  Workers: {args.compute_workers or os.cpu_count()}")
    if args.mode == "low_data":
        logger.info(
            f"  Low-data: n_per_class={args.n_per_class}, n_bootstrap={args.n_bootstrap}"
        )
    elif args.mode == "pseudobulk":
        logger.info(f"  Pseudobulk: pooling={args.pooling}")
    logger.info(f"{'=' * 70}")

    # Check which tasks are already completed
    completed_tasks = get_completed_tasks(
        temp_dir, retry_errors=not args.no_retry_errors
    )
    logger.info(f"  Already completed (in temp shards): {len(completed_tasks)}")

    # Filter to tasks not yet completed
    if args.tasks_manifest:
        # FAST: Use manifest (same as 02_run_benchmarks.py)
        logger.info(f"  Using task manifest: {args.tasks_manifest}")
        tasks_df = pd.read_parquet(args.tasks_manifest)
        tasks_df["task_id"] = tasks_df["dataset_id"] + "::" + tasks_df["cell_type"]

        logger.info(f"  Tasks in manifest: {len(tasks_df)}")

        
        # Manifest is the source of truth for per-task fold feasibility.
        min_donors_per_stratum_lookup = dict(
            zip(tasks_df["task_id"], tasks_df["min_donors_per_stratum"])
        )

        # Build cache paths from manifest
        remaining_files = []
        for _, row in tasks_df.iterrows():
            # Build cache filename (same logic as 02_run_benchmarks.py)
            safe_ct = row["cell_type"].replace(" ", "_").replace("/", "_").replace(":", "_")
            cache_filename = f"{row['dataset_id']}_{safe_ct}.h5ad"
            cache_path = Path(args.cache_dir) / cache_filename

            task_id = row["task_id"]

            # Only process if: cache file exists AND not already computed
            if cache_path.exists() and task_id not in completed_tasks:
                remaining_files.append((str(cache_path), min_donors_per_stratum_lookup[task_id]))

        logger.info(f"  Tasks with cache files: {len([p for p in Path(args.cache_dir).glob('*.h5ad')])} (checked via manifest)")
        
    else:
        # SLOW: Scan cache directory and open files
        logger.info("  No manifest provided, scanning cache directory...")
        logger.warning("  This is SLOW! Consider using --tasks_manifest for faster startup.")

        remaining_files = []
        for cache_path in cache_files:
            try:
                # Slow: load just obs to get task ID and donor count
                adata = sc.read_h5ad(cache_path, backed="r")
                dataset_id = str(adata.obs["dataset_id"].iloc[0])
                cell_type = str(adata.obs["cell_type"].iloc[0])
                task_id = f"{dataset_id}::{cell_type}"
                donors = adata.obs["donor_id"].astype(str).values
                n_donors = int(len(np.unique(donors)))

                if task_id not in completed_tasks:
                    remaining_files.append((cache_path, n_donors))
            except Exception as e:
                logger.warning(f"Could not read metadata from {cache_path}: {e}")
                # Include with a safe fallback donor count; will error in processing if truly broken
                remaining_files.append((cache_path, 2))

    logger.info(f"  Files to process: {len(remaining_files)}")

    if args.file_limit:
        remaining_files = remaining_files[: args.file_limit]
        logger.info(f"  Limited to first {args.file_limit} files")

    if len(remaining_files) == 0:
        logger.info("All cached files already processed!")
        logger.info("Merging existing shards...")
        merge_shards(temp_dir, args.out_parquet, logger)
        return

    # =========================================================================
    # PHASE 2: COMPUTE - MODE-SPECIFIC
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("Starting Phase 2: CV Computation")
    logger.info(f"{'=' * 70}")

    compute_workers = args.compute_workers or os.cpu_count()
    logger.info(f"Using {compute_workers} parallel workers")
    
    # Prepare task arguments based on mode. Each task carries its own
    # min_donors_per_stratum so the worker can cap folds without re-deriving feasibility.
    if args.mode == "standard":
        task_args = [
            (
                cache_path,
                args.embeddings,
                args.folds,
                args.alpha,
                args.pca_components,
                args.random_state,
                min_donors,
                args.baselines,
            )
            for cache_path, min_donors in remaining_files
        ]
        task_func = run_single_task_standard
    elif args.mode == "low_data":
        task_args = [
            (
                cache_path,
                args.embeddings,
                args.folds,
                args.alpha,
                args.pca_components,
                args.random_state,
                args.n_per_class,
                args.n_bootstrap,
                min_donors,
                args.baselines,
            )
            for cache_path, min_donors in remaining_files
        ]
        task_func = run_single_task_lowdata

    elif args.mode == "pseudobulk":
        task_args = [
            (
                cache_path,
                args.embeddings,
                args.folds,
                args.alpha,
                args.pca_components,
                args.random_state,
                args.pooling,
                min_donors,
                args.baselines,
            )
            for cache_path, min_donors in remaining_files
        ]
        task_func = run_single_task_pseudobulk

    all_results = []

    with ProcessPoolExecutor(max_workers=compute_workers) as executor:
        futures = {
            executor.submit(task_func, task_arg): task_arg[0]
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

        # Performance summary (if applicable)
        if "MCC" in final_df.columns and final_df["MCC"].notna().any():
            logger.info(f"\n{'=' * 70}")
            logger.info("PERFORMANCE SUMMARY")
            logger.info(f"{'=' * 70}")
            summary = (
                final_df.groupby("feature")["MCC"]
                .agg(["mean", "std", "count"])
                .sort_values("mean", ascending=False)
            )
            for feat, row in summary.iterrows():
                logger.info(
                    f"  {feat:25s}: MCC = {row['mean']:.3f} +/- {row['std']:.3f} (n={int(row['count'])})"
                     
                )

    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()