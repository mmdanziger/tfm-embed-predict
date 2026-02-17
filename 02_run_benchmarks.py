#!/usr/bin/env python3
"""
FAST Benchmark: Two-phase approach for disease prediction.

Phase 1: Fetch data from Census → cache as .h5ad AnnData files
Phase 2: Run CV experiments in parallel (CPU bound)

Features:
- Auto-detects cached .h5ad files (no flag needed)
- Resumes from existing temp shards (same as original script)
- Full compatibility with original output format
- Uses ProcessPoolExecutor for true parallelism
"""

import argparse
import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
import threading
import traceback
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cellxgene_census
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from tqdm.auto import tqdm

# Will be set from args
FETCH_SEMAPHORE = None


from benchmark import setup_logging
from preprocessing import sparse_normalize_log1p


def get_task_id(dataset_id: str, cell_type: str) -> str:
    """Create consistent task ID."""
    return f"{dataset_id}::{cell_type}"


def get_cache_filename(dataset_id: str, cell_type: str) -> str:
    """Create safe filename for cache."""
    safe_ct = cell_type.replace(" ", "_").replace("/", "_").replace(":", "_")
    return f"{dataset_id}_{safe_ct}.h5ad"


def validate_task_feasibility(
    y: np.ndarray, donors: np.ndarray
) -> tuple[bool, str, dict]:
    """Check if task has enough data for CV."""
    donor_class = {}
    for d, label in zip(donors, y):
        if d not in donor_class:
            donor_class[d] = label

    class_donor_counts = Counter(donor_class.values())
    n_classes = len(class_donor_counts)
    n_donors = len(donor_class)
    min_donors = min(class_donor_counts.values()) if class_donor_counts else 0

    stats = {
        "n_classes": n_classes,
        "n_donors": n_donors,
        "min_donors_per_class": min_donors,
    }

    if n_classes < 2:
        return False, "too_few_classes", stats
    if min_donors < 2:
        rare = [c for c, cnt in class_donor_counts.items() if cnt < 2]
        return False, f"classes_with_lt2_donors:{rare}", stats
    if n_donors < 2:
        return False, "too_few_donors", stats

    return True, "ok", stats


# =============================================================================
# PHASE 1: FETCH DATA
# =============================================================================


def fetch_single_task_with_census(
    census,
    dataset_id: str,
    cell_type: str,
    embedding_keys: list[str],
    cache_dir: str,
) -> dict:
    """Fetch data for a single task using shared Census connection."""
    task_id = get_task_id(dataset_id, cell_type)
    cache_path = Path(cache_dir) / get_cache_filename(dataset_id, cell_type)

    # Already cached?
    if cache_path.exists():
        return {"task_id": task_id, "status": "cached", "cache_path": str(cache_path)}

    try:
        obs_filter = f"is_primary_data==True and dataset_id=='{dataset_id}' and cell_type=='{cell_type}'"

        # Use semaphore to prevent TileDB context exhaustion
        with FETCH_SEMAPHORE:
            adata = cellxgene_census.get_anndata(
                census=census,
                organism="homo_sapiens",
                measurement_name="RNA",
                obs_value_filter=obs_filter,
                obs_column_names=["dataset_id", "cell_type", "donor_id", "disease"],
                obs_embeddings=embedding_keys,
            )

        if adata.n_obs == 0:
            return {"task_id": task_id, "status": "skip", "reason": "no_cells"}

        # Validate feasibility before saving
        y = adata.obs["disease"].astype(str).values
        donors = adata.obs["donor_id"].astype(str).values

        is_feasible, reason, stats = validate_task_feasibility(y, donors)
        if not is_feasible:
            return {"task_id": task_id, "status": "skip", "reason": reason, **stats}

        # Preprocess expression: filter zero genes, normalize, log1p
        X = (
            sparse.csr_matrix(adata.X)
            if not sparse.issparse(adata.X)
            else adata.X.tocsr()
        )
        gene_mask = np.array(X.sum(axis=0)).flatten() > 0
        X = X[:, gene_mask]
        X_lognorm = sparse_normalize_log1p(X)

        # Store processed data
        adata_processed = AnnData(
            X=X_lognorm,
            obs=adata.obs[["dataset_id", "cell_type", "donor_id", "disease"]].copy(),
            obsm={k: adata.obsm[k] for k in embedding_keys if k in adata.obsm},
        )
        adata_processed.uns["n_genes_original"] = int(gene_mask.sum())

        # Save
        adata_processed.write_h5ad(cache_path)

        # Clean up
        del adata
        gc.collect()

        return {
            "task_id": task_id,
            "status": "fetched",
            "cache_path": str(cache_path),
            "n_cells": adata_processed.n_obs,
            **stats,
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# PHASE 2: COMPUTE CV
# =============================================================================


def run_single_task(args_tuple) -> list[dict]:
    """Run CV benchmark on a single cached task. Returns list of result dicts."""
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


# =============================================================================
# RESUME LOGIC (compatible with original script)
# =============================================================================


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
    logger.info(f"[FLUSH] Wrote {len(df)} rows to {batch_id}")


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
# MAIN
# =============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Fast two-phase disease prediction benchmark"
    )
    ap.add_argument("--tasks_manifest", required=True, help="Task manifest parquet")
    ap.add_argument("--out_parquet", required=True, help="Output results parquet")
    ap.add_argument("--log_file", default="benchmark_fast.log", help="Log file")
    ap.add_argument("--census_uri", default=None, help="Census URI (None for S3)")
    ap.add_argument("--census_version", default="2025-01-30", help="Census version")
    ap.add_argument(
        "--embeddings",
        nargs="+",
        default=["scvi", "geneformer", "tf-sapiens", "tf-exemplar-human"],
    )
    ap.add_argument(
        "--cache_dir", default="./_adata_cache", help="Directory for cached .h5ad files"
    )
    ap.add_argument("--folds", type=int, default=5, help="CV folds")
    ap.add_argument("--alpha", type=float, default=1e-5, help="L2 regularization")
    ap.add_argument(
        "--fetch_workers", type=int, default=4, help="Parallel fetch threads"
    )
    ap.add_argument(
        "--fetch_semaphore",
        type=int,
        default=2,
        help="Max concurrent Census fetches (TileDB limit)",
    )
    ap.add_argument(
        "--compute_workers",
        type=int,
        default=None,
        help="Parallel compute workers (default: CPU count)",
    )
    ap.add_argument(
        "--task_limit", type=int, default=None, help="Limit tasks (for testing)"
    )
    ap.add_argument(
        "--keep_temp", action="store_true", help="Keep temp shards after merge"
    )
    ap.add_argument(
        "--no_retry_errors", action="store_true", help="Don't retry errored tasks"
    )
    ap.add_argument(
    "--resume_from", 
    type=int, 
    default=0, 
    help="Resume from task N in manifest (skip first N tasks)"
    )
    args = ap.parse_args()

    logger = setup_logging(args.log_file)

    # Initialize fetch semaphore
    global FETCH_SEMAPHORE
    FETCH_SEMAPHORE = threading.Semaphore(args.fetch_semaphore)

    # Paths
    temp_dir = f"{os.path.dirname(args.out_parquet) or '.'}/_temp_shards_{Path(args.out_parquet).stem}"
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Load manifest
    tasks_df = pd.read_parquet(args.tasks_manifest)
    if args.task_limit:
        tasks_df = tasks_df.head(args.task_limit)

    # Check for completed tasks (resume support)
    completed_tasks = get_completed_tasks(
        temp_dir, retry_errors=not args.no_retry_errors
    )

    logger.info(f"{'=' * 70}")
    logger.info("FAST BENCHMARK")
    logger.info(f"  Total tasks in manifest: {len(tasks_df)}")
    logger.info(f"  Already completed: {len(completed_tasks)}")
    logger.info(f"  Cache dir: {args.cache_dir}")
    logger.info(f"  Temp dir: {temp_dir}")
    logger.info(
        f"  Fetch: {args.fetch_workers} threads, semaphore={args.fetch_semaphore}"
    )
    logger.info(f"{'=' * 70}")

    # Filter to remaining tasks
    tasks_df["task_id"] = tasks_df["dataset_id"] + "::" + tasks_df["cell_type"]
    remaining_df = tasks_df[~tasks_df["task_id"].isin(completed_tasks)].copy()
    logger.info(f"Tasks to process: {len(remaining_df)}")

    if args.resume_from > 0:
        logger.info(f"  RESUMING: Skipping first {args.resume_from} tasks from ORIGINAL manifest")
        # Skip based on original tasks_df, not remaining_df
        tasks_df_filtered = tasks_df.iloc[args.resume_from:]
        remaining_df = tasks_df_filtered[~tasks_df_filtered["task_id"].isin(completed_tasks)].copy()
        logger.info(f"  Tasks after resume skip: {len(remaining_df)}")
    else:
        logger.info(f"Tasks to process: {len(remaining_df)}")



    if len(remaining_df) == 0:
        logger.info("All tasks complete. Merging shards...")
        merge_shards(temp_dir, args.out_parquet, logger)
        return

    # =========================================================================
    # PHASE 1: FETCH (only for tasks not in cache)
    # Uses shared Census connection with ThreadPoolExecutor (I/O bound)
    # =========================================================================
    logger.info("\n[PHASE 1] Checking cache and fetching missing data...")

    # Check which tasks need fetching
    tasks_to_fetch = []
    cached_paths = {}

    for _, row in remaining_df.iterrows():
        cache_path = Path(args.cache_dir) / get_cache_filename(
            row["dataset_id"], row["cell_type"]
        )
        if cache_path.exists():
            cached_paths[row["task_id"]] = str(cache_path)
        else:
            tasks_to_fetch.append(row)

    logger.info(f"  Already cached: {len(cached_paths)}")
    logger.info(f"  Need to fetch: {len(tasks_to_fetch)}")

    # Fetch missing tasks with SHARED Census connection
    if tasks_to_fetch:
        skip_results = []

        logger.info("  Opening Census connection...")
        with cellxgene_census.open_soma(
            uri=args.census_uri, census_version=args.census_version
        ) as census:
            logger.info(
                f"  Census ready. Fetching with {args.fetch_workers} workers..."
            )

            with ThreadPoolExecutor(max_workers=args.fetch_workers) as executor:
                futures = {
                    executor.submit(
                        fetch_single_task_with_census,
                        census,
                        row["dataset_id"],
                        row["cell_type"],
                        args.embeddings,
                        args.cache_dir,
                    ): row["task_id"]
                    for row in tasks_to_fetch
                }

                with tqdm(total=len(futures), desc="Fetching", unit="task") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        task_id = result["task_id"]

                        if (
                            result["status"] == "fetched"
                            or result["status"] == "cached"
                        ):
                            cached_paths[task_id] = result["cache_path"]
                        elif result["status"] == "skip":
                            parts = task_id.split("::", 1)
                            skip_results.append(
                                {
                                    "dataset_id": parts[0],
                                    "cell_type": parts[1] if len(parts) > 1 else "",
                                    "skip_reason": result.get("reason", "unknown"),
                                }
                            )
                        elif result["status"] == "error":
                            parts = task_id.split("::", 1)
                            skip_results.append(
                                {
                                    "dataset_id": parts[0],
                                    "cell_type": parts[1] if len(parts) > 1 else "",
                                    "error": result.get("error", "unknown"),
                                }
                            )

                        pbar.set_postfix_str(f"{result['status']}")
                        pbar.update(1)

        # Flush skip/error results
        if skip_results:
            flush_results(skip_results, temp_dir, logger)

    # =========================================================================
    # PHASE 2: COMPUTE (fully parallel)
    # Skip tasks that are already in temp shards
    # =========================================================================
    logger.info("\n[PHASE 2] Running CV benchmarks...")

    compute_workers = args.compute_workers or mp.cpu_count()
    logger.info(f"  Using {compute_workers} compute workers")
    logger.info(f"  Tasks with cached data: {len(cached_paths)}")

    # Check which cached tasks are already computed
    already_computed = get_completed_tasks(
        temp_dir, retry_errors=not args.no_retry_errors
    )
    tasks_to_compute = {
        task_id: path
        for task_id, path in cached_paths.items()
        if task_id not in already_computed
    }

    logger.info(f"  Already computed: {len(cached_paths) - len(tasks_to_compute)}")
    logger.info(f"  Need to compute: {len(tasks_to_compute)}")

    if not tasks_to_compute:
        logger.info("All cached tasks already computed.")
        merge_shards(temp_dir, args.out_parquet, logger)
        return

    compute_args = [
        (path, args.embeddings, args.folds, args.alpha, 50, 0)
        for path in tasks_to_compute.values()
    ]

    all_results = []
    tasks_computed = 0

    with ProcessPoolExecutor(max_workers=compute_workers) as executor:
        futures = {
            executor.submit(run_single_task, arg): arg[0] for arg in compute_args
        }

        with tqdm(total=len(futures), desc="Computing", unit="task") as pbar:
            for future in as_completed(futures):
                task_results = future.result()
                all_results.extend(task_results)
                tasks_computed += 1

                # Flush every 20 TASKS (not 500 rows) to avoid losing work
                if tasks_computed % 20 == 0:
                    flush_results(all_results, temp_dir, logger)
                    all_results = []

                pbar.update(1)

    # Final flush
    if all_results:
        flush_results(all_results, temp_dir, logger)

    # =========================================================================
    # MERGE
    # =========================================================================
    final_df = merge_shards(temp_dir, args.out_parquet, logger)

    # Summary
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
                f"  {feat:25s}: MCC = {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count'])})"
            )

    # Cleanup
    if not args.keep_temp:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up {temp_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
