#!/usr/bin/env python3
"""
Benchmark transcriptomic foundation models vs raw baselines for cell type prediction.
NOW WITH MULTIPLE MODES: standard, low_data, pseudobulk

Task: dataset_id → predict cell_type (across held-out donors)
Evaluation: Donor-stratified K-fold cross-validation (stratified on disease)

Modes:
- standard: Standard cross-validation on single-cell data
- low_data: Downsampling benchmark (low-data regime)
- pseudobulk: Donor-level classification via cell aggregation

Uses the same .h5ad cache filled by 02_run_benchmarks.py Phase 1.
"""

import argparse
import gc
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
from tqdm.auto import tqdm

from benchmark import setup_logging
from cache_io import load_and_concat_celltype_caches


def get_cache_filename(dataset_id: str, cell_type: str) -> str:
    """Return the cache filename for a (dataset_id, cell_type) pair."""
    safe_ct = cell_type.replace(" ", "_").replace("/", "_").replace(":", "_")
    return f"{dataset_id}_{safe_ct}.h5ad"


# =============================================================================
# MODE-SPECIFIC BENCHMARK FUNCTIONS
# =============================================================================


def benchmark_one_task_standard(args_tuple) -> pd.DataFrame:
    """STANDARD mode: Run cell type prediction with standard CV."""
    task_row, cache_dir, embedding_keys, n_splits, alpha, random_state = args_tuple

    dataset_id = task_row["dataset_id"]
    eligible_cell_types = task_row["cell_type"]
    logger = logging.getLogger("benchmark")

    try:
        # Load and concatenate cache files
        missing = []
        cache_paths = []
        for ct in eligible_cell_types:
            fpath = os.path.join(cache_dir, get_cache_filename(dataset_id, ct))
            if not os.path.exists(fpath):
                missing.append(ct)
            else:
                cache_paths.append(fpath)

        if missing:
            logger.warning(f"[SKIP] {dataset_id}: Missing {len(missing)} cache files")
            return pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": "missing_cache_files",
                        "n_missing": len(missing),
                    }
                ]
            )

        adata = load_and_concat_celltype_caches(cache_paths)
        gc.collect()

        if adata.n_obs == 0:
            return pd.DataFrame([{"dataset_id": dataset_id, "skip_reason": "no_cells"}])

        n_cell_types = int(adata.obs["cell_type"].nunique())
        n_genes = adata.n_vars
        k_folds = min(n_splits, int(task_row["min_donors_per_stratum"]))

        # Run benchmark
        from benchmark import build_representations, run_predictions

        representations = build_representations(adata, embedding_keys=embedding_keys)

        results_df = run_predictions(
            adata,
            label_col="cell_type",
            representations=representations,
            n_folds=k_folds,
            alpha=alpha,
            donor_col="donor_id",
            stratify_col="disease",
            random_state=random_state,
            preprocess=False,
            dataset_id=dataset_id,
            n_genes=n_genes,
            n_cell_types=n_cell_types,
        )

        results = results_df.to_dict(orient="records")
        for r in results:
            r["feature"] = r.pop("representation")

        return (
            pd.DataFrame(results)
            if results
            else pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": "all_folds_degenerate",
                    }
                ]
            )
        )

    except Exception as e:
        logger.error(f"[FAILED] {dataset_id}: {e}")
        return pd.DataFrame(
            [
                {
                    "dataset_id": dataset_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
            ]
        )
    finally:
        gc.collect()


def benchmark_one_task_lowdata(args_tuple) -> pd.DataFrame:
    """LOW_DATA mode: Run cell type prediction with downsampling."""
    (
        task_row,
        cache_dir,
        embedding_keys,
        n_splits,
        alpha,
        random_state,
        n_per_class,
        n_bootstrap,
    ) = args_tuple

    dataset_id = task_row["dataset_id"]
    eligible_cell_types = task_row["cell_type"]
    logger = logging.getLogger("benchmark")

    try:
        # Load and concatenate (same as standard)
        missing = []
        cache_paths = []
        for ct in eligible_cell_types:
            fpath = os.path.join(cache_dir, get_cache_filename(dataset_id, ct))
            if not os.path.exists(fpath):
                missing.append(ct)
            else:
                cache_paths.append(fpath)

        if missing:
            return pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": "missing_cache_files",
                        "n_missing": len(missing),
                    }
                ]
            )

        adata = load_and_concat_celltype_caches(cache_paths)
        gc.collect()

        if adata.n_obs == 0:
            return pd.DataFrame([{"dataset_id": dataset_id, "skip_reason": "no_cells"}])

        n_cell_types = int(adata.obs["cell_type"].nunique())
        n_genes = adata.n_vars
        k_folds = min(n_splits, int(task_row["min_donors_per_stratum"]))

        # Run low-data benchmark
        from benchmark import build_representations, run_predictions_downsampled

        representations = build_representations(adata, embedding_keys=embedding_keys)

        results_df = run_predictions_downsampled(
            adata,
            label_col="cell_type",
            representations=representations,
            n_folds=k_folds,
            alpha=alpha,
            donor_col="donor_id",
            stratify_col="disease",
            random_state=random_state,
            n_per_class_list=n_per_class,
            n_bootstrap=n_bootstrap,
            dataset_id=dataset_id,
            n_genes=n_genes,
            n_cell_types=n_cell_types,
        )

        results = results_df.to_dict(orient="records")
        for r in results:
            r["feature"] = r.pop("representation")

        return (
            pd.DataFrame(results)
            if results
            else pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": "all_folds_degenerate",
                    }
                ]
            )
        )

    except Exception as e:
        logger.error(f"[FAILED] {dataset_id}: {e}")
        return pd.DataFrame(
            [
                {
                    "dataset_id": dataset_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
            ]
        )
    finally:
        gc.collect()


def create_pseudobulk_adata_for_celltype(
    adata,
    donor_col="donor_id",
    celltype_col="cell_type",
    pooling="median",
):
    """
    Create pseudobulk by aggregating cells within each (donor, cell_type) pair.

    This preserves cell type labels for cell type prediction tasks.
    """
    import pandas as pd
    from scipy import sparse

    # Group by (donor_id, cell_type)
    adata.obs["_pseudobulk_group"] = (
        adata.obs[donor_col].astype(str) + ":::" + adata.obs[celltype_col].astype(str)
    )

    groups = adata.obs["_pseudobulk_group"].unique()

    pseudobulk_profiles = []
    metadata = []

    for group in groups:
        mask = adata.obs["_pseudobulk_group"] == group
        cells = adata[mask]

        # Aggregate expression
        X_group = cells.X
        if sparse.issparse(X_group):
            X_group = X_group.toarray()

        if pooling == "mean":
            profile = X_group.mean(axis=0)
        elif pooling == "median":
            profile = np.median(X_group, axis=0)
        elif pooling == "sum":
            profile = X_group.sum(axis=0)

        pseudobulk_profiles.append(profile)

        # Keep metadata
        donor_id = cells.obs[donor_col].iloc[0]
        cell_type = cells.obs[celltype_col].iloc[0]
        disease = cells.obs["disease"].iloc[0] if "disease" in cells.obs else "unknown"

        metadata.append(
            {
                donor_col: donor_id,
                celltype_col: cell_type,
                "disease": disease,
                "n_cells_aggregated": cells.n_obs,
            }
        )

    # Create new AnnData
    X_pb = np.vstack(pseudobulk_profiles)
    obs_pb = pd.DataFrame(metadata)

    # Handle embeddings
    obsm_pb = {}
    if adata.obsm:
        for key in adata.obsm.keys():
            group_embeddings = []
            for group in groups:
                mask = adata.obs["_pseudobulk_group"] == group
                emb = adata.obsm[key][mask]

                if pooling == "mean":
                    agg_emb = emb.mean(axis=0)
                elif pooling == "median":
                    agg_emb = np.median(emb, axis=0)
                else:
                    agg_emb = emb.mean(axis=0)

                group_embeddings.append(agg_emb)

            obsm_pb[key] = np.vstack(group_embeddings)

    from anndata import AnnData

    adata_pb = AnnData(
        X=X_pb,
        obs=obs_pb,
        var=adata.var.copy(),
        obsm=obsm_pb,
    )

    return adata_pb


def benchmark_one_task_pseudobulk(args_tuple) -> pd.DataFrame:
    """PSEUDOBULK mode: Aggregate by (donor, cell_type) pairs."""
    task_row, cache_dir, embedding_keys, n_splits, alpha, random_state, pooling = (
        args_tuple
    )

    dataset_id = task_row["dataset_id"]
    eligible_cell_types = task_row["cell_type"]
    logger = logging.getLogger("benchmark")

    try:
        # Load and concatenate (same as before)
        missing = []
        cache_paths = []
        for ct in eligible_cell_types:
            fpath = os.path.join(cache_dir, get_cache_filename(dataset_id, ct))
            if not os.path.exists(fpath):
                missing.append(ct)
            else:
                cache_paths.append(fpath)

        if missing:
            return pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": "missing_cache_files",
                        "n_missing": len(missing),
                    }
                ]
            )

        adata = load_and_concat_celltype_caches(cache_paths)
        gc.collect()

        # Create pseudobulk: aggregate by (donor, cell_type)
        adata_pb = create_pseudobulk_adata_for_celltype(
            adata,
            donor_col="donor_id",
            celltype_col="cell_type",
            pooling=pooling,
        )
        del adata
        gc.collect()

        n_cell_types = int(adata_pb.obs["cell_type"].nunique())
        n_genes = adata_pb.n_vars
        k_folds = min(n_splits, int(task_row["min_donors_per_stratum"]))

        # Run benchmark on pseudobulk
        from benchmark import build_representations, run_predictions

        representations = build_representations(adata_pb, embedding_keys=embedding_keys)

        results_df = run_predictions(
            adata_pb,
            label_col="cell_type",
            representations=representations,
            n_folds=k_folds,
            alpha=alpha,
            donor_col="donor_id",
            stratify_col="disease",
            random_state=random_state,
            preprocess=False,
            dataset_id=dataset_id,
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            pooling=pooling,
        )

        results = results_df.to_dict(orient="records")
        for r in results:
            r["feature"] = r.pop("representation")

        return (
            pd.DataFrame(results)
            if results
            else pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": "all_folds_degenerate",
                    }
                ]
            )
        )

    except Exception as e:
        logger.error(f"[FAILED] {dataset_id}: {e}")
        return pd.DataFrame(
            [
                {
                    "dataset_id": dataset_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
            ]
        )
    finally:
        gc.collect()


# =============================================================================
# RESUME LOGIC
# =============================================================================


def get_completed_tasks(temp_dir: str, retry_errors: bool = True) -> set[str]:
    """Scans temp directory to identify completed tasks for checkpoint resume."""
    if not os.path.exists(temp_dir):
        return set()

    completed = set()
    shard_files = glob.glob(f"{temp_dir}/batch_*.parquet")

    for shard_file in shard_files:
        try:
            df = pd.read_parquet(shard_file)

            if "dataset_id" not in df.columns:
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
                complete_mask = has_results | has_skip
            else:
                complete_mask = has_results | has_skip | has_error

            completed.update(df[complete_mask]["dataset_id"].unique())

        except Exception:
            continue

    return completed


def flush_results(
    buffer: list[pd.DataFrame], temp_dir: str, logger: logging.Logger
) -> None:
    """Flush results buffer to temp shard."""
    if not buffer:
        return
    batch_id = str(uuid.uuid4())[:8]
    batch_path = f"{temp_dir}/batch_{batch_id}.parquet"
    pd.concat(buffer, ignore_index=True).to_parquet(batch_path, index=False)
    logger.info(f"[FLUSH] Wrote batch {batch_id}: {len(buffer)} tasks")


def merge_shards(temp_dir: str, out_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Merge all shard files into final output."""
    shard_files = sorted(glob.glob(f"{temp_dir}/batch_*.parquet"))

    if not shard_files:
        logger.warning("No shard files found")
        return pd.DataFrame()

    logger.info(f"Merging {len(shard_files)} shards...")
    final_df = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)
    final_df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(final_df)} rows to {out_path}")

    return final_df


# =============================================================================
# MAIN
# =============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Celltype benchmark with multiple modes (standard/low_data/pseudobulk)"
    )
    ap.add_argument(
        "--celltype_manifest", required=True, help="Cell type manifest parquet"
    )
    ap.add_argument(
        "--cache_dir", required=True, help="Cache directory with .h5ad files"
    )
    ap.add_argument("--out_parquet", required=True, help="Output parquet path")
    ap.add_argument("--log_file", default="celltype_benchmark.log", help="Log file")

    # Mode selection
    ap.add_argument(
        "--mode",
        choices=["standard", "low_data", "pseudobulk"],
        default="standard",
        help="Benchmark mode (default: standard)",
    )

    # CV parameters
    ap.add_argument("--folds", type=int, default=5, help="CV folds (default: 5)")
    ap.add_argument(
        "--alpha", type=float, default=1e-5, help="L2 regularization (default: 1e-5)"
    )
    ap.add_argument(
        "--random_state", type=int, default=0, help="Random seed (default: 0)"
    )
    ap.add_argument(
        "--embeddings",
        nargs="+",
        default=["scvi", "geneformer", "tf-sapiens", "tf-exemplar-human", "bmfm"],
        help="Embedding keys",
    )

    # Low-data mode parameters
    ap.add_argument(
        "--n_per_class",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="Sample sizes per class for low-data (default: [10, 25, 50, 100])",
    )
    ap.add_argument(
        "--n_bootstrap",
        type=int,
        default=10,
        help="Bootstrap replicates for low-data (default: 10)",
    )

    # Pseudobulk mode parameters
    ap.add_argument(
        "--pooling",
        choices=["mean", "median", "sum"],
        default="median",
        help="Pooling strategy for pseudobulk (default: median)",
    )

    # Parallelization
    ap.add_argument(
        "--workers", type=int, default=4, help="Parallel workers (default: 4)"
    )

    # Other options
    ap.add_argument(
        "--task_limit", type=int, default=None, help="Limit datasets (for testing)"
    )
    ap.add_argument(
        "--flush_every", type=int, default=5, help="Flush every N tasks (default: 5)"
    )
    ap.add_argument(
        "--keep_temp", action="store_true", help="Keep temp shards after merge"
    )
    ap.add_argument(
        "--no_resume", action="store_true", help="Disable checkpoint resume"
    )
    ap.add_argument(
        "--no_retry_errors", action="store_true", help="Don't retry errors on resume"
    )

    args = ap.parse_args()

    logger = setup_logging(
        args.log_file,
        logger_name="benchmark",
        include_console=True,
        console_level=logging.WARNING,
    )

    # Setup paths
    temp_dir = f"{os.path.dirname(args.out_parquet) or '.'}/_temp_shards_celltype_{Path(args.out_parquet).stem}"
    os.makedirs(temp_dir, exist_ok=True)

    # Load manifest
    tasks_df = pd.read_parquet(args.celltype_manifest)
    tasks_df = tasks_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if args.task_limit:
        tasks_df = tasks_df.head(args.task_limit)

    # Resume logic
    completed_tasks = set()
    if not args.no_resume:
        completed_tasks = get_completed_tasks(
            temp_dir, retry_errors=not args.no_retry_errors
        )
        if completed_tasks:
            logger.info(f"RESUME: Found {len(completed_tasks)} completed datasets")
            tasks_df = tasks_df[~tasks_df["dataset_id"].isin(completed_tasks)].copy()

    # Exclude large datasets (run separately with --workers 1 --mem 500G)
    LARGE_DATASET_IDS = [
        '6f7fd0f1-a2ed-4ff1-80d3-33dde731cbc3',
        'd3cb449b-c2b1-4b50-a7f1-21203535fe61',
        'c2876b1b-06d8-4d96-a56b-5304f815b99a',
        '9dbab10c-118d-496b-966a-67f1763a6b7d',

    ]
    tasks_df = tasks_df[~tasks_df["dataset_id"].isin(LARGE_DATASET_IDS)].reset_index(drop=True)
    logger.info(f"Excluded {len(LARGE_DATASET_IDS)} large datasets (will run separately)")
    #tasks_df = tasks_df[tasks_df["dataset_id"].isin(LARGE_DATASET_IDS)].reset_index(drop=True)
    #logger.info(f"Only {len(LARGE_DATASET_IDS)} large datasets ")

    logger.info("=" * 80)
    logger.info(f"CELLTYPE BENCHMARK - Mode: {args.mode.upper()}")
    logger.info(
        f"  Datasets remaining: {len(tasks_df)} (completed: {len(completed_tasks)})"
    )
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Cache: {args.cache_dir}")
    logger.info(f"  Output: {args.out_parquet}")
    if args.mode == "low_data":
        logger.info(
            f"  Low-data: n_per_class={args.n_per_class}, n_bootstrap={args.n_bootstrap}"
        )
    elif args.mode == "pseudobulk":
        logger.info(f"  Pseudobulk: pooling={args.pooling}")
    logger.info("=" * 80)

    if len(tasks_df) == 0:
        logger.info("No tasks to process. Merging existing results...")
        merge_shards(temp_dir, args.out_parquet, logger)
        return

    # Prepare task arguments based on mode
    if args.mode == "standard":
        task_args = [
            (
                row,
                args.cache_dir,
                args.embeddings,
                args.folds,
                args.alpha,
                args.random_state,
            )
            for _, row in tasks_df.iterrows()
        ]
        task_func = benchmark_one_task_standard

    elif args.mode == "low_data":
        task_args = [
            (
                row,
                args.cache_dir,
                args.embeddings,
                args.folds,
                args.alpha,
                args.random_state,
                args.n_per_class,
                args.n_bootstrap,
            )
            for _, row in tasks_df.iterrows()
        ]
        task_func = benchmark_one_task_lowdata

    elif args.mode == "pseudobulk":
        task_args = [
            (
                row,
                args.cache_dir,
                args.embeddings,
                args.folds,
                args.alpha,
                args.random_state,
                args.pooling,
            )
            for _, row in tasks_df.iterrows()
        ]
        task_func = benchmark_one_task_pseudobulk

    # Run benchmarks
    buffer = []
    n_success = 0
    n_failed = 0
    n_skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(task_func, arg): arg[0]["dataset_id"] for arg in task_args
        }

        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                dataset_id = futures[future]

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

                if len(buffer) >= args.flush_every:
                    flush_results(buffer, temp_dir, logger)
                    buffer = []

                pbar.update(1)
                pbar.set_postfix(success=n_success, failed=n_failed, skipped=n_skipped)

    if buffer:
        flush_results(buffer, temp_dir, logger)

    # Merge results
    final_df = merge_shards(temp_dir, args.out_parquet, logger)

    # Summary
    has_mcc = (
        final_df["MCC"].notna()
        if "MCC" in final_df.columns
        else pd.Series(False, index=final_df.index)
    )
    logger.info("=" * 80)
    logger.info(f"COMPLETE! Output: {args.out_parquet} ({len(final_df)} rows)")
    logger.info(f"  Success: {final_df[has_mcc]['dataset_id'].nunique()} datasets")
    logger.info("=" * 80)

    if not args.keep_temp:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up {temp_dir}")


if __name__ == "__main__":
    main()
