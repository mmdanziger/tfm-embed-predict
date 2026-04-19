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

Uses the same .h5ad cache filled by 02_run_benchmarks.py Phase 1.
Cache filename format: {dataset_id}_{safe_cell_type}.h5ad
where safe_cell_type replaces spaces, slashes, colons with underscores.
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
    donor_class = {}
    for d, label in zip(donors, y):
        if d not in donor_class:
            donor_class[d] = set()
        donor_class[d].add(label)

    n_donors = len(donor_class)
    n_classes = len(np.unique(y))

    stats = {
        "n_classes": n_classes,
        "n_donors": n_donors,
    }

    if n_donors < 2:
        return False, "total_donors_lt2", stats

    if n_classes < 2:
        return False, "total_classes_lt2", stats

    return True, "ok", stats


def benchmark_one_task(args_tuple) -> pd.DataFrame:
    """
    Runs cell type prediction benchmark on a single dataset using cached .h5ad files.

    Loads per-cell-type cache files (filled by 02_run_benchmarks.py Phase 1),
    concatenates them (gene intersection), and runs CV predicting cell_type.
    """
    task_row, cache_dir, embedding_keys, n_splits, random_state = args_tuple

    dataset_id = task_row["dataset_id"]
    eligible_cell_types = task_row["cell_type"]  # List from manifest
    task_id = f"dataset::{dataset_id}"
    logger = logging.getLogger("benchmark")

    try:
        logger.info(
            f"[START] {task_id} ({len(eligible_cell_types)} cell types requested)"
        )

        # =====================================================================
        # Locate cache files for each cell type
        # =====================================================================
        missing = []
        cache_paths = []
        for ct in eligible_cell_types:
            fpath = os.path.join(cache_dir, get_cache_filename(dataset_id, ct))
            if not os.path.exists(fpath):
                missing.append(ct)
            else:
                cache_paths.append(fpath)

        if missing:
            logger.warning(
                f"[SKIP] {task_id}: Missing cache for {len(missing)}/{len(eligible_cell_types)} "
                f"cell types: {missing}"
            )
            return pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": "missing_cache_files",
                        "n_cell_types_requested": len(eligible_cell_types),
                        "n_missing": len(missing),
                    }
                ]
            )

        # =====================================================================
        # Load and concatenate cached files (identity-aware, rejects pre-v2)
        # =====================================================================
        adata = load_and_concat_celltype_caches(cache_paths)
        gc.collect()

        log_resources(logger, f"[{task_id}] Post-load")
        logger.info(
            f"[DATA LOADED] {task_id} ({adata.n_obs} cells, {adata.n_vars} genes)"
        )

        n_genes = adata.n_vars

        if adata.n_obs == 0:
            logger.warning(f"[SKIP] {task_id}: No cells after concatenation")
            return pd.DataFrame([{"dataset_id": dataset_id, "skip_reason": "no_cells"}])

        # =====================================================================
        # Validate and prepare
        # =====================================================================
        adata.obs["disease"] = adata.obs["disease"].astype(str)
        adata.obs["donor_id"] = adata.obs["donor_id"].astype(str)
        adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)

        y = adata.obs["cell_type"].values
        donors = adata.obs["donor_id"].values

        n_cell_types = len(np.unique(y))
        if n_cell_types < 2:
            logger.info(
                f"[SKIP] {task_id}: Only {n_cell_types} cell type(s) after load"
            )
            return pd.DataFrame(
                [{"dataset_id": dataset_id, "skip_reason": "too_few_celltypes"}]
            )

        is_feasible, reason, stats = validate_task_feasibility(y, donors, n_splits)
        if not is_feasible:
            logger.info(f"[SKIP] {task_id}: {reason}")
            return pd.DataFrame(
                [{"dataset_id": dataset_id, "skip_reason": reason, **stats}]
            )

        n_donors = len(np.unique(donors))
        k_folds = min(n_splits, n_donors)

        if k_folds < 2:
            logger.info(f"[SKIP] {task_id}: Only {n_donors} donor(s)")
            return pd.DataFrame(
                [{"dataset_id": dataset_id, "skip_reason": "too_few_donors"}]
            )

        # =====================================================================
        # Run unified benchmark
        # Data is already preprocessed in the cache — pass preprocess=False
        # =====================================================================
        from benchmark import build_representations, run_predictions

        representations = build_representations(adata, embedding_keys=embedding_keys)

        # Stratify on DISEASE (not cell_type) to ensure diseased donors in each fold
        results_df = run_predictions(
            adata,
            label_col="cell_type",
            representations=representations,
            n_folds=k_folds,
            alpha=1e-5,
            donor_col="donor_id",
            stratify_col="disease",
            random_state=random_state,
            preprocess=False,  # Already preprocessed in cache
            dataset_id=dataset_id,
            n_genes=n_genes,
            n_cell_types=n_cell_types,
        )

        results = results_df.to_dict(orient="records")

        for r in results:
            r["feature"] = r.pop("representation")

        if not results:
            logger.warning(f"[SKIP] {task_id}: All {k_folds} folds were degenerate")
            return pd.DataFrame(
                [
                    {
                        "dataset_id": dataset_id,
                        "skip_reason": f"all_{k_folds}_folds_degenerate",
                    }
                ]
            )

        # Suspicious-score warning: near-perfect MCC on multi-class raw features is
        # the classic symptom of the pre-v2 cache column-alignment bug. Possible on
        # clean tissues but worth flagging for review.
        if n_cell_types >= 3:
            for r in results:
                mcc = r.get("MCC")
                feat = r.get("feature", "")
                if (
                    mcc is not None
                    and not pd.isna(mcc)
                    and mcc > 0.98
                    and feat.startswith("raw_")
                ):
                    logger.warning(
                        f"[SUSPICIOUS] {task_id} {feat} MCC={mcc:.4f} on "
                        f"{n_cell_types}-class task — verify cache integrity."
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
                }
            ]
        )

    finally:
        gc.collect()


def get_completed_tasks(temp_dir: str, retry_errors: bool = True) -> set[str]:
    """
    Scans temp directory to identify completed tasks for checkpoint resume.

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

    n_datasets = combined["dataset_id"].nunique()

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

    features = (
        combined["feature"].unique().tolist() if "feature" in combined.columns else []
    )

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
    cache_dir: str,
    embedding_keys: list[str],
    out_parquet: str,
    log_file: str,
    n_splits: int = 5,
    task_limit: int | None = None,
    max_workers: int = 4,
    keep_temp: bool = False,
    resume: bool = True,
    retry_errors: bool = True,
    random_state: int = 0,
    flush_every: int = 5,
) -> None:
    """
    Runs cell type prediction benchmarks from cell type manifest using cached .h5ad files.

    Supports checkpoint/resume: if temp directory exists, skips completed tasks.
    """
    logger = setup_logging(
        log_file,
        logger_name="benchmark",
        include_console=True,
        console_level=logging.WARNING,
        include_thread_name=True,
        filter_warnings=False,
    )

    tasks_df = pd.read_parquet(celltype_manifest_path)
    tasks_df = tasks_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if task_limit:
        tasks_df = tasks_df.head(task_limit)

    temp_dir = f"{os.path.dirname(out_parquet) or '.'}/_temp_shards_celltype_{Path(out_parquet).stem}"
    os.makedirs(temp_dir, exist_ok=True)

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
    logger.info(f"  Cache directory: {cache_dir}")
    logger.info(f"  Embeddings: {embedding_keys}")
    logger.info(f"  Output: {out_parquet}")
    logger.info(f"  Temp Dir: {temp_dir}")
    logger.info(f"  Resume: {resume}, Retry errors: {retry_errors}")
    logger.info("=" * 80)

    if len(tasks_df) == 0:
        logger.info("No tasks to process. Merging existing results...")
    else:
        task_args = [
            (row, cache_dir, embedding_keys, n_splits, random_state)
            for _, row in tasks_df.iterrows()
        ]

        buffer = []
        n_success = 0
        n_failed = 0
        n_skipped = 0

        def flush_buffer(buf: list[pd.DataFrame]) -> None:
            if not buf:
                return
            batch_id = str(uuid.uuid4())[:8]
            batch_path = f"{temp_dir}/batch_{batch_id}.parquet"
            pd.concat(buf, ignore_index=True).to_parquet(batch_path, index=False)
            logger.info(f"[FLUSH] Wrote batch {batch_id}: {len(buf)} tasks")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(benchmark_one_task, arg): arg[0]["dataset_id"]
                for arg in task_args
            }

            with tqdm(total=len(futures), desc="Processing datasets") as pbar:
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

                    if len(buffer) >= flush_every:
                        flush_buffer(buffer)
                        buffer = []

                    pbar.update(1)
                    pbar.set_postfix(
                        success=n_success, failed=n_failed, skipped=n_skipped
                    )

        if buffer:
            flush_buffer(buffer)

    # =========================================================================
    # Merge shards
    # =========================================================================
    logger.info("Merging shards into final output...")
    shard_files = sorted(glob.glob(f"{temp_dir}/batch_*.parquet"))

    if not shard_files:
        logger.warning("No shard files found - no results generated")
        return

    final_df = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)
    final_df.to_parquet(out_parquet, index=False)

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

    logger.info("=" * 80)
    logger.info("CELL TYPE BENCHMARK RUN COMPLETED")
    logger.info(f"  Success: {final_df[has_mcc]['dataset_id'].nunique()} datasets")
    logger.info(f"  Failed:  {final_df[has_error]['dataset_id'].nunique()} datasets")
    logger.info(f"  Skipped: {final_df[has_skip]['dataset_id'].nunique()} datasets")
    logger.info(f"  Final output: {out_parquet} ({final_df.shape[0]} rows)")
    logger.info("=" * 80)

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
    ap.add_argument(
        "--cache_dir",
        required=True,
        help="Directory with cached .h5ad files from 02_run_benchmarks.py Phase 1",
    )
    ap.add_argument("--out_parquet", required=True, help="Output parquet path")
    ap.add_argument(
        "--log_file", default="celltype_benchmark.log", help="Log file path"
    )
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
    ap.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    ap.add_argument(
        "--flush_every",
        type=int,
        default=5,
        help="Flush results every N completed tasks (default: 5)",
    )
    args = ap.parse_args()

    run_benchmarks(
        celltype_manifest_path=args.celltype_manifest,
        cache_dir=args.cache_dir,
        embedding_keys=args.embeddings,
        out_parquet=args.out_parquet,
        log_file=args.log_file,
        n_splits=args.folds,
        task_limit=args.task_limit,
        max_workers=args.workers,
        keep_temp=args.keep_temp,
        resume=not args.no_resume,
        retry_errors=not args.no_retry_errors,
        random_state=args.random_state,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
