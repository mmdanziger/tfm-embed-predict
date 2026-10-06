#!/usr/bin/env python3
"""
04_concat_embeddings.py
=======================
Concatenates two existing obsm embeddings in per-(dataset_id, cell_type)
cache files into a new combined embedding.

Run after 03_merge_bmfm_embeddings.py has been run twice (once per embedding).

Usage:
    python 04_concat_embeddings.py \\
        --celltype_manifest ./_manifest/celltype_manifest.parquet \\
        --cache_dir         ./_adata_cache  \\
        --key_1             bmfm            \\
        --key_2             bmfm_MLMMUL     \\
        --output_key        bmfm_concat     \\
        [--workers          4]              \\
        [--task_limit       10]             \\
        [--overwrite]                       \\
        [--dry_run]
"""

import argparse
import fcntl
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str, logger_name: str = "concat_embeddings") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_cache_filename(dataset_id: str, cell_type: str) -> str:
    safe_ct = cell_type.replace(" ", "_").replace("/", "_").replace(":", "_")
    return f"{dataset_id}_{safe_ct}.h5ad"


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def concat_one_cache(
    cache_path: Path,
    key_1: str,
    key_2: str,
    output_key: str,
    overwrite: bool,
    logger: logging.Logger,
) -> dict:
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    try:
        lock_fd = open(lock_path, "w")
    except OSError as e:
        return {"path": str(cache_path), "status": "error", "error": f"lock open: {e}"}

    try:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.warning(f"  {cache_path.name}: locked - skipping")
            return {"path": str(cache_path), "status": "skip_locked"}

        return _concat_locked(cache_path, key_1, key_2, output_key, overwrite, logger)
    finally:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fd.close()
            try:
                lock_path.unlink()
            except OSError:
                pass


def _concat_locked(
    cache_path: Path,
    key_1: str,
    key_2: str,
    output_key: str,
    overwrite: bool,
    logger: logging.Logger,
) -> dict:
    try:
        adata = sc.read_h5ad(cache_path)
    except Exception as e:
        return {"path": str(cache_path), "status": "error", "error": f"read: {e}"}

    if output_key in adata.obsm and not overwrite:
        logger.debug(f"  {cache_path.name}: skip_exists")
        return {"path": str(cache_path), "status": "skip_exists"}

    # Check both keys exist
    missing_keys = [k for k in [key_1, key_2] if k not in adata.obsm]
    if missing_keys:
        return {
            "path": str(cache_path),
            "status": "error",
            "error": f"missing obsm keys: {missing_keys}. Available: {list(adata.obsm.keys())}",
        }

    emb_1 = np.array(adata.obsm[key_1], dtype=np.float32)
    emb_2 = np.array(adata.obsm[key_2], dtype=np.float32)

    if emb_1.shape[0] != emb_2.shape[0]:
        return {
            "path": str(cache_path),
            "status": "error",
            "error": f"cell count mismatch: {key_1}={emb_1.shape[0]} {key_2}={emb_2.shape[0]}",
        }

    adata.obsm[output_key] = np.concatenate([emb_1, emb_2], axis=1)

    logger.debug(
        f"  {cache_path.name}: {key_1}{emb_1.shape} + {key_2}{emb_2.shape} "
        f"-> {output_key}{adata.obsm[output_key].shape}"
    )

    # Atomic write
    tmp = cache_path.with_suffix(cache_path.suffix + ".merging")
    try:
        adata.write_h5ad(tmp)
        tmp.replace(cache_path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return {
            "path": str(cache_path),
            "status": "error",
            "error": f"write: {e}",
            "traceback": traceback.format_exc(),
        }

    return {"path": str(cache_path), "status": "done"}


# ---------------------------------------------------------------------------
# Per-dataset worker
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_id: str,
    cell_types: list[str],
    cache_dir: Path,
    key_1: str,
    key_2: str,
    output_key: str,
    overwrite: bool,
    workers: int,
    logger: logging.Logger,
) -> dict:
    cache_paths = [cache_dir / get_cache_filename(dataset_id, ct) for ct in cell_types]
    missing_files = [p for p in cache_paths if not p.exists()]
    if missing_files:
        logger.warning(
            f"[{dataset_id}] {len(missing_files)} cache files missing: "
            f"{[p.name for p in missing_files[:3]]}"
        )
        cache_paths = [p for p in cache_paths if p.exists()]

    if not cache_paths:
        return {"dataset_id": dataset_id, "status": "skip_no_files"}

    file_results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                concat_one_cache, p, key_1, key_2, output_key, overwrite, logger
            ): p
            for p in cache_paths
        }
        for fut in as_completed(futures):
            file_results.append(fut.result())

    n_done   = sum(1 for r in file_results if r["status"] == "done")
    n_skip   = sum(1 for r in file_results if r["status"].startswith("skip"))
    n_errors = sum(1 for r in file_results if r["status"] == "error")
    failures = [r for r in file_results if r["status"] == "error"]

    logger.debug(f"[{dataset_id}] done={n_done} skip={n_skip} errors={n_errors}")

    return {
        "dataset_id": dataset_id,
        "status": "error" if n_errors else "done",
        "n_done": n_done,
        "n_skip": n_skip,
        "n_errors": n_errors,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--celltype_manifest", required=True,
                   help="Manifest parquet with columns: dataset_id, cell_type (list)")
    p.add_argument("--cache_dir",         required=True,
                   help="Directory of per-(dataset,celltype) .h5ad cache files")
    p.add_argument("--key_1",             required=True,
                   help="First obsm embedding key (e.g. bmfm)")
    p.add_argument("--key_2",             required=True,
                   help="Second obsm embedding key (e.g. bmfm_MLMMUL)")
    p.add_argument("--output_key",        required=True,
                   help="Output obsm key for concatenated embedding (e.g. bmfm_concat)")
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--task_limit", type=int, default=None,
                   help="Process only first N datasets (for testing)")
    p.add_argument("--log_file",   default="04_concat_embeddings.log")
    p.add_argument("--overwrite",  action="store_true",
                   help="Overwrite output_key even if already present")
    p.add_argument("--dry_run",    action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logging(args.log_file)

    cache_dir = Path(args.cache_dir)
    tasks_df  = pd.read_parquet(args.celltype_manifest)
    # Support both manifest shapes:
    #  - celltype manifest: one row per dataset, cell_type is a list
    #  - disease manifest:   one row per (dataset_id, cell_type), cell_type is a string
    if tasks_df["cell_type"].apply(lambda x: isinstance(x, str)).all():
        logger.info("Detected per-(dataset,celltype) manifest format - grouping by dataset_id")
        tasks_df = (
            tasks_df.groupby("dataset_id")["cell_type"]
            .apply(list)
            .reset_index()
        )

    if args.task_limit:
        tasks_df = tasks_df.head(args.task_limit)

    logger.info("=" * 70)
    logger.info("CONCAT EMBEDDINGS")
    logger.info(f"  Datasets   : {len(tasks_df)}")
    logger.info(f"  Cache dir  : {cache_dir}")
    logger.info(f"  key_1      : {args.key_1}")
    logger.info(f"  key_2      : {args.key_2}")
    logger.info(f"  output_key : {args.output_key}")
    logger.info(f"  Workers    : {args.workers}")
    logger.info("=" * 70)

    if args.dry_run:
        for _, row in tasks_df.iterrows():
            print(
                f"  would concat: {row['dataset_id']} "
                f"({len(row['cell_type'])} files) "
                f"{args.key_1} + {args.key_2} -> {args.output_key}"
            )
        print(f"\n[dry_run] {len(tasks_df)} datasets would be processed.")
        return

    all_failures = []
    counts = {}

    for _, row in tqdm(tasks_df.iterrows(), total=len(tasks_df), desc="Datasets"):
        res = process_dataset(
            dataset_id=row["dataset_id"],
            cell_types=row["cell_type"],
            cache_dir=cache_dir,
            key_1=args.key_1,
            key_2=args.key_2,
            output_key=args.output_key,
            overwrite=args.overwrite,
            workers=args.workers,
            logger=logger,
        )
        status = res["status"]
        counts[status] = counts.get(status, 0) + 1
        if res.get("failures"):
            all_failures.extend(res["failures"])
            for f in res["failures"]:
                logger.warning(f"  [FAIL] {f['path']}: {f['error']}")

    logger.info(f"Done. {counts}")
    print(f"\nDone. {counts}")

    if all_failures:
        logger.warning(f"{len(all_failures)} cache files failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
