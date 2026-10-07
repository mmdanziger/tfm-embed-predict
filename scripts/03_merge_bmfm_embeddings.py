#!/usr/bin/env python3
"""
03_merge_bmfm_embeddings.py
===========================
Script 3 of 3 - Merge BMFM embeddings back into per-(dataset_id, cell_type)
cache files.

Requires v3 cache files (obs_names = soma_joinid strings). BMFM output
also has soma_joinid as obs_names (set by script 1), so alignment is exact
and unambiguous.

For each dataset in the manifest:
  1. Load the BMFM output - obs_names are soma_joinid strings
  2. For each cache file in that dataset:
       a. Load its obs_names (soma_joinid strings)
       b. Extract exactly those rows from the BMFM embedding matrix
       c. Write into obsm[--embedding_key] in place (atomic write)

Cell alignment is strict: every soma_joinid in a cache file must be present
in the BMFM output.

Usage:
    python 03_merge_bmfm_embeddings.py \\
        --celltype_manifest  ./_manifest/celltype_manifest.parquet \\
        --cache_dir          ./_adata_cache  \\
        --bmfm_output_dir    ./_bmfm_output  \\
        --embedding_key      bmfm            \\
        [--workers           4]              \\
        [--task_limit        10]             \\
        [--overwrite]                        \\
        [--dry_run]
"""

import argparse
import fcntl
import gc
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

def setup_logging(log_file: str, logger_name: str = "merge_bmfm") -> logging.Logger:
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


def load_bmfm_output(bmfm_work_dir: Path, logger: logging.Logger) -> tuple[np.ndarray, dict]:
    embeddings_path = bmfm_work_dir / "embeddings.csv"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"embeddings.csv not found in {bmfm_work_dir}")

    df = pd.read_csv(embeddings_path, index_col=0, header=None)
    df.index = df.index.astype(str)

    embedding = df.values.astype(np.float32)
    soma_joinid_to_row = {jid: i for i, jid in enumerate(df.index.tolist())}

    logger.debug(f"  loaded embeddings {embedding.shape} from {embeddings_path.name}")
    return embedding, soma_joinid_to_row

# ---------------------------------------------------------------------------
# Merge one cache file
# ---------------------------------------------------------------------------

def merge_one_cache(
    cache_path: Path,
    bmfm_embedding: np.ndarray,
    soma_joinid_to_row: dict[str, int],
    embedding_key: str,
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

        return _merge_locked(
            cache_path, bmfm_embedding, soma_joinid_to_row,
            embedding_key, overwrite, logger
        )
    finally:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fd.close()
            try:
                lock_path.unlink()
            except OSError:
                pass


def _merge_locked(
    cache_path: Path,
    bmfm_embedding: np.ndarray,
    soma_joinid_to_row: dict[str, int],
    embedding_key: str,
    overwrite: bool,
    logger: logging.Logger,
) -> dict:
    try:
        adata = sc.read_h5ad(cache_path)
    except Exception as e:
        return {"path": str(cache_path), "status": "error", "error": f"read: {e}"}

    if embedding_key in adata.obsm and not overwrite:
        logger.debug(f"  {cache_path.name}: skip_exists")
        return {"path": str(cache_path), "status": "skip_exists"}

    # obs_names are soma_joinid strings (v3 cache)
    cache_joinids = adata.obs_names.tolist()

    # Strict check - every cache soma_joinid must be in BMFM output
    missing = [jid for jid in cache_joinids if jid not in soma_joinid_to_row]
    if missing:
        return {
            "path": str(cache_path),
            "status": "error",
            "error": (
                f"{len(missing)} / {len(cache_joinids)} soma_joinids missing "
                f"from BMFM output. First few: {missing[:5]}"
            ),
        }

    # Extract rows in cache order - alignment is now exact and unambiguous
    aligned = np.stack(
        [bmfm_embedding[soma_joinid_to_row[jid]] for jid in cache_joinids], axis=0
    ).astype(np.float32)

    adata.obsm[embedding_key] = aligned

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

    logger.debug(f"  {cache_path.name}: merged {aligned.shape}")
    return {
        "path": str(cache_path),
        "status": "done",
        "n_cells": len(cache_joinids),
        "embed_shape": list(aligned.shape),
    }


# ---------------------------------------------------------------------------
# Per-dataset worker
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_id: str,
    cell_types: list[str],
    cache_dir: Path,
    bmfm_output_dir: Path,
    embedding_key: str,
    overwrite: bool,
    workers: int,
    logger: logging.Logger,
) -> dict:
    bmfm_work_dir = bmfm_output_dir / dataset_id

    if not bmfm_work_dir.exists():
        logger.warning(f"[{dataset_id}] BMFM output dir not found - skipping")
        return {"dataset_id": dataset_id, "status": "skip_no_bmfm_output"}

    try:
        embedding, soma_joinid_to_row = load_bmfm_output(bmfm_work_dir, logger)
        logger.debug(
            f"[{dataset_id}] loaded BMFM embedding {embedding.shape}, "
            f"{len(cell_types)} cache files to update"
        )
    except Exception as e:
        logger.error(f"[{dataset_id}] Failed to load BMFM output: {e}\n{traceback.format_exc()}")
        return {"dataset_id": dataset_id, "status": "error", "error": f"load: {e}"}

    cache_paths = [cache_dir / get_cache_filename(dataset_id, ct) for ct in cell_types]
    missing_files = [p for p in cache_paths if not p.exists()]
    if missing_files:
        logger.warning(
            f"[{dataset_id}] {len(missing_files)} cache files missing: "
            f"{[p.name for p in missing_files[:3]]}"
        )
        cache_paths = [p for p in cache_paths if p.exists()]

    file_results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                merge_one_cache,
                p, embedding, soma_joinid_to_row, embedding_key, overwrite, logger
            ): p
            for p in cache_paths
        }
        for fut in as_completed(futures):
            file_results.append(fut.result())

    n_done   = sum(1 for r in file_results if r["status"] == "done")
    n_skip   = sum(1 for r in file_results if r["status"].startswith("skip"))
    n_errors = sum(1 for r in file_results if r["status"] == "error")
    failures = [r for r in file_results if r["status"] == "error"]

    del embedding
    gc.collect()

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
                   help="Directory of per-(dataset,celltype) v3 .h5ad cache files")
    p.add_argument("--bmfm_output_dir",   required=True,
                   help="Directory with BMFM output (one subdir per dataset_id)")
    p.add_argument("--embedding_key",     default="bmfm",
                   help="obsm key to write the embedding under (default: 'bmfm')")
    p.add_argument("--workers",    type=int, default=4,
                   help="Parallel workers for merging cache files (default: 4)")
    p.add_argument("--task_limit", type=int, default=None,
                   help="Process only first N datasets (for testing)")
    p.add_argument("--log_file",   default="03_merge_bmfm_embeddings.log")
    p.add_argument("--overwrite",  action="store_true",
                   help="Overwrite even if embedding key already present")
    p.add_argument("--dry_run",    action="store_true",
                   help="Print what would happen, no writes")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logging(args.log_file)

    cache_dir       = Path(args.cache_dir)
    bmfm_output_dir = Path(args.bmfm_output_dir)

    tasks_df = pd.read_parquet(args.celltype_manifest)
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

    missing_bmfm = [
        row["dataset_id"] for _, row in tasks_df.iterrows()
        if not (bmfm_output_dir / row["dataset_id"]).exists()
    ]
    if missing_bmfm:
        logger.warning(
            f"{len(missing_bmfm)} datasets have no BMFM output yet - will skip."
        )

    available_df = tasks_df[~tasks_df["dataset_id"].isin(missing_bmfm)].copy()

    logger.info("=" * 70)
    logger.info("MERGE BMFM EMBEDDINGS  (soma_joinid alignment)")
    logger.info(f"  Datasets total    : {len(tasks_df)}")
    logger.info(f"  Datasets with BMFM: {len(available_df)}")
    logger.info(f"  Cache dir         : {cache_dir}")
    logger.info(f"  BMFM output dir   : {bmfm_output_dir}")
    logger.info(f"  Embedding key     : {args.embedding_key}")
    logger.info(f"  Workers           : {args.workers}")
    logger.info("=" * 70)

    if available_df.empty:
        logger.error("No datasets have BMFM output. Run script 2 first.")
        sys.exit(1)

    if args.dry_run:
        for _, row in available_df.iterrows():
            print(
                f"  would merge: {row['dataset_id']}  "
                f"({len(row['cell_type'])} cache files) → obsm['{args.embedding_key}']"
            )
        print(f"\n[dry_run] {len(available_df)} datasets would be processed.")
        return

    all_failures = []
    counts = {}

    for _, row in tqdm(available_df.iterrows(), total=len(available_df), desc="Datasets"):
        res = process_dataset(
            dataset_id=row["dataset_id"],
            cell_types=row["cell_type"],
            cache_dir=cache_dir,
            bmfm_output_dir=bmfm_output_dir,
            embedding_key=args.embedding_key,
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
