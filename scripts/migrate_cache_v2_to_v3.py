#!/usr/bin/env python3
"""
migrate_cache_v2_to_v3.py
=========================
Migrate v2 cache files to v3 by replacing integer obs_names with real
Census cell identifiers (soma_joinid).

Pre-v3 caches were written with obs_names defaulting to positional integer
strings ("0", "1", ...) — local to each (dataset_id, cell_type) file.
This makes it impossible to align cells across files or back to BMFM output.

This script restores real cell identity by re-querying Census to get the
soma_joinid for each cell, then writes it as both:
  - obs_names  (index of .obs DataFrame)
  - obs["cell_name"] column (explicit backup)

Cell count must match exactly (mirrors migrate_cache_v1_to_v2.py logic).

Requirements:
  * Network access to the same Census version used by phase 1.
  * The cache file's (dataset_id, cell_type) must still resolve to the
    same number of cells — asserted before write.

Usage:
    python migrate_cache_v2_to_v3.py \\
        --cache_dir      ./_adata_cache \\
        --census_version 2025-01-30    \\
        --workers        4             \\
        [--dry_run]
"""

import argparse
import fcntl
import logging
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cellxgene_census
import scanpy as sc
from tqdm.auto import tqdm

_FETCH_SEM = threading.Semaphore(2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str, logger_name: str = "migrate_v3") -> logging.Logger:
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
# Per-file migration
# ---------------------------------------------------------------------------

def migrate_one(path: Path, census, logger: logging.Logger) -> dict:
    """Acquire file lock then migrate."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        lock_fd = open(lock_path, "w")
    except OSError as e:
        return {"path": str(path), "status": "error", "error": f"lock open: {e}"}

    try:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"path": str(path), "status": "skip_locked"}

        return _migrate_one_locked(path, census, logger)
    finally:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fd.close()
            try:
                lock_path.unlink()
            except OSError:
                pass


def _migrate_one_locked(path: Path, census, logger: logging.Logger) -> dict:
    # Load cache file
    try:
        a = sc.read_h5ad(path)
    except Exception as e:
        return {"path": str(path), "status": "error", "error": f"read: {e}"}

    # Skip if already v3
    if a.uns.get("cache_format_version", 1) >= 3:
        logger.debug(f"[{path.name}] skip_already_v3")
        return {"path": str(path), "status": "skip_already_v3"}

    # Must be at least v2 (has real gene IDs in .var)
    if a.uns.get("cache_format_version", 1) < 2:
        return {
            "path": str(path),
            "status": "error",
            "error": "cache is pre-v2 (no gene identity). Run migrate_cache_v1_to_v2.py first.",
        }

    if "dataset_id" not in a.obs.columns or "cell_type" not in a.obs.columns:
        return {"path": str(path), "status": "error", "error": "missing obs columns"}

    dataset_id = str(a.obs["dataset_id"].iloc[0])
    cell_type  = str(a.obs["cell_type"].iloc[0])

    obs_filter = (
        f"is_primary_data==True and dataset_id=='{dataset_id}' "
        f"and cell_type=='{cell_type}'"
    )

    # Fetch soma_joinid from Census — no .X, no embeddings, just obs
    try:
        with _FETCH_SEM:
            ref = cellxgene_census.get_anndata(
                census=census,
                organism="homo_sapiens",
                measurement_name="RNA",
                obs_value_filter=obs_filter,
                obs_column_names=["soma_joinid"],
                # var_query to fetch 0 genes — we only need obs
                var_query=cellxgene_census.util.query.AxisQuery(
                    coords=(slice(0, 0),)
                ),
            )
    except Exception:
        # Fallback: fetch normally if var_query trick not supported
        try:
            with _FETCH_SEM:
                ref = cellxgene_census.get_anndata(
                    census=census,
                    organism="homo_sapiens",
                    measurement_name="RNA",
                    obs_value_filter=obs_filter,
                    obs_column_names=["soma_joinid"],
                )
        except Exception as e:
            return {
                "path": str(path),
                "status": "error",
                "error": f"census fetch: {e}",
                "traceback": traceback.format_exc(),
            }

    # Strict cell count check — mirrors migrate_cache_v1_to_v2.py
    if ref.n_obs != a.n_obs:
        return {
            "path": str(path),
            "status": "error",
            "error": (
                f"cell count drift: census={ref.n_obs} cache={a.n_obs}. "
                "Census version may have changed."
            ),
        }

    # soma_joinid as string for obs_names
    soma_joinids = ref.obs["soma_joinid"].astype(str).tolist()

    # Patch obs_names and add cell_name column
    a.obs_names      = soma_joinids
    a.obs["cell_name"] = soma_joinids
    a.uns["cache_format_version"] = 3

    # Atomic write
    tmp = path.with_suffix(path.suffix + ".migrating")
    try:
        a.write_h5ad(tmp)
        tmp.replace(path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return {
            "path": str(path),
            "status": "error",
            "error": f"write: {e}",
            "traceback": traceback.format_exc(),
        }

    logger.debug(f"[{path.name}] migrated — {a.n_obs} cells, sample id: {soma_joinids[0]}")
    return {"path": str(path), "status": "migrated", "n_cells": a.n_obs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache_dir",      required=True,
                   help="Directory of .h5ad cache files")
    p.add_argument("--census_version", default="2025-01-30",
                   help="CellXGene Census version (default: 2025-01-30)")
    p.add_argument("--census_uri",     default=None,
                   help="Optional Census URI override")
    p.add_argument("--workers",        type=int, default=4,
                   help="Parallel workers (default: 4)")
    p.add_argument("--fetch_semaphore",type=int, default=2,
                   help="Max concurrent Census fetches (default: 2)")
    p.add_argument("--log_file",       default="migrate_cache_v2_to_v3.log",
                   help="Log file path")
    p.add_argument("--dry_run",        action="store_true",
                   help="List what would migrate, no writes")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logging(args.log_file)

    global _FETCH_SEM
    _FETCH_SEM = threading.Semaphore(args.fetch_semaphore)

    cache_dir = Path(args.cache_dir)
    paths     = sorted(cache_dir.glob("*.h5ad"))

    # Warn about stale artifacts from crashed runs
    stale_tmp  = sorted(cache_dir.glob("*.h5ad.migrating"))
    stale_lock = sorted(cache_dir.glob("*.h5ad.lock"))
    if stale_tmp or stale_lock:
        logger.warning(
            f"Found {len(stale_tmp)} .migrating and {len(stale_lock)} .lock files "
            "from prior runs. Inspect and remove before re-running if no migrator is active."
        )

    logger.info("=" * 70)
    logger.info("MIGRATE CACHE v2 → v3  (add soma_joinid as obs_names)")
    logger.info(f"  Cache dir      : {cache_dir}")
    logger.info(f"  Files found    : {len(paths)}")
    logger.info(f"  Census version : {args.census_version}")
    logger.info(f"  Workers        : {args.workers}")
    logger.info("=" * 70)

    if not paths:
        logger.info("Nothing to do.")
        return

    if args.dry_run:
        for p in paths:
            print(f"  would migrate: {p.name}")
        print(f"\n[dry_run] {len(paths)} files would be processed.")
        return

    counts   = {}
    failures = []

    logger.info(f"Opening Census (version={args.census_version}) ...")
    with cellxgene_census.open_soma(
        uri=args.census_uri, census_version=args.census_version
    ) as census:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(migrate_one, p, census, logger): p
                for p in paths
            }
            with tqdm(total=len(futures), desc="Migrating") as pbar:
                for fut in as_completed(futures):
                    res    = fut.result()
                    status = res["status"]
                    counts[status] = counts.get(status, 0) + 1
                    if status == "error":
                        failures.append(res)
                        logger.warning(f"[FAIL] {res['path']}: {res['error']}")
                    pbar.update(1)
                    pbar.set_postfix_str(str(counts))

    logger.info(f"Done. {counts}")
    print(f"\nDone. {counts}")

    if failures:
        logger.warning(f"{len(failures)} files failed — rerun to retry.")
        for f in failures:
            logger.warning(f"  {f['path']}: {f['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
