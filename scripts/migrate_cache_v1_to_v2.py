#!/usr/bin/env python3
r"""
Migrate pre-v2 cache files (no gene identity in .var) to v2.

Pre-v2 caches were written without a var DataFrame, so var_names defaulted
to positional integer strings ("0", "1", ...). Concatenating such caches
across cell types in the cell-type benchmark silently mixed different genes
into the same column and produced artificially perfect predictions on raw
features.

This script restores .var in place by re-querying Census (without embeddings
to stay cheap) to recompute the gene_mask that phase 1 originally applied,
then writes the correct gene IDs back into each cache.

Requirements:
  * Network access to the same Census version used by phase 1.
  * The cache file's (dataset_id, cell_type) must still resolve to the same
    cells and the same gene_mask — asserted before write.

Usage:
  python scripts/migrate_cache_v1_to_v2.py \\
      --cache_dir ./_adata_cache \\
      --census_version 2025-01-30 \\
      --workers 4
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
import numpy as np
import scanpy as sc
from scipy import sparse
from tqdm.auto import tqdm

logger = logging.getLogger("migrate")

# TileDB context limit — matches phase 1's fetch_semaphore default.
_FETCH_SEM = threading.Semaphore(2)


def migrate_one(path: Path, census) -> dict:
    # Advisory exclusive lock on a sidecar file so concurrent invocations of
    # this script (or a stray re-submission) can't race on the same cache.
    # LOCK_NB → skip immediately if another process holds it rather than block.
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

        return _migrate_one_locked(path, census)
    finally:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fd.close()
            try:
                lock_path.unlink()
            except OSError:
                pass


def _migrate_one_locked(path: Path, census) -> dict:
    try:
        a = sc.read_h5ad(path)
    except Exception as e:
        return {"path": str(path), "status": "error", "error": f"read: {e}"}

    if a.uns.get("cache_format_version", 1) >= 2:
        return {"path": str(path), "status": "skip_already_v2"}

    if "dataset_id" not in a.obs.columns or "cell_type" not in a.obs.columns:
        return {"path": str(path), "status": "error", "error": "missing obs columns"}

    dataset_id = str(a.obs["dataset_id"].iloc[0])
    cell_type = str(a.obs["cell_type"].iloc[0])

    obs_filter = (
        f"is_primary_data==True and dataset_id=='{dataset_id}' "
        f"and cell_type=='{cell_type}'"
    )

    try:
        with _FETCH_SEM:
            ref = cellxgene_census.get_anndata(
                census=census,
                organism="homo_sapiens",
                measurement_name="RNA",
                obs_value_filter=obs_filter,
                obs_column_names=["donor_id"],
                var_column_names=["feature_id", "feature_name"],
            )
    except Exception as e:
        return {
            "path": str(path),
            "status": "error",
            "error": f"census: {e}",
            "traceback": traceback.format_exc(),
        }

    if ref.n_obs != a.n_obs:
        return {
            "path": str(path),
            "status": "error",
            "error": f"cell count drift: census={ref.n_obs} cache={a.n_obs}",
        }

    X = sparse.csr_matrix(ref.X) if not sparse.issparse(ref.X) else ref.X.tocsr()
    mask = np.array(X.sum(axis=0)).flatten() > 0

    if int(mask.sum()) != a.n_vars:
        return {
            "path": str(path),
            "status": "error",
            "error": f"gene count drift: mask_kept={int(mask.sum())} cache={a.n_vars}",
        }

    a.var = ref.var.iloc[mask].set_index("feature_id").copy()
    a.uns["cache_format_version"] = 2

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

    return {"path": str(path), "status": "migrated", "n_genes": int(mask.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", required=True, help="Directory of .h5ad cache files")
    ap.add_argument("--census_version", default="2025-01-30")
    ap.add_argument("--census_uri", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fetch_semaphore", type=int, default=2)
    ap.add_argument(
        "--dry_run", action="store_true", help="List what would migrate, no writes"
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    global _FETCH_SEM
    _FETCH_SEM = threading.Semaphore(args.fetch_semaphore)

    cache_dir = Path(args.cache_dir)
    paths = sorted(cache_dir.glob("*.h5ad"))
    logger.info(f"Found {len(paths)} cache files under {cache_dir}")

    # Flag stale artifacts from prior crashed runs. We don't auto-delete — they
    # could mask a real issue (or a live run); surface them for the operator.
    stale_tmp = sorted(cache_dir.glob("*.h5ad.migrating"))
    stale_lock = sorted(cache_dir.glob("*.h5ad.lock"))
    if stale_tmp or stale_lock:
        logger.warning(
            f"Found {len(stale_tmp)} .migrating and {len(stale_lock)} .lock files "
            "from prior runs. Inspect and remove before re-running if no migrator is active."
        )

    if args.dry_run:
        for p in paths:
            logger.info(f"  would migrate: {p}")
        return 0
    
    if not paths:
        logger.info("Nothing to do.")
        return 0
    
    to_migrate = paths

    counts = {"migrated": 0, "skip_already_v2": 0, "skip_locked": 0, "error": 0}
    failures = []

    logger.info(f"Opening Census (version={args.census_version})...")
    with cellxgene_census.open_soma(
        uri=args.census_uri, census_version=args.census_version
    ) as census:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(migrate_one, p, census): p for p in to_migrate}
            with tqdm(total=len(futures), desc="Migrating") as pbar:
                for fut in as_completed(futures):
                    res = fut.result()
                    counts[res["status"]] = counts.get(res["status"], 0) + 1
                    if res["status"] == "error":
                        failures.append(res)
                        logger.warning(f"[FAIL] {res['path']}: {res['error']}")
                    pbar.update(1)
                    pbar.set_postfix_str(str(counts))

    logger.info(f"Done. {counts}")

    exit_bad = False
    if failures:
        logger.warning(
            f"{len(failures)} files failed — rerun to retry, or rebuild via phase 1."
        )
        exit_bad = True
    if counts.get("skip_locked"):
        logger.warning(
            f"{counts['skip_locked']} files were skipped because another migrator "
            "held the lock. Rerun after it completes to cover them."
        )
        exit_bad = True
    if scan_unreadable:
        logger.warning(
            f"{len(scan_unreadable)} files were unreadable at scan time and were "
            "never attempted:"
        )
        for item in scan_unreadable:
            logger.warning(f"  {item['path']}: {item['error']}")
        exit_bad = True

    return 1 if exit_bad else 0


if __name__ == "__main__":
    sys.exit(main())
