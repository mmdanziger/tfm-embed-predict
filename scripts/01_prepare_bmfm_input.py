#!/usr/bin/env python3
"""
01_prepare_bmfm_input.py
========================
Script 1 of 3 - Prepare raw-count h5ad files for BMFM inference.

Requires v3 cache files (obs_names = soma_joinid). Run
migrate_cache_v2_to_v3.py first if not already done.

For each dataset_id in the manifest:
  1. Collect all soma_joinids from the per-(dataset_id, cell_type) cache files
  2. Re-fetch the full dataset from Census in one call using raw counts,
     requesting soma_joinid in obs
  3. Subset to exactly the cached soma_joinids (strict equality check)
  4. Set obs_names and obs["cell_name"] to soma_joinid strings
  5. Preserve obs columns (cell_type, donor_id, disease) from the cache
  6. Save one h5ad per dataset to --bmfm_input_dir

Usage:
    python 01_prepare_bmfm_input.py \\
        --celltype_manifest  ./_manifest/celltype_manifest.parquet \\
        --cache_dir          ./_adata_cache \\
        --bmfm_input_dir     ./_bmfm_input  \\
        --census_version     2025-01-30     \\
        --workers            4              \\
        [--task_limit        10]            \\
        [--overwrite]                       \\
        [--dry_run]
"""

import argparse
import gc
import logging
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cellxgene_census
import pandas as pd
import scanpy as sc
from anndata import AnnData
from tqdm.auto import tqdm

_FETCH_SEM = threading.Semaphore(2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str, logger_name: str = "prepare_bmfm") -> logging.Logger:
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


def collect_obs_from_cache(
    dataset_id: str,
    cell_types: list[str],
    cache_dir: Path,
    logger: logging.Logger,
) -> dict | None:
    """
    Load obs (not .X) from all v3 cache files for a dataset.
    obs_names are soma_joinid strings - globally unique across cell types.
    Returns dict with soma_joinids list and obs_df, or None if any file
    is missing or not v3.
    """
    obs_parts = []
    for ct in cell_types:
        fpath = cache_dir / get_cache_filename(dataset_id, ct)
        if not fpath.exists():
            logger.warning(f"[{dataset_id}] Missing cache file: {fpath.name}")
            return None
        try:
            a = sc.read_h5ad(fpath, backed="r")

            # Enforce v3
            version = a.uns.get("cache_format_version", 1)
            if version < 3:
                logger.error(
                    f"[{dataset_id}] {fpath.name} is cache v{version} - "
                    "run migrate_cache_v2_to_v3.py first."
                )
                a.file.close()
                return None

            obs_parts.append(a.obs[["cell_type", "donor_id", "disease", "cell_name"]].copy())
            a.file.close()
        except Exception as e:
            logger.error(f"[{dataset_id}] Could not read {fpath.name}: {e}")
            return None

    obs_df = pd.concat(obs_parts, axis=0)

    # soma_joinids are globally unique - no duplicates expected
    if obs_df.index.duplicated().any():
        n_dup = obs_df.index.duplicated().sum()
        logger.warning(
            f"[{dataset_id}] {n_dup} duplicate soma_joinids across cell type files - "
            "dropping duplicates (keep first)."
        )
        obs_df = obs_df[~obs_df.index.duplicated(keep="first")]

    return {
        "soma_joinids": obs_df.index.tolist(),
        "obs_df": obs_df,
    }


def fetch_dataset_raw(
    census,
    dataset_id: str,
    soma_joinids: list[str],
    logger: logging.Logger,
) -> AnnData:
    """
    Fetch raw counts for an entire dataset from Census.
    Requests soma_joinid in obs so we can align precisely.
    Subsets to exactly the cached soma_joinids - strict equality check.
    """
    obs_filter = f"is_primary_data==True and dataset_id=='{dataset_id}'"

    with _FETCH_SEM:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="homo_sapiens",
            measurement_name="RNA",
            obs_value_filter=obs_filter,
            obs_column_names=["soma_joinid", "dataset_id", "cell_type", "donor_id", "disease"],
            var_column_names=["feature_id", "feature_name"],
        )

    if adata.n_obs == 0:
        raise RuntimeError(f"Census returned 0 cells for dataset_id={dataset_id}")

    # Census obs_names are integer strings - build soma_joinid ? row mapping
    census_joinid_to_row = {
        str(jid): i
        for i, jid in enumerate(adata.obs["soma_joinid"].tolist())
    }

    # Strict check: every cached soma_joinid must be present in Census
    missing = [jid for jid in soma_joinids if jid not in census_joinid_to_row]
    if missing:
        raise RuntimeError(
            f"{len(missing)} cache soma_joinids not found in Census. "
            f"First few: {missing[:5]}"
        )

    # Subset and reorder to match cache order
    row_indices = [census_joinid_to_row[jid] for jid in soma_joinids]
    adata = adata[row_indices].copy()

    # Strict equality
    if adata.n_obs != len(soma_joinids):
        raise RuntimeError(
            f"After subset got {adata.n_obs} cells, expected {len(soma_joinids)}"
        )

    return adata


def build_and_save(
    adata_raw: AnnData,
    soma_joinids: list[str],
    obs_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Set obs_names and cell_name to soma_joinid strings.
    Attach obs metadata from cache as source of truth.
    Set var index to feature_id. Save atomically.
    """
    # Set stable obs_names = soma_joinid strings
    adata_raw.obs_names      = soma_joinids
    adata_raw.obs["cell_name"] = soma_joinids

    # Attach obs columns from cache (source of truth)
    for col in ["cell_type", "donor_id", "disease"]:
        if col in obs_df.columns:
            adata_raw.obs[col] = obs_df.loc[soma_joinids, col].values

    # Set var index to feature_id for BMFM compatibility
    if "feature_id" in adata_raw.var.columns:
        adata_raw.var = adata_raw.var.set_index("feature_id")

    tmp = out_path.with_suffix(".h5ad.tmp")
    try:
        adata_raw.write_h5ad(tmp)
        tmp.replace(out_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# ---------------------------------------------------------------------------
# Per-dataset worker
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_id: str,
    cell_types: list[str],
    cache_dir: Path,
    census,
    out_dir: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> dict:
    out_path = out_dir / f"{dataset_id}.h5ad"

    if out_path.exists() and not overwrite:
        logger.debug(f"[{dataset_id}] skip_exists")
        return {"dataset_id": dataset_id, "status": "skip_exists"}

    try:
        # Step 1: collect soma_joinids + obs from cache files
        collected = collect_obs_from_cache(dataset_id, cell_types, cache_dir, logger)
        if collected is None:
            return {"dataset_id": dataset_id, "status": "skip_missing_or_not_v3"}

        soma_joinids = collected["soma_joinids"]
        obs_df       = collected["obs_df"]
        logger.debug(
            f"[{dataset_id}] {len(soma_joinids)} cells across "
            f"{len(cell_types)} cell types - fetching from Census ..."
        )

        # Step 2: fetch raw counts from Census, aligned by soma_joinid
        adata_raw = fetch_dataset_raw(census, dataset_id, soma_joinids, logger)

        # Step 3: set obs_names to soma_joinids and save
        build_and_save(adata_raw, soma_joinids, obs_df, out_path)
        logger.debug(f"[{dataset_id}] saved ? {out_path.name}")

        del adata_raw
        gc.collect()

        return {
            "dataset_id": dataset_id,
            "status": "done",
            "n_cells": len(soma_joinids),
            "n_cell_types": len(cell_types),
        }

    except Exception as e:
        logger.error(f"[{dataset_id}] FAILED: {e}\n{traceback.format_exc()}")
        return {
            "dataset_id": dataset_id,
            "status": "error",
            "error": str(e),
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
                   help="Directory of per-(dataset,celltype) .h5ad v3 cache files")
    p.add_argument("--bmfm_input_dir",    required=True,
                   help="Output directory for per-dataset raw-count h5ad files")
    p.add_argument("--census_version",    default="2025-01-30",
                   help="CellXGene Census version (default: 2025-01-30)")
    p.add_argument("--census_uri",        default=None,
                   help="Optional Census URI override")
    p.add_argument("--workers",           type=int, default=4,
                   help="Parallel dataset workers (default: 4)")
    p.add_argument("--fetch_semaphore",   type=int, default=2,
                   help="Max concurrent Census fetches (default: 2)")
    p.add_argument("--task_limit",        type=int, default=None,
                   help="Process only first N datasets (for testing)")
    p.add_argument("--log_file",          default="01_prepare_bmfm_input.log")
    p.add_argument("--overwrite",         action="store_true",
                   help="Re-fetch even if output already exists")
    p.add_argument("--dry_run",           action="store_true",
                   help="Print what would be done, no writes")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_file)

    global _FETCH_SEM
    _FETCH_SEM = threading.Semaphore(args.fetch_semaphore)

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.bmfm_input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_df = pd.read_parquet(args.celltype_manifest)
    if args.task_limit:
        tasks_df = tasks_df.head(args.task_limit)

    logger.info("=" * 70)
    logger.info("PREPARE BMFM INPUT  (requires v3 cache files)")
    logger.info(f"  Datasets        : {len(tasks_df)}")
    logger.info(f"  Census version  : {args.census_version}")
    logger.info(f"  Cache dir       : {cache_dir}")
    logger.info(f"  Output dir      : {out_dir}")
    logger.info(f"  Workers         : {args.workers}")
    logger.info(f"  Fetch semaphore : {args.fetch_semaphore}")
    logger.info("=" * 70)

    if args.dry_run:
        for _, row in tasks_df.iterrows():
            print(f"  would process: {row['dataset_id']}  ({len(row['cell_type'])} cell types)")
        print(f"\n[dry_run] {len(tasks_df)} datasets would be processed.")
        return

    logger.info(f"Opening Census (version={args.census_version}) ...")
    with cellxgene_census.open_soma(
        uri=args.census_uri, census_version=args.census_version
    ) as census:

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    process_dataset,
                    row["dataset_id"],
                    row["cell_type"],
                    cache_dir,
                    census,
                    out_dir,
                    args.overwrite,
                    logger,
                ): row["dataset_id"]
                for _, row in tasks_df.iterrows()
            }

            counts   = {}
            failures = []

            with tqdm(total=len(futures), desc="Preparing datasets") as pbar:
                for fut in as_completed(futures):
                    res    = fut.result()
                    status = res["status"]
                    counts[status] = counts.get(status, 0) + 1
                    if status == "error":
                        failures.append(res)
                    pbar.update(1)
                    pbar.set_postfix_str(str(counts))

    logger.info(f"Done. {counts}")
    print(f"\nDone. {counts}")

    if failures:
        logger.warning(f"{len(failures)} datasets failed - rerun to retry.")
        for f in failures:
            logger.warning(f"  {f['dataset_id']}: {f['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()