#!/usr/bin/env python3
"""
01b_convert_gene_symbols.py
===========================
Converts var_names in all h5ad files in --bmfm_input_dir from Ensembl IDs
(ENSG...) to gene symbols (CD3E, GAPDH, ...) in place.

BMFM tokenizer requires gene symbols. Script 1 sets var_names to feature_id
(Ensembl IDs) -- this script swaps them to feature_name before BMFM runs.

Run between script 1 and script 2.

Usage:
    python 01b_convert_gene_symbols.py \\
        --bmfm_input_dir ./_bmfm_input \\
        [--workers 4]   \\
        [--dry_run]
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import scanpy as sc
from tqdm.auto import tqdm


def setup_logging(log_file: str, logger_name: str = "convert_symbols") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def convert_one(path: Path, logger: logging.Logger) -> dict:
    try:
        adata = sc.read_h5ad(path)

        if not adata.var_names[0].startswith("ENSG"):
            logger.debug(f"[{path.name}] already gene symbols -- skip")
            return {"path": str(path), "status": "skip_already_symbols"}

        if "feature_name" not in adata.var.columns:
            return {
                "path": str(path),
                "status": "error",
                "error": "var_names are Ensembl IDs but feature_name column is missing",
            }

        adata.var_names = adata.var["feature_name"].astype(str).tolist()
        adata.var_names_make_unique()

        # Atomic overwrite
        tmp = path.with_suffix(".h5ad.tmp")
        adata.write_h5ad(tmp)
        tmp.replace(path)

        logger.debug(f"[{path.name}] converted -- sample: {adata.var_names[0]}")
        return {"path": str(path), "status": "done"}

    except Exception as e:
        return {"path": str(path), "status": "error", "error": str(e)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bmfm_input_dir", required=True,
                   help="Directory of per-dataset h5ad files (output of script 1)")
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--log_file", default="01b_convert_gene_symbols.log")
    p.add_argument("--dry_run",  action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logging(args.log_file)

    input_dir = Path(args.bmfm_input_dir)
    files     = sorted(input_dir.glob("*.h5ad"))

    if not files:
        logger.error(f"No .h5ad files found in {input_dir}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("CONVERT VAR_NAMES TO GENE SYMBOLS")
    logger.info(f"  Input dir : {input_dir}")
    logger.info(f"  Files     : {len(files)}")
    logger.info(f"  Workers   : {args.workers}")
    logger.info("=" * 70)

    if args.dry_run:
        for f in files:
            print(f"  would convert: {f.name}")
        print(f"\n[dry_run] {len(files)} files would be processed.")
        return

    counts   = {}
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(convert_one, f, logger): f for f in files}
        with tqdm(total=len(futures), desc="Converting") as pbar:
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
        logger.warning(f"{len(failures)} files failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
