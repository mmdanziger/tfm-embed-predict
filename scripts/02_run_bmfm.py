#!/usr/bin/env python3
"""
02_run_bmfm.py
==============
Script 2 of 3 - Run BMFM inference on all per-dataset h5ad files.

Loops over every .h5ad in --bmfm_input_dir and calls bmfm-targets-run
for each one. Outputs land in --bmfm_output_dir/{dataset_id}/.

BMFM is GPU-bound - jobs run sequentially by default (--workers 1).
Increase only if you have multiple GPUs.

Usage:
    python 02_run_bmfm.py \\
        --bmfm_input_dir  ./_bmfm_input  \\
        --bmfm_output_dir ./_bmfm_output \\
        --checkpoint      ibm-research/biomed.rna.bert.110m.mlm.rda.v1 \\
        [--workers        1]   \\
        [--task_limit     10]  \\
        [--overwrite]          \\
        [--dry_run]
"""

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str, logger_name: str = "run_bmfm") -> logging.Logger:
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
# Per-dataset worker
# ---------------------------------------------------------------------------

def run_one(
    input_path: Path,
    output_dir: Path,
    checkpoint: str,
    overwrite: bool,
    logger: logging.Logger,
) -> dict:
    dataset_id = input_path.stem
    work_dir   = output_dir / dataset_id

    # Treat non-empty output dir as done
    if work_dir.exists() and any(work_dir.iterdir()) and not overwrite:
        logger.debug(f"[{dataset_id}] skip_exists")
        return {"dataset_id": dataset_id, "status": "skip_exists"}

    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bmfm-targets-run",
        "-cn", "predict_benchmark",
        f"input_file={input_path}",
        f"working_dir={work_dir}",
        "++data_module.rda_transform=auto_align",
        "data_module.log_normalize_transform=false",
        "data_module.max_length=4096",
        "data_module.collation_strategy=language_modeling",
        f"checkpoint={checkpoint}",
    ]
    logger.debug(f"[{dataset_id}] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(
            f"[{dataset_id}] bmfm-targets-run failed (exit {result.returncode}):\n"
            f"{result.stderr[-2000:]}"
        )
        return {
            "dataset_id": dataset_id,
            "status": "error",
            "error": f"exit code {result.returncode}",
            "stderr": result.stderr[-2000:],
        }

    logger.debug(f"[{dataset_id}] done ? {work_dir}")
    return {"dataset_id": dataset_id, "status": "done", "work_dir": str(work_dir)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bmfm_input_dir",  required=True,
                   help="Directory of per-dataset raw-count h5ad files (output of script 1)")
    p.add_argument("--bmfm_output_dir", required=True,
                   help="Directory where BMFM writes its output (one subdir per dataset)")
    p.add_argument("--checkpoint",      required=True,
                   help="BMFM HuggingFace checkpoint name")
    p.add_argument("--workers",  type=int, default=1,
                   help="Parallel BMFM jobs - keep at 1 unless multiple GPUs (default: 1)")
    p.add_argument("--task_limit", type=int, default=None,
                   help="Process only first N datasets (for testing)")
    p.add_argument("--log_file",   default="02_run_bmfm.log",
                   help="Log file path (default: 02_run_bmfm.log)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Re-run even if output already exists")
    p.add_argument("--dry_run",    action="store_true",
                   help="Print commands without running them")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logging(args.log_file)

    input_dir  = Path(args.bmfm_input_dir)
    output_dir = Path(args.bmfm_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.h5ad"))
    if not input_files:
        logger.error(f"No .h5ad files found in {input_dir}")
        sys.exit(1)

    if args.task_limit:
        input_files = input_files[: args.task_limit]

    logger.info("=" * 70)
    logger.info("RUN BMFM INFERENCE")
    logger.info(f"  Datasets   : {len(input_files)}")
    logger.info(f"  Checkpoint : {args.checkpoint}")
    logger.info(f"  Input dir  : {input_dir}")
    logger.info(f"  Output dir : {output_dir}")
    logger.info(f"  Workers    : {args.workers}")
    logger.info("=" * 70)

    if args.dry_run:
        for f in input_files:
            work_dir = output_dir / f.stem
            print(
                f"  would run: bmfm-targets-run -cn predict_benchmark"
                f"input_file={f} working_dir={work_dir} "
                f"++data_module.rda_transform=auto_align "
                f"data_module.log_normalize_transform=false "
                f"data_module.max_length=4096 "
                f"data_module.collation_strategy=language_modeling "
                f"checkpoint={args.checkpoint}"
            )
        print(f"\n[dry_run] {len(input_files)} datasets would be processed.")
        return

    counts   = {}
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_one, f, output_dir, args.checkpoint, args.overwrite, logger): f
            for f in input_files
        }
        with tqdm(total=len(futures), desc="Running BMFM") as pbar:
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
        logger.warning(f"{len(failures)} datasets failed.")
        for f in failures:
            logger.warning(f"  {f['dataset_id']}: {f['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()