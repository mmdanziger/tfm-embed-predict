# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Linting and formatting:**
```bash
pre-commit run --all-files   # Run ruff linting + formatting on all files
ruff check .                 # Lint only
ruff format .                # Format only
```

**Running pipelines (in order):**
```bash
python 01_prepare_task_manifest.py    # Generate eligible classification tasks
python 02_run_benchmarks.py           # Phase 1 (fetch + cache) + Phase 2 (run CV)
python 02_run_benchmarks_phase2.py    # Phase 2 only (resume from cache)
python 03_make_celltype_manifest.py   # Build cell type task manifest
python 04_run_celltype_benchmark.py   # Run cell type prediction benchmarks
```

## Architecture

This is a benchmarking suite that evaluates transcriptomic feature representations (raw counts, PCA, pre-trained embeddings) for scRNA-seq classification tasks using CellxGene Census data.

### Two prediction tasks:
1. **Disease prediction** — given cell type + dataset, predict disease status
2. **Cell type prediction** — given dataset, predict cell type

### Core modules:
- **`preprocessing.py`** — Locked preprocessing pipeline: `drop_zero_genes → normalize_log1p` (10k target). Never modify without understanding downstream impact.
- **`metrics.py`** — Canonical metric computation (balanced accuracy, MCC, F1, AUROC/AUPRC) with edge-case handling for unknown labels and imbalanced classes.
- **`benchmark.py`** — Feature extractors (raw, PCA, embeddings), fold iteration, `SGDClassifier` training, and the core `run_predictions()` function.
- **`single_dataset_benchmark.py`** — Unified wrapper exposing `run_cv_benchmark()`, `run_low_data_benchmark()`, and `run_pseudobulk_benchmark()`.
- **`evaluation.py`** — Result loading (`load_and_prepare_results()`) and 20+ visualization functions.

### Data flow:
```
CellxGene Census
  → 01_prepare_task_manifest.py → disease_manifest.parquet
  → 02_run_benchmarks.py Phase 1 → cache/{dataset}_{cell_type}.h5ad
  → 02_run_benchmarks.py Phase 2 → results.parquet
  → evaluation.py → figures

disease_manifest.parquet
  → 03_make_celltype_manifest.py → celltype_manifest.parquet
  → 04_run_celltype_benchmark.py → celltype_results.parquet
```

### Key design decisions:
- **Two-phase data handling**: Phase 1 fetches Census data once and caches as `.h5ad`; Phase 2 runs CPU-bound CV in parallel on cached files. Phase 2 is fully resumable via temp shard files (`batch_*.parquet`).
- **Feature extractors as callables**: Each feature type has a `(X, train_idx, test_idx) → (X_train, X_test)` interface, built dynamically in `build_representations()`.
- **Parallel execution**: `ProcessPoolExecutor` for true multiprocessing across tasks.
- **CV splitting**: `StratifiedGroupKFold` by default — stratified on labels, grouped by donors to prevent leakage.
- **Pseudobulk**: Cells are aggregated to donor-level before creating AnnData, then the same pipeline runs on donor-level data.

### Plotting:
See `PLOTTING_GUIDE.md` for documentation on the 6 main plotting functions in `plots/`.
