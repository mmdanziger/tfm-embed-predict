# tfm-embed-predict

Code for **Benchmarking single-cell foundation models reveals an advantage in the low-data regime for cell-type and disease-state prediction**.

We compare cell representations from four foundation models (scVI, Geneformer, TranscriptFormer-Sapiens, BMFM-RNA) with classical baselines (log-normalized counts, PCA-50, a random projection) on 868 disease-prediction tasks (82 datasets) and 65 cell-type datasets from the CZ CELLxGENE Census (release 2025-01-30). Every representation is evaluated with the same L2-regularized logistic regression under donor-stratified cross-validation, with full data, with downsampling to 2-256 cells per class, and after median pooling per donor.


## Pipeline overview

```
CELLxGENE Census
   |
01_prepare_task_manifest.py   ->  disease_manifest.parquet        (candidate disease tasks)
02_run_benchmarks.py          ->  ./_adata_cache/*.h5ad           (fetch + cache, one file per task)
03_make_celltype_manifest.py  ->  celltype_manifest.parquet       (cell-type benchmark datasets)
scripts/migrate_cache_*.py    ->  cache upgraded to current format
scripts/01_prepare_bmfm_input.py, 01b_convert_gene_symbols.py,
scripts/02_run_bmfm.py, 03_merge_bmfm_embeddings.py,
scripts/04_concat_embeddings.py   ->  BMFM-RNA embeddings added to the cache
02_run_benchmarks_multimode_phase2.py   ->  disease benchmark (standard / low_data / pseudobulk)
04_benchmark_celltype_multimode.py      ->  cell-type benchmark (standard / low_data / pseudobulk)
```

Steps 1-3 are cheap; the BMFM-RNA embedding step needs a GPU-capable environment, and the benchmark steps run one task per CPU worker.

## Installation

```bash
git clone https://github.com/mmdanziger/tfm-embed-predict.git
cd tfm-embed-predict
pip install -r requirements.txt     
```

Main dependencies: `cellxgene-census`, `scanpy`, `anndata`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `pyarrow`, `statsmodels`, `matplotlib`, `seaborn`, `tqdm`.
The BMFM-RNA steps additionally require the BMFM-RNA package and access to the public checkpoints listed below.

## Running the pipeline


### 1. Task manifests and data caching

```bash
# Candidate disease tasks (dataset x cell type) from the Census:
# >= 500 cells, >= 2 disease labels with >= 50 cells each, >= 3 donors per disease label
python 01_prepare_task_manifest.py --out_path disease_manifest.parquet

# Fetch the cells of every task, check that CV is feasible, and cache one .h5ad per task
# (log-normalized expression, genes with zero counts in the task removed, plus the
# precomputed Census embeddings for scVI, Geneformer and TranscriptFormer)
python 02_run_benchmarks.py \
  --log_file 02_run_benchmark.log \
  --tasks_manifest disease_manifest.parquet \
  --out_parquet results.parquet

# Cell-type benchmark: datasets with >= 2 cell types, >= 3 donors, >= 500 cells
python 03_make_celltype_manifest.py \
  --disease_manifest disease_manifest.parquet \
  --out_path celltype_manifest.parquet
```


### 2. BMFM-RNA embeddings

```bash

# Prepare BMFM-RNA input and convert gene identifiers to symbols
python ./scripts/01_prepare_bmfm_input.py \
  --celltype_manifest celltype_manifest.parquet \
  --cache_dir ./_adata_cache --bmfm_input_dir ./_bmfm_input --census_version 2025-01-30
python ./scripts/01b_convert_gene_symbols.py --bmfm_input_dir ./_bmfm_input

# Embed with the two BMFM-RNA checkpoints (CLS token)
python ./scripts/02_run_bmfm.py \
  --bmfm_input_dir ./_bmfm_input_have_embedding --bmfm_output_dir ./_bmfm_output_MLMMUL \
  --checkpoint ibm-research/biomed.rna.llama.32m.mlm.multitask.v1
python ./scripts/02_run_bmfm.py \
  --bmfm_input_dir ./_bmfm_input --bmfm_output_dir ./_bmfm_output \
  --checkpoint ibm-research/biomed.rna.llama.47m.wced.multitask.v1

# Add the embeddings to the cache (keys: bmfm_MLMMUL, bmfm) and concatenate them
python ./scripts/03_merge_bmfm_embeddings.py \
  --celltype_manifest ./disease_manifest.parquet --cache_dir ./_adata_cache \
  --bmfm_output_dir ./_bmfm_output_MLMMUL --embedding_key bmfm_MLMMUL \
  --log_file 03_merge_bmfm_embeddings_MLMMUL.log
python ./scripts/03_merge_bmfm_embeddings.py \
  --celltype_manifest ./disease_manifest.parquet --cache_dir ./_adata_cache \
  --bmfm_output_dir ./_bmfm_output --embedding_key bmfm \
  --log_file 03_merge_bmfm_embeddings.log
python ./scripts/04_concat_embeddings.py \
  --celltype_manifest ./disease_manifest.parquet --cache_dir ./_adata_cache \
  --key_1 bmfm --key_2 bmfm_MLMMUL --output_key bmfm_var1
```

`bmfm_var1` is the concatenation of the two checkpoints' CLS embeddings (BMFM-CONCAT in the paper).

### 3. Benchmarks

Each command runs one evaluation regime. Defaults: 5-fold donor-stratified CV (capped by the minimum number of donors per disease label), L2 penalty `alpha = 1e-5`, random seed 0.

```bash
EMB="bmfm_var1 scvi geneformer tf-sapiens"
BASE="raw_lognorm raw_pca50 random_proj50"

# Disease-state prediction
for MODE in standard low_data pseudobulk; do
  python 02_run_benchmarks_multimode_phase2.py --mode $MODE \
    --cache_dir ./_adata_cache --tasks_manifest disease_manifest.parquet \
    --out_parquet test_files_${MODE}_disease.parquet \
    --embeddings $EMB --baselines $BASE \
    $( [ $MODE = low_data ] && echo "--n_per_class 2 4 8 16 32 64 128 256 --n_bootstrap 5 --keep_temp" )
done

# Cell-type prediction
for MODE in standard low_data pseudobulk; do
  python 04_benchmark_celltype_multimode.py --mode $MODE \
    --cache_dir ./_adata_cache --celltype_manifest celltype_manifest.parquet \
    --out_parquet test_files_${MODE}_celltype.parquet \
    --embeddings $EMB --baselines $BASE \
    $( [ $MODE = low_data ] && echo "--n_per_class 2 4 8 16 32 64 128 256 --n_bootstrap 5" )
done
```

 Modes: `standard` is full-data cross-validation, `low_data` downsamples to *n* cells per class with bootstrap resamples, and `pseudobulk` (called median pooling in the paper) pools each donor's cells, expression and embeddings alike, by the median before classification.

## Outputs

Each benchmark writes one parquet file (`test_files_<mode>_<disease|celltype>.parquet`) with one row per task, fold and representation. Main columns: `dataset_id`, `cell_type`, `feature` (representation), `fold`, `n_per_class` and `bootstrap` (low-data only), and the metrics `F1_macro`, `MCC`, `balanced_acc`, `recall_macro` (plus donor-level variants). Rows with a `skip_reason` or `error` are tasks or folds that could not be evaluated.

## Statistics and figures

Statistical comparisons and figures are produced from these parquet files (notebooks and the modules `friedman_ranking.py` and `fm_difficulty_analysis.py`):

- Representations are compared per sample size with the Friedman test (Iman-Davenport correction), followed by pairwise Wilcoxon signed-rank tests with Holm correction.
- Disease results are collapsed to one score per dataset before testing (tasks within a dataset are correlated).
- Task difficulty tiers are tertiles of the mean full-data F1-macro of Raw Log-norm and PCA-50; TF-SAPIENS vs Raw Log-norm is then tested within each tier.


```

## Notes on this version

- The manifest step yields 916 disease (dataset, cell type) groups; 868 are retained after the feasibility checks at fetch time (at least two classes, and at least two donors per class).
- The cell-type manifest lists 74 datasets; 65 are benchmarked.
- Low-data results at 2-256 cells per class are computed with whatever cells a class has when it holds fewer than the requested number; the fraction of such tasks per sample size is reported in the paper's supplementary table.

## Data

No new data were generated. All single-cell data come from the CZ CELLxGENE Census, release 2025-01-30 (<https://chanzuckerberg.github.io/cellxgene-census/>).

## Citation


## Contact

