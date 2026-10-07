#!/bin/bash

python 01_prepare_task_manifest.py \
  --out_path disease_manifest.parquet

python 02_run_benchmarks.py \
  --log_file 02_run_benchmark.log \
  --tasks_manifest disease_manifest.parquet \
  --out_parquet results.parquet 

python 03_make_celltype_manifest.py \
  --disease_manifest disease_manifest.parquet \
  --out_path celltype_manifest.parquet

python ./scripts/migrate_cache_v1_to_v2.py     --cache_dir ./_adata_cache --census_version 2025-01-30 

python ./scripts/migrate_cache_v2_to_v3.py --cache_dir ./_adata_cache 


python ./scripts/01_prepare_bmfm_input.py \
        --celltype_manifest  celltype_manifest.parquet \
        --cache_dir          ./_adata_cache \
        --bmfm_input_dir     ./_bmfm_input  \
        --census_version     2025-01-30     \

python ./scripts/01b_convert_gene_symbols.py --bmfm_input_dir ./_bmfm_input 


python ./scripts/02_run_bmfm.py \
        --bmfm_input_dir  ./_bmfm_input_have_embedding  \
        --bmfm_output_dir ./_bmfm_output_MLMMUL \
        --checkpoint      ibm-research/biomed.rna.llama.32m.mlm.multitask.v1      


python ./scripts/02_run_bmfm.py \
        --bmfm_input_dir  ./_bmfm_input  \
        --bmfm_output_dir ./_bmfm_output \
        --checkpoint      ibm-research/biomed.rna.llama.47m.wced.multitask.v1   


python ./scripts/03_merge_bmfm_embeddings.py \
        --celltype_manifest  ./disease_manifest.parquet \
        --cache_dir          ./_adata_cache  \
        --bmfm_output_dir    ./_bmfm_output_MLMMUL  \
        --embedding_key      bmfm_MLMMUL            \
        --log_file 03_merge_bmfm_embeddings_MLMMUL.log


python ./scripts/03_merge_bmfm_embeddings.py \
        --celltype_manifest  ./disease_manifest.parquet \
        --cache_dir          ./_adata_cache  \
        --bmfm_output_dir    ./_bmfm_output  \
        --embedding_key      bmfm            \
        --log_file 03_merge_bmfm_embeddings.log



python ./scripts/04_concat_embeddings.py \
        --celltype_manifest ./disease_manifest.parquet \
        --cache_dir         ./_adata_cache  \
        --key_1             bmfm            \
        --key_2             bmfm_MLMMUL     \
        --output_key        bmfm_var1     \


python 02_run_benchmarks_multimode_phase2.py \
  --mode standard \
  --cache_dir ./_adata_cache \
  --out_parquet test_files_standard_disease.parquet \
  --log_file test_files_standard_disease.log \
  --embeddings bmfm_var1 scvi geneformer tf-sapiens \
  --baselines raw_lognorm raw_pca50 random_proj50

python 02_run_benchmarks_multimode_phase2.py \
  --mode low_data \
  --cache_dir ./_adata_cache \
  --out_parquet test_files_low_disease.parquet \
  --log_file test_files_low_diseaselog \
  --n_per_class 2 4 8 16 32 64 128 256 \
  --n_bootstrap 5 \
  --keep_temp \
  --tasks_manifest disease_manifest.parquet \
  --embeddings bmfm_var1 scvi geneformer tf-sapiens \
  --baselines raw_lognorm raw_pca50 random_proj50

python 02_run_benchmarks_multimode_phase2.py \
  --mode pseudobulk \
  --cache_dir ./_adata_cache \
  --out_parquet test_files_pseudobulk_disease.parquet \
  --log_file test_files_pseudobulk_disease.log \
  --embeddings bmfm_var1 scvi geneformer tf-sapiens \
  --tasks_manifest disease_manifest.parquet \
  --baselines raw_lognorm raw_pca50 random_proj50



python 04_benchmark_celltype_multimode.py \
  --mode standard    \
  --cache_dir ./_adata_cache \
  --out_parquet test_files_standard_celltype.parquet \
  --log_file test_files_standard_celltype.log \
  --celltype_manifest celltype_manifest.parquet \
  --embeddings bmfm_var1 scvi geneformer tf-sapiens \
  --baselines raw_lognorm raw_pca50 random_proj50


python 04_benchmark_celltype_multimode.py \
  --mode low_data   \
  --cache_dir ./_adata_cache \
  --out_parquet test_files_low_celltype.parquet \
  --log_file test_files_low_celltype.log \
  --celltype_manifest celltype_manifest.parquet \
  --n_per_class 2 4 8 16 32 64 128 256 \
  --n_bootstrap 5 \
  --embeddings bmfm_var1 scvi geneformer tf-sapiens \
  --baselines raw_lognorm raw_pca50 random_proj50

python 04_benchmark_celltype_multimode.py \
  --mode pseudobulk \
  --cache_dir ./_adata_cache \
  --out_parquet test_files_pseudobulk_celltype.parquet \
  --log_file test_files_pseudobulk_celltype.log \
  --celltype_manifest celltype_manifest.parquet \
  --embeddings bmfm_var1 scvi geneformer tf-sapiens \
  --baselines raw_lognorm raw_pca50 random_proj50
