"""
Shared helpers for reading per-cell-type cache files produced by
02_run_benchmarks.py phase 1.

Why this exists: cell-type benchmarks must concatenate per-cell-type caches
across a dataset. Pre-v2 caches were written without a var DataFrame, so
var_names defaulted to positional integer strings — sc.concat would silently
mix different genes into the same column and produce artificially perfect
cell type prediction on raw features. The helper here enforces v2 caches and
uses an identity-aware outer join so marker genes aren't lost.
"""

import scanpy as sc
from anndata import AnnData


def load_and_concat_celltype_caches(cache_paths: list[str]) -> AnnData:
    """Load per-cell-type caches and safely concat on real gene identity."""
    adatas = [sc.read_h5ad(p) for p in cache_paths]

    for path, a in zip(cache_paths, adatas):
        if a.uns.get("cache_format_version", 1) < 2:
            raise RuntimeError(
                f"Cache {path} is pre-v2 (no gene identity in .var). "
                "Rebuild via phase 1 or run scripts/migrate_cache_v1_to_v2.py."
            )

    if len(adatas) == 1:
        adata = adatas[0]
    else:
        # Outer join on real gene IDs with zero fill: a gene dropped from a
        # cache was zero across every cell of that cell type by construction,
        # so zero-fill reconstructs the true matrix. Inner would silently
        # discard marker genes — exactly the ones that distinguish cell types.
        adata = sc.concat(adatas, join="outer", label=None, fill_value=0)

    if not adata.var_names.str.startswith(("ENSG", "ENSMUSG")).any():
        raise RuntimeError(
            f"var_names after concat don't look like Ensembl IDs "
            f"(first few: {list(adata.var_names[:5])}). Cache integrity issue."
        )

    return adata
