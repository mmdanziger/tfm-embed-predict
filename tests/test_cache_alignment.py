"""
Regression tests for the cache gene-identity bug.

Pre-fix, per-cell-type cache files had no .var (var_names defaulted to
positional integer strings), so sc.concat across cell types silently mixed
different genes into the same column. These tests guard against that.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse


def _make_cache(
    cells: int,
    kept_gene_ids: list[str],
    nonzero_gene_id: str,
    cell_type: str,
    donor: str,
    disease: str = "normal",
    version: int = 2,
) -> AnnData:
    """Build a minimal preprocessed cache AnnData for one cell type."""
    n_genes = len(kept_gene_ids)
    X = np.zeros((cells, n_genes), dtype=np.float32)
    col = kept_gene_ids.index(nonzero_gene_id)
    X[:, col] = 3.14  # marker-ish signal for this cell type at this specific gene
    X = sparse.csr_matrix(X)

    obs = pd.DataFrame(
        {
            "dataset_id": ["ds"] * cells,
            "cell_type": [cell_type] * cells,
            "donor_id": [donor] * cells,
            "disease": [disease] * cells,
        },
        index=[f"{cell_type}_cell_{i}" for i in range(cells)],
    )
    var = pd.DataFrame(index=pd.Index(kept_gene_ids, name="feature_id"))

    a = AnnData(X=X, obs=obs, var=var)
    a.uns["cache_format_version"] = version
    return a


def test_outer_concat_preserves_gene_identity(tmp_path: Path):
    """A marker in cache A's GeneX must land in the GeneX column (not column 0)."""
    # Cache A keeps [GeneX, GeneY], with marker at GeneX.
    # Cache B keeps [GeneY, GeneZ], with marker at GeneZ.
    # Positional concat would stack them in column 0 for both → data leakage.
    # Identity-aware concat must put A's marker in GeneX, B's marker in GeneZ.
    a = _make_cache(
        cells=5,
        kept_gene_ids=["ENSG_X", "ENSG_Y"],
        nonzero_gene_id="ENSG_X",
        cell_type="type_a",
        donor="d1",
    )
    b = _make_cache(
        cells=5,
        kept_gene_ids=["ENSG_Y", "ENSG_Z"],
        nonzero_gene_id="ENSG_Z",
        cell_type="type_b",
        donor="d2",
    )

    merged = sc.concat([a, b], join="outer", label=None, fill_value=0)
    assert set(merged.var_names) == {"ENSG_X", "ENSG_Y", "ENSG_Z"}

    X = merged.X.toarray() if sparse.issparse(merged.X) else merged.X
    mask_a = merged.obs["cell_type"].values == "type_a"
    mask_b = merged.obs["cell_type"].values == "type_b"

    idx_x = list(merged.var_names).index("ENSG_X")
    idx_z = list(merged.var_names).index("ENSG_Z")

    assert (X[mask_a, idx_x] > 0).all(), "type_a marker lost from GeneX column"
    assert (X[mask_a, idx_z] == 0).all(), "type_a leaked into GeneZ column"
    assert (X[mask_b, idx_z] > 0).all(), "type_b marker lost from GeneZ column"
    assert (X[mask_b, idx_x] == 0).all(), "type_b leaked into GeneX column"


def test_positional_concat_demonstrates_original_bug(tmp_path: Path):
    """
    Sanity: with no .var, concat collapses to column position — the bug we fixed.

    Each cell type's marker gene happens to land in its cache's column 0 (a
    different real gene in each cache). Under the old positional concat, column
    0 of the merged matrix is nonzero for both cell types — but representing
    *different* genes. That's the data leak we're guarding against.
    """

    def without_var(a: AnnData) -> AnnData:
        bare = AnnData(X=a.X.copy(), obs=a.obs.copy())
        bare.uns["cache_format_version"] = 1
        return bare

    # Cache A: marker at its column 0 = GeneX.
    a_bad = without_var(_make_cache(5, ["ENSG_X", "ENSG_Y"], "ENSG_X", "type_a", "d1"))
    # Cache B: marker at its column 0 = GeneZ (different gene, same position).
    b_bad = without_var(_make_cache(5, ["ENSG_Z", "ENSG_Y"], "ENSG_Z", "type_b", "d2"))

    merged = sc.concat([a_bad, b_bad], join="inner")
    X = merged.X.toarray() if sparse.issparse(merged.X) else merged.X
    mask_a = merged.obs["cell_type"].values == "type_a"
    mask_b = merged.obs["cell_type"].values == "type_b"

    # Column 0 of the merged matrix holds GeneX for a-cells and GeneZ for b-cells
    # — positional collapse. Both groups show nonzero, confirming the mix.
    msg = (
        "expected positional collapse — if this fails, sc.concat no longer "
        "silently positionally-aligns missing-var AnnDatas and this test can retire"
    )
    assert X[mask_a, 0].mean() > 0, msg
    assert X[mask_b, 0].mean() > 0, msg


def test_load_and_concat_helper_rejects_pre_v2(tmp_path: Path):
    """The shared helper must refuse to proceed on pre-v2 caches."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cache_io import load_and_concat_celltype_caches

    a_v1 = _make_cache(5, ["ENSG_X", "ENSG_Y"], "ENSG_X", "type_a", "d1", version=1)
    p = tmp_path / "v1.h5ad"
    a_v1.write_h5ad(p)

    with pytest.raises(RuntimeError, match="pre-v2"):
        load_and_concat_celltype_caches([str(p)])
