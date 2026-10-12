"""
Preprocessing utilities for scRNA-seq data.

SINGLE SOURCE OF TRUTH for preprocessing.

Critical order (LOCKED):
1. drop_zero_genes() - Remove genes with zero expression
2. normalize_log1p() - Normalize to 10k + log1p transform
3. Scaling happens in feature extractors (MaxAbsScaler for sparse, StandardScaler for dense)
"""

import numpy as np
from scipy import sparse


def drop_zero_genes(X: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Remove genes with zero expression across all cells.

    Args:
        X: Raw count matrix (cells × genes)

    Returns:
        X_filtered: Matrix with zero-expression genes removed
        gene_mask: Boolean mask of kept genes

    """
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    gene_mask = np.array(X.sum(axis=0)).flatten() > 0
    X_filtered = X[:, gene_mask]

    return X_filtered, gene_mask


def normalize_log1p(X: sparse.csr_matrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    """
    Normalize counts per cell to target sum and apply log1p transform.

    Args:
        X: Count matrix (cells × genes)
        target_sum: Target sum for normalization (default 10k)

    Returns:
        X_normalized: Log-normalized matrix

    """
    X = X.tocsr().copy()

    # Calculate counts per cell
    counts_per_cell = np.array(X.sum(axis=1)).flatten()
    counts_per_cell[counts_per_cell == 0] = 1  # Avoid division by zero

    # Normalize to target sum
    scaling_factors = target_sum / counts_per_cell
    scale_mat = sparse.diags(scaling_factors)
    X = scale_mat @ X

    # Log1p transform
    X.data = np.log1p(X.data)

    return X


def preprocess_counts(X_raw: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Complete preprocessing pipeline.

    THIS IS THE ONLY FUNCTION TO USE EXTERNALLY.
    Order is LOCKED and cannot be changed.

    Steps:
    1. Drop zero genes globally
    2. Normalize to 10k + log1p
    3. (Scaling happens in feature extractors)

    Args:
        X_raw: Raw count matrix

    Returns:
        X_lognorm: Preprocessed log-normalized matrix
        gene_mask: Boolean mask of kept genes

    """
    X_filtered, gene_mask = drop_zero_genes(X_raw)
    X_lognorm = normalize_log1p(X_filtered, target_sum=1e4)
    return X_lognorm, gene_mask


# Alias for backward compatibility
sparse_normalize_log1p = normalize_log1p
