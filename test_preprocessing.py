#!/usr/bin/env python3
"""Quick test to verify preprocessing equivalence."""

import sys

import numpy as np
from scipy import sparse

# Add archive directory to path to import old implementation
sys.path.insert(0, "_archive")
from modular_benchmark import lognorm_cp10k, preprocess_expression_matrix

# Import new implementation
from preprocessing import preprocess_counts

# Create simple test data
np.random.seed(42)
X_test = sparse.random(100, 500, density=0.1, format="csr", random_state=42)
X_test.data = np.abs(X_test.data) * 100  # Make it look like counts

# Old preprocessing
X_old_filtered, gene_mask_old = preprocess_expression_matrix(
    type("adata", (), {"X": X_test})()
)
X_old = lognorm_cp10k(X_old_filtered)

# New preprocessing
X_new, gene_mask_new = preprocess_counts(X_test)

# Compare gene masks
print("Gene masks match:", np.array_equal(gene_mask_old, gene_mask_new))

# Compare preprocessed matrices
print("Data allclose:", np.allclose(X_old.data, X_new.data))
print("Shape matches:", X_old.shape == X_new.shape)

if np.allclose(X_old.data, X_new.data):
    print("✓ Preprocessing equivalence verified!")
else:
    print("✗ Preprocessing differs!")
    print(f"Max diff: {np.abs(X_old.data - X_new.data).max()}")
