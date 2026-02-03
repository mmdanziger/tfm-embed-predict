"""
Core benchmark functions for cross-validated predictions.

Main entry point: run_predictions() - handles CV loop and returns results DataFrame.
"""

import logging

import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from metrics import compute_donor_metrics, compute_metrics
from preprocessing import preprocess_counts

# =============================================================================
# Feature Extractors
# =============================================================================


def extract_raw(X_lognorm, train_idx, test_idx):
    """
    Extract raw log-normalized counts with MaxAbsScaler.

    MaxAbsScaler preserves sparsity and scales to [-1, 1] range.
    """
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_lognorm[train_idx])
    X_test = scaler.transform(X_lognorm[test_idx])
    return X_train, X_test


def extract_pca(X_lognorm, train_idx, test_idx, n_components=50):
    """
    Extract PCA features with StandardScaler.

    Uses TruncatedSVD for sparse matrices + StandardScaler.
    """
    # Ensure n_components doesn't exceed matrix dimensions
    n_comp = min(n_components, len(train_idx) - 1, X_lognorm.shape[1] - 1)

    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    X_train_pca = svd.fit_transform(X_lognorm[train_idx])
    X_test_pca = svd.transform(X_lognorm[test_idx])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_pca)
    X_test = scaler.transform(X_test_pca)
    return X_train, X_test


def extract_embedding(embedding_matrix, train_idx, test_idx):
    """
    Extract pre-computed embeddings with StandardScaler.

    For dense embeddings like scVI, Geneformer, etc.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(embedding_matrix[train_idx])
    X_test = scaler.transform(embedding_matrix[test_idx])
    return X_train, X_test


def build_representations(adata, embedding_keys=None):
    """
    Build representation dict for run_predictions().

    Args:
        adata: AnnData object
        embedding_keys: List of embedding keys in adata.obsm (e.g., ["scvi", "geneformer"])

    Returns:
        Dictionary of {name: extractor_func}

    """
    reps = {
        "raw_lognorm": extract_raw,
        "raw_pca50": lambda X, tr, te: extract_pca(X, tr, te, n_components=50),
    }

    if embedding_keys:
        for key in embedding_keys:
            if key in adata.obsm:
                emb = adata.obsm[key]
                # Use closure to capture embedding
                reps[f"obsm[{key}]"] = lambda X, tr, te, e=emb: extract_embedding(
                    e, tr, te
                )

    return reps


# =============================================================================
# Core Prediction Function
# =============================================================================


def run_predictions(
    adata,
    label_col: str,
    representations: dict,
    n_folds: int = 5,
    alpha: float = 1e-5,
    donor_col: str = "donor_id",
    stratify_col: str = None,
    random_state: int = 0,
    preprocess: bool = True,
    **metadata,
) -> pd.DataFrame:
    """
    Run cross-validated predictions.

    This is the CORE function that handles:
    - Preprocessing (via preprocess_counts) - optional if data already preprocessed
    - CV splitting (StratifiedGroupKFold by donors)
    - Feature extraction (via representation extractors)
    - Training (SGDClassifier)
    - Prediction and metrics computation

    Works for:
    - Cell-level or donor-level (pseudobulk adata before calling)
    - Any label column (disease, celltype, etc.)
    - Any representation set (raw, PCA, embeddings, etc.)

    Args:
        adata: AnnData object (can be cell-level or donor-level pseudobulk)
        label_col: Column in adata.obs for labels ("disease", "celltype", etc.)
        representations: Dict of {name: extractor_func(X, train_idx, test_idx)}
        n_folds: Number of CV folds
        alpha: L2 regularization for SGDClassifier
        donor_col: Column for donor/sample IDs (used for grouping)
        stratify_col: Column to stratify on (default: same as label_col)
        random_state: Random seed for reproducibility
        preprocess: Whether to preprocess counts (set False if already preprocessed)
        **metadata: Additional fields to include in results (e.g., dataset_id, cell_type)

    Returns:
        DataFrame with one row per (fold, representation) combination

    """
    logger = logging.getLogger("benchmark")

    # Preprocess counts (if needed)
    if preprocess:
        X_lognorm, gene_mask = preprocess_counts(adata.X)
    else:
        # Data already preprocessed (e.g., from cache)
        X_lognorm = adata.X if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)

    labels = adata.obs[label_col].values
    donor_ids = adata.obs[donor_col].values
    stratify_labels = adata.obs[stratify_col].values if stratify_col else labels

    # CV splits (StratifiedGroupKFold by donors)
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )

    results = []

    for fold_id, (train_idx, test_idx) in enumerate(
        sgkf.split(X_lognorm, stratify_labels, groups=donor_ids)
    ):
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        donors_test = donor_ids[test_idx]

        fold_name = f"fold_{fold_id + 1}"
        logger.info(
            f"  {fold_name}: {len(train_idx)} train, {len(test_idx)} test cells"
        )

        # For each representation
        for rep_name, extractor in representations.items():
            try:
                # Extract features
                X_train, X_test = extractor(X_lognorm, train_idx, test_idx)

                # Train classifier
                clf = SGDClassifier(
                    loss="log_loss",
                    alpha=alpha,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                    early_stopping=len(y_train) >= 10,  # Adaptive
                )
                clf.fit(X_train, y_train)

                # Predict
                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)

                # Compute metrics
                cell_metrics = compute_metrics(y_test, y_proba, y_pred, clf.classes_)
                donor_metrics = compute_donor_metrics(
                    y_test, y_proba, donors_test, clf.classes_
                )

                results.append(
                    {
                        "fold": fold_name,
                        "representation": rep_name,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        **metadata,
                        **cell_metrics,
                        **donor_metrics,
                    }
                )

            except Exception as e:
                logger.error(f"  Error with {rep_name} in {fold_name}: {e}")
                results.append(
                    {
                        "fold": fold_name,
                        "representation": rep_name,
                        "error": str(e),
                        **metadata,
                    }
                )

    return pd.DataFrame(results)


# =============================================================================
# Optional: Downsampling Wrapper
# =============================================================================


def run_predictions_downsampled(
    adata,
    label_col: str,
    representations: dict,
    n_per_class_list: list,
    n_bootstrap: int = 10,
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper for low-data regime experiments.

    Runs run_predictions() with downsampled training sets.

    Args:
        adata: AnnData object
        label_col: Label column
        representations: Representation dict
        n_per_class_list: List of sample sizes to try (e.g., [10, 25, 50, 100])
        n_bootstrap: Number of bootstrap iterations per sample size
        **kwargs: Additional arguments to pass to run_predictions()

    Returns:
        Concatenated DataFrame with all results

    """
    all_results = []

    for n_per_class in n_per_class_list:
        for bootstrap in range(n_bootstrap):
            results = run_predictions(
                adata,
                label_col,
                representations,
                n_per_class=n_per_class,
                bootstrap=bootstrap,
                **kwargs,
            )
            all_results.append(results)

    return pd.concat(all_results, ignore_index=True)
