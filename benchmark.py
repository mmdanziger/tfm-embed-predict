"""
Core benchmark functions for cross-validated predictions.

Main entry point: run_predictions() - handles CV loop and returns results DataFrame.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from typing import Dict, List, Tuple, Optional, Callable

from metrics import compute_donor_metrics, compute_metrics

# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(
    log_file: str,
    logger_name: str = "benchmark",
    include_console: bool = True,
    console_level: int = logging.INFO,
    include_thread_name: bool = False,
    filter_warnings: bool = False,
) -> logging.Logger:
    """
    Configure logging for benchmark runs.

    Parameters
    ----------
    log_file : str
        Path to log file
    logger_name : str, default="benchmark"
        Name of the logger to configure
    include_console : bool, default=True
        Whether to add console handler
    console_level : int, default=logging.INFO
        Logging level for console handler
    include_thread_name : bool, default=False
        Whether to include thread name in file log format
    filter_warnings : bool, default=False
        Whether to filter RuntimeWarning and UserWarning

    Returns
    -------
    logging.Logger
        Configured logger instance

    """
    logging.captureWarnings(True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    if include_thread_name:
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(threadName)-15s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    else:
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    logger.addHandler(fh)

    # Console handler
    if include_console:
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        if console_level == logging.INFO:
            ch.setFormatter(
                logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
            )
        else:
            ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(ch)

    # Warning filters
    if filter_warnings:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
    else:
        warnings.filterwarnings("ignore")

    # Special handling for py.warnings logger (used in 04_run_celltype_benchmark.py)
    if include_thread_name:
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.WARNING)
        warnings_logger.handlers = []
        warnings_logger.propagate = False
        warnings_logger.addHandler(fh)

    return logger


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

def fold_iterator(X, labels, sample_ids, n_folds, cv_folds=None, random_state: int = 0):
    if cv_folds is not None:
        for fold_id, test_samples in enumerate(cv_folds):
            test_mask = np.isin(sample_ids, test_samples)
            train_mask = ~test_mask
            yield fold_id, np.where(train_mask)[0], np.where(test_mask)[0]
    else:
        sgkf = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )
        for fold_id, (train_idx, test_idx) in enumerate(
            sgkf.split(X, labels, groups=sample_ids)
        ):
            yield fold_id, train_idx, test_idx

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
    cv_folds: Optional[List[List[str]]] = None,
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

    #labels = adata.obs[label_col].values
    #labels = adata.obs["disease"].astype(str).values
    labels = np.array(adata.obs[label_col], dtype=str)
    donor_ids = adata.obs[donor_col].values
    #stratify_labels = adata.obs[stratify_col].values if stratify_col else labels
    stratify_labels = np.array(adata.obs[stratify_col], dtype=str) if stratify_col else labels

    
    
    
    # ---- encode ----
    #from sklearn.preprocessing import LabelEncoder
    #label_encoder = LabelEncoder()
    #y_ = label_encoder.fit_transform(labels)
    
    #logger.info(
    #    f"Label mapping: {dict(enumerate(label_encoder.classes_))}"
    #)

    results = []
    n_per_class = metadata.pop("n_per_class", None)
    random_seed_base = metadata.get("bootstrap", 0) + random_state

    assert stratify_labels.dtype == object or np.issubdtype(stratify_labels.dtype, np.str_), \
        f"stratify_labels dtype={stratify_labels.dtype}, values={np.unique(stratify_labels)}"    
    for fold_id, train_idx, test_idx in fold_iterator(
            X_lognorm, stratify_labels, donor_ids, n_folds, cv_folds, random_state
        ):
        y_train = labels[train_idx]
        y_test  = labels[test_idx]
        donors_test = donor_ids[test_idx]
        #import pdb; pdb.set_trace()
        ### downsample X
        

        if n_per_class is not None:
            train_idx, all_classes_met = downsample_to_n_per_class(
                train_idx=train_idx,
                y_train=y_train,
                n_per_class=n_per_class,
                random_state=random_seed_base + fold_id,  # unique per fold AND bootstrap,
                logger=logger,
            )
            y_train = labels[train_idx]
        ###     
        
        fold_name = f"fold_{fold_id + 1}"
        logger.info(
            f"  {fold_name}: {len(train_idx)} train, {len(test_idx)} test cells"
        )
        #if fold_id==2: import pdb; pdb.set_trace()
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
                    early_stopping=len(y_train) >= 50,  # Adaptive
                )
                clf.fit(X_train, y_train)

                # Predict
                y_pred = clf.predict(X_test)

                # Try to get probabilities (can fail with numerical issues in multiclass)
                try:
                    y_proba = clf.predict_proba(X_test)
                    # Check for NaN in probabilities
                    if np.isnan(y_proba).any():
                        logger.warning(
                            f"  {rep_name} in {fold_name}: predict_proba produced NaN, "
                            "falling back to predict-only metrics"
                        )
                        y_proba = None
                except (ValueError, RuntimeWarning) as e:
                    logger.warning(
                        f"  {rep_name} in {fold_name}: predict_proba failed ({e}), "
                        "using predict-only metrics"
                    )
                    y_proba = None
                

                # Compute metrics (handle case where probabilities unavailable)
                cell_metrics = compute_metrics(y_test, y_proba, y_pred, clf.classes_)
                donor_metrics = compute_donor_metrics(
                    y_test, y_pred, donors_test, clf.classes_
                )

                results.append(
                    {
                        "fold": fold_name,
                        "representation": rep_name,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "n_per_class":n_per_class,
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
def downsample_to_n_per_class(
    train_idx: np.ndarray,
    y_train: np.ndarray,
    n_per_class: int,
    random_state: int,
    logger: logging.Logger
) -> Tuple[np.ndarray, bool]:
    """Downsample training data to N samples per class."""
    rng = np.random.RandomState(random_state)
    selected_idx = []
    all_classes_met = True
    
    for class_label in np.unique(y_train):
        class_mask = y_train == class_label
        class_indices = train_idx[class_mask]
        n_available = len(class_indices)
        
        if n_available < n_per_class:
            logger.warning(
                f"    Class {class_label}: only {n_available}/{n_per_class} available"
            )
            selected_idx.extend(class_indices)
            all_classes_met = False
        else:
            selected = rng.choice(class_indices, size=n_per_class, replace=False)
            selected_idx.extend(selected)
    
    return np.array(selected_idx), all_classes_met


def run_predictions_downsampled(
    adata,
    label_col: str,
    representations: dict,
    n_per_class_list: list,
    n_bootstrap: int = 10,
    cv_folds: Optional[List[List[str]]] = None,
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
                cv_folds=cv_folds,
                **kwargs,
            )
            all_results.append(results)
    return pd.concat(all_results, ignore_index=True)
