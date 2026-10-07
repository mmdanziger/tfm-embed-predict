import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from tqdm.auto import tqdm

# ============================================================
# Folds: leave out 1 diseased donor + closest healthy donor
# ============================================================


def build_leave_out_pairs(
    adata, disease_col="disease", donor_col="donor_id", normal_label="normal"
):
    obs = adata.obs.copy()
    counts = (
        obs.groupby([disease_col, donor_col], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )

    healthy = counts[counts[disease_col] == normal_label].copy()
    diseased = counts[counts[disease_col] != normal_label].copy()

    if healthy.empty:
        raise ValueError("No healthy donors found.")
    if diseased.empty:
        raise ValueError("No diseased donors found.")

    healthy_counts = healthy.set_index(donor_col)["n_cells"]

    folds = []
    for _, row in diseased.iterrows():
        dis_name = row[disease_col]
        dis_donor = row[donor_col]
        dis_n = row["n_cells"]
        healthy_donor = (healthy_counts - dis_n).abs().idxmin()
        folds.append((dis_name, dis_donor, healthy_donor))

    return folds, counts


# ============================================================
# Shared classifier for EVERYTHING (embeddings + raw variants)
# ============================================================


def make_shared_clf(random_state=0):
    """
    One classifier type for all feature sets (dense or sparse):
    SGD logistic regression (loss=log_loss) works with both.
    """
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight="balanced",
        max_iter=2000,
        tol=1e-3,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        random_state=random_state,
    )


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def proba_pos(clf, X):
    return sigmoid(clf.decision_function(X))


# ============================================================
# Metrics
# ============================================================


def compute_metrics(y_true, p_pos):
    y_hat = (p_pos >= 0.5).astype(int)
    auroc = roc_auc_score(y_true, p_pos)
    auprc = average_precision_score(y_true, p_pos)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "balanced_acc": balanced_accuracy_score(y_true, y_hat),
        "F1": f1_score(y_true, y_hat, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_hat),
        "log_loss": log_loss(y_true, np.c_[1 - p_pos, p_pos], labels=[0, 1]),
        "brier": brier_score_loss(y_true, p_pos),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "test_pos_rate": float(y_true.mean()),
    }


# ============================================================
# Raw preprocessing building blocks
# ============================================================


def to_csr(X):
    return X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)


def drop_global_all_zero_genes(X_csr):
    nnz_per_col = np.asarray((X_csr != 0).sum(axis=0)).ravel()
    keep = nnz_per_col > 0
    if keep.sum() == 0:
        raise ValueError("All genes are all-zero.")
    return X_csr[:, keep], keep


def normalize_total_csr(X_csr, target_sum=1e4):
    """
    Library-size normalize sparse counts:
      X_norm[i, :] = X[i, :] / sum_i * target_sum
    Rows with sum==0 remain all-zero.
    """
    X_csr = X_csr.tocsr(copy=True)
    row_sums = np.asarray(X_csr.sum(axis=1)).ravel()
    scale = np.zeros_like(row_sums, dtype=np.float32)
    nz = row_sums > 0
    scale[nz] = (target_sum / row_sums[nz]).astype(np.float32)

    # Multiply each row by its scale factor
    # Efficient CSR row scaling:
    X_csr = X_csr.tocsr()
    indptr = X_csr.indptr
    data = X_csr.data
    for i in range(X_csr.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if start != end:
            data[start:end] *= scale[i]
    return X_csr


def log1p_csr_inplace(X_csr):
    X_csr = X_csr.tocsr(copy=True)
    X_csr.data = np.log1p(X_csr.data)
    return X_csr


def make_presence_features(X_csr):
    """Binary presence matrix (same sparsity pattern as X, values=1)."""
    Xb = X_csr.copy()
    Xb.data[:] = 1.0
    return Xb


def hvg_topk_by_variance(X_csr, k=512):
    """
    Unsupervised HVG proxy: select top-k genes by variance in log-normalized space.
    For sparse matrices we approximate var via E[x^2] - E[x]^2.
    """
    n = X_csr.shape[0]
    mu = np.asarray(X_csr.mean(axis=0)).ravel()  # E[x]
    ex2 = np.asarray(X_csr.multiply(X_csr).mean(axis=0)).ravel()  # E[x^2]
    var = ex2 - mu * mu
    if k >= X_csr.shape[1]:
        keep = np.ones(X_csr.shape[1], dtype=bool)
        return X_csr, keep
    idx = np.argpartition(var, -k)[-k:]
    idx = np.sort(idx)
    keep = np.zeros(X_csr.shape[1], dtype=bool)
    keep[idx] = True
    return X_csr[:, keep], keep


def zscore_dense_global(X):
    return StandardScaler().fit_transform(np.asarray(X))


def scale_sparse_global(X_csr):
    return MaxAbsScaler().fit_transform(X_csr)


def pca_sparse_global(X_csr, n_components=50, random_state=0):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    return svd.fit_transform(X_csr), svd


# ============================================================
# Build feature sets (embeddings + raw variants)
# ============================================================


def build_feature_sets(
    adata,
    embedding_keys=("scvi", "geneformer"),
    # raw knobs
    target_sum=1e4,
    include_raw_lognorm=True,
    include_raw_lognorm_plus_binary=True,
    include_raw_hvg512=True,
    include_raw_best_pca=True,
    pca_components_grid=(20, 50, 100),  # small, reasonable search
    random_state=0,
):
    feats = []

    # Embeddings (zscore; optional but recommended)
    for k in embedding_keys:
        if k not in adata.obsm:
            raise KeyError(f"{k!r} not found in adata.obsm")
        X = zscore_dense_global(adata.obsm[k])
        feats.append((f"obsm[{k}] (zscore)", X, "dense"))

    # Raw counts -> global drop dead genes -> normalize_total -> log1p
    Xg = to_csr(adata.X)
    Xg, keepg = drop_global_all_zero_genes(Xg)
    kept, total = int(keepg.sum()), int(keepg.size)

    Xg_ln = normalize_total_csr(Xg, target_sum=target_sum)
    Xg_ln = log1p_csr_inplace(Xg_ln)
    Xg_ln = scale_sparse_global(Xg_ln)  # improves conditioning, still unsupervised

    if include_raw_lognorm:
        feats.append((f"X lognorm (nz {kept}/{total})", Xg_ln, "sparse"))

    if include_raw_lognorm_plus_binary:
        Xb = make_presence_features(Xg_ln)  # same pattern, values=1
        X_aug = sparse.hstack([Xg_ln, Xb], format="csr")
        feats.append((f"X lognorm + binary (nz {kept}/{total})", X_aug, "sparse"))

    if include_raw_hvg512:
        X_hvg, keep_hvg = hvg_topk_by_variance(Xg_ln, k=512)
        feats.append(
            (
                f"X HVG512 lognorm (nz {keep_hvg.sum()}/{Xg_ln.shape[1]})",
                X_hvg,
                "sparse",
            )
        )

    if include_raw_best_pca:
        # pick "best" PCA dimension using a cheap unsupervised heuristic:
        # choose the middle of the small grid by default, but expose options.
        # If you want true performance-based selection, do it only on TRAIN folds.
        # Here we keep it simple and not crazy.
        best_k = (
            int(sorted(pca_components_grid)[1])
            if len(pca_components_grid) >= 2
            else int(pca_components_grid[0])
        )
        X_pca, _ = pca_sparse_global(
            Xg_ln, n_components=best_k, random_state=random_state
        )
        X_pca = StandardScaler().fit_transform(X_pca)
        feats.append((f"X lognorm + PCA{best_k} (SVD)", X_pca, "dense"))

    return feats


# ============================================================
# Benchmark runner
# ============================================================


def run_benchmark(
    adata,
    embedding_keys=("scvi", "geneformer"),
    disease_col="disease",
    donor_col="donor_id",
    normal_label="normal",
    diseased_label="type 1 diabetes mellitus",
    # raw variants
    target_sum=1e4,
    include_raw_lognorm=True,
    include_raw_lognorm_plus_binary=True,
    include_raw_hvg512=True,
    include_raw_best_pca=True,
    pca_components_grid=(20, 50, 100),
    random_state=0,
):
    disease = adata.obs[disease_col].astype(str).to_numpy()
    donors = adata.obs[donor_col].astype(str).to_numpy()

    valid = {normal_label, diseased_label}
    bad = set(np.unique(disease)) - valid
    if bad:
        raise ValueError(f"Unexpected disease labels: {bad} (expected only {valid}).")

    y = (disease == diseased_label).astype(int)

    folds, counts = build_leave_out_pairs(
        adata, disease_col=disease_col, donor_col=donor_col, normal_label=normal_label
    )

    feats = build_feature_sets(
        adata,
        embedding_keys=embedding_keys,
        target_sum=target_sum,
        include_raw_lognorm=include_raw_lognorm,
        include_raw_lognorm_plus_binary=include_raw_lognorm_plus_binary,
        include_raw_hvg512=include_raw_hvg512,
        include_raw_best_pca=include_raw_best_pca,
        pca_components_grid=pca_components_grid,
        random_state=random_state,
    )

    rows = []
    total_tasks = len(folds) * len(feats)

    with tqdm(total=total_tasks, desc="Donor-pair benchmark", unit="fit") as pbar:
        for fold_idx, (dis_name, dis_donor, healthy_donor) in enumerate(folds):
            test_mask = (donors == dis_donor) | (donors == healthy_donor)
            train_mask = ~test_mask

            y_train = y[train_mask]
            y_test = y[test_mask]
            if len(np.unique(y_test)) < 2:
                raise RuntimeError(
                    f"Fold {fold_idx} test set has a single class; pairing failed?"
                )

            for feat_name, X_all, kind in feats:
                t0 = time.perf_counter()

                X_train = X_all[train_mask]
                X_test = X_all[test_mask]

                clf = make_shared_clf(random_state=random_state)
                clf.fit(X_train, y_train)
                p = proba_pos(clf, X_test)

                m = compute_metrics(y_test, p)
                m.update(
                    {
                        "feature": feat_name,
                        "fold": fold_idx,
                        "disease_left_out": dis_name,
                        "diseased_donor": dis_donor,
                        "healthy_donor": healthy_donor,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "elapsed_s": time.perf_counter() - t0,
                    }
                )
                rows.append(m)

                pbar.set_postfix(
                    {
                        "fold": fold_idx,
                        "feat": feat_name[:26] + ("…" if len(feat_name) > 26 else ""),
                        "AUROC": f"{m['AUROC']:.3f}",
                        "sec": f"{m['elapsed_s']:.2f}",
                    }
                )
                pbar.update(1)

    fold_df = pd.DataFrame(rows)

    metric_cols = [
        "AUROC",
        "AUPRC",
        "balanced_acc",
        "F1",
        "MCC",
        "log_loss",
        "brier",
        "elapsed_s",
    ]
    summary_df = (
        fold_df.groupby("feature", as_index=False)
        .agg(
            folds=("fold", "count"),
            **{f"{c}_mean": (c, "mean") for c in metric_cols},
            **{f"{c}_std": (c, "std") for c in metric_cols},
        )
        .sort_values(
            by=["AUROC_mean", "AUPRC_mean", "balanced_acc_mean"], ascending=False
        )
    )

    winners_df = (
        fold_df.sort_values(["fold", "MCC"], ascending=[True, False])
        .groupby("fold", as_index=False)
        .first()[
            [
                "fold",
                "feature",
                "MCC",
                "AUROC",
                "AUPRC",
                "balanced_acc",
                "F1",
                "elapsed_s",
                "diseased_donor",
                "healthy_donor",
            ]
        ]
        .rename(columns={"feature": "winner_feature"})
    )

    return fold_df, summary_df, winners_df, folds, counts


# =========================
# Example usage
# =========================
# fold_df, summary_df, winners_df, folds, counts = run_benchmark(
#     adata_hpap,
#     embedding_keys=("scvi", "geneformer"),
#     include_raw_lognorm=True,
#     include_raw_lognorm_plus_binary=True,
#     include_raw_hvg512=True,
#     include_raw_best_pca=True,
#     pca_components_grid=(20, 50, 100),
# )
# print(summary_df)
# display(winners_df)
# display(fold_df.sort_values(["feature", "fold"]))


#!/usr/bin/env python3
"""
Fast hyperparameter sweep for embedding-based cell type prediction, keeping an
"all nonzero genes" sparse baseline.

Key speed changes vs slow version:
- Keep ALL nonzero genes for raw lognorm baseline, but run ONLY fast SGD on it.
- Replace per-fold TruncatedSVD(PCA) with SparseRandomProjection (no expensive fit).
- Make SGD fast: no early_stopping, modest max_iter, tol, n_jobs=-1, average=True.
- Run heavier classifier grid only on embeddings and 50D projection baseline.
- Avoid unnecessary copies.
"""


import cellxgene_census
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.random_projection import SparseRandomProjection


def sparse_normalize_log1p(
    X: sparse.csr_matrix,
    target_sum: float = 1e4,
    dtype=np.float32,
) -> sparse.csr_matrix:
    """Normalize each row to target_sum and apply log1p, keeping CSR sparse."""
    X = X.tocsr(copy=True)
    counts = np.asarray(X.sum(axis=1)).ravel()
    counts[counts == 0] = 1.0
    scale_factors = (target_sum / counts).astype(dtype, copy=False)
    X = sparse.diags(scale_factors) @ X
    # log1p on nonzeros only
    X.data = np.log1p(X.data).astype(dtype, copy=False)
    return X.astype(dtype, copy=False)


def _clone_estimator(est):
    """Safe-ish sklearn clone without importing sklearn.base.clone (keeps minimal deps)."""
    return type(est)(**est.get_params())


def test_hyperparameters(
    dataset_id: str,
    census_uri: str | None,
    census_version: str,
    embedding_keys: list[str],
    n_splits: int = 5,
    rp_components: int = 50,
    rp_dense_output: bool = True,
) -> pd.DataFrame:
    print(f"Fetching data for dataset {dataset_id}...")

    obs_filter = f"is_primary_data==True and dataset_id=='{dataset_id}'"

    with cellxgene_census.open_soma(
        uri=census_uri, census_version=census_version
    ) as census:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="homo_sapiens",
            measurement_name="RNA",
            obs_value_filter=obs_filter,
            obs_column_names=["dataset_id", "cell_type", "donor_id", "disease"],
            obs_embeddings=embedding_keys,
        )

    print(f"Loaded {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    # Labels / groups
    adata.obs["disease"] = adata.obs["disease"].astype(str)
    adata.obs["donor_id"] = adata.obs["donor_id"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)

    y = adata.obs["cell_type"].to_numpy()
    donors = adata.obs["donor_id"].to_numpy()
    diseases = adata.obs["disease"].to_numpy()

    print(f"Cell types: {len(np.unique(y))}")
    print(f"Donors: {len(np.unique(donors))}")
    print(f"Diseases: {np.unique(diseases)}")

    # Embeddings
    embeddings: dict[str, np.ndarray] = {}
    for k in embedding_keys:
        if k in adata.obsm:
            emb = np.asarray(adata.obsm[k])
            embeddings[k] = emb
            print(f"\nEmbedding {k}: shape={emb.shape} dtype={emb.dtype}")
            print(f"  Range: [{emb.min():.3f}, {emb.max():.3f}]")
            print(f"  Mean: {emb.mean():.3f}, Std: {emb.std():.3f}")
            per_dim_std = emb.std(axis=0)
            print(
                f"  Per-dim std: min={per_dim_std.min():.3f}, max={per_dim_std.max():.3f}"
            )
        else:
            print(f"\nWarning: embedding '{k}' not found in adata.obsm")

    # Raw expression (sparse)
    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    # Keep ALL nonzero genes (drop genes that are entirely zero)
    gene_totals = np.asarray(X.sum(axis=0)).ravel()
    genes_mask = gene_totals > 0
    X_raw = X[:, genes_mask]
    print(
        f"\nGenes after filtering all-zero genes: {genes_mask.sum()} (kept all nonzero genes)"
    )

    # Normalize + log1p (sparse, float32)
    X_lognorm = sparse_normalize_log1p(X_raw, target_sum=1e4, dtype=np.float32)

    # CV setup
    n_donors = len(np.unique(donors))
    k_folds = min(n_splits, n_donors)
    sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=0)
    print(f"\nRunning {k_folds}-fold cross-validation...")

    # Scaling strategies for embeddings only
    scaling_strategies = [
        ("none", None),
        ("standard", StandardScaler()),
        ("minmax", MinMaxScaler()),
    ]

    # FAST sparse baseline configs (only these run on full gene space)
    sgd_base = {
        "loss": "log_loss",
        "penalty": "l2",
        "class_weight": "balanced",
        "max_iter": 300,
        "tol": 1e-3,
        "n_jobs": -1,
        "random_state": 0,
        "early_stopping": False,
        "average": True,
    }
    classifier_configs_sparse = [
        ("SGD_alpha1e-4", SGDClassifier(alpha=1e-4, **sgd_base)),
        ("SGD_alpha1e-5", SGDClassifier(alpha=1e-5, **sgd_base)),
    ]

    # Full grid for cheap features (embeddings + 50D projection)
    classifier_configs_dense = [
        # SGD on dense (fast)
        ("SGD_alpha1e-4", SGDClassifier(alpha=1e-4, **sgd_base)),
        ("SGD_alpha1e-5", SGDClassifier(alpha=1e-5, **sgd_base)),
        # LogisticRegression on dense low-dim (reasonable)
        (
            "LogReg_C1",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="saga",
                class_weight="balanced",
                max_iter=200,
                tol=1e-3,
                n_jobs=-1,
                random_state=0,
            ),
        ),
        (
            "LogReg_C10",
            LogisticRegression(
                penalty="l2",
                C=10.0,
                solver="saga",
                class_weight="balanced",
                max_iter=200,
                tol=1e-3,
                n_jobs=-1,
                random_state=0,
            ),
        ),
        (
            "LogReg_C100",
            LogisticRegression(
                penalty="l2",
                C=100.0,
                solver="saga",
                class_weight="balanced",
                max_iter=200,
                tol=1e-3,
                n_jobs=-1,
                random_state=0,
            ),
        ),
    ]

    # Projection baseline (no expensive fit per fold; transform is fast for sparse)
    rp50 = SparseRandomProjection(
        n_components=rp_components,
        dense_output=rp_dense_output,
        random_state=0,
    )

    results: list[dict] = []

    # Use dummy X for splitting; split is based on y + groups
    dummy_X = np.zeros_like(diseases, dtype=np.int8)

    for fold_i, (train_idx, test_idx) in enumerate(
        sgkf.split(dummy_X, diseases, groups=donors)
    ):
        y_train, y_test = y[train_idx], y[test_idx]

        # guard: need at least 2 classes in each side for meaningful metrics
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"\nFold {fold_i}: skipped (insufficient classes in train/test)")
            continue

        print(f"\nFold {fold_i}: {len(train_idx)} train, {len(test_idx)} test")

        X_train_ln = X_lognorm[train_idx]
        X_test_ln = X_lognorm[test_idx]

        # ---------------------------------------------------------------------
        # A) ALL-GENES sparse baseline (FAST only)
        # ---------------------------------------------------------------------
        for clf_name, clf in classifier_configs_sparse:
            clf_copy = _clone_estimator(clf)
            clf_copy.fit(X_train_ln, y_train)
            y_pred = clf_copy.predict(X_test_ln)

            results.append(
                {
                    "fold": fold_i,
                    "feature": "raw_lognorm_all_nonzero_genes",
                    "scaling": "none",
                    "classifier": clf_name,
                    "MCC": matthews_corrcoef(y_test, y_pred),
                    "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                    "F1_macro": f1_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                }
            )
            print(results[-1])

        # ---------------------------------------------------------------------
        # B) 50D projection baseline (PCA-like role, but cheap)
        # ---------------------------------------------------------------------
        X_train_rp = rp50.transform(X_train_ln)
        X_test_rp = rp50.transform(X_test_ln)

        scaler_rp = StandardScaler()
        X_train_rp = scaler_rp.fit_transform(X_train_rp)
        X_test_rp = scaler_rp.transform(X_test_rp)

        for clf_name, clf in classifier_configs_dense:
            clf_copy = _clone_estimator(clf)
            clf_copy.fit(X_train_rp, y_train)
            y_pred = clf_copy.predict(X_test_rp)

            results.append(
                {
                    "fold": fold_i,
                    "feature": f"rp{rp_components}",
                    "scaling": "standard",
                    "classifier": clf_name,
                    "MCC": matthews_corrcoef(y_test, y_pred),
                    "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                    "F1_macro": f1_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                }
            )
            print(results[-1])

        # ---------------------------------------------------------------------
        # C) Embeddings grid (scalers + classifiers)
        # ---------------------------------------------------------------------
        for emb_name, emb_matrix in embeddings.items():
            X_train_emb = emb_matrix[train_idx]
            X_test_emb = emb_matrix[test_idx]

            for scale_name, scaler in scaling_strategies:
                if scaler is None:
                    X_train_scaled = X_train_emb
                    X_test_scaled = X_test_emb
                else:
                    sc = (
                        _clone_estimator(scaler)
                        if hasattr(scaler, "get_params")
                        else type(scaler)()
                    )
                    X_train_scaled = sc.fit_transform(X_train_emb)
                    X_test_scaled = sc.transform(X_test_emb)

                for clf_name, clf in classifier_configs_dense:
                    clf_copy = _clone_estimator(clf)
                    clf_copy.fit(X_train_scaled, y_train)
                    y_pred = clf_copy.predict(X_test_scaled)

                    results.append(
                        {
                            "fold": fold_i,
                            "feature": emb_name,
                            "scaling": scale_name,
                            "classifier": clf_name,
                            "MCC": matthews_corrcoef(y_test, y_pred),
                            "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                            "F1_macro": f1_score(
                                y_test, y_pred, average="macro", zero_division=0
                            ),
                        }
                    )
                    print(results[-1])

    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TEST RESULTS")
    print("=" * 80)

    if results_df.empty:
        print("No results (all folds skipped).")
        return

    agg = (
        results_df.groupby(["feature", "scaling", "classifier"])
        .agg(
            {
                "MCC": ["mean", "std"],
                "balanced_acc": ["mean", "std"],
                "F1_macro": ["mean", "std"],
            }
        )
        .round(4)
    )

    agg.columns = ["_".join(col) for col in agg.columns]
    agg = agg.reset_index()

    print("\nBEST CONFIGURATION PER FEATURE (by mean MCC):")
    print("-" * 80)

    for feature in agg["feature"].unique():
        feat_data = agg[agg["feature"] == feature].sort_values(
            "MCC_mean", ascending=False
        )
        best = feat_data.iloc[0]
        print(f"\n{feature}:")
        print(f"  Best: {best['scaling']} + {best['classifier']}")
        print(f"  MCC: {best['MCC_mean']:.4f} ± {best['MCC_std']:.4f}")
        print(
            f"  Bal.Acc: {best['balanced_acc_mean']:.4f} ± {best['balanced_acc_std']:.4f}"
        )
        print(f"  F1_macro: {best['F1_macro_mean']:.4f} ± {best['F1_macro_std']:.4f}")

        print("  Top 3 configs:")
        for rank, (_, row) in enumerate(feat_data.head(3).iterrows(), start=1):
            print(
                f"    {rank}. {row['scaling']:8s} + {row['classifier']:15s} -> MCC={row['MCC_mean']:.4f}"
            )

    print("\n" + "=" * 80)
    print("SCALING STRATEGY COMPARISON (Embeddings Only):")
    print("-" * 80)

    emb_features = [
        f
        for f in agg["feature"].unique()
        if f not in ["raw_lognorm_all_nonzero_genes"] and not f.startswith("rp")
    ]
    for emb in emb_features:
        print(f"\n{emb}:")
        emb_data = (
            agg[agg["feature"] == emb]
            .groupby("scaling")["MCC_mean"]
            .mean()
            .sort_values(ascending=False)
        )
        for scale, mcc in emb_data.items():
            print(f"  {scale:8s}: MCC={mcc:.4f}")

    print("\n" + "=" * 80)
    print("CLASSIFIER COMPARISON (Embeddings, best scaling per feature):")
    print("-" * 80)

    for emb in emb_features:
        emb_data = agg[agg["feature"] == emb]
        best_scaling = emb_data.groupby("scaling")["MCC_mean"].mean().idxmax()
        print(f"\n{emb} (with {best_scaling} scaling):")
        clf_data = emb_data[emb_data["scaling"] == best_scaling].sort_values(
            "MCC_mean", ascending=False
        )
        for _, row in clf_data.iterrows():
            print(
                f"  {row['classifier']:15s}: MCC={row['MCC_mean']:.4f} ± {row['MCC_std']:.4f}"
            )

    print("\n" + "=" * 80)
    print("OVERALL WINNER:")
    print("-" * 80)
    best_overall = agg.sort_values("MCC_mean", ascending=False).iloc[0]
    print(f"Feature: {best_overall['feature']}")
    print(f"Scaling: {best_overall['scaling']}")
    print(f"Classifier: {best_overall['classifier']}")
    print(f"MCC: {best_overall['MCC_mean']:.4f} ± {best_overall['MCC_std']:.4f}")
    print(
        f"Balanced Acc: {best_overall['balanced_acc_mean']:.4f} ± {best_overall['balanced_acc_std']:.4f}"
    )
    print(
        f"F1_macro: {best_overall['F1_macro_mean']:.4f} ± {best_overall['F1_macro_std']:.4f}"
    )


from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit


def run_cv_experiment(
    adata_or_path,
    n_splits: int = 5,
    min_cells_per_class: int = 5,
    train_subsample_frac: float = 1.0,
    rp_components: int = 50,
    skip_incomplete_test_class_support: bool = True,
    # SGD params
    sgd_alpha: float = 1e-4,
    sgd_max_iter: int = 300,
    sgd_tol: float = 1e-3,
    # PCA options
    pca_genes_n_components: int = 100,  # OpenProblems uses 100
    pca_genes_whiten: bool = True,  # Whitening helps with conditioning
    run_pca_genes: bool = True,  # Run PCA on raw genes like OpenProblems
    # Problematic embedding handling
    increase_maxiter_on_nonconverge: bool = True,  # Auto-increase max_iter if needed
    # Split types
    run_donor_holdout: bool = True,
    run_simple_stratified: bool = True,
    # Logging
    log_every_folds: int = 1,
):
    """
    Run cross-validation with SGD classifiers on gene expression data.

    Two split strategies:
    - donor_holdout: Hard split where entire donors are held out (tests generalization)
    - simple_stratified: Easy split with random stratified sampling (tests separability)

    This allows diagnosing whether FM embeddings actually help with generalization
    vs just being good features for the cell types themselves.
    """

    def ts(msg: str):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    t0 = time.perf_counter()

    # Load data
    adata = (
        sc.read_h5ad(adata_or_path) if isinstance(adata_or_path, str) else adata_or_path
    )
    if not isinstance(adata, AnnData):
        raise ValueError("adata_or_path must be a .h5ad path or AnnData")

    y_str = adata.obs["cell_type"].astype(str).to_numpy()
    donors = adata.obs["donor_id"].astype(str).to_numpy()

    # === NEW: Filter rare classes ===
    if min_cells_per_class > 1:
        class_counts = pd.Series(y_str).value_counts()
        rare_classes = class_counts[class_counts < min_cells_per_class]
        if len(rare_classes) > 0:
            ts(
                f"Filtering {len(rare_classes)} rare classes (<{min_cells_per_class} cells): {rare_classes.to_dict()}"
            )
            valid_classes = class_counts[class_counts >= min_cells_per_class].index
            mask = np.isin(y_str, valid_classes)
            # Filter all arrays consistently
            adata = adata[mask].copy()
            y_str = y_str[mask]
            donors = donors[mask]
            # Also need to recompute X later, so just update adata and re-extract
            ts(f"  Remaining: {mask.sum()} cells, {len(valid_classes)} classes")

    classes, y = np.unique(y_str, return_inverse=True)
    n_classes = len(classes)
    n_donors = len(np.unique(donors))
    ts(
        f"Start | adata={adata.shape} | classes={n_classes} | donors={n_donors} | subsample={train_subsample_frac}"
    )

    # Prepare gene matrix: keep nonzero genes, log1p normalize
    X = sparse.csr_matrix(adata.X) if not sparse.issparse(adata.X) else adata.X.tocsr()
    X = X[:, (np.asarray(X.sum(axis=0)).ravel() > 0)]
    counts = np.asarray(X.sum(axis=1)).ravel()
    counts[counts == 0] = 1.0
    X = (sparse.diags((1e4 / counts).astype(np.float32)) @ X).astype(np.float32)
    X.data = np.log1p(X.data).astype(np.float32, copy=False)
    ts(f"Genes ready | X={X.shape} csr float32 | t={(time.perf_counter() - t0):.1f}s")

    # Extract embeddings: numeric 2D only
    embeddings = {
        k: np.asarray(adata.obsm[k], dtype=np.float32)
        for k in adata.obsm_keys()
        if (
            np.asarray(adata.obsm[k]).ndim == 2
            and np.issubdtype(np.asarray(adata.obsm[k]).dtype, np.number)
        )
    }
    ts(f"Embeddings | {len(embeddings)} keys: {sorted(embeddings)}")

    # SGD prototype
    sgd_proto = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=sgd_alpha,
        max_iter=sgd_max_iter,
        tol=sgd_tol,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
        average=True,
        early_stopping=False,
    )

    def scale_standard(X_tr, X_te):
        """StandardScaler for dense embeddings."""
        sc = StandardScaler()
        return sc.fit_transform(X_tr), sc.transform(X_te)

    def scale_maxabs_sparse(X_tr_sparse, X_te_sparse):
        """MaxAbsScaler for sparse gene matrices."""
        scaler = MaxAbsScaler()
        return scaler.fit_transform(X_tr_sparse), scaler.transform(X_te_sparse)

    # Track non-convergence by feature name (persists across both split types)
    nonconverge_counts = {}

    def compute_metrics(
        y_true, y_pred, fold, feature, model, split_type, subsample_frac
    ):
        """Compute metrics row with NaN detection."""
        # Check for NaN predictions
        n_nan = np.isnan(y_pred).sum() if y_pred.dtype.kind == "f" else 0
        n_inf = np.isinf(y_pred).sum() if y_pred.dtype.kind == "f" else 0

        if n_nan > 0 or n_inf > 0:
            ts("  ⚠️  NaN/Inf DETECTED in predictions!")
            ts(
                f"     fold={fold} | split={split_type} | feature={feature} | model={model}"
            )
            ts(
                f"     NaN count: {n_nan}/{len(y_pred)} ({100 * n_nan / len(y_pred):.1f}%)"
            )
            ts(
                f"     Inf count: {n_inf}/{len(y_pred)} ({100 * n_inf / len(y_pred):.1f}%)"
            )
            ts(f"     y_pred dtype: {y_pred.dtype}")
            ts(f"     y_pred shape: {y_pred.shape}")
            if n_nan > 0:
                nan_mask = np.isnan(y_pred)
                ts(f"     First NaN indices: {np.where(nan_mask)[0][:10].tolist()}")
            if n_inf > 0:
                inf_mask = np.isinf(y_pred)
                ts(f"     First Inf indices: {np.where(inf_mask)[0][:10].tolist()}")

        # Compute all metrics
        from sklearn.metrics import accuracy_score

        metrics_dict = {
            "fold": fold,
            "split_type": split_type,
            "feature": feature,
            "model": model,
            "data_fraction": subsample_frac,
            # Core metrics
            "accuracy": accuracy_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "bACC": balanced_accuracy_score(y_true, y_pred),
            # F1 variants
            "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "F1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "F1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            # Diagnostics
            "n_nan_preds": n_nan,
            "n_inf_preds": n_inf,
        }

        return metrics_dict

    def has_all_train_classes_in_test(y_tr, y_te) -> bool:
        """Check if test set contains all classes present in train."""
        return set(np.unique(y_tr)).issubset(set(np.unique(y_te)))

    results = []

    # ============================================================================
    # SPLIT TYPE 1: DONOR HOLDOUT (hard generalization test)
    # ============================================================================
    if run_donor_holdout:
        ts(f"\n{'=' * 80}\nSTART DONOR HOLDOUT SPLITS\n{'=' * 80}")

        k_folds = min(n_splits, n_donors)
        if k_folds < 2:
            ts(
                f"Warning: Need >=2 donors for donor-holdout CV; found {n_donors}. Skipping donor holdout."
            )
        else:
            sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=0)
            dummy_X = np.zeros((len(y), 1), dtype=np.int8)

            for fold, (tr_idx, te_idx) in enumerate(
                sgkf.split(dummy_X, y, groups=donors)
            ):
                if (fold % log_every_folds) == 0:
                    ts(
                        f"\nDonor-holdout fold {fold}/{k_folds - 1} | train={len(tr_idx)} test={len(te_idx)}"
                    )

                # Subsample training if requested
                if train_subsample_frac < 1.0:
                    try:
                        sub_local, _ = next(
                            StratifiedShuffleSplit(
                                n_splits=1,
                                train_size=train_subsample_frac,
                                random_state=fold,
                            ).split(
                                np.zeros((len(tr_idx), 1), dtype=np.int8), y[tr_idx]
                            )
                        )
                        tr_idx = tr_idx[sub_local]
                        if (fold % log_every_folds) == 0:
                            ts(f"  Subsample -> {len(tr_idx)} (stratified)")
                    except ValueError:
                        rng = np.random.default_rng(fold)
                        tr_idx = rng.choice(
                            tr_idx,
                            size=max(2, int(len(tr_idx) * train_subsample_frac)),
                            replace=False,
                        )
                        if (fold % log_every_folds) == 0:
                            ts(f"  Subsample -> {len(tr_idx)} (random fallback)")

                y_tr, y_te = y[tr_idx], y[te_idx]

                # Check class diversity
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    ts(f"  Skip fold {fold} (insufficient class diversity)")
                    continue
                if (
                    skip_incomplete_test_class_support
                    and not has_all_train_classes_in_test(y_tr, y_te)
                ):
                    ts(f"  Skip fold {fold} (test missing some train classes)")
                    continue

                # Run all features
                _run_fold_features(
                    fold=fold,
                    tr_idx=tr_idx,
                    te_idx=te_idx,
                    y_tr=y_tr,
                    y_te=y_te,
                    X=X,
                    embeddings=embeddings,
                    rp_components=rp_components,
                    pca_genes_n_components=pca_genes_n_components,
                    pca_genes_whiten=pca_genes_whiten,
                    run_pca_genes=run_pca_genes,
                    sgd_proto=sgd_proto,
                    scale_standard=scale_standard,
                    scale_maxabs_sparse=scale_maxabs_sparse,
                    compute_metrics=compute_metrics,
                    split_type="donor_holdout",
                    subsample_frac=train_subsample_frac,
                    results=results,
                    log_every_folds=log_every_folds,
                    ts=ts,
                    nonconverge_counts=nonconverge_counts,
                    increase_maxiter_on_nonconverge=increase_maxiter_on_nonconverge,
                )

    # ============================================================================
    # SPLIT TYPE 2: SIMPLE STRATIFIED (easy separability test)
    # ============================================================================
    if run_simple_stratified:
        ts(f"\n{'=' * 80}\nSTART SIMPLE STRATIFIED SPLITS\n{'=' * 80}")

        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
        dummy_X = np.zeros((len(y), 1), dtype=np.int8)

        for fold, (tr_idx, te_idx) in enumerate(sss.split(dummy_X, y)):
            if (fold % log_every_folds) == 0:
                ts(
                    f"\nSimple-stratified fold {fold}/{n_splits - 1} | train={len(tr_idx)} test={len(te_idx)}"
                )

            # Subsample training if requested
            if train_subsample_frac < 1.0:
                try:
                    sub_local, _ = next(
                        StratifiedShuffleSplit(
                            n_splits=1,
                            train_size=train_subsample_frac,
                            random_state=fold,
                        ).split(np.zeros((len(tr_idx), 1), dtype=np.int8), y[tr_idx])
                    )
                    tr_idx = tr_idx[sub_local]
                    if (fold % log_every_folds) == 0:
                        ts(f"  Subsample -> {len(tr_idx)} (stratified)")
                except ValueError:
                    rng = np.random.default_rng(fold)
                    tr_idx = rng.choice(
                        tr_idx,
                        size=max(2, int(len(tr_idx) * train_subsample_frac)),
                        replace=False,
                    )
                    if (fold % log_every_folds) == 0:
                        ts(f"  Subsample -> {len(tr_idx)} (random fallback)")

            y_tr, y_te = y[tr_idx], y[te_idx]

            # Run all features
            _run_fold_features(
                fold=fold,
                tr_idx=tr_idx,
                te_idx=te_idx,
                y_tr=y_tr,
                y_te=y_te,
                X=X,
                embeddings=embeddings,
                rp_components=rp_components,
                pca_genes_n_components=pca_genes_n_components,
                pca_genes_whiten=pca_genes_whiten,
                run_pca_genes=run_pca_genes,
                sgd_proto=sgd_proto,
                scale_standard=scale_standard,
                scale_maxabs_sparse=scale_maxabs_sparse,
                compute_metrics=compute_metrics,
                split_type="simple_stratified",
                subsample_frac=train_subsample_frac,
                results=results,
                log_every_folds=log_every_folds,
                ts=ts,
                nonconverge_counts=nonconverge_counts,
                increase_maxiter_on_nonconverge=increase_maxiter_on_nonconverge,
            )

    ts(f"\n{'=' * 80}")
    ts(f"COMPLETE | total_time={(time.perf_counter() - t0):.1f}s | rows={len(results)}")
    ts(f"{'=' * 80}\n")

    return pd.DataFrame(results)


def _run_fold_features(
    fold,
    tr_idx,
    te_idx,
    y_tr,
    y_te,
    X,
    embeddings,
    rp_components,
    pca_genes_n_components,
    pca_genes_whiten,
    run_pca_genes,
    sgd_proto,
    scale_standard,
    scale_maxabs_sparse,
    compute_metrics,
    split_type,
    subsample_frac,
    results,
    log_every_folds,
    ts,
    nonconverge_counts,
    increase_maxiter_on_nonconverge,
):
    """Run SGD on all feature types for a single fold."""

    def check_data_quality(X_data, name, is_sparse=False, verbose_stats=False):
        """Check for NaN/Inf in input data with optional detailed stats."""
        if is_sparse:
            data_arr = X_data.data
            n_nan = np.isnan(data_arr).sum()
            n_inf = np.isinf(data_arr).sum()
            cond_number = None
        else:
            n_nan = np.isnan(X_data).sum()
            n_inf = np.isinf(X_data).sum()
            cond_number = None

        if n_nan > 0 or n_inf > 0:
            ts(f"  ⚠️  INPUT DATA QUALITY ISSUE: {name}")
            ts(f"     NaN count: {n_nan} | Inf count: {n_inf}")
            ts(f"     Shape: {X_data.shape} | Sparse: {is_sparse}")
            if not is_sparse:
                ts(f"     Min: {np.nanmin(X_data):.4f} | Max: {np.nanmax(X_data):.4f}")
                ts(
                    f"     Mean: {np.nanmean(X_data):.4f} | Std: {np.nanstd(X_data):.4f}"
                )

        # Compute condition number for embeddings (to diagnose convergence issues and adapt alpha)
        if not is_sparse and X_data.shape[1] <= 3000:
            try:
                cov_sample = X_data[: min(1000, len(X_data))]
                U, s, Vt = np.linalg.svd(cov_sample, full_matrices=False)
                cond_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
                if verbose_stats:
                    ts(
                        f"  📊 {name}: shape={X_data.shape} | cond={cond_number:.2e} | range=[{X_data.min():.2f}, {X_data.max():.2f}]"
                    )
            except:
                pass

        return n_nan, n_inf, cond_number

    def get_adaptive_maxiter(feat_name, base_max_iter):
        """Get adaptive max_iter based on convergence history."""
        if not increase_maxiter_on_nonconverge:
            return base_max_iter

        # If this feature has failed to converge multiple times, increase max_iter
        fail_count = nonconverge_counts.get(feat_name, 0)
        if fail_count >= 3:
            return base_max_iter * 3  # 3x for persistent issues
        elif fail_count >= 1:
            return base_max_iter * 2  # 2x after first failure
        return base_max_iter

    def get_adaptive_alpha(feat_name, base_alpha, cond_number=None):
        """Get adaptive alpha (regularization) for poorly conditioned features."""
        # High condition number (>1000) suggests we need more regularization
        if cond_number is not None and cond_number > 2000:
            return base_alpha * 5  # Strong regularization for ill-conditioned problems
        elif cond_number is not None and cond_number > 1000:
            return base_alpha * 2  # Moderate boost
        return base_alpha

    # Feature 1: Raw genes with MaxAbsScaler
    t_start = time.perf_counter()
    _, _, _ = check_data_quality(X[tr_idx], "raw_genes_train", is_sparse=True)
    X_tr_scaled, X_te_scaled = scale_maxabs_sparse(X[tr_idx], X[te_idx])
    _, _, _ = check_data_quality(X_tr_scaled, "raw_genes_train_scaled", is_sparse=True)

    clf = type(sgd_proto)(**sgd_proto.get_params())
    clf.fit(X_tr_scaled, y_tr)

    # Check convergence
    if not clf.n_iter_ < sgd_proto.max_iter:
        ts(
            f"  ⚠️  SGD did NOT converge for raw_genes_log1p_maxabs (reached max_iter={sgd_proto.max_iter})"
        )

    y_pred = clf.predict(X_te_scaled)
    metrics = compute_metrics(
        y_te, y_pred, fold, "raw_genes_log1p_maxabs", "SGD", split_type, subsample_frac
    )
    results.append(metrics)
    if (fold % log_every_folds) == 0:
        ts(
            f"  {'raw_genes_log1p_maxabs':>25s} | SGD | bACC={metrics['bACC']:.3f} F1w={metrics['F1_weighted']:.3f} | iter={clf.n_iter_} | {time.perf_counter() - t_start:.2f}s"
        )

    # Feature 1b: PCA on raw genes (OpenProblems baseline)
    if run_pca_genes:
        from sklearn.decomposition import PCA

        t_start = time.perf_counter()

        # Convert sparse to dense for PCA (required)
        X_tr_dense = X[tr_idx].toarray() if hasattr(X[tr_idx], "toarray") else X[tr_idx]
        X_te_dense = X[te_idx].toarray() if hasattr(X[te_idx], "toarray") else X[te_idx]

        pca = PCA(
            n_components=pca_genes_n_components,
            whiten=pca_genes_whiten,
            random_state=fold,
        )
        X_tr_pca = pca.fit_transform(X_tr_dense)
        X_te_pca = pca.transform(X_te_dense)

        verbose = fold == 0 and (fold % log_every_folds) == 0
        _, _, cond_pca = check_data_quality(
            X_tr_pca,
            f"genes_PCA{pca_genes_n_components}_train",
            is_sparse=False,
            verbose_stats=verbose,
        )

        # No additional scaling needed - PCA with whiten=True already normalizes
        feat_name = (
            f"genes_PCA{pca_genes_n_components}{'_whiten' if pca_genes_whiten else ''}"
        )

        clf = type(sgd_proto)(**sgd_proto.get_params())
        clf.fit(X_tr_pca, y_tr)

        if not clf.n_iter_ < sgd_proto.max_iter:
            ts(
                f"  ⚠️  SGD did NOT converge for {feat_name} (reached max_iter={sgd_proto.max_iter})"
            )

        y_pred = clf.predict(X_te_pca)
        metrics = compute_metrics(
            y_te, y_pred, fold, feat_name, "SGD", split_type, subsample_frac
        )
        results.append(metrics)

        var_explained = pca.explained_variance_ratio_.sum()
        if (fold % log_every_folds) == 0:
            ts(
                f"  {feat_name:>25s} | SGD | bACC={metrics['bACC']:.3f} F1w={metrics['F1_weighted']:.3f} | var={var_explained:.3f} iter={clf.n_iter_} | {time.perf_counter() - t_start:.2f}s"
            )

    # Feature 2: Random projection of genes
    t_start = time.perf_counter()
    rp = SparseRandomProjection(
        n_components=rp_components, dense_output=True, random_state=fold
    )
    X_tr_rp = rp.fit_transform(X[tr_idx])
    X_te_rp = rp.transform(X[te_idx])
    _, _, _ = check_data_quality(X_tr_rp, f"RP_{rp_components}_train", is_sparse=False)
    X_tr_rp_scaled, X_te_rp_scaled = scale_standard(X_tr_rp, X_te_rp)
    _, _, _ = check_data_quality(
        X_tr_rp_scaled, f"RP_{rp_components}_train_scaled", is_sparse=False
    )

    feat_name = f"RP_{rp_components}"
    adaptive_max_iter = get_adaptive_maxiter(feat_name, sgd_proto.max_iter)
    clf = type(sgd_proto)(**sgd_proto.get_params())
    clf.max_iter = adaptive_max_iter
    clf.fit(X_tr_rp_scaled, y_tr)

    if not clf.n_iter_ < clf.max_iter:
        ts(
            f"  ⚠️  SGD did NOT converge for {feat_name} (reached max_iter={clf.max_iter})"
        )
        nonconverge_counts[feat_name] = nonconverge_counts.get(feat_name, 0) + 1

    y_pred = clf.predict(X_te_rp_scaled)
    metrics = compute_metrics(
        y_te, y_pred, fold, feat_name, "SGD", split_type, subsample_frac
    )
    results.append(metrics)
    if (fold % log_every_folds) == 0:
        ts(
            f"  {feat_name:>25s} | SGD | bACC={metrics['bACC']:.3f} F1w={metrics['F1_weighted']:.3f} | iter={clf.n_iter_}/{adaptive_max_iter} | {time.perf_counter() - t_start:.2f}s"
        )

    # Feature 3+: All embeddings
    for emb_name, emb_data in embeddings.items():
        t_start = time.perf_counter()
        # Verbose stats on first fold only to diagnose issues
        verbose = fold == 0 and (fold % log_every_folds) == 0
        _, _, _ = check_data_quality(
            emb_data[tr_idx],
            f"{emb_name}_train",
            is_sparse=False,
            verbose_stats=verbose,
        )
        X_tr_emb, X_te_emb = scale_standard(emb_data[tr_idx], emb_data[te_idx])
        _, _, cond_scaled = check_data_quality(
            X_tr_emb, f"{emb_name}_train_scaled", is_sparse=False, verbose_stats=verbose
        )

        adaptive_max_iter = get_adaptive_maxiter(emb_name, sgd_proto.max_iter)
        adaptive_alpha = get_adaptive_alpha(emb_name, sgd_proto.alpha, cond_scaled)

        clf = type(sgd_proto)(**sgd_proto.get_params())
        clf.max_iter = adaptive_max_iter
        clf.alpha = adaptive_alpha
        clf.fit(X_tr_emb, y_tr)

        if not clf.n_iter_ < clf.max_iter:
            ts(
                f"  ⚠️  SGD did NOT converge for {emb_name} (reached max_iter={clf.max_iter}, alpha={adaptive_alpha:.2e})"
            )
            nonconverge_counts[emb_name] = nonconverge_counts.get(emb_name, 0) + 1

        y_pred = clf.predict(X_te_emb)
        metrics = compute_metrics(
            y_te, y_pred, fold, emb_name, "SGD", split_type, subsample_frac
        )
        results.append(metrics)

        alpha_suffix = (
            f" α={adaptive_alpha:.1e}" if adaptive_alpha != sgd_proto.alpha else ""
        )
        if (fold % log_every_folds) == 0:
            ts(
                f"  {emb_name:>25s} | SGD | bACC={metrics['bACC']:.3f} F1w={metrics['F1_weighted']:.3f} | iter={clf.n_iter_}/{adaptive_max_iter}{alpha_suffix} | {time.perf_counter() - t_start:.2f}s"
            )
