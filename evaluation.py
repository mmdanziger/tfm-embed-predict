import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Optional: comment out if you keep feature_colors.py in a different location
try:
    from feature_colors import get_palette, get_color, FEATURE_COLOR_MAP
    _HAS_COLOR_MAP = True
except ImportError:
    _HAS_COLOR_MAP = False


def load_and_prepare_results(parquet_path: str | list[str]) -> pd.DataFrame:
    """
    Loads benchmark results and adds derived columns for analysis.

    Args:
        parquet_path: Path to results parquet file

    Returns:
        DataFrame with cleaned feature names and task identifiers

    """
    if isinstance(parquet_path, list):
        df = pd.concat([pd.read_parquet(f) for f in parquet_path])
    elif isinstance(parquet_path, str):
        df = pd.read_parquet(parquet_path)

    if "error" in df.columns:
        df = df[df.error.isnull() | df.error.isna()]
    if "skip_reason" in df.columns:
        df = df[df.skip_reason.isnull() | df.skip_reason.isna()]
    # Create readable feature names
    feature_map = {
        "raw_lognorm": "Raw Log-norm",
        "raw_pca50": "PCA-50",
    }

    
    # Handle embeddings dynamically
    for col in df["feature"].unique():
        if col.startswith("obsm["):
            emb_name = col.split("[")[1].split("]")[0]
            feature_map[col] = emb_name.upper()
    overrides = {
        "BMFM": "BMFM_LW",
        "BMFM_MLMMUL": "BMFM_LM",
        "BMFM_WCED10": "BMFM_BW",
        "BMFM_VAR1": "BMFM-CONCAT",
    }
    for k, v in feature_map.items():
        if v in overrides:
            feature_map[k] = overrides[v]
    df["feature_clean"] = df["feature"].map(feature_map).fillna(df["feature"])

    # Create task identifier
    if "cell_type" in df:
        df["task_id"] = df["dataset_id"] + "::" + df["cell_type"]
    else:
        df["task_id"] = df["dataset_id"]
    # Count tasks and folds per feature for filtering
    df["n_tasks"] = df.groupby("feature_clean")["task_id"].transform("nunique")
    df["n_folds"] = df.groupby(["task_id", "feature_clean"])["fold"].transform("count")

    return df


def plot_overall_performance(
    df: pd.DataFrame,
    metrics: list[str] = None,
    feature_order: list[str] = None,
    color_map: dict[str, str] | None = None,   # ← NEW: pass FEATURE_COLOR_MAP or None
) -> plt.Figure:
    """
    Creates boxplots showing overall distribution of performance across all tasks.

    Args:
        df: Results DataFrame from load_and_prepare_results()
        metrics: List of metric columns to plot (default: MCC, balanced_acc, F1_macro)
        feature_order: Explicit ordering of features on x-axis
        color_map: Optional dict mapping feature_clean → hex color.
                   If None and feature_colors.py is importable, uses FEATURE_COLOR_MAP
                   automatically.  Pass {} to force the default seaborn palette.

    Returns:
        Matplotlib figure

    """
    if metrics is None:
        metrics = ["MCC", "balanced_acc", "F1_macro"]

    if feature_order is None:
        feature_order = (
            df.groupby("feature_clean")["MCC"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

    # ── Resolve palette ──────────────────────────────────────────────────────
    # Priority: explicit color_map arg → auto-imported FEATURE_COLOR_MAP → seaborn Set2
    _cmap = color_map
    if _cmap is None and _HAS_COLOR_MAP:
        _cmap = FEATURE_COLOR_MAP

    if _cmap:
        palette = [_cmap.get(f, "#9e9e9e") for f in feature_order]
    else:
        palette = "Set2"
    # ─────────────────────────────────────────────────────────────────────────

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.boxplot(
            data=df,
            x="feature_clean",
            y=metric,
            order=feature_order,
            ax=ax,
            palette=palette,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Add mean markers
        means = df.groupby("feature_clean")[metric].mean().reindex(feature_order)
        ax.plot(
            range(len(means)),
            means,
            "D",
            color="red",
            markersize=6,
            label="Mean",
            zorder=10,
        )

    axes[0].legend()
    plt.suptitle(
        f"Overall Performance (n={df['task_id'].nunique()} tasks, "
        f"max {df['fold'].max() } folds each)",#+ 1
        y=1.02,
        fontsize=12,
    )
    plt.tight_layout()

    return fig


def plot_pairwise_comparison(
    df: pd.DataFrame, feature1: str, feature2: str, metric: str = "MCC"
) -> plt.Figure:
    """
    Creates scatter plot comparing two features on same fold/task splits.

    Shows where each approach wins and by how much. Points above diagonal
    indicate feature1 is better.

    Args:
        df: Results DataFrame
        feature1: First feature name (e.g., "Raw Log-norm")
        feature2: Second feature name (e.g., "SCVI")
        metric: Metric to compare (default: MCC)

    Returns:
        Matplotlib figure

    """
    # Pivot to get feature1 vs feature2 on same rows
    pivot = df.pivot_table(
        index=["task_id", "fold"],
        columns="feature_clean",
        values=metric,
        aggfunc="first",
    ).reset_index()

    if feature1 not in pivot.columns or feature2 not in pivot.columns:
        raise ValueError(f"Features {feature1} or {feature2} not found in data")

    # Remove NaN rows
    comparison = pivot[[feature1, feature2]].dropna()

    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter with color by delta
    delta = comparison[feature1] - comparison[feature2]
    scatter = ax.scatter(
        comparison[feature2],
        comparison[feature1],
        c=delta,
        cmap="RdBu_r",
        s=30,
        alpha=0.6,
        edgecolors="k",
        linewidth=0.3,
    )

    # Diagonal line (equality)
    lims = [
        min(comparison[feature1].min(), comparison[feature2].min()),
        max(comparison[feature1].max(), comparison[feature2].max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1.5, label="Equal performance")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"Δ{metric} ({feature1} - {feature2})", rotation=270, labelpad=20)

    # Statistics
    win_f1 = (comparison[feature1] > comparison[feature2]).sum()
    win_f2 = (comparison[feature2] > comparison[feature1]).sum()
    tie = (comparison[feature1] == comparison[feature2]).sum()
    mean_delta = delta.mean()

    # Wilcoxon signed-rank test (paired, non-parametric)
    stat, pval = stats.wilcoxon(comparison[feature1], comparison[feature2])

    ax.set_xlabel(f"{feature2} {metric}", fontsize=11)
    ax.set_ylabel(f"{feature1} {metric}", fontsize=11)
    ax.set_title(
        f"Head-to-Head: {feature1} vs {feature2}\n"
        f"Wins: {win_f1} vs {win_f2} (ties: {tie}) | "
        f"Mean Δ: {mean_delta:.3f} | p={pval:.2e}",
        fontsize=10,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_comparison_matrix(
    df: pd.DataFrame, metric: str = "MCC", features: list[str] = None
) -> plt.Figure:
    """
    Creates a matrix of pairwise scatter plots for all features.

    Lower triangle: scatter plots
    Diagonal: histograms
    Upper triangle: win rates

    Args:
        df: Results DataFrame
        metric: Metric to compare
        features: List of features to include (default: all)

    Returns:
        Matplotlib figure

    """
    if features is None:
        features = df["feature_clean"].unique().tolist()

    n_features = len(features)

    # Pivot data
    pivot = df.pivot_table(
        index=["task_id", "fold"],
        columns="feature_clean",
        values=metric,
        aggfunc="first",
    ).reset_index()

    fig, axes = plt.subplots(
        n_features, n_features, figsize=(3 * n_features, 3 * n_features)
    )

    for i, feat_i in enumerate(features):
        for j, feat_j in enumerate(features):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                data = pivot[feat_i].dropna()
                ax.hist(data, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
                ax.set_ylabel("")
                ax.set_xlabel("")
                if i == 0:
                    ax.set_title(feat_i, fontsize=9, fontweight="bold")

            elif i > j:
                # Lower triangle: scatter
                comparison = pivot[[feat_j, feat_i]].dropna()
                if len(comparison) > 0:
                    ax.scatter(
                        comparison[feat_j],
                        comparison[feat_i],
                        s=10,
                        alpha=0.4,
                        color="gray",
                    )
                    lims = [
                        min(comparison[feat_j].min(), comparison[feat_i].min()),
                        max(comparison[feat_j].max(), comparison[feat_i].max()),
                    ]
                    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
                    ax.set_aspect("equal")

                if j == 0:
                    ax.set_ylabel(feat_i, fontsize=9)
                if i == n_features - 1:
                    ax.set_xlabel(feat_j, fontsize=9)

            else:
                # Upper triangle: win rate text
                comparison = pivot[[feat_j, feat_i]].dropna()
                if len(comparison) > 0:
                    win_i = (comparison[feat_i] > comparison[feat_j]).sum()
                    total = len(comparison)
                    win_rate = win_i / total * 100

                    # Color by win rate
                    color = (
                        "green" if win_rate > 55 else "red" if win_rate < 45 else "gray"
                    )

                    ax.text(
                        0.5,
                        0.5,
                        f"{win_rate:.1f}%\n({win_i}/{total})",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color=color,
                        fontweight="bold",
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

    plt.suptitle(
        f"Pairwise Comparison Matrix ({metric})\n"
        f"Lower: scatter | Diagonal: distribution | Upper: row feature win rate vs column",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout()

    return fig


def plot_task_variability(
    df: pd.DataFrame, metric: str = "MCC", n_tasks: int = 20
) -> plt.Figure:
    """
    Shows fold-by-fold performance for top N most variable tasks.

    Useful for identifying which tasks are hard/easy and whether
    features differ in their stability.

    Args:
        df: Results DataFrame
        metric: Metric to plot
        n_tasks: Number of tasks to show (most variable)

    Returns:
        Matplotlib figure

    """
    # Calculate task-level variance across folds
    task_var = (
        df.groupby("task_id")[metric].var().sort_values(ascending=False).head(n_tasks)
    )

    selected_tasks = task_var.index.tolist()
    subset = df[df["task_id"].isin(selected_tasks)].copy()

    # Order features by mean performance
    feature_order = (
        subset.groupby("feature_clean")[metric]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    n_cols = 4
    n_rows = int(np.ceil(n_tasks / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    for idx, task in enumerate(selected_tasks):
        ax = axes[idx]
        task_data = subset[subset["task_id"] == task]

        for feat in feature_order:
            feat_data = task_data[task_data["feature_clean"] == feat].sort_values(
                "fold"
            )
            if len(feat_data) > 0:
                ax.plot(
                    feat_data["fold"],
                    feat_data[metric],
                    marker="o",
                    label=feat,
                    linewidth=2,
                    alpha=0.8,
                )

        ax.set_title(f"{task}\n(var={task_var[task]:.3f})", fontsize=8)
        ax.set_xlabel("Fold", fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(alpha=0.2)

        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    # Hide extra subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Fold-by-Fold Variability: Top {n_tasks} Most Variable Tasks",
        fontsize=13,
        y=0.995,
    )
    plt.tight_layout()

    return fig


import pandas as pd


def generate_llm_summary(results_path: str, output_path: str = None) -> str:
    """
    Generates a structured text summary optimized for LLM comprehension.

    Args:
        results_path: Path to results parquet file
        output_path: Optional path to save summary text file

    Returns:
        Formatted summary string

    """
    df = pd.read_parquet(results_path)

    # Filter to successful tasks only
    if "error" in df.columns:
        df = df[df.error.isnull() | df.error.isna()]
    if "skip_reason" in df.columns:
        df = df[df.skip_reason.isnull() | df.skip_reason.isna()]

    # Feature mapping for readability
    feature_map = {
        "raw_lognorm": "Raw_LogNorm",
        "raw_pca50": "Raw_PCA50",
    }
    for col in df["feature"].unique():
        if col.startswith("obsm["):
            emb_name = col.split("[")[1].split("]")[0].upper()
            feature_map[col] = f"FM_{emb_name}"

    df["feature_clean"] = df["feature"].map(feature_map).fillna(df["feature"])

    # Create task identifier
    if "cell_type" in df.columns:
        df["task_id"] = df["dataset_id"] + "::" + df["cell_type"]
    else:
        df["task_id"] = df["dataset_id"]

    # Separate foundation models from baselines
    df["is_foundation_model"] = df["feature_clean"].str.startswith("FM_")

    lines = []
    lines.append("=" * 80)
    lines.append(
        "BENCHMARK SUMMARY: DISEASE PREDICTION FROM SINGLE-CELL TRANSCRIPTOMICS"
    )
    lines.append("=" * 80)
    lines.append("")

    # === STUDY DESIGN ===
    lines.append("STUDY DESIGN")
    lines.append("-" * 80)
    n_tasks = df["task_id"].nunique()
    n_datasets = df["dataset_id"].nunique()
    n_folds = df.groupby("task_id")["fold"].nunique().mean()
    n_total_comparisons = len(df)

    lines.append(
        "Task Definition: Predict disease label from single-cell gene expression"
    )
    lines.append("Evaluation Strategy: Donor-stratified K-fold cross-validation")
    lines.append(
        "  - Unit of splitting: Donor (all cells from one donor stay together)"
    )
    lines.append("  - Prevents data leakage from biological replicates")
    lines.append("  - Tests generalization to new patients")
    lines.append("")
    lines.append("Scale:")
    lines.append(f"  - Tasks: {n_tasks} (dataset × cell_type combinations)")
    lines.append(f"  - Datasets: {n_datasets}")
    lines.append(f"  - Average folds per task: {n_folds:.1f}")
    lines.append(f"  - Total fold-feature comparisons: {n_total_comparisons}")
    lines.append("")

    # === METHODS COMPARED ===
    lines.append("METHODS COMPARED")
    lines.append("-" * 80)

    baselines = df[~df["is_foundation_model"]]["feature_clean"].unique()
    fms = df[df["is_foundation_model"]]["feature_clean"].unique()

    lines.append("Baseline Methods (Linear Transformations):")
    for b in sorted(baselines):
        n_folds_b = len(df[df["feature_clean"] == b])
        lines.append(f"  - {b}: {n_folds_b} evaluations")

    lines.append("")
    lines.append("Foundation Models (Pretrained Embeddings):")
    for f in sorted(fms):
        n_folds_f = len(df[df["feature_clean"] == f])
        lines.append(f"  - {f}: {n_folds_f} evaluations")

    lines.append("")
    lines.append(
        "Universal Probe: SGD Logistic Regression (L2 penalty, balanced class weights)"
    )
    lines.append("")

    # === PRIMARY RESULTS ===
    lines.append("PRIMARY RESULTS: OVERALL PERFORMANCE")
    lines.append("-" * 80)

    primary_metrics = ["MCC", "balanced_acc", "F1_macro"]

    for metric in primary_metrics:
        lines.append(f"\n{metric.replace('_', ' ').upper()}:")

        # Compute statistics per feature
        stats_df = (
            df.groupby("feature_clean")[metric]
            .agg(
                [
                    ("mean", "mean"),
                    ("std", "std"),
                    ("median", "median"),
                    ("q25", lambda x: x.quantile(0.25)),
                    ("q75", lambda x: x.quantile(0.75)),
                    ("count", "count"),
                ]
            )
            .round(4)
        )

        # Sort by mean descending
        stats_df = stats_df.sort_values("mean", ascending=False)

        for feat, row in stats_df.iterrows():
            is_fm = feat.startswith("FM_")
            marker = "[FM]" if is_fm else "[BL]"
            lines.append(
                f"  {marker} {feat:20s}: "
                f"mean={row['mean']:6.3f} ± {row['std']:5.3f} | "
                f"median={row['median']:6.3f} | "
                f"IQR=[{row['q25']:5.3f}, {row['q75']:5.3f}] | "
                f"n={int(row['count'])}"
            )

    lines.append("")

    # === PAIRWISE COMPARISONS ===
    lines.append("PAIRWISE COMPARISONS: HEAD-TO-HEAD ON SAME FOLDS")
    lines.append("-" * 80)

    # For each FM vs each baseline, compute win rate on paired folds
    pivot = df.pivot_table(
        index=["task_id", "fold"],
        columns="feature_clean",
        values="MCC",
        aggfunc="first",
    ).reset_index()

    lines.append("\nMatthews Correlation Coefficient (MCC) - Paired Comparisons:\n")

    for fm in sorted(fms):
        if fm not in pivot.columns:
            continue

        lines.append(f"{fm} vs Baselines:")

        for bl in sorted(baselines):
            if bl not in pivot.columns:
                continue

            comparison = pivot[[fm, bl]].dropna()
            if len(comparison) == 0:
                continue

            wins_fm = (comparison[fm] > comparison[bl]).sum()
            wins_bl = (comparison[bl] > comparison[fm]).sum()
            ties = (comparison[fm] == comparison[bl]).sum()
            total = len(comparison)

            mean_delta = (comparison[fm] - comparison[bl]).mean()

            # Wilcoxon signed-rank test
            stat, pval = stats.wilcoxon(comparison[fm], comparison[bl])

            sig = (
                "***"
                if pval < 0.001
                else "**"
                if pval < 0.01
                else "*"
                if pval < 0.05
                else "ns"
            )

            lines.append(
                f"  vs {bl:20s}: "
                f"Wins {wins_fm:4d} / {wins_bl:4d} (ties={ties:3d}) | "
                f"ΔμMCC={mean_delta:+7.4f} | "
                f"p={pval:.2e} {sig}"
            )

        lines.append("")

    # === DONOR-LEVEL RESULTS ===
    lines.append("DONOR-LEVEL AGGREGATION (Majority Vote)")
    lines.append("-" * 80)

    donor_metrics = ["MCC_donor", "balanced_acc_donor", "F1_macro_donor"]
    available_donor = [
        m for m in donor_metrics if m in df.columns and df[m].notna().any()
    ]

    if available_donor:
        for metric in available_donor:
            lines.append(f"\n{metric.replace('_', ' ').upper()}:")

            donor_stats = (
                df.groupby("feature_clean")[metric]
                .agg(
                    [
                        ("mean", "mean"),
                        ("std", "std"),
                        ("median", "median"),
                    ]
                )
                .round(4)
                .sort_values("mean", ascending=False)
            )

            for feat, row in donor_stats.iterrows():
                is_fm = feat.startswith("FM_")
                marker = "[FM]" if is_fm else "[BL]"
                lines.append(
                    f"  {marker} {feat:20s}: "
                    f"mean={row['mean']:6.3f} ± {row['std']:5.3f} | "
                    f"median={row['median']:6.3f}"
                )
    else:
        lines.append("  Donor-level metrics not available for this dataset.")

    lines.append("")

    # === KEY FINDINGS ===
    lines.append("KEY FINDINGS")
    lines.append("-" * 80)

    # Rank features by mean MCC
    mcc_ranking = df.groupby("feature_clean")["MCC"].mean().sort_values(ascending=False)

    best_overall = mcc_ranking.index[0]
    best_is_fm = best_overall.startswith("FM_")

    lines.append(
        f"1. Best Overall Method: {best_overall} (mean MCC = {mcc_ranking.iloc[0]:.4f})"
    )

    # Compare best FM vs best baseline
    best_fm = (
        mcc_ranking[mcc_ranking.index.str.startswith("FM_")].index[0]
        if any(mcc_ranking.index.str.startswith("FM_"))
        else None
    )
    best_bl = (
        mcc_ranking[~mcc_ranking.index.str.startswith("FM_")].index[0]
        if any(~mcc_ranking.index.str.startswith("FM_"))
        else None
    )

    if best_fm and best_bl:
        delta = mcc_ranking[best_fm] - mcc_ranking[best_bl]
        winner = best_fm if delta > 0 else best_bl
        lines.append(
            f"2. Best Foundation Model: {best_fm} (mean MCC = {mcc_ranking[best_fm]:.4f})"
        )
        lines.append(
            f"3. Best Baseline: {best_bl} (mean MCC = {mcc_ranking[best_bl]:.4f})"
        )
        lines.append(
            f"4. Performance Gap: {winner} outperforms by ΔMCC = {abs(delta):.4f}"
        )

        # Statistical significance
        comp = pivot[[best_fm, best_bl]].dropna()
        if len(comp) > 0:
            _, pval = stats.wilcoxon(comp[best_fm], comp[best_bl])
            lines.append(f"   Wilcoxon signed-rank test: p = {pval:.2e}")

    lines.append("")

    # === INTERPRETATION ===
    lines.append("INTERPRETATION")
    lines.append("-" * 80)

    if not best_is_fm:
        lines.append(
            "Foundation models do NOT outperform simple linear baselines for disease"
        )
        lines.append("prediction on held-out donors. This suggests:")
        lines.append(
            "  - Disease signatures are accessible via linear transformations of raw data"
        )
        lines.append(
            "  - Pretraining on cell type objectives may not transfer to disease tasks"
        )
        lines.append(
            "  - The latent geometry learned by foundation models does not provide"
        )
        lines.append(
            "    systematic advantage for decoding disease state across new patients"
        )
    else:
        lines.append(
            "Foundation models outperform linear baselines for disease prediction."
        )
        lines.append(
            "This suggests the pretrained embeddings capture disease-relevant structure"
        )
        lines.append("that is not accessible via simple linear transformations.")

    lines.append("")
    lines.append("=" * 80)

    summary = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(summary)
        print(f"Summary written to: {output_path}")

    return summary


def generate_celltype_summary(results_path: str, output_path: str = None) -> str:
    """
    Generates a structured text summary for cell type prediction benchmark.
    Optimized for LLM comprehension and direct comparison with disease benchmark.

    Args:
        results_path: Path to cell type results parquet file
        output_path: Optional path to save summary text file

    Returns:
        Formatted summary string

    """
    df = pd.read_parquet(results_path)

    # Filter to successful tasks only
    if "error" in df.columns:
        df = df[df.error.isnull() | df.error.isna()]
    if "skip_reason" in df.columns:
        df = df[df.skip_reason.isnull() | df.skip_reason.isna()]

    # Feature mapping for readability
    feature_map = {
        "raw_lognorm": "Raw_LogNorm",
        "raw_pca50": "Raw_PCA50",
        "random_proj50": "random_proj50",
    }
    for col in df["feature"].unique():
        if col.startswith("obsm["):
            emb_name = col.split("[")[1].split("]")[0].upper()
            feature_map[col] = f"FM_{emb_name}"

    df["feature_clean"] = df["feature"].map(feature_map).fillna(df["feature"])

    # Task identifier is just dataset_id for cell type prediction
    df["task_id"] = df["dataset_id"]

    # Separate foundation models from baselines
    df["is_foundation_model"] = df["feature_clean"].str.startswith("FM_")

    lines = []
    lines.append("=" * 80)
    lines.append(
        "BENCHMARK SUMMARY: CELL TYPE PREDICTION FROM SINGLE-CELL TRANSCRIPTOMICS"
    )
    lines.append("=" * 80)
    lines.append("")

    # === STUDY DESIGN ===
    lines.append("STUDY DESIGN")
    lines.append("-" * 80)
    n_tasks = df["task_id"].nunique()
    n_folds = df.groupby("task_id")["fold"].nunique().mean()
    n_total_comparisons = len(df)

    # Get cell type stats
    n_cell_types_mean = df.groupby("task_id")["n_cell_types"].first().mean()
    n_cell_types_median = df.groupby("task_id")["n_cell_types"].first().median()
    n_cell_types_min = df.groupby("task_id")["n_cell_types"].first().min()
    n_cell_types_max = df.groupby("task_id")["n_cell_types"].first().max()

    lines.append(
        "Task Definition: Predict cell type label from single-cell gene expression"
    )
    lines.append("Evaluation Strategy: Donor-stratified K-fold cross-validation")
    lines.append(
        "  - Unit of splitting: Donor (all cells from one donor stay together)"
    )
    lines.append("  - Prevents data leakage from biological replicates")
    lines.append("  - Tests generalization to new patients")
    lines.append("")
    lines.append(
        "CRITICAL: Cell types are taken EXACTLY from disease prediction benchmark."
    )
    lines.append(
        "This enables direct comparison: on the same cells, how do methods perform"
    )
    lines.append("for disease prediction vs cell type prediction?")
    lines.append("")
    lines.append("Scale:")
    lines.append(f"  - Tasks: {n_tasks} datasets")
    lines.append(
        f"  - Cell types per task: min={n_cell_types_min:.0f}, "
        f"median={n_cell_types_median:.0f}, max={n_cell_types_max:.0f}"
    )
    lines.append(f"  - Average folds per task: {n_folds:.1f}")
    lines.append(f"  - Total fold-feature comparisons: {n_total_comparisons}")
    lines.append("")

    # === METHODS COMPARED ===
    lines.append("METHODS COMPARED")
    lines.append("-" * 80)

    baselines = df[~df["is_foundation_model"]]["feature_clean"].unique()
    fms = df[df["is_foundation_model"]]["feature_clean"].unique()

    lines.append("Baseline Methods (Linear Transformations):")
    for b in sorted(baselines):
        n_folds_b = len(df[df["feature_clean"] == b])
        lines.append(f"  - {b}: {n_folds_b} evaluations")

    lines.append("")
    lines.append("Foundation Models (Pretrained Embeddings):")
    for f in sorted(fms):
        n_folds_f = len(df[df["feature_clean"] == f])
        lines.append(f"  - {f}: {n_folds_f} evaluations")

    lines.append("")
    lines.append(
        "Universal Probe: SGD Logistic Regression (L2 penalty, balanced class weights)"
    )
    lines.append("")

    # === PRIMARY RESULTS ===
    lines.append("PRIMARY RESULTS: OVERALL PERFORMANCE")
    lines.append("-" * 80)

    primary_metrics = ["MCC", "balanced_acc", "F1_macro"]

    for metric in primary_metrics:
        if metric not in df.columns:
            continue

        lines.append(f"\n{metric.replace('_', ' ').upper()}:")

        # Compute statistics per feature
        stats_df = (
            df.groupby("feature_clean")[metric]
            .agg(
                [
                    ("mean", "mean"),
                    ("std", "std"),
                    ("median", "median"),
                    ("q25", lambda x: x.quantile(0.25)),
                    ("q75", lambda x: x.quantile(0.75)),
                    ("count", "count"),
                ]
            )
            .round(4)
        )

        # Sort by mean descending
        stats_df = stats_df.sort_values("mean", ascending=False)

        for feat, row in stats_df.iterrows():
            is_fm = feat.startswith("FM_")
            marker = "[FM]" if is_fm else "[BL]"
            lines.append(
                f"  {marker} {feat:20s}: "
                f"mean={row['mean']:6.3f} ± {row['std']:5.3f} | "
                f"median={row['median']:6.3f} | "
                f"IQR=[{row['q25']:5.3f}, {row['q75']:5.3f}] | "
                f"n={int(row['count'])}"
            )

    lines.append("")

    # === PAIRWISE COMPARISONS ===
    lines.append("PAIRWISE COMPARISONS: HEAD-TO-HEAD ON SAME FOLDS")
    lines.append("-" * 80)

    # For each FM vs each baseline, compute win rate on paired folds
    pivot = df.pivot_table(
        index=["task_id", "fold"],
        columns="feature_clean",
        values="MCC",
        aggfunc="first",
    ).reset_index()

    lines.append("\nMatthews Correlation Coefficient (MCC) - Paired Comparisons:\n")

    for fm in sorted(fms):
        if fm not in pivot.columns:
            continue

        lines.append(f"{fm} vs Baselines:")

        for bl in sorted(baselines):
            if bl not in pivot.columns:
                continue

            comparison = pivot[[fm, bl]].dropna()
            if len(comparison) == 0:
                continue

            wins_fm = (comparison[fm] > comparison[bl]).sum()
            wins_bl = (comparison[bl] > comparison[fm]).sum()
            ties = (comparison[fm] == comparison[bl]).sum()
            total = len(comparison)

            mean_delta = (comparison[fm] - comparison[bl]).mean()

            # Wilcoxon signed-rank test (only if enough samples)
            if len(comparison) > 1:
                try:
                    stat, pval = stats.wilcoxon(comparison[fm], comparison[bl])
                    sig = (
                        "***"
                        if pval < 0.001
                        else "**"
                        if pval < 0.01
                        else "*"
                        if pval < 0.05
                        else "ns"
                    )
                    pval_str = f"p={pval:.2e} {sig}"
                except Exception as e:
                    pval_str = "p=N/A"
            else:
                pval_str = "p=N/A (n=1)"

            lines.append(
                f"  vs {bl:20s}: "
                f"Wins {wins_fm:4d} / {wins_bl:4d} (ties={ties:3d}) | "
                f"ΔμMCC={mean_delta:+7.4f} | "
                f"{pval_str}"
            )

        lines.append("")

    # === DONOR-LEVEL RESULTS ===
    lines.append("DONOR-LEVEL AGGREGATION (Majority Vote)")
    lines.append("-" * 80)

    donor_metrics = ["MCC_donor", "balanced_acc_donor", "F1_macro_donor"]
    available_donor = [
        m for m in donor_metrics if m in df.columns and df[m].notna().any()
    ]

    if available_donor:
        for metric in available_donor:
            lines.append(f"\n{metric.replace('_', ' ').upper()}:")

            donor_stats = (
                df.groupby("feature_clean")[metric]
                .agg(
                    [
                        ("mean", "mean"),
                        ("std", "std"),
                        ("median", "median"),
                    ]
                )
                .round(4)
                .sort_values("mean", ascending=False)
            )

            for feat, row in donor_stats.iterrows():
                is_fm = feat.startswith("FM_")
                marker = "[FM]" if is_fm else "[BL]"
                lines.append(
                    f"  {marker} {feat:20s}: "
                    f"mean={row['mean']:6.3f} ± {row['std']:5.3f} | "
                    f"median={row['median']:6.3f}"
                )
    else:
        lines.append("  Donor-level metrics not available for this dataset.")

    lines.append("")

    # === KEY FINDINGS ===
    lines.append("KEY FINDINGS")
    lines.append("-" * 80)

    # Rank features by mean MCC
    mcc_ranking = df.groupby("feature_clean")["MCC"].mean().sort_values(ascending=False)

    best_overall = mcc_ranking.index[0]
    best_is_fm = best_overall.startswith("FM_")

    lines.append(
        f"1. Best Overall Method: {best_overall} (mean MCC = {mcc_ranking.iloc[0]:.4f})"
    )

    # Compare best FM vs best baseline
    best_fm = (
        mcc_ranking[mcc_ranking.index.str.startswith("FM_")].index[0]
        if any(mcc_ranking.index.str.startswith("FM_"))
        else None
    )
    best_bl = (
        mcc_ranking[~mcc_ranking.index.str.startswith("FM_")].index[0]
        if any(~mcc_ranking.index.str.startswith("FM_"))
        else None
    )

    if best_fm and best_bl:
        delta = mcc_ranking[best_fm] - mcc_ranking[best_bl]
        winner = best_fm if delta > 0 else best_bl
        lines.append(
            f"2. Best Foundation Model: {best_fm} (mean MCC = {mcc_ranking[best_fm]:.4f})"
        )
        lines.append(
            f"3. Best Baseline: {best_bl} (mean MCC = {mcc_ranking[best_bl]:.4f})"
        )
        lines.append(
            f"4. Performance Gap: {winner} outperforms by ΔMCC = {abs(delta):.4f}"
        )

        # Statistical significance
        comp = pivot[[best_fm, best_bl]].dropna()
        if len(comp) > 1:
            try:
                _, pval = stats.wilcoxon(comp[best_fm], comp[best_bl])
                lines.append(f"   Wilcoxon signed-rank test: p = {pval:.2e}")
            except:
                lines.append("   Wilcoxon signed-rank test: Could not compute")

    lines.append("")

    # === INTERPRETATION ===
    lines.append("INTERPRETATION")
    lines.append("-" * 80)

    if best_fm and best_bl:
        delta = mcc_ranking[best_fm] - mcc_ranking[best_bl]

        # Define meaningful threshold (e.g., 0.05 MCC difference)
        meaningful_gap = abs(delta) > 0.05

        if not best_is_fm or not meaningful_gap:
            lines.append(
                "Foundation models do NOT meaningfully outperform simple linear baselines"
            )
            lines.append(
                "for cell type prediction on held-out donors. This is UNEXPECTED because:"
            )
            lines.append(
                "  - Foundation models are typically trained on cell type objectives"
            )
            lines.append("  - Cell type prediction is their primary use case")
            lines.append(
                "  - This suggests they may not be learning generalizable cell type structure"
            )
            lines.append("")
            lines.append(
                "Combined with disease prediction results, this indicates foundation models"
            )
            lines.append(
                "are not providing systematic advantages over simple linear transformations"
            )
            lines.append("for either task when evaluated on truly held-out donors.")
        else:
            lines.append(
                "Foundation models outperform linear baselines for cell type prediction"
            )
            lines.append(
                f"by ΔMCC = {delta:.4f}. This is EXPECTED as foundation models are trained"
            )
            lines.append("on cell type objectives.")
            lines.append("")
            lines.append(
                "However, this should be compared with disease prediction results:"
            )
            lines.append(
                "  - If FMs excel at cell types but not disease → training objective matters"
            )
            lines.append(
                "  - If FMs match baselines on both → FMs may not add value over simple methods"
            )

    lines.append("")
    lines.append("=" * 80)

    summary = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(summary)
        print(f"Summary written to: {output_path}")

    return summary


import time
from datetime import datetime

import matplotlib.colors
import scanpy as sc


def _ts(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def make_umaps_for_obsm(
    adata,
    reps=("raw", "geneformer", "scvi", "tf-exemplar-human", "tf-sapiens"),
    color_keys=("cell_type", "donor_id"),
    n_neighbors=15,
    min_dist=0.5,
    pca_n_comps=50,
    seed=0,
    save_dir=None,  # e.g. "./umaps" or None to just show
    show=True,
):
    """
    For each representation in `reps`, compute UMAP and plot twice:
      - colored by cell_type
      - colored by donor_id.

    Conventions:
      - embeddings live in adata.obsm[rep] for rep != "raw"
      - raw uses X -> normalize_total+log1p -> PCA -> neighbors(use_rep='X_pca') -> UMAP
      - each UMAP is stored as adata.obsm[f"X_umap_{rep}"] (basis="umap_{rep}")

    Notes:
      - This writes UMAP coordinates back into `adata` (no copies).
      - If you already have normalized/log1p X, this still runs fine; it just ensures raw is sane.

    """
    t0 = time.perf_counter()

    # quick sanity
    for k in color_keys:
        if k not in adata.obs:
            raise KeyError(f"adata.obs missing required key: {k}")

    # ensure output dir
    if save_dir is not None:
        import os

        os.makedirs(save_dir, exist_ok=True)

    # ---- RAW pipeline (normalize/log1p + PCA) ----
    if "raw" in reps:
        _ts("RAW: normalize_total + log1p + PCA")
        t = time.perf_counter()

        # Heuristic: if X looks like counts (ints / very large), normalize+log1p. If already log-ish, this is still ok.
        # We do it in-place to keep things simple.
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # PCA: works with sparse too; scanpy will handle it
        sc.pp.pca(adata, n_comps=pca_n_comps, svd_solver="arpack")

        _ts(f"RAW: prep done in {time.perf_counter() - t:.1f}s")

    # helper to compute/store UMAP for one rep
    def _compute_umap(rep_name: str):
        basis = f"umap_{rep_name}"
        obsm_key = f"X_{basis}"

        _ts(f"{rep_name}: neighbors+umap start")
        t = time.perf_counter()

        if rep_name == "raw":
            sc.pp.neighbors(
                adata, n_neighbors=n_neighbors, use_rep="X_pca", random_state=seed
            )
        else:
            if rep_name not in adata.obsm:
                _ts(f"  SKIP {rep_name}: not in adata.obsm")
                return False
            Xrep = adata.obsm[rep_name]
            if Xrep.ndim != 2:
                _ts(f"  SKIP {rep_name}: obsm entry is not 2D")
                return False
            sc.pp.neighbors(
                adata, n_neighbors=n_neighbors, use_rep=rep_name, random_state=seed
            )

        sc.tl.umap(adata, min_dist=min_dist, random_state=seed)

        # store under a rep-specific key (don’t overwrite global X_umap permanently)
        adata.obsm[obsm_key] = adata.obsm["X_umap"].copy()

        _ts(f"{rep_name}: umap done in {time.perf_counter() - t:.1f}s")
        return True

    # ---- compute all UMAPs ----
    for rep in reps:
        ok = _compute_umap(rep)
        if not ok:
            continue

        basis = f"umap_{rep}"

        # ---- plot (cell_type and donor_id) ----
        for ck in color_keys:
            _ts(f"{rep}: plot colored by {ck}")
            if save_dir is None:
                sc.pl.embedding(
                    adata,
                    basis=basis,
                    color=ck,
                    title=f"{rep} | {ck}",
                    show=show,
                    palette=list(matplotlib.colors.CSS4_COLORS.values()),
                )

            else:
                # scanpy save arg appends to default figdir unless you set sc.settings.figdir
                # easiest: set figdir and use save=...
                old_figdir = sc.settings.figdir
                sc.settings.figdir = save_dir
                sc.pl.embedding(
                    adata,
                    basis=basis,
                    color=ck,
                    title=f"{rep} | {ck}",
                    show=False,
                    save=f"_{rep}__{ck}.png",
                    palette=list(matplotlib.colors.CSS4_COLORS.values()),
                )

                sc.settings.figdir = old_figdir

    _ts(f"All done in {time.perf_counter() - t0:.1f}s")
    return adata


def plot_lowdata_with_standard_std(
    df_lowdata: pd.DataFrame,
    df_standard: pd.DataFrame,
    feature_order: list[str],
    palette_dict: dict[str, str],
    metrics: list[str] | str = ("MCC", "F1_macro"),
    feature_col: str = "feature_clean",
    full_data_label: int = 2048,
    figsize_per_row: tuple[float, float] = (14, 6),
    show_error: bool = True,
    title_prefix: str | None = None,
    errorbar_style: str = "band",   # "band" (fill_between) or "bar" (errorbar caps); ignored if show_error=False
    save_path: str | None = None,
    xticks: list[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
) -> plt.Figure:
    """
    Low-data performance curve + full-data reference, one row per metric --
    matched to plot_lowdata_ranks_with_standard's current styling (GridSpec
    spacing, font sizes, legend anchor, ax2 tick handling), plotting absolute
    metric values instead of Friedman ranks.

    Dataset-level means are computed first (folds/bootstraps averaged within
    each dataset), then averaged again across datasets for the plotted line.

    If you want to exclude specific n_per_class values from the plot (not
    just from the tick labels), filter df_lowdata before calling, the same
    way you'd filter ranks_df before plot_lowdata_ranks_with_standard --
    xticks only controls which tick marks are drawn, not which data plots.

    show_error: if True, shows +/-1 SD across dataset-level means, either as
    a shaded band (errorbar_style="band") or as error bars at each point
    (errorbar_style="bar"). If False, plots only the mean line/marker.
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    n_metrics = len(metrics)
    from matplotlib.gridspec import GridSpec

    low = df_lowdata.copy()
    standard = df_standard.copy()

    fig = plt.figure(figsize=(figsize_per_row[0], figsize_per_row[1] * n_metrics))

    gs = GridSpec(
        n_metrics, 2,
        figure=fig,
        width_ratios=[3, 0.4],
        wspace=0.05,
        hspace=0.3,
    )

    legend_handles = None
    legend_labels = None

    for row_idx, metric in enumerate(metrics):
        ax1 = fig.add_subplot(gs[row_idx, 0])
        ax2 = fig.add_subplot(gs[row_idx, 1], sharey=ax1)

        # --- Low-data: dataset-level mean first, then mean (+ optional SD) across datasets ---
        low_dataset_summary = (
            low.groupby(["dataset_id", feature_col, "n_per_class"])[metric]
            .mean()
            .reset_index()
        )
        low_summary = (
            low_dataset_summary
            .groupby([feature_col, "n_per_class"])[metric]
            .agg(mean="mean", std="std")
            .reset_index()
        )

        for feat in feature_order:
            d = low_summary[low_summary[feature_col] == feat].sort_values("n_per_class")
            if d.empty:
                continue
            color = palette_dict.get(feat, "#555555")

            ax1.plot(d["n_per_class"], d["mean"],
                      color=color, marker="o", markersize=4,
                      linewidth=1.5, label=feat)

            if show_error:
                if errorbar_style == "band":
                    ax1.fill_between(
                        d["n_per_class"], d["mean"] - d["std"], d["mean"] + d["std"],
                        color=color, alpha=0.15, linewidth=0,
                    )
                elif errorbar_style == "bar":
                    ax1.errorbar(
                        d["n_per_class"], d["mean"], yerr=d["std"],
                        fmt="none", ecolor=color, elinewidth=1, capsize=3, alpha=0.6,
                    )
                else:
                    raise ValueError("errorbar_style must be 'band' or 'bar'")

        ax1.set_xscale("log", base=2)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticks, fontsize=15)
        ax1.set_xlabel("Training samples per class" if row_idx == n_metrics - 1 else "", fontsize=16)
        ax1.tick_params(axis="y", labelsize=15)
        ax1.set_ylabel(metric, fontsize=16)
        ax1.grid(False)
        ax1.spines["right"].set_visible(False)
        

        if legend_handles is None:
            legend_handles, legend_labels = ax1.get_legend_handles_labels()

        # --- Standard/full-data: same dataset-level aggregation ---
        standard_dataset_summary = (
            standard.groupby(["dataset_id", feature_col])[metric]
            .mean()
            .reset_index()
        )
        standard_summary = (
            standard_dataset_summary
            .groupby(feature_col)[metric]
            .agg(mean="mean", std="std")
            .reset_index()
        )

        for feat in feature_order:
            row = standard_summary[standard_summary[feature_col] == feat]
            if row.empty:
                continue
            color = palette_dict.get(feat, "#555555")
            mean = row["mean"].iloc[0]

            if show_error:
                std = row["std"].iloc[0]
                ax2.errorbar(
                    full_data_label, mean, yerr=std,
                    fmt="_", markersize=30, color=color,
                    linewidth=1, capsize=5, capthick=1,
                )
            else:
                ax2.plot(full_data_label, mean,
                          marker="_", markersize=30, color=color, linewidth=2)

        ax2.set_xticks([full_data_label])
        ax2.set_xticklabels(["Full\nData"], fontsize=16)
        ax2.set_xlabel("")
        ax2.tick_params(
            axis="y",
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
        )
        ax2.grid(False)
        ax2.spines["left"].set_visible(False)

        # Broken-axis markers
        kwargs = dict(marker=[(-1, -1), (1, 1)], markersize=12,
                      linestyle="none", color="k", mec="k", mew=1, clip_on=False)
        ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

    fig.legend(
        legend_handles, legend_labels,
        bbox_to_anchor=(1.0, 0.5),
        bbox_transform=ax2.transAxes,
        loc="center left",
        borderaxespad=1.0,
    )

    title = f"{title_prefix} Low-data and full-data performance"
    #title += " (mean \u00b1 SD across datasets)" if show_error else " (mean across datasets)"
    fig.suptitle(title, y=1.02, fontsize=19)

    fig.subplots_adjust(right=0.82, wspace=0.05, hspace=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig

def add_task_zscore(
    df,
    metrics=("MCC", "F1_macro"),
    task_cols=("task_id",),
    feature_col="feature_clean",
):
    """
    Standard:
        task_cols=("task_id",)

    Low-data:
        task_cols=("task_id", "n_per_class")
    """

    if isinstance(metrics, str):
        metrics = [metrics]

    task_cols = list(task_cols)

    out = (
        df.groupby(task_cols + [feature_col])[list(metrics)]
        .mean()
        .reset_index()
    )

    for metric in metrics:
        stats = (
            out.groupby(task_cols)[metric]
            .agg(task_mean="mean", task_std="std")
            .reset_index()
        )

        out = out.merge(stats, on=task_cols)

        out[f"{metric}_z"] = (
            out[metric] - out["task_mean"]
        ) / out["task_std"]

        out = out.drop(columns=["task_mean", "task_std"])

    return out

def plot_overall_performance_bar(
    df,
    metrics=("MCC_z", "F1_macro_z"),
    feature_order=None,
    color_map=None,
    figsize=(9, 6),
    title="Overall Performance (Task-Normalized Z-Score)",
    xlabel="Task-Normalized Z-Score",
    bar_height=0.36,
    show_values=False,
    errorbar=None,   # None, "std", "sem", "ci95"
):
    if isinstance(metrics, str):
        metrics = [metrics]

    if len(metrics) != 2:
        raise ValueError(
            "This function expects exactly two metrics, e.g. "
            "('MCC_z', 'F1_macro_z')."
        )

    metric1, metric2 = metrics

    task_summary = (
        df.groupby(["task_id", "feature_clean"])[list(metrics)]
        .mean()
        .reset_index()
    )

    summary = (
        task_summary
        .groupby("feature_clean")[list(metrics)]
        .agg(["mean", "std", "count"])
    )

    if feature_order is not None:
        summary = summary.reindex(feature_order)

    y = np.arange(len(summary))

    colors = [
        color_map.get(f, "#999999") if color_map is not None else "#999999"
        for f in summary.index
    ]

    fig, ax = plt.subplots(figsize=figsize)

    for metric, offset, hatch, alpha in [
        (metric1, +bar_height / 2, None, 1.0),
        (metric2, -bar_height / 2, "//", 0.65),
    ]:
        mean = summary[(metric, "mean")]

        if errorbar is None:
            err = None
        elif errorbar == "std":
            err = summary[(metric, "std")]
        elif errorbar == "sem":
            err = summary[(metric, "std")] / np.sqrt(summary[(metric, "count")])
        elif errorbar == "ci95":
            err = (
                1.96
                * summary[(metric, "std")]
                / np.sqrt(summary[(metric, "count")])
            )
        else:
            raise ValueError("errorbar must be one of: None, 'std', 'sem', 'ci95'")

        ax.barh(
            y + offset,
            mean,
            xerr=err,
            height=bar_height,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
            alpha=alpha,
            capsize=3,
            label=metric,
        )

        if show_values:
            for i, value in enumerate(mean):
                if pd.isna(value):
                    continue
                # Place text inside the bar, away from the error bar at the tip
                ha = "right" if value >= 0 else "left"
                offset_text = -0.005 if value >= 0 else 0.05
                ax.text(
                    value + offset_text,
                    y[i] + offset + 0.01,  # ← add 0.15 (adjust up/down to taste)
                    f"{value:.2f}",
                    va="bottom",            # ← change to "bottom" so text sits above
                    ha=ha,
                    fontsize=8,
                )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=18)
    
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("Model / Representation", fontsize=20)
    ax.set_title(title, fontsize=21)

    ax.legend(frameon=True)

    plt.tight_layout()

    return fig, ax, summary

def plot_lowdata_with_standard_zscore(
    df_lowdata,
    df_standard,
    feature_order,
    palette_dict,
    metrics=("MCC", "F1_macro"),
    full_data_n=2048,
    figsize=(14, 10),
):
    """
    Low-data + full-data plot using article-style task z-scores.

    At each task and n_per_class:
        z = (model_metric - mean_across_models) / std_across_models

    Meaning:
        0 = average model within that task/n
        positive = above field average
        negative = below field average
    """

    if isinstance(metrics, str):
        metrics = [metrics]

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=2,
        figsize=figsize,
        sharex=False,
        sharey="row",
        width_ratios=(3, 0.5),
    )

    if len(metrics) == 1:
        axes = np.array([axes])

    legend_handles = None
    legend_labels = None

    for row_idx, metric in enumerate(metrics):
        z_metric = f"{metric}_z"

        low_z = add_task_zscore(
            df_lowdata,
            metrics=[metric],
            task_cols=("task_id", "n_per_class"),
        )
        
        standard_z = add_task_zscore(
            df_standard,
            metrics=[metric],
            task_cols=("task_id",),
        )
        standard_z["n_per_class"] = full_data_n

        ax1 = axes[row_idx, 0]
        ax2 = axes[row_idx, 1]

        sns.lineplot(
            data=low_z,
            x="n_per_class",
            y=z_metric,
            hue="feature_clean",
            estimator="mean",
            errorbar="sd",
            hue_order=feature_order,
            palette=palette_dict,
            ax=ax1,
        )

        handles, labels = ax1.get_legend_handles_labels()
        if legend_handles is None:
            legend_handles = handles
            legend_labels = labels

        if ax1.legend_ is not None:
            ax1.legend_.remove()

        ax1.axhline(0, color="black", linestyle="--", linewidth=1)
        ax1.set_xscale("log", base=2)
        ax1.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        ax1.set_xlabel("Training samples per class")
        ax1.set_ylabel(f"{metric} z-score")
        ax1.grid(False)

        standard_summary = (
            standard_z
            .groupby("feature_clean")[z_metric]
            .agg(mean="mean", std="std")
            .reset_index()
        )

        for feature in feature_order:
            row = standard_summary[
                standard_summary["feature_clean"] == feature
            ]

            if row.empty:
                continue

            ax2.errorbar(
                x=full_data_n,
                y=row["mean"].iloc[0],
                yerr=row["std"].iloc[0],
                fmt="_",
                markersize=30,
                color=palette_dict.get(feature, "#555555"),
                linewidth=1,
                capsize=5,
                capthick=1,
            )

        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_xticks([full_data_n])
        ax2.set_xticklabels(["Full\nData"])
        ax2.set_xlabel("")
        ax2.tick_params(axis="y", labelleft=False)
        ax2.grid(False)

        ax1.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.yaxis.tick_right()

        kwargs = dict(
            marker=[(-1, -1), (1, 1)],
            markersize=12,
            linestyle="none",
            color="k",
            mec="k",
            mew=1,
            clip_on=False,
        )

        ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

    fig.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
    )

    fig.suptitle(
        "Low-data and full-data task-normalized performance",
        y=1.02,
        fontsize=12,
    )

    plt.tight_layout()
    return fig

def build_standard_mode_summary_table(
    disease_full: pd.DataFrame,
    celltype_full: pd.DataFrame,
    metrics: list[str] = ("MCC", "F1_macro", "recall_macro", "balanced_acc"),
    task_col_disease: str = "task_id",
    task_col_celltype: str = "dataset_id",
    feature_col: str = "feature_clean",
    feature_order: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Builds one summary table PER BENCHMARK (Disease, Cell type), with rows =
    metric (MCC, F1_macro, recall_macro, balanced_acc) and columns = model.
    Folds are averaged to one score per task first (matching every other
    analysis in this project), then mean/SD are taken across tasks (disease)
    or datasets (cell-type). Cell values are "mean ± sd" strings (no sample
    size shown).

    Args:
        feature_order: Column order for the output table. If None, columns
                       come out in whatever order groupby returns
                       (alphabetical) -- pass e.g. the same list used to
                       order your Friedman rank figures, or
                       ["Raw Log-norm", "PCA-50", "TF-SAPIENS", "BMFM_CONCAT",
                        "GENEFORMER", "SCVI", "random_proj50"] for a
                       best-to-worst layout matching the standard-mode ranks.

    Returns a dict: {"Disease": DataFrame, "Cell type": DataFrame}, each with
    rows = metric, columns = feature_clean (in feature_order if given).
    """
    tables = {}

    for benchmark, df, task_col in [
        ("Disease", disease_full, task_col_disease),
        ("Cell type", celltype_full, task_col_celltype),
    ]:
        metric_rows = {}
        for metric in metrics:
            # Step 1: average across folds -> one score per (task, feature)
            task_feature = (
                df.groupby([task_col, feature_col])[metric]
                .mean()
                .reset_index()
            )
            # Step 2: mean/SD across tasks, per feature
            summary = (
                task_feature.groupby(feature_col)[metric]
                .agg(["mean", "std"])
            )
            metric_rows[metric] = summary.apply(
                lambda r: f"{r['mean']:.3f} \u00b1 {r['std']:.3f}",
                axis=1,
            )

        table = pd.DataFrame(metric_rows).T  # rows = metric, columns = feature_clean

        if feature_order is not None:
            # Keep only columns that exist, in the requested order; append
            # any leftover columns not in feature_order at the end so
            # nothing silently disappears.
            present = [f for f in feature_order if f in table.columns]
            leftover = [f for f in table.columns if f not in feature_order]
            table = table[present + leftover]

        tables[benchmark] = table

    return tables


def tables_to_latex(tables: dict) -> dict[str, str]:
    """
    Converts each mean±sd string table (one per benchmark) to a LaTeX
    tabular block, ready to paste into the manuscript. Uses \\pm instead of
    the unicode ± symbol, since plain ± often breaks LaTeX compilation
    depending on font/encoding.
    """
    latex_blocks = {}
    for benchmark, table in tables.items():
        table_latex_ready = table.copy()
        for col in table_latex_ready.columns:
            table_latex_ready[col] = table_latex_ready[col].str.replace(
                "\u00b1", r"$\pm$", regex=False
            )
        latex_blocks[benchmark] = table_latex_ready.to_latex(
            caption=f"Standard-mode performance, {benchmark} benchmark (mean $\\pm$ SD across "
                    f"{'tasks' if benchmark == 'Disease' else 'datasets'}).",
            label=f"tab:standard_{benchmark.lower().replace(' ', '_')}",
            escape=False,
        )
    return latex_blocks
# --- Usage ---
# adata = sc.read_h5ad("your.h5ad")
# make_umaps_for_obsm(
#     adata,
#     reps=("raw", "geneformer", "scvi", "tf-exemplar-human", "tf-sapiens"),
#     color_keys=("cell_type", "donor_id"),
#     n_neighbors=15,
#     min_dist=0.5,
#     pca_n_comps=50,
#     seed=0,
#     save_dir="./umaps",   # or None
#     show=True,
# )