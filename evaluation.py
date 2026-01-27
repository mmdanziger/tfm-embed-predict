import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


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


def plot_overall_performance(df: pd.DataFrame, metrics: list[str] = None) -> plt.Figure:
    """
    Creates boxplots showing overall distribution of performance across all tasks.

    Args:
        df: Results DataFrame from load_and_prepare_results()
        metrics: List of metric columns to plot (default: MCC, balanced_acc, F1_macro)

    Returns:
        Matplotlib figure

    """
    if metrics is None:
        metrics = ["MCC", "balanced_acc", "F1_macro"]

    # Order features by median MCC
    feature_order = (
        df.groupby("feature_clean")["MCC"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

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
            palette="Set2",
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
        f"{df['fold'].max() + 1} folds each)",
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
