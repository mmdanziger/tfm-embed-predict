"""
Plotting functions for standard cross-validation results.

Functions for visualizing performance across folds without downsampling.
Includes errorbar plots, spaghetti plots, and performance comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple


def plot_standard_cv_performance(
    results_df: pd.DataFrame,
    feature_types: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 5),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot standard CV performance with mean ± SEM across folds.
    
    Similar to low-data regime plots but without N dimension.
    
    Args:
        results_df: DataFrame with columns [fold, feature_type, metrics...]
        feature_types: List of feature types to plot (None = all)
        metrics: List of metrics to plot (None = default set)
        figsize: Figure size
        
    Returns:
        fig, axes
    """
    # Default metrics
    if metrics is None:
        metrics = ["balanced_acc", "recall", "F1", "AUROC"]
    
    metric_labels = {
        "balanced_acc": "Balanced Accuracy",
        "recall": "Recall (Sensitivity)",
        "F1": "F1 Score",
        "AUROC": "AUROC",
        "MCC": "MCC",
    }
    
    # Filter feature types
    if feature_types is not None:
        df = results_df[results_df["feature_type"].isin(feature_types)].copy()
    else:
        df = results_df.copy()
        feature_types = sorted(df["feature_type"].unique())
    
    # Setup colors
    colors = sns.color_palette("husl", len(feature_types))
    color_map = dict(zip(feature_types, colors))
    
    # Create figure
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        # Calculate statistics across folds for each feature type
        stats = df.groupby("feature_type")[metric].agg([
            ("mean", "mean"),
            ("std", "std"),
            ("sem", lambda x: x.std() / np.sqrt(len(x))),
            ("count", "count"),
        ]).reindex(feature_types).reset_index()
        
        x_pos = np.arange(len(feature_types))
        
        for i, row in stats.iterrows():
            feat = row["feature_type"]
            color = color_map[feat]
            
            ax.bar(
                i, 
                row["mean"],
                yerr=row["sem"],
                color=color,
                alpha=0.7,
                capsize=5,
                error_kw={"linewidth": 2},
            )
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_types, rotation=30, ha="right")
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=11)
        ax.set_title(metric_labels.get(metric, metric), fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        
        if metric in ["balanced_acc", "recall", "F1", "AUROC"]:
            ax.set_ylim([0, 1.05])
    
    plt.suptitle("Standard CV Performance (mean ± SEM across folds)", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    return fig, axes


def plot_errorbar_comparison(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    feature_order: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (13, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create errorbar plots (mean ± std) for multiple metrics.
    
    This recreates your first plot type with error bars.
    
    Args:
        results_df: DataFrame with fold-level results
        metrics: List of (metric_name, ylabel) tuples
        feature_order: Order of features on x-axis (None = auto by MCC)
        label_map: Dict to rename feature types for display
        figsize: Figure size
        
    Returns:
        fig, axes
    """
    df = results_df.copy()
    
    # Apply label mapping
    if label_map is not None:
        df["feature_short"] = df["feature_type"].map(label_map).fillna(df["feature_type"])
    else:
        df["feature_short"] = df["feature_type"]
    
    # Default metrics
    if metrics is None:
        metrics = [
            ("F1", "F1 (mean ± std across folds)"),
            ("balanced_acc", "Balanced accuracy (mean ± std across folds)"),
            ("MCC", "MCC (mean ± std across folds)"),
        ]
    
    # Determine feature order
    if feature_order is None:
        # Auto-order by mean MCC (descending)
        feature_order = (
            df.groupby("feature_short")["MCC"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
    
    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharex=False)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        # Calculate mean and std for each feature
        summ = (
            df.groupby("feature_short")[metric]
            .agg(["mean", "std"])
            .reindex(feature_order)
            .reset_index()
        )
        
        ax.errorbar(
            summ["feature_short"],
            summ["mean"],
            yerr=summ["std"],
            fmt="o",
            capsize=4,
            markersize=8,
            linewidth=2,
        )
        
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        
        if metric in ["balanced_acc", "recall", "F1", "AUROC"]:
            ax.set_ylim([0, 1.05])
    
    plt.suptitle("Classification quality and stability (donor-pair CV)", 
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    
    return fig, axes


def plot_spaghetti_folds(
    results_df: pd.DataFrame,
    metrics: Optional[List[Tuple[str, str]]] = None,
    feature_order: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
    fold_column: str = "fold",
    sort_folds_by: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create spaghetti plots showing fold-by-fold performance.
    
    This recreates your second plot type with lines connecting folds.
    
    Args:
        results_df: DataFrame with fold-level results
        metrics: List of (metric_name, ylabel) tuples
        feature_order: Order of features in legend (None = auto by MCC)
        label_map: Dict to rename feature types for display
        fold_column: Column name for fold identifier
        sort_folds_by: Feature type to use for sorting folds (None = natural order)
        figsize: Figure size
        
    Returns:
        fig, axes
    """
    df = results_df.copy()
    
    # Apply label mapping
    if label_map is not None:
        df["feature_short"] = df["feature_type"].map(label_map).fillna(df["feature_type"])
    else:
        df["feature_short"] = df["feature_type"]
    
    # Default metrics
    if metrics is None:
        metrics = [
            ("F1", "F1 (mean ± std across folds)"),
            ("balanced_acc", "Balanced accuracy (mean ± std across folds)"),
            ("MCC", "MCC (mean ± std across folds)"),
        ]
    
    # Determine feature order for legend
    if feature_order is None:
        feature_order = (
            df.groupby("feature_short")["MCC"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
    
    # Determine fold order
    if sort_folds_by is not None:
        # Sort folds by performance of a specific feature
        ref = (
            df[df["feature_short"] == sort_folds_by]
            .sort_values(["AUROC", fold_column], ascending=[False, True])
            [[fold_column, "AUROC"]]
            .drop_duplicates(fold_column)
        )
        fold_order = ref[fold_column].tolist()
    else:
        # Natural order
        fold_order = sorted(df[fold_column].unique())
    
    # Get fold labels (use the fold column value as label)
    fold_labels = {f: str(f) for f in fold_order}
    
    # Filter and order data
    df = df[df[fold_column].isin(fold_order)].copy()
    df["fold_cat"] = pd.Categorical(df[fold_column], categories=fold_order, ordered=True)
    
    # Setup positions
    x_pos = np.arange(len(fold_order))
    x_labels = [fold_labels[f] for f in fold_order]
    
    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    
    # Setup colors
    colors = sns.color_palette("husl", len(feature_order))
    color_map = dict(zip(feature_order, colors))
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        for feat in feature_order:
            g = df[df["feature_short"] == feat].sort_values("fold_cat")
            
            if len(g) == 0:
                continue
            
            ax.plot(
                x_pos[:len(g)],
                g[metric].to_numpy(),
                marker="o",
                linewidth=2,
                alpha=0.85,
                label=feat,
                color=color_map[feat],
                markersize=6,
            )
        
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel(f"Fold ({fold_column})", fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.axhline(0, color="0.6", lw=1, ls="--")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        
        # Light vertical guides
        for x in x_pos:
            ax.axvline(x, color="0.92", lw=1, zorder=0)
        
        if metric in ["balanced_acc", "recall", "F1", "AUROC"]:
            ax.set_ylim([0, 1.05])
    
    # One legend for all
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, 
        title="Feature Type", 
        bbox_to_anchor=(1.01, 0.98), 
        loc="upper left",
        frameon=True,
    )
    
    plt.suptitle("Fold-to-fold variability (donor generalization)", 
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    
    return fig, axes


def get_default_feature_order(
    results_df: pd.DataFrame,
    custom_order: Optional[List[str]] = None,
) -> List[str]:
    """
    Get feature ordering: raw, pca, randproj, then embeddings.
    
    Args:
        results_df: Results DataFrame
        custom_order: Optional custom ordering
        
    Returns:
        Ordered list of feature types
    """
    if custom_order is not None:
        return custom_order
    
    all_features = results_df["feature_type"].unique()
    
    # Define priority groups
    priority_features = ["raw", "pca", "randproj", "pca50", "random_proj", "random_projection"]
    
    # Separate into priority and others (embeddings)
    ordered = []
    remaining = []
    
    for feat in all_features:
        feat_lower = feat.lower()
        if any(p in feat_lower for p in priority_features):
            ordered.append(feat)
        else:
            remaining.append(feat)
    
    # Sort priority features by their position in priority list
    def priority_key(feat):
        feat_lower = feat.lower()
        for i, p in enumerate(priority_features):
            if p in feat_lower:
                return i
        return len(priority_features)
    
    ordered.sort(key=priority_key)
    
    # Add remaining (embeddings) in alphabetical order
    remaining.sort()
    
    return ordered + remaining


# Example usage
if __name__ == "__main__":
    print("Standard CV plotting functions defined.")
    print("\nAvailable functions:")
    print("  plot_standard_cv_performance() - Bar plot with error bars")
    print("  plot_errorbar_comparison() - Errorbar plot (mean ± std)")
    print("  plot_spaghetti_folds() - Fold-by-fold line plot")
    print("  get_default_feature_order() - Get feature ordering")
