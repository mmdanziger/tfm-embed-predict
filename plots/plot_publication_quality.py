import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Publication-quality settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2


def plot_publication_quality(results_df, feature_types=None, 
                            metrics=None, figsize=(16, 4),
                            color_palette='Set2', save_path=None):
    """
    Publication-quality plot with clean styling and proper error bars.
    
    Args:
        results_df: DataFrame from cross-validation
        feature_types: List of feature types to include
        metrics: List of (metric_name, display_name) tuples
        figsize: Figure size
        color_palette: Seaborn color palette name
        save_path: Path to save figure (None = don't save)
    """
    
    if metrics is None:
        metrics = [
            ("balanced_acc", "Balanced Accuracy"),
            ("AUROC", "AUROC"),
            ("F1", "F1 Score"),
            ("recall", "Recall"),
        ]
    
    if feature_types is not None:
        df = results_df[results_df["feature_type"].isin(feature_types)].copy()
    else:
        df = results_df.copy()
        feature_types = sorted(df["feature_type"].unique())
    
    # Color setup
    colors = sns.color_palette(color_palette, len(feature_types))
    color_map = dict(zip(feature_types, colors))
    
    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        
        for feature_type in feature_types:
            feature_df = df[df["feature_type"] == feature_type].copy()
            
            # Compute statistics
            grouped = feature_df.groupby("N")[metric].agg([
                ("mean", "mean"),
                ("sem", lambda x: x.std() / np.sqrt(len(x))),
                ("std", "std"),
                ("count", "count")
            ]).reset_index().sort_values("N")
            
            N_values = grouped["N"].values
            means = grouped["mean"].values
            sems = grouped["sem"].values
            
            # Plot line with markers
            color = color_map[feature_type]
            line = ax.plot(N_values, means, marker='o', color=color,
                          label=feature_type, linewidth=2.5, 
                          markersize=7, markeredgewidth=0.5,
                          markeredgecolor='white', alpha=0.9,
                          zorder=3)
            
            # Shaded SEM region
            ax.fill_between(N_values, means - sems, means + sems,
                           color=color, alpha=0.25, linewidth=0,
                           zorder=1)
        
        # Formatting
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Training cells per class", fontsize=12, fontweight='normal')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')
        
        # Grid
        ax.grid(True, linestyle=':', alpha=0.4, which='major', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Limits
        if metric in ["balanced_acc", "recall", "F1", "AUROC"]:
            ax.set_ylim([0, 1.02])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        
        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Legend (only on last plot)
        if ax == axes[-1]:
            ax.legend(loc='lower right', frameon=True, framealpha=0.95,
                     edgecolor='gray', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved to {save_path}")
    
    return fig, axes


def plot_faceted_comparison(results_df, feature_types=None, 
                           metric='balanced_acc', figsize=(12, 8)):
    """
    Create a faceted plot showing each feature type in its own subplot.
    Useful when you have many feature types to compare.
    """
    
    if feature_types is not None:
        df = results_df[results_df["feature_type"].isin(feature_types)].copy()
    else:
        df = results_df.copy()
        feature_types = sorted(df["feature_type"].unique())
    
    n_features = len(feature_types)
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                            sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    
    colors = sns.color_palette("husl", n_features)
    
    for idx, (feature_type, color) in enumerate(zip(feature_types, colors)):
        ax = axes[idx]
        
        feature_df = df[df["feature_type"] == feature_type].copy()
        
        # Aggregate across folds and bootstraps
        grouped = feature_df.groupby("N")[metric].agg([
            ("mean", "mean"),
            ("sem", lambda x: x.std() / np.sqrt(len(x))),
            ("q25", lambda x: x.quantile(0.25)),
            ("q75", lambda x: x.quantile(0.75)),
        ]).reset_index().sort_values("N")
        
        N_values = grouped["N"].values
        means = grouped["mean"].values
        sems = grouped["sem"].values
        q25 = grouped["q25"].values
        q75 = grouped["q75"].values
        
        # Plot
        ax.plot(N_values, means, 'o-', color=color, 
               linewidth=2.5, markersize=8, alpha=0.9)
        ax.fill_between(N_values, means - sems, means + sems,
                       color=color, alpha=0.3)
        
        # Add IQR as lighter shading
        ax.fill_between(N_values, q25, q75,
                       color=color, alpha=0.1)
        
        ax.set_xscale("log", base=2)
        ax.set_title(feature_type, fontsize=12, fontweight='bold', pad=8)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel("Training cells per class", fontsize=11)
        if idx % n_cols == 0:
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"{metric.upper()} Performance Across Feature Types", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig, axes


def plot_learning_curves_comparison(results_df, feature_types=None,
                                   metrics=['balanced_acc', 'AUROC'],
                                   show_confidence=True, figsize=(14, 6)):
    """
    Side-by-side learning curves for multiple metrics.
    Shows how performance improves with more training data.
    """
    
    if feature_types is not None:
        df = results_df[results_df["feature_type"].isin(feature_types)].copy()
    else:
        df = results_df.copy()
        feature_types = sorted(df["feature_type"].unique())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    colors = sns.color_palette("tab10", len(feature_types))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for ax, metric in zip(axes, metrics):
        
        for idx, feature_type in enumerate(feature_types):
            feature_df = df[df["feature_type"] == feature_type].copy()
            
            grouped = feature_df.groupby("N")[metric].agg([
                ("mean", "mean"),
                ("sem", lambda x: x.std() / np.sqrt(len(x))),
                ("ci_lower", lambda x: x.mean() - 1.96 * x.std() / np.sqrt(len(x))),
                ("ci_upper", lambda x: x.mean() + 1.96 * x.std() / np.sqrt(len(x))),
            ]).reset_index().sort_values("N")
            
            N_values = grouped["N"].values
            means = grouped["mean"].values
            
            color = colors[idx]
            marker = markers[idx % len(markers)]
            
            # Main line
            ax.plot(N_values, means, marker=marker, color=color,
                   label=feature_type, linewidth=2, markersize=9,
                   markeredgewidth=1.5, markeredgecolor='white',
                   alpha=0.85, zorder=3)
            
            if show_confidence:
                # 95% CI shading
                ci_lower = grouped["ci_lower"].values
                ci_upper = grouped["ci_upper"].values
                ax.fill_between(N_values, ci_lower, ci_upper,
                               color=color, alpha=0.2, linewidth=0)
        
        # Formatting
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Training cells per class", fontsize=13, fontweight='bold')
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=13, fontweight='bold')
        ax.set_title(f"{metric.upper()}", fontsize=14, fontweight='bold', pad=12)
        
        ax.grid(True, linestyle='--', alpha=0.3, which='both', linewidth=0.8)
        ax.set_axisbelow(True)
        
        if metric in ["balanced_acc", "recall", "F1", "AUROC"]:
            ax.set_ylim([0, 1.05])
        
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        
        # Cleaner spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        # Legend
        ax.legend(loc='best', frameon=True, framealpha=0.9,
                 edgecolor='darkgray', fontsize=10, ncol=1)
    
    confidence_note = " (shaded: 95% CI)" if show_confidence else ""
    plt.suptitle(f"Learning Curves: Low-Data Regime{confidence_note}", 
                fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig, axes


def create_summary_figure(results_df, feature_types=None, save_prefix=None):
    """
    Create a comprehensive multi-panel figure suitable for publication.
    Combines multiple views of the data.
    """
    
    if feature_types is not None:
        df = results_df[results_df["feature_type"].isin(feature_types)].copy()
    else:
        df = results_df.copy()
        feature_types = sorted(df["feature_type"].unique())
    
    # Create figure with GridSpec for flexible layout
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Top row: Main metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Middle row: Additional metrics
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Bottom row: Distribution analyses
    ax7 = fig.add_subplot(gs[2, :2])  # Wide plot
    ax8 = fig.add_subplot(gs[2, 2])
    
    axes_top = [ax1, ax2, ax3]
    axes_mid = [ax4, ax5, ax6]
    
    metrics_top = [
        ("balanced_acc", "Balanced Accuracy"),
        ("AUROC", "AUROC"),
        ("F1", "F1 Score"),
    ]
    
    metrics_mid = [
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("specificity", "Specificity"),
    ]
    
    colors = sns.color_palette("Set2", len(feature_types))
    color_map = dict(zip(feature_types, colors))
    
    # Plot top row
    for ax, (metric, ylabel) in zip(axes_top, metrics_top):
        _plot_metric(ax, df, metric, ylabel, feature_types, color_map)
        ax.set_title(ylabel, fontsize=12, fontweight='bold', pad=8)
    
    # Plot middle row
    for ax, (metric, ylabel) in zip(axes_mid, metrics_mid):
        if metric in df.columns:
            _plot_metric(ax, df, metric, ylabel, feature_types, color_map, show_legend=False)
            ax.set_title(ylabel, fontsize=12, fontweight='bold', pad=8)
        else:
            ax.text(0.5, 0.5, f"{ylabel}\nNot Available", 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Bottom left: Sample size vs performance
    _plot_sample_efficiency(ax7, df, feature_types, color_map)
    
    # Bottom right: Coefficient of variation
    _plot_stability_metric(ax8, df, feature_types, color_map)
    
    # Add legend to first plot
    axes_top[0].legend(loc='lower right', frameon=True, framealpha=0.9,
                      edgecolor='gray', fontsize=9)
    
    plt.suptitle("Comprehensive Low-Data Performance Analysis", 
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_prefix:
        plt.savefig(f"{save_prefix}_summary.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_prefix}_summary.pdf", bbox_inches='tight')
        print(f"Saved figures with prefix: {save_prefix}")
    
    return fig


def _plot_metric(ax, df, metric, ylabel, feature_types, color_map, show_legend=True):
    """Helper function for plotting a single metric."""
    for feature_type in feature_types:
        feature_df = df[df["feature_type"] == feature_type]
        
        grouped = feature_df.groupby("N")[metric].agg([
            ("mean", "mean"),
            ("sem", lambda x: x.std() / np.sqrt(len(x))),
        ]).reset_index().sort_values("N")
        
        N_values = grouped["N"].values
        means = grouped["mean"].values
        sems = grouped["sem"].values
        
        color = color_map[feature_type]
        ax.plot(N_values, means, 'o-', color=color, label=feature_type,
               linewidth=2, markersize=6, alpha=0.85)
        ax.fill_between(N_values, means - sems, means + sems,
                       color=color, alpha=0.25)
    
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (cells/class)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    
    if metric in ["balanced_acc", "recall", "F1", "AUROC", "precision", "specificity"]:
        ax.set_ylim([0, 1.02])


def _plot_sample_efficiency(ax, df, feature_types, color_map):
    """Plot showing how quickly each method reaches good performance."""
    target_performance = 0.80  # 80% balanced accuracy
    
    for feature_type in feature_types:
        feature_df = df[df["feature_type"] == feature_type]
        
        grouped = feature_df.groupby("N")["balanced_acc"].mean().reset_index().sort_values("N")
        
        N_values = grouped["N"].values
        means = grouped["balanced_acc"].values
        
        color = color_map[feature_type]
        ax.plot(N_values, means, 'o-', color=color, label=feature_type,
               linewidth=2.5, markersize=7, alpha=0.85)
    
    ax.axhline(y=target_performance, color='red', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'{target_performance:.0%} target')
    
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Training cells per class", fontsize=11, fontweight='bold')
    ax.set_ylabel("Balanced Accuracy", fontsize=11, fontweight='bold')
    ax.set_title("Sample Efficiency Analysis", fontsize=12, fontweight='bold', pad=8)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=9)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim([0, 1.02])


def _plot_stability_metric(ax, df, feature_types, color_map):
    """Plot coefficient of variation to show stability."""
    for feature_type in feature_types:
        feature_df = df[df["feature_type"] == feature_type]
        
        grouped = feature_df.groupby("N")["balanced_acc"].agg([
            ("mean", "mean"),
            ("std", "std"),
            ("cv", lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan),
        ]).reset_index().sort_values("N")
        
        N_values = grouped["N"].values
        cvs = grouped["cv"].values * 100  # Convert to percentage
        
        color = color_map[feature_type]
        ax.plot(N_values, cvs, 'o-', color=color, label=feature_type,
               linewidth=2, markersize=6, alpha=0.85)
    
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (cells/class)", fontsize=10)
    ax.set_ylabel("CV (%)", fontsize=10)
    ax.set_title("Stability (CV)", fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())


if __name__ == "__main__":
    print("Publication-quality plotting functions loaded.")
    print("\nMain functions:")
    print("  - plot_publication_quality(results_df)")
    print("  - plot_faceted_comparison(results_df)")
    print("  - plot_learning_curves_comparison(results_df)")
    print("  - create_summary_figure(results_df)")
