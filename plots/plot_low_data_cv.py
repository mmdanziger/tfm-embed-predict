import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_low_data_performance(results_df, feature_types=None, figsize=(20, 5)):
    """
    Plot model performance vs. training data size with proper error handling.
    
    Args:
        results_df: DataFrame from run_cross_validation_from_precomputed
        feature_types: List of feature types to plot (None = all)
        figsize: Figure size tuple
    
    The function properly handles:
    - Multiple folds (L5O donors)
    - Multiple bootstrap samples per fold
    - Shaded confidence intervals
    """
    
    # Define metrics to plot
    metrics = [
        ("balanced_acc", "Balanced Accuracy"),
        ("recall", "Recall (Sensitivity)"),
        ("F1", "F1 Score"),
        ("AUROC", "AUROC"),
    ]
    
    # Filter feature types if specified
    if feature_types is not None:
        df = results_df[results_df["feature_type"].isin(feature_types)].copy()
    else:
        df = results_df.copy()
        feature_types = df["feature_type"].unique()
    
    # Setup color palette
    colors = sns.color_palette("husl", len(feature_types))
    color_map = dict(zip(feature_types, colors))
    
    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        
        for feature_type in feature_types:
            # Filter data for this feature type
            feature_df = df[df["feature_type"] == feature_type].copy()
            
            # Group by N and compute statistics across folds and bootstraps
            # Strategy: For each N, we have (n_folds × n_bootstrap) measurements
            grouped = feature_df.groupby("N")[metric].agg([
                ("mean", "mean"),
                ("std", "std"),
                ("sem", lambda x: x.std() / np.sqrt(len(x))),  # Standard error
                ("count", "count")
            ]).reset_index()
            
            # Sort by N
            grouped = grouped.sort_values("N")
            
            N_values = grouped["N"].values
            means = grouped["mean"].values
            stds = grouped["std"].values
            sems = grouped["sem"].values
            
            # Use SEM for shaded region (more appropriate for mean estimates)
            lower = means - sems
            upper = means + sems
            
            # Plot
            color = color_map[feature_type]
            ax.plot(N_values, means, 'o-', color=color, 
                   label=feature_type, linewidth=2, markersize=8, alpha=0.8)
            ax.fill_between(N_values, lower, upper, 
                           color=color, alpha=0.2)
        
        # Formatting
        ax.set_xscale("log", base=2)  # Log base 2 for cleaner ticks
        ax.set_xlabel("Training cells per class (N)", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(ylabel, fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, linestyle="--", alpha=0.3, which='both')
        ax.legend(loc='best', frameon=True, framealpha=0.9)
        
        # Set y-axis limits with some padding
        if metric in ["balanced_acc", "recall", "F1", "AUROC"]:
            ax.set_ylim([0, 1.05])
        
        # Format x-axis ticks to show actual numbers
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        
    plt.suptitle("Model Performance vs. Training Data Size (Low-Data Regime)", 
                 y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, axes


def plot_low_data_performance_detailed(results_df, feature_types=None, 
                                       error_type='sem', figsize=(20, 10)):
    """
    Extended version with additional visualizations including donor-level metrics.
    
    Args:
        results_df: DataFrame from run_cross_validation_from_precomputed
        feature_types: List of feature types to plot (None = all)
        error_type: 'sem' (standard error) or 'std' (standard deviation)
        figsize: Figure size tuple
    """
    
    # Define metrics to plot
    cell_metrics = [
        ("balanced_acc", "Balanced Accuracy"),
        ("recall", "Recall (Sensitivity)"),
        ("F1", "F1 Score"),
        ("AUROC", "AUROC"),
    ]
    
    donor_metrics = [
        ("donor_balanced_acc", "Donor-Level Balanced Acc"),
        ("donor_recall", "Donor-Level Recall"),
        ("donor_F1", "Donor-Level F1"),
        ("donor_AUROC", "Donor-Level AUROC"),
    ]
    
    # Filter feature types if specified
    if feature_types is not None:
        df = results_df[results_df["feature_type"].isin(feature_types)].copy()
    else:
        df = results_df.copy()
        feature_types = df["feature_type"].unique()
    
    # Setup color palette
    colors = sns.color_palette("husl", len(feature_types))
    color_map = dict(zip(feature_types, colors))
    
    # Create figure with two rows
    fig, axes = plt.subplots(2, len(cell_metrics), figsize=figsize)
    
    # Plot cell-level metrics (top row)
    for col_idx, (metric, ylabel) in enumerate(cell_metrics):
        ax = axes[0, col_idx]
        
        for feature_type in feature_types:
            feature_df = df[df["feature_type"] == feature_type].copy()
            
            grouped = feature_df.groupby("N")[metric].agg([
                ("mean", "mean"),
                ("std", "std"),
                ("sem", lambda x: x.std() / np.sqrt(len(x))),
            ]).reset_index()
            
            grouped = grouped.sort_values("N")
            
            N_values = grouped["N"].values
            means = grouped["mean"].values
            
            # Choose error type
            if error_type == 'sem':
                errors = grouped["sem"].values
            else:
                errors = grouped["std"].values
            
            lower = means - errors
            upper = means + errors
            
            color = color_map[feature_type]
            ax.plot(N_values, means, 'o-', color=color, 
                   label=feature_type, linewidth=2, markersize=8, alpha=0.8)
            ax.fill_between(N_values, lower, upper, 
                           color=color, alpha=0.2)
        
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Training cells per class", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Cell-Level {ylabel}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.3, which='both')
        if col_idx == len(cell_metrics) - 1:
            ax.legend(loc='best', frameon=True, framealpha=0.9)
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        if metric in ["balanced_acc", "recall", "F1", "AUROC"]:
            ax.set_ylim([0, 1.05])
    
    # Plot donor-level metrics (bottom row)
    for col_idx, (metric, ylabel) in enumerate(donor_metrics):
        ax = axes[1, col_idx]
        
        # Check if donor-level metrics exist
        if metric not in df.columns:
            ax.text(0.5, 0.5, "Donor-level metrics\nnot available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel("Training cells per class", fontsize=11)
            continue
        
        for feature_type in feature_types:
            feature_df = df[df["feature_type"] == feature_type].copy()
            
            grouped = feature_df.groupby("N")[metric].agg([
                ("mean", "mean"),
                ("std", "std"),
                ("sem", lambda x: x.std() / np.sqrt(len(x))),
            ]).reset_index()
            
            grouped = grouped.sort_values("N")
            
            N_values = grouped["N"].values
            means = grouped["mean"].values
            
            if error_type == 'sem':
                errors = grouped["sem"].values
            else:
                errors = grouped["std"].values
            
            lower = means - errors
            upper = means + errors
            
            color = color_map[feature_type]
            ax.plot(N_values, means, 'o-', color=color, 
                   label=feature_type, linewidth=2, markersize=8, alpha=0.8)
            ax.fill_between(N_values, lower, upper, 
                           color=color, alpha=0.2)
        
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Training cells per class", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.3, which='both')
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        if "acc" in metric or "recall" in metric or "F1" in metric or "AUROC" in metric:
            ax.set_ylim([0, 1.05])
    
    error_label = "SEM" if error_type == 'sem' else "Std Dev"
    plt.suptitle(f"Model Performance in Low-Data Regime (shaded: ±{error_label})", 
                 y=0.995, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, axes


def plot_bootstrap_stability(results_df, feature_type, N_values=None, figsize=(15, 5)):
    """
    Visualize bootstrap stability for a specific feature type.
    Shows distribution of metric values across bootstrap samples.
    
    Args:
        results_df: DataFrame from run_cross_validation_from_precomputed
        feature_type: Which feature type to analyze
        N_values: List of N values to plot (None = all)
        figsize: Figure size
    """
    
    df = results_df[results_df["feature_type"] == feature_type].copy()
    
    if N_values is None:
        N_values = sorted(df["N"].unique())
    else:
        df = df[df["N"].isin(N_values)]
    
    metrics = ["balanced_acc", "AUROC", "F1"]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        # Create violin plot or box plot
        plot_data = []
        positions = []
        labels = []
        
        for i, N in enumerate(N_values):
            N_data = df[df["N"] == N][metric].values
            if len(N_data) > 0:
                plot_data.append(N_data)
                positions.append(i)
                labels.append(str(N))
        
        # Violin plot
        parts = ax.violinplot(plot_data, positions=positions, 
                             showmeans=True, showmedians=True)
        
        # Color the violins
        for pc in parts['bodies']:
            pc.set_facecolor('#8dd3c7')
            pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Training cells per class (N)", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"{metric.upper()} Distribution", fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        if metric in ["balanced_acc", "AUROC", "F1"]:
            ax.set_ylim([0, 1.05])
    
    plt.suptitle(f"Bootstrap Stability: {feature_type}", 
                 y=1.02, fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    return fig, axes


# Example usage:
if __name__ == "__main__":
    # Assuming results_df is already loaded
    # fig, axes = plot_low_data_performance(results_df)
    # plt.show()
    
    # Or with specific feature types
    # fig, axes = plot_low_data_performance(results_df, feature_types=['raw', 'scGPT'])
    # plt.show()
    
    # Detailed version with donor-level metrics
    # fig, axes = plot_low_data_performance_detailed(results_df, error_type='sem')
    # plt.show()
    
    # Bootstrap stability analysis
    # fig, axes = plot_bootstrap_stability(results_df, feature_type='scGPT')
    # plt.show()
    
    print("Plotting functions defined. Use:")
    print("  plot_low_data_performance(results_df)")
    print("  plot_low_data_performance_detailed(results_df)")
    print("  plot_bootstrap_stability(results_df, feature_type='raw')")
