# Example: Visualizing Low-Data Regime Cross-Validation Results
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
from plot_low_data_cv import (
    plot_low_data_performance,
    plot_low_data_performance_detailed,
    plot_bootstrap_stability
)

# Assuming you have already run your cross-validation:
# results_df, predictions_df = run_cross_validation_from_precomputed(...)

# ---------------------------------------------------------------------------
# Option 1: Simple plot with shaded SEM
# ---------------------------------------------------------------------------
fig, axes = plot_low_data_performance(
    results_df, 
    feature_types=None,  # None = all feature types, or specify ['raw', 'scGPT']
    figsize=(20, 5)
)
plt.savefig("low_data_performance.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# Option 2: Detailed plot with both cell-level and donor-level metrics
# ---------------------------------------------------------------------------
fig, axes = plot_low_data_performance_detailed(
    results_df,
    feature_types=None,  # or specify subset
    error_type='sem',    # 'sem' or 'std'
    figsize=(20, 10)
)
plt.savefig("low_data_performance_detailed.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# Option 3: Bootstrap stability analysis (per feature type)
# ---------------------------------------------------------------------------
# This shows the distribution of metrics across all folds and bootstraps
# for each N value
for feature_type in results_df['feature_type'].unique():
    fig, axes = plot_bootstrap_stability(
        results_df,
        feature_type=feature_type,
        N_values=None,  # or specify [2, 8, 32, 128, 512]
        figsize=(15, 5)
    )
    plt.savefig(f"bootstrap_stability_{feature_type}.png", dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------------------------
# Option 4: Custom comparison plot for specific metrics
# ---------------------------------------------------------------------------
# If you want to focus on specific comparisons

import seaborn as sns
import numpy as np

def plot_metric_comparison(results_df, metric='balanced_acc', 
                          feature_types=None, figsize=(10, 6)):
    """
    Create a focused plot comparing feature types on a single metric.
    """
    if feature_types is not None:
        df = results_df[results_df["feature_type"].isin(feature_types)].copy()
    else:
        df = results_df.copy()
        feature_types = df["feature_type"].unique()
    
    colors = sns.color_palette("Set2", len(feature_types))
    color_map = dict(zip(feature_types, colors))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for feature_type in feature_types:
        feature_df = df[df["feature_type"] == feature_type]
        
        grouped = feature_df.groupby("N")[metric].agg([
            ("mean", "mean"),
            ("sem", lambda x: x.std() / np.sqrt(len(x))),
            ("count", "count")
        ]).reset_index().sort_values("N")
        
        N_values = grouped["N"].values
        means = grouped["mean"].values
        sems = grouped["sem"].values
        counts = grouped["count"].values
        
        # Calculate 95% confidence interval
        ci_95 = 1.96 * sems
        
        color = color_map[feature_type]
        ax.plot(N_values, means, 'o-', color=color, 
               label=f"{feature_type} (n={counts[0]} per N)", 
               linewidth=2.5, markersize=10, alpha=0.9)
        ax.fill_between(N_values, means - ci_95, means + ci_95, 
                       color=color, alpha=0.15)
        ax.fill_between(N_values, means - sems, means + sems, 
                       color=color, alpha=0.3)
    
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Training cells per class", fontsize=14, fontweight='bold')
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=14, fontweight='bold')
    ax.set_title(f"{metric.upper()} vs. Training Data Size", 
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, linestyle="--", alpha=0.3, which='both')
    ax.legend(loc='best', frameon=True, framealpha=0.95, fontsize=11)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    
    if metric in ["balanced_acc", "recall", "F1", "AUROC"]:
        ax.set_ylim([0, 1.05])
    
    # Add note about shading
    ax.text(0.02, 0.02, "Dark shading: ±SEM\nLight shading: 95% CI", 
           transform=ax.transAxes, fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig, ax

# Use it:
fig, ax = plot_metric_comparison(results_df, metric='balanced_acc')
plt.savefig("balanced_acc_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# Option 5: Summary statistics table
# ---------------------------------------------------------------------------
def print_summary_table(results_df, N_values=None):
    """
    Print a summary table of performance across N values.
    """
    if N_values is None:
        N_values = sorted(results_df['N'].unique())
    
    for feature_type in results_df['feature_type'].unique():
        print(f"\n{'='*80}")
        print(f"Feature Type: {feature_type}")
        print(f"{'='*80}")
        
        feature_df = results_df[results_df['feature_type'] == feature_type]
        
        summary = []
        for N in N_values:
            N_df = feature_df[feature_df['N'] == N]
            if len(N_df) == 0:
                continue
            
            row = {
                'N': N,
                'n_samples': len(N_df),
                'AUROC': f"{N_df['AUROC'].mean():.3f} ± {N_df['AUROC'].std():.3f}",
                'Balanced_Acc': f"{N_df['balanced_acc'].mean():.3f} ± {N_df['balanced_acc'].std():.3f}",
                'F1': f"{N_df['F1'].mean():.3f} ± {N_df['F1'].std():.3f}",
                'Recall': f"{N_df['recall'].mean():.3f} ± {N_df['recall'].std():.3f}",
            }
            summary.append(row)
        
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
    print("\n")

# Use it:
print_summary_table(results_df)

# ---------------------------------------------------------------------------
# Option 6: Comparative improvement analysis
# ---------------------------------------------------------------------------
def plot_relative_improvement(results_df, baseline_feature='raw', 
                              comparison_features=None, figsize=(15, 5)):
    """
    Plot relative improvement of other features compared to baseline.
    """
    if comparison_features is None:
        comparison_features = [f for f in results_df['feature_type'].unique() 
                              if f != baseline_feature]
    
    metrics = ['balanced_acc', 'AUROC', 'F1']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    baseline_df = results_df[results_df['feature_type'] == baseline_feature]
    
    for ax, metric in zip(axes, metrics):
        for comp_feature in comparison_features:
            comp_df = results_df[results_df['feature_type'] == comp_feature]
            
            # Calculate mean performance at each N
            baseline_perf = baseline_df.groupby('N')[metric].mean()
            comp_perf = comp_df.groupby('N')[metric].mean()
            
            # Calculate relative improvement
            N_values = sorted(set(baseline_perf.index) & set(comp_perf.index))
            improvements = []
            for N in N_values:
                rel_imp = (comp_perf[N] - baseline_perf[N]) / baseline_perf[N] * 100
                improvements.append(rel_imp)
            
            ax.plot(N_values, improvements, 'o-', label=comp_feature, 
                   linewidth=2, markersize=8)
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Training cells per class", fontsize=12)
        ax.set_ylabel(f"Relative improvement (%)", fontsize=12)
        ax.set_title(f"{metric.upper()} vs. {baseline_feature}", 
                    fontsize=13, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc='best')
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    
    plt.suptitle(f"Relative Performance Improvement over {baseline_feature}", 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig, axes

# Use it:
fig, axes = plot_relative_improvement(results_df, baseline_feature='raw')
plt.savefig("relative_improvement.png", dpi=300, bbox_inches='tight')
plt.show()
