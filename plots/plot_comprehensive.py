"""
Comprehensive visualization functions for classification benchmarks.

Adapts to three modes:
1. Standard CV - cell-level predictions, fold-by-fold
2. Low-data regime - cell-level predictions, N × bootstrap
3. Pseudobulk - donor-level predictions (binary: 0 or 1)

Functions:
- Confusion matrices
- Patient/donor heatmaps
- Performance tables
- Bar plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# =============================================================================
# Confusion Matrix Plots
# =============================================================================

def plot_confusion_matrix_aggregated(
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    feature_types: Optional[List[str]] = None,
    normalize: str = 'true',  # 'true', 'pred', 'all', or None
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = 'Blues',
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot confusion matrices aggregated across all folds for each feature type.
    
    Works for all three modes (standard, low-data, pseudobulk).
    
    Args:
        results_df: Results DataFrame (not actually used here, just for consistency)
        predictions_df: Predictions DataFrame with columns:
            - feature_type
            - true_label
            - predicted_label
        feature_types: List of feature types to plot (None = all)
        normalize: 'true' (recall), 'pred' (precision), 'all', or None
        figsize: Figure size
        cmap: Colormap name
        
    Returns:
        fig, axes
    """
    if feature_types is None:
        feature_types = sorted(predictions_df['feature_type'].unique())
    
    n_features = len(feature_types)
    n_cols = min(4, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten() if n_features > 1 else [axes]
    
    # Get all unique labels
    all_labels = sorted(predictions_df['true_label'].unique())
    
    for idx, feature_type in enumerate(feature_types):
        ax = axes[idx]
        
        # Filter predictions for this feature
        preds = predictions_df[predictions_df['feature_type'] == feature_type]
        
        if len(preds) == 0:
            ax.text(0.5, 0.5, f"No data for\n{feature_type}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # Compute confusion matrix
        cm = confusion_matrix(
            preds['true_label'],
            preds['predicted_label'],
            labels=all_labels,
            normalize=normalize
        )
        
        # Plot
        im = ax.imshow(cm, cmap=cmap, aspect='auto', vmin=0, vmax=1 if normalize else None)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                val = cm[i, j]
                if normalize:
                    text = f'{val:.2f}'
                else:
                    text = f'{int(val)}'
                ax.text(j, i, text,
                       ha="center", va="center",
                       color="white" if val > thresh else "black",
                       fontsize=10)
        
        # Labels
        ax.set_xticks(np.arange(len(all_labels)))
        ax.set_yticks(np.arange(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.set_yticklabels(all_labels)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(feature_type, fontsize=11, fontweight='bold', pad=8)
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    normalize_label = {
        'true': ' (Normalized by True Label)',
        'pred': ' (Normalized by Prediction)',
        'all': ' (Normalized Overall)',
        None: ''
    }
    
    plt.suptitle(f'Confusion Matrices{normalize_label.get(normalize, "")}', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig, axes


def plot_confusion_matrix_per_fold(
    predictions_df: pd.DataFrame,
    feature_type: str,
    figsize: Tuple[int, int] = (16, 4),
    normalize: str = 'true',
    cmap: str = 'Blues',
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot confusion matrix for each fold separately.
    
    Useful for standard CV to see fold-to-fold variation.
    
    Args:
        predictions_df: Predictions DataFrame
        feature_type: Which feature type to plot
        figsize: Figure size
        normalize: Normalization method
        cmap: Colormap
        
    Returns:
        fig, axes
    """
    # Filter for this feature
    preds = predictions_df[predictions_df['feature_type'] == feature_type].copy()
    
    if len(preds) == 0:
        raise ValueError(f"No predictions found for feature_type='{feature_type}'")
    
    # Get unique folds
    folds = sorted(preds['fold'].unique())
    all_labels = sorted(preds['true_label'].unique())
    
    n_folds = len(folds)
    n_cols = min(5, n_folds)
    n_rows = int(np.ceil(n_folds / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten() if n_folds > 1 else [axes]
    
    for idx, fold in enumerate(folds):
        ax = axes[idx]
        
        fold_preds = preds[preds['fold'] == fold]
        
        cm = confusion_matrix(
            fold_preds['true_label'],
            fold_preds['predicted_label'],
            labels=all_labels,
            normalize=normalize
        )
        
        im = ax.imshow(cm, cmap=cmap, aspect='auto', vmin=0, vmax=1 if normalize else None)
        
        # Annotations
        thresh = cm.max() / 2.
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                val = cm[i, j]
                text = f'{val:.2f}' if normalize else f'{int(val)}'
                ax.text(j, i, text, ha="center", va="center",
                       color="white" if val > thresh else "black", fontsize=9)
        
        ax.set_xticks(np.arange(len(all_labels)))
        ax.set_yticks(np.arange(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(all_labels, fontsize=9)
        ax.set_title(f'{fold}', fontsize=10, fontweight='bold')
        
        if idx % n_cols == 0:
            ax.set_ylabel('True', fontsize=9)
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel('Predicted', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_folds, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Confusion Matrices by Fold: {feature_type}', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig, axes


# =============================================================================
# Patient/Donor Heatmaps (Your PI's Request!)
# =============================================================================

def plot_patient_performance_heatmap(
    predictions_df: pd.DataFrame,
    results_df: pd.DataFrame,
    feature_types: Optional[List[str]] = None,
    metric_col: str = 'prob',  # 'prob' for cell-level, or 'logit'
    patient_col: str = 'sample',  # or 'donor_id' for pseudobulk
    condition_col: str = 'true_label',
    figsize: Tuple[int, int] = (14, 10),
    value_type: str = 'accuracy',  # 'accuracy', 'probability', or 'logit'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Heatmap showing model performance per patient/donor.
    
    Y-axis: Patients, grouped by condition, sorted by aggregate performance
    X-axis: Models, sorted by performance
    Values: Accuracy (0/1 for pseudobulk, % for cell-level)
    
    This is exactly what your PI requested!
    
    Args:
        predictions_df: Predictions DataFrame
        results_df: Results DataFrame (for sorting models)
        feature_types: List of feature types (None = all)
        metric_col: Column for cell-level probabilities (ignored for pseudobulk)
        patient_col: Column name for patient/donor ID
        condition_col: Column name for true condition
        figsize: Figure size
        value_type: 'accuracy' (correct/incorrect), 'probability', or 'logit'
        
    Returns:
        fig, ax
    """
    if feature_types is None:
        feature_types = sorted(predictions_df['feature_type'].unique())
    
    # Check if pseudobulk (donor-level)
    is_pseudobulk = 'id_type' in predictions_df.columns and \
                    (predictions_df['id_type'] == 'donor').any()
    
    # Aggregate predictions per patient/donor and feature
    if is_pseudobulk:
        # Donor-level: already one prediction per donor
        agg_data = predictions_df.copy()
        patient_id_col = 'donor_id' if 'donor_id' in agg_data.columns else patient_col
    else:
        # Cell-level: aggregate cells to patients
        if value_type == 'accuracy':
            # Calculate accuracy per patient
            agg_data = predictions_df.groupby([patient_col, 'feature_type', condition_col]).apply(
                lambda x: (x['true_label'] == x['predicted_label']).mean()
            ).reset_index(name='accuracy')
        elif value_type == 'probability':
            # Average probability of predicted class
            def get_avg_prob(group):
                # For each cell, get probability of its predicted class
                probs = []
                for _, row in group.iterrows():
                    pred_label = row['predicted_label']
                    prob_col = f'prob_{pred_label}'
                    if prob_col in row:
                        probs.append(row[prob_col])
                return np.mean(probs) if probs else np.nan
            
            agg_data = predictions_df.groupby([patient_col, 'feature_type', condition_col]).apply(
                get_avg_prob
            ).reset_index(name='probability')
        else:  # logit
            agg_data = predictions_df.groupby([patient_col, 'feature_type', condition_col])['logit'].mean().reset_index()
        
        patient_id_col = patient_col
    
    # For pseudobulk, calculate accuracy if not already present
    if is_pseudobulk and value_type == 'accuracy':
        agg_data['accuracy'] = (agg_data['true_label'] == agg_data['predicted_label']).astype(int)
    
    # Determine value column
    value_col = {
        'accuracy': 'accuracy',
        'probability': 'probability',
        'logit': 'logit'
    }.get(value_type, 'accuracy')
    
    # Filter to selected features
    agg_data = agg_data[agg_data['feature_type'].isin(feature_types)]
    
    # Sort models by mean performance
    model_performance = agg_data.groupby('feature_type')[value_col].mean().sort_values(ascending=False)
    sorted_models = model_performance.index.tolist()
    
    # Sort patients by condition and then by aggregate performance
    patient_performance = agg_data.groupby([patient_id_col, condition_col])[value_col].mean().reset_index()
    patient_performance = patient_performance.sort_values([condition_col, value_col], ascending=[True, False])
    sorted_patients = patient_performance[patient_id_col].tolist()
    
    # Create pivot table
    pivot = agg_data.pivot_table(
        index=patient_id_col,
        columns='feature_type',
        values=value_col,
        aggfunc='mean'
    )
    
    # Reorder
    pivot = pivot.reindex(index=sorted_patients, columns=sorted_models)
    
    # Get conditions for each patient
    patient_conditions = agg_data.groupby(patient_id_col)[condition_col].first()
    pivot_conditions = patient_conditions[pivot.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose colormap based on value type
    if value_type == 'accuracy':
        if is_pseudobulk:
            cmap = sns.color_palette("RdYlGn", as_cmap=True)
            vmin, vmax = 0, 1
        else:
            cmap = sns.color_palette("RdYlGn", as_cmap=True)
            vmin, vmax = 0, 1
    else:
        cmap = 'viridis'
        vmin, vmax = None, None
    
    # Plot heatmap
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Set ticks
    ax.set_xticks(np.arange(len(sorted_models)))
    ax.set_yticks(np.arange(len(sorted_patients)))
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(sorted_patients, fontsize=7)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if value_type == 'accuracy':
        if is_pseudobulk:
            cbar.set_label('Correct (1) / Incorrect (0)', rotation=270, labelpad=20, fontsize=10)
        else:
            cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontsize=10)
    elif value_type == 'probability':
        cbar.set_label('Average Probability', rotation=270, labelpad=20, fontsize=10)
    else:
        cbar.set_label('Logit', rotation=270, labelpad=20, fontsize=10)
    
    # Add condition separators
    conditions = pivot_conditions.values
    prev_condition = None
    for i, condition in enumerate(conditions):
        if prev_condition is not None and condition != prev_condition:
            ax.axhline(y=i - 0.5, color='black', linewidth=2)
        prev_condition = condition
    
    # Add condition labels on the right
    unique_conditions = pivot_conditions.unique()
    for condition in unique_conditions:
        condition_mask = pivot_conditions == condition
        y_positions = np.where(condition_mask)[0]
        if len(y_positions) > 0:
            y_center = (y_positions[0] + y_positions[-1]) / 2
            ax.text(len(sorted_models) + 0.5, y_center, str(condition),
                   va='center', fontsize=10, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Model / Feature Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Patient / Donor ID', fontsize=11, fontweight='bold')
    
    mode_label = "Pseudobulk (Donor-Level)" if is_pseudobulk else "Cell-Level"
    value_label = {'accuracy': 'Accuracy', 'probability': 'Probability', 'logit': 'Logit'}[value_type]
    ax.set_title(f'Per-Patient Performance Heatmap ({mode_label})\n{value_label} by Model',
                 fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    return fig, ax


# =============================================================================
# Performance Summary Tables
# =============================================================================

def create_performance_table(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    feature_types: Optional[List[str]] = None,
    group_by: str = 'feature_type',  # or 'fold' for per-fold summary
    round_digits: int = 3,
) -> pd.DataFrame:
    """
    Create formatted performance summary table.
    
    Works for all three modes.
    
    Args:
        results_df: Results DataFrame
        metrics: List of metrics to include (None = default set)
        feature_types: List of feature types to include (None = all)
        group_by: 'feature_type' or 'fold'
        round_digits: Number of decimal places
        
    Returns:
        Formatted DataFrame
    """
    if metrics is None:
        metrics = ['balanced_acc', 'recall', 'F1', 'AUROC', 'MCC']
    
    if feature_types is not None:
        df = results_df[results_df['feature_type'].isin(feature_types)].copy()
    else:
        df = results_df.copy()
    
    # Aggregate
    summary = df.groupby(group_by)[metrics].agg(['mean', 'std']).round(round_digits)
    
    # Flatten column names
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    
    # Format as "mean ± std"
    formatted = pd.DataFrame(index=summary.index)
    for metric in metrics:
        if f'{metric}_mean' in summary.columns and f'{metric}_std' in summary.columns:
            formatted[metric] = summary.apply(
                lambda row: f"{row[f'{metric}_mean']:.{round_digits}f} ± {row[f'{metric}_std']:.{round_digits}f}",
                axis=1
            )
    
    return formatted


def create_comparison_table(
    results_df: pd.DataFrame,
    baseline_feature: str,
    comparison_features: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create table comparing all features to a baseline.
    
    Shows absolute performance and relative improvement.
    
    Args:
        results_df: Results DataFrame
        baseline_feature: Feature type to use as baseline
        comparison_features: Features to compare (None = all except baseline)
        metrics: Metrics to compare (None = default)
        
    Returns:
        Comparison DataFrame
    """
    if metrics is None:
        metrics = ['balanced_acc', 'F1', 'AUROC']
    
    if comparison_features is None:
        comparison_features = [f for f in results_df['feature_type'].unique() 
                              if f != baseline_feature]
    
    # Get baseline performance
    baseline_perf = results_df[results_df['feature_type'] == baseline_feature][metrics].mean()
    
    # Compare each feature
    comparison_data = []
    
    for feature in comparison_features:
        feature_perf = results_df[results_df['feature_type'] == feature][metrics].mean()
        
        row = {'feature_type': feature}
        for metric in metrics:
            baseline_val = baseline_perf[metric]
            feature_val = feature_perf[metric]
            improvement = ((feature_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else np.nan
            
            row[f'{metric}_value'] = f'{feature_val:.3f}'
            row[f'{metric}_improve'] = f'{improvement:+.1f}%'
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add baseline row
    baseline_row = {'feature_type': f'{baseline_feature} (baseline)'}
    for metric in metrics:
        baseline_row[f'{metric}_value'] = f'{baseline_perf[metric]:.3f}'
        baseline_row[f'{metric}_improve'] = '—'
    
    comparison_df = pd.concat([
        pd.DataFrame([baseline_row]),
        comparison_df
    ], ignore_index=True)
    
    return comparison_df


def plot_performance_table(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    feature_types: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot performance table as a formatted matplotlib table.
    
    Args:
        results_df: Results DataFrame
        metrics: Metrics to display
        feature_types: Features to include
        figsize: Figure size
        
    Returns:
        fig, ax
    """
    table_df = create_performance_table(results_df, metrics, feature_types)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        cellLoc='center',
        rowLoc='left',
        loc='center',
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        elif j == -1:
            cell.set_facecolor('#E8F5E9')
            cell.set_text_props(weight='bold')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#F5F5F5')
    
    plt.title('Performance Summary Table', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig, ax


# =============================================================================
# Bar Plots
# =============================================================================

def plot_performance_barplot(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    feature_types: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
    show_error_bars: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Bar plot comparing feature types across metrics.
    
    Args:
        results_df: Results DataFrame
        metrics: Metrics to plot
        feature_types: Features to include
        figsize: Figure size
        show_error_bars: Show std as error bars
        
    Returns:
        fig, axes
    """
    if metrics is None:
        metrics = ['balanced_acc', 'recall', 'F1', 'AUROC']
    
    if feature_types is not None:
        df = results_df[results_df['feature_type'].isin(feature_types)].copy()
    else:
        df = results_df.copy()
        feature_types = sorted(df['feature_type'].unique())
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    colors = sns.color_palette("Set2", len(feature_types))
    
    for ax, metric in zip(axes, metrics):
        summary = df.groupby('feature_type')[metric].agg(['mean', 'std']).reindex(feature_types)
        
        x_pos = np.arange(len(feature_types))
        
        if show_error_bars:
            ax.bar(x_pos, summary['mean'], yerr=summary['std'], 
                  color=colors, alpha=0.8, capsize=5, error_kw={'linewidth': 2})
        else:
            ax.bar(x_pos, summary['mean'], color=colors, alpha=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_types, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        if metric in ['balanced_acc', 'recall', 'F1', 'AUROC']:
            ax.set_ylim([0, 1.05])
    
    plt.suptitle('Performance Comparison Across Feature Types', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig, axes


# =============================================================================
# Combined Summary Figure
# =============================================================================

def create_comprehensive_summary(
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    feature_types: Optional[List[str]] = None,
    save_prefix: Optional[str] = None,
) -> Dict[str, Tuple[plt.Figure, any]]:
    """
    Create all visualizations at once and return them.
    
    Args:
        results_df: Results DataFrame
        predictions_df: Predictions DataFrame
        feature_types: Features to include
        save_prefix: Prefix for saving files (None = don't save)
        
    Returns:
        Dict of {name: (fig, axes)} for all created figures
    """
    figures = {}
    
    # 1. Confusion matrices
    fig, axes = plot_confusion_matrix_aggregated(results_df, predictions_df, feature_types)
    figures['confusion_matrix'] = (fig, axes)
    if save_prefix:
        fig.savefig(f'{save_prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # 2. Patient heatmap
    fig, ax = plot_patient_performance_heatmap(predictions_df, results_df, feature_types)
    figures['patient_heatmap'] = (fig, ax)
    if save_prefix:
        fig.savefig(f'{save_prefix}_patient_heatmap.png', dpi=300, bbox_inches='tight')
    
    # 3. Performance table
    fig, ax = plot_performance_table(results_df, feature_types=feature_types)
    figures['performance_table'] = (fig, ax)
    if save_prefix:
        fig.savefig(f'{save_prefix}_performance_table.png', dpi=300, bbox_inches='tight')
    
    # 4. Bar plot
    fig, axes = plot_performance_barplot(results_df, feature_types=feature_types)
    figures['barplot'] = (fig, axes)
    if save_prefix:
        fig.savefig(f'{save_prefix}_barplot.png', dpi=300, bbox_inches='tight')
    
    if save_prefix:
        print(f"Saved all figures with prefix: {save_prefix}")
    
    return figures


if __name__ == "__main__":
    print("Comprehensive visualization module loaded.")
    print("\nAvailable functions:")
    print("  - plot_confusion_matrix_aggregated()")
    print("  - plot_confusion_matrix_per_fold()")
    print("  - plot_patient_performance_heatmap()  ← Your PI's request!")
    print("  - create_performance_table()")
    print("  - create_comparison_table()")
    print("  - plot_performance_table()")
    print("  - plot_performance_barplot()")
    print("  - create_comprehensive_summary()")
