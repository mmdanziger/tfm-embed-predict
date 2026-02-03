# Quick Reference Guide for Low-Data CV Plotting
# =================================================

## Overview
This guide shows you how to visualize your low-data regime cross-validation results
with professional-looking plots featuring shaded error regions.

## Your Data Structure
After running your CV, you should have:
```python
results_df, predictions_df = run_cross_validation_from_precomputed(
    X_normalized=X_normalized,
    obs=adata.obs,
    folds=folds,  # donor-level folds (e.g., 5 donors)
    Ns=[2, 4, 8, 16, 32, 64, 128, 256, 512],  # cells per class
    n_bootstrap=10,  # bootstrap repeats per fold
    label_col="Condition",
    feature_extractors=feature_extractors,  # {'raw': ..., 'scGPT': ..., etc.}
)
```

results_df contains one row per (fold × N × bootstrap × feature_type) combination

## Quick Start: 3 Lines of Code
```python
from plot_low_data_cv import plot_low_data_performance
import matplotlib.pyplot as plt

fig, axes = plot_low_data_performance(results_df)
plt.show()
```

## Available Plotting Functions

### 1. Basic Plot (Recommended Starting Point)
```python
from plot_low_data_cv import plot_low_data_performance

fig, axes = plot_low_data_performance(
    results_df,
    feature_types=['raw', 'scGPT'],  # None = all
    figsize=(20, 5)
)
plt.savefig("performance.png", dpi=300, bbox_inches='tight')
plt.show()
```
**What it shows:** 4 metrics side-by-side with shaded SEM regions

### 2. Detailed Plot with Cell + Donor Metrics
```python
from plot_low_data_cv import plot_low_data_performance_detailed

fig, axes = plot_low_data_performance_detailed(
    results_df,
    error_type='sem',  # or 'std'
    figsize=(20, 10)
)
plt.show()
```
**What it shows:** 2 rows - cell-level metrics (top) and donor-level metrics (bottom)

### 3. Bootstrap Stability Analysis
```python
from plot_low_data_cv import plot_bootstrap_stability

# Analyze each feature type separately
for feature in ['raw', 'scGPT', 'Geneformer']:
    fig, axes = plot_bootstrap_stability(
        results_df,
        feature_type=feature,
        N_values=[2, 8, 32, 128, 512]  # None = all
    )
    plt.show()
```
**What it shows:** Violin plots showing distribution of metrics across all folds+bootstraps

### 4. Publication-Quality Figures
```python
from plot_publication_quality import plot_publication_quality

fig, axes = plot_publication_quality(
    results_df,
    feature_types=['raw', 'scGPT', 'Geneformer'],
    color_palette='Set2',  # or 'tab10', 'husl', etc.
    save_path="fig_performance.png"
)
```
**What it shows:** Clean, professional plots suitable for papers

### 5. Learning Curves Comparison
```python
from plot_publication_quality import plot_learning_curves_comparison

fig, axes = plot_learning_curves_comparison(
    results_df,
    metrics=['balanced_acc', 'AUROC'],
    show_confidence=True
)
plt.show()
```
**What it shows:** Side-by-side comparison of how performance improves with data

### 6. Comprehensive Summary Figure
```python
from plot_publication_quality import create_summary_figure

fig = create_summary_figure(
    results_df,
    save_prefix="low_data_analysis"  # saves PNG and PDF
)
plt.show()
```
**What it shows:** Multi-panel figure with 8 different views of your data

## Understanding the Error Bars

### Standard Error of Mean (SEM)
- **What it is:** `SEM = std / sqrt(n)`
- **What it shows:** Uncertainty in the estimated mean
- **Use when:** You want to show confidence in your average performance
- **In your case:** n = (n_folds × n_bootstrap) = (5 × 10) = 50 measurements per N

### Standard Deviation (STD)
- **What it is:** Spread of individual measurements
- **What it shows:** Variability across different folds/bootstraps
- **Use when:** You want to show consistency of performance

### 95% Confidence Interval
- **What it is:** `mean ± 1.96 × SEM`
- **What it shows:** Range where true mean likely falls (95% probability)

**Recommendation:** Use SEM (default) for most plots, as it shows how well you've
estimated the mean performance. Use STD if you want to emphasize variability.

## Customization Examples

### Custom Metrics Plot
```python
from example_plotting_usage import plot_metric_comparison

fig, ax = plot_metric_comparison(
    results_df,
    metric='balanced_acc',
    feature_types=['raw', 'scGPT']
)
plt.show()
```

### Relative Improvement Plot
```python
from example_plotting_usage import plot_relative_improvement

fig, axes = plot_relative_improvement(
    results_df,
    baseline_feature='raw',  # compare everything to raw
    comparison_features=['scGPT', 'Geneformer']
)
plt.show()
```

### Faceted Comparison
```python
from plot_publication_quality import plot_faceted_comparison

fig, axes = plot_faceted_comparison(
    results_df,
    metric='balanced_acc',
    figsize=(12, 8)
)
plt.show()
```

## Summary Statistics Table
```python
from example_plotting_usage import print_summary_table

print_summary_table(results_df)
```

Output:
```
================================================================================
Feature Type: raw
================================================================================
  N  n_samples        AUROC  Balanced_Acc           F1       Recall
  2         50  0.723 ± 0.045  0.689 ± 0.038  0.645 ± 0.042  0.678 ± 0.045
  4         50  0.812 ± 0.032  0.776 ± 0.034  0.751 ± 0.036  0.765 ± 0.038
...
```

## Tips for Better Visualizations

1. **Start simple:** Use `plot_low_data_performance()` first
2. **Choose colors carefully:** Use 'Set2' or 'tab10' for colorblind-friendly palettes
3. **Save high resolution:** Always use `dpi=300` for publications
4. **Label clearly:** Feature type names should be descriptive
5. **Show uncertainty:** Always include error bars/shading
6. **Log scale:** X-axis is automatically log-scaled (base 2) for N values

## Common Issues and Solutions

### Issue: Too many feature types cluttering the plot
**Solution:** Filter to 2-3 most important ones
```python
fig, axes = plot_low_data_performance(
    results_df,
    feature_types=['raw', 'scGPT']  # Only show these
)
```

### Issue: Error bars are huge
**Possible causes:**
- High variability across folds → normal, shows instability
- Small sample size → increase n_bootstrap
- Bug in CV split → check your fold generation

### Issue: Lines are jagged/non-monotonic
**Causes:**
- Random variation from bootstrap sampling → normal
- Different test sets per fold → expected with L5O
**Not a bug!** This is real variation in your data

### Issue: Can't see differences between methods
**Solutions:**
1. Zoom in on y-axis: Remove `set_ylim([0, 1.05])`
2. Plot relative improvement instead
3. Use faceted plot with one subplot per method

## File Organization
```
your_analysis/
├── plot_low_data_cv.py              # Main plotting functions
├── plot_publication_quality.py       # Publication-quality versions
├── example_plotting_usage.py         # Usage examples
└── your_analysis_notebook.ipynb      # Your analysis
```

## Complete Workflow Example

```python
# 1. Run cross-validation
results_df, predictions_df = run_cross_validation_from_precomputed(...)

# 2. Quick visualization
from plot_low_data_cv import plot_low_data_performance
fig, axes = plot_low_data_performance(results_df)
plt.show()

# 3. Create publication figure
from plot_publication_quality import plot_publication_quality
fig, axes = plot_publication_quality(
    results_df,
    save_path="figures/fig2_performance.png"
)

# 4. Generate summary statistics
from example_plotting_usage import print_summary_table
print_summary_table(results_df)

# 5. Analyze stability
from plot_low_data_cv import plot_bootstrap_stability
for ft in results_df['feature_type'].unique():
    fig, axes = plot_bootstrap_stability(results_df, feature_type=ft)
    plt.savefig(f"figures/stability_{ft}.png", dpi=300, bbox_inches='tight')
    plt.close()
```

## Key Differences from Your Original Code

**Your original approach:**
```python
summ = df.groupby("feature_short")[metric].agg(["mean", "std"])
ax.errorbar(summ["feature_short"], summ["mean"], yerr=summ["std"], ...)
```
- Used error bars with caps
- Grouped only by N (feature_short)
- No distinction between folds and bootstraps

**New approach:**
```python
grouped = df.groupby("N")[metric].agg([("mean", "mean"), ("sem", ...)])
ax.plot(...); ax.fill_between(..., alpha=0.2)
```
- Uses shaded regions (more modern/professional)
- Properly computes SEM across all fold×bootstrap combinations
- Clearer visualization of uncertainty

The shaded regions make it much easier to see:
- Where methods differ significantly
- Which N values have high/low uncertainty
- Overall trends in the data

## Questions?
If something doesn't work or you need a custom visualization, the code is
well-documented and easy to modify. Start with the basic functions and
customize as needed!
