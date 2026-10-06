"""
fm_difficulty_analysis.py

Modular functions for the difficulty-tier analysis of foundation-model
representations vs. a classical reference (default: TF-SAPIENS vs Raw
Log-norm), across the low-data sweep, for both the disease and cell-type
benchmarks.

Every metric-dependent step takes `metric` as an explicit argument (default
"MCC"), so the whole pipeline can be re-run on F1_macro, recall_macro, etc.
without touching any plotting code. `model` and `reference` are likewise
parameters, not hardcoded globals, so the same functions work for any
pairwise comparison, not just TF-SAPIENS vs Raw Log-norm.

Typical notebook usage:

    from fm_difficulty_analysis import *

    disease_full, celltype_full, disease_low, celltype_low = prepare_datasets(
        df_standard, df_celltype_standard, df_disease_low, df_celltype_low
    )

    difficulty_disease, difficulty_ct, boundaries = compute_difficulty(
        disease_full, celltype_full, metric="MCC"
    )
    disease_low, celltype_low = merge_difficulty(
        disease_low, celltype_low, difficulty_disease, difficulty_ct
    )

    results_task = compute_model_vs_reference(
        disease_low, celltype_low, metric="MCC", level="task"
    )
    results_dataset = compute_model_vs_reference(
        disease_low, celltype_low, metric="MCC", level="dataset"
    )
    fig1 = plot_absolute_metric_by_tier(results_dataset, metric="MCC")

    # Re-run the whole thing on F1_macro just by changing one argument:
    results_dataset_f1 = compute_model_vs_reference(
        disease_low, celltype_low, metric="F1_macro", level="dataset"
    )
    fig1_f1 = plot_absolute_metric_by_tier(results_dataset_f1, metric="F1_macro")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from feature_colors import get_color, FEATURE_COLOR_MAP


# ============================================================
# Module-level defaults (all overridable per function call)
# ============================================================

DEFAULT_TIERS = ("Hard", "Medium", "Easy")
DEFAULT_N_VALUES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
DEFAULT_BASELINES = ("PCA-50", "Raw Log-norm")
DEFAULT_MODEL = "TF-SAPIENS"
DEFAULT_REFERENCE = "Raw Log-norm"

# Colors for Disease vs Cell-type comparisons -- these compare BENCHMARKS,
# not models, so they deliberately sit outside FEATURE_COLOR_MAP.
BENCHMARK_COLORS = {"Disease": "#17becf", "Cell type": "#e377c2"}


# ============================================================
# 1. Data preparation
# ============================================================

def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows with a recorded error or skip_reason, and rows with no
    MCC (used as the canonical "did this row actually run" signal)."""
    df = df.copy()
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if "skip_reason" in df.columns:
        df = df[df["skip_reason"].isna()]
    df = df[df["MCC"].notna()].copy()
    return df


def add_task_ids(disease_df: pd.DataFrame, celltype_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Disease: one task = dataset_id x cell_type. Cell type: one task = dataset."""
    disease_df = disease_df.copy()
    celltype_df = celltype_df.copy()

    if "task_id" not in disease_df.columns:
        disease_df["task_id"] = (
            disease_df["dataset_id"].astype(str) + "::" + disease_df["cell_type"].astype(str)
        )
    celltype_df["task_id"] = celltype_df["dataset_id"].astype(str)

    return disease_df, celltype_df


def prepare_datasets(
    df_standard: pd.DataFrame,
    df_celltype_standard: pd.DataFrame,
    df_disease_low: pd.DataFrame,
    df_celltype_low: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-call setup: clean all four inputs and attach task_id.
    Returns (disease_full, celltype_full, disease_low, celltype_low)."""
    disease_full = clean_results(df_standard)
    celltype_full = clean_results(df_celltype_standard)
    disease_low = clean_results(df_disease_low)
    celltype_low = clean_results(df_celltype_low)

    disease_full, celltype_full = add_task_ids(disease_full, celltype_full)
    disease_low, celltype_low = add_task_ids(disease_low, celltype_low)

    return disease_full, celltype_full, disease_low, celltype_low


# ============================================================
# 2. Difficulty tiering
# ============================================================

def make_difficulty_table(
    df: pd.DataFrame,
    task_col: str,
    metric: str = "MCC",
    baselines: tuple[str, ...] = DEFAULT_BASELINES,
    tiers: tuple[str, ...] = DEFAULT_TIERS,
) -> tuple[pd.DataFrame, float, float]:
    """
    Difficulty = mean full-data `metric` of `baselines`, tertile-split into
    `tiers`. Returns (table with difficulty_score + difficulty_tier, q1, q2).
    """
    task_method = (
        df[df["feature_clean"].isin(baselines)]
        .groupby([task_col, "feature_clean"])[metric]
        .mean()
        .unstack("feature_clean")
    )
    task_method = task_method[list(baselines)].dropna()
    task_method["difficulty_score"] = task_method[list(baselines)].mean(axis=1)

    q1, q2 = task_method["difficulty_score"].quantile([1 / 3, 2 / 3])
    task_method["difficulty_tier"] = pd.cut(
        task_method["difficulty_score"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=tiers,
        include_lowest=True,
    )
    return task_method.reset_index(), q1, q2


def compute_difficulty(
    disease_full: pd.DataFrame,
    celltype_full: pd.DataFrame,
    metric: str = "MCC",
    baselines: tuple[str, ...] = DEFAULT_BASELINES,
    tiers: tuple[str, ...] = DEFAULT_TIERS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Computes difficulty tiers for both benchmarks. Returns
    (difficulty_disease, difficulty_ct, boundaries), where boundaries is
    {"disease": (q1, q2), "celltype": (q1, q2)}.
    """
    difficulty_disease, q1_dis, q2_dis = make_difficulty_table(
        disease_full, task_col="task_id", metric=metric, baselines=baselines, tiers=tiers
    )
    difficulty_disease[["dataset_id", "cell_type"]] = (
        difficulty_disease["task_id"].str.split("::", n=1, expand=True)
    )

    difficulty_ct, q1_ct, q2_ct = make_difficulty_table(
        celltype_full, task_col="task_id", metric=metric, baselines=baselines, tiers=tiers
    )
    difficulty_ct = difficulty_ct.rename(columns={"task_id": "dataset_id"})

    if verbose:
        print(f"DISEASE ({metric}) tier boundaries:", q1_dis, q2_dis)
        print(difficulty_disease["difficulty_tier"].value_counts().reindex(tiers))
        print(f"\nCELL TYPE ({metric}) tier boundaries:", q1_ct, q2_ct)
        print(difficulty_ct["difficulty_tier"].value_counts().reindex(tiers))

    boundaries = {"disease": (q1_dis, q2_dis), "celltype": (q1_ct, q2_ct)}
    return difficulty_disease, difficulty_ct, boundaries


def merge_difficulty(
    disease_low: pd.DataFrame,
    celltype_low: pd.DataFrame,
    difficulty_disease: pd.DataFrame,
    difficulty_ct: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attaches difficulty_score/difficulty_tier onto the low-data dataframes
    (fixed across all n_per_class values)."""
    disease_low = disease_low.merge(
        difficulty_disease[["task_id", "difficulty_score", "difficulty_tier"]].drop_duplicates("task_id"),
        on="task_id", how="inner",
    )
    celltype_low = celltype_low.merge(
        difficulty_ct[["dataset_id", "difficulty_score", "difficulty_tier"]].drop_duplicates("dataset_id"),
        on="dataset_id", how="inner",
    )
    return disease_low, celltype_low


# ============================================================
# 3. Paired performance helpers
# ============================================================

def get_paired_performance(
    df: pd.DataFrame,
    n,
    tier: str,
    task_col: str,
    metric: str = "MCC",
    model: str = DEFAULT_MODEL,
    reference: str = DEFAULT_REFERENCE,
) -> pd.DataFrame:
    """Task-level model/reference pairs for one (n, tier). Columns: reference, model."""
    x = df[
        (df["n_per_class"] == n)
        & (df["difficulty_tier"] == tier)
        & (df["feature_clean"].isin([reference, model]))
    ].copy()
    perf = x.groupby([task_col, "feature_clean"])[metric].mean().unstack("feature_clean")
    if reference not in perf.columns or model not in perf.columns:
        return pd.DataFrame()
    return perf[[reference, model]].dropna()


def get_paired_performance_with_dataset(
    df: pd.DataFrame,
    n,
    tier: str,
    task_col: str,
    dataset_col: str = "dataset_id",
    metric: str = "MCC",
    model: str = DEFAULT_MODEL,
    reference: str = DEFAULT_REFERENCE,
) -> pd.DataFrame:
    """Same as get_paired_performance, but keeps dataset_id attached so
    results can be collapsed to dataset level (fixes within-dataset task
    clustering, e.g. ICC ~0.7 for disease task difficulty)."""
    x = df[
        (df["n_per_class"] == n)
        & (df["difficulty_tier"] == tier)
        & (df["feature_clean"].isin([reference, model]))
    ].copy()

    group_cols = [task_col] if task_col == dataset_col else [task_col, dataset_col]
    perf = x.groupby(group_cols + ["feature_clean"])[metric].mean().unstack("feature_clean")

    if reference not in perf.columns or model not in perf.columns:
        return pd.DataFrame()

    perf = perf[[reference, model]].dropna().reset_index()
    if task_col == dataset_col:
        perf[dataset_col] = perf[task_col]
    return perf  # columns: task_col, dataset_col, reference, model


# ============================================================
# 4. Model vs reference: significance per benchmark x tier x n
# ============================================================

def compute_model_vs_reference(
    disease_low: pd.DataFrame,
    celltype_low: pd.DataFrame,
    metric: str = "MCC",
    model: str = DEFAULT_MODEL,
    reference: str = DEFAULT_REFERENCE,
    n_values=DEFAULT_N_VALUES,
    tiers=DEFAULT_TIERS,
    level: str = "task",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Paired Wilcoxon (model > reference) per benchmark x tier x n, with Holm
    correction within each (benchmark, tier) group.

    level="task":    uses raw task-level pairs (can overstate significance
                     for disease, where tasks cluster within datasets).
    level="dataset": collapses to one value per dataset before testing
                     (the corrected version -- use this for reporting).
    """
    assert level in ("task", "dataset")
    rows = []

    for benchmark, df, task_col in [("Disease", disease_low, "task_id"),
                                     ("Cell type", celltype_low, "dataset_id")]:
        for tier in tiers:
            for n in n_values:
                if level == "task":
                    perf = get_paired_performance(
                        df=df, n=n, tier=tier, task_col=task_col,
                        metric=metric, model=model, reference=reference,
                    )
                    if len(perf) == 0:
                        continue
                    delta = perf[model] - perf[reference]
                    ref_vals, model_vals = perf[reference], perf[model]
                    n_units = len(perf)
                else:
                    perf = get_paired_performance_with_dataset(
                        df=df, n=n, tier=tier, task_col=task_col,
                        metric=metric, model=model, reference=reference,
                    )
                    if len(perf) == 0:
                        continue
                    perf["delta"] = perf[model] - perf[reference]
                    delta = perf.groupby("dataset_id")["delta"].mean()
                    ref_vals = perf.groupby("dataset_id")[reference].mean()
                    model_vals = perf.groupby("dataset_id")[model].mean()
                    n_units = len(delta)

                try:
                    if np.allclose(delta.values, 0):
                        p = 1.0
                    elif n_units < 2:
                        p = np.nan
                    else:
                        _, p = wilcoxon(model_vals, ref_vals, alternative="greater")
                except ValueError:
                    p = np.nan

                rows.append({
                    "benchmark": benchmark, "tier": tier, "n_per_class": n,
                    "n_units": n_units, "level": level,
                    "reference_mean": ref_vals.mean(), "model_mean": model_vals.mean(),
                    "delta_mean": delta.mean(), "delta_median": delta.median(),
                    "win_rate": (delta > 0).mean() * 100,
                    "p_raw": p,
                })

    results = pd.DataFrame(rows)
    results["p_holm"] = np.nan
    for (benchmark, tier), g in results.groupby(["benchmark", "tier"]):
        idx = g.index[g["p_raw"].notna()]
        if len(idx) > 0:
            results.loc[idx, "p_holm"] = multipletests(results.loc[idx, "p_raw"], method="holm")[1]
    results["significant"] = results["p_holm"] < alpha
    return results


# ============================================================
# 5. Disease vs cell-type delta comparison, per tier x n
# ============================================================

def compute_benchmark_delta_comparison(
    disease_low: pd.DataFrame,
    celltype_low: pd.DataFrame,
    metric: str = "MCC",
    model: str = DEFAULT_MODEL,
    reference: str = DEFAULT_REFERENCE,
    n_values=DEFAULT_N_VALUES,
    tiers=DEFAULT_TIERS,
    level: str = "task",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Mann-Whitney comparing the disease delta distribution to the cell-type
    delta distribution, per tier x n. Same level="task"/"dataset" choice as
    compute_model_vs_reference; use "dataset" for reporting (disease tasks
    cluster within datasets, ICC ~0.7).
    """
    assert level in ("task", "dataset")
    rows = []

    for tier in tiers:
        for n in n_values:
            if level == "task":
                perf_dis = get_paired_performance(
                    df=disease_low, n=n, tier=tier, task_col="task_id",
                    metric=metric, model=model, reference=reference,
                )
                perf_ct = get_paired_performance(
                    df=celltype_low, n=n, tier=tier, task_col="dataset_id",
                    metric=metric, model=model, reference=reference,
                )
                if len(perf_dis) == 0 or len(perf_ct) == 0:
                    continue
                delta_dis = perf_dis[model] - perf_dis[reference]
                delta_ct = perf_ct[model] - perf_ct[reference]
            else:
                perf_dis = get_paired_performance_with_dataset(
                    df=disease_low, n=n, tier=tier, task_col="task_id",
                    metric=metric, model=model, reference=reference,
                )
                if len(perf_dis) == 0:
                    continue
                perf_dis["delta"] = perf_dis[model] - perf_dis[reference]
                delta_dis = perf_dis.groupby("dataset_id")["delta"].mean()

                perf_ct = get_paired_performance_with_dataset(
                    df=celltype_low, n=n, tier=tier, task_col="dataset_id",
                    metric=metric, model=model, reference=reference,
                )
                if len(perf_ct) == 0:
                    continue
                perf_ct["delta"] = perf_ct[model] - perf_ct[reference]
                delta_ct = perf_ct.groupby("dataset_id")["delta"].mean()

            if len(delta_dis) < 2 or len(delta_ct) < 2:
                continue

            U, p = mannwhitneyu(delta_dis, delta_ct, alternative="two-sided")
            rows.append({
                "tier": tier, "n_per_class": n, "level": level,
                "n_disease_units": len(delta_dis), "n_celltype_units": len(delta_ct),
                "mean_disease": delta_dis.mean(), "mean_celltype": delta_ct.mean(),
                "difference_of_means": delta_dis.mean() - delta_ct.mean(),
                "U": U, "p_raw": p,
            })

    results = pd.DataFrame(rows)
    results["p_holm"] = np.nan
    for tier in tiers:
        idx = (results["tier"] == tier) & results["p_raw"].notna()
        if idx.sum() > 0:
            results.loc[idx, "p_holm"] = multipletests(results.loc[idx, "p_raw"], method="holm")[1]
    results["significant"] = results["p_holm"] < alpha
    return results


# ============================================================
# 6. Sensitivity check: task-level vs dataset-level significance
# ============================================================

def sensitivity_check(
    results_task: pd.DataFrame,
    results_dataset: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Flags rows where the significance verdict flips between task-level
    and dataset-level testing. group_cols should match the join key, e.g.
    ["benchmark", "tier", "n_per_class"] for compute_model_vs_reference
    output, or ["tier", "n_per_class"] for compute_benchmark_delta_comparison."""
    merged = results_task.merge(
        results_dataset[group_cols + ["significant"]],
        on=group_cols, suffixes=("_task", "_dataset"),
    )
    merged["flipped"] = merged["significant_task"] != merged["significant_dataset"]
    return merged[merged["flipped"]]


# ============================================================
# 7. Plots
# ============================================================

def plot_absolute_metric_by_tier(
    results_dataset: pd.DataFrame,
    metric: str = "MCC",
    model: str = DEFAULT_MODEL,
    reference: str = DEFAULT_REFERENCE,
    tiers=DEFAULT_TIERS,
    n_values=DEFAULT_N_VALUES,
    save_path: str | None = None,
):
    """Figure 1: absolute metric value for model vs reference, one row per
    benchmark, one column per tier. Stars from results_dataset["significant"]."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
    benchmark_rows = [("Cell type", 0), ("Disease", 1)]

    for benchmark, row_i in benchmark_rows:
        r_all = results_dataset[results_dataset["benchmark"] == benchmark]
        ymax_global = max(r_all["reference_mean"].max(), r_all["model_mean"].max())
        row_ylim = (0, min(1.05, ymax_global + 0.08)) if benchmark == "Cell type" else (0, ymax_global + 0.08)

        for col_i, tier in enumerate(tiers):
            ax = axes[row_i, col_i]
            r = results_dataset[
                (results_dataset["benchmark"] == benchmark) & (results_dataset["tier"] == tier)
            ].sort_values("n_per_class")

            ax.plot(r["n_per_class"], r["reference_mean"], marker="o", linewidth=2,
                     color=get_color(reference), label=reference)
            ax.plot(r["n_per_class"], r["model_mean"], marker="o", linewidth=2,
                     color=get_color(model), label=model)

            sig = r[r["significant"]]
            for _, x in sig.iterrows():
                y = max(x["reference_mean"], x["model_mean"])
                offset = 0.025 if benchmark == "Cell type" else 0.015
                ax.text(x["n_per_class"], y + offset, "*", ha="center", va="bottom",
                        fontsize=14, fontweight="bold")

            ax.set_xscale("log", base=2)
            ax.set_xticks(n_values)
            ax.set_xticklabels(n_values, rotation=45, fontsize=10)
            ax.set_ylim(row_ylim)
            ax.grid(alpha=0.2)
            if row_i == 0:
                ax.set_title(tier, fontsize=14)
            if col_i == 0:
                ax.set_ylabel(f"{benchmark}\nMean {metric}", fontsize=14)
            if row_i == 1:
                ax.set_xlabel("Training samples per class", fontsize=14)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    star_handle = plt.Line2D([0], [0], marker=r"$\ast$", color="black",
                             linestyle="None", markersize=10)
    star_label = r"$p_{\mathrm{Holm}}<0.05$ (one-sided paired Wilcoxon)"
    fig.legend(handles + [star_handle], labels + [star_label],
               loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    fig.suptitle(
        f"{model} vs {reference} across data regimes and task difficulty ({metric})",
        y=0.995, fontsize = 16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def plot_winrate_by_tier(
    results_task: pd.DataFrame,
    tiers=DEFAULT_TIERS,
    n_values=DEFAULT_N_VALUES,
    benchmark_colors=BENCHMARK_COLORS,
    save_path: str | None = None,
):
    """Figure 2: descriptive win rate, no significance testing."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, tier in zip(axes, tiers):
        r = results_task[results_task["tier"] == tier]
        for benchmark in ["Disease", "Cell type"]:
            d = r[r["benchmark"] == benchmark].sort_values("n_per_class")
            ax.plot(d["n_per_class"], d["win_rate"], color=benchmark_colors[benchmark],
                     marker="o", linewidth=2, markersize=5, label=benchmark)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values)
        ax.set_xticklabels(n_values, rotation=45, fontsize=8)
        ax.set_ylim(-5, 108)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_title(tier)
        ax.set_xlabel("Training samples per class")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("% tasks where model > reference")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    fig.suptitle("Win rate over reference by task difficulty", y=1.15)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


# ------------------------------------------------------------
# 7b. Same as above, but pooled across all difficulty tiers
# (one line per benchmark, not one subplot per tier)
# ------------------------------------------------------------

def get_paired_performance_overall(
    df: pd.DataFrame,
    n,
    task_col: str,
    metric: str = "MCC",
    model: str = DEFAULT_MODEL,
    reference: str = DEFAULT_REFERENCE,
) -> pd.DataFrame:
    """Same as get_paired_performance, but does NOT filter by difficulty_tier
    -- pools every task at this n regardless of tier."""
    x = df[
        (df["n_per_class"] == n)
        & (df["feature_clean"].isin([reference, model]))
    ].copy()
    perf = x.groupby([task_col, "feature_clean"])[metric].mean().unstack("feature_clean")
    if reference not in perf.columns or model not in perf.columns:
        return pd.DataFrame()
    return perf[[reference, model]].dropna()


def compute_model_vs_reference_overall(
    disease_low: pd.DataFrame,
    celltype_low: pd.DataFrame,
    metric: str = "MCC",
    model: str = DEFAULT_MODEL,
    reference: str = DEFAULT_REFERENCE,
    n_values=DEFAULT_N_VALUES,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Same as compute_model_vs_reference, but pooled across difficulty tiers:
    one row per benchmark x n (not benchmark x tier x n). Does not require
    difficulty_tier to be merged onto disease_low/celltype_low at all.
    """
    rows = []
    for benchmark, df, task_col in [("Disease", disease_low, "task_id"),
                                     ("Cell type", celltype_low, "dataset_id")]:
        for n in n_values:
            perf = get_paired_performance_overall(
                df=df, n=n, task_col=task_col, metric=metric, model=model, reference=reference,
            )
            if len(perf) == 0:
                continue
            delta = perf[model] - perf[reference]
            try:
                if np.allclose(delta.values, 0):
                    p = 1.0
                else:
                    _, p = wilcoxon(perf[model], perf[reference], alternative="greater")
            except ValueError:
                p = np.nan

            rows.append({
                "benchmark": benchmark, "n_per_class": n, "n_units": len(perf),
                "reference_mean": perf[reference].mean(), "model_mean": perf[model].mean(),
                "delta_mean": delta.mean(), "delta_median": delta.median(),
                "win_rate": (delta > 0).mean() * 100,
                "p_raw": p,
            })

    results = pd.DataFrame(rows)
    results["p_holm"] = np.nan
    for benchmark, g in results.groupby("benchmark"):
        idx = g.index[g["p_raw"].notna()]
        if len(idx) > 0:
            results.loc[idx, "p_holm"] = multipletests(results.loc[idx, "p_raw"], method="holm")[1]
    results["significant"] = results["p_holm"] < alpha
    return results


def plot_winrate_overall(
    results_overall: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    reference: str = DEFAULT_REFERENCE,
    n_values=DEFAULT_N_VALUES,
    benchmark_colors=BENCHMARK_COLORS,
    save_path: str | None = None,
):
    """Single-panel win rate, Disease vs Cell type, pooled across all
    difficulty tiers -- companion to plot_winrate_by_tier."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for benchmark in ["Disease", "Cell type"]:
        d = results_overall[results_overall["benchmark"] == benchmark].sort_values("n_per_class")
        ax.plot(d["n_per_class"], d["win_rate"], color=benchmark_colors[benchmark],
                 marker="o", linewidth=2, markersize=5, label=benchmark)

    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels(n_values, rotation=45, fontsize=8)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Training samples per class")
    ax.set_ylabel(f"% tasks where {model} > {reference}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.set_title(f"Win rate of {model} over {reference}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def plot_delta_by_tier_comparison(
    results_tier_dataset: pd.DataFrame,
    metric: str = "MCC",
    model: str = DEFAULT_MODEL,
    reference: str = DEFAULT_REFERENCE,
    tiers=DEFAULT_TIERS,
    n_values=DEFAULT_N_VALUES,
    benchmark_colors=BENCHMARK_COLORS,
    save_path: str | None = None,
):
    """Figure 3: mean delta (model - reference) for disease vs cell-type,
    per tier. Stars from results_tier_dataset["significant"]."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, tier in zip(axes, tiers):
        r = results_tier_dataset[results_tier_dataset["tier"] == tier].sort_values("n_per_class")

        ax.plot(r["n_per_class"], r["mean_disease"], marker="o", linewidth=2,
                 color=benchmark_colors["Disease"], label="Disease")
        ax.plot(r["n_per_class"], r["mean_celltype"], marker="o", linestyle="--", linewidth=2,
                 color=benchmark_colors["Cell type"], label="Cell type")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values)
        ax.set_xticklabels(n_values, rotation=45, fontsize=8)
        ax.set_title(tier, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training samples per class")
        ax.grid(alpha=0.2)

        sig = r[r["significant"]]
        for _, row in sig.iterrows():
            ymax = max(row["mean_disease"], row["mean_celltype"])
            ax.text(row["n_per_class"], ymax + 0.02, "*", ha="center", va="bottom",
                    fontsize=14, fontweight="bold")

    axes[0].set_ylabel(f"Mean \u0394{metric} ({model} \u2212 {reference})")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    fig.suptitle(
        f"{model} advantage over {reference} \u2014 Disease vs cell-type ({metric})\n"
        "* disease-vs-CT difference significant, Mann-Whitney + Holm",
        y=1.15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


# ============================================================
# 8. Per-task trajectories by difficulty tier (2x2: benchmark x feature)
# ============================================================

# Default shade sets -- darker = Hard, lighter = Easy. These are fixed hex
# values (not derived via adjust_lightness/tier_shades below) because
# linear RGB lightening/darkening tends to look grey; pick your own
# {"Hard":..., "Medium":..., "Easy":...} dict per feature if you want a
# different palette.
_BLUE_DARK, _BLUE_MID, _BLUE_LIGHT = "#000080", "#1F51FF", "#89CFF0"
_PURPLE_DARK, _PURPLE_MID, _PURPLE_LIGHT = "#5D3FD3", "#7F00FF", "#CBC3E3"

DEFAULT_REFERENCE_SHADES = {"Hard": _BLUE_DARK, "Medium": _BLUE_MID, "Easy": _BLUE_LIGHT}
DEFAULT_MODEL_SHADES = {"Hard": _PURPLE_DARK, "Medium": _PURPLE_MID, "Easy": _PURPLE_LIGHT}


def adjust_lightness(hex_color: str, factor: float) -> str:
    """factor > 1 lightens, < 1 darkens, 1 leaves unchanged. Works in HLS
    space (hue/saturation preserved) to avoid the greyish look you get from
    scaling RGB linearly toward black/white. Optional alternative to picking
    fixed hex shades by hand."""
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def tier_shades(base_color: str, tiers=DEFAULT_TIERS) -> dict:
    """Derives Hard/Medium/Easy shades from one base color via adjust_lightness,
    for when you'd rather generate a palette than hand-pick hex values."""
    factors = {"Hard": 0.55, "Medium": 1.0, "Easy": 1.55}
    return {t: adjust_lightness(base_color, factors.get(t, 1.0)) for t in tiers}


def _draw_task_lines_panel(
    ax1, ax2, df_low, df_full, difficulty_table, task_col,
    feature, shades, metric="MCC", n_values=DEFAULT_N_VALUES,
    full_data_label=2048, line_alpha=0.5, line_width=1.0,
    exclude_n=(512, 1024),
):
    """Draws one panel (main low-data trajectory + a broken-axis standard-mode
    point) for a single feature, colored per-task by difficulty tier."""
    low = df_low[df_low["feature_clean"] == feature].copy()
    low = low[~low["n_per_class"].isin(exclude_n)]
    low_agg = low.groupby([task_col, "n_per_class"])[metric].mean().reset_index()

    full = df_full[df_full["feature_clean"] == feature].copy()
    full_agg = (
        full.groupby(task_col)[metric].mean().reset_index()
        .rename(columns={metric: f"{metric}_standard"})
    )

    tier_lookup = difficulty_table[[task_col, "difficulty_tier"]].drop_duplicates(task_col)
    low_agg = low_agg.merge(tier_lookup, on=task_col, how="inner")
    full_agg = full_agg.merge(tier_lookup, on=task_col, how="inner")

    for tid, g in low_agg.groupby(task_col):
        g = g.sort_values("n_per_class")
        tier = g["difficulty_tier"].iloc[0]
        ax1.plot(g["n_per_class"], g[metric],
                  color=shades.get(tier, "#999999"),
                  alpha=line_alpha, linewidth=line_width)

    for tid, row in full_agg.set_index(task_col).iterrows():
        tier = row["difficulty_tier"]
        ax2.plot(full_data_label, row[f"{metric}_standard"],
                  marker="o", markersize=4,
                  color=shades.get(tier, "#999999"), alpha=1.0)

    xticks_shown = [n for n in n_values if n not in exclude_n]
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(xticks_shown)
    ax1.set_xticklabels(xticks_shown, rotation=45, fontsize=9)
    ax1.tick_params(axis="y", left=False)

    ax1.spines["right"].set_visible(False)

    ax2.set_xticks([full_data_label])
    ax2.set_xticklabels(["Full  Data"], fontsize=9)
    ax2.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )
    ax2.spines["left"].set_visible(False)
    #ax2.yaxis.tick_right()

    kwargs = dict(marker=[(-1, -1), (1, 1)], markersize=10,
                  linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)


def plot_task_trajectories_2x2(
    disease_low: pd.DataFrame,
    disease_full: pd.DataFrame,
    celltype_low: pd.DataFrame,
    celltype_full: pd.DataFrame,
    difficulty_disease: pd.DataFrame,
    difficulty_ct: pd.DataFrame,
    metric: str = "MCC",
    model: str = DEFAULT_MODEL,
    reference: str = DEFAULT_REFERENCE,
    reference_shades: dict = None,
    model_shades: dict = None,
    tiers=DEFAULT_TIERS,
    n_values=DEFAULT_N_VALUES,
    exclude_n=(512, 1024),
    ylim: tuple[float, float] | None = (0, 1.05),
    save_path: str | None = None,
):
    """
    2x2 grid: rows = benchmark (Cell type, Disease), columns = feature
    (reference, model). Each outer cell is itself split (main trajectory +
    a broken-axis standard-mode strip) via a nested GridSpec. Each panel
    shows one line per task, colored by difficulty tier (darker = Hard,
    lighter = Easy).

    exclude_n: n_per_class values dropped from the low-data trajectories
    (default excludes 512 and 1024).
    """
    reference_shades = reference_shades or DEFAULT_REFERENCE_SHADES
    model_shades = model_shades or DEFAULT_MODEL_SHADES

    fig = plt.figure(figsize=(14, 9))
    outer_gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25)

    panels = [
        (0, celltype_low, celltype_full, difficulty_ct, "dataset_id", "Cell type", 0, reference, reference_shades, reference),
        (0, celltype_low, celltype_full, difficulty_ct, "dataset_id", "Cell type", 1, model, model_shades, model),
        (1, disease_low, disease_full, difficulty_disease, "task_id", "Disease", 0, reference, reference_shades, reference),
        (1, disease_low, disease_full, difficulty_disease, "task_id", "Disease", 1, model, model_shades, model),
    ]

    for row, df_low, df_full, diff_table, task_col, bm_label, col, feature, shades, feat_label in panels:
        inner_gs = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gs[row, col],
            width_ratios=[3, 0.5], wspace=0.05,
        )
        ax1 = fig.add_subplot(inner_gs[0, 0])
        ax2 = fig.add_subplot(inner_gs[0, 1], sharey=ax1)

        _draw_task_lines_panel(
            ax1, ax2, df_low, df_full, diff_table, task_col,
            feature, shades, metric=metric, n_values=n_values, exclude_n=exclude_n,
        )

        if ylim is not None:
            ax1.set_ylim(ylim)
        ax1.set_ylabel(f"{bm_label}\n{metric}" if col == 0 else "", fontsize=14)

        
        if row == 1:
            ax1.set_xlabel("Training samples per class", fontsize=14)
        ax1.set_title(feat_label, fontsize=14)

        # Legend moved off the lines -- anchored to ax2 (the "Full Data" strip)
        # instead of sitting inside ax1 where it got covered by dense trajectories.
        legend_handles = [
            plt.Line2D([0], [0], color=shades[t], lw=2, label=t) for t in tiers
        ]
        ax2.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.1, 1.0),
            fontsize=7,
            frameon=False,
        )

    fig.suptitle(
        f"Per-task {metric} trajectories by difficulty tier\n",
        y=0.98, fontsize=16
    )

    fig.subplots_adjust(right=0.90)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig