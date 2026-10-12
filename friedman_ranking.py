"""
20 08 after wilcoxon dependency correction

friedman_ranking.py

Friedman + Nemenyi post-hoc + Critical Difference diagram
for comparing M models over T heterogeneous tasks.

Reference: Demšar, JMLR 2006.

CORRECTED VERSION: prepare_score_matrix and run_friedman_per_nperclass
accept an optional `dataset_col` argument. When provided, task-level
scores are collapsed to one score per dataset before Friedman/Wilcoxon
testing, so that T = number of independent datasets rather than number
of (possibly clustered) tasks.

Why this matters: disease tasks are dataset_id x cell_type, so multiple
tasks can share a dataset of origin. Measured ICC ~0.71-0.73 for disease
task difficulty within a dataset -- treating those tasks as independent
inflates significance (see sensitivity checks run 2026-08). Cell-type
tasks are already one-per-dataset, so dataset_col is a no-op there.

Usage:
    from friedman_ranking import run_friedman, plot_cd_diagram

    # Task-level (original behavior, still available):
    score_matrix = prepare_score_matrix(df, task_col="task_id")

    # Dataset-collapsed (corrected, use for disease):
    score_matrix = prepare_score_matrix(df, task_col="task_id", dataset_col="dataset_id")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare


# ---------------------------------------------------------------------------
# Step 1: Prepare the (T x M) score matrix
# ---------------------------------------------------------------------------

def prepare_score_matrix(
    df: pd.DataFrame,
    task_col: str,          # "task_id" for disease, "dataset_id" for cell-type
    feature_col: str = "feature_clean",
    metric: str = "MCC",
    exclude_features: list[str] = ("random_proj50",),
    dataset_col: str | None = None,
) -> pd.DataFrame:
    """
    Averages across folds to produce one score per (task, feature),
    then pivots to a (T x M) matrix. Drops tasks with any missing feature.

    Args:
        df:               Results DataFrame. Must contain task_col, feature_col, metric.
        task_col:         Column identifying tasks.
                          - disease:   "task_id"   (dataset_id + cell_type)
                          - cell-type: "dataset_id"
        feature_col:      Column identifying models/representations.
        metric:           Metric to use (default: MCC).
        exclude_features: Features to drop before analysis.
        dataset_col:      If provided, task-level scores are further collapsed
                          to one score per (dataset_col, feature_col) before
                          pivoting -- this avoids treating multiple tasks that
                          share a dataset of origin as independent observations
                          (see ICC ~0.7 for disease tasks clustered within
                          dataset_id). Pass "dataset_id" for disease to get the
                          clustering-corrected T. For cell-type, task_col is
                          already dataset_id, so this is a no-op -- safe to
                          pass or omit.

    Returns:
        DataFrame of shape (T, M), rows = tasks (or datasets, if dataset_col
        given), columns = features. Only rows with scores for ALL features
        are kept.
    """
    df = df[~df[feature_col].isin(exclude_features)].copy()

    # Average across folds: one score per (task, feature)
    task_feature = (
        df.groupby([task_col, feature_col])[metric]
        .mean()
        .reset_index()
    )

    if dataset_col is not None:
        if task_col == dataset_col:
            # Already one row per dataset (e.g. cell-type, where task_col IS
            # dataset_id) -- nothing to collapse. Selecting df[[task_col,
            # dataset_col]] here would select the same column twice and break
            # the merge below, so skip the merge entirely in this case.
            task_feature = (
                task_feature.groupby([task_col, feature_col])[metric]
                .mean()
                .reset_index()
            )
            index_col = task_col
        else:
            # Collapse task-level scores to one score per (dataset, feature),
            # so clustered tasks within the same dataset stop being counted
            # as independent rows.
            task_to_dataset = df[[task_col, dataset_col]].drop_duplicates()
            task_feature = task_feature.merge(task_to_dataset, on=task_col, how="left")
            task_feature = (
                task_feature.groupby([dataset_col, feature_col])[metric]
                .mean()
                .reset_index()
            )
            index_col = dataset_col
    else:
        index_col = task_col

    # Pivot to wide
    wide = task_feature.pivot(index=index_col, columns=feature_col, values=metric)

    n_before = len(wide)
    wide = wide.dropna()
    n_after = len(wide)

    unit = "datasets" if dataset_col is not None else "tasks"
    print(f"{unit.capitalize()} before dropna: {n_before}, after: {n_after} "
          f"({n_before - n_after} dropped due to missing features)")
    print(f"Features: {wide.columns.tolist()}")
    print(f"Score matrix shape: {wide.shape}  (T={n_after} {unit}, M={wide.shape[1]} models)")

    return wide


# ---------------------------------------------------------------------------
# Step 2: Friedman test + average ranks
# ---------------------------------------------------------------------------

def run_friedman(
    score_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """
    Runs Friedman test (Iman-Davenport F-correction) on the score matrix.
    Returns average ranks and test results.

    Unchanged by the clustering correction: it only ever sees whatever
    score_matrix it's given (T rows, whatever T represents -- tasks or
    datasets is decided upstream in prepare_score_matrix).

    Args:
        score_matrix: DataFrame (T x M), higher = better.
        alpha:        Significance level (default: 0.05).

    Returns:
        dict with keys:
            avg_ranks   - Series: feature -> average rank (lower = better)
            friedman_stat  - chi² statistic
            friedman_p     - p-value
            iman_davenport_F - F-corrected statistic
            iman_davenport_p - F-corrected p-value
            T, M         - number of tasks and models
            significant  - bool
    """
    T, M = score_matrix.shape

    # Rank within each task (ascending rank = worse, so rank 1 = best)
    # Use ascending=False so rank 1 = highest score
    ranks = score_matrix.rank(axis=1, ascending=False, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values()

    # Friedman chi² statistic
    stat, p_friedman = friedmanchisquare(*[score_matrix[col].values for col in score_matrix.columns])

    # Iman-Davenport F-correction (less conservative)
    FF = (T - 1) * stat / (T * (M - 1) - stat)
    from scipy.stats import f as f_dist
    p_ff = 1 - f_dist.cdf(FF, dfn=M - 1, dfd=(T - 1) * (M - 1))

    print(f"\n=== Friedman Test ===")
    print(f"T={T} tasks, M={M} models")
    print(f"Chi² = {stat:.4f}, p = {p_friedman:.4e}")
    print(f"Iman-Davenport F = {FF:.4f}, p = {p_ff:.4e}")
    print(f"Significant at alpha={alpha}: {p_ff < alpha}")
    print(f"\nAverage ranks (lower = better):")
    print(avg_ranks.round(4).to_string())

    return {
        "avg_ranks": avg_ranks,
        "friedman_stat": stat,
        "friedman_p": p_friedman,
        "iman_davenport_F": FF,
        "iman_davenport_p": p_ff,
        "T": T,
        "M": M,
        "significant": p_ff < alpha,
        "ranks_matrix": ranks,
    }


# ---------------------------------------------------------------------------
# Step 3: Nemenyi critical difference
# ---------------------------------------------------------------------------

def nemenyi_cd(
    M: int,
    T: int,
    alpha: float = 0.05,
) -> float:
    """
    Computes the Nemenyi critical difference.
    Uses studentized range q_alpha values (table from Demšar 2006).

    Args:
        M:     Number of models.
        T:     Number of tasks (or datasets, if computed from a
               dataset-collapsed score matrix).
        alpha: Significance level (0.05 or 0.10).

    Returns:
        CD value (float).
    """
    # q_alpha / sqrt(2) for alpha=0.05, M=2..10 (from Demšar Table 5)
    q_table_005 = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
                   6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_table_010 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459,
                   6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}

    q_table = q_table_005 if alpha == 0.05 else q_table_010
    if M not in q_table:
        raise ValueError(f"M={M} not in Nemenyi table (supported: 2-10). "
                         f"Use pairwise Wilcoxon+Holm instead.")

    q = q_table[M]
    cd = q * np.sqrt(M * (M + 1) / (6 * T))
    print(f"Nemenyi CD (alpha={alpha}, M={M}, T={T}): {cd:.4f}")
    return cd


# ---------------------------------------------------------------------------
# Step 4: CD diagram
# ---------------------------------------------------------------------------

def plot_cd_diagram(
    avg_ranks: pd.Series,
    cd: float,
    title: str = "Critical Difference Diagram",
    color_map: dict[str, str] | None = None,
    figsize: tuple[float, float] = (8, 3),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Draws a Critical Difference diagram.
    Models are placed at their average rank on a horizontal axis.
    Horizontal bars connect groups that are NOT significantly different (diff < CD).

    Args:
        avg_ranks:  Series: feature -> average rank (lower = better, from run_friedman).
        cd:         Critical difference value (from nemenyi_cd).
        title:      Plot title.
        color_map:  Dict mapping feature -> color, or None.
        figsize:    Figure size.
        save_path:  If provided, saves figure here.

    Returns:
        Matplotlib figure.
    """
    avg_ranks = avg_ranks.sort_values()
    models = avg_ranks.index.tolist()
    ranks = avg_ranks.values
    M = len(models)

    fig, ax = plt.subplots(figsize=figsize)

    # Axis range
    rank_min = max(1, ranks.min() - 0.5)
    rank_max = min(M, ranks.max() + 0.5)
    ax.set_xlim(rank_min - 0.2, rank_max + 0.2)
    ax.set_ylim(-1, M + 1)
    ax.invert_xaxis()   # rank 1 (best) on the left
    ax.set_xlabel("Average rank  (lower = better)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    # Draw axis line
    ax.annotate("", xy=(rank_min - 0.1, M + 0.3), xytext=(rank_max + 0.1, M + 0.3),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.5))
    # Tick marks
    for r in np.arange(np.ceil(rank_min), np.floor(rank_max) + 1):
        ax.annotate("", xy=(r, M + 0.3), xytext=(r, M + 0.5),
                    arrowprops=dict(arrowstyle="-", color="black", lw=1))
        ax.text(r, M + 0.7, str(int(r)), ha="center", va="bottom", fontsize=9)

    # Place model names and dots
    for i, (model, rank) in enumerate(zip(models, ranks)):
        color = color_map.get(model, "#555555") if color_map else "#555555"
        y_pos = i
        # Dot on axis
        ax.plot(rank, M + 0.3, "o", color=color, markersize=7, zorder=5)
        # Vertical line down to label
        ax.plot([rank, rank], [M + 0.3, y_pos + 0.1], color=color, lw=1, ls="--", alpha=0.5)
        # Label
        ax.text(rank, y_pos, f"{model}  ({rank:.2f})",
                ha="center", va="top", fontsize=9, color=color, fontweight="bold")

    # Draw CD bars: connect groups not significantly different (rank diff < CD)
    # Find cliques: pairs within CD
    drawn = set()
    for i in range(M):
        group = [i]
        for j in range(i + 1, M):
            if abs(ranks[j] - ranks[i]) < cd:
                group.append(j)
        if len(group) > 1:
            key = tuple(group)
            if key not in drawn:
                drawn.add(key)
                y_bar = M + 0.15
                x_start = ranks[group[0]]
                x_end   = ranks[group[-1]]
                ax.plot([x_start, x_end], [y_bar, y_bar],
                        color="black", lw=3, solid_capstyle="round")

    # CD legend
    ax.annotate("", xy=(rank_min, -0.5), xytext=(rank_min + cd, -0.5),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(rank_min + cd / 2, -0.7, f"CD={cd:.3f}", ha="center", va="top",
            fontsize=9, color="gray")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Convenience wrapper: run everything in one call
# ---------------------------------------------------------------------------

def friedman_analysis(
    df: pd.DataFrame,
    task_col: str,
    feature_col: str = "feature_clean",
    metric: str = "MCC",
    exclude_features: list[str] = ("random_proj50",),
    alpha: float = 0.05,
    color_map: dict[str, str] | None = None,
    title: str = "Critical Difference Diagram",
    figsize: tuple[float, float] = (8, 3),
    save_path: str | None = None,
    dataset_col: str | None = None,
) -> dict:
    """
    Full pipeline: prepare matrix → Friedman → CD → plot.

    Args:
        df:               Results DataFrame.
        task_col:         "task_id" for disease, "dataset_id" for cell-type.
        feature_col:      Feature column (default: feature_clean).
        metric:           Metric to rank on (default: MCC).
        exclude_features: Features to exclude.
        alpha:            Significance level.
        color_map:        Feature -> color dict.
        title:            Plot title.
        figsize:          Figure size.
        save_path:        Save path for CD diagram.
        dataset_col:      If provided, collapses to one score per dataset
                          before testing (see prepare_score_matrix). Pass
                          "dataset_id" for disease to correct for task
                          clustering within datasets.

    Returns:
        dict with score_matrix, friedman_results, cd, fig.
    """
    score_matrix = prepare_score_matrix(
        df, task_col=task_col, feature_col=feature_col,
        metric=metric, exclude_features=exclude_features,
        dataset_col=dataset_col,
    )

    friedman_results = run_friedman(score_matrix, alpha=alpha)

    if not friedman_results["significant"]:
        print("\nFriedman test not significant — post-hoc not licensed.")
        return {"score_matrix": score_matrix, "friedman_results": friedman_results,
                "cd": None, "fig": None}

    cd = nemenyi_cd(M=friedman_results["M"], T=friedman_results["T"], alpha=alpha)

    fig = plot_cd_diagram(
        avg_ranks=friedman_results["avg_ranks"],
        cd=cd,
        title=title,
        color_map=color_map,
        figsize=figsize,
        save_path=save_path,
    )

    return {
        "score_matrix": score_matrix,
        "friedman_results": friedman_results,
        "cd": cd,
        "fig": fig,
    }


# ---------------------------------------------------------------------------
# Step 5: Pairwise Wilcoxon + Holm post-hoc on score matrix
# ---------------------------------------------------------------------------

def pairwise_wilcoxon_holm(
    score_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    All-pairs Wilcoxon signed-rank test on per-task (or per-dataset, if
    score_matrix came from a dataset-collapsed prepare_score_matrix call)
    scores, with Holm correction across all pairs.

    Unchanged by the clustering correction: operates purely on whatever
    score_matrix it receives.

    Args:
        score_matrix: DataFrame (T x M), rows = tasks or datasets, cols = features.
                      Output of prepare_score_matrix — already averaged across folds
                      (and across tasks within dataset, if dataset_col was used).
        alpha:        Significance level after correction (default: 0.05).

    Returns:
        DataFrame with columns:
            feature_i, feature_j,
            mean_i, mean_j, mean_delta_i_minus_j,
            p_raw, p_holm, significant
        One row per unique pair (upper triangle only).
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    from itertools import combinations

    features = score_matrix.columns.tolist()
    pairs = list(combinations(features, 2))

    rows = []
    for fi, fj in pairs:
        d = score_matrix[fi] - score_matrix[fj]
        if np.all(d == 0):
            p = 1.0
        else:
            _, p = wilcoxon(d, alternative="two-sided")
        rows.append({
            "feature_i": fi,
            "feature_j": fj,
            "mean_i": score_matrix[fi].mean(),
            "mean_j": score_matrix[fj].mean(),
            "mean_delta_i_minus_j": score_matrix[fi].mean() - score_matrix[fj].mean(),
            "p_raw": p,
        })

    results = pd.DataFrame(rows)

    # Holm correction across all pairs
    _, p_holm, _, _ = multipletests(results["p_raw"], method="holm")
    results["p_holm"] = p_holm
    results["p_adj"] = p_holm          # alias expected by plot_pairwise_direction_significance_heatmap
    results["significant"] = p_holm < alpha

    print(f"\n=== Pairwise Wilcoxon + Holm ({len(pairs)} pairs, alpha={alpha}) ===")
    print(results[["feature_i", "feature_j", "mean_delta_i_minus_j", "p_raw", "p_holm", "significant"]]
          .sort_values("p_holm")
          .round(4)
          .to_string(index=False))

    return results


# ---------------------------------------------------------------------------
# Wrapper: plot Friedman average ranks using plot_overall_performance_bar
# ---------------------------------------------------------------------------

def plot_friedman_as_bar(
    df: pd.DataFrame,
    plot_overall_performance_bar,
    task_col: str,
    metrics: tuple[str, str] = ("MCC", "F1_macro"),
    feature_col: str = "feature_clean",
    exclude_features: list[str] = ("random_proj50",),
    alpha: float = 0.05,
    feature_order: list[str] = None,
    color_map: dict[str, str] | None = None,
    figsize: tuple[float, float] = (9, 6),
    errorbar=None,
    save_path: str | None = None,
    dataset_col: str | None = None,
    title_prefix: str | None = None,
    reference: str = "Raw Log-norm",   # <-- NEW: star vs this model
) -> tuple:
    """
    ... (existing docstring unchanged) ...

    reference: Model to compare every other bar against. Bars significantly
               different from `reference` (pairwise Wilcoxon, Holm-corrected)
               get an asterisk. The reference bar itself is never starred.
    """
    from itertools import combinations

    metric1, metric2 = metrics
    rank_cols = [f"{m}_rank" for m in metrics]

    rank_series = {}
    cd_values = {}
    results_per_metric = {}
    pairwise_per_metric = {}   # <-- NEW

    for metric in metrics:
        score_matrix = prepare_score_matrix(
            df, task_col=task_col, feature_col=feature_col,
            metric=metric, exclude_features=exclude_features,
            dataset_col=dataset_col,
        )
        res = run_friedman(score_matrix, alpha=alpha)
        rank_series[metric] = res["avg_ranks"]
        results_per_metric[metric] = res
        if res["significant"]:
            try:
                cd_values[metric] = nemenyi_cd(M=res["M"], T=res["T"], alpha=alpha)
            except ValueError:
                cd_values[metric] = None
        else:
            cd_values[metric] = None

        # NEW: pairwise Wilcoxon+Holm for this metric, so we can star vs reference
        pairwise_per_metric[metric] = pairwise_wilcoxon_holm(score_matrix, alpha=alpha)

    if feature_order is None:
        feature_order = rank_series[metric1].sort_values().index.tolist()

    fake_rows = []
    for metric in metrics:
        col = f"{metric}_rank"
        rank_matrix = results_per_metric[metric]["ranks_matrix"]
        for feat in rank_matrix.columns:
            for task_id, rank_val in rank_matrix[feat].items():
                fake_rows.append({"task_id": task_id, feature_col: feat, col: rank_val})
    fake_df = pd.DataFrame(fake_rows)

    fig, ax, summary = plot_overall_performance_bar(
        fake_df,
        metrics=[f"{m}_rank" for m in metrics],
        feature_order=feature_order,
        color_map=color_map,
        figsize=figsize,
        title=f"{title_prefix} Friedman Average Ranks",
        xlabel="Average rank (lower = better)",
        bar_height=0.36,
        show_values=False,
        errorbar=errorbar,
    )

    ax.invert_xaxis()

    # --- NEW: asterisks vs reference, one offset per metric bar ---
    bar_height = 0.36
    y_positions = {feat: i for i, feat in enumerate(feature_order)}
    metric_offsets = [+bar_height / 2, -bar_height / 2]
    
    for metric, offset in zip(metrics, metric_offsets):
        pw = pairwise_per_metric[metric]
        for feat in feature_order:
            if feat == reference:
                continue
            row = pw[
                ((pw["feature_i"] == feat) & (pw["feature_j"] == reference)) |
                ((pw["feature_i"] == reference) & (pw["feature_j"] == feat))
            ]
            if len(row) == 0 or not row.iloc[0]["significant"]:
                continue
            rank_val = rank_series[metric].get(feat)
            if rank_val is None:
                continue
            ax.text(
                rank_val +.1 ,
                y_positions[feat] + offset - 0.1,
                "*", ha="center", va="bottom", fontsize=15,
                fontweight="bold", color="black",
            )


    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="black", edgecolor="black", label="MCC"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="F1-macro"),
    ]
    
    ax.legend(
        handles=legend_handles,
        frameon=True,
        fontsize=15, bbox_to_anchor=(0.26, 0.4)
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax, summary


# ---------------------------------------------------------------------------
# Low-data Friedman: run per n_per_class, visualize like z-score trajectory
# ---------------------------------------------------------------------------

def run_friedman_per_nperclass(
    df_lowdata: pd.DataFrame,
    df_standard: pd.DataFrame,
    task_col: str,
    feature_col: str = "feature_clean",
    metrics: list[str] | str = "MCC",
    exclude_features: list[str] = ("random_proj50",),
    alpha: float = 0.05,
    dataset_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs Friedman + pairwise Wilcoxon + Holm separately for each n_per_class
    in df_lowdata, and once for df_standard (full data), for each metric.

    Args:
        df_lowdata:       Low-data results DataFrame. Must have n_per_class column.
        df_standard:      Standard-mode results DataFrame (n_per_class is None).
        task_col:         "task_id" (disease) or "dataset_id" (cell-type).
        feature_col:      Feature column (default: feature_clean).
        metrics:          Metric or list of metrics to rank on (default: "MCC").
        exclude_features: Features to exclude.
        alpha:            Significance level.
        dataset_col:      If provided, collapses task-level scores to one
                          score per dataset before Friedman/Wilcoxon testing
                          at every n_per_class (and for standard mode) --
                          corrects for tasks clustered within a dataset.
                          Pass "dataset_id" for disease (ICC ~0.7 measured
                          for task difficulty within dataset). For cell-type,
                          task_col is already dataset_id, so this is a no-op
                          -- safe to pass or omit.

    Returns:
        ranks_df:    DataFrame with columns [metric, n_per_class, feature_clean,
                     avg_rank, friedman_p, friedman_significant, T, M].
                     Includes "standard" as a special n_per_class value.
                     T reflects datasets, not tasks, when dataset_col is given.
        pairwise_df: DataFrame with all pairwise results, with metric and
                     n_per_class columns added.
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    n_values = sorted(df_lowdata["n_per_class"].dropna().unique())
    all_ranks = []
    all_pairwise = []

    for metric in metrics:
        print(f"\n{'#'*50}\nMETRIC: {metric}")

        for n in n_values:
            df_n = df_lowdata[df_lowdata["n_per_class"] == n].copy()
            print(f"\n{'='*50}\nn_per_class = {n}")
            try:
                score_matrix = prepare_score_matrix(
                    df_n, task_col=task_col, feature_col=feature_col,
                    metric=metric, exclude_features=exclude_features,
                    dataset_col=dataset_col,
                )
                res = run_friedman(score_matrix, alpha=alpha)
                for feat, rank in res["avg_ranks"].items():
                    all_ranks.append({
                        "metric": metric, "n_per_class": n,
                        feature_col: feat, "avg_rank": rank,
                        "friedman_p": res["iman_davenport_p"],
                        "friedman_significant": res["significant"],
                        "T": res["T"], "M": res["M"],
                    })
                if res["significant"]:
                    pw = pairwise_wilcoxon_holm(score_matrix, alpha=alpha)
                    pw["n_per_class"] = n
                    pw["metric"] = metric
                    all_pairwise.append(pw)
            except Exception as e:
                print(f"  ERROR at n={n}: {e}")
                continue

        print(f"\n{'='*50}\nStandard mode (full data)")
        try:
            score_matrix_std = prepare_score_matrix(
                df_standard, task_col=task_col, feature_col=feature_col,
                metric=metric, exclude_features=exclude_features,
                dataset_col=dataset_col,
            )
            res_std = run_friedman(score_matrix_std, alpha=alpha)
            for feat, rank in res_std["avg_ranks"].items():
                all_ranks.append({
                    "metric": metric, "n_per_class": "standard",
                    feature_col: feat, "avg_rank": rank,
                    "friedman_p": res_std["iman_davenport_p"],
                    "friedman_significant": res_std["significant"],
                    "T": res_std["T"], "M": res_std["M"],
                })
            if res_std["significant"]:
                pw_std = pairwise_wilcoxon_holm(score_matrix_std, alpha=alpha)
                pw_std["n_per_class"] = "standard"
                pw_std["metric"] = metric
                all_pairwise.append(pw_std)
        except Exception as e:
            print(f"  ERROR for standard: {e}")

    ranks_df = pd.DataFrame(all_ranks)
    pairwise_df = pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame()
    return ranks_df, pairwise_df


def plot_lowdata_ranks_with_standard(
    ranks_df: pd.DataFrame,
    feature_order: list[str],
    palette_dict: dict[str, str],
    metrics: list[str] | str = "MCC",
    feature_col: str = "feature_clean",
    full_data_label: int = 2048,
    figsize_per_row: tuple[float, float] = (14, 5),
    title_prefix: str | None = None,
    save_path: str | None = None,
    xticks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
) -> plt.Figure:
    """
    Mirrors plot_lowdata_with_standard_zscore but plots average Friedman rank.
    One row of panels per metric (MCC on top, F1_macro below).

    Unchanged by the clustering correction: consumes ranks_df exactly as
    produced by run_friedman_per_nperclass, regardless of whether that was
    run task-level or dataset-collapsed.

    Args:
        ranks_df:         Output of run_friedman_per_nperclass. Must have 'metric' column.
        feature_order:    Feature order for legend/colors.
        palette_dict:     Feature -> color dict.
        metrics:          Metric or list of metrics to plot (default: "MCC").
        feature_col:      Feature column name (default: feature_clean).
        full_data_label:  x-position for the standard panel (default: 2048).
        figsize_per_row:  Size per metric row (width, height).
        save_path:        If provided, saves figure here.

    Returns:
        Matplotlib figure.
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    n_metrics = len(metrics)
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(figsize_per_row[0], figsize_per_row[1] * n_metrics))

    gs = GridSpec(
        n_metrics, 2,
        figure=fig,
        width_ratios=[3, 0.4],
        wspace=0.05,
        hspace=0.3,
    )

    legend_handles = None
    legend_labels  = None

    for row_idx, metric in enumerate(metrics):
        df_metric = ranks_df[ranks_df["metric"] == metric]
        low_ranks = df_metric[df_metric["n_per_class"] != "standard"].copy()
        low_ranks["n_per_class"] = low_ranks["n_per_class"].astype(float)
        std_ranks = df_metric[df_metric["n_per_class"] == "standard"].copy()

        ax1 = fig.add_subplot(gs[row_idx, 0])
        ax2 = fig.add_subplot(gs[row_idx, 1], sharey=ax1)

        # Left panel: rank trajectory
        for feat in feature_order:
            d = low_ranks[low_ranks[feature_col] == feat].sort_values("n_per_class")
            if d.empty:
                continue
            color = palette_dict.get(feat, "#555555")
            ax1.plot(d["n_per_class"], d["avg_rank"],
                     color=color, marker="o", markersize=4,
                     linewidth=1.5, label=feat)

        ax1.set_xscale("log", base=2)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticks, fontsize=15)
        ax1.set_xlabel("Training samples per class" if row_idx == n_metrics - 1 else "", fontsize=16)
        ax1.tick_params(axis="y", labelsize=15)
        ax1.set_ylabel(f"{metric}\nAverage rank (lower = better)", fontsize=14)
        ax1.invert_yaxis()
        #ax1.axhline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax1.grid(False)
        ax1.spines["right"].set_visible(False)

        if legend_handles is None:
            legend_handles, legend_labels = ax1.get_legend_handles_labels()

        # Right panel: standard mode
        for feat in feature_order:
            row = std_ranks[std_ranks[feature_col] == feat]
            if row.empty:
                continue
            color = palette_dict.get(feat, "#555555")
            ax2.plot(full_data_label, row["avg_rank"].iloc[0],
                     marker="_", markersize=30,
                     color=color, linewidth=2)

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

        # Broken axis markers
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

    fig.suptitle(
        f"{title_prefix} Low-data and full-data Friedman average ranks",
        y=1.02, fontsize=19,
    )

    fig.subplots_adjust(right=0.82, wspace=0.05, hspace=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Combined: Friedman ranks barplot (Panel A) + delta violin (Panel B)
# Replaces plot_lowdata_n4_zscore_and_delta with Friedman ranking in Panel A
# ---------------------------------------------------------------------------

def plot_lowdata_n_friedman_and_delta(
    df_lowdata: pd.DataFrame,
    plot_overall_performance_bar,        # pass function from evaluation.py
    feature_order: list[str],
    color_map: dict[str, str],
    n_value: int = 4,
    metrics: tuple[str, str] = ("MCC", "F1_macro"),
    task_col: str = "task_id",
    feature_col: str = "feature_clean",
    model_a: str = "TF-SAPIENS",
    model_b: str = "Raw Log-norm",
    exclude_features: list[str] = ("random_proj50",),
    alpha: float = 0.05,
    errorbar: str = "std",
    figsize: tuple[float, float] = (12, 6),
    save_path: str | None = None,
    dataset_col: str | None = None,
) -> tuple:
    """
    Two-panel figure at a single n_per_class value:

    Panel A (left):  Friedman average ranks barplot for two metrics,
                     using plot_friedman_as_bar (same style as overall analysis).
    Panel B (right): Violin of per-task delta = model_a - model_b (MCC).

    Args:
        df_lowdata:                  Low-data results DataFrame.
        plot_overall_performance_bar: Function from evaluation.py.
        feature_order:               Feature order for y-axis.
        color_map:                   Feature -> color dict.
        n_value:                     n_per_class to subset to (default: 4).
        metrics:                     Two metrics for Panel A (default: MCC, F1_macro).
        task_col:                    Task identifier column.
        feature_col:                 Feature column.
        model_a:                     FM for delta (default: TF-SAPIENS).
        model_b:                     Reference for delta (default: Raw Log-norm).
        exclude_features:            Features to exclude from Friedman.
        alpha:                       Significance level for Friedman.
        errorbar:                    Error bar type for Panel A ('std','sem','ci95',None).
        figsize:                     Figure size.
        save_path:                   If provided, saves figure here.
        dataset_col:                 If provided, Panel A's Friedman ranks are
                                     computed on dataset-collapsed scores (see
                                     prepare_score_matrix). Panel B's delta is
                                     unaffected -- it's a plain per-task paired
                                     difference, not a significance test.

    Returns:
        fig, axes, friedman_summary, delta_pivot
    """
    import matplotlib.gridspec as mgridspec
    import seaborn as sns

    # --- Subset to n_per_class ---
    df_n = df_lowdata[df_lowdata["n_per_class"] == n_value].copy()
    print(f"Subsetting to n_per_class={n_value}: {len(df_n)} rows")

    # --- Panel A: Friedman ranks via plot_friedman_as_bar ---
    # plot_friedman_as_bar creates its own figure internally, so we extract
    # the summary and re-plot manually into our axes to keep both panels together

    # Run Friedman for each metric to get avg_ranks
    rank_series = {}
    cd_values = {}
    results_per_metric = {}

    for metric in metrics:
        score_matrix = prepare_score_matrix(
            df_n, task_col=task_col, feature_col=feature_col,
            metric=metric, exclude_features=exclude_features,
            dataset_col=dataset_col,
        )
        res = run_friedman(score_matrix, alpha=alpha)
        rank_series[metric] = res["avg_ranks"]
        results_per_metric[metric] = res
        if res["significant"]:
            try:
                cd_values[metric] = nemenyi_cd(M=res["M"], T=res["T"], alpha=alpha)
            except ValueError:
                cd_values[metric] = None
        else:
            cd_values[metric] = None

    # Build fake long-format df with per-task ranks for plot_overall_performance_bar
    fake_rows = []
    for metric in metrics:
        col = f"{metric}_rank"
        rank_matrix = results_per_metric[metric]["ranks_matrix"]
        for feat in rank_matrix.columns:
            for task_id, rank_val in rank_matrix[feat].items():
                fake_rows.append({
                    "task_id": task_id,
                    feature_col: feat,
                    col: rank_val,
                })
    fake_df = pd.DataFrame(fake_rows)

    # --- Delta pivot for Panel B ---
    # NOTE: this stays task-level regardless of dataset_col -- it's a
    # descriptive per-task paired difference for the violin, not a
    # significance test, so there's no clustering issue to correct here.
    task_feature = (
        df_n.groupby([task_col, feature_col])["MCC"]
        .mean()
        .reset_index()
    )
    pivot = task_feature.pivot(
        index=task_col, columns=feature_col, values="MCC"
    )
    if model_a not in pivot.columns or model_b not in pivot.columns:
        raise ValueError(f"'{model_a}' or '{model_b}' not found in features.")
    pivot = pivot[[model_a, model_b]].dropna()
    pivot["delta"] = pivot[model_a] - pivot[model_b]

    # --- Build combined figure ---
    fig = plt.figure(figsize=figsize)
    gs = mgridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.8, 1])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # Draw Panel A using plot_overall_performance_bar into ax_a
    fig_tmp, ax_tmp, friedman_summary = plot_overall_performance_bar(
        fake_df,
        metrics=[f"{m}_rank" for m in metrics],
        feature_order=feature_order,
        color_map=color_map,
        figsize=figsize,
        title=f"A. Friedman ranks at n={n_value} (lower = better)",
        xlabel="Average rank (lower = better)",
        bar_height=0.36,
        show_values=True,
        errorbar=errorbar,
    )
    # Copy artists from tmp figure into ax_a
    for artist in fig_tmp.axes[0].get_children():
        try:
            ax_a.add_artist(artist)
        except Exception:
            pass
    # Better: just re-call the bar drawing directly
    plt.close(fig_tmp)

    # Re-draw Panel A manually (cleanest approach)
    ax_a.clear()
    rank_cols = [f"{m}_rank" for m in metrics]
    summary = (
        fake_df.groupby(feature_col)[rank_cols]
        .agg(["mean", "std", "count"])
        .reindex(feature_order)
    )
    y = np.arange(len(summary))
    bar_height = 0.36
    colors = [color_map.get(f, "#999999") for f in summary.index]

    for (metric_col, offset, hatch, alph), label in zip(
        [(rank_cols[0], +bar_height/2, None, 1.0),
         (rank_cols[1], -bar_height/2, "//", 0.65)],
        [metrics[0], metrics[1]],
    ):
        mean = summary[(metric_col, "mean")]
        if errorbar == "std":
            err = summary[(metric_col, "std")]
        elif errorbar == "sem":
            err = summary[(metric_col, "std")] / np.sqrt(summary[(metric_col, "count")])
        elif errorbar == "ci95":
            err = 1.96 * summary[(metric_col, "std")] / np.sqrt(summary[(metric_col, "count")])
        else:
            err = None

        ax_a.barh(y + offset, mean, xerr=err, height=bar_height,
                  color=colors, edgecolor="black", linewidth=0.8,
                  hatch=hatch, alpha=alph, capsize=3, label=label)

        if True:  # show values
            for i, value in enumerate(mean):
                if pd.isna(value):
                    continue
                ax_a.text(value - 0.05, y[i] + offset + 0.12,
                          f"{value:.2f}", va="bottom", ha="right", fontsize=8)

    ax_a.invert_xaxis()  # rank 1 (best) on right
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(list(summary.index))
    ax_a.set_xlabel("Average rank (lower = better)")
    ax_a.set_ylabel("Model / Representation")
    ax_a.set_title(f"A. Friedman ranks at n={n_value} (lower = better)")
    ax_a.legend(frameon=True)

    # Annotate CD
    for i, metric in enumerate(metrics):
        cd = cd_values.get(metric)
        label = f"CD({metric})={cd:.3f}" if cd is not None else f"CD({metric}): n.s."
        ax_a.annotate(label, xy=(0.02 + i*0.35, 0.02),
                      xycoords="axes fraction", fontsize=8, color="gray")

    # --- Panel B: violin ---
    mean_delta   = pivot["delta"].mean()
    median_delta = pivot["delta"].median()
    win_rate     = (pivot["delta"] > 0).mean() * 100

    sns.violinplot(data=pivot, y="delta", inner="box",
                   color="lightsteelblue", cut=0, linewidth=1, ax=ax_b)
    ax_b.axhline(0, color="red", linestyle="--", linewidth=1)
    ax_b.scatter(0, mean_delta, color="red", marker="D", s=45, zorder=10)
    ax_b.text(0.98, 0.98,
              f"{model_a} wins: {win_rate:.1f}%\n"
              f"Mean \u0394={mean_delta:.3f}\n"
              f"Median \u0394={median_delta:.3f}",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(facecolor="white", alpha=0.9))
    ax_b.set_xticks([])
    ax_b.set_xlabel("")
    ax_b.set_ylabel(f"$\\Delta$MCC ({model_a} $-$ {model_b})")
    ax_b.set_title("B. Paired difference")

    sns.despine()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, (ax_a, ax_b), friedman_summary, pivot