#!/usr/bin/env python3
"""
Generate task manifest from Census metadata for disease prediction benchmark.

A task is defined as (dataset_id, cell_type) → predict disease label.

Inclusion criteria:
- At least 2 distinct disease labels
- At least min_donors_per_stratum donors per disease stratum (default 3)
- At least min_cells total cells in the (dataset_id, cell_type) group (default 500)
- At least min_cells_per_disease cells per disease label (default 50)

These thresholds ensure:
1. Enough statistical power for meaningful classification
2. Cross-validation folds have adequate samples
3. Minority classes aren't too small to learn from
"""

import argparse

import cellxgene_census
import pandas as pd


def make_task_manifest(
    census_uri: str | None,
    census_version: str,
    out_path: str,
    organism: str = "homo_sapiens",
    min_donors_per_stratum: int = 3,
    min_cells: int = 500,
    min_cells_per_disease: int = 50,
) -> None:
    """
    Generates a parquet manifest of eligible classification tasks from Census metadata.

    Args:
        census_uri: Local Census URI or None for default S3 access
        census_version: Census version string
        out_path: Path to save the output parquet manifest
        organism: Organism to query
        min_donors_per_stratum: Minimum donors required per disease stratum (default 3)
        min_cells: Minimum total cells in (dataset_id, cell_type) group (default 500)
        min_cells_per_disease: Minimum cells per disease label (default 50)

    """
    print(f"Opening Census {census_version}...")
    print(
        f"Filters: min_donors_per_stratum={min_donors_per_stratum}, "
        f"min_cells={min_cells}, min_cells_per_disease={min_cells_per_disease}"
    )

    with cellxgene_census.open_soma(
        uri=census_uri, census_version=census_version
    ) as census:
        # Read obs metadata for primary data only
        obs_df = (
            census["census_data"][organism]
            .obs.read(
                value_filter="is_primary_data == True",
                column_names=["dataset_id", "cell_type", "disease", "donor_id"],
            )
            .concat()
            .to_pandas()
        )

    print(f"Loaded {len(obs_df):,} cells from Census")

    # Convert to string to avoid categorical issues
    for col in ["dataset_id", "cell_type", "disease", "donor_id"]:
        obs_df[col] = obs_df[col].astype(str)

    group_cols = ["dataset_id", "cell_type"]

    # ==========================================================================
    # Filter 1: Minimum cells per (dataset_id, cell_type) group
    # ==========================================================================
    cell_counts = (
        obs_df.groupby(group_cols, observed=True).size().rename("n_cells").reset_index()
    )

    groups_enough_cells = cell_counts[cell_counts["n_cells"] >= min_cells][group_cols]
    print(f"  Groups with >= {min_cells} cells: {len(groups_enough_cells):,}")

    obs_df = obs_df.merge(groups_enough_cells, on=group_cols, how="inner")

    # ==========================================================================
    # Filter 2: Minimum cells per disease within each group
    # ==========================================================================
    cells_per_disease = (
        obs_df.groupby(group_cols + ["disease"], observed=True)
        .size()
        .rename("n_cells_disease")
        .reset_index()
    )

    # Keep only diseases with enough cells
    cells_per_disease = cells_per_disease[
        cells_per_disease["n_cells_disease"] >= min_cells_per_disease
    ]

    # Keep only groups where at least 2 diseases remain after filtering
    diseases_per_group = (
        cells_per_disease.groupby(group_cols, observed=True)["disease"]
        .nunique()
        .rename("n_diseases")
        .reset_index()
    )
    groups_enough_diseases = diseases_per_group[diseases_per_group["n_diseases"] >= 2][
        group_cols
    ]
    print(
        f"  Groups with >= 2 diseases (each with >= {min_cells_per_disease} cells): {len(groups_enough_diseases):,}"
    )

    # Filter obs_df to only include valid (group, disease) combinations
    valid_group_diseases = cells_per_disease.merge(
        groups_enough_diseases, on=group_cols, how="inner"
    )
    obs_df = obs_df.merge(
        valid_group_diseases[group_cols + ["disease"]],
        on=group_cols + ["disease"],
        how="inner",
    )

    # ==========================================================================
    # Filter 3: Minimum donors per disease within each group
    # ==========================================================================
    donors_per_disease = (
        obs_df.groupby(group_cols + ["disease"], observed=True)["donor_id"]
        .nunique()
        .rename("n_donors")
        .reset_index()
    )

    # Keep only groups where EVERY disease has enough donors
    min_donors_per_group = (
        donors_per_disease.groupby(group_cols, observed=True)["n_donors"]
        .min()
        .rename("min_donors_per_stratum")
        .reset_index()
    )

    eligible_groups = min_donors_per_group[
        min_donors_per_group["min_donors_per_stratum"] >= min_donors_per_stratum
    ]
    print(
        f"  Groups with >= {min_donors_per_stratum} donors per disease stratum: {len(eligible_groups):,}"
    )

    # ==========================================================================
    # Compile final manifest
    # ==========================================================================
    # Filter to eligible groups
    obs_df = obs_df.merge(eligible_groups[group_cols], on=group_cols, how="inner")

    # Recompute disease counts after all filtering
    disease_counts = (
        obs_df.groupby(group_cols, observed=True)["disease"]
        .nunique()
        .rename("n_diseases")
        .reset_index()
    )

    # Aggregate task metadata
    task_manifest = (
        obs_df.groupby(group_cols, observed=True)
        .agg(
            diseases=("disease", lambda s: sorted(pd.unique(s.astype(str)))),
            n_cells=("donor_id", "size"),
            n_donors=("donor_id", "nunique"),
        )
        .reset_index()
        .merge(eligible_groups, on=group_cols, how="left")
        .merge(disease_counts, on=group_cols, how="left")
    )

    # Add cells per disease stats
    cells_per_disease_final = (
        obs_df.groupby(group_cols + ["disease"], observed=True)
        .size()
        .reset_index(name="n_cells_disease")
    )

    min_cells_disease = (
        cells_per_disease_final.groupby(group_cols, observed=True)["n_cells_disease"]
        .min()
        .rename("min_cells_per_disease")
        .reset_index()
    )

    task_manifest = task_manifest.merge(min_cells_disease, on=group_cols, how="left")

    # Sort by task "quality" - more diseases, more donors, more cells
    task_manifest = task_manifest.sort_values(
        ["n_diseases", "min_donors_per_stratum", "n_cells"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    task_manifest.to_parquet(out_path, index=False)

    print(f"\n{'=' * 60}")
    print("Manifest Summary")
    print(f"{'=' * 60}")
    print(f"  Total tasks: {len(task_manifest):,}")
    print(f"  Unique datasets: {task_manifest['dataset_id'].nunique():,}")
    print(f"  Unique cell types: {task_manifest['cell_type'].nunique():,}")
    print(f"  Total cells: {task_manifest['n_cells'].sum():,}")
    print(
        f"\n  Diseases per task: "
        f"min={task_manifest['n_diseases'].min()}, "
        f"median={task_manifest['n_diseases'].median():.0f}, "
        f"max={task_manifest['n_diseases'].max()}"
    )
    print(
        f"  Donors per task: "
        f"min={task_manifest['n_donors'].min()}, "
        f"median={task_manifest['n_donors'].median():.0f}, "
        f"max={task_manifest['n_donors'].max()}"
    )
    print(
        f"  Cells per task: "
        f"min={task_manifest['n_cells'].min():,}, "
        f"median={task_manifest['n_cells'].median():,.0f}, "
        f"max={task_manifest['n_cells'].max():,}"
    )
    print(
        f"  Min cells per disease: "
        f"min={task_manifest['min_cells_per_disease'].min():,}, "
        f"median={task_manifest['min_cells_per_disease'].median():,.0f}, "
        f"max={task_manifest['min_cells_per_disease'].max():,}"
    )
    print(f"\nWrote manifest to: {out_path}")

    # Preview top tasks
    print("\nTop 10 tasks (by disease diversity):")
    preview_cols = [
        "dataset_id",
        "cell_type",
        "n_diseases",
        "n_donors",
        "n_cells",
        "min_cells_per_disease",
    ]
    print(task_manifest[preview_cols].head(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description="Generate task manifest from Census metadata"
    )
    ap.add_argument("--census_uri", default=None, help="Local Census URI (None for S3)")
    ap.add_argument("--census_version", default="2025-01-30", help="Census version")
    ap.add_argument("--out_path", required=True, help="Output manifest parquet path")
    ap.add_argument("--organism", default="homo_sapiens", help="Organism")
    ap.add_argument(
        "--min_donors_per_stratum",
        type=int,
        default=3,
        help="Minimum donors per disease stratum for CV (default 3)",
    )
    ap.add_argument(
        "--min_cells",
        type=int,
        default=500,
        help="Minimum cells per (dataset, cell_type) task (default 500)",
    )
    ap.add_argument(
        "--min_cells_per_disease",
        type=int,
        default=50,
        help="Minimum cells per disease label (default 50)",
    )
    args = ap.parse_args()

    make_task_manifest(
        args.census_uri,
        args.census_version,
        args.out_path,
        args.organism,
        args.min_donors_per_stratum,
        args.min_cells,
        args.min_cells_per_disease,
    )


if __name__ == "__main__":
    main()
