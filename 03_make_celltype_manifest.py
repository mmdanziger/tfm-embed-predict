#!/usr/bin/env python3
"""
Generate cell type task manifest from disease task manifest.

Transforms:
    (dataset_id, cell_type) → predict disease
Into:
    dataset_id → predict cell_type

IMPORTANT: Cell types are taken EXACTLY from the disease manifest to ensure
the same cells are used in both benchmarks. This enables direct comparison:
"On these exact cells, how do methods perform for disease vs cell type prediction?"

Selection criteria (dataset-level only):
- At least min_cell_types distinct cell types per dataset (default 2)
- At least min_donors donors in the dataset (default 3)
- At least min_cells total cells (default 500)

No additional per-cell-type filtering is applied here - that was already done
in 01_prepare_task_manifest.py via min_cells_per_disease.
"""

import argparse

import pandas as pd


def make_celltype_manifest(
    disease_manifest_path: str,
    out_path: str,
    min_cell_types: int = 2,
    min_donors: int = 3,
    min_cells: int = 500,
) -> None:
    """
    Distills disease task manifest into cell type task manifest.

    Preserves EXACT cell types from disease manifest - no additional filtering
    at the cell type level. This ensures both benchmarks use identical cells.

    Args:
        disease_manifest_path: Path to disease task manifest parquet
        out_path: Path to save cell type manifest parquet
        min_cell_types: Minimum cell types per dataset (default 2)
        min_donors: Minimum donors per dataset (default 3)
        min_cells: Minimum total cells per dataset (default 500)

    """
    print(f"Loading disease manifest: {disease_manifest_path}")
    print(
        f"Dataset-level filters: min_cell_types={min_cell_types}, "
        f"min_donors={min_donors}, min_cells={min_cells}"
    )
    print(
        "\nNOTE: Cell types are preserved exactly from disease manifest (no additional filtering)"
    )

    # Load disease task manifest
    disease_tasks = pd.read_parquet(disease_manifest_path)

    print("\nDisease manifest stats:")
    print(f"  Tasks (dataset, cell_type pairs): {len(disease_tasks):,}")
    print(f"  Unique datasets: {disease_tasks['dataset_id'].nunique():,}")
    print(f"  Unique cell types: {disease_tasks['cell_type'].nunique():,}")

    # ==========================================================================
    # Aggregate to dataset level, preserving ALL cell types from disease manifest
    # ==========================================================================

    # Get per-cell-type stats (no filtering - preserve exactly)
    celltype_stats = (
        disease_tasks.groupby(["dataset_id", "cell_type"])
        .agg(
            n_cells_celltype=("n_cells", "first"),
        )
        .reset_index()
    )

    # Aggregate to dataset level
    celltypes_per_dataset = (
        celltype_stats.groupby("dataset_id")
        .agg(
            n_cell_types=("cell_type", "nunique"),
            cell_types=("cell_type", lambda x: sorted(x.unique())),
            total_cells=("n_cells_celltype", "sum"),
            min_cells_per_celltype=("n_cells_celltype", "min"),
        )
        .reset_index()
    )

    # Get donor counts from disease manifest
    # Use max to handle any inconsistencies (though should be same within dataset)
    donor_counts = disease_tasks.groupby("dataset_id")["n_donors"].max().reset_index()

    # Get disease info for reference
    disease_info = (
        disease_tasks.groupby("dataset_id")
        .agg(
            diseases=(
                "diseases",
                lambda x: sorted({d for sublist in x for d in sublist}),
            ),
            n_diseases=("n_diseases", "max"),
        )
        .reset_index()
    )

    # Merge all info
    celltype_tasks = celltypes_per_dataset.merge(
        donor_counts, on="dataset_id", how="left"
    ).merge(disease_info, on="dataset_id", how="left")

    # Rename for clarity
    celltype_tasks = celltype_tasks.rename(
        columns={
            "cell_types": "cell_type",  # List of cell types
            "total_cells": "n_cells",
        }
    )

    print(f"\nBefore filtering: {len(celltype_tasks):,} datasets")

    # ==========================================================================
    # Apply dataset-level filters only
    # ==========================================================================

    n_before = len(celltype_tasks)

    # Filter: minimum cell types (need at least 2 for classification)
    celltype_tasks = celltype_tasks[celltype_tasks["n_cell_types"] >= min_cell_types]
    print(
        f"  After min_cell_types >= {min_cell_types}: {len(celltype_tasks):,} datasets "
        f"(removed {n_before - len(celltype_tasks)})"
    )
    n_before = len(celltype_tasks)

    # Filter: minimum donors (need enough for CV)
    celltype_tasks = celltype_tasks[celltype_tasks["n_donors"] >= min_donors]
    print(
        f"  After min_donors >= {min_donors}: {len(celltype_tasks):,} datasets "
        f"(removed {n_before - len(celltype_tasks)})"
    )
    n_before = len(celltype_tasks)

    # Filter: minimum total cells
    celltype_tasks = celltype_tasks[celltype_tasks["n_cells"] >= min_cells]
    print(
        f"  After min_cells >= {min_cells}: {len(celltype_tasks):,} datasets "
        f"(removed {n_before - len(celltype_tasks)})"
    )

    if len(celltype_tasks) == 0:
        print("\nWARNING: No datasets passed all filters!")
        celltype_tasks.to_parquet(out_path, index=False)
        return

    # ==========================================================================
    # Add derived columns and sort
    # ==========================================================================

    celltype_tasks["n_cells_per_donor"] = (
        (celltype_tasks["n_cells"] / celltype_tasks["n_donors"]).round().astype(int)
    )

    # Sort by task "quality" - more cell types, more donors, more cells
    celltype_tasks = celltype_tasks.sort_values(
        ["n_cell_types", "n_donors", "n_cells"], ascending=[False, False, False]
    ).reset_index(drop=True)

    # Reorder columns for clarity
    column_order = [
        "dataset_id",
        "cell_type",
        "n_cell_types",
        "n_cells",
        "n_donors",
        "min_cells_per_celltype",
        "n_cells_per_donor",
        "n_diseases",
        "diseases",
    ]
    celltype_tasks = celltype_tasks[
        [c for c in column_order if c in celltype_tasks.columns]
    ]

    # Save manifest
    celltype_tasks.to_parquet(out_path, index=False)

    print(f"\n{'=' * 60}")
    print("Cell Type Manifest Summary")
    print(f"{'=' * 60}")
    print(f"  Total datasets: {len(celltype_tasks):,}")
    print(
        f"  Total cell types across all datasets: {celltype_tasks['n_cell_types'].sum():,}"
    )
    print(f"  Total cells: {celltype_tasks['n_cells'].sum():,}")
    print(
        f"\n  Cell types per dataset: "
        f"min={celltype_tasks['n_cell_types'].min()}, "
        f"median={celltype_tasks['n_cell_types'].median():.0f}, "
        f"max={celltype_tasks['n_cell_types'].max()}"
    )
    print(
        f"  Donors per dataset: "
        f"min={celltype_tasks['n_donors'].min()}, "
        f"median={celltype_tasks['n_donors'].median():.0f}, "
        f"max={celltype_tasks['n_donors'].max()}"
    )
    print(
        f"  Cells per dataset: "
        f"min={celltype_tasks['n_cells'].min():,}, "
        f"median={celltype_tasks['n_cells'].median():,.0f}, "
        f"max={celltype_tasks['n_cells'].max():,}"
    )
    print(
        f"  Min cells per cell type: "
        f"min={celltype_tasks['min_cells_per_celltype'].min():,}, "
        f"median={celltype_tasks['min_cells_per_celltype'].median():,.0f}, "
        f"max={celltype_tasks['min_cells_per_celltype'].max():,}"
    )
    print(f"\nSaved to: {out_path}")

    # Preview top tasks
    print("\nTop 10 datasets (by cell type diversity):")
    preview_cols = [
        "dataset_id",
        "n_cell_types",
        "n_donors",
        "n_cells",
        "min_cells_per_celltype",
    ]
    print(celltype_tasks[preview_cols].head(10).to_string(index=False))

    # Show cell type distribution
    print("\nCell type count distribution:")
    ct_counts = celltype_tasks["n_cell_types"].value_counts().sort_index()
    for n_ct, count in ct_counts.items():
        if count > 0:
            print(f"  {n_ct} cell types: {count} datasets")


def main():
    ap = argparse.ArgumentParser(
        description="Generate cell type task manifest from disease task manifest"
    )
    ap.add_argument(
        "--disease_manifest",
        required=True,
        help="Path to disease task manifest parquet",
    )
    ap.add_argument("--out_path", required=True, help="Output cell type manifest path")
    ap.add_argument(
        "--min_cell_types",
        type=int,
        default=2,
        help="Minimum cell types per dataset (default 2)",
    )
    ap.add_argument(
        "--min_donors",
        type=int,
        default=3,
        help="Minimum donors per dataset (default 3)",
    )
    ap.add_argument(
        "--min_cells",
        type=int,
        default=500,
        help="Minimum total cells per dataset (default 500)",
    )
    args = ap.parse_args()

    make_celltype_manifest(
        args.disease_manifest,
        args.out_path,
        args.min_cell_types,
        args.min_donors,
        args.min_cells,
    )


if __name__ == "__main__":
    main()
