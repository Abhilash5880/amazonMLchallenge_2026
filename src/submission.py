"""
Submission construction and validation for Amazon ML Challenge 2026.

Responsibilities
----------------
1. Aggregate positive inference predictions.
2. Build matching_results.tsv.
3. Build candidate_pairs.tsv from the final candidate set.
4. Validate the structural submission contract.

The module deliberately does NOT:
    - generate candidates
    - calculate model features
    - run CatBoost
    - select a threshold
"""

from pathlib import Path

import polars as pl


# ============================================================
# 1. Prediction aggregation
# ============================================================

def aggregate_prediction_partition(
    inference_partition_dir,
    output_path,
):
    """
    Aggregate positive prediction batches belonging to one
    candidate partition.

    Expected input columns:

        source1_entity_id
        match_id
        probability

    Output columns:

        source1_entity_id
        matched_entity_ids

    Duplicate (source1_entity_id, match_id) pairs are removed.
    Match IDs are sorted before being joined into the final
    comma-separated representation.
    """

    inference_partition_dir = Path(
        inference_partition_dir
    )

    output_path = Path(output_path)

    batch_files = sorted(
        inference_partition_dir.glob(
            "batch_*.parquet"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # No predictions in this partition
    # --------------------------------------------------------

    if not batch_files:

        empty = pl.DataFrame({
            "source1_entity_id": pl.Series(
                [],
                dtype=pl.String,
            ),
            "matched_entity_ids": pl.Series(
                [],
                dtype=pl.String,
            ),
        })

        empty.write_parquet(
            output_path,
            compression="zstd",
        )

        return empty

    # --------------------------------------------------------
    # Read positive prediction batches lazily
    # --------------------------------------------------------

    positives = pl.scan_parquet(
        [str(path) for path in batch_files]
    )

    # --------------------------------------------------------
    # Deduplicate + aggregate
    # --------------------------------------------------------

    aggregated = (
        positives
        .select([
            "source1_entity_id",
            "match_id",
        ])
        .unique()
        .group_by(
            "source1_entity_id"
        )
        .agg(
            pl.col("match_id")
            .sort()
            .str.join(",")
            .alias("matched_entity_ids")
        )
        .collect(
            engine="streaming"
        )
    )

    aggregated.write_parquet(
        output_path,
        compression="zstd",
    )

    return aggregated


def aggregate_all_predictions(
    inference_dir,
    aggregate_dir,
    n_partitions,
):
    """
    Aggregate prediction batches partition-by-partition.

    Every candidate partition receives an aggregate parquet,
    including partitions containing zero positive predictions.

    Returns
    -------
    list[Path]
        Paths to the aggregate partition files.
    """

    inference_dir = Path(
        inference_dir
    )

    aggregate_dir = Path(
        aggregate_dir
    )

    aggregate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_paths = []

    for part_idx in range(
        n_partitions
    ):

        inference_partition_dir = (
            inference_dir
            / f"part_{part_idx:03d}"
        )

        output_path = (
            aggregate_dir
            / f"part_{part_idx:03d}.parquet"
        )

        aggregated = aggregate_prediction_partition(
            inference_partition_dir,
            output_path,
        )

        print(
            f"[{part_idx + 1:02d}/{n_partitions}] "
            f"S1 with predictions="
            f"{aggregated.height:,}"
        )

        aggregate_paths.append(
            output_path
        )

    print(
        "✓ Partition-level aggregation complete."
    )

    return aggregate_paths


# ============================================================
# 2. Complete Source 1 population
# ============================================================

def load_complete_s1_population(
    s1_test_path,
):
    """
    Load the complete Source 1 population.

    The final submission must start from this population so
    entities with no predicted matches are retained.
    """

    return (
        pl.scan_csv(
            s1_test_path,
            separator="\t",
        )
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id")
        )
    )


# ============================================================
# 3. matching_results.tsv
# ============================================================

def build_matching_results(
    s1_test_path,
    aggregate_paths,
):
    """
    Build the final matching_results DataFrame.

    Construction:

        complete S1 population
                    ↓
        LEFT JOIN aggregated predictions
                    ↓
        fill missing matches with ""
                    ↓
        final two columns

    Returns
    -------
    polars.DataFrame
    """

    complete_s1 = load_complete_s1_population(
        s1_test_path
    )

    aggregated_files = sorted(
        Path(path)
        for path in aggregate_paths
    )

    if not aggregated_files:
        raise RuntimeError(
            "No aggregate prediction partitions found."
        )

    all_predicted = pl.scan_parquet(
        [str(path) for path in aggregated_files]
    )

    matching_results = (
        complete_s1
        .join(
            all_predicted,
            on="source1_entity_id",
            how="left",
        )
        .with_columns(
            pl.col(
                "matched_entity_ids"
            ).fill_null("")
        )
        .select([
            "source1_entity_id",
            "matched_entity_ids",
        ])
        .collect(
            engine="streaming"
        )
    )

    return matching_results


def validate_matching_results(
    matching_results,
    s1_test_path,
):
    """
    Validate the structural requirements of
    matching_results.tsv.
    """

    expected_s1_count = (
        pl.scan_csv(
            s1_test_path,
            separator="\t",
        )
        .select(
            pl.col("entity_id").n_unique()
        )
        .collect(
            engine="streaming"
        )
        .item()
    )

    actual_rows = (
        matching_results.height
    )

    unique_s1_count = (
        matching_results[
            "source1_entity_id"
        ].n_unique()
    )

    # --------------------------------------------------------
    # Column contract
    # --------------------------------------------------------

    expected_columns = [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    if matching_results.columns != expected_columns:
        raise RuntimeError(
            "Unexpected matching_results columns: "
            f"{matching_results.columns}"
        )

    # --------------------------------------------------------
    # Row / ID contract
    # --------------------------------------------------------

    if actual_rows != expected_s1_count:
        raise RuntimeError(
            "Output row count mismatch: "
            f"expected {expected_s1_count:,}, "
            f"got {actual_rows:,}"
        )

    if unique_s1_count != expected_s1_count:
        raise RuntimeError(
            "Duplicate or missing Source 1 entity IDs detected."
        )

    if (
        matching_results[
            "source1_entity_id"
        ].null_count()
        != 0
    ):
        raise RuntimeError(
            "Null Source 1 IDs found."
        )

    if (
        matching_results[
            "matched_entity_ids"
        ].null_count()
        != 0
    ):
        raise RuntimeError(
            "Null matched_entity_ids found."
        )

    # --------------------------------------------------------
    # Duplicate matched IDs inside each row
    # --------------------------------------------------------

    for row in matching_results.iter_rows(
        named=True
    ):

        value = row[
            "matched_entity_ids"
        ]

        if not value:
            continue

        ids = value.split(",")

        if len(ids) != len(set(ids)):
            raise RuntimeError(
                "Duplicate matched entity IDs detected "
                f"for S1={row['source1_entity_id']}"
            )

    print("=" * 70)
    print("MATCHING RESULTS VALIDATION")
    print("=" * 70)
    print(
        "Expected S1 entities :",
        f"{expected_s1_count:,}",
    )
    print(
        "Output rows          :",
        f"{actual_rows:,}",
    )
    print(
        "Unique S1 IDs        :",
        f"{unique_s1_count:,}",
    )

    matched_s1 = (
        matching_results
        .filter(
            pl.col(
                "matched_entity_ids"
            ) != ""
        )
        .height
    )

    unmatched_s1 = (
        actual_rows - matched_s1
    )

    print(
        "S1 with matches      :",
        f"{matched_s1:,}",
    )
    print(
        "S1 without matches   :",
        f"{unmatched_s1:,}",
    )

    print()
    print(
        "✓ Every Source 1 entity appears exactly once."
    )
    print(
        "✓ S1 entities without predictions are retained."
    )
    print(
        "✓ No duplicate matched IDs."
    )


# ============================================================
# 4. candidate_pairs.tsv
# ============================================================

def build_candidate_pairs(
    candidate_partition_paths,
):
    """
    Build candidate_pairs from the FINAL candidate partitions.

    This must be the exact candidate set actually passed into
    model inference.

    Expected partition columns:

        source1_entity_id
        match_id

    Output:

        source1_entity_id
        candidate_entity_ids
    """

    candidate_partition_paths = sorted(
        Path(path)
        for path in candidate_partition_paths
    )

    if not candidate_partition_paths:
        raise RuntimeError(
            "No final candidate partitions found."
        )

    candidates = pl.scan_parquet(
        [
            str(path)
            for path in candidate_partition_paths
        ]
    )

    candidate_pairs = (
        candidates
        .select([
            "source1_entity_id",
            "match_id",
        ])
        .unique()
        .group_by(
            "source1_entity_id"
        )
        .agg(
            pl.col("match_id")
            .sort()
            .str.join(",")
            .alias("candidate_entity_ids")
        )
        .collect(
            engine="streaming"
        )
    )

    return candidate_pairs


def build_complete_candidate_pairs(
    s1_test_path,
    candidate_partition_paths,
):
    """
    Build candidate_pairs.tsv while retaining Source 1 entities
    for which blocking produced zero candidates.
    """

    complete_s1 = load_complete_s1_population(
        s1_test_path
    )

    candidates = build_candidate_pairs(
        candidate_partition_paths
    )

    return (
        complete_s1
        .join(
            candidates.lazy(),
            on="source1_entity_id",
            how="left",
        )
        .with_columns(
            pl.col(
                "candidate_entity_ids"
            ).fill_null("")
        )
        .select([
            "source1_entity_id",
            "candidate_entity_ids",
        ])
        .collect(
            engine="streaming"
        )
    )


# ============================================================
# 5. Candidate / prediction subset validation
# ============================================================

def validate_matches_are_candidates(
    matching_results,
    candidate_pairs,
):
    """
    Verify the important invariant:

        final matches ⊆ final candidates

    Every predicted matched_entity_id must appear in the
    corresponding candidate_entity_ids list.
    """

    candidate_lookup = (
        candidate_pairs
        .select([
            "source1_entity_id",
            "candidate_entity_ids",
        ])
    )

    joined = (
        matching_results
        .join(
            candidate_lookup,
            on="source1_entity_id",
            how="left",
        )
    )

    if (
        joined[
            "candidate_entity_ids"
        ].null_count()
        != 0
    ):
        raise RuntimeError(
            "Some Source 1 entities are missing from "
            "candidate_pairs."
        )

    violations = []

    for row in joined.iter_rows(
        named=True
    ):

        matches = row[
            "matched_entity_ids"
        ]

        candidates = row[
            "candidate_entity_ids"
        ]

        if not matches:
            continue

        candidate_set = set(
            candidates.split(",")
        ) if candidates else set()

        for match_id in matches.split(","):

            if match_id not in candidate_set:
                violations.append(
                    (
                        row[
                            "source1_entity_id"
                        ],
                        match_id,
                    )
                )

    if violations:

        preview = violations[:10]

        raise RuntimeError(
            "Found predicted matches that do not "
            "appear in the final candidate set. "
            f"Examples: {preview}"
        )

    print(
        "✓ All final matches are contained "
        "in the final candidate set."
    )


# ============================================================
# 6. Write submission files
# ============================================================

def write_submission_files(
    matching_results,
    candidate_pairs,
    output_dir,
):
    """
    Write the two challenge submission files.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matching_path = (
        output_dir
        / "matching_results.tsv"
    )

    candidate_path = (
        output_dir
        / "candidate_pairs.tsv"
    )

    matching_results.write_csv(
        matching_path,
        separator="\t",
    )

    candidate_pairs.write_csv(
        candidate_path,
        separator="\t",
    )

    print()
    print("=" * 70)
    print("SUBMISSION FILES WRITTEN")
    print("=" * 70)
    print(
        "matching_results.tsv:",
        matching_path.resolve(),
    )
    print(
        "candidate_pairs.tsv :",
        candidate_path.resolve(),
    )

    return (
        matching_path,
        candidate_path,
    )


# ============================================================
# 7. Complete submission build
# ============================================================

def build_submission(
    s1_test_path,
    candidate_partition_paths,
    aggregate_paths,
    output_dir,
):
    """
    Complete submission construction.

    This function does not run the model.

    It takes:
        final candidate partitions
        aggregated positive predictions
        complete test S1 population

    and produces:

        matching_results.tsv
        candidate_pairs.tsv
    """

    print("=" * 70)
    print("BUILDING SUBMISSION")
    print("=" * 70)

    # --------------------------------------------------------
    # Candidate set
    # --------------------------------------------------------

    candidate_pairs = (
        build_complete_candidate_pairs(
            s1_test_path,
            candidate_partition_paths,
        )
    )

    print(
        f"Candidate S1 rows: "
        f"{candidate_pairs.height:,}"
    )

    # --------------------------------------------------------
    # Final predicted matches
    # --------------------------------------------------------

    matching_results = (
        build_matching_results(
            s1_test_path,
            aggregate_paths,
        )
    )

    print(
        f"Matching S1 rows: "
        f"{matching_results.height:,}"
    )

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    validate_matching_results(
        matching_results,
        s1_test_path,
    )

    # --------------------------------------------------------
    # Candidate subset validation
    # --------------------------------------------------------

    validate_matches_are_candidates(
        matching_results,
        candidate_pairs,
    )

    # --------------------------------------------------------
    # Write files
    # --------------------------------------------------------

    paths = write_submission_files(
        matching_results,
        candidate_pairs,
        output_dir,
    )

    print()
    print("✓ Submission construction complete.")

    return {
        "matching_results": matching_results,
        "candidate_pairs": candidate_pairs,
        "paths": paths,
    }