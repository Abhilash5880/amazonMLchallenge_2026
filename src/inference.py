"""
Partitioned CatBoost inference for Amazon ML Challenge 2026.

The inference pipeline processes the final candidate set in bounded
disk-backed batches:

    candidate partition
        ↓
    entity enrichment
        ↓
    V4 feature matrix
        ↓
    CatBoost predict_proba
        ↓
    threshold = 0.85
        ↓
    positive pairs written to Parquet

The implementation is intentionally partitioned because the test
candidate space is too large to materialize as one in-memory table.
"""

from pathlib import Path
import gc
import time

import numpy as np
import polars as pl

from .features import build_v4_matrix
from .models import EntityResolutionModel


DEFAULT_BATCH_SIZE = 200_000
DEFAULT_N_PARTITIONS = 64


def prepare_test_lookups(
    test_source1_path,
    test_source2_path,
    test_source3_path,
):
    """
    Build lazy lookup tables for the three test sources.

    Source 1 is used to enrich:
        source1_entity_id → S1 fields

    Sources 2 and 3 are combined into one candidate lookup:
        match_id → candidate fields
    """

    test_source1_path = Path(test_source1_path)
    test_source2_path = Path(test_source2_path)
    test_source3_path = Path(test_source3_path)

    test_s1 = pl.scan_csv(
        test_source1_path,
        separator="\t",
    )

    test_s2 = pl.scan_csv(
        test_source2_path,
        separator="\t",
    )

    test_s3 = pl.scan_csv(
        test_source3_path,
        separator="\t",
    )

    s1_lookup = (
        test_s1
        .select([
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ])
        .rename({
            "entity_id": "source1_entity_id",
            "business_name": "s1_name",
            "business_address": "s1_address",
            "country": "s1_country",
        })
    )

    s2_lookup = (
        test_s2
        .select([
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ])
        .rename({
            "entity_id": "match_id",
            "business_name": "match_name",
            "business_address": "match_address",
            "country": "match_country",
        })
    )

    s3_lookup = (
        test_s3
        .select([
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ])
        .rename({
            "entity_id": "match_id",
            "business_name": "match_name",
            "business_address": "match_address",
            "country": "match_country",
        })
    )

    candidate_lookup = pl.concat([
        s2_lookup,
        s3_lookup,
    ])

    counts = {
        "source1": (
            test_s1
            .select(pl.len())
            .collect()
            .item()
        ),
        "source2": (
            test_s2
            .select(pl.len())
            .collect()
            .item()
        ),
        "source3": (
            test_s3
            .select(pl.len())
            .collect()
            .item()
        ),
    }

    return (
        s1_lookup,
        candidate_lookup,
        counts,
    )


def enrich_candidates(
    candidates,
    s1_lookup,
    candidate_lookup,
):
    """
    Attach the raw S1 and candidate records to candidate pairs.

    Returns rows containing:

        source1_entity_id
        match_id
        s1_name
        s1_address
        s1_country
        match_name
        match_address
        match_country
    """

    enriched = (
        candidates
        .lazy()
        .join(
            s1_lookup,
            on="source1_entity_id",
            how="left",
        )
        .join(
            candidate_lookup,
            on="match_id",
            how="left",
        )
        .collect(
            engine="streaming",
        )
    )

    return enriched


def validate_entity_lookups(enriched):
    """
    Verify that candidate IDs successfully resolved to entities.

    Missing names indicate a lookup failure.

    Missing addresses are allowed because missing addresses are a
    legitimate source-data condition represented by V4 missing-value
    features.
    """

    missing_entities = (
        enriched
        .select([
            pl.col("s1_name")
            .is_null()
            .sum()
            .alias("missing_s1_name"),

            pl.col("match_name")
            .is_null()
            .sum()
            .alias("missing_match_name"),
        ])
        .row(0)
    )

    if any(
        int(value) > 0
        for value in missing_entities
    ):
        raise RuntimeError(
            "Entity lookup failure: "
            f"{missing_entities}"
        )

    return missing_entities


def run_smoke_test(
    candidate_partition,
    s1_lookup,
    candidate_lookup,
    model,
    rows=10_000,
):
    """
    Run a bounded end-to-end inference test before full inference.

    Pipeline:

        candidate rows
            ↓
        enrichment
            ↓
        18 V4 features
            ↓
        CatBoost
            ↓
        threshold
    """

    smoke_candidates = (
        pl.scan_parquet(candidate_partition)
        .head(rows)
        .collect(
            engine="streaming",
        )
    )

    start = time.time()

    smoke_enriched = enrich_candidates(
        smoke_candidates,
        s1_lookup,
        candidate_lookup,
    )

    print(
        "Smoke candidates:",
        smoke_candidates.height,
    )

    print(
        "Enriched rows:",
        smoke_enriched.height,
    )

    print(
        "Enrichment time:",
        f"{time.time() - start:.2f}s",
    )

    validate_entity_lookups(
        smoke_enriched,
    )

    X_smoke = build_v4_matrix(
        smoke_enriched,
        model.name_vectorizer,
        model.address_vectorizer,
        model.char_name_vectorizer,
        model.char_address_vectorizer,
    )

    probabilities = model.predict_proba(
        X_smoke
    )

    predictions = (
        probabilities >= model.threshold
    )

    if (
        len(probabilities)
        != smoke_candidates.height
    ):
        raise RuntimeError(
            "Smoke-test prediction count mismatch."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Smoke-test predictions contain "
            "non-finite values."
        )

    print("=" * 65)
    print("CATBOOST V4 SMOKE TEST")
    print("=" * 65)
    print(
        "Rows:",
        len(probabilities),
    )
    print(
        "Features:",
        X_smoke.shape[1],
    )
    print(
        "Threshold:",
        model.threshold,
    )
    print(
        "Predicted matches:",
        int(predictions.sum()),
    )
    print(
        "Positive rate:",
        f"{predictions.mean():.6%}",
    )
    print(
        "Probability min:",
        float(probabilities.min()),
    )
    print(
        "Probability max:",
        float(probabilities.max()),
    )
    print(
        "Probability mean:",
        float(probabilities.mean()),
    )

    print("\n✓ Smoke test passed.")


def infer_partition(
    partition_path,
    output_dir,
    s1_lookup,
    candidate_lookup,
    model,
    batch_size=DEFAULT_BATCH_SIZE,
    partition_index=None,
):
    """
    Run CatBoost inference over one candidate partition.

    Positive predictions are written to disk immediately instead
    of being accumulated in memory.

    Returns a partition summary.
    """

    partition_path = Path(
        partition_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not partition_path.exists():
        raise FileNotFoundError(
            f"Candidate partition not found: "
            f"{partition_path}"
        )

    partition_rows = (
        pl.scan_parquet(partition_path)
        .select(pl.len())
        .collect()
        .item()
    )

    part_start = time.time()

    candidates_processed = 0
    positives_written = 0
    batch_count = 0

    for offset in range(
        0,
        partition_rows,
        batch_size,
    ):
        batch_end = min(
            offset + batch_size,
            partition_rows,
        )

        # --------------------------------------------------
        # Load bounded candidate batch
        # --------------------------------------------------

        batch = (
            pl.scan_parquet(
                partition_path
            )
            .slice(
                offset,
                batch_end - offset,
            )
            .collect(
                engine="streaming",
            )
        )

        # --------------------------------------------------
        # Enrich candidate pairs
        # --------------------------------------------------

        enriched = enrich_candidates(
            batch,
            s1_lookup,
            candidate_lookup,
        )

        # --------------------------------------------------
        # Verify entity lookups
        # --------------------------------------------------

        validate_entity_lookups(
            enriched
        )

        # --------------------------------------------------
        # Build exact V4 feature matrix
        # --------------------------------------------------

        X_batch = build_v4_matrix(
            enriched,
            model.name_vectorizer,
            model.address_vectorizer,
            model.char_name_vectorizer,
            model.char_address_vectorizer,
        )

        # --------------------------------------------------
        # CatBoost probability
        # --------------------------------------------------

        probabilities = model.predict_proba(
            X_batch
        )

        # --------------------------------------------------
        # Apply frozen production threshold
        # --------------------------------------------------

        positive_mask = (
            probabilities >= model.threshold
        )

        positive_count = int(
            positive_mask.sum()
        )

        # --------------------------------------------------
        # Persist positive predictions
        # --------------------------------------------------

        if positive_count > 0:

            positive = (
                enriched
                .select([
                    "source1_entity_id",
                    "match_id",
                ])
                .filter(
                    pl.Series(
                        "positive",
                        positive_mask,
                    )
                )
                .with_columns(
                    pl.Series(
                        "probability",
                        probabilities[
                            positive_mask
                        ],
                    )
                )
            )

            batch_output = (
                output_dir
                / f"batch_{batch_count:04d}.parquet"
            )

            positive.write_parquet(
                batch_output,
                compression="zstd",
            )

            positives_written += (
                positive_count
            )

        candidates_processed += (
            batch.height
        )

        batch_count += 1

        elapsed = (
            time.time()
            - part_start
        )

        print(
            f"  batch {batch_count:4d} | "
            f"{batch_end:,}/{partition_rows:,} | "
            f"positives={positive_count:,} | "
            f"part positives={positives_written:,} | "
            f"{elapsed / 60:.1f} min"
        )

        # --------------------------------------------------
        # Release memory
        # --------------------------------------------------

        del (
            batch,
            enriched,
            X_batch,
            probabilities,
            positive_mask,
        )

        if positive_count > 0:
            del positive

        gc.collect()

    elapsed = (
        time.time()
        - part_start
    )

    summary = {
        "partition": partition_index,
        "candidate_rows": candidates_processed,
        "positive_rows": positives_written,
        "batches": batch_count,
        "minutes": elapsed / 60,
    }

    print(
        f"✓ Partition "
        f"{partition_index:03d} COMPLETE | "
        f"candidates={candidates_processed:,} | "
        f"positives={positives_written:,} | "
        f"time={elapsed / 60:.2f} min"
    )

    return summary


def run_partitioned_inference(
    candidate_dir,
    inference_dir,
    model,
    s1_lookup,
    candidate_lookup,
    start_partition=0,
    end_partition=DEFAULT_N_PARTITIONS,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Run inference over a range of candidate partitions.

    Parameters
    ----------
    candidate_dir:
        Directory containing part_000.parquet,
        part_001.parquet, etc.

    inference_dir:
        Directory where positive prediction batches
        will be written.

    model:
        Loaded EntityResolutionModel.

    start_partition:
        Inclusive starting partition.

    end_partition:
        Exclusive ending partition.

    batch_size:
        Maximum number of candidate pairs loaded into
        one inference batch.
    """

    candidate_dir = Path(
        candidate_dir
    )

    inference_dir = Path(
        inference_dir
    )

    inference_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    global_start = time.time()

    for part_idx in range(
        start_partition,
        end_partition,
    ):

        partition_path = (
            candidate_dir
            / f"part_{part_idx:03d}.parquet"
        )

        partition_output_dir = (
            inference_dir
            / f"part_{part_idx:03d}"
        )

        partition_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = infer_partition(
            partition_path=partition_path,
            output_dir=partition_output_dir,
            s1_lookup=s1_lookup,
            candidate_lookup=candidate_lookup,
            model=model,
            batch_size=batch_size,
            partition_index=part_idx,
        )

        summaries.append(summary)

    total_elapsed = (
        time.time()
        - global_start
    )

    total_candidates = sum(
        item["candidate_rows"]
        for item in summaries
    )

    total_positives = sum(
        item["positive_rows"]
        for item in summaries
    )

    print()
    print("=" * 72)
    print("SELECTED CATBOOST INFERENCE COMPLETE")
    print("=" * 72)
    print(
        f"Partitions processed: "
        f"{start_partition} → "
        f"{end_partition - 1}"
    )
    print(
        f"Batch size: "
        f"{batch_size:,}"
    )
    print(
        "Candidate rows processed:",
        f"{total_candidates:,}",
    )
    print(
        "Positive rows written:",
        f"{total_positives:,}",
    )
    print(
        "Total time:",
        f"{total_elapsed / 3600:.2f} hours",
    )
    print("=" * 72)

    if total_positives == 0:
        raise RuntimeError(
            "Zero predictions passed the production threshold."
        )

    return summaries