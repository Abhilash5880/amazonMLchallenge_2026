"""
Evaluation utilities for Amazon ML Challenge 2026.

Two evaluation levels are supported:

1. Pair-level evaluation
   Treats every candidate pair as an independent binary
   classification example.

2. Entity-level evaluation
   Computes F0.5 independently for every Source 1 entity
   and then averages those scores.

The second metric matches the competition objective.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl


# ============================================================
# F0.5
# ============================================================

def f05_from_counts(
    tp: int,
    fp: int,
    fn: int,
) -> float:
    """
    Calculate F0.5 for one Source 1 entity.

    The singleton behavior intentionally matches the
    development notebook:

        TP = 0, FP = 0, FN = 0
            -> 1.0

        TP = 0 with any error
            -> 0.0
    """

    if tp == 0:
        if fp == 0 and fn == 0:
            return 1.0

        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


# ============================================================
# Pair-level evaluation
# ============================================================

def evaluate_pair_predictions(
    true_pairs: pl.DataFrame,
    predicted_pairs: pl.DataFrame,
    source1_column: str = "source1_entity_id",
    match_column: str = "matched_entity_id",
) -> dict:
    """
    Evaluate predictions as independent binary pairs.

    Expected schema:

        source1_entity_id
        matched_entity_id

    Returns:

        TP
        FP
        FN
        precision
        recall
        f05
    """

    true_pairs = (
        true_pairs
        .select([
            source1_column,
            match_column,
        ])
        .unique()
    )

    predicted_pairs = (
        predicted_pairs
        .select([
            source1_column,
            match_column,
        ])
        .unique()
    )

    true_positive = (
        predicted_pairs
        .join(
            true_pairs,
            on=[
                source1_column,
                match_column,
            ],
            how="inner",
        )
        .height
    )

    false_positive = (
        predicted_pairs.height
        - true_positive
    )

    false_negative = (
        true_pairs.height
        - true_positive
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (
            true_positive + false_positive
        ) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (
            true_positive + false_negative
        ) > 0
        else 0.0
    )

    f05 = (
        1.25 * precision * recall
        / (0.25 * precision + recall)
        if (
            0.25 * precision + recall
        ) > 0
        else 0.0
    )

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f05": f05,
    }


# ============================================================
# Entity-level evaluation
# ============================================================

def evaluate_entity_predictions(
    all_s1: pl.DataFrame,
    true_pairs: pl.DataFrame,
    predicted_pairs: pl.DataFrame,
    source1_column: str = "source1_entity_id",
    match_column: str = "matched_entity_id",
) -> tuple[pl.DataFrame, float]:
    """
    Calculate competition-style entity-level macro F0.5.

    Every Source 1 entity is included.

    For each S1:

        true_ids
        predicted_ids
              ↓
        TP / FP / FN
              ↓
           F0.5

    The final score is the arithmetic mean of the
    per-entity F0.5 values.
    """

    # --------------------------------------------------------
    # Deduplicate pair sets
    # --------------------------------------------------------

    true_pairs = (
        true_pairs
        .select([
            source1_column,
            match_column,
        ])
        .unique()
    )

    predicted_pairs = (
        predicted_pairs
        .select([
            source1_column,
            match_column,
        ])
        .unique()
    )

    # --------------------------------------------------------
    # Every S1 entity must participate in evaluation
    # --------------------------------------------------------

    all_s1 = (
        all_s1
        .select(
            pl.col(
                source1_column
            )
        )
        .unique()
    )

    # --------------------------------------------------------
    # Ground-truth IDs grouped by S1
    # --------------------------------------------------------

    true_by_s1 = (
        true_pairs
        .group_by(source1_column)
        .agg(
            pl.col(match_column)
            .alias("true_ids")
        )
    )

    # --------------------------------------------------------
    # Predicted IDs grouped by S1
    # --------------------------------------------------------

    pred_by_s1 = (
        predicted_pairs
        .group_by(source1_column)
        .agg(
            pl.col(match_column)
            .alias("pred_ids")
        )
    )

    # --------------------------------------------------------
    # Join both onto complete S1 population
    # --------------------------------------------------------

    evaluation = (
        all_s1
        .join(
            true_by_s1,
            on=source1_column,
            how="left",
        )
        .join(
            pred_by_s1,
            on=source1_column,
            how="left",
        )
        .with_columns([
            pl.col("true_ids")
            .fill_null([]),

            pl.col("pred_ids")
            .fill_null([]),
        ])
    )

    # --------------------------------------------------------
    # Per-entity TP / FP / FN
    # --------------------------------------------------------

    evaluation = evaluation.with_columns([
        pl.col("pred_ids")
        .list.set_intersection(
            pl.col("true_ids")
        )
        .list.len()
        .alias("tp"),

        pl.col("pred_ids")
        .list.set_difference(
            pl.col("true_ids")
        )
        .list.len()
        .alias("fp"),

        pl.col("true_ids")
        .list.set_difference(
            pl.col("pred_ids")
        )
        .list.len()
        .alias("fn"),
    ])

    # --------------------------------------------------------
    # Per-entity F0.5
    # --------------------------------------------------------

    evaluation = evaluation.with_columns(
        pl.struct([
            "tp",
            "fp",
            "fn",
        ])
        .map_elements(
            lambda x: f05_from_counts(
                x["tp"],
                x["fp"],
                x["fn"],
            ),
            return_dtype=pl.Float64,
        )
        .alias("f05")
    )

    # --------------------------------------------------------
    # Macro average
    # --------------------------------------------------------

    macro_f05 = (
        evaluation["f05"]
        .mean()
    )

    return evaluation, float(macro_f05)


# ============================================================
# Threshold evaluation
# ============================================================

def evaluate_thresholds(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    thresholds: Iterable[float],
) -> pl.DataFrame:
    """
    Evaluate pair-level precision, recall and F0.5
    over a collection of probability thresholds.

    This reproduces the threshold-sweep style used during
    model experimentation.

    Note:
        This is pair-level evaluation. It should NOT be
        described as the official competition metric.
    """

    probabilities = np.asarray(
        probabilities
    )

    y_true = np.asarray(
        y_true
    )

    if probabilities.shape[0] != y_true.shape[0]:
        raise ValueError(
            "probabilities and y_true must contain "
            "the same number of rows."
        )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        )

        tp = int(
            np.sum(
                predictions & (y_true == 1)
            )
        )

        fp = int(
            np.sum(
                predictions & (y_true == 0)
            )
        )

        fn = int(
            np.sum(
                (~predictions) & (y_true == 1)
            )
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f05 = (
            1.25 * precision * recall
            / (0.25 * precision + recall)
            if (
                0.25 * precision + recall
            ) > 0
            else 0.0
        )

        results.append({
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f0.5": f05,
        })

    return pl.DataFrame(
        results
    )


# ============================================================
# Convenience summary
# ============================================================

def evaluation_summary(
    entity_evaluation: pl.DataFrame,
) -> dict:
    """
    Produce useful diagnostics from the entity-level
    evaluation table.
    """

    return {
        "macro_f05": float(
            entity_evaluation["f05"].mean()
        ),

        "s1_entities": (
            entity_evaluation.height
        ),

        "f05_perfect": (
            entity_evaluation
            .filter(
                pl.col("f05") == 1.0
            )
            .height
        ),

        "f05_zero": (
            entity_evaluation
            .filter(
                pl.col("f05") == 0.0
            )
            .height
        ),

        "total_tp": int(
            entity_evaluation["tp"].sum()
        ),

        "total_fp": int(
            entity_evaluation["fp"].sum()
        ),

        "total_fn": int(
            entity_evaluation["fn"].sum()
        ),
    }