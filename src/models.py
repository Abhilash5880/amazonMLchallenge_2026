"""
Production model loading and prediction utilities.

Amazon ML Challenge 2026
Business Entity Resolution

The production model is a frozen CatBoost V4 Depth-10 model
with an 18-feature input contract.

The deployment bundle contains:

    - CatBoost model
    - word-level name TF-IDF vectorizer
    - word-level address TF-IDF vectorizer
    - character-level name TF-IDF vectorizer
    - character-level address TF-IDF vectorizer
    - feature ordering
    - decision threshold
    - deployment configuration

IMPORTANT
---------
The TF-IDF vectorizers must be loaded from the deployment bundle.
They must NOT be refit during inference.
"""

from pathlib import Path

import joblib
import numpy as np


EXPECTED_FEATURE_COUNT = 18

EXPECTED_FEATURES = [
    "name_edit_similarity",
    "name_token_jaccard",
    "address_edit_similarity",
    "address_token_jaccard",
    "country_match",
    "name_exact_normalized",
    "name_exact_legal",
    "name_length_ratio",
    "address_exact_normalized",
    "address_length_ratio",
    "s1_name_missing",
    "match_name_missing",
    "s1_address_missing",
    "match_address_missing",
    "name_word_tfidf_cosine",
    "address_word_tfidf_cosine",
    "name_char_tfidf_cosine",
    "address_char_tfidf_cosine",
]

EXPECTED_THRESHOLD = 0.85

REQUIRED_BUNDLE_KEYS = [
    "model",
    "name_vectorizer",
    "address_vectorizer",
    "char_name_vectorizer",
    "char_address_vectorizer",
    "features",
    "threshold",
    "config",
]


class EntityResolutionModel:
    """
    Wrapper around the frozen CatBoost V4 deployment bundle.

    The wrapper does not train or refit anything.

    It only:
        1. loads the deployment artifact,
        2. validates the model contract,
        3. exposes the fitted preprocessing objects,
        4. generates probabilities,
        5. applies the frozen decision threshold.
    """

    def __init__(self, bundle_path):
        self.bundle_path = Path(bundle_path)

        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"Deployment bundle not found: {self.bundle_path}"
            )

        self.bundle = joblib.load(self.bundle_path)

        self._validate_bundle()

        self.model = self.bundle["model"]

        self.name_vectorizer = (
            self.bundle["name_vectorizer"]
        )

        self.address_vectorizer = (
            self.bundle["address_vectorizer"]
        )

        self.char_name_vectorizer = (
            self.bundle["char_name_vectorizer"]
        )

        self.char_address_vectorizer = (
            self.bundle["char_address_vectorizer"]
        )

        self.features = list(
            self.bundle["features"]
        )

        self.threshold = float(
            self.bundle["threshold"]
        )

        self.config = self.bundle["config"]

    def _validate_bundle(self):
        """Validate the deployment artifact before inference."""

        missing = [
            key
            for key in REQUIRED_BUNDLE_KEYS
            if key not in self.bundle
        ]

        if missing:
            raise RuntimeError(
                "Deployment bundle is missing required keys: "
                f"{missing}"
            )

        features = self.bundle["features"]

        if features != EXPECTED_FEATURES:
            raise RuntimeError(
                "Feature ordering mismatch.\n"
                f"Expected: {EXPECTED_FEATURES}\n"
                f"Found:    {features}"
            )

        if len(features) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "Unexpected feature count: "
                f"{len(features)}; "
                f"expected {EXPECTED_FEATURE_COUNT}"
            )

        threshold = float(
            self.bundle["threshold"]
        )

        if threshold != EXPECTED_THRESHOLD:
            raise RuntimeError(
                "Unexpected model threshold: "
                f"{threshold}; "
                f"expected {EXPECTED_THRESHOLD}"
            )

        model = self.bundle["model"]

        feature_importance = (
            model.get_feature_importance()
        )

        if len(feature_importance) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "CatBoost model feature count mismatch: "
                f"{len(feature_importance)}; "
                f"expected {EXPECTED_FEATURE_COUNT}"
            )

    def predict_proba(self, X):
        """
        Return positive-class probabilities.

        Parameters
        ----------
        X : array-like
            Matrix containing the 18 production features.

        Returns
        -------
        numpy.ndarray
            Probability of the pair being a true match.
        """

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D feature matrix, got shape {X.shape}"
            )

        if X.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                "Unexpected feature matrix width: "
                f"{X.shape[1]}; "
                f"expected {EXPECTED_FEATURE_COUNT}"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Feature matrix contains non-finite values."
            )

        probabilities = (
            self.model
            .predict_proba(X)[:, 1]
        )

        return probabilities

    def predict(self, X):
        """
        Convert match probabilities into binary predictions
        using the frozen deployment threshold.
        """

        probabilities = self.predict_proba(X)

        return (
            probabilities >= self.threshold
        ).astype(np.int8)

    def predict_with_probability(self, X):
        """
        Return both probabilities and binary predictions.

        Useful during inference and debugging.
        """

        probabilities = self.predict_proba(X)

        predictions = (
            probabilities >= self.threshold
        ).astype(np.int8)

        return probabilities, predictions