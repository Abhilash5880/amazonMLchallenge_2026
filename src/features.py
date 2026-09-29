"""
V4 feature engineering for Amazon ML Challenge 2026
Business Entity Resolution.

Feature contract
----------------
The production CatBoost model expects exactly 18 features:

 1. name_edit_similarity
 2. name_token_jaccard
 3. address_edit_similarity
 4. address_token_jaccard
 5. country_match
 6. name_exact_normalized
 7. name_exact_legal
 8. name_length_ratio
 9. address_exact_normalized
10. address_length_ratio
11. s1_name_missing
12. match_name_missing
13. s1_address_missing
14. match_address_missing
15. name_word_tfidf_cosine
16. address_word_tfidf_cosine
17. name_char_tfidf_cosine
18. address_char_tfidf_cosine

IMPORTANT
---------
The feature order is part of the trained CatBoost model contract.
Do not reorder the columns without retraining the model.
"""

import re

import numpy as np
import polars as pl

from rapidfuzz.fuzz import ratio


# ============================================================
# FEATURE CONTRACT
# ============================================================

V1_FEATURES = [
    "name_edit_similarity",
    "name_token_jaccard",
    "address_edit_similarity",
    "address_token_jaccard",
    "country_match",
]

V2_NEW_FEATURES = [
    "name_exact_normalized",
    "name_exact_legal",
    "name_length_ratio",
    "address_exact_normalized",
    "address_length_ratio",
    "s1_name_missing",
    "match_name_missing",
    "s1_address_missing",
    "match_address_missing",
]

V2_FEATURES = V1_FEATURES + V2_NEW_FEATURES

TFIDF_FEATURES = [
    "name_word_tfidf_cosine",
    "address_word_tfidf_cosine",
    "name_char_tfidf_cosine",
    "address_char_tfidf_cosine",
]

V4_FEATURES = V2_FEATURES + TFIDF_FEATURES

assert len(V1_FEATURES) == 5
assert len(V2_FEATURES) == 14
assert len(TFIDF_FEATURES) == 4
assert len(V4_FEATURES) == 18


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text using the exact V4 production behavior.

    - None -> ""
    - lowercase
    - remove non [a-z0-9 whitespace]
    - collapse whitespace
    - strip
    """
    if value is None:
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_legal_name(value):
    """
    Normalize a business name and repeatedly remove recognized
    legal suffixes from the END of the name.

    This is the V4 production implementation.
    """
    value = normalize_text(value)

    legal_suffixes = [
        "private limited",
        "pvt ltd",
        "pvt limited",
        "private ltd",
        "limited",
        "ltd",
        "llc",
        "incorporated",
        "inc",
        "corp",
        "corporation",
        "company",
        "co",
    ]

    tokens = value.split()

    while tokens:
        removed = False

        for suffix in legal_suffixes:
            suffix_tokens = suffix.split()

            if (
                len(tokens) >= len(suffix_tokens)
                and tokens[-len(suffix_tokens):] == suffix_tokens
            ):
                tokens = tokens[:-len(suffix_tokens)]
                removed = True
                break

        if not removed:
            break

    return " ".join(tokens)


# ============================================================
# BASIC NUMERIC FEATURES
# ============================================================

def safe_length_ratio(a, b):
    """
    Symmetric length ratio.

    Returns:
        min(len(a), len(b)) / max(len(a), len(b))

    Empty strings return 0.0.
    """
    a = a or ""
    b = b or ""

    if not a or not b:
        return 0.0

    return min(len(a), len(b)) / max(len(a), len(b))


def exact_normalized(a, b):
    """
    Exact equality after normalize_text().
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0

    return int(a == b)


def exact_legal_name(a, b):
    """
    Exact equality after normalize_legal_name().
    """
    a = normalize_legal_name(a)
    b = normalize_legal_name(b)

    if not a or not b:
        return 0

    return int(a == b)


def token_jaccard(a, b):
    """
    Token-set Jaccard similarity.
    """
    if not a or not b:
        return 0.0

    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def edit_similarity(a, b):
    """
    RapidFuzz ratio normalized to [0, 1].
    """
    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


# ============================================================
# V2 / HANDCRAFTED FEATURES
# ============================================================

def add_v2_features(df):
    """
    Add the 14 handcrafted V2 features to an enriched
    candidate-pair Polars DataFrame.

    Required columns:

        s1_name
        match_name
        s1_address
        match_address
        s1_country
        match_country
    """

    return df.with_columns([

        # ----------------------------------------------------
        # 1. Name edit similarity
        # ----------------------------------------------------

        pl.struct(["s1_name", "match_name"])
        .map_elements(
            lambda x: edit_similarity(
                x["s1_name"],
                x["match_name"],
            ),
            return_dtype=pl.Float64,
        )
        .alias("name_edit_similarity"),

        # ----------------------------------------------------
        # 2. Name token Jaccard
        # ----------------------------------------------------

        pl.struct(["s1_name", "match_name"])
        .map_elements(
            lambda x: token_jaccard(
                x["s1_name"],
                x["match_name"],
            ),
            return_dtype=pl.Float64,
        )
        .alias("name_token_jaccard"),

        # ----------------------------------------------------
        # 3. Address edit similarity
        # ----------------------------------------------------

        pl.struct(["s1_address", "match_address"])
        .map_elements(
            lambda x: edit_similarity(
                x["s1_address"],
                x["match_address"],
            ),
            return_dtype=pl.Float64,
        )
        .alias("address_edit_similarity"),

        # ----------------------------------------------------
        # 4. Address token Jaccard
        # ----------------------------------------------------

        pl.struct(["s1_address", "match_address"])
        .map_elements(
            lambda x: token_jaccard(
                x["s1_address"],
                x["match_address"],
            ),
            return_dtype=pl.Float64,
        )
        .alias("address_token_jaccard"),

        # ----------------------------------------------------
        # 5. Country match
        # ----------------------------------------------------

        (
            pl.col("s1_country")
            == pl.col("match_country")
        )
        .cast(pl.Int8)
        .alias("country_match"),

        # ----------------------------------------------------
        # 6. Exact normalized name
        # ----------------------------------------------------

        pl.struct(["s1_name", "match_name"])
        .map_elements(
            lambda x: exact_normalized(
                x["s1_name"],
                x["match_name"],
            ),
            return_dtype=pl.Int8,
        )
        .alias("name_exact_normalized"),

        # ----------------------------------------------------
        # 7. Exact legal-normalized name
        # ----------------------------------------------------

        pl.struct(["s1_name", "match_name"])
        .map_elements(
            lambda x: exact_legal_name(
                x["s1_name"],
                x["match_name"],
            ),
            return_dtype=pl.Int8,
        )
        .alias("name_exact_legal"),

        # ----------------------------------------------------
        # 8. Name length ratio
        # ----------------------------------------------------

        pl.struct(["s1_name", "match_name"])
        .map_elements(
            lambda x: safe_length_ratio(
                x["s1_name"],
                x["match_name"],
            ),
            return_dtype=pl.Float64,
        )
        .alias("name_length_ratio"),

        # ----------------------------------------------------
        # 9. Exact normalized address
        # ----------------------------------------------------

        pl.struct(["s1_address", "match_address"])
        .map_elements(
            lambda x: exact_normalized(
                x["s1_address"],
                x["match_address"],
            ),
            return_dtype=pl.Int8,
        )
        .alias("address_exact_normalized"),

        # ----------------------------------------------------
        # 10. Address length ratio
        # ----------------------------------------------------

        pl.struct(["s1_address", "match_address"])
        .map_elements(
            lambda x: safe_length_ratio(
                x["s1_address"],
                x["match_address"],
            ),
            return_dtype=pl.Float64,
        )
        .alias("address_length_ratio"),

        # ----------------------------------------------------
        # 11. S1 name missing
        # ----------------------------------------------------

        (
            pl.col("s1_name")
            .fill_null("")
            .str.strip_chars()
            == ""
        )
        .cast(pl.Int8)
        .alias("s1_name_missing"),

        # ----------------------------------------------------
        # 12. Candidate name missing
        # ----------------------------------------------------

        (
            pl.col("match_name")
            .fill_null("")
            .str.strip_chars()
            == ""
        )
        .cast(pl.Int8)
        .alias("match_name_missing"),

        # ----------------------------------------------------
        # 13. S1 address missing
        # ----------------------------------------------------

        (
            pl.col("s1_address")
            .fill_null("")
            .str.strip_chars()
            == ""
        )
        .cast(pl.Int8)
        .alias("s1_address_missing"),

        # ----------------------------------------------------
        # 14. Candidate address missing
        # ----------------------------------------------------

        (
            pl.col("match_address")
            .fill_null("")
            .str.strip_chars()
            == ""
        )
        .cast(pl.Int8)
        .alias("match_address_missing"),
    ])


# ============================================================
# TF-IDF COSINE
# ============================================================

def rowwise_cosine_similarity(A, B):
    """
    Compute cosine similarity between corresponding rows
    of two sparse TF-IDF matrices.

    TfidfVectorizer uses L2 normalization by default, so
    the row-wise dot product is cosine similarity.
    """
    return (
        np.asarray(
            A.multiply(B).sum(axis=1)
        )
        .ravel()
        .astype(np.float32)
    )


def transform_pairwise_tfidf(
    vectorizer,
    left_text,
    right_text,
):
    """
    Transform two text collections using an already-fitted
    TF-IDF vectorizer and calculate row-wise cosine similarity.

    The vectorizer is NEVER fitted here.

    This is important because inference must use the exact
    vectorizers saved inside the production deployment bundle.
    """

    left_matrix = vectorizer.transform(left_text)
    right_matrix = vectorizer.transform(right_text)

    result = rowwise_cosine_similarity(
        left_matrix,
        right_matrix,
    )

    # Release potentially large sparse matrices immediately.
    del left_matrix
    del right_matrix

    return result


# ============================================================
# V4 MATRIX
# ============================================================

def build_v4_matrix(
    enriched_df,
    name_vectorizer,
    address_vectorizer,
    char_name_vectorizer,
    char_address_vectorizer,
):
    """
    Build the exact 18-feature V4 matrix used by the
    production CatBoost inference pipeline.

    Parameters
    ----------
    enriched_df : polars.DataFrame
        Candidate pairs containing:

            s1_name
            match_name
            s1_address
            match_address
            s1_country
            match_country

    name_vectorizer :
        Fitted word-level name TF-IDF vectorizer.

    address_vectorizer :
        Fitted word-level address TF-IDF vectorizer.

    char_name_vectorizer :
        Fitted character-level name TF-IDF vectorizer.

    char_address_vectorizer :
        Fitted character-level address TF-IDF vectorizer.

    Returns
    -------
    np.ndarray
        Float32 matrix with shape (n_candidates, 18).
    """

    # --------------------------------------------------------
    # 1. Build the 14 handcrafted features
    # --------------------------------------------------------

    v2 = add_v2_features(enriched_df)

    # --------------------------------------------------------
    # 2. Extract text columns
    # --------------------------------------------------------

    s1_names = (
        v2["s1_name"]
        .fill_null("")
        .to_list()
    )

    match_names = (
        v2["match_name"]
        .fill_null("")
        .to_list()
    )

    s1_addresses = (
        v2["s1_address"]
        .fill_null("")
        .to_list()
    )

    match_addresses = (
        v2["match_address"]
        .fill_null("")
        .to_list()
    )

    # --------------------------------------------------------
    # 3. Word-level TF-IDF cosine features
    # --------------------------------------------------------

    name_word = transform_pairwise_tfidf(
        name_vectorizer,
        s1_names,
        match_names,
    )

    address_word = transform_pairwise_tfidf(
        address_vectorizer,
        s1_addresses,
        match_addresses,
    )

    # --------------------------------------------------------
    # 4. Character-level TF-IDF cosine features
    # --------------------------------------------------------

    name_char = transform_pairwise_tfidf(
        char_name_vectorizer,
        s1_names,
        match_names,
    )

    address_char = transform_pairwise_tfidf(
        char_address_vectorizer,
        s1_addresses,
        match_addresses,
    )

    # --------------------------------------------------------
    # 5. Extract the 14 handcrafted features IN ORDER
    # --------------------------------------------------------

    X_v2 = (
        v2
        .select([
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
        ])
        .to_numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # 6. Assemble the final 18-feature matrix
    # --------------------------------------------------------

    X_v4 = np.column_stack([
        X_v2,
        name_word,
        address_word,
        name_char,
        address_char,
    ]).astype(
        np.float32,
        copy=False,
    )

    # --------------------------------------------------------
    # 7. Production safety checks
    # --------------------------------------------------------

    expected_shape = (
        enriched_df.height,
        len(V4_FEATURES),
    )

    if X_v4.shape != expected_shape:
        raise RuntimeError(
            "Unexpected V4 matrix shape: "
            f"{X_v4.shape}; expected {expected_shape}"
        )

    if not np.isfinite(X_v4).all():
        raise RuntimeError(
            "Non-finite values found in V4 inference matrix."
        )

    return X_v4