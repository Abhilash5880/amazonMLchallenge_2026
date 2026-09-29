"""
Normalization utilities for Amazon ML Challenge 2026
business entity resolution.

IMPORTANT:
The project contains multiple normalization contracts from
different stages of the pipeline.

- normalize_basic / normalize_legal_suffix:
    Frozen V1 blocking behavior.

- normalize_text / normalize_legal_name:
    V4 model feature behavior.

Do not replace these with a single generalized normalizer
without re-validating candidate generation and model features.
"""

import re
from typing import Any


# ============================================================
# V1 BLOCKING NORMALIZATION
# ============================================================

def normalize_basic(name: Any) -> str:
    """
    Basic normalization used by the frozen V1 blocking pipeline.

    Behavior:
    - None -> ""
    - lowercase
    - replace non-word/non-whitespace characters with spaces
    - collapse whitespace
    - strip
    """
    if name is None:
        return ""

    name = str(name).lower()
    name = re.sub(r"[^\w\s]", " ", name)

    return re.sub(r"\s+", " ", name).strip()


LEGAL_SUFFIX_RE = re.compile(
    r"\b("
    r"private\s+limited|"
    r"pvt\s+limited|"
    r"pvt\s+ltd|"
    r"private\s+ltd|"
    r"limited|"
    r"ltd|"
    r"incorporated|"
    r"inc|"
    r"corporation|"
    r"corp|"
    r"llc|"
    r"llp|"
    r"plc|"
    r"gmbh|"
    r"sarl|"
    r"sas|"
    r"pte\s+ltd|"
    r"pty\s+ltd|"
    r"company"
    r")\s*$",
    re.IGNORECASE,
)


def normalize_legal_suffix(name: Any) -> str:
    """
    V1 legal-suffix normalization.

    Removes recognized legal suffixes from the END of the name.
    Repeats until no additional suffix can be removed.
    """
    if not name:
        return ""

    name = normalize_basic(name)

    previous = None

    while name != previous:
        previous = name
        name = LEGAL_SUFFIX_RE.sub("", name).strip()

    return name


# ============================================================
# ADDITIONAL RESEARCH/BLOCKING UTILITIES
# ============================================================

def normalize_research_text(value: Any) -> str:
    """
    Normalization used in the earlier research/blocking notebook.

    This is kept separately because its '&' handling differs from
    normalize_basic().
    """
    if value is None:
        return ""

    value = str(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_research_name(value: Any) -> str:
    """
    Earlier research-stage name normalization.
    """
    legal_suffixes = {
        "incorporated",
        "inc",
        "corporation",
        "corp",
        "limited",
        "ltd",
        "llc",
        "llp",
        "private",
        "pvt",
        "company",
        "co",
        "limitedliabilitycompany",
    }

    normalized = normalize_research_text(value)

    tokens = [
        token
        for token in normalized.split()
        if token not in legal_suffixes
    ]

    return " ".join(tokens)


def compact(value: str) -> str:
    """
    Remove whitespace from an already-normalized representation.
    """
    return re.sub(r"\s+", "", value)


def token_signature(value: str) -> str:
    """
    Sorted unique-token representation used during research.
    """
    return " ".join(sorted(set(value.split())))


# ============================================================
# V4 MODEL NORMALIZATION
# ============================================================

def normalize_text(value: Any) -> str:
    """
    V4 normalization used by the handcrafted feature pipeline.

    This is intentionally separate from normalize_basic().
    """
    if value is None:
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_legal_name(value: Any) -> str:
    """
    V4 legal-name normalization.

    Removes recognized legal suffixes from the END of the
    normalized business name.
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
# V4 SAFE LENGTH FEATURE
# ============================================================

def safe_length_ratio(a: Any, b: Any) -> float:
    """
    V4 symmetric length ratio.

    Returns:
        min(length(a), length(b)) / max(length(a), length(b))

    Empty values return 0.0, matching the V4 implementation.
    """
    a = a or ""
    b = b or ""

    if not a or not b:
        return 0.0

    return min(len(a), len(b)) / max(len(a), len(b))