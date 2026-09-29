from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# config.py is inside:
# AMAZON_ML_2026_github/src/
#
# parents[1] therefore points to:
# AMAZON_ML_2026_github/

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"


# ============================================================
# WORKING / INTERMEDIATE DATA
# ============================================================

# Large intermediate candidate files and partitioned inference
# artifacts should live outside the Git-tracked output folder.

WORK_DIR = DATA_DIR / "work"

CANDIDATE_DIR = WORK_DIR / "candidates"
INFERENCE_DIR = WORK_DIR / "inference"


# ============================================================
# MODEL ARTIFACT
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_BUNDLE_PATH = (
    MODEL_DIR
    / "final_tuned_catboost_v4_depth10_deployment_bundle.joblib"
)


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

MODEL_VERSION = "V4_CatBoost_Depth10"

EXPECTED_FEATURE_COUNT = 18

THRESHOLD = 0.85


# ============================================================
# INFERENCE CONFIGURATION
# ============================================================

# Candidate partitions are processed one at a time rather than
# loading the entire candidate population into memory.

N_CANDIDATE_PARTITIONS = 64

INFERENCE_BATCH_SIZE = 200_000


# ============================================================
# FINAL OUTPUT FILES
# ============================================================

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"

CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# EXPECTED V4 FEATURE ORDER
# ============================================================

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


# ============================================================
# TF-IDF CONFIGURATION
# ============================================================

WORD_TFIDF_NGRAM_RANGE = (1, 2)
WORD_TFIDF_MIN_DF = 2
WORD_TFIDF_MAX_FEATURES = 100_000

CHAR_TFIDF_NGRAM_RANGE = (2, 5)
CHAR_TFIDF_MIN_DF = 2
CHAR_TFIDF_MAX_FEATURES = 150_000


# ============================================================
# DATASET FILENAMES
# ============================================================

TRAIN_SOURCE1_PATH = DATA_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH = DATA_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH = DATA_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = DATA_DIR / "train_ground_truth.tsv"

TEST_SOURCE1_PATH = DATA_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = DATA_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = DATA_DIR / "test_source3.tsv"


# ============================================================
# CANDIDATE PARTITION PATTERN
# ============================================================

CANDIDATE_PARTITION_PATTERN = "part_*.parquet"