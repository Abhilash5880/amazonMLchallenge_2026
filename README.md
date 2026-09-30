# Amazon ML Challenge 2026: Business Entity Resolution

> A large-scale business entity-resolution system for matching noisy records across multiple data sources using multi-pass blocking, string similarity, TF-IDF representations, gradient-boosted models, disk-backed inference, and entity-level evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Why Entity Resolution Is Difficult](#why-entity-resolution-is-difficult)
- [System at a Glance](#system-at-a-glance)
- [Project Evolution](#project-evolution)
- [Research and Discovery](#research-and-discovery)
- [Data Normalization](#data-normalization)
- [Candidate Generation and Blocking](#candidate-generation-and-blocking)
- [Candidate Union and Deduplication](#candidate-union-and-deduplication)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Model Tournament](#model-tournament)
- [V3: Word-Level TF-IDF](#v3-word-level-tf-idf)
- [V4: Word + Character TF-IDF](#v4-word--character-tf-idf)
- [Final CatBoost Deployment](#final-catboost-deployment)
- [Large-Scale Inference](#large-scale-inference)
- [Submission Construction](#submission-construction)
- [Evaluation Methodology](#evaluation-methodology)
- [Problems Faced and Solutions](#problems-faced-and-solutions)
- [Important ML Lessons](#important-ml-lessons)
- [Repository Architecture](#repository-architecture)
- [Module Responsibilities](#module-responsibilities)
- [Reproducibility](#reproducibility)
- [Experiment History](#experiment-history)
- [Results](#results)
- [Current Repository Status](#current-repository-status)
- [Known Risks](#known-risks)
- [Future Improvements](#future-improvements)
- [Interview Explanation](#interview-explanation)
- [Conclusion](#conclusion)

---

# Project Overview

This project was developed for the **Amazon ML Challenge 2026 Business Entity Resolution** problem.

The objective is to identify which records in Source 2 and Source 3 correspond to each business entity in Source 1.

The same real-world business can appear differently across datasets due to:

- spelling variations
- punctuation differences
- legal suffixes
- abbreviations
- transliteration
- missing values
- address formatting differences
- tokenization differences
- noisy text
- incomplete records
- multiple valid matches

The project therefore evolved from simple string matching into a complete entity-resolution pipeline.

The final architecture is:

```text
                    SOURCE DATA
                        |
                        v
                 NORMALIZATION
                        |
                        v
              BLOCKING / RETRIEVAL
                        |
                        v
             CANDIDATE GENERATION
                        |
                        v
             UNION + DEDUPLICATION
                        |
                        v
             CANDIDATE ENRICHMENT
                        |
                        v
                FEATURE ENGINEERING
                        |
                        v
                ML MATCH CLASSIFIER
                        |
                        v
                PROBABILITY THRESHOLD
                        |
                        v
                 POSITIVE PAIRS
                        |
              +---------+---------+
              |                   |
              v                   v
    matching_results.tsv   candidate_pairs.tsv
```

The key idea is that **entity resolution is not just classification**.

It is:

```text
Retrieval
    +
Representation
    +
Classification
    +
Aggregation
    +
Systems Engineering
```

---

# Problem Statement

For every Source-1 entity, the system must identify:

- zero matching records
- one matching record
- multiple matching records

from Source 2 and Source 3.

A Source-1 entity can therefore have a match set such as:

```text
S1-001 -> S2-031,S3-892
S1-002 ->
S1-003 -> S2-441
```

This is fundamentally different from a conventional one-to-one classification problem.

The final output is an entity-level match set rather than a single class label.

---

# Evaluation Objective

The challenge uses an entity-level **macro F0.5** metric.

F0.5 places greater importance on precision than recall:

```text
                 1.25 × Precision × Recall
F0.5 = ---------------------------------------------
        0.25 × Precision + Recall
```

The important consequence is that false matches are expensive.

A true singleton:

```text
S1-A -> []
```

that is correctly predicted as:

```text
S1-A -> []
```

receives the best possible contribution for that entity.

But:

```text
S1-A -> [incorrect candidate]
```

can completely destroy that entity's score.

This makes conservative matching particularly important.

---

# Why Entity Resolution Is Difficult

Consider:

```text
Source 1:
"Acme Private Limited"
"12 Baker Street, London"

Source 2:
"ACME PVT LTD"
"12 Baker St., London"

Source 3:
"Acme Ltd."
"12 Baker Street London"
```

All three may represent the same business.

However, exact string equality fails.

At the same time, naive fuzzy matching is dangerous because unrelated businesses can share:

- common business names
- common addresses
- common tokens
- common legal suffixes

The system therefore combines several forms of evidence.

---

# The Central Engineering Problem

A brute-force comparison would look like:

```text
Every Source-1 record
        |
        +---- every Source-2 record
        |
        +---- every Source-3 record
```

This produces an enormous comparison space.

The project research estimated the theoretical space at approximately:

```text
11T+ comparisons
```

The solution was to introduce **blocking**.

```text
ALL POSSIBLE PAIRS
        |
        v
     BLOCKING
        |
        v
CANDIDATE PAIRS
        |
        v
FEATURE ENGINEERING
        |
        v
ML CLASSIFIER
```

This is one of the most important design decisions in the project.

> If a true pair is removed during blocking, no downstream classifier can recover it.

Therefore blocking creates the recall ceiling for the entire system.

---

# Project Evolution

The project evolved through six major phases:

```text
1. Research / Discovery
          |
          v
2. Blocking / Candidate Generation
          |
          v
3. ML Model Experiments
          |
          v
4. Final Model Deployment
          |
          v
5. Large-Scale Test Inference
          |
          v
6. Submission Construction
```

The model lineage became:

```text
RESEARCH
   |
   v
MULTI-PASS BLOCKING
   |
   v
V1
Logistic Regression
5 features
   |
   v
V2
14 handcrafted features
   |
   +---- Logistic Regression
   +---- Random Forest
   +---- XGBoost
   +---- Extra Trees
   +---- HistGradientBoosting
   +---- MLP
   +---- CatBoost
   +---- LightGBM
   |
   v
V3
14 handcrafted + word TF-IDF
   |
   v
V4
14 handcrafted
+
word TF-IDF
+
character TF-IDF
   |
   +---- XGBoost
   +---- CatBoost
   +---- LightGBM
   |
   v
CATBOOST DEPLOYMENT
Depth 10
18 features
threshold = 0.85
   |
   v
DEPLOYMENT BUNDLE
   |
   v
PARTITIONED TEST INFERENCE
   |
   v
SUBMISSION
```

This lineage is preserved because the final project should show **how the solution was developed**, not merely display the final model.

---

# Research and Discovery

The original research notebook was the project's experimental laboratory.

It investigated:

- baseline matching
- name normalization
- legal suffix removal
- exact matching
- fuzzy string similarity
- token similarity
- address similarity
- hard negatives
- name/address interaction
- name-token blocking
- address-token blocking
- rare-token blocking
- candidate union
- true-pair retrieval
- blocking recall

The research phase answered a critical question:

> Which signals can reduce the candidate space without removing too many true matches?

---

# Multi-Pass Blocking

A single blocking signal was not sufficient.

Different corruption patterns require different retrieval mechanisms.

The project therefore investigated multiple passes.

```text
                 SOURCE 1
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
   Exact Name   Name Tokens   Address
        |           |           |
        v           v           v
     precise     recall      independent
        |           |           |
        +-----------+-----------+
                    |
                    v
                  UNION
                    |
                    v
              CANDIDATE SET
```

The purpose was not to make every block perfect.

The purpose was to allow one retrieval mechanism to compensate for another.

---

# Rare-Token Experiment

One experiment studied how rare-token frequency thresholds affected candidate coverage.

| S2 token frequency cutoff | S1 coverage | True-pair coverage |
|---:|---:|---:|
| <= 100 | 32.40% | 43.94% |
| <= 500 | 44.86% | 56.40% |
| <= 1,000 | 55.34% | 65.22% |
| <= 5,000 | 78.31% | 83.61% |
| <= 10,000 | 86.64% | 90.07% |
| <= 50,000 | 99.75% | 99.88% |

This demonstrated a clear trade-off:

```text
More permissive blocking
        |
        +--> more candidates
        |
        +--> higher recall
        |
        +--> higher computational cost
```

The final design therefore relied on multiple candidate-generation passes rather than one aggressively tuned block.

---

# Blocking Checkpoint

A major development checkpoint produced:

| Metric | Result |
|---|---:|
| Theoretical comparisons | ~11T+ |
| Candidate pairs | 5,901,537 |
| True pairs | 27,907 |
| Recovered true pairs | 25,267 |
| Blocking recall | **90.54%** |

The significance of this result is larger than the raw numbers.

The project had established:

```text
11T+ theoretical comparisons
             |
             v
5.9M candidate pairs
             |
             v
90.54% true-pair retrieval
```

This transformed an impossible brute-force search into a candidate-classification problem.

---

# Data Normalization

Normalization was treated as a **pipeline contract**.

A major lesson from the project was that different stages should not automatically share one generalized normalizer.

There are separate normalization behaviors for:

- V1 blocking
- research experiments
- V4 model features

Changing a normalizer can change candidate generation and therefore change the entire downstream distribution.

---

# V1 Blocking Normalization

The basic blocking normalizer performs:

```text
None
  |
  v
""
```

and otherwise:

```text
lowercase
    |
remove punctuation
    |
collapse whitespace
    |
strip
```

Conceptually:

```python
def normalize_basic(name):
    if name is None:
        return ""

    name = str(name).lower()
    name = re.sub(r"[^\w\s]", " ", name)

    return re.sub(r"\s+", " ", name).strip()
```

---

# Legal Suffix Normalization

Recognized suffixes include variants such as:

```text
private limited
pvt limited
pvt ltd
private ltd
limited
ltd
incorporated
inc
corporation
corp
llc
llp
plc
gmbh
sarl
sas
pte ltd
pty ltd
company
```

Suffixes are removed from the end of the normalized business name.

This converts examples such as:

```text
Acme Private Limited
Acme Pvt Ltd
Acme Ltd
```

toward a common representation.

---

# V4 Model Normalization

The final model feature pipeline uses:

```python
def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()
```

The V4 legal-name normalization then repeatedly removes recognized legal suffixes from the end.

The project intentionally preserves these normalization contracts separately.

---

# Candidate Generation and Deduplication

Multiple blocking passes can generate the same candidate.

For example:

```text
Name block:
(S1-001, S2-010)

Address block:
(S1-001, S2-010)
```

The pair must appear only once.

The pipeline therefore uses:

```python
pair_key = (
    source1_entity_id,
    match_id,
)
```

and keeps a set of already-seen pairs.

Conceptually:

```text
name candidates
      |
      +------+
             |
address -----+----> UNION ----> DEDUP
             |
other blocks-+
```

The final candidate population is then used as the exact input population for model inference.

---

# Feature Engineering

The final CatBoost deployment expects exactly **18 features**.

The feature order is part of the trained model contract.

The production ordering is:

```text
1.  name_edit_similarity
2.  name_token_jaccard
3.  address_edit_similarity
4.  address_token_jaccard
5.  country_match
6.  name_exact_normalized
7.  name_exact_legal
8.  name_length_ratio
9.  address_exact_normalized
10. address_length_ratio
11. s1_name_missing
12. match_name_missing
13. s1_address_missing
14. match_address_missing
15. name_word_tfidf_cosine
16. address_word_tfidf_cosine
17. name_char_tfidf_cosine
18. address_char_tfidf_cosine
```

The feature contract is deliberately validated before inference.

---

# V1 Features

## 1. Name edit similarity

RapidFuzz ratio between normalized names.

The similarity is scaled to:

```text
[0, 1]
```

---

## 2. Name token Jaccard

For token sets A and B:

```text
J(A,B) = |A ∩ B| / |A ∪ B|
```

This provides a measure of token overlap.

---

## 3. Address edit similarity

RapidFuzz similarity between normalized addresses.

---

## 4. Address token Jaccard

Token overlap between normalized addresses.

---

## 5. Country match

Binary feature:

```text
1 -> countries match
0 -> countries differ
```

Country is treated as a feature rather than as a hard-coded assumption about which countries can appear.

---

# V2 Feature Expansion

The initial five features were expanded with:

```text
6.  name_exact_normalized
7.  name_exact_legal
8.  name_length_ratio
9.  address_exact_normalized
10. address_length_ratio
11. s1_name_missing
12. match_name_missing
13. s1_address_missing
14. match_address_missing
```

This increased the representation from:

```text
5 features
```

to:

```text
14 features
```

The additional features captured exact equality, structural compatibility and missingness.

---

# Length Ratio

For two strings:

```text
min(len(a), len(b))
-------------------
max(len(a), len(b))
```

Empty strings return:

```text
0.0
```

This gives the model a simple signal for whether two strings have compatible lengths.

---

# Missingness Features

Missing values are not simply discarded.

The model receives explicit indicators:

```text
s1_name_missing
match_name_missing
s1_address_missing
match_address_missing
```

This allows the model to distinguish:

```text
low similarity because the records are unrelated
```

from:

```text
low similarity because one field is unavailable
```

---

# TF-IDF Features

The V3/V4 stages introduced text-vector similarity.

## Word-level TF-IDF

The word vectorizers use:

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=100_000,
)
```

Separate vectorizers are used for:

- business names
- addresses

---

# Character-level TF-IDF

The V4 character vectorizers use:

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=150_000,
)
```

Again, separate vectorizers are used for:

- business names
- addresses

Character n-grams help preserve local spelling structure.

For example:

```text
microelectronics
micro-electronics
microelectronix
```

may still share useful character patterns even when word-level matching is weaker.

---

# TF-IDF Cosine Similarity

The vectorizers produce sparse matrices.

Because TF-IDF uses L2 normalization by default, the row-wise dot product corresponds to cosine similarity.

The pipeline computes:

- name word cosine
- address word cosine
- name character cosine
- address character cosine

These become the final four features.

---

# Final V4 Feature Vector

The final model input is:

```text
14 handcrafted features
+
4 TF-IDF cosine features
=
18 features
```

The feature matrix is converted to `float32`.

The inference pipeline validates:

```text
shape == (number_of_candidates, 18)
```

and rejects non-finite values.

---

# Model Development

## V1: Logistic Regression

The first proper ML matcher used:

```text
5 features
+
Logistic Regression
+
balanced class weights
```

The five features were:

- name edit similarity
- name token Jaccard
- address edit similarity
- address token Jaccard
- country match

One recorded local result was:

```text
Precision = 95.9824%
Recall    = 60.5185%
F0.5      = 85.9134%
```

This established the first meaningful baseline.

---

# V2: Fourteen Features

The next stage added nine additional handcrafted features.

The representation became:

```text
V1
+
exact equality
+
legal-name equality
+
length ratios
+
missingness
```

The resulting 14-feature representation was evaluated across several model families.

---

# Model Tournament

The V2 model tournament produced the following recorded local pair-level results:

| Model | Threshold | Precision | Recall | F0.5 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.92 | 91.98% | 70.77% | 86.78% |
| Random Forest | 0.69 | 97.24% | 85.83% | 94.72% |
| XGBoost | 0.81 | 98.05% | 83.79% | 94.83% |
| XGBoost variant | 0.63 | 97.21% | 86.88% | 94.95% |
| Extra Trees | 0.73 | 97.35% | 85.08% | 94.62% |
| HistGradientBoosting | 0.67 | 97.24% | 85.71% | 94.69% |
| MLP | 0.70 | 97.64% | 83.57% | 94.46% |
| CatBoost | 0.73 | 97.72% | 84.72% | 94.81% |
| LightGBM | 0.70 | 97.47% | 85.38% | 94.78% |

These results are:

> **Local pair-level validation results.**

They are not presented as the official competition score.

---

# V3: Word-Level TF-IDF

The next major improvement came from adding word-level TF-IDF.

The representation became:

```text
14 handcrafted features
+
word-level TF-IDF similarities
```

The recorded XGBoost result was:

```text
Precision = 99.0666%
Recall    = 96.6159%
F0.5      = 98.5665%
```

This was a substantial improvement over the handcrafted-only representation.

---

# V4: Word + Character TF-IDF

V4 added character-level TF-IDF.

The final representation became:

```text
14 handcrafted
+
name word TF-IDF
+
address word TF-IDF
+
name character TF-IDF
+
address character TF-IDF
```

Total:

```text
18 features
```

---

# V4 Model Comparison

Recorded local pair-level results:

| Model | Threshold | Precision | Recall | F0.5 |
|---|---:|---:|---:|---:|
| XGBoost | 0.90 | 99.2527% | 97.2492% | 98.8454% |
| CatBoost | 0.77 | 99.1374% | 97.8033% | 98.8677% |
| LightGBM | 0.88 | 99.2129% | 97.2887% | 98.8220% |

A later CatBoost Depth-10 configuration was selected for deployment development:

```text
Model:
CatBoost

Depth:
10

Features:
18

Threshold:
0.85
```

Recorded local pair-level validation:

```text
Precision = 99.2551%
Recall    = 97.5658%
F0.5      = 98.9126%
```

Again:

> These are local pair-level development metrics, not the official competition score.

---

# Final CatBoost Deployment

The deployment stage converted the experimental model into a reusable artifact.

The model was trained using the full labeled pair set used by the deployment notebook, recorded as **172,458 labeled pairs**.

The resulting artifact was:

```text
final_tuned_catboost_v4_depth10_deployment_bundle.joblib
```

The bundle contains:

```text
model
name_vectorizer
address_vectorizer
char_name_vectorizer
char_address_vectorizer
features
threshold
config
```

The production contract requires:

```text
18 features
feature order unchanged
threshold = 0.85
```

---

# Why the Model Bundle Includes the Vectorizers

The final model depends on fitted TF-IDF vocabularies and IDF statistics.

Saving only:

```text
CatBoost model
```

would not be sufficient.

Inference also needs:

```text
name vectorizer
address vectorizer
character name vectorizer
character address vectorizer
feature order
threshold
configuration
```

The deployment bundle therefore packages the full preprocessing/model state.

This prevents train/test preprocessing drift.

---

# Large-Scale Inference

The test candidate population is too large to safely materialize as one in-memory table.

The solution was a disk-backed inference architecture.

```text
Candidate partition
        |
        v
Entity enrichment
        |
        v
V4 feature matrix
        |
        v
CatBoost predict_proba
        |
        v
Threshold = 0.85
        |
        v
Positive pairs
        |
        v
Parquet batch
```

The production implementation uses:

```text
Candidate partitions: 64
Default batch size: 200,000
```

---

# Why Partitioning Was Necessary

A naive implementation would be:

```python
all_candidates = read_everything()
features = build_features(all_candidates)
predictions = model.predict(features)
```

At large scale this can require memory for:

- candidate DataFrames
- enriched DataFrames
- sparse TF-IDF matrices
- dense feature matrices
- probability arrays
- intermediate masks

The disk-backed approach instead performs:

```text
partition
    |
batch
    |
features
    |
prediction
    |
write
    |
free memory
    |
next batch
```

---

# Candidate Enrichment

Candidate pairs initially contain IDs.

The model requires actual entity information.

The inference pipeline therefore constructs:

```text
Source 1 lookup

source1_entity_id
    -> business_name
    -> business_address
    -> country
```

and:

```text
Candidate lookup

match_id
    -> business_name
    -> business_address
    -> country
```

Source 2 and Source 3 are combined into the candidate lookup.

---

# Lookup Validation

A missing address is allowed.

A missing entity name is treated differently because it can indicate an actual lookup failure.

Before feature generation, the inference pipeline checks for lookup failures.

This prevents silently generating predictions from incomplete enrichment.

---

# Smoke Testing

Before running the full candidate space, the inference pipeline supports a bounded smoke test.

The smoke test performs:

```text
Read a small candidate sample
        |
        v
Enrich entities
        |
        v
Build 18 features
        |
        v
CatBoost probabilities
        |
        v
Apply threshold
```

It verifies:

- candidate row count
- enriched row count
- feature count
- prediction count
- finite probabilities
- threshold
- positive prediction rate

This was introduced because a full inference run can be extremely expensive.

---

# Submission Construction

The final submission has two logically different files.

## `matching_results.tsv`

This contains final predicted matches.

Example:

```text
source1_entity_id    matched_entity_ids
S1-001               S2-019,S3-881
S1-002
S1-003               S2-441
```

Every Source-1 entity must appear exactly once.

Therefore the pipeline starts with the complete Source-1 population and left-joins predictions.

This guarantees that entities with no predicted matches remain in the output.

---

# `candidate_pairs.tsv`

This contains the final candidate set supplied to the matcher.

It is **not** the same thing as the predicted matches.

The relationship is:

```text
FINAL CANDIDATES
       |
       +--------------------------+
       |                          |
       v                          v
candidate_pairs.tsv         MODEL FEATURES
                                  |
                                  v
                              CATBOOST
                                  |
                                  v
                              THRESHOLD
                                  |
                                  v
                         matching_results.tsv
```

Therefore:

```text
predicted pairs ⊆ candidate pairs
```

must hold.

---

# Candidate Submission Contract

The submission construction validates:

- every Source-1 entity appears
- no duplicate Source-1 IDs
- no duplicate match IDs
- every match ID is valid
- every predicted match belongs to `candidate_pairs.tsv`

This separates:

```text
ML correctness
```

from:

```text
submission correctness
```

Both are necessary.

---

# Evaluation Methodology

The project contains two evaluation layers.

## Pair-Level Evaluation

Pair-level evaluation asks:

> Can the model classify candidate pairs correctly?

It computes:

- TP
- FP
- FN
- precision
- recall
- F0.5
- threshold curves

This is useful for model development.

---

## Entity-Level Evaluation

Entity-level evaluation asks:

> Does the complete system produce the correct match set for each Source-1 entity?

For every Source-1 entity:

```text
True match set
        |
        +---- Predicted match set
                    |
                    v
             Intersection
             Difference
                    |
                    v
                TP / FP / FN
                    |
                    v
               Entity F0.5
```

Then:

```text
Macro F0.5
=
mean(F0.5 across all Source-1 entities)
```

This is much closer to the actual competition objective.

---

# Why Pair-Level Metrics Can Mislead

Imagine:

```text
S1-A
True:      [X]
Predicted: [X]

S1-B
True:      []
Predicted: [Y]

S1-C
True:      [X, Z]
Predicted: [X]
```

When all pairs are pooled together, the classifier may still look excellent.

But entity-level evaluation sees:

- an incorrect singleton prediction
- a missed true match

This is why the final evaluation module keeps pair-level and entity-level metrics separate.

---

# Problems Faced and Solutions

## Problem 1: Brute-force matching was impossible

### Problem

The theoretical comparison space was approximately:

```text
11T+
```

comparisons.

### Solution

Introduce multi-pass blocking.

### Result

A development checkpoint reduced the population to:

```text
5,901,537 candidates
```

while recovering:

```text
25,267 / 27,907
```

known true pairs.

### Lesson

Retrieval comes before classification.

---

## Problem 2: One blocking strategy missed valid matches

### Problem

Names and addresses fail in different ways.

### Solution

Combine:

- exact name blocking
- name-token blocking
- rare-token blocking
- address-token blocking

through candidate union.

### Lesson

Entity-resolution retrieval benefits from independent evidence paths.

---

## Problem 3: Legal suffixes created artificial mismatches

### Problem

These can represent the same business:

```text
Acme Ltd
Acme Limited
Acme Pvt Ltd
Acme Private Limited
```

### Solution

Introduce legal suffix normalization.

### Lesson

Normalization should remove superficial differences without destroying useful identity information.

---

## Problem 4: Over-normalization can be dangerous

### Problem

It is tempting to build one "perfect" normalizer and use it everywhere.

### Why that is dangerous

Blocking and model features have different purposes.

Changing blocking normalization changes:

```text
candidate population
```

which changes:

```text
training distribution
inference distribution
recall ceiling
```

### Solution

Keep normalization contracts separate.

### Lesson

Preprocessing is part of the pipeline contract.

---

## Problem 5: The five-feature baseline was limited

### Problem

Basic edit/Jaccard similarity could not fully capture noisy text.

### Solution

Expand to 14 handcrafted features.

Then introduce:

```text
word TF-IDF
+
character TF-IDF
```

### Lesson

Feature representation can matter more than changing the classifier.

---

## Problem 6: Different classifiers behaved differently

### Problem

There was no reason to assume Logistic Regression would be sufficient.

### Solution

Run a model tournament.

Tested:

- Logistic Regression
- Random Forest
- XGBoost
- Extra Trees
- HistGradientBoosting
- MLP
- CatBoost
- LightGBM

### Lesson

Model selection should be evidence-driven.

---

## Problem 7: Word-level similarity missed some spelling variation

### Problem

Word-level TF-IDF can treat corrupted tokens as entirely different.

### Solution

Add character n-gram TF-IDF.

### Lesson

Character representations are particularly useful for noisy entity strings.

---

## Problem 8: TF-IDF preprocessing could not be refit during inference

### Problem

Refitting TF-IDF on test data would change the feature space.

### Solution

Package fitted vectorizers with the model.

Inference only performs:

```text
transform
```

not:

```text
fit
```

### Lesson

The preprocessing state is part of the model artifact.

---

## Problem 9: Test inference exceeded a simple in-memory design

### Problem

The final candidate population was too large to comfortably materialize at once.

### Solution

Use:

```text
64 candidate partitions
+
200,000-row batches
+
Parquet intermediate outputs
```

### Lesson

Large-scale ML inference becomes a data-engineering problem.

---

## Problem 10: Pair predictions had to become entity predictions

### Problem

The classifier produces:

```text
(S1, candidate)
```

pairs.

The submission requires:

```text
S1 -> list of candidates
```

### Solution

Separate:

```text
pair inference
```

from:

```text
entity aggregation
```

### Lesson

The prediction data structure and submission data structure are different.

---

## Problem 11: Singleton behavior is important

### Problem

An S1 entity with no true matches must remain unmatched.

### Solution

Start submission construction from the complete Source-1 population.

### Lesson

"Predict nothing" is itself a valid prediction.

---

## Problem 12: Pair-level validation can overstate real performance

### Problem

The development experiments produced very strong local pair-level metrics.

However, pair-level evaluation does not fully reproduce the entity-level competition objective.

### Solution

Implement both:

```text
pair-level evaluation
```

and:

```text
entity-level macro F0.5
```

and treat them as separate diagnostics.

### Lesson

Always evaluate at the level at which the final objective is defined.

---

## Problem 13: Validation and inference distributions can differ

A model can appear extremely strong when evaluated on a convenient sampled candidate population but behave differently when exposed to millions of realistic candidates.

This is especially relevant to threshold selection.

### Solution

Treat candidate distribution as part of validation design.

Future evaluation should use:

```text
held-out Source-1 entities
+
realistic candidate generation
+
hard negatives
+
entity-level macro F0.5
```

---

## Problem 14: Test distribution is not identical to training

The project identified country and formatting differences between development and test populations.

France is particularly relevant because it appears in the test distribution despite the development data being more concentrated around other countries.

### Risk

Hard-coded geography assumptions could cause:

```text
candidate loss
```

or:

```text
feature distribution shift
```

### Solution

Country is treated as an observed feature rather than assuming a closed list of countries.

---

## Problem 15: Too many notebooks accumulated

The project contains several notebooks because the work evolved iteratively.

The important notebooks were categorized into:

- research
- blocking
- model experiments
- CatBoost deployment
- CatBoost inference
- baseline submission

Instead of deleting the history, the repository refactor separates:

```text
experimental evidence
```

from:

```text
production code
```

---

# Important ML Lessons

## 1. Blocking determines the recall ceiling

If the true pair never reaches the classifier, the classifier cannot recover it.

Therefore candidate recall must be treated as a first-class metric.

---

## 2. Entity resolution is retrieval + classification

The system is not:

```text
text -> class
```

It is:

```text
records
  |
  v
retrieve plausible candidates
  |
  v
compare candidates
  |
  v
classify
  |
  v
aggregate
```

---

## 3. Feature representation matters

The progression:

```text
5 handcrafted
      |
      v
14 handcrafted
      |
      v
word TF-IDF
      |
      v
character TF-IDF
```

was a major improvement.

---

## 4. Threshold is part of the system

The threshold affects:

- precision
- recall
- singleton behavior
- candidate explosion
- entity-level F0.5

Therefore it must be versioned alongside the model.

---

## 5. Preprocessing is part of model state

The production artifact must include:

```text
model
+
vectorizers
+
feature ordering
+
threshold
+
configuration
```

---

## 6. Pair-level validation is not enough

A model can classify individual pairs well while producing poor entity-level match sets.

The final evaluation must match the competition objective.

---

## 7. Missing values are information

Instead of blindly dropping incomplete rows, the V2 representation explicitly includes missingness indicators.

---

## 8. Scaling changes architecture

When data is small:

```text
load -> process -> predict
```

may be sufficient.

At large scale:

```text
partition
-> batch
-> transform
-> predict
-> write
-> release
```

becomes necessary.

---

# Repository Architecture

The intended clean repository is:

```text
amazon-ml-entity-resolution/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── features.py
│   ├── models.py
│   ├── inference.py
│   ├── submission.py
│   └── evaluation.py
│
├── notebooks/
│   ├── 01_research_and_blocking.ipynb
│   ├── 02_candidate_generation.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_error_analysis.ipynb
│
├── results/
│   ├── model_comparison.csv
│   ├── blocking_results.csv
│   └── figures/
│
├── docs/
│   ├── methodology.md
│   └── architecture.md
│
└── output/
    ├── matching_results.tsv
    └── candidate_pairs.tsv
```

---

# Module Responsibilities

## `config.py`

Centralizes:

- project root
- train/test paths
- working directories
- output paths
- model artifact
- threshold
- feature count
- batch size
- partition count
- candidate partition pattern

---

## `preprocessing.py`

Contains:

- V1 normalization
- legal suffix normalization
- research normalization
- V4 model normalization
- V4 legal-name normalization
- length utilities

---

## `blocking.py`

Responsible for:

- blocking keys
- candidate retrieval
- name blocks
- address blocks
- candidate union
- candidate deduplication
- candidate partitioning
- disk-backed candidate generation

---

## `features.py`

Responsible for:

- handcrafted features
- exact-match features
- missingness features
- word TF-IDF
- character TF-IDF
- cosine similarity
- final 18-feature matrix
- feature safety checks

---

## `models.py`

Responsible for:

- loading the CatBoost deployment bundle
- validating the bundle
- validating feature ordering
- validating threshold
- exposing fitted vectorizers
- probability prediction
- thresholded prediction

---

## `inference.py`

Responsible for:

- test-source loading
- entity lookup
- candidate enrichment
- partition processing
- batch processing
- feature generation
- CatBoost inference
- probability filtering
- Parquet persistence
- memory release

---

## `submission.py`

Responsible for:

- positive prediction aggregation
- `matching_results.tsv`
- `candidate_pairs.tsv`
- S1 completeness
- duplicate detection
- candidate/prediction consistency

---

## `evaluation.py`

Responsible for:

### Pair level

- TP
- FP
- FN
- precision
- recall
- F0.5
- threshold sweep

### Entity level

- per-S1 TP
- per-S1 FP
- per-S1 FN
- per-S1 F0.5
- macro F0.5
- singleton diagnostics

---

# Reproducibility

The production inference contract is:

```text
Raw test data
      +
Final candidate partitions
      +
Frozen model bundle
      +
Frozen TF-IDF vectorizers
      +
Exact feature ordering
      +
Frozen threshold
      |
      v
Final predictions
```

The model artifact alone is not enough.

The candidate population and preprocessing state are also part of the reproducible system.

---

# Intended Reproduction Flow

Once repository integration is complete:

```bash
# Create environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt

# Compile package
python -m compileall src

# Run smoke tests
python -m ...

# Run candidate generation
python -m ...

# Run bounded inference
python -m ...

# Build submission
python -m ...

# Run official validator
python -m ...
```

The exact CLI commands will be finalized after the extracted modules pass integration testing.

---

# Experiment History

## Stage 1: Research

Goal:

```text
Understand data corruption
and identify useful signals.
```

---

## Stage 2: Blocking

Goal:

```text
Reduce the search space
without destroying recall.
```

Checkpoint:

```text
~11T+ theoretical comparisons
        |
        v
5.9M candidate pairs
        |
        v
90.54% blocking recall
```

---

## Stage 3: V1

Goal:

```text
Build a simple baseline.
```

Model:

```text
Logistic Regression
5 features
```

---

## Stage 4: V2

Goal:

```text
Increase representation quality.
```

Model input:

```text
14 handcrafted features
```

---

## Stage 5: V3

Goal:

```text
Add lexical text similarity.
```

Added:

```text
word TF-IDF
```

---

## Stage 6: V4

Goal:

```text
Capture character-level corruption.
```

Added:

```text
character TF-IDF
```

Result:

```text
18 features
```

---

## Stage 7: CatBoost Deployment

Goal:

```text
Freeze a reproducible model artifact.
```

Artifact:

```text
final_tuned_catboost_v4_depth10_deployment_bundle.joblib
```

---

## Stage 8: Test Inference

Goal:

```text
Run inference over the large candidate population.
```

Solution:

```text
64 partitions
+
bounded batches
+
Parquet intermediate outputs
```

---

## Stage 9: Submission

Goal:

```text
Convert pair predictions
into the exact submission format.
```

Outputs:

```text
matching_results.tsv
candidate_pairs.tsv
```

---

# Results

## Blocking

```text
Theoretical comparisons:  ~11T+
Candidate pairs:          5,901,537
True pairs:                  27,907
Recovered true pairs:        25,267
Blocking recall:             90.54%
```

---

## Model Development

| Version | Representation | Model | Pair-level F0.5 |
|---|---|---|---:|
| V1 | 5 handcrafted | Logistic Regression | 85.91% |
| V2 | 14 handcrafted | Logistic Regression | 86.78% |
| V2 | 14 handcrafted | Random Forest | 94.72% |
| V2 | 14 handcrafted | XGBoost | 94.83% |
| V2 | 14 handcrafted | XGBoost variant | 94.95% |
| V2 | 14 handcrafted | CatBoost | 94.81% |
| V3 | 14 + word TF-IDF | XGBoost | 98.57% |
| V4 | 14 + word + char TF-IDF | XGBoost | 98.85% |
| V4 | 14 + word + char TF-IDF | CatBoost | 98.87% |
| V4 selected | 18 features | CatBoost Depth 10 | **98.91%** |

### Important

These are **local pair-level validation results**.

They are not the official competition score.

---

# Current Repository Status

The project is currently in the **production-refactoring and reproducibility phase**.

The experimental system has been reconstructed into the following target modules:

```text
src/
├── config.py
├── preprocessing.py
├── blocking.py
├── features.py
├── models.py
├── inference.py
├── submission.py
└── evaluation.py
```

The major pipeline boundaries have been identified.

The next objective is to make those modules independently importable and end-to-end executable.

---

# Known Risks

## 1. `feature.py` vs `features.py`

The current repository contains a naming mismatch between the extracted feature module and the import used by inference.

The canonical production module should be:

```text
src/features.py
```

This must be corrected before calling the package fully integrated.

---

## 2. Blocking helper integration

The current blocking module expects helper functions that are not yet present in the extracted preprocessing module.

These must be recovered from the canonical blocking implementation rather than rewritten casually.

This is important because modifying the blocking logic could change candidate recall.

---

## 3. Repository paths

The original notebooks used notebook/environment-specific paths.

The portfolio repository should use repository-relative paths and a centralized configuration layer.

---

## 4. Model artifact location

The deployment bundle:

```text
final_tuned_catboost_v4_depth10_deployment_bundle.joblib
```

must either be:

- provided through an appropriate artifact/release mechanism, or
- documented as an externally supplied artifact.

---

## 5. Dependencies

The final `requirements.txt` must pin the versions used by the production pipeline.

The runtime includes packages such as:

```text
numpy
polars
scikit-learn
rapidfuzz
catboost
joblib
scipy
```

Exact versions should be frozen after the final runtime environment is validated.

---

## 6. Integration testing

Passing Python syntax checks is not sufficient.

The final system must be tested in this order:

```text
Compile
  |
  v
Import
  |
  v
Unit/smoke tests
  |
  v
Small candidate partition
  |
  v
Feature matrix
  |
  v
Model prediction
  |
  v
Prediction aggregation
  |
  v
Submission validation
```

Only after this chain succeeds should the repository be called fully reproducible.

---

# What Is Intentionally Not Production Code

The following should remain in notebooks or historical experiment records:

- temporary threshold sweeps
- debug cells
- exploratory dataframe dumps
- duplicate implementations
- abandoned models
- temporary path fixes
- one-off experiments
- generated candidate partitions
- local caches
- large intermediate datasets

The purpose of the refactor is not to erase the experimental history.

It is to separate:

```text
experimental evidence
```

from:

```text
validated production machinery
```

---

# Future Improvements

## 1. Source-1-level validation splits

Instead of:

```text
candidate pair -> train/validation
```

use:

```text
Source-1 entity
        |
        +---- training entities
        |
        +---- validation entities
```

This prevents leakage through repeated Source-1 entities.

---

## 2. Realistic candidate populations

Validation should reproduce the candidate environment encountered during inference.

Include:

- hard negatives
- realistic candidate density
- complete candidate blocks where feasible
- source-specific behavior

---

## 3. Optimize entity-level F0.5 directly

Threshold selection should be performed using:

```text
entity-level macro F0.5
```

rather than relying only on pair-level F0.5.

---

## 4. Singleton diagnostics

Track:

```text
singleton accuracy
singleton false-positive rate
singleton false-negative behavior
```

---

## 5. Candidate-density analysis

For every Source-1 entity, record:

```text
candidate count
true match count
predicted match count
```

This can reveal whether overmatching occurs in dense candidate regions.

---

## 6. Match-count stratification

Evaluate entities by:

```text
0 true matches
1 true match
2-3 true matches
4-6 true matches
7+ true matches
```

This helps separate singleton errors from multi-match errors.

---

## 7. Source-specific thresholds

Investigate whether Source 2 and Source 3 exhibit different noise characteristics.

If justified by validation, separate thresholds could be considered.

---

## 8. Global consistency

A future system could investigate:

```text
pair probabilities
        |
        v
global consistency
        |
        v
final match sets
```

This must account for the fact that an S1 entity can legitimately have multiple matches.

---

# Interview Explanation

A concise technical explanation of the project is:

> I worked on a business entity-resolution problem where records from multiple noisy sources had to be matched to a reference source. The first challenge was that brute-force comparison was computationally infeasible, so I developed multi-pass blocking using normalized names, name tokens and address tokens. This reduced the development candidate space to about 5.9 million pairs while recovering about 90.5% of known true pairs.
>
> I then developed the matcher iteratively. The baseline used five string-similarity features with Logistic Regression. I expanded that to fourteen handcrafted features and compared several classifiers including Random Forest, XGBoost, CatBoost, LightGBM and others. I then added word-level and character-level TF-IDF similarity, producing an 18-feature representation.
>
> The deployment branch used a Depth-10 CatBoost model with the frozen 18-feature representation and a threshold of 0.85. Because the final candidate population was too large for a simple in-memory workflow, I implemented partitioned, disk-backed inference using bounded batches and Parquet intermediate outputs.
>
> Finally, I separated pair-level inference from entity-level aggregation and constructed both `matching_results.tsv` and the exact `candidate_pairs.tsv` candidate population required by the challenge.
>
> One of the biggest lessons was that pair-level model performance does not automatically translate into competition performance. The actual objective is entity-level macro F0.5, so blocking recall, candidate distribution, singleton behavior, threshold selection and aggregation all matter.

---

# Project Takeaways

The most important lesson from the project is that the final system is not simply:

```text
CatBoost + fuzzy matching
```

It is:

```text
                    ENTITY RESOLUTION SYSTEM

                         Raw Records
                              |
                              v
                        Normalization
                              |
                              v
                     Candidate Retrieval
                              |
                              v
                     Blocking + Union
                              |
                              v
                       Candidate Pairs
                              |
                              v
                     Feature Engineering
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
          Similarity       Exactness       TF-IDF
              |               |               |
              +---------------+---------------+
                              |
                              v
                         CatBoost
                              |
                              v
                         Threshold
                              |
                              v
                      Positive Pair Set
                              |
                              v
                       Entity Aggregation
                              |
                              v
                         Submission
```

The project demonstrates:

- information retrieval
- entity resolution
- blocking
- string similarity
- feature engineering
- TF-IDF
- model comparison
- threshold tuning
- large-scale inference
- memory-aware processing
- artifact packaging
- evaluation design
- submission validation
- reproducibility

---

# Final Perspective

The project started as an investigation into:

> "How can two noisy business records be recognized as the same entity?"

It evolved into a much broader engineering problem:

> "How can we efficiently retrieve plausible matches, represent noisy records, classify millions of candidate pairs, aggregate those decisions correctly at entity level, and produce a reproducible submission?"

That distinction is the main story of this project.

The final architecture can be summarized as:

```text
RESEARCH
   |
   v
BLOCKING
   |
   v
CANDIDATE GENERATION
   |
   v
FEATURE ENGINEERING
   |
   v
MODEL EXPERIMENTATION
   |
   v
V4 CATBOOST
   |
   v
DEPLOYMENT BUNDLE
   |
   v
DISK-BACKED INFERENCE
   |
   v
ENTITY-LEVEL AGGREGATION
   |
   v
SUBMISSION
```

The goal of the repository refactor is to preserve that entire engineering story while turning the notebook-based implementation into a clean, reproducible Python project.

> **Retrieve carefully. Represent richly. Classify conservatively. Evaluate at the level that actually matters.**
