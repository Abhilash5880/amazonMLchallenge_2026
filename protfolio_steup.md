# Phase 1

### 1. Notebook-by-notebook inventory
- Research / Discovery
> notebook0910fa3ddf(5).ipynb

### 2. Candidate-generation engineering
> amazon-ml-challenge-2026-entity-resolution03-disk-backed(1).ipynb

### 3. ML experiments
> modelFineTuning_v4TF_IDF+XGB(1).ipynb

> for fine tuning_completed_.989(3).ipynb

### 4. CatBoost production branch
> final-catboost-deployment-kaggle(5).ipynb

### 5. Actual CatBoost inference
> FINAL_CATBOOST_INFERENCE_PATCHED.ipynb

### 6. Frozen V1 baseline
> amazonML_challenge_2026_FINAL_BASELINE_CLEAN(20260928-112931).ipynb

> amazonML_challenge_2026_FINAL_BASELINE_CLEAN(20260928-113011).ipynb

## Project Architectture After Phase 1

                    ┌──────────────────────┐
                    │  RESEARCH NOTEBOOK   │
                    │   notebook0910...    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ BLOCKING / ER PIPELINE│
                    │ entity-resolution03  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  MODEL EXPERIMENTS   │
                    │ TF-IDF + XGB + etc.  │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴──────────────┐
                 ▼                            ▼
       ┌──────────────────┐         ┌──────────────────┐
       │   XGB BRANCH     │         │  CATBOOST BRANCH │
       │ fine tuning      │         │ V4 deployment    │
       └──────────────────┘         └────────┬─────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │ CatBoost inference  │
                                  │ FINAL_PATCHED       │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                      TEST PREDICTIONS
                                             │
                                             ▼
                                      SUBMISSION FILES

# Phase 2: Experiment + Model Lineage

                    RESEARCH
                       │
                       ▼
              MULTI-PASS BLOCKING
                       │
                       ▼
              5.90M candidates
                 90.54% recall
                       │
                       ▼
                     V1
              LR + 5 features
                       │
                       ▼
                     V2
              14 engineered features
                       │
                       ▼
                     V3
                + word TF-IDF
                       │
                       ▼
                     V4
              + character TF-IDF
                       │
                       ▼
             CatBoost Depth 10
                  18 features
                 threshold .85
                       │
                       ▼
             DEPLOYMENT BUNDLE
                       │
                       ▼
              LOCAL INFERENCE
                       │
                       ▼
                SUBMISSION

# PHASE 3: Target Architecture/ Repo Structure
```
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
│   ├── config.py
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── features.py
│   ├── train.py
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
```
REPRODUCIBILITY AUDIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 Model artifact exists
🟢 TF-IDF preprocessing artifacts are preserved with V4 bundle
🟢 Test inference procedure exists
🟢 Candidate generation exists
🟢 Disk-backed strategy exists
🟢 Submission construction exists
🟢 Official validator exists

🟡 Pipeline logic is distributed across notebooks
🟡 Paths/configuration aren't yet centralized
🟡 Candidate artifacts need explicit provenance
🟡 Dependency versions need pinning
🟡 Exact end-to-end command sequence needs to be written

🔴 No clean standalone src/ pipeline yet
🔴 Another person cannot currently reproduce everything
    from a fresh clone without reconstructing notebook state
```

### Extract the proven machinery from the notebooks into a reproducible entity-resolution system, while preserving the notebooks as experimental evidence.

# Phase 4: Code Extraction Blueprint
```
Raw data
   │
   ▼
preprocessing.py
   │
   ▼
blocking.py
   │
   ▼
candidate pairs
   │
   ▼
features.py
   │
   ▼
model
   │
   ▼
inference.py
   │
   ▼
submission.py
   │
   ├── matching_results.tsv
   └── candidate_pairs.tsv
```

### Extraction Map :

| Existing notebook material | Final location            |
| -------------------------- | ------------------------- |
| normalization              | `src/preprocessing.py`    |
| blocking functions         | `src/blocking.py`         |
| candidate generation       | `src/blocking.py`         |
| V4 handcrafted features    | `src/features.py`         |
| TF-IDF transformation      | `src/features.py`         |
| CatBoost bundle loading    | `src/models.py`           |
| partitioned inference      | `src/inference.py`        |
| aggregation                | `src/submission.py`       |
| output validation          | `src/submission.py`       |
| pair-level metrics         | `src/evaluation.py`       |
| entity-level metrics       | `src/evaluation.py`       |
| thresholds/config          | `src/config.py`           |
| EDA                        | `notebooks/`              |
| model tournament           | `notebooks/` + `results/` |
| failed experiments         | `notebooks/` / report     |
| final figures              | `results/figures/`        |

### Final Structure :
```
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
```
PHASE 4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Production pipeline identified
✅ Notebook → module mapping established
✅ V4 feature pipeline identified
✅ CatBoost deployment boundary identified
✅ Disk-backed inference preserved
✅ Submission contract separated from inference
✅ Pair/entity evaluation separated

Next:
🔲 Extract actual functions from notebooks
🔲 Compare duplicate implementations
🔲 Decide canonical implementation for each function
🔲 Build src/
🔲 Run smoke tests against existing artifacts
```
# Phase 5: Actual code extraction

### 1. `amazon-ml-challenge-2026-entity-resolution03-disk-backed(1).ipynb`

> What we are extracting?
```
src/
├── config.py
├── preprocessing.py
├── blocking.py
├── features.py
├── evaluation.py
├── models.py
├── inference.py
└── submission.py
```
---
```
Source 1
   │
   ├── normalize name
   ├── normalize address
   └── normalize country
          │
          ▼
      exact / blocking keys
          │
     ┌────┼───────────────┐
     ▼    ▼               ▼
 exact  name-token    address-token
 block    block           block
     │    │               │
     └────┼───────────────┘
          ▼
       UNION
          │
          ▼
   candidate pairs
          │
          ▼
 candidate enrichment
          │
          ▼
       features
          │
          ▼
       matcher
```
