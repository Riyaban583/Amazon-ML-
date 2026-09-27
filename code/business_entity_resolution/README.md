# Amazon Business Entity Resolution Pipeline

This repository contains the complete, production-grade, end-to-end Machine Learning pipeline for the Amazon Business Entity Resolution Challenge.

## Team Contributions & Architecture Alignment

The pipeline is modularly organized according to the competition roles:

| Member | Focus Area | Core Responsibilities & Modules |
| :--- | :--- | :--- |
| **Riya** | **Data & Scalable Blocking** | `src/preprocess.py`, `src/blocking.py`: Multi-country text normalization, legal suffix removal, address expansion, country partitioning, multi-method inverted indexing, candidate reduction, and generation of `output/candidate_pairs.tsv`. |
| **Khushi** | **Matching Features & Similarity** | `src/features.py`: Dense string similarity metrics (Levenshtein, Jaro-Winkler, Token Sort/Set ratios, Char 3-gram & Word Jaccard), address numbers & postal code overlap, and cross-field signals. |
| **Akshat** | **ML, Evaluation & Prediction** | `src/model.py`, `src/evaluate.py`, `src/pipeline.py`: LightGBM / Gradient Boosting classifier, macro-averaged $F_{0.5}$ metric computation, singleton threshold optimization, test inference, generation of `output/matching_results.tsv`, and submission validation. |

---

## Directory Structure

```text
├── output/
│   ├── matching_results.tsv        # Final entity matches (leaderboard submission)
│   └── candidate_pairs.tsv         # Blocking candidate set (candidate generation audit)
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py           # Paths and hyperparameters
│       │   ├── preprocess.py       # Riya's text cleaning & abbreviation expansion
│       │   ├── blocking.py         # Riya's scalable candidate generation
│       │   ├── features.py         # Khushi's feature extraction engine
│       │   ├── model.py            # Akshat's ML classification model
│       │   ├── evaluate.py         # Akshat's macro F_0.5 metric & threshold optimizer
│       │   └── pipeline.py         # End-to-end pipeline orchestrator
│       ├── README.md               # Reproduction guide
│       └── requirements.txt        # Pinned dependencies
├── utils/
│   └── validate_submission.py      # Standalone submission validator
├── Documentation_template.md       # Detailed technical methodology report
└── run_pipeline.py                 # One-click execution script
```

---

## Installation & Setup

1. **Prerequisites**: Python 3.9+ (tested on Python 3.10 - 3.14).
2. **Install Dependencies**:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## How to Reproduce End-to-End

### Step 1: Place Competition Dataset
Ensure your dataset files are placed in the `dataset/` directory:
```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

*(Note: If testing without raw files, you can generate synthetic benchmark data anytime by running `python generate_sample_data.py`)*

### Step 2: Run End-to-End Pipeline
Execute the master runner:
```bash
python run_pipeline.py
```
This single command automatically:
1. Loads training data across US and India.
2. Runs Riya's blocking module to extract candidate pairs with high reduction ratio.
3. Computes Khushi's 21+ dense similarity features.
4. Trains Akshat's ML model with class-balancing.
5. Optimizes thresholds on validation splits for the competition's macro-averaged $F_{0.5}$ metric (specifically handling singletons).
6. Runs inference on the test set (including unseen countries like France).
7. Generates `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
8. Automatically runs `utils/validate_submission.py` to confirm 100% compliance with submission criteria.

### Step 3: Run Standalone Submission Validation
You can independently verify your submission files at any time:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
A successful validation outputs `PASS` with exit code `0`.

---

## Key Algorithmic Strengths

1. **Ultra-Scalable Blocking ($>99.9\%$ Reduction)**:
   - Partitioning by country drastically reduces candidate search space.
   - Token inverted index on discriminative terms and n-grams limits candidate sets to a small, high-quality pool per Source 1 entity (crucial for Amazon's ranking criteria).
2. **Open-Set Country Generalization**:
   - The test set introduces **France** (unseen in training data). The pipeline handles countries as open string sets and normalizes French legal suffixes (`SA`, `SARL`, `SAS`) and diacritics (`é, è, ç`).
3. **Precision-Weighted Optimization for Singletons**:
   - $F_{0.5}$ penalizes false positives twice as heavily as false negatives.
   - Predicting any false match on a singleton entity drops its score from 1.0 to 0.0. The pipeline enforces a high confidence barrier ($\tau_{\text{singleton}} \ge 0.60$) before declaring any match, preserving maximal singleton reward.
