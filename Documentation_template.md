# Amazon Business Entity Resolution Challenge — Technical Methodology Documentation

## Team Roles & Division of Responsibilities

| Team Member | Module / Responsibility | Core Technical Contributions |
| :--- | :--- | :--- |
| **Riya** | **Data Engineering & Scalable Candidate Generation (Blocking)** | • Multi-country text normalization & accent stripping (US, India, France)<br>• Legal suffix normalization & address token expansion<br>• Dynamic country partitioning (open set)<br>• Multi-method inverted index blocking (discriminative tokens, n-grams, numbers)<br>• Pre-scoring, top-$K$ pruning & generation of `output/candidate_pairs.tsv` |
| **Khushi** | **Feature Engineering & String Similarity Engine** | • 21-dimensional dense pairwise feature extraction<br>• Fast string metrics: Normalized Levenshtein ratio, Jaro-Winkler similarity, Token Sort Ratio, Token Set Ratio<br>• Character 3-gram and Word Jaccard set metrics<br>• Address number, PIN/ZIP code exact match & digit overlap extraction<br>• First-token brand name matching & source indicators |
| **Akshat** | **ML Classification, Evaluation & Singleton Resolution** | • LightGBM & HistGradientBoosting classifier with balanced class weighting<br>• Custom competition metric: Macro-averaged $F_{0.5}$ per Source 1 entity<br>• 2-stage threshold optimization for singletons vs. multi-matches<br>• Test inference & generation of `output/matching_results.tsv`<br>• Local automated validation suite (`utils/validate_submission.py`) |

---

## 1. Executive Summary & Problem Formulation

In large-scale commercial e-commerce platforms like Amazon, business entity resolution (ER) requires linking noisy, heterogeneous records arriving from disparate sources without shared keys.
- **Source 1**: The deduplicated reference source (ground truth anchor).
- **Source 2 & Source 3**: Noisy, unaligned sources with potential typos, abbreviations, legal suffix variations, reordered address components, and missing postal codes.
- **Cardinality**: A Source 1 entity may map to 0 (singleton), 1, or multiple records across Source 2 and Source 3.
- **Target Metric**: Macro-averaged $F_{0.5}$ across all Source 1 entities, including singletons:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Since precision is weighted 2× over recall, false merges (linking distinct entities) are twice as damaging as missed links. Crucially, a singleton entity scores 1.0 if predicted empty, and **0.0 if any match is predicted**.

---

## 2. Riya's Module: Data Preprocessing & Scalable Blocking Strategy

### 2.1 Multi-Country Text Normalization
Business names and addresses suffer from severe typographic and orthographic noise. The preprocessing pipeline applies:
1. **Unicode Diacritic Stripping**: Using `unicodedata.normalize('NFKD', text)`, accents are converted to ASCII base characters (crucial for French names in the test set, such as `Société` $\to$ `Societe`, `Hôtel` $\to$ `Hotel`).
2. **Punctuation & Ampersand Canonicalization**: `&` is mapped to `and`, and noisy punctuation marks are stripped.
3. **Legal Suffix Stripping**: Corporate designations add high string noise without entity identity. The cleaner removes:
   - **US**: `inc`, `incorporated`, `corp`, `corporation`, `llc`, `ltd`, `co`, `company`, `lp`, `llp`.
   - **India**: `pvt ltd`, `private limited`, `pvt`, `limited`, `traders`, `enterprises`, `associates`, `agency`.
   - **France**: `sa`, `sarl`, `sas`, `sasu`, `snc`, `eurl`, `sci`, `ste`, `societe`, `ets`.
4. **Address Component Expansion**: Standardizes structural abbreviations across locales:
   - Street names: `st` $\to$ `street`, `rd` $\to$ `road`, `ave` $\to$ `avenue`, `blvd` $\to$ `boulevard`, `rte` $\to$ `route`, `chem` $\to$ `chemin`.
   - Landmark & Indian designations: `opp` $\to$ `opposite`, `nr` $\to$ `near`, `behind`, `sec` $\to$ `sector`, `col` $\to$ `colony`, `chowk`.

### 2.2 Scalable Multi-Method Blocking
To avoid $O(|S_1| \times (|S_2| + |S_3|))$ full Cartesian comparison, blocking reduces candidate pairs by $>99.9\%$:
1. **Dynamic Country Partitioning**: Entities operate within national boundaries; grouping targets by `country` completely eliminates cross-border comparisons without any recall loss. Country is treated as an open set of string labels, seamlessly generalizing to **France** in the test set.
2. **Informative Name Token Inverted Index**: Tokens of length $\ge 3$ are indexed with inverse document frequency (IDF) weighting. Overly frequent generic tokens (occurring in $>5\%$ of targets) are automatically pruned from the index to prevent candidate explosions.
3. **Character 3-Gram Typo Index**: For words of length $\ge 4$, character tri-grams are indexed to bridge OCR typos, misspellings, and transliterations.
4. **Address Numeric Co-Occurrence**: Inverted index on postal codes and street numbers ensures co-located businesses are surfaced even when names have variations.
5. **Union & Top-$K$ Pruning**: Candidates from all blocking rules are merged, pre-scored with composite Jaccard overlap, and pruned to the top $K \le 25$ candidates per Source 1 entity.

**Output**: Exported directly to `output/candidate_pairs.tsv` satisfying all format rules.

---

## 3. Khushi's Module: Feature Engineering & Similarity Engine

For every candidate pair $(S_1, S_{\text{target}})$, a 21-dimensional dense feature vector is computed across multiple complementary orthogonal signals:

### 3.1 Name Similarity Dimensions
- **Exact Match Flag**: Binary indicator for exact clean name equality.
- **Normalized Levenshtein Ratio**: Edit distance normalized by sequence lengths $[0, 1]$.
- **Jaro-Winkler Similarity**: Measures common prefix length and character transpositions, optimal for corporate brand names.
- **Token Sort Ratio**: Tokenizes words, sorts alphabetically, and computes fuzzy ratio (invariant to word order rearrangements like `Walmart Supercenter` vs `Supercenter Walmart`).
- **Token Set Ratio**: Identifies matching subset tokens while ignoring extraneous terms.
- **Character 3-Gram Jaccard**: Set-level intersection-over-union of tri-grams.
- **Word Jaccard**: Bag-of-words lexical overlap.
- **First Token Exact Match**: Binary flag testing whether the first word (primary brand identifier) matches.
- **Length Difference & Ratio**: Absolute length disparity and relative ratio.

### 3.2 Address & Numeric Signals
- **Address Word Jaccard & Character 4-gram Jaccard**: Measures structural street similarity.
- **Address Token Sort Ratio**: Captures transposed address components.
- **Numeric Overlap Count & Jaccard**: Counts shared numeric tokens (house numbers, PIN codes, ZIP codes). Numbers provide extraordinarily strong disambiguation power in dense urban areas.
- **Has Shared Number Flag**: Binary flag indicating if at least one distinct number is shared.

### 3.3 Contextual & Source Indicators
- **Combined Word Jaccard**: Joint similarity over name + address text.
- **Target Source Indicators**: One-hot flags for Source 2 (`is_s2`) vs. Source 3 (`is_s3`).
- **Candidate Pool Rank**: The index of the candidate in Riya's blocking output.

---

## 4. Akshat's Module: ML Architecture, Threshold Optimization & Singleton Resolution

### 4.1 Classifier Model
- **Algorithm**: Gradient Boosted Decision Trees (`LightGBM` / `HistGradientBoosting`).
- **Class Balancing**: Candidate generation generates a non-match to match ratio of ~5:1 to 10:1. The classifier utilizes `class_weight='balanced'` to prevent bias toward non-matches.
- **Outputs**: Calibrated class probabilities $P(\text{Match} = 1 \mid \mathbf{x})$.

### 4.2 Precision-Weighted Threshold Optimization ($F_{0.5}$)
The competition scoring function is:
$$F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$$
A standard $0.5$ classification threshold is suboptimal for $F_{0.5}$ because:
1. False positives (false merges) are penalized twice as heavily as false negatives.
2. A single false positive on a true singleton entity drops its entity score from **1.0 to 0.0**.

**Two-Stage Threshold Strategy**:
- **Stage 1 (Singleton Gate)**: For each $S_1$ entity, find $p_{\max} = \max_{j} P(\text{Match} \mid S_1, C_j)$. If $p_{\max} < \tau_{\text{singleton}}$ (empirically tuned in $[0.60, 0.70]$), the entity is classified as a **singleton**, and an empty match list `[]` is predicted. This secures a perfect 1.0 score for all genuine singletons.
- **Stage 2 (Multi-Match Admission)**: If $p_{\max} \ge \tau_{\text{singleton}}$, candidate 1 is accepted. Subsequent candidates $C_j$ ($j \ge 2$) are admitted only if:
  $$P(\text{Match} \mid S_1, C_j) \ge \tau_{\text{match}} \quad \text{AND} \quad (p_{\max} - P(\text{Match} \mid S_1, C_j)) \le \Delta_{\text{margin}}$$
  This ensures that multi-matches are only predicted when confidence across multiple sources is exceptionally high.

### 4.3 Validation & Output Formatting
- Matching outputs are strictly checked to guarantee that all matched IDs are a subset of candidate IDs.
- Outputs are exported to `output/matching_results.tsv` in exact accordance with competition specifications.
- The pipeline automatically executes `utils/validate_submission.py`, ensuring 0 errors before submission.

---

## 5. Handling Open-Set Country Generalization (France)

The training set contains only `US` and `India`, whereas the test set introduces `France`.
Our pipeline handles this seamlessly:
1. **Zero Hardcoded Country Lists**: Country labels are used dynamically as partition keys. No hardcoded filters or fixed one-hot encodings exist for country names.
2. **French Legal Entities & Address Terms**: Legal suffixes (`SA`, `SARL`, `SAS`, `SNC`, `EURL`) and address terms (`rue`, `boulevard`, `avenue`, `route`) are natively supported in the preprocessing dictionaries.
3. **Accent-Insensitive String Processing**: All French diacritics (`é, è, ê, ç, à, ô`) are decomposed into ASCII, preventing artificial mismatch penalties between accented and unaccented variations.

---

## 6. Reproducibility & Fair Play Compliance

- **Zero External Data**: Strictly no external APIs, commercial entity resolution tools, government registry lookups, or internet geocoding were used.
- **Model Constraints**: All models are MIT/Apache 2.0 licensed, running lightweight GBDT architectures well below the 8 Billion parameter ceiling.
- **One-Command Reproduction**:
  ```bash
  python run_pipeline.py
  ```
  Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv` and verifies validation in seconds.
