# Amazon ML Challenge 2026 — Business Entity Resolution

## Overview

A precision-first entity resolution pipeline that resolves business entities
across 3 noisy data sources using business names and addresses. The solution
uses hierarchical blocking, multi-signal feature engineering, and a LightGBM
classifier with singleton detection and graph consistency checks.

## Architecture

```
Raw Data → Preprocessing → Hierarchical Blocking → Feature Extraction
→ LightGBM Matching → Singleton Detection → Graph Consistency → Output
```

### Key Differentiators
1. **4-Layer Hierarchical Blocking** (token, phonetic, MinHash LSH, FAISS ANN)
2. **35+ Similarity Features** (name, address, combined, meta)
3. **Margin-Based Confidence Gating** — when in doubt, don't merge
4. **Graph Consistency** — resolves conflicting multi-source matches
5. **Country-Agnostic Design** — handles unseen countries (France)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train & Predict (End-to-End)

```bash
cd code/business_entity_resolution/src
python pipeline.py --mode full
```

### 3. Train Only (with validation)

```bash
python pipeline.py --mode train --val-ratio 0.2
```

### 4. Predict Only (using saved model)

```bash
python pipeline.py --mode predict
```

### 5. Evaluate

```bash
python evaluate.py --predictions ../../output/matching_results.tsv \
                    --ground-truth ../../dataset/student_resource/student_resource/dataset/train/train_ground_truth.tsv
```

### 6. Validate Submission Format

```bash
python ../../dataset/student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/student_resource/student_resource/dataset/test
```

## Output Files

- `output/matching_results.tsv` — Final entity matches (scored on leaderboard)
- `output/candidate_pairs.tsv` — Blocking candidate set (reviewed for blocking quality)
- `output/matcher_model.pkl` — Trained LightGBM model
- `output/threshold.txt` — Optimized match threshold
- `output/pipeline.log` — Full execution log

## Project Structure

```
code/business_entity_resolution/
├── src/
│   ├── config.py          # Paths, constants, normalization dictionaries
│   ├── preprocessing.py   # Business name & address normalization
│   ├── blocking.py        # 4-layer hierarchical candidate generation
│   ├── features.py        # 35+ similarity feature computation
│   ├── matcher.py         # LightGBM matching model + margin gating
│   ├── graph_consistency.py # Bipartite graph conflict resolution
│   ├── evaluate.py        # F₀.₅ scoring and validation split
│   └── pipeline.py        # End-to-end orchestrator (main entry point)
├── requirements.txt
└── README.md
```

## Methodology

See `Documentation_template.md` in the submission root for the full methodology
write-up including blocking recall analysis, feature importance, and threshold
tuning results.
