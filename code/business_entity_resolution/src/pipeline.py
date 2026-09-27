"""
pipeline.py — Scalable End-to-End Entity Resolution Pipeline

Fixes vs v1:
  - Wires in margin-based gating via generate_matching_results()
  - Replaces iterrows()-based lookup construction with dict-from-zip (O(n) not O(n²))
  - Vectorized TF-IDF lookup via pre-built index arrays
  - Handles 2M+ S1, 10M+ S2/S3 records
"""

import sys
import os
import argparse
import logging
import time
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from preprocessing import preprocess_dataframe
from blocking import run_blocking, measure_blocking_recall, candidates_to_tsv
from features import compute_pair_features
from matcher import (
    EntityMatcher, generate_matching_results, results_to_tsv,
    apply_margin_gating,
)
from graph_consistency import apply_graph_consistency
from evaluate import load_ground_truth, macro_f05, split_validation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fast row lookup (NO iterrows!)
# ---------------------------------------------------------------------------

def _build_row_lookup(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build {entity_id: row} dict WITHOUT iterrows.

    Uses positional indexing which is ~100x faster than iterrows on large DFs.
    """
    eid_col = df["entity_id"].values
    lookup: dict[str, int] = {}
    for i, eid in enumerate(eid_col):
        lookup[str(eid)] = i
    return df, lookup  # return df + index map; access via df.iloc[idx]


# ---------------------------------------------------------------------------
# Training pair generation
# ---------------------------------------------------------------------------

def create_training_pairs(
    candidates: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    max_neg_ratio: int = 5,
) -> list[tuple[str, str, int]]:
    random.seed(cfg.RANDOM_SEED)
    pairs = []
    for s1_id, cands in candidates.items():
        if not cands:
            continue
        true_matches = ground_truth.get(s1_id, set())
        positives = cands & true_matches
        negatives = list(cands - true_matches)

        for cid in positives:
            pairs.append((s1_id, cid, 1))

        n_neg = min(len(negatives), max(len(positives) * max_neg_ratio, 3))
        if len(negatives) > n_neg:
            negatives = random.sample(negatives, n_neg)
        for cid in negatives:
            pairs.append((s1_id, cid, 0))

    n_pos = sum(1 for _, _, l in pairs if l == 1)
    n_neg = sum(1 for _, _, l in pairs if l == 0)
    logger.info("Training pairs: %d total (%d pos, %d neg, ratio 1:%.1f)",
                len(pairs), n_pos, n_neg, n_neg / max(n_pos, 1))
    return pairs


# ---------------------------------------------------------------------------
# Feature extraction (fast — no iterrows)
# ---------------------------------------------------------------------------

def extract_features_for_pairs(
    pairs: list[tuple[str, str]],
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    layer_counts: dict[str, dict[str, int]],
    candidates: dict[str, set[str]],
) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame()

    logger.info("Extracting features for %d pairs...", len(pairs))

    # Build fast index lookup (NOT iterrows)
    s1_eid = s1_df["entity_id"].astype(str).values
    s23_eid = s23_df["entity_id"].astype(str).values
    s1_idx_map = {eid: i for i, eid in enumerate(s1_eid)}
    s23_idx_map = {eid: i for i, eid in enumerate(s23_eid)}

    feature_rows = []
    skipped = 0
    for s1_id, cid in tqdm(pairs, desc="Features", mininterval=10):
        s1_i = s1_idx_map.get(s1_id)
        s23_i = s23_idx_map.get(cid)
        if s1_i is None or s23_i is None:
            feature_rows.append({})
            skipped += 1
            continue

        s1_row = s1_df.iloc[s1_i]
        cand_row = s23_df.iloc[s23_i]

        meta = {
            "blocking_layers_count": layer_counts.get(s1_id, {}).get(cid, 1),
            "num_candidates": len(candidates.get(s1_id, set())),
        }
        feats = compute_pair_features(s1_row, cand_row, meta)
        feature_rows.append(feats)

    if skipped:
        logger.warning("Skipped %d pairs (missing entity IDs)", skipped)

    return pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# Pipeline Class
# ---------------------------------------------------------------------------

class EntityResolutionPipeline:
    def __init__(self):
        cfg.setup_logging(log_file=True)
        cfg.ensure_directories()
        self.matcher = EntityMatcher()
        self.model_path = str(cfg.MODEL_PATH)
        self.s1_df = None
        self.s23_df = None
        self.ground_truth = None

    def load_data(self, mode: str):
        logger.info("Loading data (mode=%s)...", mode)
        if mode in ("train", "full"):
            self.s1_df = pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t", dtype=str)
            s2 = pd.read_csv(cfg.TRAIN_SOURCE2, sep="\t", dtype=str)
            s3 = pd.read_csv(cfg.TRAIN_SOURCE3, sep="\t", dtype=str)
            self.s23_df = pd.concat([s2, s3], ignore_index=True)
            del s2, s3
            self.ground_truth = load_ground_truth(cfg.TRAIN_GROUND_TRUTH)
        else:
            self.s1_df = pd.read_csv(cfg.TEST_SOURCE1, sep="\t", dtype=str)
            s2 = pd.read_csv(cfg.TEST_SOURCE2, sep="\t", dtype=str)
            s3 = pd.read_csv(cfg.TEST_SOURCE3, sep="\t", dtype=str)
            self.s23_df = pd.concat([s2, s3], ignore_index=True)
            del s2, s3
        logger.info("Loaded S1=%d, S23=%d", len(self.s1_df), len(self.s23_df))

    def preprocess(self):
        logger.info("Preprocessing S1 (%d)...", len(self.s1_df))
        self.s1_df = preprocess_dataframe(self.s1_df)
        logger.info("Preprocessing S23 (%d)...", len(self.s23_df))
        self.s23_df = preprocess_dataframe(self.s23_df)
        logger.info("Preprocessing done.")

    def train(self, val_ratio=0.2):
        start = time.time()
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE")
        logger.info("=" * 60)

        self.load_data("train")
        self.preprocess()

        train_gt, val_gt = split_validation(self.ground_truth, val_ratio, cfg.RANDOM_SEED)

        # Blocking
        candidates, layer_counts = run_blocking(
            self.s1_df, self.s23_df, max_candidates=cfg.BLOCKING_MAX_CANDIDATES,
        )
        metrics = measure_blocking_recall(candidates, self.ground_truth)
        logger.info("Blocking recall: %.4f (%d/%d)",
                     metrics["blocking_recall"],
                     metrics["captured_pairs"],
                     metrics["total_true_pairs"])

        # Training pairs
        train_triples = create_training_pairs(candidates, train_gt)
        val_triples = create_training_pairs(candidates, val_gt)

        # Features
        train_pairs = [(s1, c) for s1, c, _ in train_triples]
        train_labels = np.array([l for _, _, l in train_triples])
        train_features = extract_features_for_pairs(
            train_pairs, self.s1_df, self.s23_df, layer_counts, candidates
        )

        val_pairs = [(s1, c) for s1, c, _ in val_triples]
        val_labels = np.array([l for _, _, l in val_triples])
        val_features = pd.DataFrame()
        if val_triples:
            val_features = extract_features_for_pairs(
                val_pairs, self.s1_df, self.s23_df, layer_counts, candidates
            )

        # Train model
        X_train = train_features.fillna(0.0)
        X_val = val_features.fillna(0.0) if len(val_features) > 0 else None
        y_val = val_labels if len(val_triples) > 0 else None

        self.matcher.train(X_train, train_labels, X_val, y_val)

        # Find optimal threshold
        opt_threshold = cfg.MATCH_THRESHOLD
        if X_val is not None and len(X_val) > 0:
            opt_threshold = self.matcher.find_optimal_threshold(
                X_val.values, y_val, val_gt,
                [s1 for s1, _, _ in val_triples],
                [c for _, c, _ in val_triples],
            )
            logger.info("Optimal threshold: %.3f", opt_threshold)

        # Evaluate on validation WITH margin gating
        if val_triples:
            val_probs = self.matcher.predict_proba(X_val.values)

            # Build predictions DataFrame for generate_matching_results
            pred_df = pd.DataFrame({
                "source1_entity_id": [s1 for s1, _, _ in val_triples],
                "candidate_entity_id": [c for _, c, _ in val_triples],
                "match_prob": val_probs,
            })

            all_val_s1 = list(val_gt.keys())
            val_results = generate_matching_results(
                pred_df, all_val_s1,
                threshold=opt_threshold,
                margin_threshold=cfg.MARGIN_THRESHOLD,
            )

            # Graph consistency
            match_scores = {(row["source1_entity_id"], row["candidate_entity_id"]): row["match_prob"]
                           for _, row in pred_df.iterrows()}
            val_results = apply_graph_consistency(val_results, match_scores)

            f05 = macro_f05(val_results, val_gt)
            logger.info("=" * 60)
            logger.info("VALIDATION F0.5: %.6f", f05)
            logger.info("=" * 60)

        # Save
        self.matcher.save(self.model_path)
        with open(str(cfg.THRESHOLD_PATH), "w") as f:
            f.write(str(opt_threshold))

        logger.info("Training done in %.1f seconds", time.time() - start)

    def predict(self):
        start = time.time()
        logger.info("=" * 60)
        logger.info("PREDICTION PIPELINE")
        logger.info("=" * 60)

        self.load_data("predict")
        self.preprocess()
        self.matcher.load(self.model_path)

        threshold = cfg.MATCH_THRESHOLD
        if os.path.exists(str(cfg.THRESHOLD_PATH)):
            with open(str(cfg.THRESHOLD_PATH)) as f:
                threshold = float(f.read().strip())

        # Blocking
        candidates, layer_counts = run_blocking(
            self.s1_df, self.s23_df, max_candidates=cfg.BLOCKING_MAX_CANDIDATES,
        )
        candidates_to_tsv(candidates, str(cfg.CANDIDATE_PAIRS_PATH))

        # Flatten pairs
        pairs = [(s1, c) for s1, cands in candidates.items() for c in cands]
        logger.info("Total candidate pairs to score: %d", len(pairs))

        if not pairs:
            results = {str(eid): set() for eid in self.s1_df["entity_id"].values}
            results_to_tsv(results, str(cfg.MATCHING_RESULTS_PATH))
            return

        # Features
        features = extract_features_for_pairs(
            pairs, self.s1_df, self.s23_df, layer_counts, candidates
        )

        # Predict
        X = features.fillna(0.0).values
        probs = self.matcher.predict_proba(X)

        # Build predictions DataFrame and use margin gating
        pred_df = pd.DataFrame({
            "source1_entity_id": [s1 for s1, _ in pairs],
            "candidate_entity_id": [c for _, c in pairs],
            "match_prob": probs,
        })

        all_s1 = [str(eid) for eid in self.s1_df["entity_id"].values]
        results = generate_matching_results(
            pred_df, all_s1,
            threshold=threshold,
            margin_threshold=cfg.MARGIN_THRESHOLD,
        )

        # Graph consistency
        match_scores = {(pairs[i][0], pairs[i][1]): float(probs[i]) for i in range(len(pairs))}
        results = apply_graph_consistency(results, match_scores)

        # Write output
        results_to_tsv(results, str(cfg.MATCHING_RESULTS_PATH))

        n_matched = sum(1 for v in results.values() if v)
        n_singleton = sum(1 for v in results.values() if not v)
        logger.info("RESULTS: %d matched, %d singletons, %d total links",
                     n_matched, n_singleton, sum(len(v) for v in results.values()))
        logger.info("Prediction done in %.1f seconds", time.time() - start)

    def train_and_predict(self):
        self.train()
        self.predict()


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["train", "predict", "full"], default="full")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    pipeline = EntityResolutionPipeline()
    if args.mode == "train":
        pipeline.train(args.val_ratio)
    elif args.mode == "predict":
        pipeline.predict()
    else:
        pipeline.train_and_predict()


if __name__ == "__main__":
    main()
