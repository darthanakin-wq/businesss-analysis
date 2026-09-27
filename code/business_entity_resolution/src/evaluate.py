"""Evaluation module for Amazon ML Challenge 2026: Business Entity Resolution.

This module computes the macro-averaged F0.5 score for entity resolution,
accommodating singletons (records with no matching entities in other sources)
and rewarding precision twice as heavily as recall.

Core functions:
- f05_per_entity: Computes F0.5 for an individual entity pair set.
- macro_f05: Computes macro-averaged F0.5 across all entities.
- load_ground_truth: Reads train_ground_truth.tsv into a dict.
- load_predictions: Reads matching_results.tsv into a dict.
- evaluate_submission: Computes F0.5 and comprehensive diagnostic metrics.
- split_validation: Performs stratified train/validation split.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def f05_per_entity(predicted_ids: set[str], true_ids: set[str]) -> float:
    """Compute the F0.5 score for a single Source 1 entity.

    Scoring rules:
    - If both predicted and true match sets are empty (true singleton, predicted singleton): return 1.0
    - If true is empty but predicted is not (false merge on singleton): return 0.0
    - If true is not empty but predicted is empty (missed all matches): return 0.0
    - Otherwise:
        precision = |predicted ∩ true| / |predicted|
        recall = |predicted ∩ true| / |true|
        F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
        Handle division by zero cleanly (e.g. disjoint non-empty sets -> 0.0).

    Args:
        predicted_ids: Set of predicted match entity IDs (from Source 2 / Source 3).
        true_ids: Set of ground truth match entity IDs.

    Returns:
        F0.5 score in range [0.0, 1.0].
    """
    pred_set = set(predicted_ids) if predicted_ids is not None else set()
    true_set = set(true_ids) if true_ids is not None else set()

    # Case 1: Both sets empty (correctly identified singleton)
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0

    # Case 2: True singleton, but predicted matches (false merge on singleton)
    if len(true_set) == 0 and len(pred_set) > 0:
        return 0.0

    # Case 3: True matches exist, but predicted empty (missed all matches)
    if len(true_set) > 0 and len(pred_set) == 0:
        return 0.0

    # Case 4: Both non-empty
    common = pred_set & true_set
    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(true_set)

    denominator = 0.25 * precision + recall
    if denominator == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def macro_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> float:
    """Compute macro-averaged F0.5 score across all entities in ground truth.

    Every entity present in ground_truth is evaluated. If an entity is missing
    from predictions, it is evaluated as an empty set (i.e. predicted singleton).

    Args:
        predictions: Mapping {source1_entity_id: set(matched_entity_ids)}.
        ground_truth: Mapping {source1_entity_id: set(matched_entity_ids)}.

    Returns:
        Macro-averaged F0.5 score across all ground truth entities.
    """
    if not ground_truth:
        logger.warning("ground_truth dictionary is empty. Returning 0.0.")
        return 0.0

    total_score = 0.0
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        total_score += f05_per_entity(pred_ids, true_ids)

    return total_score / len(ground_truth)


def _load_mapping_tsv(path: str | Path) -> dict[str, set[str]]:
    """Internal helper to load a tab-separated file mapping Source 1 IDs to matched ID sets.

    Handles NaN and empty values as empty sets.

    Args:
        path: Path to the TSV file.

    Returns:
        Dictionary mapping source1_entity_id to set of matched_entity_ids.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns cannot be determined.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.debug("Loading TSV mapping from %s", file_path)
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=True)

    # Normalize column names
    col_lookup = {str(c).strip().lower(): c for c in df.columns}
    s1_col = col_lookup.get("source1_entity_id")
    matched_col = col_lookup.get("matched_entity_ids")

    if not s1_col or not matched_col:
        # Fall back to first two columns if present
        if len(df.columns) >= 2:
            s1_col, matched_col = df.columns[0], df.columns[1]
            logger.warning(
                "Columns 'source1_entity_id' or 'matched_entity_ids' not found in %s. "
                "Using first two columns: %s, %s",
                file_path,
                s1_col,
                matched_col,
            )
        else:
            raise ValueError(
                f"File at {file_path} must have at least 2 columns ('source1_entity_id', 'matched_entity_ids')."
            )

    df[matched_col] = df[matched_col].fillna("")

    mapping: dict[str, set[str]] = {}
    for s1_raw, matched_raw in zip(df[s1_col].astype(str), df[matched_col].astype(str)):
        s1 = s1_raw.strip()
        if not s1 or s1.lower() == "nan":
            continue

        m_str = matched_raw.strip()
        if not m_str or m_str.lower() == "nan":
            mapping[s1] = set()
        else:
            mapping[s1] = {mid.strip() for mid in m_str.split(",") if mid.strip()}

    logger.info("Loaded %d entities from %s", len(mapping), file_path)
    return mapping


def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    """Load train_ground_truth.tsv and return {source1_entity_id: set(matched_ids)}.

    Empty or NaN matched_entity_ids are loaded as empty sets (singletons).

    Args:
        path: Path to the ground truth TSV file.

    Returns:
        Mapping {source1_entity_id: set(matched_entity_ids)}.
    """
    return _load_mapping_tsv(path)


def load_predictions(path: str | Path) -> dict[str, set[str]]:
    """Load matching_results.tsv and return {source1_entity_id: set(matched_ids)}.

    Empty or NaN matched_entity_ids are loaded as empty sets.

    Args:
        path: Path to the predictions TSV file.

    Returns:
        Mapping {source1_entity_id: set(matched_entity_ids)}.
    """
    return _load_mapping_tsv(path)


def save_mapping_tsv(data: dict[str, set[str]], output_path: str | Path) -> None:
    """Save an entity resolution mapping to a submission-compliant TSV file.

    Args:
        data: Mapping {source1_entity_id: set(matched_entity_ids)}.
        output_path: Path where the TSV file will be written.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id, matched_ids in data.items():
        matched_str = ",".join(sorted(matched_ids)) if matched_ids else ""
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})

    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(out, sep="\t", index=False)
    logger.info("Saved %d entities to %s", len(df), out)


def evaluate_submission(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
) -> dict[str, Any]:
    """Load predictions and ground truth, compute macro F0.5 and diagnostic metrics.

    Prints a comprehensive summary report and returns detailed evaluation statistics.

    Args:
        predictions_path: Path to matching_results.tsv.
        ground_truth_path: Path to train_ground_truth.tsv.

    Returns:
        Dictionary containing:
        - macro_f05: Macro-averaged F0.5 score
        - precision: Macro-averaged precision across all entities
        - recall: Macro-averaged recall across all entities
        - micro_precision: Global micro precision across all matched pairs
        - micro_recall: Global micro recall across all matched pairs
        - total_entities: Total number of entities evaluated
        - total_singletons: Total singletons in ground truth
        - total_singletons_in_ground_truth: Alias for total_singletons
        - singletons_correct: Number of singletons correctly predicted as empty
        - singletons_correctly_predicted: Alias for singletons_correct
        - singletons_missed: Number of true singletons where matches were predicted (false merges)
        - non_singletons_total: Total entities with at least one true match
        - non_singletons_missed: Total non-singletons where no matches were predicted
        - true_positives: Total true positive links
        - false_positives: Total false positive links
        - false_negatives: Total false negative links
    """
    predictions = load_predictions(predictions_path)
    ground_truth = load_ground_truth(ground_truth_path)

    total_entities = len(ground_truth)
    if total_entities == 0:
        raise ValueError(f"Ground truth file at {ground_truth_path} has 0 entities.")

    total_singletons = 0
    singletons_correct = 0
    singletons_missed = 0

    non_singletons_total = 0
    non_singletons_missed = 0

    sum_f05 = 0.0
    sum_macro_precision = 0.0
    sum_macro_recall = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())

        # F0.5
        score = f05_per_entity(pred_ids, true_ids)
        sum_f05 += score

        is_true_singleton = len(true_ids) == 0
        if is_true_singleton:
            total_singletons += 1
            if len(pred_ids) == 0:
                singletons_correct += 1
                sum_macro_precision += 1.0
                sum_macro_recall += 1.0
            else:
                singletons_missed += 1
                sum_macro_precision += 0.0
                sum_macro_recall += 0.0
                total_fp += len(pred_ids)
        else:
            non_singletons_total += 1
            if len(pred_ids) == 0:
                non_singletons_missed += 1
                sum_macro_precision += 0.0
                sum_macro_recall += 0.0
                total_fn += len(true_ids)
            else:
                common = pred_ids & true_ids
                p = len(common) / len(pred_ids)
                r = len(common) / len(true_ids)
                sum_macro_precision += p
                sum_macro_recall += r

                total_tp += len(common)
                total_fp += len(pred_ids - true_ids)
                total_fn += len(true_ids - pred_ids)

    macro_score = sum_f05 / total_entities
    macro_prec = sum_macro_precision / total_entities
    macro_rec = sum_macro_recall / total_entities

    micro_prec = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_rec = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )

    singleton_accuracy = (
        (singletons_correct / total_singletons * 100.0)
        if total_singletons > 0
        else 0.0
    )

    report_lines = [
        "",
        "=" * 64,
        "             ENTITY RESOLUTION EVALUATION REPORT",
        "=" * 64,
        f" Predictions:  {predictions_path}",
        f" Ground Truth: {ground_truth_path}",
        "-" * 64,
        f" Macro F0.5 Score:                    {macro_score:.6f}",
        f" Precision (Macro / Micro):           {macro_prec:.6f} / {micro_prec:.6f}",
        f" Recall (Macro / Micro):              {macro_rec:.6f} / {micro_rec:.6f}",
        "-" * 64,
        f" Total Entities Evaluated:            {total_entities:,}",
        f" Ground Truth Singletons:             {total_singletons:,} ({total_singletons / total_entities * 100:.1f}%)",
        f"   - Correctly Predicted:             {singletons_correct:,} ({singleton_accuracy:.1f}%)",
        f"   - Missed (False Merges):           {singletons_missed:,} ({singletons_missed / max(1, total_singletons) * 100:.1f}%)",
        f" Ground Truth Non-Singletons:         {non_singletons_total:,} ({non_singletons_total / total_entities * 100:.1f}%)",
        f"   - Completely Missed:               {non_singletons_missed:,} ({non_singletons_missed / max(1, non_singletons_total) * 100:.1f}%)",
        f" True Positives (Link Level):         {total_tp:,}",
        f" False Positives (Link Level):        {total_fp:,}",
        f" False Negatives (Link Level):        {total_fn:,}",
        "=" * 64,
        "",
    ]
    report = "\n".join(report_lines)
    print(report)
    logger.info("Evaluation completed. Macro F0.5: %.6f", macro_score)

    return {
        "macro_f05": macro_score,
        "precision": macro_prec,
        "recall": macro_rec,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "total_entities": total_entities,
        "total_singletons": total_singletons,
        "total_singletons_in_ground_truth": total_singletons,
        "singletons_correct": singletons_correct,
        "singletons_correctly_predicted": singletons_correct,
        "singletons_missed": singletons_missed,
        "non_singletons_total": non_singletons_total,
        "non_singletons_missed": non_singletons_missed,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }


def split_validation(
    ground_truth: dict[str, set[str]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Split ground truth into train and validation dicts with stratification.

    Preserves the proportion of singletons vs non-singletons across both splits.

    Args:
        ground_truth: Full ground truth mapping {source1_entity_id: set(matched_ids)}.
        val_ratio: Fraction of entities to assign to validation split (0.0 to 1.0).
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple of (train_gt, val_gt) dictionaries.

    Raises:
        ValueError: If val_ratio is not between 0.0 and 1.0.
    """
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")

    if val_ratio == 0.0:
        return ground_truth.copy(), {}
    if val_ratio == 1.0:
        return {}, ground_truth.copy()

    singletons = [s1 for s1, matched in ground_truth.items() if len(matched) == 0]
    non_singletons = [s1 for s1, matched in ground_truth.items() if len(matched) > 0]

    # Sort for deterministic behavior across Python runs
    singletons.sort()
    non_singletons.sort()

    rng = random.Random(seed)
    rng.shuffle(singletons)
    rng.shuffle(non_singletons)

    n_val_sing = int(round(len(singletons) * val_ratio))
    n_val_non = int(round(len(non_singletons) * val_ratio))

    val_keys = set(singletons[:n_val_sing] + non_singletons[:n_val_non])

    train_gt: dict[str, set[str]] = {}
    val_gt: dict[str, set[str]] = {}

    for s1_id, matched_ids in ground_truth.items():
        if s1_id in val_keys:
            val_gt[s1_id] = matched_ids
        else:
            train_gt[s1_id] = matched_ids

    total_gt = len(ground_truth)
    logger.info(
        "Split %d entities into train (%d, %.1f%%) and val (%d, %.1f%%). "
        "Singletons in train: %.1f%%, val: %.1f%% (original: %.1f%%)",
        total_gt,
        len(train_gt),
        len(train_gt) / total_gt * 100 if total_gt else 0,
        len(val_gt),
        len(val_gt) / total_gt * 100 if total_gt else 0,
        (sum(1 for s in train_gt.values() if not s) / max(1, len(train_gt))) * 100,
        (sum(1 for s in val_gt.values() if not s) / max(1, len(val_gt))) * 100,
        (len(singletons) / max(1, total_gt)) * 100,
    )

    return train_gt, val_gt


def main() -> None:
    """CLI entrypoint for evaluating entity resolution predictions."""
    parser = argparse.ArgumentParser(
        description="Evaluate Amazon ML Challenge 2026 entity resolution predictions (Macro F0.5)."
    )
    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        help="Path to matching_results.tsv file.",
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        required=True,
        help="Path to train_ground_truth.tsv file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: %(default)s).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    evaluate_submission(args.predictions, args.ground_truth)


if __name__ == "__main__":
    main()
