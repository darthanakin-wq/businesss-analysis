"""Unit tests for evaluate.py in business_entity_resolution."""

import math
import os
import tempfile
import unittest

from business_entity_resolution.src.evaluate import (
    f05_per_entity,
    macro_f05,
    load_ground_truth,
    load_predictions,
    evaluate_submission,
    split_validation,
    save_mapping_tsv,
)


class TestEvaluate(unittest.TestCase):
    def test_f05_both_empty_singleton(self):
        # Both sets empty: true singleton correctly predicted
        self.assertEqual(f05_per_entity(set(), set()), 1.0)
        self.assertEqual(f05_per_entity(None, set()), 1.0)
        self.assertEqual(f05_per_entity(set(), None), 1.0)

    def test_f05_false_merge_on_singleton(self):
        # True is singleton, predicted matches -> false merge
        self.assertEqual(f05_per_entity({"S2-100"}, set()), 0.0)
        self.assertEqual(f05_per_entity({"S2-100", "S3-200"}, set()), 0.0)

    def test_f05_missed_matches(self):
        # True has matches, predicted is empty -> missed all matches
        self.assertEqual(f05_per_entity(set(), {"S2-100"}), 0.0)
        self.assertEqual(f05_per_entity(set(), {"S2-100", "S3-200"}), 0.0)

    def test_f05_disjoint(self):
        # Both non-empty but disjoint
        self.assertEqual(f05_per_entity({"S2-100"}, {"S3-200"}), 0.0)

    def test_f05_perfect_match(self):
        # Both non-empty and exact match
        self.assertEqual(f05_per_entity({"S2-100", "S3-200"}, {"S2-100", "S3-200"}), 1.0)

    def test_f05_partial_match_example(self):
        # Example from competition README:
        # Predicted: S1 matches [S2-00047, S2-00193, S3-00812]
        # Ground truth: S1 matches [S2-00047, S3-00812]
        # Precision = 2/3, Recall = 1.0
        # F0.5 = (1.25 * (2/3) * 1) / (0.25 * (2/3) + 1) = (5/6) / (7/6) = 5/7 ≈ 0.7142857
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        true = {"S2-00047", "S3-00812"}
        score = f05_per_entity(pred, true)
        self.assertAlmostEqual(score, 5.0 / 7.0, places=5)

    def test_macro_f05(self):
        gt = {
            "S1-1": set(),                         # singleton
            "S1-2": {"S2-10"},                     # 1 match
            "S1-3": {"S2-20", "S3-30"},            # 2 matches
            "S1-4": set(),                         # singleton
        }
        pred = {
            "S1-1": set(),                         # correct singleton -> 1.0
            "S1-2": set(),                         # missed match -> 0.0
            "S1-3": {"S2-20", "S3-30"},            # perfect match -> 1.0
            "S1-4": {"S2-40"},                     # false merge on singleton -> 0.0
        }
        # Scores: 1.0 + 0.0 + 1.0 + 0.0 = 2.0 / 4 = 0.5
        score = macro_f05(pred, gt)
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_macro_f05_missing_from_predictions(self):
        # If an entity is omitted from predictions dict, it is treated as predicting empty set
        gt = {
            "S1-1": set(),              # true singleton -> pred is empty set -> score = 1.0
            "S1-2": {"S2-10"},          # non-singleton -> pred is empty set -> score = 0.0
        }
        pred = {}  # omitted both
        score = macro_f05(pred, gt)
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_load_and_evaluate_tsv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = os.path.join(tmpdir, "gt.tsv")
            pred_path = os.path.join(tmpdir, "pred.tsv")

            # Write TSV files
            with open(gt_path, "w", encoding="utf-8") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                f.write("S1-01\tS2-10,S3-20\n")
                f.write("S1-02\t\n")  # singleton
                f.write("S1-03\tNaN\n")  # singleton with NaN string
                f.write("S1-04\tS2-30\n")

            with open(pred_path, "w", encoding="utf-8") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                f.write("S1-01\tS2-10,S3-20\n")  # perfect match -> 1.0
                f.write("S1-02\t\n")  # correct singleton -> 1.0
                f.write("S1-03\tS2-99\n")  # false merge on singleton -> 0.0
                f.write("S1-04\t\n")  # missed match -> 0.0

            gt = load_ground_truth(gt_path)
            self.assertEqual(gt["S1-01"], {"S2-10", "S3-20"})
            self.assertEqual(gt["S1-02"], set())
            self.assertEqual(gt["S1-03"], set())
            self.assertEqual(gt["S1-04"], {"S2-30"})

            pred = load_predictions(pred_path)
            self.assertEqual(pred["S1-01"], {"S2-10", "S3-20"})
            self.assertEqual(pred["S1-02"], set())
            self.assertEqual(pred["S1-03"], {"S2-99"})
            self.assertEqual(pred["S1-04"], set())

            metrics = evaluate_submission(pred_path, gt_path)
            self.assertAlmostEqual(metrics["macro_f05"], 0.5, places=5)
            self.assertEqual(metrics["total_entities"], 4)
            self.assertEqual(metrics["total_singletons"], 2)
            self.assertEqual(metrics["singletons_correct"], 1)
            self.assertEqual(metrics["singletons_missed"], 1)
            self.assertEqual(metrics["non_singletons_total"], 2)

    def test_split_validation(self):
        # 100 entities: 60 singletons, 40 non-singletons
        gt = {f"S1-sing-{i}": set() for i in range(60)}
        gt.update({f"S1-pair-{i}": {f"S2-{i}"} for i in range(40)})

        train_gt, val_gt = split_validation(gt, val_ratio=0.25, seed=123)
        self.assertEqual(len(train_gt) + len(val_gt), 100)
        self.assertEqual(len(val_gt), 25)
        self.assertEqual(len(train_gt), 75)

        val_sing = sum(1 for m in val_gt.values() if len(m) == 0)
        train_sing = sum(1 for m in train_gt.values() if len(m) == 0)

        # 60 * 0.25 = 15 val singletons
        self.assertEqual(val_sing, 15)
        self.assertEqual(train_sing, 45)


if __name__ == "__main__":
    unittest.main()
