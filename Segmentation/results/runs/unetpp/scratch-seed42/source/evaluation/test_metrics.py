import json
import unittest

import numpy as np
import torch

from evaluation.metrics import confusion_matrix, dice_score, summarize_confusion


class MetricTests(unittest.TestCase):
    def test_ignore_and_undefined_classes(self):
        matrix = confusion_matrix(torch.tensor([0, 1, 2]), torch.tensor([0, 255, 2]), 3)
        report = summarize_confusion(matrix, ["background", "absent", "defect"], [2])
        self.assertEqual(report["valid_pixels"], 2)
        self.assertIsNone(report["per_class"][1]["iou"])
        self.assertEqual(report["macro_iou"], 1.0)
        json.dumps(report, allow_nan=False)

    def test_absent_gt_false_positive_is_penalized(self):
        matrix = confusion_matrix(torch.tensor([0, 1]), torch.tensor([0, 0]), 2)
        report = summarize_confusion(matrix, ["background", "absent"])
        self.assertEqual(report["per_class"][1]["iou"], 0.0)
        self.assertAlmostEqual(report["macro_iou"], 0.25)

    def test_dice_is_multiclass(self):
        score = dice_score(torch.tensor([1, 0]), torch.tensor([1, 2]), num_classes=3)
        self.assertAlmostEqual(score, 1 / 3)

    def test_counts_accumulate_before_ratios(self):
        first = confusion_matrix(torch.tensor([0, 1]), torch.tensor([0, 1]), 2)
        second = confusion_matrix(torch.tensor([0]), torch.tensor([1]), 2)
        report = summarize_confusion(first + second, ["background", "defect"])
        self.assertAlmostEqual(report["macro_iou"], 0.5)

    def test_all_ignored(self):
        matrix = confusion_matrix(torch.tensor([0]), torch.tensor([255]), 2)
        report = summarize_confusion(matrix, ["background", "defect"])
        self.assertIsNone(report["macro_iou"])
        np.testing.assert_array_equal(matrix, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()