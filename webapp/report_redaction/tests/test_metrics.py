"""Tests for the redaction-evaluation metrics.

``calculate_metrics`` scores an automatic redaction against a ground-truth
redaction character-by-character (ignoring whitespace/punctuation) and is what
the report-redaction evaluation UI reports. A miscount here would misrepresent
how well de-identification performed, so the confusion-matrix arithmetic is
pinned here.
"""

import os
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: never open a GUI window during tests

from webapp.report_redaction.utils import calculate_metrics, find_fuzzy_matches

R = "■"  # redacted character


class TestCalculateMetrics(unittest.TestCase):
    def test_one_of_each_confusion_cell(self):
        # original: all non-special so every position counts
        original = "ABCD"
        ground_truth = R + "XX" + R  # ■ X X ■
        automatic = R + R + "YZ"  # ■ ■ Y Z
        # pos0 TP, pos1 FP, pos2 TN, pos3 FN
        (precision, recall, accuracy, f1, specificity, fpr, fnr,
         path, tp, fp, tn, fn) = calculate_metrics(
            ground_truth, automatic, original, R
        )
        self.assertEqual((tp, fp, tn, fn), (1, 1, 1, 1))
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(accuracy, 0.5)
        self.assertAlmostEqual(f1, 0.5)
        self.assertAlmostEqual(specificity, 0.5)
        self.assertAlmostEqual(fpr, 0.5)
        self.assertAlmostEqual(fnr, 0.5)
        self.assertTrue(os.path.exists(path))

    def test_special_characters_are_ignored(self):
        # The comma position must not be counted at all.
        original = "A,B"
        ground_truth = R + "," + R
        automatic = "X,X"  # both real positions are missed redactions -> FN
        (_, _, accuracy, _, _, _, _, _, tp, fp, tn, fn) = calculate_metrics(
            ground_truth, automatic, original, R
        )
        self.assertEqual((tp, fp, tn, fn), (0, 0, 0, 2))
        self.assertEqual(accuracy, 0)

    def test_perfect_redaction(self):
        original = "ABCD"
        (precision, recall, accuracy, f1, *_rest) = calculate_metrics(
            R * 4, R * 4, original, R
        )
        self.assertEqual((precision, recall, accuracy, f1), (1, 1, 1.0, 1))

    def test_length_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            calculate_metrics(R, R * 2, "AB", R)


class TestFindFuzzyMatches(unittest.TestCase):
    def test_exact_word_matched(self):
        matches = find_fuzzy_matches("Patient Johnson admitted", ["Johnson"])
        self.assertTrue(any(m == "Johnson" and s >= 90 for m, s in matches))

    def test_below_threshold_excluded(self):
        matches = find_fuzzy_matches(
            "Patient Anderson", ["Johnson"], threshold=99
        )
        self.assertFalse(any(m == "Anderson" for m, s in matches))

    def test_invalid_scorer_raises(self):
        with self.assertRaises(ValueError):
            find_fuzzy_matches("text", ["info"], scorer="bogus")


if __name__ == "__main__":
    unittest.main()
