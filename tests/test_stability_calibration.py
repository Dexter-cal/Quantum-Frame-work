import unittest
import numpy as np
import qai
import qai.math as qmath

class TestStabilityCalibration(unittest.TestCase):
    def test_log_sum_exp_overflow_prevention(self):
        large_x = [1000.0, 1001.0, 1002.0]
        lse = qmath.log_sum_exp(large_x)
        self.assertFalse(np.isnan(lse))
        self.assertFalse(np.isinf(lse))
        self.assertGreater(lse, 1002.0)

    def test_safe_divide_zero_protection(self):
        self.assertAlmostEqual(qmath.safe_divide(10.0, 0.0, eps=1e-6), 10.0 / 1e-6)
        div_arr = qmath.safe_divide([1.0, 2.0], [0.0, 4.0], eps=1e-5)
        self.assertEqual(len(div_arr), 2)
        self.assertFalse(np.isnan(div_arr[0]))

    def test_brier_score_and_ece(self):
        y_true = [1, 0, 1, 1, 0, 0]
        y_prob = [0.9, 0.1, 0.8, 0.7, 0.2, 0.3]

        bs = qai.brier_score(y_true, y_prob)
        ece = qai.expected_calibration_error(y_true, y_prob, n_bins=5)

        self.assertGreaterEqual(bs, 0.0)
        self.assertLessEqual(bs, 1.0)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_levenshtein_and_jaccard_distance(self):
        self.assertEqual(qmath.levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(qmath.levenshtein_distance("qai", "qai"), 0)
        self.assertAlmostEqual(qmath.jaccard_similarity(["a", "b"], ["a", "c"]), 0.33333333, places=4)

    def test_smooth_l1_and_clip(self):
        loss = qmath.smooth_l1_loss([1.0, 2.0], [1.2, 4.0], beta=1.0)
        self.assertGreater(loss, 0)
        clipped = qmath.clip_by_value([-100, 0, 100], min_val=-10, max_val=10)
        self.assertTrue(np.array_equal(clipped, [-10, 0, 10]))

if __name__ == "__main__":
    unittest.main()
