import unittest
import numpy as np
import qai

class TestFiftyNewFeatures(unittest.TestCase):
    def test_new_preprocessing_transformers(self):
        X = [[-10, 100], [0, 200], [10, 300]]
        scaler = qai.MaxAbsScaler()
        X_trans = scaler.fit_transform(X)
        self.assertEqual(X_trans.shape, (3, 2))
        self.assertAlmostEqual(X_trans[2, 1], 1.0)

        X_cat = [["red", "S"], ["blue", "M"], ["red", "L"]]
        ord_enc = qai.OrdinalEncoder()
        X_ord = ord_enc.fit_transform(X_cat)
        self.assertEqual(X_ord.shape, (3, 2))

        target_enc = qai.TargetEncoder()
        X_targ = target_enc.fit_transform(X_cat, [1.0, 0.0, 1.0])
        self.assertEqual(X_targ.shape, (3, 2))

        kbins = qai.KBinsDiscretizer(n_bins=3)
        X_bins = kbins.fit_transform(X)
        self.assertEqual(X_bins.shape, (3, 2))

    def test_fairness_governance_metrics(self):
        y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 1, 1, 0, 0, 1, 0]
        sf = [0, 0, 0, 0, 1, 1, 1, 1]

        dp_diff = qai.demographic_parity_difference(y_pred, sf)
        eo_diff = qai.equalized_odds_difference(y_true, y_pred, sf)
        di_ratio = qai.disparate_impact_ratio(y_pred, sf)
        audit = qai.fairness_audit(y_true, y_pred, sf)

        self.assertGreaterEqual(dp_diff, 0.0)
        self.assertGreaterEqual(eo_diff, 0.0)
        self.assertGreaterEqual(di_ratio, 0.0)
        self.assertIn("demographic_parity_difference", audit)

    def test_additional_evaluation_metrics(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1, 2.9]

        self.assertGreaterEqual(qai.mean_squared_log_error(y_true, y_pred), 0.0)
        self.assertGreaterEqual(qai.mean_absolute_percentage_error(y_true, y_pred), 0.0)
        self.assertAlmostEqual(qai.median_absolute_error(y_true, y_pred), 0.1, places=4)
        self.assertGreater(qai.explained_variance_score(y_true, y_pred), 0.8)
        self.assertAlmostEqual(qai.max_error(y_true, y_pred), 0.1, places=4)

    def test_error_and_edge_cases(self):
        scaler = qai.MaxAbsScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform([[1, 2]])

        with self.assertRaises(ValueError):
            qai.mean_squared_log_error([-1.0, 2.0], [1.0, 2.0])

if __name__ == "__main__":
    unittest.main()
