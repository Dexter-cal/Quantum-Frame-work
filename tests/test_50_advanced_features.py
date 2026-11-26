import unittest
import numpy as np
import qai

class Test50AdvancedFeatures(unittest.TestCase):
    def test_function_and_column_transformers(self):
        X = np.array([[1, 2], [3, 4]])
        func_trans = qai.FunctionTransformer(func=lambda x: x * 2)
        self.assertTrue(np.array_equal(func_trans.fit_transform(X), X * 2))

        col_trans = qai.ColumnTransformer([
            ("scale", qai.StandardScaler(), [0]),
            ("pass", qai.FunctionTransformer(), [1])
        ])
        ct_out = col_trans.fit_transform(X)
        self.assertEqual(ct_out.shape, (2, 2))

    def test_quantile_and_power_transformers(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        qt = qai.QuantileTransformer(n_quantiles=3, output_distribution="normal")
        qt_out = qt.fit_transform(X)
        self.assertEqual(qt_out.shape, (3, 2))

        pt = qai.PowerTransformer(method="yeo-johnson")
        pt_out = pt.fit_transform(X)
        self.assertEqual(pt_out.shape, (3, 2))

        pt_bc = qai.PowerTransformer(method="box-cox")
        with self.assertRaises(ValueError):
            pt_bc.transform([[-1.0, 10.0]])

    def test_reduce_lr_on_plateau(self):
        lr_scheduler = qai.ReduceLROnPlateau(initial_lr=0.1, factor=0.5, patience=2)
        self.assertEqual(lr_scheduler.step(1.0), 0.1)
        self.assertEqual(lr_scheduler.step(1.0), 0.1)
        self.assertEqual(lr_scheduler.step(1.0), 0.05)

    def test_curve_and_ranking_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])

        prec, rec, thresh = qai.precision_recall_curve(y_true, y_scores)
        self.assertGreater(len(prec), 0)

        fpr, tpr, roc_thresh = qai.roc_curve(y_true, y_scores)
        self.assertGreater(len(fpr), 0)

        fpr_det, fnr_det, det_thresh = qai.det_curve(y_true, y_scores)
        self.assertGreater(len(fpr_det), 0)

        y_multilabel = np.array([[1, 0], [0, 1]])
        y_score_multi = np.array([[0.8, 0.2], [0.1, 0.9]])
        self.assertGreater(qai.coverage_error(y_multilabel, y_score_multi), 0.0)
        self.assertGreater(qai.label_ranking_average_precision_score(y_multilabel, y_score_multi), 0.0)

if __name__ == "__main__":
    unittest.main()
