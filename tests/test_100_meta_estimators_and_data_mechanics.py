import unittest
import numpy as np
import qai
import qai.data as qdata

class Test100MetaEstimatorsAndDataMechanics(unittest.TestCase):
    def test_multi_output_classifier_and_regressor(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        Y_cls = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        Y_reg = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

        base_cls = qai.build(type="classifier")
        mo_cls = qai.MultiOutputClassifier(estimator=base_cls)
        mo_cls.train(X, Y_cls)
        preds_cls = mo_cls.predict(X)
        self.assertEqual(preds_cls.shape, (4, 2))

        base_reg = qai.build(type="regression")
        mo_reg = qai.MultiOutputRegressor(estimator=base_reg)
        mo_reg.train(X, Y_reg)
        preds_reg = mo_reg.predict(X)
        self.assertEqual(preds_reg.shape, (4, 2))

    def test_classifier_chain_and_stacking(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        Y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])

        base_cls = qai.build(type="classifier")
        chain = qai.ClassifierChain(estimator=base_cls)
        chain.train(X, Y)
        chain_preds = chain.predict(X)
        self.assertEqual(chain_preds.shape, (4, 2))

        m1 = qai.build(type="decision_tree")
        m2 = qai.build(type="knn")
        final_m = qai.build(type="classifier")
        stack_cls = qai.StackingClassifier(estimators=[m1, m2], final_estimator=final_m)
        stack_cls.train(X, Y[:, 0])
        stack_preds = stack_cls.predict(X)
        self.assertEqual(len(stack_preds), 4)

    def test_data_health_check_outliers_interpolation(self):
        raw = [
            [1, 10.0],
            [2, 20.0],
            [3, 1000.0], # outlier
            [4, None]   # missing
        ]
        ds = qdata.DataDataset(raw, ["id", "val"])
        health = ds.data_health_check()
        self.assertFalse(health["is_empty"])
        self.assertEqual(health["null_cells"], 1)

        outliers = ds.detect_outliers_iqr("val")
        self.assertIn(2, outliers)

        ds.interpolate_missing_values()
        self.assertIsNotNone(ds.data[3][1])

if __name__ == "__main__":
    unittest.main()
