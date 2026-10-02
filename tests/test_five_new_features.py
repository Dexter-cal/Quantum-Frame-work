import unittest
import numpy as np
import tempfile
import os
import qai
from fixtures_huggingface_iris import load as load_hf_iris

class TestFiveNewFeatures(unittest.TestCase):
    def test_feature1_explainability(self):
        # Tree feature attribution
        X, y = load_hf_iris()
        model = qai.build(type="decision_tree", max_depth=3)
        model.train(X, y, verbose=False)

        exp = model.explain(X[0])
        self.assertIn("feature_attributions", exp)
        self.assertIn("most_influential_feature", exp)

    def test_feature2_benchmark(self):
        X, y = load_hf_iris()
        res = qai.benchmark(["decision_tree", "knn", "naive_bayes"], X, y)

        self.assertIn("leaderboard", res)
        self.assertEqual(len(res["leaderboard"]), 3)
        self.assertIn(res["best_technique"], ["decision_tree", "knn", "naive_bayes"])

    def test_feature3_tsne(self):
        X, _ = load_hf_iris()
        model = qai.build(type="tsne", n_components=2, perplexity=10.0)
        model.train(X)

        self.assertEqual(model.embedding().shape, (30, 2))
        pred = model.predict(X[0])
        self.assertEqual(len(pred), 2)

    def test_feature4_model_persistence(self):
        X, y = load_hf_iris()
        model = qai.build(type="knn", k=3)
        model.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            qai.save_model(model, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            reloaded = qai.load_model(tmp_path)
            pred_orig = model.predict(X[0])
            pred_reloaded = reloaded.predict(X[0])
            self.assertEqual(pred_orig, pred_reloaded)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_feature5_clean_dataset(self):
        X_nan = [[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]]
        y_str = ["cat", "dog", "cat"]

        X_clean, y_clean = qai.clean_dataset(X_nan, y_str, impute_strategy="mean", scale=True)
        self.assertFalse(np.isnan(X_clean).any())
        self.assertTrue(np.issubdtype(y_clean.dtype, np.integer))

    def test_security_misuse_scenarios(self):
        # Benchmark with non-existent technique name -> records error
        res = qai.benchmark(["invalid_technique_name"], [[1.0, 2.0]], [1])
        self.assertEqual(res["leaderboard"][0]["status"], "failed")

        # load_model with missing file -> FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            qai.load_model("non_existent_file.pkl")

if __name__ == "__main__":
    unittest.main()
