import unittest
import numpy as np
import os
import qai.data as qdata

class TestQAIDataGaps(unittest.TestCase):
    def setUp(self):
        self.raw_data = [
            [1, "Alice Smith", "The quick brown fox jumps", 25, 100.0, 0.9],
            [2, "Alice Smth", "The fast brown fox jumps", 25, 105.0, 0.1], # fuzzy dup
            [3, "Bob Jones", "A lazy dog was sleeping", 30, 200.0, 0.5],
            [4, "Charlie Brown", "Python machine learning models", 35, 300.0, 0.2]
        ]
        self.columns = ["id", "name", "text", "age", "score", "weight"]
        self.ds = qdata.DataDataset(self.raw_data, self.columns)

    def test_fuzzy_deduplicate(self):
        self.ds.fuzzy_deduplicate(threshold=0.85)
        self.assertLessEqual(len(self.ds.data), 4)

    def test_generate_synthetic_data(self):
        syn_ds = self.ds.generate_synthetic_data(n_samples=5)
        self.assertEqual(len(syn_ds.data), 5)
        self.assertEqual(len(syn_ds.columns), 6)

    def test_active_learning_queue(self):
        dummy_predict_probs = lambda X: np.array([[0.5, 0.5], [0.9, 0.1], [0.8, 0.2], [0.6, 0.4]])
        queue_ds = self.ds.active_learning_queue(dummy_predict_probs, top_k=2)
        self.assertEqual(len(queue_ds.data), 2)

    def test_rolling_window_and_lag_feature(self):
        self.ds.rolling_window(size=2, column="score")
        self.assertIn("score_rolling_2", self.ds.columns)

        self.ds.lag_feature(column="score", periods=1)
        self.assertIn("score_lag_1", self.ds.columns)

    def test_nlp_stopwords_and_stemming(self):
        self.ds.remove_stopwords("text")
        self.assertNotIn("The", self.ds.data[0][2].split())

        self.ds.stem("text")
        self.assertIn("jump", self.ds.data[0][2])

    def test_weighted_sampling(self):
        sampled = self.ds.sample_weighted(weights_column="weight", n_samples=3)
        self.assertEqual(len(sampled.data), 3)

    def test_project_catalog_and_estimate_cost(self):
        cat = qdata.project_catalog()
        self.assertIn("catalog_files", cat)

        temp_path = "/tmp/test_cost.csv"
        with open(temp_path, "w") as f:
            f.write("id,value\n1,100\n2,200\n")

        cost = qdata.estimate_load_cost(temp_path)
        self.assertTrue(cost["exists"])
        self.assertGreater(cost["disk_size_mb"], 0)

        # Test incremental sync
        ds2 = qdata.incremental_sync(self.ds, temp_path, key_col="id")
        self.assertGreaterEqual(len(ds2.data), 4)
        os.remove(temp_path)

if __name__ == "__main__":
    unittest.main()
