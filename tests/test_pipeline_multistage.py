import unittest
import numpy as np
import qai

class TestPipelineMultistage(unittest.TestCase):
    def test_pipeline_sequential_training_and_predict_happy_path(self):
        # PCA 4D -> 2D, then Classifier
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = np.array(["A" if row[0] > 0 else "B" for row in X])

        pca = qai.build(type="pca", n_components=2)
        clf = qai.build(type="classifier")
        pipeline = pca.pipe_to(clf)

        # Train pipeline stages sequentially
        pipeline.train(X, y)

        pred_single = pipeline.predict(X[0])
        self.assertIn(pred_single, ["A", "B"])

        pred_batch = pipeline.predict(X[:5])
        self.assertEqual(len(pred_batch), 5)

    def test_pipeline_chaining_edge_cases(self):
        # Chaining multiple pipelines together
        X = np.random.randn(20, 5)

        m1 = qai.build(type="pca", n_components=3)
        m2 = qai.build(type="pca", n_components=2)
        m3 = qai.build(type="kmeans", n_clusters=2)

        pipe1 = m1.pipe_to(m2)
        full_pipe = pipe1.pipe_to(m3)

        full_pipe.train(X)
        res = full_pipe.predict(X[0])
        self.assertIsInstance(res, (int, np.integer))

    def test_pipeline_error_cases(self):
        # Empty pipeline list
        with self.assertRaises(ValueError):
            qai.Pipeline([])

        # Invalid pipe_to target type
        m1 = qai.build(type="pca", n_components=2)
        pipe = qai.Pipeline([m1])
        with self.assertRaises(TypeError):
            pipe.pipe_to("invalid_non_model_object")

    def test_security_misuse_scenario(self):
        pca = qai.build(type="pca", n_components=2)
        clf = qai.build(type="classifier")
        pipe = pca.pipe_to(clf)
        X = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
        y = np.array(["A", "B"])
        pipe.train(X, y)

        # Misuse case: passing malformed string list
        with self.assertRaises(ValueError):
            pipe.predict(["invalid", "garbage"])

if __name__ == "__main__":
    unittest.main()
