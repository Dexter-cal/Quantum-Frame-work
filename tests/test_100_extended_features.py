import unittest
import numpy as np
import qai

class Test100ExtendedFeatures(unittest.TestCase):
    def test_advanced_loss_functions(self):
        c_loss = qai.contrastive_loss([1, 0], [0.1, 0.9])
        self.assertGreaterEqual(c_loss, 0.0)

        cos_loss = qai.cosine_embedding_loss([[1, 0]], [[0, 1]], [1])
        self.assertGreaterEqual(cos_loss, 0.0)

        mr_loss = qai.margin_ranking_loss([1.0], [0.5], [1])
        self.assertGreaterEqual(mr_loss, 0.0)

        sm_loss = qai.soft_margin_loss([1], [0.5])
        self.assertGreater(sm_loss, 0.0)

        pnll = qai.poisson_nll_loss([2.0], [0.5])
        self.assertIsInstance(pnll, float)

    def test_extended_metrics(self):
        y_t = [0, 1, 1, 0]
        y_p = [0, 1, 0, 0]
        self.assertAlmostEqual(qai.zero_one_loss(y_t, y_p), 0.25)
        self.assertAlmostEqual(qai.hamming_loss(y_t, y_p), 0.25)
        self.assertGreater(qai.jaccard_score(y_t, y_p), 0.0)

        mcm = qai.multilabel_confusion_matrix([[0, 1]], [[0, 1]])
        self.assertEqual(mcm.shape, (1, 2, 2))

        bsl = qai.brier_score_loss([1, 0], [0.9, 0.2])
        self.assertGreaterEqual(bsl, 0.0)

    def test_vision_filters_and_histograms(self):
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        blurred = qai.vision.gaussian_blur_filter(img, kernel_size=3)
        self.assertEqual(blurred.shape, (20, 20, 3))

        edges = qai.vision.edge_detection_sobel(img[:, :, 0])
        self.assertEqual(edges.shape, (20, 20))

        hist = qai.vision.color_histogram(img, bins=8)
        self.assertEqual(len(hist), 24)

    def test_audio_and_nlp_extended(self):
        sig = np.sin(np.linspace(0, 100, 1000))
        zcr = qai.audio.zero_crossing_rate(sig)
        self.assertGreater(zcr, 0.0)

        sc = qai.audio.spectral_centroid(sig)
        self.assertGreater(sc, 0.0)

        ngrams = qai.nlp.character_ngram_tokenizer("sample text", n=3)
        self.assertIn("sam", ngrams)

        readability = qai.nlp.readability_score_flesch("The quick brown fox jumps over the lazy dog.")
        self.assertIsInstance(readability, float)

    def test_system_profiling(self):
        def sample_fn(x):
            return x * 2

        res, latency = qai.profile_execution_time(sample_fn, 10)
        self.assertEqual(res, 20)
        self.assertGreaterEqual(latency, 0.0)

if __name__ == "__main__":
    unittest.main()
