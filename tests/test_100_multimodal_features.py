import unittest
import numpy as np
import qai

class Test100MultimodalFeatures(unittest.TestCase):
    def test_vision_module(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        resized = qai.vision.resize_image(img, (50, 50))
        self.assertEqual(resized.shape, (50, 50, 3))

        normed = qai.vision.normalize_image(resized)
        self.assertEqual(normed.shape, (50, 50, 3))

        cropped = qai.vision.center_crop(img, (30, 30))
        self.assertEqual(cropped.shape, (30, 30, 3))

        flipped = qai.vision.random_flip(img, p=1.0)
        self.assertEqual(flipped.shape, (100, 100, 3))

        tensor = qai.vision.image_to_tensor(img)
        self.assertEqual(tensor.shape, (3, 100, 100))

    def test_audio_module(self):
        waveform = np.sin(np.linspace(0, 100, 16000))
        spec = qai.audio.spectrogram(waveform, n_fft=256, hop_length=128)
        self.assertGreater(spec.shape[0], 0)

        melspec = qai.audio.melspectrogram(waveform, n_mels=40)
        self.assertEqual(melspec.shape[0], 40)

        mfcc_feats = qai.audio.mfcc(waveform, n_mfcc=13)
        self.assertEqual(mfcc_feats.shape[0], 13)

        shifted = qai.audio.pitch_shift(waveform, semitones=2.0)
        self.assertGreater(len(shifted), 0)

        stretched = qai.audio.time_stretch(waveform, rate=1.2)
        self.assertLess(len(stretched), len(waveform))

    def test_nlp_module(self):
        text = "Hello World! This is QAI NLP test."
        cleaned = qai.nlp.text_cleaner(text)
        self.assertNotIn("!", cleaned)

        corpus = ["The quick brown fox", "Python machine learning QAI"]
        tfidf, vocab = qai.nlp.tf_idf_vectorizer(corpus)
        self.assertEqual(tfidf.shape[0], 2)

        tokenizer = qai.nlp.WordPieceTokenizer(vocab=["the", "quick", "brown"])
        tokens = tokenizer.tokenize("The quick cat")
        self.assertEqual(tokens, ["the", "quick", "[UNK]"])

        embeds = qai.nlp.word_embeddings_lookup(["quick", "fox"], embedding_dim=8)
        self.assertEqual(embeds.shape, (2, 8))

    def test_graph_module(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        dc = qai.graph.degree_centrality(adj)
        self.assertEqual(len(dc), 3)

        bc = qai.graph.betweenness_centrality(adj)
        self.assertEqual(len(bc), 3)

        dist = qai.graph.shortest_path_dijkstra(adj, start_node=0)
        self.assertEqual(dist[1], 1.0)

    def test_time_series_module(self):
        series = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
        smoothed = qai.time_series.exponential_smoothing(series, alpha=0.3)
        self.assertEqual(len(smoothed), 8)

        acf_vals = qai.time_series.autocorrelation_acf(series, max_lag=3)
        self.assertEqual(len(acf_vals), 4)

        t, s, r = qai.time_series.seasonal_decompose(series, period=2)
        self.assertEqual(len(t), 8)

if __name__ == "__main__":
    unittest.main()
