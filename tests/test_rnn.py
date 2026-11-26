"""Tests for qai.techniques.rnn — RNN/LSTM/GRU/Seq2Seq."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qai
from qai.techniques import RNN, LSTM, GRU, Seq2Seq


CORPUS = [
    "def add(a, b):",
    "    return a + b",
    "def sub(a, b):",
    "    return a - b",
    "def mul(a, b):",
    "    return a * b",
] * 3

TEXTS = ["i love this", "this is great", "i hate this", "this is bad"] * 5
LABELS = [1, 1, 0, 0] * 5


class TestRNNBuild(unittest.TestCase):
    def test_build_rnn(self):
        m = qai.build(technique="rnn", vocab_size=50, d_model=16)
        self.assertIsInstance(m.technique, RNN)

    def test_build_lstm(self):
        m = qai.build(technique="lstm", vocab_size=50, d_model=16)
        self.assertIsInstance(m.technique, LSTM)

    def test_build_gru(self):
        m = qai.build(technique="gru", vocab_size=50, d_model=16)
        self.assertIsInstance(m.technique, GRU)

    def test_build_seq2seq(self):
        m = qai.build(technique="seq2seq", vocab_size=50, d_model=16)
        self.assertIsInstance(m.technique, Seq2Seq)

    def test_invalid_task_type_raises(self):
        with self.assertRaises(AssertionError):
            qai.build(technique="lstm", task_type="invalid")

    def test_bidirectional_generation_rejected(self):
        with self.assertRaises(ValueError):
            qai.build(technique="gru", task_type="generation", bidirectional=True)


class TestLSTMClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = qai.CharTokenizer()
        cls.tok.fit(TEXTS)
        cls.model = qai.build(
            technique="lstm", task_type="classification", n_classes=2,
            vocab_size=cls.tok.vocab_size, d_model=32, n_layers=1,
            epochs=20, lr=1e-3, batch_size=4,
        )
        cls.model.train(TEXTS, y=LABELS, tokenizer=cls.tok, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_loss_decreases(self):
        losses = self.model.technique.loss_history()
        self.assertGreater(losses[0], losses[-1])

    def test_predict_returns_int(self):
        pred = self.model.predict("i love this")
        self.assertIsInstance(pred, (int, np.integer))

    def test_accuracy(self):
        acc = self.model.technique.accuracy()
        self.assertGreater(acc, 0.5)

    def test_parameter_count(self):
        self.assertGreater(self.model.technique.parameter_count(), 0)

    def test_string_labels(self):
        y_str = np.array(["pos", "neg"])[np.array(LABELS)]
        m = qai.build(technique="lstm", task_type="classification", n_classes=2,
                      vocab_size=self.tok.vocab_size, d_model=16, n_layers=1, epochs=5)
        m.train(TEXTS, y=y_str, tokenizer=self.tok, verbose=False)
        pred = m.predict("i love this")
        self.assertIn(pred, ["pos", "neg"])

    def test_no_tokenizer_raises(self):
        m = qai.build(technique="lstm", task_type="classification", n_classes=2,
                      vocab_size=50, d_model=16, epochs=1)
        with self.assertRaises(ValueError):
            m.train(["hello"], y=[0], verbose=False)

    def test_predict_before_train_raises(self):
        m = qai.build(technique="lstm", vocab_size=50, d_model=16)
        with self.assertRaises(RuntimeError):
            m.predict("test")


class TestGRUGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = qai.CharTokenizer()
        cls.tok.fit(CORPUS)
        cls.model = qai.build(
            technique="gru", task_type="generation",
            vocab_size=cls.tok.vocab_size, d_model=32, n_layers=1,
            max_seq_len=32, epochs=15, lr=1e-3, batch_size=4,
        )
        cls.model.train(CORPUS, tokenizer=cls.tok, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_generate_returns_string(self):
        out = self.model.generate("def ", max_new_tokens=10, tokenizer=self.tok, temperature=0)
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("def "))

    def test_generate_with_seed_reproducible(self):
        out1 = self.model.generate("def ", max_new_tokens=10, tokenizer=self.tok,
                                    temperature=0.8, seed=42)
        out2 = self.model.generate("def ", max_new_tokens=10, tokenizer=self.tok,
                                    temperature=0.8, seed=42)
        self.assertEqual(out1, out2)

    def test_generate_before_train_raises(self):
        m = qai.build(technique="gru", task_type="generation", vocab_size=50, d_model=16)
        with self.assertRaises(RuntimeError):
            m.generate("test", max_new_tokens=5, tokenizer=self.tok)


class TestBidirectional(unittest.TestCase):
    def test_bidirectional_lstm_classification(self):
        tok = qai.CharTokenizer()
        tok.fit(TEXTS)
        m = qai.build(
            technique="lstm", task_type="classification", n_classes=2,
            vocab_size=tok.vocab_size, d_model=32, n_layers=1,
            bidirectional=True, epochs=10, lr=1e-3, batch_size=4,
        )
        m.train(TEXTS, y=LABELS, tokenizer=tok, verbose=False)
        self.assertTrue(m.technique.is_trained())
        self.assertTrue(m.technique.bidirectional)

    def test_bidirectional_more_params(self):
        uni = qai.build(technique="lstm", n_classes=2, vocab_size=50,
                        d_model=32, n_layers=1, bidirectional=False)
        uni.train(TEXTS, y=LABELS, tokenizer=qai.CharTokenizer().fit(TEXTS) or qai.CharTokenizer(),
                  verbose=False)
        bi = qai.build(technique="lstm", n_classes=2, vocab_size=50,
                       d_model=32, n_layers=1, bidirectional=True)
        bi.train(TEXTS, y=LABELS, tokenizer=qai.CharTokenizer().fit(TEXTS) or qai.CharTokenizer(),
                 verbose=False)
        # Bidirectional should have ~2x the RNN params
        self.assertGreater(bi.technique.parameter_count(), uni.technique.parameter_count())


class TestMultiLayer(unittest.TestCase):
    def test_two_layers(self):
        tok = qai.CharTokenizer()
        tok.fit(TEXTS)
        m = qai.build(
            technique="lstm", task_type="classification", n_classes=2,
            vocab_size=tok.vocab_size, d_model=32, n_layers=2,
            epochs=5, lr=1e-3, batch_size=4,
        )
        m.train(TEXTS, y=LABELS, tokenizer=tok, verbose=False)
        self.assertTrue(m.technique.is_trained())
        self.assertEqual(m.technique.n_layers, 2)


class TestSeq2Seq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = qai.CharTokenizer()
        # Simple reversal task: "ab" -> "ba"
        pairs = [("ab", "ba"), ("cd", "dc"), ("ef", "fe"), ("gh", "hg")] * 5
        all_text = [s for p in pairs for s in p]
        cls.tok.fit(all_text)
        cls.model = qai.build(
            technique="seq2seq", vocab_size=cls.tok.vocab_size,
            d_model=32, n_layers=1, rnn_type="lstm",
            max_seq_len=8, epochs=30, lr=1e-3, batch_size=4,
        )
        cls.model.train(pairs, tokenizer=cls.tok, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_translate_returns_string(self):
        out = self.model.translate("ab", tokenizer=self.tok)
        self.assertIsInstance(out, str)

    def test_forward_returns_ids(self):
        ids = self.model.forward("ab")
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_rnn_type_gru(self):
        m = qai.build(technique="seq2seq", vocab_size=50, d_model=16,
                      rnn_type="gru", epochs=1)
        self.assertEqual(m.technique.rnn_type, "gru")

    def test_invalid_rnn_type_raises(self):
        with self.assertRaises(AssertionError):
            qai.build(technique="seq2seq", rnn_type="invalid")

    def test_parameter_count(self):
        self.assertGreater(self.model.technique.parameter_count(), 0)

    def test_predict_before_train_raises(self):
        m = qai.build(technique="seq2seq", vocab_size=50, d_model=16)
        with self.assertRaises(RuntimeError):
            m.forward("test")


class TestArchitectureSummary(unittest.TestCase):
    def test_lstm_summary(self):
        tok = qai.CharTokenizer()
        tok.fit(TEXTS)
        m = qai.build(technique="lstm", task_type="classification", n_classes=2,
                      vocab_size=tok.vocab_size, d_model=32, n_layers=2,
                      bidirectional=True)
        m.train(TEXTS, y=LABELS, tokenizer=tok, verbose=False)
        arch = m.technique.architecture_summary()
        self.assertEqual(arch["technique"], "lstm")
        self.assertEqual(arch["d_model"], 32)
        self.assertEqual(arch["n_layers"], 2)
        self.assertTrue(arch["bidirectional"])
        self.assertGreater(arch["parameters"], 0)

    def test_seq2seq_summary(self):
        m = qai.build(technique="seq2seq", vocab_size=50, d_model=16, rnn_type="gru")
        arch = m.technique.architecture_summary()
        self.assertEqual(arch["technique"], "seq2seq")
        self.assertEqual(arch["rnn_type"], "gru")


class TestIntegrationWithQai(unittest.TestCase):
    def test_in_registry(self):
        from qai.techniques import _REGISTRY
        for name in ("rnn", "lstm", "gru", "seq2seq"):
            self.assertIn(name, _REGISTRY)

    def test_in_supervised_list(self):
        from qai.techniques import SUPERVISED
        for name in ("rnn", "lstm", "gru", "seq2seq"):
            self.assertIn(name, SUPERVISED)

    def test_loss_history_via_getattr(self):
        tok = qai.CharTokenizer()
        tok.fit(TEXTS)
        m = qai.build(technique="lstm", task_type="classification", n_classes=2,
                      vocab_size=tok.vocab_size, d_model=16, epochs=3)
        m.train(TEXTS, y=LABELS, tokenizer=tok, verbose=False)
        # Via passthrough
        self.assertGreater(len(m.loss_history()), 0)
        self.assertGreater(m.parameter_count(), 0)


if __name__ == "__main__":
    unittest.main()
