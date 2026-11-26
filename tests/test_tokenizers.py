"""
Tests for qai.tokenizers — character, word, and BPE tokenizers.
Covers: fit/encode/decode round-trip, special tokens, batch ops,
padding/truncation, save/load, error cases.
"""
import sys, os, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qai
from qai.tokenizers import (
    CharTokenizer, WordTokenizer, BPETokenizer, BaseTokenizer,
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
    PAD_ID, UNK_ID, BOS_ID, EOS_ID,
)


class TestCharTokenizer(unittest.TestCase):
    def test_fit_and_encode(self):
        tok = CharTokenizer()
        tok.fit(["hello world", "hello there"])
        self.assertTrue(tok._fitted)
        ids = tok.encode("hello world")
        self.assertEqual(len(ids), 11)
        # Decoded text should match (since char-level is lossless)
        self.assertEqual(tok.decode(ids), "hello world")

    def test_special_tokens_reserved(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        # First 4 IDs should be special tokens
        self.assertEqual(tok.token_to_id[PAD_TOKEN], 0)
        self.assertEqual(tok.token_to_id[UNK_TOKEN], 1)
        self.assertEqual(tok.token_to_id[BOS_TOKEN], 2)
        self.assertEqual(tok.token_to_id[EOS_TOKEN], 3)

    def test_bos_eos_wrapping(self):
        tok = CharTokenizer()
        tok.fit(["hi"])
        ids = tok.encode("hi", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tok.token_to_id[BOS_TOKEN])
        self.assertEqual(ids[-1], tok.token_to_id[EOS_TOKEN])

    def test_padding_and_truncation(self):
        tok = CharTokenizer()
        tok.fit(["abcdef"])
        # Padding
        ids = tok.encode("abc", max_length=6, padding=True)
        self.assertEqual(len(ids), 6)
        self.assertEqual(ids[3:], [PAD_ID, PAD_ID, PAD_ID])
        # Truncation
        ids = tok.encode("abcdef", max_length=3, truncation=True)
        self.assertEqual(len(ids), 3)
        # Truncation preserves EOS if added
        ids = tok.encode("abcdef", max_length=4, add_eos=True, truncation=True)
        self.assertEqual(len(ids), 4)
        self.assertEqual(ids[-1], tok.token_to_id[EOS_TOKEN])

    def test_lowercase(self):
        tok = CharTokenizer(lowercase=True)
        tok.fit(["Hello"])
        self.assertIn("h", tok.token_to_id)
        self.assertNotIn("H", tok.token_to_id)
        # Encoding "HELLO" should work after lowercasing
        ids = tok.encode("HELLO")
        self.assertEqual(tok.decode(ids), "hello")

    def test_unknown_chars_mapped_to_unk(self):
        # If we fit on "abc" and encode "abcd", 'd' should map to UNK
        tok = CharTokenizer()
        tok.fit(["abc"])
        ids = tok.encode("abcd")
        self.assertEqual(ids[-1], tok.token_to_id[UNK_TOKEN])

    def test_encode_before_fit_raises(self):
        tok = CharTokenizer()
        with self.assertRaises(RuntimeError):
            tok.encode("hello")


class TestWordTokenizer(unittest.TestCase):
    def test_fit_and_encode(self):
        tok = WordTokenizer(lowercase=True)
        tok.fit(["the cat sat on the mat"])
        ids = tok.encode("the cat")
        self.assertEqual(len(ids), 2)
        # Round-trip
        decoded = tok.decode(ids)
        self.assertIn("the", decoded)
        self.assertIn("cat", decoded)

    def test_min_freq_filter(self):
        tok = WordTokenizer(min_freq=2)
        tok.fit(["rare common common common"])
        # "rare" appears once -> not in vocab -> UNK
        self.assertNotIn("rare", tok.token_to_id)
        self.assertIn("common", tok.token_to_id)
        ids = tok.encode("rare common")
        self.assertEqual(ids[0], tok.token_to_id[UNK_TOKEN])

    def test_max_vocab(self):
        tok = WordTokenizer(max_vocab=2)
        tok.fit(["alpha beta gamma delta"])
        # Only 2 most-frequent words should be in vocab (all freq 1, so alphabetical)
        # Plus 4 special tokens
        self.assertEqual(tok.vocab_size, 6)

    def test_punctuation_split(self):
        tok = WordTokenizer()
        tok.fit(["hello, world!"])
        self.assertIn(",", tok.token_to_id)
        self.assertIn("!", tok.token_to_id)

    def test_unknown_word_to_unk(self):
        tok = WordTokenizer()
        tok.fit(["the cat"])
        ids = tok.encode("the dog")
        self.assertEqual(ids[1], tok.token_to_id[UNK_TOKEN])


class TestBPETokenizer(unittest.TestCase):
    def test_fit_basic(self):
        corpus = ["low low low low low lower newest newest newest newest newest newest widest widest widest"]
        tok = BPETokenizer(vocab_size=50, min_frequency=2)
        tok.fit(corpus)
        self.assertTrue(tok._fitted)
        self.assertGreater(len(tok.merges), 0)
        self.assertLessEqual(tok.vocab_size, 50)

    def test_encode_decode_round_trip(self):
        corpus = ["def add(a, b):", "    return a + b"] * 20
        tok = BPETokenizer(vocab_size=80, min_frequency=2)
        tok.fit(corpus)
        text = "def add(a, b):"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        # The decoded text should be close to the original (may differ in spacing)
        self.assertEqual(decoded.replace(" ", ""), text.replace(" ", ""))

    def test_unseen_word_decomposes_into_subwords(self):
        corpus = ["running", "runner", "runs", "ran"] * 10
        tok = BPETokenizer(vocab_size=50, min_frequency=2)
        tok.fit(corpus)
        # "runners" wasn't in corpus but should tokenize via subwords
        ids = tok.encode("runners")
        self.assertGreater(len(ids), 0)
        # None of the tokens should be UNK (since BPE should decompose)
        unk_id = tok.token_to_id[UNK_TOKEN]
        # Allow at most 1 UNK (in case a char wasn't seen)
        unk_count = sum(1 for i in ids if i == unk_id)
        self.assertLessEqual(unk_count, 1)

    def test_merges_recorded(self):
        corpus = ["abababab"] * 10  # high-frequency "ab" pair
        tok = BPETokenizer(vocab_size=20, min_frequency=2)
        tok.fit(corpus)
        # Should have merged 'a' + 'b' into 'ab'
        merged_symbols = [m for (_, m) in tok.merges]
        self.assertIn("ab", merged_symbols)

    def test_vocab_size_respected(self):
        corpus = ["abcdefghijklmnop"] * 5
        tok = BPETokenizer(vocab_size=30, min_frequency=2)
        tok.fit(corpus)
        self.assertLessEqual(tok.vocab_size, 30)


class TestSaveLoad(unittest.TestCase):
    def test_char_save_load(self):
        tok = CharTokenizer()
        tok.fit(["hello world"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = BaseTokenizer.load(path)
            self.assertIsInstance(tok2, CharTokenizer)
            self.assertEqual(tok.vocab_size, tok2.vocab_size)
            self.assertEqual(tok.encode("hello"), tok2.encode("hello"))
        finally:
            os.unlink(path)

    def test_word_save_load(self):
        tok = WordTokenizer()
        tok.fit(["the cat sat on the mat"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = BaseTokenizer.load(path)
            self.assertIsInstance(tok2, WordTokenizer)
            self.assertEqual(tok.encode("the cat"), tok2.encode("the cat"))
        finally:
            os.unlink(path)

    def test_bpe_save_load(self):
        corpus = ["def add(a, b):", "    return a + b"] * 10
        tok = BPETokenizer(vocab_size=50, min_frequency=2)
        tok.fit(corpus)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = BaseTokenizer.load(path)
            self.assertIsInstance(tok2, BPETokenizer)
            self.assertEqual(tok.merges, tok2.merges)
            self.assertEqual(tok.encode("def"), tok2.encode("def"))
        finally:
            os.unlink(path)

    def test_load_tokenizer_module_function(self):
        tok = CharTokenizer()
        tok.fit(["test"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = qai.tokenizers.load(path)
            self.assertEqual(tok.encode("test"), tok2.encode("test"))
        finally:
            os.unlink(path)


class TestBatchOperations(unittest.TestCase):
    def test_encode_batch_padded(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        batch = tok.encode_batch(["a", "ab", "abc"], padding=True)
        # All should be padded to length 3
        self.assertEqual(len(batch), 3)
        self.assertEqual(len(batch[0]), 3)
        self.assertEqual(len(batch[1]), 3)
        self.assertEqual(len(batch[2]), 3)
        # First sequence should end with PAD_ID
        self.assertEqual(batch[0][1:], [PAD_ID, PAD_ID])

    def test_encode_batch_max_length(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        batch = tok.encode_batch(["a", "abc", "abcdef"], max_length=4, padding=True, truncation=True)
        for ids in batch:
            self.assertEqual(len(ids), 4)

    def test_decode_batch(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        batch = tok.encode_batch(["a", "b", "c"], padding=True)
        decoded = tok.decode_batch(batch)
        self.assertEqual(decoded[0], "a")
        self.assertEqual(decoded[1], "b")
        self.assertEqual(decoded[2], "c")


class TestVocabIntrospection(unittest.TestCase):
    def test_vocab_size_property(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        # 4 special + 3 chars = 7
        self.assertEqual(tok.vocab_size, 7)

    def test_get_vocab(self):
        tok = CharTokenizer()
        tok.fit(["ab"])
        v = tok.get_vocab()
        self.assertIn(PAD_TOKEN, v)
        self.assertIn(UNK_TOKEN, v)
        self.assertIn("a", v)
        self.assertIn("b", v)

    def test_token_to_id_get(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        self.assertEqual(tok.token_to_id_get("a"), tok.token_to_id["a"])
        # Unknown token returns UNK_ID
        self.assertEqual(tok.token_to_id_get("z"), tok.token_to_id[UNK_TOKEN])

    def test_id_to_token_get(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        self.assertEqual(tok.id_to_token_get(0), PAD_TOKEN)
        # Out of range returns unk_token
        self.assertEqual(tok.id_to_token_get(99999), UNK_TOKEN)


class TestIntegrationWithQai(unittest.TestCase):
    def test_tokenizers_accessible_from_qai_top_level(self):
        self.assertTrue(hasattr(qai, "CharTokenizer"))
        self.assertTrue(hasattr(qai, "WordTokenizer"))
        self.assertTrue(hasattr(qai, "BPETokenizer"))
        self.assertTrue(hasattr(qai, "tokenizers"))

    def test_tokenizers_use_with_numpy(self):
        tok = CharTokenizer()
        tok.fit(["hello"])
        ids = tok.encode("hello")
        arr = np.array(ids, dtype=np.int64)
        self.assertEqual(arr.dtype, np.int64)
        self.assertEqual(len(arr), 5)


if __name__ == "__main__":
    unittest.main()
