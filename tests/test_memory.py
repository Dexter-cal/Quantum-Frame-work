"""Tests for qai.memory — three-tier memory system."""
import sys, os, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qai
from qai.memory import (
    Memory, WorkingMemory, WorkingItem, RetrievalMemory, RetrievalResult,
    WeightsMemory, TFIDFEmbedder, hash_embedding, random_embedding,
)


class TestEmbeddingFunctions(unittest.TestCase):
    def test_hash_embedding_deterministic(self):
        v1 = hash_embedding("hello world", dim=64)
        v2 = hash_embedding("hello world", dim=64)
        self.assertTrue(np.allclose(v1, v2))

    def test_hash_embedding_normalized(self):
        v = hash_embedding("test", dim=64)
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=5)

    def test_hash_embedding_different_texts_different_vectors(self):
        v1 = hash_embedding("hello", dim=64)
        v2 = hash_embedding("world", dim=64)
        self.assertFalse(np.allclose(v1, v2))

    def test_random_embedding_deterministic_with_seed(self):
        v1 = random_embedding("hello", dim=64, seed=42)
        v2 = random_embedding("hello", dim=64, seed=42)
        self.assertTrue(np.allclose(v1, v2))

    def test_tfidf_embedder(self):
        emb = TFIDFEmbedder(dim=32)
        v1 = emb.embed("the cat sat on the mat")
        v2 = emb.embed("the dog sat on the log")
        v3 = emb.embed("python programming language")
        # Similar texts should have higher cosine sim than dissimilar
        sim_similar = float(np.dot(v1, v2))
        sim_diff = float(np.dot(v1, v3))
        self.assertGreater(sim_similar, sim_diff)


class TestWorkingMemory(unittest.TestCase):
    def test_add_and_recent(self):
        wm = WorkingMemory(capacity=10)
        wm.add("first")
        wm.add("second")
        wm.add("third")
        self.assertEqual(len(wm), 3)
        self.assertEqual(wm.recent(2), ["second", "third"])

    def test_capacity_eviction_fifo(self):
        wm = WorkingMemory(capacity=3, policy="fifo")
        wm.add("a"); wm.add("b"); wm.add("c"); wm.add("d")
        self.assertEqual(len(wm), 3)
        # FIFO evicts "a" first
        self.assertEqual(wm.recent(3), ["b", "c", "d"])

    def test_capacity_eviction_lru(self):
        wm = WorkingMemory(capacity=3, policy="lru")
        wm.add("a"); wm.add("b"); wm.add("c")
        # Touch "a" to make it recently used
        wm.get("a")
        wm.add("d")  # should evict "b" (least recently used)
        texts = wm.recent(3)
        self.assertIn("a", texts)
        self.assertIn("c", texts)
        self.assertIn("d", texts)
        self.assertNotIn("b", texts)

    def test_capacity_eviction_importance(self):
        wm = WorkingMemory(capacity=3, policy="importance")
        wm.add("low", importance=0.1)
        wm.add("mid", importance=0.5)
        wm.add("high", importance=0.9)
        wm.add("new", importance=0.7)  # should evict "low"
        texts = wm.recent(4)
        self.assertIn("mid", texts)
        self.assertIn("high", texts)
        self.assertIn("new", texts)
        self.assertNotIn("low", texts)

    def test_invalid_policy_raises(self):
        with self.assertRaises(ValueError):
            WorkingMemory(policy="invalid")

    def test_search_substring(self):
        wm = WorkingMemory()
        wm.add("the cat sat on the mat")
        wm.add("the dog sat on the log")
        results = wm.search("cat")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "the cat sat on the mat")

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("a"); wm.add("b")
        wm.clear()
        self.assertEqual(len(wm), 0)

    def test_summary(self):
        wm = WorkingMemory(capacity=5, policy="lru")
        wm.add("a", importance=0.5)
        s = wm.summary()
        self.assertEqual(s["tier"], "working")
        self.assertEqual(s["capacity"], 5)
        self.assertEqual(s["size"], 1)
        self.assertEqual(s["policy"], "lru")

    def test_audit_log(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        log = wm.audit_log()
        self.assertGreaterEqual(len(log), 2)
        self.assertEqual(log[0]["action"], "add")
        self.assertEqual(log[1]["action"], "add")

    def test_custom_eviction_fn(self):
        # Custom: always evict the longest text (including the new one if it's longest)
        def evict_longest(items, n):
            indexed = sorted(range(len(items)), key=lambda i: -len(items[i].text))
            return indexed[:n]
        wm = WorkingMemory(capacity=2, eviction_fn=evict_longest)
        wm.add("short")         # items: ["short"]
        wm.add("medium length") # items: ["short", "medium length"]
        wm.add("very long text here")  # over capacity, evict longest
        # At eviction time, items are: ["short", "medium length", "very long text here"]
        # Longest is "very long text here" (20 chars) → evicted
        # Final: ["short", "medium length"]
        texts = wm.recent(3)
        self.assertIn("short", texts)
        self.assertIn("medium length", texts)
        self.assertEqual(len(wm), 2)


class TestRetrievalMemory(unittest.TestCase):
    def test_add_and_search(self):
        rm = RetrievalMemory(embed_dim=64)
        rm.add("Paris is the capital of France")
        rm.add("France is in Europe")
        rm.add("Python is a programming language")
        results = rm.search("capital of France", k=2)
        self.assertLessEqual(len(results), 2)
        # Top result should be the Paris fact
        self.assertGreater(results[0].score, 0)
        self.assertIn("Paris", results[0].text)

    def test_search_empty_returns_empty(self):
        rm = RetrievalMemory()
        self.assertEqual(rm.search("anything"), [])

    def test_get_by_id(self):
        rm = RetrievalMemory(embed_dim=32)
        id1, _ = rm.add("first fact")
        id2, _ = rm.add("second fact")
        result = rm.get(id1)
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "first fact")
        self.assertEqual(result.id, id1)
        self.assertIsNone(rm.get(99999))

    def test_remove(self):
        rm = RetrievalMemory(embed_dim=32)
        id1, _ = rm.add("fact")
        self.assertTrue(rm.remove(id1))
        self.assertEqual(len(rm), 0)
        self.assertFalse(rm.remove(id1))

    def test_update_metadata(self):
        rm = RetrievalMemory(embed_dim=32)
        id1, _ = rm.add("fact", metadata={"source": "test"})
        self.assertTrue(rm.update_metadata(id1, {"verified": True}))
        result = rm.get(id1)
        self.assertTrue(result.metadata["verified"])
        self.assertEqual(result.metadata["source"], "test")

    def test_filter_fn(self):
        rm = RetrievalMemory(embed_dim=32)
        rm.add("public fact", metadata={"visibility": "public"})
        rm.add("private fact", metadata={"visibility": "private"})
        rm.add("another public", metadata={"visibility": "public"})
        results = rm.search("fact", k=10, filter_fn=lambda m: m.get("visibility") == "public")
        self.assertTrue(all(r.metadata["visibility"] == "public" for r in results))

    def test_conflict_detection(self):
        # Use TFIDFEmbedder for more semantic similarity
        emb = TFIDFEmbedder(dim=64)
        rm = RetrievalMemory(embed_fn=emb, conflict_threshold=0.5)
        rm.add("Paris is the capital of France", metadata={"source": "wiki", "fact": "capital"})
        _, conflicts = rm.add("Paris is the capital of France", metadata={"source": "wiki", "fact": "capital"})
        # Exact duplicate should be flagged
        self.assertGreater(len(conflicts), 0)
        self.assertEqual(conflicts[0]["type"], "duplicate")

    def test_save_load_round_trip(self):
        rm = RetrievalMemory(embed_dim=32)
        rm.add("first fact", metadata={"source": "test"})
        rm.add("second fact", metadata={"source": "test"})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            rm.save(path)
            rm2 = RetrievalMemory.load(path)
            self.assertEqual(len(rm2), 2)
            # Search should still work
            results = rm2.search("first", k=1)
            self.assertGreater(len(results), 0)
        finally:
            os.unlink(path)

    def test_tfidf_embedder_save_load(self):
        emb = TFIDFEmbedder(dim=32)
        rm = RetrievalMemory(embed_fn=emb, conflict_threshold=0.5)
        rm.add("first fact")
        rm.add("second fact")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            rm.save(path)
            rm2 = RetrievalMemory.load(path)
            self.assertEqual(len(rm2), 2)
            # The TFIDFEmbedder state should be restored
            self.assertIsInstance(rm2.embed_fn, TFIDFEmbedder)
        finally:
            os.unlink(path)


class TestWeightsMemory(unittest.TestCase):
    def test_attach_and_hash(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)
        wmem = WeightsMemory()
        wmem.attach(m)
        h = wmem.hash()
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)  # SHA-256

    def test_diff_detects_changes(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)
        wmem = WeightsMemory()
        wmem.attach(m)
        # Modify a weight
        m.technique.w[0] = 99.0
        diff = wmem.diff(m)
        self.assertFalse(diff["overall"]["hash_match"])
        self.assertGreater(diff["overall"]["max_abs_change"], 1.0)

    def test_diff_no_changes(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)
        wmem = WeightsMemory()
        wmem.attach(m)
        diff = wmem.diff(m)
        self.assertTrue(diff["overall"]["hash_match"])
        self.assertAlmostEqual(diff["overall"]["max_abs_change"], 0.0, places=6)

    def test_restore_regression(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)
        original_pred = m.predict([4.0])

        wmem = WeightsMemory()
        wmem.attach(m)
        # Mess up
        m.technique.w[0] = 99.0
        m.technique.b = 99.0
        messed_pred = m.predict([4.0])
        self.assertNotAlmostEqual(original_pred, messed_pred, places=2)
        # Restore
        wmem.restore(m)
        restored_pred = m.predict([4.0])
        self.assertAlmostEqual(original_pred, restored_pred, places=4)

    def test_restore_transformer(self):
        # Tiny transformer test
        tok = qai.CharTokenizer()
        tok.fit(["ab", "cd"])
        m = qai.build(technique="transformer", vocab_size=tok.vocab_size,
                      d_model=16, n_heads=2, n_layers=1, max_seq_len=8, epochs=1)
        m.train(["abcd"], tokenizer=tok, verbose=False)
        wmem = WeightsMemory()
        wmem.attach(m)
        # Diff with no changes -> hash should match
        diff = wmem.diff(m)
        self.assertTrue(diff["overall"]["hash_match"])
        # Restore should not crash
        wmem.restore(m)

    def test_summary_unattached(self):
        wmem = WeightsMemory()
        s = wmem.summary()
        self.assertFalse(s["attached"])

    def test_summary_attached(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)
        wmem = WeightsMemory()
        wmem.attach(m)
        s = wmem.summary()
        self.assertTrue(s["attached"])
        self.assertGreater(s["n_tensors"], 0)
        self.assertGreater(s["total_params"], 0)

    def test_attach_unsupported_model(self):
        wmem = WeightsMemory()
        with self.assertRaises(ValueError):
            wmem.attach("not a model")


class TestMemoryCoordinator(unittest.TestCase):
    def test_default_construction(self):
        mem = Memory()
        self.assertIsInstance(mem.working, WorkingMemory)
        self.assertIsInstance(mem.retrieval, RetrievalMemory)
        self.assertIsInstance(mem.weights, WeightsMemory)

    def test_auto_link_promotion(self):
        mem = Memory(auto_link=True)
        # auto_link sets threshold=5
        self.assertEqual(mem.working.auto_promote_threshold, 5)
        # Add same item 5 times via .get() to bump access_count
        mem.working.add("important fact", metadata={"source": "test"})
        for _ in range(5):
            mem.working.get("important fact")
        # Should have been auto-promoted to retrieval
        self.assertEqual(len(mem.retrieval), 1)

    def test_attach(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)
        mem = Memory()
        mem.attach(m)
        self.assertTrue(mem.weights.summary()["attached"])

    def test_save_load_round_trip(self):
        mem = Memory()
        mem.working.add("working item 1")
        mem.working.add("working item 2")
        mem.retrieval.add("retrieval item 1")
        mem.retrieval.add("retrieval item 2")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            mem.save(path)
            mem2 = Memory.load(path)
            self.assertEqual(len(mem2.working), 2)
            self.assertEqual(len(mem2.retrieval), 2)
            # Working items preserved
            texts = mem2.working.recent(2)
            self.assertIn("working item 1", texts)
            self.assertIn("working item 2", texts)
        finally:
            os.unlink(path)
            # Sibling retrieval file
            retrieval_path = path.replace(".json", ".retrieval.json")
            if os.path.exists(retrieval_path):
                os.unlink(retrieval_path)

    def test_summary(self):
        mem = Memory()
        mem.working.add("a")
        mem.retrieval.add("b")
        s = mem.summary()
        self.assertEqual(s["working"]["size"], 1)
        self.assertEqual(s["retrieval"]["size"], 1)
        self.assertEqual(s["weights"]["attached"], False)

    def test_full_audit(self):
        mem = Memory()
        mem.working.add("a")
        mem.retrieval.add("b")
        audit = mem.full_audit()
        self.assertIn("working", audit)
        self.assertIn("retrieval", audit)
        self.assertIn("weights", audit)
        self.assertGreaterEqual(len(audit["working"]), 1)
        self.assertGreaterEqual(len(audit["retrieval"]), 1)


class TestIntegrationWithQai(unittest.TestCase):
    def test_accessible_from_top_level(self):
        self.assertTrue(hasattr(qai, "Memory"))
        self.assertTrue(hasattr(qai, "WorkingMemory"))
        self.assertTrue(hasattr(qai, "RetrievalMemory"))
        self.assertTrue(hasattr(qai, "WeightsMemory"))
        self.assertTrue(hasattr(qai, "TFIDFEmbedder"))
        self.assertTrue(hasattr(qai, "memory"))

    def test_memory_with_logic_decorator(self):
        """A @qai.logic function can read/write memory."""
        mem = qai.Memory()

        @qai.logic
        def answer_with_context(question):
            # Look up in retrieval
            results = mem.retrieval.search(question, k=1)
            context = results[0].text if results else "(no context)"
            # Add question to working
            mem.working.add(f"Q: {question}")
            return f"answer based on: {context}"

        mem.retrieval.add("Paris is the capital of France")
        result = answer_with_context("capital of France")
        self.assertIn("Paris", result)
        # Working memory should have the question
        self.assertEqual(len(mem.working), 1)


if __name__ == "__main__":
    unittest.main()
