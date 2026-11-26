"""Tests for qai.advanced — RAG, Knowledge Distillation, AI Agent + qai.techniques.mamba."""
import sys, os, unittest, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qai


class TestMamba(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = qai.CharTokenizer()
        cls.corpus = ["def add(a, b):", "    return a + b", "def sub(a, b):", "    return a - b"] * 5
        cls.tok.fit(cls.corpus)
        cls.model = qai.build(technique="mamba", vocab_size=cls.tok.vocab_size,
                              d_model=32, n_layers=2, state_dim=8, max_seq_len=32,
                              task_type="generation", epochs=10, lr=1e-3, batch_size=4)
        cls.model.train(cls.corpus, tokenizer=cls.tok, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_generate(self):
        out = self.model.generate("def ", max_new_tokens=10, tokenizer=self.tok, temperature=0)
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("def "))

    def test_classification(self):
        texts = ["hello world", "goodbye world"] * 5
        labels = [1, 0] * 5
        tok2 = qai.CharTokenizer(); tok2.fit(texts)
        m = qai.build(technique="mamba", vocab_size=tok2.vocab_size, d_model=16,
                      n_layers=2, state_dim=4, max_seq_len=16,
                      task_type="classification", n_classes=2, epochs=5)
        m.train(texts, y=labels, tokenizer=tok2, verbose=False)
        pred = m.predict("hello world")
        self.assertIsInstance(pred, (int, np.integer))

    def test_hybrid_mode(self):
        m = qai.build(technique="mamba", vocab_size=self.tok.vocab_size,
                      d_model=16, n_layers=4, state_dim=4, max_seq_len=16,
                      use_attention_layers=True, attention_ratio=4, epochs=2)
        m.train(self.corpus, tokenizer=self.tok, verbose=False)
        self.assertTrue(m.technique.is_trained())

    def test_parameter_count(self):
        self.assertGreater(self.model.technique.parameter_count(), 0)

    def test_in_registry(self):
        from qai.techniques import _REGISTRY
        self.assertIn("mamba", _REGISTRY)


class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = qai.CharTokenizer()
        corpus = ["def add(a, b):", "    return a + b"] * 5
        cls.tok.fit(corpus)
        cls.model = qai.build(technique="mamba", vocab_size=cls.tok.vocab_size,
                              d_model=16, n_layers=2, state_dim=4, max_seq_len=16,
                              task_type="generation", epochs=5, lr=1e-3, batch_size=4)
        cls.model.train(corpus, tokenizer=cls.tok, verbose=False)

    def test_add_and_retrieve(self):
        rag = qai.RAGPipeline(model=self.model, tokenizer=self.tok, top_k=2)
        rag.add_document("Paris is the capital of France")
        rag.add_document("France is in Europe")
        docs = rag.retrieve("capital of France")
        self.assertGreater(len(docs), 0)

    def test_query(self):
        rag = qai.RAGPipeline(model=self.model, tokenizer=self.tok, top_k=1)
        rag.add_document("def add(a, b): return a + b")
        result = rag.query("def add", max_new_tokens=5, return_context=True)
        self.assertIn("answer", result)
        self.assertIn("retrieved_docs", result)
        self.assertIn("prompt", result)

    def test_add_multiple(self):
        rag = qai.RAGPipeline(model=self.model, tokenizer=self.tok)
        ids = rag.add_documents(["doc1", "doc2", "doc3"])
        self.assertEqual(len(ids), 3)

    def test_query_log(self):
        rag = qai.RAGPipeline(model=self.model, tokenizer=self.tok)
        rag.add_document("test doc")
        rag.query("test", max_new_tokens=2)
        self.assertGreater(len(rag.query_log()), 0)


class TestKnowledgeDistillation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = np.random.randn(100, 8).astype(np.float32)
        cls.y = np.random.randint(0, 3, 100)
        cls.teacher = qai.build(technique="neural_network", hidden_dim=64, epochs=10, lr=0.01)
        cls.teacher.train(cls.X, cls.y, verbose=False)

    def test_distill_neural_network(self):
        student = qai.build(technique="neural_network", hidden_dim=16, epochs=1)
        distiller = qai.KnowledgeDistiller(teacher=self.teacher, student=student,
                                            temperature=4.0, alpha=0.5)
        distiller.distill(self.X, self.y, epochs=5, lr=1e-3, batch_size=16, verbose=False)
        self.assertTrue(student.technique.is_trained())
        self.assertGreater(len(distiller.losses()), 0)

    def test_compression_ratio(self):
        student = qai.build(technique="neural_network", hidden_dim=8, epochs=1)
        distiller = qai.KnowledgeDistiller(teacher=self.teacher, student=student)
        distiller.distill(self.X, self.y, epochs=2, lr=1e-3, batch_size=16, verbose=False)
        ratio = distiller.compression_ratio()
        self.assertGreater(ratio, 1.0)  # teacher should be bigger

    def test_sklearn_fallback(self):
        # Distillation from sklearn-backed model should use fallback
        teacher_sk = qai.build(technique="random_forest")
        teacher_sk.train(self.X, self.y, verbose=False)
        student_sk = qai.build(technique="decision_tree")
        distiller = qai.KnowledgeDistiller(teacher=teacher_sk, student=student_sk)
        distiller.distill(self.X, self.y, epochs=1, verbose=False)
        self.assertTrue(student_sk.technique.is_trained())


class TestAIAgent(unittest.TestCase):
    def setUp(self):
        def calc(expression):
            try:
                return {"value": eval(expression), "done": False}
            except:
                return {"error": "bad expression"}

        def write_file(content, path):
            with open(path, "w") as f:
                f.write(content)
            return {"written": True, "path": path, "done": True}

        self.agent = qai.AIAgent(
            name="test-agent",
            tools={"calculator": calc, "write_file": write_file},
            max_steps=5,
        )

    def test_add_remove_tool(self):
        self.agent.add_tool("echo", lambda: "hello")
        self.assertIn("echo", self.agent.list_tools())
        self.agent.remove_tool("echo")
        self.assertNotIn("echo", self.agent.list_tools())

    def test_run_returns_dict(self):
        result = self.agent.run("Calculate 2 + 3")
        self.assertIsInstance(result, dict)
        self.assertIn("task", result)
        self.assertIn("steps", result)
        self.assertIn("completed", result)
        self.assertIn("thinking", result)

    def test_execution_log(self):
        self.agent.run("Calculate 2 + 3")
        self.assertGreater(len(self.agent.execution_log()), 0)

    def test_thinking_log(self):
        self.agent.run("Calculate 2 + 3")
        self.assertGreater(len(self.agent.thinking()), 0)

    def test_max_steps(self):
        agent = qai.AIAgent(tools={"noop": lambda: {"value": 1}}, max_steps=2)
        result = agent.run("do something")
        self.assertLessEqual(result["n_steps"], 2)

    def test_tool_error_handling(self):
        def bad_tool():
            raise ValueError("crash")
        agent = qai.AIAgent(tools={"bad": bad_tool}, max_steps=1, retry_failed=False)
        result = agent.run("call bad tool")
        # Should not crash — errors are caught
        self.assertIsInstance(result, dict)

    def test_repr(self):
        self.assertIn("test-agent", repr(self.agent))


class TestIntegration(unittest.TestCase):
    def test_advanced_accessible_from_top_level(self):
        self.assertTrue(hasattr(qai, "RAGPipeline"))
        self.assertTrue(hasattr(qai, "KnowledgeDistiller"))
        self.assertTrue(hasattr(qai, "AIAgent"))
        self.assertTrue(hasattr(qai, "advanced"))

    def test_mamba_accessible(self):
        from qai.techniques import _REGISTRY
        self.assertIn("mamba", _REGISTRY)

    def test_rag_with_memory_integration(self):
        """RAG uses qai.memory.RetrievalMemory internally."""
        tok = qai.CharTokenizer()
        tok.fit(["hello world"])
        model = qai.build(technique="mamba", vocab_size=tok.vocab_size,
                          d_model=16, n_layers=1, state_dim=4, max_seq_len=8,
                          task_type="generation", epochs=2)
        model.train(["hello world"], tokenizer=tok, verbose=False)
        rag = qai.RAGPipeline(model=model, tokenizer=tok)
        rag.add_document("hello world document")
        result = rag.query("hello", max_new_tokens=3)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
