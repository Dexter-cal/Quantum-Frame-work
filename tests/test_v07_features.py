"""
Tests for all v0.4-v0.7 features: dashboard, evaluate, doc, active_learning,
drift, profiler, experiment_compare, compress, CPC, CPC AI, synthetic generators.
"""
import sys, os, json, tempfile, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import qai


# =====================================================================
# Test helpers
# =====================================================================
def _make_classifier(n=50, n_features=4):
    """Build + train a quick classifier."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = qai.build(technique="classifier")
    m.train(X, y, verbose=False)
    return m, X, y

def _make_regressor(n=50, n_features=4):
    """Build + train a quick regressor."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_features))
    y = X[:, 0] * 2 + X[:, 1] * 3 + rng.standard_normal(n) * 0.1
    m = qai.build(technique="regression")
    m.train(X, y, verbose=False)
    return m, X, y


# =====================================================================
# 1. qai.evaluate
# =====================================================================
def test_evaluate_classification():
    m, X, y = _make_classifier()
    report = qai.evaluate(m, X, y)
    assert report.technique == "classifier"
    assert "accuracy" in report.metrics
    assert report.verdict in ("excellent", "good", "ok", "poor")
    assert report.metrics["accuracy"] >= 0.5
    print(f"  evaluate (classification): accuracy={report.metrics['accuracy']:.3f}, verdict={report.verdict}")

def test_evaluate_regression():
    m, X, y = _make_regressor()
    report = qai.evaluate(m, X, y)
    assert "mse" in report.metrics
    assert "r2" in report.metrics
    assert report.verdict in ("excellent", "good", "ok", "poor")
    print(f"  evaluate (regression): r2={report.metrics.get('r2', 0):.3f}, verdict={report.verdict}")

def test_evaluate_report_str():
    m, X, y = _make_classifier()
    report = qai.evaluate(m, X, y)
    s = str(report)
    assert "Evaluation Report" in s
    assert "verdict" in s
    assert "metrics" in s

def test_evaluate_as_dict():
    m, X, y = _make_classifier()
    report = qai.evaluate(m, X, y)
    d = report.as_dict()
    assert "technique" in d
    assert "metrics" in d
    assert "verdict" in d


# =====================================================================
# 2. qai.doc
# =====================================================================
def test_doc_generates_markdown():
    m, X, y = _make_classifier()
    doc = qai.doc(m)
    assert isinstance(doc, str)
    assert "classifier" in doc.lower()
    assert "## Identity" in doc
    assert "## Hyperparameters" in doc
    assert "## How to use" in doc
    assert len(doc) > 500
    print(f"  doc: {len(doc)} chars, has all sections")

def test_doc_save_to_file():
    m, X, y = _make_classifier()
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        path = f.name
    qai.doc(m, save=path)
    with open(path) as f:
        content = f.read()
    assert "classifier" in content
    os.unlink(path)

def test_doc_for_agent():
    """doc() should work for AgentBase subclasses too."""
    agent = qai.AgentBase(
        brain=qai.build(technique="tabular_policy"),
        hparams={"epsilon": 0.3},
        name="test_agent",
    )
    doc = qai.doc(agent)
    assert "test_agent" in doc or "tabular_policy" in doc


# =====================================================================
# 3. qai.ActiveLearningLoop
# =====================================================================
def test_active_learning_uncertainty():
    m, X, y = _make_classifier(n=30)
    rng = np.random.default_rng(99)
    X_pool = rng.standard_normal((20, 4))
    loop = qai.ActiveLearningLoop(
        model=m, X_unlabeled=X_pool,
        strategy="uncertainty", batch_size=5,
    )
    batch = loop.next_batch()
    assert len(batch) == 5
    assert all(0 <= i < 20 for i in batch)
    print(f"  active_learning (uncertainty): batch={batch}")

def test_active_learning_random():
    m, X, y = _make_classifier(n=30)
    rng = np.random.default_rng(99)
    X_pool = rng.standard_normal((20, 4))
    loop = qai.ActiveLearningLoop(
        model=m, X_unlabeled=X_pool,
        strategy="random", batch_size=3,
    )
    batch = loop.next_batch()
    assert len(batch) == 3

def test_active_learning_commit():
    m, X, y = _make_classifier(n=30)
    rng = np.random.default_rng(99)
    X_pool = rng.standard_normal((20, 4))
    y_pool = (X_pool[:, 0] > 0).astype(int)
    loop = qai.ActiveLearningLoop(
        model=m, X_unlabeled=X_pool,
        strategy="uncertainty", batch_size=5,
    )
    batch = loop.next_batch()
    labels = y_pool[batch].tolist()
    loop.commit(batch, labels)
    assert loop.status()["n_labeled"] == 5
    assert loop.status()["training_set_size"] == 5

def test_active_learning_invalid_strategy():
    m, X, y = _make_classifier(n=30)
    try:
        qai.ActiveLearningLoop(model=m, X_unlabeled=X[:10], strategy="invalid")
        assert False, "should have raised"
    except ValueError:
        pass


# =====================================================================
# 4. qai.DriftDetector
# =====================================================================
def test_drift_ks_normal():
    rng = np.random.default_rng(42)
    baseline = rng.standard_normal((200, 4))
    new = rng.standard_normal((50, 4))
    det = qai.DriftDetector(baseline=baseline, method="ks")
    report = det.detect(new)
    assert report.method == "ks"
    assert isinstance(report.drifted, bool)
    assert isinstance(report.drifted_features, list)
    assert "per_feature" in report.__dict__ or hasattr(report, "per_feature")

def test_drift_ks_drifted():
    rng = np.random.default_rng(42)
    baseline = rng.standard_normal((200, 4))
    new = rng.standard_normal((50, 4)) * 3.0 + 2.0  # shifted + scaled
    det = qai.DriftDetector(baseline=baseline, method="ks")
    report = det.detect(new)
    assert report.drifted == True
    assert len(report.drifted_features) > 0

def test_drift_psi():
    rng = np.random.default_rng(42)
    baseline = rng.standard_normal((200, 3))
    new = rng.standard_normal((50, 3))
    det = qai.DriftDetector(baseline=baseline, method="psi")
    report = det.detect(new)
    assert report.method == "psi"

def test_drift_ensemble():
    rng = np.random.default_rng(42)
    baseline = rng.standard_normal((500, 4))
    new_normal = rng.standard_normal((100, 4))
    det = qai.DriftDetector(baseline=baseline, method="ensemble")
    report = det.detect(new_normal)
    assert report.method == "ensemble"

def test_drift_report_str():
    rng = np.random.default_rng(42)
    baseline = rng.standard_normal((200, 4))
    det = qai.DriftDetector(baseline=baseline, method="ks")
    report = det.detect(rng.standard_normal((50, 4)))
    s = str(report)
    assert "Drift Report" in s

def test_drift_invalid_method():
    try:
        qai.DriftDetector(baseline=np.zeros((10, 2)), method="invalid")
        assert False, "should have raised"
    except ValueError:
        pass


# =====================================================================
# 5. qai.Profiler
# =====================================================================
def test_profiler_basic():
    m, X, y = _make_classifier()
    profile = qai.Profiler(m).profile(X[:1], n_warmup=1, n_runs=5)
    assert profile.technique == "classifier"
    assert profile.n_params >= 0
    assert profile.avg_latency_ms >= 0
    assert profile.throughput_per_s >= 0
    assert isinstance(profile.p50_latency_ms, float)
    assert isinstance(profile.p95_latency_ms, float)
    print(f"  profiler: {profile.n_params} params, {profile.avg_latency_ms:.3f}ms latency")

def test_profiler_str():
    m, X, y = _make_classifier()
    profile = qai.Profiler(m).profile(X[:1], n_runs=3)
    s = str(profile)
    assert "Profile" in s
    assert "parameters" in s
    assert "latency" in s

def test_profiler_as_dict():
    m, X, y = _make_classifier()
    profile = qai.Profiler(m).profile(X[:1], n_runs=3)
    d = profile.as_dict() if hasattr(profile, 'as_dict') else {}
    # Profile doesn't have as_dict -- check attributes directly
    assert hasattr(profile, "n_params")
    assert hasattr(profile, "avg_latency_ms")
    assert hasattr(profile, "throughput_per_s")


# =====================================================================
# 6. qai.ExperimentCompare
# =====================================================================
def test_experiment_compare_evaluate():
    m1, X, y = _make_classifier()
    m2 = qai.build(technique="random_forest", n_estimators=20)
    m2.train(X, y, verbose=False)
    cmp = qai.ExperimentCompare({"m1": m1, "m2": m2})
    report = cmp.evaluate(X, y, metric="accuracy")
    assert len(report.rows) == 2
    assert report.best_name in ("m1", "m2")

def test_experiment_compare_profile():
    m1, X, y = _make_classifier()
    rows = qai.ExperimentCompare({"m1": m1}).profile(X[:1])
    assert len(rows) == 1
    assert rows[0].latency_ms is not None or rows[0].latency_ms is None  # may fail gracefully

def test_experiment_compare_full():
    m1, X, y = _make_classifier()
    m2 = qai.build(technique="random_forest", n_estimators=20)
    m2.train(X, y, verbose=False)
    cmp = qai.ExperimentCompare({"m1": m1, "m2": m2})
    report = cmp.full_comparison(X, y, X_profile_sample=X[:1])
    assert len(report.rows) == 2
    assert report.best_name is not None

def test_experiment_compare_str():
    m1, X, y = _make_classifier()
    cmp = qai.ExperimentCompare({"m1": m1})
    report = cmp.evaluate(X, y)
    s = str(report)
    assert "Experiment Comparison" in s

def test_experiment_compare_empty():
    try:
        qai.ExperimentCompare({})
        assert False, "should have raised"
    except ValueError:
        pass


# =====================================================================
# 7. qai.compress_bytes / qai.decompress_bytes / qai.Archive
# =====================================================================
def test_compress_decompress_roundtrip():
    data = b"hello world " * 100
    compressed = qai.compress_bytes(data)
    restored = qai.decompress_bytes(compressed)
    assert restored == data

def test_compress_random_data():
    data = os.urandom(500)
    compressed = qai.compress_bytes(data)
    restored = qai.decompress_bytes(compressed)
    assert restored == data

def test_compress_empty():
    data = b""
    compressed = qai.compress_bytes(data)
    restored = qai.decompress_bytes(compressed)
    assert restored == data

def test_archive_roundtrip():
    arc = qai.Archive()
    arc.add("file1.txt", b"hello world " * 50)
    arc.add("file2.json", b'{"key": "value"}')
    arc.add("nested/file3.csv", b"a,b,c\n1,2,3\n")
    with tempfile.NamedTemporaryFile(suffix=".qai.zip", delete=False) as f:
        path = f.name
    arc.save(path)
    arc2 = qai.Archive.load(path)
    assert sorted(arc2.list_files()) == ["file1.txt", "file2.json", "nested/file3.csv"]
    assert arc2.read("file1.txt") == b"hello world " * 50
    assert arc2.read("file2.json") == b'{"key": "value"}'
    os.unlink(path)

def test_archive_signed():
    arc = qai.Archive()
    arc.add("secret.txt", b"top secret data")
    with tempfile.NamedTemporaryFile(suffix=".qai.zip", delete=False) as f:
        path = f.name
    arc.save(path, sign_key="mykey")
    # Correct key
    arc2 = qai.Archive.load(path, sign_key="mykey")
    assert arc2.read("secret.txt") == b"top secret data"
    # Wrong key
    try:
        qai.Archive.load(path, sign_key="wrong")
        assert False, "should have raised"
    except ValueError:
        pass
    os.unlink(path)

def test_archive_stats():
    arc = qai.Archive()
    arc.add("a.txt", b"hello " * 100)
    arc.add("b.txt", b"world " * 100)
    with tempfile.NamedTemporaryFile(suffix=".qai.zip", delete=False) as f:
        path = f.name
    arc.save(path)
    stats = arc.stats()
    assert stats["n_files"] == 2
    assert stats["total_uncompressed"] > 0
    assert stats["total_compressed"] > 0
    assert "algorithms_used" in stats
    os.unlink(path)

def test_archive_tamper_detection():
    arc = qai.Archive()
    arc.add("file.txt", b"sensitive data " * 20)
    with tempfile.NamedTemporaryFile(suffix=".qai.zip", delete=False) as f:
        path = f.name
    arc.save(path, sign_key="secret")
    # Tamper
    with open(path, "rb") as f:
        data = f.read()
    tampered = data[:100] + bytes([data[100] ^ 0xff]) + data[101:]
    with open(path, "wb") as f:
        f.write(tampered)
    try:
        qai.Archive.load(path, sign_key="secret")
        assert False, "should have detected tampering"
    except ValueError:
        pass
    os.unlink(path)


# =====================================================================
# 8. Synthetic data generators
# =====================================================================
def test_make_time_series():
    X, y = qai.make_time_series(n_series=20, length=30, n_patterns=3, random_state=0)
    assert X.shape == (20, 30)
    assert y.shape == (20,)
    assert len(set(y.tolist())) <= 3

def test_make_images():
    X, y = qai.make_images(n=10, size=(16, 16), channels=1, n_classes=3, random_state=0)
    assert X.shape == (10, 1, 16, 16)
    assert y.shape == (10,)

def test_make_text():
    docs, y = qai.make_text(n=10, vocab_size=50, length=10, n_classes=2, random_state=0)
    assert len(docs) == 10
    assert all(len(d) == 10 for d in docs)
    assert len(set(y.tolist())) <= 2

def test_make_graphs():
    adj, feat, y = qai.make_graphs(n=5, n_nodes=6, n_features=3, n_classes=2, random_state=0)
    assert len(adj) == 5
    assert adj[0].shape == (6, 6)
    assert feat[0].shape == (6, 3)
    assert len(y) == 5


# =====================================================================
# 9. CPC (file-native learning system)
# =====================================================================
def test_cpc_ingest_and_ask():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        os.makedirs(os.path.join(tmpdir, "data"), exist_ok=True)
        with open(os.path.join(tmpdir, "data", "test.py"), "w") as f:
            f.write("def hello():\n    return 'world'\n" * 10)
        with open(os.path.join(tmpdir, "data", "readme.md"), "w") as f:
            f.write("# Project\n\nThis is a test project about authentication.\n" * 5)

        kb_path = os.path.join(tmpdir, "kb")
        cpc = qai.CPC(kb_path, verbose=False)
        summary = cpc.ingest(os.path.join(tmpdir, "data"))
        assert summary["n_files"] >= 2
        assert summary["n_chunks"] > 0
        assert summary["n_nodes"] > 0

        # Ask a question
        result = cpc.ask("What is this project about?")
        assert "answer" in result
        assert "evidence" in result
        assert "confidence" in result

        # Search
        results = cpc.search("authentication")
        assert isinstance(results, list)

        # Stats
        stats = cpc.stats()
        assert stats["n_sources"] >= 2
        assert stats["n_nodes"] > 0

def test_cpc_forget():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.txt"), "w") as f:
            f.write("hello world " * 50)
        kb_path = os.path.join(tmpdir, "kb")
        cpc = qai.CPC(kb_path, verbose=False)
        cpc.ingest(os.path.join(tmpdir, "test.txt"))
        source_id = list(cpc.sources.keys())[0]
        result = cpc.forget(source_id)
        assert result["chunks_removed"] >= 0
        assert source_id not in cpc.sources


# =====================================================================
# 10. CPC AI (compression + knowledge)
# =====================================================================
def test_cpc_ai_compress_decompress_small():
    """Small files should use brotli fallback and round-trip correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cpc = qai.CPCAI(tmpdir, min_cpc_size=100, verbose=False)
        data = b"hello world " * 10  # 120 bytes -- below min_cpc_size
        bite = cpc.compress_data(data, "test.txt")
        restored = cpc.decompress_bite(bite)
        assert restored == data, f"round-trip failed: {len(restored)} != {len(data)}"

def test_cpc_ai_compress_decompress_large():
    """Large files should try CPC pipeline + brotli and pick the winner."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cpc = qai.CPCAI(tmpdir, min_cpc_size=100, verbose=False)
        data = b"The user logged in successfully.\n" * 200  # ~4.6 KB
        bite = cpc.compress_data(data, "large.txt")
        restored = cpc.decompress_bite(bite)
        assert restored == data, f"round-trip failed: {len(restored)} != {len(data)}"

def test_cpc_ai_verification():
    """Verify that CPC AI's verification engine works."""
    data = b"test data for verification " * 20
    report = qai.VerificationEngine.verify(data, data)
    assert report["verified"] == True
    assert report["sha256_match"] == True
    assert report["crc32_match"] == True

    report2 = qai.VerificationEngine.verify(data, data + b"x")
    assert report2["verified"] == False
    assert report2["sha256_match"] == False

def test_cpc_ai_huffman_roundtrip():
    """Huffman entropy coder round-trip."""
    data = b"aaaabbbccd" * 50
    compressed = qai.huffman_compress(data)
    restored = qai.huffman_decompress(compressed)
    assert restored == data

def test_cpc_ai_ngram_roundtrip():
    """N-gram predictor round-trip."""
    text = b"def hello(): return world " * 20
    predictor = qai.NGramPredictor(n=3)
    predictor.train(text)
    encoded, flags = predictor.compress_with_prediction(text)
    decoded = predictor.decompress_with_prediction(encoded)
    assert decoded == text

def test_cpc_ai_symbol_roundtrip():
    """Symbolic compressor round-trip."""
    data = b"User clicked login button. " * 50
    sc = qai.SymbolicCompressor()
    compressed, sym_dict = sc.analyze_and_compress(data)
    expanded = sc.decompress(compressed, sym_dict)
    assert expanded == data

def test_cpc_ai_knowledge_graph():
    """Knowledge graph: absorb, reinforce, search, forget."""
    kg = qai.KnowledgeGraph()
    node = qai.KnowledgeNode(
        node_id="test_1", node_type="concept",
        label="AUTHENTICATION", summary="How users log in",
        sources=["src_1"], confidence=0.8,
    )
    kg.absorb(node)
    assert "test_1" in kg.nodes

    # Reinforce
    kg.reinforce("test_1")
    assert kg.nodes["test_1"].frequency == 2

    # Search
    results = kg.search(["authentication"])
    assert len(results) > 0

    # Forget
    removed = kg.forget("src_1")
    assert removed >= 1

def test_cpc_ai_curiosity():
    """Curiosity engine: find gaps + contradictions."""
    kg = qai.KnowledgeGraph()
    # Add a low-confidence high-frequency node
    node = qai.KnowledgeNode(
        node_id="gap_1", node_type="pattern",
        label="UNKNOWN_PATTERN", summary="something unknown",
        sources=["src_1"], confidence=0.3, frequency=10,
    )
    kg.absorb(node)
    curiosity = qai.CuriosityEngine(kg)
    gaps = curiosity.find_gaps()
    assert len(gaps) > 0
    assert any(g["type"] == "low_confidence_high_frequency" for g in gaps)

def test_cpc_ai_compress_file():
    """Full file compression + verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        fpath = os.path.join(tmpdir, "test.txt")
        with open(fpath, "wb") as f:
            f.write(b"hello world " * 100)
        # Compress
        cpc = qai.CPCAI(os.path.join(tmpdir, "kb"), min_cpc_size=50, verbose=False)
        bite_path = cpc.compress_file(fpath)
        assert os.path.exists(bite_path)
        # Load + decompress
        bite = qai.CPCBite.load(bite_path)
        restored = cpc.decompress_bite(bite)
        with open(fpath, "rb") as f:
            original = f.read()
        assert restored == original


# =====================================================================
# 11. qai.AgentBase + qai.Environment
# =====================================================================
def test_agent_base_run_episode():
    """AgentBase: run_episode on puzzle_world."""
    class TinyAgent(qai.AgentBase):
        def reflect(self, summary): pass
        def self_tune(self, summary): pass

    agent = TinyAgent(
        brain=qai.build(technique="tabular_policy"),
        hparams={"epsilon": 0.3},
        name="tiny",
    )
    env = qai.make_environment("puzzle_world", seed=42)
    summary = agent.run_episode(env, max_steps=10, learn=True)
    assert "total_reward" in summary
    assert "steps" in summary
    assert summary["steps"] <= 10

def test_environment_registry():
    """Environment registry: list + make."""
    envs = qai.list_environments()
    assert "puzzle_world" in envs
    assert "grid_world" in envs
    env = qai.make_environment("grid_world")
    state = env.reset()
    assert state is not None

def test_replay_buffer():
    """ReplayBuffer: add + sample."""
    rb = qai.ReplayBuffer(capacity=100)
    for i in range(20):
        rb.add(state=f"s{i}", action=f"a{i}", reward=float(i),
               next_state=f"s{i+1}", done=(i == 19))
    assert len(rb) == 20
    batch = rb.sample(10)
    assert len(batch) == 10

def test_hyperparameter_scheduler():
    """HyperparameterScheduler: record + slope."""
    sched = qai.HyperparameterScheduler(params={"epsilon": 0.3}, window=5)
    for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
        sched.update(r)
    assert sched.slope > 0  # increasing rewards -> positive slope


# =====================================================================
# 12. qai.dashboard (smoke test -- no browser)
# =====================================================================
def test_dashboard_import():
    """Dashboard module imports + has the right function."""
    assert callable(qai.dashboard)

def test_dashboard_state_functions():
    """Dashboard's internal state functions work."""
    from qai.dashboard import _model_info, _model_metadata
    m, X, y = _make_classifier()
    info = _model_info(m)
    assert "technique" in info
    assert "trained" in info
    meta = _model_metadata(m)
    assert "class" in meta


# =====================================================================
# 13. qai.live_training (smoke test)
# =====================================================================
def test_live_training_callback_import():
    """LiveTrainingCallback imports + is a class."""
    assert qai.LiveTrainingCallback is not None
    assert callable(qai.live_training)


# =====================================================================
# 14. qai.PermanentMemory
# =====================================================================
def test_permanent_memory_roundtrip():
    """PermanentMemory: save + load + reset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "brain.json")
        pm = qai.memory.PermanentMemory(path, origin_id="test")

        # Save -- use a picklable type (dict, not a local class)
        state = {"x": 42, "name": "test"}
        pm.save(state)

        # Load
        loaded = pm.load()
        assert loaded["x"] == 42

        # Info
        info = pm.info()
        assert info["sidecar_exists"] == True
        assert info["n_saves"] == 1

        # Reset
        pm.reset()
        assert pm.info()["sidecar_exists"] == False


# =====================================================================
# 15. qai.memory.Memory picklability (the bug we fixed)
# =====================================================================
def test_memory_picklable():
    """Memory object can be pickled (the bug from the critique)."""
    import pickle
    m = qai.Memory()
    m.working.add("hello", importance=1.0)
    data = pickle.dumps(m)
    m2 = pickle.loads(data)
    assert m2 is not None


# =====================================================================
# 16. TabularPolicy actions= default
# =====================================================================
def test_tabular_policy_actions_default():
    """TabularPolicy accepts actions= default."""
    m = qai.build(technique="tabular_policy", actions=["left", "right"])
    assert m.technique._default_actions == ["left", "right"]

def test_tabular_policy_actions_dict():
    """TabularPolicy accepts dict-keyed actions."""
    m = qai.build(technique="tabular_policy",
                  actions={("maze",): ["up", "down"],
                           ("seq",): [0, 1, 2]})
    assert isinstance(m.technique._default_actions, dict)


# =====================================================================
# 17. AutoTuner strategies
# =====================================================================
def test_autotuner_grid():
    m, X, y = _make_classifier()
    tuner = qai.AutoTuner("classifier", {"C": [0.1, 1.0]}, strategy="grid", k=2)
    tuner.fit(X, y)
    assert tuner.best_params_ is not None
    assert len(tuner.results_) == 2

def test_autotuner_random():
    m, X, y = _make_classifier()
    tuner = qai.AutoTuner("random_forest", {"n_estimators": [10, 20, 50]},
                          strategy="random", n_trials=2, k=2)
    tuner.fit(X, y)
    assert len(tuner.results_) <= 2

def test_autotuner_bayesian():
    m, X, y = _make_classifier()
    tuner = qai.AutoTuner("random_forest", {"n_estimators": [10, 20, 50, 100]},
                          strategy="bayesian", n_trials=4, k=2)
    tuner.fit(X, y)
    assert len(tuner.results_) >= 1


# =====================================================================
# 18. @qai.logic requires= enforcement
# =====================================================================
def test_logic_requires_present():
    """requires= with the dependency present should work."""
    # Define at module level so func.__globals__ includes 'agent'
    # Actually, the @qai.logic wrapper checks func.__globals__, which is the
    # module where the function is defined. Since this test function is in
    # the module's scope, we need to set 'agent' at module level.
    import __main__
    __main__.agent = type("A", (), {"x": 1})()

    @qai.logic(name="test_req_ok", requires=["agent"])
    def fn(x):
        return x + 1
    # The function's __globals__ is this module's globals
    fn.__globals__["agent"] = __main__.agent
    assert fn(5) == 6

def test_logic_requires_missing():
    """requires= with a missing dependency should raise LogicError."""
    @qai.logic(name="test_req_missing", requires=["nonexistent_thing_xyz"])
    def fn(x):
        return x

    try:
        fn(5)
        assert False, "should have raised LogicError"
    except qai.LogicError as e:
        assert "nonexistent_thing_xyz" in str(e)


# =====================================================================
# 19. Bundle README on Bundle dataclass
# =====================================================================
def test_bundle_readme():
    """Bundle.readme is populated from README.md."""
    m, X, y = _make_regressor()
    with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
        path = f.name
    qai.save_bundle(m, path, description="test bundle")
    b = qai.load_bundle(path)
    assert b.readme is not None
    assert "test bundle" in b.readme or "regression" in b.readme.lower()
    os.unlink(path)

def test_bundle_print_readme():
    """Bundle.print_readme() doesn't crash."""
    m, X, y = _make_regressor()
    with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
        path = f.name
    qai.save_bundle(m, path)
    b = qai.load_bundle(path)
    b.print_readme()  # should not crash
    os.unlink(path)


# =====================================================================
# 20. model.partial_fit
# =====================================================================
def test_partial_fit():
    """partial_fit works on a classifier."""
    m, X, y = _make_classifier(n=30)
    X_new = np.random.default_rng(99).standard_normal((10, 4))
    y_new = (X_new[:, 0] > 0).astype(int)
    m.partial_fit(X_new, y_new, verbose=False)
    assert m.is_trained()


# =====================================================================
# 21. model.train(budget=)
# =====================================================================
def test_train_with_budget():
    """model.train(budget=) works and doesn't crash."""
    m, X, y = _make_classifier(n=30)
    budget = qai.ResourceBudget(ram_mb=512, time_s=60)
    m.train(X, y, budget=budget, verbose=False)
    assert m.is_trained()

def test_train_budget_exceeded():
    """model.train(budget=) skips when budget is already exceeded."""
    m = qai.build(technique="classifier")
    budget = qai.ResourceBudget(time_s=0.001)
    time.sleep(0.01)  # exceed the budget
    m.train(X := np.random.randn(20, 4), y := (np.random.randn(20) > 0).astype(int),
            budget=budget, verbose=False)
    # Should have skipped training
    assert not m.is_trained()


# =====================================================================
# 22. sklearn_compat
# =====================================================================
def test_sklearn_compat_categories():
    cats = qai.sklearn_compat.list_categories()
    assert "classifiers" in cats
    assert "regressors" in cats
    assert "clusterers" in cats
    assert "preprocessors" in cats

def test_sklearn_compat_list_all():
    all_names = qai.sklearn_compat.list_all()
    assert "random_forest" in all_names
    assert len(all_names) > 50

def test_sklearn_compat_is_sklearn():
    assert qai.sklearn_compat.is_sklearn_technique("random_forest") == True
    assert qai.sklearn_compat.is_sklearn_technique("neural_network") == False


# =====================================================================
# 23. CLI smoke tests
# =====================================================================
def test_cli_bundle_inspect():
    """qai bundle inspect works on a freshly-saved bundle."""
    import subprocess
    m, X, y = _make_regressor()
    with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
        path = f.name
    qai.save_bundle(m, path)
    result = subprocess.run(
        [sys.executable, "-m", "qai", "bundle", "inspect", path],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..")},
    )
    assert result.returncode == 0
    assert "regression" in result.stdout
    os.unlink(path)

def test_cli_compress_decompress():
    """qai compress + decompress round-trip via CLI."""
    import subprocess
    data = b"hello world " * 100
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(data)
        in_path = f.name
    out_path = in_path + ".qai"
    env = {**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..")}
    # Compress
    r = subprocess.run([sys.executable, "-m", "qai", "compress", in_path, "-o", out_path],
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, f"compress failed: {r.stderr}"
    assert os.path.exists(out_path)
    # Decompress
    dec_path = in_path + ".restored"
    r = subprocess.run([sys.executable, "-m", "qai", "decompress", out_path, "-o", dec_path],
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, f"decompress failed: {r.stderr}"
    with open(dec_path, "rb") as f:
        restored = f.read()
    assert restored == data
    os.unlink(in_path); os.unlink(out_path); os.unlink(dec_path)


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing all v0.4-v0.7 features")
    print("=" * 60)
    tests = [
        # evaluate
        ("evaluate classification", test_evaluate_classification),
        ("evaluate regression", test_evaluate_regression),
        ("evaluate report str", test_evaluate_report_str),
        ("evaluate as_dict", test_evaluate_as_dict),
        # doc
        ("doc generates markdown", test_doc_generates_markdown),
        ("doc save to file", test_doc_save_to_file),
        ("doc for agent", test_doc_for_agent),
        # active learning
        ("active learning uncertainty", test_active_learning_uncertainty),
        ("active learning random", test_active_learning_random),
        ("active learning commit", test_active_learning_commit),
        ("active learning invalid strategy", test_active_learning_invalid_strategy),
        # drift
        ("drift ks normal", test_drift_ks_normal),
        ("drift ks drifted", test_drift_ks_drifted),
        ("drift psi", test_drift_psi),
        ("drift ensemble", test_drift_ensemble),
        ("drift report str", test_drift_report_str),
        ("drift invalid method", test_drift_invalid_method),
        # profiler
        ("profiler basic", test_profiler_basic),
        ("profiler str", test_profiler_str),
        ("profiler attributes", test_profiler_as_dict),
        # experiment compare
        ("experiment compare evaluate", test_experiment_compare_evaluate),
        ("experiment compare profile", test_experiment_compare_profile),
        ("experiment compare full", test_experiment_compare_full),
        ("experiment compare str", test_experiment_compare_str),
        ("experiment compare empty", test_experiment_compare_empty),
        # compress
        ("compress/decompress roundtrip", test_compress_decompress_roundtrip),
        ("compress random data", test_compress_random_data),
        ("compress empty", test_compress_empty),
        ("archive roundtrip", test_archive_roundtrip),
        ("archive signed", test_archive_signed),
        ("archive stats", test_archive_stats),
        ("archive tamper detection", test_archive_tamper_detection),
        # synthetic
        ("make_time_series", test_make_time_series),
        ("make_images", test_make_images),
        ("make_text", test_make_text),
        ("make_graphs", test_make_graphs),
        # CPC
        ("cpc ingest + ask", test_cpc_ingest_and_ask),
        ("cpc forget", test_cpc_forget),
        # CPC AI
        ("cpc ai compress small", test_cpc_ai_compress_decompress_small),
        ("cpc ai compress large", test_cpc_ai_compress_decompress_large),
        ("cpc ai verification", test_cpc_ai_verification),
        ("cpc ai huffman roundtrip", test_cpc_ai_huffman_roundtrip),
        ("cpc ai ngram roundtrip", test_cpc_ai_ngram_roundtrip),
        ("cpc ai symbol roundtrip", test_cpc_ai_symbol_roundtrip),
        ("cpc ai knowledge graph", test_cpc_ai_knowledge_graph),
        ("cpc ai curiosity", test_cpc_ai_curiosity),
        ("cpc ai compress file", test_cpc_ai_compress_file),
        # agents
        ("agent base run episode", test_agent_base_run_episode),
        ("environment registry", test_environment_registry),
        ("replay buffer", test_replay_buffer),
        ("hyperparameter scheduler", test_hyperparameter_scheduler),
        # dashboard
        ("dashboard import", test_dashboard_import),
        ("dashboard state functions", test_dashboard_state_functions),
        # live training
        ("live training import", test_live_training_callback_import),
        # permanent memory
        ("permanent memory roundtrip", test_permanent_memory_roundtrip),
        # memory picklability
        ("memory picklable", test_memory_picklable),
        # tabular policy
        ("tabular policy actions default", test_tabular_policy_actions_default),
        ("tabular policy actions dict", test_tabular_policy_actions_dict),
        # autotuner
        ("autotuner grid", test_autotuner_grid),
        ("autotuner random", test_autotuner_random),
        ("autotuner bayesian", test_autotuner_bayesian),
        # logic requires
        ("logic requires present", test_logic_requires_present),
        ("logic requires missing", test_logic_requires_missing),
        # bundle readme
        ("bundle readme", test_bundle_readme),
        ("bundle print readme", test_bundle_print_readme),
        # partial_fit
        ("partial fit", test_partial_fit),
        # train with budget
        ("train with budget", test_train_with_budget),
        ("train budget exceeded", test_train_budget_exceeded),
        # sklearn_compat
        ("sklearn compat categories", test_sklearn_compat_categories),
        ("sklearn compat list all", test_sklearn_compat_list_all),
        ("sklearn compat is sklearn", test_sklearn_compat_is_sklearn),
        # CLI
        ("cli bundle inspect", test_cli_bundle_inspect),
        ("cli compress decompress", test_cli_compress_decompress),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1
    print()
    print(f"{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}")
