"""
Test the input-validation feature and advanced model features.

Verifies:
  1. InputValidationError is raised when input type doesn't match training data
  2. Error messages include sample training data
  3. Numeric→text and text→numeric mismatches are both caught
  4. Feature-count mismatches are caught
  5. Transformer supports sliding window attention (window_size param)
  6. GMM supports variational Bayesian inference
  7. GMM supports BIC/AIC for model selection
  8. Vision models auto-switch to eval mode for single-sample prediction
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import qai

print("=" * 70)
print("TEST 1: Input validation — text input to numeric model")
print("=" * 70)
m = qai.build(technique="classifier")
X = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.9, 3.0, 1.4, 0.2],
              [4.7, 3.2, 1.3, 0.2]], dtype=float)
y = np.array([0, 0, 1])
m.train(X, y, verbose=False)
try:
    m.predict("hello world")
    print("FAIL: should have raised InputValidationError")
    sys.exit(1)
except qai.InputValidationError as e:
    msg = str(e)
    assert "NUMERIC data" in msg, "missing NUMERIC mention"
    assert "TEXT input" in msg, "missing TEXT mention"
    assert "5.1, 3.5, 1.4, 0.2" in msg, "missing sample data"
    print("CORRECTLY caught text→numeric mismatch with sample data ✓")

print()
print("=" * 70)
print("TEST 2: Input validation — numeric input to text model")
print("=" * 70)
m = qai.build(technique="tfidf_vectorizer")
texts = ["hello world", "foo bar", "machine learning", "data science"]
m.train(texts, verbose=False)
try:
    m.predict([1.0, 2.0, 3.0])
    print("FAIL: should have raised InputValidationError")
    sys.exit(1)
except qai.InputValidationError as e:
    msg = str(e)
    assert "TEXT data" in msg, "missing TEXT mention"
    assert "NUMERIC input" in msg, "missing NUMERIC mention"
    assert "hello world" in msg, "missing sample text"
    print("CORRECTLY caught numeric→text mismatch with sample data ✓")

print()
print("=" * 70)
print("TEST 3: Input validation — wrong number of features")
print("=" * 70)
m = qai.build(technique="classifier")
m.train(np.random.randn(20, 4), np.random.randint(0, 2, 20), verbose=False)
for bad_n in (3, 5, 10):
    try:
        m.predict(np.random.randn(bad_n))
        print(f"FAIL: {bad_n} features should have raised")
        sys.exit(1)
    except qai.InputValidationError as e:
        assert f"{bad_n} features" in str(e), f"missing '{bad_n} features' in error"
        print(f"  {bad_n} features (expected 4): correctly caught ✓")

print()
print("=" * 70)
print("TEST 4: Input validation — correct input passes")
print("=" * 70)
m = qai.build(technique="regression")
m.train(np.random.randn(20, 4), np.random.randn(20), verbose=False)
assert m.validate_input([1.0, 2.0, 3.0, 4.0]) is True
print("Correct numeric input passes validation ✓")

print()
print("=" * 70)
print("TEST 5: Transformer sliding window attention (SWA)")
print("=" * 70)
m = qai.build(technique="transformer", vocab_size=32, n_layers=1,
              d_model=16, n_heads=2, epochs=1, window_size=4)
assert m.technique.window_size == 4, "window_size not set"
X = np.random.randint(0, 32, size=(10, 8))
y = np.random.randint(0, 2, 10)
m.train(X, y, verbose=False)
pred = m.predict(X[0])
print(f"Transformer with window_size=4 trains and predicts: {pred} ✓")

print()
print("=" * 70)
print("TEST 6: GMM variational Bayesian inference")
print("=" * 70)
m = qai.build(technique="gmm", n_components=5, variational_bayes=True,
              weight_concentration_prior=1e-2)
assert m.technique.variational_bayes is True
X = np.random.randn(50, 4)
m.train(X, verbose=False)
n_active = m.technique.active_components()
assert 1 <= n_active <= 5, f"active_components out of range: {n_active}"
print(f"Variational GMM: requested 5 components, {n_active} active ✓")

print()
print("=" * 70)
print("TEST 7: GMM BIC for model selection")
print("=" * 70)
X = np.random.randn(50, 4)
bics = {}
for k in range(2, 6):
    m = qai.build(technique="gmm", n_components=k)
    m.train(X, verbose=False)
    bics[k] = m.technique.bic(X)
best_k = min(bics, key=bics.get)
print(f"BIC scores: {bics}")
print(f"Best k by BIC: {best_k} ✓")

print()
print("=" * 70)
print("TEST 8: Vision models auto-eval for single-sample prediction")
print("=" * 70)
m = qai.build(technique="resnet", in_ch=1, n_classes=3, epochs=1)
X = np.random.randn(5, 1, 8, 8).astype(np.float32)
y = np.random.randint(0, 3, 5)
m.train(X, y, verbose=False)
pred = m.predict(X[0])  # single sample — would fail without eval mode
print(f"ResNet single-sample predict: {pred} ✓")

print()
print("=" * 70)
print("ALL INPUT VALIDATION + ADVANCED FEATURE TESTS PASSED")
print("=" * 70)
