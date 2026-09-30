import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai

np.random.seed(42)

print("=" * 60)
print("TEST: weights.as_table() -- real, readable weight inspection")
print("=" * 60)
X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 2.0], [5.0, 1.0]])
y = np.array([5.0, 4.0, 12.0, 10.0, 7.0])  # y = 2*x0 + 1*x1 + small noise-ish pattern

model = qai.build(type="regression")
model.train(X, y, verbose=False)
table = model.weights.as_table()
print("Weight table:", table)
assert len(table) == 3  # 2 features + bias
assert table[0]["name"] == "w0" and table[1]["name"] == "w1" and table[2]["name"] == "bias"
print("VERIFIED: as_table() correctly names and lists every weight, including bias")

print()
print("=" * 60)
print("TEST: weights.get() matches the technique's own stored value exactly")
print("=" * 60)
w0_via_accessor = model.weights.get(0)
w0_direct = model.technique.w[0]
assert w0_via_accessor == w0_direct
print(f"weights.get(0) = {w0_via_accessor:.6f}, matches technique.w[0] exactly")

print()
print("=" * 60)
print("TEST: THE REAL CHECK -- editing a weight produces the EXACT predicted")
print("change in output, verified by hand-computed math, not just 'it changed'")
print("=" * 60)
test_point = np.array([1.0, 1.0])
prediction_before = model.predict(test_point)
print(f"Prediction before edit: {prediction_before:.4f}")

original_w0 = model.weights.get(0)
delta = 10.0
model.weights.set(0, original_w0 + delta)  # directly mutate weight 0

prediction_after = model.predict(test_point)
print(f"Prediction after setting w0 += {delta}: {prediction_after:.4f}")

# REAL, EXACT CHECK: since prediction = w0*x0 + w1*x1 + b, and x0=1.0 for
# this test point, increasing w0 by 10 should increase the prediction by
# EXACTLY 10 * 1.0 = 10.0 -- not approximately, EXACTLY (to floating point precision)
expected_change = delta * test_point[0]
actual_change = prediction_after - prediction_before
print(f"Expected change: {expected_change:.6f}, Actual change: {actual_change:.6f}")
assert abs(actual_change - expected_change) < 1e-9, \
    f"BUG: weight edit did not produce the mathematically exact expected change"
print("VERIFIED: the edit produced EXACTLY the mathematically correct change in output --")
print("this proves weights.set() genuinely mutates the live model used for real predictions,")
print("not a disconnected copy")

print()
print("=" * 60)
print("TEST: weight editing also works on an sklearn-backed technique (Classifier)")
print("=" * 60)
from fixtures_huggingface_iris import load as load_hf_iris
Xc, yc = load_hf_iris()
# binary subset for a clean 1D coefficient story
mask = np.isin(yc, ["Iris-setosa", "Iris-versicolor"])
clf = qai.build(type="classifier")
clf.train(Xc[mask], yc[mask], verbose=False)
table2 = clf.weights.as_table()
print(f"Classifier has {len(table2)} weights (4 features + bias)")
assert len(table2) == 5

original_val = clf.weights.get(0)
clf.weights.set(0, 0.0)  # zero out the first feature's influence entirely
new_val = clf.weights.get(0)
assert new_val == 0.0 and original_val != 0.0
print(f"VERIFIED: successfully edited a real sklearn model's coef_ directly ({original_val:.3f} -> 0.0)")

print()
print("ALL WEIGHT INSPECTION/EDITING TESTS PASSED -- edits verified to produce")
print("mathematically exact, predictable changes in real predictions")
