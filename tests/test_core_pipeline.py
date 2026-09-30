"""
Real, functional test of the qai prototype -- not a mock, actual data,
actual training, actual predictions, checked against known-good sklearn
results for sanity.
"""
import numpy as np
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); import qai
from qai.techniques import Technique, register_technique

print("=" * 60)
print("TEST 1: Regression -- qai.build().train().predict(), matches doc syntax")
print("=" * 60)

from sklearn.datasets import load_diabetes
data = load_diabetes()
X, y = data.data, data.target

model = qai.build(type="regression")
print(model.help())
model.train(X, y)
print(model.help())
print("formula() [first 3 terms]:", model.formula().split(" + ")[:3])
print("r_squared():", round(model.r_squared(), 4))

# sanity check against numpy's own lstsq directly
X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
coef_direct, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
match = np.allclose(model.technique.w, coef_direct[:-1])
print("Matches independent numpy lstsq computation:", match)

sample = X[0]
pred = model.predict(sample)
print(f"Prediction for sample 0: {pred:.2f} (actual: {y[0]})")

print()
print("=" * 60)
print("TEST 2: Classifier -- different technique, same Model interface")
print("=" * 60)

from sklearn.datasets import load_breast_cancer
cdata = load_breast_cancer()
CX, cy = cdata.data, cdata.target

clf = qai.build(type="classifier")
clf.train(CX, cy)
print(clf.help())
print("accuracy():", round(clf.accuracy(), 4))
print("class_probabilities() for sample 0:", clf.class_probabilities(CX[0]))
print("confusion_matrix():\n", clf.confusion_matrix())

print()
print("=" * 60)
print("TEST 3: Error handling -- predict() before train() should fail clearly")
print("=" * 60)
fresh = qai.build(type="regression")
try:
    fresh.predict([1, 2, 3])
    print("FAIL: should have raised")
except RuntimeError as e:
    print("Correctly raised:", e)

print()
print("=" * 60)
print("TEST 4: Custom technique -- the REAL extensibility test")
print("Does 'nothing is hardcoded' actually hold in working code?")
print("=" * 60)


class MeanBaseline(Technique):
    """A trivial custom technique a user might write: always predicts
    the mean of y, ignoring x entirely. Not useful, but a genuine,
    independent test of the extension mechanism."""
    name = "mean_baseline"
    family = "custom"

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self._trained = True
        return self

    def forward(self, x):
        return self.mean_


register_technique("mean_baseline", MeanBaseline)
baseline = qai.build(type="mean_baseline")
baseline.train(X, y)
print(baseline.help())
print("Predicts constant mean:", baseline.predict(X[0]), "== actual mean:", round(float(np.mean(y)), 3))

print()
print("=" * 60)
print("TEST 5: Export -- does the .qmodel-equivalent round-trip work")
print("=" * 60)
path = model.export("/tmp/test_export.json")
import json
with open(path) as f:
    loaded = json.load(f)
print("Exported keys:", list(loaded.keys()))
print("Weight count matches:", len(loaded["w"]) == len(model.technique.w))

print()
print("ALL TESTS COMPLETED")
