import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai
from fixtures_huggingface_iris import load as load_hf_iris

X, y = load_hf_iris()

print("=" * 60)
print("TEST: .explain() -- linear technique (Regression)")
print("Using diabetes data since regression needs numeric y")
print("=" * 60)
from sklearn.datasets import load_diabetes
d = load_diabetes()
reg = qai.build(type="regression")
reg.train(d.data, d.target, verbose=False)
result = reg.explain(d.data[0])
print("prediction:", round(result["prediction"], 3))
print("method:", result["method"])
print("most_influential_feature:", result["most_influential_feature"])
print("bias_contribution:", result["bias_contribution"])
print("reasoning:", result["reasoning"])

# REAL CHECK: does most_influential_feature actually have the largest |contribution|
# AMONG FEATURES (bias correctly excluded from this comparison now)?
contribs = result["feature_contributions"]
claimed_max = result["most_influential_feature"]
actual_max = max(contribs, key=lambda k: abs(contribs[k]))
assert claimed_max == actual_max, f"BUG: claimed {claimed_max} but {actual_max} is actually largest"
print("VERIFIED: most_influential_feature claim is mathematically correct, bias correctly excluded")

print()
print("=" * 60)
print("TEST: .explain() -- tree-based technique (Random Forest)")
print("=" * 60)
rf = qai.build(type="random_forest", n_estimators=50)
rf.train(X, y, verbose=False)
result = rf.explain(X[0])
print("prediction:", result["prediction"])
print("method:", result["method"])
print("top feature:", result["feature_ranking"][0])
print("reasoning:", result["reasoning"])

# REAL CHECK: is the ranking actually sorted descending?
importances = [f["importance"] for f in result["feature_ranking"]]
assert importances == sorted(importances, reverse=True), "BUG: feature_ranking not actually sorted"
print("VERIFIED: feature_ranking is genuinely sorted by importance")

print()
print("=" * 60)
print("TEST: .explain() -- probabilistic technique (Naive Bayes)")
print("=" * 60)
nb = qai.build(type="naive_bayes")
nb.train(X, y, verbose=False)
result = nb.explain(X[0])
print("prediction:", result["prediction"])
print("class_probabilities:", result["class_probabilities"])
print("reasoning:", result["reasoning"])

# REAL CHECK: do probabilities actually sum to ~1.0?
total = sum(result["class_probabilities"].values())
assert abs(total - 1.0) < 0.01, f"BUG: probabilities sum to {total}, not 1.0"
print(f"VERIFIED: probabilities sum to {total:.4f} (~1.0, correct)")

print()
print("=" * 60)
print("TEST: .explain() on an unsupported technique -- should degrade honestly")
print("Using K-Means: genuinely has no .w, no feature_importances_, no predict_proba")
print("(earlier test wrongly assumed KNN had no explanation -- it actually DOES,")
print(" via neighbor-vote probabilities, which is correct, not a bug)")
print("=" * 60)
km = qai.build(type="kmeans", n_clusters=3)
km.train(X, verbose=False)
result = km.explain(X[0])
print("method:", result["method"])
print("reasoning:", result["reasoning"])
assert "No explanation method" in result["reasoning"], "should honestly admit no explanation exists"
print("VERIFIED: honestly reports no explanation available, rather than faking one")

print()
print("ALL .explain() TESTS PASSED -- including mathematical correctness checks,")
print("not just 'did it return something'")
