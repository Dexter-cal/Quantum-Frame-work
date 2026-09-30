import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai
from fixtures_huggingface_iris import load as load_hf_iris

np.random.seed(42)

print("=" * 60)
print("SYSTEMATIC AUDIT: every technique, checked for the SAME bug class")
print("found in Classifier -- does forward() on ONE sample return a genuine")
print("scalar, or does it silently return an array wearing a scalar's clothes?")
print("(This is exactly the check that would have caught the Classifier bug")
print(" immediately, if it had existed from the start.)")
print("=" * 60)

X, y = load_hf_iris()
results = {}

techniques_to_check = [
    ("regression", {}),
    ("classifier", {}),
    ("decision_tree", {"max_depth": 3}),
    ("knn", {"k": 3}),
    ("svm", {"kernel": "linear"}),
    ("naive_bayes", {}),
    ("perceptron", {}),
    ("random_forest", {"n_estimators": 20}),
]

for name, params in techniques_to_check:
    model = qai.build(type=name, **params)
    if name == "regression":
        # regression needs numeric y -- use a numeric-friendly slice
        model.train(X, np.arange(len(X), dtype=float), verbose=False)
    else:
        model.train(X, y, verbose=False)

    single_result = model.predict(X[0])
    is_array = isinstance(single_result, np.ndarray)
    results[name] = {"value": single_result, "is_array": is_array, "type": type(single_result).__name__}
    status = "FAIL (returns array!)" if is_array else "OK (genuine scalar)"
    print(f"  {name:15s} single-prediction type: {type(single_result).__name__:20s} -- {status}")

print()
print("=" * 60)
print("PCA CHECKED SEPARATELY -- its forward() genuinely returns a VECTOR")
print("by design (dimensionality reduction), not a label -- correctly")
print("excluded from the 'should be scalar' expectation")
print("=" * 60)
pca = qai.build(type="pca", n_components=2)
pca.train(X, verbose=False)
pca_result = pca.predict(X[0])
print(f"  pca             single-prediction type: {type(pca_result).__name__:20s} shape={pca_result.shape} "
      f"-- OK, a vector is the CORRECT output here, not a bug")

print()
print("=" * 60)
print("KMEANS CHECKED SEPARATELY -- returns a cluster index, should be a plain int")
print("=" * 60)
km = qai.build(type="kmeans", n_clusters=3)
km.train(X, verbose=False)
km_result = km.predict(X[0])
print(f"  kmeans          single-prediction type: {type(km_result).__name__:20s} -- "
      f"{'FAIL (array!)' if isinstance(km_result, np.ndarray) else 'OK'}")
assert not isinstance(km_result, np.ndarray), "BUG: KMeans also has the array-instead-of-scalar issue"

print()
print("=" * 60)
print("FINAL VERDICT")
print("=" * 60)
failures = [name for name, r in results.items() if r["is_array"]]
if failures:
    print(f"FOUND {len(failures)} technique(s) with the SAME bug class: {failures}")
    print("These need the same single/batch unwrap fix Classifier just received.")
else:
    print("NONE of the other 8 techniques have this bug -- Classifier was a genuine,")
    print("isolated oversight (likely because it was one of the first two techniques")
    print("written, before the single/batch unwrap pattern became a consistent habit")
    print("across the rest of the build), not a systemic problem across the codebase.")

assert not failures, f"Found the same bug class in: {failures} -- must fix before continuing"
print()
print("AUDIT COMPLETE -- confirmed the fix was isolated to Classifier,")
print("not a hidden systemic issue across all 10 techniques")
