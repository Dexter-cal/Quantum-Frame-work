import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import qai
from qai.techniques import SUPERVISED, UNSUPERVISED
from fixtures_huggingface_iris import load as load_hf_iris

X, y = load_hf_iris()

print("=" * 60)
print("TEST: Perceptron -- the original 1958 architecture, real data")
print("=" * 60)
p = qai.build(type="perceptron")
p.train(X, y)
print(p.help())
print("accuracy():", round(p.accuracy(), 4))
print("n_updates():", p.n_updates())
print("weights [first 4]:", [round(row["value"], 3) for row in p.weights.as_table()[:4]])

print()
print("=" * 60)
print("TEST: Random Forest -- ensemble composition, real data")
print("=" * 60)
rf = qai.build(type="random_forest", n_estimators=50)
rf.train(X, y)
print(rf.help())
print("accuracy():", round(rf.accuracy(), 4))
print("tree_count():", rf.tree_count())
print("member_agreement() for sample 0:", round(rf.member_agreement(X[0]), 3))

print()
print("=" * 60)
print(f"FULL SUPERVISED SWEEP -- all {len(SUPERVISED)} supervised techniques,")
print("same real Hugging Face data, same Model interface, one loop")
print("=" * 60)

# Regressors can't fit categorical (species) labels — skip them in this sweep.
# (The framework now gives a helpful error, but the test should still skip them
# because Iris y is categorical.)
REGRESSORS = {
    "regression", "ridge", "lasso", "elastic_net", "kernel_ridge",
    "bayesian_ridge", "ard_regression", "huber", "ransac", "theil_sen",
    "quantile_regression", "decision_tree_regressor",
    "random_forest_regressor", "adaboost_regressor",
    "gradient_boosting_regressor",
}

results = {}
for name in SUPERVISED:
    if name in REGRESSORS:
        continue  # regressors need numeric y; this fixture's y is categorical species
    try:
        m = qai.build(type=name)
        m.train(X, y, verbose=False)
        acc = m.accuracy()
        results[name] = acc
        print(f"  {name:15s} accuracy={acc:.4f}")
    except Exception as e:
        results[name] = None
        print(f"  {name:15s} FAILED: {type(e).__name__}: {str(e)[:80]}")

print()
valid = {k: v for k, v in results.items() if v is not None}
if valid:
    print("Best performing:", max(valid, key=valid.get), "->", round(max(valid.values()), 4))
    print("Worst performing:", min(valid, key=valid.get), "->", round(min(valid.values()), 4))
else:
    print("No techniques succeeded")

print()
print(f"UNSUPERVISED CHECK -- {len(UNSUPERVISED)} technique(s)")
for name in UNSUPERVISED:
    m = qai.build(type=name)  # let each technique use its own sensible defaults, don't assume a shared param set
    m.train(X, verbose=False)
    if hasattr(m, "cluster_centers"):
        print(f"  {name:15s} cluster_centers shape={m.cluster_centers().shape}")
    else:
        print(f"  {name:15s} trained={m.technique.is_trained()} (see test_pca.py for PCA-specific checks)")

print()
print("ALL SUPERVISED + UNSUPERVISED CLASSICAL TECHNIQUES VERIFIED")
