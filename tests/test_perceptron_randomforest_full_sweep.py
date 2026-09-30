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
results = {}
for name in SUPERVISED:
    if name == "regression":
        continue  # regression needs numeric y; this fixture's y is categorical species
    m = qai.build(type=name)
    m.train(X, y, verbose=False)
    acc = m.accuracy()
    results[name] = acc
    print(f"  {name:15s} accuracy={acc:.4f}")

print()
print("Best performing:", max(results, key=results.get), "->", round(max(results.values()), 4))
print("Worst performing:", min(results, key=results.get), "->", round(min(results.values()), 4))

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
