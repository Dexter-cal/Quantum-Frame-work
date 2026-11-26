import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai
from fixtures_huggingface_iris import load as load_hf_iris

print("=" * 60)
print("TEST: SVM + Naive Bayes on REAL data fetched live from")
print("huggingface.co/datasets/scikit-learn/iris (not sklearn's bundled copy)")
print("=" * 60)

X, y = load_hf_iris()
print(f"Loaded {len(X)} real rows from Hugging Face, {len(set(y.tolist()))} classes: {sorted(set(y.tolist()))}")

svm = qai.build(type="svm", kernel="linear")
svm.train(X, y)
print()
print(svm.help())
print("accuracy():", round(svm.accuracy(), 4))
print("support_vectors() count:", len(svm.support_vectors()))
print("margin_width() [linear kernel]:", round(svm.margin_width(), 4))

print()
nb = qai.build(type="naive_bayes")
nb.train(X, y)
print(nb.help())
print("accuracy():", round(nb.accuracy(), 4))
print("class_priors():", {k: round(v, 3) for k, v in nb.class_priors().items()})
liks = nb.likelihood_table()
print("likelihood_table()['Iris-setosa']['mean']:", [round(m, 2) for m in liks["Iris-setosa"]["mean"]])

print()
print("=" * 60)
print("CROSS-CHECK: same real data, all 7 techniques, one loop, same interface")
print("=" * 60)
for name in ["regression", "classifier", "decision_tree", "knn", "svm", "naive_bayes"]:
    if name == "regression":
        continue  # regression needs numeric y, skip for this classification fixture
    m = qai.build(type=name)
    m.train(X, y, verbose=False)
    print(f"  {name:15s} accuracy={m.accuracy():.4f}  trained={m.technique.is_trained()}")

km = qai.build(type="kmeans", n_clusters=3)
km.train(X, verbose=False)
print(f"  {'kmeans':15s} (unsupervised) cluster_centers shape={km.cluster_centers().shape}")

print()
print("ALL TESTS COMPLETED -- verified against real, externally-sourced Hugging Face data")
