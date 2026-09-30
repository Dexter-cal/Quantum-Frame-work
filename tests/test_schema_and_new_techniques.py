import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai
from qai.core.schema import SchemaError

print("=" * 60)
print("TEST: Schema validation actually rejects mismatched data")
print("=" * 60)
model = qai.build(type="regression", input_schema={"a": float, "b": float, "c": float})
X_good = np.random.rand(10, 3)
y_good = np.random.rand(10)
model.train(X_good, y_good)
print("Correctly ACCEPTED 3-column data matching a 3-field schema")

model2 = qai.build(type="regression", input_schema={"a": float, "b": float, "c": float})
X_bad = np.random.rand(10, 5)  # wrong number of columns
try:
    model2.train(X_bad, y_good)
    print("FAIL: should have raised SchemaError")
except SchemaError as e:
    print("Correctly REJECTED mismatched data:")
    print(str(e))

print()
print("=" * 60)
print("TEST: Decision Tree, on real Iris data")
print("=" * 60)
from sklearn.datasets import load_iris
iris = load_iris()
tree = qai.build(type="decision_tree", max_depth=3)
tree.train(iris.data, iris.target)
print(tree.help())
print("accuracy():", round(tree.accuracy(), 4))
print("tree_depth():", tree.tree_depth())
print("feature_importance():", {k: round(v, 3) for k, v in tree.feature_importance().items()})

print()
print("=" * 60)
print("TEST: K-Nearest Neighbors, on real Iris data")
print("=" * 60)
knn = qai.build(type="knn", k=5)
knn.train(iris.data, iris.target)
print(knn.help())
print("accuracy():", round(knn.accuracy(), 4))
print("nearest_neighbors() for sample 0:", knn.nearest_neighbors(iris.data[0], n=3))

print()
print("=" * 60)
print("TEST: K-Means, UNSUPERVISED -- no y at all, real interface stress test")
print("=" * 60)
km = qai.build(type="kmeans", n_clusters=3)
km.train(iris.data)  # deliberately no y -- tests fit(X, y=None) actually works end to end
print(km.help())
cluster_for_sample_0 = km.predict(iris.data[0])
print("Cluster assigned to sample 0:", cluster_for_sample_0)
print("cluster_centers() shape:", km.cluster_centers().shape)

print()
print("ALL TESTS COMPLETED")
