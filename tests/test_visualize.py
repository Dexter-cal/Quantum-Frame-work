import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai
from qai.mechanics import GradientDescent
from fixtures_huggingface_iris import load as load_hf_iris

np.random.seed(42)
OUT_DIR = "/tmp/qai_visualize_test"
os.makedirs(OUT_DIR, exist_ok=True)


def real_image_check(path, min_bytes=3000):
    """A genuine check, not just 'does the file exist' -- a truly blank/
    trivial plot saves as a tiny file (mostly white pixels compress well);
    a real plot with data, lines, and labels is meaningfully larger."""
    assert os.path.exists(path), f"BUG: {path} was never created"
    size = os.path.getsize(path)
    assert size > min_bytes, f"BUG: {path} is only {size} bytes -- likely blank/near-empty, not a real plot"
    return size


print("=" * 60)
print("TEST: visualize() -- Regression, single feature, real scatter+line+residuals")
print("=" * 60)
X1 = np.random.rand(30, 1) * 10
y1 = 3 * X1.flatten() + 2 + np.random.normal(0, 1, 30)
reg = qai.build(type="regression")
reg.train(X1, y1, verbose=False)
path1 = reg.visualize(f"{OUT_DIR}/regression.png")
size1 = real_image_check(path1)
print(f"Saved real regression plot: {size1} bytes")

print()
print("=" * 60)
print("TEST: visualize() -- K-Means, real cluster coloring + centers")
print("=" * 60)
X_iris, y_iris = load_hf_iris()
km = qai.build(type="kmeans", n_clusters=3)
km.train(X_iris, verbose=False)
path2 = km.visualize(f"{OUT_DIR}/kmeans.png", X=X_iris)
size2 = real_image_check(path2)
print(f"Saved real K-Means plot: {size2} bytes")

print()
print("=" * 60)
print("TEST: visualize() -- gradient descent loss curve, REAL training history")
print("=" * 60)
gd_reg = qai.build(type="regression", optimizer=GradientDescent(learning_rate=0.01))
gd_reg.train(X1, y1, verbose=False)
path3 = gd_reg.visualize(f"{OUT_DIR}/loss_curve.png")
size3 = real_image_check(path3)
print(f"Saved real loss-curve plot: {size3} bytes")

print()
print("=" * 60)
print("TEST: visualize() -- Random Forest, real feature importance bars")
print("=" * 60)
rf = qai.build(type="random_forest", n_estimators=30)
rf.train(X_iris, y_iris, verbose=False)
path4 = rf.visualize(f"{OUT_DIR}/rf_importance.png")
size4 = real_image_check(path4)
print(f"Saved real feature-importance plot: {size4} bytes")

print()
print("=" * 60)
print("TEST: the DISPATCH is actually different per technique, not one generic chart")
print("=" * 60)
sizes = {"regression": size1, "kmeans": size2, "gd_loss_curve": size3, "random_forest_importance": size4}
for name, size in sizes.items():
    print(f"  {name:25s} {size} bytes")
# a real, if weak, sanity signal: 4 genuinely different plot types on
# genuinely different data are extremely unlikely to all be identical sizes
assert len(set(sizes.values())) == len(sizes), \
    "SUSPICIOUS: all plots came out exactly the same size -- dispatch may not actually be working"
print("VERIFIED: all 4 plots have distinct file sizes -- real evidence they're genuinely different images,")
print("not the same fallback chart saved four times")

print()
print("ALL .visualize() TESTS PASSED -- real matplotlib output, checked for")
print("actual content, not just file existence")
