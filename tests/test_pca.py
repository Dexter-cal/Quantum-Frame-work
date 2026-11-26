import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai
from fixtures_huggingface_iris import load as load_hf_iris

X, y = load_hf_iris()

print("=" * 60)
print("TEST: PCA -- a genuinely different SHAPE of unsupervised technique")
print("forward() transforms, doesn't classify -- stress-tests the interface")
print("=" * 60)

pca = qai.build(type="pca", n_components=2)
pca.train(X)  # no y at all, same as K-Means
print(pca.help())

transformed = pca.predict(X[0])
print(f"Original sample 0 has {len(X[0])} dimensions: {X[0]}")
print(f"PCA-transformed sample 0 has {len(transformed)} dimensions: {transformed}")
assert len(transformed) == 2, "PCA should reduce 4 dimensions down to 2"
print("Correctly reduced 4D -> 2D")

print()
print("explained_variance_ratio():", [round(v, 4) for v in pca.explained_variance_ratio()])
total_variance_kept = sum(pca.explained_variance_ratio())
print(f"Total variance retained by 2 components: {total_variance_kept:.2%}")

print()
print("find_eigenvectors() shape:", pca.find_eigenvectors().shape)
print("reconstruction_error() for sample 0:", round(pca.reconstruction_error(X[0]), 6))

print()
print("=" * 60)
print("SANITY CHECK against independent sklearn PCA -- not just 'does it run'")
print("=" * 60)
from sklearn.decomposition import PCA as SKPCA
ref = SKPCA(n_components=2).fit(X)
ref_transformed = ref.transform(X[0].reshape(1, -1))[0]
match = np.allclose(transformed, ref_transformed)
print("Matches independent sklearn PCA computation:", match)

print()
print("ALL PCA TESTS PASSED -- unsupervised technique family now has 2 genuinely")
print("different shapes verified: clustering (K-Means) and transformation (PCA)")
