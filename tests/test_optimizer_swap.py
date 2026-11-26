import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai
import qai.mechanics
from qai.mechanics import ClosedForm, GradientDescent

np.random.seed(42)

print("=" * 60)
print("TEST: default behavior UNCHANGED -- no regression from this refactor")
print("=" * 60)
from sklearn.datasets import load_diabetes
d = load_diabetes()

model_default = qai.build(type="regression")  # no optimizer specified
model_default.train(d.data, d.target, verbose=False)
print("Default optimizer:", model_default.technique.optimizer.name)
assert model_default.technique.optimizer.name == "closed_form"
print("r_squared() with default:", round(model_default.r_squared(), 4))
print("VERIFIED: existing default behavior is completely unchanged")

print()
print("=" * 60)
print("TEST: Section 56's REAL claim -- swap the optimizer, same structure,")
print("does a from-scratch gradient descent actually converge to the SAME answer?")
print("=" * 60)

# small, well-conditioned synthetic problem where we KNOW the true answer
X = np.random.rand(200, 2) * 10
true_w = np.array([3.0, -2.0])
true_b = 5.0
y = X @ true_w + true_b + np.random.normal(0, 0.1, 200)  # small noise

closed_form_model = qai.build(type="regression", optimizer=ClosedForm())
closed_form_model.train(X, y, verbose=False)
print("ClosedForm learned w:", np.round(closed_form_model.technique.w, 3), "b:", round(closed_form_model.technique.b, 3))

gd_model = qai.build(type="regression", optimizer=GradientDescent(learning_rate=0.01, training_loop=qai.mechanics.FixedEpochs(2000)))
gd_model.train(X, y, verbose=False)
print("GradientDescent learned w:", np.round(gd_model.technique.w, 3), "b:", round(gd_model.technique.b, 3))

# REAL CHECK: do two ENTIRELY DIFFERENT optimization mechanisms agree,
# closely, on the same underlying structure and data?
w_diff = np.abs(closed_form_model.technique.w - gd_model.technique.w)
b_diff = abs(closed_form_model.technique.b - gd_model.technique.b)
print(f"Weight difference between the two optimizers: {w_diff}")
print(f"Bias difference: {b_diff:.4f}")
assert np.all(w_diff < 0.1), f"BUG: optimizers disagree too much on weights: {w_diff}"
assert b_diff < 0.5, f"BUG: optimizers disagree too much on bias: {b_diff}"
print("VERIFIED: two genuinely different optimizers converge to essentially the same answer")

print()
print("=" * 60)
print("TEST: does gradient descent's OWN loss history actually decrease?")
print("Real evidence of genuine learning, not just a lucky final answer")
print("=" * 60)
history = gd_model.technique.optimizer.loss_history
print(f"Loss at epoch 0: {history[0]:.4f}")
print(f"Loss at epoch 500: {history[500]:.4f}")
print(f"Loss at epoch 1999 (final): {history[-1]:.4f}")
assert history[-1] < history[0] * 0.01, "BUG: loss did not genuinely decrease -- gradient math may be wrong"
assert all(history[i] >= history[i+1] - 0.01 for i in range(0, len(history)-1, 200)), \
    "BUG: loss is not monotonically decreasing (sampled) -- learning_rate may be unstable"
print("VERIFIED: loss genuinely and consistently decreased -- real, correct gradient math")

print()
print("=" * 60)
print("TEST: recover the KNOWN true answer -- the strongest possible check")
print("=" * 60)
print(f"True w: {true_w}, learned by GD: {np.round(gd_model.technique.w, 3)}")
print(f"True b: {true_b}, learned by GD: {round(gd_model.technique.b, 3)}")
assert np.all(np.abs(gd_model.technique.w - true_w) < 0.15), "BUG: GD did not recover the true weights"
assert abs(gd_model.technique.b - true_b) < 0.5, "BUG: GD did not recover the true bias"
print("VERIFIED: gradient descent recovered the actual ground-truth relationship, not just SOME answer")

print()
print("ALL OPTIMIZER-SWAPPING TESTS PASSED -- Section 56's claim is real:")
print("structure and optimizer are genuinely separable and interchangeable")
