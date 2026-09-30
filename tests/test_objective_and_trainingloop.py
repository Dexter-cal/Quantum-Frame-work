import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai
from qai.mechanics import GradientDescent, MSE, MAE, FixedEpochs, RetrainUntil

np.random.seed(42)

print("=" * 60)
print("TEST: Objective swap -- MSE vs MAE, a REAL behavioral difference")
print("(outlier robustness), not just a different formula on paper")
print("=" * 60)

# clean linear data, PLUS one extreme outlier
X = np.random.rand(50, 1) * 10
y = (2 * X.flatten()) + 3 + np.random.normal(0, 0.2, 50)
y[0] = 500.0  # one wildly extreme outlier

mse_model = qai.build(type="regression", optimizer=GradientDescent(
    learning_rate=0.001, objective=MSE(), training_loop=FixedEpochs(3000)))
mse_model.train(X, y, verbose=False)

mae_model = qai.build(type="regression", optimizer=GradientDescent(
    learning_rate=0.001, objective=MAE(), training_loop=FixedEpochs(3000)))
mae_model.train(X, y, verbose=False)

print(f"True underlying slope (ignoring the outlier): ~2.0")
print(f"MSE-trained slope: {mse_model.technique.w[0]:.3f}  (should be PULLED toward the outlier)")
print(f"MAE-trained slope: {mae_model.technique.w[0]:.3f}  (should stay CLOSER to the true 2.0)")

# REAL CHECK: MAE should be measurably more robust to the outlier than MSE
mse_error_from_truth = abs(mse_model.technique.w[0] - 2.0)
mae_error_from_truth = abs(mae_model.technique.w[0] - 2.0)
print(f"MSE's distance from true slope: {mse_error_from_truth:.3f}")
print(f"MAE's distance from true slope: {mae_error_from_truth:.3f}")
assert mae_error_from_truth < mse_error_from_truth, \
    "BUG: MAE should be more robust to the outlier than MSE, but wasn't"
print("VERIFIED: MAE is genuinely more outlier-robust than MSE -- a real behavioral difference, not just a label swap")

print()
print("=" * 60)
print("TEST: TrainingLoop swap -- does RetrainUntil ACTUALLY stop early?")
print("=" * 60)
X2 = np.random.rand(100, 2) * 5
y2 = X2 @ np.array([1.5, -0.5]) + 2.0 + np.random.normal(0, 0.05, 100)

fixed_model = qai.build(type="regression", optimizer=GradientDescent(
    learning_rate=0.05, training_loop=FixedEpochs(5000)))
fixed_model.train(X2, y2, verbose=False)
fixed_epochs_run = len(fixed_model.technique.optimizer.loss_history)
print(f"FixedEpochs(5000): ran {fixed_epochs_run} epochs (no matter what)")

retrain_model = qai.build(type="regression", optimizer=GradientDescent(
    learning_rate=0.05,
    training_loop=RetrainUntil(condition=lambda history: history[-1] < 0.01, max_epochs=5000)))
retrain_model.train(X2, y2, verbose=False)
retrain_epochs_run = len(retrain_model.technique.optimizer.loss_history)
print(f"RetrainUntil(loss < 0.01): ran {retrain_epochs_run} epochs (stopped once condition met)")

# REAL CHECK: RetrainUntil should have stopped noticeably earlier, and its
# final loss should actually satisfy the condition it was given
assert retrain_epochs_run < fixed_epochs_run, \
    f"BUG: RetrainUntil ran {retrain_epochs_run} epochs, not fewer than FixedEpochs's {fixed_epochs_run}"
final_loss = retrain_model.technique.optimizer.loss_history[-1]
print(f"RetrainUntil's actual final loss: {final_loss:.5f} (condition was < 0.01)")
assert final_loss < 0.01, "BUG: RetrainUntil stopped without actually satisfying its own condition"
print(f"VERIFIED: RetrainUntil genuinely stopped early ({retrain_epochs_run} vs {fixed_epochs_run} epochs), "
      f"and only after truly satisfying its condition")

print()
print("ALL OBJECTIVE + TRAININGLOOP TESTS PASSED -- Section 56's full three-part")
print("decomposition (Optimizer, Objective, TrainingLoop) is now genuinely real,")
print("each piece independently swappable, each swap verified to behave differently")
print("in a real, meaningful, checkable way -- not just accepted as a parameter")
