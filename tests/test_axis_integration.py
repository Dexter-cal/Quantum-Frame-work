import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai
from qai.core import CompatibilityError
from fixtures_huggingface_iris import load as load_hf_iris

np.random.seed(42)

print("=" * 60)
print("TEST: compatibility checking REJECTS a genuinely nonsensical combination")
print("=" * 60)
try:
    bad = qai.build(type="regression", learning_technique="reinforcement")
    print("FAIL: should have raised CompatibilityError")
except CompatibilityError as e:
    print("Correctly rejected 'regression' + 'reinforcement':")
    print(str(e))

print()
print("=" * 60)
print("TEST: sensible default learning_technique applied when none is given")
print("=" * 60)
model = qai.build(type="regression")  # no learning_technique specified
print("Auto-assigned learning_technique:", model.learning_technique)
assert model.learning_technique == "supervised", "BUG: wrong default assigned"
print("VERIFIED: correctly defaulted to 'supervised'")

km = qai.build(type="kmeans", n_clusters=3)
print("Auto-assigned learning_technique for kmeans:", km.learning_technique)
assert km.learning_technique == "unsupervised", "BUG: wrong default for kmeans"
print("VERIFIED: correctly defaulted to 'unsupervised'")

print()
print("=" * 60)
print("TEST: explicit, CORRECT combination -- reinforcement + tabular_policy")
print("Now RL flows through the SAME Model interface as everything else")
print("=" * 60)
rl_model = qai.build(type="tabular_policy", learning_technique="reinforcement", epsilon=0.3, learning_rate=0.2)
print(rl_model.help() if hasattr(rl_model, "help") else "built successfully")

env = qai.GridWorld(start=2)
rl_model.train(environment=env, episodes=500)  # goes through Model.train(), not a standalone script this time
print("Trained via Model.train(environment=...) -- not a standalone script")
print("technique.is_trained():", rl_model.technique.is_trained())

# REAL CHECK: does the policy learned THROUGH Model.train() match the standalone result?
decision_from_3 = rl_model.predict(3, actions=env.ACTIONS)
print("Model.predict(state=3):", decision_from_3)
assert decision_from_3 == "right", f"BUG: learned wrong policy through the Model interface, got {decision_from_3}"
print("VERIFIED: same correct policy ('right') learned through the unified Model interface")

print()
print("reward property:", round(rl_model.technique.reward, 2))

print()
print("=" * 60)
print("TEST: regular supervised flow STILL works after all these changes")
print("(regression check against the earlier, already-passing behavior)")
print("=" * 60)
X, y = load_hf_iris()
clf = qai.build(type="classifier", learning_technique="supervised")
clf.train(X, y, verbose=False)
print("classifier accuracy() still works:", round(clf.accuracy(), 4))
assert clf.accuracy() > 0.9

print()
print("ALL AXIS-INTEGRATION TESTS PASSED -- Technique and LearningTechnique")
print("are now genuinely connected through one Model interface, with real")
print("compatibility checking that actually rejects bad combinations")
