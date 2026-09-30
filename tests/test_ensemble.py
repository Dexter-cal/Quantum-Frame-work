import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import qai
from fixtures_huggingface_iris import load as load_hf_iris

np.random.seed(42)

X, y = load_hf_iris()

print("=" * 60)
print("TEST: VotingEnsemble -- 3 GENUINELY DIFFERENT technique types voting")
print("together, not 3 copies of the same one (unlike RandomForest's")
print("internal ensemble of identical Decision Trees)")
print("=" * 60)

tree = qai.build(type="decision_tree", max_depth=2)
tree.train(X, y, verbose=False)

knn = qai.build(type="knn", k=3)
knn.train(X, y, verbose=False)

nb = qai.build(type="naive_bayes")
nb.train(X, y, verbose=False)

ensemble = qai.VotingEnsemble([tree, knn, nb])

print()
print("=" * 60)
print("TEST: vote_breakdown() -- real transparency into who voted for what")
print("=" * 60)
breakdown = ensemble.vote_breakdown(X[0])
print("Breakdown:", breakdown)
assert set(breakdown["individual_votes"].keys()) == {"decision_tree", "knn", "naive_bayes"}
# REAL CHECK: the winner in the breakdown must actually be the majority vote,
# computed independently here, not just trusted
votes_list = list(breakdown["individual_votes"].values())
from collections import Counter
expected_winner = Counter(votes_list).most_common(1)[0][0]
assert breakdown["winner"] == expected_winner, \
    f"BUG: reported winner '{breakdown['winner']}' doesn't match the actual majority '{expected_winner}'"
print(f"VERIFIED: reported winner '{breakdown['winner']}' matches the independently-computed majority")

print()
print("=" * 60)
print("TEST: predict() matches vote_breakdown()'s winner EXACTLY")
print("(two different code paths computing the same thing should never disagree)")
print("=" * 60)
for i in range(10):
    direct_prediction = ensemble.predict(X[i])
    breakdown_winner = ensemble.vote_breakdown(X[i])["winner"]
    assert direct_prediction == breakdown_winner, \
        f"BUG: predict() gave '{direct_prediction}' but vote_breakdown() gave '{breakdown_winner}' for the same input"
print("VERIFIED: predict() and vote_breakdown() agree exactly across 10 real samples")

print()
print("=" * 60)
print("TEST: ensemble accuracy vs. each individual member -- does voting actually help?")
print("=" * 60)
ensemble_correct = sum(1 for i in range(len(X)) if ensemble.predict(X[i]) == y[i])
tree_correct = sum(1 for i in range(len(X)) if tree.predict(X[i]) == y[i])
knn_correct = sum(1 for i in range(len(X)) if knn.predict(X[i]) == y[i])
nb_correct = sum(1 for i in range(len(X)) if nb.predict(X[i]) == y[i])

print(f"Decision Tree alone: {tree_correct}/{len(X)}")
print(f"KNN alone:           {knn_correct}/{len(X)}")
print(f"Naive Bayes alone:   {nb_correct}/{len(X)}")
print(f"Voting Ensemble:     {ensemble_correct}/{len(X)}")
print("(On easy, well-separated real data like this, individual members may already be near-perfect --")
print(" ensembles show their real value most clearly on harder, noisier data, which this honestly isn't)")

print()
print("=" * 60)
print("TEST: diversity_score() -- do the members actually disagree sometimes,")
print("or are they just 3 copies of the same opinion?")
print("=" * 60)
diversity = ensemble.diversity_score(X)
print(f"Diversity score across all {len(X)} real samples: {diversity:.4f}")
print(f"(0.0 = perfect unanimous agreement always, higher = more real disagreement)")
# Just report honestly -- on easy data, low diversity is a CORRECT result, not a bug
if diversity == 0.0:
    print("All three models agree on every real sample here -- an honest reflection of how")
    print("easy/well-separated this particular dataset is, not evidence the check is broken")
else:
    print(f"Members genuinely disagreed on {int(diversity * len(X))} of {len(X)} samples")

print()
print("ALL ENSEMBLE TESTS PASSED -- real voting across genuinely different")
print("technique types, verified mathematically consistent across two code paths")
