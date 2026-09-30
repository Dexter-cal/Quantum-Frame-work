"""
qai.core.ensemble -- Section 24/46: combining several independently
trained models' predictions via voting. Genuinely different from
Pipeline (sequential, output->input) -- this is PARALLEL, same input to
every model, votes combined.
"""
from __future__ import annotations
from collections import Counter
import numpy as np


class VotingEnsemble:
    def __init__(self, models: list):
        self.models = models  # can be genuinely DIFFERENT technique types

    def predict(self, x):
        votes = [model.predict(x) for model in self.models]
        # majority vote, real tie-breaking by first-seen order (deterministic, not random)
        counts = Counter(votes)
        winner = counts.most_common(1)[0][0]
        return winner

    def vote_breakdown(self, x):
        """Real transparency: which model voted for what, and the final tally --
        Section 46's Ensemble.voting_breakdown() made concrete."""
        votes = {model.technique_name: model.predict(x) for model in self.models}
        tally = Counter(votes.values())
        return {"individual_votes": votes, "tally": dict(tally), "winner": tally.most_common(1)[0][0]}

    def diversity_score(self, X):
        """How often do the ensemble members actually DISAGREE with each
        other, across real data -- Section 46's diversity_score(), made
        concrete. 0.0 = always unanimous, higher = more disagreement."""
        X = np.asarray(X)
        disagreements = 0
        for x in X:
            votes = set(model.predict(x) if not isinstance(model.predict(x), np.ndarray)
                        else tuple(model.predict(x)) for model in self.models)
            if len(votes) > 1:
                disagreements += 1
        return disagreements / len(X)
