"""
qai.learning.base -- Section 66/67 of the design doc: LEARNING TECHNIQUE
is a separate axis from TYPE/TECHNIQUE. A Technique (Regression, CNN...)
answers "what shape is the model." A LearningTechnique (Supervised,
Reinforcement...) answers "how does it learn." Independent concerns.
"""
from __future__ import annotations


class LearningTechnique:
    """Base class every learning paradigm implements. Each one declares
    its own on_step signature and its own tunable params -- genuinely
    different across paradigms, per Section 105's signature table."""

    name: str = "learning_technique"
    on_step_signature: tuple = ()  # e.g. ("state", "decision", "outcome") for RL

    def __init__(self, **params):
        self.params = params

    def requires(self) -> str:
        """What kind of data/setup this paradigm needs -- Section 105's
        'Needs' column, made real and queryable, not just documentation."""
        raise NotImplementedError

    def help(self) -> str:
        sig = ", ".join(self.on_step_signature) if self.on_step_signature else "(none)"
        return f"[{self.name}] on_step signature: ({sig}) | requires: {self.requires()} | params: {self.params}"
