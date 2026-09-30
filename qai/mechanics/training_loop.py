"""
qai.mechanics.training_loop -- the third piece of Section 56's
decomposition. Answers "when does training stop," separable from both
Optimizer (how weights get found) and Objective (how wrongness is measured).
"""
from __future__ import annotations


class TrainingLoop:
    name = "training_loop"

    def should_continue(self, epoch: int, loss_history: list) -> bool:
        raise NotImplementedError


class FixedEpochs(TrainingLoop):
    """The original, implicit behavior -- run exactly N epochs, no matter
    what the loss is doing. Formalized as one option, not the only one."""
    name = "fixed_epochs"

    def __init__(self, epochs=1000):
        self.epochs = epochs

    def should_continue(self, epoch, loss_history):
        return epoch < self.epochs


class RetrainUntil(TrainingLoop):
    """Section 56's actual example from the design doc:
    `TrainingLoop: RetrainUntil(loss < 0.1)` -- stop as soon as a real
    condition on the loss is satisfied, not after some fixed count."""
    name = "retrain_until"

    def __init__(self, condition, max_epochs=10000):
        self.condition = condition  # fn(loss_history) -> bool
        self.max_epochs = max_epochs

    def should_continue(self, epoch, loss_history):
        if epoch >= self.max_epochs:
            return False
        if not loss_history:
            return True
        return not self.condition(loss_history)
