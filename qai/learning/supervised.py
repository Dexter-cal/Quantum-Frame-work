from .base import LearningTechnique


class Supervised(LearningTechnique):
    name = "supervised"
    on_step_signature = ("input", "expected_output", "actual_output")

    def requires(self) -> str:
        return "a labeled dataset (X, y)"


class Unsupervised(LearningTechnique):
    name = "unsupervised"
    on_step_signature = ("input", "discovered_pattern")

    def requires(self) -> str:
        return "raw, unlabeled data (X only)"
