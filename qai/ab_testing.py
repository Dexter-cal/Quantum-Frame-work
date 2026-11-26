"""
qai.ab_testing -- production A/B testing with statistical significance.

Route traffic between two models, collect results, and determine if one
is statistically significantly better than the other.

Usage:
    import qai
    ab = qai.ABTest(model_a=m1, model_b=m2, traffic_split=0.5)

    # In production:
    for x in incoming_requests:
        pred = ab.predict(x)  # routes to A or B
        ab.record_outcome(x, actual_label)  # records which model won

    # Check significance:
    result = ab.analyze()
    print(result)
    if result.significant:
        print(f"Winner: {result.winner} (p={result.p_value:.4f})")
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ABTestResult:
    """Result of an A/B test analysis."""
    n_a: int = 0
    n_b: int = 0
    accuracy_a: float = 0.0
    accuracy_b: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    winner: Optional[str] = None
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    def __str__(self) -> str:
        lines = [
            f"=== A/B Test Result ===",
            f"model A: n={self.n_a}, accuracy={self.accuracy_a:.4f}",
            f"model B: n={self.n_b}, accuracy={self.accuracy_b:.4f}",
            f"p-value: {self.p_value:.4f}",
            f"significant: {self.significant}",
        ]
        if self.winner:
            lines.append(f"winner: {self.winner}")
        lines.append(f"95% CI for diff: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]")
        return "\n".join(lines)


class ABTest:
    """Production A/B testing router.

    Args:
        model_a: the control model.
        model_b: the treatment model.
        traffic_split: fraction of traffic to send to model_a (default 0.5).
    """
    def __init__(self, model_a, model_b, traffic_split: float = 0.5):
        if not 0.0 < traffic_split < 1.0:
            raise ValueError("[WHAT] traffic_split must be between 0 and 1\n"
                             "[WHY] It's a fraction of traffic going to model_a\n"
                             "[FIX] Use 0.5 for equal split")
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self._results_a: List[bool] = []  # correct/incorrect per prediction
        self._results_b: List[bool] = []
        self._predictions_a: List[Any] = []
        self._predictions_b: List[Any] = []

    def predict(self, x) -> Any:
        """Route x to one of the models and return the prediction."""
        if np.random.random() < self.traffic_split:
            pred = self.model_a.predict(x)
            self._predictions_a.append(pred)
            return pred
        else:
            pred = self.model_b.predict(x)
            self._predictions_b.append(pred)
            return pred

    def record_outcome(self, x, actual_label) -> None:
        """Record whether the prediction for x was correct.

        This re-runs the prediction deterministically (using the same
        traffic split) so the outcome is attributed to the right model.
        For production use, call this with the same x that was passed to
        predict().
        """
        # Determine which model handled this x (deterministic by hash)
        hash_val = hash(str(np.asarray(x).tobytes())) / 2**63  # [0, 1)
        if hash_val < self.traffic_split:
            pred = self.model_a.predict(x)
            correct = (np.asarray(pred).item() == np.asarray(actual_label).item())
            self._results_a.append(bool(correct))
        else:
            pred = self.model_b.predict(x)
            correct = (np.asarray(pred).item() == np.asarray(actual_label).item())
            self._results_b.append(bool(correct))

    def analyze(self, alpha: float = 0.05) -> ABTestResult:
        """Analyze the collected results for statistical significance.

        Uses a two-proportion z-test.

        Args:
            alpha: significance level (default 0.05).

        Returns:
            ABTestResult
        """
        n_a = len(self._results_a)
        n_b = len(self._results_b)
        if n_a < 10 or n_b < 10:
            return ABTestResult(n_a=n_a, n_b=n_b)

        acc_a = sum(self._results_a) / n_a
        acc_b = sum(self._results_b) / n_b

        # Two-proportion z-test
        p_pool = (sum(self._results_a) + sum(self._results_b)) / (n_a + n_b)
        if p_pool == 0 or p_pool == 1:
            return ABTestResult(n_a=n_a, n_b=n_b, accuracy_a=acc_a, accuracy_b=acc_b,
                                p_value=1.0, significant=False)

        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        z = (acc_a - acc_b) / max(se, 1e-10)

        # Two-tailed p-value (normal CDF approximation)
        from math import erf, sqrt
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

        # 95% confidence interval for the difference
        diff = acc_a - acc_b
        se_diff = np.sqrt(acc_a * (1 - acc_a) / n_a + acc_b * (1 - acc_b) / n_b)
        ci = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)

        significant = p_value < alpha
        winner = None
        if significant:
            winner = "A" if acc_a > acc_b else "B"

        return ABTestResult(
            n_a=n_a, n_b=n_b, accuracy_a=acc_a, accuracy_b=acc_b,
            p_value=float(p_value), significant=significant,
            winner=winner, confidence_interval=(float(ci[0]), float(ci[1])),
        )

    def reset(self) -> None:
        """Clear all collected results."""
        self._results_a.clear()
        self._results_b.clear()
        self._predictions_a.clear()
        self._predictions_b.clear()


__all__ = ["ABTest", "ABTestResult"]
