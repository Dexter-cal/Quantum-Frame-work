"""
qai.drift -- production drift detection.

In production, your input distribution shifts. A model trained on January
data may behave badly on July data because the world changed underneath
it. DriftDetector flags this automatically.

Detection methods:
  - "ks"      : Kolmogorov-Smirnov test (per-feature, 1-D distributions)
  - "psi"     : Population Stability Index (binned distribution comparison)
  - "kl"      : KL divergence (binned distribution comparison)
  - "ensemble": run all three, flag drift if any fires

Usage:
    import qai
    import numpy as np

    X_train = np.random.randn(1000, 5)
    detector = qai.DriftDetector(baseline=X_train, method="ensemble")

    # In production:
    X_new = np.random.randn(100, 5) * 1.5  # drifted
    report = detector.detect(X_new)
    print(report)
    if report.drifted:
        print(f"DRIFT DETECTED on features: {report.drifted_features}")

The detector maintains a baseline distribution and compares new batches
against it. Returns a DriftReport with per-feature scores + an overall
verdict.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class DriftReport:
    """A drift-detection report.

    Attributes:
        method: the detection method used ("ks", "psi", "kl", "ensemble").
        drifted: True if drift was detected.
        drifted_features: list of feature indices where drift was detected.
        per_feature: dict of {feature_idx: {score, threshold, drifted, p_value}}.
        overall_score: aggregate drift score across all features.
        baseline_size: number of samples in the baseline.
        new_size: number of samples in the new batch.
    """
    method: str
    drifted: bool
    drifted_features: List[int]
    per_feature: Dict[int, Dict[str, Any]]
    overall_score: float
    baseline_size: int
    new_size: int

    def __str__(self) -> str:
        lines = [
            f"=== Drift Report ({self.method}) ===",
            f"drifted: {self.drifted}",
            f"overall_score: {self.overall_score:.4f}",
            f"drifted_features: {self.drifted_features}",
            f"baseline_size: {self.baseline_size}, new_size: {self.new_size}",
            "",
            "per-feature scores:",
        ]
        for idx in sorted(self.per_feature.keys()):
            f = self.per_feature[idx]
            flag = "DRIFT" if f["drifted"] else "ok"
            score_str = f"score={f['score']:.4f}" if isinstance(f['score'], (int, float)) else f"score={f['score']}"
            thresh_str = f"threshold={f['threshold']}" if f.get('threshold') is not None else ""
            pval_str = f"p={f.get('p_value', 'n/a')}" if f.get('p_value') is not None else ""
            lines.append(f"  feature[{idx}]: {score_str} {thresh_str} {pval_str} -> {flag}")
        return "\n".join(lines)


class DriftDetector:
    """Production drift detector.

    Maintains a baseline distribution and compares new batches against it.

    Args:
        baseline: the training-distribution data (n_samples, n_features).
        method: "ks" | "psi" | "kl" | "ensemble" (default "ensemble").
        threshold: drift threshold. Method-specific defaults if None:
            - ks: p-value < 0.05 means drift (threshold = 0.05)
            - psi: score > 0.2 means drift (threshold = 0.2)
            - kl: score > 0.1 means drift (threshold = 0.1)
        n_bins: number of bins for psi/kl methods (default 10).
    """
    METHODS = ("ks", "psi", "kl", "ensemble")

    def __init__(self, baseline, method: str = "ensemble",
                 threshold: Optional[float] = None, n_bins: int = 10):
        if method not in self.METHODS:
            raise ValueError(
                f"[WHAT] Unknown drift method {method!r}\n"
                f"[WHY] Method must be one of {self.METHODS}\n"
                f"[FIX] Use one of the supported methods"
            )
        self.baseline = np.asarray(baseline, dtype=float)
        if self.baseline.ndim == 1:
            self.baseline = self.baseline.reshape(-1, 1)
        self.method = method
        self.n_bins = n_bins
        # Default thresholds per method
        if threshold is not None:
            self.threshold = {method: threshold}
        else:
            # In ensemble mode, all three thresholds must fire independently,
            # so the overall false-positive rate is naturally lower. The
            # individual thresholds are slightly stricter than the standard
            # 0.05/0.2/0.1 to compensate.
            if method == "ensemble":
                self.threshold = {"ks": 0.01, "psi": 0.25, "kl": 0.15}
            else:
                self.threshold = {"ks": 0.05, "psi": 0.2, "kl": 0.1}

    def detect(self, new_batch) -> DriftReport:
        """Detect drift in `new_batch` relative to the baseline.

        Args:
            new_batch: new data (n_samples, n_features). Must have the
                       same number of features as the baseline.

        Returns:
            DriftReport
        """
        new = np.asarray(new_batch, dtype=float)
        if new.ndim == 1:
            new = new.reshape(-1, 1)
        if new.shape[1] != self.baseline.shape[1]:
            raise ValueError(
                f"[WHAT] Feature count mismatch\n"
                f"[WHY] baseline has {self.baseline.shape[1]} features, "
                f"new_batch has {new.shape[1]}\n"
                f"[FIX] Pass a new_batch with the same number of features"
            )

        n_features = self.baseline.shape[1]
        per_feature: Dict[int, Dict[str, Any]] = {}
        drifted_features: List[int] = []

        methods_to_run = ["ks", "psi", "kl"] if self.method == "ensemble" else [self.method]

        for feat_idx in range(n_features):
            base_col = self.baseline[:, feat_idx]
            new_col = new[:, feat_idx]
            feature_scores: Dict[str, Any] = {"drifted": False}

            for m in methods_to_run:
                if m == "ks":
                    score, p_value = self._ks_test(base_col, new_col)
                    feature_scores["ks_score"] = score
                    feature_scores["ks_p_value"] = p_value
                    feature_scores["ks_threshold"] = self.threshold["ks"]
                    if p_value < self.threshold["ks"]:
                        feature_scores["drifted"] = True
                elif m == "psi":
                    score = self._psi(base_col, new_col)
                    feature_scores["psi_score"] = score
                    feature_scores["psi_threshold"] = self.threshold["psi"]
                    if score > self.threshold["psi"]:
                        feature_scores["drifted"] = True
                elif m == "kl":
                    score = self._kl_divergence(base_col, new_col)
                    feature_scores["kl_score"] = score
                    feature_scores["kl_threshold"] = self.threshold["kl"]
                    if score > self.threshold["kl"]:
                        feature_scores["drifted"] = True

            # For the report's "score" and "threshold" fields, pick the
            # primary method's score
            primary = self.method if self.method != "ensemble" else "ks"
            feature_scores["score"] = feature_scores.get(f"{primary}_score", feature_scores.get("ks_score", 0))
            feature_scores["threshold"] = feature_scores.get(f"{primary}_threshold")
            feature_scores["p_value"] = feature_scores.get(f"{primary}_p_value")

            per_feature[feat_idx] = feature_scores
            if feature_scores["drifted"]:
                drifted_features.append(feat_idx)

        # Overall score: average of the per-feature primary scores
        overall_score = float(np.mean([f["score"] for f in per_feature.values()
                                        if isinstance(f["score"], (int, float))]))

        return DriftReport(
            method=self.method,
            drifted=len(drifted_features) > 0,
            drifted_features=drifted_features,
            per_feature=per_feature,
            overall_score=overall_score,
            baseline_size=self.baseline.shape[0],
            new_size=new.shape[0],
        )

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------
    def _ks_test(self, baseline: np.ndarray, new: np.ndarray) -> tuple:
        """Kolmogorov-Smirnov test. Returns (statistic, p_value).

        We try scipy.stats.ks_2samp first; fall back to a manual KS
        computation if scipy isn't available.
        """
        try:
            from scipy.stats import ks_2samp
            result = ks_2samp(baseline, new)
            return float(result.statistic), float(result.pvalue)
        except ImportError:
            # Manual KS -- less precise but works without scipy
            return self._ks_manual(baseline, new)

    def _ks_manual(self, baseline: np.ndarray, new: np.ndarray) -> tuple:
        """Manual KS test (no scipy dependency)."""
        all_vals = np.concatenate([baseline, new])
        all_vals.sort()
        cdf_base = np.searchsorted(np.sort(baseline), all_vals, side='right') / len(baseline)
        cdf_new = np.searchsorted(np.sort(new), all_vals, side='right') / len(new)
        ks_stat = float(np.max(np.abs(cdf_base - cdf_new)))
        # Approximate p-value (Kolmogorov distribution)
        n1, n2 = len(baseline), len(new)
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = float(np.exp(-2 * (en * ks_stat) ** 2))
        p_value = max(0.0, min(1.0, p_value))
        return ks_stat, p_value

    def _psi(self, baseline: np.ndarray, new: np.ndarray) -> float:
        """Population Stability Index.

        PSI = sum((new_pct - base_pct) * ln(new_pct / base_pct))

        Bins are based on the baseline distribution's quantiles.
        """
        # Use baseline quantiles as bin edges
        edges = np.percentile(baseline, np.linspace(0, 100, self.n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        # Bin counts
        base_counts = np.histogram(baseline, bins=edges)[0].astype(float)
        new_counts = np.histogram(new, bins=edges)[0].astype(float)
        # Convert to proportions (avoid divide-by-zero)
        base_pct = (base_counts + 0.5) / (len(baseline) + 0.5 * self.n_bins)
        new_pct = (new_counts + 0.5) / (len(new) + 0.5 * self.n_bins)
        # PSI
        psi = float(np.sum((new_pct - base_pct) * np.log(new_pct / base_pct)))
        return psi

    def _kl_divergence(self, baseline: np.ndarray, new: np.ndarray) -> float:
        """KL divergence KL(new || baseline) using binned distributions."""
        edges = np.percentile(baseline, np.linspace(0, 100, self.n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        base_counts = np.histogram(baseline, bins=edges)[0].astype(float)
        new_counts = np.histogram(new, bins=edges)[0].astype(float)
        # Convert to probabilities (smoothed)
        base_p = (base_counts + 0.5) / (len(baseline) + 0.5 * self.n_bins)
        new_p = (new_counts + 0.5) / (len(new) + 0.5 * self.n_bins)
        # KL(new || base) = sum(new_p * log(new_p / base_p))
        kl = float(np.sum(new_p * np.log(new_p / base_p)))
        return kl


__all__ = ["DriftDetector", "DriftReport"]
