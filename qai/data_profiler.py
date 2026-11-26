"""
qai.data_profiler -- dataset health checker.

Profiles a dataset and reports: missing values, outliers, type inference,
distribution stats, correlations, duplicate rows, class balance.

Usage:
    import qai
    import numpy as np
    X = np.random.randn(100, 5)
    X[5, 2] = np.nan  # add a missing value
    X[10, 0] = 1000   # add an outlier
    report = qai.DataProfiler(X).profile(y=(X[:, 0] > 0).astype(int))
    print(report)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DataProfile:
    """A dataset health report."""
    n_samples: int = 0
    n_features: int = 0
    n_missing: int = 0
    n_duplicates: int = 0
    n_outliers: int = 0
    feature_types: Dict[str, str] = field(default_factory=dict)
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    correlations: Optional[Dict[str, float]] = None
    class_balance: Optional[Dict[str, float]] = None
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"=== Data Profile ===",
            f"samples:      {self.n_samples}",
            f"features:     {self.n_features}",
            f"missing:      {self.n_missing} ({self.n_missing/max(self.n_samples*self.n_features,1):.1%})",
            f"duplicates:   {self.n_duplicates}",
            f"outliers:     {self.n_outliers}",
        ]
        if self.class_balance:
            lines.append(f"class balance: {self.class_balance}")
        if self.warnings:
            lines.append("")
            lines.append("warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.feature_stats:
            lines.append("")
            lines.append(f"{'feature':10s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'type':>6s}")
            lines.append("-" * 55)
            for name, stats in list(self.feature_stats.items())[:10]:
                lines.append(f"{name:10s} {stats.get('mean', 0):>8.3f} {stats.get('std', 0):>8.3f} "
                             f"{stats.get('min', 0):>8.3f} {stats.get('max', 0):>8.3f} "
                             f"{self.feature_types.get(name, '?'):>6s}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples, "n_features": self.n_features,
            "n_missing": self.n_missing, "n_duplicates": self.n_duplicates,
            "n_outliers": self.n_outliers, "warnings": self.warnings,
            "feature_stats": self.feature_stats, "class_balance": self.class_balance,
        }


class DataProfiler:
    """Profile a dataset for health, quality, and distribution.

    Args:
        X: input data (n_samples, n_features).
    """
    def __init__(self, X):
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)

    def profile(self, y=None) -> DataProfile:
        """Run the full profile. Returns a DataProfile."""
        X = self.X
        n, d = X.shape
        report = DataProfile(n_samples=n, n_features=d)

        # 1. Missing values
        nan_mask = np.isnan(X)
        report.n_missing = int(np.sum(nan_mask))
        if report.n_missing > 0:
            report.warnings.append(f"{report.n_missing} missing values detected")

        # 2. Duplicates
        try:
            _, counts = np.unique(X, axis=0, return_counts=True)
            report.n_duplicates = int(np.sum(counts[counts > 1] - 1))
            if report.n_duplicates > n * 0.1:
                report.warnings.append(f"{report.n_duplicates} duplicate rows (>10% of data)")
        except Exception:
            pass

        # 3. Outliers (IQR method per feature)
        outlier_count = 0
        for j in range(d):
            col = X[:, j]
            col_clean = col[~np.isnan(col)]
            if len(col_clean) < 4:
                continue
            q1, q3 = np.percentile(col_clean, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = np.sum((col_clean < lower) | (col_clean > upper))
            outlier_count += int(outliers)
        report.n_outliers = outlier_count
        if outlier_count > n * 0.05:
            report.warnings.append(f"{outlier_count} outliers detected (>5% of data)")

        # 4. Feature stats + type inference
        for j in range(d):
            col = X[:, j]
            col_clean = col[~np.isnan(col)]
            if len(col_clean) == 0:
                report.feature_stats[f"x{j}"] = {"mean": 0, "std": 0, "min": 0, "max": 0}
                report.feature_types[f"x{j}"] = "all_missing"
                continue
            mean = float(np.mean(col_clean))
            std = float(np.std(col_clean))
            report.feature_stats[f"x{j}"] = {
                "mean": mean, "std": std,
                "min": float(np.min(col_clean)),
                "max": float(np.max(col_clean)),
                "n_missing": int(np.sum(np.isnan(col))),
            }
            # Type inference
            unique_vals = np.unique(col_clean)
            if len(unique_vals) <= 2:
                report.feature_types[f"x{j}"] = "binary"
            elif len(unique_vals) <= 20 and all(v == int(v) for v in unique_vals):
                report.feature_types[f"x{j}"] = "categorical"
            else:
                report.feature_types[f"x{j}"] = "continuous"

        # 5. Correlations (top 5 pairs)
        if d > 1 and d <= 50:
            try:
                X_clean = np.nan_to_num(X, nan=0.0)
                corr = np.corrcoef(X_clean.T)
                pairs = []
                for i in range(d):
                    for j in range(i+1, d):
                        if not np.isnan(corr[i, j]):
                            pairs.append((f"x{i}_x{j}", float(corr[i, j])))
                pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                report.correlations = dict(pairs[:5])
                # Warn about high correlations
                for name, val in report.correlations.items():
                    if abs(val) > 0.95:
                        report.warnings.append(f"High correlation: {name} = {val:.3f}")
            except Exception:
                pass

        # 6. Class balance (if y provided)
        if y is not None:
            y_arr = np.asarray(y)
            classes, counts = np.unique(y_arr, return_counts=True)
            total = len(y_arr)
            report.class_balance = {str(c): float(count / total) for c, count in zip(classes, counts)}
            # Warn about imbalance
            if len(classes) > 1:
                min_ratio = min(counts) / max(counts)
                if min_ratio < 0.2:
                    report.warnings.append(f"Class imbalance: min/max ratio = {min_ratio:.2f}")

        return report


__all__ = ["DataProfiler", "DataProfile"]
