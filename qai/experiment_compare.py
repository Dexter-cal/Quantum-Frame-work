"""
qai.experiment_compare -- compare multiple experiment runs side by side.

This module provides a richer, side-by-side comparison toolkit for experiment
runs / models. It is intended to complement ``qai.experiment_tracking`` and
``qai.benchmark_suite`` -- you can collect runs from an ``ExperimentTracker``
(or from a JSON storage directory), build an :class:`ExperimentComparison`,
rank runs by metrics, find Pareto-optimal tradeoffs, and render the whole thing
as markdown, JSON, or self-contained HTML with inline SVG charts.

Public API
----------
Classes:
    ExperimentRun          -- a single experiment run (model + metrics + params)
    ExperimentComparison   -- compares multiple ExperimentRuns
    ComparisonVisualizer   -- renders comparisons as markdown / HTML / SVG
    RunCollector           -- auto-collects runs from a tracker or storage path

    (Legacy, kept for backward compatibility:)
    ExperimentCompare      -- compare models on the same test set
    ComparisonReport       -- formatted comparison report (legacy)
    ComparisonRow          -- single row in a legacy comparison report

Functions:
    compare(models_or_runs, X=None, y=None, metrics=None) -> ExperimentComparison
    from_tracker(tracker, n=10) -> ExperimentComparison
    leaderboard(runs, metric='accuracy', higher_is_better=True) -> str
    pareto_front(runs, metric_a, metric_b) -> list[ExperimentRun]

Design
------
* Stdlib + numpy only at import time. ``scipy`` is imported lazily for the
  paired t-test; if it is missing, :meth:`ExperimentComparison.statistical_significance`
  returns ``{'error': 'scipy not installed'}`` instead of raising.
* No external CSS/JS for the HTML output -- all charts are inline SVG.
* Edge cases handled: runs with different metrics, missing values, a single
  run, identical runs.

Usage
-----
    from qai.experiment_compare import (
        ExperimentRun, ExperimentComparison, ComparisonVisualizer,
        RunCollector, compare, from_tracker, leaderboard, pareto_front,
    )

    runs = [
        ExperimentRun('mlp_v1', metrics={'accuracy': 0.85, 'latency_ms': 12},
                       params={'lr': 0.01}),
        ExperimentRun('mlp_v2', metrics={'accuracy': 0.91, 'latency_ms': 15},
                       params={'lr': 0.005}),
    ]
    comp = ExperimentComparison(runs)
    print(comp.metric_table())
    print('Best:', comp.best('accuracy').name)

    viz = ComparisonVisualizer()
    html = viz.to_html(comp)        # self-contained
    svg = viz.radar_chart(comp, ['accuracy', 'latency_ms'])
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Optional heavy deps for the legacy ExperimentCompare class. If they are not
# importable we degrade gracefully.
try:
    from .profiler import Profiler, Profile  # type: ignore
    from .evaluate import evaluate, EvaluationReport  # type: ignore
except ImportError:  # pragma: no cover - optional deps
    Profiler = None  # type: ignore
    Profile = None  # type: ignore
    evaluate = None  # type: ignore
    EvaluationReport = None  # type: ignore


# =====================================================================
# Helpers
# =====================================================================
_LOWER_IS_BETTER_HINTS: Tuple[str, ...] = (
    "latency", "loss", "error", "time", "mse", "rmse", "mae", "ms",
    "sec", "duration", "memory", "size", "complexity", "perplexity",
    "wer", "cer", "mer", "fpr", "fnr", "far", "frr", "fp", "fn",
)


def _is_number(v: Any) -> bool:
    """True if v is a real number (not bool, not None)."""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        # NaN/inf are not "real" for ranking purposes
        try:
            return not (math.isnan(float(v)) or math.isinf(float(v)))
        except (TypeError, ValueError):
            return False
    return False


def _higher_is_better(metric: str) -> bool:
    """Heuristic: guess whether higher is better for a metric name.

    Names containing 'latency', 'loss', 'error', 'time', 'mse' etc. are
    treated as lower-is-better. Everything else defaults to higher-is-better.
    """
    name = (metric or "").lower()
    return not any(h in name for h in _LOWER_IS_BETTER_HINTS)


def _fmt_metric(v: Any) -> str:
    """Format a metric value for a markdown table cell."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        if math.isnan(f) or math.isinf(f):
            return str(v)
        # Compact float formatting
        if abs(f) >= 1000 or (abs(f) < 0.001 and f != 0):
            return f"{f:.3e}"
        if f == int(f) and abs(f) < 1e6:
            return f"{int(f)}"
        return f"{f:.4g}"
    s = str(v)
    if len(s) > 40:
        s = s[:37] + "..."
    return s


def _html_escape(s: Any) -> str:
    """Minimal HTML escaping for text content."""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _md_table_to_html(md: str) -> str:
    """Convert a simple markdown table (no nested pipes) to an HTML table."""
    lines = [ln for ln in md.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return _html_escape(md)
    header = [c.strip() for c in lines[0].split("|")[1:-1]]
    parts = ['<table border="1" cellpadding="6" cellspacing="0" '
             'style="border-collapse:collapse;">']
    parts.append("<thead><tr>")
    for h in header:
        parts.append(f"<th>{_html_escape(h)}</th>")
    parts.append("</tr></thead><tbody>")
    # Skip separator row (line index 1)
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.split("|")[1:-1]]
        parts.append("<tr>")
        for c in cells:
            parts.append(f"<td>{_html_escape(c)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


# A small fixed palette -- colorblind-friendly-ish defaults.
_COLORS: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


# =====================================================================
# ExperimentRun
# =====================================================================
class ExperimentRun:
    """A single experiment run.

    Parameters
    ----------
    name : str
        Human-readable run name (used in tables and charts).
    model : Any, optional
        The actual model object. Not serialized by :meth:`to_dict`.
    metrics : dict, optional
        Mapping of metric name -> value (accuracy, loss, latency_ms, ...).
        Values may be scalars (numbers/strings) or, for statistical tests,
        lists/arrays of per-sample observations.
    params : dict, optional
        Hyperparameters used for this run.
    dataset : str, optional
        Name of the dataset this run was evaluated on.
    timestamp : float, optional
        Unix timestamp. Defaults to ``time.time()``.
    """

    def __init__(
        self,
        name: str,
        model: Any = None,
        metrics: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        dataset: Optional[str] = None,
        timestamp: Optional[float] = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("[WHAT] ExperimentRun requires a non-empty name\n"
                             "[FIX] Pass name='...'")
        self.name: str = name
        self.model: Any = model
        self.metrics: Dict[str, Any] = dict(metrics or {})
        self.params: Dict[str, Any] = dict(params or {})
        self.dataset: Optional[str] = dataset
        self.timestamp: float = float(timestamp) if timestamp is not None else time.time()

    # -- serialization ------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict.

        The ``model`` attribute is intentionally omitted -- it is usually
        not picklable/JSON-serializable.
        """
        return {
            "name": self.name,
            "metrics": self.metrics,
            "params": self.params,
            "dataset": self.dataset,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRun":
        """Reconstruct an :class:`ExperimentRun` from a dict."""
        return cls(
            name=d.get("name", "run"),
            metrics=d.get("metrics") or {},
            params=d.get("params") or {},
            dataset=d.get("dataset"),
            timestamp=d.get("timestamp"),
        )

    # -- convenience --------------------------------------------------
    def summary(self) -> str:
        """One-line summary: ``name | metric=value, metric=value``."""
        parts = []
        for k, v in self.metrics.items():
            parts.append(f"{k}={_fmt_metric(v)}")
        metric_str = ", ".join(parts) if parts else "(no metrics)"
        return f"{self.name} | {metric_str}"

    def __repr__(self) -> str:
        return f"<ExperimentRun name={self.name!r} metrics={list(self.metrics)}>"


# =====================================================================
# ExperimentComparison
# =====================================================================
class ExperimentComparison:
    """Compares multiple :class:`ExperimentRun` objects.

    Parameters
    ----------
    runs : list[ExperimentRun]
        The runs to compare. May be empty (methods will return trivial
        results -- e.g. ``metric_table()`` returns just the header).
    """

    def __init__(self, runs: Optional[Sequence[ExperimentRun]] = None):
        self.runs: List[ExperimentRun] = list(runs or [])
        self._by_name: Dict[str, ExperimentRun] = {}
        for r in self.runs:
            # Last one wins for duplicates (mirrors add_run() behavior).
            self._by_name[r.name] = r

    # -- mutation -----------------------------------------------------
    def add_run(self, run: ExperimentRun) -> None:
        """Add a run. If a run with the same name exists, it is replaced."""
        if not isinstance(run, ExperimentRun):
            raise TypeError("[WHAT] add_run requires an ExperimentRun\n"
                            "[FIX] Wrap your data in ExperimentRun first")
        if run.name in self._by_name:
            self.runs = [r for r in self.runs if r.name != run.name]
        self.runs.append(run)
        self._by_name[run.name] = run

    def remove_run(self, name: str) -> None:
        """Remove a run by name. No-op if not found."""
        self.runs = [r for r in self.runs if r.name != name]
        self._by_name.pop(name, None)

    def get_run(self, name_or_run: Union[str, ExperimentRun]) -> ExperimentRun:
        """Resolve a name or run object to a run in this comparison."""
        if isinstance(name_or_run, ExperimentRun):
            return name_or_run
        if name_or_run in self._by_name:
            return self._by_name[name_or_run]
        raise KeyError(f"No run named {name_or_run!r} in this comparison")

    # -- introspection ------------------------------------------------
    def all_metrics(self) -> List[str]:
        """Union of metric names across all runs, in first-seen order."""
        seen = set()
        ordered: List[str] = []
        for r in self.runs:
            for k in r.metrics.keys():
                if k not in seen:
                    seen.add(k)
                    ordered.append(k)
        return ordered

    def all_params(self) -> List[str]:
        """Union of param names across all runs, in first-seen order."""
        seen = set()
        ordered: List[str] = []
        for r in self.runs:
            for k in r.params.keys():
                if k not in seen:
                    seen.add(k)
                    ordered.append(k)
        return ordered

    # -- tables -------------------------------------------------------
    def metric_table(self) -> str:
        """Markdown table of all runs x all metrics."""
        metrics = self.all_metrics()
        header = ["Run"] + metrics
        sep = ["---"] * len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for r in self.runs:
            row = [r.name]
            for m in metrics:
                row.append(_fmt_metric(r.metrics.get(m)))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def param_table(self) -> str:
        """Markdown table of all runs x all params."""
        params = self.all_params()
        header = ["Run"] + params
        sep = ["---"] * len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for r in self.runs:
            row = [r.name]
            for p in params:
                row.append(_fmt_metric(r.params.get(p)))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    # -- ranking ------------------------------------------------------
    def _scored(self, metric: str) -> List[Tuple[ExperimentRun, float]]:
        out: List[Tuple[ExperimentRun, float]] = []
        for r in self.runs:
            v = r.metrics.get(metric)
            if _is_number(v):
                out.append((r, float(v)))
        return out

    def best(self, metric: str = "accuracy",
             higher_is_better: bool = True) -> ExperimentRun:
        """Return the best run for a metric.

        Raises
        ------
        ValueError
            If no runs have a numeric value for ``metric``.
        """
        scored = self._scored(metric)
        if not scored:
            raise ValueError(
                f"[WHAT] No runs have a numeric value for metric {metric!r}\n"
                f"[FIX] Pick a metric that exists: {self.all_metrics()}"
            )
        return max(scored, key=lambda x: x[1])[0] if higher_is_better else \
               min(scored, key=lambda x: x[1])[0]

    def worst(self, metric: str = "accuracy",
              higher_is_better: bool = True) -> ExperimentRun:
        """Return the worst run for a metric (opposite of :meth:`best`)."""
        return self.best(metric=metric, higher_is_better=not higher_is_better)

    def rank(self, metric: str = "accuracy",
             higher_is_better: bool = True
             ) -> List[Tuple[int, ExperimentRun]]:
        """Return a ranked list of (rank, run) tuples.

        Runs that lack a numeric value for ``metric`` are excluded from the
        ranking (they cannot be ordered).
        """
        scored = self._scored(metric)
        scored.sort(key=lambda x: x[1], reverse=higher_is_better)
        return [(i + 1, r) for i, (r, _v) in enumerate(scored)]

    def pairwise_diff(self, run_a: Union[str, ExperimentRun],
                      run_b: Union[str, ExperimentRun]) -> Dict[str, Any]:
        """Diff two runs' metrics.

        Returns a dict keyed by metric name. Each entry has
        ``{a, b, delta, pct_change}``. ``delta = a - b``. ``pct_change``
        is the relative change ``((a - b) / b) * 100`` (or ``None`` if ``b``
        is zero or non-numeric).
        """
        a = self.get_run(run_a)
        b = self.get_run(run_b)
        all_metrics = list(set(a.metrics) | set(b.metrics))
        all_metrics.sort()
        out: Dict[str, Any] = {}
        for m in all_metrics:
            va = a.metrics.get(m)
            vb = b.metrics.get(m)
            entry: Dict[str, Any] = {"a": va, "b": vb, "delta": None, "pct_change": None}
            if _is_number(va) and _is_number(vb):
                fa, fb = float(va), float(vb)
                entry["delta"] = fa - fb
                if fb != 0:
                    entry["pct_change"] = (fa - fb) / abs(fb) * 100.0
            out[m] = entry
        return out

    def statistical_significance(self, metric: str = "accuracy") -> Dict[str, Any]:
        """Run paired t-tests across runs (if scipy is available).

        A run is eligible only if its value for ``metric`` is array-like
        (list / 1-D numpy array / tuple). Scalar-valued metrics cannot be
        paired-t-tested on their own.

        Returns
        -------
        dict
            ``{'metric': ..., 'pairwise': {f'{a} vs {b}': {...}, ...}}``
            on success, or ``{'error': 'scipy not installed'}`` if scipy
            is missing, or ``{'error': '... explanation ...'}`` if there
            is not enough data.
        """
        try:
            from scipy import stats  # type: ignore
        except ImportError:
            return {"error": "scipy not installed"}

        arrays: List[Tuple[str, np.ndarray]] = []
        for r in self.runs:
            v = r.metrics.get(metric)
            if isinstance(v, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(v, dtype=float).flatten()
                except Exception:
                    continue
                if arr.size >= 2:
                    arrays.append((r.name, arr))
        if len(arrays) < 2:
            return {
                "error": ("paired t-test requires at least 2 runs with "
                          "array-valued metrics (>=2 observations each)"),
                "metric": metric,
                "n_eligible": len(arrays),
            }

        results: Dict[str, Any] = {}
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                na, a = arrays[i]
                nb, b = arrays[j]
                n = min(len(a), len(b))
                if n < 2:
                    results[f"{na} vs {nb}"] = {
                        "error": "need >=2 paired observations",
                        "n_pairs": n,
                    }
                    continue
                a_p = a[:n]
                b_p = b[:n]
                # Mask NaNs/inf in either array
                mask = np.isfinite(a_p) & np.isfinite(b_p)
                a_p, b_p = a_p[mask], b_p[mask]
                if len(a_p) < 2:
                    results[f"{na} vs {nb}"] = {
                        "error": "need >=2 finite paired observations",
                        "n_pairs": int(mask.sum()),
                    }
                    continue
                try:
                    t_stat, p_val = stats.ttest_rel(a_p, b_p)
                    if math.isnan(float(t_stat)):
                        results[f"{na} vs {nb}"] = {
                            "error": "t-statistic undefined (e.g. zero variance)",
                            "n_pairs": len(a_p),
                        }
                    else:
                        results[f"{na} vs {nb}"] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(p_val),
                            "significant_at_0.05": float(p_val) < 0.05,
                            "n_pairs": len(a_p),
                            "mean_diff": float(np.mean(a_p - b_p)),
                        }
                except Exception as e:
                    results[f"{na} vs {nb}"] = {"error": str(e)}
        return {"metric": metric, "pairwise": results}

    def tradeoffs(self, metric_a: str,
                  metric_b: str) -> List[Dict[str, Any]]:
        """Find runs that are Pareto-optimal w.r.t. two metrics.

        Whether higher-is-better is inferred from the metric name using the
        :func:`_higher_is_better` heuristic (e.g. ``latency_ms`` and
        ``loss`` are lower-is-better; ``accuracy`` and ``f1`` are higher).

        Returns
        -------
        list of dict
            One entry per Pareto-optimal run. Each entry is
            ``{'name': ..., 'run': ExperimentRun, metric_a: ..., metric_b: ...}``.
        """
        a_higher = _higher_is_better(metric_a)
        b_higher = _higher_is_better(metric_b)

        candidates: List[Dict[str, Any]] = []
        for r in self.runs:
            va = r.metrics.get(metric_a)
            vb = r.metrics.get(metric_b)
            if _is_number(va) and _is_number(vb):
                candidates.append({
                    "run": r,
                    "name": r.name,
                    metric_a: float(va),
                    metric_b: float(vb),
                    "_a": float(va),
                    "_b": float(vb),
                })
        if not candidates:
            return []

        def dominates(x: Dict[str, Any], y: Dict[str, Any]) -> bool:
            """True if x dominates y (>= in both, > in at least one)."""
            ax, bx = x["_a"], x["_b"]
            ay, by = y["_a"], y["_b"]
            # x is at least as good as y in both metrics:
            a_ok = (ax >= ay) if a_higher else (ax <= ay)
            b_ok = (bx >= by) if b_higher else (bx <= by)
            if not (a_ok and b_ok):
                return False
            # x is strictly better in at least one:
            a_strict = (ax > ay) if a_higher else (ax < ay)
            b_strict = (bx > by) if b_higher else (bx < by)
            return a_strict or b_strict

        pareto: List[Dict[str, Any]] = []
        for c in candidates:
            if not any(dominates(other, c) for other in candidates if other is not c):
                # Strip private keys
                pareto.append({k: v for k, v in c.items() if not k.startswith("_")})
        return pareto

    # -- repr ---------------------------------------------------------
    def __repr__(self) -> str:
        return f"<ExperimentComparison n_runs={len(self.runs)}>"


# =====================================================================
# ComparisonVisualizer
# =====================================================================
class ComparisonVisualizer:
    """Renders an :class:`ExperimentComparison` as markdown, HTML, or SVG.

    All rendering is dependency-free: SVGs are emitted as strings, the HTML
    report inlines the SVGs and uses inline ``<style>`` (no external CSS or
    JS).
    """

    # -- markdown -----------------------------------------------------
    def to_markdown(self, comparison: ExperimentComparison) -> str:
        """Full markdown report: tables + per-metric best-run analysis."""
        runs = comparison.runs
        lines: List[str] = ["# Experiment Comparison Report", ""]
        lines.append(f"**{len(runs)} run(s) compared.**")
        lines.append("")

        # Per-run one-line summaries
        lines.append("## Runs")
        lines.append("")
        for r in runs:
            lines.append(f"- {r.summary()}")
        lines.append("")

        # Metric table
        lines.append("## Metrics")
        lines.append("")
        lines.append(comparison.metric_table())
        lines.append("")

        # Param table
        if comparison.all_params():
            lines.append("## Parameters")
            lines.append("")
            lines.append(comparison.param_table())
            lines.append("")

        # Per-metric best run
        lines.append("## Best Run Per Metric")
        lines.append("")
        if not runs:
            lines.append("_(no runs)_")
        else:
            for m in comparison.all_metrics():
                hb = _higher_is_better(m)
                try:
                    b = comparison.best(m, higher_is_better=hb)
                    arrow = "↑" if hb else "↓"
                    lines.append(f"- **{m}** {arrow}: `{b.name}` "
                                 f"({_fmt_metric(b.metrics.get(m))})")
                except ValueError:
                    lines.append(f"- **{m}**: _(no numeric values)_")
        lines.append("")

        # Pareto hint for any pair of metrics where one is higher-is-better
        # and the other is lower-is-better (the interesting tradeoff case).
        metrics = comparison.all_metrics()
        if len(metrics) >= 2:
            trade_pairs: List[Tuple[str, str]] = []
            for i, ma in enumerate(metrics):
                for mb in metrics[i + 1:]:
                    if _higher_is_better(ma) != _higher_is_better(mb):
                        trade_pairs.append((ma, mb))
            if trade_pairs:
                lines.append("## Pareto-Optimal Tradeoffs")
                lines.append("")
                for ma, mb in trade_pairs:
                    front = comparison.tradeoffs(ma, mb)
                    if front:
                        names = [e["name"] for e in front]
                        lines.append(f"- **{ma}** vs **{mb}**: "
                                     f"{', '.join('`' + n + '`' for n in names)}")
                lines.append("")

        return "\n".join(lines)

    # -- dict ---------------------------------------------------------
    def to_dict(self, comparison: ExperimentComparison) -> Dict[str, Any]:
        """JSON-serializable representation of the comparison."""
        return {
            "n_runs": len(comparison.runs),
            "runs": [r.to_dict() for r in comparison.runs],
            "all_metrics": comparison.all_metrics(),
            "all_params": comparison.all_params(),
            "best_per_metric": self._best_per_metric(comparison),
        }

    @staticmethod
    def _best_per_metric(comparison: ExperimentComparison) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for m in comparison.all_metrics():
            hb = _higher_is_better(m)
            try:
                b = comparison.best(m, higher_is_better=hb)
                out[m] = {
                    "name": b.name,
                    "value": b.metrics.get(m),
                    "higher_is_better": hb,
                }
            except ValueError:
                out[m] = None
        return out

    # -- SVG: bar chart ----------------------------------------------
    def bar_chart(self, comparison: ExperimentComparison,
                  metric: str) -> str:
        """SVG bar chart for one metric across all runs."""
        runs = comparison.runs
        pairs: List[Tuple[str, Optional[float]]] = []
        for r in runs:
            v = r.metrics.get(metric)
            pairs.append((r.name, float(v) if _is_number(v) else None))

        width = 600
        height = 360
        margin_l, margin_r = 70, 30
        margin_t, margin_b = 50, 80
        plot_w = width - margin_l - margin_r
        plot_h = height - margin_t - margin_b

        valid = [v for _n, v in pairs if v is not None]
        if not valid:
            return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                    f'height="{height}"><rect width="{width}" height="{height}" '
                    f'fill="white"/><text x="{width/2}" y="{height/2}" '
                    f'text-anchor="middle" font-family="sans-serif" '
                    f'fill="gray">No data for {metric}</text></svg>')

        vmin = min(min(valid), 0.0)
        vmax = max(max(valid), 0.0)
        if vmax == vmin:
            vmax = vmin + 1.0

        n = len(pairs)
        bar_w = max(8, min(60, plot_w / max(n, 1) * 0.6))
        gap = (plot_w - n * bar_w) / (n + 1)

        parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text x="{width/2}" y="22" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" font-weight="bold">'
            f'{_html_escape(metric)} (n={n})</text>',
        ]

        # Y axis + ticks
        x_axis_y = height - margin_b
        for i in range(5):
            t = vmin + (vmax - vmin) * (4 - i) / 4
            y_pos = margin_t + i * plot_h / 4
            parts.append(f'<line x1="{margin_l-5}" y1="{y_pos:.1f}" '
                         f'x2="{margin_l}" y2="{y_pos:.1f}" stroke="black"/>')
            parts.append(f'<text x="{margin_l-9}" y="{y_pos+4:.1f}" '
                         f'text-anchor="end" font-family="sans-serif" '
                         f'font-size="10">{t:.3g}</text>')
        parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" '
                     f'y2="{x_axis_y}" stroke="black"/>')
        parts.append(f'<line x1="{margin_l}" y1="{x_axis_y}" '
                     f'x2="{margin_l + plot_w}" y2="{x_axis_y}" stroke="black"/>')

        # Bars
        x = margin_l + gap
        for i, (name, v) in enumerate(pairs):
            color = _color(i)
            if v is None:
                # Placeholder (gray, zero height)
                parts.append(f'<rect x="{x:.1f}" y="{x_axis_y-3:.1f}" '
                             f'width="{bar_w:.1f}" height="3" fill="gray" '
                             f'opacity="0.4"/>')
            else:
                y_top = margin_t + (vmax - v) / (vmax - vmin) * plot_h
                bar_h = x_axis_y - y_top
                parts.append(f'<rect x="{x:.1f}" y="{y_top:.1f}" '
                             f'width="{bar_w:.1f}" height="{bar_h:.1f}" '
                             f'fill="{color}"/>')
                parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y_top-5:.1f}" '
                             f'text-anchor="middle" font-family="sans-serif" '
                             f'font-size="10">{v:.3g}</text>')
            # X-axis label (rotated)
            label_x = x + bar_w / 2
            label_y = x_axis_y + 12
            parts.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" '
                         f'text-anchor="end" font-family="sans-serif" '
                         f'font-size="10" '
                         f'transform="rotate(-30 {label_x:.1f} {label_y:.1f})">'
                         f'{_html_escape(name)}</text>')
            x += bar_w + gap

        parts.append(f'<text x="{margin_l + plot_w/2:.1f}" y="{height-5}" '
                     f'text-anchor="middle" font-family="sans-serif" '
                     f'font-size="10" fill="gray">{_html_escape(metric)}</text>')
        parts.append("</svg>")
        return "\n".join(parts)

    # -- SVG: radar chart --------------------------------------------
    def radar_chart(self, comparison: ExperimentComparison,
                    metrics: Sequence[str]) -> str:
        """SVG radar chart comparing runs across the given metrics.

        Each metric is min-max normalized to ``[0, 1]`` across runs so that
        runs can be compared on a common radar. Metrics where every run has
        the same value are drawn at the outer ring for all runs.
        """
        runs = comparison.runs
        metric_list = [m for m in metrics if m]
        if not runs or not metric_list:
            return ('<svg xmlns="http://www.w3.org/2000/svg" width="500" '
                    'height="450"><rect width="500" height="450" fill="white"/>'
                    '<text x="250" y="225" text-anchor="middle" '
                    'font-family="sans-serif" fill="gray">'
                    'No data for radar chart</text></svg>')

        # Normalize per metric
        norm: Dict[str, List[Optional[float]]] = {}
        for m in metric_list:
            vals: List[Optional[float]] = []
            for r in runs:
                v = r.metrics.get(m)
                vals.append(float(v) if _is_number(v) else None)
            valid = [v for v in vals if v is not None]
            if not valid:
                norm[m] = [None] * len(vals)
                continue
            vmin, vmax = min(valid), max(valid)
            if vmax == vmin:
                norm[m] = [0.5 if v is not None else None for v in vals]
            else:
                # For lower-is-better metrics, invert so that higher normalized
                # value always means "better" (so the radar reads intuitively).
                invert = not _higher_is_better(m)
                norm[m] = []
                for v in vals:
                    if v is None:
                        norm[m].append(None)
                    else:
                        x = (v - vmin) / (vmax - vmin)
                        norm[m].append(1.0 - x if invert else x)

        # Geometry
        width = 560
        height = 480
        cx, cy = width / 2, height / 2 + 10
        radius = 160
        n_metrics = len(metric_list)
        # Start at top (-pi/2), go clockwise
        angles = [(-math.pi / 2 + 2 * math.pi * i / n_metrics)
                  for i in range(n_metrics)]

        parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text x="{width/2}" y="22" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" font-weight="bold">'
            f'Radar Chart</text>',
        ]

        # Grid (concentric polygons)
        for level in (0.2, 0.4, 0.6, 0.8, 1.0):
            pts = []
            for a in angles:
                x = cx + radius * level * math.cos(a)
                y = cy + radius * level * math.sin(a)
                pts.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polygon points="{" ".join(pts)}" fill="none" '
                         f'stroke="#dddddd" stroke-width="1"/>')

        # Axes + labels
        for i, m in enumerate(metric_list):
            a = angles[i]
            x_end = cx + radius * math.cos(a)
            y_end = cy + radius * math.sin(a)
            parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x_end:.1f}" '
                         f'y2="{y_end:.1f}" stroke="#bbbbbb"/>')
            lx = cx + (radius + 18) * math.cos(a)
            ly = cy + (radius + 18) * math.sin(a)
            parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                         f'font-family="sans-serif" font-size="11" '
                         f'dominant-baseline="middle">{_html_escape(m)}</text>')

        # One polygon per run
        for ri, r in enumerate(runs):
            color = _color(ri)
            pts: List[str] = []
            for mi, m in enumerate(metric_list):
                n = norm[m][ri]
                if n is None:
                    # Use center to indicate missing data
                    pts.append(f"{cx:.1f},{cy:.1f}")
                else:
                    a = angles[mi]
                    x = cx + radius * n * math.cos(a)
                    y = cy + radius * n * math.sin(a)
                    pts.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polygon points="{" ".join(pts)}" '
                         f'fill="{color}" fill-opacity="0.15" '
                         f'stroke="{color}" stroke-width="2"/>')
            # Vertices
            for mi, m in enumerate(metric_list):
                n = norm[m][ri]
                if n is None:
                    continue
                a = angles[mi]
                x = cx + radius * n * math.cos(a)
                y = cy + radius * n * math.sin(a)
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" '
                             f'fill="{color}"/>')

        # Legend (top-right)
        legend_x = width - 130
        legend_y = 40
        parts.append(f'<rect x="{legend_x-8}" y="{legend_y-15}" width="120" '
                     f'height="{15 + 18 * len(runs) + 5}" fill="white" '
                     f'stroke="#cccccc" rx="3"/>')
        for ri, r in enumerate(runs):
            color = _color(ri)
            ly = legend_y + ri * 18
            parts.append(f'<rect x="{legend_x}" y="{ly}" width="12" '
                         f'height="12" fill="{color}"/>')
            parts.append(f'<text x="{legend_x + 18}" y="{ly + 10}" '
                         f'font-family="sans-serif" font-size="11">'
                         f'{_html_escape(r.name)}</text>')

        parts.append("</svg>")
        return "\n".join(parts)

    # -- HTML ---------------------------------------------------------
    def to_html(self, comparison: ExperimentComparison,
                title: str = "Experiment Comparison") -> str:
        """Self-contained HTML report with inline SVG charts."""
        runs = comparison.runs
        metrics = comparison.all_metrics()

        # Charts
        radar_svg = self.radar_chart(comparison, metrics) if metrics else ""
        bar_svgs = "\n".join(
            f'<div class="chart"><h3>{_html_escape(m)}</h3>'
            f'{self.bar_chart(comparison, m)}</div>'
            for m in metrics
        )

        # Best-per-metric list
        best_items: List[str] = []
        for m in metrics:
            hb = _higher_is_better(m)
            try:
                b = comparison.best(m, higher_is_better=hb)
                arrow = "&uarr;" if hb else "&darr;"
                best_items.append(
                    f"<li><b>{_html_escape(m)}</b> {arrow}: "
                    f"<code>{_html_escape(b.name)}</code> "
                    f"({_fmt_metric(b.metrics.get(m))})</li>"
                )
            except ValueError:
                best_items.append(
                    f"<li><b>{_html_escape(m)}</b>: <i>no numeric values</i></li>"
                )
        best_html = "\n".join(best_items) if best_items else "<p><i>No metrics.</i></p>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{_html_escape(title)}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    max-width: 1200px;
    margin: 24px auto;
    padding: 0 24px;
    color: #222;
    background: #fafafa;
  }}
  h1, h2, h3 {{ color: #1a1a1a; }}
  h1 {{ border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #f9f9f9; }}
  .chart {{
    background: white;
    padding: 16px;
    margin: 16px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    overflow-x: auto;
  }}
  .chart h3 {{ margin-top: 0; }}
  ul {{ line-height: 1.6; }}
  .summary {{ background: white; padding: 12px 16px; margin: 16px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-left: 4px solid #1f77b4; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>{_html_escape(title)}</h1>
<div class="summary">{len(runs)} run(s) compared
across {len(metrics)} metric(s).</div>

<h2>Runs</h2>
<ul>
{chr(10).join(f'<li><code>{_html_escape(r.name)}</code> &mdash; {_html_escape(r.summary())}</li>' for r in runs)}
</ul>

<h2>Metrics Table</h2>
{_md_table_to_html(comparison.metric_table())}

<h2>Parameters Table</h2>
{_md_table_to_html(comparison.param_table()) if comparison.all_params() else '<p><i>No parameters.</i></p>'}

<h2>Best Run Per Metric</h2>
<ul>
{best_html}
</ul>

<h2>Radar Chart</h2>
<div class="chart">
{radar_svg if radar_svg else '<p><i>No metrics to plot.</i></p>'}
</div>

<h2>Bar Charts</h2>
{bar_svgs if bar_svgs else '<p><i>No metrics to plot.</i></p>'}

<hr>
<p style="color:#888;font-size:12px;">Generated by
<code>qai.experiment_compare.ComparisonVisualizer</code>.</p>
</body>
</html>"""
        return html


# =====================================================================
# RunCollector
# =====================================================================
class RunCollector:
    """Auto-collects :class:`ExperimentRun` objects from a tracker or storage.

    Parameters
    ----------
    tracker : object, optional
        Anything that exposes ``.runs`` (a list of dicts with at least
        ``params``, ``metrics``, ``timestamp`` -- e.g.
        ``qai.tracking.ExperimentTracker``), or ``get_all_runs()`` returning
        either dicts or ``qai.experiment_tracking.ExperimentRun`` objects
        (e.g. ``qai.ExperimentLogger``).
    storage_path : str or Path, optional
        Either a directory of ``*.json`` files (one per run, as written by
        ``JSONLogger``) or a single JSON file (a dict or a list of run dicts
        or a dict with a ``"runs"`` key).
    """

    def __init__(self, tracker: Any = None,
                 storage_path: Optional[Union[str, Path]] = None):
        self.tracker = tracker
        self.storage_path = storage_path

    # -- sources ------------------------------------------------------
    def _all_runs(self) -> List[ExperimentRun]:
        out: List[ExperimentRun] = []
        if self.tracker is not None:
            out.extend(self._from_tracker(self.tracker))
        if self.storage_path is not None:
            out.extend(self._from_storage(self.storage_path))
        return out

    @staticmethod
    def _from_tracker(tracker: Any) -> List[ExperimentRun]:
        runs: List[ExperimentRun] = []

        # Case 1: tracker has a `.runs` list of dicts
        # (qai.tracking.ExperimentTracker)
        if hasattr(tracker, "runs") and isinstance(tracker.runs, list):
            for i, r in enumerate(tracker.runs):
                if isinstance(r, dict):
                    name = r.get("name") or r.get("run_id") or f"run_{i}"
                    runs.append(ExperimentRun(
                        name=str(name),
                        metrics=r.get("metrics") or {},
                        params=r.get("params") or {},
                        dataset=r.get("dataset"),
                        timestamp=r.get("timestamp"),
                    ))
                elif isinstance(r, ExperimentRun):
                    runs.append(r)

        # Case 2: tracker has get_all_runs() (qai.ExperimentLogger / JSONLogger)
        if hasattr(tracker, "get_all_runs"):
            try:
                raw = tracker.get_all_runs() or []
            except Exception:
                raw = []
            for i, r in enumerate(raw):
                if isinstance(r, ExperimentRun):
                    runs.append(r)
                elif isinstance(r, dict):
                    name = r.get("name") or r.get("run_id") or f"run_{i}"
                    runs.append(ExperimentRun(
                        name=str(name),
                        metrics=r.get("metrics") or {},
                        params=r.get("params") or {},
                        dataset=r.get("dataset"),
                        timestamp=r.get("timestamp"),
                    ))
                else:
                    # experiment_tracking.ExperimentRun dataclass
                    name = getattr(r, "name", None) or \
                           getattr(r, "run_id", f"run_{i}")
                    runs.append(ExperimentRun(
                        name=str(name),
                        metrics=dict(getattr(r, "metrics", {}) or {}),
                        params=dict(getattr(r, "params", {}) or {}),
                        dataset=getattr(r, "dataset", None),
                        timestamp=getattr(r, "timestamp", None),
                    ))
        return runs

    @staticmethod
    def _from_storage(path: Union[str, Path]) -> List[ExperimentRun]:
        p = Path(path)
        runs: List[ExperimentRun] = []
        if not p.exists():
            return runs
        if p.is_dir():
            for f in sorted(p.glob("*.json")):
                if f.name == "index.json":
                    continue
                try:
                    with open(f) as fh:
                        d = json.load(fh)
                    runs.append(ExperimentRun.from_dict(d))
                except Exception:
                    continue
        elif p.is_file():
            try:
                with open(p) as fh:
                    d = json.load(fh)
                if isinstance(d, dict) and "runs" in d and isinstance(d["runs"], list):
                    for r in d["runs"]:
                        if isinstance(r, dict):
                            runs.append(ExperimentRun.from_dict(r))
                elif isinstance(d, list):
                    for r in d:
                        if isinstance(r, dict):
                            runs.append(ExperimentRun.from_dict(r))
                elif isinstance(d, dict):
                    runs.append(ExperimentRun.from_dict(d))
            except Exception:
                return runs
        return runs

    # -- public API ---------------------------------------------------
    def collect(self, n: int = 10) -> List[ExperimentRun]:
        """Get the last ``n`` runs, sorted by timestamp descending.

        If ``n <= 0``, returns all runs.
        """
        all_runs = self._all_runs()
        all_runs.sort(key=lambda r: r.timestamp or 0.0, reverse=True)
        if n and n > 0:
            return all_runs[:n]
        return all_runs

    def filter_by_metric(self, metric: str, op: str,
                         value: float) -> List[ExperimentRun]:
        """Filter runs by a metric comparison.

        Parameters
        ----------
        metric : str
            Metric name to compare.
        op : str
            One of ``'>', '<', '>=', '<=', '==', '!='``.
        value : float
            RHS of the comparison.
        """
        ops: Dict[str, Callable[[float, float], bool]] = {
            ">":  lambda a, b: a > b,
            "<":  lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        if op not in ops:
            raise ValueError(
                f"[WHAT] Unknown operator {op!r}\n"
                f"[FIX] Use one of {list(ops.keys())}"
            )
        out: List[ExperimentRun] = []
        for r in self._all_runs():
            v = r.metrics.get(metric)
            if _is_number(v):
                try:
                    if ops[op](float(v), float(value)):
                        out.append(r)
                except Exception:
                    continue
        return out

    def filter_by_param(self, param: str, value: Any) -> List[ExperimentRun]:
        """Filter runs where ``params[param] == value``."""
        out: List[ExperimentRun] = []
        for r in self._all_runs():
            if param in r.params and r.params[param] == value:
                out.append(r)
        return out


# =====================================================================
# Top-level convenience functions
# =====================================================================
def compare(models_or_runs: Sequence[Any],
            X: Any = None,
            y: Any = None,
            metrics: Optional[Sequence[str]] = None) -> ExperimentComparison:
    """Quick comparison of multiple runs or models.

    If ``models_or_runs`` is a list of :class:`ExperimentRun` objects, they
    are wrapped directly in an :class:`ExperimentComparison`.

    Otherwise the items are treated as models (or ``(name, model)`` tuples)
    and ``model.predict(X)`` is called for each, computing accuracy against
    ``y`` (if provided) and assembling metrics from the model's
    ``.metrics_``/``.scores_`` attributes if present.

    Parameters
    ----------
    models_or_runs : sequence
        Either :class:`ExperimentRun` objects or model objects / tuples.
    X : array-like, optional
        Test inputs (required when comparing raw models).
    y : array-like, optional
        Test targets.
    metrics : sequence of str, optional
        Reserved for future use; currently ignored.
    """
    items = list(models_or_runs or [])
    if not items:
        return ExperimentComparison([])

    # Case A: already ExperimentRun objects
    if all(isinstance(x, ExperimentRun) for x in items):
        return ExperimentComparison(items)

    # Case B: models (or (name, model) tuples)
    if X is None:
        raise ValueError(
            "[WHAT] compare() needs X to evaluate raw models\n"
            "[FIX] Pass X=test_inputs, or pass ExperimentRun objects instead"
        )
    runs: List[ExperimentRun] = []
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            name, model = item
        else:
            name = type(item).__name__
            model = item
        metrics_dict: Dict[str, Any] = {}
        try:
            preds = np.asarray(model.predict(X)).flatten()
            if y is not None:
                y_arr = np.asarray(y).flatten()
                n = min(len(preds), len(y_arr))
                if n > 0:
                    metrics_dict["accuracy"] = float(
                        np.mean(preds[:n] == y_arr[:n])
                    )
        except Exception as e:
            metrics_dict["evaluate_error"] = str(e)
        # Pick up any pre-computed scores on the model
        for attr in ("metrics_", "scores_", "eval_metrics"):
            v = getattr(model, attr, None)
            if isinstance(v, dict):
                metrics_dict.update(v)
                break
        runs.append(ExperimentRun(name=name, model=model, metrics=metrics_dict))
    return ExperimentComparison(runs)


def from_tracker(tracker: Any, n: int = 10) -> ExperimentComparison:
    """Build an :class:`ExperimentComparison` from the last ``n`` tracker runs."""
    collector = RunCollector(tracker=tracker)
    return ExperimentComparison(collector.collect(n=n))


def leaderboard(runs: Sequence[ExperimentRun],
                metric: str = "accuracy",
                higher_is_better: bool = True) -> str:
    """Render a markdown leaderboard of ``runs`` ranked by ``metric``."""
    comp = ExperimentComparison(runs)
    ranked = comp.rank(metric=metric, higher_is_better=higher_is_better)
    arrow = "↑" if higher_is_better else "↓"
    lines = [
        f"# Leaderboard — `{metric}` {arrow}",
        "",
        f"_{len(ranked)} of {len(runs)} run(s) have a numeric value for "
        f"`{metric}`._",
        "",
        "| Rank | Run | " + metric + " |",
        "| --- | --- | --- |",
    ]
    for rank, run in ranked:
        v = run.metrics.get(metric)
        lines.append(f"| {rank} | {run.name} | {_fmt_metric(v)} |")
    if not ranked:
        lines.append("| _none_ | _no runs qualify_ | — |")
    return "\n".join(lines)


def pareto_front(runs: Sequence[ExperimentRun],
                 metric_a: str,
                 metric_b: str) -> List[ExperimentRun]:
    """Return the Pareto-optimal runs w.r.t. two metrics.

    Whether higher is better is inferred from the metric name (see
    :func:`_higher_is_better`).
    """
    comp = ExperimentComparison(runs)
    return [entry["run"] for entry in comp.tradeoffs(metric_a, metric_b)]


# =====================================================================
# Legacy classes (kept for backward compatibility with qai/__init__.py)
# =====================================================================
@dataclass
class ComparisonRow:
    """A single row in a (legacy) comparison report."""
    name: str
    technique: str
    metric_value: Optional[float]
    metric_name: str
    latency_ms: Optional[float] = None
    n_params: Optional[int] = None
    model_size_mb: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """A formatted (legacy) comparison of multiple experiments.

    Attributes:
        metric_name: the metric used for ranking ("accuracy", "mse", etc.)
        rows: list of ComparisonRow, sorted by metric descending (or
              ascending for error metrics like MSE).
        best_name: name of the winning experiment.
    """
    metric_name: str
    rows: List[ComparisonRow]
    best_name: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "best_name": self.best_name,
            "rows": [
                {
                    "name": r.name, "technique": r.technique,
                    "metric_value": r.metric_value,
                    "latency_ms": r.latency_ms,
                    "n_params": r.n_params,
                    "model_size_mb": r.model_size_mb,
                }
                for r in self.rows
            ],
        }

    def __str__(self) -> str:
        lines = [
            f"=== Experiment Comparison ({self.metric_name}) ===",
            f"best: {self.best_name}",
            "",
            (f"{'name':20s} {'technique':15s} {self.metric_name:>12s} "
             f"{'latency_ms':>12s} {'n_params':>12s} {'size_mb':>10s}"),
            "-" * 90,
        ]
        for r in self.rows:
            mv = f"{r.metric_value:.4f}" if r.metric_value is not None else "?"
            lat = f"{r.latency_ms:.3f}" if r.latency_ms is not None else "?"
            np_str = f"{r.n_params:,}" if r.n_params is not None else "?"
            sz = f"{r.model_size_mb:.3f}" if r.model_size_mb is not None else "?"
            lines.append(f"{r.name:20s} {r.technique:15s} {mv:>12s} "
                         f"{lat:>12s} {np_str:>12s} {sz:>10s}")
        return "\n".join(lines)


class ExperimentCompare:
    """Compare multiple models on the same test set (legacy).

    Args:
        models: dict of {name: model} to compare. Names are used in the
                comparison report.
    """
    def __init__(self, models: Dict[str, Any]):
        if not models:
            raise ValueError("[WHAT] ExperimentCompare needs at least one model\n"
                             "[FIX] Pass a non-empty dict of {name: model}")
        self.models = models

    def evaluate(self, X, y=None, metric: str = "accuracy",
                 higher_is_better: bool = True) -> ComparisonReport:
        """Run `evaluate()` on each model and produce a comparison."""
        rows: List[ComparisonRow] = []
        for name, model in self.models.items():
            tech = getattr(model, "technique", None)
            technique_name = (getattr(tech, "name", type(model).__name__)
                               if tech else type(model).__name__)
            try:
                if evaluate is not None:
                    report: EvaluationReport = evaluate(model, X, y, verbose=False)
                    metric_value = report.metrics.get(metric)
                    if metric_value is None:
                        for k, v in report.metrics.items():
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                metric_value = float(v)
                                metric = k
                                break
                    notes = report.notes
                else:
                    preds = np.asarray(model.predict(X)).flatten()
                    y_arr = np.asarray(y).flatten()
                    metric_value = float(np.mean(preds == y_arr))
                    notes = []
            except Exception as e:
                metric_value = None
                notes = [f"evaluate failed: {e}"]
            rows.append(ComparisonRow(
                name=name, technique=technique_name,
                metric_value=metric_value, metric_name=metric,
                notes=notes,
            ))
        rows.sort(key=lambda r: r.metric_value if r.metric_value is not None
                  else float("-inf"), reverse=higher_is_better)
        best_name = rows[0].name if rows and rows[0].metric_value is not None else None
        return ComparisonReport(metric_name=metric, rows=rows, best_name=best_name)

    def profile(self, X_sample) -> List[ComparisonRow]:
        """Profile each model's latency / params / size."""
        if Profiler is None:
            return [ComparisonRow(name=n, technique="?",
                                   metric_value=None, metric_name="profile")
                    for n in self.models]
        rows = []
        for name, model in self.models.items():
            try:
                p: Profile = Profiler(model).profile(X_sample)
                tech = getattr(model, "technique", None)
                technique_name = (getattr(tech, "name", type(model).__name__)
                                   if tech else type(model).__name__)
                rows.append(ComparisonRow(
                    name=name, technique=technique_name,
                    metric_value=None, metric_name="profile",
                    latency_ms=p.avg_latency_ms,
                    n_params=p.n_params,
                    model_size_mb=p.model_size_mb,
                    notes=p.notes,
                ))
            except Exception as e:
                rows.append(ComparisonRow(
                    name=name, technique="?",
                    metric_value=None, metric_name="profile",
                    notes=[f"profile failed: {e}"],
                ))
        rows.sort(key=lambda r: r.latency_ms if r.latency_ms is not None
                  else float("inf"))
        return rows

    def full_comparison(self, X, y=None, X_profile_sample=None,
                        metric: str = "accuracy",
                        higher_is_better: bool = True) -> ComparisonReport:
        """Run both evaluate() and profile(), merging the results."""
        eval_report = self.evaluate(X, y, metric=metric,
                                     higher_is_better=higher_is_better)
        if X_profile_sample is None:
            X_profile_sample = X[:1] if hasattr(X, "__getitem__") else X
        profile_rows = self.profile(X_profile_sample)
        profile_by_name = {r.name: r for r in profile_rows}
        for row in eval_report.rows:
            pr = profile_by_name.get(row.name)
            if pr is not None:
                row.latency_ms = pr.latency_ms
                row.n_params = pr.n_params
                row.model_size_mb = pr.model_size_mb
                row.notes.extend(pr.notes)
        return eval_report


__all__ = [
    # New API
    "ExperimentRun",
    "ExperimentComparison",
    "ComparisonVisualizer",
    "RunCollector",
    "compare",
    "from_tracker",
    "leaderboard",
    "pareto_front",
    # Legacy (backward compat)
    "ExperimentCompare",
    "ComparisonReport",
    "ComparisonRow",
]
