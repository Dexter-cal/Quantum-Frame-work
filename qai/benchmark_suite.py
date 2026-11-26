"""
qai.benchmark_suite — comprehensive, offline, synthetic-but-realistic benchmark suite.

This module provides a self-contained, internet-free benchmark suite for
evaluating language/chat models. It ships with small, hand-authored sets of
realistic benchmark items inspired by standard ML evaluation suites:

    - MMLU           (multiple-choice knowledge, 6 subjects x 5 = 30 items)
    - HumanEval      (code generation, 15 problems)
    - GSM8K          (grade-school math word problems, 20 items)
    - GLUE           (SST-2 + MRPC + QQP subsets, 30 items total)
    - CommonsenseQA  (commonsense multiple choice, 20 items)
    - ARC            (AI2 Reasoning Challenge — easy set, 20 items)

Design goals
------------
* **No internet required.** All questions and answers are embedded in this
  module as Python literals.
* **Tolerant evaluation.** A model can answer with a string, a number, a
  class label, or a bare letter (``"A"``, ``"B"``, ``"C"``, ``"D"``).
* **Pluggable model interface.** A model is anything that responds to one of:
  ``model.chat(prompt)`` -> ``model.predict(text)`` -> ``model(text)``.
  This means scikit-learn-style predictors, qai chat models, and plain
  callables all work out of the box.
* **Stdlib + numpy only.** No external dependencies beyond what qai already
  requires.

Public API
----------
Classes:
    Benchmark          — abstract base class
    MMLUBenchmark, HumanEvalBenchmark, GSM8KBenchmark, GLUEBenchmark,
    CommonsenseQABenchmark, ARCBenchmark
    BenchmarkSuite     — runs several benchmarks and aggregates scores
    SuiteReport        — result container for a BenchmarkSuite run
    Leaderboard        — ranking of models across the suite

Functions:
    run_benchmark(name, model, n=None) -> dict
    run_full_suite(model, benchmarks=None) -> SuiteReport
    compare_models(models, benchmarks=None) -> Leaderboard
    list_benchmarks() -> list[str]
    benchmark_info(name) -> dict

Usage
-----
    from qai.benchmark_suite import run_full_suite, compare_models

    class MyModel:
        def chat(self, prompt): return "A"
        def predict(self, x):   return "A"

    report = run_full_suite(MyModel())
    print(report.summary())

    board = compare_models([("model_a", MyModel()), ("model_b", OtherModel())])
    print(board.to_markdown())
"""
from __future__ import annotations

import re
import time
import json
import math
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# =====================================================================
# Model invocation helpers
# =====================================================================
def _call_model(model: Any, prompt: str, max_tokens: int = 256) -> str:
    """Call a model with a chat-style prompt and return a string.

    The model interface is intentionally tolerant. We try, in order:
        1. ``model.chat(prompt)``            — chat-style models
        2. ``model.generate(prompt, ...)``    — generative models
        3. ``model.predict(prompt)``          — sklearn-style predictors
        4. ``model(prompt)``                 — plain callables
        5. ``str(model)``                    — last-resort fallback

    Any exception is swallowed and we fall through to the next strategy;
    if everything fails we return an empty string.
    """
    # Strategy 1: chat interface (preferred for chat models)
    try:
        if hasattr(model, "chat"):
            out = model.chat(prompt)
            if out is not None:
                return str(out)
    except Exception:
        pass

    # Strategy 2: generate interface
    try:
        if hasattr(model, "generate"):
            # Try with max_new_tokens kwarg first; fall back to positional.
            try:
                out = model.generate(prompt, max_new_tokens=max_tokens)
            except TypeError:
                try:
                    out = model.generate(prompt, max_tokens)
                except TypeError:
                    out = model.generate(prompt)
            if out is not None:
                return str(out)
    except Exception:
        pass

    # Strategy 3: predict interface
    try:
        if hasattr(model, "predict"):
            out = model.predict(prompt)
            if out is not None:
                return str(out)
    except Exception:
        pass

    # Strategy 4: plain callable
    try:
        out = model(prompt)
        if out is not None:
            return str(out)
    except Exception:
        pass

    # Strategy 5: str(model) — extremely degenerate fallback
    try:
        return str(model)
    except Exception:
        return ""


def _extract_letter(text: str, valid: str = "ABCDEFGH") -> Optional[str]:
    """Extract a multiple-choice letter from a model response.

    Looks for patterns like ``"A"``, ``"Answer: B"``, ``"The answer is C."``
    Returns the uppercase letter or ``None`` if nothing matched.
    """
    if not text:
        return None
    t = str(text).strip().upper()
    if not t:
        return None
    # A bare single letter answer.
    if len(t) == 1 and t in valid:
        return t
    # Look for explicit "Answer: X" or "answer is X" phrasings first — these
    # are more reliable than the first letter of the response.
    m = re.search(r"(?:ANSWER|ANS|FINAL\s*ANSWER)\s*[:\.\-]?\s*\(?([A-E])\)?", t)
    if m:
        return m.group(1)
    m = re.search(r"(?:ANSWER\s+IS|THE\s+ANSWER\s+IS|OPTION)\s*\(?([A-E])\)?", t)
    if m:
        return m.group(1)
    # A leading "(B)" or "[B]".
    m = re.match(r"^[\(\[]\s*([A-E])\s*[\)\]]", t)
    if m:
        return m.group(1)
    # First single-letter token at the start.
    m = re.match(r"^([A-E])\b", t)
    if m:
        return m.group(1)
    # Last resort: any standalone letter A-E in the text.
    m = re.search(r"\b([A-E])\b", t)
    if m:
        return m.group(1)
    return None


def _extract_number(text: str) -> Optional[float]:
    """Extract a numeric answer from a model response.

    Handles commas in thousands, dollar signs, percentages, simple fractions,
    and "Answer: 42" phrasings. Returns ``None`` if no number was found.
    """
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    # Try to find an "answer is X" / "answer: X" / "final answer: X" cue
    # and restrict to the trailing portion after that cue.
    cue_match = re.search(
        r"(?:final\s+answer|answer|result|total|=)\s*[:\-]?\s*([\$]?\s*[\d,\./]+)",
        t, flags=re.IGNORECASE,
    )
    candidate = cue_match.group(1) if cue_match else t
    candidate = candidate.strip()
    # Strip currency / percent symbols and surrounding parentheses.
    candidate = re.sub(r"[\$%]", "", candidate)
    # Try as fraction like "3/4".
    frac = re.search(r"-?\d+\s*/\s*\d+", candidate)
    if frac:
        try:
            num, den = frac.group().split("/")
            den = float(den.strip())
            if den != 0:
                return float(num.strip()) / den
        except Exception:
            pass
    # Remove commas in numbers like "1,234.56".
    candidate = candidate.replace(",", "")
    # Find the last number-like token (most likely the final answer).
    nums = re.findall(r"-?\d+(?:\.\d+)?", candidate)
    if nums:
        try:
            return float(nums[-1])
        except Exception:
            return None
    # Fall back to scanning the whole text.
    nums = re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    if nums:
        try:
            return float(nums[-1])
        except Exception:
            return None
    return None


def _numbers_close(a: Optional[float], b: float, tol: float = 1e-2) -> bool:
    """Compare two numbers with a relative+absolute tolerance."""
    if a is None:
        return False
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= max(tol, tol * abs(b))


def _model_name(model: Any) -> str:
    """Best-effort human-readable name for a model object."""
    try:
        for attr in ("name", "model_name", "__name__"):
            v = getattr(model, attr, None)
            if isinstance(v, str) and v:
                return v
    except Exception:
        pass
    return type(model).__name__


# =====================================================================
# Benchmark base class
# =====================================================================
class Benchmark:
    """Abstract base class for all benchmarks in the suite.

    Subclasses must override:
        - ``name``        : str identifier (e.g. ``"mmlu"``)
        - ``task_type``   : one of "classification", "generation",
                            "multiple_choice", "code", "math", "retrieval"
        - ``_load_items()``: returns a list of item dicts
        - ``evaluate(model, n=None)``: runs the model and returns a dict
    """

    name: str = "benchmark"
    version: str = "1.0.0"
    task_type: str = "classification"
    description: str = ""
    # Optional weight used by BenchmarkSuite when computing a weighted average.
    weight: float = 1.0

    def __init__(self, name: Optional[str] = None, version: Optional[str] = None):
        # Cached items + leaderboard state live on the instance so that
        # repeated evaluate() calls accumulate a leaderboard.
        self._items: Optional[List[Dict[str, Any]]] = None
        self._leaderboard: List[Dict[str, Any]] = []
        if name is not None:
            self.name = name
        if version is not None:
            self.version = version

    # -- core API ------------------------------------------------------
    @property
    def n_samples(self) -> int:
        """Total number of items in the benchmark."""
        return len(self.load())

    def load(self) -> List[Dict[str, Any]]:
        """Return the (cached) list of benchmark items."""
        if self._items is None:
            self._items = self._load_items()
        return self._items

    def _load_items(self) -> List[Dict[str, Any]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def evaluate(self, model: Any, n: Optional[int] = None) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- shared helpers ------------------------------------------------
    def _slice_items(self, n: Optional[int]) -> List[Dict[str, Any]]:
        items = self.load()
        if n is None or n <= 0:
            return items
        return items[:n]

    def leaderboard(self) -> List[Dict[str, Any]]:
        """Return cached scores from previous runs (highest accuracy first)."""
        return sorted(
            list(self._leaderboard),
            key=lambda r: r.get("score", r.get("accuracy", 0.0)),
            reverse=True,
        )

    def summary(self, result: Dict[str, Any]) -> str:
        """Render a one-paragraph human-readable summary of a result dict."""
        name = result.get("benchmark", self.name)
        acc = result.get("accuracy", result.get("pass_at_1", result.get("score", 0.0)))
        n_correct = result.get("n_correct", result.get("n_passed", 0))
        n_total = result.get("n_total", 0)
        lines = [f"[{name}] accuracy={acc:.2%} ({n_correct}/{n_total} correct)"]
        per = result.get("per_subject") or result.get("per_task") or {}
        if per:
            for subj, val in per.items():
                if isinstance(val, dict):
                    s = val.get("accuracy", 0.0)
                    n_c = val.get("n_correct", 0)
                    n_t = val.get("n_total", 0)
                    lines.append(f"  - {subj}: {s:.2%} ({n_c}/{n_t})")
                else:
                    try:
                        lines.append(f"  - {subj}: {float(val):.2%}")
                    except (TypeError, ValueError):
                        lines.append(f"  - {subj}: {val}")
        if result.get("errors"):
            lines.append(f"  ({len(result['errors'])} errors during evaluation)")
        return "\n".join(lines)

    # -- registration helpers -----------------------------------------
    def _record(self, model_name: str, score: float, extras: Optional[Dict[str, Any]] = None) -> None:
        entry: Dict[str, Any] = {
            "model": model_name,
            "benchmark": self.name,
            "score": score,
            "ts": time.time(),
        }
        if extras:
            entry.update(extras)
        self._leaderboard.append(entry)

    def info(self) -> Dict[str, Any]:
        """Return metadata about this benchmark."""
        return {
            "name": self.name,
            "version": self.version,
            "task_type": self.task_type,
            "description": self.description,
            "n_samples": self.n_samples,
            "weight": self.weight,
        }


# =====================================================================
# MMLU — Massive Multitask Language Understanding
# =====================================================================
_MMLU_ITEMS: List[Dict[str, Any]] = [
    # --- elementary_mathematics (5) ---
    {"subject": "elementary_mathematics",
     "question": "What is 7 × 8?",
     "choices": ["54", "56", "58", "64"], "answer": "B"},
    {"subject": "elementary_mathematics",
     "question": "Which fraction is equivalent to 0.75?",
     "choices": ["1/2", "2/3", "3/4", "4/5"], "answer": "C"},
    {"subject": "elementary_mathematics",
     "question": "How many minutes are there in 2 hours and 15 minutes?",
     "choices": ["125", "135", "145", "215"], "answer": "B"},
    {"subject": "elementary_mathematics",
     "question": "What is the next number in the sequence 2, 4, 8, 16, ...?",
     "choices": ["20", "24", "32", "64"], "answer": "C"},
    {"subject": "elementary_mathematics",
     "question": "If a rectangle has length 6 and width 4, what is its perimeter?",
     "choices": ["10", "20", "24", "48"], "answer": "B"},

    # --- high_school_biology (5) ---
    {"subject": "high_school_biology",
     "question": "Which organelle is the site of photosynthesis in plant cells?",
     "choices": ["Mitochondrion", "Chloroplast", "Nucleus", "Ribosome"], "answer": "B"},
    {"subject": "high_school_biology",
     "question": "DNA is replicated in which phase of the cell cycle?",
     "choices": ["G1 phase", "S phase", "G2 phase", "M phase"], "answer": "B"},
    {"subject": "high_school_biology",
     "question": "Which molecule carries oxygen in the human bloodstream?",
     "choices": ["Hemoglobin", "Insulin", "Albumin", "Collagen"], "answer": "A"},
    {"subject": "high_school_biology",
     "question": "What is the basic monomer unit of proteins?",
     "choices": ["Nucleotides", "Amino acids", "Monosaccharides", "Fatty acids"], "answer": "B"},
    {"subject": "high_school_biology",
     "question": "In Mendelian genetics, the observable trait of an organism is called its:",
     "choices": ["Genotype", "Phenotype", "Karyotype", "Allele"], "answer": "B"},

    # --- college_physics (5) ---
    {"subject": "college_physics",
     "question": "The SI unit of electric resistance is the:",
     "choices": ["Volt", "Ampere", "Ohm", "Watt"], "answer": "C"},
    {"subject": "college_physics",
     "question": "Which law states that the current through a conductor is proportional to the voltage across it?",
     "choices": ["Newton's law", "Ohm's law", "Hooke's law", "Coulomb's law"], "answer": "B"},
    {"subject": "college_physics",
     "question": "A photon of which color has the highest energy?",
     "choices": ["Red", "Yellow", "Green", "Violet"], "answer": "D"},
    {"subject": "college_physics",
     "question": "The speed of light in vacuum is approximately:",
     "choices": ["3 × 10^5 m/s", "3 × 10^6 m/s", "3 × 10^8 m/s", "3 × 10^10 m/s"], "answer": "C"},
    {"subject": "college_physics",
     "question": "Which particle has no electric charge?",
     "choices": ["Proton", "Electron", "Neutron", "Positron"], "answer": "C"},

    # --- formal_logic (5) ---
    {"subject": "formal_logic",
     "question": "The contrapositive of 'If P then Q' is:",
     "choices": ["If Q then P", "If not P then not Q", "If not Q then not P", "If Q then not P"], "answer": "C"},
    {"subject": "formal_logic",
     "question": "A statement that is always false is called a:",
     "choices": ["Tautology", "Contradiction", "Contingency", "Paradox"], "answer": "B"},
    {"subject": "formal_logic",
     "question": "Which logical connective is represented by '∧'?",
     "choices": ["OR", "AND", "NOT", "IMPLIES"], "answer": "B"},
    {"subject": "formal_logic",
     "question": "In a truth table, how many rows are needed for 3 propositional variables?",
     "choices": ["3", "6", "8", "9"], "answer": "C"},
    {"subject": "formal_logic",
     "question": "Modus ponens has the form: If P then Q; P; therefore:",
     "choices": ["not P", "Q", "not Q", "P and Q"], "answer": "B"},

    # --- world_religions (5) ---
    {"subject": "world_religions",
     "question": "Which religion's sacred text is the Quran?",
     "choices": ["Hinduism", "Buddhism", "Islam", "Judaism"], "answer": "C"},
    {"subject": "world_religions",
     "question": "The Eightfold Path is central to which religion?",
     "choices": ["Buddhism", "Sikhism", "Jainism", "Taoism"], "answer": "A"},
    {"subject": "world_religions",
     "question": "How many noble truths are there in Buddhism?",
     "choices": ["Two", "Three", "Four", "Eight"], "answer": "C"},
    {"subject": "world_religions",
     "question": "The Torah is most closely associated with which religion?",
     "choices": ["Christianity", "Judaism", "Islam", "Hinduism"], "answer": "B"},
    {"subject": "world_religions",
     "question": "Which religion recognizes the Five Pillars?",
     "choices": ["Hinduism", "Islam", "Buddhism", "Sikhism"], "answer": "B"},

    # --- business_ethics (5) ---
    {"subject": "business_ethics",
     "question": "The 'categorical imperative' was proposed by which philosopher?",
     "choices": ["John Stuart Mill", "Immanuel Kant", "Aristotle", "Jeremy Bentham"], "answer": "B"},
    {"subject": "business_ethics",
     "question": "A conflict of interest is best described as:",
     "choices": ["A legal violation", "A situation where personal interest conflicts with professional duty",
                 "A financial loss", "An HR policy"], "answer": "B"},
    {"subject": "business_ethics",
     "question": "Whistleblowing is most appropriate when:",
     "choices": ["A coworker is rude", "An employee is late",
                 "There is serious wrongdoing the organization will not address",
                 "A colleague disagrees with you"], "answer": "C"},
    {"subject": "business_ethics",
     "question": "Utilitarianism holds that the right action is the one that:",
     "choices": ["Follows the rules", "Maximizes overall happiness",
                 "Respects individual rights above all", "Is most profitable"], "answer": "B"},
    {"subject": "business_ethics",
     "question": "Corporate Social Responsibility (CSR) refers to:",
     "choices": ["Maximizing shareholder returns",
                 "A company's obligation to society beyond profit-making",
                 "Government regulation of business", "Charitable donations only"], "answer": "B"},
]


class MMLUBenchmark(Benchmark):
    """MMLU-style multi-subject multiple-choice benchmark.

    Includes 5 hand-authored questions per subject across 6 subjects
    (30 items total): elementary_mathematics, high_school_biology,
    college_physics, formal_logic, world_religions, business_ethics.
    """

    name = "mmlu"
    version = "1.0.0"
    task_type = "multiple_choice"
    description = "Massive Multitask Language Understanding (synthetic subset, 6 subjects)"
    weight = 1.0

    def _load_items(self) -> List[Dict[str, Any]]:
        return [dict(it) for it in _MMLU_ITEMS]

    def _format_prompt(self, item: Dict[str, Any]) -> str:
        lines = [f"Subject: {item['subject'].replace('_', ' ').title()}",
                 f"Q: {item['question']}", ""]
        for i, choice in enumerate(item["choices"]):
            lines.append(f"  {chr(65 + i)}. {choice}")
        lines.append("")
        lines.append("Answer with a single letter (A, B, C, or D).")
        return "\n".join(lines)

    def evaluate(self, model: Any, n: Optional[int] = None) -> Dict[str, Any]:
        items = self._slice_items(n)
        n_correct = 0
        per_subject: Dict[str, Dict[str, int]] = {}
        samples: List[Dict[str, Any]] = []
        errors: List[str] = []
        for item in items:
            prompt = self._format_prompt(item)
            try:
                response = _call_model(model, prompt, max_tokens=8)
            except Exception as e:
                response = ""
                errors.append(f"{item.get('subject','?')}: {e}")
            pred = _extract_letter(response)
            expected = item["answer"]
            correct = (pred is not None and pred == expected)
            if correct:
                n_correct += 1
            subj = item["subject"]
            bucket = per_subject.setdefault(subj, {"n_correct": 0, "n_total": 0})
            bucket["n_total"] += 1
            if correct:
                bucket["n_correct"] += 1
            samples.append({
                "subject": subj,
                "question": item["question"],
                "expected": expected,
                "predicted": pred,
                "correct": correct,
                "response": response[:80],
            })
        n_total = len(items)
        accuracy = n_correct / n_total if n_total else 0.0
        per_subject_acc = {
            s: {
                "accuracy": v["n_correct"] / v["n_total"] if v["n_total"] else 0.0,
                "n_correct": v["n_correct"],
                "n_total": v["n_total"],
            }
            for s, v in per_subject.items()
        }
        result = {
            "benchmark": self.name,
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_subject": per_subject_acc,
            "samples": samples,
            "errors": errors,
        }
        self._record(_model_name(model), accuracy, {"n_total": n_total})
        return result


# =====================================================================
# HumanEval — code generation
# =====================================================================
_HUMANEVAL_ITEMS: List[Dict[str, Any]] = [
    {"task_id": "HumanEval/0",
     "signature": "def has_close_elements(numbers: list, threshold: float) -> bool:",
     "docstring": "Check if in given list of numbers, are any two numbers closer to each other than threshold.",
     "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than threshold.\n    \"\"\"\n",
     "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True",
     "entry_point": "has_close_elements"},
    {"task_id": "HumanEval/1",
     "signature": "def separate_paren_groups(paren_string: str) -> list:",
     "docstring": "Split the input string into separate balanced parentheses groups, ignoring spaces.",
     "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Split the input string into separate balanced parentheses groups, ignoring spaces.\n    \"\"\"\n",
     "test": "assert separate_paren_groups('(()) ((())) () ()') == ['(())', '((()))', '()', '()']\nassert separate_paren_groups('() () ()') == ['()', '()', '()']\nassert separate_paren_groups('((()))') == ['((()))']",
     "entry_point": "separate_paren_groups"},
    {"task_id": "HumanEval/2",
     "signature": "def truncate_number(number: float) -> float:",
     "docstring": "Return the decimal part of a positive floating point number.",
     "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Return the decimal part of a positive floating point number.\n    \"\"\"\n",
     "test": "assert truncate_number(3.5) == 0.5\nassert truncate_number(1.0) == 0.0\nassert truncate_number(10.25) == 0.25",
     "entry_point": "truncate_number"},
    {"task_id": "HumanEval/3",
     "signature": "def below_zero(operations: list) -> bool:",
     "docstring": "Given a list of deposit (positive) and withdrawal (negative) operations, return True if the balance falls below zero at any point.",
     "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" Given a list of deposit (positive) and withdrawal (negative) operations, return True if the balance falls below zero at any point.\n    \"\"\"\n",
     "test": "assert below_zero([1, 2, 3]) == False\nassert below_zero([1, 2, -4, 5, 6]) == True\nassert below_zero([-1]) == True",
     "entry_point": "below_zero"},
    {"task_id": "HumanEval/4",
     "signature": "def mean_absolute_deviation(numbers: list) -> float:",
     "docstring": "Compute the mean absolute deviation of a list of numbers.",
     "prompt": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" Compute the mean absolute deviation of a list of numbers.\n    \"\"\"\n",
     "test": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-9\nassert abs(mean_absolute_deviation([1.0]) - 0.0) < 1e-9\nassert abs(mean_absolute_deviation([2.0, 2.0, 2.0]) - 0.0) < 1e-9",
     "entry_point": "mean_absolute_deviation"},
    {"task_id": "HumanEval/5",
     "signature": "def intersperse(numbers: list, delimiter: int) -> list:",
     "docstring": "Insert the delimiter between every two consecutive items in the list.",
     "prompt": "from typing import List\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert the delimeter between every two consecutive items in the list.\n    \"\"\"\n",
     "test": "assert intersperse([], 4) == []\nassert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]\nassert intersperse([1], 4) == [1]",
     "entry_point": "intersperse"},
    {"task_id": "HumanEval/6",
     "signature": "def parse_nested_parens(paren_string: str) -> list:",
     "docstring": "Return the maximum nesting depth of each group in the string.",
     "prompt": "from typing import List\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\" Return the maximum nesting depth of each group in the string.\n    \"\"\"\n",
     "test": "assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]\nassert parse_nested_parens('() (())') == [1, 2]\nassert parse_nested_parens('((()))') == [3]",
     "entry_point": "parse_nested_parens"},
    {"task_id": "HumanEval/7",
     "signature": "def filter_by_substring(strings: list, substring: str) -> list:",
     "docstring": "Return only the strings that contain the given substring.",
     "prompt": "from typing import List\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Return only the strings that contain the given substring.\n    \"\"\"\n",
     "test": "assert filter_by_substring([], 'a') == []\nassert filter_by_substring(['abc', 'bac', 'bcde', 'cde'], 'bc') == ['abc', 'bcde']\nassert filter_by_substring(['xxx', 'xXx', 'XXx'], 'x') == ['xxx', 'xXx']",
     "entry_point": "filter_by_substring"},
    {"task_id": "HumanEval/8",
     "signature": "def sum_product(numbers: list) -> tuple:",
     "docstring": "Return a tuple of (sum, product) of the input numbers.",
     "prompt": "from typing import List, Tuple\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n    \"\"\" Return a tuple of (sum, product) of the input numbers.\n    \"\"\"\n",
     "test": "assert sum_product([]) == (0, 1)\nassert sum_product([1, 2, 3, 4]) == (10, 24)\nassert sum_product([5]) == (5, 5)",
     "entry_point": "sum_product"},
    {"task_id": "HumanEval/9",
     "signature": "def rolling_max(numbers: list) -> list:",
     "docstring": "Return the running maximum at each position of the list.",
     "prompt": "from typing import List\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\" Return the running maximum at each position of the list.\n    \"\"\"\n",
     "test": "assert rolling_max([]) == []\nassert rolling_max([1, 2, 3, 4]) == [1, 2, 3, 4]\nassert rolling_max([4, 3, 2, 1]) == [4, 4, 4, 4]\nassert rolling_max([3, 1, 4, 1, 5]) == [3, 3, 4, 4, 5]",
     "entry_point": "rolling_max"},
    {"task_id": "HumanEval/10",
     "signature": "def is_palindrome(string: str) -> bool:",
     "docstring": "Return True if the input string is a palindrome.",
     "prompt": "def is_palindrome(string: str) -> bool:\n    \"\"\" Return True if the input string is a palindrome.\n    \"\"\"\n",
     "test": "assert is_palindrome('') == True\nassert is_palindrome('aba') == True\nassert is_palindrome('abc') == False\nassert is_palindrome('racecar') == True",
     "entry_point": "is_palindrome"},
    {"task_id": "HumanEval/11",
     "signature": "def add(x: int, y: int) -> int:",
     "docstring": "Add two integers and return the result.",
     "prompt": "def add(x: int, y: int) -> int:\n    \"\"\" Add two integers and return the result.\n    \"\"\"\n",
     "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
     "entry_point": "add"},
    {"task_id": "HumanEval/12",
     "signature": "def string_xor(a: str, b: str) -> str:",
     "docstring": "Return the XOR of two equal-length binary strings.",
     "prompt": "def string_xor(a: str, b: str) -> str:\n    \"\"\" Return the XOR of two equal-length binary strings.\n    \"\"\"\n",
     "test": "assert string_xor('010', '110') == '100'\nassert string_xor('111', '111') == '000'\nassert string_xor('0', '1') == '1'",
     "entry_point": "string_xor"},
    {"task_id": "HumanEval/13",
     "signature": "def longest(strings: list) -> str:",
     "docstring": "Return the longest string in a list; return None if the list is empty.",
     "prompt": "from typing import List, Optional\n\ndef longest(strings: List[str]) -> Optional[str]:\n    \"\"\" Return the longest string in a list; return None if the list is empty.\n    \"\"\"\n",
     "test": "assert longest([]) is None\nassert longest(['a', 'bb', 'ccc']) == 'ccc'\nassert longest(['x', 'yy', 'zz']) == 'yy'",
     "entry_point": "longest"},
    {"task_id": "HumanEval/14",
     "signature": "def greatest_common_divisor(a: int, b: int) -> int:",
     "docstring": "Return the greatest common divisor of two integers using Euclid's algorithm.",
     "prompt": "def greatest_common_divisor(a: int, b: int) -> int:\n    \"\"\" Return the greatest common divisor of two integers using Euclid's algorithm.\n    \"\"\"\n",
     "test": "assert greatest_common_divisor(48, 18) == 6\nassert greatest_common_divisor(7, 13) == 1\nassert greatest_common_divisor(100, 10) == 10",
     "entry_point": "greatest_common_divisor"},
]


class HumanEvalBenchmark(Benchmark):
    """HumanEval-style code generation benchmark.

    The model is given a function signature + docstring prompt and asked
    to produce a Python implementation. We score a problem as passed if:

      1. The response contains a function definition matching the entry
         point name AND a non-trivial body, and
      2. The combined (response + test) code executes without assertion
         errors (best effort — if execution fails for environmental
         reasons we still award partial credit for (1)).

    Pass@1 is the fraction of problems fully passing (1)+(2).
    """

    name = "humaneval"
    version = "1.0.0"
    task_type = "code"
    description = "HumanEval-style code generation benchmark (synthetic, 15 problems)"
    weight = 1.0

    def _load_items(self) -> List[Dict[str, Any]]:
        return [dict(it) for it in _HUMANEVAL_ITEMS]

    def _has_function_def(self, code: str, entry_point: str) -> bool:
        """Check that the response defines the target function with a body."""
        if not code or not entry_point:
            return False
        # Find a top-level "def <entry_point>(...):" with at least one
        # non-trivial body line after it.
        pattern = rf"def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->\s*[^:\n]+\s*)?:"
        for match in re.finditer(pattern, code):
            # Check that there's something after the colon (the body).
            tail = code[match.end():]
            # Skip the rest of the def line if continuation.
            body_lines = [ln for ln in tail.split("\n") if ln.strip()]
            if len(body_lines) >= 1:
                # Filter out trivial bodies (only "pass", "return None", "...").
                meaningful = [ln for ln in body_lines
                              if ln.strip() not in {"pass", "...", "return None", "return"}]
                if meaningful:
                    return True
        return False

    def _try_run_tests(self, code: str, item: Dict[str, Any]) -> bool:
        """Attempt to exec the code + tests; return True if all pass."""
        entry = item["entry_point"]
        full = code + "\n" + item["test"]
        env: Dict[str, Any] = {}
        try:
            exec(compile(full, f"<humaneval:{entry}>", "exec"), env)
            return True
        except Exception:
            return False

    def evaluate(self, model: Any, n: Optional[int] = None) -> Dict[str, Any]:
        items = self._slice_items(n)
        n_passed = 0
        n_has_def = 0
        samples: List[Dict[str, Any]] = []
        errors: List[str] = []
        for item in items:
            prompt = (
                item["prompt"]
                + "\n# Complete the function above. Respond with Python code only.\n"
            )
            try:
                response = _call_model(model, prompt, max_tokens=512)
            except Exception as e:
                response = ""
                errors.append(f"{item['task_id']}: {e}")
            has_def = self._has_function_def(response, item["entry_point"])
            if has_def:
                n_has_def += 1
            tests_passed = self._try_run_tests(response, item) if has_def else False
            if tests_passed:
                n_passed += 1
            samples.append({
                "task_id": item["task_id"],
                "entry_point": item["entry_point"],
                "has_function_def": has_def,
                "tests_passed": tests_passed,
                "response_preview": response[:120],
            })
        n_total = len(items)
        accuracy = n_passed / n_total if n_total else 0.0
        result = {
            "benchmark": self.name,
            "accuracy": accuracy,           # pass@1, alias for suite aggregation
            "pass_at_1": accuracy,
            "n_correct": n_passed,
            "n_passed": n_passed,
            "n_total": n_total,
            "n_has_function_def": n_has_def,
            "per_subject": {
                "function_def_rate": n_has_def / n_total if n_total else 0.0,
                "pass_rate": accuracy,
            },
            "samples": samples,
            "errors": errors,
        }
        self._record(_model_name(model), accuracy, {"n_total": n_total})
        return result


# =====================================================================
# GSM8K — grade school math
# =====================================================================
_GSM8K_ITEMS: List[Dict[str, Any]] = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "answer": 18},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": 3},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": 70000},
    {"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "answer": 540},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She has 20 chickens. How many cups of feed does she need total?", "answer": 60},
    {"question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the first. Kylar wants to buy 16 glasses. How much does he need to pay?", "answer": 64},
    {"question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?", "answer": 260},
    {"question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart. How many minutes does the full download take?", "answer": 160},
    {"question": "Marissa is making lemonade. She needs 12 lemons for each pitcher. She has 96 lemons. How many pitchers of lemonade can she make?", "answer": 8},
    {"question": "A store sells pencils in packs of 6. If a student needs 24 pencils, how many packs should they buy?", "answer": 4},
    {"question": "A car drives 60 miles per hour for 2.5 hours. How many miles does it travel?", "answer": 150},
    {"question": "A pizza is cut into 8 equal slices. If 3 friends each eat 2 slices, how many slices remain?", "answer": 2},
    {"question": "Maya's book has 240 pages. She reads 30 pages a day. How many days will it take her to finish?", "answer": 8},
    {"question": "A box contains 144 pens. If they are distributed equally among 12 students, how many pens does each student get?", "answer": 12},
    {"question": "John buys 3 shirts for $20 each and 2 pairs of pants for $35 each. How much does he spend in total?", "answer": 130},
    {"question": "A train travels 240 miles in 4 hours. What is its average speed in miles per hour?", "answer": 60},
    {"question": "There are 5 shelves with 18 books each. How many books are there in total?", "answer": 90},
    {"question": "Sara baked 48 cookies. She gave 1/4 of them to her brother and 1/3 of the remainder to her friend. How many cookies does she have left?", "answer": 24},
    {"question": "A movie is 2 hours and 15 minutes long. If it starts at 7:30 PM, at what time does it end? Express the answer as the hour on a 24-hour clock (e.g. 21).", "answer": 21},
    {"question": "A rectangle has length 12 cm and width 8 cm. What is its area in square centimeters?", "answer": 96},
]


class GSM8KBenchmark(Benchmark):
    """GSM8K-style grade-school math word problems (20 items).

    Each problem has a single numeric answer. The model's response is parsed
    for the last numeric token; answers within a 1% relative tolerance are
    marked correct.
    """

    name = "gsm8k"
    version = "1.0.0"
    task_type = "math"
    description = "Grade School Math 8K (synthetic subset, 20 problems)"
    weight = 1.0

    def _load_items(self) -> List[Dict[str, Any]]:
        return [dict(it) for it in _GSM8K_ITEMS]

    def _format_prompt(self, item: Dict[str, Any]) -> str:
        return (
            f"Q: {item['question']}\n\n"
            "Think step by step, then end with a line of the form "
            "'Answer: <number>'."
        )

    def evaluate(self, model: Any, n: Optional[int] = None) -> Dict[str, Any]:
        items = self._slice_items(n)
        n_correct = 0
        samples: List[Dict[str, Any]] = []
        errors: List[str] = []
        for item in items:
            prompt = self._format_prompt(item)
            try:
                response = _call_model(model, prompt, max_tokens=256)
            except Exception as e:
                response = ""
                errors.append(f"gsm8k: {e}")
            pred = _extract_number(response)
            expected = item["answer"]
            correct = _numbers_close(pred, float(expected), tol=1e-2)
            if correct:
                n_correct += 1
            samples.append({
                "question": item["question"][:100],
                "expected": expected,
                "predicted": pred,
                "correct": correct,
                "response_preview": response[:80],
            })
        n_total = len(items)
        accuracy = n_correct / n_total if n_total else 0.0
        result = {
            "benchmark": self.name,
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_subject": {
                "all": {
                    "accuracy": accuracy,
                    "n_correct": n_correct,
                    "n_total": n_total,
                }
            },
            "samples": samples,
            "errors": errors,
        }
        self._record(_model_name(model), accuracy, {"n_total": n_total})
        return result


# =====================================================================
# GLUE — SST-2, MRPC, QQP subsets
# =====================================================================
_GLUE_ITEMS: List[Dict[str, Any]] = [
    # --- SST-2: binary sentiment (positive=1, negative=0) ---
    {"task": "sst2", "text": "this movie was absolutely wonderful and uplifting", "label": 1},
    {"task": "sst2", "text": "a complete waste of time and money", "label": 0},
    {"task": "sst2", "text": "i loved every minute of this heartwarming film", "label": 1},
    {"task": "sst2", "text": "boring, predictable, and badly acted", "label": 0},
    {"task": "sst2", "text": "the best documentary i have seen all year", "label": 1},
    {"task": "sst2", "text": "avoid this film at all costs", "label": 0},
    {"task": "sst2", "text": "the cinematography was stunning and the script sharp", "label": 1},
    {"task": "sst2", "text": "the plot made no sense and the ending was awful", "label": 0},
    {"task": "sst2", "text": "a delightful and charming romantic comedy", "label": 1},
    {"task": "sst2", "text": "the dullest two hours i have ever spent", "label": 0},

    # --- MRPC: paraphrase (1=is_paraphrase, 0=not) ---
    {"task": "mrpc", "text1": "The company announced a merger yesterday.", "text2": "Yesterday the company said it would merge.", "label": 1},
    {"task": "mrpc", "text1": "He walked to the store to buy milk.", "text2": "She drove to the bank to cash a check.", "label": 0},
    {"task": "mrpc", "text1": "The cat sat on the mat.", "text2": "A cat is sitting on a mat.", "label": 1},
    {"task": "mrpc", "text1": "The stock market crashed in 2008.", "text2": "She loves hiking in the mountains.", "label": 0},
    {"task": "mrpc", "text1": "They are going to the concert tonight.", "text2": "Tonight they will attend the concert.", "label": 1},
    {"task": "mrpc", "text1": "The book is on the table.", "text2": "She is reading a magazine.", "label": 0},
    {"task": "mrpc", "text1": "The recipe calls for two cups of flour.", "text2": "You need two cups of flour for the recipe.", "label": 1},
    {"task": "mrpc", "text1": "It rained heavily last night.", "text2": "The sun is shining brightly today.", "label": 0},
    {"task": "mrpc", "text1": "The team won the championship game.", "text2": "The squad took the title in the final match.", "label": 1},
    {"task": "mrpc", "text1": "He fixed the broken pipe.", "text2": "She planted a tree in the garden.", "label": 0},

    # --- QQP: question equivalence (1=equivalent, 0=not) ---
    {"task": "qqp", "text1": "How do I learn Python?", "text2": "What is the best way to learn Python?", "label": 1},
    {"task": "qqp", "text1": "What is the capital of France?", "text2": "How tall is the Eiffel Tower?", "label": 0},
    {"task": "qqp", "text1": "Why is the sky blue?", "text2": "What causes the sky to appear blue?", "label": 1},
    {"task": "qqp", "text1": "How do I bake a cake?", "text2": "Where can I buy a car?", "label": 0},
    {"task": "qqp", "text1": "What time is it in Tokyo?", "text2": "What is the current time in Tokyo?", "label": 1},
    {"task": "qqp", "text1": "Who wrote Hamlet?", "text2": "What is the boiling point of water?", "label": 0},
    {"task": "qqp", "text1": "Is it going to rain tomorrow?", "text2": "Will it rain tomorrow?", "label": 1},
    {"task": "qqp", "text1": "How much does a Tesla cost?", "text2": "What is the recipe for pancakes?", "label": 0},
    {"task": "qqp", "text1": "What is machine learning?", "text2": "Can you explain machine learning?", "label": 1},
    {"task": "qqp", "text1": "Where is the nearest coffee shop?", "text2": "What is quantum physics?", "label": 0},
]


class GLUEBenchmark(Benchmark):
    """GLUE subset benchmark covering SST-2, MRPC, and QQP.

    Each item is a binary classification task. For MRPC/QQP the model is
    presented with two sentences and asked whether they have the same
    meaning (yes/no → 1/0). For SST-2 the model judges sentiment
    (positive/negative → 1/0). Both numeric and textual ("yes"/"no",
    "positive"/"negative") responses are accepted.
    """

    name = "glue"
    version = "1.0.0"
    task_type = "classification"
    description = "GLUE subset (SST-2, MRPC, QQP) — synthetic items"
    weight = 1.0

    def _load_items(self) -> List[Dict[str, Any]]:
        return [dict(it) for it in _GLUE_ITEMS]

    def _format_prompt(self, item: Dict[str, Any]) -> str:
        if item["task"] == "sst2":
            return (
                f"Sentence: {item['text']}\n\n"
                "Is the sentiment of this sentence positive or negative?\n"
                "Answer with a single word: positive or negative."
            )
        # MRPC and QQP — both ask about semantic equivalence.
        return (
            f"Sentence 1: {item['text1']}\n"
            f"Sentence 2: {item['text2']}\n\n"
            "Do these two sentences have the same meaning?\n"
            "Answer with a single word: yes or no."
        )

    def _parse_response(self, response: str, task: str) -> Optional[int]:
        """Map a free-form response to a 0/1 label, or None if unparseable."""
        if response is None:
            return None
        t = str(response).strip().lower()
        if not t:
            return None
        # Numeric tokens 0/1.
        nums = re.findall(r"\b[01]\b", t)
        if nums:
            return int(nums[0])
        if task == "sst2":
            if "pos" in t:
                return 1
            if "neg" in t:
                return 0
        else:  # MRPC / QQP
            # "yes"/"equivalent"/"same" => 1; "no"/"different"/"not" => 0.
            if re.search(r"\b(yes|equivalent|same|identical)\b", t):
                return 1
            if re.search(r"\b(no|different|not equivalent|not the same)\b", t):
                return 0
            if re.search(r"\b(true)\b", t):
                return 1
            if re.search(r"\b(false)\b", t):
                return 0
        return None

    def evaluate(self, model: Any, n: Optional[int] = None) -> Dict[str, Any]:
        items = self._slice_items(n)
        n_correct = 0
        per_task: Dict[str, Dict[str, int]] = {}
        samples: List[Dict[str, Any]] = []
        errors: List[str] = []
        for item in items:
            prompt = self._format_prompt(item)
            try:
                response = _call_model(model, prompt, max_tokens=16)
            except Exception as e:
                response = ""
                errors.append(f"{item['task']}: {e}")
            pred = self._parse_response(response, item["task"])
            expected = item["label"]
            correct = (pred is not None and pred == expected)
            if correct:
                n_correct += 1
            bucket = per_task.setdefault(item["task"], {"n_correct": 0, "n_total": 0})
            bucket["n_total"] += 1
            if correct:
                bucket["n_correct"] += 1
            samples.append({
                "task": item["task"],
                "expected": expected,
                "predicted": pred,
                "correct": correct,
                "response_preview": response[:60],
            })
        n_total = len(items)
        accuracy = n_correct / n_total if n_total else 0.0
        per_task_acc = {
            t: {
                "accuracy": v["n_correct"] / v["n_total"] if v["n_total"] else 0.0,
                "n_correct": v["n_correct"],
                "n_total": v["n_total"],
            }
            for t, v in per_task.items()
        }
        result = {
            "benchmark": self.name,
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_subject": per_task_acc,
            "samples": samples,
            "errors": errors,
        }
        self._record(_model_name(model), accuracy, {"n_total": n_total})
        return result


# =====================================================================
# CommonsenseQA
# =====================================================================
_COMMONSENSEQA_ITEMS: List[Dict[str, Any]] = [
    {"question": "Where would you most likely find a satellite?",
     "choices": ["in the ocean", "in orbit around Earth", "in a kitchen", "in a library"], "answer": "B"},
    {"question": "If you want to keep food cold, what appliance do you use?",
     "choices": ["oven", "microwave", "refrigerator", "toaster"], "answer": "C"},
    {"question": "What do people typically do with a book?",
     "choices": ["eat it", "read it", "drive it", "wear it"], "answer": "B"},
    {"question": "Which of these is a primary color?",
     "choices": ["green", "orange", "purple", "red"], "answer": "D"},
    {"question": "A person who treats sick animals is called a:",
     "choices": ["doctor", "veterinarian", "dentist", "lawyer"], "answer": "B"},
    {"question": "What season comes after winter in the Northern Hemisphere?",
     "choices": ["fall", "summer", "spring", "winter"], "answer": "C"},
    {"question": "Which instrument has 88 keys?",
     "choices": ["guitar", "violin", "piano", "flute"], "answer": "C"},
    {"question": "What gas do humans need to breathe to survive?",
     "choices": ["carbon dioxide", "nitrogen", "oxygen", "helium"], "answer": "C"},
    {"question": "How many sides does a hexagon have?",
     "choices": ["5", "6", "7", "8"], "answer": "B"},
    {"question": "What is the capital city of Japan?",
     "choices": ["Beijing", "Seoul", "Bangkok", "Tokyo"], "answer": "D"},
    {"question": "Which planet is known as the Red Planet?",
     "choices": ["Venus", "Mars", "Jupiter", "Mercury"], "answer": "B"},
    {"question": "What do you call a baby dog?",
     "choices": ["kitten", "calf", "puppy", "foal"], "answer": "C"},
    {"question": "Which is the largest ocean on Earth?",
     "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "answer": "D"},
    {"question": "What is the freezing point of water in degrees Celsius?",
     "choices": ["0", "32", "100", "212"], "answer": "A"},
    {"question": "What sound does a cat make?",
     "choices": ["bark", "meow", "moo", "quack"], "answer": "B"},
    {"question": "How many days are in a leap year?",
     "choices": ["364", "365", "366", "367"], "answer": "C"},
    {"question": "Which body organ pumps blood through the body?",
     "choices": ["liver", "brain", "heart", "lungs"], "answer": "C"},
    {"question": "What is the chemical symbol for water?",
     "choices": ["H2", "O2", "H2O", "CO2"], "answer": "C"},
    {"question": "Which of these is NOT a continent?",
     "choices": ["Africa", "Europe", "Greenland", "Asia"], "answer": "C"},
    {"question": "What do bees collect from flowers?",
     "choices": ["milk", "nectar", "water", "salt"], "answer": "B"},
]


class CommonsenseQABenchmark(Benchmark):
    """CommonsenseQA-style multiple-choice commonsense reasoning (20 items)."""

    name = "commonsense_qa"
    version = "1.0.0"
    task_type = "multiple_choice"
    description = "CommonsenseQA (synthetic subset, 20 commonsense MCQs)"
    weight = 1.0

    def _load_items(self) -> List[Dict[str, Any]]:
        return [dict(it) for it in _COMMONSENSEQA_ITEMS]

    def _format_prompt(self, item: Dict[str, Any]) -> str:
        lines = [f"Q: {item['question']}", ""]
        for i, choice in enumerate(item["choices"]):
            lines.append(f"  {chr(65 + i)}. {choice}")
        lines.append("")
        lines.append("Answer with a single letter (A, B, C, or D).")
        return "\n".join(lines)

    def evaluate(self, model: Any, n: Optional[int] = None) -> Dict[str, Any]:
        items = self._slice_items(n)
        n_correct = 0
        samples: List[Dict[str, Any]] = []
        errors: List[str] = []
        for item in items:
            prompt = self._format_prompt(item)
            try:
                response = _call_model(model, prompt, max_tokens=8)
            except Exception as e:
                response = ""
                errors.append(f"csqa: {e}")
            pred = _extract_letter(response)
            expected = item["answer"]
            correct = (pred is not None and pred == expected)
            if correct:
                n_correct += 1
            samples.append({
                "question": item["question"],
                "expected": expected,
                "predicted": pred,
                "correct": correct,
                "response_preview": response[:60],
            })
        n_total = len(items)
        accuracy = n_correct / n_total if n_total else 0.0
        result = {
            "benchmark": self.name,
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_subject": {
                "all": {
                    "accuracy": accuracy,
                    "n_correct": n_correct,
                    "n_total": n_total,
                }
            },
            "samples": samples,
            "errors": errors,
        }
        self._record(_model_name(model), accuracy, {"n_total": n_total})
        return result


# =====================================================================
# ARC — AI2 Reasoning Challenge (easy set)
# =====================================================================
_ARC_ITEMS: List[Dict[str, Any]] = [
    {"question": "Which of the following is an example of a chemical change?",
     "choices": ["ice melting", "wood burning", "paper folding", "water evaporating"], "answer": "B"},
    {"question": "What force keeps the planets in orbit around the Sun?",
     "choices": ["magnetism", "gravity", "friction", "electricity"], "answer": "B"},
    {"question": "Plants make their own food through a process called:",
     "choices": ["respiration", "digestion", "photosynthesis", "transpiration"], "answer": "C"},
    {"question": "Which of these animals is a mammal?",
     "choices": ["shark", "eagle", "whale", "lizard"], "answer": "C"},
    {"question": "What is the main function of the roots of a plant?",
     "choices": ["to make food", "to absorb water and nutrients",
                 "to attract insects", "to produce seeds"], "answer": "B"},
    {"question": "Which state of matter has a definite volume but no definite shape?",
     "choices": ["solid", "liquid", "gas", "plasma"], "answer": "B"},
    {"question": "What is the Earth's primary source of energy?",
     "choices": ["the Moon", "the Sun", "volcanoes", "wind"], "answer": "B"},
    {"question": "Which gas in the atmosphere do plants use for photosynthesis?",
     "choices": ["oxygen", "nitrogen", "carbon dioxide", "hydrogen"], "answer": "C"},
    {"question": "A change in an object's position over time is called:",
     "choices": ["speed", "motion", "force", "energy"], "answer": "B"},
    {"question": "Which part of the plant conducts photosynthesis?",
     "choices": ["roots", "stem", "leaves", "flowers"], "answer": "C"},
    {"question": "What instrument is used to measure temperature?",
     "choices": ["barometer", "thermometer", "ruler", "scale"], "answer": "B"},
    {"question": "Which of the following is a renewable energy source?",
     "choices": ["coal", "oil", "solar power", "natural gas"], "answer": "C"},
    {"question": "What is the center of an atom called?",
     "choices": ["electron", "nucleus", "proton shell", "molecule"], "answer": "B"},
    {"question": "Which of these is a conductor of electricity?",
     "choices": ["rubber", "copper", "glass", "plastic"], "answer": "B"},
    {"question": "The water cycle is driven primarily by energy from:",
     "choices": ["the wind", "the Sun", "the Moon", "Earth's core"], "answer": "B"},
    {"question": "Which of the following describes a solid?",
     "choices": ["takes the shape of its container", "has a definite shape and volume",
                 "flows easily", "has no fixed volume"], "answer": "B"},
    {"question": "What is the process by which liquid water becomes water vapor?",
     "choices": ["condensation", "evaporation", "precipitation", "freezing"], "answer": "B"},
    {"question": "Which organ in the human body is responsible for pumping blood?",
     "choices": ["lungs", "liver", "heart", "brain"], "answer": "C"},
    {"question": "Which of these is NOT a simple machine?",
     "choices": ["lever", "pulley", "wheel and axle", "computer"], "answer": "D"},
    {"question": "What causes day and night on Earth?",
     "choices": ["Earth's revolution around the Sun", "Earth's rotation on its axis",
                 "the Moon's orbit", "the Sun's movement"], "answer": "B"},
]


class ARCBenchmark(Benchmark):
    """ARC-Easy style science reasoning benchmark (20 items)."""

    name = "arc"
    version = "1.0.0"
    task_type = "multiple_choice"
    description = "AI2 Reasoning Challenge — Easy set (synthetic, 20 science MCQs)"
    weight = 1.0

    def _load_items(self) -> List[Dict[str, Any]]:
        return [dict(it) for it in _ARC_ITEMS]

    def _format_prompt(self, item: Dict[str, Any]) -> str:
        lines = [f"Q: {item['question']}", ""]
        for i, choice in enumerate(item["choices"]):
            lines.append(f"  {chr(65 + i)}. {choice}")
        lines.append("")
        lines.append("Answer with a single letter (A, B, C, or D).")
        return "\n".join(lines)

    def evaluate(self, model: Any, n: Optional[int] = None) -> Dict[str, Any]:
        items = self._slice_items(n)
        n_correct = 0
        samples: List[Dict[str, Any]] = []
        errors: List[str] = []
        for item in items:
            prompt = self._format_prompt(item)
            try:
                response = _call_model(model, prompt, max_tokens=8)
            except Exception as e:
                response = ""
                errors.append(f"arc: {e}")
            pred = _extract_letter(response)
            expected = item["answer"]
            correct = (pred is not None and pred == expected)
            if correct:
                n_correct += 1
            samples.append({
                "question": item["question"],
                "expected": expected,
                "predicted": pred,
                "correct": correct,
                "response_preview": response[:60],
            })
        n_total = len(items)
        accuracy = n_correct / n_total if n_total else 0.0
        result = {
            "benchmark": self.name,
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_subject": {
                "all": {
                    "accuracy": accuracy,
                    "n_correct": n_correct,
                    "n_total": n_total,
                }
            },
            "samples": samples,
            "errors": errors,
        }
        self._record(_model_name(model), accuracy, {"n_total": n_total})
        return result


# =====================================================================
# Registry of built-in benchmarks
# =====================================================================
_BENCHMARK_REGISTRY: Dict[str, Callable[[], Benchmark]] = {
    "mmlu": MMLUBenchmark,
    "humaneval": HumanEvalBenchmark,
    "gsm8k": GSM8KBenchmark,
    "glue": GLUEBenchmark,
    "commonsense_qa": CommonsenseQABenchmark,
    "arc": ARCBenchmark,
}


def list_benchmarks() -> List[str]:
    """Return the list of available benchmark names (registry keys)."""
    return sorted(_BENCHMARK_REGISTRY.keys())


def benchmark_info(name: str) -> Dict[str, Any]:
    """Return metadata about a benchmark by name.

    Raises ``KeyError`` if the name is not registered.
    """
    name = name.lower().strip()
    if name not in _BENCHMARK_REGISTRY:
        # Allow some common aliases.
        aliases = {
            "mmlu_subset": "mmlu",
            "commonsenseqa": "commonsense_qa",
            "csqa": "commonsense_qa",
            "arc_easy": "arc",
            "arc-easy": "arc",
        }
        name = aliases.get(name, name)
    if name not in _BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown benchmark: {name!r}. Available: {list_benchmarks()}")
    return _BENCHMARK_REGISTRY[name]().info()


def _get_benchmark(name_or_instance: Union[str, Benchmark]) -> Benchmark:
    """Resolve a benchmark name OR an instance into a Benchmark instance."""
    if isinstance(name_or_instance, Benchmark):
        return name_or_instance
    name = str(name_or_instance).lower().strip()
    aliases = {
        "mmlu_subset": "mmlu",
        "commonsenseqa": "commonsense_qa",
        "csqa": "commonsense_qa",
        "arc_easy": "arc",
    }
    name = aliases.get(name, name)
    if name not in _BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown benchmark: {name!r}. Available: {list_benchmarks()}")
    return _BENCHMARK_REGISTRY[name]()


# =====================================================================
# SuiteReport + BenchmarkSuite
# =====================================================================
class SuiteReport:
    """Container for the results of running a BenchmarkSuite on a model.

    Attributes
    ----------
    model_name : str
    results : dict[name -> result_dict]
    weighted_average : float
    """

    def __init__(self, model_name: str,
                 results: Dict[str, Dict[str, Any]],
                 weights: Dict[str, float],
                 elapsed_seconds: float = 0.0):
        self.model_name = model_name
        self.results = results
        self.weights = weights
        self.elapsed_seconds = elapsed_seconds

    @property
    def weighted_average(self) -> float:
        total_w = 0.0
        total_score = 0.0
        for name, res in self.results.items():
            if "error" in res and "accuracy" not in res:
                continue
            w = self.weights.get(name, 1.0)
            s = res.get("accuracy", res.get("pass_at_1", res.get("score", 0.0)))
            try:
                s = float(s)
            except (TypeError, ValueError):
                continue
            total_w += w
            total_score += w * s
        return total_score / total_w if total_w > 0 else 0.0

    @property
    def overall_score(self) -> float:
        return self.weighted_average

    def summary(self) -> str:
        lines = [f"=== Suite Report: {self.model_name} ==="]
        lines.append(f"  benchmarks run: {len(self.results)}")
        lines.append(f"  weighted average: {self.weighted_average:.2%}")
        lines.append(f"  elapsed: {self.elapsed_seconds:.2f}s")
        lines.append("")
        lines.append(f"  {'benchmark':<18} {'accuracy':>10} {'correct':>10} {'weight':>8}")
        lines.append("  " + "-" * 50)
        for name in sorted(self.results.keys()):
            res = self.results[name]
            acc = res.get("accuracy", res.get("pass_at_1", res.get("score", 0.0)))
            n_c = res.get("n_correct", res.get("n_passed", 0))
            w = self.weights.get(name, 1.0)
            if "error" in res and "accuracy" not in res:
                lines.append(f"  {name:<18} {'ERROR':>10} {'-':>10} {w:>8.2f}")
            else:
                lines.append(f"  {name:<18} {acc:>10.2%} {n_c:>10} {w:>8.2f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "weighted_average": self.weighted_average,
            "elapsed_seconds": self.elapsed_seconds,
            "weights": dict(self.weights),
            "results": self.results,
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"<SuiteReport model={self.model_name!r} avg={self.weighted_average:.2%}>"


class BenchmarkSuite:
    """Aggregate runner over multiple benchmarks.

    Parameters
    ----------
    benchmarks : list, optional
        List of benchmark names or ``Benchmark`` instances. Defaults to all
        registered benchmarks.
    weights : dict, optional
        Mapping of benchmark name -> weight for the weighted average.
    """

    def __init__(self,
                 benchmarks: Optional[Sequence[Union[str, Benchmark]]] = None,
                 weights: Optional[Dict[str, float]] = None):
        if benchmarks is None:
            benchmarks = list(list_benchmarks())
        self.benchmarks: List[Benchmark] = [_get_benchmark(b) for b in benchmarks]
        self.weights: Dict[str, float] = {b.name: b.weight for b in self.benchmarks}
        if weights:
            self.weights.update(weights)

    def run(self, model: Any, n: Optional[int] = None) -> SuiteReport:
        """Run all benchmarks on a model and return a SuiteReport."""
        model_name = _model_name(model)
        results: Dict[str, Dict[str, Any]] = {}
        t0 = time.time()
        for bench in self.benchmarks:
            try:
                res = bench.evaluate(model, n=n)
                results[bench.name] = res
            except Exception as e:
                tb = traceback.format_exc(limit=2)
                results[bench.name] = {
                    "benchmark": bench.name,
                    "error": str(e),
                    "traceback": tb,
                }
        elapsed = time.time() - t0
        return SuiteReport(model_name, results, dict(self.weights), elapsed)

    def compare(self, models: Sequence[Any]) -> "Leaderboard":
        """Run all benchmarks on every model and rank them.

        ``models`` may be either model instances or ``(name, model)`` tuples.
        Returns a populated :class:`Leaderboard`.
        """
        board = Leaderboard(suite=self)
        for m in models:
            if isinstance(m, tuple) and len(m) == 2:
                name, model = m
            else:
                name, model = _model_name(m), m
            report = self.run(model)
            board.rank(name, report)
        return board

    def summary(self) -> str:
        lines = ["BenchmarkSuite:"]
        for b in self.benchmarks:
            info = b.info()
            lines.append(
                f"  - {info['name']:<18} ({info['task_type']:<16}) "
                f"items={info['n_samples']:>3}  weight={self.weights.get(info['name'], 1.0):.2f}"
            )
        return "\n".join(lines)


# =====================================================================
# Leaderboard
# =====================================================================
class Leaderboard:
    """A simple model leaderboard.

    Entries may be added via ``rank(model_name, score_or_report)``. ``score_or_report``
    can be a float (interpreted as the overall score) or a :class:`SuiteReport`
    (in which case its ``weighted_average`` is used and the per-benchmark
    breakdown is preserved).
    """

    def __init__(self, suite: Optional[BenchmarkSuite] = None):
        self.suite = suite
        self.entries: List[Dict[str, Any]] = []

    def rank(self, model_name: str, score: Union[float, SuiteReport, Dict[str, Any]]) -> "Leaderboard":
        """Add a model's score to the leaderboard and return self."""
        if isinstance(score, SuiteReport):
            entry = {
                "model": model_name,
                "score": score.weighted_average,
                "per_benchmark": {k: v.get("accuracy", v.get("pass_at_1", v.get("score", 0.0)))
                                  for k, v in score.results.items()},
                "ts": time.time(),
            }
        elif isinstance(score, dict) and "weighted_average" in score:
            entry = {
                "model": model_name,
                "score": score["weighted_average"],
                "per_benchmark": score.get("results", {}),
                "ts": time.time(),
            }
        else:
            try:
                sc = float(score)
            except (TypeError, ValueError):
                sc = 0.0
            entry = {
                "model": model_name,
                "score": sc,
                "per_benchmark": {},
                "ts": time.time(),
            }
        self.entries.append(entry)
        return self

    def top(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the top-N entries sorted by score descending."""
        return sorted(self.entries, key=lambda e: e.get("score", 0.0), reverse=True)[:n]

    def to_markdown(self) -> str:
        """Render the leaderboard as a markdown table."""
        if not self.entries:
            return "_(empty leaderboard)_"
        # Collect the set of benchmarks seen across entries.
        bench_names: List[str] = []
        seen = set()
        for e in self.entries:
            for b in e.get("per_benchmark", {}).keys():
                if b not in seen:
                    seen.add(b)
                    bench_names.append(b)
        bench_names.sort()
        header = ["Rank", "Model", "Overall"]
        header.extend(bench_names)
        sep = ["---"] * len(header)
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join(sep) + " |"]
        for i, e in enumerate(self.top(len(self.entries)), start=1):
            row = [str(i), str(e.get("model", "?")), f"{e.get('score', 0.0):.2%}"]
            per = e.get("per_benchmark", {}) or {}
            for b in bench_names:
                v = per.get(b)
                if isinstance(v, dict):
                    v = v.get("accuracy", v.get("score", 0.0))
                if v is None:
                    row.append("-")
                else:
                    try:
                        row.append(f"{float(v):.2%}")
                    except (TypeError, ValueError):
                        row.append(str(v))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def to_dataframe(self) -> Dict[str, List[Any]]:
        """Return a dict-of-columns suitable for ``pandas.DataFrame(...)``.

        Falls back to plain Python lists so callers can do
        ``pd.DataFrame(leaderboard.to_dataframe())`` without us importing pandas.
        """
        bench_names: List[str] = []
        seen = set()
        for e in self.entries:
            for b in e.get("per_benchmark", {}).keys():
                if b not in seen:
                    seen.add(b)
                    bench_names.append(b)
        bench_names.sort()
        cols: Dict[str, List[Any]] = {
            "rank": [],
            "model": [],
            "overall": [],
        }
        for b in bench_names:
            cols[b] = []
        for i, e in enumerate(self.top(len(self.entries)), start=1):
            cols["rank"].append(i)
            cols["model"].append(e.get("model", "?"))
            cols["overall"].append(float(e.get("score", 0.0)))
            per = e.get("per_benchmark", {}) or {}
            for b in bench_names:
                v = per.get(b)
                if isinstance(v, dict):
                    v = v.get("accuracy", v.get("score", None))
                cols[b].append(v)
        return cols

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": list(self.entries),
            "n_models": len(self.entries),
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"<Leaderboard n_models={len(self.entries)}>"


# =====================================================================
# Top-level convenience functions
# =====================================================================
def run_benchmark(name: str, model: Any, n: Optional[int] = None) -> Dict[str, Any]:
    """Run a single named benchmark on a model.

    Parameters
    ----------
    name : str
        Benchmark name (e.g. ``"mmlu"``). See :func:`list_benchmarks`.
    model : Any
        A model that responds to ``chat`` / ``generate`` / ``predict`` / ``__call__``.
    n : int, optional
        Limit to the first ``n`` items.

    Returns
    -------
    dict
        The benchmark's result dict (with at least ``accuracy``,
        ``n_correct``, ``n_total``, ``per_subject``, ``samples``).
    """
    bench = _get_benchmark(name)
    return bench.evaluate(model, n=n)


def run_full_suite(model: Any,
                   benchmarks: Optional[Sequence[Union[str, Benchmark]]] = None) -> SuiteReport:
    """Run all (or a subset of) benchmarks on a model.

    Parameters
    ----------
    model : Any
        A model with a chat/generate/predict interface.
    benchmarks : list, optional
        List of benchmark names or instances. Defaults to all registered.

    Returns
    -------
    SuiteReport
    """
    suite = BenchmarkSuite(benchmarks=benchmarks)
    return suite.run(model)


def compare_models(models: Sequence[Any],
                   benchmarks: Optional[Sequence[Union[str, Benchmark]]] = None) -> Leaderboard:
    """Compare multiple models across the benchmark suite.

    Parameters
    ----------
    models : sequence
        Either model instances or ``(name, model)`` tuples.
    benchmarks : list, optional
        Benchmarks to include. Defaults to all registered.

    Returns
    -------
    Leaderboard
        Populated with one entry per model.
    """
    suite = BenchmarkSuite(benchmarks=benchmarks)
    return suite.compare(models)


__all__ = [
    # Base classes
    "Benchmark",
    # Concrete benchmarks
    "MMLUBenchmark",
    "HumanEvalBenchmark",
    "GSM8KBenchmark",
    "GLUEBenchmark",
    "CommonsenseQABenchmark",
    "ARCBenchmark",
    # Suite + report
    "BenchmarkSuite",
    "SuiteReport",
    # Leaderboard
    "Leaderboard",
    # Convenience functions
    "run_benchmark",
    "run_full_suite",
    "compare_models",
    "list_benchmarks",
    "benchmark_info",
]
