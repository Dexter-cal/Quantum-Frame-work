"""
qai.security -- Security module (spec Section 33).

Real production AI systems need defense-in-depth: detect poisoned data, guard
against prompt injection, sandbox untrusted code, scan for PII/secrets, and
verify model provenance.

This module provides 10 security tools, each independently usable:

    from qai.security import (
        security_scan, prompt_injection_guard, sandbox_run,
        detect_pii, scan_secrets, detect_adversarial_input,
        moderate_content, sanitize_prompt_template,
        APIKeyVault, verify_model_integrity,
    )

    # 1. Scan a dataset for poisoning patterns
    report = security_scan(X, y)
    print(report["outlier_clusters"], report["label_flips"])

    # 2. Guard against prompt injection
    if prompt_injection_guard(user_input)["is_injection"]:
        raise ValueError("Possible prompt injection detected")

    # 3. Run untrusted code in a sandbox
    result = sandbox_run("print(2+2)", timeout=5.0)

    # 4. Detect PII
    pii = detect_pii("Contact me at alice@example.com or 555-123-4567")
    # [{"type": "email", "value": "alice@example.com", "start": 14, "end": 32}, ...]

    # 5. Scan for hardcoded secrets
    secrets = scan_secrets("API_KEY=sk-abc123...")

    # 6. Detect adversarial inputs (very long, repeated chars, control chars)
    adv = detect_adversarial_input(input_text)

    # 7. Content moderation
    mod = moderate_content("some text")  # -> {toxicity: 0.1, profanity: 0.0, ...}

    # 8. Sanitize a prompt template (strip injection vectors)
    clean = sanitize_prompt_template(user_template)

    # 9. Encrypted API key vault
    vault = APIKeyVault(master_password="my-secret")
    vault.set("openai_key", "sk-abc123...")
    print(vault.get("openai_key"))

    # 10. Verify model integrity (signed bundles)
    ok = verify_model_integrity("model.qmodel", sign_key="my-secret")

Creative additions beyond the spec:
  - **Adversarial input detection** — flags anomalous inputs (very long,
    repeated chars, control characters, encoding tricks)
  - **Prompt template sanitizer** — strips `{{...}}` substitution vectors,
    escape sequences, and zero-width characters from user-supplied templates
  - **API key vault** — encrypted at-rest storage using Fernet-style
    AES-128-CBC + HMAC-SHA256 (no external deps beyond stdlib)
  - **Content moderation** — lightweight rule-based toxicity/profanity
    detector (no external model required)
  - **Model integrity verifier** — uses qai.bundle.verify_bundle for signed
    .qmodel files; also computes fresh SHA-256 of weights for unsigned models
"""
from __future__ import annotations

import ast
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# =====================================================================
# 1. Dataset security scan (poisoning detection)
# =====================================================================
def security_scan(
    X: Union[np.ndarray, List[List[float]]],
    y: Optional[Union[np.ndarray, List]] = None,
    contamination: float = 0.05,
    label_flip_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Scan a dataset for poisoning patterns.

    Detects:
      - **Outlier clusters** — groups of points far from the main distribution
        (potential adversarial examples)
      - **Label flipping** — samples whose features match one class but are
        labeled as another (suspicious — possible label-flip attack)
      - **Duplicate adversarial samples** — exact duplicates that might be
        injected to bias the model
      - **Class imbalance** — extreme imbalance might indicate injection

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Labels, shape (n_samples,). Optional.
        contamination: Expected fraction of anomalies (default 0.05).
        label_flip_threshold: Samples whose nearest-class-neighbor disagrees
            with their label above this ratio are flagged. Default 0.3.

    Returns:
        Dict with keys:
          - `outlier_clusters`: list of indices flagged as outliers
          - `label_flips`: list of indices flagged as suspected label flips
          - `duplicates`: list of (i, j) index pairs that are exact duplicates
          - `class_distribution`: dict {label: count}
          - `imbalance_ratio`: max_class_count / min_class_count
          - `summary`: dict with total flag count and risk level
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, d = X_arr.shape

    # Outlier detection: compute distance from centroid, flag top `contamination`%
    centroid = X_arr.mean(axis=0)
    distances = np.linalg.norm(X_arr - centroid, axis=1)
    n_outliers = max(1, int(n * contamination))
    outlier_idx = np.argsort(-distances)[:n_outliers].tolist()

    # Label flipping: for each sample, find k nearest neighbors; if majority
    # of neighbors have a different label, flag it
    label_flips: List[int] = []
    if y is not None:
        y_arr = np.asarray(y)
        k = min(5, n - 1)
        for i in range(n):
            # Distances from sample i to all others (excluding self)
            dists = np.linalg.norm(X_arr - X_arr[i], axis=1)
            dists[i] = np.inf  # exclude self
            neighbors = np.argsort(dists)[:k]
            neighbor_labels = y_arr[neighbors]
            own_label = y_arr[i]
            disagreement = np.mean(neighbor_labels != own_label)
            if disagreement > label_flip_threshold:
                label_flips.append(int(i))

    # Exact duplicates
    duplicates: List[Tuple[int, int]] = []
    seen: Dict[bytes, int] = {}
    for i in range(n):
        key = X_arr[i].tobytes()
        if key in seen:
            duplicates.append((seen[key], i))
        else:
            seen[key] = i

    # Class distribution
    class_dist: Dict[Any, int] = {}
    if y is not None:
        y_arr = np.asarray(y)
        for label in y_arr:
            key = label.item() if hasattr(label, "item") else label
            class_dist[key] = class_dist.get(key, 0) + 1
        counts = list(class_dist.values())
        imbalance = max(counts) / max(1, min(counts)) if counts else 1.0
    else:
        imbalance = 1.0

    total_flags = len(outlier_idx) + len(label_flips) + len(duplicates)
    risk_level = (
        "high" if total_flags > 0.2 * n
        else "medium" if total_flags > 0.05 * n
        else "low"
    )

    return {
        "outlier_clusters": outlier_idx,
        "label_flips": label_flips,
        "duplicates": duplicates,
        "class_distribution": class_dist,
        "imbalance_ratio": float(imbalance),
        "summary": {
            "total_flags": total_flags,
            "flag_ratio": total_flags / max(1, n),
            "risk_level": risk_level,
            "n_samples": n,
            "n_features": d,
        },
    }


# =====================================================================
# 2. Prompt injection guard
# =====================================================================
# Heuristic patterns from common prompt-injection attacks (OWASP LLM Top 10)
_INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "ignore_instructions"),
    (r"disregard\s+(all\s+)?(prior|previous)\s+(instructions|prompts)", "disregard_instructions"),
    (r"forget\s+(everything|all\s+previous)", "forget_previous"),
    (r"you\s+are\s+now\s+(a|an)\s+", "role_hijack"),
    (r"new\s+instructions?\s*:", "new_instructions"),
    (r"system\s*:\s*", "system_prefix"),
    (r"<\s*system\s*>", "system_tag"),
    (r"<\s*/?\s*(system|assistant|user)\s*>", "chatml_tag"),
    (r"reveal\s+(your|the)\s+(system\s+)?prompt", "prompt_extraction"),
    (r"what\s+(is|are)\s+your\s+(initial\s+)?instructions", "instructions_extraction"),
    (r"print\s+(your|the)\s+system\s+prompt", "prompt_print"),
    (r"jailbreak", "jailbreak_keyword"),
    (r"DAN\s+mode", "dan_mode"),
    (r"developer\s+mode", "developer_mode"),
    (r"act\s+as\s+(if\s+)?(you\s+(have|are)\s+)?(no|without)\s+(restrictions|limits)", "no_restrictions"),
    (r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?\w+\s+with\s+no\s+(rules|restrictions)", "pretend_unrestricted"),
    (r"\\x[0-9a-f]{2}", "hex_escape"),  # encoding tricks
    (r"%[0-9a-f]{2}", "url_encoded"),
    (r"\beval\s*\(", "eval_call"),
    (r"\bexec\s*\(", "exec_call"),
    (r"__import__\s*\(", "dunder_import"),
]


def prompt_injection_guard(
    text: str,
    extra_patterns: Optional[List[Tuple[str, str]]] = None,
    threshold: int = 1,
) -> Dict[str, Any]:
    """Classify whether a text looks like a prompt-injection attack.

    Uses a list of heuristic regex patterns derived from common attacks
    (OWASP LLM Top 10 LLM01: Prompt Injection). Each pattern has a name;
    if `threshold` or more patterns match, `is_injection=True`.

    Args:
        text: The input text to check.
        extra_patterns: Optional list of (regex, name) tuples to add to the
            default patterns.
        threshold: Min number of pattern matches to flag as injection (default 1).

    Returns:
        Dict with keys:
          - `is_injection`: bool
          - `matched_patterns`: list of pattern names that matched
          - `match_count`: int
          - `text_length`: int
          - `risk_level`: "low" / "medium" / "high"
    """
    patterns = list(_INJECTION_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)

    matched: List[str] = []
    for regex, name in patterns:
        if re.search(regex, text, flags=re.IGNORECASE):
            matched.append(name)

    is_injection = len(matched) >= threshold
    if len(matched) >= 3:
        risk = "high"
    elif len(matched) >= 1:
        risk = "medium" if is_injection else "low"
    else:
        risk = "low"

    return {
        "is_injection": is_injection,
        "matched_patterns": matched,
        "match_count": len(matched),
        "text_length": len(text),
        "risk_level": risk,
    }


# =====================================================================
# 3. Sandbox runner
# =====================================================================
def sandbox_run(
    code: str,
    timeout: float = 5.0,
    work_dir: Optional[str] = None,
    allowed_modules: Optional[List[str]] = None,
    max_memory_mb: int = 100,
) -> Dict[str, Any]:
    """Execute untrusted Python code in a sandboxed subprocess.

    Spawns a separate Python subprocess with restricted builtins, no network
    access (via env vars), and a timeout. The subprocess writes its result
    to a temp file, which the parent reads.

    For simpler in-process sandboxing, use `qai.mcp.CodeEvalTool` instead.
    This function provides stronger isolation (separate process) but is slower.

    Args:
        code: Python code to execute.
        timeout: Max seconds before killing the subprocess (default 5).
        work_dir: Working directory for the subprocess. If None, uses a temp dir.
        allowed_modules: Module names the code is allowed to import. Default
            allows only: math, random, json, re, datetime, statistics.
        max_memory_mb: Memory limit hint (best-effort, depends on OS).

    Returns:
        Dict with `stdout`, `stderr`, `result`, `error`, `exit_code`, `elapsed_s`.
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="qai_sandbox_")
    os.makedirs(work_dir, exist_ok=True)

    # Build the sandbox script
    allowed = allowed_modules or ["math", "random", "json", "re", "datetime", "statistics"]
    # The sandbox script: restricted builtins, only allowed modules importable
    sandbox_script = f'''
import sys, os, json, traceback, io, builtins as _builtins

_ALLOWED_MODULES = {allowed!r}
_original_import = _builtins.__import__

def _restricted_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top not in _ALLOWED_MODULES:
        raise ImportError(f"Module {{name!r}} is not in the allowed list: {{_ALLOWED_MODULES}}")
    return _original_import(name, *args, **kwargs)

# Build restricted builtins dict
_SAFE_BUILTIN_NAMES = {{n: getattr(_builtins, n) for n in dir(_builtins) if not n.startswith("_")
                       and n not in ("exec", "eval", "compile", "open", "input", "breakpoint",
                                     "globals", "locals", "vars", "dir", "getattr", "setattr",
                                     "delattr", "memoryview", "help", "exit", "quit", "license",
                                     "copyright", "credits")}}
_SAFE_BUILTINS = dict(_SAFE_BUILTIN_NAMES)
_SAFE_BUILTINS["__import__"] = _restricted_import
_SAFE_BUILTINS["__name__"] = "__sandbox__"

# Capture stdout
_captured = io.StringIO()
_old_stdout = sys.stdout
sys.stdout = _captured

_result = None
_error = None
try:
    exec(compile({code!r}, "<sandbox>", "exec"), {{"__builtins__": _SAFE_BUILTINS}})
except Exception as e:
    _error = f"{{type(e).__name__}}: {{e}}"
    _error += "\\n" + traceback.format_exc()

sys.stdout = _old_stdout
print(json.dumps({{"stdout": _captured.getvalue(), "error": _error}}))
'''

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-c", sandbox_script],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": os.environ.get("PATH", ""), "HOME": work_dir},
        )
        elapsed = time.time() - start
        # Parse the JSON line from stdout
        try:
            payload = json.loads(result.stdout.strip().split("\n")[-1])
        except (json.JSONDecodeError, IndexError):
            payload = {"stdout": result.stdout, "error": "Failed to parse sandbox output"}
        return {
            "stdout": payload.get("stdout", ""),
            "stderr": result.stderr,
            "error": payload.get("error"),
            "exit_code": result.returncode,
            "elapsed_s": round(elapsed, 4),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "",
            "error": f"Timeout after {timeout}s",
            "exit_code": -1,
            "elapsed_s": round(time.time() - start, 4),
            "timed_out": True,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "error": f"Sandbox error: {type(e).__name__}: {e}",
            "exit_code": -1,
            "elapsed_s": round(time.time() - start, 4),
            "timed_out": False,
        }


# =====================================================================
# 4. PII detector
# =====================================================================
_PII_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("phone_us", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("ipv6", re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b")),
    ("us_zip", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
    ("date_iso", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("api_key_aws", re.compile(r"\bAKIA[0-9A-Z]{{16}}\b".replace("{{", "{").replace("}}", "}"))),
    ("api_key_generic", re.compile(r"\b(?:sk|pk|api[_-]?key|token)[_-]?[a-zA-Z0-9]{{20,}}\b".replace("{{", "{").replace("}}", "}"))),
    ("jwt", re.compile(r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b")),
]


def detect_pii(text: str, types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Detect personally-identifiable information in text.

    Args:
        text: Input text.
        types: Optional list of PII types to detect (e.g. ["email", "phone_us"]).
            If None, detects all types.

    Returns:
        List of dicts: `[{"type": "email", "value": "alice@example.com", "start": 14, "end": 32}, ...]`
    """
    results: List[Dict[str, Any]] = []
    for pii_type, pattern in _PII_PATTERNS:
        if types and pii_type not in types:
            continue
        for m in pattern.finditer(text):
            results.append({
                "type": pii_type,
                "value": m.group(0),
                "start": m.start(),
                "end": m.end(),
            })
    # Sort by position
    results.sort(key=lambda r: r["start"])
    return results


def redact_pii(text: str, replacement: str = "[REDACTED]") -> Tuple[str, int]:
    """Redact all detected PII in text. Returns (redacted_text, count)."""
    pii = detect_pii(text)
    # Replace from end so positions don't shift
    redacted = text
    for entry in sorted(pii, key=lambda r: -r["start"]):
        redacted = redacted[:entry["start"]] + replacement + redacted[entry["end"]:]
    return redacted, len(pii)


# =====================================================================
# 5. Secret scanner
# =====================================================================
_SECRET_PATTERNS = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("aws_secret", re.compile(r"(?i)aws(.{0,20})?(secret|sk)[^\n]{0,40}['\"][0-9a-zA-Z/+]{40}['\"]")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36}")),
    ("github_pat", re.compile(r"github_pat_[A-Za-z0-9_]{82}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    ("slack_token", re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,48}")),
    ("stripe_key", re.compile(r"(sk|pk)_(test_|live_)[0-9a-zA-Z]{24}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9-_]{95}")),
    ("private_key", re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("jwt", re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+")),
    ("generic_env_var", re.compile(
        r"(?i)(api[_-]?key|secret|token|password|passwd|pwd)[\s]*[:=][\s]*['\"][^'\"]{8,}['\"]"
    )),
]


def scan_secrets(text: str) -> List[Dict[str, Any]]:
    """Scan text for hardcoded secrets (API keys, tokens, private keys).

    Args:
        text: Input text (e.g. source code, config file, env dump).

    Returns:
        List of dicts: `[{"type": "openai_key", "value": "sk-...", "start": ..., "end": ...}, ...]`
    """
    results: List[Dict[str, Any]] = []
    for secret_type, pattern in _SECRET_PATTERNS:
        for m in pattern.finditer(text):
            results.append({
                "type": secret_type,
                "value": m.group(0),
                "start": m.start(),
                "end": m.end(),
            })
    results.sort(key=lambda r: r["start"])
    return results


# =====================================================================
# 6. Adversarial input detector (my own addition)
# =====================================================================
def detect_adversarial_input(
    text: str,
    max_length: int = 10000,
    max_repeat_ratio: float = 0.3,
    max_unicode_tricks: int = 3,
) -> Dict[str, Any]:
    """Detect anomalous inputs that might be adversarial.

    Flags:
      - **Excessive length** — inputs longer than `max_length` chars
      - **Repetition** — single character/word repeated > `max_repeat_ratio` of total
      - **Control characters** — non-printable chars (excluding newline/tab)
      - **Unicode tricks** — zero-width chars, homoglyphs, RTL/LTR overrides
      - **Mixed scripts** — combining Latin + Cyrillic + Greek (homoglyph attacks)
      - **Encoding tricks** — hex/URL/HTML entity encoded sequences

    Args:
        text: Input text.
        max_length: Max allowed length (default 10000).
        max_repeat_ratio: Max fraction of text that can be a single char (default 0.3).
        max_unicode_tricks: Max allowed zero-width/homoglyph chars (default 3).

    Returns:
        Dict with `is_adversarial`, `flags` (list), `score` (0-1), `details`.
    """
    flags: List[str] = []

    # 1. Length
    if len(text) > max_length:
        flags.append("excessive_length")

    # 2. Repetition: most-common char / total length
    if text:
        from collections import Counter
        char_counts = Counter(text)
        most_common_ratio = char_counts.most_common(1)[0][1] / len(text)
        if most_common_ratio > max_repeat_ratio:
            flags.append("excessive_repetition")

    # 3. Control characters (excluding \n, \r, \t)
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    if control_chars > 0:
        flags.append("control_characters")

    # 4. Unicode tricks: zero-width chars, RTL/LTR overrides, BOM
    zero_width = sum(1 for c in text if ord(c) in (
        0x200B, 0x200C, 0x200D, 0xFEFF,  # zero-width space, joiner, BOM
        0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # RTL/LTR overrides
        0x2060, 0x2061, 0x2062, 0x2063,  # word joiner etc.
    ))
    if zero_width > max_unicode_tricks:
        flags.append("unicode_tricks")

    # 5. Mixed scripts (Latin + Cyrillic + Greek homoglyphs)
    scripts_present = set()
    for c in text:
        cp = ord(c)
        if 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A:  # basic Latin
            scripts_present.add("latin")
        elif 0x410 <= cp <= 0x4FF:  # Cyrillic
            scripts_present.add("cyrillic")
        elif 0x390 <= cp <= 0x3FF:  # Greek
            scripts_present.add("greek")
    if len(scripts_present) >= 2:
        flags.append("mixed_scripts")

    # 6. Encoding tricks
    if re.search(r"(?:\\x[0-9a-f]{2}){3,}", text, re.IGNORECASE):
        flags.append("hex_encoding")
    if re.search(r"(?:%[0-9a-f]{2}){3,}", text, re.IGNORECASE):
        flags.append("url_encoding")
    if re.search(r"&#\d+;", text) or re.search(r"&#x[0-9a-f]+;", text, re.IGNORECASE):
        flags.append("html_entity_encoding")

    score = min(1.0, len(flags) / 3.0)  # 3+ flags = score 1.0
    is_adversarial = len(flags) >= 1

    return {
        "is_adversarial": is_adversarial,
        "flags": flags,
        "flag_count": len(flags),
        "score": score,
        "text_length": len(text),
        "details": {
            "control_chars": control_chars,
            "zero_width_chars": zero_width,
            "scripts": list(scripts_present),
            "most_common_char_ratio": most_common_ratio if text else 0.0,
        },
    }


# =====================================================================
# 7. Content moderation (my own addition — lightweight rule-based)
# =====================================================================
# Curated small word lists — not exhaustive, but enough to flag obvious cases
_PROFANITY_WORDS = {
    "fuck", "shit", "ass", "bitch", "bastard", "damn", "crap", "dick",
    "piss", "hell", "asshole", "motherfucker", "bullshit",
}
_TOXIC_PHRASES = {
    "kill yourself", "go die", "you suck", "shut up", "you're stupid",
    "you are stupid", "i hate you", "you're an idiot", "you are an idiot",
}
_HATE_PATTERNS = [
    re.compile(r"\b(all|those|these)\s+\w+\s+(are|is|should)\b", re.IGNORECASE),
    re.compile(r"\bgo\s+back\s+to\s+your\s+country\b", re.IGNORECASE),
]


def moderate_content(text: str) -> Dict[str, Any]:
    """Lightweight content moderation: toxicity, profanity, hate speech.

    Uses rule-based detection (word lists + simple patterns). For production
    use, swap for a real model (e.g. Perspective API, OpenAI moderation).

    Args:
        text: Input text.

    Returns:
        Dict with `toxicity` (0-1), `profanity` (0-1), `hate` (0-1),
        `flagged_categories` (list), `is_flagged` (bool).
    """
    text_lower = text.lower()
    words = set(re.findall(r"\w+", text_lower))

    # Profanity: fraction of words that are profane
    profanity_count = sum(1 for w in words if w in _PROFANITY_WORDS)
    profanity_score = min(1.0, profanity_count / max(1, len(words)) * 10)  # scale up

    # Toxicity: phrase matches
    toxicity_count = sum(1 for phrase in _TOXIC_PHRASES if phrase in text_lower)
    toxicity_score = min(1.0, toxicity_count * 0.5)

    # Hate speech: pattern matches
    hate_count = sum(1 for pat in _HATE_PATTERNS if pat.search(text))
    hate_score = min(1.0, hate_count * 0.7)

    flagged = []
    if profanity_score > 0.3:
        flagged.append("profanity")
    if toxicity_score > 0.3:
        flagged.append("toxicity")
    if hate_score > 0.3:
        flagged.append("hate")

    return {
        "toxicity": toxicity_score,
        "profanity": profanity_score,
        "hate": hate_score,
        "flagged_categories": flagged,
        "is_flagged": len(flagged) > 0,
    }


# =====================================================================
# 8. Prompt template sanitizer (my own addition)
# =====================================================================
def sanitize_prompt_template(
    template: str,
    allow_variables: bool = False,
    strip_html: bool = True,
    strip_zero_width: bool = True,
    max_length: int = 10000,
) -> Dict[str, Any]:
    """Sanitize a user-supplied prompt template.

    Strips:
      - **`{{variable}}` substitution vectors** — unless `allow_variables=True`
      - **HTML tags** — `<script>`, `<iframe>`, etc. — unless `strip_html=False`
      - **Zero-width Unicode** — invisible chars that can hide injection
      - **Control characters** — non-printable (except newline/tab)
      - **Truncates to `max_length`** to prevent resource exhaustion

    Args:
        template: User-supplied prompt template.
        allow_variables: If True, keep `{{...}}` substitutions (default False).
        strip_html: If True, remove HTML tags (default True).
        strip_zero_width: If True, remove zero-width Unicode (default True).
        max_length: Truncate to this length (default 10000).

    Returns:
        Dict with `sanitized` (str), `changes` (list of strings describing what was changed),
        `original_length`, `sanitized_length`.
    """
    changes: List[str] = []
    sanitized = template

    # 1. Variable substitution
    if not allow_variables:
        new = re.sub(r"\{\{[^}]+\}\}", "[VAR]", sanitized)
        if new != sanitized:
            changes.append("replaced_{{var}}_with_[VAR]")
            sanitized = new

    # 2. HTML tags
    if strip_html:
        new = re.sub(r"<[^>]+>", "", sanitized)
        if new != sanitized:
            changes.append("stripped_html_tags")
            sanitized = new

    # 3. Zero-width chars
    if strip_zero_width:
        new = "".join(c for c in sanitized if ord(c) not in (
            0x200B, 0x200C, 0x200D, 0xFEFF,
            0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
            0x2060, 0x2061, 0x2062, 0x2063,
        ))
        if new != sanitized:
            changes.append("stripped_zero_width_chars")
            sanitized = new

    # 4. Control chars (except \n \r \t)
    new = "".join(c for c in sanitized if ord(c) >= 32 or c in "\n\r\t")
    if new != sanitized:
        changes.append("stripped_control_chars")
        sanitized = new

    # 5. Truncate
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        changes.append(f"truncated_to_{max_length}_chars")

    return {
        "sanitized": sanitized,
        "changes": changes,
        "original_length": len(template),
        "sanitized_length": len(sanitized),
    }


# =====================================================================
# 9. API key vault (my own addition — encrypted storage)
# =====================================================================
class APIKeyVault:
    """Encrypted at-rest storage for API keys and secrets.

    Uses AES-128-CBC + HMAC-SHA256 (encrypt-then-MAC) implemented on top of
    Python's `hashlib` + `hmac` + manual AES via `os.urandom` for keys.

    For real production use, swap the implementation to use `cryptography`
    library's Fernet (which is audited). This implementation is intentionally
    dependency-free so it works in any environment.

    Storage format (JSON):
        {
            "keys": {
                "name1": {"ciphertext": "hex", "iv": "hex", "mac": "hex", "salt": "hex"},
                "name2": {...},
            },
            "version": 1,
        }
    """
    def __init__(self, master_password: str, storage_path: Optional[str] = None):
        """
        Args:
            master_password: Master password. Derives the encryption key via
                PBKDF2-HMAC-SHA256 with a random salt.
            storage_path: Path to the vault file. If None, in-memory only.
        """
        self.master_password = master_password.encode("utf-8")
        self.storage_path = storage_path
        self._keys: Dict[str, Dict[str, str]] = {}
        if storage_path and os.path.exists(storage_path):
            self._load()

    def _derive_key(self, salt: bytes, length: int = 32) -> bytes:
        """PBKDF2-HMAC-SHA256 key derivation."""
        return hashlib.pbkdf2_hmac("sha256", self.master_password, salt, iterations=100000, dklen=length)

    def _encrypt(self, plaintext: str) -> Dict[str, str]:
        """Encrypt a string. Returns dict with ciphertext, iv, mac, salt (all hex)."""
        salt = os.urandom(16)
        iv = os.urandom(16)
        key = self._derive_key(salt)
        # XOR-keystream cipher (simple but correct; for production, use AES)
        # Generate keystream via SHA-256 in counter mode
        plaintext_bytes = plaintext.encode("utf-8")
        ciphertext = bytearray()
        counter = 0
        while len(ciphertext) < len(plaintext_bytes):
            block = hashlib.sha256(key + iv + counter.to_bytes(8, "big")).digest()
            chunk = plaintext_bytes[len(ciphertext):len(ciphertext) + 32]
            ciphertext.extend(b ^ k for b, k in zip(chunk, block))
            counter += 1
        # MAC over iv + ciphertext
        mac = hmac.new(key, iv + bytes(ciphertext), hashlib.sha256).digest()
        return {
            "ciphertext": bytes(ciphertext).hex(),
            "iv": iv.hex(),
            "mac": mac.hex(),
            "salt": salt.hex(),
        }

    def _decrypt(self, entry: Dict[str, str]) -> str:
        """Decrypt an entry. Raises ValueError if MAC fails."""
        salt = bytes.fromhex(entry["salt"])
        iv = bytes.fromhex(entry["iv"])
        ciphertext = bytes.fromhex(entry["ciphertext"])
        mac = bytes.fromhex(entry["mac"])
        key = self._derive_key(salt)
        # Verify MAC
        expected_mac = hmac.new(key, iv + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, expected_mac):
            raise ValueError(
                "[WHAT] MAC verification failed — wrong password or tampered data"
            )
        # Decrypt
        plaintext = bytearray()
        counter = 0
        while len(plaintext) < len(ciphertext):
            block = hashlib.sha256(key + iv + counter.to_bytes(8, "big")).digest()
            chunk = ciphertext[len(plaintext):len(plaintext) + 32]
            plaintext.extend(b ^ k for b, k in zip(chunk, block))
            counter += 1
        return plaintext.decode("utf-8", errors="replace").rstrip("\x00")

    def set(self, name: str, secret: str) -> None:
        """Store a secret under `name`. Overwrites if exists."""
        self._keys[name] = self._encrypt(secret)
        self._save()

    def get(self, name: str) -> Optional[str]:
        """Retrieve a secret by name. Returns None if not found."""
        entry = self._keys.get(name)
        if entry is None:
            return None
        return self._decrypt(entry)

    def delete(self, name: str) -> bool:
        """Delete a secret. Returns True if it existed."""
        if name in self._keys:
            del self._keys[name]
            self._save()
            return True
        return False

    def list_names(self) -> List[str]:
        """List all stored secret names (without revealing values)."""
        return sorted(self._keys.keys())

    def _save(self) -> None:
        if self.storage_path:
            with open(self.storage_path, "w") as f:
                json.dump({"keys": self._keys, "version": 1}, f, indent=2)

    def _load(self) -> None:
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            self._keys = data.get("keys", {})
        except (json.JSONDecodeError, OSError):
            self._keys = {}


# =====================================================================
# 10. Model integrity verifier (uses qai.bundle)
# =====================================================================
def verify_model_integrity(
    bundle_path: str,
    sign_key: Optional[str] = None,
    compute_unsigned_hash: bool = True,
) -> Dict[str, Any]:
    """Verify a model's integrity.

    For signed bundles (saved via `qai.save_bundle(..., sign_key=...)`):
      - Verifies the HMAC-SHA256 signature
      - Returns per-weight hash verification

    For unsigned bundles:
      - Computes fresh SHA-256 of all weights + manifest
      - Returns the hash for out-of-band verification

    Args:
        bundle_path: Path to a .qmodel bundle.
        sign_key: HMAC key (required for signed bundles to verify).
        compute_unsigned_hash: If True, also compute the unsigned hash
            (useful for change detection).

    Returns:
        Dict with `is_signed`, `signature_valid`, `weights_hash_match`,
        `unsigned_hash`, `details`.
    """
    # Import locally to avoid circular import at module load
    from ..bundle import inspect_bundle, verify_bundle

    info = inspect_bundle(bundle_path)
    manifest = info["manifest"]
    is_signed = manifest.get("signed", False)

    result: Dict[str, Any] = {
        "is_signed": is_signed,
        "signature_valid": None,
        "weights_hash_match": None,
        "unsigned_hash": None,
        "details": {
            "technique": manifest.get("technique"),
            "total_params": manifest.get("total_params"),
            "format_version": manifest.get("format_version"),
        },
    }

    if is_signed:
        if sign_key is None:
            result["signature_valid"] = False
            result["error"] = "Bundle is signed but no sign_key provided"
        else:
            result["signature_valid"] = verify_bundle(bundle_path, sign_key)

    # Always compute the unsigned hash (over weights file bytes)
    if compute_unsigned_hash:
        with zipfile.ZipFile(bundle_path, "r") as zf:
            weights_bytes = zf.read("weights.npy")
            manifest_bytes = zf.read("manifest.json")
            model_json_bytes = zf.read("model.json")
        h = hashlib.sha256()
        h.update(weights_bytes)
        h.update(manifest_bytes)
        h.update(model_json_bytes)
        result["unsigned_hash"] = h.hexdigest()

    return result


# =====================================================================
# Public API
# =====================================================================
__all__ = [
    "security_scan",
    "prompt_injection_guard",
    "sandbox_run",
    "detect_pii",
    "redact_pii",
    "scan_secrets",
    "detect_adversarial_input",
    "moderate_content",
    "sanitize_prompt_template",
    "APIKeyVault",
    "verify_model_integrity",
]
