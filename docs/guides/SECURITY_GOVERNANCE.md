# Guide: Security & Governance — Defensive AI

*Protect the model, the data, and the people in the data.*

> Back to [Guides index](README.md) · Related: [SERVE_DEPLOY.md](SERVE_DEPLOY.md) (signing) · [MEMORY_RAG.md](MEMORY_RAG.md)

---

## What you'll learn

1. The ten-tool security module (`qai.security`)
2. Prompt-injection and adversarial-input defense
3. PII and secret scanning
4. Sandboxing untrusted code
5. Fairness auditing and drift detection (governance)
6. Model watermarking and integrity verification

---

## 1. The security toolkit (`qai.security`)

```python
from qai.security import (
    security_scan,                # dataset poisoning detection
    prompt_injection_guard,       # 20+ heuristic patterns (OWASP LLM01)
    sandbox_run,                  # restricted subprocess
    detect_pii, redact_pii,       # emails, phones, SSNs, cards, IPs, JWTs...
    scan_secrets,                 # AWS, GitHub, Google, Slack, Stripe, OpenAI, Anthropic keys
    detect_adversarial_input,     # unicode tricks, repetition, encoding attacks
    moderate_content,             # rule-based toxicity/profanity/hate
    sanitize_prompt_template,     # strip {{var}}, HTML tags, zero-width chars
    APIKeyVault,                  # encrypted at-rest key storage
    verify_model_integrity,       # signed .qmodel verification
)
```

### Poisoning detection (before training)

```python
report = security_scan(X, y)
# flags: statistical outliers, label flips, duplicates, class imbalance
if not report["clean"]:
    print(report["findings"])
```

### Prompt injection (before your LLM sees user text)

```python
verdict = prompt_injection_guard(user_text)
if verdict["blocked"]:
    log.warning(f"Injection attempt: {verdict['patterns_matched']}")
```

### Adversarial input (beyond injection)

```python
verdict = detect_adversarial_input(text)
# catches: excessive length, repetition, control chars, zero-width/RTL/LTR
# unicode, mixed scripts, hex/URL/HTML-entity encoding
```

### PII and secrets

```python
findings = detect_pii(text)      # structured findings with spans
clean = redact_pii(text)         # masked version for logs/training

leaks = scan_secrets(config_text)  # find committed API keys before the scanners do
```

### Content moderation

```python
verdict = moderate_content(text)    # toxicity / profanity / hate flags
```

### Prompt template hygiene

```python
safe = sanitize_prompt_template(template)  # strips {{var}}, HTML, zero-width, control chars
```

---

## 2. Sandboxing untrusted code

```python
from qai.security import sandbox_run

result = sandbox_run(
    untrusted_code,
    timeout_s=5,                  # hard kill
    allowed_modules=["math", "json"],
    workdir="/tmp/sandbox",       # isolated working dir
)
print(result["stdout"], result["error"])
```

Restricted `__import__` and limited builtins — pairs with MCP's
`ExecTool`/`CodeEvalTool` (see [AGENTS_MCP.md](AGENTS_MCP.md)).

---

## 3. APIKeyVault (encrypted at rest)

```python
from qai.security import APIKeyVault

vault = APIKeyVault(master_password)     # PBKDF2-HMAC-SHA256 derivation,
vault.set("openai", sk)                  # SHA-256 CTR keystream + HMAC-SHA256 MAC
sk = vault.get("openai")                 # no external crypto deps
```

---

## 4. Model integrity & watermarking

```python
# Sign on save, verify on load (HMAC-SHA256):
m.save("model.qmodel", sign=True)
qai.Model.load("model.qmodel", verify=True)

# Out-of-band hash verification:
digest = verify_model_integrity("model.qmodel")   # returns hash + signature status

# Publisher trust:
from qai.bundle import trust_add
trust_add(pubkey, "trusted-publisher")

# Invisible watermark inside the weights:
from qai.governance import watermark_model
watermark_model(m, owner_id="my-org")   # survives export round-trips
```

Runtime-side signing, staging, and rollback live in
[SERVE_DEPLOY.md §4–5](SERVE_DEPLOY.md).

---

## 5. Governance — fairness, drift, model cards

### Fairness audit

```python
from qai.governance import fairness_audit

report = fairness_audit(
    m, X, y,
    sensitive_features=["gender", "region"],
)
# → demographic_parity_difference
#   equalized_odds_difference
#   disparate_impact_ratio
```

### Drift detection (production data moving under you)

```python
from qai.governance import detect_drift

drift = detect_drift(X_train, X_production)
if drift["significant"]:
    retrain()   # or let ContinuousLearner do it — see below
```

### Model cards (auto-filled)

```python
from qai.core.model_card import ModelCard

card = ModelCard.from_model(m, X, y)
print(card.to_markdown())   # intended use, metrics, bias audit, limitations
```

### Self-improving guardrails

```python
from qai.self_improving import ContinuousLearner, ModelGuardian, SelfEvaluator

guardian = ModelGuardian(model=m)
guardian.watch(X_production_stream)   # drift → auto-retrain → A/B → promote/rollback
SelfEvaluator(m).find_blind_spots(X, y)   # where is the model weakest + calibration errors
```

### Risk scoring (multi-factor)

```python
from qai.core import RiskScorer

risk = RiskScorer()
score = risk.score(text)   # toxicity + PII + hallucination + injection + length factors
```

---

## Defense-in-depth recipe (production)

```python
# 1. Train-time
security_scan(X, y)                    # poison check
card = ModelCard.from_model(m, X, y)   # document it
watermark_model(m, "my-org")           # mark it

# 2. Ship-time
m.save("model.qmodel", sign=True)      # sign it
# publisher trust keys distributed separately

# 3. Serve-time (per request)
if prompt_injection_guard(text)["blocked"]: return 400
if detect_adversarial_input(text)["suspicious"]: return 400
clean = redact_pii(text)               # don't log PII
moderate_content(clean)                # content check

# 4. Runtime
wrapper.enable_auto_rollback(...)      # degrade → roll back (see SERVE_DEPLOY.md)
detect_drift(X_train, X_live)          # watch for data drift
guardian.watch(X_live_stream)          # auto-retrain pipeline
```

---

## Cheat sheet

| Task | Call |
|---|---|
| Check training data | `qai.security.security_scan(X, y)` |
| Block injection | `prompt_injection_guard(text)` |
| Find PII | `detect_pii(text)` / `redact_pii(text)` |
| Find leaked keys | `scan_secrets(text)` |
| Sandbox code | `sandbox_run(code, timeout_s=5, allowed_modules=[...])` |
| Store keys safely | `APIKeyVault(password).set(name, key)` |
| Sign a model | `m.save(path, sign=True)` |
| Verify a model | `verify_model_integrity(path)` |
| Watermark | `qai.governance.watermark_model(m, owner_id)` |
| Fairness audit | `qai.governance.fairness_audit(m, X, y, sensitive_features=...)` |
| Detect drift | `qai.governance.detect_drift(X_train, X_live)` |
| Auto model card | `ModelCard.from_model(m, X, y).to_markdown()` |
| Full guardian | `qai.self_improving.ModelGuardian(model)` |
| Risk score text | `RiskScorer().score(text)` |

---

## Common gotchas

- **`prompt_injection_guard` is heuristic.** It catches known patterns; novel injections need human review + output-side checks. Defense in depth, not silver bullets.
- **`redact_pii` before logging, always.** Logs are the most-leaked artifact in practice.
- **`sandbox_run` is process-level isolation, not a VM.** For hostile multi-tenant code, run it in a container too.
- **Fairness metrics need a sensitive-attribute column.** If you don't collect it, you can't audit it — plan data schema accordingly.

---

*Next: [MLOPS.md](MLOPS.md) — operationalize all of this, or [MULTIMODAL.md](MULTIMODAL.md) for non-tabular data.*
