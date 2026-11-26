# Guide: Fine-Tuning & Alignment

*You have a base model. Adapt it to your data (PEFT) and your values (alignment).*

> Back to [Guides index](README.md) · Prev: [EVALUATION.md](EVALUATION.md) · Full API: [User Manual §8–9](../USER_MANUAL.md)

---

## What you'll learn

1. Six PEFT methods: LoRA, QLoRA, Adapter, IA³, Prefix, Prompt
2. Eight alignment methods: DPO, GRPO, RLHF, KTO, IPO, SimPO, ORPO, RLVR
3. Which to pick, with real trade-offs

---

## 1. PEFT — parameter-efficient fine-tuning

Train <1% of the weights, keep the rest frozen. Works on any torch-backed
technique (transformer, cnn, neural_network, mamba, ...).

```python
m = qai.build(technique="transformer", vocab_size=1000, d_model=256)
m.train(corpus)                       # SFT first — the usual full fine-tune

# LoRA — low-rank adaptation (the default choice)
m.finetune(X_new, y_new, method="lora", rank=8, alpha=16)

# QLoRA — LoRA on top of a 4-bit quantized base (fits in tiny VRAM)
m.finetune(X_new, y_new, method="qlora", rank=8, quantize="nf4")

# Adapter — bottleneck layers inserted per block
m.finetune(X_new, y_new, method="adapter", bottleneck=64)

# IA³ — per-channel scaling vectors (even smaller than LoRA)
m.finetune(X_new, y_new, method="ia3")

# Prefix / Prompt — learn soft tokens, zero weight changes
m.finetune(X_new, y_new, method="prefix", n_tokens=10)
m.finetune(X_new, y_new, method="prompt", n_tokens=20)
```

**Pick between them:**

| Method | Trainable params | Best for |
|---|---|---|
| LoRA | ~0.1–1% | default for most fine-tuning |
| QLoRA | ~0.1% | big base model, small GPU |
| Adapter | ~1–5% | multiple task heads per base |
| IA³ | ~0.05% | extreme memory budget |
| Prefix / Prompt | ~0.01% | frozen base, tiny data |

Universal finetuning for non-PEFT cases: `qai.finetune_model(model, X, y,
strategy="incremental")` — retrain / incremental / warm-start strategies for
any technique.

---

## 2. Alignment — teach the model *what* to prefer

All trainers accept a torch-backed qai model. Preference data means
`(prompt, chosen, rejected)` triples; unary data means `(prompt, label)` pairs.

```python
m.train(corpus)   # SFT first for DPO-family methods

# DPO — Direct Preference Optimization (paired preferences)
dpo = qai.DPOTrainer(model=m, beta=0.1, lr=5e-7)
dpo.train(preference_data, epochs=1)

# KTO — no pairs needed, unary thumbs-up/down labels
kto = qai.KTOTrainer(model=m, beta=0.1)
kto.train(unary_data, epochs=1)

# SimPO — no reference model needed
simpo = qai.SimPOTrainer(model=m, beta=2.0, gamma=0.5)
simpo.train(preference_data, epochs=1)

# ORPO — combined SFT + alignment, skip the separate SFT stage
orpo = qai.ORPOTrainer(model=m, lambda_=0.5)
orpo.train(preference_data, epochs=1)

# GRPO — group relative policy optimization
grpo = qai.GRPOTrainer(model=m)
grpo.train(prompts, reward_fn=my_reward, epochs=1)

# RLVR — verifiable rewards (math, code — anything you can check)
rlvr = qai.RLVRTrainer(model=m, reward_fn=math_verifier)
rlvr.train(math_problems, epochs=10)

# RLHF — classic reward-model + PPO pipeline
rlhf = qai.RLHFTrainer(model=m, reward_model=rm)

# IPO — identity preference optimization (more conservative than DPO)
ipo = qai.IPOTrainer(model=m)
```

### One-call auto-select

```python
qai.align(model=m, data=data, method="dpo")   # or "kto", "simpo", "orpo", "rlvr", ...
```

**Pick between them:**

| Method | Data needed | Reference model | When |
|---|---|---|---|
| DPO | pairs | yes | you have chosen/rejected pairs |
| KTO | unary | yes | you only have 👍/👎 feedback |
| SimPO | pairs | **no** | want simpler, stronger default |
| ORPO | pairs | **no** | want SFT + alignment in one stage |
| GRPO | prompts + reward | no | group-based RL, reasoning tasks |
| RLVR | prompts + verifier | no | answers can be *checked* (math/code) |
| RLHF | prompts + RM | RM | you trained a reward model |
| IPO | pairs | yes | DPO overfits — go conservative |

---

## 3. Knowledge distillation (related — compress via teaching)

Teacher → student transfer. Full treatment in [COMPRESS.md](COMPRESS.md):

```python
qai.distill(teacher, student, X, y)                    # single teacher
qai.multi_distill([teacher_a, teacher_b], student, X)  # multi-teacher
qai.self_distill(model, X, y)                          # self-distillation
```

---

## Cheat sheet

| Task | Call |
|---|---|
| Default fine-tune | `m.finetune(X, y, method="lora", rank=8)` |
| Tiny GPU | `method="qlora", quantize="nf4"` |
| Many tasks, one base | `method="adapter"` |
| Pairs of preferences | `qai.DPOTrainer` or `qai.SimPOTrainer` |
| Only 👍/👎 | `qai.KTOTrainer` |
| Skip SFT stage | `qai.ORPOTrainer` |
| Math/code rewards | `qai.RLVRTrainer` |
| Auto-pick | `qai.align(model, data, method=...)` |
| Universal finetune | `qai.finetune_model(m, X, y, strategy=...)` |

---

## Common gotchas

- **DPO needs SFT first.** Aligning a random-init model produces garbage.
- **beta matters.** Too high → model drifts from base; too low → no learning. 0.1 is the sane start.
- **QLoRA needs the quantize kwarg** — `quantize="nf4"` is the standard choice.
- **RLVR reward must be checkable.** If you can't verify the answer programmatically, use DPO/KTO instead.

---

*Next: [SERVE_DEPLOY.md](SERVE_DEPLOY.md) to ship your aligned model.*
