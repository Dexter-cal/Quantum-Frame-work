# Guide: Compression — Quantization, Distillation, Pruning, NAS

*Make the model smaller, faster, and cheaper — with controlled quality loss.*

> Back to [Guides index](README.md) · Prev: [EVALUATION.md](EVALUATION.md) · Ship after: [SERVE_DEPLOY.md](SERVE_DEPLOY.md)

---

## What you'll learn

1. Quantization: int8, int4 (NF4), GPTQ, AWQ, GGUF, EXL2 + QAT
2. Knowledge distillation v2: single/multi-teacher, self-distillation
3. Pruning: structured sparsity
4. NAS: search for the right architecture instead of guessing

---

## 1. Quantization

```python
# int8 — dynamic quantization, no retraining needed
qai.quantize_dynamic(m)

# int4 — NF4 (the QLoRA data type)
qai.quantize_to_int4(m, "nf4")

# Undo (lossy — keep your original!)
qai.dequantize(m)
```

**Full format matrix:**

| Format | Bits | Use case |
|---|---|---|
| int8 (dynamic) | 8 | CPU serving, default choice |
| int4 NF4 | 4 | QLoRA base, tiny GPU |
| GPTQ | 4/3/2 | GPU inference, per-layer calibration |
| AWQ | 4 | activation-aware, protects salient weights |
| GGUF | 2–8 | llama.cpp / Ollama |
| EXL2 | variable | ExLlamaV2 |

```python
m.save("model.gguf", format="gguf")     # for llama.cpp / Ollama
m.save("model.onnx", format="onnx")     # for ONNX Runtime
```

### QAT — quantization-aware training (minimize quality loss)

```python
qat = qai.QATTrainer(model=m, quantization="int8")
qat.train(X, y, epochs=3)         # model learns to live in low precision
qai.quantize_dynamic(m)           # now minimal quality loss
```

### Precision management

```python
pm = qai.PrecisionManager()
print(pm.recommend_precision())   # bf16 on Ampere+, fp16 on Volta+, fp32 on CPU

qai.set_precision(m, "bf16")
qai.compare_precision(m, X[0])    # side-by-side fp32 vs fp16 vs bf16
```

---

## 2. Knowledge distillation v2 (`qai.distillation_v2`)

**What:** make a small model (student) behave like a big one (teacher) by
training it on the teacher's soft outputs.

### Single teacher

```python
from qai.distillation_v2 import Distiller, distill

teacher = qai.build(technique="transformer", d_model=512, n_layers=8); teacher.train(corpus)
student = qai.build(technique="neural_network", n_layers=2)

result = distill(teacher, student, X, y, temperature=4.0, alpha=0.7)
# temperature: softens teacher logits (4.0 = classic Hinton setting)
# alpha:       balance soft-target loss vs hard-label loss
print(result.compression_ratio())   # e.g. 12.5x smaller
```

### Multi-teacher

```python
from qai.distillation_v2 import MultiTeacherDistiller, multi_distill

result = multi_distill([teacher_a, teacher_b], student, X, y)
# teachers are weighted by their validation performance
```

### Self-distillation

```python
from qai.distillation_v2 import self_distill

result = self_distill(model, X, y)   # the model teaches a copy of itself
```

Works with any torch-backed qai technique (transformer, neural_network, cnn...)
and falls back gracefully for sklearn-style models. Config via
`DistillationConfig` — temperature, alpha, epochs, lr, intermediate-feature
matching (feature-based distillation).

---

## 3. Pruning

```python
from qai.core.quantize import prune_structured_sparsity

prune_structured_sparsity(m, sparsity=0.3)   # zero out 30% of weights, structured
```

Pair with fine-tuning afterward to recover accuracy (see
[FINETUNING_ALIGNMENT.md](FINETUNING_ALIGNMENT.md)).

---

## 4. NAS — Neural Architecture Search (`qai.nas_search`)

**What:** instead of guessing layer counts and widths, search for them.

```python
from qai.nas_search import RandomSearch, GridSearch, EvolutionarySearch, ProgressiveGrowth

space = {"n_layers": [2, 4, 8], "d_model": [64, 128, 256], "dropout": [0.0, 0.1, 0.3]}

# Random search — good default
nas = RandomSearch(space, n_trials=20)

# Exhaustive — 3x3x3 = 27 trials
nas = GridSearch(space)

# Evolutionary — mutate + select over generations
nas = EvolutionarySearch(space, population=10, generations=5)

# Progressive — start small, grow what works
nas = ProgressiveGrowth(space)

result = nas.search(build_fn=qai.build, X=X, y=y, metric="accuracy")
best = result.best_spec        # ArchitectureSpec
print(result.history)          # TrialResult per trial
```

NAS + distillation is the power combo: search a small architecture, then
distill your big model into it.

---

## The standard compression recipe

```python
# 1. Train big
big = qai.build(technique="transformer", d_model=512, n_layers=8)
big.train(corpus)

# 2. Search small architecture (optional)
nas = RandomSearch({"n_layers": [2, 4], "d_model": [64, 128]}, n_trials=8)
small_spec = nas.search(build_fn=qai.build, X=X, y=y).best_spec

# 3. Distill big → small
student = qai.build(technique="neural_network", **small_spec.to_kwargs())
qai.distill(big, student, X, y, temperature=4.0)

# 4. QAT on the student
qai.QATTrainer(model=student, quantization="int8").train(X, y, epochs=2)

# 5. Quantize + export
qai.quantize_dynamic(student)
student.save("student.qmodel", sign=True)
```

---

## Cheat sheet

| Task | Call |
|---|---|
| int8, no retraining | `qai.quantize_dynamic(m)` |
| int4 NF4 | `qai.quantize_to_int4(m, "nf4")` |
| llama.cpp / Ollama | `m.save("model.gguf", format="gguf")` |
| Minimal quality loss | `qai.QATTrainer(...).train(...)` before quantizing |
| Teacher → student | `qai.distill(teacher, student, X, y)` |
| Multiple teachers | `qai.multi_distill([t1, t2], student, X, y)` |
| Self-distill | `qai.self_distill(m, X, y)` |
| Structured pruning | `prune_structured_sparsity(m, sparsity=0.3)` |
| Search architecture | `qai.nas_search.RandomSearch(space).search(...)` |
| Compare precisions | `qai.compare_precision(m, x)` |
| Recommend precision | `qai.PrecisionManager().recommend_precision()` |

---

## Common gotchas

- **Always keep the original.** `dequantize` is lossy — your fp32 weights are the only ground truth.
- **QAT before quantize, not after.** Post-hoc int8 without QAT costs 1–3% accuracy on small models.
- **Distillation temperature 4.0 is the classic** — lower it only if the teacher is undertrained.
- **NAS trials are full trainings.** Budget accordingly: 20 random trials = 20 train runs.

---

*Next: [SERVE_DEPLOY.md](SERVE_DEPLOY.md) — version, sign, and serve the compressed model.*
