# Guide: Memory & RAG — Models That Remember

*Working memory, long-term retrieval, persistent knowledge, and retrieval-augmented generation.*

> Back to [Guides index](README.md) · Related: [AGENTS_MCP.md](AGENTS_MCP.md) · Data loading: [DATA.md](DATA.md)

---

## What you'll learn

1. The three memory tiers (`qai.memory`)
2. Vector retrieval with conflict detection
3. Permanent knowledge across sessions
4. End-to-end RAG pipeline
5. Context-span configuration for long contexts

---

## 1. Three-tier memory (`qai.memory`)

| Tier | Class | Lifespan | Use |
|---|---|---|---|
| Working | `WorkingMemory` | current session, sliding window | recent context |
| Retrieval | `RetrievalMemory` | persistent vector store | long-term facts |
| Weights | `WeightsMemory` | model checkpoints | weight snapshots, drift |

```python
from qai.memory import Memory, WorkingMemory, RetrievalMemory, WeightsMemory

mem = Memory()   # coordinates all three tiers
```

### Working memory — the sliding window

```python
wm = WorkingMemory(max_items=100, policy="lru")   # lru / lfu / time-decay / importance / fifo
wm.add("user asked about refund policy")
wm.add("user is from Berlin", importance=0.8)
wm.context(k=5)          # top-5 items for the next prompt
```

### Retrieval memory — the vector store

```python
rm = RetrievalMemory()
rm.add("Refunds are processed within 14 business days.")
rm.add("The office closes at 5pm on Fridays.")

hits = rm.search("How long do refunds take?", k=3)   # cosine similarity
# → ranked matches with scores
```

**Conflict detection is built in:** adding a fact that contradicts an existing
one is flagged instead of silently duplicated.

```python
rm.add("Refunds are processed within 30 business days.")   # conflicts with the 14-day fact
# → conflict reported; you decide which to keep
```

### Weights memory — snapshots and drift

```python
weights_mem = WeightsMemory(model=m)
weights_mem.snapshot("before-finetune-v2")
weights_mem.diff("before-finetune-v2")   # how far have the weights moved?
weights_mem.restore("before-finetune-v2")
```

### Auto-promotion (facts graduate tiers)

Facts repeated across working-memory sessions are auto-promoted into retrieval
memory. Every tier keeps an audit log.

---

## 2. Knowledge accumulation across sessions

```python
from qai.self_improving import KnowledgeAccumulator

ka = KnowledgeAccumulator()
ka.learn(question, answer, confidence=0.9)
ka.recall(similar_question)   # retrieves learned knowledge
```

Pair with `SelfEvaluator` (find blind spots) and `ContinuousLearner`
(auto-retrain on drift) — see the self-improving section in
[User Manual](../USER_MANUAL.md).

---

## 3. RAG — retrieval-augmented generation (`qai.RAGPipeline`)

```python
from qai import RAGPipeline

rag = RAGPipeline(
    retriever=rm,                 # a RetrievalMemory
    model=my_language_model,      # any qai LM
)

# Add documents (chunks come straight from qai.data loaders)
rag.add_documents([
    "Refunds are processed within 14 business days.",
    "Premium accounts get priority support.",
])

answer = rag.query("How long do refunds take?")
print(answer.text)       # grounded answer citing retrieved context
print(answer.sources)    # which chunks were used

# Every query is logged (question, retrieved chunks, response)
for entry in rag.query_log():
    print(entry)
```

Document ingestion pairs with [DATA.md](DATA.md):

```python
ds = qai.Dataset.from_pdf("handbook.pdf")     # text chunks
rag.add_documents(ds.texts())
```

---

## 4. Long-context configuration (`memory_span`)

For transformer-based models, control context spans explicitly:

```python
from qai.memory import MemorySpanConfig, configure_memory_span

cfg = MemorySpanConfig(
    memory_span=8192,       # total span
    focus_width=2048,       # tightly attended region
    window_size=4096,       # sliding window
)
configure_memory_span(m, cfg)   # applies rope_theta / rope_scaling / max_seq_len
```

Alias parameters (`focus_width`, `position_scale`, `stretch_factor`) map onto
the underlying RoPE settings — see [User Manual](../USER_MANUAL.md).

---

## 5. Persist everything

```python
rm.save("knowledge.qmem")
rm2 = RetrievalMemory.load("knowledge.qmem")

weights_mem.snapshot_to_disk("ckpt/")
```

`.qmodel` bundles can carry a tokenizer + logic + weights together — see
[SERVE_DEPLOY.md §5](SERVE_DEPLOY.md).

---

## Cheat sheet

| Task | Call |
|---|---|
| Recent context | `WorkingMemory(policy="lru")` |
| Long-term facts | `RetrievalMemory` + `.add()` + `.search()` |
| Contradiction check | built into `RetrievalMemory.add()` |
| Weight snapshots | `WeightsMemory(model).snapshot(label)` |
| Weight drift | `WeightsMemory.diff(label)` |
| Permanent knowledge | `KnowledgeAccumulator.learn / recall` |
| RAG | `qai.RAGPipeline(retriever, model)` |
| Load docs for RAG | `qai.Dataset.from_pdf(...)` → `rag.add_documents(...)` |
| Long context | `configure_memory_span(m, MemorySpanConfig(...))` |
| Persist memory | `rm.save(path)` / `RetrievalMemory.load(path)` |

---

## Common gotchas

- **Chunk size matters for RAG.** `from_document()` chunks by structure — very small chunks lose context, huge ones dilute retrieval.
- **Conflicts need a policy.** RetrievalMemory flags contradictions; decide keep-new vs keep-both and enforce it in your app.
- **WorkingMemory is not storage.** It's a window — anything you need later must go to RetrievalMemory (or get auto-promoted).
- **Restoring old weights undoes fine-tuning.** Snapshot labels are your undo log — name them well.

---

*Next: [SECURITY_GOVERNANCE.md](SECURITY_GOVERNANCE.md) or [AGENTS_MCP.md](AGENTS_MCP.md) to give the memory a brain.*
