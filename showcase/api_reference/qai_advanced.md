# `qai.advanced`

> qai.advanced -- RAG pipeline, Knowledge Distillation, and AI Agent framework.

**Module summary:**
- Classes: 3 | Functions: 0 | Constants: 0
- Source: `qai/advanced.py`

## Table of Contents

**Classes:**
- [`RAGPipeline`](#ragpipeline)
- [`KnowledgeDistiller`](#knowledgedistiller)
- [`AIAgent`](#aiagent)

## Classes

<a id="ragpipeline"></a>
### `RAGPipeline`  
`CLASS` | `qai.advanced.RAGPipeline`

**Signature:**
```python
class RAGPipeline
```

> End-to-end Retrieval-Augmented Generation pipeline.

End-to-end Retrieval-Augmented Generation pipeline.

Combines a document store (qai.memory.RetrievalMemory) with a language
model (any qai generation model: Transformer, Mamba, LSTM, etc.) to
answer questions using retrieved context.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `Any` | `` |  |
| `tokenizer` | `Any` | `` |  |
| `retriever` | `Optional[Any]` | `None` |  |
| `top_k` | `int` | `3` |  |
| `max_context_chars` | `int` | `500` |  |
| `prompt_template` | `Optional[str]` | `None` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(model: Any, tokenizer: Any, retriever: Optional[Any] = None, top_k: int = 3, max_context_chars: int = 500, prompt_template: Optional[str] = None)` |  |
| `add_document()` | `(text: str, metadata: Optional[Dict] = None)` | Add a document to the retrieval store. Returns the document ID. |
| `add_documents()` | `(texts: List[str], metadatas: Optional[List[Dict]] = None)` | Add multiple documents at once. |
| `retrieve()` | `(query: str, k: Optional[int] = None)` | Retrieve top-k documents for a query. Returns list of {text, score, metadata}. |
| `construct_prompt()` | `(query: str, retrieved_docs: List[Dict])` | Build the prompt with retrieved context. |
| `query()` | `(question: str, max_new_tokens: int = 50, temperature: float = 0.3, return_context: bool = False)` | Answer a question using RAG. |
| `query_log()` | `(last_n: int = 50)` |  |

_Defined in `qai/advanced.py:30`._

---

<a id="knowledgedistiller"></a>
### `KnowledgeDistiller`  
`CLASS` | `qai.advanced.KnowledgeDistiller`

**Signature:**
```python
class KnowledgeDistiller
```

> Teacher→Student model compression (spec Section 23).

Teacher→Student model compression (spec Section 23).

Trains a smaller "student" model to mimic a larger "teacher" model by
matching the teacher's soft output probabilities (KL divergence) in
addition to the true labels (cross-entropy).

The classic loss: L = alpha * CE(student, true_labels) + (1-alpha) * T² * KL(student_soft, teacher_soft)
where T is the temperature for softening probabilities.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `teacher` | `Any` | `` |  |
| `student` | `Any` | `` |  |
| `temperature` | `float` | `4.0` |  |
| `alpha` | `float` | `0.5` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(teacher: Any, student: Any, temperature: float = 4.0, alpha: float = 0.5)` | Args: |
| `distill()` | `(X, y, epochs: int = 30, lr: float = 0.001, batch_size: int = 32, verbose: bool = False)` | Train the student model using knowledge distillation. |
| `losses()` | `()` |  |
| `compression_ratio()` | `()` | Ratio of teacher params to student params (higher = more compression). |

_Defined in `qai/advanced.py:160`._

---

<a id="aiagent"></a>
### `AIAgent`  
`CLASS` | `qai.advanced.AIAgent`

**Signature:**
```python
class AIAgent
```

> Autonomous AI agent with multi-step tool use + planning.

Autonomous AI agent with multi-step tool use + planning.

An agent receives a task, plans steps, executes tools, observes results,
and iterates until the task is complete or max steps is reached.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `name` | `str` | `'qai-agent'` |  |
| `tools` | `Optional[Dict[str, Callable]]` | `None` |  |
| `max_steps` | `int` | `10` |  |
| `retry_failed` | `bool` | `True` |  |
| `max_retries` | `int` | `2` |  |
| `verbose` | `bool` | `False` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(name: str = 'qai-agent', tools: Optional[Dict[str, Callable]] = None, max_steps: int = 10, retry_failed: bool = True, max_retries: int = 2, verbose: bool = False)` | Args: |
| `add_tool()` | `(name: str, fn: Callable)` | Add a tool to the agent's toolkit. |
| `remove_tool()` | `(name: str)` |  |
| `list_tools()` | `()` |  |
| `run()` | `(task: str, **initial_context)` | Execute a task autonomously. |
| `execution_log()` | `()` |  |
| `thinking()` | `()` |  |

_Defined in `qai/advanced.py:346`._

---
