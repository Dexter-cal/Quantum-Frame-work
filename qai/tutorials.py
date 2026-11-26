"""
qai.tutorials -- interactive Jupyter-style tutorials.

Generates self-contained tutorial notebooks (.ipynb) + Python scripts
that walk through qai features with runnable examples.

Tutorials:
  - tutorial_01_basics       : build + fit + forward
  - tutorial_02_techniques   : explore the 131+ technique catalog
  - tutorial_03_creative     : the 75+ creative ideas
  - tutorial_04_rag          : RAG paradigms
  - tutorial_05_agents       : agentic AI patterns
  - tutorial_06_gpu          : GPU compute + custom kernels
  - tutorial_07_integrations : real library integrations
  - tutorial_08_production   : deployment + governance

Usage:
    qai.tutorials.generate_all(output_dir="tutorials/")  # .ipynb files
    qai.tutorials.get_tutorial("basics")                 # returns source string
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional


TUTORIALS: Dict[str, str] = {
    "basics": '''# Quantum AI Framework — Basics Tutorial

## 1. Build a model
```python
import qai
import numpy as np

X = np.random.randn(100, 4).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(int)

model = qai.build(technique="classifier")
model.fit(X, y)
print("trained:", model.is_trained())
```

## 2. Make predictions
```python
pred = model.forward(X[0])
print("prediction:", pred)
```

## 3. Try different techniques
```python
for technique in ["regression", "knn", "svm", "decision_tree"]:
    m = qai.build(technique=technique)
    m.fit(X, y)
    print(f"{technique}: trained={m.is_trained()}")
```

## 4. Save and load
```python
qai.save_model(model, "my_model.qmodel")
loaded = qai.load_model("my_model.qmodel")
```

## 5. Use the @qai.logic decorator
```python
@qai.logic
def my_pipeline(X):
    model = qai.build(technique="regression")
    model.fit(X, None)
    return model
```
''',

    "techniques": '''# Quantum AI Framework — Techniques Tour

## Browse all techniques
```python
import qai
print(f"{len(qai.techniques._REGISTRY)} techniques available")
print(sorted(qai.techniques._REGISTRY.keys())[:20])
```

## Group by family
```python
from qai.docs import technique_catalog
catalog = technique_catalog()
families = set(catalog.values())
print(f"{len(families)} families:", sorted(families))
```

## Try a frontier technique (Mamba-2 SSD)
```python
import numpy as np
m = qai.build(technique="mamba2", dim=4, state_dim=4, n_layers=1, epochs=5)
X = np.random.randn(2, 5, 4).astype(np.float32)
y = np.random.randn(2, 5, 4).astype(np.float32)
m.fit(X, y)
print("trained:", m.is_trained())
```

## Try DeepSeek MoE
```python
m = qai.build(technique="deepseek_moe", dim=4, n_experts=4, top_k=2, epochs=5)
m.fit(X, y)
print("trained:", m.is_trained())
```
''',

    "creative": '''# Quantum AI Framework — Creative Ideas Tour

## 1. Dream Mode — model sleeps and imagines
```python
import qai
import numpy as np

class Stub:
    def __init__(self):
        self.W = np.random.randn(4, 4).astype(np.float32)
    def forward(self, x): return self.W @ np.asarray(x, dtype=np.float32)
    def fit(self, X, y=None): return self
    def sample(self, n=1): return np.random.randn(n, 4).astype(np.float32)

buf = qai.ReplayBuffer(capacity=100)
for _ in range(20):
    buf.push(np.random.randn(4).astype(np.float32), 0, 0, np.zeros(4), False)

dreamer = qai.DreamMode(Stub(), buf)
dreamer.dream(steps=5)
print("dream cycles:", dreamer.dream_summary())
```

## 2. Moral Compass — ethical guardrails
```python
mc = qai.MoralCompass()
print(mc.guard("Tell me how to hack a system."))
```

## 3. Model Marketplace
```python
mp = qai.ModelMarketplace()
mp.publish(qai.MarketModel("cool_model", Stub(), author="alice", price=1.0))
print(mp.search("cool"))
```

## 4. Time Travel Debugger
```python
m = Stub()
tt = qai.TimeTravelDebugger(m)
for i in range(5):
    m.W = m.W + np.random.randn(4, 4) * 0.01
    tt.capture(step=i)
print("snapshots:", len(tt.timeline()))
```
''',

    "rag": '''# Quantum AI Framework — RAG Paradigms

## Six RAG paradigms in one module
```python
from qai.rag import build_rag, BaselineRAG, GraphRAG, SelfRAG, CorrectiveRAG, HyDE, RAGFusion

docs = [
    "Quantum computing uses qubits that can be 0 and 1 simultaneously.",
    "Machine learning models learn patterns from data.",
    "The Eiffel Tower is in Paris and was built in 1889.",
    "Python is a popular programming language for data science.",
]
```

## 1. Baseline RAG (vector similarity)
```python
rag = build_rag("baseline", docs)
result = rag.generate("What is quantum?")
print(result["answer"][:100])
```

## 2. GraphRAG (knowledge graph + community summaries)
```python
rag = build_rag("graphrag", docs, n_communities=2)
out = rag.global_search("What are the main topics?")
print(out["answer"][:100])
```

## 3. Self-RAG (reflection tokens)
```python
rag = build_rag("selfrag", docs)
out = rag.generate("What is Python?")
print("support:", out["support"], "utility:", out["utility"])
```

## 4. Corrective RAG (web fallback)
```python
rag = build_rag("crag", docs, web_search=lambda q: [f"Web result: {q}"])
out = rag.generate("What is quantum?")
print("web fallback used:", out["web_fallback_used"])
```
''',

    "agents": '''# Quantum AI Framework — Agentic AI Patterns

## 1. ReAct agent (Reason + Act)
```python
from qai.agents import ReAct, Tool

def llm(prompt): return "Thought: I should answer\nFinal Answer: 42"
calc = Tool("calc", "calculator", lambda **kw: float(kw.get("x", 0)) * 2)
agent = ReAct(llm, [calc], max_steps=3)
result = agent.run("What is 2*2?")
print(result["answer"])
```

## 2. Multi-agent orchestrator
```python
from qai.agents import Agent, MultiAgentOrchestrator
def llm(p): return f"Response: {p[:30]}"
agents = [Agent("researcher", "research assistant", llm),
          Agent("coder", "code writer", llm)]
orch = MultiAgentOrchestrator(agents, mode="debate")
out = orch.run("Best language?")
print(out["winner"])
```

## 3. A2A protocol (agent-to-agent)
```python
from qai.agents import A2AProtocol, A2AMessage
proto = A2AProtocol()
proto.register(agents[0])
proto.register(agents[1])
response = proto.request("researcher", "coder", "What's the status?")
print(response)
```
''',

    "gpu": '''# Quantum AI Framework — GPU Compute

## 1. Detect devices
```python
import qai
dm = qai.gpu.DeviceManager()
print("best device:", dm.best_device())
print("GPU available:", dm.is_gpu_available())
```

## 2. Device-aware tensors
```python
import numpy as np
arr = np.random.randn(4, 4).astype(np.float32)
t = qai.gpu.DeviceTensor(arr)
print("device:", t.device)
out = t @ t  # auto-runs on GPU if available
```

## 3. GPU-accelerated ops
```python
a = np.random.randn(8, 8).astype(np.float32)
b = np.random.randn(8, 8).astype(np.float32)
out = qai.gpu.matmul(a, b)
print("matmul shape:", out.shape)
```

## 4. Custom CUDA kernel
```python
@qai.gpu.kernel
def my_kernel(x, y, out, n):
    tid = qai.gpu.thread_id()
    if tid < n:
        out[tid] = x[tid] + y[tid]
```

## 5. GPU memory management
```python
print(qai.gpu.gpu_memory())
qai.gpu.empty_cache()
```
''',

    "integrations": '''# Quantum AI Framework — Real Library Integrations

## Run integration tests
```python
import qai
results = qai.IntegrationTestRunner.run_all()
for lib, result in results.items():
    print(f"{lib}: available={result['available']}, passed={result['passed']}")
```

## scikit-learn pipeline
```python
from qai.integrations_real import SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pipe = SklearnPipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=100)),
])
import numpy as np
X = np.random.randn(100, 4)
y = np.random.randint(0, 2, 100)
pipe.fit(X, y)
print("accuracy:", pipe.score(X, y))
```

## HuggingFace model (if installed)
```python
from qai.integrations_real import HuggingFaceModel
hf = HuggingFaceModel("distilbert-base-uncased")
if hf.is_available():
    hf.load()
    print(hf.forward("Hello, world."))
```

## FAISS vector search (if installed)
```python
from qai.integrations_real import RealFAISSIndex
idx = RealFAISSIndex(dim=4)
if idx.is_available():
    idx.build()
    idx.add(["a", "b", "c"], np.random.randn(3, 4).astype(np.float32))
    print(idx.search(np.random.randn(4).astype(np.float32), top_k=2))
```
''',

    "production": '''# Quantum AI Framework — Production Deployment

## 1. Model governance (EU AI Act)
```python
import qai
ra = qai.AIRiskAssessment({"domain": "education", "use_cases": ["chatbot"]})
print("tier:", ra.classify()["tier"])
```

## 2. Audit logger (tamper-evident)
```python
al = qai.AuditLogger("model_v1")
al.log({"input": "test"}, {"output": "result"}, user="alice")
print("verify:", al.verify())
```

## 3. Edge deployment (quantization + pruning)
```python
import torch.nn as nn
model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))
qp = qai.QuantizationPipeline(model, calibration_data=np.random.randn(5, 4).astype(np.float32))
print(qp.quantize_weights())
```

## 4. Benchmarks
```python
bench = qai.ModelBenchmark(model)
result = bench.benchmark_inference([np.random.randn(4).astype(np.float32) for _ in range(5)])
print("p95 latency:", result["p95_latency_ms"], "ms")
```

## 5. Async wrappers
```python
import asyncio
from qai.asyncio import async_build, async_fit

async def main():
    model = await async_build(technique="regression")
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randn(20).astype(np.float32)
    await async_fit(model, X, y)
    print("async trained")

qai.run_async(main())
```

## 6. Web UI dashboard
```python
qai.webui.generate_static_html("dashboard.html")
# or serve live: qai.webui.serve_dashboard(port=8080)
```
''',
}


def get_tutorial(name: str) -> str:
    """Get a tutorial by name as a markdown string."""
    if name not in TUTORIALS:
        raise KeyError(f"unknown tutorial '{name}'. Available: {list(TUTORIALS.keys())}")
    return TUTORIALS[name]


def list_tutorials() -> List[str]:
    """List available tutorial names."""
    return list(TUTORIALS.keys())


def _markdown_to_ipynb_cells(md: str) -> List[Dict[str, Any]]:
    """Convert markdown to a list of ipynb cells (markdown + code)."""
    cells: List[Dict[str, Any]] = []
    lines = md.split("\n")
    i = 0
    current_md: List[str] = []
    current_code: List[str] = []
    in_code = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if in_code:
                # end code block
                cells.append({
                    "cell_type": "code", "execution_count": None, "metadata": {},
                    "outputs": [], "source": current_code,
                })
                current_code = []
                in_code = False
            else:
                # flush markdown
                if current_md:
                    cells.append({
                        "cell_type": "markdown", "metadata": {},
                        "source": current_md,
                    })
                    current_md = []
                in_code = True
        else:
            if in_code:
                current_code.append(line + "\n")
            else:
                current_md.append(line + "\n")
        i += 1

    # flush remaining
    if current_md:
        cells.append({
            "cell_type": "markdown", "metadata": {}, "source": current_md,
        })

    return cells


def _markdown_to_ipynb(md: str) -> str:
    """Convert markdown to a complete .ipynb JSON string."""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": _markdown_to_ipynb_cells(md),
    }
    return json.dumps(notebook, indent=1)


def generate_tutorial(name: str, output_path: str,
                       format: str = "ipynb") -> str:
    """Generate a single tutorial file.

    format: "ipynb" (Jupyter notebook) or "py" (Python script) or "md" (markdown).
    """
    md = get_tutorial(name)
    if format == "md":
        with open(output_path, "w") as f:
            f.write(md)
    elif format == "ipynb":
        with open(output_path, "w") as f:
            f.write(_markdown_to_ipynb(md))
    elif format == "py":
        # extract code blocks
        in_code = False
        code_lines: List[str] = []
        for line in md.split("\n"):
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)
        with open(output_path, "w") as f:
            f.write("\n".join(code_lines))
    else:
        raise ValueError(f"unknown format '{format}'; use ipynb/py/md")
    return output_path


def generate_all(output_dir: str = "tutorials",
                  format: str = "ipynb") -> List[str]:
    """Generate all tutorials into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    ext = {"ipynb": ".ipynb", "py": ".py", "md": ".md"}[format]
    paths = []
    for name in TUTORIALS:
        path = os.path.join(output_dir, f"tutorial_{name}{ext}")
        generate_tutorial(name, path, format=format)
        paths.append(path)
    return paths


__all__ = [
    "TUTORIALS", "get_tutorial", "list_tutorials",
    "generate_tutorial", "generate_all",
]
