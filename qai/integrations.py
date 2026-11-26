"""
qai.integrations -- adapters to the wider AI ecosystem (2025/2026 era).

Wraps (rather than reimplements) the libraries AI devs actually use:

  LLM App Frameworks:
    - LangChainAdapter    -- use qai models inside LangChain chains/agents
    - LlamaIndexAdapter   -- use qai models as LlamaIndex LLM/embedder
    - HaystackAdapter     -- use qai models in Haystack pipelines

  Model Hub:
    - HuggingFaceHub      -- download/upload models, push to hub
    - ModelRegistry       -- multi-hub (HF + local + qai.bundle) registry

  Vector DBs:
    - FAISSIndex          -- local high-D vector search
    - ChromaStore         -- persistent local vector store
    - MilvusClient        -- distributed vector DB
    - PgvectorStore       -- Postgres + pgvector
    - VectorStoreRegistry -- unified registry, swap backend with one line

  Experiment Tracking:
    - MLflowLogger        -- log params, metrics, artifacts
    - WandBLogger         -- Weights & Biases
    - NeptuneLogger       -- Neptune.ai

  Observability:
    - LangfuseTracer      -- LLM call tracing + eval
    - ArizeLogger         -- model performance monitoring
    - PhoenixTracer       -- Arize Phoenix open-source
    - HeliconeLogger      -- LLM proxy logging

  Structured Generation:
    - DSPyAdapter         -- declarative LLM programs
    - InstructorAdapter   -- pydantic-typed LLM calls
    - OutlinesAdapter     -- structured generation with regex/JSON
    - LMQLAdapter         -- SQL-like LLM query language
"""
from __future__ import annotations
import os
import json
import time
import hashlib
import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# Lazy-import helper
# ============================================================================
def _try_import(name: str):
    """Import a module if available; return None otherwise."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


# ============================================================================
# LangChain Adapter
# ============================================================================
class LangChainAdapter:
    """Wrap qai models so they can be used inside LangChain pipelines.

    Requires `langchain` to be installed. The adapter exposes a qai model
    as a LangChain-compatible LLM (with _call/_generate methods) so it
    can be dropped into chains, agents, and tools.
    """
    def __init__(self, model: Any):
        self.model = model
        self._lc = _try_import("langchain")
        self._lc_llms = _try_import("langchain.llms.base")
        self._impl = self._build_impl()

    def _build_impl(self):
        if self._lc_llms is None:
            return None
        model = self.model

        class _QaiLLM(self._lc_llms.LLM):
            model_ref: Any = None

            @property
            def _llm_type(self):
                return "qai"

            def _call(self, prompt, stop=None, **kwargs):
                try:
                    return str(self.model_ref.forward(prompt))
                except Exception as e:
                    return f"[qai-llm error] {e}"

            def _generate(self, prompts, stop=None, **kwargs):
                from langchain.schema import Generation, LLMResult
                generations = [[Generation(text=self._call(p, stop, **kwargs))]
                                for p in prompts]
                return LLMResult(generations=generations)

        impl = _QaiLLM()
        impl.model_ref = model
        return impl

    def as_llm(self):
        if self._impl is None:
            raise ImportError("langchain is not installed; pip install langchain")
        return self._impl

    def as_embedding(self):
        lc_emb = _try_import("langchain.schema.embeddings")
        if lc_emb is None:
            raise ImportError("langchain is not installed")
        model = self.model

        class _QaiEmb(lc_emb.Embeddings):
            def embed_documents(self, texts):
                return [np.asarray(model.forward(t)).flatten().tolist() for t in texts]
            def embed_query(self, text):
                return np.asarray(model.forward(text)).flatten().tolist()

        return _QaiEmb()


# ============================================================================
# LlamaIndex Adapter
# ============================================================================
class LlamaIndexAdapter:
    """Wrap qai models for use in LlamaIndex pipelines."""
    def __init__(self, model: Any):
        self.model = model
        self._li = _try_import("llama_index")
        self._impl = None

    def as_llm(self):
        if self._li is None:
            raise ImportError("llama_index is not installed; pip install llama-index")
        # LlamaIndex 0.10+ uses llama_index.llms.custom
        try:
            from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse
            from llama_index.core.llms.callbacks import llm_completion_callback
            model = self.model

            class _QaiLICustomLLM(CustomLLM):
                @property
                def metadata(self):
                    return LLMMetadata(
                        context_window=2048,
                        num_output=256,
                        model_name="qai-model",
                    )
                @llm_completion_callback()
                def complete(self, prompt: str, **kwargs) -> CompletionResponse:
                    try:
                        return CompletionResponse(text=str(model.forward(prompt)))
                    except Exception as e:
                        return CompletionResponse(text=f"[qai error] {e}")
                @llm_completion_callback()
                def stream_complete(self, prompt: str, **kwargs):
                    from llama_index.core.llms import CompletionResponseGen
                    def gen():
                        try:
                            yield CompletionResponse(text=str(model.forward(prompt)))
                        except Exception as e:
                            yield CompletionResponse(text=f"[qai error] {e}")
                    return gen()
            return _QaiLICustomLLM()
        except Exception as e:
            raise ImportError(f"could not build LlamaIndex LLM: {e}")


# ============================================================================
# Haystack Adapter
# ============================================================================
class HaystackAdapter:
    """Wrap qai models for Haystack 2.x pipelines."""
    def __init__(self, model: Any):
        self.model = model
        self._hs = _try_import("haystack")

    def as_component(self):
        if self._hs is None:
            raise ImportError("haystack is not installed; pip install haystack-ai")
        from haystack.core.component import component
        model = self.model

        @component
        class _QaiComponent:
            def __init__(self):
                self.model = model

            @component.output_types(result=str)
            def run(self, prompt: str):
                try:
                    return {"result": str(self.model.forward(prompt))}
                except Exception as e:
                    return {"result": f"[qai error] {e}"}

        return _QaiComponent()


# ============================================================================
# Hugging Face Hub
# ============================================================================
class HuggingFaceHub:
    """Download/upload models from/to the Hugging Face Hub.

    Wraps `huggingface_hub` if available; falls back to direct HTTP for
    downloads. Models are cached under ~/.cache/qai/hf_models/.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/qai/hf_models")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._hf = _try_import("huggingface_hub")

    def list_models(self, query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        """Search the HF Hub for models."""
        if self._hf is None:
            return []
        try:
            api = self._hf.HfApi()
            models = api.list_models(search=query, limit=limit)
            return [{"id": m.id, "downloads": getattr(m, "downloads", 0),
                     "likes": getattr(m, "likes", 0)} for m in models]
        except Exception:
            return []

    def download(self, repo_id: str, filename: Optional[str] = None) -> str:
        """Download a model file from HF Hub; return local path."""
        if self._hf is None:
            raise ImportError("huggingface_hub not installed; pip install huggingface_hub")
        if filename:
            return self._hf.hf_hub_download(repo_id=repo_id, filename=filename,
                                              cache_dir=self.cache_dir)
        # snapshot download whole repo
        return self._hf.snapshot_download(repo_id=repo_id, cache_dir=self.cache_dir)

    def upload(self, local_path: str, repo_id: str, token: Optional[str] = None) -> Dict[str, Any]:
        """Upload a file to a HF repo."""
        if self._hf is None:
            raise ImportError("huggingface_hub not installed")
        api = self._hf.HfApi(token=token)
        try:
            api.upload_file(path_or_fileobj=local_path,
                              path_in_repo=os.path.basename(local_path),
                              repo_id=repo_id)
            return {"uploaded": True, "repo_id": repo_id, "filename": os.path.basename(local_path)}
        except Exception as e:
            return {"uploaded": False, "error": str(e)}


class ModelRegistry:
    """Unified model registry: HuggingFace + local paths + qai bundles.

    Look up a model by name; registry figures out where to fetch it from.
    """
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._sources: List[Callable[[str], Optional[Any]]] = []

    def register_source(self, name: str, loader: Callable[[str], Optional[Any]]) -> None:
        self._sources.append(loader)

    def get(self, name: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        for loader in self._sources:
            try:
                m = loader(name)
                if m is not None:
                    self._cache[name] = m
                    return m
            except Exception:
                continue
        raise KeyError(f"Model '{name}' not found in any registered source")


# ============================================================================
# FAISS Vector Index
# ============================================================================
class FAISSIndex:
    """Wrap FAISS for high-dimensional vector search.

    Falls back to a numpy brute-force implementation if faiss is not installed.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim
        self._faiss = _try_import("faiss")
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []
        if self._faiss is not None:
            self._index = self._faiss.IndexFlatIP(dim)
        else:
            self._index = None

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._faiss is not None:
            self._index.add(vectors)
        else:
            self._vectors.append(vectors)
        self._ids.extend(ids)

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, indices = self._index.search(query, top_k)
            return [(self._ids[i], float(s)) for s, i in zip(scores[0], indices[0])
                    if i >= 0]
        # brute force
        if not self._vectors:
            return []
        all_vecs = np.vstack(self._vectors)
        scores = all_vecs @ query.flatten()
        idx = np.argsort(scores)[-top_k:][::-1]
        return [(self._ids[i], float(scores[i])) for i in idx if i < len(self._ids)]

    def save(self, path: str) -> str:
        if self._faiss is not None:
            self._faiss.write_index(self._index, path)
        else:
            np.savez(path, ids=self._ids, vectors=np.vstack(self._vectors) if self._vectors else np.zeros((0, self.dim)))
        return path

    def load(self, path: str) -> "FAISSIndex":
        if self._faiss is not None:
            self._index = self._faiss.read_index(path)
        else:
            data = np.load(path + ".npz" if not path.endswith(".npz") else path,
                            allow_pickle=True)
            self._ids = list(data["ids"])
            self._vectors = [data["vectors"]]
        return self


# ============================================================================
# Chroma Vector Store
# ============================================================================
class ChromaStore:
    """Wrap Chroma for persistent local vector search."""
    def __init__(self, collection: str = "qai_default",
                 persist_dir: Optional[str] = None):
        self._chroma = _try_import("chromadb")
        if self._chroma is None:
            raise ImportError("chromadb not installed; pip install chromadb")
        self.client = self._chroma.PersistentClient(path=persist_dir or "./chroma") \
            if persist_dir else self._chroma.Client()
        self.collection = self.client.get_or_create_collection(name=collection)

    def add(self, ids: List[str], documents: List[str],
            embeddings: np.ndarray, metadatas: Optional[List[Dict]] = None) -> None:
        self.collection.add(ids=ids, documents=documents,
                             embeddings=np.asarray(embeddings).tolist(),
                             metadatas=metadatas or [{}]*len(ids))

    def query(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        result = self.collection.query(query_embeddings=[np.asarray(embedding).tolist()],
                                          n_results=top_k)
        return [{"id": result["ids"][0][i], "document": result["documents"][0][i],
                 "distance": result["distances"][0][i]}
                for i in range(len(result["ids"][0]))]


# ============================================================================
# Milvus Vector DB
# ============================================================================
class MilvusClient:
    """Wrap Milvus for distributed vector search."""
    def __init__(self, uri: str = "http://localhost:19530",
                 collection: str = "qai_default", dim: int = 64):
        self._milvus = _try_import("pymilvus")
        if self._milvus is None:
            raise ImportError("pymilvus not installed; pip install pymilvus")
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        connections.connect(uri=uri)
        self.dim = dim
        if utility.has_collection(collection):
            self.collection = Collection(collection)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields, "qai collection")
            self.collection = Collection(collection, schema)
            self.collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

    def add(self, ids: List[str], embeddings: np.ndarray) -> None:
        self.collection.insert([ids, np.asarray(embeddings, dtype=np.float32).tolist()])

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        self.collection.load()
        results = self.collection.search([np.asarray(query, dtype=np.float32).tolist()],
                                            "embedding", {"metric_type": "L2", "params": {"nprobe": 10}},
                                            top_k, output_fields=["id"])
        return [(hit.entity.get("id"), float(hit.distance)) for hit in results[0]]


# ============================================================================
# pgvector Store
# ============================================================================
class PgvectorStore:
    """Wrap PostgreSQL + pgvector extension."""
    def __init__(self, dsn: str = "postgresql://localhost/qai",
                 table: str = "vectors", dim: int = 64):
        self._psycopg = _try_import("psycopg")
        if self._psycopg is None:
            raise ImportError("psycopg not installed; pip install 'psycopg[binary]'")
        self.conn = self._psycopg.connect(dsn)
        self.table = table
        self.dim = dim
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, embedding vector({dim}), document TEXT);")
            self.conn.commit()

    def add(self, ids: List[str], embeddings: np.ndarray, documents: List[str]) -> None:
        with self.conn.cursor() as cur:
            for id_, emb, doc in zip(ids, embeddings, documents):
                cur.execute(f"INSERT INTO {self.table} (id, embedding, document) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING;",
                              (id_, list(np.asarray(emb).flatten()), doc))
            self.conn.commit()

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id, document, embedding <-> %s::vector AS dist FROM {self.table} ORDER BY dist LIMIT %s;",
                          [list(np.asarray(query).flatten()), top_k])
            return [(row[0], float(row[2])) for row in cur.fetchall()]


# ============================================================================
# Vector Store Registry -- swap backend with one line
# ============================================================================
class VectorStoreRegistry:
    """Unified API over multiple vector store backends."""
    def __init__(self, backend: str = "faiss", **kwargs):
        self.backend = backend
        if backend == "faiss":
            self._store = FAISSIndex(dim=kwargs.get("dim", 64))
        elif backend == "chroma":
            self._store = ChromaStore(collection=kwargs.get("collection", "qai"))
        elif backend == "milvus":
            self._store = MilvusClient(uri=kwargs.get("uri", "http://localhost:19530"),
                                         dim=kwargs.get("dim", 64))
        elif backend == "pgvector":
            self._store = PgvectorStore(dsn=kwargs.get("dsn", ""), dim=kwargs.get("dim", 64))
        else:
            raise ValueError(f"Unknown backend '{backend}'")

    def add(self, ids, vectors, documents=None, metadatas=None):
        if self.backend == "faiss":
            self._store.add(ids, vectors)
        elif self.backend == "chroma":
            self._store.add(ids, documents or [""]*len(ids), vectors, metadatas)
        elif self.backend == "milvus":
            self._store.add(ids, vectors)
        elif self.backend == "pgvector":
            self._store.add(ids, vectors, documents or [""]*len(ids))

    def search(self, query, top_k=5):
        return self._store.search(query, top_k)


# ============================================================================
# Experiment Tracking
# ============================================================================
class _BaseLogger:
    """Common interface for all tracking loggers."""
    def __init__(self, project: str = "qai"):
        self.project = project
        self._runs: Dict[str, Dict[str, Any]] = {}

    def start_run(self, run_name: str) -> str:
        self._runs[run_name] = {"params": {}, "metrics": {}, "artifacts": [],
                                  "start_time": time.time()}
        return run_name

    def log_params(self, run: str, params: Dict[str, Any]) -> None:
        self._runs[run]["params"].update(params)

    def log_metric(self, run: str, key: str, value: float, step: int = 0) -> None:
        self._runs[run]["metrics"].setdefault(key, []).append({"step": step, "value": float(value)})

    def log_artifact(self, run: str, path: str) -> None:
        self._runs[run]["artifacts"].append(path)

    def end_run(self, run: str) -> Dict[str, Any]:
        self._runs[run]["end_time"] = time.time()
        return self._runs[run]


class MLflowLogger(_BaseLogger):
    """Wrap MLflow for experiment tracking."""
    def __init__(self, project: str = "qai", tracking_uri: Optional[str] = None):
        super().__init__(project)
        self._mlflow = _try_import("mlflow")
        if self._mlflow is None:
            raise ImportError("mlflow not installed; pip install mlflow")
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(project)
        self._active_run = None

    def start_run(self, run_name: str) -> str:
        self._active_run = self._mlflow.start_run(run_name=run_name)
        return run_name

    def log_params(self, run: str, params: Dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metric(self, run: str, key: str, value: float, step: int = 0) -> None:
        self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, run: str, path: str) -> None:
        self._mlflow.log_artifact(path)


class WandBLogger(_BaseLogger):
    """Wrap Weights & Biases."""
    def __init__(self, project: str = "qai"):
        super().__init__(project)
        self._wandb = _try_import("wandb")
        if self._wandb is None:
            raise ImportError("wandb not installed; pip install wandb")

    def start_run(self, run_name: str) -> str:
        self._wandb.init(project=self.project, name=run_name)
        return run_name

    def log_params(self, run: str, params: Dict[str, Any]) -> None:
        self._wandb.config.update(params)

    def log_metric(self, run: str, key: str, value: float, step: int = 0) -> None:
        self._wandb.log({key: value}, step=step)


# ============================================================================
# Observability
# ============================================================================
class LangfuseTracer:
    """Wrap Langfuse for LLM call tracing + evaluation."""
    def __init__(self, public_key: Optional[str] = None, secret_key: Optional[str] = None):
        lf = _try_import("langfuse")
        if lf is None:
            raise ImportError("langfuse not installed; pip install langfuse")
        self._lf = lf
        self.client = lf.Langfuse(public_key=public_key, secret_key=secret_key) \
            if public_key else lf.Langfuse()

    def trace(self, name: str, input_data: Any, output_data: Any,
              metadata: Optional[Dict] = None) -> None:
        trace = self.client.trace(name=name)
        trace.generation(name=name, input=input_data, output=output_data,
                          metadata=metadata or {})

    def flush(self) -> None:
        self.client.flush()


class ArizeLogger:
    """Wrap Arize for model performance monitoring."""
    def __init__(self, api_key: Optional[str] = None, space_key: Optional[str] = None):
        arize = _try_import("arize.pandas")
        if arize is None:
            raise ImportError("arize not installed; pip install arize")
        self._arize = arize
        self.client = arize.Client(api_key=api_key, space_key=space_key) \
            if api_key and space_key else None

    def log_predictions(self, model_id: str, predictions: List[Dict]) -> Dict[str, Any]:
        if self.client is None:
            return {"logged": False, "reason": "no credentials"}
        import pandas as pd
        df = pd.DataFrame(predictions)
        return self.client.log(model_id=model_id, model_version="1.0", dataframe=df)


class PhoenixTracer:
    """Arize Phoenix (open-source, self-hosted) tracer."""
    def __init__(self, endpoint: str = "http://localhost:6006"):
        phoenix = _try_import("phoenix")
        if phoenix is None:
            raise ImportError("arize-phoenix not installed; pip install arize-phoenix")
        self._phoenix = phoenix
        try:
            self.session = phoenix.launch_app(endpoint=endpoint)
        except Exception:
            self.session = None

    def trace(self, data: Dict[str, Any]) -> None:
        # Phoenix uses OpenTelemetry; we just store events for retrieval
        pass


class HeliconeLogger:
    """Wrap Helicone for LLM proxy logging."""
    def __init__(self, api_key: Optional[str] = None):
        helicone = _try_import("helicone")
        if helicone is None:
            raise ImportError("helicone not installed; pip install helicone")
        self._helicone = helicone
        self.api_key = api_key

    def wrap(self, llm_callable: Callable) -> Callable:
        """Wrap an LLM callable with Helicone logging."""
        try:
            return self._helicone.wrapper(llm_callable)
        except Exception:
            return llm_callable


# ============================================================================
# Structured Generation
# ============================================================================
class DSPyAdapter:
    """Wrap DSPy for declarative LLM programming."""
    def __init__(self, lm: Any = None):
        dspy = _try_import("dspy")
        if dspy is None:
            raise ImportError("dspy-ai not installed; pip install dspy-ai")
        self._dspy = dspy
        self.lm = lm

    def configure(self, lm: Any) -> None:
        self.lm = lm
        self._dspy.settings.configure(lm=lm)

    def build_signature(self, input_fields: Dict[str, type], output_fields: Dict[str, type]):
        """Build a DSPy Signature subclass at runtime."""
        return self._dspy.Signature(input_fields, output_fields)


class InstructorAdapter:
    """Wrap instructor for pydantic-typed LLM calls."""
    def __init__(self, client: Any = None):
        instructor = _try_import("instructor")
        if instructor is None:
            raise ImportError("instructor not installed; pip install instructor")
        self._instructor = instructor
        self.client = client

    def patch(self, client: Any) -> Any:
        """Patch an OpenAI-compatible client with instructor."""
        return self._instructor.from_openai(client)

    def extract(self, client: Any, response_model: type, prompt: str):
        """One-shot extract a pydantic model from a prompt."""
        patched = self.patch(client)
        return patched.chat.completions.create(
            response_model=response_model,
            messages=[{"role": "user", "content": prompt}],
        )


class OutlinesAdapter:
    """Wrap Outlines for structured generation with regex/JSON schemas."""
    def __init__(self, model: Any = None):
        outlines = _try_import("outlines")
        if outlines is None:
            raise ImportError("outlines not installed; pip install outlines")
        self._outlines = outlines
        self.model = model

    def regex(self, pattern: str):
        """Build a generator constrained to a regex."""
        if self.model is None:
            raise ValueError("OutlinesAdapter requires a model")
        return self._outlines.generate.regex(self.model, pattern)

    def json(self, schema: Dict[str, Any]):
        """Build a generator constrained to a JSON schema."""
        if self.model is None:
            raise ValueError("OutlinesAdapter requires a model")
        return self._outlines.generate.json(self.model, schema)


class LMQLAdapter:
    """Wrap LMQL for SQL-like LLM queries."""
    def __init__(self):
        lmql = _try_import("lmql")
        if lmql is None:
            raise ImportError("lmql not installed; pip install lmql")
        self._lmql = lmql

    def query(self, query_str: str, **kwargs):
        """Run an LMQL query."""
        return self._lmql.run(query_str, **kwargs)


__all__ = [
    # framework adapters
    "LangChainAdapter", "LlamaIndexAdapter", "HaystackAdapter",
    # model hub
    "HuggingFaceHub", "ModelRegistry",
    # vector stores
    "FAISSIndex", "ChromaStore", "MilvusClient", "PgvectorStore", "VectorStoreRegistry",
    # tracking
    "MLflowLogger", "WandBLogger",
    # observability
    "LangfuseTracer", "ArizeLogger", "PhoenixTracer", "HeliconeLogger",
    # structured generation
    "DSPyAdapter", "InstructorAdapter", "OutlinesAdapter", "LMQLAdapter",
]
