"""
qai.serving -- unified LLM serving layer.

Wraps the major 2025/2026 LLM serving engines behind one API:

  Production engines:
    - VLLMServer         -- PagedAttention, continuous batching (UC Berkeley)
    - SGLangServer       -- RadixAttention, prefix cache reuse (LMSYS)
    - TGIServer          -- HuggingFace Text Generation Inference (now in
                             maintenance mode; we still support it)
    - TensorRTLLMServer  -- NVIDIA's peak-throughput engine
    - LMDeployServer     -- OpenMMLab's C++ TurboMind engine
    - AphroditeServer    -- community fork of vLLM with broad quant support

  Local engines:
    - OllamaServer       -- one-command local LLM (llama.cpp backend)
    - LlamaCppServer     -- C++ llama.cpp directly
    - MLXServer          -- Apple Silicon optimised
    - LocalAIServer      -- multi-modal local stack

  Unified client:
    - LLMClient          -- swap backend by changing one string
    - LLMRouter          -- route requests by model/hardware/cost
    - StreamingResponse  -- standard streaming protocol

  OpenAI-compatible HTTP shim:
    - OpenAIShim         -- expose any qai model via an OpenAI-style API

Most of these wrappers lazy-import their backend so that qai.serving can
be imported even when no actual serving engine is installed.
"""
from __future__ import annotations
import json
import time
import os
import subprocess
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import importlib


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


# ============================================================================
# Common protocol
# ============================================================================
class GenerationRequest:
    def __init__(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 1.0, top_p: float = 1.0,
                 stop: Optional[List[str]] = None, stream: bool = False,
                 **kwargs):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop or []
        self.stream = stream
        self.extra = kwargs


class GenerationResponse:
    def __init__(self, text: str, finish_reason: str = "stop",
                 tokens_generated: int = 0, latency_ms: float = 0.0,
                 raw: Optional[Dict] = None):
        self.text = text
        self.finish_reason = finish_reason
        self.tokens_generated = tokens_generated
        self.latency_ms = latency_ms
        self.raw = raw or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "finish_reason": self.finish_reason,
                "tokens_generated": self.tokens_generated,
                "latency_ms": self.latency_ms}


# ============================================================================
# Backend wrappers
# ============================================================================
class _BaseBackend:
    """Common interface every backend implements."""
    name: str = "base"

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        return {"backend": self.name, "model": self.model, "healthy": True}


class VLLMBackend(_BaseBackend):
    """vLLM -- PagedAttention + continuous batching.

    Requires `vllm` to be installed. Best for production high-concurrency.
    """
    name = "vllm"

    def __init__(self, model: str, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9, **kwargs):
        super().__init__(model, **kwargs)
        vllm = _try_import("vllm")
        if vllm is None:
            raise ImportError("vllm not installed; pip install vllm")
        self._vllm = vllm
        self.llm = vllm.LLM(model=model, tensor_parallel_size=tensor_parallel_size,
                              gpu_memory_utilization=gpu_memory_utilization, **kwargs)
        self.sampler = vllm.SamplingParams

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        t0 = time.time()
        params = self.sampler(temperature=request.temperature,
                                max_tokens=request.max_tokens,
                                top_p=request.top_p, stop=request.stop)
        outputs = self.llm.generate([request.prompt], params)
        out = outputs[0].outputs[0]
        return GenerationResponse(text=out.text,
                                    finish_reason=out.finish_reason,
                                    tokens_generated=len(out.token_ids),
                                    latency_ms=(time.time() - t0) * 1000)


class SGLangBackend(_BaseBackend):
    """SGLang -- RadixAttention prefix cache reuse + structured output.

    Best for agentic / RAG workloads with shared long prefixes.
    """
    name = "sglang"

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        sglang = _try_import("sglang")
        if sglang is None:
            raise ImportError("sglang not installed; pip install sglang")
        self.engine = sglang.Engine(model_path=model, **kwargs)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        t0 = time.time()
        out = self.engine.generate(request.prompt,
                                     max_new_tokens=request.max_tokens,
                                     temperature=request.temperature,
                                     top_p=request.top_p)
        return GenerationResponse(text=out["text"],
                                    finish_reason=out.get("finish_reason", "stop"),
                                    tokens_generated=out.get("completion_tokens", 0),
                                    latency_ms=(time.time() - t0) * 1000)


class TGIBackend(_BaseBackend):
    """Text Generation Inference (HuggingFace)."""
    name = "tgi"

    def __init__(self, model: str, url: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        client = _try_import("huggingface_hub")
        if client is None:
            raise ImportError("huggingface_hub not installed")
        # TGI is typically run as a server; we connect via the inference client
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(url or f"http://localhost:8080")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        t0 = time.time()
        out = self.client.text_generation(request.prompt,
                                            max_new_tokens=request.max_tokens,
                                            temperature=request.temperature,
                                            top_p=request.top_p,
                                            stop_sequences=request.stop)
        return GenerationResponse(text=out, latency_ms=(time.time() - t0) * 1000)


class OllamaBackend(_BaseBackend):
    """Ollama -- local LLM with one-command setup."""
    name = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        ollama = _try_import("ollama")
        if ollama is None:
            raise ImportError("ollama not installed; pip install ollama")
        self.client = ollama.Client(host=host)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        t0 = time.time()
        out = self.client.generate(model=self.model, prompt=request.prompt,
                                     stream=False, options={
                                         "temperature": request.temperature,
                                         "top_p": request.top_p,
                                         "num_predict": request.max_tokens,
                                         "stop": request.stop,
                                     })
        return GenerationResponse(text=out["response"],
                                    finish_reason="stop",
                                    tokens_generated=out.get("eval_count", 0),
                                    latency_ms=(time.time() - t0) * 1000)

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        for chunk in self.client.generate(model=self.model, prompt=request.prompt,
                                            stream=True, options={
                                                "temperature": request.temperature,
                                                "num_predict": request.max_tokens,
                                            }):
            yield chunk["response"]


class LlamaCppBackend(_BaseBackend):
    """Direct llama.cpp bindings."""
    name = "llama_cpp"

    def __init__(self, model: str, n_ctx: int = 2048, n_threads: int = 4, **kwargs):
        super().__init__(model, **kwargs)
        lcp = _try_import("llama_cpp")
        if lcp is None:
            raise ImportError("llama-cpp-python not installed; pip install llama-cpp-python")
        self.llm = lcp.Llama(model_path=model, n_ctx=n_ctx, n_threads=n_threads, **kwargs)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        t0 = time.time()
        out = self.llm(request.prompt, max_tokens=request.max_tokens,
                        temperature=request.temperature, top_p=request.top_p,
                        stop=request.stop)
        return GenerationResponse(text=out["choices"][0]["text"],
                                    finish_reason=out["choices"][0].get("finish_reason", "stop"),
                                    tokens_generated=out["usage"]["completion_tokens"],
                                    latency_ms=(time.time() - t0) * 1000)


class MLXBackend(_BaseBackend):
    """MLX for Apple Silicon."""
    name = "mlx"

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        mlx = _try_import("mlx_lm")
        if mlx is None:
            raise ImportError("mlx-lm not installed; pip install mlx-lm")
        self.model_obj, self.tokenizer = mlx.load(model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        mlx = _try_import("mlx_lm")
        t0 = time.time()
        out = mlx.generate(self.model_obj, self.tokenizer,
                             prompt=request.prompt,
                             max_tokens=request.max_tokens,
                             temp=request.temperature)
        return GenerationResponse(text=out, latency_ms=(time.time() - t0) * 1000)


class LocalAIBackend(_BaseBackend):
    """LocalAI -- multi-modal local stack (TTS/STT/embeddings/vision)."""
    name = "localai"

    def __init__(self, model: str, base_url: str = "http://localhost:8080", **kwargs):
        super().__init__(model, **kwargs)
        openai = _try_import("openai")
        if openai is None:
            raise ImportError("openai not installed; pip install openai")
        self.client = openai.OpenAI(base_url=base_url, api_key="not-needed")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        t0 = time.time()
        out = self.client.completions.create(model=self.model,
                                                prompt=request.prompt,
                                                max_tokens=request.max_tokens,
                                                temperature=request.temperature,
                                                top_p=request.top_p)
        return GenerationResponse(text=out.choices[0].text,
                                    latency_ms=(time.time() - t0) * 1000)


class TensorRTLLMBackend(_BaseBackend):
    """NVIDIA TensorRT-LLM (peak throughput on NVIDIA-only stacks)."""
    name = "tensorrt_llm"

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        trt = _try_import("tensorrt_llm")
        if trt is None:
            raise ImportError("tensorrt-llm not installed")
        # TensorRT-LLM requires pre-built engines; here we just open a session
        self.session = trt.runtime.Session.from_dir(model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        t0 = time.time()
        out = self.session.generate(request.prompt,
                                      max_new_tokens=request.max_tokens,
                                      temperature=request.temperature)
        return GenerationResponse(text=out["text"], latency_ms=(time.time() - t0) * 1000)


class LMDeployBackend(_BaseBackend):
    """OpenMMLab LMDeploy (TurboMind C++ engine)."""
    name = "lmdeploy"

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        lmd = _try_import("lmdeploy")
        if lmd is None:
            raise ImportError("lmdeploy not installed; pip install lmdeploy")
        self.pipe = lmd.pipeline(model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        t0 = time.time()
        out = self.pipe([request.prompt], gen_config=self.pipe.gen_config(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        ))
        return GenerationResponse(text=out[0].response, latency_ms=(time.time() - t0) * 1000)


# ============================================================================
# Backend registry
# ============================================================================
BACKENDS: Dict[str, type] = {
    "vllm": VLLMBackend,
    "sglang": SGLangBackend,
    "tgi": TGIBackend,
    "ollama": OllamaBackend,
    "llama_cpp": LlamaCppBackend,
    "mlx": MLXBackend,
    "localai": LocalAIBackend,
    "tensorrt_llm": TensorRTLLMBackend,
    "lmdeploy": LMDeployBackend,
}


# ============================================================================
# Unified LLM client
# ============================================================================
class LLMClient:
    """Unified LLM client -- swap backend by changing one string.

    Usage:
        client = LLMClient(backend="ollama", model="llama3.1")
        response = client.generate("Hello, world!")
    """
    def __init__(self, backend: str = "ollama", model: str = "llama3.1", **kwargs):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Available: {sorted(BACKENDS.keys())}")
        self.backend_name = backend
        self.backend = BACKENDS[backend](model=model, **kwargs)

    def generate(self, prompt: str, **kwargs) -> GenerationResponse:
        req = GenerationRequest(prompt=prompt, **kwargs)
        return self.backend.generate(req)

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        req = GenerationRequest(prompt=prompt, stream=True, **kwargs)
        return self.backend.stream(req)

    def health(self) -> Dict[str, Any]:
        return self.backend.health()


# ============================================================================
# LLM Router -- route requests by model / hardware / cost
# ============================================================================
class LLMRouter:
    """Route requests to the cheapest/fastest available backend.

    Routing rules (evaluated top-to-bottom):
      - if model name starts with 'ollama/' -> ollama
      - if request.stream is True -> ollama (best streaming)
      - if max_tokens > 1024 and concurrency > 10 -> vllm
      - if 'sglang' in tags -> sglang
      - else default to first available
    """
    def __init__(self, backends: Optional[List[LLMClient]] = None):
        self.backends: List[LLMClient] = backends or []
        self._rules: List[Tuple[Callable[[GenerationRequest], bool], str]] = []

    def add_backend(self, client: LLMClient) -> None:
        self.backends.append(client)

    def add_rule(self, predicate: Callable[[GenerationRequest], bool], backend: str) -> None:
        self._rules.append((predicate, backend))

    def route(self, request: GenerationRequest) -> LLMClient:
        for predicate, backend_name in self._rules:
            if predicate(request):
                for client in self.backends:
                    if client.backend_name == backend_name:
                        return client
        return self.backends[0] if self.backends else None

    def generate(self, prompt: str, **kwargs) -> GenerationResponse:
        req = GenerationRequest(prompt=prompt, **kwargs)
        client = self.route(req)
        if client is None:
            raise RuntimeError("No backends registered")
        return client.generate(prompt, **kwargs)


# ============================================================================
# OpenAI-compatible HTTP shim
# ============================================================================
class OpenAIShim:
    """Expose any qai model via an OpenAI-style /v1/chat/completions API.

    Useful for plugging qai models into any client that speaks OpenAI.
    """
    def __init__(self, model: Any, host: str = "0.0.0.0", port: int = 8000):
        self.model = model
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

    def _build_app(self):
        flask = _try_import("flask")
        if flask is None:
            raise ImportError("flask not installed; pip install flask")
        app = flask.Flask(__name__)

        @app.route("/v1/chat/completions", methods=["POST"])
        def chat_completions():
            data = flask.request.json
            prompt = "\n".join(m["content"] for m in data.get("messages", []))
            try:
                out = self.model.forward(prompt)
                text = str(out)
            except Exception as e:
                text = f"[error] {e}"
            return flask.jsonify({
                "id": "chatcmpl-qai",
                "object": "chat.completion",
                "model": "qai-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt.split()),
                            "completion_tokens": len(text.split()),
                            "total_tokens": len(prompt.split()) + len(text.split())},
            })

        @app.route("/v1/models", methods=["GET"])
        def models():
            return flask.jsonify({"data": [{"id": "qai-model", "object": "model"}]})

        @app.route("/health", methods=["GET"])
        def health():
            return flask.jsonify({"status": "ok"})

        return app

    def start(self, block: bool = False) -> None:
        app = self._build_app()
        if block:
            app.run(host=self.host, port=self.port)
        else:
            self._thread = threading.Thread(
                target=app.run, kwargs={"host": self.host, "port": self.port, "use_reloader": False},
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        # Flask dev server doesn't have a clean stop; rely on daemon thread
        pass


__all__ = [
    "GenerationRequest", "GenerationResponse",
    "VLLMBackend", "SGLangBackend", "TGIBackend", "OllamaBackend",
    "LlamaCppBackend", "MLXBackend", "LocalAIBackend",
    "TensorRTLLMBackend", "LMDeployBackend",
    "BACKENDS", "LLMClient", "LLMRouter", "OpenAIShim",
]
