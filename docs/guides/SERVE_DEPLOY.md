# Guide: Serving & Deployment

*Your model is trained and evaluated. Put it behind an API, version it, A/B test it, and keep it healthy.*

> Back to [Guides index](README.md) · Prev: [FINETUNING_ALIGNMENT.md](FINETUNING_ALIGNMENT.md) · Shrink first: [COMPRESS.md](COMPRESS.md) · Monitor after: [MLOPS.md](MLOPS.md)

---

## What you'll learn

1. Production server with dynamic batching and streaming
2. Multi-model Gateway routing
3. Versioning, staging promotion, A/B testing, auto-rollback (`qai.runtime`)
4. Cryptographic model signing (supply-chain integrity)
5. One-line deployment targets

---

## 1. ProductionServer

```python
server = qai.ProductionServer(
    model=m, port=8000,
    max_batch_size=8,          # dynamic batching — groups concurrent requests
    max_concurrent=64,
    enable_streaming=True,     # SSE for token-by-token generation
)
server.start()   # blocks

# Endpoints:
#   POST /predict        single prediction
#   POST /predict_batch  batched predictions
#   POST /generate       text generation (streams when enable_streaming)
#   GET  /health         liveness + model info
#   GET  /metrics        latency, throughput, error rates
```

## 2. FastInferenceEngine (vLLM-style batching)

```python
engine = qai.FastInferenceEngine(model=m, max_batch_size=64)
engine.start()
future = engine.generate(prompt, max_new_tokens=100)
tokens = future.result()
engine.stop()
```

Under the hood: **PagedAttention** — the KV cache is stored in fixed-size
pages (like virtual memory), eliminating fragmentation. CUDA is used
automatically when available (`torch.cuda.is_available()`), CPU otherwise.

## 3. Multi-model Gateway

```python
from qai.serving_gateway import Gateway, DeployedModel

gw = Gateway()
gw.deploy(DeployedModel(name="sentiment", model=clf_a, version="1.0.0"))
gw.deploy(DeployedModel(name="sentiment", model=clf_b, version="1.1.0"))

gw.route("sentiment", version="1.1.0")     # pin a version
result = gw.predict("sentiment", payload)
gw.status()                                 # health of every deployment
```

## 4. Versioning, staging, A/B, rollback (`qai.runtime`)

```python
from qai.runtime import ModelVersion, RuntimeModelWrapper

wrapper = RuntimeModelWrapper(model=m)

# Semantic-version every artifact
v1 = wrapper.register(version="1.0.0", stage="production")

# Promote through stages: dev → staging → production (gated)
wrapper.promote("1.1.0", to_stage="staging")
wrapper.promote("1.1.0", to_stage="production")

# A/B traffic routing
wrapper.start_ab_test("1.0.0", "1.1.0", traffic_split=0.2)
report = wrapper.ab_report()         # per-version metrics

# Auto-rollback when the new version degrades
wrapper.enable_auto_rollback(metric="error_rate", threshold=0.05)
```

CLI: `qai-runtime` ships as a command-line router for the same registry.

## 5. Cryptographic signing (supply-chain integrity)

Every `.qmodel` bundle supports HMAC-SHA256 signing — any tampering invalidates:

```python
m.save("model.qmodel", sign=True)             # sign on save
loaded = qai.Model.load("model.qmodel", verify=True)   # verify on load

# Publisher trust (public keys, not just symmetric HMAC):
from qai.bundle import trust_add
trust_add(publisher_key, "trusted-publisher")  # pin a publisher
```

More security context: [SECURITY_GOVERNANCE.md](SECURITY_GOVERNANCE.md).

## 6. One-line deploy targets

```python
m.deploy(target="server")   # full precision, wired into ProductionServer
m.deploy(target="edge")     # int8-quantized artifact
m.deploy(target="mobile")   # int4-quantized, smallest footprint
```

Export formats (10 of them): `qai.export(model, format="onnx", path=...)` —
onnx, torchscript, tflite, coreml, tensorrt, openvino, gguf, safetensors,
pickle, qai_native. Conversion details and verification in
[COMPRESS.md](COMPRESS.md).

## 7. Serving with Flask directly (simple cases)

```python
from qai.core.serve import serve, build_flask_app

app = build_flask_app(model=m)   # POST /predict endpoint
serve(model=m, port=5000)        # or one call
```

---

## Cheat sheet

| Task | Call |
|---|---|
| Serve one model | `qai.ProductionServer(model=m, port=8000).start()` |
| Streaming generation | `enable_streaming=True` |
| High-throughput LLM | `qai.FastInferenceEngine(model, max_batch_size=64)` |
| Many models | `qai.serving_gateway.Gateway` |
| Version a model | `RuntimeModelWrapper.register(version=...)` |
| A/B test | `wrapper.start_ab_test(a, b, traffic_split)` |
| Auto-rollback | `wrapper.enable_auto_rollback(...)` |
| Sign a bundle | `m.save("model.qmodel", sign=True)` |
| Deploy target | `m.deploy(target="edge")` |
| Export format | `qai.export(m, format="onnx")` |
| Simple Flask | `qai.serve(model, port=5000)` |
| Health endpoint | `GET /health` |

---

## Common gotchas

- **Sign + verify both ways.** `sign=True` on save and `verify=True` on load — one without the other is theater.
- **A/B tests need traffic.** A 1% split gives you statistically nothing; use ≥20% per arm.
- **Quantize before serving**, not after — see [COMPRESS.md](COMPRESS.md) for the QAT trick that minimizes quality loss.
- **Watch for drift** once live — that's [MLOPS.md](MLOPS.md) territory, and `qai.runtime` auto-rollback is your safety net.

---

*Next: [MLOPS.md](MLOPS.md) to monitor what you just deployed.*
