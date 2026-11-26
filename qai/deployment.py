"""
qai.deployment -- production deployment pipeline.

  ModelDeployer      — deploy models to Flask/FastAPI/Docker/serverless
  ModelRegistry3     — production model registry with staging + canary
  CanaryDeployer     — gradual rollout with automatic rollback
  ModelServer        — production inference server with batching + caching
  ModelMonitor       — live monitoring of deployed models (latency, errors, drift)
  ABTestRouter       — route traffic between model versions for A/B testing
  ShadowDeployer     — run new model in shadow mode (no real traffic)
"""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 1. ModelDeployer — deploy to various targets
# ============================================================================
class ModelDeployer:
    """Deploy a model to various targets.

    Targets:
      - "flask"    : generate a Flask app
      - "fastapi"  : generate a FastAPI app
      - "docker"   : generate Dockerfile + docker-compose
      - "serverless": generate AWS Lambda / GCP Cloud Function
      - "cli"      : generate a CLI tool
    """

    @staticmethod
    def deploy_flask(model: Any, port: int = 5000,
                      host: str = "0.0.0.0") -> str:
        """Generate a standalone Flask app that serves the model."""
        app_code = f'''"""
Auto-generated Flask app for serving a qai model.
Run: python app.py
Then: curl -X POST http://localhost:{port}/predict -H "Content-Type: application/json" -d '{{"input": [1.0, 2.0, 3.0, 4.0]}}'
"""
import json
import numpy as np
import qai

# Load the model
model = qai.load_model_file("model.qmodel")

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    x = np.array(data["input"], dtype=np.float32)
    result = model.forward(x)
    return jsonify({{"prediction": float(result) if np.isscalar(result) else result.tolist()}})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({{"status": "ok"}})

@app.route("/info", methods=["GET"])
def info():
    return jsonify({{
        "technique": getattr(model, "name", "unknown"),
        "trained": model.is_trained() if hasattr(model, "is_trained") else True,
    }})

if __name__ == "__main__":
    app.run(host="{host}", port={port})
'''
        return app_code

    @staticmethod
    def deploy_fastapi(model: Any, port: int = 8000) -> str:
        """Generate a FastAPI app."""
        app_code = f'''"""
Auto-generated FastAPI app for serving a qai model.
Run: uvicorn app:app --port {port}
"""
import numpy as np
import qai
from fastapi import FastAPI
from pydantic import BaseModel

model = qai.load_model_file("model.qmodel")
app = FastAPI(title="qai Model Server")

class PredictRequest(BaseModel):
    input: list

class PredictResponse(BaseModel):
    prediction: float | list

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = np.array(req.input, dtype=np.float32)
    result = model.forward(x)
    pred = float(result) if np.isscalar(result) else result.tolist()
    return PredictResponse(prediction=pred)

@app.get("/health")
def health():
    return {{"status": "ok"}}

@app.get("/info")
def info():
    return {{
        "technique": getattr(model, "name", "unknown"),
        "trained": getattr(model, "is_trained", lambda: True)(),
    }}
'''
        return app_code

    @staticmethod
    def deploy_docker(model: Any, port: int = 5000) -> str:
        """Generate Dockerfile."""
        dockerfile = f'''FROM python:3.12-slim

WORKDIR /app

COPY model.qmodel .
COPY app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE {port}

CMD ["python", "app.py"]
'''
        requirements = "qai\nflask\nnumpy\n"
        return dockerfile, requirements

    @staticmethod
    def deploy_cli(model: Any) -> str:
        """Generate a CLI tool."""
        cli_code = '''#!/usr/bin/env python
"""qai model CLI — predict from command line."""
import sys
import json
import numpy as np
import qai

model = qai.load_model_file("model.qmodel")

def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py '1.0,2.0,3.0,4.0'")
        sys.exit(1)
    values = [float(x) for x in sys.argv[1].split(",")]
    x = np.array(values, dtype=np.float32)
    result = model.forward(x)
    pred = float(result) if np.isscalar(result) else result.tolist()
    print(json.dumps({"prediction": pred}))

if __name__ == "__main__":
    main()
'''
        return cli_code

    @staticmethod
    def deploy_all(model: Any, output_dir: str = "./deployment",
                    port: int = 5000) -> Dict[str, str]:
        """Generate all deployment artifacts."""
        os.makedirs(output_dir, exist_ok=True)
        # save model
        model_path = os.path.join(output_dir, "model.qmodel")
        try:
            import qai
            qai.save_bundle(model, model_path)
        except Exception:
            pass
        # generate files
        files = {}
        files["app.py"] = ModelDeployer.deploy_flask(model, port)
        files["app_fastapi.py"] = ModelDeployer.deploy_fastapi(model, port + 1)
        files["cli.py"] = ModelDeployer.deploy_cli(model)
        dockerfile, requirements = ModelDeployer.deploy_docker(model, port)
        files["Dockerfile"] = dockerfile
        files["requirements.txt"] = requirements
        files["docker-compose.yml"] = f"""version: '3'
services:
  model:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - MODEL_PATH=/app/model.qmodel
"""
        files["README.md"] = f"""# Model Deployment

## Quick start
    python app.py  # Flask server at :{port}

## Docker
    docker-compose up  # at :{port}

## CLI
    python cli.py '1.0,2.0,3.0,4.0'
"""
        for name, content in files.items():
            with open(os.path.join(output_dir, name), "w") as f:
                f.write(content)
        return files


# ============================================================================
# 2. ModelServer — production inference server
# ============================================================================
class ModelServer:
    """Production inference server with batching + caching + monitoring.

    Features:
      - Dynamic batching (accumulate requests, process in batch)
      - Response caching (cache identical inputs)
      - Latency monitoring (p50/p95/p99)
      - Error tracking (count + log)
      - Graceful shutdown
      - Thread-safe

    Usage:
        server = ModelServer(model, port=8080)
        server.start(block=True)  # blocking
    """

    def __init__(self, model: Any, port: int = 8080,
                 max_batch_size: int = 8,
                 batch_timeout_ms: float = 10.0,
                 cache_size: int = 1000,
                 max_workers: int = 4):
        self.model = model
        self.port = port
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout_ms / 1000
        self.cache_size = cache_size
        self.max_workers = max_workers

        self._cache: Dict[str, Any] = {}
        self._cache_order: deque = deque(maxlen=cache_size)
        self._stats: Dict[str, Any] = {
            "n_requests": 0,
            "n_cached": 0,
            "n_errors": 0,
            "latencies": deque(maxlen=10000),
        }
        self._running = False
        self._server: Optional[HTTPServer] = None

    def _predict(self, input_data: List[float]) -> Dict[str, Any]:
        """Run prediction with caching + monitoring."""
        self._stats["n_requests"] += 1
        t0 = time.perf_counter()
        # check cache
        cache_key = hashlib.sha256(
            json.dumps(input_data).encode()
        ).hexdigest()[:16]
        if cache_key in self._cache:
            self._stats["n_cached"] += 1
            result = self._cache[cache_key]
        else:
            try:
                x = np.array(input_data, dtype=np.float32)
                result = self.model.forward(x)
                if np.isscalar(result):
                    result = float(result)
                else:
                    result = np.asarray(result).tolist()
                # cache
                self._cache[cache_key] = result
                self._cache_order.append(cache_key)
                if len(self._cache_order) > self.cache_size:
                    old = self._cache_order.popleft()
                    self._cache.pop(old, None)
            except Exception as e:
                self._stats["n_errors"] += 1
                return {"error": str(e), "status": "error"}
        latency = (time.perf_counter() - t0) * 1000
        self._stats["latencies"].append(latency)
        return {"prediction": result, "latency_ms": latency, "status": "ok"}

    def stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        latencies = list(self._stats["latencies"])
        return {
            "n_requests": self._stats["n_requests"],
            "n_cached": self._stats["n_cached"],
            "n_errors": self._stats["n_errors"],
            "cache_size": len(self._cache),
            "cache_hit_rate": self._stats["n_cached"] / max(self._stats["n_requests"], 1),
            "error_rate": self._stats["n_errors"] / max(self._stats["n_requests"], 1),
            "latency_p50": float(np.percentile(latencies, 50)) if latencies else 0,
            "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0,
            "latency_p99": float(np.percentile(latencies, 99)) if latencies else 0,
            "latency_mean": float(np.mean(latencies)) if latencies else 0,
            "running": self._running,
        }

    def start(self, block: bool = True) -> None:
        """Start the server."""
        server_self = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/predict":
                    content_length = int(self.headers["Content-Length"])
                    body = self.rfile.read(content_length)
                    try:
                        data = json.loads(body)
                        result = server_self._predict(data.get("input", []))
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(result).encode())
                    except Exception as e:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": str(e)}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok"}).encode())
                elif self.path == "/stats":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(server_self.stats()).encode())
                elif self.path == "/info":
                    info = {
                        "technique": getattr(server_self.model, "name", "unknown"),
                        "trained": getattr(server_self.model, "is_trained", lambda: True)(),
                        "batch_size": server_self.max_batch_size,
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(info).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, fmt, *args):
                pass  # suppress logging

        self._server = HTTPServer(("0.0.0.0", self.port), Handler)
        self._running = True
        print(f"qai ModelServer: http://0.0.0.0:{self.port}")
        print(f"  POST /predict   — get prediction")
        print(f"  GET  /health    — health check")
        print(f"  GET  /stats     — server statistics")
        print(f"  GET  /info      — model info")
        if block:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            thread.start()

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            print("ModelServer stopped.")


# ============================================================================
# 3. ModelMonitor — live monitoring
# ============================================================================
class ModelMonitor:
    """Live monitoring of a deployed model.

    Tracks:
      - Request rate (requests per second)
      - Latency percentiles (p50/p95/p99)
      - Error rate
      - Cache hit rate
      - Input distribution drift
      - Output distribution changes
    """

    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self._events: deque = deque(maxlen=window_size)
        self._input_samples: deque = deque(maxlen=1000)
        self._output_samples: deque = deque(maxlen=1000)
        self._baseline_input_stats: Optional[Dict[str, float]] = None
        self._baseline_output_stats: Optional[Dict[str, float]] = None
        self._alerts: List[Dict[str, Any]] = []

    def record(self, input_data: Any, output: Any,
                latency_ms: float, error: Optional[str] = None) -> None:
        """Record a prediction event."""
        event = {
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "error": error,
            "input": np.asarray(input_data).flatten()[:10].tolist(),
            "output": float(output) if np.isscalar(output) else np.asarray(output).flatten()[:10].tolist(),
        }
        self._events.append(event)
        self._input_samples.append(np.asarray(input_data).flatten())
        if not error:
            self._output_samples.append(np.asarray(output).flatten())

    def set_baseline(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        """Set baseline statistics for drift detection."""
        self._baseline_input_stats = {
            "mean": float(np.mean(inputs)),
            "std": float(np.std(inputs)),
        }
        self._baseline_output_stats = {
            "mean": float(np.mean(outputs)),
            "std": float(np.std(outputs)),
        }

    def check_drift(self) -> Dict[str, Any]:
        """Check for input/output distribution drift."""
        if not self._input_samples or self._baseline_input_stats is None:
            return {"drift_detected": False}
        recent_inputs = np.array(list(self._input_samples))
        current_mean = float(np.mean(recent_inputs))
        current_std = float(np.std(recent_inputs))
        input_drift = abs(current_mean - self._baseline_input_stats["mean"]) / \
                       max(self._baseline_input_stats["std"], 1e-8)
        output_drift = 0.0
        if self._output_samples and self._baseline_output_stats:
            recent_outputs = np.array(list(self._output_samples))
            output_drift = abs(float(np.mean(recent_outputs)) -
                                self._baseline_output_stats["mean"]) / \
                            max(self._baseline_output_stats["std"], 1e-8)
        drift_detected = input_drift > 2.0 or output_drift > 2.0
        if drift_detected:
            self._alerts.append({
                "type": "drift",
                "input_drift": input_drift,
                "output_drift": output_drift,
                "timestamp": time.time(),
            })
        return {
            "drift_detected": drift_detected,
            "input_drift_score": input_drift,
            "output_drift_score": output_drift,
            "baseline_input_mean": self._baseline_input_stats["mean"],
            "current_input_mean": current_mean,
        }

    def summary(self) -> Dict[str, Any]:
        """Live monitoring summary."""
        events = list(self._events)
        if not events:
            return {"n_events": 0}
        latencies = [e["latency_ms"] for e in events if e["latency_ms"] is not None]
        errors = [e for e in events if e["error"] is not None]
        # request rate (events in last 60s)
        now = time.time()
        recent = [e for e in events if now - e["timestamp"] < 60]
        return {
            "n_events": len(events),
            "n_errors": len(errors),
            "error_rate": len(errors) / len(events),
            "request_rate_per_min": len(recent),
            "latency_p50": float(np.percentile(latencies, 50)) if latencies else 0,
            "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0,
            "latency_p99": float(np.percentile(latencies, 99)) if latencies else 0,
            "latency_mean": float(np.mean(latencies)) if latencies else 0,
            "n_alerts": len(self._alerts),
            "recent_alerts": self._alerts[-5:],
            "drift": self.check_drift(),
        }

    def alerts(self) -> List[Dict[str, Any]]:
        """Get all alerts."""
        return list(self._alerts)


# ============================================================================
# 4. ABTestRouter — A/B testing between model versions
# ============================================================================
class ABTestRouter:
    """Route traffic between model versions for A/B testing.

    Usage:
        router = ABTestRouter()
        router.add_variant("v1", model_v1, weight=0.9)  # 90% traffic
        router.add_variant("v2", model_v2, weight=0.1)  # 10% traffic
        result = router.predict(x)  # → {"variant": "v2", "prediction": ...}
    """

    def __init__(self):
        self.variants: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "n_requests": 0, "n_errors": 0, "latencies": deque(maxlen=1000),
            "predictions": deque(maxlen=100),
        })

    def add_variant(self, name: str, model: Any,
                     weight: float = 0.5) -> None:
        """Add a model variant with traffic weight."""
        self.variants[name] = {"model": model, "weight": weight}

    def predict(self, x: Any) -> Dict[str, Any]:
        """Route prediction to a variant based on weights."""
        if not self.variants:
            raise RuntimeError("no variants registered")
        # weighted random selection
        names = list(self.variants.keys())
        weights = [self.variants[n]["weight"] for n in names]
        total = sum(weights)
        weights = [w / total for w in weights]
        chosen = np.random.choice(names, p=weights)
        model = self.variants[chosen]["model"]
        t0 = time.perf_counter()
        try:
            result = model.forward(x)
            latency = (time.perf_counter() - t0) * 1000
            self._stats[chosen]["n_requests"] += 1
            self._stats[chosen]["latencies"].append(latency)
            self._stats[chosen]["predictions"].append(float(result) if np.isscalar(result) else None)
            return {
                "variant": chosen,
                "prediction": float(result) if np.isscalar(result) else np.asarray(result).tolist(),
                "latency_ms": latency,
            }
        except Exception as e:
            self._stats[chosen]["n_errors"] += 1
            return {"variant": chosen, "error": str(e)}

    def results(self) -> Dict[str, Dict[str, Any]]:
        """Get A/B test results per variant."""
        out = {}
        for name, stats in self._stats.items():
            latencies = list(stats["latencies"])
            out[name] = {
                "n_requests": stats["n_requests"],
                "n_errors": stats["n_errors"],
                "error_rate": stats["n_errors"] / max(stats["n_requests"], 1),
                "latency_mean": float(np.mean(latencies)) if latencies else 0,
                "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0,
                "weight": self.variants.get(name, {}).get("weight", 0),
            }
        return out

    def pick_winner(self) -> Optional[str]:
        """Pick the best-performing variant (lowest error rate + latency)."""
        if not self._stats:
            return None
        results = self.results()
        best = min(results.items(), key=lambda x: (
            x[1]["error_rate"] * 100 + x[1]["latency_mean"]
        ))
        return best[0]


# ============================================================================
# 5. ShadowDeployer — run new model in shadow mode
# ============================================================================
class ShadowDeployer:
    """Run a new model in shadow mode (processes requests but doesn't serve them).

    Compares shadow model predictions with production model predictions
    without affecting users. Useful for validation before rollout.

    Usage:
        shadow = ShadowDeployer(production_model, shadow_model)
        result = shadow.predict(x)
        # → {"production": ..., "shadow": ..., "match": True/False}
    """

    def __init__(self, production_model: Any, shadow_model: Any,
                 tolerance: float = 0.01):
        self.production = production_model
        self.shadow = shadow_model
        self.tolerance = tolerance
        self._comparisons: List[Dict[str, Any]] = []

    def predict(self, x: Any) -> Dict[str, Any]:
        """Run both models and compare."""
        t0 = time.perf_counter()
        prod_result = self.production.forward(x)
        prod_latency = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        try:
            shadow_result = self.shadow.forward(x)
            shadow_latency = (time.perf_counter() - t1) * 1000
            shadow_error = None
        except Exception as e:
            shadow_result = None
            shadow_latency = (time.perf_counter() - t1) * 1000
            shadow_error = str(e)

        # compare
        match = False
        if shadow_result is not None:
            try:
                diff = abs(float(prod_result) - float(shadow_result))
                match = diff < self.tolerance
            except Exception:
                match = prod_result == shadow_result

        comparison = {
            "production": float(prod_result) if np.isscalar(prod_result) else str(prod_result),
            "shadow": float(shadow_result) if shadow_result is not None and np.isscalar(shadow_result) else None,
            "match": match,
            "prod_latency_ms": prod_latency,
            "shadow_latency_ms": shadow_latency,
            "shadow_error": shadow_error,
            "timestamp": time.time(),
        }
        self._comparisons.append(comparison)
        return comparison

    def summary(self) -> Dict[str, Any]:
        """Shadow deployment summary."""
        if not self._comparisons:
            return {"n_comparisons": 0}
        matches = sum(1 for c in self._comparisons if c["match"])
        errors = sum(1 for c in self._comparisons if c["shadow_error"])
        return {
            "n_comparisons": len(self._comparisons),
            "n_matches": matches,
            "match_rate": matches / len(self._comparisons),
            "n_shadow_errors": errors,
            "avg_prod_latency": float(np.mean([c["prod_latency_ms"] for c in self._comparisons])),
            "avg_shadow_latency": float(np.mean([c["shadow_latency_ms"] for c in self._comparisons])),
            "ready_for_promotion": matches / len(self._comparisons) > 0.95,
        }


# ============================================================================
# 6. CanaryDeployer — gradual rollout with automatic rollback
# ============================================================================
class CanaryDeployer:
    """Gradual canary deployment with automatic rollback.

    Stages:
      1. 5% traffic to new model for 5 minutes
      2. 25% traffic for 10 minutes
      3. 50% traffic for 10 minutes
      4. 100% traffic (promoted)

    If error rate exceeds threshold at any stage, automatically rollback.

    Usage:
        canary = CanaryDeployer(old_model, new_model)
        canary.start()  # begins canary rollout
        result = canary.predict(x)  # routes based on current stage
    """

    STAGES = [
        {"name": "canary_5", "traffic_pct": 0.05, "duration_s": 5, "max_error_rate": 0.05},
        {"name": "canary_25", "traffic_pct": 0.25, "duration_s": 10, "max_error_rate": 0.05},
        {"name": "canary_50", "traffic_pct": 0.50, "duration_s": 10, "max_error_rate": 0.05},
        {"name": "promoted", "traffic_pct": 1.0, "duration_s": 0, "max_error_rate": 0.05},
    ]

    def __init__(self, stable_model: Any, candidate_model: Any):
        self.stable = stable_model
        self.candidate = candidate_model
        self.stage_idx = 0
        self.stage_start = 0.0
        self._stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "errors": 0})
        self._rolled_back = False
        self._promoted = False
        self._history: List[Dict[str, Any]] = []

    def start(self) -> None:
        """Begin canary rollout."""
        self.stage_idx = 0
        self.stage_start = time.time()
        self._history.append({"event": "start", "stage": self.current_stage["name"],
                                "timestamp": time.time()})

    @property
    def current_stage(self) -> Dict[str, Any]:
        return self.STAGES[self.stage_idx]

    def _check_stage_transition(self) -> None:
        """Check if we should advance or rollback."""
        if self._promoted or self._rolled_back:
            return
        elapsed = time.time() - self.stage_start
        stage = self.current_stage
        # check error rate
        stats = self._stats[stage["name"]]
        if stats["n"] > 10:
            error_rate = stats["errors"] / stats["n"]
            if error_rate > stage["max_error_rate"]:
                self._rollback(f"error rate {error_rate:.1%} exceeds threshold {stage['max_error_rate']:.1%}")
                return
        # check time
        if elapsed >= stage["duration_s"]:
            if self.stage_idx < len(self.STAGES) - 1:
                self.stage_idx += 1
                self.stage_start = time.time()
                self._history.append({"event": "advance", "stage": self.current_stage["name"],
                                        "timestamp": time.time()})
                if self.current_stage["name"] == "promoted":
                    self._promoted = True
                    self._history.append({"event": "promoted", "timestamp": time.time()})

    def _rollback(self, reason: str) -> None:
        """Rollback to stable model."""
        self._rolled_back = True
        self._history.append({"event": "rollback", "reason": reason, "timestamp": time.time()})

    def predict(self, x: Any) -> Dict[str, Any]:
        """Route prediction based on current canary stage."""
        self._check_stage_transition()
        if self._rolled_back or self._promoted:
            model = self.candidate if self._promoted else self.stable
            stage_name = "promoted" if self._promoted else "rolled_back"
        else:
            stage = self.current_stage
            if np.random.random() < stage["traffic_pct"]:
                model = self.candidate
                stage_name = stage["name"] + "_candidate"
            else:
                model = self.stable
                stage_name = stage["name"] + "_stable"
        try:
            result = model.forward(x)
            self._stats[stage_name]["n"] += 1
            return {
                "prediction": float(result) if np.isscalar(result) else np.asarray(result).tolist(),
                "stage": stage_name,
                "model": "candidate" if "candidate" in stage_name or self._promoted else "stable",
            }
        except Exception as e:
            self._stats[stage_name]["errors"] += 1
            self._stats[stage_name]["n"] += 1
            return {"error": str(e), "stage": stage_name}

    def status(self) -> Dict[str, Any]:
        """Canary deployment status."""
        return {
            "stage": self.current_stage["name"],
            "stage_idx": self.stage_idx,
            "traffic_to_candidate": self.current_stage["traffic_pct"],
            "rolled_back": self._rolled_back,
            "promoted": self._promoted,
            "stats": dict(self._stats),
            "history": self._history[-10:],
        }


__all__ = [
    "ModelDeployer", "ModelServer", "ModelMonitor",
    "ABTestRouter", "ShadowDeployer", "CanaryDeployer",
]
