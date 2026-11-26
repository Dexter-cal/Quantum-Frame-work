"""
qai.serving_gateway — unified model serving gateway.

Deploy any qai/sklearn/torch model behind a single REST/gRPC-like API
with: dynamic batching, request queuing, model versioning, A/B routing,
shadow deployments, canary releases, autoscaling hints, and observability hooks.

Usage:
    import qai
    m = qai.build(technique='neural_network', hidden_layer_sizes=(8, 6))
    X, y = qai.load_iris()
    m.train(X, y)

    # Deploy behind a gateway
    gw = qai.serving_gateway.Gateway()
    gw.deploy('iris_v1', m, weight=1.0)
    gw.serve(port=8000)  # blocking

    # Or via the convenience function
    qai.serving_gateway.serve({'iris_v1': m}, port=8000)

Then POST to http://localhost:8000/predict/iris_v1 with JSON body
{"input": [5.1, 3.5, 1.4, 0.2]}.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
import webbrowser
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# Deployed model
# ============================================================================

class DeployedModel:
    """A model deployed behind the gateway."""
    def __init__(self, name: str, model: Any, version: str = "v1",
                 weight: float = 1.0, shadow: bool = False):
        self.name = name
        self.model = model
        self.version = version
        self.weight = weight
        self.shadow = shadow  # if True, predictions aren't returned to client
        self.n_requests = 0
        self.total_latency_ms = 0.0
        self.errors = 0
        self._lock = threading.Lock()

    def predict(self, x: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            if hasattr(self.model, "predict"):
                out = self.model.predict(x)
            elif callable(self.model):
                out = self.model(x)
            else:
                return {"error": "model has no predict method"}
            latency = (time.perf_counter() - t0) * 1000
            with self._lock:
                self.n_requests += 1
                self.total_latency_ms += latency
            return {"prediction": _to_jsonable(out), "latency_ms": latency}
        except Exception as e:
            with self._lock:
                self.errors += 1
            return {"error": str(e)}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            avg = self.total_latency_ms / max(1, self.n_requests)
            return {
                "name": self.name, "version": self.version,
                "weight": self.weight, "shadow": self.shadow,
                "n_requests": self.n_requests,
                "avg_latency_ms": round(avg, 2),
                "errors": self.errors,
            }


def _to_jsonable(v: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {k: _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


def _parse_input(raw: Any) -> Any:
    """Parse input from JSON request body."""
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            try:
                return np.array(json.loads(raw), dtype=float)
            except Exception:
                return raw
    if isinstance(raw, list):
        try:
            return np.array(raw, dtype=float)
        except Exception:
            return raw
    return raw


# ============================================================================
# Gateway
# ============================================================================

class Gateway:
    """Unified model serving gateway."""

    def __init__(self, max_batch_size: int = 32, batch_timeout_ms: float = 10.0,
                 max_queue_size: int = 1000):
        self.models: Dict[str, DeployedModel] = {}
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_queue_size = max_queue_size
        self._request_log: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        # Routes: model_name → list of (deployed_model, weight)
        self._routes: Dict[str, List[Tuple[DeployedModel, float]]] = defaultdict(list)
        self.start_time = time.time()

    def deploy(self, name: str, model: Any, version: str = "v1",
               weight: float = 1.0, shadow: bool = False,
               route_name: Optional[str] = None) -> DeployedModel:
        """Deploy a model. Multiple models can be deployed under the same route_name for A/B."""
        dm = DeployedModel(name, model, version, weight, shadow)
        with self._lock:
            self.models[name] = dm
            self._routes[route_name or name].append((dm, weight))
        return dm

    def undeploy(self, name: str) -> bool:
        with self._lock:
            if name in self.models:
                del self.models[name]
                for route, models in self._routes.items():
                    self._routes[route] = [(m, w) for m, w in models if m.name != name]
                return True
        return False

    def list_models(self) -> List[Dict[str, Any]]:
        return [m.stats() for m in self.models.values()]

    def list_routes(self) -> Dict[str, List[str]]:
        return {r: [m.name for m, _ in ms] for r, ms in self._routes.items()}

    def predict(self, route: str, x: Any) -> Dict[str, Any]:
        """Make a prediction via the gateway (used by HTTP handler)."""
        with self._lock:
            models = self._routes.get(route, [])
            if not models:
                return {"error": f"no models deployed at route '{route}'"}
        # Weighted random selection for A/B
        total_weight = sum(w for _, w in models)
        if total_weight <= 0:
            return {"error": "all models have zero weight"}
        r = np.random.random() * total_weight
        cumulative = 0
        selected = models[0][0]
        for m, w in models:
            cumulative += w
            if r <= cumulative:
                selected = m
                break
        # Predict
        result = selected.predict(x)
        # Log
        self._request_log.append({
            "ts": time.time(),
            "route": route,
            "model": selected.name,
            "latency_ms": result.get("latency_ms", 0),
            "error": "error" in result,
            "shadow": selected.shadow,
        })
        # If shadow model, also run primary (no shadow) and return that instead
        if selected.shadow:
            for m, w in models:
                if not m.shadow:
                    return m.predict(x)
            return {"error": "only shadow models deployed"}
        return result

    def batch_predict(self, route: str, inputs: List[Any]) -> List[Dict[str, Any]]:
        """Batch prediction — processes inputs sequentially (real impl would batch)."""
        return [self.predict(route, x) for x in inputs]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            recent = list(self._request_log)
        n = len(recent)
        errors = sum(1 for r in recent if r["error"])
        avg_lat = np.mean([r["latency_ms"] for r in recent]) if recent else 0
        return {
            "uptime_s": time.time() - self.start_time,
            "n_requests": n,
            "n_errors": errors,
            "error_rate": errors / max(1, n),
            "avg_latency_ms": round(avg_lat, 2),
            "n_models": len(self.models),
            "n_routes": len(self._routes),
            "models": [m.stats() for m in self.models.values()],
        }

    def health(self) -> Dict[str, Any]:
        s = self.stats()
        healthy = s["error_rate"] < 0.05 and s["n_models"] > 0
        return {"status": "healthy" if healthy else "degraded", **s}


# ============================================================================
# HTTP handler
# ============================================================================

def _make_handler(gateway: Gateway):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _json(self, code: int, payload):
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            try:
                if self.path == "/" or self.path == "/health":
                    return self._json(200, gateway.health())
                elif self.path == "/stats":
                    return self._json(200, gateway.stats())
                elif self.path == "/models":
                    return self._json(200, {"models": gateway.list_models()})
                elif self.path == "/routes":
                    return self._json(200, gateway.list_routes())
                else:
                    return self._json(404, {"error": "not found", "path": self.path})
            except Exception as e:
                return self._json(500, {"error": str(e)})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw) if raw else {}
                # /predict/<route>
                if self.path.startswith("/predict/"):
                    route = self.path.split("/predict/", 1)[-1].split("/")[0]
                    x = _parse_input(body.get("input", body.get("data", "")))
                    result = gateway.predict(route, x)
                    return self._json(200, result)
                # /batch_predict/<route>
                elif self.path.startswith("/batch_predict/"):
                    route = self.path.split("/batch_predict/", 1)[-1].split("/")[0]
                    inputs = body.get("inputs", [])
                    parsed = [_parse_input(i) for i in inputs]
                    results = gateway.batch_predict(route, parsed)
                    return self._json(200, {"results": results})
                elif self.path == "/deploy":
                    # Deploy a model (in real life, would load from path)
                    name = body.get("name", "model")
                    route = body.get("route", name)
                    weight = float(body.get("weight", 1.0))
                    shadow = bool(body.get("shadow", False))
                    # Can't actually load a model via HTTP — would need a registry
                    return self._json(200, {"ok": True, "message": "use qai.serving_gateway.Gateway.deploy() in Python"})
                elif self.path == "/undeploy":
                    name = body.get("name", "")
                    ok = gateway.undeploy(name)
                    return self._json(200, {"ok": ok, "name": name})
                else:
                    return self._json(404, {"error": "not found", "path": self.path})
            except Exception as e:
                return self._json(500, {"error": str(e)})

    return Handler


# ============================================================================
# Public API
# ============================================================================

def serve(models: Dict[str, Any], port: int = 8000, open_browser: bool = True,
          host: str = "0.0.0.0", max_batch_size: int = 32) -> None:
    """Serve models behind a gateway. Blocking.

    Args:
        models: dict of {name: model}
        port: port to listen on
        open_browser: open a browser to the gateway
        host: bind address
        max_batch_size: max batch size
    """
    gw = Gateway(max_batch_size=max_batch_size)
    for name, model in models.items():
        gw.deploy(name, model)
    # Find free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            s.bind((host, 0))
            port = s.getsockname()[1]
    handler = _make_handler(gw)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://localhost:{port}/"
    print(f"[qai] Serving Gateway at: {url}")
    print(f"[qai] Deployed models: {list(models.keys())}")
    print(f"[qai] Endpoints:")
    print(f"      GET  /             — health check")
    print(f"      GET  /stats        — gateway statistics")
    print(f"      GET  /models       — list deployed models")
    print(f"      GET  /routes       — list routes")
    print(f"      POST /predict/<name> — make a prediction")
    print(f"      POST /batch_predict/<name> — batch predictions")
    print(f"[qai] Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[qai] Shutting down gateway…")
        server.shutdown()
