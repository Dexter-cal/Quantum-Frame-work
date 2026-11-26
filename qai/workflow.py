"""
qai.workflow -- node-based visual workflow / pipeline builder for ML.

Think n8n / Node-RED / LangFlow / Microsoft Fabric: users compose a directed
acyclic graph of operations (load_dataset -> split -> build -> train ->
evaluate -> save -> deploy) by adding nodes, connecting them with edges
(named ports), and executing the whole graph in topological order.

Quickstart:
    import qai.workflow as wf

    # 1. Build a workflow from a pre-built template
    g = wf.create('classification').build()
    print(g.visualize())         # Mermaid markdown
    result = g.execute()          # run end-to-end on iris

    # 2. Build a custom workflow programmatically
    g = (wf.create()
            .add('load_dataset', name='load', dataset='iris')
            .add('split_train_test', name='split', test_size=0.2)
            .add('build_model', name='build', technique='random_forest')
            .add('train_model', name='train')
            .add('evaluate_model', name='evaluate')
            .connect('load', 'split', 'dataset', 'dataset')
            .connect('split', 'train', 'train', 'X')
            .connect('build', 'train', 'model', 'model')
            .connect('train', 'evaluate', 'trained_model', 'model')
            .connect('split', 'evaluate', 'test', 'X')
            .build())
    result = g.execute()

    # 3. One-shot execute a template
    metrics = wf.execute('classification')

Persistence:
    g.save('workflow.json')      # JSON serialization
    g2 = wf.load('workflow.json')

Visualization:
    g.visualize()                # Mermaid markdown
    WorkflowVisualizer().to_html(g)  # self-contained HTML
    WorkflowVisualizer().to_d3(g)    # D3.js node-link data

Design:
  - All qai operations are lazy-imported so importing this module never
    crashes even if torch / sklearn / onnx aren't installed.
  - Each operation handles errors gracefully and returns {'_error': str}
    instead of raising — so a failure in one node doesn't take down the
    whole graph; downstream nodes see the error string in their input.
  - Topological sort uses Kahn's algorithm. Cycles raise ValueError.
  - Nodes receive their inputs in declared port order. Unconnected ports
    pass None. Tuples like (X, y) are auto-unpacked where appropriate.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


# =====================================================================
# Helpers
# =====================================================================
def _json_safe(obj: Any) -> Any:
    """Coerce `obj` into something JSON-serializable (str fallback)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    # Anything non-serializable (callables, models, arrays, ...) -> repr
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)[:200]


def _sanitize_mermaid_id(s: str) -> str:
    """Mermaid IDs must be alphanumeric + underscore."""
    out = []
    for ch in str(s):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "node"


def _unpack_xy(X: Any, y: Any = None):
    """If X is a (X, y) tuple and y is None, unpack it."""
    if isinstance(X, tuple) and len(X) == 2 and y is None:
        return X[0], X[1]
    return X, y


# =====================================================================
# Built-in operations
# =====================================================================
# Each entry is a dict with keys:
#   fn:          callable(*inputs, **config) -> dict {port: value}
#   inputs:      list of input port names (in order)
#   outputs:     list of output port names
#   description: short human-readable description
#
# Operations must:
#   - lazy-import qai (so the module is import-safe without torch/sklearn)
#   - return {'_error': str} on failure instead of raising
#   - accept extra **config gracefully


def _op_load_dataset(*args, **config) -> dict:
    """Load a built-in dataset (iris, digits, wine, breast_cancer, diabetes, ...)."""
    name = config.get("dataset", config.get("name", "iris"))
    try:
        import qai
        loaders = {
            "iris": qai.load_iris,
            "digits": qai.load_digits,
            "wine": qai.load_wine,
            "breast_cancer": qai.load_breast_cancer,
            "diabetes": qai.load_diabetes,
            "california_housing": qai.load_california_housing,
        }
        loader = loaders.get(name)
        if loader is None:
            return {"_error": f"unknown dataset '{name}'. Available: {list(loaders)}"}
        X, y = loader()
        return {"dataset": (X, y)}
    except Exception as e:
        return {"_error": f"load_dataset failed: {e}"}


def _op_split_train_test(dataset=None, **config) -> dict:
    """Split a (X, y) dataset into train and test partitions."""
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    if dataset is None:
        return {"_error": "split_train_test requires a dataset input"}
    X, y = _unpack_xy(dataset)
    try:
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        try:
            from sklearn.model_selection import train_test_split as _split
            if y is not None:
                X_tr, X_te, y_tr, y_te = _split(
                    X, y, test_size=test_size, random_state=random_state)
            else:
                X_tr, X_te = _split(X, test_size=test_size,
                                     random_state=random_state)
                y_tr = y_te = None
        except ImportError:
            n = len(X)
            rng = np.random.RandomState(random_state)
            idx = rng.permutation(n)
            split = int(n * (1 - test_size))
            tr_idx, te_idx = idx[:split], idx[split:]
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr = y[tr_idx] if y is not None else None
            y_te = y[te_idx] if y is not None else None
        return {"train": (X_tr, y_tr), "test": (X_te, y_te)}
    except Exception as e:
        return {"_error": f"split_train_test failed: {e}"}


def _op_build_model(*args, **config) -> dict:
    """Build a qai model from a technique name + params."""
    technique = config.get("technique", "neural_network")
    params = {k: v for k, v in config.items() if k != "technique"}
    try:
        import qai
        m = qai.build(technique=technique, **params)
        return {"model": m}
    except Exception as e:
        return {"_error": f"build_model failed: {e}"}


def _op_train_model(model=None, X=None, y=None, **config) -> dict:
    """Train a model on (X, y). X may be a (X, y) tuple."""
    if model is None:
        return {"_error": "train_model requires a model input"}
    if X is None:
        return {"_error": "train_model requires training data (X)"}
    X, y = _unpack_xy(X, y)
    try:
        verbose = bool(config.get("verbose", False))
        trained = False
        if y is not None:
            try:
                model.train(X, y, verbose=verbose)
                trained = True
            except (TypeError, ValueError):
                model.train(X, verbose=verbose)
                trained = True
        if not trained:
            model.train(X, verbose=verbose)
        return {"trained_model": model}
    except Exception as e:
        return {"_error": f"train_model failed: {e}"}


def _op_evaluate_model(model=None, X=None, y=None, **config) -> dict:
    """Evaluate a trained model on (X, y)."""
    if model is None:
        return {"_error": "evaluate_model requires a model input"}
    if X is None:
        return {"_error": "evaluate_model requires data (X)"}
    X, y = _unpack_xy(X, y)
    try:
        X_arr = np.asarray(X)
        n = len(X_arr)
        preds = []
        for i in range(n):
            try:
                preds.append(model.predict(X_arr[i]))
            except Exception:
                # Some models accept batched input directly
                try:
                    preds = list(model.predict(X_arr))
                    break
                except Exception:
                    preds.append(None)
        metrics: Dict[str, Any] = {
            "n_samples": int(n),
            "n_predictions": int(len(preds)),
        }
        if y is not None:
            y_arr = np.asarray(y)
            try:
                correct = sum(1 for p, t in zip(preds, y_arr)
                              if p is not None and p == t)
                metrics["accuracy"] = float(correct) / max(len(y_arr), 1)
            except Exception:
                # Regression-style metrics
                try:
                    errors = [(p - t) ** 2 for p, t in zip(preds, y_arr)
                              if p is not None]
                    if errors:
                        metrics["mse"] = float(np.mean(errors))
                        metrics["rmse"] = float(np.sqrt(np.mean(errors)))
                except Exception:
                    pass
        return {"metrics": metrics}
    except Exception as e:
        return {"_error": f"evaluate_model failed: {e}"}


def _op_predict(model=None, X=None, **config) -> dict:
    """Run model.predict(X) and return predictions."""
    if model is None:
        return {"_error": "predict requires a model input"}
    if X is None:
        return {"_error": "predict requires data (X)"}
    try:
        if isinstance(X, tuple) and len(X) == 2:
            X = X[0]
        X_arr = np.asarray(X)
        preds = []
        for i in range(len(X_arr)):
            try:
                preds.append(model.predict(X_arr[i]))
            except Exception:
                try:
                    preds = list(model.predict(X_arr))
                    break
                except Exception as inner:
                    return {"_error": f"predict failed: {inner}"}
        return {"predictions": preds}
    except Exception as e:
        return {"_error": f"predict failed: {e}"}


def _op_save_model(*args, **config) -> dict:
    """Save a model to disk. Path can come from a port or config.

    Using *args (rather than named `path`) so the same op can accept
    a value on its `path` input port AND a `path=` config fallback
    without Python's 'got multiple values for argument' error.
    """
    model = args[0] if len(args) > 0 else None
    path_arg = args[1] if len(args) > 1 else None
    if model is None:
        return {"_error": "save_model requires a model input"}
    # Priority: input port value > config['path'] > default temp path
    path = path_arg if path_arg is not None else config.get("path")
    if not path:
        path = os.path.join(tempfile_get(), "qai_workflow_model.qmodel")
    try:
        import qai
        qai.save_model(model, str(path))
        return {"path": str(path)}
    except Exception as e:
        return {"_error": f"save_model failed: {e}"}


def _op_load_model(*args, **config) -> dict:
    """Load a model from disk or from HuggingFace model zoo.

    Using *args so the same op can accept a value on its `path` input
    port AND a `path=` config fallback without collision.
    """
    path_arg = args[0] if len(args) > 0 else None
    path = path_arg if path_arg is not None else config.get("path")
    if not path:
        return {"_error": "load_model requires a path"}
    try:
        import qai
        if os.path.exists(str(path)):
            m = qai.load_model(str(path))
        else:
            # Treat as a HuggingFace / pretrained zoo alias
            m = None
            try:
                m = qai.from_huggingface(str(path))
            except Exception:
                try:
                    m = qai.load_pretrained(str(path))
                except Exception as inner:
                    return {"_error": f"load_model failed: cannot resolve '{path}': {inner}"}
        return {"model": m}
    except Exception as e:
        return {"_error": f"load_model failed: {e}"}


def _op_quantize(model=None, **config) -> dict:
    """Quantize a model (int8 dynamic / int4 / static)."""
    if model is None:
        return {"_error": "quantize requires a model input"}
    fmt = config.get("format", "int8")
    try:
        import qai
        if fmt in ("int8", "dynamic"):
            qm = qai.quantize_dynamic(model)
        elif fmt in ("int4", "nf4", "fp4"):
            qt = fmt if fmt in ("nf4", "fp4") else "nf4"
            qm = qai.quantize_to_int4(model, quant_type=qt)
        elif fmt == "static":
            calib = config.get("calibration_data")
            if calib is None:
                return {"_error": "static quantization requires 'calibration_data'"}
            qm = qai.quantize_static(model, calib)
        else:
            return {"_error": f"unknown quantization format '{fmt}'"}
        return {"quantized_model": qm}
    except Exception as e:
        return {"_error": f"quantize failed: {e}"}


def _op_convert(*args, **config) -> dict:
    """Convert a model to another format (onnx, tflite, gguf, ...).

    Using *args so the `format` input port and `format=` config
    fallback don't collide on Python's keyword-argument binding.
    """
    model = args[0] if len(args) > 0 else None
    format_arg = args[1] if len(args) > 1 else None
    if model is None:
        return {"_error": "convert requires a model input"}
    fmt = format_arg if format_arg is not None else config.get("format", "onnx")
    try:
        import qai
        path = qai.conversion.export(model, fmt)
        try:
            loaded = qai.conversion.import_model(path)
        except Exception:
            loaded = None
        return {"converted_model": loaded if loaded is not None else model,
                "path": str(path)}
    except Exception as e:
        return {"_error": f"convert failed: {e}"}


def _op_explain(model=None, X=None, **config) -> dict:
    """Explain a model's prediction for a single sample (or first row of X)."""
    if model is None:
        return {"_error": "explain requires a model input"}
    if X is None:
        return {"_error": "explain requires data (X)"}
    try:
        if isinstance(X, tuple) and len(X) == 2:
            X = X[0]
        X_arr = np.asarray(X)
        sample = X_arr[0] if X_arr.ndim >= 1 and len(X_arr) > 0 else X_arr
        if hasattr(model, "explain"):
            result = model.explain(sample)
        else:
            try:
                from .core.explain import explain as _core_explain
                result = _core_explain(model, sample)
            except Exception:
                result = {"method": "none", "reasoning": "explain() not available"}
        return {"explanations": result}
    except Exception as e:
        return {"_error": f"explain failed: {e}"}


def _op_benchmark(model=None, **config) -> dict:
    """Benchmark a model: parameters, latency, prediction throughput."""
    if model is None:
        return {"_error": "benchmark requires a model input"}
    try:
        report: Dict[str, Any] = {
            "technique": getattr(model, "technique_name", "unknown"),
        }
        # Trained status
        try:
            tech = getattr(model, "technique", None)
            if tech is not None and hasattr(tech, "is_trained"):
                report["trained"] = bool(tech.is_trained())
        except Exception:
            pass
        # Parameter count
        try:
            if hasattr(model, "n_parameters"):
                params = model.n_parameters()
                if isinstance(params, dict):
                    report["parameters"] = params
                else:
                    report["parameters"] = {"total": int(params)}
        except Exception:
            pass
        # Inference latency on a tiny sample (if possible)
        try:
            tech = getattr(model, "technique", None)
            if tech is not None and hasattr(tech, "is_trained") and tech.is_trained():
                # Find a representative input
                sample_input = None
                last_in = getattr(model, "_last_input", None)
                if last_in is not None:
                    sample_input = last_in
                if sample_input is not None:
                    t0 = time.perf_counter()
                    for _ in range(5):
                        try:
                            model.predict(sample_input)
                        except Exception:
                            break
                    elapsed = time.perf_counter() - t0
                    report["inference_latency_ms"] = round(elapsed * 1000 / 5, 3)
        except Exception:
            pass
        return {"report": report}
    except Exception as e:
        return {"_error": f"benchmark failed: {e}"}


def _op_deploy(*args, **config) -> dict:
    """Generate deployment artifacts (Flask / FastAPI / Docker / CLI).

    Using *args so the `target` input port and `target=` config
    fallback don't collide on Python's keyword-argument binding.
    """
    model = args[0] if len(args) > 0 else None
    target_arg = args[1] if len(args) > 1 else None
    if model is None:
        return {"_error": "deploy requires a model input"}
    target = target_arg if target_arg is not None else config.get("target", "flask")
    try:
        import qai
        deployer = qai.ModelDeployer
        method_map = {
            "flask": "deploy_flask",
            "fastapi": "deploy_fastapi",
            "docker": "deploy_docker",
            "cli": "deploy_cli",
        }
        method_name = method_map.get(target, "deploy_flask")
        method = getattr(deployer, method_name, None)
        if method is None:
            return {"_error": f"unknown deploy target '{target}'"}
        result = method(model)
        # deploy_docker returns (dockerfile, requirements); others return code
        if isinstance(result, tuple):
            code = result[0]
        else:
            code = result
        endpoint = f"http://localhost:5000/predict ({target} generated)"
        return {"endpoint": endpoint, "code": code}
    except Exception as e:
        return {"_error": f"deploy failed: {e}"}


def tempfile_get() -> str:
    """Return a writable temp directory (cached)."""
    import tempfile
    return tempfile.gettempdir()


# The registry of built-in operations.
OPERATIONS: Dict[str, Dict[str, Any]] = {
    "load_dataset": {
        "fn": _op_load_dataset,
        "inputs": [],
        "outputs": ["dataset"],
        "description": "Load a built-in dataset (iris, digits, wine, ...).",
    },
    "split_train_test": {
        "fn": _op_split_train_test,
        "inputs": ["dataset"],
        "outputs": ["train", "test"],
        "description": "Split a (X, y) dataset into train/test partitions.",
    },
    "build_model": {
        "fn": _op_build_model,
        "inputs": [],
        "outputs": ["model"],
        "description": "Build a qai model from a technique name + params.",
    },
    "train_model": {
        "fn": _op_train_model,
        "inputs": ["model", "X", "y"],
        "outputs": ["trained_model"],
        "description": "Train a model on (X, y). X may be a (X, y) tuple.",
    },
    "evaluate_model": {
        "fn": _op_evaluate_model,
        "inputs": ["model", "X", "y"],
        "outputs": ["metrics"],
        "description": "Evaluate a trained model on (X, y).",
    },
    "predict": {
        "fn": _op_predict,
        "inputs": ["model", "X"],
        "outputs": ["predictions"],
        "description": "Run model.predict(X) and return predictions.",
    },
    "save_model": {
        "fn": _op_save_model,
        "inputs": ["model", "path"],
        "outputs": ["path"],
        "description": "Save a model to disk. Path can come from a port or config.",
    },
    "load_model": {
        "fn": _op_load_model,
        "inputs": ["path"],
        "outputs": ["model"],
        "description": "Load a model from disk or from HuggingFace / pretrained zoo.",
    },
    "quantize": {
        "fn": _op_quantize,
        "inputs": ["model"],
        "outputs": ["quantized_model"],
        "description": "Quantize a model (int8 dynamic / int4 / static).",
    },
    "convert": {
        "fn": _op_convert,
        "inputs": ["model", "format"],
        "outputs": ["converted_model"],
        "description": "Convert a model to another format (onnx, tflite, gguf, ...).",
    },
    "explain": {
        "fn": _op_explain,
        "inputs": ["model", "X"],
        "outputs": ["explanations"],
        "description": "Explain a model's prediction for a single sample.",
    },
    "benchmark": {
        "fn": _op_benchmark,
        "inputs": ["model"],
        "outputs": ["report"],
        "description": "Benchmark a model: parameters, latency, throughput.",
    },
    "deploy": {
        "fn": _op_deploy,
        "inputs": ["model", "target"],
        "outputs": ["endpoint"],
        "description": "Generate deployment artifacts (Flask / FastAPI / Docker / CLI).",
    },
}


def register_operation(name: str, fn: Callable, inputs: List[str],
                        outputs: List[str], description: str = "") -> None:
    """Register a custom operation in the global OPERATIONS registry."""
    OPERATIONS[name] = {
        "fn": fn, "inputs": list(inputs), "outputs": list(outputs),
        "description": description or (fn.__doc__ or "").strip().split("\n")[0],
    }


# =====================================================================
# WorkflowNode
# =====================================================================
class WorkflowNode:
    """A single node in the workflow graph.

    Args:
        node_id: unique identifier within the graph.
        op:      either a string naming a registered operation in OPERATIONS,
                 or a callable (*inputs, **config) -> dict.
        name:    human-readable name (defaults to node_id).
        config:  dict of configuration params forwarded to the op as **kwargs.
    """

    def __init__(self, node_id: str, op: Union[str, Callable],
                 name: Optional[str] = None, config: Optional[dict] = None):
        if not node_id or not isinstance(node_id, str):
            raise ValueError("node_id must be a non-empty string")
        self.node_id = node_id
        self.name = name or node_id
        self.config = dict(config or {})

        if isinstance(op, str):
            if op not in OPERATIONS:
                raise ValueError(
                    f"[WHAT] Unknown operation '{op}'\n"
                    f"[WHY] No operation with that name is registered\n"
                    f"[FIX] Available: {sorted(OPERATIONS.keys())}")
            entry = OPERATIONS[op]
            self.op_name = op
            self.op: Callable = entry["fn"]
            self.inputs: List[str] = list(entry.get("inputs", []))
            self.outputs: List[str] = list(entry.get("outputs", ["out"]))
            self.description: str = entry.get("description", "")
        elif callable(op):
            self.op_name = getattr(op, "__name__", "callable")
            self.op = op
            self.inputs = list(getattr(op, "_inputs", []))
            self.outputs = list(getattr(op, "_outputs", ["out"]))
            doc = (op.__doc__ or "").strip()
            self.description = doc.split("\n")[0] if doc else "custom callable"
        else:
            raise TypeError(
                f"op must be a string or callable, got {type(op).__name__}")

    def execute(self, *input_values: Any) -> dict:
        """Run the op with input_values (in port order) + config.

        Returns:
            dict mapping output port names to values. On failure, returns
            {'_error': str} instead of raising.
        """
        try:
            result = self.op(*input_values, **self.config)
            if isinstance(result, dict):
                return result
            # Wrap scalar/None returns in the first declared output port.
            if self.outputs:
                return {self.outputs[0]: result}
            return {"out": result}
        except Exception as e:
            return {"_error": f"node '{self.node_id}' ({self.op_name}) failed: {e}"}

    def validate(self) -> List[str]:
        """Return a list of validation error strings (empty if valid)."""
        errors: List[str] = []
        if not self.node_id:
            errors.append("node_id is empty")
        if not callable(self.op):
            errors.append(f"op is not callable: {self.op_name!r}")
        if not isinstance(self.config, dict):
            errors.append(f"config must be a dict, got {type(self.config).__name__}")
        if not isinstance(self.inputs, list):
            errors.append(f"inputs must be a list, got {type(self.inputs).__name__}")
        if not isinstance(self.outputs, list):
            errors.append(f"outputs must be a list, got {type(self.outputs).__name__}")
        # Duplicate port names
        if len(set(self.inputs)) != len(self.inputs):
            errors.append(f"duplicate input ports: {self.inputs}")
        if len(set(self.outputs)) != len(self.outputs):
            errors.append(f"duplicate output ports: {self.outputs}")
        return errors

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "op": self.op_name,
            "config": _json_safe(self.config),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowNode":
        op_name = d.get("op", "callable")
        # If op_name is a registered operation, resolve it; otherwise we
        # can't reconstruct a custom callable from JSON — fall back to a
        # no-op stub so the graph still loads (with a clear marker).
        if op_name in OPERATIONS:
            op: Union[str, Callable] = op_name
        else:
            op = op_name  # will raise in __init__ if not registered
        node = cls(
            node_id=d["node_id"],
            op=op,
            name=d.get("name"),
            config=d.get("config", {}),
        )
        # Preserve custom port info if the original was a callable
        if "inputs" in d:
            node.inputs = list(d["inputs"])
        if "outputs" in d:
            node.outputs = list(d["outputs"])
        if "description" in d:
            node.description = d["description"]
        return node

    def __repr__(self) -> str:
        return (f"<WorkflowNode {self.node_id!r} op={self.op_name!r} "
                f"in={self.inputs} out={self.outputs}>")


# =====================================================================
# WorkflowEdge
# =====================================================================
class WorkflowEdge:
    """A directed edge: source_node.source_port -> target_node.target_port."""

    def __init__(self, source_node_id: str, source_port: str,
                 target_node_id: str, target_port: str):
        self.source_node_id = source_node_id
        self.source_port = source_port
        self.target_node_id = target_node_id
        self.target_port = target_port

    @property
    def edge_id(self) -> str:
        return (f"{self.source_node_id}.{self.source_port}"
                f"->{self.target_node_id}.{self.target_port}")

    def to_dict(self) -> dict:
        return {
            "source_node_id": self.source_node_id,
            "source_port": self.source_port,
            "target_node_id": self.target_node_id,
            "target_port": self.target_port,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowEdge":
        return cls(
            source_node_id=d["source_node_id"],
            source_port=d["source_port"],
            target_node_id=d["target_node_id"],
            target_port=d["target_port"],
        )

    def __repr__(self) -> str:
        return f"<WorkflowEdge {self.edge_id}>"


# =====================================================================
# WorkflowGraph
# =====================================================================
class WorkflowGraph:
    """A directed acyclic graph of WorkflowNodes connected by WorkflowEdges."""

    def __init__(self, name: str = "workflow"):
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, WorkflowEdge] = {}

    # -- mutation -------------------------------------------------------
    def add_node(self, node: WorkflowNode) -> "WorkflowGraph":
        if node.node_id in self.nodes:
            raise ValueError(
                f"[WHAT] Node '{node.node_id}' already exists in graph\n"
                f"[WHY] Duplicate node id\n"
                f"[FIX] Use a unique node id")
        self.nodes[node.node_id] = node
        return self

    def remove_node(self, node_id: str) -> "WorkflowGraph":
        if node_id not in self.nodes:
            return self
        del self.nodes[node_id]
        # Remove all edges touching this node
        to_remove = [eid for eid, e in self.edges.items()
                     if e.source_node_id == node_id or e.target_node_id == node_id]
        for eid in to_remove:
            del self.edges[eid]
        return self

    def add_edge(self, edge: WorkflowEdge) -> "WorkflowGraph":
        if edge.source_node_id not in self.nodes:
            raise ValueError(
                f"[WHAT] Edge source node '{edge.source_node_id}' not in graph\n"
                f"[WHY] Cannot connect to a non-existent node\n"
                f"[FIX] Add the node before connecting it")
        if edge.target_node_id not in self.nodes:
            raise ValueError(
                f"[WHAT] Edge target node '{edge.target_node_id}' not in graph\n"
                f"[WHY] Cannot connect to a non-existent node\n"
                f"[FIX] Add the node before connecting it")
        self.edges[edge.edge_id] = edge
        return self

    def remove_edge(self, edge_id: str) -> "WorkflowGraph":
        self.edges.pop(edge_id, None)
        return self

    # -- analysis -------------------------------------------------------
    def topological_sort(self) -> List[str]:
        """Return node ids in topological order (Kahn's algorithm).

        Raises:
            ValueError: if the graph contains a cycle.
        """
        in_degree = {nid: 0 for nid in self.nodes}
        adj: Dict[str, List[str]] = defaultdict(list)
        for edge in self.edges.values():
            adj[edge.source_node_id].append(edge.target_node_id)
            in_degree[edge.target_node_id] += 1
        queue = deque([nid for nid, d in in_degree.items() if d == 0])
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(self.nodes):
            stuck = [nid for nid, d in in_degree.items() if d > 0]
            raise ValueError(
                f"[WHAT] Workflow graph has a cycle\n"
                f"[WHY] {len(stuck)} nodes are part of a cycle: {stuck}\n"
                f"[FIX] Remove edges that form cycles")
        return order

    def validate(self) -> List[str]:
        """Return a list of validation error strings. Empty list = valid."""
        errors: List[str] = []
        # Cycle check
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))
        # Node validation
        for nid, node in self.nodes.items():
            for err in node.validate():
                errors.append(f"node '{nid}': {err}")
        # Edge validation
        for eid, edge in self.edges.items():
            if edge.source_node_id not in self.nodes:
                errors.append(f"edge '{eid}': source node "
                              f"'{edge.source_node_id}' not in graph")
            if edge.target_node_id not in self.nodes:
                errors.append(f"edge '{eid}': target node "
                              f"'{edge.target_node_id}' not in graph")
            src = self.nodes.get(edge.source_node_id)
            tgt = self.nodes.get(edge.target_node_id)
            if src and edge.source_port not in src.outputs:
                errors.append(
                    f"edge '{eid}': source port '{edge.source_port}' not in "
                    f"node '{src.node_id}' outputs {src.outputs}")
            if tgt and edge.target_port not in tgt.inputs:
                errors.append(
                    f"edge '{eid}': target port '{edge.target_port}' not in "
                    f"node '{tgt.node_id}' inputs {tgt.inputs}")
        # Each input port can receive at most one edge
        seen_targets: Dict[tuple, str] = {}
        for eid, edge in self.edges.items():
            key = (edge.target_node_id, edge.target_port)
            if key in seen_targets:
                errors.append(
                    f"multiple edges target '{edge.target_node_id}."
                    f"{edge.target_port}' (e.g. '{eid}' and '{seen_targets[key]}')")
            else:
                seen_targets[key] = eid
        return errors

    # -- execution ------------------------------------------------------
    def execute(self, input_data: Optional[dict] = None,
                return_all: bool = False) -> dict:
        """Execute the graph in topological order.

        Args:
            input_data: optional dict providing initial values for nodes
                that have no incoming edges on a given port. Three forms
                are accepted:

                  * `{(node_id, port): value}`          - explicit per-port
                  * `{node_id: value}`                  - first input port
                  * `{node_id: [v0, v1, ...]}`          - unpack across
                                                           all input ports
            return_all: if True, return ALL node outputs (not just terminal
                nodes). Useful for debugging intermediate stages.

        Returns:
            dict mapping each terminal node id (nodes with no outgoing
            edges) to its outputs dict (port -> value). If `return_all=True`,
            returns outputs for every node. If the graph has no terminal
            nodes (a cycle, only reachable if `return_all=True` is forced
            through a non-cyclic graph), all node outputs are returned.

        Raises:
            ValueError: if the graph contains a cycle.
        """
        # Reject cycles up front
        order = self.topological_sort()

        # Build edge index: (target_node, target_port) -> (source_node, source_port)
        edge_index: Dict[tuple, tuple] = {}
        for edge in self.edges.values():
            edge_index[(edge.target_node_id, edge.target_port)] = (
                edge.source_node_id, edge.source_port)

        # Normalize input_data into a (node_id, port) -> value map.
        input_map: Dict[tuple, Any] = {}
        if input_data:
            for k, v in input_data.items():
                if isinstance(k, tuple):
                    input_map[k] = v
                    continue
                node = self.nodes.get(k)
                if node is None:
                    input_map[(k, "in")] = v
                    continue
                # If value is a list/tuple matching the node's input port
                # count, unpack it across the ports.
                if (node.inputs and isinstance(v, (list, tuple))
                        and len(v) == len(node.inputs)
                        and not (len(v) == 2 and node.inputs == ["dataset"])):
                    for port, sub_val in zip(node.inputs, v):
                        input_map[(k, port)] = sub_val
                elif node.inputs:
                    input_map[(k, node.inputs[0])] = v
                else:
                    input_map[(k, "in")] = v

        # Outputs per node: {node_id: {port: value}}
        outputs: Dict[str, dict] = {nid: {} for nid in self.nodes}

        for nid in order:
            node = self.nodes[nid]
            input_values: List[Any] = []
            for port in node.inputs:
                key = (nid, port)
                if key in edge_index:
                    src_nid, src_port = edge_index[key]
                    input_values.append(outputs.get(src_nid, {}).get(src_port))
                elif key in input_map:
                    input_values.append(input_map[key])
                else:
                    input_values.append(None)
            result = node.execute(*input_values)
            outputs[nid] = result if isinstance(result, dict) else {"out": result}

        # Cache the full output map for debugging
        self._last_outputs = outputs

        if return_all:
            return outputs
        # Terminal nodes = nodes with no outgoing edges
        has_outgoing = {e.source_node_id for e in self.edges.values()}
        terminal = {nid: outputs[nid]
                     for nid in self.nodes if nid not in has_outgoing}
        return terminal if terminal else outputs

    # -- visualization --------------------------------------------------
    def visualize(self, path: Optional[str] = None) -> str:
        """Render the graph as Mermaid markdown (and optionally save PNG)."""
        mermaid = WorkflowVisualizer().to_mermaid(self)
        if path:
            # Always write the Mermaid source as a .mmd alongside
            try:
                with open(path, "w") as f:
                    f.write(mermaid)
            except OSError:
                pass
            # If graphviz is available, also try to render a PNG
            try:
                import graphviz  # type: ignore
                dot = graphviz.Digraph(self.name)
                for nid, node in self.nodes.items():
                    dot.node(nid, f"{nid}\\n({node.op_name})")
                for edge in self.edges.values():
                    dot.edge(edge.source_node_id, edge.target_node_id,
                             label=f"{edge.source_port}->{edge.target_port}")
                png_path = os.path.splitext(path)[0] or "workflow"
                try:
                    dot.render(png_path, format="png", cleanup=True)
                except Exception:
                    pass
            except ImportError:
                # graphviz not installed — Mermaid text is still saved above
                pass
        return mermaid

    # -- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowGraph":
        g = cls(name=d.get("name", "workflow"))
        for nd in d.get("nodes", []):
            try:
                g.add_node(WorkflowNode.from_dict(nd))
            except Exception as e:
                # Skip nodes that reference unregistered custom callables
                # so a corrupt / partial JSON doesn't break the whole load.
                continue
        for ed in d.get("edges", []):
            try:
                g.add_edge(WorkflowEdge.from_dict(ed))
            except Exception:
                continue
        return g

    def save(self, path: str) -> str:
        """Save the graph to a JSON file. Returns the path."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "WorkflowGraph":
        """Load a graph from a JSON file saved by .save()."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    def __repr__(self) -> str:
        return (f"<WorkflowGraph {self.name!r} "
                f"nodes={len(self.nodes)} edges={len(self.edges)}>")


# =====================================================================
# WorkflowBuilder (fluent API)
# =====================================================================
class WorkflowBuilder:
    """Fluent builder for constructing WorkflowGraphs programmatically.

    Example:
        g = (WorkflowBuilder('clf')
                .add('load_dataset', name='load', dataset='iris')
                .add('build_model', name='build', technique='random_forest')
                .add('train_model', name='train')
                .connect('load', 'train', 'dataset', 'X')
                .connect('build', 'train', 'model', 'model')
                .build())
    """

    def __init__(self, name: str = "workflow"):
        self._graph = WorkflowGraph(name=name)
        self._counter = 0

    def add(self, op: Union[str, Callable], name: Optional[str] = None,
            **config) -> "WorkflowBuilder":
        """Add a node to the workflow.

        Args:
            op:    registered operation name (str) or callable.
            name:  unique node id within this workflow. If None, auto-generated.
            **config: configuration params forwarded to the op as **kwargs.

        Returns:
            self (for chaining).
        """
        if name is None:
            while f"node_{self._counter}" in self._graph.nodes:
                self._counter += 1
            name = f"node_{self._counter}"
            self._counter += 1
        node = WorkflowNode(node_id=name, op=op, name=name, config=config)
        self._graph.add_node(node)
        return self

    def connect(self, source_name: str, target_name: str,
                source_port: str = "out", target_port: str = "in"
                ) -> "WorkflowBuilder":
        """Connect two nodes with an edge.

        Args:
            source_name: id of the source node (must already be added).
            target_name: id of the target node (must already be added).
            source_port: output port on the source node (default 'out').
            target_port: input port on the target node (default 'in').

        Returns:
            self (for chaining).
        """
        edge = WorkflowEdge(source_name, source_port, target_name, target_port)
        self._graph.add_edge(edge)
        return self

    def build(self) -> WorkflowGraph:
        """Return the constructed WorkflowGraph."""
        return self._graph

    def __repr__(self) -> str:
        return (f"<WorkflowBuilder nodes={len(self._graph.nodes)} "
                f"edges={len(self._graph.edges)}>")


# =====================================================================
# WorkflowVisualizer (HTML / Mermaid / D3)
# =====================================================================
class WorkflowVisualizer:
    """Renders a WorkflowGraph in different formats for embedding in docs."""

    def to_mermaid(self, graph: WorkflowGraph) -> str:
        """Return Mermaid markdown for the graph."""
        lines = [f"graph TD"]
        # Node declarations
        for nid, node in graph.nodes.items():
            safe = _sanitize_mermaid_id(nid)
            label = f"{nid} ({node.op_name})"
            # Escape double quotes for Mermaid
            label = label.replace('"', "'")
            lines.append(f'    {safe}["{label}"]')
        # Edges
        for edge in graph.edges.values():
            src = _sanitize_mermaid_id(edge.source_node_id)
            tgt = _sanitize_mermaid_id(edge.target_node_id)
            port_label = f"{edge.source_port} -> {edge.target_port}"
            lines.append(f'    {src} -->|{port_label}| {tgt}')
        return "\n".join(lines)

    def to_html(self, graph: WorkflowGraph) -> str:
        """Return a self-contained HTML document with the Mermaid graph."""
        mermaid = self.to_mermaid(graph)
        d3_data = self.to_d3(graph)
        d3_json = json.dumps(d3_data, indent=2, default=str)
        # Use mermaid.js from CDN to render the graph inline
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{graph.name} - QAI Workflow</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #222; }}
  h1 {{ font-size: 20px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 12px; }}
  .mermaid {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 16px; }}
  pre.data {{ background: #f6f8fa; padding: 12px; border-radius: 6px; overflow: auto; font-size: 12px; }}
</style>
</head>
<body>
<h1>QAI Workflow: {graph.name}</h1>
<div class="meta">Nodes: {len(graph.nodes)} &middot; Edges: {len(graph.edges)}</div>
<div class="mermaid">
{mermaid}
</div>
<h2>D3.js node-link data</h2>
<pre class="data">{d3_json}</pre>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
</script>
</body>
</html>"""

    def to_d3(self, graph: WorkflowGraph) -> dict:
        """Return D3.js-compatible node-link data."""
        nodes = []
        for nid, node in graph.nodes.items():
            nodes.append({
                "id": nid,
                "label": node.name,
                "op": node.op_name,
                "inputs": list(node.inputs),
                "outputs": list(node.outputs),
                "config": _json_safe(node.config),
            })
        links = []
        for edge in graph.edges.values():
            links.append({
                "source": edge.source_node_id,
                "target": edge.target_node_id,
                "source_port": edge.source_port,
                "target_port": edge.target_port,
            })
        return {"nodes": nodes, "links": links, "name": graph.name}


# =====================================================================
# Pre-built workflow templates
# =====================================================================
def _build_classification(b: WorkflowBuilder) -> WorkflowBuilder:
    """load -> split -> build -> train -> evaluate -> save."""
    (b.add("load_dataset", name="load", dataset="iris")
      .add("split_train_test", name="split", test_size=0.2)
      .add("build_model", name="build", technique="neural_network")
      .add("train_model", name="train")
      .add("evaluate_model", name="evaluate")
      .add("save_model", name="save", path=os.path.join(tempfile_get(), "qai_clf.qmodel"))
      .connect("load", "split", "dataset", "dataset")
      .connect("split", "train", "train", "X")
      .connect("build", "train", "model", "model")
      .connect("train", "evaluate", "trained_model", "model")
      .connect("split", "evaluate", "test", "X")
      .connect("train", "save", "trained_model", "model"))
    return b


def _build_regression(b: WorkflowBuilder) -> WorkflowBuilder:
    """load -> split -> build -> train -> evaluate -> save."""
    (b.add("load_dataset", name="load", dataset="diabetes")
      .add("split_train_test", name="split", test_size=0.2)
      .add("build_model", name="build", technique="regression")
      .add("train_model", name="train")
      .add("evaluate_model", name="evaluate")
      .add("save_model", name="save", path=os.path.join(tempfile_get(), "qai_reg.qmodel"))
      .connect("load", "split", "dataset", "dataset")
      .connect("split", "train", "train", "X")
      .connect("build", "train", "model", "model")
      .connect("train", "evaluate", "trained_model", "model")
      .connect("split", "evaluate", "test", "X")
      .connect("train", "save", "trained_model", "model"))
    return b


def _build_llm_finetune(b: WorkflowBuilder) -> WorkflowBuilder:
    """load_dataset -> load_pretrained -> quantize -> finetune -> evaluate -> deploy."""
    (b.add("load_dataset", name="dataset", dataset="text")
      .add("load_model", name="pretrained", path="gpt2")
      .add("quantize", name="quantize", format="int8")
      .add("train_model", name="finetune")
      .add("evaluate_model", name="evaluate")
      .add("deploy", name="deploy", target="flask")
      .connect("pretrained", "quantize", "model", "model")
      .connect("dataset", "finetune", "dataset", "X")
      .connect("quantize", "finetune", "quantized_model", "model")
      .connect("finetune", "evaluate", "trained_model", "model")
      .connect("dataset", "evaluate", "dataset", "X")
      .connect("finetune", "deploy", "trained_model", "model"))
    return b


def _build_clustering(b: WorkflowBuilder) -> WorkflowBuilder:
    """load -> build -> fit -> predict -> visualize."""
    (b.add("load_dataset", name="load", dataset="iris")
      .add("build_model", name="build", technique="kmeans", n_clusters=3)
      .add("train_model", name="fit")
      .add("predict", name="predict")
      .add("explain", name="visualize")
      .connect("load", "fit", "dataset", "X")
      .connect("build", "fit", "model", "model")
      .connect("fit", "predict", "trained_model", "model")
      .connect("load", "predict", "dataset", "X")
      .connect("fit", "visualize", "trained_model", "model")
      .connect("load", "visualize", "dataset", "X"))
    return b


def _build_auto_ml(b: WorkflowBuilder) -> WorkflowBuilder:
    """load -> auto_train -> evaluate -> save_best."""
    (b.add("load_dataset", name="load", dataset="iris")
      .add("split_train_test", name="split", test_size=0.2)
      .add("build_model", name="auto_train", technique="classifier")
      .add("train_model", name="train")
      .add("evaluate_model", name="evaluate")
      .add("save_model", name="save_best",
            path=os.path.join(tempfile_get(), "qai_automl.qmodel"))
      .connect("load", "split", "dataset", "dataset")
      .connect("split", "train", "train", "X")
      .connect("auto_train", "train", "model", "model")
      .connect("train", "evaluate", "trained_model", "model")
      .connect("split", "evaluate", "test", "X")
      .connect("train", "save_best", "trained_model", "model"))
    return b


def _build_production_deploy(b: WorkflowBuilder) -> WorkflowBuilder:
    """load_model -> quantize -> convert -> benchmark -> deploy."""
    (b.add("load_model", name="load",
            path=os.path.join(tempfile_get(), "qai_model.qmodel"))
      .add("quantize", name="quantize", format="int8")
      .add("convert", name="convert", format="onnx")
      .add("benchmark", name="benchmark")
      .add("deploy", name="deploy", target="flask")
      .connect("load", "quantize", "model", "model")
      .connect("quantize", "convert", "quantized_model", "model")
      .connect("convert", "benchmark", "converted_model", "model")
      .connect("convert", "deploy", "converted_model", "model"))
    return b


def _build_compare_models(b: WorkflowBuilder) -> WorkflowBuilder:
    """load -> split -> [build x N] -> train x N -> evaluate x N."""
    b.add("load_dataset", name="load", dataset="iris")
    b.add("split_train_test", name="split", test_size=0.2)
    b.connect("load", "split", "dataset", "dataset")
    techniques = ["neural_network", "random_forest", "svm"]
    for i, tech in enumerate(techniques):
        b_name, t_name, e_name = f"build_{i}", f"train_{i}", f"eval_{i}"
        (b.add("build_model", name=b_name, technique=tech)
          .add("train_model", name=t_name)
          .add("evaluate_model", name=e_name)
          .connect("split", t_name, "train", "X")
          .connect(b_name, t_name, "model", "model")
          .connect(t_name, e_name, "trained_model", "model")
          .connect("split", e_name, "test", "X"))
    return b


TEMPLATES: Dict[str, Dict[str, Any]] = {
    "classification": {
        "name": "classification",
        "description": "load -> split -> build -> train -> evaluate -> save",
        "build": _build_classification,
        "nodes": ["load", "split", "build", "train", "evaluate", "save"],
    },
    "regression": {
        "name": "regression",
        "description": "load -> split -> build -> train -> evaluate -> save (regression)",
        "build": _build_regression,
        "nodes": ["load", "split", "build", "train", "evaluate", "save"],
    },
    "llm_finetune": {
        "name": "llm_finetune",
        "description": "load_dataset -> load_pretrained -> quantize -> finetune -> evaluate -> deploy",
        "build": _build_llm_finetune,
        "nodes": ["dataset", "pretrained", "quantize", "finetune", "evaluate", "deploy"],
    },
    "clustering": {
        "name": "clustering",
        "description": "load -> build -> fit -> predict -> visualize",
        "build": _build_clustering,
        "nodes": ["load", "build", "fit", "predict", "visualize"],
    },
    "auto_ml": {
        "name": "auto_ml",
        "description": "load -> split -> auto_train -> evaluate -> save_best",
        "build": _build_auto_ml,
        "nodes": ["load", "split", "auto_train", "train", "evaluate", "save_best"],
    },
    "production_deploy": {
        "name": "production_deploy",
        "description": "load_model -> quantize -> convert -> benchmark -> deploy",
        "build": _build_production_deploy,
        "nodes": ["load", "quantize", "convert", "benchmark", "deploy"],
    },
    "compare_models": {
        "name": "compare_models",
        "description": "load -> split -> [build x N] -> train x N -> evaluate x N",
        "build": _build_compare_models,
        "nodes": ["load", "split",
                   "build_0", "train_0", "eval_0",
                   "build_1", "train_1", "eval_1",
                   "build_2", "train_2", "eval_2"],
    },
}


# =====================================================================
# Top-level convenience functions
# =====================================================================
def create(template: Optional[str] = None) -> WorkflowBuilder:
    """Start a new workflow builder, optionally seeded from a template.

    Args:
        template: name of a template in TEMPLATES. If None, returns an
            empty WorkflowBuilder.

    Returns:
        A WorkflowBuilder ready to be customized or built.
    """
    if template is None:
        return WorkflowBuilder()
    if template not in TEMPLATES:
        raise ValueError(
            f"[WHAT] Unknown template '{template}'\n"
            f"[WHY] No template with that name is registered\n"
            f"[FIX] Available templates: {sorted(TEMPLATES.keys())}")
    builder = WorkflowBuilder(name=template)
    return TEMPLATES[template]["build"](builder)


def load(path: str) -> WorkflowGraph:
    """Load a saved workflow graph from a JSON file."""
    return WorkflowGraph.load(path)


def list_templates() -> List[str]:
    """Return the names of all available pre-built workflow templates."""
    return list(TEMPLATES.keys())


def template_info(name: str) -> dict:
    """Return metadata about a template (name, description, nodes)."""
    if name not in TEMPLATES:
        raise ValueError(
            f"[WHAT] Unknown template '{name}'\n"
            f"[WHY] No template with that name is registered\n"
            f"[FIX] Available templates: {sorted(TEMPLATES.keys())}")
    t = TEMPLATES[name]
    return {
        "name": t["name"],
        "description": t["description"],
        "nodes": list(t.get("nodes", [])),
    }


def execute(template: str, **overrides: dict) -> dict:
    """Build and execute a template end-to-end.

    Args:
        template: name of a template in TEMPLATES.
        **overrides: optional config overrides keyed by node name. Each
            override value should be a dict of config keys to merge into
            that node's existing config.

            Example:
                execute('classification', build={'technique': 'random_forest'})

    Returns:
        The terminal-node outputs dict from the executed graph.
    """
    builder = create(template)
    # Apply overrides before building
    for node_name, config_update in overrides.items():
        if node_name in builder._graph.nodes and isinstance(config_update, dict):
            builder._graph.nodes[node_name].config.update(config_update)
    graph = builder.build()
    return graph.execute()


__all__ = [
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowGraph",
    "WorkflowBuilder",
    "WorkflowVisualizer",
    "OPERATIONS",
    "TEMPLATES",
    "register_operation",
    "create",
    "load",
    "list_templates",
    "template_info",
    "execute",
]
