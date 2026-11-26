"""
qai.mlops — integrated MLOps pipeline tying together all production features.

Provides:
  - MLOpsPipeline — end-to-end: train → track → register → serve → monitor
  - Integration with qai.Trainer, ModelRegistry, ModelMonitor, DriftDetector,
    ABTest, ModelVersioner, ModelCard, ProductionServer, Quantization

This is the "top of the stack" — a single class that orchestrates the full
production ML lifecycle.

Usage:
    import qai

    pipeline = qai.MLOpsPipeline(
        name="iris-classifier",
        registry_dir="./model_registry",
        tracking_dir="./experiments",
        serving_port=8000,
    )

    # 1. Train
    run = pipeline.train(
        model=m, train_data=X_train, train_targets=y_train,
        eval_data=X_val, eval_targets=y_val,
        epochs=10, lr=3e-4,
        method="lora",  # optional PEFT
    )

    # 2. Register the trained model
    pipeline.register(run, description="v1 with LoRA fine-tuning", tags=["v1", "lora"])

    # 3. Quantize for deployment
    pipeline.quantize(method="int8")

    # 4. Serve
    pipeline.serve()  # blocks — production server

    # 5. Monitor (in a separate process/thread)
    pipeline.monitor(X_prod, y_prod)
"""
from __future__ import annotations
import os, time, json, hashlib, threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
import numpy as np

# Lazy imports — only require what's actually used
def _lazy_import(name: str):
    """Lazy-import a qai submodule."""
    import importlib
    return importlib.import_module(f".{name}", package="qai")


# =====================================================================
# Run record
# =====================================================================
@dataclass
class RunRecord:
    """A single training run's record."""
    run_id: str
    timestamp: float
    technique: str
    params: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    quantized_path: Optional[str] = None
    model_card_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# The pipeline
# =====================================================================
class MLOpsPipeline:
    """End-to-end MLOps pipeline.

    Orchestrates: train → track → register → quantize → serve → monitor.

    Args:
        name: pipeline name (e.g. "iris-classifier")
        registry_dir: where to store registered model versions
        tracking_dir: where to store experiment tracking logs
        serving_port: port for the production server (default 8000)
        monitoring_window: how many recent predictions to keep for monitoring (default 10000)
        drift_baseline: optional baseline X for drift detection
        auto_register: if True, automatically register every trained model
        auto_card: if True, auto-generate a model card after training
    """
    def __init__(self, name: str = "qai_pipeline",
                 registry_dir: str = "./model_registry",
                 tracking_dir: str = "./experiments",
                 serving_port: int = 8000,
                 monitoring_window: int = 10000,
                 drift_baseline: Optional[np.ndarray] = None,
                 auto_register: bool = False,
                 auto_card: bool = True):
        self.name = name
        self.registry_dir = registry_dir
        self.tracking_dir = tracking_dir
        self.serving_port = serving_port
        self.monitoring_window = monitoring_window
        self.auto_register = auto_register
        self.auto_card = auto_card
        os.makedirs(registry_dir, exist_ok=True)
        os.makedirs(tracking_dir, exist_ok=True)

        # State
        self.runs: List[RunRecord] = []
        self.current_run: Optional[RunRecord] = None
        self.current_model = None
        self._monitor = None
        self._drift_detector = None
        self._server = None

        # Initialize drift detector if baseline provided
        if drift_baseline is not None:
            try:
                self._drift_detector = _lazy_import("drift").DriftDetector(
                    baseline=drift_baseline, method="ensemble")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 1. Train
    # ------------------------------------------------------------------
    def train(self, model, train_data, train_targets=None,
              eval_data=None, eval_targets=None,
              epochs: int = 10, lr: float = 3e-4,
              batch_size: int = 32, optimizer: str = "adamw",
              lr_schedule: str = "cosine", warmup_steps: int = 100,
              max_grad_norm: float = 1.0, weight_decay: float = 0.01,
              fp16: bool = False, bf16: bool = False,
              method: Optional[str] = None,  # PEFT method
              tags: Optional[List[str]] = None,
              notes: str = "",
              **kwargs) -> RunRecord:
        """Train a model and record the run.

        Args:
            model: a qai.Model (untrained or pre-trained for fine-tuning)
            train_data, train_targets: training data
            eval_data, eval_targets: optional eval data
            epochs, lr, batch_size, ...: training hyperparams
            method: optional PEFT method ("lora", "qlora", "adapter", "ia3")
            tags: list of tags for this run
            notes: free-form notes

        Returns:
            RunRecord with metrics + paths
        """
        # Generate run ID
        run_id = f"{self.name}_{int(time.time())}_{len(self.runs):04d}"
        params = {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "optimizer": optimizer, "lr_schedule": lr_schedule,
            "warmup_steps": warmup_steps, "max_grad_norm": max_grad_norm,
            "weight_decay": weight_decay, "fp16": fp16, "bf16": bf16,
            "method": method,
        }
        run = RunRecord(
            run_id=run_id, timestamp=time.time(),
            technique=model.technique_name,
            params=params, tags=tags or [], notes=notes,
        )
        self.current_run = run
        self.current_model = model

        # Train
        t0 = time.time()
        if method is not None:
            # PEFT fine-tuning
            model.finetune(train_data, train_targets, method=method,
                            verbose=True, **kwargs)
        else:
            # Use the universal Trainer
            import qai
            trainer = qai.Trainer(
                model=model, train_data=train_data, train_targets=train_targets,
                eval_data=eval_data, eval_targets=eval_targets,
                optimizer=optimizer, lr=lr, lr_schedule=lr_schedule,
                warmup_steps=warmup_steps, weight_decay=weight_decay,
                max_grad_norm=max_grad_norm, batch_size=batch_size,
                fp16=fp16, bf16=bf16,
                output_dir=os.path.join(self.tracking_dir, run_id),
                log_every=50, verbose=True,
            )
            trainer.train(epochs=epochs)
            run.metrics["train_history"] = trainer.history
        train_time = time.time() - t0
        run.metrics["train_time_s"] = round(train_time, 2)

        # Evaluate
        if eval_data is not None and eval_targets is not None:
            eval_metrics = self._evaluate(model, eval_data, eval_targets)
            run.metrics.update(eval_metrics)
        elif eval_data is not None:
            eval_metrics = self._evaluate(model, eval_data, None)
            run.metrics.update(eval_metrics)

        # Save model
        model_path = os.path.join(self.tracking_dir, run_id, "model.qmodel")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            import qai
            qai.save_model(model, model_path)
            run.model_path = model_path
        except Exception as e:
            print(f"[qai.MLOps] model save failed: {e}")

        # Auto-generate model card
        if self.auto_card and eval_data is not None:
            try:
                card_path = self._generate_card(
                    model, eval_data, eval_targets, run)
                run.model_card_path = card_path
            except Exception as e:
                print(f"[qai.MLOps] card generation failed: {e}")

        # Log to tracking
        self._log_run(run)

        # Auto-register if enabled
        if self.auto_register:
            self.register(run)

        self.runs.append(run)
        print(f"[qai.MLOps] run {run_id} complete — metrics: "
              f"{ {k: v for k, v in run.metrics.items() if k != 'train_history'} }")
        return run

    def _evaluate(self, model, X, y) -> Dict[str, Any]:
        """Run evaluation metrics."""
        metrics = {}
        try:
            preds = []
            n = min(len(X), 200)
            for i in range(n):
                try:
                    preds.append(model.predict(X[i]))
                except Exception:
                    preds.append(None)
            valid = [(p, t) for p, t in zip(preds, y[:n] if y is not None else [None]*n)
                      if p is not None]
            if y is not None and len(valid) > 0:
                y_arr = np.array([t for _, t in valid])
                p_arr = np.array([p for p, _ in valid])
                if p_arr.dtype.kind in ("i", "f") and y_arr.dtype.kind in ("i", "f"):
                    accuracy = float(np.mean(p_arr == y_arr))
                    metrics["accuracy"] = accuracy
                    metrics["n_eval_samples"] = len(valid)
        except Exception:
            pass
        return metrics

    def _generate_card(self, model, X, y, run: RunRecord) -> Optional[str]:
        """Generate a model card."""
        try:
            import qai
            card = qai.ModelCardGenerator(model).generate(
                X_test=X, y_test=y if y is not None else None,
                description=f"Run {run.run_id}: {run.notes or 'No description'}")
            path = os.path.join(self.tracking_dir, run.run_id, "model_card.md")
            # ModelCardContent has .markdown attr, or use str()
            content = card.markdown if hasattr(card, "markdown") else str(card)
            with open(path, "w") as f:
                f.write(content)
            return path
        except Exception as e:
            print(f"[qai.MLOps] card gen failed: {e}")
            return None

    def _log_run(self, run: RunRecord):
        """Log run to tracking file."""
        path = os.path.join(self.tracking_dir, run.run_id, "run.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(run.to_dict(), f, indent=2, default=str)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2. Register
    # ------------------------------------------------------------------
    def register(self, run: RunRecord, description: str = "",
                 tags: Optional[List[str]] = None,
                 version: Optional[str] = None) -> Dict[str, Any]:
        """Register a trained model in the model registry.

        Args:
            run: RunRecord from train()
            description: human-readable description
            tags: list of tags
            version: semantic version (default: auto-incremented)

        Returns:
            Registry entry dict (or string path on failure)
        """
        if self.current_model is None:
            raise ValueError("No current model — call train() first")
        try:
            import qai
            registry = qai.ModelRegistry(self.registry_dir)
            ver = version or f"{len(self.runs)}.0.0"
            entry = registry.publish(
                self.current_model,  # pass the Model object, not a path
                name=self.name,
                version=ver,
                description=description or run.notes or f"Run {run.run_id}",
                tags=tags or run.tags,
            )
            # entry may be a dict or a string path — normalize
            if isinstance(entry, dict):
                print(f"[qai.MLOps] registered {self.name} v{entry.get('version', ver)}")
            else:
                print(f"[qai.MLOps] registered {self.name} v{ver} → {entry}")
                entry = {"name": self.name, "version": ver, "path": str(entry)}
            return entry
        except Exception as e:
            print(f"[qai.MLOps] register failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # 3. Quantize
    # ------------------------------------------------------------------
    def quantize(self, method: str = "int8",
                  calibration_data: Optional[np.ndarray] = None
                  ) -> Optional[str]:
        """Quantize the current model for deployment.

        Args:
            method: "int8" (dynamic), "int4" (NF4), "static" (calibration)
            calibration_data: required for "static"

        Returns:
            Path to the quantized model, or None on failure.
        """
        if self.current_model is None:
            raise ValueError("No current model — call train() first")
        try:
            import qai
            if method == "int8":
                qai.quantize_dynamic(self.current_model)
            elif method == "int4":
                qai.quantize_to_int4(self.current_model, quant_type="nf4")
            elif method == "static":
                if calibration_data is None:
                    raise ValueError("static quantization requires calibration_data")
                qai.quantize_static(self.current_model, calibration_data)
            # Save quantized model
            if self.current_run is not None:
                path = os.path.join(self.tracking_dir,
                                     self.current_run.run_id,
                                     f"model_{method}.qmodel")
                qai.save_model(self.current_model, path)
                self.current_run.quantized_path = path
                print(f"[qai.MLOps] quantized ({method}) → {path}")
                return path
        except Exception as e:
            print(f"[qai.MLOps] quantize failed: {e}")
        return None

    # ------------------------------------------------------------------
    # 4. Serve
    # ------------------------------------------------------------------
    def serve(self, max_batch_size: int = 8, max_concurrent: int = 64,
              enable_streaming: bool = True):
        """Start the production server (blocks).

        Args:
            max_batch_size: maximum requests per batch
            max_concurrent: maximum concurrent requests
            enable_streaming: if True, support /generate (SSE)
        """
        if self.current_model is None:
            raise ValueError("No current model — call train() first")
        try:
            import qai
            self._server = qai.ProductionServer(
                model=self.current_model,
                port=self.serving_port,
                max_batch_size=max_batch_size,
                max_concurrent=max_concurrent,
                enable_streaming=enable_streaming,
            )
            print(f"[qai.MLOps] serving at http://0.0.0.0:{self.serving_port}/")
            self._server.start()
        except KeyboardInterrupt:
            print("\n[qai.MLOps] stopping server...")
            self.stop_serve()

    def stop_serve(self):
        if self._server is not None:
            self._server.stop()
            self._server = None

    # ------------------------------------------------------------------
    # 5. Monitor
    # ------------------------------------------------------------------
    def monitor(self, X_prod: Optional[np.ndarray] = None,
                 y_prod: Optional[np.ndarray] = None,
                 window: Optional[int] = None) -> Dict[str, Any]:
        """Run monitoring checks on production data.

        Args:
            X_prod: recent production inputs
            y_prod: optional ground-truth labels (for accuracy tracking)
            window: number of recent samples to use (default: monitoring_window)

        Returns:
            Monitoring report dict
        """
        report = {"alerts": [], "metrics": {}}
        # Drift detection
        if self._drift_detector is not None and X_prod is not None:
            try:
                drift_report = self._drift_detector.detect(X_prod)
                if drift_report.drifted:
                    report["alerts"].append({
                        "type": "drift",
                        "message": f"Drift detected on features: {drift_report.drifted_features}",
                        "severity": "high",
                    })
                report["metrics"]["drift"] = {
                    "drifted": drift_report.drifted,
                    "n_drifted_features": len(drift_report.drifted_features),
                }
            except Exception as e:
                report["alerts"].append({"type": "drift_error", "message": str(e)})

        # Accuracy tracking
        if self.current_model is not None and X_prod is not None and y_prod is not None:
            try:
                preds = []
                n = min(len(X_prod), window or self.monitoring_window)
                for i in range(n):
                    try:
                        preds.append(self.current_model.predict(X_prod[i]))
                    except Exception:
                        preds.append(None)
                valid = [(p, t) for p, t in zip(preds, y_prod[:n]) if p is not None]
                if valid:
                    accuracy = float(np.mean(
                        np.array([p for p, _ in valid]) ==
                        np.array([t for _, t in valid])))
                    report["metrics"]["accuracy"] = accuracy
                    if accuracy < 0.7:  # threshold
                        report["alerts"].append({
                            "type": "low_accuracy",
                            "message": f"Accuracy dropped to {accuracy:.1%}",
                            "severity": "high",
                        })
            except Exception:
                pass

        # Model health check
        if self.current_model is not None and X_prod is not None:
            try:
                import qai
                health = qai.health_check(self.current_model, X_prod, y_prod)
                report["metrics"]["health"] = health["overall_status"]
                if health["overall_status"] != "healthy":
                    report["alerts"].append({
                        "type": "health",
                        "message": f"Model health: {health['overall_status']}",
                        "severity": "medium",
                    })
            except Exception:
                pass

        return report

    # ------------------------------------------------------------------
    # 6. A/B test
    # ------------------------------------------------------------------
    def ab_test(self, model_a, model_b, X_prod, y_prod,
                 traffic_split: float = 0.5) -> Dict[str, Any]:
        """Run an A/B test between two models on production data.

        Args:
            model_a, model_b: the two models to compare
            X_prod, y_prod: production data + ground truth
            traffic_split: fraction of traffic to model B (default 0.5)

        Returns:
            A/B test result dict
        """
        try:
            import qai
            ab = qai.ABTest(model_a=model_a, model_b=model_b,
                             traffic_split=traffic_split)
            n = min(len(X_prod), 200)
            for i in range(n):
                try:
                    ab.predict(X_prod[i])
                    ab.record_outcome(X_prod[i], y_prod[i])
                except Exception:
                    pass
            result = ab.analyze()
            return {
                "significant": result.significant,
                "winner": result.winner,
                "p_value": result.p_value,
                "metric_a": result.metric_a,
                "metric_b": result.metric_b,
            }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all training runs."""
        return [r.to_dict() for r in self.runs]

    def get_best_run(self, metric: str = "accuracy") -> Optional[RunRecord]:
        """Get the best run by a metric."""
        scored = [(r, r.metrics.get(metric, 0)) for r in self.runs
                   if metric in r.metrics]
        if not scored:
            return None
        return max(scored, key=lambda x: x[1])[0]

    def compare_runs(self, metric: str = "accuracy") -> str:
        """Compare all runs by a metric, sorted descending."""
        scored = [(r, r.metrics.get(metric, None)) for r in self.runs]
        scored.sort(key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
        lines = [f"{'Run ID':40s} {metric:>10s}  Notes"]
        lines.append("-" * 80)
        for r, m in scored:
            lines.append(f"{r.run_id:40s} {str(m):>10s}  {r.notes[:30]}")
        return "\n".join(lines)


# =====================================================================
# Convenience: one-shot pipeline
# =====================================================================
def quick_pipeline(name: str, technique: str, X_train, y_train,
                    X_val=None, y_val=None, epochs: int = 10,
                    quantize: Optional[str] = None,
                    serve: bool = False, port: int = 8000,
                    **kwargs) -> MLOpsPipeline:
    """Quick one-shot MLOps pipeline: build → train → quantize → serve.

    Args:
        name: pipeline name
        technique: qai technique name (e.g. "classifier", "transformer")
        X_train, y_train: training data
        X_val, y_val: optional validation data
        epochs: training epochs
        quantize: optional quantization method ("int8", "int4")
        serve: if True, start the production server after training
        port: server port
        **kwargs: passed to qai.build()

    Returns:
        The MLOpsPipeline (already trained; serving if requested)
    """
    import qai
    pipeline = MLOpsPipeline(name=name, serving_port=port,
                              drift_baseline=X_val if X_val is not None else X_train)
    model = qai.build(technique=technique, **kwargs)
    pipeline.train(model=model, train_data=X_train, train_targets=y_train,
                    eval_data=X_val, eval_targets=y_val, epochs=epochs)
    if quantize is not None:
        pipeline.quantize(method=quantize)
    if serve:
        pipeline.serve()
    return pipeline


__all__ = ["MLOpsPipeline", "RunRecord", "quick_pipeline"]
