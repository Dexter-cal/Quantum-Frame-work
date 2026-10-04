"""
Experiment tracking and metrics logging for qai workflows.
"""
import time
import json

class ExperimentTracker:
    """In-memory and file-persistent experiment logger for qai training runs."""
    def __init__(self, experiment_name="default_experiment"):
        self.experiment_name = experiment_name
        self.runs = []

    def log_run(self, params, metrics, tags=None):
        run_data = {
            "timestamp": time.time(),
            "params": params,
            "metrics": metrics,
            "tags": tags or {}
        }
        self.runs.append(run_data)
        return run_data

    def get_best_run(self, metric_name="accuracy", mode="max"):
        if not self.runs:
            return None

        valid_runs = [r for r in self.runs if metric_name in r["metrics"]]
        if not valid_runs:
            return None

        if mode == "max":
            return max(valid_runs, key=lambda r: r["metrics"][metric_name])
        return min(valid_runs, key=lambda r: r["metrics"][metric_name])

    def export(self, filepath):
        with open(filepath, "w") as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "runs": self.runs
            }, f, indent=2)
        return filepath


def log_telemetry(metric_name: str, value: float, step: int = 0) -> Dict[str, Any]:
    """Logs live experiment training telemetry."""
    return {"metric": metric_name, "value": float(value), "step": step}


def sync_tensorboard(log_dir: str = "./logs") -> str:
    """Syncs live experiment telemetry to Tensorboard directory."""
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
