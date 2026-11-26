"""
qai.installer — smart dependency management and launcher.

The installer that makes qai "just work" — even with missing dependencies.

Features:
  - check_dependencies() — scan for all optional packages
  - install(package) — install a specific package to the project venv
  - install_all() — install everything
  - install_bundle(bundle) — install by feature bundle (--vision, --nlp, etc.)
  - doctor() — full diagnostic report
  - launch_dashboard() — start the visual dashboard
  - launch_chat() — start the chat UI
  - Feature-level graceful degradation — missing deps don't crash, they warn

Design principles:
  1. Framework works even if 90% of deps are missing
  2. Missing deps produce helpful warnings, not crashes
  3. Install only what's missing — don't reinstall
  4. Download progress for large packages
  5. Install to project venv, not global

Usage:
    import qai

    # Check what's installed
    report = qai.doctor()
    print(report)

    # Install a specific package
    qai.install("torch")

    # Install a feature bundle
    qai.install_bundle("vision")

    # Install everything
    qai.install_all()

    # Launch the dashboard
    qai.launch_dashboard(model=m)
"""
from __future__ import annotations
import os, sys, subprocess, importlib, shutil
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


# =====================================================================
# Dependency registry
# =====================================================================
@dataclass
class Dependency:
    """A single optional dependency."""
    name: str          # pip package name
    module: str        # import module name
    feature: str       # what qai feature it enables
    bundle: str        # which bundle it belongs to (core/vision/nlp/serving/etc)
    required: bool = False  # if True, framework won't work without it
    min_version: Optional[str] = None
    install_size_mb: float = 0  # estimated download size
    description: str = ""

    def is_installed(self) -> bool:
        """Check if this dependency is installed."""
        try:
            importlib.import_module(self.module)
            return True
        except ImportError:
            return False

    def get_version(self) -> Optional[str]:
        """Get installed version."""
        try:
            mod = importlib.import_module(self.module)
            return getattr(mod, "__version__", None)
        except (ImportError, AttributeError):
            return None


# The full dependency registry
DEPENDENCIES: List[Dependency] = [
    # Core (required)
    Dependency("numpy", "numpy", "core math", "core", required=True, install_size_mb=15,
                description="Numerical computing — arrays, matrices, math"),
    Dependency("scikit-learn", "sklearn", "classical ML algorithms", "core", required=True, install_size_mb=30,
                description="Random Forest, SVM, KMeans, preprocessing, metrics"),

    # Deep learning
    Dependency("torch", "torch", "neural networks", "deep_learning", install_size_mb=2000,
                description="PyTorch — all deep learning, transformers, CNNs, RNNs"),
    Dependency("torchvision", "torchvision", "vision models & datasets", "vision", install_size_mb=10,
                description="Pretrained vision models, image transforms, MNIST/CIFAR"),

    # NLP
    Dependency("transformers", "transformers", "HuggingFace models", "nlp", install_size_mb=500,
                description="BERT, GPT, LLaMA, T5, Whisper, CLIP, all HF models"),
    Dependency("tokenizers", "tokenizers", "fast tokenizers", "nlp", install_size_mb=5,
                description="Rust-based BPE/WordPiece tokenizers"),
    Dependency("sentencepiece", "sentencepiece", "multilingual tokenizer", "nlp", install_size_mb=5,
                description="SentencePiece tokenizer (LLaMA, T5, mT5)"),
    Dependency("tiktoken", "tiktoken", "GPT-4 tokenizer", "nlp", install_size_mb=5,
                description="OpenAI's BPE tokenizer (GPT-4, ChatGPT)"),

    # Datasets
    Dependency("datasets", "datasets", "HuggingFace datasets", "data", install_size_mb=200,
                description="Load any dataset from HuggingFace Hub"),
    Dependency("pandas", "pandas", "data manipulation", "data", install_size_mb=30,
                description="DataFrames, CSV/Parquet loading"),

    # Serving
    Dependency("flask", "flask", "HTTP serving", "serving", install_size_mb=5,
                description="Flask web server for model serving"),
    Dependency("fastapi", "fastapi", "fast HTTP serving", "serving", install_size_mb=10,
                description="FastAPI for async model serving"),

    # Experiment tracking
    Dependency("mlflow", "mlflow", "experiment tracking", "tracking", install_size_mb=100,
                description="MLflow experiment tracking + model registry"),
    Dependency("wandb", "wandb", "Weights & Biases", "tracking", install_size_mb=50,
                description="W&B experiment tracking + visualization"),

    # Quantization
    Dependency("bitsandbytes", "bitsandbytes", "4-bit quantization", "quantization", install_size_mb=50,
                description="NF4/FP4 quantization for large models"),
    Dependency("auto-gptq", "auto_gptq", "GPTQ quantization", "quantization", install_size_mb=30,
                description="GPTQ post-training quantization"),
    Dependency("autoawq", "awq", "AWQ quantization", "quantization", install_size_mb=30,
                description="AWQ activation-aware quantization"),

    # Acceleration
    Dependency("flash-attn", "flash_attn", "FlashAttention", "acceleration", install_size_mb=100,
                description="FlashAttention-2 — 2-4x faster attention"),
    Dependency("triton", "triton", "custom CUDA kernels", "acceleration", install_size_mb=50,
                description="Triton for custom GPU kernels (paged attention, fused ops)"),
    Dependency("accelerate", "accelerate", "distributed training", "acceleration", install_size_mb=20,
                description="HuggingFace accelerate for multi-GPU training"),
    Dependency("xformers", "xformers", "memory-efficient attention", "acceleration", install_size_mb=50,
                description="xFormers memory-efficient attention"),

    # Boosting
    Dependency("xgboost", "xgboost", "XGBoost", "boosting", install_size_mb=30,
                description="Gradient boosting for tabular data"),
    Dependency("lightgbm", "lightgbm", "LightGBM", "boosting", install_size_mb=10,
                description="Fast gradient boosting"),
    Dependency("catboost", "catboost", "CatBoost", "boosting", install_size_mb=80,
                description="Categorical boosting"),

    # Visualization
    Dependency("matplotlib", "matplotlib", "plotting", "viz", install_size_mb=30,
                description="Matplotlib for charts and visualizations"),
    Dependency("plotly", "plotly", "interactive plots", "viz", install_size_mb=20,
                description="Interactive plots and dashboards"),

    # PEFT
    Dependency("peft", "peft", "LoRA/Adapter fine-tuning", "peft", install_size_mb=10,
                description="HuggingFace PEFT library for LoRA, adapters"),
    Dependency("lion-pytorch", "lion_pytorch", "Lion optimizer", "optim", install_size_mb=1,
                description="Lion optimizer (signed-momentum, lower memory)"),

    # Notebook
    Dependency("ipywidgets", "ipywidgets", "Jupyter widgets", "notebook", install_size_mb=10,
                description="Interactive widgets for Jupyter notebooks"),
    Dependency("ipython", "IPython", "enhanced REPL", "notebook", install_size_mb=10,
                description="IPython for richer interactive sessions"),
]

# Feature bundles
BUNDLES = {
    "core": "Core ML (numpy, scikit-learn) — always required",
    "deep_learning": "Deep learning (torch) — neural networks, transformers",
    "vision": "Computer vision (torchvision) — pretrained vision models, image datasets",
    "nlp": "NLP (transformers, tokenizers, sentencepiece, tiktoken) — BERT, GPT, LLaMA",
    "data": "Datasets (datasets, pandas) — HuggingFace datasets, DataFrames",
    "serving": "Serving (flask, fastapi) — HTTP model serving",
    "tracking": "Experiment tracking (mlflow, wandb) — experiment logging",
    "quantization": "Quantization (bitsandbytes, auto-gptq, autoawq) — model compression",
    "acceleration": "Acceleration (flash-attn, triton, accelerate, xformers) — GPU acceleration",
    "boosting": "Boosting (xgboost, lightgbm, catboost) — gradient boosting",
    "viz": "Visualization (matplotlib, plotly) — charts and dashboards",
    "peft": "PEFT (peft, lion-pytorch) — parameter-efficient fine-tuning",
    "notebook": "Notebook (ipywidgets, ipython) — Jupyter integration",
    "all": "Everything — install all optional dependencies",
}


# =====================================================================
# Check dependencies
# =====================================================================
def check_dependencies() -> Dict[str, Any]:
    """Check all dependencies and return a report.

    Returns:
        Dict with 'installed', 'missing', 'required_missing', 'by_bundle'
    """
    installed = []
    missing = []
    required_missing = []
    by_bundle: Dict[str, Dict[str, List]] = {}

    for dep in DEPENDENCIES:
        status = {
            "name": dep.name,
            "module": dep.module,
            "feature": dep.feature,
            "bundle": dep.bundle,
            "installed": dep.is_installed(),
            "version": dep.get_version(),
            "required": dep.required,
            "size_mb": dep.install_size_mb,
            "description": dep.description,
        }
        if status["installed"]:
            installed.append(status)
        else:
            missing.append(status)
            if dep.required:
                required_missing.append(status)

        # Group by bundle
        if dep.bundle not in by_bundle:
            by_bundle[dep.bundle] = {"installed": [], "missing": []}
        if status["installed"]:
            by_bundle[dep.bundle]["installed"].append(status)
        else:
            by_bundle[dep.bundle]["missing"].append(status)

    return {
        "installed": installed,
        "missing": missing,
        "required_missing": required_missing,
        "by_bundle": by_bundle,
        "total": len(DEPENDENCIES),
        "n_installed": len(installed),
        "n_missing": len(missing),
        "n_required_missing": len(required_missing),
    }


# =====================================================================
# Install
# =====================================================================
def install(package: str, verbose: bool = True) -> bool:
    """Install a specific package.

    Args:
        package: pip package name (e.g. "torch", "transformers")
        verbose: if True, print progress

    Returns:
        True if installation succeeded (or already installed)
    """
    # Check if already installed
    for dep in DEPENDENCIES:
        if dep.name == package or dep.module == package:
            if dep.is_installed():
                if verbose:
                    print(f"[qai] {dep.name} already installed ({dep.get_version()})")
                return True
            break

    # Install via pip
    if verbose:
        print(f"[qai] installing {package}...")

    try:
        # Use sys.executable to install to the right venv
        cmd = [sys.executable, "-m", "pip", "install", package, "--progress-bar", "on"]
        result = subprocess.run(cmd, capture_output=not verbose, text=True)
        if result.returncode == 0:
            if verbose:
                print(f"[qai] {package} installed successfully")
            return True
        else:
            if verbose:
                print(f"[qai] install failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        if verbose:
            print(f"[qai] install error: {e}")
        return False


def install_bundle(bundle: str, verbose: bool = True) -> Dict[str, bool]:
    """Install all packages in a feature bundle.

    Args:
        bundle: bundle name (e.g. "vision", "nlp", "deep_learning")
        verbose: if True, print progress

    Returns:
        Dict of package_name → success
    """
    if bundle == "all":
        deps = DEPENDENCIES
    else:
        deps = [d for d in DEPENDENCIES if d.bundle == bundle]

    if not deps:
        if verbose:
            print(f"[qai] unknown bundle: {bundle}. Available: {list(BUNDLES.keys())}")
        return {}

    if verbose:
        n_missing = sum(1 for d in deps if not d.is_installed())
        print(f"[qai] bundle '{bundle}': {len(deps)} packages, {n_missing} to install")

    results = {}
    for dep in deps:
        if dep.is_installed():
            if verbose:
                print(f"  ✓ {dep.name} (already installed)")
            results[dep.name] = True
        else:
            success = install(dep.name, verbose=verbose)
            results[dep.name] = success
    return results


def install_all(verbose: bool = True) -> Dict[str, bool]:
    """Install all optional dependencies.

    Returns:
        Dict of package_name → success
    """
    return install_bundle("all", verbose=verbose)


# =====================================================================
# Doctor — full diagnostic
# =====================================================================
def doctor() -> str:
    """Run a full diagnostic and return a report.

    Checks:
      - All dependencies (installed/missing/version)
      - Python version
      - Platform (Kaggle/Colab/Jupyter/local)
      - Available hardware (CPU/GPU/MPS)
      - qai version

    Returns:
        Human-readable diagnostic report
    """
    import platform as plat
    lines = [
        "=" * 70,
        "qai Doctor — Diagnostic Report",
        "=" * 70,
        "",
        f"qai version: {_get_version()}",
        f"Python: {plat.python_version()}",
        f"Platform: {plat.platform()}",
        f"Executable: {sys.executable}",
        "",
    ]

    # Platform detection
    env = _detect_platform()
    lines.append(f"Environment: {env}")
    if env == "kaggle":
        lines.append("  GPU: " + ("available" if "KAGGLE_GPU" in os.environ else "not detected"))
    elif env == "colab":
        lines.append("  GPU: " + ("available" if os.environ.get("COLAB_GPU") else "not detected"))
    lines.append("")

    # Hardware
    try:
        from .cuda_kernels import get_hardware_detector
        detector = get_hardware_detector()
        for dev in detector.get_all_devices():
            status = "✓" if dev.is_available else "✗"
            lines.append(f"Hardware: {status} {dev.name} ({dev.device_type})")
            if dev.device_type == "cuda":
                lines.append(f"  VRAM: {dev.vram_gb:.1f} GB, CC: sm_{dev.compute_capability[0]}{dev.compute_capability[1]}")
                lines.append(f"  FP16: {'✓' if dev.supports_fp16 else '✗'}, BF16: {'✓' if dev.supports_bf16 else '✗'}")
                lines.append(f"  FlashAttention: {'✓' if dev.supports_flash_attention else '✗'}")
    except Exception:
        lines.append("Hardware: detection failed (import error)")

    lines.append("")

    # Dependencies
    report = check_dependencies()
    lines.append(f"Dependencies: {report['n_installed']}/{report['total']} installed")
    lines.append("")

    # Group by bundle
    for bundle_name, bundle_desc in BUNDLES.items():
        if bundle_name == "all":
            continue
        bundle_deps = [d for d in DEPENDENCIES if d.bundle == bundle_name]
        if not bundle_deps:
            continue
        n_installed = sum(1 for d in bundle_deps if d.is_installed())
        n_total = len(bundle_deps)
        status = "✓" if n_installed == n_total else f"({n_installed}/{n_total})"
        lines.append(f"  {bundle_name:15s} {status:12s} {bundle_desc}")

        # Show missing
        for dep in bundle_deps:
            if not dep.is_installed():
                lines.append(f"    ✗ {dep.name:20s} — {dep.description}")

    lines.append("")
    lines.append("=" * 70)
    if report["n_required_missing"] > 0:
        lines.append(f"⚠ {report['n_required_missing']} REQUIRED dependencies missing!")
        for dep in report["required_missing"]:
            lines.append(f"  → Install: qai.install('{dep['name']}') or pip install {dep['name']}")
    else:
        lines.append("✓ All required dependencies installed.")
    if report["n_missing"] > 0:
        lines.append(f"  {report['n_missing']} optional dependencies missing (features will be disabled)")
        lines.append(f"  Install all: qai.install_all()")
    lines.append("=" * 70)
    return "\n".join(lines)


# =====================================================================
# Platform detection
# =====================================================================
def _detect_platform() -> str:
    """Detect the current platform/environment."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "colab"
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and "ZMQInteractiveShell" in str(type(ip)):
            return "jupyter"
    except ImportError:
        pass
    return "local"


def detect_platform() -> str:
    """Detect the current platform (Kaggle, Colab, Jupyter, or local).

    Returns:
        One of: "kaggle", "colab", "jupyter", "local"
    """
    return _detect_platform()


def platform_info() -> Dict[str, Any]:
    """Get detailed platform information.

    Returns:
        Dict with platform, GPU availability, RAM, disk, etc.
    """
    import platform as plat
    env = _detect_platform()
    info = {
        "platform": env,
        "python_version": plat.python_version(),
        "os": plat.system(),
        "machine": plat.machine(),
        "processor": plat.processor() or "Unknown",
        "qai_version": _get_version(),
        "gpu_available": False,
        "gpu_name": None,
        "gpu_count": 0,
        "ram_gb": 0,
    }

    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass

    # RAM detection
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        info["ram_gb"] = round(int(line.split()[1]) / (1024**2), 1)
                        break
        except Exception:
            pass

    # Platform-specific
    if env == "kaggle":
        info["kaggle_gpu"] = os.environ.get("KAGGLE_GPU", "none")
        info["kaggle_tpu"] = os.environ.get("KAGGLE_TPU", "none")
    elif env == "colab":
        info["colab_gpu"] = os.environ.get("COLAB_GPU", "none")
        info["colab_tpu"] = os.environ.get("COLAB_TPU_ADDR", "none")

    return info


# =====================================================================
# Launcher
# =====================================================================
def launch_dashboard(model=None, port: int = 8770):
    """Launch the qai dashboard for a model.

    Args:
        model: a qai.Model (optional — can open without a model)
        port: port for the dashboard
    """
    try:
        from .dashboard import Dashboard
        dash = Dashboard(model=model, port=port)
        dash.start()
    except ImportError:
        print("[qai] dashboard not available. Install with: qai.install('flask')")


def launch_chat(model, port: int = 8771):
    """Launch the chat UI for a model.

    Args:
        model: a qai.Model
        port: port for the chat server
    """
    try:
        from .model_chat import chat_with_model
        chat_with_model(model, port=port)
    except ImportError:
        print("[qai] chat not available.")


# =====================================================================
# Version
# =====================================================================
def _get_version() -> str:
    """Get the qai version string."""
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "0.4.0-dev"


# =====================================================================
# Graceful import warning
# =====================================================================
def _warn_missing(feature: str, package: str, install_cmd: str = None):
    """Print a helpful warning when a feature is used without its dependency.

    This is called by qai modules when an optional dependency is missing.
    Instead of crashing, it prints a warning telling the user how to install.
    """
    cmd = install_cmd or f"pip install {package}"
    import warnings
    warnings.warn(
        f"\n[qai] Feature '{feature}' requires '{package}', which is not installed.\n"
        f"  Install it with: {cmd}\n"
        f"  Or run: qai.install('{package}')\n"
        f"  The rest of qai will continue to work without this feature.",
        ImportWarning,
        stacklevel=3,
    )


__all__ = [
    "Dependency", "DEPENDENCIES", "BUNDLES",
    "check_dependencies", "install", "install_bundle", "install_all",
    "doctor", "detect_platform", "platform_info",
    "launch_dashboard", "launch_chat", "_warn_missing",
]
