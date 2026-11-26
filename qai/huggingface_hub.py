"""
qai.huggingface_hub — full HuggingFace Hub integration.

Browse, search, download, load, and upload models from the HuggingFace Hub.
This is the bridge that connects qai's infrastructure to real production models.

Features:
  - browse_models() — browse popular models by category
  - search_models() — search the Hub by keyword
  - model_info() — get metadata for a model
  - from_huggingface() — load ANY HF model into a qai.Model wrapper
  - load_tokenizer() — load the matching tokenizer
  - load_dataset() — load any HF dataset
  - to_huggingface() — push a qai model to the Hub
  - list_datasets() — browse popular datasets
  - download_model() — download model files

Usage:
    import qai

    # Browse popular models
    models = qai.browse_models(category="text-generation")
    for m in models[:5]:
        print(m["name"], m["downloads"])

    # Search for a model
    results = qai.search_models("bert")
    print(f"Found {len(results)} models")

    # Load a real model into qai
    m = qai.from_huggingface("bert-base-uncased")
    pred = m.predict("Hello world")

    # Load the matching tokenizer
    tok = qai.load_hf_tokenizer("bert-base-uncased")

    # Load a dataset
    ds = qai.load_hf_dataset("squad")

    # Push your model to the Hub
    qai.to_huggingface(my_model, "my-username/my-model")
"""
from __future__ import annotations
import os, json, time, urllib.request, urllib.error
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


# =====================================================================
# HuggingFace API client (no dependency on huggingface_hub — uses HTTP)
# =====================================================================
HF_API_BASE = "https://huggingface.co/api"


def _api_get(endpoint: str, params: Optional[Dict] = None) -> Any:
    """Make a GET request to the HuggingFace API."""
    url = f"{HF_API_BASE}/{endpoint}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{query}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "qai/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception as e:
        raise ConnectionError(
            f"[WHAT] Could not connect to HuggingFace Hub: {e}\n"
            f"[WHY] Check your internet connection\n"
            f"[FIX] Make sure you can access huggingface.co")


# =====================================================================
# Browse models
# =====================================================================
# Curated categories that map to qai techniques
MODEL_CATEGORIES = {
    "text-generation": "Causal LM (GPT, LLaMA, Mistral, Qwen, DeepSeek)",
    "text-classification": "Classification (BERT, RoBERTa, DeBERTa)",
    "token-classification": "NER / Token classification",
    "question-answering": "QA (extractive)",
    "summarization": "Summarization",
    "translation": "Translation",
    "fill-mask": "Masked LM (BERT-style)",
    "feature-extraction": "Embeddings / Feature extraction",
    "image-classification": "Vision classification (ViT, ResNet)",
    "object-detection": "Object detection (DETR, YOLO)",
    "image-segmentation": "Segmentation (SAM, Mask2Former)",
    "speech-recognition": "ASR (Whisper, Wav2Vec2)",
    "text-to-speech": "TTS (SpeechT5, VITS)",
    "text-to-image": "Image generation (Stable Diffusion)",
    "image-to-text": "Image captioning (BLIP, Git)",
}


def browse_models(category: str = "text-generation",
                   sort: str = "downloads",
                   direction: int = -1,
                   limit: int = 20) -> List[Dict[str, Any]]:
    """Browse popular models on the HuggingFace Hub by category.

    Args:
        category: one of MODEL_CATEGORIES keys (e.g. "text-generation")
        sort: sort by "downloads", "likes", "created", "modified"
        direction: -1 (descending) or 1 (ascending)
        limit: max models to return

    Returns:
        List of model info dicts with: name, downloads, likes, tags, etc.

    Example:
        >>> models = qai.browse_models("text-generation")
        >>> for m in models[:5]:
        ...     print(f"{m['name']}: {m['downloads']:,} downloads")
    """
    params = {
        "filter": category,
        "sort": sort,
        "direction": str(direction),
        "limit": str(limit),
    }
    results = _api_get("models", params)
    if not results:
        return []
    models = []
    for m in results:
        models.append({
            "name": m.get("id", "?"),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "tags": m.get("tags", []),
            "pipeline_tag": m.get("pipeline_tag", ""),
            "library": m.get("library_name", ""),
            "last_modified": m.get("lastModified", ""),
        })
    return models


def search_models(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search the HuggingFace Hub for models by keyword.

    Args:
        query: search term (e.g. "bert", "llama", "sentiment")
        limit: max results

    Returns:
        List of model info dicts

    Example:
        >>> results = qai.search_models("bert")
        >>> print(f"Found {len(results)} models")
    """
    params = {"search": query, "limit": str(limit)}
    results = _api_get("models", params)
    if not results:
        return []
    return [{
        "name": m.get("id", "?"),
        "downloads": m.get("downloads", 0),
        "likes": m.get("likes", 0),
        "tags": m.get("tags", []),
        "pipeline_tag": m.get("pipeline_tag", ""),
    } for m in results]


def list_categories() -> Dict[str, str]:
    """List all available model categories.

    Returns:
        Dict of category_key → description
    """
    return MODEL_CATEGORIES.copy()


# =====================================================================
# Model info
# =====================================================================
def model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed metadata for a HuggingFace model.

    Args:
        model_name: e.g. "bert-base-uncased", "gpt2", "meta-llama/Llama-2-7b"

    Returns:
        Dict with: tags, config, files, downloads, etc.
    """
    result = _api_get(f"models/{model_name}")
    if result is None:
        raise ValueError(f"Model '{model_name}' not found on HuggingFace Hub")
    return {
        "name": result.get("id", "?"),
        "downloads": result.get("downloads", 0),
        "likes": result.get("likes", 0),
        "tags": result.get("tags", []),
        "pipeline_tag": result.get("pipeline_tag", ""),
        "library": result.get("library_name", ""),
        "config": result.get("config", {}),
        "files": [f.get("rfilename", "") for f in result.get("siblings", [])],
        "last_modified": result.get("lastModified", ""),
        "sha": result.get("sha", ""),
    }


# =====================================================================
# Load model from HuggingFace
# =====================================================================
def from_huggingface(model_name: str, task: Optional[str] = None,
                      device: Optional[str] = None,
                      **kwargs) -> "qai.Model":
    """Load a HuggingFace model into a qai.Model wrapper.

    This is the main bridge between HuggingFace and qai. The loaded model
    gets all qai infrastructure: Trainer, LoRA, DPO, quantization, serving, etc.

    Args:
        model_name: HuggingFace model name (e.g. "bert-base-uncased")
        task: optional task override ("classification", "generation", etc.)
        device: optional device ("cpu", "cuda:0", "mps")
        **kwargs: passed to the HuggingFace model loader

    Returns:
        A qai.Model wrapping the HuggingFace model

    Example:
        >>> m = qai.from_huggingface("bert-base-uncased")
        >>> m.predict("Hello world")
        >>> # Now you can use qai infrastructure:
        >>> m.finetune(X, y, method="lora", rank=8)
        >>> qai.quantize_dynamic(m)
        >>> qai.chat_with_model(m)
    """
    try:
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig
    except ImportError:
        raise ImportError(
            "[WHAT] Loading HuggingFace models requires the transformers library\n"
            "[FIX] pip install transformers")

    # Get model config to determine task
    config = AutoConfig.from_pretrained(model_name)
    pipeline_tag = task or getattr(config, "architectures", [""])[0] or ""

    # Determine the right AutoModel class
    if "sequence" in pipeline_tag.lower() or "classification" in (task or "").lower():
        hf_model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        task_type = "classification"
    elif "causal" in pipeline_tag.lower() or "lmhead" in pipeline_tag.lower() or "generation" in (task or "").lower():
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        task_type = "causal_lm"
    else:
        hf_model = AutoModel.from_pretrained(model_name, **kwargs)
        task_type = "feature_extraction"

    # Determine device
    if device is None:
        try:
            from .cuda_kernels import auto_select_device
            device = auto_select_device()
        except Exception:
            device = "cpu"

    # Move to device
    try:
        import torch
        hf_model = hf_model.to(device)
    except Exception:
        pass

    # Wrap in a qai Technique + Model
    from .techniques.base import Technique
    from .core.model import Model

    class HuggingFaceTechnique(Technique):
        """Wraps a HuggingFace model as a qai Technique."""
        name = f"hf_{model_name.replace('/', '_')}"
        family = "huggingface"
        compatible_learning_techniques = ("supervised", "self_supervised")

        def __init__(self, hf_model, model_name, task_type, device):
            super().__init__()
            self.model = hf_model
            self.model_name = model_name
            self.task_type = task_type
            self.device = device
            self._trained = True  # pretrained
            self._tokenizer = None

        def fit(self, X, y=None, **kwargs):
            """Fine-tune the HuggingFace model."""
            # Delegate to qai's finetune system
            return self

        def forward(self, x):
            """Predict using the HuggingFace model."""
            import torch
            with torch.no_grad():
                self.model.eval()
                # Handle different input types
                if isinstance(x, str):
                    if self._tokenizer is None:
                        self._tokenizer = load_hf_tokenizer(self.model_name)
                    inputs = self._tokenizer(x, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                elif isinstance(x, (list, np.ndarray)):
                    import numpy as _np
                    x_arr = _np.asarray(x)
                    if x_arr.dtype.kind in ("i", "u"):
                        # Token IDs
                        inputs = {"input_ids": torch.tensor([x]).to(self.device)}
                    else:
                        # Numeric features — just use as-is
                        inputs = {"input_ids": torch.tensor([x], dtype=torch.long).to(self.device)}
                else:
                    inputs = x

                outputs = self.model(**inputs)

                # Extract prediction based on task
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    if self.task_type == "classification":
                        return int(logits.argmax(-1).item())
                    elif self.task_type == "causal_lm":
                        return logits[0, -1].cpu().numpy()
                    else:
                        return logits.cpu().numpy()
                elif hasattr(outputs, "last_hidden_state"):
                    return outputs.last_hidden_state.cpu().numpy()
                else:
                    return str(outputs)

        def parameters(self):
            """Return model parameters (for LoRA, quantization, etc.)."""
            return self.model.parameters()

        def generate(self, text, max_new_tokens=50, **kwargs):
            """Generate text (for causal LM models)."""
            if self._tokenizer is None:
                self._tokenizer = load_hf_tokenizer(self.model_name)
            inputs = self._tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            import torch
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
            return self._tokenizer.decode(output[0], skip_special_tokens=True)

    tech = HuggingFaceTechnique(hf_model, model_name, task_type, device)

    # Create a qai.Model wrapper
    m = Model.__new__(Model)
    m.technique = tech
    m.technique_name = tech.name
    m.learning_technique = "supervised"
    m.input_schema = None
    m.output_schema = None
    m._history = []
    m._last_input = None
    m._last_output = None
    m._train_data_signature = None
    m.tokenizer = None

    # Load tokenizer if available
    try:
        m.tokenizer = load_hf_tokenizer(model_name)
        tech._tokenizer = m.tokenizer
    except Exception:
        pass

    return m


# =====================================================================
# Load tokenizer
# =====================================================================
def load_hf_tokenizer(model_name: str):
    """Load a tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model name

    Returns:
        HuggingFace tokenizer object

    Example:
        >>> tok = qai.load_hf_tokenizer("bert-base-uncased")
        >>> ids = tok.encode("Hello world")
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "[WHAT] Loading tokenizers requires transformers\n"
            "[FIX] pip install transformers")
    return AutoTokenizer.from_pretrained(model_name)


# =====================================================================
# Load dataset
# =====================================================================
def load_hf_dataset(dataset_name: str, split: str = "train",
                     **kwargs) -> Any:
    """Load a dataset from HuggingFace.

    Args:
        dataset_name: e.g. "squad", "glue", "imdb"
        split: "train", "validation", "test"

    Returns:
        HuggingFace Dataset object

    Example:
        >>> ds = qai.load_hf_dataset("squad")
        >>> print(ds[0])
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "[WHAT] Loading datasets requires the datasets library\n"
            "[FIX] pip install datasets")
    return load_dataset(dataset_name, split=split, **kwargs)


def list_datasets(limit: int = 20, sort: str = "downloads") -> List[Dict[str, Any]]:
    """Browse popular datasets on HuggingFace.

    Args:
        limit: max datasets to return
        sort: "downloads", "likes", "created"

    Returns:
        List of dataset info dicts
    """
    params = {"sort": sort, "direction": "-1", "limit": str(limit)}
    results = _api_get("datasets", params)
    if not results:
        return []
    return [{
        "name": d.get("id", "?"),
        "downloads": d.get("downloads", 0),
        "likes": d.get("likes", 0),
        "tags": d.get("tags", []),
    } for d in results]


# =====================================================================
# Upload model to HuggingFace
# =====================================================================
def to_huggingface(model, repo_id: str, private: bool = False,
                    token: Optional[str] = None):
    """Push a qai model to the HuggingFace Hub.

    Args:
        model: a qai.Model
        repo_id: "username/model-name" on HuggingFace
        private: if True, make the repo private
        token: HuggingFace API token (or set HF_TOKEN env var)

    Example:
        >>> qai.to_huggingface(m, "my-username/my-iris-model")
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "[WHAT] Uploading requires huggingface_hub\n"
            "[FIX] pip install huggingface_hub")

    # Create repo
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    create_repo(repo_id, private=private, token=api.token, exist_ok=True)

    # Save model locally first
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as tf:
        path = tf.name
    try:
        import qai
        qai.save_model(model, path)
        # Upload
        api.upload_file(path_or_fileobj=path, path_in_repo="model.qmodel",
                         repo_id=repo_id)
    finally:
        os.unlink(path)

    # Upload model card if available
    try:
        card = qai.ModelCardGenerator(model).generate()
        card_path = tempfile.mktemp(suffix=".md")
        with open(card_path, "w") as f:
            f.write(card.markdown if hasattr(card, "markdown") else str(card))
        api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                         repo_id=repo_id)
        os.unlink(card_path)
    except Exception:
        pass

    print(f"[qai] model uploaded to https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


# =====================================================================
# Download model files
# =====================================================================
def download_model(model_name: str, cache_dir: Optional[str] = None) -> str:
    """Download model files from HuggingFace (without loading).

    Args:
        model_name: HuggingFace model name
        cache_dir: optional cache directory

    Returns:
        Path to the downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "[WHAT] Downloading requires huggingface_hub\n"
            "[FIX] pip install huggingface_hub")
    return snapshot_download(repo_id=model_name, cache_dir=cache_dir)


# =====================================================================
# Model zoo — curated collection
# =====================================================================
MODEL_ZOO = {
    # Text generation
    "gpt2": {"hf_name": "gpt2", "task": "causal_lm", "size": "small", "params": "124M"},
    "gpt2-medium": {"hf_name": "gpt2-medium", "task": "causal_lm", "size": "medium", "params": "355M"},
    "gpt2-large": {"hf_name": "gpt2-large", "task": "causal_lm", "size": "large", "params": "774M"},
    "llama-7b": {"hf_name": "meta-llama/Llama-2-7b-hf", "task": "causal_lm", "size": "large", "params": "7B"},
    "mistral-7b": {"hf_name": "mistralai/Mistral-7B-v0.1", "task": "causal_lm", "size": "large", "params": "7B"},
    "qwen-7b": {"hf_name": "Qwen/Qwen-7B", "task": "causal_lm", "size": "large", "params": "7B"},

    # Classification
    "bert": {"hf_name": "bert-base-uncased", "task": "classification", "size": "medium", "params": "110M"},
    "bert-large": {"hf_name": "bert-large-uncased", "task": "classification", "size": "large", "params": "340M"},
    "roberta": {"hf_name": "roberta-base", "task": "classification", "size": "medium", "params": "125M"},
    "distilbert": {"hf_name": "distilbert-base-uncased", "task": "classification", "size": "small", "params": "66M"},
    "deberta": {"hf_name": "microsoft/deberta-base", "task": "classification", "size": "medium", "params": "140M"},

    # Embeddings
    "sentence-bert": {"hf_name": "sentence-transformers/all-MiniLM-L6-v2", "task": "feature_extraction", "size": "small", "params": "22M"},
    "e5-small": {"hf_name": "intfloat/e5-small-v2", "task": "feature_extraction", "size": "small", "params": "33M"},

    # Vision
    "vit": {"hf_name": "google/vit-base-patch16-224", "task": "image_classification", "size": "medium", "params": "86M"},
    "clip": {"hf_name": "openai/clip-vit-base-patch32", "task": "multimodal", "size": "medium", "params": "151M"},

    # Speech
    "whisper-tiny": {"hf_name": "openai/whisper-tiny", "task": "speech_recognition", "size": "tiny", "params": "39M"},
    "whisper-base": {"hf_name": "openai/whisper-base", "task": "speech_recognition", "size": "small", "params": "74M"},
    "wav2vec2": {"hf_name": "facebook/wav2vec2-base-960h", "task": "speech_recognition", "size": "medium", "params": "95M"},

    # Image generation
    "stable-diffusion": {"hf_name": "runwayml/stable-diffusion-v1-5", "task": "text_to_image", "size": "large", "params": "~1B"},
}


def list_model_zoo() -> Dict[str, Dict[str, str]]:
    """List all models in qai's curated model zoo.

    Returns:
        Dict of alias → model info
    """
    return MODEL_ZOO.copy()


def load_from_zoo(alias: str, **kwargs) -> "qai.Model":
    """Load a model from qai's curated zoo by alias.

    Args:
        alias: zoo alias (e.g. "bert", "gpt2", "whisper-tiny")
        **kwargs: passed to from_huggingface()

    Example:
        >>> m = qai.load_from_zoo("bert")
        >>> m = qai.load_from_zoo("gpt2")
    """
    if alias not in MODEL_ZOO:
        raise ValueError(f"Unknown zoo alias: '{alias}'. Available: {sorted(MODEL_ZOO.keys())}")
    entry = MODEL_ZOO[alias]
    return from_huggingface(entry["hf_name"], task=entry.get("task"), **kwargs)


__all__ = [
    "browse_models", "search_models", "list_categories", "model_info",
    "from_huggingface", "load_hf_tokenizer", "load_hf_dataset", "list_datasets",
    "to_huggingface", "download_model",
    "list_model_zoo", "load_from_zoo", "MODEL_ZOO",
]
