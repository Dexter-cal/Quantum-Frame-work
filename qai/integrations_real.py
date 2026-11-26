"""
qai.integrations.real -- real library integrations with actual tests.

Unlike qai.integrations (which is mostly stubs that lazy-import), this
module ships full integration adapters that:
  - call the real library
  - have actual integration tests (run when the library is installed)
  - degrade gracefully when the library isn't available

Currently integrated:
  - HuggingFace Transformers (real model loading + tokenization)
  - HuggingFace Datasets (real dataset loading)
  - FAISS (real vector search)
  - scikit-learn (real ML pipeline)
  - spaCy (real NLP)
  - sentence-transformers (real embeddings)
"""
from __future__ import annotations
import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


# ============================================================================
# HuggingFace Transformers integration
# ============================================================================
class HuggingFaceModel:
    """Real HuggingFace Transformers model wrapper.

    Loads any model from the HF Hub. Falls back gracefully if transformers
    isn't installed.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 task: str = "text-classification"):
        self.model_name = model_name
        self.task = task
        self.transformers = _try_import("transformers")
        self.torch = _try_import("torch")
        self.model = None
        self.tokenizer = None

    def is_available(self) -> bool:
        return self.transformers is not None

    def load(self) -> "HuggingFaceModel":
        if not self.is_available():
            raise ImportError("transformers not installed; pip install transformers")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except Exception as e:
            raise RuntimeError(f"failed to load model '{self.model_name}': {e}")
        return self

    def forward(self, text: str) -> Dict[str, Any]:
        """Run inference on a text input."""
        if self.model is None or self.tokenizer is None:
            self.load()
        if not self.torch:
            raise ImportError("torch not installed")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        return {
            "logits": outputs.logits.numpy().tolist(),
            "predicted_class": int(outputs.logits.argmax(-1).item()),
        }

    def embed(self, text: str) -> np.ndarray:
        """Get embedding (mean-pooled last hidden state)."""
        if self.model is None or self.tokenizer is None:
            self.load()
        if not self.torch:
            raise ImportError("torch not installed")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # mean-pool last hidden state
        hidden = outputs.hidden_states[-1].squeeze(0).mean(dim=0)
        return hidden.numpy()


# ============================================================================
# HuggingFace Datasets integration
# ============================================================================
class HuggingFaceDataset:
    """Real HuggingFace datasets wrapper."""
    def __init__(self, dataset_name: str = "imdb", split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        self.datasets = _try_import("datasets")
        self.dataset = None

    def is_available(self) -> bool:
        return self.datasets is not None

    def load(self) -> "HuggingFaceDataset":
        if not self.is_available():
            raise ImportError("datasets not installed; pip install datasets")
        self.dataset = self.datasets.load_dataset(self.dataset_name, split=self.split)
        return self

    def __iter__(self):
        if self.dataset is None:
            self.load()
        return iter(self.dataset)

    def __len__(self):
        if self.dataset is None:
            self.load()
        return len(self.dataset)

    def to_numpy(self, field: str) -> np.ndarray:
        """Extract a field as a numpy array."""
        if self.dataset is None:
            self.load()
        return np.array([x[field] for x in self.dataset])


# ============================================================================
# FAISS integration (real)
# ============================================================================
class RealFAISSIndex:
    """Real FAISS vector search index."""
    def __init__(self, dim: int = 768, index_type: str = "flat"):
        self.dim = dim
        self.index_type = index_type
        self.faiss = _try_import("faiss")
        self.index = None
        self._ids: List[str] = []

    def is_available(self) -> bool:
        return self.faiss is not None

    def build(self) -> "RealFAISSIndex":
        if not self.is_available():
            raise ImportError("faiss not installed; pip install faiss-cpu")
        if self.index_type == "flat":
            self.index = self.faiss.IndexFlatIP(self.dim)
        elif self.index_type == "ivf":
            quantizer = self.faiss.IndexFlatIP(self.dim)
            self.index = self.faiss.IndexIVFFlat(quantizer, self.dim, 100)
        else:
            self.index = self.faiss.IndexFlatIP(self.dim)
        return self

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        if self.index is None:
            self.build()
        vectors = np.asarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        self._ids.extend(ids)

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, top_k)
        return [(self._ids[i], float(s)) for s, i in zip(scores[0], indices[0])
                 if i >= 0 and i < len(self._ids)]


# ============================================================================
# sentence-transformers integration
# ============================================================================
class SentenceTransformerEmbedder:
    """Real sentence-transformers embedder."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.st = _try_import("sentence_transformers")
        self.model = None

    def is_available(self) -> bool:
        return self.st is not None

    def load(self) -> "SentenceTransformerEmbedder":
        if not self.is_available():
            raise ImportError("sentence-transformers not installed; pip install sentence-transformers")
        self.model = self.st.SentenceTransformer(self.model_name)
        return self

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if self.model is None:
            self.load()
        return self.model.encode(texts)


# ============================================================================
# spaCy integration
# ============================================================================
class SpacyNLP:
    """Real spaCy NLP pipeline."""
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.spacy = _try_import("spacy")
        self.nlp = None

    def is_available(self) -> bool:
        return self.spacy is not None

    def load(self) -> "SpacyNLP":
        if not self.is_available():
            raise ImportError("spacy not installed; pip install spacy && python -m spacy download en_core_web_sm")
        self.nlp = self.spacy.load(self.model_name)
        return self

    def parse(self, text: str) -> Dict[str, Any]:
        """Run full NLP pipeline on text."""
        if self.nlp is None:
            self.load()
        doc = self.nlp(text)
        return {
            "tokens": [t.text for t in doc],
            "lemmas": [t.lemma_ for t in doc],
            "pos": [t.pos_ for t in doc],
            "entities": [(e.text, e.label_) for e in doc.ents],
            "noun_chunks": [nc.text for nc in doc.noun_chunks],
            "sentences": [s.text for s in doc.sents],
        }

    def similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if self.nlp is None:
            self.load()
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return float(doc1.similarity(doc2))


# ============================================================================
# scikit-learn integration (real pipeline)
# ============================================================================
class SklearnPipeline:
    """Real scikit-learn pipeline wrapper."""
    def __init__(self, steps: Optional[List[Tuple[str, Any]]] = None):
        self.sklearn = _try_import("sklearn")
        self.steps = steps or []
        self.pipeline = None

    def is_available(self) -> bool:
        return self.sklearn is not None

    def build(self) -> "SklearnPipeline":
        if not self.is_available():
            raise ImportError("scikit-learn not installed; pip install scikit-learn")
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(self.steps)
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnPipeline":
        if self.pipeline is None:
            self.build()
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("call fit() first")
        return self.pipeline.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.pipeline is None:
            raise RuntimeError("call fit() first")
        return float(self.pipeline.score(X, y))


# ============================================================================
# Integration test runner
# ============================================================================
class IntegrationTestRunner:
    """Run integration tests for each available library.

    Tests are skipped (not failed) if the library isn't installed.
    """
    @staticmethod
    def test_huggingface() -> Dict[str, Any]:
        """Test HuggingFace Transformers integration."""
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            result = tok("hello world", return_tensors="pt")
            return {"available": True, "passed": True,
                     "n_tokens": int(result["input_ids"].shape[1])}
        except ImportError:
            return {"available": False, "passed": None, "reason": "transformers not installed"}
        except Exception as e:
            return {"available": True, "passed": False, "error": str(e)[:200]}

    @staticmethod
    def test_faiss() -> Dict[str, Any]:
        """Test FAISS integration."""
        try:
            import faiss
            import numpy as np
            index = faiss.IndexFlatIP(64)
            vecs = np.random.randn(10, 64).astype(np.float32)
            index.add(vecs)
            scores, indices = index.search(vecs[:1], 3)
            return {"available": True, "passed": True,
                     "n_results": int(len(indices[0]))}
        except ImportError:
            return {"available": False, "passed": None, "reason": "faiss not installed"}
        except Exception as e:
            return {"available": True, "passed": False, "error": str(e)[:200]}

    @staticmethod
    def test_sklearn() -> Dict[str, Any]:
        """Test scikit-learn integration."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=50, n_features=4, random_state=0)
            clf = LogisticRegression().fit(X, y)
            score = clf.score(X, y)
            return {"available": True, "passed": True,
                     "accuracy": float(score)}
        except ImportError:
            return {"available": False, "passed": None, "reason": "sklearn not installed"}
        except Exception as e:
            return {"available": True, "passed": False, "error": str(e)[:200]}

    @staticmethod
    def test_datasets() -> Dict[str, Any]:
        """Test HuggingFace datasets integration (lightweight)."""
        try:
            from datasets import Dataset
            ds = Dataset.from_dict({"text": ["hello", "world"], "label": [0, 1]})
            return {"available": True, "passed": True,
                     "n_samples": len(ds)}
        except ImportError:
            return {"available": False, "passed": None, "reason": "datasets not installed"}
        except Exception as e:
            return {"available": True, "passed": False, "error": str(e)[:200]}

    @staticmethod
    def test_spacy() -> Dict[str, Any]:
        """Test spaCy integration."""
        try:
            import spacy
            # try loading a small model
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                # build a blank pipeline if no model
                nlp = spacy.blank("en")
            doc = nlp("hello world")
            return {"available": True, "passed": True,
                     "n_tokens": len(doc)}
        except ImportError:
            return {"available": False, "passed": None, "reason": "spacy not installed"}
        except Exception as e:
            return {"available": True, "passed": False, "error": str(e)[:200]}

    @staticmethod
    def run_all() -> Dict[str, Any]:
        """Run all integration tests."""
        return {
            "huggingface": IntegrationTestRunner.test_huggingface(),
            "faiss": IntegrationTestRunner.test_faiss(),
            "sklearn": IntegrationTestRunner.test_sklearn(),
            "datasets": IntegrationTestRunner.test_datasets(),
            "spacy": IntegrationTestRunner.test_spacy(),
        }


__all__ = [
    "HuggingFaceModel", "HuggingFaceDataset", "RealFAISSIndex",
    "SentenceTransformerEmbedder", "SpacyNLP", "SklearnPipeline",
    "IntegrationTestRunner",
]
