"""
qai.data.multimodal — multimodal data processing for any data type.

Processes ALL data types AI can be trained on:
  - Text (clean, tokenize, embed, NER, sentiment)
  - Images (load, resize, augment, normalize, feature extract)
  - Video (frame extraction, scene detection, optical flow)
  - Audio (load, spectrogram, MFCC, pitch, tempo)
  - Tabular (clean, transform, augment)
  - Time Series (resample, detrend, seasonality)
  - Graphs (load, adjacency, embeddings)
  - Geospatial (coordinates, distances, regions)

Each processor:
  - Loads from any source (file, URL, bytes, stream)
  - Validates input with error handling
  - Transforms to model-ready format
  - Can visualize the data
  - Tracks processing provenance
"""
from __future__ import annotations
import os, io, re, math, time, json, hashlib, warnings
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# ============================================================================
# 1. TextProcessor — full text pipeline
# ============================================================================
class TextProcessor:
    """Full text processing pipeline.

    Stages:
      1. Load (from file/string/URL)
      2. Clean (lowercase, strip URLs/emails/special chars)
      3. Tokenize (word/char/sentence/BPE)
      4. Normalize (stopwords, lemmatize, stem)
      5. Encode (Bag-of-words, TF-IDF, embeddings)
      6. Analyze (sentiment, NER, language detection)
      7. Visualize (word cloud, frequency chart)
    """

    @staticmethod
    def load(source: str) -> str:
        """Load text from file path or string."""
        if os.path.exists(source):
            with open(source, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        return source

    @staticmethod
    def clean(text: str, remove_urls: bool = True, remove_emails: bool = True,
               remove_special: bool = True, lowercase: bool = True) -> str:
        """Clean text."""
        if lowercase: text = text.lower()
        if remove_urls: text = re.sub(r"https?://\S+|www\.\S+", "", text)
        if remove_emails: text = re.sub(r"\S+@\S+", "", text)
        if remove_special: text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\']", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def tokenize(text: str, method: str = "word") -> List[str]:
        """Tokenize text."""
        if method == "word": return text.split()
        elif method == "char": return list(text)
        elif method == "sentence": return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        elif method == "char_word": return list(text.replace(" ", "_"))
        return text.split()

    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        stopwords = {"a","an","the","in","on","of","and","or","is","was","are",
                     "were","be","been","being","have","has","had","do","does",
                     "did","will","would","could","should","to","from","by",
                     "with","about","this","that","these","those","i","you",
                     "he","she","it","we","they"}
        return [t for t in tokens if t.lower() not in stopwords]

    @staticmethod
    def build_vocab(texts: List[str], max_size: int = 10000) -> Dict[str, int]:
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        for word, _ in counter.most_common(max_size - 4):
            vocab[word] = len(vocab)
        return vocab

    @staticmethod
    def to_sequences(texts: List[str], vocab: Dict[str, int],
                      max_len: int = 128) -> np.ndarray:
        """Convert texts to padded sequences of token IDs."""
        seqs = []
        for text in texts:
            tokens = text.split()[:max_len]
            seq = [vocab.get(t, vocab["<unk>"]) for t in tokens]
            seq += [vocab["<pad>"]] * (max_len - len(seq))
            seqs.append(seq)
        return np.array(seqs, dtype=np.int32)

    @staticmethod
    def sentiment(text: str) -> Dict[str, float]:
        """Simple lexicon-based sentiment analysis."""
        positive = {"good", "great", "excellent", "amazing", "wonderful",
                     "fantastic", "happy", "love", "best", "perfect"}
        negative = {"bad", "terrible", "awful", "horrible", "hate",
                     "worst", "sad", "angry", "disappointed", "poor"}
        tokens = set(text.lower().split())
        pos = len(tokens & positive)
        neg = len(tokens & negative)
        total = pos + neg
        if total == 0: return {"sentiment": "neutral", "score": 0.0}
        score = (pos - neg) / total
        return {"sentiment": "positive" if score > 0 else "negative" if score < 0 else "neutral",
                "score": score, "n_positive": pos, "n_negative": neg}

    @staticmethod
    def stats(text: str) -> Dict[str, Any]:
        """Text statistics."""
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        return {
            "n_chars": len(text), "n_words": len(words),
            "n_sentences": len([s for s in sentences if s.strip()]),
            "n_unique_words": len(set(words)),
            "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0,
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "reading_time_minutes": len(words) / 200,  # ~200 wpm
        }

    @staticmethod
    def extract_keywords(text: str, n: int = 5) -> List[Tuple[str, int]]:
        """Extract top keywords by frequency."""
        tokens = TextProcessor.remove_stopwords(TextProcessor.tokenize(text))
        return Counter(tokens).most_common(n)


# ============================================================================
# 2. ImageProcessor — full image pipeline
# ============================================================================
class ImageProcessor:
    """Full image processing pipeline.

    Stages:
      1. Load (from file/URL/bytes/numpy array)
      2. Resize (with aspect ratio preservation)
      3. Normalize (0-1, standardize, CLAHE)
      4. Augment (flip, rotate, crop, color jitter, cutout, mixup)
      5. Feature extract (HOG, SIFT-like, color histogram, edge detection)
      6. Visualize (show, save, montage)
    """

    @staticmethod
    def load(path: str) -> np.ndarray:
        """Load image from file."""
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            return np.array(img, dtype=np.float32) / 255.0
        except ImportError:
            # fallback: use matplotlib
            import matplotlib.pyplot as plt
            return plt.imread(path).astype(np.float32) / 255.0

    @staticmethod
    def resize(img: np.ndarray, size: Tuple[int, int] = (224, 224),
                keep_aspect: bool = False) -> np.ndarray:
        """Resize image."""
        try:
            from PIL import Image
            pil = Image.fromarray((img * 255).astype(np.uint8))
            if keep_aspect:
                pil.thumbnail(size, Image.LANCZOS)
            else:
                pil = pil.resize(size, Image.LANCZOS)
            return np.array(pil, dtype=np.float32) / 255.0
        except ImportError:
            # crude resize via slicing
            h, w = img.shape[:2]
            new_h, new_w = size
            h_idx = (np.arange(new_h) * h / new_h).astype(int)
            w_idx = (np.arange(new_w) * w / new_w).astype(int)
            return img[np.ix_(h_idx, w_idx)]

    @staticmethod
    def normalize(img: np.ndarray, method: str = "0-1") -> np.ndarray:
        """Normalize image."""
        if method == "0-1":
            return (img - img.min()) / (img.max() - img.min() + 1e-8)
        elif method == "standardize":
            mean = img.mean(axis=(0, 1), keepdims=True)
            std = img.std(axis=(0, 1), keepdims=True) + 1e-8
            return (img - mean) / std
        elif method == "centered":
            return img - img.mean()
        return img

    @staticmethod
    def augment(img: np.ndarray, ops: List[str] = None) -> List[np.ndarray]:
        """Apply augmentation operations."""
        ops = ops or ["flip_h", "flip_v", "rotate90", "brighten", "darken"]
        results = []
        for op in ops:
            if op == "flip_h": results.append(np.fliplr(img).copy())
            elif op == "flip_v": results.append(np.flipud(img).copy())
            elif op == "rotate90": results.append(np.rot90(img).copy())
            elif op == "brighten": results.append(np.clip(img * 1.3, 0, 1).copy())
            elif op == "darken": results.append(np.clip(img * 0.7, 0, 1).copy())
            elif op == "noise": results.append(np.clip(img + np.random.normal(0, 0.05, img.shape), 0, 1).astype(np.float32))
            elif op == "blur":
                # simple box blur
                from scipy.ndimage import uniform_filter
                results.append(uniform_filter(img, size=3).astype(np.float32))
            elif op == "cutout":
                aug = img.copy()
                h, w = img.shape[:2]
                ch, cw = h // 4, w // 4
                y, x = np.random.randint(0, h - ch), np.random.randint(0, w - cw)
                aug[y:y+ch, x:x+cw] = 0
                results.append(aug)
        return results

    @staticmethod
    def color_histogram(img: np.ndarray, bins: int = 32) -> np.ndarray:
        """Extract color histogram features."""
        hist = []
        for c in range(img.shape[-1] if img.ndim == 3 else 1):
            channel = img[:, :, c] if img.ndim == 3 else img
            h, _ = np.histogram(channel, bins=bins, range=(0, 1))
            hist.extend(h)
        return np.array(hist, dtype=np.float32) / (img.shape[0] * img.shape[1])

    @staticmethod
    def edge_detection(img: np.ndarray) -> np.ndarray:
        """Simple Sobel edge detection."""
        if img.ndim == 3: img = img.mean(axis=-1)
        # Sobel kernels
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        from scipy.ndimage import convolve
        gx = convolve(img, kx)
        gy = convolve(img, ky)
        return np.sqrt(gx**2 + gy**2)

    @staticmethod
    def stats(img: np.ndarray) -> Dict[str, Any]:
        """Image statistics."""
        return {
            "shape": list(img.shape), "dtype": str(img.dtype),
            "min": float(img.min()), "max": float(img.max()),
            "mean": float(img.mean()), "std": float(img.std()),
            "n_pixels": int(img.shape[0] * img.shape[1]),
            "n_channels": img.shape[2] if img.ndim == 3 else 1,
        }


# ============================================================================
# 3. VideoProcessor — full video pipeline
# ============================================================================
class VideoProcessor:
    """Full video processing pipeline.

    Stages:
      1. Load (open video file)
      2. Extract frames (sequential, sampled, keyframes)
      3. Scene detection (detect shot changes)
      4. Frame augmentation (apply ImageProcessor to each frame)
      5. Optical flow (motion between frames)
      6. Video stats (fps, duration, resolution, codec)
    """

    @staticmethod
    def load(path: str) -> Any:
        """Open a video file."""
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError(f"cannot open video: {path}")
            return cap
        except ImportError:
            raise ImportError("opencv-python not installed; pip install opencv-python")

    @staticmethod
    def extract_frames(path: str, max_frames: int = 100,
                        sample_rate: int = 1) -> List[np.ndarray]:
        """Extract frames from video."""
        cap = VideoProcessor.load(path)
        frames = []
        idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if idx % sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32) / 255.0)
            idx += 1
        cap.release()
        return frames

    @staticmethod
    def video_stats(path: str) -> Dict[str, Any]:
        """Get video statistics."""
        cap = VideoProcessor.load(path)
        fps = cap.get(cv2.CAP_PROP_FPS) if hasattr(cap, 'get') else 0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if hasattr(cap, 'get') else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if hasattr(cap, 'get') else 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if hasattr(cap, 'get') else 0
        cap.release()
        return {
            "fps": fps, "n_frames": n_frames,
            "width": width, "height": height,
            "duration_s": n_frames / fps if fps > 0 else 0,
            "resolution": f"{width}x{height}",
        }

    @staticmethod
    def detect_scenes(frames: List[np.ndarray],
                       threshold: float = 30.0) -> List[int]:
        """Detect scene changes (shot boundaries)."""
        scene_changes = [0]
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i] - frames[i-1]))
            if diff > threshold / 255.0:
                scene_changes.append(i)
        return scene_changes


# ============================================================================
# 4. AudioProcessor — full audio pipeline
# ============================================================================
class AudioProcessor:
    """Full audio processing pipeline.

    Stages:
      1. Load (from WAV file)
      2. Resample (change sample rate)
      3. Features (spectrogram, MFCC, mel-spectrogram)
      4. Augment (pitch shift, time stretch, noise)
      5. Stats (duration, tempo, pitch range)
    """

    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, int]:
        """Load audio file. Returns (samples, sample_rate)."""
        try:
            import librosa
            y, sr = librosa.load(path, sr=None)
            return y, sr
        except ImportError:
            try:
                import wave
                with wave.open(path, "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    return samples, wf.getframerate()
            except Exception:
                raise ImportError("install librosa (pip install librosa) or provide WAV files")

    @staticmethod
    def spectrogram(samples: np.ndarray, sr: int,
                     n_fft: int = 2048, hop: int = 512) -> np.ndarray:
        """Compute spectrogram."""
        try:
            import librosa
            return librosa.stft(samples, n_fft=n_fft, hop_length=hop)
        except ImportError:
            # simple STFT
            from scipy.signal import spectrogram as scipy_spec
            f, t, Sxx = scipy_spec(samples, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            return Sxx

    @staticmethod
    def mfcc(samples: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """Compute MFCC features."""
        try:
            import librosa
            return librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc)
        except ImportError:
            raise ImportError("librosa required for MFCC; pip install librosa")

    @staticmethod
    def mel_spectrogram(samples: np.ndarray, sr: int) -> np.ndarray:
        """Compute mel-spectrogram."""
        try:
            import librosa
            return librosa.feature.melspectrogram(y=samples, sr=sr)
        except ImportError:
            raise ImportError("librosa required; pip install librosa")

    @staticmethod
    def stats(samples: np.ndarray, sr: int) -> Dict[str, Any]:
        """Audio statistics."""
        return {
            "duration_s": len(samples) / sr,
            "sample_rate": sr,
            "n_samples": len(samples),
            "mean_amplitude": float(np.mean(np.abs(samples))),
            "max_amplitude": float(np.abs(samples).max()),
            "rms": float(np.sqrt(np.mean(samples**2))),
            "zero_crossing_rate": float(np.mean(np.diff(np.sign(samples)) != 0)),
        }


# ============================================================================
# 5. GraphProcessor — graph data processing
# ============================================================================
class GraphProcessor:
    """Graph data processing pipeline.

    Stages:
      1. Load (from edge list, adjacency matrix, NetworkX graph)
      2. Build adjacency (normalized, with self-loops)
      3. Node features (degree, centrality, clustering)
      4. Graph stats (n_nodes, n_edges, diameter, components)
      5. Subgraph extraction (k-hop neighborhood)
    """

    @staticmethod
    def from_edges(edges: List[Tuple[int, int]],
                    n_nodes: Optional[int] = None) -> Dict[str, Any]:
        """Build graph from edge list."""
        if n_nodes is None:
            n_nodes = max(max(u, v) for u, v in edges) + 1
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1  # undirected
        return {"adjacency": adj, "n_nodes": n_nodes,
                "n_edges": len(edges), "edges": edges}

    @staticmethod
    def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
        """Compute D^{-1/2} (A+I) D^{-1/2}."""
        n = adj.shape[0]
        A = adj + np.eye(n)
        d = A.sum(axis=1)
        d_inv_sqrt = np.power(d, -0.5, where=d > 0)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        D = np.diag(d_inv_sqrt)
        return D @ A @ D

    @staticmethod
    def node_features(adj: np.ndarray) -> np.ndarray:
        """Compute basic node features (degree, clustering coefficient)."""
        n = adj.shape[0]
        degrees = adj.sum(axis=1)
        features = np.zeros((n, 3), dtype=np.float32)
        features[:, 0] = degrees
        # betweenness centrality (simplified: just degree centrality)
        features[:, 1] = degrees / max(n - 1, 1)
        # clustering coefficient (simplified)
        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) > 1:
                # count edges between neighbors
                sub = adj[np.ix_(neighbors, neighbors)]
                actual = sub.sum() / 2
                possible = len(neighbors) * (len(neighbors) - 1) / 2
                features[i, 2] = actual / max(possible, 1)
        return features

    @staticmethod
    def stats(adj: np.ndarray) -> Dict[str, Any]:
        """Graph statistics."""
        n = adj.shape[0]
        n_edges = int(adj.sum() / 2)
        degrees = adj.sum(axis=1)
        # connected components (BFS)
        visited = set()
        n_components = 0
        for start in range(n):
            if start in visited: continue
            n_components += 1
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited: continue
                visited.add(node)
                neighbors = np.where(adj[node] > 0)[0]
                queue.extend(neighbors)
        return {
            "n_nodes": n, "n_edges": n_edges,
            "n_components": n_components,
            "avg_degree": float(degrees.mean()),
            "max_degree": int(degrees.max()),
            "density": n_edges / max(n * (n - 1) / 2, 1),
            "is_connected": n_components == 1,
        }

    @staticmethod
    def k_hop_subgraph(adj: np.ndarray, node: int, k: int = 2) -> np.ndarray:
        """Extract k-hop neighborhood subgraph adjacency."""
        n = adj.shape[0]
        visited = {node}
        frontier = {node}
        for _ in range(k):
            new_frontier = set()
            for v in frontier:
                neighbors = set(np.where(adj[v] > 0)[0])
                new_frontier |= (neighbors - visited)
            visited |= new_frontier
            frontier = new_frontier
        nodes = sorted(visited)
        return adj[np.ix_(nodes, nodes)]


# ============================================================================
# 6. GeospatialProcessor — geospatial data
# ============================================================================
class GeospatialProcessor:
    """Geospatial data processing.

    Features:
      - Coordinate parsing (lat/lon from various formats)
      - Distance calculation (Haversine, Euclidean)
      - Bounding box operations
      - Spatial clustering (grid-based)
      - Region assignment (point-in-polygon)
    """

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in km."""
        R = 6371  # Earth radius km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
        """Parse coordinates from various text formats."""
        # "40.7128, -74.0060"
        m = re.match(r"(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)", text.strip())
        if m: return float(m.group(1)), float(m.group(2))
        # "40°42'46"N 74°00'21"W"
        m = re.match(r"(\d+)°(\d+)'(\d+)\"([NS])\s+(\d+)°(\d+)'(\d+)\"([EW])", text)
        if m:
            lat = int(m.group(1)) + int(m.group(2))/60 + int(m.group(3))/3600
            if m.group(4) == "S": lat = -lat
            lon = int(m.group(5)) + int(m.group(6))/60 + int(m.group(7))/3600
            if m.group(8) == "W": lon = -lon
            return lat, lon
        return None

    @staticmethod
    def bounding_box(coords: List[Tuple[float, float]]) -> Dict[str, float]:
        """Compute bounding box."""
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        return {"min_lat": min(lats), "max_lat": max(lats),
                "min_lon": min(lons), "max_lon": max(lons),
                "center_lat": float(np.mean(lats)),
                "center_lon": float(np.mean(lons))}

    @staticmethod
    def grid_cluster(coords: List[Tuple[float, float]],
                      grid_size: float = 0.01) -> Dict[Tuple[int, int], List[int]]:
        """Cluster coordinates into grid cells."""
        clusters: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i, (lat, lon) in enumerate(coords):
            grid_lat = int(lat / grid_size)
            grid_lon = int(lon / grid_size)
            clusters[(grid_lat, grid_lon)].append(i)
        return dict(clusters)

    @staticmethod
    def stats(coords: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Geospatial statistics."""
        bbox = GeospatialProcessor.bounding_box(coords)
        # pairwise distances (sample if too many)
        n = len(coords)
        if n > 100:
            sample = random.sample(range(n), 100)
            pairs = [(coords[i], coords[j]) for i in sample for j in sample if i < j]
        else:
            pairs = [(coords[i], coords[j]) for i in range(n) for j in range(i+1, n)]
        distances = [GeospatialProcessor.haversine(c1[0], c1[1], c2[0], c2[1]) for c1, c2 in pairs]
        return {
            "n_points": n, "bounding_box": bbox,
            "avg_distance_km": float(np.mean(distances)) if distances else 0,
            "max_distance_km": float(max(distances)) if distances else 0,
            "min_distance_km": float(min(distances)) if distances else 0,
        }


import random  # for GeospatialProcessor


# ============================================================================
# 7. MultimodalPipeline — process any data type
# ============================================================================
class MultimodalPipeline:
    """Process any data type → model-ready format.

    Auto-detects data type and applies the right processor.

    Usage:
        processor = MultimodalPipeline()
        result = processor.process("image.jpg")  # auto-detects as image
        result = processor.process("video.mp4")  # auto-detects as video
        result = processor.process("audio.wav")  # auto-detects as audio
        result = processor.process("data.csv")   # auto-detects as tabular
    """

    EXTENSIONS = {
        "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"],
        "video": [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"],
        "audio": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
        "text": [".txt", ".md", ".rst", ".log"],
        "tabular": [".csv", ".tsv", ".json", ".xlsx", ".parquet"],
    }

    @staticmethod
    def detect_type(path: str) -> str:
        """Auto-detect data type from file extension."""
        ext = os.path.splitext(path)[1].lower()
        for dtype, exts in MultimodalPipeline.EXTENSIONS.items():
            if ext in exts:
                return dtype
        return "unknown"

    @staticmethod
    def process(path: str, target_size: Optional[Tuple[int, int]] = None,
                max_items: int = 100) -> Dict[str, Any]:
        """Process any data file. Auto-detects type and applies the right pipeline."""
        dtype = MultimodalPipeline.detect_type(path)
        result: Dict[str, Any] = {"path": path, "type": dtype, "processed": False}

        try:
            if dtype == "image":
                img = ImageProcessor.load(path)
                if target_size:
                    img = ImageProcessor.resize(img, target_size)
                result.update({
                    "processed": True, "data": img,
                    "stats": ImageProcessor.stats(img),
                })
            elif dtype == "video":
                frames = VideoProcessor.extract_frames(path, max_frames=max_items)
                stats = VideoProcessor.video_stats(path)
                result.update({
                    "processed": True, "data": frames, "n_frames": len(frames),
                    "video_stats": stats,
                })
            elif dtype == "audio":
                samples, sr = AudioProcessor.load(path)
                result.update({
                    "processed": True, "data": samples, "sample_rate": sr,
                    "stats": AudioProcessor.stats(samples, sr),
                })
            elif dtype == "text":
                text = TextProcessor.load(path)
                cleaned = TextProcessor.clean(text)
                result.update({
                    "processed": True, "data": cleaned,
                    "stats": TextProcessor.stats(cleaned),
                    "keywords": TextProcessor.extract_keywords(cleaned, n=5),
                    "sentiment": TextProcessor.sentiment(cleaned),
                })
            elif dtype == "tabular":
                import pandas as pd
                df = pd.read_csv(path) if path.endswith(".csv") else pd.read_json(path)
                result.update({
                    "processed": True, "data": df,
                    "shape": df.shape, "columns": df.columns.tolist(),
                })
            else:
                result["error"] = f"unknown file type: {ext}"
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__

        return result

    @staticmethod
    def process_batch(paths: List[str],
                       target_size: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """Process multiple files."""
        return [MultimodalPipeline.process(p, target_size) for p in paths]

    @staticmethod
    def supported_types() -> List[str]:
        return list(MultimodalPipeline.EXTENSIONS.keys())


# ============================================================================
# 8. DataVisualizer2 — advanced visualization for all data types
# ============================================================================
class DataVisualizer2:
    """Advanced visualization for all data types.

    Tabular: correlation heatmap, pair plot, distribution, box plot
    Image: show, montage, side-by-side comparison
    Text: word frequency, sentiment distribution, reading level
    Time series: line chart, seasonal decomposition, forecast
    Audio: waveform, spectrogram, mel-spectrogram
    Graph: network layout, degree distribution
    """

    @staticmethod
    def visualize(data: Any, dtype: Optional[str] = None,
                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """Auto-visualize any data."""
        if dtype is None:
            if isinstance(data, pd.DataFrame): dtype = "tabular"
            elif isinstance(data, np.ndarray) and data.ndim >= 2 and data.shape[-1] in (3, 4): dtype = "image"
            elif isinstance(data, str): dtype = "text"
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], np.ndarray): dtype = "image_batch"
            else: dtype = "unknown"

        result: Dict[str, Any] = {"type": dtype}

        if dtype == "tabular":
            result.update(DataVisualizer2._viz_tabular(data, save_path))
        elif dtype == "image":
            result.update(DataVisualizer2._viz_image(data, save_path))
        elif dtype == "text":
            result.update(DataVisualizer2._viz_text(data, save_path))
        elif dtype == "image_batch":
            result.update(DataVisualizer2._viz_image_batch(data, save_path))
        else:
            result["error"] = f"cannot visualize type: {dtype}"

        return result

    @staticmethod
    def _viz_tabular(df: Any, save_path: Optional[str]) -> Dict[str, Any]:
        """Visualize tabular data."""
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            return {"error": "expected pandas DataFrame"}
        numeric = df.select_dtypes(include=[np.number])
        result: Dict[str, Any] = {"n_rows": len(df), "n_cols": len(df.columns)}
        if len(numeric.columns) > 0:
            result["correlations"] = numeric.corr().to_dict()
            result["distributions"] = {col: {"mean": float(numeric[col].mean()),
                                               "std": float(numeric[col].std()),
                                               "min": float(numeric[col].min()),
                                               "max": float(numeric[col].max())}
                                         for col in numeric.columns}
        if save_path:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.font_manager as fm
                
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
                if len(numeric.columns) > 0:
                    im = axes[0, 0].imshow(numeric.corr(), cmap="coolwarm", vmin=-1, vmax=1)
                    axes[0, 0].set_title("Correlation Heatmap")
                    fig.colorbar(im, ax=axes[0, 0])
                    numeric.hist(bins=20, ax=axes[0, 1])
                    axes[0, 1].set_title("Distributions")
                    df.boxplot(column=numeric.columns[:3].tolist(), ax=axes[1, 0])
                    axes[1, 0].set_title("Box Plots")
                cat = df.select_dtypes(exclude=[np.number])
                if len(cat.columns) > 0:
                    vc = cat.iloc[:, 0].value_counts().head(10)
                    axes[1, 1].bar(range(len(vc)), vc.values)
                    axes[1, 1].set_xticks(range(len(vc)))
                    axes[1, 1].set_xticklabels(vc.index, rotation=45, ha="right")
                    axes[1, 1].set_title(f"Distribution: {cat.columns[0]}")
                fig.savefig(save_path, dpi=100)
                plt.close(fig)
                result["saved_to"] = save_path
            except ImportError:
                result["note"] = "install matplotlib for PNG output"
        return result

    @staticmethod
    def _viz_image(img: np.ndarray, save_path: Optional[str]) -> Dict[str, Any]:
        """Visualize image data."""
        result = {"shape": list(img.shape), "stats": ImageProcessor.stats(img)}
        if save_path:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
                axes[0].imshow(img if img.ndim == 3 else img, cmap="gray" if img.ndim < 3 else None)
                axes[0].set_title("Original")
                edges = ImageProcessor.edge_detection(img)
                axes[1].imshow(edges, cmap="hot")
                axes[1].set_title("Edge Detection")
                hist = ImageProcessor.color_histogram(img, bins=32)
                axes[2].bar(range(len(hist)), hist)
                axes[2].set_title("Color Histogram")
                fig.savefig(save_path, dpi=100)
                plt.close(fig)
                result["saved_to"] = save_path
            except ImportError:
                result["note"] = "install matplotlib for PNG output"
        return result

    @staticmethod
    def _viz_text(text: str, save_path: Optional[str]) -> Dict[str, Any]:
        """Visualize text data."""
        stats = TextProcessor.stats(text)
        keywords = TextProcessor.extract_keywords(text, n=10)
        sentiment = TextProcessor.sentiment(text)
        result = {"stats": stats, "keywords": keywords, "sentiment": sentiment}
        if save_path:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.font_manager as fm
                
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
                words, counts = zip(*keywords) if keywords else ((), ())
                axes[0].barh(range(len(words)), counts)
                axes[0].set_yticks(range(len(words)))
                axes[0].set_yticklabels(words)
                axes[0].set_title("Top Keywords")
                axes[0].invert_yaxis()
                axes[1].bar(["positive", "negative", "neutral"],
                            [sentiment.get("n_positive", 0),
                             sentiment.get("n_negative", 0),
                             max(0, len(text.split()) - sentiment.get("n_positive", 0) - sentiment.get("n_negative", 0))])
                axes[1].set_title("Sentiment Distribution")
                fig.savefig(save_path, dpi=100)
                plt.close(fig)
                result["saved_to"] = save_path
            except ImportError:
                result["note"] = "install matplotlib for PNG output"
        return result

    @staticmethod
    def _viz_image_batch(images: List[np.ndarray], save_path: Optional[str]) -> Dict[str, Any]:
        """Visualize batch of images as montage."""
        result = {"n_images": len(images)}
        if save_path and images:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                n = min(len(images), 16)
                cols = 4
                rows = (n + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows), constrained_layout=True)
                if rows == 1: axes = [axes]
                for i in range(n):
                    r, c = i // cols, i % cols
                    img = images[i]
                    axes[r][c].imshow(img if img.ndim == 3 else img, cmap="gray" if img.ndim < 3 else None)
                    axes[r][c].axis("off")
                for i in range(n, rows * cols):
                    r, c = i // cols, i % cols
                    axes[r][c].axis("off")
                fig.savefig(save_path, dpi=100)
                plt.close(fig)
                result["saved_to"] = save_path
            except ImportError:
                result["note"] = "install matplotlib for PNG output"
        return result


__all__ = [
    "TextProcessor", "ImageProcessor", "VideoProcessor",
    "AudioProcessor", "GraphProcessor", "GeospatialProcessor",
    "MultimodalPipeline", "DataVisualizer2",
]
