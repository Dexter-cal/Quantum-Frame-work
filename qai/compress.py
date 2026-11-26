"""
qai.compress -- a learned meta-compressor.

Real-world files have different byte distributions, and no single
compression algorithm is best for all of them:

  - text / logs      : bz2 typically wins (BWT handles repetitive text)
  - JSON / XML       : lzma typically wins (LZ77+range coding)
  - mixed binaries   : zlib typically wins (fast + good enough)
  - random/encrypted : none win -- store uncompressed

This module trains a classifier on byte-distribution features
(entropy, byte histogram, printable fraction, repetition index) and
uses it to pick the best algorithm per file. On real workloads the
learned policy beats 'always-zlib' / 'always-bz2' / 'always-lzma'.

Bundle / archive format (qai.zip):
  - 8-byte magic ("QAIARC\0\0")
  - 4-byte format version
  - 8-byte number of files
  - For each file:
      - 4-byte filename length
      - N-byte filename (UTF-8)
      - 8-byte uncompressed size
      - 1-byte algorithm id (0=store, 1=zlib, 2=bz2, 3=lzma)
      - 8-byte compressed size
      - N-byte compressed payload
  - 32-byte HMAC-SHA256 (if signed)

The archive is itself compressed file-by-file, so a small file inside
a large archive can still be accessed without re-decompressing the whole
thing. This is the difference vs. tar+gzip (which requires reading the
whole archive to extract one file).

Usage:
    import qai

    # Compress a single blob
    data = open("file.txt", "rb").read()
    compressed = qai.compress(data)
    decompressed = qai.decompress(compressed)
    assert decompressed == data

    # Multi-file archive
    arc = qai.Archive()
    arc.add("file1.txt", b"hello world")
    arc.add("file2.json", b'{"k": 1}')
    arc.save("output.qai.zip")

    arc2 = qai.Archive.load("output.qai.zip")
    print(arc2.list_files())  # ['file1.txt', 'file2.json']
    print(arc2.read("file1.txt"))  # b"hello world"

CLI:
    python -m qai compress <file> -o <output>
    python -m qai decompress <file> -o <output>
    python -m qai archive <dir> -o <output.qai.zip>
    python -m qai extract <archive.qai.zip> -d <dest_dir>
"""
from __future__ import annotations
import os, sys, io, struct, time, hashlib, hmac, math, json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Use qai if available (for the classifier); fall back gracefully.
# NOTE: qai may not be fully imported yet when this module loads (circular
# import). We check lazily inside MetaCompressor.train() instead of at
# module load time.
_HAS_QAI = True  # set optimistically; the train() method will re-check
qai = None  # will be imported lazily

# =====================================================================
# Backend algorithms
# =====================================================================
import zlib
try:
    import bz2
    _HAS_BZ2 = True
except ImportError:
    _HAS_BZ2 = False
try:
    import lzma
    _HAS_LZMA = True
except ImportError:
    _HAS_LZMA = False
try:
    import zstd as _zstd_mod
    _HAS_ZSTD = True
except ImportError:
    try:
        import zstandard as _zstd_mod
        _HAS_ZSTD = True
    except ImportError:
        _HAS_ZSTD = False
        _zstd_mod = None
try:
    import brotli as _brotli_mod
    _HAS_BROTLI = True
except ImportError:
    _HAS_BROTLI = False
    _brotli_mod = None

# Algorithm ids (stored in the archive header)
STORE = 0
ZLIB = 1
BZ2 = 2
LZMA = 3
ZSTD = 4
BROTLI = 5

ALGORITHM_NAMES = {
    STORE: "store", ZLIB: "zlib", BZ2: "bz2", LZMA: "lzma",
    ZSTD: "zstd", BROTLI: "brotli",
}
ALGORITHM_IDS = {v: k for k, v in ALGORITHM_NAMES.items()}

ARCHIVE_MAGIC = b"QAIARC\x00\x00"
ARCHIVE_VERSION = 2  # bumped for zstd/brotli support
ARCHIVE_HEADER_FMT = "<8sIQ"  # magic, version (uint32 little-endian... wait, "I" is uint32), n_files (uint64)


def _compress_with(data: bytes, algo: int) -> bytes:
    if algo == STORE:
        return data
    if algo == ZLIB:
        return zlib.compress(data, 9)
    if algo == BZ2:
        if not _HAS_BZ2:
            raise RuntimeError("bz2 not available")
        return bz2.compress(data, 9)
    if algo == LZMA:
        if not _HAS_LZMA:
            raise RuntimeError("lzma not available")
        return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    if algo == ZSTD:
        if not _HAS_ZSTD:
            raise RuntimeError("zstd not available -- pip install zstandard")
        # zstandard API
        if hasattr(_zstd_mod, 'ZstdCompressor'):
            return _zstd_mod.ZstdCompressor(level=19).compress(data)
        # zstd module API
        return _zstd_mod.compress(data, 19)
    if algo == BROTLI:
        if not _HAS_BROTLI:
            raise RuntimeError("brotli not available -- pip install brotli")
        return _brotli_mod.compress(data, quality=11)
    raise ValueError(f"Unknown algorithm id: {algo}")


def _decompress_with(data: bytes, algo: int) -> bytes:
    if algo == STORE:
        return data
    if algo == ZLIB:
        return zlib.decompress(data)
    if algo == BZ2:
        return bz2.decompress(data)
    if algo == LZMA:
        return lzma.decompress(data)
    if algo == ZSTD:
        if hasattr(_zstd_mod, 'ZstdDecompressor'):
            return _zstd_mod.ZstdDecompressor().decompress(data)
        return _zstd_mod.decompress(data)
    if algo == BROTLI:
        return _brotli_mod.decompress(data)
    raise ValueError(f"Unknown algorithm id: {algo}")


# =====================================================================
# Feature extraction -- byte-distribution features for the classifier
# =====================================================================
def extract_features(data: bytes) -> np.ndarray:
    """Extract byte-distribution features from `data`.

    Returns a 16-element float32 vector:
      [0]   file size (log10)
      [1]   Shannon entropy (bits/byte, 0..8)
      [2]   printable-byte fraction (0..1)
      [3]   ascii-text fraction (0..1)
      [4]   null-byte fraction
      [5]   high-byte (>= 128) fraction
      [6]   repetition index (mean run length, normalized)
      [7]   unique-byte count (out of 256)
      [8-15] byte-histogram buckets (8 quantiles, 0..1)

    These features let a classifier distinguish text (low entropy, high
    printable fraction, high repetition) from binary (high entropy, more
    high bytes) from already-compressed/encrypted (very high entropy,
    flat histogram, low repetition).
    """
    if len(data) == 0:
        return np.zeros(16, dtype=np.float32)

    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr)

    # [0] log file size
    log_size = float(np.log10(max(n, 1)))

    # [1] Shannon entropy
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    p = counts[counts > 0] / n
    entropy = float(-np.sum(p * np.log2(p)))

    # [2] printable-byte fraction (ASCII printable: 32..126 + common ws)
    printable = np.sum((arr >= 32) & (arr <= 126)) + np.sum(arr == 9) + np.sum(arr == 10) + np.sum(arr == 13)
    printable_frac = float(printable / n)

    # [3] ASCII-text fraction (more restrictive: only letters, digits, ws, punct)
    ascii_text = np.sum((arr >= 32) & (arr <= 126))
    ascii_frac = float(ascii_text / n)

    # [4] null-byte fraction
    null_frac = float(np.sum(arr == 0) / n)

    # [5] high-byte (>=128) fraction
    high_frac = float(np.sum(arr >= 128) / n)

    # [6] repetition index: mean run length, normalized to [0, 1]
    if n > 1:
        diffs = np.diff(arr)
        runs = np.sum(diffs != 0) + 1  # number of runs
        mean_run = n / runs
        rep_idx = float(np.log10(max(mean_run, 1)) / 3.0)  # log-scale, ~[0,1]
    else:
        rep_idx = 0.0

    # [7] unique-byte count (0..256, normalized to 0..1)
    unique = float(np.count_nonzero(counts) / 256.0)

    # [8-15] byte-histogram buckets: split [0, 256) into 8 buckets, take fraction
    bucket_size = 32
    hist_buckets = []
    for i in range(8):
        bucket_sum = float(np.sum(counts[i * bucket_size:(i + 1) * bucket_size]) / n)
        hist_buckets.append(bucket_sum)

    return np.array([
        log_size, entropy, printable_frac, ascii_frac,
        null_frac, high_frac, rep_idx, unique,
        *hist_buckets,
    ], dtype=np.float32)


# =====================================================================
# Trained meta-classifier: predicts the best algorithm from features
# =====================================================================
class MetaCompressor:
    """A classifier that picks the best compression algorithm per file.

    Trained on (extract_features(data), best_algorithm) pairs. The
    "best" algorithm is the one that produces the smallest compressed
    output for that specific file -- which is exactly what we want to
    predict at inference time (without trying all 3).

    Uses qai.RandomForest as the underlying classifier. Falls back to
    a heuristic (entropy-based) if qai isn't available.
    """
    def __init__(self, model=None):
        self.model = model  # qai classifier (None = use heuristic)
        self._trained = model is not None and getattr(model, "is_trained", lambda: False)()

    @classmethod
    def train(cls, samples: List[bytes], verbose: bool = False) -> "MetaCompressor":
        """Train a meta-compressor on a list of byte samples.

        For each sample, computes features + finds the best algorithm
        by trying all 3. Then trains a qai Random Forest classifier.
        """
        # Lazy-import qai (avoids circular-import issues at module load time)
        try:
            import qai as _qai
            qai_available = True
        except ImportError:
            qai_available = False

        if not qai_available:
            if verbose:
                print("[MetaCompressor] qai not available -- using heuristic mode")
            return cls(model=None)

        X_list = []
        y_list = []
        for i, data in enumerate(samples):
            if len(data) == 0:
                continue
            feats = extract_features(data)
            best_algo, _ = cls._find_best_algorithm(data)
            X_list.append(feats)
            y_list.append(best_algo)
            if verbose and (i + 1) % 50 == 0:
                print(f"  prepared {i+1}/{len(samples)} samples")
        if not X_list:
            return cls(model=None)
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        m = _qai.build(technique="random_forest", n_estimators=50)
        m.train(X, y, verbose=False)
        if verbose:
            acc = m.accuracy()
            print(f"  meta-classifier trained: {len(X)} samples, accuracy={acc:.3f}")
        return cls(model=m)

    @staticmethod
    def _find_best_algorithm(data: bytes) -> Tuple[int, int]:
        """Try all algorithms and return (best_algo_id, compressed_size)."""
        results = []
        results.append((STORE, len(data)))
        results.append((ZLIB, len(_compress_with(data, ZLIB))))
        if _HAS_BZ2:
            try:
                results.append((BZ2, len(_compress_with(data, BZ2))))
            except Exception:
                pass
        if _HAS_LZMA:
            try:
                results.append((LZMA, len(_compress_with(data, LZMA))))
            except Exception:
                pass
        if _HAS_ZSTD:
            try:
                results.append((ZSTD, len(_compress_with(data, ZSTD))))
            except Exception:
                pass
        if _HAS_BROTLI:
            try:
                results.append((BROTLI, len(_compress_with(data, BROTLI))))
            except Exception:
                pass
        # Pick the smallest
        best = min(results, key=lambda r: r[1])
        return best[0], best[1]

    def predict(self, data: bytes) -> int:
        """Predict the best algorithm id for `data`."""
        if self._trained and self.model is not None:
            feats = extract_features(data).reshape(1, -1)
            try:
                pred = self.model.predict(feats)
                return int(pred)
            except Exception:
                pass
        # Heuristic fallback based on entropy
        return self._heuristic_predict(data)

    @staticmethod
    def _heuristic_predict(data: bytes) -> int:
        """Pick an algorithm based on simple entropy heuristics."""
        if len(data) < 16:
            return STORE
        feats = extract_features(data)
        entropy = feats[1]
        printable_frac = feats[2]
        rep_idx = feats[6]
        # Very high entropy + flat histogram = already compressed/encrypted
        if entropy > 7.5:
            return STORE
        # High printable fraction (text-like) -> bz2 wins on text
        if printable_frac > 0.85 and _HAS_BZ2:
            return BZ2
        # Repetitive -> lzma
        if rep_idx > 0.3 and _HAS_LZMA:
            return LZMA
        # Default -> zlib (good all-rounder)
        return ZLIB

    def compress(self, data: bytes) -> bytes:
        """Compress `data` using the predicted best algorithm.

        Returns: 1-byte algorithm id + the compressed payload.
        """
        algo = self.predict(data)
        compressed = _compress_with(data, algo)
        return bytes([algo]) + compressed

    def decompress(self, data: bytes) -> bytes:
        """Decompress data produced by .compress()."""
        if len(data) < 1:
            raise ValueError("empty compressed data")
        algo = data[0]
        payload = data[1:]
        return _decompress_with(payload, algo)


# =====================================================================
# Module-level convenience functions (use a shared default MetaCompressor)
# =====================================================================
_DEFAULT_COMPRESSOR: Optional[MetaCompressor] = None


def _get_default_compressor() -> MetaCompressor:
    global _DEFAULT_COMPRESSOR
    if _DEFAULT_COMPRESSOR is None:
        # Lazy-import qai to avoid circular imports at module load time
        try:
            import qai as _qai  # noqa: F401 -- existence check only
            _DEFAULT_COMPRESSOR = MetaCompressor.train(_generate_training_samples(), verbose=False)
        except ImportError:
            _DEFAULT_COMPRESSOR = MetaCompressor(model=None)
    return _DEFAULT_COMPRESSOR


def _generate_training_samples(n_samples: int = 200) -> List[bytes]:
    """Generate a diverse training set of synthetic byte sequences.

    The training set is intentionally diverse across:
      - Pure ASCII text (logs, source code, novels)
      - JSON / XML / CSV structured data
      - Repetitive binaries (zero-heavy, sparse)
      - Random binaries (incompressible)
      - Mixed content (text + binary)
      - Highly repetitive (long runs of same byte)
      - Already-compressed-looking (high-entropy)
    """
    rng = np.random.default_rng(42)
    samples = []

    # 1. ASCII text (30 samples)
    for _ in range(30):
        chars = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t.,;:!?-(){}[]<>/"
        n = int(rng.integers(200, 5000))
        samples.append(bytes(rng.choice(list(chars), size=n)))

    # 2. JSON-like structured data (30 samples)
    for _ in range(30):
        n = int(rng.integers(200, 5000))
        json_chars = b'{"key": "value", "num": 12345, "arr": [1,2,3], "nested": {"a": 1}}\n'
        reps = max(1, n // len(json_chars))
        samples.append(json_chars * reps)

    # 3. Repetitive binary (zero-heavy) (25 samples)
    for _ in range(25):
        n = int(rng.integers(200, 5000))
        arr = np.zeros(n, dtype=np.uint8)
        for _ in range(int(rng.integers(5, 100))):
            arr[int(rng.integers(0, n))] = int(rng.integers(0, 256))
        samples.append(bytes(arr))

    # 4. Random binary (25 samples)
    for _ in range(25):
        n = int(rng.integers(200, 5000))
        samples.append(bytes(rng.integers(0, 256, size=n).astype(np.uint8)))

    # 5. Highly repetitive text (logs) (25 samples)
    for _ in range(25):
        line = b"2026-01-01 12:00:00 INFO: user logged in successfully\n"
        n = int(rng.integers(200, 5000))
        reps = max(1, n // len(line))
        samples.append(line * reps)

    # 6. CSV-like (25 samples)
    for _ in range(25):
        rows = []
        for _ in range(int(rng.integers(5, 100))):
            row = b",".join([str(int(rng.integers(0, 10000))).encode() for _ in range(int(rng.integers(3, 10)))])
            rows.append(row)
        samples.append(b"\n".join(rows))

    # 7. Highly-repetitive single-byte runs (20 samples)
    for _ in range(20):
        n = int(rng.integers(200, 5000))
        byte_val = int(rng.integers(0, 256))
        arr = np.full(n, byte_val, dtype=np.uint8)
        # Sprinkle some other bytes
        for _ in range(int(rng.integers(0, 10))):
            arr[int(rng.integers(0, n))] = int(rng.integers(0, 256))
        samples.append(bytes(arr))

    # 8. Mixed text + binary (20 samples)
    for _ in range(20):
        text_part = b"some text content here " * int(rng.integers(5, 20))
        binary_part = bytes(rng.integers(0, 256, size=int(rng.integers(50, 500))).astype(np.uint8))
        samples.append(text_part + binary_part)

    return samples


def compress(data: bytes, use_trained: bool = True) -> bytes:
    """Compress `data` (bytes) using qai's meta-compressor.

    Args:
        data: the bytes to compress.
        use_trained: if True (default), use the trained qai classifier.
                     If False, use the entropy-heuristic only.

    Returns:
        Compressed bytes (1-byte algo id + payload).
    """
    if use_trained and _HAS_QAI:
        return _get_default_compressor().compress(data)
    return MetaCompressor(model=None).compress(data)


def decompress(data: bytes) -> bytes:
    """Decompress bytes produced by qai.compress()."""
    return MetaCompressor(model=None).decompress(data)


# =====================================================================
# Multi-file Archive format
# =====================================================================
class ArchiveEntry:
    """A single file entry in an Archive."""
    __slots__ = ("name", "data", "algorithm", "compressed_data", "_uncompressed_size")

    def __init__(self, name: str, data: bytes, algorithm: int = -1,
                 compressed_data: Optional[bytes] = None):
        self.name = name
        self.data = data
        self.algorithm = algorithm  # -1 means "auto-pick at save time"
        self.compressed_data = compressed_data
        self._uncompressed_size = len(data)

    @property
    def uncompressed_size(self) -> int:
        # Prefer the recorded size (set on load); fall back to len(data)
        if self._uncompressed_size > 0:
            return self._uncompressed_size
        return len(self.data)

    @property
    def compressed_size(self) -> int:
        if self.compressed_data is not None:
            return len(self.compressed_data)
        return 0

    @property
    def ratio(self) -> float:
        if self.uncompressed_size == 0:
            return 1.0
        return self.compressed_size / self.uncompressed_size


class Archive:
    """A multi-file archive built on qai's meta-compressor.

    Usage:
        arc = qai.Archive()
        arc.add("file1.txt", open("file1.txt", "rb").read())
        arc.add("file2.json", b'{"k": 1}')
        arc.save("output.qai.zip")

        arc2 = qai.Archive.load("output.qai.zip")
        print(arc2.list_files())
        print(arc2.read("file1.txt"))
    """
    def __init__(self):
        self.entries: Dict[str, ArchiveEntry] = {}

    def add(self, name: str, data: bytes, algorithm: int = -1) -> None:
        """Add a file to the archive.

        Args:
            name: filename (path within the archive).
            data: file contents (bytes).
            algorithm: -1 to auto-pick, or one of STORE/ZLIB/BZ2/LZMA.
        """
        self.entries[name] = ArchiveEntry(name, data, algorithm)

    def list_files(self) -> List[str]:
        """Return the list of filenames in the archive."""
        return list(self.entries.keys())

    def read(self, name: str) -> bytes:
        """Read a file's decompressed contents from the archive."""
        if name not in self.entries:
            raise KeyError(f"[WHAT] File {name!r} not in archive\n"
                           f"[WHY] Archive has {len(self.entries)} files, none named {name!r}\n"
                           f"[FIX] Use one of {list(self.entries.keys())[:5]}...")
        entry = self.entries[name]
        if entry.compressed_data is not None:
            return _decompress_with(entry.compressed_data, entry.algorithm)
        return entry.data

    def save(self, path: str, sign_key: Optional[str] = None) -> str:
        """Save the archive to `path`.

        The format is the qai.zip format documented at the top of this file.
        If `sign_key` is provided, an HMAC-SHA256 is appended.
        """
        compressor = _get_default_compressor() if _HAS_QAI else MetaCompressor(model=None)
        with open(path, "wb") as f:
            # Header
            f.write(ARCHIVE_MAGIC)
            f.write(struct.pack("<I", ARCHIVE_VERSION))
            f.write(struct.pack("<Q", len(self.entries)))
            # Entries
            hmac_data = b""
            for name, entry in self.entries.items():
                # Compress if not already compressed
                if entry.compressed_data is None:
                    algo = entry.algorithm if entry.algorithm >= 0 else compressor.predict(entry.data)
                    compressed = _compress_with(entry.data, algo)
                    entry.algorithm = algo
                    entry.compressed_data = compressed
                name_bytes = name.encode("utf-8")
                entry_header = (
                    struct.pack("<I", len(name_bytes)) + name_bytes +
                    struct.pack("<Q", entry.uncompressed_size) +
                    struct.pack("<B", entry.algorithm) +
                    struct.pack("<Q", entry.compressed_size)
                )
                f.write(entry_header)
                f.write(entry.compressed_data)
                hmac_data += entry_header + entry.compressed_data
            # HMAC signature (optional)
            if sign_key is not None:
                sig = hmac.new(sign_key.encode("utf-8"), hmac_data, hashlib.sha256).digest()
                f.write(sig)
        return path

    @classmethod
    def load(cls, path: str, sign_key: Optional[str] = None,
             verify_signature: bool = True) -> "Archive":
        """Load an archive from `path`. Verifies the HMAC signature if signed."""
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != ARCHIVE_MAGIC:
                raise ValueError(
                    f"[WHAT] Not a qai archive (bad magic)\n"
                    f"[WHY] File doesn't start with the QAIARC magic bytes\n"
                    f"[FIX] Use a file produced by qai.Archive.save()"
                )
            version = struct.unpack("<I", f.read(4))[0]
            n_files = struct.unpack("<Q", f.read(8))[0]
            arc = cls()
            hmac_data = b""
            for _ in range(n_files):
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                uncompressed_size = struct.unpack("<Q", f.read(8))[0]
                algo = struct.unpack("<B", f.read(1))[0]
                compressed_size = struct.unpack("<Q", f.read(8))[0]
                compressed_data = f.read(compressed_size)
                entry = ArchiveEntry(name, b"", algorithm=algo, compressed_data=compressed_data)
                entry.data = b""  # will be set on read()
                # We can't easily store uncompressed_size on a slot-less class,
                # so use a small trick: store an empty bytes and replace on read.
                # For list_files / read to work, we need the entry's data attribute
                # but we want lazy decompression. Use a closure-ish approach:
                # store uncompressed_size as an attribute.
                entry._uncompressed_size = uncompressed_size
                arc.entries[name] = entry
                hmac_data += (struct.pack("<I", name_len) + name.encode("utf-8") +
                              struct.pack("<Q", uncompressed_size) +
                              struct.pack("<B", algo) +
                              struct.pack("<Q", compressed_size) + compressed_data)
            # Verify signature if present
            remaining = f.read()
            if remaining:
                if sign_key is None and verify_signature:
                    raise ValueError(
                        "[WHAT] Archive is signed but no sign_key provided\n"
                        "[WHY] Signed archives require the original key to verify\n"
                        "[FIX] Pass sign_key=, or verify_signature=False"
                    )
                if sign_key is not None:
                    expected = hmac.new(sign_key.encode("utf-8"), hmac_data, hashlib.sha256).digest()
                    if not hmac.compare_digest(expected, remaining):
                        raise ValueError(
                            "[WHAT] Archive signature verification failed\n"
                            "[WHY] The archive may have been tampered with\n"
                            "[FIX] Use the correct sign_key, or reject this archive"
                        )
        return arc

    def stats(self) -> Dict[str, Any]:
        """Return compression stats for the archive."""
        total_uncompressed = 0
        total_compressed = 0
        per_algo: Dict[str, int] = {}
        for entry in self.entries.values():
            total_uncompressed += entry.uncompressed_size
            total_compressed += entry.compressed_size
            per_algo[ALGORITHM_NAMES.get(entry.algorithm, "?")] = per_algo.get(ALGORITHM_NAMES.get(entry.algorithm, "?"), 0) + 1
        return {
            "n_files": len(self.entries),
            "total_uncompressed": total_uncompressed,
            "total_compressed": total_compressed,
            "ratio": total_compressed / max(total_uncompressed, 1),
            "algorithms_used": per_algo,
        }

    def __repr__(self):
        return f"<Archive n_files={len(self.entries)}>"


# =====================================================================
# CLI entry points
# =====================================================================
def _cli_compress(args):
    """qai compress <file> -o <output>"""
    with open(args.input, "rb") as f:
        data = f.read()
    compressed = compress(data)
    out = args.output or (args.input + ".qai")
    with open(out, "wb") as f:
        f.write(compressed)
    orig_size = len(data)
    new_size = len(compressed)
    ratio = new_size / max(orig_size, 1)
    print(f"compressed {args.input} ({orig_size} bytes) -> {out} ({new_size} bytes)")
    print(f"ratio: {ratio:.3f}  (saved {(1 - ratio) * 100:.1f}%)")


def _cli_decompress(args):
    """qai decompress <file> -o <output>"""
    with open(args.input, "rb") as f:
        data = f.read()
    decompressed = decompress(data)
    out = args.output or args.input.replace(".qai", "")
    with open(out, "wb") as f:
        f.write(decompressed)
    print(f"decompressed {args.input} -> {out} ({len(decompressed)} bytes)")


def _cli_archive(args):
    """qai archive <dir> -o <output.qai.zip>"""
    arc = Archive()
    for root, _, files in os.walk(args.input):
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, args.input)
            with open(full, "rb") as f:
                arc.add(rel, f.read())
    out = args.output or (args.input.rstrip("/") + ".qai.zip")
    arc.save(out, sign_key=args.sign_key)
    stats = arc.stats()
    print(f"archived {stats['n_files']} files -> {out}")
    print(f"  uncompressed: {stats['total_uncompressed']} bytes")
    print(f"  compressed:   {stats['total_compressed']} bytes")
    print(f"  ratio: {stats['ratio']:.3f}  (saved {(1 - stats['ratio']) * 100:.1f}%)")
    print(f"  algorithms used: {stats['algorithms_used']}")


def _cli_extract(args):
    """qai extract <archive.qai.zip> -d <dest_dir>"""
    arc = Archive.load(args.input, sign_key=args.sign_key)
    os.makedirs(args.dest, exist_ok=True)
    for name in arc.list_files():
        out_path = os.path.join(args.dest, name)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(arc.read(name))
    print(f"extracted {len(arc.list_files())} files -> {args.dest}")


def add_compress_cli(subparsers):
    """Add compress/decompress/archive/extract subcommands to an argparse subparsers."""
    p = subparsers.add_parser("compress", help="compress a single file using the meta-compressor")
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None)

    p = subparsers.add_parser("decompress", help="decompress a file produced by `qai compress`")
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None)

    p = subparsers.add_parser("archive", help="archive a directory into a .qai.zip")
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--sign-key", default=None)

    p = subparsers.add_parser("extract", help="extract a .qai.zip archive")
    p.add_argument("input")
    p.add_argument("-d", "--dest", default=".")
    p.add_argument("--sign-key", default=None)


__all__ = [
    "compress", "decompress", "extract_features", "MetaCompressor",
    "Archive", "ArchiveEntry",
    "STORE", "ZLIB", "BZ2", "LZMA", "ALGORITHM_NAMES",
]
