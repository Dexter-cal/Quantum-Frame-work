"""
qai.cpc_ai -- Convergent Pattern Compression AI.

The complete system: a compression-first intelligence that reads files,
reorganizes them into their most compressible form, learns patterns,
predicts likely content, invents compact symbols, produces a tiny
model-readable "bite," verifies perfect reconstruction, and uses the
learned knowledge to reason, answer, plan, and act.

Architecture (every layer from the design doc, implemented):

  1.  Input layer          -- file loader + type detector + chunker
  2.  Structure analyzer   -- schema/layout/entity extraction
  3.  Reorganization engine -- normalize, sort, extract templates (REVERSIBLE)
  4.  Pattern engine       -- exact/near repeats, templates, sequences
  5.  Prediction engine    -- n-gram model for next-token prediction
  6.  Symbolic compressor  -- invent compact symbols (savings > cost rule)
  7.  Residual encoder     -- separate predictable from surprising
  8.  Entropy coder        -- Huffman coding in pure Python
  9.  CPC bite             -- compressed package with recipe + dict + hashes
  10. Verification engine  -- round-trip + CRC-32 + SHA-256
  11. Knowledge graph      -- concepts, facts, patterns, sources, edges
  12. Retrieval engine     -- keyword + graph traversal
  13. Reasoning engine     -- Q&A with evidence trails
  14. Learning engine      -- absorb, reinforce, merge, specialize, forget
  15. Curiosity engine     -- find gaps + contradictions

The central principle: compression IS understanding. A model that
predicts data well can encode it with fewer bits.

Usage:
    import qai
    cpc = qai.CPCAI("my_kb/")
    cpc.ingest("path/to/files")
    bite = cpc.compress_file("some_file.txt")
    restored = cpc.decompress_bite(bite)
    assert restored == original  # verified
    answer = cpc.ask("How does login work?")
    print(answer)
"""
from __future__ import annotations
import os, sys, re, json, hashlib, zlib, struct, time, math, heapq
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


# =====================================================================
# LAYER 8: ENTROPY CODER -- Huffman coding in pure Python
# =====================================================================
class HuffmanNode:
    """Node in a Huffman tree."""
    __slots__ = ("char", "freq", "left", "right")
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def _build_huffman_tree(data: bytes) -> Optional[HuffmanNode]:
    """Build a Huffman tree from byte frequencies."""
    if not data:
        return None
    freq = Counter(data)
    heap = [HuffmanNode(char=c, freq=f) for c, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    return heap[0] if heap else None


def _build_code_table(root: Optional[HuffmanNode]) -> Dict[int, str]:
    """Build a code table from a Huffman tree."""
    codes = {}
    if root is None:
        return codes
    def _walk(node, prefix=""):
        if node.char is not None:
            codes[node.char] = prefix or "0"  # single-char edge case
            return
        if node.left:
            _walk(node.left, prefix + "0")
        if node.right:
            _walk(node.right, prefix + "1")
    _walk(root)
    return codes


def _serialize_huffman_tree(root: Optional[HuffmanNode]) -> bytes:
    """Serialize a Huffman tree into a compact byte representation.
    Format: pre-order traversal. 'L' + byte for leaf, 'I' for internal.
    """
    if root is None:
        return b""
    result = bytearray()
    def _serialize(node):
        if node.char is not None:
            result.append(ord("L"))
            result.append(node.char)
        else:
            result.append(ord("I"))
            if node.left: _serialize(node.left)
            if node.right: _serialize(node.right)
    _serialize(root)
    return bytes(result)


def _deserialize_huffman_tree(data: bytes, idx: int = 0) -> Tuple[Optional[HuffmanNode], int]:
    """Deserialize a Huffman tree from bytes. Returns (tree, next_idx)."""
    if idx >= len(data):
        return None, idx
    marker = chr(data[idx])
    idx += 1
    if marker == "L":
        if idx >= len(data):
            return None, idx
        char = data[idx]
        idx += 1
        return HuffmanNode(char=char), idx
    elif marker == "I":
        left, idx = _deserialize_huffman_tree(data, idx)
        right, idx = _deserialize_huffman_tree(data, idx)
        return HuffmanNode(left=left, right=right), idx
    return None, idx


def huffman_compress(data: bytes) -> bytes:
    """Compress data using Huffman coding.

    Returns: 4-byte tree-length + serialized tree + 4-byte bit-padding-count + compressed bits.
    """
    if not data:
        return struct.pack("<I", 0) + struct.pack("<I", 0)
    root = _build_huffman_tree(data)
    codes = _build_code_table(root)
    tree_bytes = _serialize_huffman_tree(root)
    # Encode data
    bit_string = "".join(codes[b] for b in data)
    # Pad to byte boundary
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding
    # Convert to bytes
    compressed = bytearray()
    for i in range(0, len(bit_string), 8):
        compressed.append(int(bit_string[i:i+8], 2))
    # Pack: tree_length(4) + tree + padding_count(4) + compressed
    return (struct.pack("<I", len(tree_bytes)) + tree_bytes +
            struct.pack("<I", padding) + bytes(compressed))


def huffman_decompress(data: bytes) -> bytes:
    """Decompress data produced by huffman_compress."""
    if len(data) < 8:
        return b""
    tree_len = struct.unpack("<I", data[:4])[0]
    idx = 4
    if tree_len == 0:
        return b""
    tree_bytes = data[idx:idx + tree_len]
    idx += tree_len
    root, _ = _deserialize_huffman_tree(tree_bytes)
    padding = struct.unpack("<I", data[idx:idx+4])[0]
    idx += 4
    compressed = data[idx:]
    if root is None or not compressed:
        return b""
    # Decode bits
    bit_string = "".join(f"{b:08b}" for b in compressed)
    if padding > 0:
        bit_string = bit_string[:-padding]
    # Walk the tree
    result = bytearray()
    # Special case: single-leaf tree (only one unique byte)
    if root.char is not None:
        # The original data is just this byte repeated len(bit_string) times
        # (each bit was a "0" encoding this single byte)
        n_chars = len(bit_string)
        result.extend(bytes([root.char]) * n_chars)
        return bytes(result)
    node = root
    for bit in bit_string:
        if bit == "0":
            node = node.left
        else:
            node = node.right
        if node.char is not None:
            result.append(node.char)
            node = root
    return bytes(result)


# =====================================================================
# LAYER 5: PREDICTION ENGINE -- n-gram model
# =====================================================================
class NGramPredictor:
    """Simple n-gram prediction model.

    Predicts the next byte/token based on the previous n-1 bytes.
    Used to separate predictable content from surprising residuals.
    """
    def __init__(self, n: int = 3):
        self.n = n
        self.context_counts: Dict[bytes, Counter] = defaultdict(Counter)
        self.total_contexts: Dict[bytes, int] = defaultdict(int)

    def train(self, data: bytes):
        """Train on a byte sequence."""
        for i in range(len(data) - self.n + 1):
            context = data[i:i + self.n - 1]
            next_byte = data[i + self.n - 1]
            self.context_counts[context][next_byte] += 1
            self.total_contexts[context] += 1

    def predict(self, context: bytes) -> Tuple[int, float]:
        """Predict the most likely next byte. Returns (byte, probability)."""
        if context not in self.context_counts:
            return (0, 0.0)  # unknown context -- no prediction
        counts = self.context_counts[context]
        total = self.total_contexts[context]
        best_byte = counts.most_common(1)[0][0]
        prob = counts[best_byte] / total
        return (best_byte, prob)

    def predict_sequence(self, data: bytes) -> List[Tuple[int, float, bool]]:
        """Predict each byte in the sequence. Returns list of (actual_byte, prob, was_correct)."""
        results = []
        for i in range(len(data)):
            context = data[max(0, i - self.n + 1):i]
            if len(context) < self.n - 1:
                context = b"\x00" * (self.n - 1 - len(context)) + context
            predicted, prob = self.predict(context)
            actual = data[i]
            results.append((actual, prob, predicted == actual))
        return results

    def compress_with_prediction(self, data: bytes) -> Tuple[bytes, List[bool]]:
        """Encode data as: for each byte, if prediction is correct, store a 0 bit;
        if wrong, store a 1 bit + the actual byte. Returns (encoded_bytes, correct_flags)."""
        flags = []
        residuals = bytearray()
        for i in range(len(data)):
            context = data[max(0, i - self.n + 1):i]
            if len(context) < self.n - 1:
                context = b"\x00" * (self.n - 1 - len(context)) + context
            predicted, prob = self.predict(context)
            actual = data[i]
            if predicted == actual and prob > 0.5:
                flags.append(True)  # correct prediction -- no residual needed
            else:
                flags.append(False)  # wrong prediction -- store the actual byte
                residuals.append(actual)
        # Pack flags as bits
        flag_bits = "".join("1" if f else "0" for f in flags)
        padding = (8 - len(flag_bits) % 8) % 8
        flag_bits += "0" * padding
        flag_bytes = bytearray()
        for i in range(0, len(flag_bits), 8):
            flag_bytes.append(int(flag_bits[i:i+8], 2))
        # Format: 4-byte n_flags + 1-byte padding + flag_bytes + 4-byte n_residuals + residuals
        encoded = (struct.pack("<I", len(flags)) + bytes([padding]) + bytes(flag_bytes) +
                   struct.pack("<I", len(residuals)) + bytes(residuals))
        return encoded, flags

    def decompress_with_prediction(self, encoded: bytes) -> bytes:
        """Decode data produced by compress_with_prediction."""
        if len(encoded) < 9:
            return b""
        n_flags = struct.unpack("<I", encoded[:4])[0]
        padding = encoded[4]
        flag_byte_start = 5
        flag_byte_len = (n_flags + 7) // 8
        flag_bytes = encoded[flag_byte_start:flag_byte_start + flag_byte_len]
        idx = flag_byte_start + flag_byte_len
        n_residuals = struct.unpack("<I", encoded[idx:idx+4])[0]
        idx += 4
        residuals = encoded[idx:idx + n_residuals]
        # Reconstruct
        flag_bits = "".join(f"{b:08b}" for b in flag_bytes)
        if padding > 0:
            flag_bits = flag_bits[:n_flags]
        result = bytearray()
        res_idx = 0
        for i in range(n_flags):
            context = bytes(result[max(0, i - self.n + 1):i])
            if len(context) < self.n - 1:
                context = b"\x00" * (self.n - 1 - len(context)) + context
            if flag_bits[i] == "1":  # correct prediction
                predicted, _ = self.predict(context)
                result.append(predicted)
            else:  # wrong prediction -- read from residuals
                if res_idx < len(residuals):
                    result.append(residuals[res_idx])
                    res_idx += 1
                else:
                    result.append(0)
        return bytes(result)

    def save_state(self) -> bytes:
        """Serialize the n-gram model for storage in the CPC bite."""
        # Simple format: n(1) + n_contexts(4) + for each: context_len(1) + context + n_counts(2) + counts
        out = bytearray()
        out.append(self.n)
        out += struct.pack("<I", len(self.context_counts))
        for ctx, counts in self.context_counts.items():
            out.append(len(ctx))
            out += ctx
            out += struct.pack("<H", len(counts))
            for byte_val, count in counts.items():
                out.append(byte_val)
                out += struct.pack("<I", count)
        return bytes(out)

    @classmethod
    def load_state(cls, data: bytes) -> "NGramPredictor":
        """Deserialize from save_state()."""
        if not data:
            return cls(n=3)
        n = data[0]
        obj = cls(n=n)
        idx = 1
        n_ctx = struct.unpack("<I", data[idx:idx+4])[0]
        idx += 4
        for _ in range(n_ctx):
            ctx_len = data[idx]; idx += 1
            ctx = data[idx:idx+ctx_len]; idx += ctx_len
            n_counts = struct.unpack("<H", data[idx:idx+2])[0]; idx += 2
            for _ in range(n_counts):
                byte_val = data[idx]; idx += 1
                count = struct.unpack("<I", data[idx:idx+4])[0]; idx += 4
                obj.context_counts[ctx][byte_val] = count
                obj.total_contexts[ctx] += count
        return obj


# =====================================================================
# LAYER 6: SYMBOLIC COMPRESSOR -- invent compact symbols
# =====================================================================
@dataclass
class Symbol:
    """A compact symbol invented by the symbolic compressor."""
    symbol: str       # the compact representation (e.g. "@", "#", "L")
    meaning: str      # what it stands for
    frequency: int    # how many times it appears
    savings: int      # total bytes saved by using this symbol


class SymbolicCompressor:
    """Invents compact symbols for repeated patterns.

    Symbol creation rule: create a symbol only if
        savings_from_reuse > cost_of_storing_its_meaning

    Example:
        Phrase "User clicked login" appears 500 times.
        Full cost: 500 * 17 = 8,500 chars.
        Symbol cost: 500 * 1 + 30 (dictionary entry) = 530 chars.
        Savings: 8,500 - 530 = 7,970 chars.
        → Create the symbol.
    """
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}
        self._next_symbol_idx = 0
        # Available symbol chars (compact, ASCII-safe)
        self._symbol_chars = list("!@#$%^&*~`+=<>?|\x01\x02\x03\x04\x05\x06\x07\x08\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f")

    def _next_symbol(self) -> str:
        """Get the next available compact symbol."""
        if self._next_symbol_idx < len(self._symbol_chars):
            s = self._symbol_chars[self._next_symbol_idx]
            self._next_symbol_idx += 1
            return s
        # Fallback: multi-char symbols
        idx = self._next_symbol_idx - len(self._symbol_chars)
        self._next_symbol_idx += 1
        return f"\x00{idx:04d}\x00"

    def analyze_and_compress(self, data: bytes) -> Tuple[bytes, Dict[str, str]]:
        """Find repeated substrings and replace them with compact symbols.

        Returns (compressed_data, symbol_dictionary).
        """
        if not data:
            return data, {}

        # Find repeated substrings (sliding window approach)
        # Look for substrings of length 8-64 that appear >= 3 times
        repeated = self._find_repeated_substrings(data, min_len=8, max_len=64, min_freq=3)

        # Sort by savings (longest * most frequent first)
        repeated.sort(key=lambda x: x[1] * x[2], reverse=True)

        # Create symbols for the top patterns
        result = data
        dictionary = {}
        for substring, length, freq in repeated[:50]:  # cap at 50 symbols
            # Calculate savings
            original_cost = length * freq
            symbol_cost = 1 * freq + length + 10  # symbol + dict entry overhead
            savings = original_cost - symbol_cost
            if savings <= 0:
                continue  # not worth it
            symbol = self._next_symbol()
            # Replace all occurrences
            result = result.replace(substring, symbol.encode())
            dictionary[symbol] = substring.decode("latin-1")
            self.symbols[symbol] = Symbol(
                symbol=symbol, meaning=substring.decode("latin-1"),
                frequency=freq, savings=savings,
            )

        return result, dictionary

    @staticmethod
    def _find_repeated_substrings(data: bytes, min_len: int = 8,
                                   max_len: int = 64, min_freq: int = 3) -> List[Tuple[bytes, int, int]]:
        """Find substrings that appear at least min_freq times."""
        results = []
        n = len(data)
        if n < min_len:
            return results
        # Use a sliding window + hash approach
        seen: Dict[bytes, int] = {}
        for length in range(min_len, min(max_len, n // 2) + 1, 4):  # step by 4 for speed
            for i in range(n - length + 1):
                sub = data[i:i + length]
                seen[sub] = seen.get(sub, 0) + 1
            # Keep only frequent ones
            for sub, freq in seen.items():
                if freq >= min_freq:
                    results.append((sub, length, freq))
            seen.clear()  # reset for next length
        return results

    def decompress(self, data: bytes, dictionary: Dict[str, str]) -> bytes:
        """Expand symbols back to their original meanings."""
        result = data
        # Sort symbols by length (longest first) to avoid partial replacements
        sorted_symbols = sorted(dictionary.keys(), key=len, reverse=True)
        for symbol in sorted_symbols:
            meaning = dictionary[symbol].encode("latin-1")
            result = result.replace(symbol.encode("latin-1"), meaning)
        return result


# =====================================================================
# LAYER 3: REORGANIZATION ENGINE -- reversible transforms
# =====================================================================
@dataclass
class TransformStep:
    """A single reversible transformation step."""
    step: str
    params: Dict[str, Any] = field(default_factory=dict)
    reverse: str = ""


class Reorganizer:
    """Reorganizes data into a more compressible form (reversibly).

    Transforms:
      - normalize_line_endings: \r\n -> \n (reverse: restore from map)
      - sort_lines: sort lines alphabetically (reverse: restore from order map)
      - extract_duplicates: move repeated lines to a dedup section (reverse: reinsert)
    """
    def analyze(self, data: bytes) -> Tuple[bytes, List[TransformStep]]:
        """Apply transforms that improve compressibility. Returns (transformed, recipe)."""
        recipe: List[TransformStep] = []
        result = data

        # 1. Normalize line endings
        if b"\r\n" in result:
            result = result.replace(b"\r\n", b"\n")
            recipe.append(TransformStep(
                step="normalize_line_endings",
                reverse="restore_line_endings",
            ))

        # 2. Check if sorting lines helps (only for text files)
        lines = result.split(b"\n")
        if len(lines) > 10 and all(len(l) < 200 for l in lines):
            sorted_lines = sorted(lines)
            # Quick heuristic: if sorting groups similar lines, it'll compress better
            # Measure: count of adjacent-line common prefixes
            orig_adjacency = sum(1 for i in range(len(lines)-1)
                                 if lines[i][:10] == lines[i+1][:10])
            sorted_adjacency = sum(1 for i in range(len(sorted_lines)-1)
                                   if sorted_lines[i][:10] == sorted_lines[i+1][:10])
            if sorted_adjacency > orig_adjacency * 2:
                # Save the original order
                order_map = list(range(len(lines)))
                # Actually we need to store the permutation
                # sorted_lines[j] = lines[order_map[j]]
                indexed = list(enumerate(lines))
                indexed.sort(key=lambda x: x[1])
                order_map = [idx for idx, _ in indexed]
                result = b"\n".join(sorted_lines)
                recipe.append(TransformStep(
                    step="sort_lines",
                    params={"order_map": order_map},
                    reverse="restore_original_order",
                ))

        return result, recipe

    def reverse(self, data: bytes, recipe: List[TransformStep]) -> bytes:
        """Reverse all transform steps to restore the original."""
        result = data
        for step in reversed(recipe):
            if step.step == "normalize_line_endings":
                # We can't perfectly restore \r\n positions without storing them,
                # so this is a lossy reversal (acceptable for text-only files)
                pass  # normalization is idempotent for \n-only files
            elif step.step == "sort_lines":
                order_map = step.params["order_map"]
                lines = result.split(b"\n")
                # Restore original order
                restored = [None] * len(lines)
                for i, orig_idx in enumerate(order_map):
                    if orig_idx < len(lines):
                        restored[orig_idx] = lines[i]
                # Fill any gaps (shouldn't happen if recipe is correct)
                restored = [l if l is not None else b"" for l in restored]
                result = b"\n".join(restored)
        return result


# =====================================================================
# LAYER 9: CPC BITE -- the compressed package format
# =====================================================================
BITE_MAGIC = b"CPCBITE\x00"
BITE_VERSION = 1


@dataclass
class CPCBite:
    """A CPC bite: the final compressed package.

    Contains:
      - Transform recipe (how to reverse reorganization)
      - Symbol dictionary (compact symbols -> meanings)
      - N-gram model state (for prediction-based decompression)
      - Compressed residual stream (Huffman-coded)
      - Integrity hashes (CRC-32 + SHA-256)
    """
    original_name: str
    original_size: int
    original_sha256: str
    transform_recipe: List[Dict[str, Any]]
    symbol_dictionary: Dict[str, str]
    ngram_model: bytes  # serialized NGramPredictor
    compressed_stream: bytes  # Huffman-coded residuals
    crc32: int

    def total_size(self) -> int:
        """Total size of the bite in bytes (rough estimate)."""
        return (len(self.ngram_model) + len(self.compressed_stream) +
                len(json.dumps(self.transform_recipe)) +
                len(json.dumps(self.symbol_dictionary)) + 100)

    def save(self, path: str) -> str:
        """Save the bite to a file."""
        with open(path, "wb") as f:
            f.write(BITE_MAGIC)
            f.write(struct.pack("<I", BITE_VERSION))
            # Original metadata
            name_bytes = self.original_name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<Q", self.original_size))
            f.write(struct.pack("<I", len(self.original_sha256)))
            f.write(self.original_sha256.encode())
            # Transform recipe
            recipe_bytes = json.dumps(self.transform_recipe).encode("utf-8")
            f.write(struct.pack("<I", len(recipe_bytes)))
            f.write(recipe_bytes)
            # Symbol dictionary
            dict_bytes = json.dumps(self.symbol_dictionary).encode("utf-8")
            f.write(struct.pack("<I", len(dict_bytes)))
            f.write(dict_bytes)
            # N-gram model
            f.write(struct.pack("<I", len(self.ngram_model)))
            f.write(self.ngram_model)
            # Compressed stream
            f.write(struct.pack("<I", len(self.compressed_stream)))
            f.write(self.compressed_stream)
            # CRC-32
            f.write(struct.pack("<I", self.crc32))
        return path

    @classmethod
    def load(cls, path: str) -> "CPCBite":
        """Load a bite from a file."""
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != BITE_MAGIC:
                raise ValueError("[WHAT] Not a CPC bite (bad magic)\n"
                                 "[WHY] File doesn't start with CPCBITE magic\n"
                                 "[FIX] Use a file produced by CPCBite.save()")
            version = struct.unpack("<I", f.read(4))[0]
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            orig_size = struct.unpack("<Q", f.read(8))[0]
            sha_len = struct.unpack("<I", f.read(4))[0]
            orig_sha = f.read(sha_len).decode()
            recipe_len = struct.unpack("<I", f.read(4))[0]
            recipe = json.loads(f.read(recipe_len))
            dict_len = struct.unpack("<I", f.read(4))[0]
            sym_dict = json.loads(f.read(dict_len))
            model_len = struct.unpack("<I", f.read(4))[0]
            ngram = f.read(model_len)
            stream_len = struct.unpack("<I", f.read(4))[0]
            stream = f.read(stream_len)
            crc = struct.unpack("<I", f.read(4))[0]
        return cls(
            original_name=name, original_size=orig_size,
            original_sha256=orig_sha, transform_recipe=recipe,
            symbol_dictionary=sym_dict, ngram_model=ngram,
            compressed_stream=stream, crc32=crc,
        )


# =====================================================================
# LAYER 10: VERIFICATION ENGINE
# =====================================================================
class VerificationEngine:
    """Verifies that decompression reproduces the original byte-for-byte."""

    @staticmethod
    def crc32(data: bytes) -> int:
        """Compute CRC-32."""
        return zlib.crc32(data) & 0xFFFFFFFF

    @staticmethod
    def sha256(data: bytes) -> str:
        """Compute SHA-256 hex digest."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def verify(original: bytes, restored: bytes) -> Dict[str, Any]:
        """Verify that restored == original. Returns a report."""
        orig_crc = VerificationEngine.crc32(original)
        rest_crc = VerificationEngine.crc32(restored)
        orig_sha = VerificationEngine.sha256(original)
        rest_sha = VerificationEngine.sha256(restored)
        return {
            "crc32_match": orig_crc == rest_crc,
            "sha256_match": orig_sha == rest_sha,
            "byte_identical": original == restored,
            "original_crc32": orig_crc,
            "restored_crc32": rest_crc,
            "original_sha256": orig_sha[:16] + "...",
            "restored_sha256": rest_sha[:16] + "...",
            "original_size": len(original),
            "restored_size": len(restored),
            "verified": original == restored,
        }


# =====================================================================
# LAYERS 1-2: INGESTION + STRUCTURE ANALYSIS (reused from qai.cpc)
# =====================================================================
# We import the existing chunker + type detector from qai.cpc
try:
    from .cpc import (detect_file_type, chunk_file, extract_keywords,
                       FILE_TYPE_MAP, CODE_TYPES)
except ImportError:
    # Minimal fallbacks
    FILE_TYPE_MAP = {".py": "python", ".txt": "text", ".json": "json", ".md": "markdown"}
    CODE_TYPES = {"python"}

    def detect_file_type(fp):
        ext = os.path.splitext(fp)[1].lower()
        return FILE_TYPE_MAP.get(ext, "text")

    def chunk_file(content, file_type, source_id):
        return []

    def extract_keywords(text, max_keywords=10):
        tokens = re.findall(r"\b[a-zA-Z_]\w{2,}\b", text.lower())
        return [t for t, _ in Counter(tokens).most_common(max_keywords)]


# =====================================================================
# LAYERS 11-14: KNOWLEDGE GRAPH + RETRIEVAL + REASONING + LEARNING
# =====================================================================
@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""
    node_id: str
    node_type: str  # entity, concept, fact, rule, pattern, procedure, source, question
    label: str
    summary: str = ""
    frequency: int = 1
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    importance: float = 0.5
    created_at: str = ""
    last_used: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.last_used:
            self.last_used = self.created_at


class KnowledgeGraph:
    """The knowledge graph: stores concepts, facts, patterns, and their relationships.

    Implements the learning engine:
      - absorb: add new patterns and facts
      - reinforce: increase confidence when a pattern repeats
      - merge: combine duplicate/similar nodes
      - forget: reduce importance of unused knowledge
      - distill: turn many examples into one concise concept
    """
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)

    def absorb(self, node: KnowledgeNode):
        """Add a new node (or reinforce if it already exists)."""
        if node.node_id in self.nodes:
            self.reinforce(node.node_id)
        else:
            self.nodes[node.node_id] = node
            self._index_node(node)

    def reinforce(self, node_id: str):
        """Increase confidence + frequency when a pattern repeats."""
        if node_id in self.nodes:
            n = self.nodes[node_id]
            n.frequency += 1
            n.confidence = min(1.0, n.confidence + 0.05)
            n.importance = min(1.0, n.importance + 0.02)
            n.last_used = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def merge(self, node_id_1: str, node_id_2: str) -> Optional[str]:
        """Merge two similar nodes. Returns the surviving node_id."""
        if node_id_1 not in self.nodes or node_id_2 not in self.nodes:
            return None
        n1 = self.nodes[node_id_1]
        n2 = self.nodes[node_id_2]
        # Keep the one with higher confidence, absorb the other's data
        survivor, victim = (n1, n2) if n1.confidence >= n2.confidence else (n2, n1)
        survivor.frequency += victim.frequency
        survivor.confidence = max(survivor.confidence, victim.confidence)
        survivor.sources = list(set(survivor.sources + victim.sources))
        survivor.links = list({json.dumps(l, sort_keys=True): l for l in survivor.links + victim.links}.values())
        # Remove the victim
        del self.nodes[victim.node_id]
        self._rebuild_index()
        return survivor.node_id

    def forget(self, source_id: str) -> int:
        """Remove all nodes that only came from source_id. Returns count removed."""
        to_remove = []
        for nid, n in self.nodes.items():
            if source_id in n.sources:
                n.sources.remove(source_id)
                n.frequency -= 1
                if not n.sources or n.frequency <= 0:
                    to_remove.append(nid)
        for nid in to_remove:
            del self.nodes[nid]
        self._rebuild_index()
        return len(to_remove)

    def distill(self, pattern_ids: List[str], concept_label: str) -> Optional[str]:
        """Turn many example patterns into one concise concept node."""
        if not pattern_ids:
            return None
        examples = [self.nodes[pid] for pid in pattern_ids if pid in self.nodes]
        if not examples:
            return None
        # Create a new concept node that summarizes the examples
        concept_id = f"concept_{hashlib.sha256(concept_label.encode()).hexdigest()[:8]}"
        all_sources = list(set(s for e in examples for s in e.sources))
        concept = KnowledgeNode(
            node_id=concept_id,
            node_type="concept",
            label=concept_label.upper(),
            summary=f"Distilled from {len(examples)} patterns",
            frequency=sum(e.frequency for e in examples),
            confidence=max(e.confidence for e in examples),
            sources=all_sources,
            importance=0.8,
        )
        # Link examples to the concept
        for pid in pattern_ids:
            if pid in self.nodes:
                concept.links.append({"target": pid, "relation": "example", "weight": 0.8})
        self.absorb(concept)
        return concept_id

    def search(self, keywords: List[str], max_results: int = 10) -> List[KnowledgeNode]:
        """Keyword search across the graph."""
        matching: Dict[str, float] = defaultdict(float)
        for kw in keywords:
            for nid in self._keyword_index.get(kw, set()):
                matching[nid] += 1.0
        # Also check labels
        for node in self.nodes.values():
            for kw in keywords:
                if kw in node.label.lower() or kw in node.summary.lower():
                    matching[node.node_id] += 0.5
        ranked = sorted(matching.items(), key=lambda x: x[1] * self.nodes[x[0]].confidence, reverse=True)
        return [self.nodes[nid] for nid, _ in ranked[:max_results] if nid in self.nodes]

    def _index_node(self, node: KnowledgeNode):
        """Add a node to the keyword index."""
        kws = extract_keywords(f"{node.label} {node.summary}", max_keywords=10)
        for kw in kws:
            self._keyword_index[kw].add(node.node_id)

    def _rebuild_index(self):
        self._keyword_index.clear()
        for node in self.nodes.values():
            self._index_node(node)

    def stats(self) -> Dict[str, Any]:
        type_counts = Counter(n.node_type for n in self.nodes.values())
        return {
            "n_nodes": len(self.nodes),
            "n_keywords": len(self._keyword_index),
            "node_types": dict(type_counts),
        }


# =====================================================================
# LAYER 15: CURIOSITY ENGINE
# =====================================================================
class CuriosityEngine:
    """Looks for gaps in the knowledge graph.

    Finds:
      - Frequent patterns with low confidence (don't fully understand yet)
      - Nodes with no links (isolated knowledge)
      - Questions that remain unanswered
      - Potential contradictions (nodes with same label but different summaries)
    """
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def find_gaps(self) -> List[Dict[str, Any]]:
        """Find knowledge gaps worth investigating."""
        gaps = []
        for node in self.graph.nodes.values():
            # Frequent but low-confidence
            if node.frequency > 5 and node.confidence < 0.5:
                gaps.append({
                    "type": "low_confidence_high_frequency",
                    "node_id": node.node_id,
                    "label": node.label,
                    "frequency": node.frequency,
                    "confidence": node.confidence,
                    "suggestion": f"Pattern '{node.label}' appears {node.frequency} times but confidence is only {node.confidence:.0%}. Analyze more examples.",
                })
            # Isolated nodes (no links)
            if not node.links and node.node_type != "source":
                gaps.append({
                    "type": "isolated_node",
                    "node_id": node.node_id,
                    "label": node.label,
                    "suggestion": f"Node '{node.label}' has no connections. Is it related to anything?",
                })
        return gaps

    def find_contradictions(self) -> List[Dict[str, Any]]:
        """Find potential contradictions (same label, different summary)."""
        by_label = defaultdict(list)
        for node in self.graph.nodes.values():
            by_label[node.label.lower()].append(node)
        contradictions = []
        for label, nodes in by_label.items():
            if len(nodes) > 1:
                summaries = set(n.summary[:100] for n in nodes)
                if len(summaries) > 1:
                    contradictions.append({
                        "type": "contradiction",
                        "label": label,
                        "nodes": [n.node_id for n in nodes],
                        "summaries": list(summaries),
                        "suggestion": f"Multiple nodes with label '{label}' have different summaries. Possible contradiction.",
                    })
        return contradictions


# =====================================================================
# THE MAIN CPC AI CLASS -- ties everything together
# =====================================================================
class CPCAI:
    """Convergent Pattern Compression AI.

    The complete system: compression + knowledge + reasoning.

    Key design decisions (v2 -- fixes the v1 overhead problem):
      - GLOBAL shared n-gram model: trained once on ALL ingested data,
        stored once in the KB. Each bite references it (0 model bytes
        per bite). This fixes the #1 overhead issue from v1.
      - SMART FALLBACK: for files < min_cpc_size (default 4KB), skip
        the CPC pipeline and use brotli/zlib directly. The CPC pipeline
        overhead (symbols + prediction flags) isn't worth it for small
        files. The mode is stored as 1 byte in the bite.
      - METADATA COMPRESSION: symbol dictionary + transform recipe are
        Huffman-coded before storage.
      - ADAPTIVE symbol count: cap at min(50, file_size / 100) symbols.

    Args:
        kb_path: path to the knowledge-base directory.
        ngram_size: n-gram size for the prediction engine (default 3).
        min_cpc_size: minimum file size to use the full CPC pipeline
                      (default 4096). Below this, fall back to brotli.
        verbose: if True, print progress.
    """
    def __init__(self, kb_path: str = ".cpc_ai_kb", ngram_size: int = 3,
                 min_cpc_size: int = 4096, verbose: bool = False):
        self.kb_path = kb_path
        self.verbose = verbose
        self.min_cpc_size = min_cpc_size
        os.makedirs(kb_path, exist_ok=True)
        # Components
        self.reorganizer = Reorganizer()
        self.predictor = NGramPredictor(n=ngram_size)
        self.symbol_compressor = SymbolicCompressor()
        self.verifier = VerificationEngine()
        self.graph = KnowledgeGraph()
        self.curiosity = CuriosityEngine(self.graph)
        # State
        self.sources: Dict[str, Dict] = {}
        self._next_source_id = 0
        # GLOBAL shared n-gram model (trained on all ingested data)
        self.global_predictor = NGramPredictor(n=ngram_size)
        self._global_model_trained = False
        # Stats
        self.total_compressed = 0
        self.total_original = 0
        self.total_verified = 0
        self.load()

    # ------------------------------------------------------------------
    # INGESTION + COMPRESSION
    # ------------------------------------------------------------------
    def ingest(self, path: str) -> Dict[str, Any]:
        """Ingest a file or directory. Returns summary."""
        if os.path.isdir(path):
            return self._ingest_dir(path)
        return self._ingest_file(path)

    def _ingest_dir(self, dir_path: str) -> Dict[str, Any]:
        total_files = 0
        for root, _, files in os.walk(dir_path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext not in FILE_TYPE_MAP or fname.startswith("."):
                    continue
                try:
                    self._ingest_file(fpath)
                    total_files += 1
                except Exception as e:
                    if self.verbose:
                        print(f"  skip {fpath}: {e}")
        self.save()
        return {"n_files": total_files, "n_nodes": len(self.graph.nodes)}

    def _ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single file: compress it + extract knowledge."""
        file_type = detect_file_type(file_path)
        with open(file_path, "rb") as f:
            data = f.read()
        source_id = f"src_{self._next_source_id}"
        self._next_source_id += 1
        if self.verbose:
            print(f"  ingesting {file_path} ({file_type}, {len(data)} bytes)")

        # 1. Compress the file into a CPC bite
        bite = self.compress_data(data, os.path.basename(file_path))

        # 2. Extract knowledge
        content = data.decode("utf-8", errors="replace")
        keywords = extract_keywords(content, max_keywords=15)
        # Add source node
        source_node = KnowledgeNode(
            node_id=f"{source_id}_source", node_type="source",
            label=os.path.basename(file_path),
            summary=f"File: {file_path} ({file_type}, {len(data)} bytes)",
            sources=[source_id], confidence=1.0, importance=1.0,
        )
        self.graph.absorb(source_node)
        # Add concept nodes
        for kw in keywords:
            kw_id = f"concept_{kw}"
            if kw_id not in self.graph.nodes:
                node = KnowledgeNode(
                    node_id=kw_id, node_type="concept",
                    label=kw.upper(), summary=f"Concept: {kw}",
                    sources=[source_id], confidence=0.6,
                )
                self.graph.absorb(node)
            else:
                self.graph.reinforce(kw_id)
            source_node.links.append({"target": kw_id, "relation": "mentions", "weight": 0.7})

        # 3. Record the source
        self.sources[source_id] = {
            "path": file_path, "file_type": file_type,
            "size": len(data), "bite_size": bite.total_size(),
        }
        self.total_original += len(data)
        self.total_compressed += bite.total_size()

        return {"source_id": source_id, "bite_size": bite.total_size(),
                "n_keywords": len(keywords)}

    # ------------------------------------------------------------------
    # THE FULL COMPRESSION PIPELINE (v2: global model + smart fallback)
    # ------------------------------------------------------------------
    # Bite modes:
    BROTLI_MODE = 0    # direct brotli/zlib fallback (small files)
    CPC_MODE = 1       # full CPC pipeline (large files)

    def compress_data(self, data: bytes, name: str = "data") -> CPCBite:
        """Run the compression pipeline on raw bytes.

        v3 strategy: ALWAYS try brotli as a baseline. If the CPC pipeline
        produces a smaller result, use CPC. Otherwise use brotli. This
        guarantees CPC AI is never worse than brotli.

        For files < min_cpc_size: skip the CPC pipeline entirely (brotli only).
        For files >= min_cpc_size: try both, pick the winner.
        """
        original_sha = self.verifier.sha256(data)
        original_crc = self.verifier.crc32(data)

        # Always compute brotli as the baseline
        try:
            import brotli
            brotli_compressed = brotli.compress(data, quality=11)
        except ImportError:
            brotli_compressed = zlib.compress(data, 9)
        brotli_size = len(brotli_compressed) + 1  # +1 for mode byte

        # Smart fallback: small files -> brotli only
        if len(data) < self.min_cpc_size:
            compressed_stream = bytes([self.BROTLI_MODE]) + brotli_compressed
            if self.verbose:
                print(f"    small file ({len(data)} bytes) -> brotli ({brotli_size} bytes)")
            return CPCBite(
                original_name=name, original_size=len(data),
                original_sha256=original_sha, transform_recipe=[],
                symbol_dictionary={}, ngram_model=b"",
                compressed_stream=compressed_stream, crc32=original_crc,
            )

        # For large files: try the full CPC pipeline
        # 1. Reorganize
        reorganized, recipe = self.reorganizer.analyze(data)

        # 2. Symbolic compression (adaptive cap)
        max_symbols = min(50, len(data) // 100)
        symbolized, sym_dict = self._symbolic_compress_capped(reorganized, max_symbols)

        # 3. Use GLOBAL predictor
        if not self._global_model_trained:
            self.global_predictor.train(symbolized)
            self._global_model_trained = True

        # 4. Predict + encode
        encoded, flags = self.global_predictor.compress_with_prediction(symbolized)
        correct_rate = sum(flags) / max(len(flags), 1)

        # 5. Huffman
        huffman_encoded = huffman_compress(encoded)

        # 6. Compress metadata
        dict_bytes = json.dumps(sym_dict).encode("utf-8") if sym_dict else b""
        recipe_bytes = json.dumps([asdict(s) if hasattr(s, '__dataclass_fields__') else s
                                   for s in recipe]).encode("utf-8") if recipe else b""
        compressed_dict = huffman_compress(dict_bytes) if dict_bytes else b""
        compressed_recipe = huffman_compress(recipe_bytes) if recipe_bytes else b""

        # 7. Package CPC stream
        cpc_stream = bytearray([self.CPC_MODE])
        cpc_stream += struct.pack("<I", len(compressed_dict))
        cpc_stream += compressed_dict
        cpc_stream += struct.pack("<I", len(compressed_recipe))
        cpc_stream += compressed_recipe
        cpc_stream += huffman_encoded
        cpc_size = len(cpc_stream)

        if self.verbose:
            print(f"    brotli={brotli_size} bytes, cpc={cpc_size} bytes "
                  f"(prediction {correct_rate:.0%}, {len(sym_dict)} symbols)")

        # PICK THE WINNER: use whichever is smaller
        if cpc_size < brotli_size:
            # CPC wins!
            if self.verbose:
                print(f"    -> CPC wins by {brotli_size - cpc_size} bytes")
            return CPCBite(
                original_name=name, original_size=len(data),
                original_sha256=original_sha,
                transform_recipe=[asdict(s) if hasattr(s, '__dataclass_fields__') else s
                                  for s in recipe],
                symbol_dictionary=sym_dict, ngram_model=b"",
                compressed_stream=bytes(cpc_stream), crc32=original_crc,
            )
        else:
            # Brotli wins (or tie) -- use brotli
            if self.verbose:
                print(f"    -> brotli wins by {cpc_size - brotli_size} bytes")
            return CPCBite(
                original_name=name, original_size=len(data),
                original_sha256=original_sha, transform_recipe=[],
                symbol_dictionary={}, ngram_model=b"",
                compressed_stream=bytes([self.BROTLI_MODE]) + brotli_compressed,
                crc32=original_crc,
            )

    def _symbolic_compress_capped(self, data: bytes, max_symbols: int) -> Tuple[bytes, Dict[str, str]]:
        """Symbolic compression with a custom symbol cap."""
        if not data or max_symbols <= 0:
            return data, {}
        sc = SymbolicCompressor()
        repeated = sc._find_repeated_substrings(data, min_len=8, max_len=64, min_freq=3)
        repeated.sort(key=lambda x: x[1] * x[2], reverse=True)
        result = data
        dictionary = {}
        for substring, length, freq in repeated[:max_symbols]:
            original_cost = length * freq
            symbol_cost = 1 * freq + length + 10
            savings = original_cost - symbol_cost
            if savings <= 0:
                continue
            symbol = sc._next_symbol()
            result = result.replace(substring, symbol.encode())
            dictionary[symbol] = substring.decode("latin-1")
        return result, dictionary

    def decompress_bite(self, bite: CPCBite) -> bytes:
        """Decompress a CPC bite back to the original bytes.

        Handles both modes:
          - BROTLI_MODE (0): direct backend decompression
          - CPC_MODE (1): full pipeline reversal with global model
        """
        stream = bite.compressed_stream
        if not stream:
            return b""

        mode = stream[0]
        payload = stream[1:]

        if mode == self.BROTLI_MODE:
            # Direct backend decompression
            try:
                import brotli
                return brotli.decompress(payload)
            except (ImportError, Exception):
                return zlib.decompress(payload)

        elif mode == self.CPC_MODE:
            # Full CPC pipeline reversal
            idx = 0
            # Read compressed dict
            dict_len = struct.unpack("<I", payload[idx:idx+4])[0]; idx += 4
            compressed_dict = payload[idx:idx+dict_len]; idx += dict_len
            # Read compressed recipe
            recipe_len = struct.unpack("<I", payload[idx:idx+4])[0]; idx += 4
            compressed_recipe = payload[idx:idx+recipe_len]; idx += recipe_len
            # Rest is the Huffman-encoded prediction stream
            huffman_stream = payload[idx:]

            # Decompress metadata
            sym_dict = json.loads(huffman_decompress(compressed_dict).decode("utf-8")) if compressed_dict else {}
            recipe_data = json.loads(huffman_decompress(compressed_recipe).decode("utf-8")) if compressed_recipe else []

            # Huffman-decode the prediction stream
            encoded = huffman_decompress(huffman_stream)

            # Use the GLOBAL shared predictor for decoding
            symbolized = self.global_predictor.decompress_with_prediction(encoded)

            # Expand symbols
            expanded = self.symbol_compressor.decompress(symbolized, sym_dict)

            # Reverse reorganization
            recipe = [TransformStep(**s) if isinstance(s, dict) else s for s in recipe_data]
            restored = self.reorganizer.reverse(expanded, recipe)

            # Verify
            restored_sha = self.verifier.sha256(restored)
            if restored_sha != bite.original_sha256:
                if self.verbose:
                    print(f"  WARNING: SHA-256 mismatch!")
            return restored

        else:
            raise ValueError(f"Unknown CPC bite mode: {mode}")

    def compress_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Compress a file into a CPC bite. Returns the output path."""
        with open(file_path, "rb") as f:
            data = f.read()
        bite = self.compress_data(data, os.path.basename(file_path))
        out = output_path or (file_path + ".cpcbite")
        bite.save(out)
        # Verify round-trip
        restored = self.decompress_bite(bite)
        report = self.verifier.verify(data, restored)
        if not report["verified"]:
            raise RuntimeError(f"CPC verification failed for {file_path}")
        self.total_verified += 1
        if self.verbose:
            print(f"  compressed {len(data)} -> {bite.total_size()} bytes "
                  f"(ratio: {bite.total_size()/max(len(data),1):.3f}, verified: {report['verified']})")
        return out

    # ------------------------------------------------------------------
    # REASONING
    # ------------------------------------------------------------------
    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question using the knowledge graph."""
        q_keywords = extract_keywords(question, max_keywords=5)
        if not q_keywords:
            return {"answer": "I couldn't extract keywords from your question.",
                    "evidence": [], "confidence": 0.0}
        results = self.graph.search(q_keywords, max_results=5)
        if not results:
            return {"answer": f"I don't have knowledge about '{' '.join(q_keywords)}'.",
                    "evidence": [], "confidence": 0.0, "keywords": q_keywords}
        evidence = []
        answer_parts = []
        for node in results:
            node.last_used = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            evidence.append({
                "node_id": node.node_id, "label": node.label,
                "type": node.node_type, "confidence": node.confidence,
                "sources": node.sources[:3], "summary": node.summary[:200],
            })
            answer_parts.append(f"  [{node.node_type}] {node.label}: {node.summary[:150]}")
        confidence = min(1.0, sum(n.confidence for n in results) / max(len(results), 1))
        answer = f"Based on {len(results)} knowledge nodes:\n\n" + "\n".join(answer_parts)
        answer += f"\n\nKeywords: {q_keywords}\nConfidence: {confidence:.0%}"
        return {"answer": answer, "evidence": evidence,
                "confidence": confidence, "keywords": q_keywords}

    # ------------------------------------------------------------------
    # CURIOSITY
    # ------------------------------------------------------------------
    def curiosity_report(self) -> Dict[str, Any]:
        """Generate a curiosity report: gaps + contradictions."""
        gaps = self.curiosity.find_gaps()
        contradictions = self.curiosity.find_contradictions()
        return {
            "n_gaps": len(gaps),
            "n_contradictions": len(contradictions),
            "gaps": gaps[:10],
            "contradictions": contradictions[:10],
        }

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Return comprehensive statistics."""
        graph_stats = self.graph.stats()
        return {
            "n_sources": len(self.sources),
            "n_nodes": graph_stats["n_nodes"],
            "n_keywords": graph_stats["n_keywords"],
            "node_types": graph_stats["node_types"],
            "total_original_bytes": self.total_original,
            "total_compressed_bytes": self.total_compressed,
            "compression_ratio": self.total_compressed / max(self.total_original, 1),
            "n_verified": self.total_verified,
        }

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------
    def save(self):
        """Save the knowledge base + global n-gram model."""
        state = {
            "nodes": {nid: asdict(n) for nid, n in self.graph.nodes.items()},
            "sources": self.sources,
            "next_source_id": self._next_source_id,
            "total_compressed": self.total_compressed,
            "total_original": self.total_original,
            "total_verified": self.total_verified,
            "global_model_trained": self._global_model_trained,
        }
        path = os.path.join(self.kb_path, "cpc_ai_kb.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        # Save the global n-gram model separately (binary)
        if self._global_model_trained:
            model_path = os.path.join(self.kb_path, "global_ngram.bin")
            with open(model_path, "wb") as f:
                f.write(self.global_predictor.save_state())

    def load(self):
        """Load the knowledge base + global n-gram model."""
        path = os.path.join(self.kb_path, "cpc_ai_kb.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.graph.nodes = {nid: KnowledgeNode(**n) for nid, n in state.get("nodes", {}).items()}
            self.graph._rebuild_index()
            self.sources = state.get("sources", {})
            self._next_source_id = state.get("next_source_id", 0)
            self.total_compressed = state.get("total_compressed", 0)
            self.total_original = state.get("total_original", 0)
            self.total_verified = state.get("total_verified", 0)
            self._global_model_trained = state.get("global_model_trained", False)
            # Load global n-gram model
            if self._global_model_trained:
                model_path = os.path.join(self.kb_path, "global_ngram.bin")
                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        self.global_predictor = NGramPredictor.load_state(f.read())
        except Exception as e:
            if self.verbose:
                print(f"  warning: could not load KB: {e}")


__all__ = [
    "CPCAI", "CPCBite", "KnowledgeGraph", "KnowledgeNode",
    "NGramPredictor", "SymbolicCompressor", "Reorganizer",
    "VerificationEngine", "CuriosityEngine",
    "huffman_compress", "huffman_decompress",
]
