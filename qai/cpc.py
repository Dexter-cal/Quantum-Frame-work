"""
qai.cpc -- Convergent Pattern Compression.

A file-native learning system. You show it files; it studies their
structure and meaning, extracts reusable patterns, compresses them into
a knowledge graph, and uses that graph to answer questions.

This is NOT a traditional compressor. It's a "semantic compressor" --
it turns files into compressed, connected, reusable understanding.

Key operations:
  cpc = qai.CPC("my_knowledge_base/")    # init with a storage path
  cpc.ingest("path/to/file_or_dir")      # show it files
  cpc.ask("How is authentication handled?")  # query the knowledge graph
  cpc.search("error handling")           # keyword search
  cpc.graph("authentication")            # show graph around a topic
  cpc.stats()                            # knowledge base statistics
  cpc.forget("source_id")                # remove a source + derived knowledge
  cpc.save() / cpc.load()                # persist / restore

Supported file types (text-based):
  .py .js .ts .java .c .cpp .go .rs .rb .php  (code)
  .txt .md .rst .log                          (text)
  .json .yaml .yml .xml .csv .tsv             (structured)
  .html .css .sql                             (misc text)

For each file, CPC:
  1. Detects the type
  2. Splits into chunks (by function, paragraph, record, or line)
  3. Extracts patterns (exact repeats, near-repeats, templates)
  4. Builds knowledge-graph nodes + edges
  5. Stores compressed representations

Compression modes:
  "lossless"       -- store exact chunks + indexes (full reconstruction)
  "smart_lossless" -- deduplicate repeated chunks, keep references
  "semantic"       -- store concepts + summaries only (no reconstruction)

CLI:
  python -m qai cpc ingest <path> --kb <kb_dir>
  python -m qai cpc ask "question" --kb <kb_dir>
  python -m qai cpc search "query" --kb <kb_dir>
  python -m qai cpc graph "topic" --kb <kb_dir>
  python -m qai cpc stats --kb <kb_dir>
  python -m qai cpc forget <source_id> --kb <kb_dir>
"""
from __future__ import annotations
import os, sys, re, json, hashlib, time, math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


# =====================================================================
# 1. Data model -- nodes, edges, chunks, patterns
# =====================================================================
@dataclass
class Chunk:
    """A unit of content extracted from a file."""
    chunk_id: str
    source_id: str
    file_type: str
    content: str
    chunk_type: str  # "function", "paragraph", "record", "line", "section"
    line_start: int = 0
    line_end: int = 0
    content_hash: str = ""  # SHA-256 of content (for dedup)

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]


@dataclass
class Pattern:
    """A reusable pattern detected across files."""
    pattern_id: str
    label: str
    pattern_type: str  # "exact_repeat", "near_repeat", "template", "concept"
    compressed_form: str
    frequency: int = 1
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)  # source_ids
    links: List[str] = field(default_factory=list)    # other pattern_ids


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    node_id: str
    node_type: str  # "entity", "concept", "fact", "rule", "pattern", "procedure", "source"
    label: str
    summary: str = ""
    content: str = ""
    frequency: int = 1
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)  # [{"target": id, "relation": "uses", "weight": 0.9}]
    importance: float = 0.5
    created_at: str = ""
    last_used: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.last_used:
            self.last_used = self.created_at


# =====================================================================
# 2. File type detection
# =====================================================================
FILE_TYPE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c_header",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".txt": "text", ".md": "markdown", ".rst": "rst", ".log": "log",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".xml": "xml",
    ".csv": "csv", ".tsv": "tsv", ".html": "html", ".css": "css",
    ".sql": "sql", ".sh": "shell", ".toml": "toml", ".ini": "ini",
    ".cfg": "ini", ".conf": "ini",
}

CODE_TYPES = {"python", "javascript", "typescript", "java", "c", "cpp",
              "c_header", "go", "rust", "ruby", "php", "shell", "sql"}


def detect_file_type(filepath: str) -> str:
    """Detect the file type from the extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return FILE_TYPE_MAP.get(ext, "text")


# =====================================================================
# 3. Chunker -- splits files into chunks
# =====================================================================
def chunk_file(content: str, file_type: str, source_id: str) -> List[Chunk]:
    """Split file content into chunks based on the file type."""
    chunks = []
    if file_type in CODE_TYPES:
        chunks = _chunk_code(content, file_type, source_id)
    elif file_type == "json":
        chunks = _chunk_json(content, source_id)
    elif file_type in ("csv", "tsv"):
        chunks = _chunk_csv(content, file_type, source_id)
    elif file_type == "markdown":
        chunks = _chunk_markdown(content, source_id)
    else:
        chunks = _chunk_text(content, source_id)
    return chunks


def _chunk_code(content: str, file_type: str, source_id: str) -> List[Chunk]:
    """Split code by function/class definitions."""
    chunks = []
    lines = content.split("\n")
    # Pattern: lines starting with def/class/function/func (at any indentation level
    # for Python, or function keyword for JS)
    if file_type == "python":
        func_pattern = re.compile(r"^\s*(def |class |async def )")
    else:
        func_pattern = re.compile(r"^\s*(function |class |def |func |pub fn |fn )")

    current_start = 0
    current_lines = []
    current_type = "header"
    chunk_idx = 0

    for i, line in enumerate(lines):
        if func_pattern.match(line) and current_lines:
            # Flush previous chunk
            chunk_content = "\n".join(current_lines)
            chunks.append(Chunk(
                chunk_id=f"{source_id}_c{chunk_idx}",
                source_id=source_id, file_type=file_type,
                content=chunk_content, chunk_type=current_type,
                line_start=current_start + 1, line_end=i,
            ))
            chunk_idx += 1
            current_lines = [line]
            current_start = i
            current_type = "function" if "def " in line or "function" in line else "class"
        else:
            current_lines.append(line)

    # Flush last chunk
    if current_lines:
        chunk_content = "\n".join(current_lines)
        chunks.append(Chunk(
            chunk_id=f"{source_id}_c{chunk_idx}",
            source_id=source_id, file_type=file_type,
            content=chunk_content, chunk_type=current_type,
            line_start=current_start + 1, line_end=len(lines),
        ))
    return chunks


def _chunk_json(content: str, source_id: str) -> List[Chunk]:
    """Split JSON by top-level keys (best effort)."""
    chunks = []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return _chunk_text(content, source_id)
    if isinstance(data, dict):
        for i, (key, value) in enumerate(data.items()):
            chunks.append(Chunk(
                chunk_id=f"{source_id}_c{i}",
                source_id=source_id, file_type="json",
                content=json.dumps({key: value}, indent=2),
                chunk_type="record", line_start=0, line_end=0,
            ))
    elif isinstance(data, list):
        for i, item in enumerate(data[:100]):  # cap at 100 records
            chunks.append(Chunk(
                chunk_id=f"{source_id}_c{i}",
                source_id=source_id, file_type="json",
                content=json.dumps(item, indent=2),
                chunk_type="record", line_start=0, line_end=0,
            ))
    return chunks


def _chunk_csv(content: str, file_type: str, source_id: str) -> List[Chunk]:
    """Split CSV by rows."""
    lines = content.strip().split("\n")
    if not lines:
        return []
    delimiter = "\t" if file_type == "tsv" else ","
    header = lines[0]
    chunks = [Chunk(
        chunk_id=f"{source_id}_c0",
        source_id=source_id, file_type=file_type,
        content=header, chunk_type="header", line_start=1, line_end=1,
    )]
    for i, line in enumerate(lines[1:101], start=1):  # cap at 100 rows
        chunks.append(Chunk(
            chunk_id=f"{source_id}_c{i}",
            source_id=source_id, file_type=file_type,
            content=line, chunk_type="record", line_start=i+1, line_end=i+1,
        ))
    return chunks


def _chunk_markdown(content: str, source_id: str) -> List[Chunk]:
    """Split markdown by headers."""
    lines = content.split("\n")
    header_pattern = re.compile(r"^(#{1,6}\s+)")
    chunks = []
    current_start = 0
    current_lines = []
    chunk_idx = 0
    current_header = "intro"

    for i, line in enumerate(lines):
        if header_pattern.match(line) and current_lines:
            chunks.append(Chunk(
                chunk_id=f"{source_id}_c{chunk_idx}",
                source_id=source_id, file_type="markdown",
                content="\n".join(current_lines),
                chunk_type="section",
                line_start=current_start + 1, line_end=i,
            ))
            chunk_idx += 1
            current_header = line.strip()
            current_lines = [line]
            current_start = i
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append(Chunk(
            chunk_id=f"{source_id}_c{chunk_idx}",
            source_id=source_id, file_type="markdown",
            content="\n".join(current_lines),
            chunk_type="section",
            line_start=current_start + 1, line_end=len(lines),
        ))
    return chunks


def _chunk_text(content: str, source_id: str) -> List[Chunk]:
    """Split text by paragraphs (double newlines) or by ~500-char blocks."""
    chunks = []
    # Try paragraph splitting first
    paragraphs = re.split(r"\n\s*\n", content)
    if len(paragraphs) > 1:
        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append(Chunk(
                    chunk_id=f"{source_id}_c{i}",
                    source_id=source_id, file_type="text",
                    content=para.strip(), chunk_type="paragraph",
                    line_start=0, line_end=0,
                ))
    else:
        # Fallback: fixed-size blocks
        block_size = 500
        for i in range(0, len(content), block_size):
            chunks.append(Chunk(
                chunk_id=f"{source_id}_c{len(chunks)}",
                source_id=source_id, file_type="text",
                content=content[i:i+block_size], chunk_type="block",
                line_start=0, line_end=0,
            ))
    return chunks


# =====================================================================
# 4. Pattern detector -- finds repeats, templates, concepts
# =====================================================================
def detect_patterns(chunks: List[Chunk]) -> List[Pattern]:
    """Detect patterns across chunks."""
    patterns = []

    # 1. Exact repeats: chunks with the same content_hash
    hash_groups = defaultdict(list)
    for chunk in chunks:
        hash_groups[chunk.content_hash].append(chunk)
    for h, group in hash_groups.items():
        if len(group) > 1:
            label = group[0].content[:60].replace("\n", " ").strip()
            patterns.append(Pattern(
                pattern_id=f"pat_exact_{h}",
                label=f"EXACT_REPEAT: {label}...",
                pattern_type="exact_repeat",
                compressed_form=group[0].content,
                frequency=len(group),
                confidence=1.0,
                sources=[c.source_id for c in group],
            ))

    # 2. Near-repeats: chunks with similar line structure
    # Use a simple fingerprint: first line + number of lines
    struct_groups = defaultdict(list)
    for chunk in chunks:
        first_line = chunk.content.split("\n")[0][:80] if chunk.content else ""
        n_lines = chunk.content.count("\n") + 1
        fingerprint = f"{n_lines}L:{first_line}"
        struct_groups[fingerprint].append(chunk)
    for fp, group in struct_groups.items():
        if len(group) > 1 and len(group) <= 20:  # skip overly generic patterns
            label = group[0].content[:60].replace("\n", " ").strip()
            patterns.append(Pattern(
                pattern_id=f"pat_near_{hashlib.sha256(fp.encode()).hexdigest()[:8]}",
                label=f"NEAR_REPEAT: {label}...",
                pattern_type="near_repeat",
                compressed_form=fp,
                frequency=len(group),
                confidence=0.8,
                sources=[c.source_id for c in group],
            ))

    # 3. Templates: code patterns like "def <name>(<args>):"
    template_patterns = defaultdict(list)
    for chunk in chunks:
        if chunk.chunk_type in ("function", "class"):
            # Abstract the function signature
            first_line = chunk.content.split("\n")[0] if chunk.content else ""
            # Replace identifiers with placeholders
            template = re.sub(r"\b[a-zA-Z_]\w*\b", "<ID>", first_line)
            template_patterns[template].append(chunk)
    for template, group in template_patterns.items():
        if len(group) > 1:
            patterns.append(Pattern(
                pattern_id=f"pat_tmpl_{hashlib.sha256(template.encode()).hexdigest()[:8]}",
                label=f"TEMPLATE: {template[:80]}",
                pattern_type="template",
                compressed_form=template,
                frequency=len(group),
                confidence=0.9,
                sources=[c.source_id for c in group],
            ))

    return patterns


# =====================================================================
# 5. Keyword extraction (simple but effective)
# =====================================================================
def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using frequency + IDF-like scoring."""
    # Tokenize: split on non-alphanumeric, keep tokens >= 3 chars
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text.lower())
    if not tokens:
        return []
    # Stop words (common English + code keywords)
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
        "was", "one", "our", "out", "has", "have", "had", "how", "its", "may",
        "them", "than", "this", "that", "with", "from", "they", "will", "would",
        "there", "their", "what", "about", "which", "when", "make", "can",
        "like", "into", "def", "class", "return", "import", "from", "self",
        "none", "true", "false", "print", "self", "args", "kwargs",
    }
    filtered = [t for t in tokens if t not in stop_words and len(t) >= 3]
    if not filtered:
        return []
    counts = Counter(filtered)
    # Return top keywords by frequency
    return [kw for kw, _ in counts.most_common(max_keywords)]


# =====================================================================
# 6. CPC -- the main system
# =====================================================================
class CPC:
    """Convergent Pattern Compression -- a file-native learning system.

    Args:
        kb_path: path to the knowledge-base directory (stores the graph).
        mode: compression mode ("lossless" | "smart_lossless" | "semantic").
        verbose: if True, print progress.
    """
    def __init__(self, kb_path: str = ".cpc_kb", mode: str = "smart_lossless",
                 verbose: bool = False):
        self.kb_path = kb_path
        self.mode = mode
        self.verbose = verbose
        os.makedirs(kb_path, exist_ok=True)

        # In-memory state
        self.chunks: Dict[str, Chunk] = {}  # chunk_id -> Chunk
        self.patterns: Dict[str, Pattern] = {}  # pattern_id -> Pattern
        self.nodes: Dict[str, GraphNode] = {}  # node_id -> GraphNode
        self.sources: Dict[str, Dict] = {}  # source_id -> {path, file_type, n_chunks, size}
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> {node_id, ...}
        self._next_source_id = 0

        # Try to load existing state
        self.load()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, path: str) -> Dict[str, Any]:
        """Ingest a file or directory into the knowledge base.

        Args:
            path: path to a file or directory.

        Returns:
            Summary dict with n_files, n_chunks, n_patterns, n_nodes.
        """
        if os.path.isdir(path):
            return self._ingest_dir(path)
        else:
            return self._ingest_file(path)

    def _ingest_dir(self, dir_path: str) -> Dict[str, Any]:
        """Recursively ingest all files in a directory."""
        total_files = 0
        total_chunks = 0
        total_patterns = 0
        total_nodes = 0
        for root, _, files in os.walk(dir_path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                # Skip binary files and hidden files
                ext = os.path.splitext(fname)[1].lower()
                if ext not in FILE_TYPE_MAP or fname.startswith("."):
                    continue
                try:
                    summary = self._ingest_file(fpath)
                    total_files += 1
                    total_chunks += summary["n_chunks"]
                    total_patterns += summary["n_patterns"]
                    total_nodes += summary["n_nodes"]
                except Exception as e:
                    if self.verbose:
                        print(f"  skip {fpath}: {e}")
        self._rebuild_keyword_index()
        self.save()
        return {
            "n_files": total_files, "n_chunks": total_chunks,
            "n_patterns": total_patterns, "n_nodes": total_nodes,
        }

    def _ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single file."""
        file_type = detect_file_type(file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        source_id = f"src_{self._next_source_id}"
        self._next_source_id += 1

        if self.verbose:
            print(f"  ingesting {file_path} ({file_type}, {len(content)} bytes)")

        # 1. Chunk the file
        chunks = chunk_file(content, file_type, source_id)
        # 2. Detect patterns
        patterns = detect_patterns(chunks)
        # 3. Build graph nodes
        nodes = self._build_nodes(source_id, file_path, file_type, chunks, patterns)

        # Store
        for c in chunks:
            # In semantic mode, don't store the full content -- just metadata
            if self.mode == "semantic":
                c.content = f"[semantic: {len(c.content)} chars, type={c.chunk_type}]"
            self.chunks[c.chunk_id] = c
        for p in patterns:
            self.patterns[p.pattern_id] = p
        for n in nodes:
            self.nodes[n.node_id] = n
        self.sources[source_id] = {
            "path": file_path,
            "file_type": file_type,
            "n_chunks": len(chunks),
            "size": len(content),
        }

        return {
            "source_id": source_id,
            "n_chunks": len(chunks),
            "n_patterns": len(patterns),
            "n_nodes": len(nodes),
        }

    def _build_nodes(self, source_id: str, file_path: str, file_type: str,
                     chunks: List[Chunk], patterns: List[Pattern]) -> List[GraphNode]:
        """Build knowledge-graph nodes from chunks + patterns."""
        nodes = []
        # 1. Source node
        source_node = GraphNode(
            node_id=f"{source_id}_source",
            node_type="source",
            label=os.path.basename(file_path),
            summary=f"File: {file_path} ({file_type})",
            content=f"[source: {file_path}]",
            sources=[source_id],
            importance=1.0,
            confidence=1.0,
        )
        nodes.append(source_node)

        # 2. Pattern nodes
        for p in patterns:
            node = GraphNode(
                node_id=p.pattern_id,
                node_type="pattern",
                label=p.label,
                summary=p.compressed_form[:200],
                content=p.compressed_form,
                frequency=p.frequency,
                confidence=p.confidence,
                sources=p.sources,
                importance=min(1.0, p.frequency / 10.0),
            )
            # Link to source node
            node.links.append({"target": source_node.node_id, "relation": "found_in", "weight": 1.0})
            nodes.append(node)

        # 3. Concept nodes (from keywords)
        all_text = " ".join(c.content for c in chunks)
        keywords = extract_keywords(all_text, max_keywords=15)
        for kw in keywords:
            kw_node_id = f"concept_{kw}"
            if kw_node_id not in self.nodes:
                node = GraphNode(
                    node_id=kw_node_id,
                    node_type="concept",
                    label=kw.upper(),
                    summary=f"Concept: {kw}",
                    frequency=1,
                    confidence=0.6,
                    sources=[source_id],
                    importance=0.5,
                )
                nodes.append(node)
            # Link source to concept
            for n in nodes:
                if n.node_id == source_node.node_id:
                    n.links.append({"target": kw_node_id, "relation": "mentions", "weight": 0.7})

        return nodes

    def _rebuild_keyword_index(self):
        """Rebuild the keyword -> node_id index."""
        self._keyword_index.clear()
        for node in self.nodes.values():
            keywords = extract_keywords(f"{node.label} {node.summary}", max_keywords=10)
            for kw in keywords:
                self._keyword_index[kw].add(node.node_id)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question using the knowledge graph.

        Args:
            question: a natural-language question.

        Returns:
            Dict with:
              - answer: text answer
              - evidence: list of {node_id, label, confidence, source}
              - confidence: overall confidence (0..1)
        """
        # Extract keywords from the question
        q_keywords = extract_keywords(question, max_keywords=5)
        if not q_keywords:
            return {
                "answer": "I couldn't extract keywords from your question. Try rephrasing.",
                "evidence": [],
                "confidence": 0.0,
            }

        # Find matching nodes
        matching_nodes: Dict[str, float] = defaultdict(float)
        for kw in q_keywords:
            for node_id in self._keyword_index.get(kw, set()):
                matching_nodes[node_id] += 1.0

        # Also check node labels directly
        for node in self.nodes.values():
            for kw in q_keywords:
                if kw in node.label.lower() or kw in node.summary.lower():
                    matching_nodes[node.node_id] += 0.5

        if not matching_nodes:
            return {
                "answer": f"I don't have any knowledge about '{' '.join(q_keywords)}'. "
                          f"Try ingesting more files related to this topic.",
                "evidence": [],
                "confidence": 0.0,
                "keywords_searched": q_keywords,
            }

        # Rank nodes by score * confidence
        ranked = sorted(matching_nodes.items(),
                        key=lambda x: x[1] * self.nodes[x[0]].confidence,
                        reverse=True)

        # Build the answer from the top nodes
        top_nodes = ranked[:5]
        evidence = []
        answer_parts = []
        for node_id, score in top_nodes:
            node = self.nodes[node_id]
            # Update last_used
            node.last_used = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            evidence.append({
                "node_id": node_id,
                "label": node.label,
                "type": node.node_type,
                "confidence": node.confidence,
                "sources": node.sources[:3],  # top 3 sources
                "summary": node.summary[:200],
            })
            answer_parts.append(f"  [{node.node_type}] {node.label}: {node.summary[:150]}")

        overall_confidence = min(1.0, sum(s for _, s in top_nodes) / max(len(q_keywords), 1) * 0.3)

        answer = f"Based on {len(top_nodes)} relevant knowledge nodes:\n\n"
        answer += "\n".join(answer_parts)
        answer += f"\n\nKeywords matched: {q_keywords}"
        answer += f"\nOverall confidence: {overall_confidence:.1%}"

        return {
            "answer": answer,
            "evidence": evidence,
            "confidence": overall_confidence,
            "keywords_searched": q_keywords,
        }

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Keyword search across the knowledge base."""
        q_keywords = extract_keywords(query, max_keywords=5)
        results = []
        seen_nodes = set()
        for kw in q_keywords:
            for node_id in self._keyword_index.get(kw, set()):
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)
                node = self.nodes[node_id]
                results.append({
                    "node_id": node_id,
                    "label": node.label,
                    "type": node.node_type,
                    "confidence": node.confidence,
                    "frequency": node.frequency,
                    "sources": node.sources[:3],
                })
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break
        return results

    def graph(self, topic: str, depth: int = 2) -> Dict[str, Any]:
        """Show the knowledge graph around a topic."""
        # Find nodes matching the topic
        q_keywords = extract_keywords(topic, max_keywords=3)
        start_nodes = set()
        for kw in q_keywords:
            start_nodes.update(self._keyword_index.get(kw, set()))

        # BFS to depth
        visited = set()
        edges = []
        queue = [(nid, 0) for nid in start_nodes]
        while queue:
            node_id, d = queue.pop(0)
            if node_id in visited or d > depth:
                continue
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if not node:
                continue
            for link in node.links:
                target = link["target"]
                edges.append({
                    "from": node_id, "to": target,
                    "relation": link["relation"], "weight": link["weight"],
                })
                if target not in visited:
                    queue.append((target, d + 1))

        nodes_info = []
        for nid in visited:
            n = self.nodes.get(nid)
            if n:
                nodes_info.append({
                    "node_id": nid, "label": n.label,
                    "type": n.node_type, "confidence": n.confidence,
                })

        return {
            "topic": topic,
            "keywords": q_keywords,
            "n_nodes": len(nodes_info),
            "n_edges": len(edges),
            "nodes": nodes_info,
            "edges": edges,
        }

    def stats(self) -> Dict[str, Any]:
        """Return knowledge-base statistics."""
        type_counts = Counter(n.node_type for n in self.nodes.values())
        file_type_counts = Counter(s["file_type"] for s in self.sources.values())
        total_source_size = sum(s["size"] for s in self.sources.values())
        kb_size = 0
        for fname in os.listdir(self.kb_path):
            kb_size += os.path.getsize(os.path.join(self.kb_path, fname))
        return {
            "n_sources": len(self.sources),
            "n_chunks": len(self.chunks),
            "n_patterns": len(self.patterns),
            "n_nodes": len(self.nodes),
            "n_keywords_indexed": len(self._keyword_index),
            "node_types": dict(type_counts),
            "file_types": dict(file_type_counts),
            "original_size_bytes": total_source_size,
            "kb_size_bytes": kb_size,
            "compression_ratio": kb_size / max(total_source_size, 1),
            "mode": self.mode,
        }

    def forget(self, source_id: str) -> Dict[str, Any]:
        """Remove a source and all derived knowledge."""
        n_chunks_removed = 0
        n_patterns_removed = 0
        n_nodes_removed = 0

        # Remove chunks
        to_remove = [cid for cid, c in self.chunks.items() if c.source_id == source_id]
        for cid in to_remove:
            del self.chunks[cid]
            n_chunks_removed += 1

        # Remove patterns that only came from this source
        to_remove = []
        for pid, p in self.patterns.items():
            if source_id in p.sources:
                p.sources.remove(source_id)
                p.frequency -= 1
                if not p.sources or p.frequency <= 0:
                    to_remove.append(pid)
        for pid in to_remove:
            del self.patterns[pid]
            n_patterns_removed += 1

        # Remove nodes
        to_remove = []
        for nid, n in self.nodes.items():
            if source_id in n.sources:
                n.sources.remove(source_id)
                n.frequency -= 1
                if not n.sources or n.frequency <= 0:
                    to_remove.append(nid)
        for nid in to_remove:
            del self.nodes[nid]
            n_nodes_removed += 1

        # Remove the source record
        if source_id in self.sources:
            del self.sources[source_id]

        self._rebuild_keyword_index()
        self.save()
        return {
            "source_id": source_id,
            "chunks_removed": n_chunks_removed,
            "patterns_removed": n_patterns_removed,
            "nodes_removed": n_nodes_removed,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self):
        """Save the knowledge base to disk."""
        state = {
            "chunks": {cid: asdict(c) for cid, c in self.chunks.items()},
            "patterns": {pid: asdict(p) for pid, p in self.patterns.items()},
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            "sources": self.sources,
            "next_source_id": self._next_source_id,
            "mode": self.mode,
        }
        path = os.path.join(self.kb_path, "kb.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self):
        """Load the knowledge base from disk."""
        path = os.path.join(self.kb_path, "kb.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.chunks = {cid: Chunk(**c) for cid, c in state.get("chunks", {}).items()}
            self.patterns = {pid: Pattern(**p) for pid, p in state.get("patterns", {}).items()}
            self.nodes = {nid: GraphNode(**n) for nid, n in state.get("nodes", {}).items()}
            self.sources = state.get("sources", {})
            self._next_source_id = state.get("next_source_id", 0)
            self._rebuild_keyword_index()
        except Exception as e:
            if self.verbose:
                print(f"  warning: could not load KB: {e}")


# =====================================================================
# 7. CLI
# =====================================================================
def _cli_cpc_ingest(args):
    cpc = CPC(args.kb, mode=args.mode, verbose=True)
    summary = cpc.ingest(args.path)
    print(f"\nIngestion complete:")
    print(f"  files:   {summary['n_files']}")
    print(f"  chunks:  {summary['n_chunks']}")
    print(f"  patterns: {summary['n_patterns']}")
    print(f"  nodes:   {summary['n_nodes']}")
    s = cpc.stats()
    print(f"  KB size: {s['kb_size_bytes']:,} bytes (ratio: {s['compression_ratio']:.3f})")


def _cli_cpc_ask(args):
    cpc = CPC(args.kb, verbose=False)
    result = cpc.ask(args.question)
    print(result["answer"])
    if result["evidence"]:
        print("\nEvidence:")
        for ev in result["evidence"]:
            print(f"  [{ev['type']}] {ev['label']} (confidence: {ev['confidence']:.0%})")
            print(f"    sources: {ev['sources']}")


def _cli_cpc_search(args):
    cpc = CPC(args.kb, verbose=False)
    results = cpc.search(args.query)
    if not results:
        print("No results found.")
        return
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  [{r['type']}] {r['label']} (conf: {r['confidence']:.0%}, freq: {r['frequency']})")


def _cli_cpc_graph(args):
    cpc = CPC(args.kb, verbose=False)
    result = cpc.graph(args.topic)
    print(f"Graph around '{args.topic}': {result['n_nodes']} nodes, {result['n_edges']} edges")
    print("\nNodes:")
    for n in result["nodes"]:
        print(f"  [{n['type']}] {n['label']} (conf: {n['confidence']:.0%})")
    print("\nEdges:")
    for e in result["edges"]:
        print(f"  {e['from']} --{e['relation']}--> {e['to']}")


def _cli_cpc_stats(args):
    cpc = CPC(args.kb, verbose=False)
    s = cpc.stats()
    print(f"=== CPC Knowledge Base Stats ===")
    print(f"  mode:             {s['mode']}")
    print(f"  sources:          {s['n_sources']}")
    print(f"  chunks:           {s['n_chunks']}")
    print(f"  patterns:         {s['n_patterns']}")
    print(f"  nodes:            {s['n_nodes']}")
    print(f"  keywords indexed: {s['n_keywords_indexed']}")
    print(f"  node types:       {s['node_types']}")
    print(f"  file types:       {s['file_types']}")
    print(f"  original size:    {s['original_size_bytes']:,} bytes")
    print(f"  KB size:          {s['kb_size_bytes']:,} bytes")
    print(f"  compression ratio: {s['compression_ratio']:.3f}")


def _cli_cpc_forget(args):
    cpc = CPC(args.kb, verbose=False)
    result = cpc.forget(args.source_id)
    print(f"Forgot {args.source_id}:")
    print(f"  chunks removed:   {result['chunks_removed']}")
    print(f"  patterns removed: {result['patterns_removed']}")
    print(f"  nodes removed:    {result['nodes_removed']}")


def add_cpc_cli(subparsers):
    """Add CPC subcommands to an argparse subparsers."""
    p = subparsers.add_parser("cpc", help="Convergent Pattern Compression -- file-native learning system")
    cpc_sub = p.add_subparsers(dest="cpc_command")

    p_ingest = cpc_sub.add_parser("ingest", help="ingest a file or directory")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--kb", default=".cpc_kb")
    p_ingest.add_argument("--mode", default="smart_lossless",
                          choices=["lossless", "smart_lossless", "semantic"])

    p_ask = cpc_sub.add_parser("ask", help="ask a question about the ingested content")
    p_ask.add_argument("question")
    p_ask.add_argument("--kb", default=".cpc_kb")

    p_search = cpc_sub.add_parser("search", help="keyword search")
    p_search.add_argument("query")
    p_search.add_argument("--kb", default=".cpc_kb")

    p_graph = cpc_sub.add_parser("graph", help="show graph around a topic")
    p_graph.add_argument("topic")
    p_graph.add_argument("--kb", default=".cpc_kb")

    p_stats = cpc_sub.add_parser("stats", help="show knowledge-base statistics")
    p_stats.add_argument("--kb", default=".cpc_kb")

    p_forget = cpc_sub.add_parser("forget", help="remove a source + derived knowledge")
    p_forget.add_argument("source_id")
    p_forget.add_argument("--kb", default=".cpc_kb")


__all__ = [
    "CPC", "Chunk", "Pattern", "GraphNode",
    "chunk_file", "detect_patterns", "detect_file_type", "extract_keywords",
]
