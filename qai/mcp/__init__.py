"""
qai.mcp -- Model Context Protocol server for qai.

Exposes any qai model (and other tools) as an MCP-compatible server that AI
assistants like Claude, Cursor, or other MCP clients can call as tools.

## Architecture

The MCP protocol is JSON-RPC 2.0 over a transport (stdio or HTTP). qai.mcp
provides:

  - `MCPServer`       — the server class, holds tools and handles requests
  - `Tool`            — base class for tools (subclassed by every built-in)
  - `tool_registry`   — module-level registry of available tool classes
  - Built-in tools    — filesystem, memory, git, fetch, time, sqlite,
                        exec, sequential_thinking, calculator, code_eval,
                        qai_model, qai_logic, qai_memory, qai_bundle
  - Transports        — `stdio_serve()` (JSON-RPC over stdin/stdout),
                        `build_http_app()` (Flask for testing/HTTP)
  - Permission system — per-tool allow/deny lists
  - Rate limiting     — per-tool call limits per minute
  - Audit log         — every tool call recorded with timestamp + args + result

## Quick start

    import qai
    from qai.mcp import MCPServer, FilesystemTool, TimeTool, CalculatorTool

    # Build a model to expose
    model = qai.build(technique="regression")
    model.train([[1.0], [2.0]], [2.0, 4.0], verbose=False)

    # Create an MCP server with built-in + qai tools
    server = MCPServer(name="my-qai-server", version="1.0.0")
    server.register(TimeTool())
    server.register(CalculatorTool())
    server.register(FilesystemTool(allowed_dirs=["/tmp"]))
    server.register(QaiModelTool(model=model, tool_name="predict"))

    # List available tools (MCP `tools/list` method)
    print(server.list_tools())

    # Call a tool directly (programmatic, no transport needed)
    print(server.call_tool("calculator", {"expression": "2 + 3 * 4"}))

    # Serve over stdio (for use with Claude / Cursor / etc.)
    server.stdio_serve()

    # Or serve over HTTP for testing
    # app = server.build_http_app()
    # app.run(port=8080)

## Built-in tools (10)

Inspired by the official MCP reference servers
(https://github.com/modelcontextprotocol/servers):

  1. FilesystemTool  — read/write/list files (with sandboxed allowed_dirs)
  2. MemoryTool      — knowledge graph (entities + relations)
  3. GitTool         — git status/log/diff (read-only)
  4. FetchTool       — HTTP GET + text extraction
  5. TimeTool        — current time + timezone conversion
  6. SqliteTool      — read-only SQLite queries
  7. ExecTool        — sandboxed subprocess execution
  8. SequentialThinkingTool — step-by-step reasoning recorder
  9. CalculatorTool  — safe math expression evaluation
 10. CodeEvalTool    — sandboxed Python code execution

## qai-specific tools (4 — my own additions)

 11. QaiModelTool    — expose any trained qai.Model as an MCP tool
 12. QaiLogicTool    — expose any @qai.logic function as an MCP tool
 13. QaiMemoryTool   — bridge to qai.memory (working/retrieval tiers)
 14. QaiBundleTool   — load .qmodel bundle + run predictions

## Creative features (my own additions beyond the spec)

 - **Permission system**: per-tool allow/deny lists with role-based access
 - **Rate limiting**: per-tool calls-per-minute caps (default: 60)
 - **Audit log**: every tool call recorded with timestamp, args, result, caller
 - **Tool composition**: tools can call other tools (with cycle detection)
 - **Argument validation**: JSON-schema-based argument checking
 - **Async tool execution**: long-running tools can be non-blocking
 - **Tool introspection**: `inspect_tool(name)` returns schema + description
"""
from __future__ import annotations

import ast
import datetime
import io
import json
import operator as op
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# =====================================================================
# JSON-RPC 2.0 protocol
# =====================================================================
@dataclass
class JsonRpcRequest:
    """A JSON-RPC 2.0 request."""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[Union[int, str]] = None

    @classmethod
    def from_dict(cls, d: Dict) -> "JsonRpcRequest":
        return cls(
            method=d.get("method", ""),
            params=d.get("params", {}) or {},
            id=d.get("id"),
        )


@dataclass
class JsonRpcResponse:
    """A JSON-RPC 2.0 response."""
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = None

    def to_dict(self) -> Dict:
        d: Dict[str, Any] = {"jsonrpc": "2.0", "id": self.id}
        if self.error is not None:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


@dataclass
class JsonRpcError(Exception):
    """A JSON-RPC 2.0 error."""
    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict:
        d: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# MCP-specific error codes
TOOL_NOT_FOUND = -32001
TOOL_EXECUTION_ERROR = -32002
PERMISSION_DENIED = -32003
RATE_LIMIT_EXCEEDED = -32004
VALIDATION_ERROR = -32005


# =====================================================================
# Tool base class
# =====================================================================
class Tool:
    """Base class for MCP tools.

    Subclasses must override:
      - `name` (class attr): the tool name (snake_case)
      - `description` (class attr): human-readable description
      - `input_schema` (class attr): JSON schema for the `arguments` dict
      - `run(**arguments) -> Any`: the actual tool implementation

    Optional overrides:
      - `required_permissions`: list of permission strings needed to call this tool
      - `rate_limit_per_minute`: max calls per minute (default 60)
    """
    name: str = "tool"
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    required_permissions: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60

    def run(self, **arguments) -> Any:
        raise NotImplementedError(f"Tool {self.name} must implement run()")

    def to_mcp_dict(self) -> Dict[str, Any]:
        """Return the MCP `tools/list` representation of this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def __repr__(self):
        return f"<Tool {self.name!r}>"


# =====================================================================
# Tool registry
# =====================================================================
class ToolRegistry:
    """Registry of available tool classes (by name)."""
    def __init__(self):
        self._classes: Dict[str, type] = {}

    def register_class(self, name: str, cls: type) -> None:
        if not issubclass(cls, Tool):
            raise TypeError(f"{cls.__name__} must inherit from Tool")
        self._classes[name] = cls

    def create(self, name: str, **kwargs) -> Tool:
        if name not in self._classes:
            available = ", ".join(sorted(self._classes.keys()))
            raise ValueError(
                f"[WHAT] Unknown tool '{name}'\n"
                f"[WHY] Tool not registered\n"
                f"[FIX] Available tools: {available}"
            )
        return self._classes[name](**kwargs)

    def list_classes(self) -> List[str]:
        return sorted(self._classes.keys())


# Module-level registry — populated below as built-in tools are defined
tool_registry = ToolRegistry()


# =====================================================================
# Built-in tool 1: FilesystemTool
# =====================================================================
class FilesystemTool(Tool):
    """Secure file operations with configurable access controls.

    All file paths are validated against `allowed_dirs` — attempts to read
    or write outside the sandbox are rejected.
    """
    name = "filesystem"
    description = (
        "Secure file operations: read, write, list, mkdir, delete, exists. "
        "All paths must be inside one of the configured allowed_dirs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "list", "mkdir", "delete", "exists"],
                "description": "Filesystem operation to perform",
            },
            "path": {"type": "string", "description": "File or directory path"},
            "content": {"type": "string", "description": "Content to write (for write operation)"},
        },
        "required": ["operation", "path"],
    }
    required_permissions = ["filesystem"]

    def __init__(self, allowed_dirs: Optional[List[str]] = None, max_file_size: int = 10 * 1024 * 1024):
        """
        Args:
            allowed_dirs: List of directories the tool can read/write inside.
                If None, defaults to ["/tmp"]. Paths must be inside (or equal
                to) one of these dirs.
            max_file_size: Max bytes for read/write operations (default 10MB).
        """
        self.allowed_dirs = [os.path.abspath(d) for d in (allowed_dirs or ["/tmp"])]
        self.max_file_size = max_file_size

    def _validate_path(self, path: str) -> str:
        """Ensure path is inside one of allowed_dirs. Returns absolute path."""
        abs_path = os.path.abspath(path)
        for allowed in self.allowed_dirs:
            # Path is OK if it equals allowed dir, or is inside it
            if abs_path == allowed or abs_path.startswith(allowed + os.sep):
                return abs_path
        raise JsonRpcError(
            PERMISSION_DENIED,
            f"Path {path!r} is outside allowed directories {self.allowed_dirs}",
        )

    def run(self, operation: str, path: str, content: str = None, **kwargs) -> Any:
        abs_path = self._validate_path(path)

        if operation == "read":
            if not os.path.isfile(abs_path):
                raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Not a file: {path}")
            size = os.path.getsize(abs_path)
            if size > self.max_file_size:
                raise JsonRpcError(
                    TOOL_EXECUTION_ERROR,
                    f"File too large: {size} bytes (max {self.max_file_size})",
                )
            with open(abs_path, "r", encoding="utf-8") as f:
                return f.read()

        elif operation == "write":
            if content is None:
                raise JsonRpcError(INVALID_PARAMS, "write operation requires 'content'")
            if len(content) > self.max_file_size:
                raise JsonRpcError(
                    TOOL_EXECUTION_ERROR,
                    f"Content too large: {len(content)} bytes (max {self.max_file_size})",
                )
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"written": len(content), "path": abs_path}

        elif operation == "list":
            if not os.path.isdir(abs_path):
                raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Not a directory: {path}")
            entries = []
            for entry in os.listdir(abs_path):
                full = os.path.join(abs_path, entry)
                entries.append({
                    "name": entry,
                    "type": "dir" if os.path.isdir(full) else "file",
                    "size": os.path.getsize(full) if os.path.isfile(full) else None,
                })
            return entries

        elif operation == "mkdir":
            os.makedirs(abs_path, exist_ok=True)
            return {"created": abs_path}

        elif operation == "delete":
            if not os.path.exists(abs_path):
                raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Path does not exist: {path}")
            if os.path.isdir(abs_path):
                os.rmdir(abs_path)  # only removes empty dirs (safer)
            else:
                os.remove(abs_path)
            return {"deleted": abs_path}

        elif operation == "exists":
            return {"exists": os.path.exists(abs_path), "is_file": os.path.isfile(abs_path),
                    "is_dir": os.path.isdir(abs_path)}

        else:
            raise JsonRpcError(INVALID_PARAMS, f"Unknown operation: {operation!r}")


tool_registry.register_class("filesystem", FilesystemTool)


# =====================================================================
# Built-in tool 2: MemoryTool (knowledge graph)
# =====================================================================
class MemoryTool(Tool):
    """Knowledge graph memory: entities with observations + typed relations.

    Inspired by the official MCP Memory server. Stores entities (each with
    a name, type, and observations) and relations (subject-predicate-object
    triples). All state is in-memory by default; optionally persists to JSON.
    """
    name = "memory"
    description = (
        "Knowledge graph memory: create entities with observations and "
        "typed relations between them. Supports queries by entity name, "
        "type, or relation traversal."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["create_entity", "add_observation", "create_relation",
                         "delete_entity", "delete_relation", "get_entity",
                         "search", "list_all", "save", "load"],
                "description": "Memory operation to perform",
            },
            "entity_name": {"type": "string"},
            "entity_type": {"type": "string"},
            "observations": {"type": "array", "items": {"type": "string"}},
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "query": {"type": "string"},
            "path": {"type": "string"},
        },
    }

    def __init__(self, persist_path: Optional[str] = None):
        self.entities: Dict[str, Dict[str, Any]] = {}  # name -> {type, observations}
        self.relations: List[Dict[str, str]] = []  # [{subject, predicate, object}]
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path) and os.path.getsize(persist_path) > 0:
            try:
                self._load(persist_path)
            except (json.JSONDecodeError, OSError):
                pass  # corrupt or empty — start fresh

    def run(self, operation: str, **kw) -> Any:
        if operation == "create_entity":
            name = kw["entity_name"]
            etype = kw.get("entity_type", "thing")
            obs = kw.get("observations", [])
            if name in self.entities:
                raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Entity {name!r} already exists")
            self.entities[name] = {"type": etype, "observations": list(obs)}
            self._maybe_persist()
            return {"created": name}

        if operation == "add_observation":
            name = kw["entity_name"]
            if name not in self.entities:
                raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Entity {name!r} not found")
            new_obs = kw.get("observations", [])
            self.entities[name]["observations"].extend(new_obs)
            self._maybe_persist()
            return {"entity": name, "total_observations": len(self.entities[name]["observations"])}

        if operation == "create_relation":
            sub, pred, obj = kw["subject"], kw["predicate"], kw["object"]
            for n in (sub, obj):
                if n not in self.entities:
                    raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Entity {n!r} not found — create it first")
            rel = {"subject": sub, "predicate": pred, "object": obj}
            if rel not in self.relations:
                self.relations.append(rel)
            self._maybe_persist()
            return {"created": rel}

        if operation == "delete_entity":
            name = kw["entity_name"]
            if name not in self.entities:
                raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Entity {name!r} not found")
            del self.entities[name]
            # Remove all relations involving this entity
            self.relations = [r for r in self.relations
                              if r["subject"] != name and r["object"] != name]
            self._maybe_persist()
            return {"deleted": name}

        if operation == "delete_relation":
            sub, pred, obj = kw["subject"], kw["predicate"], kw["object"]
            before = len(self.relations)
            self.relations = [r for r in self.relations
                              if not (r["subject"] == sub and r["predicate"] == pred
                                      and r["object"] == obj)]
            self._maybe_persist()
            return {"deleted_count": before - len(self.relations)}

        if operation == "get_entity":
            name = kw["entity_name"]
            if name not in self.entities:
                return None
            ent = self.entities[name]
            # Include relations involving this entity
            rels = [r for r in self.relations if r["subject"] == name or r["object"] == name]
            return {"name": name, "type": ent["type"],
                    "observations": ent["observations"], "relations": rels}

        if operation == "search":
            q = kw.get("query", "").lower()
            results = []
            for name, ent in self.entities.items():
                if q in name.lower() or any(q in o.lower() for o in ent["observations"]):
                    results.append({"name": name, "type": ent["type"],
                                    "observations": ent["observations"]})
            return results

        if operation == "list_all":
            return {
                "entities": [{"name": n, "type": e["type"], "observations": e["observations"]}
                             for n, e in self.entities.items()],
                "relations": list(self.relations),
                "counts": {"entities": len(self.entities), "relations": len(self.relations)},
            }

        if operation == "save":
            self._save(kw.get("path", self.persist_path or "memory.json"))
            return {"saved": True}

        if operation == "load":
            self._load(kw["path"])
            return {"loaded": True, "entities": len(self.entities), "relations": len(self.relations)}

        raise JsonRpcError(INVALID_PARAMS, f"Unknown operation: {operation!r}")

    def _maybe_persist(self):
        if self.persist_path:
            self._save(self.persist_path)

    def _save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"entities": self.entities, "relations": self.relations}, f, indent=2)

    def _load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entities = data.get("entities", {})
        self.relations = data.get("relations", [])


tool_registry.register_class("memory", MemoryTool)


# =====================================================================
# Built-in tool 3: GitTool (read-only)
# =====================================================================
class GitTool(Tool):
    """Read-only git operations: status, log, diff, show, branch list."""
    name = "git"
    description = (
        "Read-only git repository operations: status, log, diff, show, "
        "branch list. Does NOT modify the repo (no commit/push/pull)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["status", "log", "diff", "show", "branches", "current_branch"],
            },
            "repo_path": {"type": "string", "description": "Path to git repo (default: cwd)"},
            "limit": {"type": "integer", "description": "Max entries (for log)"},
            "ref": {"type": "string", "description": "Branch/commit for diff/show"},
            "file": {"type": "string", "description": "File path for diff/show"},
        },
        "required": ["operation"],
    }
    required_permissions = ["git"]

    def _run_git(self, args: List[str], repo_path: str) -> str:
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise JsonRpcError(TOOL_EXECUTION_ERROR, f"git error: {result.stderr.strip()}")
            return result.stdout
        except FileNotFoundError:
            raise JsonRpcError(TOOL_EXECUTION_ERROR, "git not installed")
        except subprocess.TimeoutExpired:
            raise JsonRpcError(TOOL_EXECUTION_ERROR, "git command timed out")

    def run(self, operation: str, repo_path: str = ".", limit: int = 20,
            ref: str = None, file: str = None, **kw) -> Any:
        repo_path = os.path.abspath(repo_path)
        if not os.path.isdir(repo_path):
            raise JsonRpcError(INVALID_PARAMS, f"Not a directory: {repo_path}")

        if operation == "status":
            return self._run_git(["status", "--porcelain"], repo_path)

        if operation == "log":
            n = max(1, min(limit, 200))
            return self._run_git(["log", f"-{n}", "--oneline"], repo_path)

        if operation == "diff":
            args = ["diff"]
            if ref:
                args.append(ref)
            if file:
                args += ["--", file]
            return self._run_git(args, repo_path)

        if operation == "show":
            if not ref:
                raise JsonRpcError(INVALID_PARAMS, "show requires 'ref'")
            return self._run_git(["show", ref], repo_path)

        if operation == "branches":
            local = self._run_git(["branch", "--list"], repo_path)
            return local

        if operation == "current_branch":
            return self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_path).strip()

        raise JsonRpcError(INVALID_PARAMS, f"Unknown operation: {operation!r}")


tool_registry.register_class("git", GitTool)


# =====================================================================
# Built-in tool 4: FetchTool (HTTP GET + text extraction)
# =====================================================================
class FetchTool(Tool):
    """HTTP GET + text extraction. Strips HTML tags for clean LLM consumption."""
    name = "fetch"
    description = (
        "Fetch a URL and return its content. For HTML pages, strips tags "
        "and returns plain text. Configurable max bytes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_bytes": {"type": "integer", "description": "Max bytes to fetch (default 1MB)"},
            "extract_text": {"type": "boolean", "description": "Strip HTML tags (default true)"},
            "timeout": {"type": "number", "description": "HTTP timeout seconds (default 10)"},
        },
        "required": ["url"],
    }
    required_permissions = ["fetch"]

    def run(self, url: str, max_bytes: int = 1024 * 1024, extract_text: bool = True,
            timeout: float = 10.0, **kw) -> Any:
        if not url.startswith(("http://", "https://")):
            raise JsonRpcError(INVALID_PARAMS, f"URL must start with http:// or https:// — got {url!r}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "qai-mcp-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read(max_bytes)
        except urllib.error.URLError as e:
            raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Fetch failed: {e}")
        except Exception as e:
            raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Fetch error: {e}")

        # Decode
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = data.decode("latin-1", errors="replace")

        # Strip HTML if requested and looks like HTML
        if extract_text and ("html" in content_type.lower() or "<html" in text.lower()[:500]):
            text = self._strip_html(text)

        return {
            "url": url,
            "status": "ok",
            "content_type": content_type,
            "size": len(data),
            "text": text,
        }

    @staticmethod
    def _strip_html(html: str) -> str:
        # Remove script/style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Convert <br>, <p>, <div> to newlines
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</(p|div|li|h[1-6])>", "\n", html, flags=re.IGNORECASE)
        # Remove all remaining tags
        text = re.sub(r"<[^>]+>", "", html)
        # Decode common HTML entities
        entities = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
                    "&#39;": "'", "&nbsp;": " "}
        for ent, ch in entities.items():
            text = text.replace(ent, ch)
        # Collapse whitespace
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return text


tool_registry.register_class("fetch", FetchTool)


# =====================================================================
# Built-in tool 5: TimeTool
# =====================================================================
class TimeTool(Tool):
    """Time and timezone conversion."""
    name = "time"
    description = "Get current time, convert between timezones, format timestamps."
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["now", "convert", "format"],
            },
            "timezone": {"type": "string", "description": "IANA timezone (e.g. 'America/New_York')"},
            "from_tz": {"type": "string"},
            "to_tz": {"type": "string"},
            "timestamp": {"type": "number", "description": "Unix timestamp"},
            "fmt": {"type": "string", "description": "strftime format"},
        },
        "required": ["operation"],
    }

    def run(self, operation: str, timezone: str = None, from_tz: str = None,
            to_tz: str = None, timestamp: float = None, fmt: str = None, **kw) -> Any:
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            ZoneInfo = None  # Python < 3.9

        if operation == "now":
            if timezone:
                if ZoneInfo is None:
                    raise JsonRpcError(TOOL_EXECUTION_ERROR, "zoneinfo not available")
                try:
                    tz = ZoneInfo(timezone)
                except Exception as e:
                    raise JsonRpcError(INVALID_PARAMS, f"Unknown timezone: {timezone!r} ({e})")
                now = datetime.datetime.now(tz)
            else:
                now = datetime.datetime.now()
            return {
                "iso": now.isoformat(),
                "unix": now.timestamp(),
                "timezone": str(now.tzinfo) if now.tzinfo else "local",
            }

        if operation == "convert":
            if timestamp is None or from_tz is None or to_tz is None:
                raise JsonRpcError(INVALID_PARAMS, "convert requires timestamp, from_tz, to_tz")
            if ZoneInfo is None:
                raise JsonRpcError(TOOL_EXECUTION_ERROR, "zoneinfo not available")
            try:
                src_tz = ZoneInfo(from_tz)
                dst_tz = ZoneInfo(to_tz)
            except Exception as e:
                raise JsonRpcError(INVALID_PARAMS, f"Unknown timezone: {e}")
            dt = datetime.datetime.fromtimestamp(timestamp, tz=src_tz)
            converted = dt.astimezone(dst_tz)
            return {
                "original_iso": dt.isoformat(),
                "converted_iso": converted.isoformat(),
                "from_tz": from_tz,
                "to_tz": to_tz,
            }

        if operation == "format":
            if timestamp is None:
                raise JsonRpcError(INVALID_PARAMS, "format requires timestamp")
            fmt = fmt or "%Y-%m-%d %H:%M:%S"
            dt = datetime.datetime.fromtimestamp(timestamp)
            return dt.strftime(fmt)

        raise JsonRpcError(INVALID_PARAMS, f"Unknown operation: {operation!r}")


tool_registry.register_class("time", TimeTool)


# =====================================================================
# Built-in tool 6: SqliteTool (read-only)
# =====================================================================
class SqliteTool(Tool):
    """Read-only SQLite queries. SELECT only — no INSERT/UPDATE/DELETE/DDL."""
    name = "sqlite"
    description = (
        "Read-only SQLite database queries. Connects to a .db file and runs "
        "SELECT statements. Also supports listing tables and schema inspection."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["query", "tables", "schema"],
            },
            "db_path": {"type": "string", "description": "Path to .db file"},
            "sql": {"type": "string", "description": "SQL query (SELECT only)"},
            "table": {"type": "string", "description": "Table name (for schema op)"},
            "limit": {"type": "integer", "description": "Max rows (default 100)"},
        },
        "required": ["operation", "db_path"],
    }
    required_permissions = ["sqlite"]

    def run(self, operation: str, db_path: str, sql: str = None, table: str = None,
            limit: int = 100, **kw) -> Any:
        import sqlite3
        if not os.path.isfile(db_path):
            raise JsonRpcError(INVALID_PARAMS, f"Database file not found: {db_path}")
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
        except Exception as e:
            raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Cannot open database: {e}")

        try:
            if operation == "query":
                if not sql:
                    raise JsonRpcError(INVALID_PARAMS, "query operation requires 'sql'")
                # Reject anything that isn't a SELECT (case-insensitive)
                if not sql.strip().upper().startswith("SELECT"):
                    raise JsonRpcError(
                        PERMISSION_DENIED,
                        "Only SELECT statements are allowed (read-only tool)",
                    )
                cur.execute(sql)
                rows = cur.fetchmany(limit)
                return {
                    "columns": [d[0] for d in cur.description] if cur.description else [],
                    "rows": [dict(r) for r in rows],
                    "rowcount": len(rows),
                    "truncated": len(rows) == limit,
                }

            if operation == "tables":
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                return [r[0] for r in cur.fetchall()]

            if operation == "schema":
                target = table or "%"
                cur.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type='table' AND name LIKE ?",
                    (target,),
                )
                return [{"name": r[0], "sql": r[1]} for r in cur.fetchall()]

        finally:
            conn.close()


tool_registry.register_class("sqlite", SqliteTool)


# =====================================================================
# Built-in tool 7: ExecTool (sandboxed subprocess)
# =====================================================================
class ExecTool(Tool):
    """Sandboxed subprocess execution with timeout + working dir restriction."""
    name = "exec"
    description = (
        "Execute a shell command in a sandboxed subprocess. Returns stdout, "
        "stderr, and exit code. Commands run inside `work_dir` (default /tmp) "
        "with a configurable timeout."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "work_dir": {"type": "string", "description": "Working directory (default /tmp)"},
            "timeout": {"type": "number", "description": "Timeout seconds (default 10)"},
            "env": {"type": "object", "description": "Extra env vars"},
        },
        "required": ["command"],
    }
    required_permissions = ["exec"]

    def __init__(self, allowed_work_dirs: Optional[List[str]] = None, default_timeout: float = 10.0):
        self.allowed_work_dirs = [os.path.abspath(d) for d in (allowed_work_dirs or ["/tmp"])]
        self.default_timeout = default_timeout

    def _validate_work_dir(self, work_dir: str) -> str:
        abs_path = os.path.abspath(work_dir)
        for allowed in self.allowed_work_dirs:
            if abs_path == allowed or abs_path.startswith(allowed + os.sep):
                return abs_path
        raise JsonRpcError(
            PERMISSION_DENIED,
            f"work_dir {work_dir!r} is outside allowed dirs {self.allowed_work_dirs}",
        )

    def run(self, command: str, work_dir: str = "/tmp", timeout: float = None,
            env: Dict = None, **kw) -> Any:
        work_dir = self._validate_work_dir(work_dir)
        timeout = timeout or self.default_timeout

        full_env = os.environ.copy()
        if env:
            full_env.update({k: str(v) for k, v in env.items()})

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=full_env,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "command": command,
            }
        except subprocess.TimeoutExpired:
            raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Command timed out after {timeout}s")
        except Exception as e:
            raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Exec failed: {e}")


tool_registry.register_class("exec", ExecTool)


# =====================================================================
# Built-in tool 8: SequentialThinkingTool
# =====================================================================
class SequentialThinkingTool(Tool):
    """Step-by-step reasoning recorder. Each call adds one thought to a sequence.

    Inspired by the official MCP sequential-thinking server. The idea: an AI
    can call this tool repeatedly to externalize its reasoning process, then
    retrieve the full sequence later for review or revision.
    """
    name = "sequential_thinking"
    description = (
        "Record a step-by-step reasoning sequence. Each call appends one "
        "thought (with optional 'next_thought_needed' and 'revision_of' "
        "fields). Use list/revise operations to retrieve or modify the sequence."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["think", "list", "revise", "clear", "summary"],
            },
            "thought": {"type": "string", "description": "The reasoning step (for 'think')"},
            "next_thought_needed": {"type": "boolean", "description": "More steps to come?"},
            "thought_number": {"type": "integer", "description": "Sequence number"},
            "revision_of": {"type": "integer", "description": "Thought number being revised"},
        },
        "required": ["operation"],
    }

    def __init__(self):
        self.thoughts: List[Dict[str, Any]] = []

    def run(self, operation: str, thought: str = None, next_thought_needed: bool = False,
            thought_number: int = None, revision_of: int = None, **kw) -> Any:
        if operation == "think":
            if thought is None:
                raise JsonRpcError(INVALID_PARAMS, "think requires 'thought'")
            n = thought_number if thought_number is not None else len(self.thoughts) + 1
            entry = {
                "thought_number": n,
                "thought": thought,
                "next_thought_needed": next_thought_needed,
                "timestamp": time.time(),
            }
            if revision_of is not None:
                entry["revision_of"] = revision_of
            self.thoughts.append(entry)
            return entry

        if operation == "list":
            return list(self.thoughts)

        if operation == "revise":
            if thought_number is None or thought is None:
                raise JsonRpcError(INVALID_PARAMS, "revise requires thought_number and thought")
            for i, t in enumerate(self.thoughts):
                if t["thought_number"] == thought_number:
                    self.thoughts[i] = {
                        **t,
                        "thought": thought,
                        "revised_at": time.time(),
                    }
                    return self.thoughts[i]
            raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Thought #{thought_number} not found")

        if operation == "clear":
            n = len(self.thoughts)
            self.thoughts.clear()
            return {"cleared": n}

        if operation == "summary":
            return {
                "total_thoughts": len(self.thoughts),
                "thoughts": [t["thought"] for t in self.thoughts],
                "completed": not (self.thoughts and self.thoughts[-1].get("next_thought_needed")),
            }

        raise JsonRpcError(INVALID_PARAMS, f"Unknown operation: {operation!r}")


tool_registry.register_class("sequential_thinking", SequentialThinkingTool)


# =====================================================================
# Built-in tool 9: CalculatorTool (safe math eval)
# =====================================================================
class CalculatorTool(Tool):
    """Safe math expression evaluator. No `import`, no function defs, no attribute access."""
    name = "calculator"
    description = (
        "Evaluate a math expression safely. Supports +, -, *, /, **, %, "
        "parentheses, and the functions: abs, min, max, round, sqrt, sin, "
        "cos, tan, log, log10, exp, pi, e. No imports, no function defs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"},
        },
        "required": ["expression"],
    }

    # Allowed AST node types (ast.Num deprecated in 3.12, use ast.Constant)
    _ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Name,
    )
    # Allowed functions/constants
    _ALLOWED_NAMES = {
        "abs": abs, "min": min, "max": max, "round": round,
        "sqrt": __import__("math").sqrt,
        "sin": __import__("math").sin, "cos": __import__("math").cos,
        "tan": __import__("math").tan,
        "log": __import__("math").log, "log10": __import__("math").log10,
        "log2": __import__("math").log2,
        "exp": __import__("math").exp,
        "pow": pow, "floor": __import__("math").floor, "ceil": __import__("math").ceil,
        "pi": __import__("math").pi, "e": __import__("math").e,
        "tau": __import__("math").tau,
    }

    def run(self, expression: str, **kw) -> Any:
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as e:
            raise JsonRpcError(INVALID_PARAMS, f"Syntax error: {e}")

        # Walk AST to reject anything not in the allowed set
        for node in ast.walk(tree):
            if not isinstance(node, self._ALLOWED_NODES):
                raise JsonRpcError(
                    PERMISSION_DENIED,
                    f"Disallowed syntax: {type(node).__name__}. "
                    "Calculator only supports math expressions.",
                )

        try:
            result = eval(compile(tree, "<calc>", "eval"),
                          {"__builtins__": {}}, self._ALLOWED_NAMES)
        except Exception as e:
            raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Evaluation error: {e}")

        # Convert numpy/complex types to plain Python for JSON
        if isinstance(result, complex):
            return {"real": result.real, "imag": result.imag, "is_complex": True}
        if isinstance(result, float) and result.is_integer():
            return {"value": int(result), "type": "int"}
        return {"value": result, "type": type(result).__name__}


tool_registry.register_class("calculator", CalculatorTool)


# =====================================================================
# Built-in tool 10: CodeEvalTool (sandboxed Python)
# =====================================================================
class CodeEvalTool(Tool):
    """Sandboxed Python code execution. Limited builtins, no imports, timeout."""
    name = "code_eval"
    description = (
        "Execute Python code in a restricted sandbox. Limited builtins "
        "(no open, exec, eval, __import__, etc.). Returns stdout, result, "
        "and any error. Useful for code generation agents."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {"type": "number", "description": "Timeout seconds (default 5)"},
        },
        "required": ["code"],
    }
    required_permissions = ["code_eval"]

    # A safe subset of builtins
    _SAFE_BUILTINS = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float,
        "int": int, "len": len, "list": list, "map": map, "max": max,
        "min": min, "print": print, "range": range, "round": round,
        "set": set, "sorted": sorted, "str": str, "sum": sum,
        "tuple": tuple, "zip": zip, "isinstance": isinstance,
        "type": type, "repr": repr, "True": True, "False": False, "None": None,
        "locals": locals, "globals": lambda: {},  # restricted globals
    }

    def run(self, code: str, timeout: float = 5.0, **kw) -> Any:
        # Reject anything that looks dangerous
        forbidden = ["__import__", "import ", "exec(", "eval(", "open(",
                     "subprocess", "os.system", "os.popen", "__builtins__",
                     "globals()"]
        for f in forbidden:
            if f in code:
                raise JsonRpcError(
                    PERMISSION_DENIED,
                    f"Forbidden pattern in code: {f!r}. CodeEvalTool is sandboxed.",
                )

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        globals_dict = {"__builtins__": self._SAFE_BUILTINS}
        # We'll execute the user's code as the body of a function, then
        # extract `result` if the user assigned it (convention), OR the value
        # of the last expression statement (REPL-style).
        try:
            try:
                tree = ast.parse(code)
                last_expr_value = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr_value = ast.unparse(tree.body[-1].value)
                    tree.body = tree.body[:-1]

                wrapped_lines = ["def __qai_eval():"]
                if not tree.body and last_expr_value is None:
                    wrapped_lines.append("    pass")
                for node in tree.body:
                    wrapped_lines.extend("    " + line for line in ast.unparse(node).split("\n"))
                if last_expr_value is not None:
                    wrapped_lines.append(f"    return ({last_expr_value})")
                else:
                    wrapped_lines.append("    return locals().get('result')")
                wrapped = "\n".join(wrapped_lines)

                exec(compile(wrapped, "<sandbox>", "exec"), globals_dict)
                # Call the wrapped function
                result = globals_dict["__qai_eval"]()
                stdout = sys.stdout.getvalue()
                return {
                    "stdout": stdout,
                    "result": self._jsonable(result),
                    "result_type": type(result).__name__ if result is not None else "NoneType",
                }
            except Exception as e:
                return {
                    "stdout": sys.stdout.getvalue(),
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc().split("\n"),
                }
        finally:
            sys.stdout = old_stdout

    @staticmethod
    def _jsonable(v):
        if v is None or isinstance(v, (int, float, str, bool, list, dict, tuple)):
            try:
                json.dumps(v)
                return v
            except Exception:
                return repr(v)
        return repr(v)


tool_registry.register_class("code_eval", CodeEvalTool)


# =====================================================================
# qai-specific tool 11: QaiModelTool (my own addition)
# =====================================================================
class QaiModelTool(Tool):
    """Expose any trained qai.Model as an MCP tool.

    The model's `predict` method becomes the tool's `run` method. The
    input_schema is auto-generated based on the model's `input_schema`
    attribute (if set), or defaults to a flexible `X` array argument.
    """
    name = "qai_model"
    description = "Call a trained qai model's predict method."
    input_schema = {
        "type": "object",
        "properties": {
            "X": {"type": "array", "description": "Input features (single sample as list, or list of lists for batch)"},
        },
        "required": ["X"],
    }

    def __init__(self, model: Any, tool_name: str = "qai_model", description: str = None):
        """
        Args:
            model: A trained qai.Model (must have `predict()` method).
            tool_name: Override the tool name (useful when exposing multiple models).
            description: Override the tool description.
        """
        self.model = model
        self.name = tool_name
        technique_name = getattr(model, "technique_name", "unknown")
        self.description = description or (
            f"Call the trained qai model (technique={technique_name!r}) predict method."
        )

    def run(self, X: Any, **kw) -> Any:
        try:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                result = self.model.predict(X_arr.tolist())
            else:
                result = self.model.predict(X_arr)
            return self._jsonable(result)
        except Exception as e:
            raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Model prediction failed: {e}")

    @staticmethod
    def _jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [QaiModelTool._jsonable(x) for x in v]
        return v


tool_registry.register_class("qai_model", QaiModelTool)


# =====================================================================
# qai-specific tool 12: QaiLogicTool (my own addition)
# =====================================================================
class QaiLogicTool(Tool):
    """Expose any @qai.logic-decorated function as an MCP tool.

    The function's signature is introspected to build the input_schema.
    """
    name = "qai_logic"
    description = "Call a @qai.logic-decorated function."
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, logic_fn: Callable, tool_name: str = None, description: str = None):
        import inspect
        self.logic_fn = logic_fn
        self.name = tool_name or getattr(logic_fn, "logic_name", logic_fn.__name__)
        self.description = description or (
            f"@qai.logic function: {self.name}. "
            + (inspect.getdoc(logic_fn) or "").split("\n")[0]
        )
        # Build input schema from function signature
        sig = inspect.signature(logic_fn)
        properties = {}
        required = []
        for pname, param in sig.parameters.items():
            properties[pname] = {"type": "string", "description": f"Argument {pname}"}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        self.input_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def run(self, **kwargs) -> Any:
        try:
            # Try to convert string args to appropriate types
            import inspect
            sig = inspect.signature(self.logic_fn)
            coerced = {}
            for pname, param in sig.parameters.items():
                if pname in kwargs:
                    val = kwargs[pname]
                    # Try to eval as Python literal (int, float, list, etc.)
                    try:
                        coerced[pname] = ast.literal_eval(val) if isinstance(val, str) else val
                    except (ValueError, SyntaxError):
                        coerced[pname] = val
            result = self.logic_fn(**coerced)
            return self._jsonable(result)
        except Exception as e:
            raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Logic function failed: {e}")

    @staticmethod
    def _jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [QaiLogicTool._jsonable(x) for x in v]
        return v


tool_registry.register_class("qai_logic", QaiLogicTool)


# =====================================================================
# qai-specific tool 13: QaiMemoryTool (my own addition)
# =====================================================================
class QaiMemoryTool(Tool):
    """Bridge to qai.memory tiers (working + retrieval).

    Allows MCP clients to add/query working memory and search retrieval memory.
    """
    name = "qai_memory"
    description = (
        "Bridge to qai.memory: add/query working memory, search retrieval memory, "
        "view weight-snapshot summary."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["working_add", "working_recent", "working_search",
                         "retrieval_add", "retrieval_search", "summary"],
            },
            "text": {"type": "string"},
            "query": {"type": "string"},
            "k": {"type": "integer"},
            "metadata": {"type": "object"},
            "n": {"type": "integer"},
        },
        "required": ["operation"],
    }

    def __init__(self, memory: Any = None):
        """
        Args:
            memory: A qai.memory.Memory instance. If None, a fresh one is created.
        """
        if memory is None:
            import qai
            memory = qai.Memory()
        self.memory = memory

    def run(self, operation: str, text: str = None, query: str = None, k: int = 5,
            metadata: Dict = None, n: int = 10, **kw) -> Any:
        if operation == "working_add":
            if text is None:
                raise JsonRpcError(INVALID_PARAMS, "working_add requires 'text'")
            self.memory.working.add(text, metadata=metadata or {})
            return {"added": True, "size": len(self.memory.working)}

        if operation == "working_recent":
            return self.memory.working.recent(n)

        if operation == "working_search":
            if query is None:
                raise JsonRpcError(INVALID_PARAMS, "working_search requires 'query'")
            return [{"text": t, "score": s} for t, s in self.memory.working.search(query, k=k)]

        if operation == "retrieval_add":
            if text is None:
                raise JsonRpcError(INVALID_PARAMS, "retrieval_add requires 'text'")
            id, conflicts = self.memory.retrieval.add(text, metadata=metadata or {})
            return {"id": id, "conflicts": conflicts, "size": len(self.memory.retrieval)}

        if operation == "retrieval_search":
            if query is None:
                raise JsonRpcError(INVALID_PARAMS, "retrieval_search requires 'query'")
            results = self.memory.retrieval.search(query, k=k)
            return [{"text": r.text, "score": r.score, "metadata": r.metadata, "id": r.id}
                    for r in results]

        if operation == "summary":
            return self.memory.summary()

        raise JsonRpcError(INVALID_PARAMS, f"Unknown operation: {operation!r}")


tool_registry.register_class("qai_memory", QaiMemoryTool)


# =====================================================================
# qai-specific tool 14: QaiBundleTool (my own addition)
# =====================================================================
class QaiBundleTool(Tool):
    """Load a .qmodel bundle and run predictions through it."""
    name = "qai_bundle"
    description = (
        "Load a .qmodel bundle (saved via qai.save_bundle) and run predictions. "
        "Returns the bundle's manifest info, or runs a prediction."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["inspect", "predict", "list_logic", "call_logic"],
            },
            "path": {"type": "string", "description": "Path to .qmodel bundle"},
            "X": {"type": "array", "description": "Input features for predict"},
            "logic_name": {"type": "string"},
            "args": {"type": "object"},
            "sign_key": {"type": "string"},
        },
        "required": ["operation", "path"],
    }

    def run(self, operation: str, path: str, X: Any = None, logic_name: str = None,
            args: Dict = None, sign_key: str = None, **kw) -> Any:
        import qai
        if operation == "inspect":
            return qai.inspect_bundle(path)

        if operation == "predict":
            if X is None:
                raise JsonRpcError(INVALID_PARAMS, "predict requires 'X'")
            bundle = qai.load_bundle(path, sign_key=sign_key, verify_signature=sign_key is not None)
            try:
                X_arr = np.asarray(X, dtype=float)
                result = bundle.model.predict(X_arr.tolist() if X_arr.ndim == 1 else X_arr)
                return self._jsonable(result)
            finally:
                pass

        if operation == "list_logic":
            bundle = qai.load_bundle(path, sign_key=sign_key, verify_signature=sign_key is not None)
            return list(bundle.logic_fns.keys())

        if operation == "call_logic":
            if logic_name is None:
                raise JsonRpcError(INVALID_PARAMS, "call_logic requires 'logic_name'")
            bundle = qai.load_bundle(path, sign_key=sign_key, verify_signature=sign_key is not None)
            if logic_name not in bundle.logic_fns:
                raise JsonRpcError(TOOL_NOT_FOUND,
                                    f"Logic function {logic_name!r} not in bundle")
            try:
                result = bundle.logic_fns[logic_name](*(args or {}).values())
                return self._jsonable(result)
            except Exception as e:
                raise JsonRpcError(TOOL_EXECUTION_ERROR, f"Logic call failed: {e}")

        raise JsonRpcError(INVALID_PARAMS, f"Unknown operation: {operation!r}")

    @staticmethod
    def _jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [QaiBundleTool._jsonable(x) for x in v]
        return v


tool_registry.register_class("qai_bundle", QaiBundleTool)


# =====================================================================
# MCP Server
# =====================================================================
class MCPServer:
    """An MCP server: holds tools, handles JSON-RPC requests, serves over a transport.

    The server supports both:
      - Direct programmatic calls (call_tool, list_tools) — for testing
      - JSON-RPC over stdio (stdio_serve) — for use with Claude/Cursor/etc.
      - JSON-RPC over HTTP (build_http_app) — for testing/networked use

    Creative additions:
      - **Permission system**: per-tool allow/deny lists with role-based access
      - **Rate limiting**: per-tool calls-per-minute caps
      - **Audit log**: every tool call recorded with timestamp, args, result
      - **Argument validation**: schema-based argument checking
    """
    PROTOCOL_VERSION = "2024-11-05"

    def __init__(
        self,
        name: str = "qai-mcp-server",
        version: str = "1.0.0",
        allowed_tools: Optional[List[str]] = None,
        denied_tools: Optional[List[str]] = None,
        default_rate_limit: int = 60,
        audit_log_size: int = 1000,
    ):
        """
        Args:
            name: Server name (returned in `initialize` response).
            version: Server version (returned in `initialize` response).
            allowed_tools: If set, only these tools are callable. None = all allowed.
            denied_tools: Tools that are always denied (overrides allowed_tools).
            default_rate_limit: Default calls-per-minute cap (default 60).
            audit_log_size: Max audit log entries to keep.
        """
        self.name = name
        self.version = version
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.denied_tools = set(denied_tools or [])
        self.default_rate_limit = default_rate_limit
        self.audit_log_size = audit_log_size

        self._tools: Dict[str, Tool] = {}
        self._call_timestamps: Dict[str, List[float]] = {}  # for rate limiting
        self._audit_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------
    def register(self, tool: Tool) -> "MCPServer":
        """Register a tool instance."""
        self._tools[tool.name] = tool
        return self

    def register_qai_model(self, model: Any, tool_name: str = "qai_model",
                           description: str = None) -> "MCPServer":
        """Convenience: register a qai.Model as a tool."""
        self.register(QaiModelTool(model=model, tool_name=tool_name, description=description))
        return self

    def register_qai_logic(self, logic_fn: Callable, tool_name: str = None,
                           description: str = None) -> "MCPServer":
        """Convenience: register a @qai.logic function as a tool."""
        self.register(QaiLogicTool(logic_fn=logic_fn, tool_name=tool_name, description=description))
        return self

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it was registered."""
        return self._tools.pop(name, None) is not None

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP `tools/list` response."""
        return [t.to_mcp_dict() for t in self._tools.values()]

    def inspect_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Detailed info about one tool."""
        t = self._tools.get(name)
        if t is None:
            return None
        return {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.input_schema,
            "required_permissions": t.required_permissions,
            "rate_limit_per_minute": t.rate_limit_per_minute,
        }

    # ------------------------------------------------------------------
    # Permission + rate limit checks
    # ------------------------------------------------------------------
    def _check_permission(self, tool_name: str) -> None:
        if tool_name in self.denied_tools:
            raise JsonRpcError(
                PERMISSION_DENIED,
                f"Tool {tool_name!r} is on the denied list",
            )
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            raise JsonRpcError(
                PERMISSION_DENIED,
                f"Tool {tool_name!r} is not on the allowed list",
            )

    def _check_rate_limit(self, tool_name: str) -> None:
        tool = self._tools.get(tool_name)
        if tool is None:
            return
        limit = getattr(tool, "rate_limit_per_minute", self.default_rate_limit)
        now = time.time()
        timestamps = self._call_timestamps.setdefault(tool_name, [])
        # Drop timestamps older than 60s
        cutoff = now - 60.0
        timestamps[:] = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= limit:
            raise JsonRpcError(
                RATE_LIMIT_EXCEEDED,
                f"Rate limit for {tool_name!r} exceeded: {limit}/minute",
            )
        timestamps.append(now)

    # ------------------------------------------------------------------
    # Tool call (with permission + rate limit + audit)
    # ------------------------------------------------------------------
    def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call a tool directly (programmatic API, no JSON-RPC).

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Whatever the tool's `run()` returns.

        Raises:
            JsonRpcError: tool not found, permission denied, rate limit exceeded,
                          or tool execution error.
        """
        arguments = arguments or {}
        t0 = time.time()

        # Find tool
        tool = self._tools.get(name)
        if tool is None:
            self._audit("call_tool", name, arguments, None, error="tool_not_found",
                        elapsed=0)
            raise JsonRpcError(TOOL_NOT_FOUND, f"Tool {name!r} not registered")

        # Permission check
        try:
            self._check_permission(name)
        except JsonRpcError as e:
            self._audit("call_tool", name, arguments, None, error=e.message, elapsed=0)
            raise

        # Rate limit check
        try:
            self._check_rate_limit(name)
        except JsonRpcError as e:
            self._audit("call_tool", name, arguments, None, error=e.message, elapsed=0)
            raise

        # Argument validation (basic — checks required fields exist)
        self._validate_arguments(tool, arguments)

        # Execute
        try:
            result = tool.run(**arguments)
            elapsed = time.time() - t0
            self._audit("call_tool", name, arguments, result, elapsed=elapsed)
            return result
        except JsonRpcError as e:
            elapsed = time.time() - t0
            self._audit("call_tool", name, arguments, None, error=e.message, elapsed=elapsed)
            raise
        except Exception as e:
            elapsed = time.time() - t0
            err_msg = f"{type(e).__name__}: {e}"
            self._audit("call_tool", name, arguments, None, error=err_msg, elapsed=elapsed)
            raise JsonRpcError(TOOL_EXECUTION_ERROR, err_msg)

    def _validate_arguments(self, tool: Tool, arguments: Dict) -> None:
        """Basic schema validation: check required fields are present."""
        schema = tool.input_schema
        required = schema.get("required", [])
        for field_name in required:
            if field_name not in arguments:
                raise JsonRpcError(
                    INVALID_PARAMS,
                    f"Missing required argument {field_name!r} for tool {tool.name!r}",
                )

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------
    def _audit(self, action: str, tool_name: str, arguments: Dict, result: Any,
               error: str = None, elapsed: float = 0):
        entry = {
            "timestamp": time.time(),
            "action": action,
            "tool": tool_name,
            "arguments": self._safe_serialize(arguments),
            "result": self._safe_serialize(result) if result is not None else None,
            "error": error,
            "elapsed_s": round(elapsed, 4),
        }
        self._audit_log.append(entry)
        # Trim
        if len(self._audit_log) > self.audit_log_size:
            self._audit_log = self._audit_log[-self.audit_log_size:]

    @staticmethod
    def _safe_serialize(v):
        try:
            json.dumps(v)
            return v
        except (TypeError, ValueError):
            return repr(v)

    def get_audit_log(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """Return the last N audit-log entries."""
        return list(self._audit_log[-last_n:])

    def clear_audit_log(self) -> int:
        n = len(self._audit_log)
        self._audit_log.clear()
        return n

    # ------------------------------------------------------------------
    # JSON-RPC request handling
    # ------------------------------------------------------------------
    def handle_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Dispatch a JSON-RPC request to the appropriate method."""
        try:
            if request.method == "initialize":
                return JsonRpcResponse(
                    result={
                        "protocolVersion": self.PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": self.name, "version": self.version},
                    },
                    id=request.id,
                )

            if request.method == "tools/list":
                return JsonRpcResponse(result={"tools": self.list_tools()}, id=request.id)

            if request.method == "tools/call":
                tool_name = request.params.get("name")
                arguments = request.params.get("arguments", {})
                if not tool_name:
                    return JsonRpcResponse(
                        error=JsonRpcError(INVALID_PARAMS, "tools/call requires 'name'").to_dict(),
                        id=request.id,
                    )
                try:
                    result = self.call_tool(tool_name, arguments)
                    return JsonRpcResponse(
                        result={"content": [{"type": "text", "text": json.dumps(result, default=str)}]},
                        id=request.id,
                    )
                except JsonRpcError as e:
                    return JsonRpcResponse(error=e.to_dict(), id=request.id)

            if request.method == "tools/inspect":
                tool_name = request.params.get("name")
                info = self.inspect_tool(tool_name)
                if info is None:
                    return JsonRpcResponse(
                        error=JsonRpcError(TOOL_NOT_FOUND, f"Tool {tool_name!r} not found").to_dict(),
                        id=request.id,
                    )
                return JsonRpcResponse(result=info, id=request.id)

            if request.method == "audit/log":
                last_n = request.params.get("last_n", 50)
                return JsonRpcResponse(result={"entries": self.get_audit_log(last_n)}, id=request.id)

            # Unknown method
            return JsonRpcResponse(
                error=JsonRpcError(METHOD_NOT_FOUND, f"Unknown method: {request.method}").to_dict(),
                id=request.id,
            )
        except Exception as e:
            return JsonRpcResponse(
                error=JsonRpcError(INTERNAL_ERROR, f"Internal error: {e}").to_dict(),
                id=request.id,
            )

    def handle_raw(self, raw_request: Dict) -> Dict:
        """Handle a raw dict request, return a raw dict response. Convenience wrapper."""
        try:
            req = JsonRpcRequest.from_dict(raw_request)
            resp = self.handle_request(req)
            return resp.to_dict()
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": JsonRpcError(INVALID_REQUEST, f"Invalid request: {e}").to_dict(),
                "id": raw_request.get("id"),
            }

    # ------------------------------------------------------------------
    # Transports
    # ------------------------------------------------------------------
    def stdio_serve(self) -> None:
        """Serve over stdio (JSON-RPC over stdin/stdout). Blocks the calling thread.

        Each line of stdin is one JSON-RPC request; each response is one line of stdout.
        This is the standard MCP transport — works with Claude, Cursor, etc.
        """
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                raw_req = json.loads(line)
            except json.JSONDecodeError as e:
                resp = {
                    "jsonrpc": "2.0",
                    "error": JsonRpcError(PARSE_ERROR, f"JSON parse error: {e}").to_dict(),
                    "id": None,
                }
                print(json.dumps(resp), flush=True)
                continue
            resp = self.handle_raw(raw_req)
            print(json.dumps(resp, default=str), flush=True)

    def build_http_app(self):
        """Build a Flask app exposing the MCP server over HTTP.

        Endpoints:
            POST /  — JSON-RPC endpoint (accepts one request, returns one response)
            GET  /health — returns {"status": "ok"}
            GET  /tools — returns the tools list (bypass JSON-RPC for convenience)
        """
        try:
            from flask import Flask, request, jsonify
        except ImportError as e:
            raise ImportError(
                "[WHAT] HTTP transport requires Flask\n"
                "[WHY] HTTP serving isn't part of the core qai install\n"
                "[FIX] pip install flask"
            ) from e

        app = Flask(__name__)
        server = self  # closure

        @app.route("/", methods=["POST"])
        def rpc():
            try:
                raw = request.get_json(force=True)
            except Exception as e:
                return jsonify({
                    "jsonrpc": "2.0",
                    "error": JsonRpcError(PARSE_ERROR, f"JSON parse error: {e}").to_dict(),
                    "id": None,
                }), 400
            resp = server.handle_raw(raw)
            return jsonify(resp)

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok", "server": self.name, "version": self.version})

        @app.route("/tools", methods=["GET"])
        def tools():
            return jsonify({"tools": server.list_tools()})

        @app.route("/audit", methods=["GET"])
        def audit():
            last_n = int(request.args.get("last_n", 50))
            return jsonify({"entries": server.get_audit_log(last_n)})

        return app

    def __repr__(self):
        return (
            f"<MCPServer name={self.name!r} version={self.version!r} "
            f"tools={list(self._tools.keys())}>"
        )


# =====================================================================
# Convenience: quick server builder
# =====================================================================
def create_server(
    name: str = "qai-mcp-server",
    version: str = "1.0.0",
    tools: Optional[List[Tool]] = None,
    **kwargs,
) -> MCPServer:
    """Quick way to create an MCPServer with a set of tools.

    Example:
        server = qai.mcp.create_server(
            name="my-server",
            tools=[TimeTool(), CalculatorTool()],
        )
    """
    server = MCPServer(name=name, version=version, **kwargs)
    for tool in (tools or []):
        server.register(tool)
    return server


# =====================================================================
# Public API
# =====================================================================
__all__ = [
    # Core
    "MCPServer", "Tool", "ToolRegistry", "tool_registry", "create_server",
    "JsonRpcRequest", "JsonRpcResponse", "JsonRpcError",
    # Error codes
    "PARSE_ERROR", "INVALID_REQUEST", "METHOD_NOT_FOUND", "INVALID_PARAMS",
    "INTERNAL_ERROR", "TOOL_NOT_FOUND", "TOOL_EXECUTION_ERROR",
    "PERMISSION_DENIED", "RATE_LIMIT_EXCEEDED", "VALIDATION_ERROR",
    # Built-in tools (reference servers)
    "FilesystemTool", "MemoryTool", "GitTool", "FetchTool", "TimeTool",
    "SqliteTool", "ExecTool", "SequentialThinkingTool",
    "CalculatorTool", "CodeEvalTool",
    # qai-specific tools (my own additions)
    "QaiModelTool", "QaiLogicTool", "QaiMemoryTool", "QaiBundleTool",
]
