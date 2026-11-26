"""
qai.api_reference -- API reference auto-generator for the qai package.

Walks the entire ``qai`` package using AST parsing (no imports of qai
submodules are performed at scan time -- this keeps the scan fast and
side-effect-free even when qai has heavy optional dependencies).  For
every public module it extracts classes, functions and constants, pulls
their docstrings, signatures and type hints, and renders structured
Markdown (or HTML with syntax highlighting) reference documentation.

Quickstart::

    from qai import api_reference

    # one-shot generation
    files = api_reference.generate(format='markdown', output_dir='docs/api/')

    # or build a scanner once and reuse it
    from qai.api_reference import ApiReference
    ref = ApiReference(output_format='markdown', output_dir='docs/api/')
    ref.generate()
    ref.stats()
    ref.search('neural')

The module is stdlib-only (``ast``, ``inspect``, ``json``, ``os``, ``re``,
``html``, ``typing``).
"""
from __future__ import annotations

import ast
import html
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    "ApiEntry",
    "ApiScanner",
    "MarkdownGenerator",
    "HtmlGenerator",
    "ApiReference",
    "generate",
    "stats",
    "search",
    "find_class",
    "find_function",
]


# ---------------------------------------------------------------------------
# Minimal stdlib-only helpers (replacements for difflib.SequenceMatcher and
# textwrap.dedent -- kept in-module so the public surface is fully stdlib).
# ---------------------------------------------------------------------------

def _dedent(text: str) -> str:
    """Remove the longest common leading whitespace from all non-blank lines."""
    if not text:
        return text
    lines = text.split("\n")
    indents: List[int] = []
    for ln in lines:
        if not ln.strip():
            continue
        stripped = ln.lstrip(" \t")
        indents.append(len(ln) - len(stripped))
    if not indents:
        return text
    common = min(indents)
    if common == 0:
        return text
    out_lines = []
    for ln in lines:
        if ln[:common].strip():
            # Mixed tabs/spaces edge case -- leave as-is.
            out_lines.append(ln)
        else:
            out_lines.append(ln[common:])
    return "\n".join(out_lines)


def _similarity(a: str, b: str) -> float:
    """Return a 0..1 similarity ratio between two strings.

    A tiny implementation of the Gestalt pattern-matching ratio (the same
    algorithm ``difflib.SequenceMatcher.ratio`` uses, with a simpler
    implementation that is good enough for typo-tolerant search).
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    # Count matching blocks via a simple two-pointer scan.
    matches = 0
    ai = 0
    bj = 0
    # Use a quick longest-common-subsequence-like scan.
    # Build a 2D table only if both strings are short; otherwise use
    # a positional match-count heuristic.
    if la <= 256 and lb <= 256:
        # Standard LCS via two-row DP.
        prev = [0] * (lb + 1)
        for i in range(1, la + 1):
            cur = [0] * (lb + 1)
            ai_ch = a[i - 1]
            for j in range(1, lb + 1):
                if ai_ch == b[j - 1]:
                    cur[j] = prev[j - 1] + 1
                else:
                    cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
            prev = cur
        matches = prev[lb]
        return (2.0 * matches) / (la + lb)
    # Fallback: count shared characters (case-insensitive position-bucketed).
    return 0.0


# ===========================================================================
# Docstring parsing helpers
# ===========================================================================

_SECTION_HEADERS_GOOGLE = {
    "args", "arguments", "parameters", "params",
    "returns", "return", "yields", "yield",
    "raises", "raise",
    "examples", "example",
    "note", "notes",
    "see also", "see",
    "warns", "warning", "warnings",
    "attributes",
    "references",
    "todo",
}

_SECTION_HEADERS_NUMPY = {
    "parameters", "returns", "yields", "raises", "examples",
    "notes", "see also", "warnings", "attributes", "references", "todo",
}


def _split_sections(docstring: str) -> Tuple[str, str, Dict[str, str]]:
    """Split a docstring into (brief, body, sections_dict).

    Recognises both Google-style ("Args:") and NumPy-style ("Parameters\\n---")
    section headers.  The brief is the first non-empty line; the body is
    everything before the first section; sections_dict maps lowercase section
    name -> raw text body (with common leading whitespace removed so that
    parsers can treat the first arg line as the base indent).
    """
    if not docstring:
        return "", "", {}
    # Normalise newlines.
    text = docstring.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    # Brief = first non-empty line.
    brief = ""
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx < len(lines):
        brief = lines[idx].strip()
    # Find section headers.
    sections: Dict[str, str] = {}
    body_lines: List[str] = []
    i = 0
    current_section: Optional[str] = None
    current_buf: List[str] = []
    section_start_idx = -1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        low = stripped.lower().rstrip(":").strip()
        # Google-style header: "Args:" or "Args :"
        is_google = bool(re.match(r"^[A-Z][A-Za-z ]+:\s*$", stripped))
        # NumPy-style: a header word on its own line followed by dashes.
        is_numpy = False
        if (
            i + 1 < len(lines)
            and stripped
            and low in _SECTION_HEADERS_NUMPY
            and re.match(r"^\s*-{3,}\s*$", lines[i + 1])
        ):
            is_numpy = True
        if is_google or is_numpy:
            # flush previous section
            if current_section is not None:
                sections[current_section] = _dedent(
                    "\n".join(current_buf)
                ).strip("\n")
            elif section_start_idx == -1:
                # first section discovered -- body is everything seen so far
                body_lines = lines[:i]
            current_section = low
            current_buf = []
            section_start_idx = i
            i += 2 if is_numpy else 1
            continue
        if current_section is not None:
            current_buf.append(line)
        i += 1
    # Flush final section.
    if current_section is not None:
        sections[current_section] = _dedent(
            "\n".join(current_buf)
        ).strip("\n")
    else:
        body_lines = lines
    body = "\n".join(body_lines).strip()
    return brief, body, sections


_ARG_LINE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<name>\*{0,2}[\w.]+)"
    r"(?:\s*\((?P<type>[^)]*)\))?"
    r"(?:\s*:\s*(?P<desc>.*))?$"
)


def _parse_args_section(text: str) -> List[Dict[str, str]]:
    """Parse a Google-style Args section into a list of parameter dicts.

    Each new arg starts at the same base indent; continuation lines are more
    deeply indented.  Works with both Google-style (``name: desc``) and the
    ``name (type): desc`` variant.
    """
    if not text:
        return []
    lines = text.split("\n")
    # First pass: find the indent of the first line that looks like an arg
    # definition (matches the name pattern with a colon after it).  That
    # indent becomes the "base indent" -- lines at this indent that match
    # the pattern start a new arg; lines at greater indent are continuations.
    base_indent: Optional[int] = None
    for ln in lines:
        if not ln.strip():
            continue
        m = _ARG_LINE_RE.match(ln)
        if not m or not m.group("name"):
            continue
        after_name = ln[m.end("name"):]
        if ":" not in after_name:
            continue
        base_indent = len(ln) - len(ln.lstrip(" "))
        break

    out: List[Dict[str, str]] = []
    cur_name: Optional[str] = None
    cur_type = ""
    cur_desc: List[str] = []
    blocks: List[Tuple[str, str, str]] = []
    for line in lines:
        if not line.strip():
            if cur_name is not None:
                cur_desc.append("")
            continue
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        m = _ARG_LINE_RE.match(line)
        is_new_arg = (
            m is not None
            and m.group("name")
            and (base_indent is None or indent <= base_indent)
        )
        if is_new_arg and m is not None:
            after_name = line[m.end("name"):]
            has_colon = ":" in after_name
            if has_colon or cur_name is None:
                if cur_name is not None:
                    blocks.append((cur_name, cur_type, "\n".join(cur_desc).strip()))
                cur_name = m.group("name")
                cur_type = (m.group("type") or "").strip()
                cur_desc = [m.group("desc") or ""]
                if base_indent is None:
                    base_indent = indent
                continue
        # Otherwise, continuation of current arg description.
        if cur_name is not None:
            cur_desc.append(line.strip())
    if cur_name is not None:
        blocks.append((cur_name, cur_type, "\n".join(cur_desc).strip()))
    for name, typ, desc in blocks:
        out.append({"name": name.strip(), "type": typ, "description": desc.strip()})
    return out


def _parse_raises_section(text: str) -> List[str]:
    """Parse a Raises section into a list of 'ExceptionType: description' strings."""
    if not text:
        return []
    out: List[str] = []
    cur: Optional[str] = None
    cur_desc: List[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        # Match "ValueError: if ..." or "ValueError\n    description"
        m = re.match(r"^([A-Za-z_][\w.]*)\s*:\s*(.*)$", s)
        if m and not line.startswith(" "):
            if cur:
                out.append(f"{cur}: {' '.join(cur_desc).strip()}".rstrip(": ").strip())
            cur = m.group(1)
            cur_desc = [m.group(2)] if m.group(2) else []
        else:
            # NumPy-style: line is just the type, description on indented lines.
            if cur is None and re.match(r"^[A-Za-z_][\w.]*$", s):
                cur = s
                cur_desc = []
            elif cur is not None:
                cur_desc.append(s)
    if cur:
        out.append(f"{cur}: {' '.join(cur_desc).strip()}".rstrip(": ").strip())
    return [r for r in out if r]


def _parse_examples_section(text: str) -> List[str]:
    """Extract code examples from an Examples section."""
    if not text:
        return []
    blocks: List[str] = []
    cur: List[str] = []
    in_code = False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            if in_code:
                blocks.append("\n".join(cur))
                cur = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            cur.append(line)
        elif s.startswith(">>>") or s.startswith("..."):
            # doctest block
            cur.append(s[4:] if s.startswith(">>> ") else s.lstrip(".> "))
            in_code = True
        elif in_code and (not s):
            blocks.append("\n".join(cur).rstrip())
            cur = []
            in_code = False
    if cur:
        blocks.append("\n".join(cur).rstrip())
    return [b for b in blocks if b.strip()]


# ===========================================================================
# AST helpers
# ===========================================================================

def _safe_unparse(node: Optional[ast.AST]) -> str:
    """Best-effort unparse of an AST node (Python 3.9+)."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:
        # Fallback: name-based rendering for common node types.
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Attribute):
            return f"{_safe_unparse(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return f"{_safe_unparse(node.value)}[{_safe_unparse(node.slice)}]"
        return ""


def _args_to_params(node: ast.FunctionDef) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Convert a FunctionDef into (parameters, signature_str, return_annotation).

    parameters: list of dicts {name, annotation, default, kind} where kind is
    one of 'positional', 'positional_only', 'var_positional', 'keyword_only',
    'var_keyword'.
    """
    args = node.args
    params: List[Dict[str, Any]] = []

    # Positional-only args
    pos_only = list(args.posonlyargs)
    regular = list(args.args)
    defaults = list(args.defaults)
    # Defaults align to the END of (pos_only + regular).
    n_all_pos = len(pos_only) + len(regular)
    n_defaults = len(defaults)
    first_default_idx = n_all_pos - n_defaults

    def _default_for(i: int) -> str:
        if i >= first_default_idx:
            return _safe_unparse(defaults[i - first_default_idx])
        return ""

    pos_kind = "positional_only"
    for i, a in enumerate(pos_only):
        params.append({
            "name": a.arg,
            "annotation": _safe_unparse(a.annotation),
            "default": _default_for(i),
            "kind": pos_kind,
        })

    pos_kind = "positional"
    for i, a in enumerate(regular):
        idx = len(pos_only) + i
        params.append({
            "name": a.arg,
            "annotation": _safe_unparse(a.annotation),
            "default": _default_for(idx),
            "kind": pos_kind,
        })

    if args.vararg is not None:
        params.append({
            "name": f"*{args.vararg.arg}",
            "annotation": _safe_unparse(args.vararg.annotation),
            "default": "",
            "kind": "var_positional",
        })

    # Keyword-only args
    for i, a in enumerate(args.kwonlyargs):
        d = ""
        if args.kw_defaults and args.kw_defaults[i] is not None:
            d = _safe_unparse(args.kw_defaults[i])
        params.append({
            "name": a.arg,
            "annotation": _safe_unparse(a.annotation),
            "default": d,
            "kind": "keyword_only",
        })

    if args.kwarg is not None:
        params.append({
            "name": f"**{args.kwarg.arg}",
            "annotation": _safe_unparse(args.kwarg.annotation),
            "default": "",
            "kind": "var_keyword",
        })

    # Build signature string.
    sig_parts: List[str] = []
    for p in params:
        s = p["name"]
        if p["annotation"]:
            s += f": {p['annotation']}"
        if p["default"]:
            s += f" = {p['default']}"
        sig_parts.append(s)
    # Handle bare * marker (positional vs keyword) if no vararg but kwonly exists.
    if args.vararg is None and args.kwonlyargs and not pos_only:
        # Insert a bare '*' before kwonly entries.
        # Find first kwonly index
        for i, p in enumerate(params):
            if p["kind"] == "keyword_only":
                params.insert(i, {"name": "*", "annotation": "", "default": "", "kind": "marker"})
                sig_parts.insert(i, "*")
                break
    # Handle bare / marker for positional-only if mixed.
    if pos_only and regular:
        # Insert '/' after last positional-only entry.
        for i, p in enumerate(params):
            if p["kind"] == "positional_only" and (
                i + 1 >= len(params) or params[i + 1]["kind"] != "positional_only"
            ):
                params.insert(i + 1, {"name": "/", "annotation": "", "default": "", "kind": "marker"})
                sig_parts.insert(i + 1, "/")
                break

    sig_str = f"({', '.join(sig_parts)})"
    ret = _safe_unparse(node.returns) or None
    return params, sig_str, ret


def _decorators(node: ast.AST) -> List[str]:
    """Return list of decorator names for a class/function."""
    out: List[str] = []
    for d in getattr(node, "decorator_list", []) or []:
        if isinstance(d, ast.Name):
            out.append(d.id)
        elif isinstance(d, ast.Attribute):
            out.append(_safe_unparse(d))
        elif isinstance(d, ast.Call):
            out.append(_safe_unparse(d.func))
        else:
            out.append(_safe_unparse(d))
    return out


def _is_constant_assign(node: ast.Assign) -> bool:
    """Heuristic: is this a module-level constant assignment?

    Treats simple Name targets whose id is UPPER_CASE (with at least one
    underscore OR all chars uppercase) as constants.  Skips dunder names
    like __all__, __version__.
    """
    if not node.targets:
        return False
    for t in node.targets:
        if not isinstance(t, ast.Name):
            return False
        name = t.id
        if name.startswith("__") and name.endswith("__"):
            return False
        if name.startswith("_"):
            return False
        # UPPER_CASE_WITH_UNDERSCORES, or short all-caps, OR a single ALL_CAPS word.
        if name.isupper():
            return True
        # Mixed: like 'PI' (upper) or 'MAX_LEN'
        if re.match(r"^[A-Z][A-Z0-9_]+$", name):
            return True
        return False
    return True


# ===========================================================================
# ApiEntry
# ===========================================================================

class ApiEntry:
    """A single API entry -- class, function, or constant.

    Fields:
        name:           The simple name (e.g. ``ContinuousLearner``).
        kind:           ``'class'`` | ``'function'`` | ``'constant'``.
        module:         Dotted module path (e.g. ``qai.self_improving``).
        signature:      Source signature string, e.g. ``(x: int = 1) -> bool``.
        docstring:      Full docstring (with leading/trailing whitespace stripped).
        line_number:    1-based source line where the entry is defined.
        source_file:    Path of the source file relative to the package root.
        type_hints:     Dict mapping parameter name -> type hint string.
        parameters:     List of param dicts {name, annotation, default, kind}.
        returns:        Return type string, or None.
        raises:         List of 'ExceptionType: description' strings parsed
                        from the docstring Raises section.
        bases:          (classes) list of base class names.
        methods:        (classes) list of method dicts {name, signature, docstring, kind}.
        value:          (constants) source representation of the value.
        brief:          First line of the docstring.
        body:           Long description (docstring minus brief, sections).
        examples:       List of example code blocks parsed from the docstring.
    """

    __slots__ = (
        "name", "kind", "module", "signature", "docstring", "line_number",
        "source_file", "type_hints", "parameters", "returns", "raises",
        "bases", "methods", "value", "brief", "body", "examples",
    )

    def __init__(
        self,
        name: str,
        kind: str,
        module: str,
        signature: str = "",
        docstring: Optional[str] = None,
        line_number: int = 0,
        source_file: str = "",
        type_hints: Optional[Dict[str, str]] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
        returns: Optional[str] = None,
        raises: Optional[List[str]] = None,
        bases: Optional[List[str]] = None,
        methods: Optional[List[Dict[str, Any]]] = None,
        value: Optional[str] = None,
        brief: Optional[str] = None,
        body: Optional[str] = None,
        examples: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.module = module
        self.signature = signature or ""
        self.docstring = docstring or ""
        self.line_number = int(line_number or 0)
        self.source_file = source_file or ""
        self.type_hints = dict(type_hints or {})
        self.parameters = list(parameters or [])
        self.returns = returns
        self.raises = list(raises or [])
        self.bases = list(bases or [])
        self.methods = list(methods or [])
        self.value = value
        self.brief = brief or ""
        self.body = body or ""
        self.examples = list(examples or [])

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation suitable for JSON serialisation."""
        return {
            "name": self.name,
            "kind": self.kind,
            "module": self.module,
            "signature": self.signature,
            "docstring": self.docstring,
            "line_number": self.line_number,
            "source_file": self.source_file,
            "type_hints": dict(self.type_hints),
            "parameters": list(self.parameters),
            "returns": self.returns,
            "raises": list(self.raises),
            "bases": list(self.bases),
            "methods": list(self.methods),
            "value": self.value,
            "brief": self.brief,
            "body": self.body,
            "examples": list(self.examples),
        }

    def to_markdown(self) -> str:
        """Return a Markdown rendering of this entry."""
        lines: List[str] = []
        anchor = self.name.lower().replace(" ", "-")
        kind_badge = self.kind.upper()
        lines.append(f'<a id="{anchor}"></a>')
        lines.append(f"### `{self.name}`  ")
        lines.append(f"`{kind_badge}` | `{self.module}.{self.name}`")
        lines.append("")

        if self.kind == "class":
            bases = ", ".join(self.bases) if self.bases else ""
            cls_decl = f"class {self.name}"
            if bases:
                cls_decl += f"({bases})"
            lines.append("**Signature:**")
            lines.append("```python")
            lines.append(cls_decl)
            lines.append("```")
            lines.append("")
        elif self.kind == "function":
            sig = f"{self.name}{self.signature}"
            if self.returns:
                sig += f" -> {self.returns}"
            lines.append("**Signature:**")
            lines.append("```python")
            lines.append(sig)
            lines.append("```")
            lines.append("")
        elif self.kind == "constant":
            lines.append("**Signature:**")
            lines.append("```python")
            lines.append(f"{self.name} = {self.value or '...'}")
            lines.append("```")
            lines.append("")

        # Description.
        if self.brief:
            lines.append(f"> {self.brief}")
            lines.append("")
        if self.body:
            lines.append(self.body)
            lines.append("")

        # Parameters table.
        if self.parameters:
            real_params = [p for p in self.parameters if p.get("kind") != "marker"]
            if real_params:
                lines.append("**Parameters:**")
                lines.append("")
                lines.append("| Name | Type | Default | Description |")
                lines.append("| ---- | ---- | ------- | ----------- |")
                # Merge with docstring-derived descriptions.
                doc_args = {p["name"].lstrip("*"): p for p in _parse_args_section(self._doc_args_text())}
                for p in real_params:
                    pname = p["name"]
                    doc_p = doc_args.get(pname.lstrip("*"))
                    desc = (doc_p or {}).get("description", "")
                    ptype = p.get("annotation") or (doc_p or {}).get("type", "") or ""
                    default = p.get("default", "")
                    if default == "" and p.get("kind") not in ("positional", "positional_only"):
                        default = ""
                    lines.append(
                        f"| `{pname}` | `{ptype}` | `{default}` | {desc} |"
                    )
                lines.append("")

        # Return type.
        if self.kind == "function" and self.returns:
            lines.append(f"**Returns:** `{self.returns}`")
            ret_desc = self._doc_returns_text()
            if ret_desc:
                lines.append("")
                lines.append(ret_desc)
            lines.append("")

        # Raises.
        if self.raises:
            lines.append("**Raises:**")
            lines.append("")
            for r in self.raises:
                lines.append(f"- `{r}`")
            lines.append("")

        # Methods (for classes).
        if self.methods:
            lines.append("**Methods:**")
            lines.append("")
            lines.append("| Method | Signature | Description |")
            lines.append("| ------ | --------- | ----------- |")
            for m in self.methods:
                mname = m.get("name", "")
                msig = m.get("signature", "")
                mdesc = (m.get("docstring", "") or "").split("\n", 1)[0].strip()
                tag = m.get("kind", "")
                label = f"`{mname}()`"
                if tag and tag != "method":
                    label = f"`{tag} {mname}()`"
                lines.append(f"| {label} | `{msig}` | {mdesc} |")
            lines.append("")

        # Examples.
        if self.examples:
            lines.append("**Examples:**")
            lines.append("")
            for ex in self.examples:
                lines.append("```python")
                lines.append(ex.strip())
                lines.append("```")
                lines.append("")

        # Source ref.
        if self.source_file:
            loc = f"{self.source_file}:{self.line_number}" if self.line_number else self.source_file
            lines.append(f"_Defined in `{loc}`._")
            lines.append("")
        lines.append("---")
        return "\n".join(lines)

    # Internal helpers used by to_markdown -- lazy-parse docstring sections.

    def _doc_args_text(self) -> str:
        _, _, secs = _split_sections(self.docstring)
        for key in ("args", "arguments", "parameters", "params", "attributes"):
            if key in secs:
                return secs[key]
        return ""

    def _doc_returns_text(self) -> str:
        _, _, secs = _split_sections(self.docstring)
        for key in ("returns", "return"):
            if key in secs:
                return secs[key]
        return ""

    def __repr__(self) -> str:
        return f"<ApiEntry {self.kind} {self.module}.{self.name}>"


# ===========================================================================
# ApiScanner
# ===========================================================================

class ApiScanner:
    """Scan the qai package and collect its full public API surface.

    Uses AST parsing only -- it never imports qai submodules, so the scan is
    fast and has no side effects.

    Parameters:
        package_path: Path to the qai package directory.  Defaults to the
            directory containing this file (which lives inside qai/).
        exclude_private: Skip names starting with a single underscore.
        exclude_tests: Skip files/dirs named ``tests`` or starting with ``test_``.
    """

    def __init__(
        self,
        package_path: Optional[str] = None,
        exclude_private: bool = True,
        exclude_tests: bool = True,
    ) -> None:
        if package_path is None:
            package_path = os.path.dirname(os.path.abspath(__file__))
        self.package_path = os.path.abspath(package_path)
        self.exclude_private = exclude_private
        self.exclude_tests = exclude_tests
        self._entries: Dict[str, List[ApiEntry]] = {}
        self._all_entries_cache: Optional[List[ApiEntry]] = None
        self._scanned = False

    # ------------------------------------------------------------------ #
    # Walking
    # ------------------------------------------------------------------ #

    def _module_dotted_name(self, file_path: str) -> str:
        """Compute the dotted module name for a .py file inside the package."""
        pkg_root = os.path.dirname(self.package_path)  # parent of qai/
        rel = os.path.relpath(file_path, pkg_root)
        if rel.endswith(os.sep + "__init__.py"):
            rel = rel[: -len(os.sep + "__init__.py")]
        elif rel.endswith("__init__.py"):
            rel = rel[: -len("__init__.py")]
        else:
            rel = rel[:-3] if rel.endswith(".py") else rel
        rel = rel.replace(os.sep, ".")
        return rel

    def _iter_py_files(self) -> Iterable[str]:
        skip_dirs = {"__pycache__"}
        if self.exclude_tests:
            skip_dirs.update({"tests", "test", "_tests"})
        for root, dirs, files in os.walk(self.package_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in files:
                if not f.endswith(".py"):
                    continue
                if self.exclude_tests and (
                    f.startswith("test_") or f == "conftest.py"
                ):
                    continue
                if f == "__main__.py":
                    # Still scan __main__.py -- it is sometimes part of the
                    # public API.  Comment out the next line if you want to skip.
                    pass
                yield os.path.join(root, f)

    # ------------------------------------------------------------------ #
    # AST extraction
    # ------------------------------------------------------------------ #

    def _parse_module(self, file_path: str) -> Tuple[Optional[ast.Module], str]:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                src = fh.read()
        except (OSError, UnicodeDecodeError):
            return None, ""
        try:
            tree = ast.parse(src, filename=file_path)
        except SyntaxError:
            return None, src
        return tree, src

    def _extract_methods(self, cls_node: ast.ClassDef) -> List[Dict[str, Any]]:
        methods: List[Dict[str, Any]] = []
        for node in cls_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if name.startswith("_") and name not in ("__init__", "__call__"):
                    if self.exclude_private:
                        continue
                decorators = _decorators(node)
                kind = "method"
                if "property" in decorators:
                    kind = "property"
                elif "staticmethod" in decorators:
                    kind = "staticmethod"
                elif "classmethod" in decorators:
                    kind = "classmethod"
                elif node.name == "__init__":
                    kind = "constructor"
                params, _sig, ret = _args_to_params(node)
                # Drop self/cls/marker entries from user-facing params + sig.
                if kind in ("method", "constructor", "classmethod", "property"):
                    params = [p for p in params
                              if p.get("name") not in ("self", "cls")
                              and p.get("kind") != "marker"]
                else:
                    params = [p for p in params if p.get("kind") != "marker"]
                sig = self._sig_from_params(params)
                doc = ast.get_docstring(node) or ""
                methods.append({
                    "name": name,
                    "signature": sig,
                    "docstring": doc,
                    "returns": ret or "",
                    "parameters": params,
                    "decorators": decorators,
                    "kind": kind,
                    "line_number": node.lineno,
                })
        return methods

    @staticmethod
    def _sig_from_params(params: List[Dict[str, Any]]) -> str:
        """Rebuild a signature string like ``(x: int = 1, *args, **kw)`` from params."""
        parts: List[str] = []
        for p in params:
            if p.get("kind") == "marker":
                parts.append(p["name"])
                continue
            s = p["name"]
            if p.get("annotation"):
                s += f": {p['annotation']}"
            if p.get("default"):
                s += f" = {p['default']}"
            parts.append(s)
        return f"({', '.join(parts)})"

    def _extract_from_module(self, tree: ast.Module, dotted: str, file_path: str) -> List[ApiEntry]:
        entries: List[ApiEntry] = []
        rel_path = os.path.relpath(file_path, os.path.dirname(self.package_path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                name = node.name
                if name.startswith("_") and self.exclude_private:
                    continue
                bases = [_safe_unparse(b) for b in node.bases]
                methods = self._extract_methods(node)
                # Try to find __init__ to derive class-level parameters.
                init_params: List[Dict[str, Any]] = []
                init_sig = ""
                init_ret: Optional[str] = None
                for m in methods:
                    if m.get("kind") == "constructor":
                        init_params = [p for p in m.get("parameters", [])
                                       if p.get("name") not in ("self", "cls")
                                       and p.get("kind") != "marker"]
                        init_sig = self._sig_from_params(init_params)
                        init_ret = m.get("returns") or None
                        break
                docstring = ast.get_docstring(node) or ""
                brief, body, secs = _split_sections(docstring)
                raises = _parse_raises_section(secs.get("raises", ""))
                examples = _parse_examples_section(secs.get("examples", ""))
                type_hints = {p["name"].lstrip("*"): p["annotation"]
                              for p in init_params if p.get("annotation")}
                entry = ApiEntry(
                    name=name,
                    kind="class",
                    module=dotted,
                    signature=init_sig if init_sig else "()",
                    docstring=docstring,
                    line_number=node.lineno,
                    source_file=rel_path,
                    type_hints=type_hints,
                    parameters=init_params,
                    returns=init_ret,
                    raises=raises,
                    bases=bases,
                    methods=methods,
                    brief=brief,
                    body=body,
                    examples=examples,
                )
                entries.append(entry)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if name.startswith("_") and self.exclude_private:
                    continue
                params, sig, ret = _args_to_params(node)
                docstring = ast.get_docstring(node) or ""
                brief, body, secs = _split_sections(docstring)
                raises = _parse_raises_section(secs.get("raises", ""))
                examples = _parse_examples_section(secs.get("examples", ""))
                type_hints = {p["name"].lstrip("*"): p["annotation"]
                              for p in params if p.get("annotation")}
                sig_full = f"{name}{sig}"
                if ret:
                    sig_full += f" -> {ret}"
                # Mark async functions in the docstring brief.
                is_async = isinstance(node, ast.AsyncFunctionDef)
                decorators = _decorators(node)
                entry = ApiEntry(
                    name=name,
                    kind="function",
                    module=dotted,
                    signature=sig_full,
                    docstring=docstring,
                    line_number=node.lineno,
                    source_file=rel_path,
                    type_hints=type_hints,
                    parameters=params,
                    returns=ret,
                    raises=raises,
                    brief=("async " + brief) if is_async else brief,
                    body=body,
                    examples=examples,
                )
                entry.bases = list(decorators)  # repurpose to record decorators
                entries.append(entry)
            elif isinstance(node, ast.Assign):
                if not _is_constant_assign(node):
                    continue
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    name = target.id
                    if name.startswith("_") and self.exclude_private:
                        continue
                    val_src = _safe_unparse(node.value)
                    # Try to extract a docstring from a trailing string expr
                    # immediately after the assignment.
                    doc = ""
                    parent_body = tree.body
                    idx = parent_body.index(node)
                    if idx + 1 < len(parent_body):
                        nxt = parent_body[idx + 1]
                        if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) \
                                and isinstance(nxt.value.value, str):
                            doc = nxt.value.value
                    brief, body, _ = _split_sections(doc)
                    entry = ApiEntry(
                        name=name,
                        kind="constant",
                        module=dotted,
                        signature=name,
                        docstring=doc,
                        line_number=node.lineno,
                        source_file=rel_path,
                        type_hints={},
                        parameters=[],
                        returns=None,
                        raises=[],
                        value=val_src,
                        brief=brief,
                        body=body,
                    )
                    entries.append(entry)
            # Annotated assignments (X: int = 5) -- not commonly constants,
            # but capture UPPER_CASE ones.
            elif isinstance(node, ast.AnnAssign):
                if not isinstance(node.target, ast.Name):
                    continue
                name = node.target.id
                if name.startswith("_") and self.exclude_private:
                    continue
                if not (name.isupper() or re.match(r"^[A-Z][A-Z0-9_]+$", name)):
                    continue
                val_src = _safe_unparse(node.value) if node.value is not None else ""
                ann = _safe_unparse(node.annotation)
                entry = ApiEntry(
                    name=name,
                    kind="constant",
                    module=dotted,
                    signature=name,
                    docstring="",
                    line_number=node.lineno,
                    source_file=rel_path,
                    type_hints={name: ann},
                    parameters=[],
                    returns=None,
                    raises=[],
                    value=val_src,
                    brief="",
                    body="",
                )
                entries.append(entry)
        return entries

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def scan(self) -> Dict[str, List[Dict[str, Any]]]:
        """Scan the package and return a dict ``{module: [entry_dict, ...]}``.

        Each entry dict has keys: name, kind, signature, docstring, line_number.
        The full ApiEntry objects are also stored internally for use by the
        other methods (classes(), functions(), find(), ...).
        """
        self._entries = {}
        for file_path in self._iter_py_files():
            dotted = self._module_dotted_name(file_path)
            tree, _src = self._parse_module(file_path)
            if tree is None:
                self._entries[dotted] = []
                continue
            entries = self._extract_from_module(tree, dotted, file_path)
            self._entries[dotted] = entries
        self._all_entries_cache = None
        self._scanned = True
        return {mod: [e.to_dict() for e in ents] for mod, ents in self._entries.items()}

    def _ensure_scanned(self) -> None:
        if not self._scanned:
            self.scan()

    def _all_entries(self) -> List[ApiEntry]:
        if self._all_entries_cache is not None:
            return self._all_entries_cache
        out: List[ApiEntry] = []
        for ents in self._entries.values():
            out.extend(ents)
        self._all_entries_cache = out
        return out

    def modules(self) -> List[str]:
        """Return a sorted list of all module names that contain at least one entry."""
        self._ensure_scanned()
        return sorted(m for m, ents in self._entries.items() if ents)

    def classes(self) -> List[ApiEntry]:
        """Return all class entries."""
        self._ensure_scanned()
        return [e for e in self._all_entries() if e.kind == "class"]

    def functions(self) -> List[ApiEntry]:
        """Return all function entries."""
        self._ensure_scanned()
        return [e for e in self._all_entries() if e.kind == "function"]

    def constants(self) -> List[ApiEntry]:
        """Return all constant entries."""
        self._ensure_scanned()
        return [e for e in self._all_entries() if e.kind == "constant"]

    def entries_for(self, module_name: str) -> List[ApiEntry]:
        """Return all entries defined in ``module_name``."""
        self._ensure_scanned()
        return list(self._entries.get(module_name, []))

    def stats(self) -> Dict[str, int]:
        """Return counts: total modules, classes, functions, constants, total."""
        self._ensure_scanned()
        n_mods = sum(1 for ents in self._entries.values() if ents)
        n_classes = sum(1 for e in self._all_entries() if e.kind == "class")
        n_funcs = sum(1 for e in self._all_entries() if e.kind == "function")
        n_consts = sum(1 for e in self._all_entries() if e.kind == "constant")
        return {
            "modules": n_mods,
            "classes": n_classes,
            "functions": n_funcs,
            "constants": n_consts,
            "total": n_classes + n_funcs + n_consts,
        }

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def find(self, query: str, limit: Optional[int] = None) -> List[ApiEntry]:
        """Fuzzy-search across all entries.

        Combines exact / substring matching, token overlap, docstring
        substring and a small LCS-based similarity ratio.  Returns a list
        of ``ApiEntry`` sorted by descending score.

        A result is only kept if at least one *strong* signal fires -- a
        substring hit on name/module/docstring, a token/CamelCase overlap,
        or a very high sequence-ratio (>= 0.6).  Pure low-quality ratio
        matches are filtered out so that, e.g., searching for ``"neural"``
        returns the dozens of actually-relevant entries rather than
        thousands of near-noise matches.
        """
        self._ensure_scanned()
        q = (query or "").lower().strip()
        if not q:
            return []
        q_tokens = set(re.findall(r"[a-z0-9_]+", q))
        scored: List[Tuple[float, ApiEntry]] = []
        for e in self._all_entries():
            name = e.name.lower()
            mod = e.module.lower()
            doc = (e.docstring or "").lower()
            score = 0.0
            strong = False
            if name == q:
                score += 100.0
                strong = True
            elif q in name:
                score += 60.0 + (10.0 if name.startswith(q) else 0.0)
                strong = True
            name_tokens = set(re.findall(r"[a-z0-9_]+", name))
            overlap = q_tokens & name_tokens
            if overlap:
                score += 15.0 * len(overlap)
                strong = True
            if q in mod:
                score += 20.0
                strong = True
            if q in doc:
                score += 8.0
                strong = True
            # Typo tolerance (only counts as strong if very high).
            ratio = _similarity(q, name)
            score += 30.0 * ratio
            if ratio >= 0.6:
                strong = True
            # CamelCase token match: "ContinuousLearner" -> {continuous, learner}.
            cc_tokens = set(re.findall(r"[A-Z][a-z0-9]+", e.name))
            cc_tokens = {t.lower() for t in cc_tokens}
            if q_tokens & cc_tokens:
                score += 25.0 * len(q_tokens & cc_tokens)
                strong = True
            if strong and score > 8.0:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [e for _, e in scored]
        if limit is not None:
            out = out[:limit]
        return out

    def find_class(self, name: str) -> Optional[ApiEntry]:
        """Find the first class whose name matches exactly (case-insensitive)."""
        self._ensure_scanned()
        target = (name or "").strip().lower()
        for e in self._all_entries():
            if e.kind == "class" and e.name.lower() == target:
                return e
        return None

    def find_function(self, name: str) -> Optional[ApiEntry]:
        """Find the first function whose name matches exactly (case-insensitive)."""
        self._ensure_scanned()
        target = (name or "").strip().lower()
        for e in self._all_entries():
            if e.kind == "function" and e.name.lower() == target:
                return e
        return None


# ===========================================================================
# Markdown generator
# ===========================================================================

class MarkdownGenerator:
    """Generate a per-module Markdown API reference."""

    def __init__(self, scanner: ApiScanner, output_dir: str = "docs/api/") -> None:
        self.scanner = scanner
        self.output_dir = output_dir

    # ------------------------------------------------------------------ #

    def generate_module(self, module_name: str) -> str:
        """Render the Markdown for a single module and return it as a string."""
        entries = self.scanner.entries_for(module_name)
        classes = [e for e in entries if e.kind == "class"]
        functions = [e for e in entries if e.kind == "function"]
        constants = [e for e in entries if e.kind == "constant"]
        lines: List[str] = []
        lines.append(f"# `{module_name}`")
        lines.append("")
        # Module-level brief from the source file's module docstring.
        brief = self._module_brief(module_name)
        if brief:
            lines.append(f"> {brief}")
            lines.append("")
        # Quick stats.
        lines.append("**Module summary:**")
        lines.append(
            f"- Classes: {len(classes)} | Functions: {len(functions)} | "
            f"Constants: {len(constants)}"
        )
        if entries:
            src = entries[0].source_file
            lines.append(f"- Source: `{src}`")
        lines.append("")

        # Table of contents.
        lines.append("## Table of Contents")
        lines.append("")
        if classes:
            lines.append("**Classes:**")
            for e in classes:
                anchor = e.name.lower()
                lines.append(f"- [`{e.name}`](#{anchor})")
            lines.append("")
        if functions:
            lines.append("**Functions:**")
            for e in functions:
                anchor = e.name.lower()
                lines.append(f"- [`{e.name}`](#{anchor})")
            lines.append("")
        if constants:
            lines.append("**Constants:**")
            for e in constants:
                anchor = e.name.lower()
                lines.append(f"- [`{e.name}`](#{anchor})")
            lines.append("")

        if classes:
            lines.append("## Classes")
            lines.append("")
            for e in classes:
                lines.append(e.to_markdown())
                lines.append("")

        if functions:
            lines.append("## Functions")
            lines.append("")
            for e in functions:
                lines.append(e.to_markdown())
                lines.append("")

        if constants:
            lines.append("## Constants")
            lines.append("")
            for e in constants:
                lines.append(e.to_markdown())
                lines.append("")

        return "\n".join(lines)

    def _module_brief(self, module_name: str) -> str:
        # Best-effort: read the module docstring from the source file.
        try:
            file_path = self._module_file_path(module_name)
        except Exception:
            return ""
        if not file_path or not os.path.isfile(file_path):
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                src = fh.read()
            tree = ast.parse(src, filename=file_path)
            doc = ast.get_docstring(tree) or ""
            if doc:
                return doc.split("\n", 1)[0].strip()
        except Exception:
            pass
        return ""

    def _module_file_path(self, module_name: str) -> str:
        rel = module_name.replace(".", os.sep)
        candidate = os.path.join(self.scanner.package_path, os.path.basename(rel))
        # module_name is like 'qai.foo.bar'; package_path is the qai/ dir.
        # Convert dotted name back to path under qai/.
        parts = module_name.split(".")
        if parts[0] == "qai":
            parts = parts[1:]
        candidate = os.path.join(self.scanner.package_path, *parts) + ".py"
        if os.path.isfile(candidate):
            return candidate
        # Maybe __init__.py
        candidate2 = os.path.join(self.scanner.package_path, *parts, "__init__.py")
        return candidate2

    # ------------------------------------------------------------------ #

    def generate_all(self) -> List[str]:
        """Write a Markdown file per module.  Returns the list of file paths."""
        os.makedirs(self.output_dir, exist_ok=True)
        written: List[str] = []
        for mod in self.scanner.modules():
            md = self.generate_module(mod)
            slug = mod.replace(".", "_")
            path = os.path.join(self.output_dir, f"{slug}.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(md)
            written.append(path)
        # Index + search index.
        idx_path = self.generate_index()
        written.append(idx_path)
        search_path = self.generate_search_index()
        written.append(search_path)
        return written

    def generate_index(self) -> str:
        """Write the top-level Markdown index and return its file path."""
        stats = self.scanner.stats()
        mods = self.scanner.modules()
        lines: List[str] = []
        lines.append("# Quantum AI Framework — API Reference")
        lines.append("")
        lines.append("## Statistics")
        lines.append("")
        lines.append(f"- **Modules:** {stats['modules']}")
        lines.append(f"- **Classes:** {stats['classes']}")
        lines.append(f"- **Functions:** {stats['functions']}")
        lines.append(f"- **Constants:** {stats['constants']}")
        lines.append(f"- **Total entries:** {stats['total']}")
        lines.append("")
        lines.append("## Modules")
        lines.append("")
        for m in mods:
            slug = m.replace(".", "_")
            brief = self._module_brief(m)
            tail = f" — {brief}" if brief else ""
            lines.append(f"- [`{m}`]({slug}.md){tail}")
        lines.append("")
        lines.append("## Search")
        lines.append("")
        lines.append("A JSON search index is available at [`search_index.json`](search_index.json).")
        lines.append("")
        path = os.path.join(self.output_dir, "index.md")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return path

    def generate_search_index(self) -> str:
        """Write a JSON search index file and return its path."""
        os.makedirs(self.output_dir, exist_ok=True)
        entries = []
        for e in self.scanner.classes() + self.scanner.functions() + self.scanner.constants():
            entries.append({
                "name": e.name,
                "kind": e.kind,
                "module": e.module,
                "signature": e.signature,
                "brief": e.brief,
                "docstring": (e.docstring or "")[:500],
                "source_file": e.source_file,
                "line_number": e.line_number,
            })
        path = os.path.join(self.output_dir, "search_index.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2)
        return path


# ===========================================================================
# HTML generator
# ===========================================================================

class HtmlGenerator:
    """Generate a per-module HTML API reference with syntax highlighting.

    Uses only stdlib: HTML is built via ``html.escape`` and a small CSS/JS
    snippet for syntax highlighting and client-side search filtering.
    """

    def __init__(self, scanner: ApiScanner, output_dir: str = "docs/api_html/") -> None:
        self.scanner = scanner
        self.output_dir = output_dir

    # ------------------------------------------------------------------ #
    # Syntax highlighting (very small, regex-based)
    # ------------------------------------------------------------------ #

    _PY_KEYWORDS = (
        "False None True and as assert async await break class continue def "
        "del elif else except finally for from global if import in is lambda "
        "nonlocal not or pass raise return try while with yield self cls"
    ).split()

    def _highlight_python(self, code: str) -> str:
        """Very small syntax highlighter for Python code -> HTML string."""
        if not code:
            return ""
        # Escape first.
        out = html.escape(code)
        # Triple-quoted strings (either single or double quotes).
        out = re.sub(
            r"(&#39;&#39;&#39;.*?&#39;&#39;&#39;|&quot;&quot;&quot;.*?&quot;&quot;&quot;)",
            r'<span class="tok-str">\1</span>', out, flags=re.DOTALL,
        )
        # Single-line single-quoted strings.
        out = re.sub(
            r"(&#39;(?:\\.|[^\\&#39;\n])*&#39;)",
            r'<span class="tok-str">\1</span>', out,
        )
        # Single-line double-quoted strings.
        out = re.sub(
            r"(&quot;(?:\\.|[^\\&quot;\n])*&quot;)",
            r'<span class="tok-str">\1</span>', out,
        )
        # Comments.
        out = re.sub(r"(#[^\n]*)", r'<span class="tok-com">\1</span>', out)
        # Keywords (word boundaries).
        kw_re = "|".join(re.escape(k) for k in self._PY_KEYWORDS)
        out = re.sub(rf"\b({kw_re})\b", r'<span class="tok-kw">\1</span>', out)
        # Numbers.
        out = re.sub(r"\b(\d+\.?\d*)\b", r'<span class="tok-num">\1</span>', out)
        # Decorators.
        out = re.sub(r"(^|\n)(@\w[\w.]*)", r'\1<span class="tok-dec">\2</span>', out)
        return out

    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return """
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               margin: 0; color: #1f2328; background: #f6f8fa; }
        .layout { display: flex; min-height: 100vh; }
        .sidebar { width: 280px; background: #ffffff; border-right: 1px solid #d0d7de;
                   overflow-y: auto; padding: 16px; position: sticky; top: 0; height: 100vh; }
        .sidebar h1 { font-size: 18px; margin: 0 0 8px; }
        .sidebar .search { width: 100%; padding: 6px 8px; border: 1px solid #d0d7de;
                           border-radius: 4px; margin-bottom: 12px; font-size: 13px; }
        .sidebar ul { list-style: none; padding: 0; margin: 0 0 12px; }
        .sidebar li { padding: 2px 0; font-size: 13px; }
        .sidebar li a { color: #0969da; text-decoration: none; }
        .sidebar li a:hover { text-decoration: underline; }
        .sidebar .pkg { font-weight: 600; color: #24292f; margin-top: 8px; }
        .sidebar .pkg.collapsed > ul { display: none; }
        .main { flex: 1; padding: 32px 48px; max-width: 1100px; }
        h1 { font-size: 28px; border-bottom: 1px solid #d0d7de; padding-bottom: 8px; }
        h2 { font-size: 22px; margin-top: 32px; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }
        h3 { font-size: 18px; margin-top: 24px; }
        h4 { font-size: 15px; margin-top: 16px; color: #57606a; }
        pre { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px;
              padding: 12px 14px; overflow-x: auto; font-size: 13px; line-height: 1.5;
              font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace; }
        code { background: #eff1f3; padding: 1px 5px; border-radius: 4px; font-size: 90%; }
        pre code { background: transparent; padding: 0; }
        table { border-collapse: collapse; margin: 12px 0; width: 100%; }
        th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; font-size: 13px; }
        th { background: #f6f8fa; }
        .kind-badge { display: inline-block; padding: 1px 6px; border-radius: 10px;
                      font-size: 11px; font-weight: 600; }
        .kind-class { background: #ddf4ff; color: #0969da; }
        .kind-function { background: #dafbe1; color: #1a7f37; }
        .kind-constant { background: #fff8c5; color: #9a6700; }
        blockquote { border-left: 4px solid #d0d7de; padding: 4px 14px; color: #57606a;
                     background: #f6f8fa; margin: 12px 0; }
        a.xref { color: #0969da; text-decoration: none; }
        a.xref:hover { text-decoration: underline; }
        .tok-kw { color: #cf222e; }
        .tok-str { color: #0a3069; }
        .tok-com { color: #6e7781; font-style: italic; }
        .tok-num { color: #0550ae; }
        .tok-dec { color: #8250df; }
        .entry { margin-bottom: 36px; }
        .loc { color: #6e7781; font-size: 12px; font-style: italic; }
        .stats { display: flex; gap: 16px; margin: 12px 0; }
        .stats div { background: #fff; border: 1px solid #d0d7de; border-radius: 6px;
                     padding: 8px 14px; font-size: 13px; }
        .stats strong { display: block; font-size: 18px; color: #0969da; }
        """

    def _js(self) -> str:
        return """
        document.addEventListener('DOMContentLoaded', function() {
            var input = document.getElementById('search-box');
            if (!input) return;
            input.addEventListener('input', function() {
                var q = input.value.toLowerCase().trim();
                var items = document.querySelectorAll('.sidebar li[data-name]');
                items.forEach(function(li) {
                    var name = (li.getAttribute('data-name') || '').toLowerCase();
                    var mod = (li.getAttribute('data-module') || '').toLowerCase();
                    if (!q || name.indexOf(q) !== -1 || mod.indexOf(q) !== -1) {
                        li.style.display = '';
                    } else {
                        li.style.display = 'none';
                    }
                });
            });
            // collapsible packages
            document.querySelectorAll('.sidebar .pkg').forEach(function(p) {
                p.addEventListener('click', function() {
                    p.parentElement.classList.toggle('collapsed');
                });
            });
        });
        """

    def _sidebar_html(self, current_module: Optional[str] = None) -> str:
        mods = self.scanner.modules()
        # Group modules by top-level package.
        tree: Dict[str, Any] = {}
        for m in mods:
            parts = m.split(".")
            node = tree
            for i, p in enumerate(parts):
                if p not in node:
                    node[p] = {"__mod__": ".".join(parts[: i + 1]) if i == len(parts) - 1 else None,
                               "__children__": {}}
                node = node[p]["__children__"]
        # Render.
        def render(node: Dict[str, Any], depth: int) -> str:
            out: List[str] = []
            for key in sorted(k for k in node if not k.startswith("__")):
                sub = node[key]
                mod_path = sub.get("__mod__")
                children = sub.get("__children__", {})
                if children:
                    out.append(f'<div class="pkg" style="cursor:pointer;">{"  " * depth}{key}</div>')
                    out.append("<ul>")
                    out.append(render(children, depth + 1))
                    out.append("</ul>")
                else:
                    if mod_path:
                        slug = mod_path.replace(".", "_")
                        cls = ' class="active"' if mod_path == current_module else ""
                        out.append(
                            f'<li data-name="{html.escape(key)}" '
                            f'data-module="{html.escape(mod_path)}">'
                            f'<a href="{slug}.html"{cls}>{html.escape(mod_path)}</a></li>'
                        )
            return "".join(out)

        return render(tree, 0)

    def _page(self, title: str, body: str, sidebar: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{self._css()}</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <h1>qai API</h1>
    <input type="search" id="search-box" class="search" placeholder="Filter modules..." autocomplete="off">
    {sidebar}
  </aside>
  <main class="main">
{body}
  </main>
</div>
<script>{self._js()}</script>
</body>
</html>
"""

    # ------------------------------------------------------------------ #

    def generate_module(self, module_name: str) -> str:
        """Render the HTML for one module as a string."""
        entries = self.scanner.entries_for(module_name)
        classes = [e for e in entries if e.kind == "class"]
        functions = [e for e in entries if e.kind == "function"]
        constants = [e for e in entries if e.kind == "constant"]
        body: List[str] = []
        body.append(f"<h1><code>{html.escape(module_name)}</code></h1>")
        brief = self._module_brief(module_name)
        if brief:
            body.append(f"<blockquote>{html.escape(brief)}</blockquote>")
        # Stats.
        body.append('<div class="stats">')
        body.append(f"<div><strong>{len(classes)}</strong>classes</div>")
        body.append(f"<div><strong>{len(functions)}</strong>functions</div>")
        body.append(f"<div><strong>{len(constants)}</strong>constants</div>")
        body.append("</div>")
        if entries:
            body.append(f'<p class="loc">Source: <code>{html.escape(entries[0].source_file)}</code></p>')

        # TOC.
        body.append("<h2>Contents</h2>")
        body.append("<ul>")
        for e in classes + functions + constants:
            anchor = e.name.lower()
            body.append(
                f'<li><a href="#{anchor}" class="xref">{html.escape(e.kind)} '
                f"{html.escape(e.name)}</a></li>"
            )
        body.append("</ul>")

        if classes:
            body.append("<h2>Classes</h2>")
            for e in classes:
                body.append(self._render_entry(e))
        if functions:
            body.append("<h2>Functions</h2>")
            for e in functions:
                body.append(self._render_entry(e))
        if constants:
            body.append("<h2>Constants</h2>")
            for e in constants:
                body.append(self._render_entry(e))

        sidebar = self._sidebar_html(current_module=module_name)
        return self._page(module_name, "\n".join(body), sidebar)

    def _render_entry(self, e: ApiEntry) -> str:
        anchor = e.name.lower()
        out: List[str] = [f'<div class="entry" id="{anchor}">']
        badge = f'<span class="kind-badge kind-{e.kind}">{e.kind}</span>'
        out.append(f"<h3>{badge} <code>{html.escape(e.name)}</code></h3>")
        # Signature.
        if e.kind == "class":
            decl = f"class {e.name}"
            if e.bases:
                # bases may carry decorators for functions; only render real bases.
                real_bases = [b for b in e.bases if not b.startswith("@")]
                if real_bases:
                    decl += f"({', '.join(real_bases)})"
            sig_html = self._highlight_python(decl)
        elif e.kind == "function":
            sig_html = self._highlight_python(e.signature or "")
        else:
            sig_html = self._highlight_python(f"{e.name} = {e.value or '...'}")
        out.append(f"<pre><code>{sig_html}</code></pre>")
        if e.brief:
            out.append(f"<blockquote>{html.escape(e.brief)}</blockquote>")
        if e.body:
            out.append(f"<p>{html.escape(e.body).replace(chr(10), '<br>')}</p>")
        # Parameters.
        real_params = [p for p in e.parameters if p.get("kind") != "marker"]
        if real_params:
            out.append("<h4>Parameters</h4>")
            out.append("<table><tr><th>Name</th><th>Type</th><th>Default</th></tr>")
            for p in real_params:
                out.append(
                    "<tr>"
                    f"<td><code>{html.escape(p['name'])}</code></td>"
                    f"<td><code>{html.escape(p.get('annotation',''))}</code></td>"
                    f"<td><code>{html.escape(p.get('default',''))}</code></td>"
                    "</tr>"
                )
            out.append("</table>")
        if e.kind == "function" and e.returns:
            out.append(f"<h4>Returns</h4><p><code>{html.escape(e.returns)}</code></p>")
        if e.raises:
            out.append("<h4>Raises</h4><ul>")
            for r in e.raises:
                out.append(f"<li><code>{html.escape(r)}</code></li>")
            out.append("</ul>")
        if e.methods:
            out.append("<h4>Methods</h4><table><tr><th>Name</th><th>Signature</th></tr>")
            for m in e.methods:
                out.append(
                    "<tr>"
                    f"<td><code>{html.escape(m.get('name',''))}()</code></td>"
                    f"<td><code>{html.escape(m.get('signature',''))}</code></td>"
                    "</tr>"
                )
            out.append("</table>")
        if e.examples:
            out.append("<h4>Examples</h4>")
            for ex in e.examples:
                out.append(f"<pre><code>{self._highlight_python(ex)}</code></pre>")
        if e.source_file:
            loc = f"{e.source_file}:{e.line_number}" if e.line_number else e.source_file
            out.append(f'<p class="loc">Defined in <code>{html.escape(loc)}</code></p>')
        out.append("</div>")
        return "\n".join(out)

    def _module_brief(self, module_name: str) -> str:
        try:
            file_path = self._module_file_path(module_name)
        except Exception:
            return ""
        if not file_path or not os.path.isfile(file_path):
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                src = fh.read()
            tree = ast.parse(src, filename=file_path)
            doc = ast.get_docstring(tree) or ""
            if doc:
                return doc.split("\n", 1)[0].strip()
        except Exception:
            pass
        return ""

    def _module_file_path(self, module_name: str) -> str:
        parts = module_name.split(".")
        if parts and parts[0] == "qai":
            parts = parts[1:]
        candidate = os.path.join(self.scanner.package_path, *parts) + ".py"
        if os.path.isfile(candidate):
            return candidate
        return os.path.join(self.scanner.package_path, *parts, "__init__.py")

    # ------------------------------------------------------------------ #

    def generate_all(self) -> List[str]:
        os.makedirs(self.output_dir, exist_ok=True)
        written: List[str] = []
        for mod in self.scanner.modules():
            page = self.generate_module(mod)
            slug = mod.replace(".", "_")
            path = os.path.join(self.output_dir, f"{slug}.html")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(page)
            written.append(path)
        idx_path = self.generate_index()
        written.append(idx_path)
        return written

    def generate_index(self) -> str:
        stats = self.scanner.stats()
        mods = self.scanner.modules()
        body: List[str] = []
        body.append("<h1>Quantum AI Framework — API Reference</h1>")
        body.append('<div class="stats">')
        body.append(f"<div><strong>{stats['modules']}</strong>modules</div>")
        body.append(f"<div><strong>{stats['classes']}</strong>classes</div>")
        body.append(f"<div><strong>{stats['functions']}</strong>functions</div>")
        body.append(f"<div><strong>{stats['constants']}</strong>constants</div>")
        body.append("</div>")
        body.append("<h2>Modules</h2><ul>")
        for m in mods:
            brief = self._module_brief(m)
            slug = m.replace(".", "_")
            tail = f" — {html.escape(brief)}" if brief else ""
            body.append(f'<li><a href="{slug}.html" class="xref">{html.escape(m)}</a>{tail}</li>')
        body.append("</ul>")
        sidebar = self._sidebar_html()
        page = self._page("qai API Reference", "\n".join(body), sidebar)
        path = os.path.join(self.output_dir, "index.html")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(page)
        return path


# ===========================================================================
# Top-level orchestrator
# ===========================================================================

class ApiReference:
    """Top-level orchestrator.

    Builds a single :class:`ApiScanner` and delegates to the appropriate
    generator based on ``output_format``.
    """

    def __init__(
        self,
        output_format: str = "markdown",
        output_dir: str = "docs/api/",
        package_path: Optional[str] = None,
        exclude_private: bool = True,
        exclude_tests: bool = True,
    ) -> None:
        self.output_format = output_format.lower()
        self.output_dir = output_dir
        self.scanner = ApiScanner(
            package_path=package_path,
            exclude_private=exclude_private,
            exclude_tests=exclude_tests,
        )
        self.scanner.scan()

    def generate(self) -> List[str]:
        """Generate the full reference and return the list of files written."""
        if self.output_format in ("html", "htm"):
            gen = HtmlGenerator(self.scanner, output_dir=self.output_dir)
        else:
            gen = MarkdownGenerator(self.scanner, output_dir=self.output_dir)
        return gen.generate_all()

    def stats(self) -> Dict[str, int]:
        return self.scanner.stats()

    def search(self, query: str, limit: Optional[int] = None) -> List[ApiEntry]:
        return self.scanner.find(query, limit=limit)


# ===========================================================================
# Module-level convenience API
# ===========================================================================

_DEFAULT_SCANNER: Optional[ApiScanner] = None


def _get_scanner() -> ApiScanner:
    global _DEFAULT_SCANNER
    if _DEFAULT_SCANNER is None:
        _DEFAULT_SCANNER = ApiScanner()
        _DEFAULT_SCANNER.scan()
    return _DEFAULT_SCANNER


def generate(format: str = "markdown", output_dir: str = "docs/api/") -> List[str]:
    """Generate the full API reference.

    Parameters:
        format: ``'markdown'`` or ``'html'``.
        output_dir: Directory to write files into.

    Returns:
        List of file paths that were written.
    """
    ref = ApiReference(output_format=format, output_dir=output_dir)
    return ref.generate()


def stats() -> Dict[str, int]:
    """Return scanner statistics (modules, classes, functions, constants)."""
    return _get_scanner().stats()


def search(query: str, limit: Optional[int] = None) -> List[ApiEntry]:
    """Fuzzy-search the qai API surface for ``query``."""
    return _get_scanner().find(query, limit=limit)


def find_class(name: str) -> Optional[ApiEntry]:
    """Find the first class named ``name`` (case-insensitive)."""
    return _get_scanner().find_class(name)


def find_function(name: str) -> Optional[ApiEntry]:
    """Find the first function named ``name`` (case-insensitive)."""
    return _get_scanner().find_function(name)
