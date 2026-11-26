"""
qai.chainable_output — chainable model output formatting (PDF spec feature).

The PDF spec says:
  "model.output() as a real, chainable, inspectable object, not just a string:
   let result = model.output()
   .format("json")
   .redact(pattern: "emails")"

This module implements:
  - ModelOutput — chainable output object returned by model.output()
  - .format(fmt) — format as "json", "yaml", "csv", "text", "html"
  - .redact(pattern) — redact sensitive data (emails, phones, SSNs)
  - .mask(fields) — mask specific fields
  - .filter(keys) — keep only specified keys
  - .sort(key) — sort results
  - .limit(n) — limit number of results
  - .keys() — list available keys
  - .summary() — one-line summary
  - .save(path) — save to file

Usage:
    import qai
    m = qai.build(technique="classifier")
    m.train(X, y)

    result = m.output()
    result.format("json")          # → JSON string
    result.format("json").redact("emails")  # → JSON with emails redacted
    result.format("text").mask("confidence")  # → text with confidence hidden
    result.save("output.json")     # → save to file
"""
from __future__ import annotations
import json, re, os
from typing import Any, Dict, List, Optional, Union


# =====================================================================
# Redaction patterns
# =====================================================================
REDACT_PATTERNS = {
    "emails": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]'),
    "phones": (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]'),
    "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]'),
    "credit_cards": (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[REDACTED_CC]'),
    "ips": (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[REDACTED_IP]'),
    "urls": (r'https?://\S+', '[REDACTED_URL]'),
}


class ModelOutput:
    """Chainable model output object.

    Returned by model.output() — supports method chaining for formatting,
    redaction, filtering, and saving.

    Example:
        >>> result = m.output()
        >>> result.format("json")
        '{"prediction": 1, "confidence": 0.95}'

        >>> result.format("json").redact("emails")
        '{"prediction": 1, "confidence": 0.95, "text": "[REDACTED_EMAIL]"}'

        >>> result.format("text").save("output.txt")
        'output.txt'
    """
    def __init__(self, data: Any, model=None):
        self._data = data
        self._model = model
        self._format: Optional[str] = None
        self._redactions: List[str] = []
        self._masks: List[str] = []
        self._filters: Optional[List[str]] = None
        self._sort_key: Optional[str] = None
        self._limit_n: Optional[int] = None

    def format(self, fmt: str = "json") -> "ModelOutput":
        """Set the output format.

        Args:
            fmt: "json", "yaml", "csv", "text", "html"

        Returns:
            self (for chaining)
        """
        self._format = fmt
        return self

    def redact(self, pattern: Union[str, List[str]]) -> "ModelOutput":
        """Redact sensitive data matching the pattern.

        Args:
            pattern: "emails", "phones", "ssn", "credit_cards", "ips", "urls",
                     or a regex string, or a list of any of these.

        Returns:
            self (for chaining)
        """
        if isinstance(pattern, str):
            patterns = [pattern]
        else:
            patterns = pattern
        self._redactions.extend(patterns)
        return self

    def mask(self, fields: Union[str, List[str]]) -> "ModelOutput":
        """Mask specific fields (replace value with '***').

        Args:
            fields: field name or list of field names to mask

        Returns:
            self (for chaining)
        """
        if isinstance(fields, str):
            fields = [fields]
        self._masks.extend(fields)
        return self

    def filter(self, keys: Union[str, List[str]]) -> "ModelOutput":
        """Keep only the specified keys in the output.

        Args:
            keys: key name or list of key names

        Returns:
            self (for chaining)
        """
        if isinstance(keys, str):
            keys = [keys]
        self._filters = keys
        return self

    def sort(self, key: str, descending: bool = True) -> "ModelOutput":
        """Sort the output by a key.

        Args:
            key: field name to sort by
            descending: if True, sort descending

        Returns:
            self (for chaining)
        """
        self._sort_key = key
        self._sort_desc = descending
        return self

    def limit(self, n: int) -> "ModelOutput":
        """Limit the number of results.

        Args:
            n: maximum number of items

        Returns:
            self (for chaining)
        """
        self._limit_n = n
        return self

    def keys(self) -> List[str]:
        """List available keys in the output."""
        if isinstance(self._data, dict):
            return list(self._data.keys())
        return []

    def summary(self) -> str:
        """One-line summary of the output."""
        if isinstance(self._data, dict):
            parts = [f"{k}={v}" for k, v in list(self._data.items())[:3]]
            extra = f" (+{len(self._data)-3} more)" if len(self._data) > 3 else ""
            return f"ModelOutput({', '.join(parts)}{extra})"
        return f"ModelOutput({str(self._data)[:80]})"

    def save(self, path: str) -> str:
        """Save the formatted output to a file.

        Args:
            path: file path

        Returns:
            The path (for chaining)
        """
        content = str(self)
        with open(path, "w") as f:
            f.write(content)
        return path

    def _apply_transforms(self) -> Any:
        """Apply all transforms (filter, mask, redact, sort, limit) to the data."""
        data = self._data

        # Filter
        if self._filters and isinstance(data, dict):
            data = {k: v for k, v in data.items() if k in self._filters}

        # Mask
        if self._masks and isinstance(data, dict):
            data = dict(data)  # copy
            for field in self._masks:
                if field in data:
                    data[field] = "***"

        # Redact
        if self._redactions:
            data = self._apply_redactions(data)

        # Sort
        if self._sort_key and isinstance(data, list):
            reverse = getattr(self, "_sort_desc", True)
            data = sorted(data, key=lambda x: x.get(self._sort_key, 0), reverse=reverse)

        # Limit
        if self._limit_n is not None and isinstance(data, list):
            data = data[:self._limit_n]

        return data

    def _apply_redactions(self, data: Any) -> Any:
        """Apply redaction patterns to data."""
        def redact_str(s: str) -> str:
            for pattern_name in self._redactions:
                if pattern_name in REDACT_PATTERNS:
                    regex, replacement = REDACT_PATTERNS[pattern_name]
                    s = re.sub(regex, replacement, s)
                else:
                    # Treat as a regex
                    try:
                        s = re.sub(pattern_name, "[REDACTED]", s)
                    except re.error:
                        pass
            return s

        if isinstance(data, str):
            return redact_str(data)
        elif isinstance(data, dict):
            return {k: self._apply_redactions(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._apply_redactions(item) for item in data]
        return data

    def _format_output(self, data: Any) -> str:
        """Format data according to self._format."""
        fmt = self._format or "json"

        if fmt == "json":
            return json.dumps(data, indent=2, default=str)
        elif fmt == "yaml":
            return self._to_yaml(data)
        elif fmt == "csv":
            return self._to_csv(data)
        elif fmt == "text":
            if isinstance(data, dict):
                return "\n".join(f"{k}: {v}" for k, v in data.items())
            return str(data)
        elif fmt == "html":
            if isinstance(data, dict):
                rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data.items())
                return f"<table>{rows}</table>"
            return f"<pre>{data}</pre>"
        else:
            return str(data)

    def _to_yaml(self, data: Any, indent: int = 0) -> str:
        """Simple YAML formatter (no dependency on pyyaml)."""
        pad = "  " * indent
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    lines.append(self._to_yaml(v, indent + 1))
                else:
                    lines.append(f"{pad}{k}: {v}")
            return "\n".join(lines)
        elif isinstance(data, list):
            lines = []
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.append(self._to_yaml(item, indent + 1))
                else:
                    lines.append(f"{pad}- {item}")
            return "\n".join(lines)
        return f"{pad}{data}"

    def _to_csv(self, data: Any) -> str:
        """Simple CSV formatter."""
        if isinstance(data, dict):
            header = ",".join(data.keys())
            row = ",".join(str(v) for v in data.values())
            return f"{header}\n{row}"
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            keys = list(data[0].keys())
            header = ",".join(keys)
            rows = [",".join(str(item.get(k, "")) for k in keys) for item in data]
            return "\n".join([header] + rows)
        return str(data)

    def __str__(self) -> str:
        """Render the output with all transforms applied."""
        data = self._apply_transforms()
        return self._format_output(data)

    def __repr__(self) -> str:
        return self.summary()

    # Allow dict-like access
    def __getitem__(self, key):
        if isinstance(self._data, dict):
            return self._data[key]
        raise KeyError(f"Key '{key}' not found in output")

    def __contains__(self, key):
        if isinstance(self._data, dict):
            return key in self._data
        return False

    # Allow iteration
    def __iter__(self):
        if isinstance(self._data, dict):
            return iter(self._data)
        elif isinstance(self._data, list):
            return iter(self._data)
        return iter([])

    def __len__(self):
        if isinstance(self._data, dict):
            return len(self._data)
        elif isinstance(self._data, list):
            return len(self._data)
        return 0


__all__ = ["ModelOutput"]
