#!/usr/bin/env python3
"""Patch all technique files that have `try: import torch` but no torch stub,
so that `import qai` works without PyTorch installed.

Handles multiple patterns:
  - `_TORCH_AVAILABLE = True/False` (cnn.py, transformer.py, rnn.py style)
  - `_TORCH = True/False` (most other files)

Idempotent: skips files that already import from qai._torch_stub.
"""
import os
import re

ROOT = "/home/z/my-project/qai"

# Pattern A: `_TORCH_AVAILABLE = True/False`
PATTERN_A = re.compile(
    r"(except ImportError:\n"
    r"(\s+)_TORCH_AVAILABLE = False\n)"
)
# Pattern B: `_TORCH = True/False`
PATTERN_B = re.compile(
    r"(except ImportError:\n"
    r"(\s+)_TORCH = False\n)"
)


def patch_file(path: str) -> str:
    with open(path) as f:
        src = f.read()
    if "qai._torch_stub" in src:
        return "skip (already stubbed)"

    # Check which pattern matches
    stub_a = (
        "except ImportError:\n"
        "{ind}_TORCH_AVAILABLE = False\n"
        "{ind}from qai._torch_stub import _TorchStubNamespace, _NNStubNamespace\n"
        "{ind}torch = _TorchStubNamespace()\n"
        "{ind}nn = _NNStubNamespace()\n"
        "{ind}F = nn\n"
    )
    stub_b = (
        "except ImportError:\n"
        "{ind}_TORCH = False\n"
        "{ind}from qai._torch_stub import _TorchStubNamespace, _NNStubNamespace\n"
        "{ind}torch = _TorchStubNamespace()\n"
        "{ind}nn = _NNStubNamespace()\n"
        "{ind}F = nn\n"
    )

    new_src = src
    n = 0
    for pat, tpl in ((PATTERN_A, stub_a), (PATTERN_B, stub_b)):
        for m in pat.finditer(new_src):
            ind = m.group(2)
            replacement = tpl.format(ind=ind)
            new_src = new_src.replace(m.group(0), replacement, 1)
            n += 1

    if new_src != src:
        with open(path, "w") as f:
            f.write(new_src)
        return f"patched ({n} site(s))"
    return "skip (no try/except torch pattern found)"


def main():
    patched = []
    skipped = []
    for dirpath, _, filenames in os.walk(ROOT):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            if fn == "_torch_stub.py":
                continue
            p = os.path.join(dirpath, fn)
            with open(p) as f:
                src = f.read()
            # Only consider files that ACTUALLY use nn.Module as a base class
            if not re.search(r"class\s+\w+\s*\(\s*nn\.Module", src):
                continue
            # Must have one of the try/except torch patterns
            if not (PATTERN_A.search(src) or PATTERN_B.search(src)):
                continue
            if "qai._torch_stub" in src:
                skipped.append((p, "already stubbed"))
                continue
            result = patch_file(p)
            if result.startswith("patched"):
                patched.append((p, result))
            else:
                skipped.append((p, result))

    print(f"Patched {len(patched)} file(s):")
    for p, r in patched:
        print(f"  + {p}  ({r})")
    if skipped:
        print(f"Skipped {len(skipped)} file(s):")
        for p, r in skipped:
            print(f"  - {p}  ({r})")


if __name__ == "__main__":
    main()
