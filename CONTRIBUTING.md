# Contributing to QAI

Thanks for your interest in improving QAI. This document covers the workflow, code style, and test policy.

> See also: [README](README.md) · [Roadmap](docs/ROADMAP.md) · [Implementation Status](docs/UNBUILT_FEATURES.md)

---

## Workflow

1. **Fork** the repo and clone your fork locally.
2. **Branch** off `main`:
   ```bash
   git checkout -b feat/my-new-feature
   ```
3. **Make your changes.** Add tests for any new behavior (see Test Policy below).
4. **Run the test suite**:
   ```bash
   pytest tests/ -v
   ```
5. **Commit** with a clear message:
   ```
   feat(federation): add FedNova aggregator
   
    - implements normalized-averaging strategy
    - 4 tests cover 2-client iris, 4-client digits
    - documents the algorithm in qai/federation.py
   ```
   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation only
   - `refactor:` no behavior change
   - `test:` test additions
   - `chore:` tooling, deps, cleanup
6. **Push** and open a PR against `main`. Fill in the PR template:
   - What changed and why
   - How to test it
   - Anything that needs review attention

---

## Code Style

- **Python 3.9+** — type hints on all public functions.
- **Line length: 100 chars.** Use Black defaults if in doubt.
- **Imports:** stdlib → third-party → `qai.*`, alphabetized within each group.
- **Lazy imports for optional deps.** Never import `torch`, `transformers`, `cv2`, etc. at module top-level if the module can work without them. Wrap in functions or use `qai.LazyLoader`.
- **No breaking changes to the public API.** If you must change a signature, deprecate the old one for one minor release first.
- **Docstrings on every public class and function.** Triple-quoted, first line imperative.
- **No emojis in source files.** Plain text only — they break rendering in some terminals and PDF builds.

### Single-import surface

The top-level `qai/__init__.py` is the user-facing API. Every public symbol
lives there. When adding a new module:

1. Implement it in `qai/<module>.py` (or `qai/<package>/__init__.py` for sub-packages).
2. Export its public symbols in `qai/__init__.py`.
3. Document the new symbols in [docs/USER_MANUAL.md](docs/USER_MANUAL.md).
4. Add at least one test that exercises the new symbol end-to-end.

### Technique registration

New model techniques use the `register_technique` decorator:

```python
from qai.techniques import register_technique, Technique

@register_technique("my_model", family="custom")
class MyModel(Technique):
    def fit(self, X, y):
        ...
    def forward(self, x):
        ...
```

Once registered, it's available as `qai.build(technique="my_model")`.

---

## Test Policy

- **Every new technique, function, or class needs at least one test.**
- Tests live in `tests/` and follow the naming convention `test_<area>_<feature>.py`.
- Use real data where possible — `qai.load_iris()`, `qai.load_digits()`, or small synthetic arrays.
- Test the **public API**, not the internals. If a user can call it, it gets a test.
- For bug fixes, add a regression test that would have caught the bug.
- Run the full suite before submitting: `pytest tests/ -v`. All tests must pass on a clean checkout.

### Test structure

```python
import qai

def test_my_model_predicts_correctly():
    """MyModel should predict y=2x for linear data."""
    m = qai.build(technique="my_model")
    m.train(X=[[1.0], [2.0], [3.0]], y=[2.0, 4.0, 6.0])
    assert abs(m.predict([4.0]) - 8.0) < 0.01
```

---

## Documentation

- **User-facing changes** → update [docs/USER_MANUAL.md](docs/USER_MANUAL.md) and [docs/TUTORIAL.md](docs/TUTORIAL.md).
- **New features** → add an entry to [CHANGELOG.md](CHANGELOG.md) under the next version.
- **Architecture decisions** → add to the Decision Log in [docs/ROADMAP.md](docs/ROADMAP.md).
- **Engineering gotchas** → add to [docs/FINDINGS.md](docs/FINDINGS.md).
- **Spec-section coverage** → update [docs/UNBUILT_FEATURES.md](docs/UNBUILT_FEATURES.md).

All docs cross-link to each other. When you add a new doc, link it from the relevant existing docs.

---

## Reporting Bugs

Open a GitHub Issue with:

1. **QAI version:** `python -c "import qai; print(qai.__version__)"`
2. **Python version:** `python --version`
3. **OS:** Linux / macOS / Windows
4. **Reproducer:** minimal code that triggers the bug
5. **Expected vs actual behavior**
6. **Stack trace** (if applicable)

---

## Reporting Security Issues

**Do not open a public issue for security bugs.** Email the maintainer privately instead. The security module (`qai.security`) has its own test coverage; please don't disclose vulnerabilities in the public issue tracker before a fix is released.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## Code of Conduct

Be kind. Be specific. Be patient. We're all here to build something useful.
