# QAI Engineering Findings & Architecture Notes

This document logs critical discoveries, architectural insights, edge cases, and runtime behavior observations uncovered during the implementation and auditing of `qai`.

## Findings Log

### 1. Loose Equality Truthiness Risk in Numpy Arrays
- **Discovery**: When testing model prediction outputs against scalar ground truth values, Python equality checks like `assert prediction == expected` can evaluate to truthy numpy boolean arrays when `prediction` is actually an `np.ndarray` of shape `(1,)`.
- **Impact**: Code returning single-element 1D arrays instead of unboxed scalars went unnoticed across several standard test runs.
- **Resolution**: Implemented strict scalar unboxing checks using `isinstance(val, (int, float, str, np.number))` and `isinstance(val, np.ndarray) == False` to guarantee unboxed prediction types for single inputs.

### 2. Logic Function Deserialization Namespace Requirements
- **Discovery**: When loading decorated `@qai.logic` function source via `load_logic_source()`, executing source code dynamically in an isolated scope failed if the `qai` module itself was not bound in the execution globals.
- **Impact**: `@qai.logic` decorator syntax failed inside `exec()` during subprocess/cross-process reloading.
- **Resolution**: Injected `qai` explicitly into the dynamic execution globals dictionary during `load_logic_source()`.

### 3. Absolute Path Export Hardcoding
- **Discovery**: `tests/test_core_pipeline.py` referenced a hardcoded user path (`/home/claude/...`), causing test suite failures in generic containerized or fresh sandbox environments.
- **Impact**: Non-portable test setup.
- **Resolution**: Updated export path targets to use standard system temporary directories (`/tmp/`).

### 4. Consolidated Framework Single-Import Pattern
- **Discovery**: Users building workflows expect a complete end-to-end framework without requiring secondary third-party package imports.
- **Resolution**: Standardized top-level exports in `qai/__init__.py` to expose all core models, techniques, evaluation tools, pipelines, datasets, logic decorators, and serving handlers.
