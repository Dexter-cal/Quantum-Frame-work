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

### 5. Dict-Row Input & Output Schema Validation Ergonomics
- **Discovery**: Real-world REST and pipeline payloads often deliver named JSON/dict records rather than raw NumPy 2D matrices.
- **Impact**: Requiring users to manually unpack dictionaries into positional matrices reduced usability.
- **Resolution**: Enhanced `Schema` with `validate_row()` and `dict_to_array()`. Updated `Model.train()` and `Model.predict()` to automatically accept dictionary payloads, validate fields/types, and map keys to column order.

### 6. Deep Learning Integration via PyTorch
- **Discovery**: Real-world deep learning workflows require standard dynamic computation graphs and tensor backpropagation rather than manually coded gradient matrices.
- **Impact**: PyTorch integration (`torch`) allows `qai` to seamlessly offer deep neural network capabilities while keeping the clean `qai.build(type="neural_network")` interface.

### 7. Section 56 Complete Mechanics Decomposition Wiring
- **Discovery**: In the initial design, `Objective` and `TrainingLoop` objects existed as standalone classes in `qai.mechanics` but were not directly wired into `Model.build()`.
- **Impact**: Explicitly wiring `objective` and `training_loop` into `qai.build()` fulfills Section 56's core design goal: structure (what the model IS) is completely decoupled from optimization, loss measurement, and termination criteria.
