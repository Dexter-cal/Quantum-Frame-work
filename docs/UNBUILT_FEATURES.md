# QAI Unbuilt Features & Roadmap

This document maintains a prioritized inventory of pending features, design spec requirements, and framework extensions required for complete AI lifecycle coverage.

## Design Specification Pending Sections

1. **Output Schema Validation & Named Field Validation**
   - *Status*: Pending (Section 91)
   - *Spec Target*: Validate output row structure and dict/named-field inputs against declared schemas.

2. **Unsupervised Historical Techniques Suite Expansion**
   - *Status*: Pending (Section 32)
   - *Spec Target*: Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models (GMM) with native `qai` model ergonomics.

3. **Complete Mechanics Decomposition (Objective & TrainingLoop)**
   - *Status*: Partial (Section 56)
   - *Spec Target*: Explicit `Objective` (loss function abstraction) and `TrainingLoop` objects fully integrated into `Model.build()`.

4. **Deep Learning Types Abstraction Layer**
   - *Status*: Blocked / Lightweight Fallback (Section 12/56)
   - *Spec Target*: Pure-Python/NumPy neural network runtime for environments without PyTorch/TensorFlow.

5. **Advanced Model Governance & Comprehensive AI Utilities**
   - *Status*: Proposed Roadmap
   - *Features*: Model drift detection, automated data preprocessing/cleaning utilities, experiment tracking, and metrics logging natively accessible under `qai.*`.
