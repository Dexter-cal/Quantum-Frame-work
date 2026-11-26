# Guide: Visual Workspace — The Model Microscope

*See your model think: architecture canvas, weight sliders, forward-pass animation, story mode.*

> Back to [Guides index](README.md) · Design spec: [VISUAL_DESIGN.md](../VISUAL_DESIGN.md) · Full API: [User Manual](../USER_MANUAL.md)

---

## What you'll learn

1. Launch the interactive workspace
2. The three-column layout and what each does
3. Weight editing with live re-prediction
4. Story mode, compare mode, adversarial probe
5. DEMO mode (no model needed)

---

## 1. Launch it

```python
import qai

m = qai.build(technique="classifier")
m.train(X, y)

qai.workspace(m)     # opens the microscope in your browser
```

Pure HTML/CSS/JS — no npm, no build step, no D3. A small Flask JSON backend
(7 endpoints) feeds it live data. Works on localhost and in notebooks.

## 2. The three columns

| Column | What it shows |
|---|---|
| **Input (left)** | chat-style input box, example buttons, file drop (CSV/text) |
| **Architecture canvas (center)** | live diagram, weights as colors, active path highlighted |
| **Output (right)** | prediction, confidence bar, latency, top contributing features |

The whole point is **cause and effect**: change input → watch the path light
up → watch the prediction move.

## 3. Rendering modes (auto by model size)

| Mode | Trigger | What you see |
|---|---|---|
| Full | < 10K params | every node + weight, all editable |
| Layer-summary | 10K–1M params | layer boxes with histograms, click to drill |
| Block-summary | > 1M params | block-level, click → layer → neuron |

"Focus on active nodes": only the top-k neurons by |activation| render, plus
a "47 more hidden" chip. Click it to show all.

## 4. Weight inspector (bottom panel)

Four tabs: **Heatmap** (matrix grid) · **Sliders** (top-K influential weights,
draggable) · **Sensitivity** (sweep a weight, plot the response curve) ·
**History** (undo/redo stack).

Drag a slider → 200ms debounced re-predict → output updates with a diff:
"weight 0.42→0.67, prediction flipped: A→B". Range is clamped to [μ−3σ, μ+3σ]
to prevent NaN. "Reset to trained" restores the original weights.

## 5. Story mode (guided tour)

A 6-step narrated walkthrough of the last forward pass: "Step 2: Token 'cat'
activates neuron 247 (weight 0.83)..." — exportable as a self-contained HTML
file to share.

## 6. Compare / adversarial / health

- **Compare two inputs** — pin input A, send input B; differing paths highlight orange
- **Adversarial probe** — one click adds minimal noise that flips the prediction
- **Health tab** — dead neurons (gray X), saturated activations (yellow), overconfidence on OOD inputs (red), MoE routing collapse, attention sinks

## 7. DEMO and STORY modes

```python
qai.workspace()              # DEMO — no model needed, synthetic data
qai.workspace(m, mode="story")   # STORY — static HTML snapshot
```

## 8. Node labels (the key feature)

Nodes are labeled with what they *mean*:
- tabular classifier → `petal_length`, `sepal_width`
- transformer → actual tokens `"the"`, `"cat"`, `"sat"`
- CNN → `conv1.ch3 (edges)`
- KMeans → `cluster_2 (5.0, 3.4)`
- random forest → `tree_7 → setosa 0.9`

---

## Cheat sheet

| Task | Call |
|---|---|
| Open the microscope | `qai.workspace(model)` |
| No model handy | `qai.workspace()` (DEMO) |
| Shareable snapshot | `qai.workspace(model, mode="story")` |
| Edit a weight live | drag a slider in the Inspector |
| See what matters | click a node → weights appear below |
| Find dead neurons | Health tab |
| Break the model | Adversarial probe button |
| Export the story | Story mode → Export |

---

## Common gotchas

- **Big models render in block-summary mode.** That's deliberate — 1M+ nodes will hang a browser tab.
- **Slider edits are temporary.** "Reset to trained" restores originals; to keep edits, use the Weight Safety Toolkit (`qai.core` weights API) instead.
- **The JSON backend is localhost-only by default.** Don't expose it publicly without auth — see [SECURITY_GOVERNANCE.md](SECURITY_GOVERNANCE.md).

---

*Next: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) when something breaks, or back to the [Guides index](README.md).*
