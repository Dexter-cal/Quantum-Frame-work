# QAI Visual UI Design

*The interactive model microscope — see your model think in real time.*

> See also: [README](../README.md) · [Roadmap](ROADMAP.md) (Decision #9: pure HTML/CSS/JS) · [User Manual](USER_MANUAL.md)

---

## Design Philosophy

The qai visual system is NOT just an architecture diagram. It's a **living
workspace** where you can:
- Watch the forward pass animate as your input flows through the model
- Drag a weight slider and see the prediction change in real time
- Click any neuron and see what concept it represents
- Compare two inputs side by side to understand *why* they get different predictions

This is the "model microscope" — the first tool that makes neural network
internals as inspectable as a spreadsheet.

---

## Layout: Three-Column Workspace

```
┌────────────────────────────────────────────────────────────────────┐
│ TOP BAR:  [model: classifier ▾]  [technique badge]  train  reset  │
├──────────────┬───────────────────────────────┬─────────────────────┤
│  INPUT       │  ARCHITECTURE CANVAS          │  OUTPUT              │
│  (320px)     │  (flexible)                   │  (320px)             │
│              │                               │                     │
│ chat input   │  live diagram — click any     │  prediction (big)   │
│ batch input  │  layer / node to inspect       │  confidence bar     │
│ examples     │  weights overlaid as colors   │  latency, tokens/s  │
│ file upload  │  active path highlighted       │  explanation        │
│              │  from current input            │  before/after diff  │
│              │                               │                     │
├──────────────┴───────────────────────────────┴─────────────────────┤
│ WEIGHT INSPECTOR (collapsible, 240px tall)                          │
│ [heatmap] [sliders] [sensitivity] [history]                         │
│  layer2.w[3,7] = 0.42  [─────●─────]  sensitivity: ████████░░ 0.81 │
└────────────────────────────────────────────────────────────────────┘
```

### Why three columns?

The whole point is **cause and effect**. User changes input → sees path light
up → sees prediction move. Hide any column and the loop breaks.

---

## Column 1: Input Panel (left, 320px)

### Chat/textbook interface
- Text input box at the bottom (always visible)
- User types: `[1.0, 2.0, 3.0, 4.0]` or `"Hello world"` or uploads a file
- Enter key sends the input to the model
- Previous messages shown above (scrollable history)
- Batch mode: one input per line

### Example inputs
- Quick-fill buttons with sample data from the training set
- "Try a random training sample" button
- "Try an out-of-distribution sample" button

### File upload
- Drag-and-drop CSV, image, or text file
- Auto-detects format and converts to model input

---

## Column 2: Architecture Canvas (center, flexible)

### Rendering modes (auto-switched by model size)

| Mode | Trigger | What you see |
|------|----------|-------------|
| **Full** | <10K params | Every node + every weight, all editable |
| **Layer-summary** | 10K–1M params | Layer boxes with histograms; click to drill |
| **Block-summary** | >1M params | Block-level boxes; click → layer → neuron |

### Forward-pass animation
When the user submits an input:
1. Input layer nodes glow (intensity = activation magnitude)
2. Activation propagates layer by layer (~600ms total)
3. Winning output node gets a golden ring
4. Path through the network is highlighted

### Node labels (the key feature)
Every node is labeled with what it represents:
- **Classifier on tabular data**: `petal_length`, `sepal_width`
- **Transformer**: `"the"`, `"cat"`, `"sat"` (actual tokens)
- **CNN**: `conv1.ch3 (edges)` (channel + detected pattern)
- **KMeans**: `cluster_2 (5.0, 3.4)` (cluster ID + centroid)
- **Random Forest**: `tree_7 → setosa 0.9` (tree + leaf prediction)

### "Focus on active nodes" rendering
Instead of rendering all 1B params:
1. Forward pass runs, captures activations for every layer
2. For each layer, rank neurons by |activation|
3. Render only top-k (default 50) + "47 more hidden" indicator
4. User can click "show all" for full layer

### Interactive elements
- **Click a node** → its weights appear in the bottom inspector
- **Click a layer** → expands to show all neurons in that layer
- **Hover** → tooltip with exact value + contribution score
- **Drag** → pan the canvas
- **Scroll** → zoom in/out
- **Right-click** → context menu (freeze, mask, edit weight)

### Layout options
- **Vertical** (default): layers stacked top-to-bottom
- **Horizontal**: layers left-to-right (wider models)
- **Radial**: for graphs and trees (GNNs, decision trees)

---

## Column 3: Output Panel (right, 320px)

### Prediction display
- Large text showing the prediction value
- For classification: class name + colored badge
- For regression: numerical value + unit
- For generation: streaming text output

### Confidence bar
- Horizontal bar (0-100%)
- Color: green (>80%), yellow (50-80%), red (<50%)
- Click to see full probability distribution

### Metrics
- **Latency**: milliseconds for this prediction
- **Tokens/s**: for generative models
- **Params active**: "1.2K / 4.3M (0.03%)" — how much of the model was used
- **Surprise score**: how different is this from the training distribution

### Explanation
- Top 3 contributing features (with bars)
- "This prediction was most influenced by petal_length (0.83)"
- Before/after diff when weights are edited

---

## Bottom: Weight Inspector (collapsible, 240px)

### Tabs
1. **Heatmap** — weight matrix as a color-coded grid
2. **Sliders** — top-K most influential weights as draggable sliders
3. **Sensitivity** — response curve for each weight (sweep → plot)
4. **History** — undo/redo stack of weight edits

### Slider behavior
- Each slider controls one weight
- Range: [μ−3σ, μ+3σ] by default (prevents NaN)
- Drag → 200ms debounced re-predict → output updates
- Output shows diff: "weight 0.42→0.67, prediction flipped: A→B"
- Undo/redo stack
- "Reset to trained" button

### Sensitivity sweep
- Pick a weight
- Click "Sweep" → UI scrubs from min to max over 2 seconds
- Output plots response curve: prediction vs weight value

---

## Status Bar (bottom)

```
tokens: 47 in / 12 out │ latency: 23ms │ conf: 94.2% │ mem: 2.1MB │
tokens/sec: 521 │ params active: 1.2K / 4.3M (0.03%) │ #experts: 2/8
```

---

## Family-Specific Visualizations

### MoE (Mixture of Experts)
- Router heatmap showing routing distribution for current input
- Expert utilization timeline (stacked area chart)
- Toggle individual experts on/off → see prediction change

### Attention (Transformer)
- Attention matrix as a heatmap (query × key)
- Top-K attention links as curved arrows
- Click a head → see what it attends to

### Decision Tree
- Tree structure with highlighted path (root → leaf)
- Split thresholds shown as draggable lines
- "What if sepal_length was 6.3?" → path updates

### KMeans
- 2D scatter with colored clusters
- Drag a centroid → see points reassign
- Click a cluster → see its members

### CNN
- Feature map tiles for each conv channel
- Receptive field overlay on input image
- Activation heatmaps

### RL Agent
- Environment state as a small grid
- Q-values as a bar chart (one bar per action)
- Action probabilities animate as state changes

---

## Responsive Design

| Device | Layout |
|--------|--------|
| Desktop ≥1200px | Three columns side-by-side |
| Tablet 768-1199px | Canvas (50%) + chat (50%); inspector as drawer |
| Mobile <768px | Tabbed: [canvas] [chat] [inspect]; swipe to switch |

---

## Blind Spot Detection

The "Health" tab in the output panel:
- **Dead neurons**: gray X overlay (never fire across 100 inputs)
- **Saturated activations**: yellow border (stuck at tanh±1)
- **Overconfidence**: red if input is out-of-distribution
- **Routing collapse** (MoE): red if one expert gets >95% traffic
- **Attention sink** (transformers): highlight token that hogs attention

---

## Creative Features

### Story Mode
Guided tour: "Step 1: Input enters embedding. Step 2: Token 'cat' activates
neuron 247 (weight 0.83). Step 3: Attention head 4 connects 'cat' to 'sat'.
Step 4: Top logit is 'on'."

### Counterfactual Explorer
"What's the smallest change that flips the prediction?" → gradient × input
→ modify most influential feature → show diff

### Compare Two Inputs
Pin input A, send input B → two forward passes overlaid → differing paths
highlighted in orange

### Adversarial Probe
One-click "add tiny noise that breaks the model" → shows vulnerability

### Training Replay
Slider that scrubs through training epochs → watch confidence evolve

### Export the Story
Self-contained HTML with model + input + visualization + chat → shareable

---

## Tech Stack — Decision (locked)

The Visual Workspace is **pure HTML/CSS/JS** — no build step, no React, no
D3 install. It opens in any browser via `qai.workspace(model)` which serves
a static HTML page plus a small Flask JSON backend.

This is decision #9 in [ROADMAP.md](ROADMAP.md) — chosen deliberately so
that:

- No `npm install` or `node_modules/` is ever required
- The HTML can be embedded directly in a notebook, blog post, or HTML email
- The same file works in DEMO mode (no model) and STORY mode (static snapshot)
- The HTTP backend exposes 7 JSON endpoints that any client can call

The Canvas API is used for the architecture canvas (handles thousands of
nodes without DOM overhead). The weight inspector uses native HTML range
inputs. The chat panel uses a plain `<div>` with text content.

If a richer client is ever needed, the JSON endpoints are stable and any
React/Vue/Svelte app can consume them — but the reference implementation
ships as pure HTML/CSS/JS.

---

## Implementation Priority

1. Three-column layout (HTML/CSS grid)
2. Architecture canvas (SVG for small, Canvas for large)
3. Weight inspector (heatmap + sliders)
4. Forward-pass animation
5. Node labels with actual tokens/features
6. "Focus on active nodes" rendering
7. Responsive breakpoints
8. Family-specific visualizations
9. Story mode, counterfactual, compare, adversarial
10. Export the story
