# `qai.advanced_architectures_2`

> qai.advanced_architectures_2 — second batch of advanced architectures.

**Module summary:**
- Classes: 15 | Functions: 0 | Constants: 1
- Source: `qai/advanced_architectures_2.py`

## Table of Contents

**Classes:**
- [`SwitchTransformer`](#switchtransformer)
- [`Mixtral`](#mixtral)
- [`YOLO`](#yolo)
- [`SAM`](#sam)
- [`ControlNet`](#controlnet)
- [`LatentDiffusion`](#latentdiffusion)
- [`DeepONet`](#deeponet)
- [`NeuralSDE`](#neuralsde)
- [`LiquidStateMachine`](#liquidstatemachine)
- [`Hyena`](#hyena)
- [`S5`](#s5)
- [`EpisodicMemoryNetwork`](#episodicmemorynetwork)
- [`SparseAccessMemory`](#sparseaccessmemory)
- [`ProductKeyMemory`](#productkeymemory)
- [`FastWeightNetwork`](#fastweightnetwork)

**Constants:**
- [`ADVANCED_ARCHITECTURES_2`](#advanced_architectures_2)

## Classes

<a id="switchtransformer"></a>
### `SwitchTransformer`  
`CLASS` | `qai.advanced_architectures_2.SwitchTransformer`

**Signature:**
```python
class SwitchTransformer(Technique)
```

> Switch Transformer — sparse MoE with top-1 expert routing (Fedus 2021).

Switch Transformer — sparse MoE with top-1 expert routing (Fedus 2021).

Each token is routed to exactly 1 of N experts, drastically reducing
compute at inference (vs. dense MoE that uses top-K).

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `vocab_size` | `int` | `1000` | vocabulary size |
| `dim` | `int` | `64` | model dimension |
| `n_layers` | `int` | `2` | number of switch blocks |
| `n_heads` | `int` | `4` | attention heads |
| `n_experts` | `int` | `4` | number of experts per layer |
| `n_classes` | `Optional[int]` | `None` | if set, classification head
epochs, lr, batch_size: training params |
| `epochs` | `int` | `10` |  |
| `lr` | `float` | `0.0003` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(vocab_size: int = 1000, dim: int = 64, n_layers: int = 2, n_heads: int = 4, n_experts: int = 4, n_classes: Optional[int] = None, epochs: int = 10, lr: float = 0.0003, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:91`._

---

<a id="mixtral"></a>
### `Mixtral`  
`CLASS` | `qai.advanced_architectures_2.Mixtral`

**Signature:**
```python
class Mixtral(Technique)
```

> Mixtral — sparse MoE decoder LLM (Jiang et al. 2023).

Mixtral — sparse MoE decoder LLM (Jiang et al. 2023).

Like Switch Transformer but with top-K routing (K=2 by default), so each
token is processed by 2 experts and their outputs are weighted.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `vocab_size` | `int` | `1000` |  |
| `dim` | `int` | `64` |  |
| `n_layers` | `int` | `2` |  |
| `n_heads` | `int` | `4` |  |
| `n_experts` | `int` | `8` |  |
| `top_k` | `int` | `2` |  |
| `n_classes` | `Optional[int]` | `None` |  |
| `epochs` | `int` | `10` |  |
| `lr` | `float` | `0.0003` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(vocab_size: int = 1000, dim: int = 64, n_layers: int = 2, n_heads: int = 4, n_experts: int = 8, top_k: int = 2, n_classes: Optional[int] = None, epochs: int = 10, lr: float = 0.0003, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:188`._

---

<a id="yolo"></a>
### `YOLO`  
`CLASS` | `qai.advanced_architectures_2.YOLO`

**Signature:**
```python
class YOLO(Technique)
```

> YOLO — You Only Look Once (simplified, Redmon et al. 2016).

YOLO — You Only Look Once (simplified, Redmon et al. 2016).

Single-stage object detection: image → grid → per-cell bbox + class
predictions. The output is (S, S, B*5 + C) where S=grid size, B=bboxes
per cell, C=n_classes.

This is a simplified single-class YOLO. Real YOLO uses DarkNet backbone.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `img_size` | `int` | `32` | input image size (square) |
| `grid_size` | `int` | `4` | detection grid (S x S) |
| `n_classes` | `int` | `1` | number of object classes |
| `n_boxes` | `int` | `1` | bounding boxes per grid cell |
| `epochs` | `int` | `10` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `8` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(img_size: int = 32, grid_size: int = 4, n_classes: int = 1, n_boxes: int = 1, epochs: int = 10, lr: float = 0.001, batch_size: int = 8, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:317`._

---

<a id="sam"></a>
### `SAM`  
`CLASS` | `qai.advanced_architectures_2.SAM`

**Signature:**
```python
class SAM(Technique)
```

> SAM — Segment Anything Model (simplified, Kirillov 2023).

SAM — Segment Anything Model (simplified, Kirillov 2023).

Image + prompt → segmentation mask. Real SAM uses ViT backbone + prompt
encoder + mask decoder. This simplified version uses a small CNN.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `img_size` | `int` | `32` | input image size |
| `hidden` | `int` | `32` | backbone hidden dim |
| `epochs` | `int` | `10` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `8` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(img_size: int = 32, hidden: int = 32, epochs: int = 10, lr: float = 0.001, batch_size: int = 8, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:410`._

---

<a id="controlnet"></a>
### `ControlNet`  
`CLASS` | `qai.advanced_architectures_2.ControlNet`

**Signature:**
```python
class ControlNet(Technique)
```

> ControlNet — conditional diffusion with side-network (Zhang 2023).

ControlNet — conditional diffusion with side-network (Zhang 2023).

Adds a trainable side-network that injects conditioning (e.g. edges,
depth maps) into a frozen base model. The base is unaffected; the side
learns to add the conditioning.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input_dim` | `int` | `4` | data dimension |
| `condition_dim` | `int` | `4` | conditioning signal dimension |
| `hidden` | `int` | `32` | hidden width |
| `n_steps` | `int` | `100` | diffusion timesteps |
| `epochs` | `int` | `10` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(input_dim: int = 4, condition_dim: int = 4, hidden: int = 32, n_steps: int = 100, epochs: int = 10, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, conditions = None)` |  |
| `forward()` | `(x, condition = None)` |  |

_Defined in `qai/advanced_architectures_2.py:487`._

---

<a id="latentdiffusion"></a>
### `LatentDiffusion`  
`CLASS` | `qai.advanced_architectures_2.LatentDiffusion`

**Signature:**
```python
class LatentDiffusion(Technique)
```

> Latent Diffusion — diffusion in a compressed latent space.

Latent Diffusion — diffusion in a compressed latent space.

Two-stage: (1) autoencoder compresses data to latent, (2) diffusion runs
in latent space. Much faster than pixel-space diffusion.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input_dim` | `int` | `4` | data dimension |
| `latent_dim` | `int` | `2` | compressed latent dimension |
| `hidden` | `int` | `16` | hidden width |
| `n_steps` | `int` | `100` | diffusion timesteps |
| `epochs` | `int` | `10` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(input_dim: int = 4, latent_dim: int = 2, hidden: int = 16, n_steps: int = 100, epochs: int = 10, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:617`._

---

<a id="deeponet"></a>
### `DeepONet`  
`CLASS` | `qai.advanced_architectures_2.DeepONet`

**Signature:**
```python
class DeepONet(Technique)
```

> DeepONet — operator learning for PDEs (Lu et al. 2021).

DeepONet — operator learning for PDEs (Lu et al. 2021).

Learns a mapping between function spaces: takes function evaluations as
input, produces function evaluations as output. Two networks:
  - Branch: encodes the input function evaluated at fixed points
  - Trunk:  encodes the query locations
Output = sum_k branch_k(x) * trunk_k(y)

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_sensors` | `int` | `8` | number of input function evaluation points |
| `n_outputs` | `int` | `4` | number of output function evaluation points |
| `hidden` | `int` | `32` | hidden width of branch and trunk nets |
| `n_basis` | `int` | `8` |  |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(n_sensors: int = 8, n_outputs: int = 4, hidden: int = 32, n_basis: int = 8, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` | X: (B, n_sensors) — input function evaluations |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:717`._

---

<a id="neuralsde"></a>
### `NeuralSDE`  
`CLASS` | `qai.advanced_architectures_2.NeuralSDE`

**Signature:**
```python
class NeuralSDE(Technique)
```

> Neural SDE — stochastic differential equation network.

Neural SDE — stochastic differential equation network.

Like Neural ODE but with a stochastic diffusion term. Models:
    dh = f(h, t) dt + g(h, t) dW
where dW is Brownian motion. Useful for probabilistic time series.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `2` | input dimension |
| `out_dim` | `int` | `2` | output dimension |
| `hidden` | `int` | `16` | hidden width |
| `n_steps` | `int` | `10` | Euler-Maruyama integration steps |
| `noise_scale` | `float` | `0.1` | diffusion coefficient (g) |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 2, out_dim: int = 2, hidden: int = 16, n_steps: int = 10, noise_scale: float = 0.1, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:803`._

---

<a id="liquidstatemachine"></a>
### `LiquidStateMachine`  
`CLASS` | `qai.advanced_architectures_2.LiquidStateMachine`

**Signature:**
```python
class LiquidStateMachine(Technique)
```

> Liquid State Machine — spiking reservoir computing (Maass 2002).

Liquid State Machine — spiking reservoir computing (Maass 2002).

Input → random spiking reservoir → linear readout. The reservoir
transforms temporal input into a high-dimensional "liquid state" that
a simple linear readout can decode.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `4` | input dimension |
| `n_reservoir` | `int` | `50` | number of spiking neurons |
| `out_dim` | `int` | `2` | output dimension |
| `spectral_radius` | `float` | `0.9` | controls reservoir dynamics (typical: 0.9-0.95) |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 4, n_reservoir: int = 50, out_dim: int = 2, spectral_radius: float = 0.9, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:906`._

---

<a id="hyena"></a>
### `Hyena`  
`CLASS` | `qai.advanced_architectures_2.Hyena`

**Signature:**
```python
class Hyena(Technique)
```

> Hyena — long convolution alternative to attention (Poli 2023).

Hyena — long convolution alternative to attention (Poli 2023).

Replaces self-attention with a long implicit convolution (via FFT).
Scales subquadratically: O(N log N) instead of O(N²).

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `4` | input dimension |
| `hidden` | `int` | `32` | hidden width |
| `n_layers` | `int` | `2` | number of Hyena blocks |
| `n_classes` | `Optional[int]` | `None` | if set, classification head |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 4, hidden: int = 32, n_layers: int = 2, n_classes: Optional[int] = None, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:1013`._

---

<a id="s5"></a>
### `S5`  
`CLASS` | `qai.advanced_architectures_2.S5`

**Signature:**
```python
class S5(Technique)
```

> S5 — Simplified Structured State Space Model (Smith 2022).

S5 — Simplified Structured State Space Model (Smith 2022).

Simplification of S4 with a single multi-input multi-output (MIMO) state.
Parallelizable during training, recurrent during inference.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `4` | input dimension |
| `out_dim` | `int` | `2` | output dimension |
| `state_dim` | `int` | `16` | SSM state dimension |
| `hidden` | `int` | `32` | hidden width |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 4, out_dim: int = 2, state_dim: int = 16, hidden: int = 32, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:1130`._

---

<a id="episodicmemorynetwork"></a>
### `EpisodicMemoryNetwork`  
`CLASS` | `qai.advanced_architectures_2.EpisodicMemoryNetwork`

**Signature:**
```python
class EpisodicMemoryNetwork(Technique)
```

> Episodic Memory Network — external memory for OOD generalization.

Episodic Memory Network — external memory for OOD generalization.

Stores (key, value) pairs in an external memory matrix. At inference,
retrieves the top-K most similar memory entries and uses them to inform
the prediction. Allows the model to "remember" specific examples.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `4` | input dimension |
| `out_dim` | `int` | `2` | output dimension |
| `hidden` | `int` | `16` | hidden width |
| `n_memory` | `int` | `100` | number of memory slots |
| `key_dim` | `int` | `8` | key dimension for memory lookup |
| `top_k` | `int` | `3` | number of memory slots to retrieve per query |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 4, out_dim: int = 2, hidden: int = 16, n_memory: int = 100, key_dim: int = 8, top_k: int = 3, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:1231`._

---

<a id="sparseaccessmemory"></a>
### `SparseAccessMemory`  
`CLASS` | `qai.advanced_architectures_2.SparseAccessMemory`

**Signature:**
```python
class SparseAccessMemory(Technique)
```

> Sparse Access Memory — sparse read/write external memory.

Sparse Access Memory — sparse read/write external memory.

Unlike DNC/NTM (which read all memory slots), Sparse Access Memory
reads only the top-K most relevant slots. Scales to large memory sizes.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `4` | input dimension |
| `out_dim` | `int` | `2` | output dimension |
| `hidden` | `int` | `16` | controller hidden width |
| `n_memory` | `int` | `32` | number of memory slots |
| `mem_dim` | `int` | `8` | memory vector dimension |
| `top_k` | `int` | `4` | number of slots to read |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 4, out_dim: int = 2, hidden: int = 16, n_memory: int = 32, mem_dim: int = 8, top_k: int = 4, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:1346`._

---

<a id="productkeymemory"></a>
### `ProductKeyMemory`  
`CLASS` | `qai.advanced_architectures_2.ProductKeyMemory`

**Signature:**
```python
class ProductKeyMemory(Technique)
```

> Product-Key Memory — huge memory via product keys (Lample 2019).

Product-Key Memory — huge memory via product keys (Lample 2019).

Memory with N² slots but only 2N key parameters: keys are constructed
as the outer product of two N-dim key sets. Allows billions of memory
slots with minimal parameters.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `4` | input dimension |
| `out_dim` | `int` | `2` | output dimension |
| `hidden` | `int` | `16` | hidden width |
| `n_keys` | `int` | `16` | dimension of each key set (memory has n_keys² slots) |
| `key_dim` | `int` | `4` | dimension of each key vector |
| `top_k` | `int` | `4` |  |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 4, out_dim: int = 2, hidden: int = 16, n_keys: int = 16, key_dim: int = 4, top_k: int = 4, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:1460`._

---

<a id="fastweightnetwork"></a>
### `FastWeightNetwork`  
`CLASS` | `qai.advanced_architectures_2.FastWeightNetwork`

**Signature:**
```python
class FastWeightNetwork(Technique)
```

> Fast-Weight Network — fast-weight programmers (Schlag 2021).

Fast-Weight Network — fast-weight programmers (Schlag 2021).

Instead of fixed weights, the network "programs" a fast-weight matrix
on the fly from the input. The fast weights are then used to process
the same input (or a follow-up input). Allows O(N²) capacity with
O(N) parameters.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `4` | input dimension |
| `out_dim` | `int` | `2` | output dimension |
| `hidden` | `int` | `16` | hidden width |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 4, out_dim: int = 2, hidden: int = 16, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures_2.py:1580`._

---

## Constants

<a id="advanced_architectures_2"></a>
### `ADVANCED_ARCHITECTURES_2`  
`CONSTANT` | `qai.advanced_architectures_2.ADVANCED_ARCHITECTURES_2`

**Signature:**
```python
ADVANCED_ARCHITECTURES_2 = {'switch_transformer': SwitchTransformer, 'mixtral': Mixtral, 'yolo': YOLO, 'sam': SAM, 'controlnet': ControlNet, 'latent_diffusion': LatentDiffusion, 'deeponet': DeepONet, 'neural_sde': NeuralSDE, 'liquid_state_machine': LiquidStateMachine, 'hyena': Hyena, 's5': S5, 'episodic_memory_network': EpisodicMemoryNetwork, 'sparse_access_memory': SparseAccessMemory, 'product_key_memory': ProductKeyMemory, 'fast_weight_network': FastWeightNetwork}
```

_Defined in `qai/advanced_architectures_2.py:1694`._

---
