# `qai.advanced_architectures`

> qai.advanced_architectures — production-grade modern architectures.

**Module summary:**
- Classes: 10 | Functions: 0 | Constants: 1
- Source: `qai/advanced_architectures.py`

## Table of Contents

**Classes:**
- [`RetNet`](#retnet)
- [`xLSTM`](#xlstm)
- [`PerceiverIO`](#perceiverio)
- [`ModernHopfield`](#modernhopfield)
- [`KAN`](#kan)
- [`NeuralODE`](#neuralode)
- [`PINN`](#pinn)
- [`FNO`](#fno)
- [`DiT`](#dit)
- [`VQVAE`](#vqvae)

**Constants:**
- [`ADVANCED_ARCHITECTURES`](#advanced_architectures)

## Classes

<a id="retnet"></a>
### `RetNet`  
`CLASS` | `qai.advanced_architectures.RetNet`

**Signature:**
```python
class RetNet(Technique)
```

> RetNet — Retention Network (Sun et al. 2023).

RetNet — Retention Network (Sun et al. 2023).

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `vocab_size` | `int` | `1000` | vocabulary size |
| `dim` | `int` | `64` | model dimension (d_model) |
| `n_layers` | `int` | `2` | number of retention blocks |
| `n_heads` | `int` | `4` | number of retention heads |
| `ffn_multiplier` | `int` | `4` | FFN inner dim = ffn_multiplier * dim |
| `max_seq_len` | `int` | `256` | maximum sequence length |
| `n_classes` | `Optional[int]` | `None` | if set, classification head; else causal LM head
epochs, lr, batch_size: training params |
| `epochs` | `int` | `10` |  |
| `lr` | `float` | `0.0003` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(vocab_size: int = 1000, dim: int = 64, n_layers: int = 2, n_heads: int = 4, ffn_multiplier: int = 4, max_seq_len: int = 256, n_classes: Optional[int] = None, epochs: int = 10, lr: float = 0.0003, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures.py:132`._

---

<a id="xlstm"></a>
### `xLSTM`  
`CLASS` | `qai.advanced_architectures.xLSTM`

**Signature:**
```python
class xLSTM(Technique)
```

> xLSTM — Extended LSTM (Beck et al. 2024).

xLSTM — Extended LSTM (Beck et al. 2024).

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input_size` | `int` | `4` | input feature dimension |
| `hidden_size` | `int` | `32` | LSTM hidden dimension |
| `n_layers` | `int` | `1` | number of stacked mLSTM layers |
| `n_classes` | `Optional[int]` | `None` | if set, classification head; else regression
epochs, lr, batch_size: training params |
| `epochs` | `int` | `10` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(input_size: int = 4, hidden_size: int = 32, n_layers: int = 1, n_classes: Optional[int] = None, epochs: int = 10, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures.py:290`._

---

<a id="perceiverio"></a>
### `PerceiverIO`  
`CLASS` | `qai.advanced_architectures.PerceiverIO`

**Signature:**
```python
class PerceiverIO(Technique)
```

> Perceiver IO — modality-agnostic architecture with asymmetric attention.

Perceiver IO — modality-agnostic architecture with asymmetric attention.

Uses a small set of learned latent arrays that attend to the input,
process it among themselves, then attend back to produce outputs of
any shape. Scales linearly with input/output size.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input_dim` | `int` | `4` | input feature dimension |
| `latent_dim` | `int` | `32` | latent array dimension |
| `n_latents` | `int` | `8` | number of latent vectors |
| `n_layers` | `int` | `2` | number of self-attention layers among latents |
| `n_heads` | `int` | `4` | attention heads |
| `output_dim` | `Optional[int]` | `None` | output dimension (n_classes for classification) |
| `n_classes` | `Optional[int]` | `None` | if set, classification head
epochs, lr, batch_size: training params |
| `epochs` | `int` | `10` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(input_dim: int = 4, latent_dim: int = 32, n_latents: int = 8, n_layers: int = 2, n_heads: int = 4, output_dim: Optional[int] = None, n_classes: Optional[int] = None, epochs: int = 10, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures.py:386`._

---

<a id="modernhopfield"></a>
### `ModernHopfield`  
`CLASS` | `qai.advanced_architectures.ModernHopfield`

**Signature:**
```python
class ModernHopfield(Technique)
```

> Modern Hopfield Network — dense associative memory with exponential attention.

Modern Hopfield Network — dense associative memory with exponential attention.

Stores patterns as a weight matrix. Lookup is a single-step attention
that retrieves the closest stored pattern with exponential precision.

Unlike classical Hopfield (which stores ~0.14N patterns for N neurons),
the modern version stores exponentially many patterns (2^N theoretical
limit) and retrieves in one step.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input_dim` | `int` | `32` | dimension of stored patterns |
| `n_patterns` | `int` | `100` | capacity (max patterns storable) |
| `beta` | `float` | `1.0` | inverse temperature (higher = sharper retrieval) |
| `epochs` | `int` | `1` |  |
| `lr` | `float` | `0.001` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(input_dim: int = 32, n_patterns: int = 100, beta: float = 1.0, epochs: int = 1, lr: float = 0.001, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |
| `retrieve()` | `(x, n_steps: int = 1)` | Retrieve a pattern from partial/noisy input. |

_Defined in `qai/advanced_architectures.py:518`._

---

<a id="kan"></a>
### `KAN`  
`CLASS` | `qai.advanced_architectures.KAN`

**Signature:**
```python
class KAN(Technique)
```

> KAN — Kolmogorov-Arnold Network (Liu et al. 2024).

KAN — Kolmogorov-Arnold Network (Liu et al. 2024).

Replaces MLPs with learnable activation functions on EDGES instead of
fixed activations on NODES. Each edge has its own learnable spline-based
activation. Theoretical advantages:
  - Smaller networks for same accuracy
  - Better interpretability (edge activations are visualizable)
  - Faster neural scaling laws

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `4` | input dimension |
| `hidden` | `List[int]` | `[8, 8]` | list of hidden layer widths |
| `out_dim` | `int` | `2` | output dimension |
| `n_knots` | `int` | `8` | number of B-spline knots per edge
epochs, lr, batch_size: training params |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 4, hidden: List[int] = [8, 8], out_dim: int = 2, n_knots: int = 8, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures.py:617`._

---

<a id="neuralode"></a>
### `NeuralODE`  
`CLASS` | `qai.advanced_architectures.NeuralODE`

**Signature:**
```python
class NeuralODE(Technique)
```

> Neural ODE — continuous-depth network via ODE solvers.

Neural ODE — continuous-depth network via ODE solvers.

Instead of discrete layers, defines a vector field dh/dt = f(h, t) and
integrates it from t=0 to t=1 using an ODE solver (Euler by default).
Memory: O(1) (constant, regardless of "depth").

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `2` | input dimension |
| `out_dim` | `int` | `2` | output dimension |
| `hidden` | `int` | `16` | hidden layer width in the ODE function |
| `n_steps` | `int` | `10` | number of Euler integration steps (acts like "depth")
epochs, lr, batch_size: training params |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 2, out_dim: int = 2, hidden: int = 16, n_steps: int = 10, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures.py:704`._

---

<a id="pinn"></a>
### `PINN`  
`CLASS` | `qai.advanced_architectures.PINN`

**Signature:**
```python
class PINN(Technique)
```

> PINN — Physics-Informed Neural Network for PDE solving.

PINN — Physics-Informed Neural Network for PDE solving.

Standard MLP that incorporates PDE residual loss during training.
The network takes spatial/temporal coordinates as input and outputs
the solution field. The loss = data_loss + physics_loss, where
physics_loss is the PDE residual computed via autodiff.

Example PDE (heat equation): du/dt = α d²u/dx²
The user provides the PDE residual function; the trainer handles the
derivatives via torch.autograd.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `2` | input dimension (spatial + temporal coords) |
| `out_dim` | `int` | `1` | output dimension (solution field) |
| `hidden` | `int` | `32` | hidden layer width |
| `n_layers` | `int` | `3` | number of hidden layers |
| `alpha` | `float` | `1.0` | PDE coefficient (e.g. thermal diffusivity) |
| `pde_type` | `str` | `'heat'` | "heat", "wave", "burgers", or "custom"
epochs, lr, batch_size: training params |
| `epochs` | `int` | `100` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 2, out_dim: int = 1, hidden: int = 32, n_layers: int = 3, alpha: float = 1.0, pde_type: str = 'heat', epochs: int = 100, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` | Train PINN. X = collocation points, y = boundary/initial conditions (optional). |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures.py:798`._

---

<a id="fno"></a>
### `FNO`  
`CLASS` | `qai.advanced_architectures.FNO`

**Signature:**
```python
class FNO(Technique)
```

> FNO — Fourier Neural Operator for PDE solving.

FNO — Fourier Neural Operator for PDE solving.

Replaces spatial convolutions with spectral convolutions in Fourier space.
Captures low-frequency global structure efficiently; resolution-invariant
(train on small grid, evaluate on larger grid).

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `in_dim` | `int` | `2` | input dimension (spatial coords) |
| `out_dim` | `int` | `1` | output dimension (solution field) |
| `modes` | `int` | `8` | number of Fourier modes to keep (lower = faster, less accurate) |
| `width` | `int` | `16` | hidden channel width |
| `n_layers` | `int` | `2` | number of spectral conv layers
epochs, lr, batch_size: training params |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(in_dim: int = 2, out_dim: int = 1, modes: int = 8, width: int = 16, n_layers: int = 2, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` |  |

_Defined in `qai/advanced_architectures.py:942`._

---

<a id="dit"></a>
### `DiT`  
`CLASS` | `qai.advanced_architectures.DiT`

**Signature:**
```python
class DiT(Technique)
```

> DiT — Diffusion Transformer.

DiT — Diffusion Transformer.

Replaces the standard U-Net diffusion backbone with a transformer.
Each transformer block is conditioned on the diffusion timestep via
adaptive layer norm (AdaLN).

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input_dim` | `int` | `4` | data dimension (input to be denoised) |
| `hidden` | `int` | `32` | hidden dimension of the transformer |
| `n_layers` | `int` | `2` | number of transformer blocks |
| `n_heads` | `int` | `4` | attention heads |
| `n_classes` | `Optional[int]` | `None` | optional class-conditional generation |
| `n_steps` | `int` | `100` | number of diffusion timesteps
epochs, lr, batch_size: training params |
| `epochs` | `int` | `10` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(input_dim: int = 4, hidden: int = 32, n_layers: int = 2, n_heads: int = 4, n_classes: Optional[int] = None, n_steps: int = 100, epochs: int = 10, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` | Generate: start from x (treated as noise) and denoise. |

_Defined in `qai/advanced_architectures.py:1045`._

---

<a id="vqvae"></a>
### `VQVAE`  
`CLASS` | `qai.advanced_architectures.VQVAE`

**Signature:**
```python
class VQVAE(Technique)
```

> VQ-VAE — Vector-Quantized Variational Autoencoder.

VQ-VAE — Vector-Quantized Variational Autoencoder.

Like a VAE but with a DISCRETE latent space: the encoder outputs continuous
vectors that are quantized to the nearest codebook entry. This gives a
discrete bottleneck useful for representation learning and generation.

The discrete latents can be autoregressively modeled (e.g. with a
transformer) to generate new samples.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input_dim` | `int` | `4` | input data dimension |
| `latent_dim` | `int` | `2` | codebook vector dimension |
| `hidden` | `int` | `16` | encoder/decoder hidden width |
| `n_codes` | `int` | `64` | number of codebook entries (vocabulary size) |
| `commitment_cost` | `float` | `0.25` | weight on the commitment loss
epochs, lr, batch_size: training params |
| `epochs` | `int` | `50` |  |
| `lr` | `float` | `0.001` |  |
| `batch_size` | `int` | `32` |  |
| `**kw` | `` | `` |  |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(input_dim: int = 4, latent_dim: int = 2, hidden: int = 16, n_codes: int = 64, commitment_cost: float = 0.25, epochs: int = 50, lr: float = 0.001, batch_size: int = 32, **kw)` |  |
| `parameters()` | `()` |  |
| `fit()` | `(X, y = None)` |  |
| `forward()` | `(x)` | Reconstruct input through the VQ-VAE. |
| `encode()` | `(x)` | Encode input to discrete codebook indices. |
| `decode()` | `(indices)` | Decode codebook indices back to data. |

_Defined in `qai/advanced_architectures.py:1183`._

---

## Constants

<a id="advanced_architectures"></a>
### `ADVANCED_ARCHITECTURES`  
`CONSTANT` | `qai.advanced_architectures.ADVANCED_ARCHITECTURES`

**Signature:**
```python
ADVANCED_ARCHITECTURES = {'retnet': RetNet, 'xlstm': xLSTM, 'perceiver_io': PerceiverIO, 'modern_hopfield': ModernHopfield, 'kan': KAN, 'neural_ode': NeuralODE, 'pinn': PINN, 'fno': FNO, 'dit': DiT, 'vq_vae': VQVAE}
```

_Defined in `qai/advanced_architectures.py:1310`._

---
