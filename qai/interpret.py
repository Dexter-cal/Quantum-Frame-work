"""
qai.interpret -- mechanistic interpretability toolkit (2025 era).

Five core tools inspired by Anthropic / OpenAI / DeepMind interpretability work:

  1. ActivationLogger      -- capture every module's forward activations
  2. ActivationPatcher     -- patch activations from one run into another
  3. CircuitTracer         -- find minimal circuits responsible for a behavior
  4. SparseAutoencoder     -- SAE for finding monosemantic features (Bricken 2023)
  5. LogitLens             -- project intermediate hidden states to vocab logits
  6. AttributionPatching   -- fast approximate patching via gradients
"""
from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# ============================================================================
# 1. Activation Logger
# ============================================================================
class ActivationLogger:
    """Register forward hooks on every named module; store activations."""

    def __init__(self, model):
        self.model = model
        self.activations: Dict[str, np.ndarray] = {}
        self._hooks: List[Any] = []

    def attach(self) -> "ActivationLogger":
        if not _TORCH:
            return self
        for name, module in self.model.named_modules():
            if not name:
                continue
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)
        return self

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            with torch.no_grad():
                if isinstance(out, torch.Tensor):
                    self.activations[name] = out.detach().cpu().numpy().copy()
                elif isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                    self.activations[name] = out[0].detach().cpu().numpy().copy()
        return hook

    def run(self, x) -> Dict[str, np.ndarray]:
        self.activations.clear()
        with torch.no_grad():
            self.model(x)
        return self.activations

    def get(self, name: str) -> Optional[np.ndarray]:
        return self.activations.get(name)

    def top_k_neurons(self, layer_name: str, k: int = 10) -> List[Tuple[int, float]]:
        """Return indices of the k highest-activating neurons in a layer."""
        a = self.activations.get(layer_name)
        if a is None:
            return []
        # average across batch + sequence dims
        while a.ndim > 1:
            a = a.mean(axis=0)
        idx = np.argsort(a)[-k:][::-1]
        return [(int(i), float(a[i])) for i in idx]


# ============================================================================
# 2. Activation Patcher
# ============================================================================
class ActivationPatcher:
    """Run the model on x_A, capture activations; then run on x_B with
    selected layers' activations patched from x_A. See whether the
    prediction flips.
    """
    def __init__(self, model):
        self.model = model
        self.logger_a = ActivationLogger(model)
        self.logger_b = ActivationLogger(model)

    def patch(self, x_a, x_b, layers_to_patch: List[str]) -> Dict[str, Any]:
        # capture A's activations
        self.logger_a.attach()
        with torch.no_grad():
            out_a = self.model(x_a)
        self.logger_a.detach()

        # patch hooks: replace B's activation with A's at specified layers
        patches = {n: self.logger_a.activations[n] for n in layers_to_patch
                   if n in self.logger_a.activations}

        def make_patch_hook(name, saved):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor) and saved.shape == out.shape:
                    out.copy_(torch.from_numpy(saved))
            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if name in patches:
                h = module.register_forward_hook(make_patch_hook(name, patches[name]))
                hooks.append(h)
        with torch.no_grad():
            out_b_patched = self.model(x_b)
        for h in hooks:
            h.remove()
        return {"out_a": out_a, "out_b_patched": out_b_patched,
                "patched_layers": list(patches.keys())}


# ============================================================================
# 3. Circuit Tracer (simplified)
# ============================================================================
class CircuitTracer:
    """Find the minimal set of edges (module -> module) that causally
    contribute to a specific output logit.

    Uses edge-ablation: zero out each edge's output and measure the logit
    drop. Edges with drop > threshold form the circuit.
    """
    def __init__(self, model, threshold: float = 0.05):
        self.model = model
        self.threshold = threshold

    def trace(self, x, target_logit: int) -> Dict[str, Any]:
        if not _TORCH:
            return {"circuit": []}
        # baseline output
        with torch.no_grad():
            base_out = self.model(x)
        if not isinstance(base_out, torch.Tensor):
            return {"circuit": [], "reason": "model output is not a tensor"}
        base_logit = float(base_out.flatten()[target_logit])

        # ablate each module's output individually
        contributions = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            # zero-out hook
            def make_hook():
                def hook(m, inp, out):
                    if isinstance(out, torch.Tensor):
                        out.zero_()
                return hook
            h = module.register_forward_hook(make_hook())
            with torch.no_grad():
                ablated_out = self.model(x)
            h.remove()
            if isinstance(ablated_out, torch.Tensor):
                ablated_logit = float(ablated_out.flatten()[target_logit])
                drop = base_logit - ablated_logit
                if drop > self.threshold:
                    contributions.append({"module": name, "drop": drop})
        contributions.sort(key=lambda c: -c["drop"])
        return {"circuit": contributions, "base_logit": base_logit,
                "target_logit": target_logit}


# ============================================================================
# 4. Sparse Autoencoder (Bricken et al. 2023)
# ============================================================================
class SparseAutoencoder(nn.Module if _TORCH else object):
    """SAE for finding monosemantic features in dense activations.

    encoder: x (d_model) -> z (n_features) with ReLU + L1 sparsity
    decoder: z -> x_hat
    """
    def __init__(self, d_model: int = 64, n_features: int = 256,
                 l1_coef: float = 1e-3, lr: float = 1e-3):
        if not _TORCH:
            raise ImportError("SparseAutoencoder requires PyTorch")
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)
        self.l1_coef = l1_coef
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.feature_freq = np.zeros(n_features)  # track dead vs alive features

    def forward(self, x):
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def fit(self, X, n_steps: int = 100):
        """X: (N, d_model) array of activation samples."""
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        for _ in range(n_steps):
            self.opt.zero_grad()
            x_hat, z = self(X)
            recon_loss = F.mse_loss(x_hat, X)
            l1_loss = z.abs().sum(-1).mean()
            loss = recon_loss + self.l1_coef * l1_loss
            loss.backward()
            self.opt.step()
            # update frequency tracking
            with torch.no_grad():
                alive = (z > 1e-6).float().sum(0).cpu().numpy()
                self.feature_freq = 0.99 * self.feature_freq + 0.01 * alive
        return self

    def encode(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            return F.relu(self.encoder(x)).cpu().numpy()

    def alive_features(self, threshold: float = 0.01) -> List[int]:
        return [int(i) for i, f in enumerate(self.feature_freq) if f > threshold]

    def dead_feature_count(self, threshold: float = 0.01) -> int:
        return int((self.feature_freq < threshold).sum())

    def top_features(self, x, k: int = 10) -> List[Tuple[int, float]]:
        z = self.encode(x)
        idx = np.argsort(z)[-k:][::-1]
        return [(int(i), float(z[i])) for i in idx if z[i] > 0]


# ============================================================================
# 5. Logit Lens -- project intermediate hidden states to vocab logits
# ============================================================================
class LogitLens:
    """Apply the model's final unembedding to intermediate hidden states.

    Reveals what the model "thinks" at each layer.
    """
    def __init__(self, model, embed_layer_name: str = "embed",
                 unembed_layer_name: str = "lm_head"):
        self.model = model
        self.embed_name = embed_layer_name
        self.unembed_name = unembed_layer_name

    def _find_module(self, name: str):
        for n, m in self.model.named_modules():
            if n == name:
                return m
        return None

    def project(self, hidden_states: np.ndarray) -> np.ndarray:
        """Project hidden states to vocab logits using unembed."""
        unembed = self._find_module(self.unembed_name)
        if unembed is None:
            return hidden_states
        with torch.no_grad():
            h = torch.as_tensor(hidden_states, dtype=torch.float32)
            logits = unembed(h)
        return logits.cpu().numpy()

    def lens(self, x) -> List[Dict[str, Any]]:
        """Run the model with activation logging; project each layer."""
        logger = ActivationLogger(self.model).attach()
        with torch.no_grad():
            self.model(x)
        logger.detach()
        results = []
        for name, act in logger.activations.items():
            if act.ndim < 2:
                continue
            logits = self.project(act)
            top_token = int(np.argmax(logits.mean(axis=tuple(range(logits.ndim - 1)))))
            results.append({"layer": name, "top_token": top_token,
                             "mean_logit": float(logits.mean())})
        return results


# ============================================================================
# 6. Attribution Patching (fast approximate via gradients)
# ============================================================================
class AttributionPatcher:
    """Use gradients to estimate the effect of patching each activation.

    Approximation: patch_effect(x_a, x_b) ≈ grad * (x_a - x_b)
    Much faster than running N ablation passes.
    """
    def __init__(self, model):
        self.model = model

    def attribute(self, x_a, x_b, target_logit: int) -> Dict[str, Any]:
        if not _TORCH:
            return {"attributions": {}}
        x_a = torch.as_tensor(np.asarray(x_a, dtype=np.float32), requires_grad=True)
        x_b = torch.as_tensor(np.asarray(x_b, dtype=np.float32))
        self.model.zero_grad()
        out_a = self.model(x_a)
        if not isinstance(out_a, torch.Tensor):
            return {"attributions": {}}
        target = out_a.flatten()[target_logit]
        target.backward()
        # attribution = grad(x_a) * (x_a - x_b)
        grad = x_a.grad
        diff = x_a.detach() - x_b
        attribution = (grad * diff).sum().item()
        return {"total_attribution": attribution,
                "target_logit": target_logit,
                "grad_norm": float(grad.norm().item()),
                "diff_norm": float(diff.norm().item())}


# Need F for SAE
try:
    import torch.nn.functional as F
except ImportError:
    pass


__all__ = [
    "ActivationLogger", "ActivationPatcher", "CircuitTracer",
    "SparseAutoencoder", "LogitLens", "AttributionPatcher",
]
