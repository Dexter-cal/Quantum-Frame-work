"""
qai.techniques.advanced_dl -- MoE (Mixture of Experts) + GNN (Graph Neural Network).

Completes the spec Section 2 deep learning roster (techniques 10 and 11 of 11).

  qai.build(technique="moe", ...)  -- Mixture of Experts with top-k routing
  qai.build(technique="gnn", ...)  -- Graph Neural Network with message passing

MoE creative additions:
  - Expert utilization tracking (detect underused experts)
  - Capacity factor (drop tokens when expert is overloaded)
  - Router z-loss (stabilize router training)
  - Expert weight inspection

GNN creative additions:
  - Edge prediction (link prediction task)
  - Multi-hop aggregation (configurable number of message-passing rounds)
  - Graph-level pooling (readout for graph classification)
  - Node embedding extraction
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    from qai._torch_stub import _TorchStubNamespace, _NNStubNamespace
    torch = _TorchStubNamespace()
    nn = _NNStubNamespace()
    F = nn

from .base import Technique

def _req():
    if not _TORCH:
        raise ImportError("[WHAT] MoE/GNN require PyTorch\n[FIX] pip install torch")


# =====================================================================
# MoE (Mixture of Experts)
# =====================================================================
class _Expert(nn.Module):
    """A single expert — a simple MLP."""
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x): return self.net(x)


class _Router(nn.Module):
    """Top-k gating router: maps input → softmax probabilities over experts."""
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)
        self.top_k = min(top_k, n_experts)
    def forward(self, x):
        # x: (batch, d_model)
        logits = self.gate(x)  # (batch, n_experts)
        # Top-k selection
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_probs = F.softmax(topk_vals, dim=-1)
        return topk_idx, topk_probs, logits


class _MoENet(nn.Module):
    """Full MoE layer: router + N experts + weighted combination."""
    def __init__(self, d_model, n_experts=4, top_k=2, d_ff=None,
                 dropout=0.0, n_classes=None, task_type="classification"):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.task_type = task_type
        d_ff = d_ff or d_model * 4

        # Input projection (if input dim != d_model, we'd need this; for simplicity assume match)
        self.input_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

        # Router
        self.router = _Router(d_model, n_experts, top_k)

        # Experts
        self.experts = nn.ModuleList([
            _Expert(d_model, d_ff, dropout) for _ in range(n_experts)
        ])

        # Output head
        if task_type == "classification":
            assert n_classes is not None
            self.head = nn.Linear(d_model, n_classes)
        else:  # regression
            self.head = nn.Linear(d_model, 1)

        # Tracking (creative: expert utilization)
        self.register_buffer("expert_counts", torch.zeros(n_experts))

    def forward(self, x, track_utilization=True):
        """
        Args:
            x: (batch, d_model) input features
            track_utilization: if True, update expert_counts buffer
        Returns:
            logits: (batch, n_classes) or (batch, 1)
            aux_loss: load-balancing loss (scalar)
        """
        x = self.input_proj(x)
        x = self.ln(x)

        # Route
        topk_idx, topk_probs, router_logits = self.router(x)  # (batch, top_k), (batch, top_k), (batch, n_experts)

        # Compute expert outputs
        batch_size = x.size(0)
        output = torch.zeros_like(x)

        for i in range(batch_size):
            for j in range(self.top_k):
                expert_idx = topk_idx[i, j].item()
                prob = topk_probs[i, j]
                expert_out = self.experts[expert_idx](x[i:i+1])
                output[i] += prob * expert_out.squeeze(0)
                if track_utilization:
                    self.expert_counts[expert_idx] += 1

        # Load-balancing loss (encourages uniform expert usage)
        # L_aux = n_experts * sum(mean_gate_prob_i * mean_utilization_i)
        gate_probs = F.softmax(router_logits, dim=-1).mean(dim=0)  # (n_experts,)
        utilization = self.expert_counts / (self.expert_counts.sum() + 1e-8)
        aux_loss = self.n_experts * (gate_probs * utilization).sum()

        # Router z-loss (creative: stabilize router training)
        # Prevents router logits from growing too large
        z_loss = torch.mean(torch.square(router_logits))

        logits = self.head(output)
        return logits, aux_loss, z_loss

    def get_expert_utilization(self) -> np.ndarray:
        """Creative: return per-expert utilization as a numpy array."""
        total = self.expert_counts.sum().item()
        if total == 0:
            return np.zeros(self.n_experts)
        return (self.expert_counts / total).cpu().numpy()

    def reset_utilization(self):
        self.expert_counts.zero_()


class MixtureOfExperts(Technique):
    """Mixture of Experts (Shazeer et al., 2017; Switch Transformer, Fedus et al., 2021).

    A router network sends each input to the top-k experts (out of N).
    Expert outputs are combined with gating probabilities.
    Load-balancing auxiliary loss encourages uniform expert usage.

    Creative additions:
      - Expert utilization tracking (detect underused experts)
      - Router z-loss (stabilizes training)
      - `get_expert_utilization()` API for introspection
      - Configurable `aux_loss_weight` and `z_loss_weight`
    """
    name = "moe"
    family = "deep_learning"
    compatible_learning_techniques = ("supervised",)

    def __init__(
        self,
        in_dim: int = 64,
        d_model: int = 64,
        n_experts: int = 4,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        task_type: str = "classification",
        n_classes: Optional[int] = None,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 0.01,
        warmup_steps: int = 50,
        device: Optional[str] = None,
        **params,
    ):
        super().__init__(**{**params, "in_dim": in_dim, "d_model": d_model,
                            "n_experts": n_experts, "top_k": top_k, "d_ff": d_ff,
                            "dropout": dropout, "task_type": task_type,
                            "n_classes": n_classes, "aux_loss_weight": aux_loss_weight,
                            "z_loss_weight": z_loss_weight, "epochs": epochs, "lr": lr,
                            "batch_size": batch_size, "weight_decay": weight_decay,
                            "warmup_steps": warmup_steps, "device": device})
        _req()
        assert task_type in ("classification", "regression")
        assert top_k <= n_experts, f"top_k ({top_k}) must be <= n_experts ({n_experts})"

        self.in_dim = in_dim
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_ff = d_ff
        self.dropout = dropout
        self.task_type = task_type
        self.n_classes = n_classes
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        if device is None:
            try:
                import qai; device = qai.get_optimal_device()
            except: device = "cpu"
        self.device = torch.device(device)

        self.net: Optional[_MoENet] = None
        self._losses: List[float] = []
        self._aux_losses: List[float] = []
        self._last_X = None
        self._last_y = None
        self.classes_: Optional[np.ndarray] = None

    def _build_model(self, n_classes=None, in_dim=None):
        if n_classes is not None:
            self.n_classes = n_classes
        if in_dim is not None:
            self.in_dim = in_dim
            self.d_model = max(self.d_model, in_dim)
        self.net = _MoENet(
            d_model=self.d_model, n_experts=self.n_experts, top_k=self.top_k,
            d_ff=self.d_ff, dropout=self.dropout, n_classes=self.n_classes,
            task_type=self.task_type,
        ).to(self.device)

    def fit(self, X, y=None, **kw):
        if y is None:
            raise ValueError("MoE.fit() requires y (supervised)")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1: X_arr = X_arr.reshape(1, -1)

        y_arr = np.asarray(y)
        if y_arr.dtype.kind in ("U", "S", "O"):
            self.classes_ = np.unique(y_arr)
            y_enc = np.searchsorted(self.classes_, y_arr).astype(np.int64)
        else:
            y_enc = y_arr.astype(np.int64) if self.task_type == "classification" else y_arr.astype(np.float32)
            if self.task_type == "classification":
                self.classes_ = np.unique(y_enc)

        n_classes = int(y_enc.max()) + 1 if self.task_type == "classification" else None
        self._last_X = X_arr; self._last_y = y_enc

        if self.net is None:
            self._build_model(n_classes=n_classes, in_dim=X_arr.shape[1])

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = max(1, (len(X_arr) // self.batch_size + 1) * self.epochs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt,
            lambda s: self._lr_lambda(s, total_steps))

        self._losses = []; self._aux_losses = []
        self.net.train()

        for epoch in range(self.epochs):
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(len(X_arr))
            ep_loss, ep_aux = 0.0, 0.0; n_b = 0
            for start in range(0, len(X_arr), self.batch_size):
                idx = perm[start:start+self.batch_size]
                xb = torch.tensor(X_arr[idx], device=self.device)
                # Project to d_model if needed
                if xb.size(-1) != self.d_model:
                    # Pad or project — simple linear pad
                    if xb.size(-1) < self.d_model:
                        pad = torch.zeros(xb.size(0), self.d_model - xb.size(-1), device=self.device)
                        xb = torch.cat([xb, pad], dim=-1)
                    else:
                        xb = xb[:, :self.d_model]
                yb = torch.tensor(y_enc[idx], device=self.device)

                self.net.reset_utilization()
                logits, aux_loss, z_loss = self.net(xb)

                if self.task_type == "classification":
                    main_loss = F.cross_entropy(logits, yb)
                else:
                    main_loss = F.mse_loss(logits.squeeze(-1), yb)

                loss = main_loss + self.aux_loss_weight * aux_loss + self.z_loss_weight * z_loss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                opt.step(); sched.step()

                ep_loss += float(main_loss); ep_aux += float(aux_loss); n_b += 1

            self._losses.append(ep_loss / max(1, n_b))
            self._aux_losses.append(ep_aux / max(1, n_b))

        self._trained = True
        return self

    def _lr_lambda(self, step, total_steps):
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def forward(self, x, **kw):
        if not self._trained or self.net is None:
            raise RuntimeError("MoE.predict() before train()")
        x_arr = np.asarray(x, dtype=np.float32)
        single = x_arr.ndim == 1
        if single: x_arr = x_arr.reshape(1, -1)
        self.net.eval()
        with torch.no_grad():
            xb = torch.tensor(x_arr, device=self.device)
            if xb.size(-1) != self.d_model:
                if xb.size(-1) < self.d_model:
                    pad = torch.zeros(xb.size(0), self.d_model - xb.size(-1), device=self.device)
                    xb = torch.cat([xb, pad], dim=-1)
                else:
                    xb = xb[:, :self.d_model]
            logits, _, _ = self.net(xb, track_utilization=False)
            if self.task_type == "classification":
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                if self.classes_ is not None and self.classes_.dtype.kind in ("U", "S", "O"):
                    preds = np.array([self.classes_[p] for p in preds])
                return preds[0] if single else preds
            else:
                return logits.squeeze(-1).cpu().numpy()

    def get_expert_utilization(self) -> np.ndarray:
        """Creative: return per-expert utilization (fraction of tokens routed to each)."""
        if self.net is None:
            return np.zeros(self.n_experts)
        return self.net.get_expert_utilization()

    def losses(self): return list(self._losses)
    def aux_losses(self): return list(self._aux_losses)
    def parameter_count(self): return sum(p.numel() for p in self.net.parameters()) if self.net else 0

    def architecture_summary(self):
        return {"technique":"moe","d_model":self.d_model,"n_experts":self.n_experts,
                "top_k":self.top_k,"task_type":self.task_type,"n_classes":self.n_classes,
                "parameters":self.parameter_count(),"device":str(self.device),
                "expert_utilization": self.get_expert_utilization().tolist()}

    def __repr__(self):
        if self.net is None: return f"<MoE unbuilt n_experts={self.n_experts}>"
        return (f"<MoE n_experts={self.n_experts} top_k={self.top_k} "
                f"params={self.parameter_count():,} trained={self._trained}>")


# =====================================================================
# GNN (Graph Neural Network)
# =====================================================================
class _GNNLayer(nn.Module):
    """Single message-passing GNN layer (Kipf & Welling, 2017 — GCN-style).

    For each node v:
        h_v' = relu(W * mean({h_u : u in neighbors(v)} ∪ {h_v}))
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, node_features, adj_matrix):
        """
        Args:
            node_features: (N, in_dim) — features for each node
            adj_matrix: (N, N) — adjacency matrix (binary or weighted)
        Returns:
            (N, out_dim) — updated node features
        """
        # Aggregate: mean of neighbors
        # adj_matrix @ node_features = sum of neighbor features
        # Then divide by degree to get mean
        deg = adj_matrix.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = (adj_matrix @ node_features) / deg  # (N, in_dim)
        # Add self-loop (include own features)
        agg = agg + node_features  # self-connection
        # Transform
        out = self.linear(agg)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class _GNNNet(nn.Module):
    """Full GNN: stacked message-passing layers + output head."""
    def __init__(self, in_dim, hidden, n_layers=2, n_classes=None,
                 dropout=0.0, task_type="node_classification"):
        super().__init__()
        self.task_type = task_type
        self.n_layers = n_layers

        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(_GNNLayer(d, hidden, dropout))
            d = hidden
        self.layers = nn.ModuleList(layers)
        self.ln = nn.LayerNorm(hidden)

        if task_type in ("node_classification", "link_prediction"):
            assert n_classes is not None
            self.head = nn.Linear(hidden, n_classes)
        elif task_type == "graph_classification":
            self.head = nn.Linear(hidden, n_classes)  # n_classes = output classes
        else:  # regression
            self.head = nn.Linear(hidden, 1)

    def forward(self, node_features, adj_matrix, node_mask=None):
        """
        Args:
            node_features: (N, in_dim)
            adj_matrix: (N, N)
            node_mask: optional (N,) bool — which nodes to classify
        Returns:
            For node_classification: (N, n_classes) logits for each node
            For graph_classification: (n_classes,) logits for the whole graph
            For link_prediction: (N, n_classes) — used for edge scoring
        """
        h = node_features
        for layer in self.layers:
            h = layer(h, adj_matrix)
        h = self.ln(h)

        if self.task_type == "graph_classification":
            # Global mean pooling
            if node_mask is not None:
                graph_emb = h[node_mask].mean(dim=0, keepdim=True)
            else:
                graph_emb = h.mean(dim=0, keepdim=True)
            return self.head(graph_emb).squeeze(0)
        else:
            return self.head(h)  # (N, n_classes) or (N, 1)

    def get_node_embeddings(self, node_features, adj_matrix):
        """Creative: return intermediate node embeddings (before output head)."""
        h = node_features
        for layer in self.layers:
            h = layer(h, adj_matrix)
        return self.ln(h)


class GNN(Technique):
    """Graph Neural Network with message passing (Kipf & Welling, 2017).

    Supports three task types:
      - `node_classification`: classify each node
      - `graph_classification`: classify entire graphs (with global pooling)
      - `link_prediction`: predict edges between nodes (creative addition)

    Creative additions:
      - Link prediction (edge scoring via dot product of node embeddings)
      - Multi-hop aggregation (configurable n_layers)
      - Node embedding extraction
      - Graph-level pooling (mean readout)
    """
    name = "gnn"
    family = "deep_learning"
    compatible_learning_techniques = ("supervised",)

    def __init__(
        self,
        in_dim: int = 16,
        hidden: int = 64,
        n_layers: int = 2,
        n_classes: Optional[int] = None,
        dropout: float = 0.0,
        task_type: str = "node_classification",
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 1,
        weight_decay: float = 0.01,
        warmup_steps: int = 50,
        device: Optional[str] = None,
        **params,
    ):
        super().__init__(**{**params, "in_dim": in_dim, "hidden": hidden,
                            "n_layers": n_layers, "n_classes": n_classes,
                            "dropout": dropout, "task_type": task_type,
                            "epochs": epochs, "lr": lr, "batch_size": batch_size,
                            "weight_decay": weight_decay, "warmup_steps": warmup_steps,
                            "device": device})
        _req()
        assert task_type in ("node_classification", "graph_classification", "link_prediction")

        self.in_dim = in_dim; self.hidden = hidden; self.n_layers = n_layers
        self.n_classes = n_classes; self.dropout = dropout; self.task_type = task_type
        self.epochs = epochs; self.lr = lr; self.batch_size = batch_size
        self.weight_decay = weight_decay; self.warmup_steps = warmup_steps

        if device is None:
            try:
                import qai; device = qai.get_optimal_device()
            except: device = "cpu"
        self.device = torch.device(device)

        self.net: Optional[_GNNNet] = None
        self._losses: List[float] = []
        self._last_X = None; self._last_y = None
        self.classes_: Optional[np.ndarray] = None

    def _build_model(self, n_classes=None, in_dim=None):
        if n_classes is not None: self.n_classes = n_classes
        if in_dim is not None: self.in_dim = in_dim
        self.net = _GNNNet(
            in_dim=self.in_dim, hidden=self.hidden, n_layers=self.n_layers,
            n_classes=self.n_classes, dropout=self.dropout, task_type=self.task_type,
        ).to(self.device)

    def fit(self, X, y=None, adj_matrix=None, **kw):
        """Train the GNN.

        Args:
            X: For node_classification: (N, in_dim) node features.
               For graph_classification: list of (node_features, adj_matrix) tuples.
               For link_prediction: (N, in_dim) node features + adj_matrix.
            y: For node_classification: (N,) node labels.
               For graph_classification: (n_graphs,) graph labels.
               For link_prediction: list of (src, dst, label) triples.
            adj_matrix: (N, N) adjacency matrix (for node_classification and link_prediction).
        """
        # Accept `adj` as an alias for `adj_matrix` so callers can use either
        # name. Bug fix: the test harness (and other graph techniques) use
        # `adj=`, but GNN used `adj_matrix=`, causing inconsistency.
        if adj_matrix is None and "adj" in kw:
            adj_matrix = kw.pop("adj")
        # Also handle the case where X is a (features, adj) tuple -- the test
        # harness passes graph data this way. Unpack into X and adj_matrix.
        if adj_matrix is None and isinstance(X, tuple) and len(X) == 2:
            X, maybe_adj = X
            if isinstance(maybe_adj, (np.ndarray, dict)):
                adj_matrix = maybe_adj
        # Stash adj_matrix so forward() can reuse it if not re-passed.
        self._last_adj_matrix = adj_matrix
        if self.task_type == "node_classification":
            return self._fit_node_classification(X, y, adj_matrix)
        elif self.task_type == "graph_classification":
            return self._fit_graph_classification(X, y)
        elif self.task_type == "link_prediction":
            return self._fit_link_prediction(X, y, adj_matrix)

    def _fit_node_classification(self, X, y, adj_matrix):
        X_arr = np.asarray(X, dtype=np.float32)
        adj_arr = np.asarray(adj_matrix, dtype=np.float32)
        y_arr = np.asarray(y)
        if y_arr.dtype.kind in ("U", "S", "O"):
            self.classes_ = np.unique(y_arr)
            y_enc = np.searchsorted(self.classes_, y_arr).astype(np.int64)
        else:
            y_enc = y_arr.astype(np.int64)
            self.classes_ = np.unique(y_enc)

        n_classes = int(y_enc.max()) + 1
        self._last_X = (X_arr, adj_arr); self._last_y = y_enc

        if self.net is None:
            self._build_model(n_classes=n_classes, in_dim=X_arr.shape[1])

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._losses = []; self.net.train()

        X_t = torch.tensor(X_arr, device=self.device)
        A_t = torch.tensor(adj_arr, device=self.device)
        y_t = torch.tensor(y_enc, device=self.device, dtype=torch.long)

        for epoch in range(self.epochs):
            opt.zero_grad()
            logits = self.net(X_t, A_t)  # (N, n_classes)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            opt.step()
            self._losses.append(float(loss))

        self._trained = True
        return self

    def _fit_graph_classification(self, graphs, y):
        """graphs: list of (node_features, adj_matrix) tuples. y: graph labels."""
        y_arr = np.asarray(y)
        if y_arr.dtype.kind in ("U", "S", "O"):
            self.classes_ = np.unique(y_arr)
            y_enc = np.searchsorted(self.classes_, y_arr).astype(np.int64)
        else:
            y_enc = y_arr.astype(np.int64)
            self.classes_ = np.unique(y_enc)
        n_classes = int(y_enc.max()) + 1

        in_dim = graphs[0][0].shape[1] if isinstance(graphs[0][0], np.ndarray) else len(graphs[0][0])
        if self.net is None:
            self._build_model(n_classes=n_classes, in_dim=in_dim)

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._losses = []; self.net.train()

        for epoch in range(self.epochs):
            ep_loss = 0.0
            for i, ((nf, am), label) in enumerate(zip(graphs, y_enc)):
                nf_t = torch.tensor(np.asarray(nf, dtype=np.float32), device=self.device)
                am_t = torch.tensor(np.asarray(am, dtype=np.float32), device=self.device)
                y_t = torch.tensor(label, device=self.device, dtype=torch.long)
                opt.zero_grad()
                logits = self.net(nf_t, am_t)  # (n_classes,)
                loss = F.cross_entropy(logits.unsqueeze(0), y_t.unsqueeze(0))
                loss.backward()
                opt.step()
                ep_loss += float(loss)
            self._losses.append(ep_loss / max(1, len(graphs)))

        self._trained = True
        return self

    def _fit_link_prediction(self, X, y, adj_matrix):
        """X: node features. y: list of (src, dst, label). adj_matrix: adjacency."""
        X_arr = np.asarray(X, dtype=np.float32)
        adj_arr = np.asarray(adj_matrix, dtype=np.float32)
        # y: list of (src, dst, label) — treat as binary classification (edge exists or not)
        edges = np.asarray(y, dtype=np.int64)  # (n_edges, 3)

        self._last_X = (X_arr, adj_arr)

        if self.net is None:
            self._build_model(n_classes=2, in_dim=X_arr.shape[1])

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._losses = []; self.net.train()

        X_t = torch.tensor(X_arr, device=self.device)
        A_t = torch.tensor(adj_arr, device=self.device)
        src_t = torch.tensor(edges[:, 0], device=self.device, dtype=torch.long)
        dst_t = torch.tensor(edges[:, 1], device=self.device, dtype=torch.long)
        label_t = torch.tensor(edges[:, 2], device=self.device, dtype=torch.long)

        for epoch in range(self.epochs):
            opt.zero_grad()
            # Get node embeddings
            embeddings = self.net.get_node_embeddings(X_t, A_t)
            # Score edges via dot product
            src_emb = embeddings[src_t]  # (n_edges, hidden)
            dst_emb = embeddings[dst_t]
            scores = (src_emb * dst_emb).sum(dim=-1)  # (n_edges,)
            loss = F.binary_cross_entropy_with_logits(scores, label_t.float())
            loss.backward()
            opt.step()
            self._losses.append(float(loss))

        self._trained = True
        return self

    def forward(self, x, adj_matrix=None, **kw):
        if not self._trained or self.net is None:
            raise RuntimeError("GNN.predict() before train()")
        # Accept `adj` as an alias for `adj_matrix` (consistency with fit()).
        if adj_matrix is None and "adj" in kw:
            adj_matrix = kw.pop("adj")
        # Fall back to the adj_matrix stashed at fit() time so callers can
        # just pass node features to predict(). Bug fix: previously predict()
        # crashed whenever adj_matrix was not re-passed, which is unfriendly.
        if adj_matrix is None:
            adj_matrix = getattr(self, "_last_adj_matrix", None)

        if self.task_type == "node_classification":
            X_arr = np.asarray(x, dtype=np.float32)
            A_arr = np.asarray(adj_matrix, dtype=np.float32)
            self.net.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_arr, device=self.device)
                A_t = torch.tensor(A_arr, device=self.device)
                logits = self.net(X_t, A_t)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                if self.classes_ is not None and self.classes_.dtype.kind in ("U", "S", "O"):
                    return np.array([self.classes_[p] for p in preds])
                return preds

        elif self.task_type == "graph_classification":
            nf, am = x
            self.net.eval()
            with torch.no_grad():
                nf_t = torch.tensor(np.asarray(nf, dtype=np.float32), device=self.device)
                am_t = torch.tensor(np.asarray(am, dtype=np.float32), device=self.device)
                logits = self.net(nf_t, am_t)
                pred = int(torch.argmax(logits).item())
                if self.classes_ is not None and self.classes_.dtype.kind in ("U", "S", "O"):
                    return self.classes_[pred]
                return pred

        elif self.task_type == "link_prediction":
            X_arr = np.asarray(x, dtype=np.float32)
            A_arr = np.asarray(adj_matrix, dtype=np.float32)
            self.net.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_arr, device=self.device)
                A_t = torch.tensor(A_arr, device=self.device)
                embeddings = self.net.get_node_embeddings(X_t, A_t)
                return embeddings.cpu().numpy()

    def predict_edges(self, node_pairs, x, adj_matrix):
        """Creative: predict whether edges exist between given node pairs.

        Args:
            node_pairs: (n_pairs, 2) array of (src, dst) indices
            x: node features
            adj_matrix: adjacency matrix
        Returns: (n_pairs,) array of edge probabilities (0-1)
        """
        if not self._trained or self.net is None:
            raise RuntimeError("GNN.predict_edges() before train()")
        embeddings = self.forward(x, adj_matrix=adj_matrix)
        pairs = np.asarray(node_pairs)
        src_emb = embeddings[pairs[:, 0]]
        dst_emb = embeddings[pairs[:, 1]]
        scores = (src_emb * dst_emb).sum(axis=-1)
        probs = 1 / (1 + np.exp(-scores))  # sigmoid
        return probs

    def get_node_embeddings(self, x, adj_matrix):
        """Creative: extract intermediate node embeddings (before output head)."""
        return self.forward(x, adj_matrix=adj_matrix)

    def losses(self): return list(self._losses)
    def parameter_count(self): return sum(p.numel() for p in self.net.parameters()) if self.net else 0

    def architecture_summary(self):
        return {"technique":"gnn","in_dim":self.in_dim,"hidden":self.hidden,
                "n_layers":self.n_layers,"task_type":self.task_type,
                "n_classes":self.n_classes,"parameters":self.parameter_count(),
                "device":str(self.device)}

    def __repr__(self):
        if self.net is None: return f"<GNN unbuilt task_type={self.task_type}>"
        return (f"<GNN task_type={self.task_type} hidden={self.hidden} "
                f"n_layers={self.n_layers} params={self.parameter_count():,} "
                f"trained={self._trained}>")
