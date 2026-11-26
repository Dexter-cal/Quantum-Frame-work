"""
qai.techniques.frontier_meta -- meta-learning and metric-learning architectures
for few-shot classification.

  qai.build(technique="siamese", ...)     — Siamese network + contrastive loss
  qai.build(technique="prototypical", ...) — Prototypical Networks (Snell 2017)
  qai.build(technique="matching", ...)     — Matching Networks (Vinyals 2016)
  qai.build(technique="maml", ...)         — Model-Agnostic Meta-Learning (Finn 2017)
  qai.build(technique="relation", ...)     — Relation Networks (Sung 2018)

These are the canonical few-shot learning methods. Each supports an episodic
fit() interface and an inference-time few-shot predict(support_x, support_y, query_x).
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

from .base import Technique

def _req():
    if not _TORCH:
        raise ImportError(
            "[WHAT] Meta-learning techniques require PyTorch\n"
            "[FIX] pip install torch"
        )


class _EmbeddingNet(nn.Module):
    def __init__(self, in_dim, emb_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================
# 1. SIAMESE NETWORK (Koch et al. 2015)
# =====================================================================
class SiameseNetwork(Technique):
    """Siamese network with contrastive loss for verification.

    Two twin networks share weights; the distance between embeddings of two
    inputs drives the contrastive loss L = (1-y) * d^2 + y * max(0, m-d)^2.
    fit() consumes pairs (X1, X2, y_same_class); predict(x1, x2) returns
    the similarity score.
    """
    name = "siamese"
    family = "metric_learning"

    def __init__(self, in_dim: int = 16, emb_dim: int = 64,
                 margin: float = 1.0, lr: float = 1e-3,
                 epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, emb_dim=emb_dim, margin=margin,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.margin = margin
        self.lr, self.epochs = lr, epochs
        self.embed = _EmbeddingNet(in_dim, emb_dim)
        self.opt = torch.optim.Adam(self.embed.parameters(), lr=lr)

    def _contrastive(self, x1, x2, y):
        e1 = self.embed(x1)
        e2 = self.embed(x2)
        d = F.pairwise_distance(e1, e2)
        loss = ((1 - y) * d.pow(2) + y * F.relu(self.margin - d).pow(2)).mean()
        return loss

    def fit(self, X, y):
        """X: (N, 2, in_dim), y: (N,) in {0,1}."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            x1 = torch.from_numpy(X[:, 0, :])
            x2 = torch.from_numpy(X[:, 1, :])
            yt = torch.from_numpy(y)
            loss = self._contrastive(x1, x2, yt)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, pair):
        """pair: (2, in_dim)."""
        pair = np.asarray(pair, dtype=np.float32)
        x1 = torch.from_numpy(pair[0:1])
        x2 = torch.from_numpy(pair[1:2])
        with torch.no_grad():
            e1 = self.embed(x1)
            e2 = self.embed(x2)
            d = F.pairwise_distance(e1, e2)
        return float(d.item())


# =====================================================================
# 2. PROTOTYPICAL NETWORKS (Snell et al. 2017)
# =====================================================================
class PrototypicalNetwork(Technique):
    """Prototypical Networks — class prototypes computed as the mean embedding
    of support examples. Query points are classified to the nearest prototype.
    """
    name = "prototypical"
    family = "few_shot"

    def __init__(self, in_dim: int = 16, emb_dim: int = 64,
                 n_way: int = 5, n_support: int = 5, n_query: int = 5,
                 lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, emb_dim=emb_dim, n_way=n_way,
                         n_support=n_support, n_query=n_query,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.n_way, self.n_support, self.n_query = n_way, n_support, n_query
        self.lr, self.epochs = lr, epochs
        self.embed = _EmbeddingNet(in_dim, emb_dim)
        self.opt = torch.optim.Adam(self.embed.parameters(), lr=lr)

    def fit(self, X, y):
        """X: (N, in_dim), y: (N,) integer labels."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes = np.unique(y)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            loss = 0.0
            for c in classes:
                idx_c = np.where(y == c)[0]
                if len(idx_c) < self.n_support + 1:
                    continue
                sup = idx_c[:self.n_support]
                qry = idx_c[self.n_support:self.n_support + self.n_query]
                if len(qry) == 0:
                    qry = idx_c[-1:]
                proto = self.embed(torch.from_numpy(X[sup])).mean(0)
                emb_q = self.embed(torch.from_numpy(X[qry]))
                d = torch.cdist(emb_q, proto.unsqueeze(0)).squeeze(-1)
                loss = loss + d.mean()
            (-loss).backward()
            self.opt.step()
        self._trained = True
        return self

    def few_shot_predict(self, support_x, support_y, query_x):
        support_x = torch.from_numpy(np.asarray(support_x, dtype=np.float32))
        query_x = torch.from_numpy(np.asarray(query_x, dtype=np.float32))
        support_y = np.asarray(support_y)
        classes = np.unique(support_y)
        with torch.no_grad():
            prototypes = []
            for c in classes:
                idx = np.where(support_y == c)[0]
                p = self.embed(support_x[idx]).mean(0)
                prototypes.append(p)
            prototypes = torch.stack(prototypes)
            emb_q = self.embed(query_x)
            d = torch.cdist(emb_q, prototypes)
            preds = d.argmin(-1).numpy()
            return classes[preds]

    def forward(self, x):
        return self.embed(torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)).squeeze(0).detach().numpy()


# =====================================================================
# 3. MATCHING NETWORKS (Vinyals et al. 2016)
# =====================================================================
class MatchingNetwork(Technique):
    """Matching Networks — attention over support embeddings.

    Query prediction = weighted sum of support labels, weights = softmax of
    cosine similarity between query and support embeddings.
    """
    name = "matching"
    family = "few_shot"

    def __init__(self, in_dim: int = 16, emb_dim: int = 64,
                 n_way: int = 5, n_support: int = 5,
                 lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, emb_dim=emb_dim, n_way=n_way,
                         n_support=n_support, lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.n_way, self.n_support = n_way, n_support
        self.lr, self.epochs = lr, epochs
        self.embed = _EmbeddingNet(in_dim, emb_dim)
        self.opt = torch.optim.Adam(self.embed.parameters(), lr=lr)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes = np.unique(y)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            loss = 0.0
            for c in classes:
                idx_c = np.where(y == c)[0]
                if len(idx_c) < self.n_support + 1:
                    continue
                sup = idx_c[:self.n_support]
                qry = idx_c[self.n_support]
                e_s = F.normalize(self.embed(torch.from_numpy(X[sup])), dim=-1)
                e_q = F.normalize(self.embed(torch.from_numpy(X[qry:qry+1])), dim=-1)
                sim = (e_q @ e_s.t()).squeeze(0)
                log_attn = F.log_softmax(sim, dim=-1)
                target = torch.tensor(list(range(len(sup)))).argmax()
                loss = loss - log_attn[target]
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def few_shot_predict(self, support_x, support_y, query_x):
        support_x = torch.from_numpy(np.asarray(support_x, dtype=np.float32))
        query_x = torch.from_numpy(np.asarray(query_x, dtype=np.float32))
        support_y = np.asarray(support_y)
        with torch.no_grad():
            e_s = F.normalize(self.embed(support_x), dim=-1)
            e_q = F.normalize(self.embed(query_x), dim=-1)
            sim = e_q @ e_s.t()
            attn = F.softmax(sim, dim=-1)
            # one-hot support labels
            classes = np.unique(support_y)
            label_map = {c: i for i, c in enumerate(classes)}
            onehot = np.zeros((len(support_y), len(classes)))
            for i, c in enumerate(support_y):
                onehot[i, label_map[c]] = 1
            onehot = torch.from_numpy(onehot).float()
            preds = (attn @ onehot).argmax(-1).numpy()
            return classes[preds]

    def forward(self, x):
        return self.embed(torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)).squeeze(0).detach().numpy()


# =====================================================================
# 4. MAML — Model-Agnostic Meta-Learning (Finn et al. 2017)
# =====================================================================
class MAML(Technique):
    """MAML — learns an init that adapts in a few gradient steps.

    Inner loop: take K gradient steps on the task's support set.
    Outer loop: optimise the init against the query loss after adaptation.

    This implementation uses a simple linear head so the second-order
    gradient through the inner step is tractable on CPU.
    """
    name = "maml"
    family = "meta_learning"

    def __init__(self, in_dim: int = 16, n_classes: int = 5,
                 inner_lr: float = 0.01, outer_lr: float = 1e-3,
                 inner_steps: int = 1, epochs: int = 50, **kw):
        super().__init__(in_dim=in_dim, n_classes=n_classes,
                         inner_lr=inner_lr, outer_lr=outer_lr,
                         inner_steps=inner_steps, epochs=epochs, **kw)
        _req()
        self.in_dim, self.n_classes = in_dim, n_classes
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.epochs = epochs
        self.W = torch.zeros(n_classes, in_dim, requires_grad=True)
        self.b = torch.zeros(n_classes, requires_grad=True)
        self.opt = torch.optim.Adam([self.W, self.b], lr=outer_lr)

    def _loss(self, W, b, X, y):
        logits = X @ W.t() + b
        return F.cross_entropy(logits, y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            classes = np.unique(y)
            total_loss = 0.0
            for c in classes:
                idx = np.where(y == c)[0]
                if len(idx) < 6:
                    continue
                sup = idx[:5]
                qry = idx[5:]
                Xs = torch.from_numpy(X[sup])
                ys = torch.full((5,), c, dtype=torch.long)
                Xq = torch.from_numpy(X[qry])
                yq = torch.full((len(qry),), c, dtype=torch.long)
                # inner loop
                W_fast = self.W.clone()
                b_fast = self.b.clone()
                for _ in range(self.inner_steps):
                    inner_loss = self._loss(W_fast, b_fast, Xs, ys)
                    gW, gb = torch.autograd.grad(inner_loss, [W_fast, b_fast], create_graph=True)
                    W_fast = W_fast - self.inner_lr * gW
                    b_fast = b_fast - self.inner_lr * gb
                total_loss = total_loss + self._loss(W_fast, b_fast, Xq, yq)
            total_loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = x @ self.W.t() + self.b
        return logits.squeeze(0).argmax().item()


# =====================================================================
# 5. RELATION NETWORK (Sung et al. 2018)
# =====================================================================
class _RelationNet(nn.Module):
    def __init__(self, emb_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, e_s, e_q):
        # e_s: (S, D), e_q: (Q, D) -> (Q, S)
        S = e_s.shape[0]
        Q = e_q.shape[0]
        e_s_exp = e_s.unsqueeze(0).expand(Q, -1, -1)
        e_q_exp = e_q.unsqueeze(1).expand(-1, S, -1)
        return self.net(torch.cat([e_s_exp, e_q_exp], dim=-1)).squeeze(-1)


class RelationNetwork(Technique):
    """Relation Network — learns a metric via a small relation module.

    The relation module g(e_s, e_q) outputs a similarity score in [0,1].
    Trained with BCE against the same-class indicator.
    """
    name = "relation"
    family = "few_shot"

    def __init__(self, in_dim: int = 16, emb_dim: int = 64,
                 lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, emb_dim=emb_dim, lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.lr, self.epochs = lr, epochs
        self.embed = _EmbeddingNet(in_dim, emb_dim)
        self.relation = _RelationNet(emb_dim)
        self.opt = torch.optim.Adam(
            list(self.embed.parameters()) + list(self.relation.parameters()), lr=lr)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes = np.unique(y)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            loss = 0.0
            for c in classes:
                idx = np.where(y == c)[0]
                if len(idx) < 6:
                    continue
                sup = idx[:5]
                qry = idx[5:]
                e_s = self.embed(torch.from_numpy(X[sup]))
                e_q = self.embed(torch.from_numpy(X[qry]))
                sim = self.relation(e_s, e_q)
                target = torch.zeros_like(sim)
                target[:, 0] = 1.0  # support[0] is same class as query
                loss = loss + F.mse_loss(sim, target)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def few_shot_predict(self, support_x, support_y, query_x):
        support_x = torch.from_numpy(np.asarray(support_x, dtype=np.float32))
        query_x = torch.from_numpy(np.asarray(query_x, dtype=np.float32))
        support_y = np.asarray(support_y)
        classes = np.unique(support_y)
        with torch.no_grad():
            e_s = self.embed(support_x)
            e_q = self.embed(query_x)
            sim = self.relation(e_s, e_q)
            preds = sim.argmax(-1).numpy()
            return classes[preds]

    def forward(self, x):
        return self.embed(torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)).squeeze(0).detach().numpy()
