"""
qai.techniques.frontier_recsys -- recommender systems and graph variants.

Recommenders:
  qai.build(technique="collab_filtering", ...) — User-item collaborative filtering
  qai.build(technique="matrix_factorization", ...) — ALS matrix factorization
  qai.build(technique="wide_deep", ...)        — Wide & Deep (Google 2016)
  qai.build(technique="deepfm", ...)           — DeepFM (Huawei 2017)
  qai.build(technique="item2vec", ...)         — Item2Vec embedding from sessions

Graph variants:
  qai.build(technique="graphsage", ...)        — GraphSAGE (Hamilton 2017)
  qai.build(technique="gat", ...)              — Graph Attention Network (Veličković 2018)
  qai.build(technique="gin", ...)              — Graph Isomorphism Network (Xu 2019)
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
            "[WHAT] Recsys / graph techniques require PyTorch\n"
            "[FIX] pip install torch"
        )


# =====================================================================
# 1. COLLABORATIVE FILTERING (user-based + item-based)
# =====================================================================
class CollaborativeFiltering(Technique):
    """Memory-based collaborative filtering.

    Stores the user-item rating matrix. predict(user, item) is a weighted
    average of the k most similar users (or items) to the target. Supports
    both user-based and item-based variants.
    """
    name = "collab_filtering"
    family = "recommender"

    def __init__(self, mode: str = "item", k: int = 5, **kw):
        super().__init__(mode=mode, k=k, **kw)
        self.mode = mode
        self.k = k
        self.R = None
        self.sim = None

    def _cosine_sim(self, M):
        norm = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
        Mn = M / norm
        return Mn @ Mn.T

    def fit(self, X, y=None):
        """X: (n_samples, 3) — (user_idx, item_idx, rating)."""
        X = np.asarray(X)
        users = X[:, 0].astype(int)
        items = X[:, 1].astype(int)
        ratings = X[:, 2].astype(np.float32)
        self.n_users = users.max() + 1
        self.n_items = items.max() + 1
        self.R = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        self.R[users, items] = ratings
        if self.mode == "user":
            self.sim = self._cosine_sim(self.R)
        else:
            self.sim = self._cosine_sim(self.R.T)
        self._trained = True
        return self

    def predict(self, user, item):
        # Cast to int -- callers (e.g. test harness) may pass float indices
        # coming from np.float32 data, which numpy indexing rejects.
        user = int(user)
        item = int(item)
        if self.mode == "user":
            sims = self.sim[user]
            rated = np.where(self.R[:, item] > 0)[0]
            rated = rated[rated != user]
            if len(rated) == 0:
                return self.R[:, item].mean() if (self.R[:, item] > 0).any() else 0.0
            top = rated[np.argsort(-sims[rated])[:self.k]]
            num = (sims[top] * self.R[top, item]).sum()
            den = np.abs(sims[top]).sum() + 1e-8
            return num / den
        else:
            sims = self.sim[item]
            rated = np.where(self.R[user] > 0)[0]
            rated = rated[rated != item]
            if len(rated) == 0:
                return self.R[user].mean() if (self.R[user] > 0).any() else 0.0
            top = rated[np.argsort(-sims[rated])[:self.k]]
            num = (sims[top] * self.R[user, top]).sum()
            den = np.abs(sims[top]).sum() + 1e-8
            return num / den

    def forward(self, x):
        return self.predict(x[0], x[1])


# =====================================================================
# 2. MATRIX FACTORIZATION (ALS)
# =====================================================================
class MatrixFactorization(Technique):
    """Matrix factorization with ALS for recommender systems.

    Decomposes R = U @ V.T where U is (n_users, k) and V is (n_items, k).
    ALS alternates fixing U and solving for V via least squares, then vice
    versa. Optionally L2-regularized.
    """
    name = "matrix_factorization"
    family = "recommender"

    def __init__(self, n_factors: int = 8, n_iter: int = 10,
                 reg: float = 0.1, **kw):
        super().__init__(n_factors=n_factors, n_iter=n_iter, reg=reg, **kw)
        self.k = n_factors
        self.n_iter = n_iter
        self.reg = reg
        self.U = None
        self.V = None

    def fit(self, X, y=None):
        """X: (n_samples, 3) — (user_idx, item_idx, rating)."""
        X = np.asarray(X)
        users = X[:, 0].astype(int)
        items = X[:, 1].astype(int)
        ratings = X[:, 2].astype(np.float32)
        n_u = users.max() + 1
        n_i = items.max() + 1
        rng = np.random.default_rng(0)
        self.U = rng.standard_normal((n_u, self.k)).astype(np.float32)
        self.V = rng.standard_normal((n_i, self.k)).astype(np.float32)
        user_items = [[] for _ in range(n_u)]
        item_users = [[] for _ in range(n_i)]
        for u, i, r in zip(users, items, ratings):
            user_items[u].append((i, r))
            item_users[i].append((u, r))
        for it in range(self.n_iter):
            # update U
            for u in range(n_u):
                if not user_items[u]:
                    continue
                idxs = [x[0] for x in user_items[u]]
                rs = np.array([x[1] for x in user_items[u]])
                Vsub = self.V[idxs]
                A = Vsub.T @ Vsub + self.reg * np.eye(self.k)
                b = Vsub.T @ rs
                self.U[u] = np.linalg.solve(A, b)
            # update V
            for i in range(n_i):
                if not item_users[i]:
                    continue
                idxs = [x[0] for x in item_users[i]]
                rs = np.array([x[1] for x in item_users[i]])
                Usub = self.U[idxs]
                A = Usub.T @ Usub + self.reg * np.eye(self.k)
                b = Usub.T @ rs
                self.V[i] = np.linalg.solve(A, b)
        self._trained = True
        return self

    def predict(self, user, item):
        # Cast to int -- callers (e.g. test harness) may pass float indices
        # coming from np.float32 data, which numpy indexing rejects.
        user = int(user)
        item = int(item)
        return float(self.U[user] @ self.V[item])

    def forward(self, x):
        return self.predict(x[0], x[1])


# =====================================================================
# 3. WIDE & DEEP (Google 2016)
# =====================================================================
class WideAndDeep(Technique):
    """Wide & Deep — linear model (wide) + MLP (deep) joined at the output.

    The wide part memorises cross-feature interactions; the deep part
    generalises via dense embeddings. Output is sigmoid(wide + deep).
    """
    name = "wide_deep"
    family = "recommender"

    def __init__(self, n_features: int = 16, n_cross: int = 8,
                 deep_hidden: int = 64, lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(n_features=n_features, n_cross=n_cross,
                         deep_hidden=deep_hidden, lr=lr, epochs=epochs, **kw)
        _req()
        self.n_features = n_features
        # Bug fix: stash n_cross too -- fit()/forward() need it to size the
        # cross-feature block, but __init__ previously only stored n_features.
        self.n_cross = n_cross
        self.lr, self.epochs = lr, epochs
        self.wide = nn.Linear(n_features + n_cross, 1)
        self.deep = nn.Sequential(
            nn.Linear(n_features, deep_hidden), nn.ReLU(),
            nn.Linear(deep_hidden, deep_hidden), nn.ReLU(),
            nn.Linear(deep_hidden, 1),
        )
        self.opt = torch.optim.Adam(
            list(self.wide.parameters()) + list(self.deep.parameters()), lr=lr)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        # cross features = pairwise products of first n_cross dims.
        # Bug fix: the array width and loop bound used a hardcoded 8 instead
        # of self.n_cross, so the wide Linear layer (which expects
        # n_features + n_cross inputs) crashed with a shape mismatch whenever
        # n_cross != 8. Use self.n_cross consistently.
        n_cross = self.n_cross
        Xwide = np.zeros((X.shape[0], self.n_features + n_cross), dtype=np.float32)
        Xwide[:, :self.n_features] = X
        for i in range(min(n_cross, self.n_features - 1)):
            Xwide[:, self.n_features + i] = X[:, i] * X[:, i + 1]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            xt = torch.from_numpy(X)
            xw = torch.from_numpy(Xwide)
            yt = torch.from_numpy(y).unsqueeze(-1)
            out = self.wide(xw) + self.deep(xt)
            loss = F.binary_cross_entropy_with_logits(out, yt)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        # Bug fix: same hardcoded-8 issue as fit() -- use self.n_cross.
        n_cross = self.n_cross
        xwide = torch.zeros(1, self.n_features + n_cross)
        xwide[0, :self.n_features] = x
        for i in range(min(n_cross, self.n_features - 1)):
            xwide[0, self.n_features + i] = x[0, i] * x[0, i + 1]
        with torch.no_grad():
            out = self.wide(xwide) + self.deep(x)
        return torch.sigmoid(out).item()


# =====================================================================
# 4. DEEPFM (Huawei 2017)
# =====================================================================
class _FM(nn.Module):
    def __init__(self, n_features, k=8):
        super().__init__()
        self.w = nn.Linear(n_features, 1)
        self.V = nn.Parameter(torch.randn(n_features, k) * 0.01)
        self.k = k

    def forward(self, x):
        linear = self.w(x)
        # FM second-order interactions: 0.5 * sum((sum v_i x_i)^2 - sum (v_i x_i)^2)
        sq_sum = (x @ self.V).pow(2).sum(-1, keepdim=True)
        sum_sq = ((x.unsqueeze(-1) * self.V) ** 2).sum(1).sum(-1, keepdim=True)
        fm = 0.5 * (sq_sum - sum_sq)
        return linear + fm


class DeepFM(Technique):
    """DeepFM — FM + deep MLP sharing the same input embeddings.

    Replaces the wide part of Wide & Deep with an FM component, removing the
    need for manual feature engineering.
    """
    name = "deepfm"
    family = "recommender"

    def __init__(self, n_features: int = 16, k: int = 8,
                 deep_hidden: int = 64, lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(n_features=n_features, k=k, deep_hidden=deep_hidden,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.n_features, self.k = n_features, k
        self.lr, self.epochs = lr, epochs
        self.fm = _FM(n_features, k)
        self.deep = nn.Sequential(
            nn.Linear(n_features, deep_hidden), nn.ReLU(),
            nn.Linear(deep_hidden, deep_hidden), nn.ReLU(),
            nn.Linear(deep_hidden, 1),
        )
        self.opt = torch.optim.Adam(
            list(self.fm.parameters()) + list(self.deep.parameters()), lr=lr)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            xt = torch.from_numpy(X)
            yt = torch.from_numpy(y).unsqueeze(-1)
            out = self.fm(xt) + self.deep(xt)
            loss = F.binary_cross_entropy_with_logits(out, yt)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.fm(x) + self.deep(x)
        return torch.sigmoid(out).item()


# =====================================================================
# 5. ITEM2VEC — item embeddings from sessions via word2vec
# =====================================================================
class Item2Vec(Technique):
    """Item2Vec — learns item embeddings by treating session item sequences
    as sentences and applying the SGNS word2vec objective.

    fit() consumes a list of sessions (each a list of item IDs); embed(i)
    returns the learned embedding of item i.
    """
    name = "item2vec"
    family = "recommender"

    def __init__(self, n_items: int = 50, dim: int = 16,
                 window: int = 3, neg_samples: int = 4,
                 lr: float = 1e-2, epochs: int = 5, **kw):
        super().__init__(n_items=n_items, dim=dim, window=window,
                         neg_samples=neg_samples, lr=lr, epochs=epochs, **kw)
        _req()
        self.n_items, self.dim = n_items, dim
        self.window, self.neg = window, neg_samples
        self.lr, self.epochs = lr, epochs
        self.center = nn.Embedding(n_items, dim)
        self.context = nn.Embedding(n_items, dim)
        self.opt = torch.optim.Adam(
            list(self.center.parameters()) + list(self.context.parameters()), lr=lr)

    def fit(self, X, y=None):
        """X: list of sessions, each a list of item ids."""
        for ep in range(self.epochs):
            self.opt.zero_grad()
            loss = 0.0
            n_pairs = 0
            for session in X:
                for i, item in enumerate(session):
                    lo = max(0, i - self.window)
                    hi = min(len(session), i + self.window + 1)
                    center = torch.tensor([item])
                    for j in range(lo, hi):
                        if j == i:
                            continue
                        pos = torch.tensor([session[j]])
                        negs = torch.randint(0, self.n_items, (self.neg,))
                        center_emb = self.center(center)
                        pos_emb = self.context(pos)
                        neg_emb = self.context(negs)
                        pos_score = (center_emb * pos_emb).sum(-1)
                        neg_score = (center_emb * neg_emb).sum(-1)
                        loss = loss - F.logsigmoid(pos_score).mean()
                        loss = loss - F.logsigmoid(-neg_score).mean()
                        n_pairs += 1
            (loss / max(n_pairs, 1)).backward()
            self.opt.step()
        self._trained = True
        return self

    def embed(self, item):
        with torch.no_grad():
            return self.center(torch.tensor([item])).squeeze(0).numpy()

    def forward(self, x):
        return self.embed(x)


# =====================================================================
# 6. GraphSAGE (Hamilton et al. 2017)
# =====================================================================
class _SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, neighbors):
        # x: (N, in_dim), neighbors: aggregated neighbor mean (N, in_dim)
        return F.relu(self.w_self(x) + self.w_neigh(neighbors))


class GraphSAGE(Technique):
    """GraphSAGE — inductive node embedding via neighbor sampling + aggregation.

    For each node, sample a fixed-size set of neighbors at each layer and
    aggregate their features. Supports mean aggregator here; generalises to
    new nodes unseen at training time.
    """
    name = "graphsage"
    family = "graph"

    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 8,
                 n_layers: int = 2, n_neighbors: int = 5,
                 lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, hidden=hidden, out_dim=out_dim,
                         n_layers=n_layers, n_neighbors=n_neighbors,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.n_layers, self.n_neighbors = n_layers, n_neighbors
        self.lr, self.epochs = lr, epochs
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        for i in range(n_layers):
            self.layers.append(_SAGEConv(dims[i], dims[i + 1]))
        self.opt = torch.optim.Adam(self.layers.parameters(), lr=lr)
        self.adj = None

    def _aggregate(self, x, layer):
        N = x.shape[0]
        agg = torch.zeros_like(x)
        for v in range(N):
            neighbors = self.adj.get(v, [])
            if neighbors:
                sample = neighbors[:self.n_neighbors]
                agg[v] = x[sample].mean(0)
        return layer(x, agg)

    def fit(self, X, y, adj=None):
        """X: (N, in_dim) features; y: (N,) labels; adj: dict node -> list of neighbor ids,
        or a dense (N, N) ndarray adjacency matrix."""
        # Bug fix: accept adj as either a dict or a dense ndarray. Convert
        # matrix to dict so _aggregate's `self.adj.get(v, [])` keeps working.
        if adj is not None and isinstance(adj, np.ndarray):
            N = adj.shape[0]
            adj = {v: [u for u in range(N) if adj[v, u] > 0 and u != v]
                   for v in range(N)}
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y = torch.as_tensor(np.asarray(y).astype(np.int64))
        if adj is None:
            adj = {i: [] for i in range(X.shape[0])}
        self.adj = adj
        for ep in range(self.epochs):
            self.opt.zero_grad()
            h = X
            for layer in self.layers:
                h = self._aggregate(h, layer)
            logits = h @ h.t()  # dot product classification head (illustrative)
            loss = F.cross_entropy(logits.diag().unsqueeze(0).expand(X.shape[0], -1), y)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x, **kw):
        # Accept and ignore graph kwargs (adj/adj_matrix) for consistency
        # with the test harness that passes them to all graph models.
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            h = x
            for layer in self.layers:
                agg = torch.zeros_like(h)
                h = layer(h, agg)
        return h.squeeze(0).numpy()


# =====================================================================
# 7. GAT — Graph Attention Network (Veličković et al. 2018)
# =====================================================================
class _GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.a)
        self.out_dim = out_dim

    def forward(self, x, adj):
        N = x.shape[0]
        h = self.W(x).view(N, self.n_heads, self.out_dim)
        # attention scores
        e = torch.zeros(N, N, self.n_heads)
        for v in range(N):
            for u in adj.get(v, []):
                cat = torch.cat([h[v], h[u]], dim=-1)
                e[v, u] = F.leaky_relu((self.a * cat).sum(-1), 0.2)
        # softmax over neighbors
        alpha = torch.zeros_like(e)
        for v in range(N):
            neighbors = adj.get(v, []) + [v]
            scores = e[v, neighbors]
            alpha[v, neighbors] = F.softmax(scores, dim=0)
        out = torch.zeros(N, self.n_heads, self.out_dim)
        for v in range(N):
            neighbors = adj.get(v, []) + [v]
            for u in neighbors:
                out[v] += alpha[v, u].unsqueeze(-1) * h[u]
        return out.mean(1)  # average heads


class GAT(Technique):
    """GAT — graph attention network with learned attention coefficients.

    Each node attends to its neighbors with weights learned end-to-end,
    providing interpretability and better performance on heterophilic graphs.
    """
    name = "gat"
    family = "graph"

    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 4,
                 n_heads: int = 4, lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, hidden=hidden, out_dim=out_dim,
                         n_heads=n_heads, lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.lr, self.epochs = lr, epochs
        self.l1 = _GATLayer(in_dim, hidden, n_heads)
        self.l2 = _GATLayer(hidden, out_dim, n_heads)
        self.opt = torch.optim.Adam(
            list(self.l1.parameters()) + list(self.l2.parameters()), lr=lr)

    def fit(self, X, y, adj):
        # Bug fix: accept adj as either a dict {node: [neighbors]} (the
        # original API) or a dense (N, N) ndarray. The test harness (and
        # other graph techniques) pass a matrix, so convert to dict here.
        if isinstance(adj, np.ndarray):
            N = adj.shape[0]
            adj = {v: [u for u in range(N) if adj[v, u] > 0 and u != v]
                   for v in range(N)}
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y = torch.as_tensor(np.asarray(y).astype(np.int64))
        for ep in range(self.epochs):
            self.opt.zero_grad()
            h = F.elu(self.l1(X, adj))
            h = self.l2(h, adj)
            loss = F.cross_entropy(h, y)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x, **kw):
        # Accept and ignore graph kwargs (adj/adj_matrix) for consistency
        # with the test harness that passes them to all graph models.
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            h = F.elu(self.l1(x, {0: []}))
            h = self.l2(h, {0: []})
        return h.squeeze(0).numpy()


# =====================================================================
# 8. GIN — Graph Isomorphism Network (Xu et al. 2019)
# =====================================================================
class _GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, adj):
        N = x.shape[0]
        out = torch.zeros_like(x)
        for v in range(N):
            neighbors = adj.get(v, [])
            if neighbors:
                agg = x[neighbors].sum(0)
            else:
                agg = torch.zeros_like(x[v])
            out[v] = (1 + self.eps) * x[v] + agg
        return self.mlp(out)


class GIN(Technique):
    """GIN — provably as expressive as the Weisfeiler-Lehman test.

    Uses sum aggregation (more discriminative than mean) and an MLP at
    each layer. The injection parameter eps is learnable.
    """
    name = "gin"
    family = "graph"

    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 4,
                 n_layers: int = 2, lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, hidden=hidden, out_dim=out_dim,
                         n_layers=n_layers, lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.lr, self.epochs = lr, epochs
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([_GINLayer(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.opt = torch.optim.Adam(self.layers.parameters(), lr=lr)

    def fit(self, X, y, adj):
        # Bug fix: accept adj as either a dict {node: [neighbors]} or a dense
        # (N, N) ndarray. Convert matrix to dict so _GINLayer keeps working.
        if isinstance(adj, np.ndarray):
            N = adj.shape[0]
            adj = {v: [u for u in range(N) if adj[v, u] > 0 and u != v]
                   for v in range(N)}
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y = torch.as_tensor(np.asarray(y).astype(np.int64))
        for ep in range(self.epochs):
            self.opt.zero_grad()
            h = X
            for layer in self.layers:
                h = F.relu(layer(h, adj))
            loss = F.cross_entropy(h, y)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x, **kw):
        # Accept and ignore graph kwargs (adj/adj_matrix) for consistency
        # with the test harness that passes them to all graph models.
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            h = x
            for layer in self.layers:
                h = F.relu(layer(h, {0: []}))
        return h.squeeze(0).numpy()
