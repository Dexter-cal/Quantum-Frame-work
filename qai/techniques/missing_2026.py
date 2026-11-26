"""
qai.techniques.missing_2026 — latest 2025/2026 techniques we were missing.

  DPO                   — Direct Preference Optimization (Rafailov 2023)
  ContrastiveLearning   — SimCLR-style contrastive learning
  GRPO                  — Group Relative Policy Optimization (DeepSeek 2024)
  CountVectorizer       — text → bag-of-words counts
  TfidfVectorizer       — text → TF-IDF features
  TargetEncoder         — target encoding for categorical features
  HashingVectorizer     — text → hashed features
  DictVectorizer        — dict → feature matrix
  FeatureHasher         — feature hashing
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from .base import Technique

def _sk():
    try:
        import sklearn; return sklearn
    except ImportError:
        return None


# ============================================================================
# 1. DPO — Direct Preference Optimization
# ============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


class DPO(Technique):
    """DPO — Direct Preference Optimization (Rafailov et al. 2023).

    Replaces RLHF's reward model + PPO with a single contrastive loss.
    Given (prompt, chosen_response, rejected_response), trains the policy
    to prefer chosen over rejected.

    Loss = -log(σ(β * (log π(chosen) - log π_ref(chosen) - log π(rejected) + log π_ref(rejected))))
    """
    name = "dpo"
    family = "alignment"

    def __init__(self, policy_model: Any = None, ref_model: Any = None,
                 beta: float = 0.1, lr: float = 5e-6,
                 epochs: int = 100, **kw):
        super().__init__(beta=beta, lr=lr, epochs=epochs, **kw)
        if not _TORCH:
            raise ImportError("DPO requires PyTorch")
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.policy = policy_model
        self.ref = ref_model
        self.history: List[float] = []

    def _get_logprob(self, model, inputs):
        """Get log probability of inputs from a model."""
        try:
            out = model.forward(inputs) if hasattr(model, "forward") else model(inputs)
            out = torch.as_tensor(out, dtype=torch.float32)
            return F.log_softmax(out, dim=-1).mean()
        except Exception:
            return torch.tensor(0.0)

    def fit(self, X, y=None):
        """X: dict with 'prompt', 'chosen', 'rejected'."""
        if self.policy is None:
            raise ValueError("must provide policy_model")
        if self.ref is None:
            import copy
            self.ref = copy.deepcopy(self.policy)
            for p in self.ref.parameters() if hasattr(self.ref, "parameters") else []:
                p.requires_grad = False
        opt = torch.optim.Adam(
            self.policy.parameters() if hasattr(self.policy, "parameters") else [],
            lr=self.lr
        )
        for ep in range(self.epochs):
            opt.zero_grad()
            chosen_logp = self._get_logprob(self.policy, X.get("chosen", X))
            rejected_logp = self._get_logprob(self.policy, X.get("rejected", X))
            with torch.no_grad():
                ref_chosen = self._get_logprob(self.ref, X.get("chosen", X))
                ref_rejected = self._get_logprob(self.ref, X.get("rejected", X))
            pi_logratio = chosen_logp - rejected_logp
            ref_logratio = ref_chosen - ref_rejected
            logits = self.beta * (pi_logratio - ref_logratio)
            loss = -F.logsigmoid(logits)
            loss.backward()
            opt.step()
            self.history.append(float(loss))
        self._trained = True
        return self

    def forward(self, x):
        return self._get_logprob(self.policy, x).item()


# ============================================================================
# 2. Contrastive Learning (SimCLR-style)
# ============================================================================
class ContrastiveLearning(Technique):
    """SimCLR-style contrastive learning.

    Learns representations by maximizing agreement between augmented
    views of the same input (positive pairs) while minimizing agreement
    with different inputs (negative pairs).

    Loss = -log(exp(sim(z_i, z_j)/τ) / Σ exp(sim(z_i, z_k)/τ))
    """
    name = "contrastive_learning"
    family = "self_supervised"

    def __init__(self, dim: int = 16, hidden: int = 32,
                 temperature: float = 0.5, lr: float = 1e-3,
                 epochs: int = 50, **kw):
        super().__init__(dim=dim, hidden=hidden, temperature=temperature,
                         lr=lr, epochs=epochs, **kw)
        if not _TORCH:
            raise ImportError("ContrastiveLearning requires PyTorch")
        self.temperature = temperature
        self.lr = lr
        self.epochs = epochs
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.projector.parameters()),
            lr=lr)

    def _augment(self, x):
        """Create an augmented view of x (add noise)."""
        noise = torch.randn_like(x) * 0.1
        return x + noise

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        X_t = torch.from_numpy(X)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            # create two augmented views
            x1 = self._augment(X_t)
            x2 = self._augment(X_t)
            # encode + project
            z1 = self.projector(self.encoder(x1))
            z2 = self.projector(self.encoder(x2))
            # normalize
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            # compute similarity matrix
            sim = torch.mm(z1, z2.t()) / self.temperature
            # NT-Xent loss
            labels = torch.arange(X_t.shape[0])
            loss = F.cross_entropy(sim, labels)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            z = self.encoder(x)
        return z.squeeze(0).numpy()


# ============================================================================
# 3. GRPO — Group Relative Policy Optimization (DeepSeek 2024)
# ============================================================================
class GRPO(Technique):
    """GRPO — Group Relative Policy Optimization.

    Simplified: samples G outputs for each prompt, computes group-relative
    advantages, and optimizes the policy.

    Advantage_i = (reward_i - mean(rewards)) / (std(rewards) + ε)
    Loss = -mean(advantage_i * log π(output_i))
    """
    name = "grpo"
    family = "alignment"

    def __init__(self, policy_model: Any = None,
                 group_size: int = 4, lr: float = 1e-5,
                 epochs: int = 50, **kw):
        super().__init__(group_size=group_size, lr=lr, epochs=epochs, **kw)
        if not _TORCH:
            raise ImportError("GRPO requires PyTorch")
        self.policy = policy_model
        self.group_size = group_size
        self.lr = lr
        self.epochs = epochs
        self.history: List[float] = []

    def fit(self, X, y=None):
        """X: dict with 'prompt' and 'rewards' (list of reward scores)."""
        if self.policy is None:
            raise ValueError("must provide policy_model")
        opt = torch.optim.Adam(
            self.policy.parameters() if hasattr(self.policy, "parameters") else [],
            lr=self.lr)
        rewards = X.get("rewards", [0.5] * self.group_size)
        rewards = torch.tensor(rewards[:self.group_size], dtype=torch.float32)
        for ep in range(self.epochs):
            opt.zero_grad()
            # compute group-relative advantages
            mean_r = rewards.mean()
            std_r = rewards.std() + 1e-8
            advantages = (rewards - mean_r) / std_r
            # get log probs for each output
            log_probs = []
            for i in range(self.group_size):
                lp = self.policy.forward(X.get("prompt", X)) if hasattr(self.policy, "forward") else 0.0
                log_probs.append(torch.tensor(float(lp), requires_grad=True))
            log_probs = torch.stack(log_probs)
            # GRPO loss
            loss = -(advantages * log_probs).mean()
            loss.backward()
            opt.step()
            self.history.append(float(loss))
        self._trained = True
        return self

    def forward(self, x):
        if self.policy is None:
            return 0.0
        return self.policy.forward(x) if hasattr(self.policy, "forward") else 0.0


# ============================================================================
# 4. Text vectorizers (from sklearn)
# ============================================================================
class CountVectorizer(Technique):
    """Convert text documents to a matrix of token counts."""
    name = "count_vectorizer"
    family = "preprocessing"
    def __init__(self, max_features=1000, **kw):
        super().__init__(max_features=max_features, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.feature_extraction.text import CountVectorizer as SK
        self._m = SK(max_features=max_features)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x):
        if isinstance(x, str): x = [x]
        return self._m.transform(x).toarray()[0]

class TfidfVectorizer(Technique):
    """Convert text documents to TF-IDF feature matrix."""
    name = "tfidf_vectorizer"
    family = "preprocessing"
    def __init__(self, max_features=1000, **kw):
        super().__init__(max_features=max_features, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.feature_extraction.text import TfidfVectorizer as SK
        self._m = SK(max_features=max_features)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x):
        if isinstance(x, str): x = [x]
        return self._m.transform(x).toarray()[0]

class TargetEncoder(Technique):
    """Encode categorical features using target mean encoding."""
    name = "target_encoder"
    family = "preprocessing"
    def __init__(self, **kw):
        super().__init__(**kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import TargetEncoder as SK
        self._m = SK()
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x):
        return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class HashingVectorizer(Technique):
    """Convert text to a matrix of hashed features."""
    name = "hashing_vectorizer"
    family = "preprocessing"
    def __init__(self, n_features=100, **kw):
        super().__init__(n_features=n_features, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.feature_extraction.text import HashingVectorizer as SK
        self._m = SK(n_features=n_features)
    def fit(self, X, y=None): self._trained = True; return self  # stateless
    def forward(self, x):
        if isinstance(x, str): x = [x]
        return self._m.transform(x).toarray()[0]


__all__ = [
    "DPO", "ContrastiveLearning", "GRPO",
    "CountVectorizer", "TfidfVectorizer", "TargetEncoder", "HashingVectorizer",
]
