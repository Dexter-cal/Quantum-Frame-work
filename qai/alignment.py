"""
qai.alignment — preference alignment trainers.

Provides:
  - DPO (Direct Preference Optimization) — Rafailov et al. 2023
  - GRPO (Group Relative Policy Optimization) — DeepSeek 2024
  - RLHF (Reinforcement Learning from Human Feedback) — Christiano et al. 2017

These are alignment methods for fine-tuning language models on human
preferences (preferred vs. rejected responses).

Usage:
    import qai
    m = qai.build(technique="transformer", vocab_size=1000, ...)
    dpo = qai.DPOTrainer(model=m, ref_model=m_ref, beta=0.1)
    dpo.train(preference_data, epochs=3)
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


def _require_torch():
    if not _TORCH:
        raise ImportError("[WHAT] Alignment trainers require PyTorch\n[FIX] pip install torch")


# =====================================================================
# DPO — Direct Preference Optimization (Rafailov et al. 2023)
# =====================================================================
class DPOTrainer:
    """DPO — Direct Preference Optimization.

    Fine-tunes a model to prefer "chosen" responses over "rejected" ones,
    using a reference model to prevent drift. No reward model needed — the
    implicit reward is computed from the log-probability ratio.

    Loss: L = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

    Args:
        model: the policy model to align (will be updated)
        ref_model: the reference model (frozen). If None, uses a copy of `model`
        beta: DPO temperature (typical: 0.1-0.5; higher = stronger preference)
        lr: learning rate (typical: 5e-7 for DPO)
    """
    def __init__(self, model, ref_model=None, beta: float = 0.1,
                 lr: float = 5e-7):
        _require_torch()
        if not model.technique.is_trained():
            raise RuntimeError(
                "[WHAT] DPOTrainer requires a trained model (the policy must "
                "already be fine-tuned via SFT before alignment)\n"
                "[FIX] Call model.train(corpus) first, then DPOTrainer(model)")
        self.model = model
        self.ref_model = ref_model if ref_model is not None else self._copy_model(model)
        self.beta = beta
        self.lr = lr
        # Freeze ref model
        for p in self._get_params(self.ref_model.technique):
            p.requires_grad = False
        # Optimizer for policy only
        self.opt = torch.optim.AdamW(self._get_params(self.model.technique), lr=lr)

    def _copy_model(self, model):
        """Deep copy a model (for reference model)."""
        import copy, qai
        # Build a fresh model with the same technique + params
        params = getattr(model.technique, "params", {}).copy()
        new = qai.build(technique=model.technique_name, **params)
        # Force the technique to build its inner net (if it's a transformer-style model)
        tech_src = model.technique
        tech_dst = new.technique
        # If the source has a net (trained), force the destination to build its own
        if hasattr(tech_src, "net") and tech_src.net is not None:
            if hasattr(tech_dst, "_build_model"):
                # Get the vocab_size and n_classes from the source
                vocab = getattr(tech_src, "vocab_size", 256)
                n_classes = getattr(tech_src, "n_classes", None)
                tech_dst._build_model(vocab_size=vocab, n_classes=n_classes)
            # Copy state dict from src.net to dst.net
            if hasattr(tech_dst, "net") and tech_dst.net is not None:
                try:
                    tech_dst.net.load_state_dict(tech_src.net.state_dict())
                except Exception:
                    pass
            tech_dst._trained = True
        return new

    def _get_params(self, tech):
        """Get all trainable parameters — try technique.parameters(), then
        scan sub-attributes for nn.Module instances."""
        params = []
        if hasattr(tech, "parameters"):
            try:
                params = list(tech.parameters())
            except Exception:
                params = []
        if not params:
            for attr in vars(tech).values():
                if isinstance(attr, nn.Parameter) and attr.requires_grad:
                    params.append(attr)
                elif isinstance(attr, nn.Module):
                    params += [p for p in attr.parameters() if p.requires_grad]
                elif isinstance(attr, (list, tuple)):
                    for x in attr:
                        if isinstance(x, nn.Module):
                            params += [p for p in x.parameters() if p.requires_grad]
        return [p for p in params if p.requires_grad]

    def _log_prob(self, model, tokens, targets):
        """Compute log p(targets | tokens) under model."""
        tech = model.technique
        if hasattr(tech, "_forward_logits"):
            logits = tech._forward_logits(tokens)
            log_probs = F.log_softmax(logits, dim=-1)
            # Gather log probs of target tokens
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            return token_log_probs.sum(-1)  # sum over sequence
        return torch.zeros(tokens.shape[0])

    def train(self, preference_data, epochs: int = 1, batch_size: int = 4,
              verbose: bool = True):
        """Train on preference data.

        Args:
            preference_data: list of dicts with keys:
                - "prompt": token ids (list of int)
                - "chosen": token ids of preferred response
                - "rejected": token ids of rejected response
            epochs: number of epochs
            batch_size: batch size
            verbose: if True, print progress
        """
        for ep in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(preference_data), batch_size):
                batch = preference_data[i:i+batch_size]
                self.opt.zero_grad()
                losses = []
                for ex in batch:
                    prompt = torch.tensor([ex["prompt"]], dtype=torch.long)
                    chosen = torch.tensor([ex["chosen"]], dtype=torch.long)
                    rejected = torch.tensor([ex["rejected"]], dtype=torch.long)
                    # Concat prompt + response for log-prob computation
                    chosen_full = torch.cat([prompt, chosen], dim=1)
                    rejected_full = torch.cat([prompt, rejected], dim=1)
                    # Log probs under policy
                    pi_chosen = self._log_prob(self.model, chosen_full[:, :-1], chosen_full[:, 1:])
                    pi_rejected = self._log_prob(self.model, rejected_full[:, :-1], rejected_full[:, 1:])
                    # Log probs under reference (no grad)
                    with torch.no_grad():
                        ref_chosen = self._log_prob(self.ref_model, chosen_full[:, :-1], chosen_full[:, 1:])
                        ref_rejected = self._log_prob(self.ref_model, rejected_full[:, :-1], rejected_full[:, 1:])
                    # DPO loss: -log σ(β * (log π(w)/π_ref(w) - log π(l)/π_ref(l)))
                    log_ratio_chosen = pi_chosen - ref_chosen
                    log_ratio_rejected = pi_rejected - ref_rejected
                    logits = self.beta * (log_ratio_chosen - log_ratio_rejected)
                    loss = -F.logsigmoid(logits).mean()
                    losses.append(loss)
                loss = torch.stack(losses).mean()
                loss.backward()
                self.opt.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            if verbose:
                print(f"  [DPO] epoch {ep+1}/{epochs} loss={avg:.4f}")
        return self.model


# =====================================================================
# GRPO — Group Relative Policy Optimization (DeepSeek 2024)
# =====================================================================
class GRPOTrainer:
    """GRPO — Group Relative Policy Optimization.

    Like PPO but uses group-relative advantages (no separate value function).
    For each prompt, sample G responses, compute rewards, normalize within
    the group, then update the policy.

    Args:
        model: the policy model to align
        ref_model: reference model (frozen). If None, copies `model`.
        beta: KL penalty coefficient
        group_size: number of responses per prompt (typical: 4-8)
    """
    def __init__(self, model, ref_model=None, beta: float = 0.04,
                 group_size: int = 4, lr: float = 1e-6):
        _require_torch()
        if not model.technique.is_trained():
            raise RuntimeError(
                "[WHAT] GRPOTrainer requires a trained model\n"
                "[FIX] Call model.train(corpus) first")
        self.model = model
        self.ref_model = ref_model if ref_model is not None else DPOTrainer(model)._copy_model(model)
        self.beta = beta
        self.group_size = group_size
        self.lr = lr
        for p in self._get_params(self.ref_model.technique):
            p.requires_grad = False
        self.opt = torch.optim.AdamW(self._get_params(self.model.technique), lr=lr)

    def _get_params(self, tech):
        params = []
        if hasattr(tech, "parameters"):
            try:
                params = list(tech.parameters())
            except Exception:
                params = []
        if not params:
            for attr in vars(tech).values():
                if isinstance(attr, nn.Parameter) and attr.requires_grad:
                    params.append(attr)
                elif isinstance(attr, nn.Module):
                    params += [p for p in attr.parameters() if p.requires_grad]
                elif isinstance(attr, (list, tuple)):
                    for x in attr:
                        if isinstance(x, nn.Module):
                            params += [p for p in x.parameters() if p.requires_grad]
        return [p for p in params if p.requires_grad]

    def _log_prob(self, model, tokens, targets):
        tech = model.technique
        if hasattr(tech, "_forward_logits"):
            logits = tech._forward_logits(tokens)
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            return token_log_probs.sum(-1)
        return torch.zeros(tokens.shape[0])

    def train(self, prompts_with_rewards, epochs: int = 1,
              verbose: bool = True):
        """Train on prompts with reward functions.

        Args:
            prompts_with_rewards: list of (prompt_tokens, reward_fn) where
                reward_fn(response_tokens) → float
            epochs: number of epochs
        """
        for ep in range(epochs):
            epoch_loss = 0.0
            n = 0
            for prompt_tokens, reward_fn in prompts_with_rewards:
                # Sample G responses from policy
                responses = []
                rewards = []
                for _ in range(self.group_size):
                    resp = self._sample_response(prompt_tokens)
                    responses.append(resp)
                    rewards.append(reward_fn(resp))
                rewards = np.array(rewards, dtype=np.float32)
                # Group-relative advantage: (r - mean) / (std + eps)
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                # Update policy
                self.opt.zero_grad()
                losses = []
                for resp, adv in zip(responses, advantages):
                    prompt = torch.tensor([prompt_tokens], dtype=torch.long)
                    response = torch.tensor([resp], dtype=torch.long)
                    full = torch.cat([prompt, response], dim=1)
                    pi_log_prob = self._log_prob(self.model, full[:, :-1], full[:, 1:])
                    with torch.no_grad():
                        ref_log_prob = self._log_prob(self.ref_model, full[:, :-1], full[:, 1:])
                    # Policy gradient with KL penalty
                    kl = pi_log_prob - ref_log_prob
                    loss = -adv * pi_log_prob + self.beta * kl
                    losses.append(loss)
                loss = torch.stack(losses).mean()
                loss.backward()
                self.opt.step()
                epoch_loss += float(loss.item())
                n += 1
            if verbose:
                print(f"  [GRPO] epoch {ep+1}/{epochs} loss={epoch_loss/max(n,1):.4f}")
        return self.model

    def _sample_response(self, prompt_tokens, max_len: int = 32):
        """Sample a response from the policy (greedy for simplicity)."""
        tech = self.model.technique
        # Try _forward_logits-based greedy decoding (works for transformer, retnet)
        if hasattr(tech, "_forward_logits"):
            try:
                import torch
                tokens = list(prompt_tokens)
                for _ in range(max_len):
                    with torch.no_grad():
                        t = torch.tensor([tokens], dtype=torch.long)
                        logits = tech._forward_logits(t)
                        next_tok = int(logits[0, -1].argmax().item())
                    tokens.append(next_tok)
                    if next_tok in (0, 2):
                        break
                return tokens[len(prompt_tokens):]
            except Exception:
                pass
        # Fallback: random tokens
        rng = np.random.default_rng()
        return list(rng.integers(0, 32, size=max_len))


# =====================================================================
# RLHF — Reinforcement Learning from Human Feedback (simplified PPO)
# =====================================================================
class RLHFTrainer:
    """RLHF — simplified PPO with a learned reward model.

    Pipeline:
      1. Train a reward model on (prompt, chosen, rejected) triples
      2. Use the reward model to score policy responses
      3. Update policy via PPO with KL penalty to reference

    Args:
        model: the policy model
        ref_model: reference model (frozen)
        reward_model: a model that takes (prompt, response) → scalar reward
            If None, a small MLP reward model is created
        beta: KL penalty coefficient
        clip_epsilon: PPO clip range
    """
    def __init__(self, model, ref_model=None, reward_model=None,
                 beta: float = 0.04, clip_epsilon: float = 0.2,
                 lr: float = 1e-6):
        _require_torch()
        if not model.technique.is_trained():
            raise RuntimeError(
                "[WHAT] RLHFTrainer requires a trained model\n"
                "[FIX] Call model.train(corpus) first")
        self.model = model
        self.ref_model = ref_model if ref_model is not None else DPOTrainer(model)._copy_model(model)
        self.beta = beta
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        # Reward model: simple MLP if not provided
        if reward_model is None:
            self.reward_model = self._build_reward_model()
        else:
            self.reward_model = reward_model
        for p in self._get_params(self.ref_model.technique):
            p.requires_grad = False
        self.opt = torch.optim.AdamW(self._get_params(self.model.technique), lr=lr)
        self.reward_opt = torch.optim.AdamW(self.reward_model.parameters(), lr=1e-4)

    def _build_reward_model(self):
        """Build a small reward model: token embeddings → MLP → scalar."""
        d = 64
        class RewardModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(1000, d)
                self.mlp = nn.Sequential(
                    nn.Linear(d, d), nn.ReLU(),
                    nn.Linear(d, 1),
                )
            def forward(self, tokens):
                emb = self.embed(tokens).mean(1)
                return self.mlp(emb).squeeze(-1)
        return RewardModel()

    def _get_params(self, tech):
        params = []
        if hasattr(tech, "parameters"):
            try:
                params = list(tech.parameters())
            except Exception:
                params = []
        if not params:
            for attr in vars(tech).values():
                if isinstance(attr, nn.Parameter) and attr.requires_grad:
                    params.append(attr)
                elif isinstance(attr, nn.Module):
                    params += [p for p in attr.parameters() if p.requires_grad]
                elif isinstance(attr, (list, tuple)):
                    for x in attr:
                        if isinstance(x, nn.Module):
                            params += [p for p in x.parameters() if p.requires_grad]
        return [p for p in params if p.requires_grad]

    def _log_prob(self, model, tokens, targets):
        tech = model.technique
        if hasattr(tech, "_forward_logits"):
            logits = tech._forward_logits(tokens)
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            return token_log_probs.sum(-1)
        return torch.zeros(tokens.shape[0])

    def train_reward_model(self, preference_data, epochs: int = 1):
        """Train the reward model on preference data."""
        for ep in range(epochs):
            for ex in preference_data:
                self.reward_opt.zero_grad()
                prompt = torch.tensor([ex["prompt"]], dtype=torch.long)
                chosen = torch.tensor([ex["chosen"]], dtype=torch.long)
                rejected = torch.tensor([ex["rejected"]], dtype=torch.long)
                r_chosen = self.reward_model(torch.cat([prompt, chosen], dim=1))
                r_rejected = self.reward_model(torch.cat([prompt, rejected], dim=1))
                # Bradley-Terry loss: -log σ(r_chosen - r_rejected)
                loss = -F.logsigmoid(r_chosen - r_rejected).mean()
                loss.backward()
                self.reward_opt.step()
        return self

    def train(self, prompts, epochs: int = 1, max_response_len: int = 32,
              verbose: bool = True):
        """Train policy via PPO with the reward model.

        Args:
            prompts: list of prompt token-id lists
        """
        for ep in range(epochs):
            epoch_loss = 0.0
            n = 0
            for prompt_tokens in prompts:
                # Sample response
                response = self._sample_response(prompt_tokens, max_response_len)
                prompt_t = torch.tensor([prompt_tokens], dtype=torch.long)
                response_t = torch.tensor([response], dtype=torch.long)
                full = torch.cat([prompt_t, response_t], dim=1)
                # Compute reward
                with torch.no_grad():
                    reward = self.reward_model(full).item()
                # Compute log probs under policy and ref
                pi_log_prob = self._log_prob(self.model, full[:, :-1], full[:, 1:])
                with torch.no_grad():
                    ref_log_prob = self._log_prob(self.ref_model, full[:, :-1], full[:, 1:])
                # PPO ratio
                ratio = torch.exp(pi_log_prob - ref_log_prob)
                # Advantage = reward (no value function in simplified version)
                advantage = torch.tensor([reward])
                # PPO clipped loss
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                # KL penalty
                kl = pi_log_prob - ref_log_prob
                loss = policy_loss + self.beta * kl
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                epoch_loss += float(loss.item())
                n += 1
            if verbose:
                print(f"  [RLHF] epoch {ep+1}/{epochs} loss={epoch_loss/max(n,1):.4f}")
        return self.model

    def _sample_response(self, prompt_tokens, max_len):
        """Sample a response from the policy (greedy for simplicity)."""
        tech = self.model.technique
        # Try _forward_logits-based greedy decoding (works for transformer, retnet)
        if hasattr(tech, "_forward_logits"):
            try:
                import torch
                tokens = list(prompt_tokens)
                for _ in range(max_len):
                    with torch.no_grad():
                        t = torch.tensor([tokens], dtype=torch.long)
                        logits = tech._forward_logits(t)
                        next_tok = int(logits[0, -1].argmax().item())
                    tokens.append(next_tok)
                    if next_tok in (0, 2):
                        break
                return tokens[len(prompt_tokens):]
            except Exception:
                pass
        # Fallback: random tokens
        rng = np.random.default_rng()
        return list(rng.integers(0, 32, size=max_len))


__all__ = ["DPOTrainer", "GRPOTrainer", "RLHFTrainer"]
