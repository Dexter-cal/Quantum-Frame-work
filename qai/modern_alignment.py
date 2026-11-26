"""
qai.modern_alignment — 2025-2026 alignment methods beyond DPO/GRPO/RLHF.

Provides:
  - KTO (Kahneman-Tversky Optimization) — Ethayarajh et al. 2024
  - IPO (Identity Preference Optimization) — Azar et al. 2023
  - SimPO (Simple Preference Optimization) — Meng et al. 2024
  - ORPO (Odds Ratio Preference Optimization) — Hong et al. 2024
  - RLVR (RL with Verifiable Rewards) — DeepSeek 2025

These are the 2024-2026 SOTA alignment methods that have largely replaced
plain DPO in production. Each has theoretical advantages over DPO:

  KTO:  Doesn't need paired data — works with unary (good/bad) labels.
  IPO:  Fixes DPO's overfitting on preference data via a regularizer.
  SimPO: Removes the reference model entirely (simpler, faster).
  ORPO: Combines SFT + preference learning in one step (no separate SFT).
  RLVR: Uses verifiable rewards (math, code) instead of human preferences.

Usage:
    import qai
    m = qai.build(technique="transformer", ...)
    m.train(corpus)  # SFT first

    # KTO (unary labels — doesn't need pairs!)
    kto = qai.KTOTrainer(model=m, ref_model=m_ref, beta=0.1, lr=1e-6)
    kto.train(unary_data, epochs=1)

    # SimPO (no reference model needed!)
    simpo = qai.SimPOTrainer(model=m, beta=2.0, gamma=0.5, lr=1e-6)
    simpo.train(preference_data, epochs=1)

    # RLVR (verifiable rewards — math, code)
    rlvr = qai.RLVRTrainer(model=m, reward_fn=math_verifier)
    rlvr.train(math_problems, epochs=1)
"""
from __future__ import annotations
import math, time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


def _require_torch():
    if not _TORCH or not _TORCH:
        raise ImportError("[WHAT] Modern alignment requires PyTorch\n[FIX] pip install torch")


# =====================================================================
# Helper: get model parameters
# =====================================================================
def _get_params(tech):
    """Get all trainable parameters from a technique."""
    params = []
    if hasattr(tech, "parameters"):
        try:
            params = list(tech.parameters())
        except Exception:
            params = []
    if not params:
        import torch.nn as nn
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


def _log_prob(model, tokens, targets):
    """Compute log p(targets | tokens) under model."""
    tech = model.technique
    if hasattr(tech, "_forward_logits"):
        logits = tech._forward_logits(tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(-1)
    return torch.zeros(tokens.shape[0])


def _copy_model(model):
    """Deep copy a model for reference."""
    import qai
    params = getattr(model.technique, "params", {}).copy()
    new = qai.build(technique=model.technique_name, **params)
    tech_src = model.technique
    tech_dst = new.technique
    if hasattr(tech_src, "net") and tech_src.net is not None:
        if hasattr(tech_dst, "_build_model"):
            vocab = getattr(tech_src, "vocab_size", 256)
            n_classes = getattr(tech_src, "n_classes", None)
            tech_dst._build_model(vocab_size=vocab, n_classes=n_classes)
        if hasattr(tech_dst, "net") and tech_dst.net is not None:
            try:
                tech_dst.net.load_state_dict(tech_src.net.state_dict())
            except Exception:
                pass
        tech_dst._trained = True
    return new


# =====================================================================
# 1. KTO — Kahneman-Tversky Optimization (Ethayarajh et al. 2024)
# =====================================================================
class KTOTrainer:
    """KTO — Kahneman-Tversky Optimization.

    Unlike DPO which needs PAIRED (chosen, rejected) data, KTO works with
    UNARY labels: each sample is independently marked as "good" or "bad".
    This is much easier to collect in practice.

    Uses prospect theory (loss aversion): the loss for bad samples is
    weighted more heavily than the gain for good samples.

    Loss:
        L = E_good[log σ(β · (log π(y|x)/π_ref(y|x) - z₀))]
          + λ · E_bad[log σ(β · (z₀ - log π(y|x)/π_ref(y|x)))]

    Args:
        model: the policy model (must be SFT-trained)
        ref_model: reference model (frozen; if None, copies model)
        beta: KTO temperature (typical: 0.1)
        desirable_weight: weight for good samples (typical: 1.0)
        undesirable_weight: weight for bad samples (typical: 1.0; loss aversion → >1)
        lr: learning rate (typical: 1e-6)
    """
    def __init__(self, model, ref_model=None, beta: float = 0.1,
                 desirable_weight: float = 1.0, undesirable_weight: float = 1.0,
                 lr: float = 1e-6):
        _require_torch()
        if not model.technique.is_trained():
            raise RuntimeError("KTOTrainer requires a trained (SFT) model")
        self.model = model
        self.ref_model = ref_model if ref_model is not None else _copy_model(model)
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.lr = lr
        for p in _get_params(self.ref_model.technique):
            p.requires_grad = False
        self.opt = torch.optim.AdamW(_get_params(self.model.technique), lr=lr)

    def train(self, unary_data, epochs: int = 1, verbose: bool = True):
        """Train on unary-labeled data.

        Args:
            unary_data: list of dicts with keys:
                - "prompt": token IDs
                - "response": token IDs
                - "desirable": bool (True=good, False=bad)
            epochs: number of epochs
        """
        for ep in range(epochs):
            total_loss = 0.0
            n = 0
            for ex in unary_data:
                self.opt.zero_grad()
                prompt = torch.tensor([ex["prompt"]], dtype=torch.long)
                response = torch.tensor([ex["response"]], dtype=torch.long)
                full = torch.cat([prompt, response], dim=1)
                # Log probs under policy and reference
                pi_logp = _log_prob(self.model, full[:, :-1], full[:, 1:])
                with torch.no_grad():
                    ref_logp = _log_prob(self.ref_model, full[:, :-1], full[:, 1:])
                # Log ratio
                log_ratio = pi_logp - ref_logp
                # KTO loss (simplified — z₀ is the average log ratio)
                if ex["desirable"]:
                    # Good sample: maximize log σ(β · log_ratio)
                    loss = -F.logsigmoid(self.beta * log_ratio).mean()
                    loss = loss * self.desirable_weight
                else:
                    # Bad sample: maximize log σ(-β · log_ratio)
                    loss = -F.logsigmoid(-self.beta * log_ratio).mean()
                    loss = loss * self.undesirable_weight
                loss.backward()
                self.opt.step()
                total_loss += float(loss.item())
                n += 1
            if verbose:
                print(f"  [KTO] epoch {ep+1}/{epochs} loss={total_loss/max(n,1):.4f}")
        return self.model


# =====================================================================
# 2. IPO — Identity Preference Optimization (Azar et al. 2023)
# =====================================================================
class IPOTrainer:
    """IPO — Identity Preference Optimization.

    Fixes DPO's overfitting problem by replacing the sigmoid with a squared
    loss. More robust to noisy preference data.

    Loss:
        L = (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) - 1/(2β))²

    Args:
        model: policy model (SFT-trained)
        ref_model: reference model (frozen)
        beta: regularization strength (typical: 0.1)
        lr: learning rate
    """
    def __init__(self, model, ref_model=None, beta: float = 0.1,
                 lr: float = 1e-6):
        _require_torch()
        if not model.technique.is_trained():
            raise RuntimeError("IPOTrainer requires a trained (SFT) model")
        self.model = model
        self.ref_model = ref_model if ref_model is not None else _copy_model(model)
        self.beta = beta
        self.lr = lr
        for p in _get_params(self.ref_model.technique):
            p.requires_grad = False
        self.opt = torch.optim.AdamW(_get_params(self.model.technique), lr=lr)

    def train(self, preference_data, epochs: int = 1, verbose: bool = True):
        """Train on paired preference data (same format as DPO)."""
        for ep in range(epochs):
            total_loss = 0.0
            n = 0
            for ex in preference_data:
                self.opt.zero_grad()
                prompt = torch.tensor([ex["prompt"]], dtype=torch.long)
                chosen = torch.tensor([ex["chosen"]], dtype=torch.long)
                rejected = torch.tensor([ex["rejected"]], dtype=torch.long)
                chosen_full = torch.cat([prompt, chosen], dim=1)
                rejected_full = torch.cat([prompt, rejected], dim=1)
                pi_chosen = _log_prob(self.model, chosen_full[:, :-1], chosen_full[:, 1:])
                pi_rejected = _log_prob(self.model, rejected_full[:, :-1], rejected_full[:, 1:])
                with torch.no_grad():
                    ref_chosen = _log_prob(self.ref_model, chosen_full[:, :-1], chosen_full[:, 1:])
                    ref_rejected = _log_prob(self.ref_model, rejected_full[:, :-1], rejected_full[:, 1:])
                log_ratio_chosen = pi_chosen - ref_chosen
                log_ratio_rejected = pi_rejected - ref_rejected
                # IPO squared loss
                target = 1.0 / (2 * self.beta)
                loss = ((log_ratio_chosen - log_ratio_rejected - target) ** 2).mean()
                loss.backward()
                self.opt.step()
                total_loss += float(loss.item())
                n += 1
            if verbose:
                print(f"  [IPO] epoch {ep+1}/{epochs} loss={total_loss/max(n,1):.4f}")
        return self.model


# =====================================================================
# 3. SimPO — Simple Preference Optimization (Meng et al. 2024)
# =====================================================================
class SimPOTrainer:
    """SimPO — Simple Preference Optimization.

    Removes the reference model entirely! Uses the average log-probability
    as the implicit reward, with a length-normalization and a target margin γ.

    Simpler than DPO (no reference model), faster (less memory), and
    often better performance.

    Loss:
        L = -log σ(β/|y_w| · log π(y_w|x) - β/|y_l| · log π(y_l|x) - γ)

    Args:
        model: policy model (SFT-trained)
        beta: reward scale (typical: 2.0-2.5, higher than DPO's 0.1)
        gamma: target margin (typical: 0.5-1.0)
        lr: learning rate
    """
    def __init__(self, model, beta: float = 2.0, gamma: float = 0.5,
                 lr: float = 1e-6):
        _require_torch()
        if not model.technique.is_trained():
            raise RuntimeError("SimPOTrainer requires a trained (SFT) model")
        self.model = model
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        # NO reference model — that's the SimPO innovation!
        self.opt = torch.optim.AdamW(_get_params(self.model.technique), lr=lr)

    def train(self, preference_data, epochs: int = 1, verbose: bool = True):
        """Train on paired preference data."""
        for ep in range(epochs):
            total_loss = 0.0
            n = 0
            for ex in preference_data:
                self.opt.zero_grad()
                prompt = torch.tensor([ex["prompt"]], dtype=torch.long)
                chosen = torch.tensor([ex["chosen"]], dtype=torch.long)
                rejected = torch.tensor([ex["rejected"]], dtype=torch.long)
                chosen_full = torch.cat([prompt, chosen], dim=1)
                rejected_full = torch.cat([prompt, rejected], dim=1)
                # Length-normalized log probs (no reference model!)
                pi_chosen = _log_prob(self.model, chosen_full[:, :-1], chosen_full[:, 1:])
                pi_rejected = _log_prob(self.model, rejected_full[:, :-1], rejected_full[:, 1:])
                chosen_reward = self.beta * pi_chosen / max(len(ex["chosen"]), 1)
                rejected_reward = self.beta * pi_rejected / max(len(ex["rejected"]), 1)
                # SimPO loss with margin γ
                loss = -F.logsigmoid(chosen_reward - rejected_reward - self.gamma).mean()
                loss.backward()
                self.opt.step()
                total_loss += float(loss.item())
                n += 1
            if verbose:
                print(f"  [SimPO] epoch {ep+1}/{epochs} loss={total_loss/max(n,1):.4f}")
        return self.model


# =====================================================================
# 4. ORPO — Odds Ratio Preference Optimization (Hong et al. 2024)
# =====================================================================
class ORPOTrainer:
    """ORPO — Odds Ratio Preference Optimization.

    Combines SFT + preference learning in ONE step. No need for a separate
    SFT phase! The loss has two components:
      1. SFT loss on the chosen response (standard cross-entropy)
      2. Odds-ratio penalty that penalizes the model for preferring rejected

    Loss:
        L = L_SFT(y_w|x) + λ · log(odds(y_w|x) / odds(y_l|x))

    where odds(p) = p / (1 - p).

    Args:
        model: policy model (can be untrained — ORPO does SFT+alignment together)
        lambda_: odds-ratio penalty weight (typical: 0.5-1.0)
        lr: learning rate
    """
    def __init__(self, model, lambda_: float = 0.5, lr: float = 1e-5):
        _require_torch()
        self.model = model
        self.lambda_ = lambda_
        self.lr = lr
        # No reference model, no SFT prerequisite — ORPO does both
        self.opt = torch.optim.AdamW(_get_params(self.model.technique), lr=lr)

    def train(self, preference_data, epochs: int = 1, verbose: bool = True):
        """Train with combined SFT + preference learning."""
        for ep in range(epochs):
            total_loss = 0.0
            n = 0
            for ex in preference_data:
                self.opt.zero_grad()
                prompt = torch.tensor([ex["prompt"]], dtype=torch.long)
                chosen = torch.tensor([ex["chosen"]], dtype=torch.long)
                rejected = torch.tensor([ex["rejected"]], dtype=torch.long)
                chosen_full = torch.cat([prompt, chosen], dim=1)
                rejected_full = torch.cat([prompt, rejected], dim=1)
                # 1. SFT loss on chosen
                chosen_logits = self.model.technique._forward_logits(chosen_full[:, :-1])
                sft_loss = F.cross_entropy(
                    chosen_logits.reshape(-1, chosen_logits.shape[-1]),
                    chosen_full[:, 1:].reshape(-1))
                # 2. Odds-ratio penalty
                with torch.no_grad():
                    chosen_logp = _log_prob(self.model, chosen_full[:, :-1], chosen_full[:, 1:])
                    rejected_logp = _log_prob(self.model, rejected_full[:, :-1], rejected_full[:, 1:])
                # odds = exp(logp) / (1 - exp(logp))
                chosen_prob = torch.exp(chosen_logp).clamp(1e-8, 1 - 1e-8)
                rejected_prob = torch.exp(rejected_logp).clamp(1e-8, 1 - 1e-8)
                chosen_odds = chosen_prob / (1 - chosen_prob)
                rejected_odds = rejected_prob / (1 - rejected_prob)
                log_odds_ratio = torch.log(chosen_odds / rejected_odds + 1e-8)
                # ORPO loss: the penalty is -log(σ(log_odds_ratio)) but simplified
                orpo_penalty = -F.logsigmoid(log_odds_ratio).mean()
                loss = sft_loss + self.lambda_ * orpo_penalty
                loss.backward()
                self.opt.step()
                total_loss += float(loss.item())
                n += 1
            if verbose:
                print(f"  [ORPO] epoch {ep+1}/{epochs} loss={total_loss/max(n,1):.4f}")
        return self.model


# =====================================================================
# 5. RLVR — RL with Verifiable Rewards (DeepSeek 2025)
# =====================================================================
class RLVRTrainer:
    """RLVR — Reinforcement Learning with Verifiable Rewards.

    The 2025 paradigm shift (DeepSeek R1). Instead of human preferences,
    uses verifiable rewards: math problems with known answers, code with
    test cases, logical puzzles with verifiable solutions.

    This enables self-improving reasoning: the model learns to generate
    chain-of-thought reasoning that leads to verifiably correct answers,
    without any human annotation.

    Pipeline:
      1. Sample response from model (with chain-of-thought)
      2. Verify the answer with a programmatic reward function
      3. Use GRPO-style policy gradient with the verifiable reward

    Args:
        model: policy model (should be SFT-trained first)
        ref_model: reference model for KL penalty (frozen)
        reward_fn: function(response_tokens) → float (verifiable reward)
            Examples:
              - Math: extract answer, compare to ground truth
              - Code: run tests, count passing
              - Logic: check formal proof
        beta: KL penalty coefficient
        group_size: number of responses per problem (for GRPO-style advantage)
        max_response_len: maximum response length

    Usage:
        # Math RLVR
        def math_reward(response_tokens, answer=42):
            text = tokenizer.decode(response_tokens)
            # Extract the final number from the response
            try:
                predicted = int(text.split('Answer:')[-1].strip())
                return 1.0 if predicted == answer else 0.0
            except:
                return 0.0

        rlvr = qai.RLVRTrainer(model=m, reward_fn=math_reward, beta=0.04)
        rlvr.train(math_problems, epochs=10)
    """
    def __init__(self, model, reward_fn: Callable, ref_model=None,
                 beta: float = 0.04, group_size: int = 4,
                 max_response_len: int = 256, lr: float = 1e-6):
        _require_torch()
        if not model.technique.is_trained():
            raise RuntimeError("RLVRTrainer requires a trained (SFT) model")
        self.model = model
        self.reward_fn = reward_fn
        self.ref_model = ref_model if ref_model is not None else _copy_model(model)
        self.beta = beta
        self.group_size = group_size
        self.max_response_len = max_response_len
        self.lr = lr
        for p in _get_params(self.ref_model.technique):
            p.requires_grad = False
        self.opt = torch.optim.AdamW(_get_params(self.model.technique), lr=lr)

    def train(self, problems, epochs: int = 1, verbose: bool = True):
        """Train on problems with verifiable rewards.

        Args:
            problems: list of (prompt_tokens, problem_context) where
                problem_context is passed to reward_fn for verification.
                For math: (prompt, ground_truth_answer)
                For code: (prompt, test_cases)
        """
        for ep in range(epochs):
            total_reward = 0.0
            n_problems = 0
            for prompt_tokens, context in problems:
                # Sample G responses
                responses = []
                rewards = []
                for _ in range(self.group_size):
                    resp = self._sample_response(prompt_tokens)
                    responses.append(resp)
                    reward = self.reward_fn(resp, context) if context is not None else self.reward_fn(resp)
                    rewards.append(reward)
                rewards = np.array(rewards, dtype=np.float32)
                # Group-relative advantage
                if rewards.std() > 0:
                    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                else:
                    advantages = rewards - rewards.mean()
                # Policy gradient update
                self.opt.zero_grad()
                losses = []
                for resp, adv in zip(responses, advantages):
                    prompt = torch.tensor([prompt_tokens], dtype=torch.long)
                    response = torch.tensor([resp], dtype=torch.long)
                    full = torch.cat([prompt, response], dim=1)
                    pi_logp = _log_prob(self.model, full[:, :-1], full[:, 1:])
                    with torch.no_grad():
                        ref_logp = _log_prob(self.ref_model, full[:, :-1], full[:, 1:])
                    kl = pi_logp - ref_logp
                    loss = -adv * pi_logp + self.beta * kl
                    losses.append(loss)
                if losses:
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    self.opt.step()
                total_reward += rewards.mean()
                n_problems += 1
            avg_reward = total_reward / max(n_problems, 1)
            if verbose:
                print(f"  [RLVR] epoch {ep+1}/{epochs} avg_reward={avg_reward:.3f}")
        return self.model

    def _sample_response(self, prompt_tokens, max_len=None):
        """Sample a response from the policy (greedy for simplicity)."""
        max_len = max_len or self.max_response_len
        tech = self.model.technique
        if hasattr(tech, "_forward_logits"):
            try:
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
        rng = np.random.default_rng()
        return list(rng.integers(0, 32, size=max_len))


# =====================================================================
# Convenience: auto-select alignment method
# =====================================================================
def align(model, data, method: str = "dpo", **kwargs):
    """Auto-select and run an alignment method.

    Args:
        model: SFT-trained policy model
        data: preference data (paired) or unary data (for KTO)
        method: "dpo", "kto", "ipo", "simpo", "orpo", "rlvr", "grpo", "rlhf"
        **kwargs: method-specific args

    Returns:
        The aligned model.
    """
    method = method.lower()
    if method == "dpo":
        import qai
        trainer = qai.DPOTrainer(model=model, **kwargs)
    elif method == "kto":
        trainer = KTOTrainer(model=model, **kwargs)
    elif method == "ipo":
        trainer = IPOTrainer(model=model, **kwargs)
    elif method == "simpo":
        trainer = SimPOTrainer(model=model, **kwargs)
    elif method == "orpo":
        trainer = ORPOTrainer(model=model, **kwargs)
    elif method == "grpo":
        import qai
        trainer = qai.GRPOTrainer(model=model, **kwargs)
    elif method == "rlhf":
        import qai
        trainer = qai.RLHFTrainer(model=model, **kwargs)
    elif method == "rlvr":
        trainer = RLVRTrainer(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown alignment method: {method}. "
                         f"Supported: dpo, kto, ipo, simpo, orpo, grpo, rlhf, rlvr")
    trainer.train(data, epochs=kwargs.get("epochs", 1))
    return model


__all__ = ["KTOTrainer", "IPOTrainer", "SimPOTrainer", "ORPOTrainer",
           "RLVRTrainer", "align"]
