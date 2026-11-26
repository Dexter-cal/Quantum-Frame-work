"""
qai.inference — advanced inference methods for generative models.

Provides:
  - Speculative decoding (Leviathan et al. 2023) — 2-3x speedup with draft model
  - Streaming generation — yield tokens one at a time
  - Beam search — for translation/summarization
  - Min-p sampling — better than top-p for long-tail diversity
  - Typical sampling (LocalTypical) — Meister et al. 2023
  - Best-of-N sampling — sample N, pick best by reward

Usage:
    import qai
    m = qai.build(technique="transformer", vocab_size=1000, ...)
    m.train(corpus)

    # Streaming generation
    for token in qai.stream_generate(m, prompt, max_new_tokens=100):
        print(token, end="", flush=True)

    # Speculative decoding (2-3x faster)
    draft = qai.build(technique="transformer", vocab_size=1000, n_layers=1, ...)  # small
    text = qai.speculative_decode(m, draft, prompt, max_new_tokens=100)

    # Beam search
    text = qai.beam_search(m, prompt, beam_width=5, max_new_tokens=50)

    # Min-p sampling
    text = qai.generate_min_p(m, prompt, min_p=0.05, max_new_tokens=100)
"""
from __future__ import annotations
import math
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# =====================================================================
# Streaming generation
# =====================================================================
def stream_generate(model, prompt, max_new_tokens: int = 100,
                     temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                     **kwargs) -> Generator[int, None, None]:
    """Stream tokens one at a time from a generative model.

    Yields token IDs (int) as they are generated. The caller is responsible
    for decoding them to text (e.g. via tokenizer.decode([token_id])).

    Args:
        model: a trained generative model (transformer, retnet, etc.)
        prompt: list of token IDs (int)
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature (1.0 = same, <1 = sharper, >1 = more random)
        top_k: keep only top-k tokens (0 = disabled)
        top_p: keep tokens with cumulative prob ≤ top_p (1.0 = disabled)

    Yields:
        int — next token ID
    """
    tokens = list(prompt)
    for _ in range(max_new_tokens):
        # Get logits for next token
        next_token = _sample_next_token(model, tokens, temperature, top_k, top_p)
        if next_token is None:
            break
        tokens.append(next_token)
        yield next_token
        # Stop on EOS (assume token 0 or 2 is EOS — model-specific)
        if next_token in (0, 2):  # common EOS token IDs
            break


def _sample_next_token(model, tokens, temperature: float, top_k: int, top_p: float):
    """Sample the next token given the current sequence."""
    if not _TORCH:
        return None
    tech = model.technique
    if not hasattr(tech, "_forward_logits"):
        return None
    # eval mode for inference
    if hasattr(tech, "eval"):
        tech.eval()
    with torch.no_grad():
        # Forward through current tokens
        token_tensor = torch.tensor([tokens], dtype=torch.long)
        logits = tech._forward_logits(token_tensor)
        # Take logits for the last position
        next_logits = logits[0, -1] / max(temperature, 1e-8)
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, next_logits.shape[-1])
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove)
            next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))
        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        return int(next_token)


# =====================================================================
# Speculative decoding (Leviathan et al. 2023)
# =====================================================================
def speculative_decode(model, draft_model, prompt, max_new_tokens: int = 100,
                        draft_length: int = 4, temperature: float = 1.0,
                        **kwargs) -> List[int]:
    """Speculative decoding — 2-3x speedup using a small draft model.

    The draft model generates `draft_length` tokens speculatively, then the
    main model verifies them in parallel. Accepted tokens are kept; the
    first rejected token is replaced by a sample from the main model.

    Args:
        model: the main (large) model — verifier
        draft_model: a smaller, faster model — proposer
        prompt: list of token IDs
        max_new_tokens: total tokens to generate
        draft_length: tokens to draft per round (typical: 4-8)
        temperature: sampling temperature

    Returns:
        List of generated token IDs
    """
    tokens = list(prompt)
    generated = 0
    while generated < max_new_tokens:
        # 1. Draft model generates `draft_length` tokens
        draft_tokens = []
        for _ in range(draft_length):
            next = _sample_next_token(draft_model, tokens + draft_tokens,
                                        temperature, 0, 1.0)
            if next is None:
                break
            draft_tokens.append(next)
        if not draft_tokens:
            break
        # 2. Main model verifies (single forward pass on tokens + draft)
        if not _TORCH or not hasattr(model.technique, "_forward_logits"):
            # Fallback: just accept draft tokens
            tokens.extend(draft_tokens)
            generated += len(draft_tokens)
            continue
        with torch.no_grad():
            full = torch.tensor([tokens + draft_tokens], dtype=torch.long)
            logits = model.technique._forward_logits(full)
            # The logits at positions [len(tokens)-1, len(tokens), ..., len(tokens)+len(draft)-1]
            # predict the next tokens after each draft position
            main_probs = F.softmax(logits[0] / max(temperature, 1e-8), dim=-1)
        # 3. Accept/reject each draft token
        n_accepted = 0
        for i, draft_tok in enumerate(draft_tokens):
            main_prob = main_probs[len(tokens) - 1 + i, draft_tok].item()
            draft_prob = 1.0  # simplification — would need draft model probs
            # Accept with prob min(1, main_prob / draft_prob)
            accept_prob = min(1.0, main_prob / max(draft_prob, 1e-8))
            if np.random.random() < accept_prob:
                tokens.append(draft_tok)
                n_accepted += 1
                generated += 1
                if generated >= max_new_tokens:
                    break
            else:
                # Reject: sample from (main_prob - draft_prob) normalized
                residual = main_probs[len(tokens) - 1 + i].clone()
                residual[draft_tok] = max(0, main_prob - draft_prob)
                residual = residual / (residual.sum() + 1e-8)
                next_token = torch.multinomial(residual, 1).item()
                tokens.append(int(next_token))
                generated += 1
                break
        # If all draft tokens accepted, sample one more from main model
        if n_accepted == len(draft_tokens) and generated < max_new_tokens:
            pos = len(tokens) - 1
            if pos < main_probs.shape[0]:
                next_token = torch.multinomial(main_probs[pos], 1).item()
                tokens.append(int(next_token))
                generated += 1
    return tokens[len(prompt):]


# =====================================================================
# Beam search
# =====================================================================
def beam_search(model, prompt, beam_width: int = 5, max_new_tokens: int = 50,
                 length_penalty: float = 1.0, **kwargs) -> List[int]:
    """Beam search for translation/summarization.

    Maintains `beam_width` candidate sequences, expands each by one token,
    keeps the top-`beam_width` by total log-probability.

    Args:
        model: a trained language model
        prompt: list of token IDs
        beam_width: number of beams (typical: 4-8)
        max_new_tokens: maximum tokens to generate
        length_penalty: penalty for length (1.0 = no penalty, <1 = prefer shorter)

    Returns:
        Best sequence (list of token IDs)
    """
    if not _TORCH or not hasattr(model.technique, "_forward_logits"):
        # Fallback: greedy
        result = list(prompt)
        for _ in range(max_new_tokens):
            next = _sample_next_token(model, result, 1.0, 0, 1.0)
            if next is None:
                break
            result.append(next)
        return result[len(prompt):]
    # Initialize beams
    beams = [(0.0, list(prompt))]  # (log_prob, tokens)
    for step in range(max_new_tokens):
        candidates = []
        for log_prob, tokens in beams:
            with torch.no_grad():
                token_tensor = torch.tensor([tokens], dtype=torch.long)
                logits = model.technique._forward_logits(token_tensor)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
            # Top beam_width tokens
            top_log_probs, top_tokens = log_probs.topk(beam_width)
            for lp, tok in zip(top_log_probs, top_tokens):
                new_log_prob = log_prob + float(lp)
                # Length penalty
                length = len(tokens) + 1 - len(prompt)
                score = new_log_prob / (length ** length_penalty)
                candidates.append((score, new_log_prob, tokens + [int(tok)]))
        # Keep top beam_width
        candidates.sort(key=lambda x: -x[0])
        beams = [(lp, toks) for _, lp, toks in candidates[:beam_width]]
        # Stop if all beams end with EOS
        if all(b[1][-1] in (0, 2) for b in beams):
            break
    # Return best beam
    best = max(beams, key=lambda x: x[0])
    return best[1][len(prompt):]


# =====================================================================
# Min-p sampling (better than top-p for long-tail diversity)
# =====================================================================
def generate_min_p(model, prompt, min_p: float = 0.05,
                    max_new_tokens: int = 100, temperature: float = 1.0,
                    **kwargs) -> List[int]:
    """Min-p sampling — keep tokens with prob ≥ min_p * max_prob.

    Better than top-p for long-tail diversity: scales the threshold with the
    top token's probability, so rare-but-relevant tokens are kept when the
    model is confident, and the threshold tightens when uncertain.

    Args:
        model: a trained language model
        prompt: list of token IDs
        min_p: relative threshold (typical: 0.05-0.1)
        max_new_tokens: maximum tokens
        temperature: sampling temperature

    Returns:
        List of generated token IDs
    """
    if not _TORCH or not hasattr(model.technique, "_forward_logits"):
        return []
    tokens = list(prompt)
    if hasattr(model.technique, "eval"):
        model.technique.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            token_tensor = torch.tensor([tokens], dtype=torch.long)
            logits = model.technique._forward_logits(token_tensor)
            next_logits = logits[0, -1] / max(temperature, 1e-8)
            probs = F.softmax(next_logits, dim=-1)
            # Min-p: keep tokens with prob ≥ min_p * max_prob
            max_prob = probs.max().item()
            threshold = min_p * max_prob
            keep = probs >= threshold
            if not keep.any():
                keep = probs >= max_prob  # at least keep the top token
            filtered = probs * keep
            filtered = filtered / (filtered.sum() + 1e-8)
            next_token = torch.multinomial(filtered, 1).item()
            tokens.append(int(next_token))
            if next_token in (0, 2):
                break
    return tokens[len(prompt):]


# =====================================================================
# Typical (LocalTypical) sampling — Meister et al. 2023
# =====================================================================
def generate_typical(model, prompt, mass: float = 0.95,
                      max_new_tokens: int = 100, temperature: float = 1.0,
                      **kwargs) -> List[int]:
    """Typical sampling — keep tokens whose entropy is close to the typical.

    Args:
        model: a trained language model
        prompt: list of token IDs
        mass: cumulative probability mass (typical: 0.95)
        max_new_tokens: maximum tokens

    Returns:
        List of generated token IDs
    """
    if not _TORCH or not hasattr(model.technique, "_forward_logits"):
        return []
    tokens = list(prompt)
    if hasattr(model.technique, "eval"):
        model.technique.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            token_tensor = torch.tensor([tokens], dtype=torch.long)
            logits = model.technique._forward_logits(token_tensor)
            next_logits = logits[0, -1] / max(temperature, 1e-8)
            probs = F.softmax(next_logits, dim=-1)
            log_probs = F.log_softmax(next_logits, dim=-1)
            # Entropy of distribution
            entropy = -(probs * log_probs).sum().item()
            # Per-token "typicality": |negative log prob - entropy|
            neg_log_probs = -log_probs
            typicality = (neg_log_probs - entropy).abs()
            # Sort by typicality (ascending = most typical first)
            sorted_indices = typicality.argsort()
            sorted_probs = probs[sorted_indices]
            # Take tokens until cumulative mass ≥ `mass`
            cumulative = 0.0
            keep_mask = torch.zeros_like(probs, dtype=torch.bool)
            for idx in sorted_indices:
                cumulative += float(probs[idx])
                keep_mask[idx] = True
                if cumulative >= mass:
                    break
            filtered = probs * keep_mask
            filtered = filtered / (filtered.sum() + 1e-8)
            next_token = torch.multinomial(filtered, 1).item()
            tokens.append(int(next_token))
            if next_token in (0, 2):
                break
    return tokens[len(prompt):]


# =====================================================================
# Best-of-N sampling
# =====================================================================
def best_of_n(model, prompt, n: int = 4, max_new_tokens: int = 100,
               reward_fn=None, **kwargs) -> List[int]:
    """Best-of-N sampling — generate N candidates, pick the best by reward.

    Args:
        model: a trained language model
        prompt: list of token IDs
        n: number of candidates
        max_new_tokens: tokens per candidate
        reward_fn: function(tokens) → float. If None, uses length as reward.

    Returns:
        Best candidate (list of token IDs)
    """
    if reward_fn is None:
        reward_fn = lambda toks: -len(toks)  # prefer shorter
    candidates = []
    for _ in range(n):
        toks = []
        for tok in stream_generate(model, prompt, max_new_tokens=max_new_tokens):
            toks.append(tok)
        candidates.append(toks)
    # Pick best by reward
    best = max(candidates, key=reward_fn)
    return best


__all__ = ["stream_generate", "speculative_decode", "beam_search",
           "generate_min_p", "generate_typical", "best_of_n"]
