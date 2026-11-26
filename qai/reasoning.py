"""
qai.reasoning -- test-time compute scaling (2024-2026 era).

  1. ChainOfThought  -- prompt the model to think step-by-step
  2. SelfConsistency  -- sample N CoT chains, majority-vote the answer
  3. TreeOfThoughts   -- already in qai.agents; thin wrapper here
  4. BestOfN          -- sample N, score with a verifier, return best
  5. BudgetForcing    -- s1-style (Muennighoff 2025): cap tokens, force "wait"
                         / "end" tokens to control reasoning depth
  6. StepVerifier     -- verify each step of a CoT against the previous
  7. ProcessReward    -- train a small reward model on intermediate steps
"""
from __future__ import annotations
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# 1. Chain of Thought
# ============================================================================
class ChainOfThought:
    """Wrap an LLM with a step-by-step reasoning prompt."""

    def __init__(self, llm: Callable[[str], str],
                 cot_prompt: str = "Let's think step by step."):
        self.llm = llm
        self.cot_prompt = cot_prompt

    def run(self, query: str) -> Dict[str, Any]:
        prompt = f"{query}\n\n{self.cot_prompt}"
        response = self.llm(prompt)
        # split reasoning from answer
        parts = re.split(r"(?i)answer\s*:", response, maxsplit=1)
        reasoning = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else response
        return {"reasoning": reasoning, "answer": answer, "raw": response}


# ============================================================================
# 2. Self-Consistency (Wang et al. 2022)
# ============================================================================
class SelfConsistency:
    """Sample N CoT chains, majority-vote the final answer."""

    def __init__(self, llm: Callable[[str], str], n_samples: int = 5,
                 cot_prompt: str = "Let's think step by step.",
                 answer_extractor: Optional[Callable[[str], str]] = None):
        self.llm = llm
        self.n_samples = n_samples
        self.cot_prompt = cot_prompt
        self.extractor = answer_extractor or self._default_extractor

    @staticmethod
    def _default_extractor(response: str) -> str:
        # take last line, strip non-alphanumeric prefixes
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        if not lines:
            return response[:50]
        last = lines[-1]
        last = re.sub(r"^(the answer is|answer|final answer|result)[:\s]*", "", last, flags=re.I)
        return last[:100]

    def run(self, query: str) -> Dict[str, Any]:
        chains = []
        answers = []
        for _ in range(self.n_samples):
            prompt = f"{query}\n\n{self.cot_prompt}"
            response = self.llm(prompt)
            chains.append(response)
            answers.append(self.extractor(response))
        # majority vote
        counter = Counter(answers)
        most_common, count = counter.most_common(1)[0]
        return {
            "answer": most_common,
            "confidence": count / self.n_samples,
            "n_samples": self.n_samples,
            "chains": chains,
            "all_answers": answers,
            "vote_distribution": dict(counter),
        }


# ============================================================================
# 3. Best-of-N with verifier
# ============================================================================
class BestOfN:
    """Sample N responses, score with a verifier, return best."""

    def __init__(self, llm: Callable[[str], str],
                 verifier: Callable[[str, str], float],
                 n: int = 4):
        self.llm = llm
        self.verifier = verifier
        self.n = n

    def run(self, query: str) -> Dict[str, Any]:
        samples = [self.llm(query) for _ in range(self.n)]
        scores = [self.verifier(query, s) for s in samples]
        best_idx = max(range(self.n), key=lambda i: scores[i])
        return {"answer": samples[best_idx], "score": scores[best_idx],
                "all_samples": samples, "all_scores": scores, "best_idx": best_idx}


# ============================================================================
# 4. Budget Forcing (s1, Muennighoff et al. 2025)
# ============================================================================
class BudgetForcing:
    """s1-style budget forcing: control reasoning depth via tokens.

    The model is told to think between <think> and </think> tags. If the
    response is too short, append "Wait" to force more reasoning. If too
    long, append the end token to cut off.
    """

    def __init__(self, llm: Callable[[str], str],
                 min_tokens: int = 100,
                 max_tokens: int = 2000,
                 wait_token: str = "Wait",
                 end_token: str = "</think>"):
        self.llm = llm
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.wait_token = wait_token
        self.end_token = end_token

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def run(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        prompt = f"{query}\n\n<think>"
        response = self.llm(prompt)
        n_tokens = self._count_tokens(response)
        iterations = 0
        while iterations < max_iterations:
            if n_tokens < self.min_tokens:
                # too short; force more reasoning
                response = response + f"\n{self.wait_token}"
                extension = self.llm(prompt + response[len("<think>"):])
                response = response + "\n" + extension
                n_tokens = self._count_tokens(response)
            elif n_tokens > self.max_tokens:
                # too long; cut off
                response = response + self.end_token
                break
            else:
                break
            iterations += 1
        # extract final answer
        answer_match = re.search(r"</think>\s*(.+)", response, re.DOTALL)
        final_answer = answer_match.group(1).strip() if answer_match else response[-200:]
        return {
            "reasoning": response,
            "answer": final_answer,
            "n_tokens": n_tokens,
            "iterations": iterations,
            "budget_enforced": iterations > 0,
        }


# ============================================================================
# 5. Step Verifier -- verify each CoT step
# ============================================================================
class StepVerifier:
    """Split a CoT into steps; have a verifier check each transition."""

    def __init__(self, llm: Callable[[str], str],
                 verifier: Optional[Callable[[str, str], bool]] = None):
        self.llm = llm
        self.verifier = verifier or self._default_verifier

    @staticmethod
    def _default_verifier(prev: str, curr: str) -> bool:
        """Default: check that current step references keywords from previous."""
        prev_words = set(prev.lower().split())
        curr_words = set(curr.lower().split())
        return len(prev_words & curr_words) > 0

    def verify(self, reasoning: str) -> Dict[str, Any]:
        steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reasoning) if s.strip()]
        if len(steps) < 2:
            return {"valid": True, "n_steps": len(steps), "verified": []}
        verified = []
        for i in range(1, len(steps)):
            ok = self.verifier(steps[i-1], steps[i])
            verified.append({"step": i, "from": steps[i-1][:80], "to": steps[i][:80],
                              "valid": ok})
        all_valid = all(v["valid"] for v in verified)
        return {"valid": all_valid, "n_steps": len(steps), "verified": verified}

    def run(self, query: str) -> Dict[str, Any]:
        cot = ChainOfThought(self.llm).run(query)
        verification = self.verify(cot["reasoning"])
        return {**cot, "verification": verification}


# ============================================================================
# 6. Process Reward Model
# ============================================================================
class ProcessRewardModel:
    """Score each intermediate step of a reasoning chain.

    A lightweight rule-based reward model. Replaces each step with an
    embedding (hash trick) and computes a learned scalar reward.
    """
    def __init__(self, dim: int = 32, lr: float = 0.01):
        self.dim = dim
        self.lr = lr
        self.weights = {}  # step_hash -> reward

    def _hash(self, text: str) -> int:
        import hashlib
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % self.dim

    def score_step(self, step: str) -> float:
        return self.weights.get(self._hash(step), 0.5)

    def score_chain(self, chain: List[str]) -> List[float]:
        return [self.score_step(s) for s in chain]

    def update(self, step: str, target_reward: float) -> None:
        h = self._hash(step)
        current = self.weights.get(h, 0.5)
        self.weights[h] = current + self.lr * (target_reward - current)

    def best_chain(self, chains: List[List[str]]) -> Tuple[int, float]:
        scored = [(i, float(sum(self.score_chain(c)) / max(len(c), 1))) for i, c in enumerate(chains)]
        scored.sort(key=lambda x: -x[1])
        return scored[0]


__all__ = [
    "ChainOfThought", "SelfConsistency", "BestOfN", "BudgetForcing",
    "StepVerifier", "ProcessRewardModel",
]
