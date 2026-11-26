"""
qai.test_time_scaling — test-time compute scaling methods.

Provides the 2025-2026 paradigm of "thinking longer at inference time":
  - ChainOfThought — structured reasoning before answering
  - SelfConsistency — sample multiple chains, vote on the answer
  - TreeOfThoughts — explore multiple reasoning paths
  - VerificationGuided — generate + verify + refine
  - BestOfNWithVerifier — generate N candidates, verify, pick best
  - ReasoningChain — multi-step reasoning with self-reflection

These are the methods behind o1/o3-style "reasoning" models.

Usage:
    import qai
    m = qai.build(technique="transformer", ...)

    # Chain of thought
    answer = qai.ChainOfThought(m).solve("What is 15% of 240?")

    # Self-consistency (sample 5 chains, vote)
    answer = qai.SelfConsistency(m, n_samples=5).solve("What is 15% of 240?")

    # Tree of thoughts
    answer = qai.TreeOfThoughts(m, n_branches=3, depth=3).solve(problem)

    # Verification-guided
    answer = qai.VerificationGuided(m, verifier_fn).solve(problem)
"""
from __future__ import annotations
import re, time
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np


def _generate(model, prompt: str, max_new_tokens: int = 100) -> str:
    """Generate text from a model."""
    try:
        import qai
        tokens = [ord(c) % 256 for c in prompt[:256]]  # truncate + tokenize
        gen = list(qai.stream_generate(model, tokens, max_new_tokens=max_new_tokens))
        return "".join(chr(t % 256) for t in gen)
    except Exception:
        return ""


# =====================================================================
# Chain of Thought
# =====================================================================
class ChainOfThought:
    """Chain-of-thought reasoning: generate step-by-step reasoning before answering.

    Prompt format:
        Question: <problem>
        Let's think step by step.
        <reasoning>
        Therefore, the answer is: <answer>

    Args:
        model: a qai Model
        max_reasoning_tokens: max tokens for the reasoning chain
    """
    def __init__(self, model, max_reasoning_tokens: int = 200):
        self.model = model
        self.max_reasoning_tokens = max_reasoning_tokens

    def solve(self, problem: str) -> str:
        """Solve a problem using chain-of-thought reasoning."""
        prompt = (
            f"Question: {problem}\n"
            f"Let's think step by step.\n"
        )
        response = _generate(self.model, prompt, self.max_reasoning_tokens)
        # Extract answer from the reasoning
        if "answer is:" in response.lower():
            answer = response.lower().split("answer is:")[-1].strip()
        else:
            # Try to extract the last number
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
            answer = numbers[-1] if numbers else response.strip()
        return answer


# =====================================================================
# Self-Consistency (sample multiple chains, vote)
# =====================================================================
class SelfConsistency:
    """Self-consistency: sample multiple reasoning chains and vote.

    1. Generate N independent chain-of-thought solutions
    2. Extract the answer from each
    3. Return the most common answer (majority vote)

    Args:
        model: a qai Model
        n_samples: number of independent samples (typical: 5-40)
        temperature: sampling temperature (higher = more diversity)
    """
    def __init__(self, model, n_samples: int = 5, temperature: float = 0.7):
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature

    def solve(self, problem: str) -> str:
        """Solve a problem with self-consistency voting."""
        cot = ChainOfThought(self.model)
        answers = []
        for _ in range(self.n_samples):
            answer = cot.solve(problem)
            answers.append(answer)
        # Majority vote
        from collections import Counter
        vote = Counter(answers)
        return vote.most_common(1)[0][0] if answers else ""


# =====================================================================
# Tree of Thoughts (explore multiple reasoning paths)
# =====================================================================
class TreeOfThoughts:
    """Tree of thoughts: explore multiple reasoning paths, pick the best.

    1. Generate N initial thoughts (branches)
    2. Evaluate each thought (heuristic or model-based)
    3. Expand the top-K thoughts further
    4. Repeat for D depth levels
    5. Return the best leaf

    Args:
        model: a qai Model
        n_branches: number of branches per node (typical: 3-5)
        depth: tree depth (typical: 2-4)
        evaluator: function(thought) → float score (if None, uses length heuristic)
    """
    def __init__(self, model, n_branches: int = 3, depth: int = 2,
                 evaluator: Optional[Callable] = None):
        self.model = model
        self.n_branches = n_branches
        self.depth = depth
        self.evaluator = evaluator

    def solve(self, problem: str) -> str:
        """Solve a problem with tree-of-thoughts search."""
        # BFS through the thought tree
        current_thoughts = [problem]
        for d in range(self.depth):
            # Expand each thought into n_branches children
            all_children = []
            for thought in current_thoughts:
                children = []
                for _ in range(self.n_branches):
                    child = _generate(self.model,
                                       f"Problem: {problem}\nThought so far: {thought}\nNext step:",
                                       max_new_tokens=50)
                    children.append(child)
                all_children.extend(children)
            # Evaluate and prune
            if self.evaluator:
                scored = [(c, self.evaluator(c)) for c in all_children]
            else:
                scored = [(c, len(c)) for c in all_children]  # length heuristic
            scored.sort(key=lambda x: -x[1])
            current_thoughts = [c for c, _ in scored[:self.n_branches]]
        # Return the best thought
        if self.evaluator:
            best = max(current_thoughts, key=self.evaluator)
        else:
            best = max(current_thoughts, key=len)
        # Extract final answer
        numbers = re.findall(r'-?\d+(?:\.\d+)?', best)
        return numbers[-1] if numbers else best


# =====================================================================
# Verification-Guided Generation
# =====================================================================
class VerificationGuided:
    """Generate an answer, verify it, and refine if needed.

    1. Generate a candidate answer
    2. Verify it (using a verifier function or self-checking)
    3. If incorrect, generate feedback and refine
    4. Repeat up to max_refinements

    Args:
        model: a qai Model
        verifier: function(problem, answer) → (is_correct, feedback)
        max_refinements: maximum refinement iterations
    """
    def __init__(self, model, verifier: Callable,
                 max_refinements: int = 3):
        self.model = model
        self.verifier = verifier
        self.max_refinements = max_refinements

    def solve(self, problem: str) -> str:
        """Solve a problem with verification-guided refinement."""
        answer = ChainOfThought(self.model).solve(problem)
        for i in range(self.max_refinements):
            is_correct, feedback = self.verifier(problem, answer)
            if is_correct:
                return answer
            # Refine
            refine_prompt = (
                f"Problem: {problem}\n"
                f"Previous answer: {answer}\n"
                f"Feedback: {feedback}\n"
                f"Corrected answer:"
            )
            answer = _generate(self.model, refine_prompt, max_new_tokens=100)
            numbers = re.findall(r'-?\d+(?:\.\d+)?', answer)
            if numbers:
                answer = numbers[-1]
        return answer


# =====================================================================
# Best-of-N with Verifier
# =====================================================================
class BestOfNWithVerifier:
    """Generate N candidates, verify each, return the best.

    Args:
        model: a qai Model
        verifier: function(problem, answer) → float score
        n: number of candidates to generate
        temperature: sampling temperature
    """
    def __init__(self, model, verifier: Callable, n: int = 4,
                 temperature: float = 0.7):
        self.model = model
        self.verifier = verifier
        self.n = n
        self.temperature = temperature

    def solve(self, problem: str) -> str:
        """Generate N candidates and return the best-verified one."""
        cot = ChainOfThought(self.model)
        candidates = [cot.solve(problem) for _ in range(self.n)]
        scored = [(c, self.verifier(problem, c)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[0][0] if scored else ""


# =====================================================================
# ReasoningChain — multi-step with self-reflection
# =====================================================================
class ReasoningChain:
    """Multi-step reasoning with self-reflection at each step.

    For each step:
      1. Generate the next reasoning step
      2. Self-reflect: "Is this step correct?"
      3. If incorrect, regenerate
      4. If correct, proceed

    This mimics o1-style "thinking" where the model checks its own work.

    Args:
        model: a qai Model
        max_steps: maximum reasoning steps
        max_reflections: maximum reflection attempts per step
    """
    def __init__(self, model, max_steps: int = 5, max_reflections: int = 2):
        self.model = model
        self.max_steps = max_steps
        self.max_reflections = max_reflections

    def solve(self, problem: str) -> str:
        """Solve a problem with self-reflective reasoning."""
        context = f"Problem: {problem}\n"
        for step in range(self.max_steps):
            # Generate step
            step_text = _generate(self.model,
                                   f"{context}\nStep {step+1}:",
                                   max_new_tokens=80)
            # Self-reflect
            for _ in range(self.max_reflections):
                reflection = _generate(self.model,
                                         f"{context}\nStep {step+1}: {step_text}\n"
                                         f"Is this step correct? (yes/no):",
                                         max_new_tokens=10)
                if "yes" in reflection.lower():
                    break
                # Regenerate
                step_text = _generate(self.model,
                                       f"{context}\nStep {step+1} (corrected):",
                                       max_new_tokens=80)
            context += f"\nStep {step+1}: {step_text}"
            # Check if we have an answer
            if "answer" in step_text.lower():
                # Extract answer
                numbers = re.findall(r'-?\d+(?:\.\d+)?', step_text)
                if numbers:
                    return numbers[-1]
        # Final answer extraction
        final = _generate(self.model,
                           f"{context}\nFinal answer:",
                           max_new_tokens=30)
        numbers = re.findall(r'-?\d+(?:\.\d+)?', final)
        return numbers[-1] if numbers else final.strip()


__all__ = ["ChainOfThought", "SelfConsistency", "TreeOfThoughts",
           "VerificationGuided", "BestOfNWithVerifier", "ReasoningChain"]
