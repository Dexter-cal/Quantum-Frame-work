"""
qai.creative.llm_enhanced -- real-LLM-backed versions of key creative modules.

These replace the rule-based stubs in MoralCompass, Reflexion, VibeCoder,
PromptEvolver, and HallucinationDetector with versions that call a real
LLM (any callable returning text).

Usage:
    from qai.creative.llm_enhanced import LLMEnhancedMoralCompass, LLMEnhancedVibeCoder
    mc = LLMEnhancedMoralCompass(llm=my_real_llm)
    result = mc.guard("Describe how to hack a system.")
"""
from __future__ import annotations
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .alignment import MoralPrinciple
from .ecosystem import AdversarialPartner
from .experimental2 import HallucinationDetector


# ============================================================================
# Common LLM interface
# ============================================================================
class LLMCallable:
    """Wrap a callable as an LLM with system + user message support.

    The underlying callable can be:
      - a function (prompt: str) -> str
      - an OpenAI-compatible client with .chat.completions.create
      - a qai.serving.LLMClient
    """
    def __init__(self, llm: Any, model: str = "default",
                 system_prompt: str = "You are a helpful assistant."):
        self.llm = llm
        self.model = model
        self.system_prompt = system_prompt

    def __call__(self, prompt: str, **kwargs) -> str:
        # OpenAI-style client
        if hasattr(self.llm, "chat") and hasattr(self.llm.chat, "completions"):
            try:
                resp = self.llm.chat.completions.create(
                    model=kwargs.get("model", self.model),
                    messages=[{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 500),
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f"[llm error] {e}"
        # qai.serving.LLMClient
        if hasattr(self.llm, "generate"):
            try:
                resp = self.llm.generate(prompt, **kwargs)
                return getattr(resp, "text", str(resp))
            except Exception as e:
                return f"[llm error] {e}"
        # plain callable
        try:
            return str(self.llm(prompt, **kwargs))
        except Exception as e:
            return f"[llm error] {e}"


# ============================================================================
# 1. LLM-Enhanced Moral Compass
# ============================================================================
class LLMEnhancedMoralCompass:
    """MoralCompass backed by a real LLM critic.

    The LLM evaluates each output against the principles and returns a
    JSON-structured verdict: {per_principle: scores, overall: float,
    violations: [...], rewrite: str}
    """
    JUDGE_PROMPT = """You are a strict ethics reviewer. Evaluate the following text against these principles:

{principles}

Text to evaluate: "{text}"

Return JSON only:
{{
  "per_principle": {{"{principle_names}": score in [-1, 1], ...}},
  "overall": average score,
  "violations": [list of principle names that scored < 0],
  "rewrite": a safer version if any violations, else empty string
}}
"""

    def __init__(self, llm: Any,
                 principles: Optional[List[MoralPrinciple]] = None,
                 temperature: float = 0.3):
        self.llm = LLMCallable(llm, system_prompt="You are a strict ethics judge. Output JSON only.")
        self.principles = principles or []
        self.temperature = temperature

    def evaluate(self, text: str) -> Dict[str, Any]:
        if not self.principles:
            return {"overall": 1.0, "violations": [], "rewrite": "", "per_principle": {}}
        principle_str = "\n".join(f"- {p.name}: {p.description}" for p in self.principles)
        principle_names = ", ".join(p.name for p in self.principles)
        prompt = self.JUDGE_PROMPT.format(
            principles=principle_str,
            text=text,
            principle_names=principle_names,
        )
        response = self.llm(prompt, temperature=self.temperature)
        try:
            # find first { and last }
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                return parsed
        except Exception:
            pass
        return {"overall": 0.0, "violations": ["parse_failed"],
                 "rewrite": response, "per_principle": {}, "raw_response": response}

    def guard(self, text: str) -> Dict[str, Any]:
        eval_result = self.evaluate(text)
        if eval_result.get("violations"):
            return {
                "original": text,
                "blocked": True,
                "violations": eval_result["violations"],
                "rewrite": eval_result.get("rewrite", text),
                "score": eval_result.get("overall", 0.0),
            }
        return {"original": text, "blocked": False, "rewrite": text,
                 "score": eval_result.get("overall", 1.0)}


# ============================================================================
# 2. LLM-Enhanced Reflexion Agent
# ============================================================================
class LLMEnhancedReflexion:
    """Reflexion agent that uses an LLM for reflection.

    Pipeline:
      1. Actor attempts the task
      2. LLM evaluates the attempt and writes a reflection
      3. Reflection is added to episodic memory
      4. Actor retries with the augmented prompt
    """
    def __init__(self, llm: Any, actor: Optional[Any] = None,
                 max_trials: int = 3):
        self.llm = LLMCallable(llm, system_prompt="You are a self-reflective agent.")
        self.actor = actor
        self.max_trials = max_trials
        self.episodic_memory: List[str] = []

    def _reflect(self, query: str, attempt: str,
                  feedback: str) -> str:
        prompt = f"""You tried to answer: {query}
Your attempt: {attempt}
Feedback: {feedback}

Write a 1-2 sentence reflection on what to do differently next time.
Reflection:"""
        return self.llm(prompt)

    def run(self, query: str,
            evaluator: Optional[Callable[[str], Tuple[bool, str]]] = None) -> Dict[str, Any]:
        history = []
        for trial in range(self.max_trials):
            memory_str = "\n".join(f"- {m}" for m in self.episodic_memory[-5:])
            augmented_query = f"{query}\n\nPast reflections:\n{memory_str}" if memory_str else query
            # actor attempt
            if self.actor is not None:
                try:
                    attempt = str(self.actor.forward(augmented_query))
                except Exception:
                    attempt = self.llm(augmented_query)
            else:
                attempt = self.llm(augmented_query)
            # evaluate
            if evaluator:
                success, feedback = evaluator(attempt)
            else:
                # LLM self-evaluate
                eval_prompt = f"Question: {query}\nAnswer: {attempt}\nIs this answer correct? Reply yes/no + brief reason."
                eval_response = self.llm(eval_prompt)
                success = eval_response.lower().startswith("yes")
                feedback = eval_response
            history.append({"trial": trial, "answer": attempt,
                              "success": success, "feedback": feedback})
            if success:
                return {"answer": attempt, "history": history, "n_trials": trial + 1}
            reflection = self._reflect(query, attempt, feedback)
            self.episodic_memory.append(reflection)
        return {"answer": history[-1]["answer"], "history": history,
                 "n_trials": self.max_trials}


# ============================================================================
# 3. LLM-Enhanced Vibe Coder
# ============================================================================
class LLMEnhancedVibeCoder:
    """VibeCoder backed by a real LLM for code generation.

    The LLM generates Python code that uses qai. The coder validates
    syntax and optionally executes in a sandbox.
    """
    SYSTEM = """You are an expert Python programmer who uses the qai framework.
Generate clean, working code. Only output Python code, no prose."""

    def __init__(self, llm: Any, execute_sandbox: bool = False):
        self.llm = LLMCallable(llm, system_prompt=self.SYSTEM)
        self.execute_sandbox = execute_sandbox

    def generate(self, request: str, **kwargs) -> Dict[str, Any]:
        prompt = f"""Generate Python code using the qai framework for:

{request}

Use qai.build() or import from qai.* modules. Return only the code, no explanation.
"""
        response = self.llm(prompt, max_tokens=kwargs.get("max_tokens", 800))
        # extract code block if present
        code_match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        code = code_match.group(1).strip() if code_match else response.strip()
        result = {
            "code": code,
            "request": request,
            "raw_response": response,
        }
        # syntax check
        try:
            import ast
            ast.parse(code)
            result["syntax_valid"] = True
        except SyntaxError as e:
            result["syntax_valid"] = False
            result["syntax_error"] = str(e)
        # sandbox execution
        if self.execute_sandbox and result.get("syntax_valid"):
            result["execution"] = self._sandbox_exec(code)
        return result

    def _sandbox_exec(self, code: str) -> Dict[str, Any]:
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            proc = subprocess.run(
                ["python", path], capture_output=True, text=True, timeout=10
            )
            return {"stdout": proc.stdout[:5000], "stderr": proc.stderr[:5000],
                    "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "timeout", "returncode": -1}
        finally:
            os.unlink(path)


# ============================================================================
# 4. LLM-Enhanced Prompt Evolver
# ============================================================================
class LLMEnhancedPromptEvolver:
    """PromptEvolver with LLM-driven mutation + LLM judge.

    The LLM is used for both:
      - mutation: paraphrase prompts intelligently
      - judging: score outputs as the fitness function
    """
    def __init__(self, task: str, llm: Any,
                 population_size: int = 6, mutation_rate: float = 0.5,
                 seed_prompts: Optional[List[str]] = None):
        self.task = task
        self.mutator_llm = LLMCallable(llm, system_prompt="You rewrite prompts to be clearer and more effective.")
        self.judge_llm = LLMCallable(llm, system_prompt="You score outputs from 0 to 1.")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.seed = seed_prompts or [
            f"Solve: {task}",
            f"Step by step, address: {task}",
            f"You are an expert. {task}",
        ]
        self.history: List[Dict[str, Any]] = []

    def _mutate(self, prompt: str) -> str:
        mutation_prompt = f"""Rewrite this prompt to be more effective for the task: "{self.task}"

Original prompt: "{prompt}"

Return only the rewritten prompt, nothing else."""
        return self.mutator_llm(mutation_prompt).strip()

    def _crossover(self, p1: str, p2: str) -> str:
        words1 = p1.split()
        words2 = p2.split()
        if not words1 or not words2:
            return p1
        half1 = words1[:len(words1) // 2]
        half2 = words2[len(words2) // 2:]
        return " ".join(half1 + half2)

    def _fitness(self, prompt: str, executor_llm: Any) -> Tuple[float, str]:
        # generate output using executor LLM
        output = executor_llm(prompt)
        # ask judge LLM to score
        judge_prompt = f"""Task: {self.task}
Output to evaluate: {output[:500]}

Score this output from 0.0 to 1.0. Return only a number."""
        score_response = self.judge_llm(judge_prompt)
        try:
            # extract first float
            match = re.search(r"(\d+\.?\d*)", score_response)
            if match:
                score = float(match.group(1))
                score = max(0.0, min(1.0, score))
                return score, output
        except Exception:
            pass
        return 0.5, output

    def evolve(self, executor_llm: Any, generations: int = 3) -> Dict[str, Any]:
        executor = LLMCallable(executor_llm) if not isinstance(executor_llm, LLMCallable) else executor_llm
        population = list(self.seed[:self.population_size])
        while len(population) < self.population_size:
            population.append(self._mutate(population[0]))
        for gen in range(generations):
            scored = []
            for p in population:
                score, output = self._fitness(p, executor)
                scored.append((p, score, output))
            scored.sort(key=lambda x: -x[1])
            top = [s[0] for s in scored[:max(2, self.population_size // 2)]]
            new_pop = list(top)
            while len(new_pop) < self.population_size:
                if random.random() < 0.5 and len(top) >= 2:
                    p1, p2 = random.sample(top, 2)
                    child = self._crossover(p1, p2)
                else:
                    child = self._mutate(random.choice(top))
                new_pop.append(child)
            population = new_pop
            self.history.append({
                "generation": gen,
                "best_prompt": scored[0][0],
                "best_fitness": scored[0][1],
                "best_output": scored[0][2],
                "mean_fitness": float(np.mean([s[1] for s in scored])),
            })
        return self.history[-1] if self.history else {}

    def best_prompt(self) -> Optional[str]:
        if not self.history:
            return None
        return max(self.history, key=lambda h: h["best_fitness"])["best_prompt"]


import random  # for PromptEvolver


# ============================================================================
# 5. LLM-Enhanced Hallucination Detector
# ============================================================================
class LLMEnhancedHallucinationDetector:
    """HallucinationDetector backed by an LLM verifier.

    The LLM checks each claim against retrieved evidence and returns
    a support score in [0, 1].
    """
    VERIFIER_PROMPT = """You are a fact-checker. Decide if the claim is supported by the evidence.

Claim: "{claim}"
Evidence: "{evidence}"

Return JSON only:
{{"supported": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}
"""

    def __init__(self, llm: Any,
                 knowledge_base: Optional[Callable[[str], List[str]]] = None):
        self.llm = LLMCallable(llm, system_prompt="You are a fact-checker. Output JSON only.")
        self.kb = knowledge_base or (lambda q: [])

    @staticmethod
    def _extract_claims(text: str) -> List[str]:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.split()) >= 3]

    def verify_claim(self, claim: str, evidence: str) -> Dict[str, Any]:
        prompt = self.VERIFIER_PROMPT.format(claim=claim, evidence=evidence)
        response = self.llm(prompt)
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except Exception:
            pass
        return {"supported": False, "confidence": 0.5, "reason": "parse_failed",
                 "raw_response": response}

    def check(self, output: str) -> Dict[str, Any]:
        claims = self._extract_claims(output)
        results = []
        for claim in claims:
            evidence_list = self.kb(claim)
            best = {"supported": False, "confidence": 0.0, "reason": "no_evidence"}
            for ev in evidence_list:
                verdict = self.verify_claim(claim, ev)
                if verdict.get("confidence", 0) > best.get("confidence", 0):
                    best = verdict
                    best["evidence"] = ev
            results.append({"claim": claim, **best})
        n_supported = sum(1 for r in results if r.get("supported"))
        hallucination_score = 1.0 - (n_supported / max(len(results), 1))
        return {
            "output": output,
            "n_claims": len(results),
            "n_supported": n_supported,
            "hallucination_score": hallucination_score,
            "claims": results,
        }


__all__ = [
    "LLMCallable",
    "LLMEnhancedMoralCompass",
    "LLMEnhancedReflexion",
    "LLMEnhancedVibeCoder",
    "LLMEnhancedPromptEvolver",
    "LLMEnhancedHallucinationDetector",
]
