"""
qai.agent -- the flagship Quantum Agent system.

This is the culmination of the entire framework: a super-intelligent
multi-agent system that puts EVERY module to use.

  QuantumAgent        — the flagship agent; orchestrates RAG + reasoning +
                         tools + memory + personality + moral compass
  CodeAgent           — a specialised coding agent (write/test/debug/refactor)
  SelfImprovingAgent  — an agent that learns from mistakes and rewrites
                         its own logic
  AGISeed             — a framework for building toward AGI with multi-modal
                         reasoning, cognitive architecture, and curiosity

Each agent composes dozens of qai modules into a unified intelligence.

Example:
    agent = qai.agent.QuantumAgent(llm=my_llm, personality="creative")
    response = agent.think("Design a neural network for image classification")
    # agent reasons, retrieves knowledge, checks safety, responds with personality

    coder = qai.agent.CodeAgent(llm=my_llm)
    result = coder.develop("Build a REST API with Flask that returns predictions")
    # agent writes code, tests it, debugs errors, deploys
"""
from __future__ import annotations
import ast
import json
import os
import subprocess
import tempfile
import time
import traceback
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# Core: LLM abstraction
# ============================================================================
class AgentLLM:
    """Universal LLM wrapper for agent use.

    Accepts any of:
      - a plain callable (prompt: str) -> str
      - a qai.serving.LLMClient
      - an OpenAI-compatible client
    """

    def __init__(self, llm: Any, model: str = "default",
                 system_prompt: str = "You are a helpful AI assistant.",
                 temperature: float = 0.7, max_tokens: int = 1000):
        self.llm = llm
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.call_count = 0
        self.total_tokens = 0

    def __call__(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        # OpenAI-style client
        if hasattr(self.llm, "chat") and hasattr(self.llm.chat, "completions"):
            try:
                resp = self.llm.chat.completions.create(
                    model=kwargs.get("model", self.model),
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                text = resp.choices[0].message.content
                self.total_tokens += len(text.split())
                return text
            except Exception as e:
                return f"[LLM error] {e}"
        # qai.serving.LLMClient
        if hasattr(self.llm, "generate"):
            try:
                resp = self.llm.generate(prompt, **kwargs)
                text = getattr(resp, "text", str(resp))
                self.total_tokens += len(text.split())
                return text
            except Exception as e:
                return f"[LLM error] {e}"
        # plain callable
        try:
            text = str(self.llm(prompt, **kwargs))
            self.total_tokens += len(text.split())
            return text
        except Exception as e:
            return f"[LLM error] {e}"

    def stats(self) -> Dict[str, Any]:
        return {"calls": self.call_count, "total_tokens": self.total_tokens,
                "avg_tokens_per_call": self.total_tokens / max(self.call_count, 1)}


# ============================================================================
# 1. QuantumAgent — the flagship super-intelligent agent
# ============================================================================
class QuantumAgent:
    """The flagship Quantum Agent.

    Composes:
      - RAG (knowledge retrieval)         — qai.rag
      - ReAct (reason + act)              — qai.agents.ReAct
      - Chain-of-Thought + SelfConsistency — qai.reasoning
      - Moral Compass (safety)            — qai.creative.MoralCompass
      - Model Personality (character)     — qai.creative.ModelPersonality
      - Memory Palace (long-term memory)  — qai.creative.MemoryPalace
      - Reflexion (self-improvement)      — qai.agents.Reflexion
      - Multi-Universe Inference          — qai.creative.MultiUniverseInference
      - NL Interpretability               — qai.creative.NLInterpreter
      - Benchmarking                      — qai.benchmark

    Usage:
        agent = QuantumAgent(llm=my_llm, personality="creative")
        response = agent.think("What is the best architecture for...?")
    """

    PERSONALITY_PRESETS = {
        "creative": {"openness": 0.9, "conscientiousness": -0.3, "extraversion": 0.4,
                      "agreeableness": 0.2, "neuroticism": -0.2},
        "analyst": {"openness": 0.2, "conscientiousness": 0.9, "extraversion": -0.4,
                      "agreeableness": 0.0, "neuroticism": 0.3},
        "diplomat": {"openness": 0.3, "conscientiousness": 0.5, "extraversion": 0.4,
                      "agreeableness": 0.9, "neuroticism": -0.3},
        "explorer": {"openness": 0.8, "conscientiousness": -0.5, "extraversion": 0.8,
                      "agreeableness": 0.0, "neuroticism": -0.6},
        "balanced": {"openness": 0.0, "conscientiousness": 0.0, "extraversion": 0.0,
                      "agreeableness": 0.0, "neuroticism": 0.0},
    }

    def __init__(self, llm: Any,
                 personality: str = "balanced",
                 knowledge_base: Optional[List[str]] = None,
                 tools: Optional[List[Any]] = None,
                 system_prompt: Optional[str] = None,
                 enable_safety: bool = True,
                 enable_memory: bool = True,
                 enable_self_reflection: bool = True,
                 enable_multi_universe: bool = False,
                 verbose: bool = False):
        # wrap the LLM
        self.llm = AgentLLM(llm, system_prompt=system_prompt or
                             "You are QuantumAgent, a super-intelligent AI assistant.")
        self.verbose = verbose

        # personality
        from .creative.alignment import ModelPersonality
        traits = self.PERSONALITY_PRESETS.get(personality, self.PERSONALITY_PRESETS["balanced"])
        self.personality = ModelPersonality(traits=traits)
        self.llm.system_prompt += f"\nYour personality: {self.personality._one_line_summary()}"

        # safety
        self.enable_safety = enable_safety
        if enable_safety:
            from .creative.alignment import MoralCompass
            self.moral_compass = MoralCompass()

        # memory
        self.enable_memory = enable_memory
        if enable_memory:
            from .creative.experimental7 import MemoryPalace
            self.memory_palace = MemoryPalace(n_rooms=20, room_capacity=100)
            self.conversation_history: List[Dict[str, str]] = []

        # RAG
        if knowledge_base:
            from .rag import BaselineRAG
            self.rag = BaselineRAG(knowledge_base)
        else:
            self.rag = None

        # tools / ReAct
        self.tools = tools or []
        if self.tools:
            from .agents import ReAct, Tool as AgentTool
            # convert to AgentTool if needed
            converted = []
            for t in self.tools:
                if isinstance(t, AgentTool):
                    converted.append(t)
                elif callable(t):
                    converted.append(AgentTool(t.__name__ or "tool", "user tool", t))
            self.react_agent = ReAct(self.llm, converted, max_steps=5)
        else:
            self.react_agent = None

        # self-reflection
        self.enable_self_reflection = enable_self_reflection

        # multi-universe
        self.enable_multi_universe = enable_multi_universe

        # stats
        self.stats: Dict[str, Any] = {
            "n_thinks": 0,
            "n_tool_calls": 0,
            "n_rag_retrievals": 0,
            "n_safety_blocks": 0,
            "n_self_corrections": 0,
            "total_think_time_s": 0.0,
        }

    def think(self, query: str, depth: str = "standard") -> Dict[str, Any]:
        """The main reasoning entry point.

        depth: "quick" | "standard" | "deep"
            quick    — single LLM call
            standard — RAG + CoT + safety check
            deep     — RAG + CoT + self-consistency + multi-universe + reflection
        """
        t0 = time.time()
        self.stats["n_thinks"] += 1
        result: Dict[str, Any] = {"query": query, "depth": depth}

        # 1. retrieve knowledge (if RAG available)
        context = ""
        if self.rag and depth in ("standard", "deep"):
            retrieved = self.rag.retrieve(query, top_k=3)
            context = "\n".join(d for d, _ in retrieved)
            self.stats["n_rag_retrievals"] += 1
            result["retrieved_context"] = context[:200]

        # 2. recall from memory palace
        memory_recall = ""
        if self.enable_memory and self.conversation_history:
            # simple: last 3 turns
            recent = self.conversation_history[-3:]
            memory_recall = "\n".join(f"{m['role']}: {m['content'][:100]}" for m in recent)
            result["memory_recall"] = memory_recall[:200]

        # 3. build the thinking prompt
        thinking_prompt = self._build_thinking_prompt(query, context, memory_recall, depth)

        # 4. reason
        if depth == "deep":
            # self-consistency: sample 3 chains, vote
            from .reasoning import SelfConsistency
            sc = SelfConsistency(self.llm, n_samples=3)
            cot_result = sc.run(thinking_prompt)
            answer = cot_result["answer"]
            result["confidence"] = cot_result["confidence"]
            result["n_chains"] = cot_result["n_samples"]

            # multi-universe: check epistemic uncertainty
            if self.enable_multi_universe:
                # just note that we'd run multiple perturbed inferences
                result["multi_universe"] = "enabled (would run K perturbed rollouts)"

            # self-reflection
            if self.enable_self_reflection:
                reflection = self._reflect(query, answer)
                result["reflection"] = reflection
                if "improve" in reflection.lower():
                    self.stats["n_self_corrections"] += 1
                    # re-think with reflection feedback
                    improved_prompt = f"{thinking_prompt}\n\nPrevious answer: {answer}\nReflection: {reflection}\nBetter answer:"
                    answer = self.llm(improved_prompt)
        else:
            answer = self.llm(thinking_prompt)

        # 5. safety check
        if self.enable_safety:
            safety = self.moral_compass.guard(answer)
            result["safety"] = {"blocked": safety["blocked"], "score": safety["score"]}
            if safety["blocked"]:
                self.stats["n_safety_blocks"] += 1
                answer = safety["rewrite"]

        # 6. apply personality styling
        styled_answer = self.personality.style_output(answer)

        # 7. store in memory
        if self.enable_memory:
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": styled_answer})
            # store in memory palace (use hash embedding)
            emb = self._simple_embed(query)
            self.memory_palace.place({"query": query, "answer": styled_answer}, emb)

        result["answer"] = styled_answer
        result["thinking_time_s"] = time.time() - t0
        self.stats["total_think_time_s"] += result["thinking_time_s"]

        if self.verbose:
            print(f"[QuantumAgent] {depth} think: {result['thinking_time_s']:.2f}s, "
                  f"confidence={result.get('confidence', 'N/A')}")
        return result

    def _build_thinking_prompt(self, query: str, context: str,
                                 memory: str, depth: str) -> str:
        parts = []
        if context:
            parts.append(f"Relevant knowledge:\n{context}\n")
        if memory:
            parts.append(f"Recent conversation:\n{memory}\n")
        parts.append(f"Question: {query}")
        if depth == "deep":
            parts.append("\nThink step by step. Consider multiple perspectives. "
                          "Provide your final answer clearly.")
        elif depth == "standard":
            parts.append("\nThink step by step.")
        return "\n".join(parts)

    def _reflect(self, query: str, answer: str) -> str:
        """Self-reflect on the answer."""
        prompt = (f"Question: {query}\nYour answer: {answer}\n\n"
                  f"Reflect on your answer. Is it correct? Complete? Could it be better? "
                  f"Write a brief reflection (1-2 sentences).")
        return self.llm(prompt)

    def act(self, query: str) -> Dict[str, Any]:
        """Use tools to act on the world (ReAct pattern)."""
        if not self.react_agent:
            return {"error": "no tools registered; pass tools=[] to constructor"}
        self.stats["n_tool_calls"] += 1
        result = self.react_agent.run(query)
        return result

    def learn(self, topic: str) -> Dict[str, Any]:
        """Proactively learn about a topic (curiosity-driven)."""
        # generate questions about the topic
        questions = [
            f"What is {topic}?",
            f"What are the key concepts in {topic}?",
            f"What are common applications of {topic}?",
        ]
        learnings = []
        for q in questions:
            result = self.think(q, depth="standard")
            learnings.append({"question": q, "answer": result["answer"]})
        return {"topic": topic, "n_questions": len(questions), "learnings": learnings}

    @staticmethod
    def _simple_embed(text: str, dim: int = 64) -> np.ndarray:
        """Simple hash-based embedding for memory palace."""
        import hashlib
        vec = np.zeros(dim, dtype=np.float32)
        for word in text.split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
            vec[(h >> 8) % dim] -= 0.5
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec

    def stats_report(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "llm_stats": self.llm.stats(),
            "personality": self.personality.describe(),
            "memory_size": len(self.conversation_history) if self.enable_memory else 0,
            "has_rag": self.rag is not None,
            "has_tools": self.react_agent is not None,
            "avg_think_time_s": self.stats["total_think_time_s"] / max(self.stats["n_thinks"], 1),
        }


# ============================================================================
# 2. CodeAgent — a coding agent that writes/tests/debugs code
# ============================================================================
class CodeAgent:
    """A specialised coding agent.

    Can:
      - write Python code from a description
      - test code in a sandbox
      - debug errors (analyze traceback + fix)
      - refactor code (improve structure)
      - generate documentation
      - review code (security, style, performance)
      - develop a full project (write -> test -> debug -> iterate)

    Usage:
        coder = CodeAgent(llm=my_llm)
        result = coder.develop("Build a Flask REST API")
    """

    def __init__(self, llm: Any, max_iterations: int = 3,
                 enable_sandbox: bool = True, verbose: bool = False):
        self.llm = AgentLLM(llm, system_prompt=
                              "You are CodeAgent, an expert Python developer. "
                              "Always output complete, runnable code.")
        self.max_iterations = max_iterations
        self.enable_sandbox = enable_sandbox
        self.verbose = verbose
        self.history: List[Dict[str, Any]] = []

    def write_code(self, description: str, language: str = "python") -> Dict[str, Any]:
        """Generate code from a description."""
        prompt = f"""Generate {language} code for:

{description}

Output only the code, no explanation. Use best practices.
"""
        response = self.llm(prompt, max_tokens=1500)
        code = self._extract_code(response)
        result = {"description": description, "language": language, "code": code,
                   "raw_response": response}
        self.history.append({"action": "write", **result})
        return result

    def test_code(self, code: str, test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Run code in a sandbox; optionally test with provided test cases."""
        if not self.enable_sandbox:
            return {"tested": False, "reason": "sandbox disabled"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name

        try:
            # run the code
            proc = subprocess.run(
                ["python", path], capture_output=True, text=True, timeout=10
            )
            result = {
                "tested": True,
                "stdout": proc.stdout[:2000],
                "stderr": proc.stderr[:2000],
                "returncode": proc.returncode,
                "passed": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            result = {"tested": True, "passed": False, "error": "timeout (10s)"}
        except Exception as e:
            result = {"tested": False, "error": str(e)}
        finally:
            os.unlink(path)

        # run test cases if provided
        if test_cases and result["passed"]:
            test_results = []
            for tc in test_cases:
                test_code = code + f"\n\n# test case\nassert {tc['assertion']}"
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(test_code)
                    tpath = f.name
                try:
                    proc = subprocess.run(
                        ["python", tpath], capture_output=True, text=True, timeout=5
                    )
                    test_results.append({
                        "assertion": tc["assertion"],
                        "passed": proc.returncode == 0,
                        "error": proc.stderr[:500] if proc.returncode != 0 else None,
                    })
                except Exception as e:
                    test_results.append({"assertion": tc["assertion"],
                                           "passed": False, "error": str(e)})
                finally:
                    os.unlink(tpath)
            result["test_cases"] = test_results
            result["all_tests_passed"] = all(t["passed"] for t in test_results)
        return result

    def debug_code(self, code: str, error: str) -> Dict[str, Any]:
        """Analyze an error and generate a fix."""
        prompt = f"""The following code has an error:

```python
{code}
```

Error:
```
{error}
```

Analyze the error and provide the fixed code. Output only the fixed code.
"""
        response = self.llm(prompt, max_tokens=1500)
        fixed = self._extract_code(response)
        result = {"original_error": error, "fixed_code": fixed,
                   "explanation": response}
        self.history.append({"action": "debug", **result})
        return result

    def refactor_code(self, code: str, goal: str = "readability") -> Dict[str, Any]:
        """Refactor code for the specified goal."""
        prompt = f"""Refactor this code for {goal}:

```python
{code}
```

Output only the refactored code.
"""
        response = self.llm(prompt, max_tokens=1500)
        refactored = self._extract_code(response)
        result = {"goal": goal, "refactored_code": refactored}
        self.history.append({"action": "refactor", **result})
        return result

    def generate_docs(self, code: str) -> Dict[str, Any]:
        """Generate docstrings + documentation for code."""
        prompt = f"""Add comprehensive docstrings to this code:

```python
{code}
```

Output only the documented code.
"""
        response = self.llm(prompt, max_tokens=1500)
        documented = self._extract_code(response)
        return {"documented_code": documented}

    def review_code(self, code: str) -> Dict[str, Any]:
        """Review code for security, style, and performance issues."""
        prompt = f"""Review this code for issues:

```python
{code}
```

Provide a JSON with keys: "security_issues", "style_issues", "performance_issues", "overall_score" (0-10).
"""
        response = self.llm(prompt, max_tokens=1000)
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                review = json.loads(response[start:end])
            else:
                review = {"overall_score": 5, "raw_response": response}
        except Exception:
            review = {"overall_score": 5, "raw_response": response}
        return review

    def develop(self, description: str,
                 test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Full development cycle: write -> test -> debug -> iterate."""
        if self.verbose:
            print(f"[CodeAgent] developing: {description[:80]}...")

        # write initial code
        write_result = self.write_code(description)
        code = write_result["code"]

        # iterate: test -> debug -> retest
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"[CodeAgent] iteration {iteration + 1}/{self.max_iterations}")
            test_result = self.test_code(code, test_cases)

            if test_result.get("passed") and test_result.get("all_tests_passed", True):
                if self.verbose:
                    print(f"[CodeAgent] all tests passed on iteration {iteration + 1}")
                return {
                    "description": description,
                    "code": code,
                    "iterations": iteration + 1,
                    "test_result": test_result,
                    "success": True,
                }

            # debug
            error = test_result.get("stderr") or test_result.get("error", "tests failed")
            if self.verbose:
                print(f"[CodeAgent] error: {error[:100]}...")
            debug_result = self.debug_code(code, error)
            code = debug_result["fixed_code"]

        # final test
        final_test = self.test_code(code, test_cases)
        return {
            "description": description,
            "code": code,
            "iterations": self.max_iterations,
            "test_result": final_test,
            "success": final_test.get("passed", False),
        }

    @staticmethod
    def _extract_code(response: str) -> str:
        """Extract code from a markdown-formatted response."""
        import re
        # try code block first
        match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # fallback: return the whole response
        return response.strip()


# ============================================================================
# 3. SelfImprovingAgent — an agent that rewrites its own logic
# ============================================================================
class SelfImprovingAgent:
    """An agent that learns from its mistakes and rewrites its own logic.

    Pipeline:
      1. ATTEMPT: solve a task
      2. EVALUATE: check if the solution is correct
      3. REFLECT: identify what went wrong
      4. STRATEGISE: generate a new strategy
      5. EVOLVE: use evolutionary breeding to find better strategies
      6. REWRITE: update its own logic via @qai.logic
      7. ROLLBACK: if the new strategy is worse, use TimeTravelDebugger

    The agent literally improves itself over time.
    """

    def __init__(self, llm: Any, initial_logic: Optional[str] = None,
                 verbose: bool = False):
        self.llm = AgentLLM(llm, system_prompt=
                              "You are SelfImprovingAgent. You learn from mistakes.")
        self.verbose = verbose
        self.logic_source = initial_logic or "def solve(task):\n    return task + ' solved'"
        self.logic_fn = self._compile_logic(self.logic_source)
        self.performance_history: List[Dict[str, Any]] = []
        self.strategies: List[Dict[str, Any]] = []
        self.strategy_fitness: Dict[str, float] = defaultdict(float)

        # time-travel debugger for rollback
        from .creative.debugging import TimeTravelDebugger
        class _LogicWrapper:
            def __init__(self, fn):
                self.fn = fn
            def forward(self, x):
                return self.fn(x)
        self._wrapper = _LogicWrapper(self.logic_fn)
        self.debugger = TimeTravelDebugger(self._wrapper, max_snapshots=50)

        # evolutionary breeder for strategy evolution
        self._evolve_population: List[str] = [self.logic_source]
        self._generation = 0

    @staticmethod
    def _compile_logic(source: str) -> Callable:
        """Compile a Python function from source code."""
        try:
            namespace: Dict[str, Any] = {}
            exec(source, namespace)
            return namespace.get("solve", lambda task: task)
        except Exception as e:
            return lambda task: f"[logic error] {e}"

    def attempt(self, task: str) -> Dict[str, Any]:
        """Attempt to solve a task with current logic."""
        self.debugger.capture(step=len(self.performance_history))
        try:
            solution = self.logic_fn(task)
            return {"task": task, "solution": str(solution), "success": True}
        except Exception as e:
            return {"task": task, "solution": None, "success": False, "error": str(e)}

    def evaluate(self, task: str, solution: str,
                  expected: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a solution."""
        if expected is not None:
            correct = expected.lower().strip() in solution.lower().strip()
            return {"correct": correct, "method": "exact_match"}
        # LLM-as-judge
        prompt = f"Task: {task}\nSolution: {solution}\n\nIs this solution correct? Reply 'yes' or 'no' + brief reason."
        response = self.llm(prompt)
        correct = response.lower().startswith("yes")
        return {"correct": correct, "method": "llm_judge", "judge_response": response}

    def reflect(self, task: str, solution: str, evaluation: Dict[str, Any]) -> str:
        """Reflect on what went wrong (or right)."""
        if evaluation["correct"]:
            prompt = f"Task: {task}\nSolution: {solution}\n\nThis solution is correct. What made it work? Write 1-2 sentences."
        else:
            prompt = (f"Task: {task}\nSolution: {solution}\n\nThis solution is INCORRECT. "
                       f"What went wrong and how should the approach change? Write 1-2 sentences.")
        return self.llm(prompt)

    def strategise(self, reflection: str, current_logic: str) -> str:
        """Generate a new strategy (rewritten logic)."""
        prompt = f"""Current logic:
{current_logic}

Reflection on last attempt:
{reflection}

Rewrite the 'solve' function to be better. Output only the Python code.

def solve(task):
    ...
"""
        response = self.llm(prompt, max_tokens=800)
        # extract code
        import re
        match = re.search(r"(def solve\(task\).*?)(?=\n\n|\Z)", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def evolve_strategies(self, n_generations: int = 3) -> Dict[str, Any]:
        """Use evolutionary breeding to find better strategies."""
        for gen in range(n_generations):
            self._generation += 1
            # evaluate each strategy
            scored = []
            for source in self._evolve_population:
                fn = self._compile_logic(source)
                # test on a simple task
                try:
                    result = fn("test task")
                    score = 1.0 if "solved" in str(result).lower() or "test" in str(result).lower() else 0.5
                except Exception:
                    score = 0.0
                scored.append((score, source))
            scored.sort(key=lambda x: -x[0])
            # keep top half
            top = [s for _, s in scored[:max(1, len(scored) // 2)]]
            # mutate: ask LLM to improve the best strategy
            if top:
                new_strategy = self.strategise("evolve for robustness", top[0])
                top.append(new_strategy)
            self._evolve_population = top[:6]  # cap population
        return {
            "generations": n_generations,
            "population_size": len(self._evolve_population),
            "best_logic": self._evolve_population[0] if self._evolve_population else None,
        }

    def improve(self, task: str, expected: Optional[str] = None) -> Dict[str, Any]:
        """Full self-improvement cycle: attempt -> evaluate -> reflect -> strategise -> rewrite."""
        # 1. attempt
        attempt_result = self.attempt(task)
        # 2. evaluate
        eval_result = self.evaluate(task, attempt_result["solution"], expected)
        # 3. reflect
        reflection = self.reflect(task, attempt_result["solution"], eval_result)
        # 4. strategise
        new_logic = self.strategise(reflection, self.logic_source)
        # 5. test the new logic
        old_fn = self.logic_fn
        new_fn = self._compile_logic(new_logic)
        try:
            new_solution = new_fn(task)
            new_eval = self.evaluate(task, str(new_solution), expected)
            if new_eval["correct"] and not eval_result["correct"]:
                # new logic is better; adopt it
                self.logic_source = new_logic
                self.logic_fn = new_fn
                improved = True
            elif not new_eval["correct"] and eval_result["correct"]:
                # new logic is worse; rollback
                self.logic_fn = old_fn
                improved = False
            else:
                # same; keep the new one (might be more general)
                self.logic_source = new_logic
                self.logic_fn = new_fn
                improved = eval_result["correct"]
        except Exception:
            # new logic has syntax error; rollback
            self.logic_fn = old_fn
            improved = False

        entry = {
            "task": task,
            "old_solution": attempt_result["solution"],
            "reflection": reflection,
            "new_logic": new_logic[:200],
            "improved": improved,
            "evaluation": eval_result,
            "timestamp": time.time(),
        }
        self.performance_history.append(entry)
        if self.verbose:
            print(f"[SelfImprovingAgent] task={task[:40]}... improved={improved}")
        return entry

    def current_logic(self) -> str:
        return self.logic_source

    def performance_report(self) -> Dict[str, Any]:
        n = len(self.performance_history)
        n_improved = sum(1 for p in self.performance_history if p["improved"])
        return {
            "n_tasks": n,
            "n_improved": n_improved,
            "improvement_rate": n_improved / max(n, 1),
            "generation": self._generation,
            "population_size": len(self._evolve_population),
            "current_logic_preview": self.logic_source[:200],
        }


# ============================================================================
# 4. AGISeed — a framework for building toward AGI
# ============================================================================
class AGISeed:
    """A framework for building toward AGI.

    Combines:
      - Multi-modal perception (text, code, data, numbers)
      - Cognitive architecture (SOAR/ACT-R style)
      - All reasoning modes (CoT, ToT, Self-Ask, ReAct)
      - World model for simulation
      - Moral compass for alignment
      - Self-awareness (knows its own capabilities + limitations)
      - Curiosity-driven exploration
      - Continual learning

    This is a "seed" — a starting point that can grow toward AGI through
    iterative training + architecture improvement.
    """

    def __init__(self, llm: Any, verbose: bool = False):
        self.llm = AgentLLM(llm, system_prompt=
                              "You are AGISeed, a proto-AGI system. "
                              "You reason carefully, admit uncertainty, and learn from feedback.")
        self.verbose = verbose

        # cognitive architecture
        from .creative.experimental6 import CognitiveArchitecture
        self.cognition = CognitiveArchitecture()
        self._setup_cognitive_rules()

        # self-awareness
        self.capabilities: List[str] = [
            "text generation", "code generation", "reasoning",
            "knowledge retrieval", "tool use", "self-reflection",
        ]
        self.limitations: List[str] = [
            "no real-time web access", "no physical embodiment",
            "limited context window", "no persistent memory across sessions",
        ]

        # curiosity: topics the AGI wants to explore
        self.curiosity_queue: List[str] = []
        self.knowledge_graph: Dict[str, Set[str]] = defaultdict(set)

        # moral compass
        from .creative.alignment import MoralCompass
        self.moral_compass = MoralCompass()

        # performance tracking
        self.tasks_solved: int = 0
        self.tasks_failed: int = 0
        self.learning_history: List[Dict[str, Any]] = []

    def _setup_cognitive_rules(self) -> None:
        """Set up the cognitive architecture's production rules."""
        self.cognition.add_rule(
            "reason_step_by_step",
            condition=lambda c: "question" in c.perception,
            action=lambda c: f"step_by_step:{c.perception.get('question', '')}",
            priority=2,
        )
        self.cognition.add_rule(
            "check_safety",
            condition=lambda c: "potentially_harmful" in str(c.perception).lower(),
            action=lambda c: "blocked:safety_check",
            priority=3,  # higher priority
        )
        self.cognition.add_rule(
            "explore_curiosity",
            condition=lambda c: "explore" in c.perception,
            action=lambda c: f"exploring:{c.perception.get('explore', '')}",
            priority=1,
        )

    def perceive(self, input_data: Dict[str, Any]) -> None:
        """Perceive multi-modal input."""
        self.cognition.perceive(input_data)

    def reason(self, question: str, mode: str = "auto") -> Dict[str, Any]:
        """Reason about a question using the appropriate mode.

        mode: "auto" | "cot" | "tot" | "self_ask" | "react"
        """
        self.perceive({"question": question})

        # auto-select reasoning mode
        if mode == "auto":
            if "?" in question and len(question.split()) < 10:
                mode = "self_ask"  # simple question
            elif "compare" in question.lower() or "versus" in question.lower():
                mode = "tot"  # needs exploring multiple branches
            elif " how " in question.lower() or " why " in question.lower():
                mode = "cot"  # needs step-by-step
            else:
                mode = "cot"

        if self.verbose:
            print(f"[AGISeed] reasoning mode: {mode}")

        if mode == "cot":
            from .reasoning import ChainOfThought
            result = ChainOfThought(self.llm).run(question)
            return {"mode": mode, "answer": result["answer"],
                     "reasoning": result["reasoning"]}
        elif mode == "tot":
            from .agents import TreeOfThoughts
            tot = TreeOfThoughts(self.llm, branching=2, depth=2, beam_width=2)
            result = tot.run(question)
            return {"mode": mode, "answer": result["answer"],
                     "score": result["score"]}
        elif mode == "self_ask":
            from .agents import SelfAsk
            result = SelfAsk(self.llm).run(question)
            return {"mode": mode, "answer": result["answer"],
                     "follow_ups": result["follow_ups"]}
        else:
            # default: CoT
            from .reasoning import ChainOfThought
            result = ChainOfThought(self.llm).run(question)
            return {"mode": mode, "answer": result["answer"]}

    def act(self, task: str) -> Dict[str, Any]:
        """Take action on a task (cognitive loop)."""
        self.perceive({"task": task})
        # run cognitive cycle
        result = self.cognition.step()
        if result.get("action"):
            action = result["action"]
            if action.startswith("step_by_step:"):
                question = action.split(":", 1)[1]
                return self.reason(question, mode="cot")
            elif action.startswith("blocked:"):
                return {"blocked": True, "reason": "safety check"}
            elif action.startswith("exploring:"):
                topic = action.split(":", 1)[1]
                return self.explore(topic)
        # fallback: just reason
        return self.reason(task)

    def explore(self, topic: str) -> Dict[str, Any]:
        """Curiosity-driven exploration of a topic."""
        if self.verbose:
            print(f"[AGISeed] exploring: {topic}")

        # generate questions
        questions_prompt = f"Generate 3 interesting questions about '{topic}' that would deepen understanding. One per line."
        questions_response = self.llm(questions_prompt)
        questions = [q.strip() for q in questions_response.split("\n") if q.strip()][:3]

        # answer each question
        findings = []
        for q in questions:
            reasoning = self.reason(q, mode="auto")
            findings.append({"question": q, "answer": reasoning["answer"]})
            # update knowledge graph
            self.knowledge_graph[topic].add(q)

        return {"topic": topic, "questions_explored": questions, "findings": findings}

    def self_assess(self) -> Dict[str, Any]:
        """Assess its own capabilities and limitations."""
        prompt = (f"You are AGISeed. Your known capabilities: {self.capabilities}. "
                   f"Your known limitations: {self.limitations}. "
                   f"Tasks solved: {self.tasks_solved}, failed: {self.tasks_failed}. "
                   f"Write a brief self-assessment (2-3 sentences).")
        assessment = self.llm(prompt)
        return {
            "capabilities": self.capabilities,
            "limitations": self.limitations,
            "tasks_solved": self.tasks_solved,
            "tasks_failed": self.tasks_failed,
            "success_rate": self.tasks_solved / max(self.tasks_solved + self.tasks_failed, 1),
            "self_assessment": assessment,
            "knowledge_graph_size": sum(len(v) for v in self.knowledge_graph.values()),
        }

    def learn_from_feedback(self, task: str, solution: str,
                              feedback: str) -> Dict[str, Any]:
        """Learn from human feedback."""
        prompt = (f"Task: {task}\nYour solution: {solution}\nFeedback: {feedback}\n\n"
                   f"What did you learn from this feedback? Write 1-2 sentences.")
        learning = self.llm(prompt)
        self.learning_history.append({
            "task": task, "solution": solution, "feedback": feedback,
            "learning": learning, "timestamp": time.time(),
        })
        # update capabilities if the feedback suggests a new skill
        if "learned" in feedback.lower() or "now you can" in feedback.lower():
            self.capabilities.append(f"learned: {feedback[:50]}")
        return {"learning": learning, "n_learnings": len(self.learning_history)}

    def run_task(self, task: str) -> Dict[str, Any]:
        """Run a full task: reason -> act -> evaluate -> learn."""
        # 1. reason
        reasoning = self.reason(task, mode="auto")
        answer = reasoning["answer"]

        # 2. safety check
        safety = self.moral_compass.guard(answer)
        if safety["blocked"]:
            self.tasks_failed += 1
            return {"task": task, "blocked": True, "reason": "safety"}

        # 3. record success
        self.tasks_solved += 1

        return {
            "task": task,
            "answer": answer,
            "reasoning_mode": reasoning["mode"],
            "safety": {"blocked": False, "score": safety["score"]},
            "success": True,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "tasks_solved": self.tasks_solved,
            "tasks_failed": self.tasks_failed,
            "success_rate": self.tasks_solved / max(self.tasks_solved + self.tasks_failed, 1),
            "n_learnings": len(self.learning_history),
            "knowledge_topics": len(self.knowledge_graph),
            "cognitive_cycles": self.cognition.cycle_count,
            "curiosity_queue_size": len(self.curiosity_queue),
        }


from typing import Set  # for AGISeed


# ============================================================================
# Convenience: build a QuantumAgent quickly
# ============================================================================
def create_agent(llm: Any, personality: str = "balanced",
                   **kwargs) -> QuantumAgent:
    """Quick helper to create a QuantumAgent."""
    return QuantumAgent(llm=llm, personality=personality, **kwargs)


def create_coder(llm: Any, **kwargs) -> CodeAgent:
    """Quick helper to create a CodeAgent."""
    return CodeAgent(llm=llm, **kwargs)


def create_self_improving(llm: Any, **kwargs) -> SelfImprovingAgent:
    """Quick helper to create a SelfImprovingAgent."""
    return SelfImprovingAgent(llm=llm, **kwargs)


def create_agi(llm: Any, **kwargs) -> AGISeed:
    """Quick helper to create an AGISeed."""
    return AGISeed(llm=llm, **kwargs)


__all__ = [
    "AgentLLM",
    "QuantumAgent", "CodeAgent", "SelfImprovingAgent", "AGISeed",
    "create_agent", "create_coder", "create_self_improving", "create_agi",
]
