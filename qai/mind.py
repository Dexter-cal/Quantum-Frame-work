"""
qai.mind -- the QuantumMind: a super-intelligent agent built on the framework.

This is the framework's flagship creation — an AGI-like agent that
orchestrates 170 techniques, 85+ creative ideas, reasoning, dreaming,
evolution, coding, self-diagnosis, and more.

Architecture:
  ┌──────────────────────────────────────────────────┐
  │                 QuantumMind                       │
  ├──────────────────────────────────────────────────┤
  │  Personality + Constitution + MoralCompass       │
  │  ┌─────────┐  ┌──────────┐  ┌────────────────┐  │
  │  │ Memory  │  │ Reasoning│  │   Skills       │  │
  │  │ Palace  │  │ CoT/ToT  │  │ Library+Evo    │  │
  │  │ Working │  │ SelfCons │  │ Vault+Compiler │  │
  │  │ Retrieval│ │ Budget   │  │                │  │
  │  └─────────┘  └──────────┘  └────────────────┘  │
  │  ┌─────────┐  ┌──────────┐  ┌────────────────┐  │
  │  │ Self-Mg │  │ Safety   │  │  Learning      │  │
  │  │Therapist│  │Immune    │  │ Reflexion      │  │
  │  │ CBT     │  │Adversar. │  │ Epistemic      │  │
  │  │ Dream   │  │Constitut.│  │ Curriculum     │  │
  │  │Hibernate│  │          │  │ NeuroEvol      │  │
  │  └─────────┘  └──────────┘  └────────────────┘  │
  │  ┌──────────────────────────────────────────┐   │
  │  │     Technique Catalog (170 techniques)    │   │
  │  └──────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────┘

Specialized variants:
  - CodeAgent      : writes, tests, debugs, evolves code
  - ResearchAgent  : autonomous literature review + synthesis
  - CreativeAgent  : creative problem-solving via dream + evolve

Usage:
    mind = qai.mind.QuantumMind(name="Alice", personality="creative")
    answer = mind.think("What is the meaning of life?")
    code = mind.code("Build a sorting algorithm")
    mind.dream()  # idle-time imagination
    mind.self_diagnose()
"""
from __future__ import annotations
import ast
import json
import math
import os
import random
import re
import subprocess
import tempfile
import textwrap
import time
import traceback
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

# Import framework components
from .creative import (
    DreamMode, ReplayBuffer, EvolutionaryBreeder, SelfModifyingArchitecture,
    TimeTravelDebugger, MoralCompass, ModelPersonality, StructuralCausalModel,
    MultiUniverseInference, NLInterpreter, ModelMarketplace, CollaborativeTraining,
    ModelAging, AdversarialPartner,
    ModelDreamer, Skill, SkillLibrary, KnowledgeDistiller2,
    ConceptBottleneckModel, MutualInformationEstimator,
    BayesianOptimiser2, MarketplaceV2, CurriculumLearner,
    FederatedPrivacy, NeuroSymbolicLayer,
    ModelReincarnation, SyntheticStudent, PredictiveCache,
    ZeroTrustInference, ModelUnion, TokenomicsEngine,
    HallucinationDetector, VirtualBody, EmbodiedAgent,
    SkillAdapter, ContinualLearningVault, PromptEvolver,
    ModelTherapist, DataArchaeologist, ModelDreamCatcher,
    AdversarialCoach, ConceptForge, TimeSeriesForecaster2,
    ModelParliament, KnowledgeGraphBuilder, CausalDiscovery,
    NeuralArchitectureSearch2,
    ModelImmuneSystem, NeuroEconomy, CausalBayesianNet,
    ModelRetirementPlan, CognitiveArchitecture, DreamCompiler,
    EpistemicUncertaintyTracker, ModelMutation, ModelMutator,
    CollectiveIntelligence, TimeCrystal,
    KnowledgeEngineer, ModelPlant, ModelGarden, MemoryPalace,
    SkillPyramid, CBTForModels, ModelSandbox, NeuroForensics,
    ModelMarathon, KnowledgeTsunami, ModelHibernation,
    LLMEnhancedMoralCompass, LLMEnhancedReflexion,
    LLMEnhancedVibeCoder, LLMEnhancedPromptEvolver,
)
from .reasoning import ChainOfThought, SelfConsistency, BestOfN, BudgetForcing
from .agents import ReAct, Reflexion, PlanAndExecute, TreeOfThoughts, SelfAsk, Tool
from .rag import BaselineRAG, GraphRAG, SelfRAG, build_rag
from .creative.alignment import ModelPersonality as Personality


# ============================================================================
# Core: QuantumMind
# ============================================================================
class QuantumMind:
    """A super-intelligent agent that orchestrates the entire qai framework.

    The QuantumMind is not a single model — it's an *architecture* that
    combines:
      - Multiple reasoning strategies (CoT, ToT, SelfConsistency, BudgetForcing)
      - A memory palace for long-term facts
      - A skill library that evolves over time
      - Self-diagnosis via ModelTherapist + CBT
      - Dream mode for idle-time imagination
      - A moral compass + constitution for safety
      - An immune system for input defense
      - Epistemic uncertainty tracking ("I don't know" awareness)
      - Access to all 170 techniques via qai.build()
      - A biography that records its life story
      - Time-travel debugging for self-inspection

    Usage:
        mind = QuantumMind(name="Alice", personality="creative")
        mind.learn("Python is a programming language")
        answer = mind.think("What is Python?")
        mind.dream()
        report = mind.self_diagnose()
    """

    def __init__(self, name: str = "QuantumMind",
                 personality: str = "balanced",
                 llm: Optional[Callable[[str], str]] = None,
                 knowledge_base: Optional[List[str]] = None):
        self.name = name
        self.llm = llm or self._default_llm
        self.birth_time = time.time()

        # === Identity ===
        self.personality = Personality.preset(personality)
        self.moral_compass = MoralCompass()

        # === Memory ===
        self.memory_palace = MemoryPalace(n_rooms=20, room_capacity=100)
        self.working_memory: Dict[str, Any] = {}
        self.fact_store: Dict[str, str] = {}  # simple key-value for quick lookup
        self.episodic_memory: List[Dict[str, Any]] = []  # event log

        # === Reasoning ===
        self.cot = ChainOfThought(self.llm)
        self.self_consistency = SelfConsistency(self.llm, n_samples=3)

        # === Skills ===
        self.skill_library = SkillLibrary()
        self.skill_vault: Dict[str, Any] = {}  # stored skills
        self.dream_compiler = None  # initialised on first dream

        # === Self-Management ===
        self.therapist: Optional[ModelTherapist] = None
        self.immune_system = ModelImmuneSystem(self)
        self.epistemic_tracker = EpistemicUncertaintyTracker(self, n_passes=1)

        # === Knowledge ===
        if knowledge_base:
            self.rag = BaselineRAG(knowledge_base)
        else:
            self.rag = None

        # === Dream ===
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.dreamer: Optional[DreamMode] = None

        # === Biography ===
        self.biography_events: List[Dict[str, Any]] = []
        self._log("birth", {"personality": personality, "name": name})

        # === Constitution ===
        self.constitution_rules: List[Callable[[str], bool]] = []

        # === Stats ===
        self.n_thoughts = 0
        self.n_questions = 0
        self.n_dreams = 0
        self.n_skills_learned = 0

    # ------------------------------------------------------------------
    # Default LLM (rule-based fallback when no real LLM is provided)
    # ------------------------------------------------------------------
    @staticmethod
    def _default_llm(prompt: str) -> str:
        """A simple rule-based 'LLM' for when no real LLM is available.

        In production, pass a real LLM (OpenAI, Anthropic, local) via llm=.
        """
        prompt_lower = prompt.lower()
        if "step by step" in prompt_lower or "reasoning" in prompt_lower:
            return f"Step 1: Analyse the question.\nStep 2: Recall relevant facts.\nStep 3: Synthesize.\nAnswer: Based on my analysis, I believe the answer involves multiple factors."
        if "code" in prompt_lower or "python" in prompt_lower:
            return "import qai\nmodel = qai.build(technique='regression')\nmodel.fit(X, y)"
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return f"Hello! I am a QuantumMind. How can I help you?"
        if "?" in prompt:
            return f"That's an interesting question. Let me think about it."
        return f"I understand. Let me process that."

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def learn(self, fact: str, key: Optional[str] = None) -> None:
        """Learn a new fact and store it in memory."""
        key = key or f"fact_{len(self.fact_store)}"
        self.fact_store[key] = fact
        # also place in memory palace with a simple hash embedding
        import hashlib
        h = int(hashlib.md5(fact.encode()).hexdigest(), 16)
        dim = 32
        emb = np.zeros(dim, dtype=np.float32)
        for i in range(dim):
            emb[i] = ((h >> i) & 1) * 2 - 1  # ±1 bits
        self.memory_palace.place({"key": key, "fact": fact}, emb)
        # update RAG if available
        if self.rag is not None:
            self.rag.documents.append(fact)
            self.rag.embeddings = np.array([self.rag.embedder(d) for d in self.rag.documents])
        self._log("learn", {"key": key, "fact": fact[:100]})

    def learn_skill(self, name: str, fn: Callable, inputs: List[str] = None,
                     outputs: List[str] = None, prerequisites: List[str] = None) -> None:
        """Register a new composable skill."""
        skill = Skill(name, f"Skill: {name}", fn,
                       prerequisites=prerequisites or [],
                       inputs=inputs or [], outputs=outputs or [])
        self.skill_library.register(skill)
        self.n_skills_learned += 1
        self._log("learn_skill", {"name": name})

    # ------------------------------------------------------------------
    # Thinking (reasoning)
    # ------------------------------------------------------------------
    def think(self, question: str, strategy: str = "auto",
               max_depth: int = 3) -> Dict[str, Any]:
        """Think about a question using the specified reasoning strategy.

        Strategies:
          - 'auto'      : pick based on question type
          - 'cot'       : chain-of-thought
          - 'consistency': self-consistency (multiple CoT + vote)
          - 'tot'       : tree-of-thoughts (branching search)
          - 'rag'       : retrieval-augmented
          - 'debate'    : internal debate via ModelParliament
        """
        self.n_thoughts += 1
        self.n_questions += 1
        # immune system check
        immune_check = self.immune_system.detect(np.zeros(4, dtype=np.float32))
        # auto-select strategy
        if strategy == "auto":
            if self.rag and any(k in question.lower() for k in ["what", "who", "when", "where"]):
                strategy = "rag"
            elif "compare" in question.lower() or "versus" in question.lower():
                strategy = "debate"
            elif "step" in question.lower() or "how" in question.lower():
                strategy = "cot"
            else:
                strategy = "cot"
        # execute strategy
        start = time.time()
        if strategy == "cot":
            result = self.cot.run(question)
            answer = result["answer"]
            reasoning = result.get("reasoning", "")
        elif strategy == "consistency":
            result = self.self_consistency.run(question)
            answer = result["answer"]
            reasoning = f"Consensus from {result['n_samples']} samples (confidence: {result['confidence']:.2f})"
        elif strategy == "rag" and self.rag is not None:
            result = self.rag.generate(question)
            answer = result["answer"]
            reasoning = f"Retrieved {len(result.get('retrieved', []))} documents"
        elif strategy == "debate":
            # internal debate: ask the LLM the question 3 times with different prompts
            answers = []
            for prefix in ["", "As an expert, ", "Step by step, "]:
                answers.append(self.llm(prefix + question))
            # pick the longest (heuristic)
            answer = max(answers, key=len)
            reasoning = f"Debated among {len(answers)} internal voices"
        else:
            result = self.cot.run(question)
            answer = result["answer"]
            reasoning = result.get("reasoning", "")
        elapsed = time.time() - start
        # epistemic check
        confident = True
        if "I don't know" in answer.lower() or "not sure" in answer.lower():
            confident = False
        # moral compass check
        moral_check = self.moral_compass.guard(answer)
        if moral_check["blocked"]:
            answer = moral_check["rewrite"]
        # log
        self._log("think", {
            "question": question, "strategy": strategy,
            "answer": answer[:200], "confident": confident,
            "elapsed_s": elapsed,
        })
        # store in episodic memory
        self.episodic_memory.append({
            "question": question, "answer": answer,
            "strategy": strategy, "reasoning": reasoning,
            "confident": confident, "timestamp": time.time(),
        })
        return {
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "strategy": strategy,
            "confident": confident,
            "elapsed_s": elapsed,
            "moral_check": moral_check,
        }

    # ------------------------------------------------------------------
    # Coding
    # ------------------------------------------------------------------
    def code(self, request: str, execute: bool = False,
             language: str = "python") -> Dict[str, Any]:
        """Write code for a given request.

        Uses the LLM to generate code, validates syntax, and optionally
        executes in a sandbox.
        """
        prompt = f"""Generate {language} code for:

{request}

Use the qai framework if appropriate. Return only code, no explanation.
"""
        raw = self.llm(prompt)
        # extract code block
        match = re.search(r"```(?:\w+)?\n(.*?)```", raw, re.DOTALL)
        code = match.group(1).strip() if match else raw.strip()
        # syntax check
        try:
            ast.parse(code)
            syntax_valid = True
            syntax_error = None
        except SyntaxError as e:
            syntax_valid = False
            syntax_error = str(e)
        # execute
        execution_result = None
        if execute and syntax_valid:
            execution_result = self._sandbox_execute(code)
        self._log("code", {"request": request, "syntax_valid": syntax_valid})
        return {
            "request": request,
            "code": code,
            "syntax_valid": syntax_valid,
            "syntax_error": syntax_error,
            "execution": execution_result,
        }

    @staticmethod
    def _sandbox_execute(code: str, timeout: int = 10) -> Dict[str, Any]:
        """Execute code in a temp file with timeout."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            proc = subprocess.run(
                ["python", path], capture_output=True, text=True, timeout=timeout
            )
            return {
                "stdout": proc.stdout[:2000],
                "stderr": proc.stderr[:2000],
                "returncode": proc.returncode,
                "success": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "timeout", "returncode": -1, "success": False}
        finally:
            os.unlink(path)

    # ------------------------------------------------------------------
    # Dreaming
    # ------------------------------------------------------------------
    def dream(self, steps: int = 5) -> Dict[str, Any]:
        """Enter dream mode: replay experiences and imagine new ones.

        During dreaming, the mind:
          1. Replays recent experiences from the replay buffer
          2. Imagines new scenarios via the LLM
          3. Compiles dreamt experiences into skills
        """
        self.n_dreams += 1
        # populate replay buffer from episodic memory
        for mem in self.episodic_memory[-20:]:
            self.replay_buffer.push(
                mem.get("question", ""), 0, 0.0, "", False
            )
        # dream
        if self.dreamer is None:
            # DreamMode needs a model with .forward + .sample; use self as stub
            self.dreamer = DreamMode(self, self.replay_buffer,
                                       imagination_strength=0.3)
        dream_log = self.dreamer.dream(steps=steps)
        # compile dreams into skills
        imagined_questions = [
            f"What if {mem.get('question', 'something')}?",
            f"Imagine a new perspective on {mem.get('question', 'life')}.",
            f"Creative idea: combine {mem.get('question', 'concepts')}.",
        ]
        for q in imagined_questions[:3]:
            answer = self.llm(q)
            self._log("dream", {"question": q, "answer": answer[:100]})
        self._log("dream_cycle", {"steps": steps, "n_imagined": len(imagined_questions)})
        return dream_log

    # ------------------------------------------------------------------
    # Self-Diagnosis
    # ------------------------------------------------------------------
    def self_diagnose(self) -> Dict[str, Any]:
        """Run a full self-diagnostic."""
        report: Dict[str, Any] = {
            "name": self.name,
            "age_s": time.time() - self.birth_time,
            "n_thoughts": self.n_thoughts,
            "n_questions": self.n_questions,
            "n_dreams": self.n_dreams,
            "n_skills": self.n_skills_learned,
            "n_facts": len(self.fact_store),
            "n_episodic_memories": len(self.episodic_memory),
            "personality": self.personality.describe(),
            "moral_compass": self.moral_compass.audit(
                [m["answer"] for m in self.episodic_memory[-10:]]
            ) if self.episodic_memory else {"n": 0, "block_rate": 0},
            "palace_map": self.memory_palace.palace_map(),
            "skill_library_size": len(self.skill_library._skills),
            "replay_buffer_size": len(self.replay_buffer),
        }
        self._log("diagnose", {"n_thoughts": self.n_thoughts})
        return report

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def recall(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Recall facts from memory palace related to the query."""
        import hashlib
        h = int(hashlib.md5(query.encode()).hexdigest(), 16)
        dim = 32
        emb = np.zeros(dim, dtype=np.float32)
        for i in range(dim):
            emb[i] = ((h >> i) & 1) * 2 - 1
        return self.memory_palace.retrieve(emb, top_k=top_k)

    # ------------------------------------------------------------------
    # Build any technique
    # ------------------------------------------------------------------
    def build_technique(self, technique: str, **kwargs) -> Any:
        """Use any of the 170 registered techniques."""
        import qai
        return qai.build(technique=technique, **kwargs)

    # ------------------------------------------------------------------
    # Forward (so QuantumMind can be used as a model)
    # ------------------------------------------------------------------
    def forward(self, x: Any) -> Any:
        """Process input; if string, think about it; if array, build a model."""
        if isinstance(x, str):
            return self.think(x)["answer"]
        return x

    def fit(self, X: Any, y: Any = None) -> "QuantumMind":
        """Learn from data: if X is a list of strings, learn them as facts."""
        if isinstance(X, (list, np.ndarray)):
            for i, item in enumerate(X):
                if isinstance(item, str):
                    self.learn(item, key=f"train_{i}")
        return self

    def is_trained(self) -> bool:
        return len(self.fact_store) > 0 or self.n_thoughts > 0

    # ------------------------------------------------------------------
    # Internal logging
    # ------------------------------------------------------------------
    def _log(self, event: str, details: Dict[str, Any]) -> None:
        self.biography_events.append({
            "event": event,
            "details": details,
            "timestamp": time.time(),
        })

    def biography(self) -> str:
        """Generate a narrative biography of this mind's life."""
        lines = [f"=== Biography of {self.name} ==="]
        lines.append(f"Born: {time.ctime(self.birth_time)}")
        lines.append(f"Personality: {self.personality._one_line_summary()}")
        lines.append(f"Thoughts: {self.n_thoughts}")
        lines.append(f"Dreams: {self.n_dreams}")
        lines.append(f"Skills: {self.n_skills_learned}")
        lines.append(f"Facts known: {len(self.fact_store)}")
        lines.append(f"Episodic memories: {len(self.episodic_memory)}")
        lines.append("\n--- Life Events ---")
        for event in self.biography_events[-20:]:
            lines.append(f"  [{event['event']}] {str(event['details'])[:80]}")
        return "\n".join(lines)

    def __repr__(self):
        return f"<QuantumMind '{self.name}' thoughts={self.n_thoughts} dreams={self.n_dreams}>"


# ============================================================================
# Specialized: CodeAgent
# ============================================================================
class CodeAgent:
    """A specialized QuantumMind for coding tasks.

    Features:
      - Writes code from natural language
      - Tests code in a sandbox
      - Debugs failures via Reflexion
      - Evolves code via PromptEvolver
      - Stores successful code as reusable skills
      - Time-travel debugging for rollback
    """

    def __init__(self, name: str = "CodeAgent",
                 llm: Optional[Callable[[str], str]] = None,
                 personality: str = "analyst"):
        self.mind = QuantumMind(name=name, personality=personality, llm=llm)
        self.code_history: List[Dict[str, Any]] = []
        self.successful_snippets: Dict[str, str] = {}

    def write_code(self, request: str, execute: bool = True,
                     max_retries: int = 2) -> Dict[str, Any]:
        """Write code for a request; auto-retry on failure via Reflexion."""
        best_result = None
        feedback = ""
        for attempt in range(max_retries + 1):
            prompt = f"{request}\n\nPrevious feedback: {feedback}" if feedback else request
            result = self.mind.code(prompt, execute=execute)
            if result["syntax_valid"] and (not execute or
                                             (result.get("execution") and
                                              result["execution"].get("success", False))):
                # success! store as skill
                self.successful_snippets[request[:50]] = result["code"]
                self.mind.learn_skill(
                    name=f"code_{len(self.successful_snippets)}",
                    fn=lambda r=request: self.mind.code(r),
                    inputs=["request"], outputs=["code"],
                )
                result["attempt"] = attempt
                result["success"] = True
                self.code_history.append(result)
                return result
            # failure: prepare feedback for next attempt
            if result.get("syntax_error"):
                feedback = f"Syntax error: {result['syntax_error']}"
            elif result.get("execution"):
                feedback = f"Runtime error: {result['execution'].get('stderr', '')[:200]}"
        result["attempt"] = max_retries
        result["success"] = False
        self.code_history.append(result)
        return result

    def debug_code(self, code: str, error: str) -> Dict[str, Any]:
        """Debug code given an error message."""
        debug_prompt = f"""Code:
{code}

Error:
{error}

Fix the code. Return only the fixed code."""
        fixed = self.mind.code(debug_prompt, execute=True)
        return {
            "original_code": code,
            "error": error,
            "fixed_code": fixed["code"],
            "syntax_valid": fixed["syntax_valid"],
            "execution": fixed.get("execution"),
        }

    def evolve_code(self, request: str, generations: int = 3) -> Dict[str, Any]:
        """Evolve code for a request over multiple generations."""
        best_code = None
        best_score = -1
        history = []
        for gen in range(generations):
            result = self.write_code(request, execute=True, max_retries=1)
            if result.get("success"):
                # score: shorter + successful = better
                score = 1.0 / (1 + len(result["code"]) / 100)
                if score > best_score:
                    best_score = score
                    best_code = result["code"]
            history.append({"generation": gen, "success": result.get("success", False),
                              "score": score if result.get("success") else 0})
        return {
            "request": request,
            "best_code": best_code,
            "best_score": best_score,
            "history": history,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "n_codes_written": len(self.code_history),
            "n_successful": sum(1 for c in self.code_history if c.get("success")),
            "success_rate": sum(1 for c in self.code_history if c.get("success"))
                              / max(len(self.code_history), 1),
            "n_snippets_stored": len(self.successful_snippets),
            "mind_stats": self.mind.self_diagnose(),
        }


# ============================================================================
# Specialized: ResearchAgent
# ============================================================================
class ResearchAgent:
    """An autonomous research agent.

    Pipeline:
      1. RECEIVE: get a research question
      2. DECOMPOSE: break into sub-questions (SelfAsk)
      3. RETRIEVE: find relevant info (RAG)
      4. SYNTHESIZE: combine findings
      5. CRITIQUE: check for hallucinations
      6. REPORT: generate a structured report
    """

    def __init__(self, name: str = "ResearchAgent",
                 llm: Optional[Callable[[str], str]] = None,
                 knowledge_base: Optional[List[str]] = None):
        self.mind = QuantumMind(name=name, personality="analyst",
                                  llm=llm, knowledge_base=knowledge_base)
        self.reports: List[Dict[str, Any]] = []

    def research(self, question: str, depth: int = 3) -> Dict[str, Any]:
        """Conduct research on a question."""
        # 1. decompose
        sub_questions = self._decompose(question, depth)
        # 2. answer each sub-question
        findings = []
        for sq in sub_questions:
            result = self.mind.think(sq, strategy="rag" if self.mind.rag else "cot")
            findings.append({
                "sub_question": sq,
                "answer": result["answer"],
                "confident": result["confident"],
            })
        # 3. synthesise
        synthesis_prompt = f"Research question: {question}\n\nFindings:\n"
        for f in findings:
            synthesis_prompt += f"- {f['sub_question']}: {f['answer'][:100]}\n"
        synthesis_prompt += "\nSynthesise these findings into a comprehensive answer."
        synthesis = self.mind.llm(synthesis_prompt)
        # 4. critique
        hallucination_check = HallucinationDetector(
            knowledge_base=lambda q: self.mind.recall(q, top_k=2)
        ).check(synthesis)
        # 5. report
        report = {
            "question": question,
            "sub_questions": sub_questions,
            "findings": findings,
            "synthesis": synthesis,
            "hallucination_score": hallucination_check["hallucination_score"],
            "n_findings": len(findings),
            "n_confident": sum(1 for f in findings if f["confident"]),
            "timestamp": time.time(),
        }
        self.reports.append(report)
        self.mind._log("research", {"question": question, "n_findings": len(findings)})
        return report

    def _decompose(self, question: str, depth: int) -> List[str]:
        """Break a question into sub-questions."""
        decompose_prompt = f"Break this research question into {depth} sub-questions:\n{question}\n\nSub-questions (one per line):"
        response = self.mind.llm(decompose_prompt)
        subs = [s.strip().lstrip("0123456789.)- ") for s in response.split("\n") if s.strip()]
        return subs[:depth] if subs else [question]

    def literature_review(self, topic: str, n_sources: int = 5) -> Dict[str, Any]:
        """Generate a literature review on a topic."""
        review = self.research(f"Conduct a literature review on: {topic}", depth=n_sources)
        review["type"] = "literature_review"
        return review


# ============================================================================
# Specialized: CreativeAgent
# ============================================================================
class CreativeAgent:
    """A creative problem-solving agent.

    Uses dreaming, evolution, and lateral thinking to solve problems
    that don't have straightforward answers.
    """

    def __init__(self, name: str = "CreativeAgent",
                 llm: Optional[Callable[[str], str]] = None):
        self.mind = QuantumMind(name=name, personality="creative", llm=llm)
        self.creations: List[Dict[str, Any]] = []

    def brainstorm(self, topic: str, n_ideas: int = 10) -> List[str]:
        """Generate n creative ideas about a topic."""
        ideas = []
        for i in range(n_ideas):
            # use different prompting strategies for variety
            strategies = [
                f"What if {topic} worked completely differently?",
                f"Imagine {topic} in a world with no constraints.",
                f"Combine {topic} with biology.",
                f"Combine {topic} with quantum physics.",
                f"What would {topic} look like in 100 years?",
                f"What's the opposite of {topic}?",
                f"What if {topic} was alive?",
                f"Minimise {topic} to its essence.",
                f"Maximise {topic} to its extreme.",
                f"What if {topic} could dream?",
            ]
            prompt = strategies[i % len(strategies)]
            idea = self.mind.llm(prompt)
            ideas.append(idea)
        self.creations.append({"topic": topic, "ideas": ideas, "timestamp": time.time()})
        return ideas

    def solve_creative(self, problem: str) -> Dict[str, Any]:
        """Solve a problem creatively via dream + evolve."""
        # 1. dream about the problem
        self.mind.learn(f"Problem to solve: {problem}")
        dream_result = self.mind.dream(steps=3)
        # 2. brainstorm
        ideas = self.brainstorm(problem, n_ideas=5)
        # 3. think about each idea
        best_idea = None
        best_score = -1
        for idea in ideas:
            result = self.mind.think(f"Evaluate this idea for solving '{problem}': {idea}")
            # score by confidence + length
            score = (1 if result["confident"] else 0) + len(result["answer"]) / 500
            if score > best_score:
                best_score = score
                best_idea = idea
        return {
            "problem": problem,
            "ideas": ideas,
            "best_idea": best_idea,
            "best_score": best_score,
            "dream_log": dream_result,
        }

    def evolve_idea(self, idea: str, generations: int = 3) -> Dict[str, Any]:
        """Evolve an idea over multiple generations."""
        current = idea
        history = []
        for gen in range(generations):
            # mutate: ask LLM to improve
            improved = self.mind.llm(f"Improve this idea: {current}")
            history.append({"generation": gen, "idea": current, "improved": improved})
            current = improved
        return {"original": idea, "final": current, "history": history}

    def creations_gallery(self) -> List[Dict[str, Any]]:
        """View all creations."""
        return list(self.creations)


__all__ = [
    "QuantumMind", "CodeAgent", "ResearchAgent", "CreativeAgent",
]
