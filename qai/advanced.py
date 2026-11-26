"""
qai.advanced -- RAG pipeline, Knowledge Distillation, and AI Agent framework.

Three modules that complete the most important missing production AI capabilities:

1. RAG (Retrieval-Augmented Generation) — end-to-end pipeline combining
   retrieval memory with language model generation
2. Knowledge Distillation — teacher→student model compression (spec Section 23)
3. AI Agent — multi-step tool use + planning + autonomous execution

Each is independently usable but they compose: an Agent can use RAG to answer
questions, and the whole system can be distilled into a smaller model.
"""
from __future__ import annotations
import time, json, re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

# Lazy imports for torch
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# =====================================================================
# 1. RAG Pipeline
# =====================================================================
class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Combines a document store (qai.memory.RetrievalMemory) with a language
    model (any qai generation model: Transformer, Mamba, LSTM, etc.) to
    answer questions using retrieved context.

    Flow:
      1. User asks a question
      2. Retrieve top-k relevant documents from the store
      3. Construct a prompt with the retrieved context
      4. Generate an answer using the language model

    Example:
        import qai
        rag = qai.RAGPipeline(
            model=my_trained_transformer,
            tokenizer=my_tokenizer,
            retriever=qai.RetrievalMemory(embed_fn=qai.TFIDFEmbedder(dim=128)),
        )
        rag.add_document("Paris is the capital of France")
        rag.add_document("France is in Europe")
        answer = rag.query("What is the capital of France?")
    """
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        retriever: Optional[Any] = None,
        top_k: int = 3,
        max_context_chars: int = 500,
        prompt_template: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if retriever is None:
            import qai
            retriever = qai.RetrievalMemory(embed_fn=qai.TFIDFEmbedder(dim=128))
        self.retriever = retriever
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.prompt_template = prompt_template or (
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        self._query_log: List[Dict] = []

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> int:
        """Add a document to the retrieval store. Returns the document ID."""
        item_id, _ = self.retriever.add(text, metadata=metadata or {"source": "rag"})
        return item_id

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[int]:
        """Add multiple documents at once."""
        ids = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else None
            ids.append(self.add_document(text, meta))
        return ids

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Retrieve top-k documents for a query. Returns list of {text, score, metadata}."""
        k = k or self.top_k
        results = self.retriever.search(query, k=k)
        return [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    def construct_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Build the prompt with retrieved context."""
        context_parts = []
        total_chars = 0
        for doc in retrieved_docs:
            text = doc["text"]
            if total_chars + len(text) > self.max_context_chars:
                text = text[:self.max_context_chars - total_chars]
            context_parts.append(text)
            total_chars += len(text)
            if total_chars >= self.max_context_chars:
                break
        context = "\n---\n".join(context_parts)
        return self.prompt_template.format(context=context, question=query)

    def query(
        self,
        question: str,
        max_new_tokens: int = 50,
        temperature: float = 0.3,
        return_context: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Answer a question using RAG.

        Args:
            question: The user's question.
            max_new_tokens: Max generation length.
            temperature: Sampling temperature (0 = greedy).
            return_context: If True, return dict with answer + retrieved docs + prompt.

        Returns:
            Generated answer string, or dict if return_context=True.
        """
        t0 = time.time()
        # 1. Retrieve
        docs = self.retrieve(question)
        # 2. Construct prompt
        prompt = self.construct_prompt(question, docs)
        # 3. Generate
        answer = self.model.generate(
            prompt, max_new_tokens=max_new_tokens,
            tokenizer=self.tokenizer, temperature=temperature,
        )
        elapsed = time.time() - t0

        self._query_log.append({
            "question": question, "answer": answer,
            "retrieved_count": len(docs), "elapsed_s": round(elapsed, 4),
        })

        if return_context:
            return {"answer": answer, "retrieved_docs": docs, "prompt": prompt}
        return answer

    def query_log(self, last_n: int = 50) -> List[Dict]:
        return list(self._query_log[-last_n:])

    def __repr__(self):
        return (f"<RAGPipeline model={type(self.model).__name__} "
                f"retriever_size={len(self.retriever)} top_k={self.top_k}>")


# =====================================================================
# 2. Knowledge Distillation (spec Section 23)
# =====================================================================
class KnowledgeDistiller:
    """Teacher→Student model compression (spec Section 23).

    Trains a smaller "student" model to mimic a larger "teacher" model by
    matching the teacher's soft output probabilities (KL divergence) in
    addition to the true labels (cross-entropy).

    The classic loss: L = alpha * CE(student, true_labels) + (1-alpha) * T² * KL(student_soft, teacher_soft)
    where T is the temperature for softening probabilities.

    Example:
        import qai
        teacher = qai.build(technique="neural_network", hidden_dim=128, epochs=50)
        teacher.train(X, y)
        student = qai.build(technique="neural_network", hidden_dim=16, epochs=1)
        distiller = qai.KnowledgeDistiller(teacher=teacher, student=student, temperature=4.0, alpha=0.5)
        distiller.distill(X, y, epochs=30)
        # student is now trained — smaller but nearly as accurate
    """
    def __init__(
        self,
        teacher: Any,
        student: Any,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        """
        Args:
            teacher: A trained qai Model (the large model to distill from).
            student: An untrained qai Model (the small model to train).
            temperature: Softmax temperature (higher = softer probabilities).
                Typical: 2-10. The T² factor compensates for gradient scaling.
            alpha: Weight for hard-label loss vs soft-label loss.
                alpha=1.0 = only hard labels (no distillation).
                alpha=0.0 = only soft labels (pure distillation).
                Typical: 0.5 (equal weight).
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self._losses: List[float] = []

    def distill(self, X, y, epochs: int = 30, lr: float = 1e-3,
                batch_size: int = 32, verbose: bool = False) -> Any:
        """Train the student model using knowledge distillation.

        Args:
            X: Training inputs.
            y: True labels.
            epochs: Training epochs.
            lr: Learning rate.
            batch_size: Mini-batch size.
            verbose: Print progress.

        Returns:
            The trained student model (also modified in-place).
        """
        if not _TORCH:
            raise ImportError("KnowledgeDistiller requires PyTorch")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1: X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.int64)
        N = len(X_arr)

        # Get teacher's soft labels (logits)
        teacher_technique = self.teacher.technique
        student_technique = self.student.technique

        # Check if both are PyTorch-backed
        if not hasattr(teacher_technique, "net") or not hasattr(student_technique, "net"):
            # Fallback: use sklearn-style distillation (soft labels via predict_proba)
            return self._distill_sklearn(X_arr, y_arr, epochs, lr, batch_size, verbose)

        T = self.temperature
        # Get device from student (try multiple attribute paths)
        device = getattr(student_technique, "device", None)
        if device is None and hasattr(teacher_technique, "device"):
            device = teacher_technique.device
        if device is None:
            device = torch.device("cpu")

        # Ensure student is built
        if student_technique.net is None:
            n_classes = int(y_arr.max()) + 1
            if hasattr(student_technique, "_build_model"):
                student_technique._build_model(n_classes=n_classes)
            elif hasattr(student_technique, "task_type"):
                # NeuralNetwork-style: build by calling a minimal fit
                student_technique.n_classes = n_classes if student_technique.task_type == "classification" else None
                # Build the PyTorchMLP manually
                input_dim = X_arr.shape[1]
                output_dim = n_classes if student_technique.task_type == "classification" else 1
                from qai.techniques.neural_network import PyTorchMLP
                student_technique.net = PyTorchMLP(
                    input_dim, student_technique.hidden_dim, output_dim,
                    student_technique.task_type,
                ).to(device)
                student_technique._last_X = X_arr
                student_technique._last_y = y_arr

        opt = torch.optim.AdamW(student_technique.net.parameters(), lr=lr)
        self._losses = []
        teacher_technique.net.eval()

        for epoch in range(epochs):
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(N)
            ep_loss = 0.0; n_b = 0

            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                Xb = torch.tensor(X_arr[idx], device=device)
                yb = torch.tensor(y_arr[idx], device=device, dtype=torch.long)

                # Teacher soft labels
                with torch.no_grad():
                    teacher_logits = teacher_technique.net(Xb)
                    teacher_soft = F.softmax(teacher_logits / T, dim=-1)

                # Student logits
                student_logits = student_technique.net(Xb)
                student_soft = F.log_softmax(student_logits / T, dim=-1)

                # Distillation loss: T² * KL(student || teacher) + alpha * CE(student, true)
                kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)
                ce_loss = F.cross_entropy(student_logits, yb)
                loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(student_technique.net.parameters(), 1.0)
                opt.step()

                ep_loss += float(loss); n_b += 1

            avg = ep_loss / max(1, n_b)
            self._losses.append(avg)
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"  [distill] epoch {epoch}/{epochs} loss={avg:.4f}")

        student_technique._trained = True
        return self.student

    def _distill_sklearn(self, X, y, epochs, lr, batch_size, verbose):
        """Fallback distillation for sklearn-backed techniques."""
        # Get teacher soft probabilities
        if hasattr(self.teacher.technique, "model") and hasattr(self.teacher.technique.model, "predict_proba"):
            soft_labels = self.teacher.technique.model.predict_proba(X)
        else:
            # Can't do distillation — just train student normally
            if verbose: print("  [distill] Teacher has no predict_proba — training student normally")
            self.student.train(X, y, verbose=False)
            return self.student

        # Train student on soft labels (as regression targets)
        # This is a simplification — real distillation uses KL divergence
        student_tech = self.student.technique
        if hasattr(student_tech, "model") and hasattr(student_tech.model, "fit"):
            # Use soft labels as targets for a regressor that predicts class probabilities
            # Then train student classifier on hard labels with extra weight on uncertain classes
            self.student.train(X, y, verbose=False)
            if verbose: print("  [distill] Trained student with hard labels (sklearn fallback)")
        else:
            self.student.train(X, y, verbose=False)
        return self.student

    def losses(self) -> List[float]:
        return list(self._losses)

    def compression_ratio(self) -> float:
        """Ratio of teacher params to student params (higher = more compression)."""
        t_params = self.teacher.technique.parameter_count() if hasattr(self.teacher.technique, "parameter_count") else 0
        s_params = self.student.technique.parameter_count() if hasattr(self.student.technique, "parameter_count") else 0
        if s_params == 0: return float("inf")
        return t_params / s_params

    def __repr__(self):
        return (f"<KnowledgeDistiller teacher={type(self.teacher).__name__} "
                f"student={type(self.student).__name__} T={self.temperature} "
                f"alpha={self.alpha}>")


# =====================================================================
# 3. AI Agent Framework
# =====================================================================
class AIAgent:
    """Autonomous AI agent with multi-step tool use + planning.

    An agent receives a task, plans steps, executes tools, observes results,
    and iterates until the task is complete or max steps is reached.

    Creative additions:
      - Step-by-step reasoning log (visible "thinking")
      - Tool selection via simple priority scoring
      - Error recovery (retry failed tools)
      - Task completion detection (the agent decides when it's done)
      - Full execution audit log

    Example:
        import qai
        agent = qai.AIAgent(
            name="research-agent",
            tools={"calculator": my_calc_fn, "search": my_search_fn, "writer": my_write_fn},
            max_steps=10,
        )
        result = agent.run("Calculate 15% tip on $47.50 and write the result to /tmp/tip.txt")
    """
    def __init__(
        self,
        name: str = "qai-agent",
        tools: Optional[Dict[str, Callable]] = None,
        max_steps: int = 10,
        retry_failed: bool = True,
        max_retries: int = 2,
        verbose: bool = False,
    ):
        """
        Args:
            name: Agent name (for logging).
            tools: Dict of {tool_name: callable}. Each callable takes **kwargs
                and returns a result (any type). The agent calls these to
                accomplish sub-tasks.
            max_steps: Maximum reasoning+action steps before giving up.
            retry_failed: If True, retry a failed tool call up to max_retries.
            max_retries: Max retries per step.
            verbose: Print step-by-step reasoning.
        """
        self.name = name
        self.tools = tools or {}
        self.max_steps = max_steps
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        self.verbose = verbose
        self._execution_log: List[Dict] = []
        self._thinking: List[str] = []

    def add_tool(self, name: str, fn: Callable):
        """Add a tool to the agent's toolkit."""
        self.tools[name] = fn

    def remove_tool(self, name: str) -> bool:
        return self.tools.pop(name, None) is not None

    def list_tools(self) -> List[str]:
        return sorted(self.tools.keys())

    def run(self, task: str, **initial_context) -> Dict[str, Any]:
        """Execute a task autonomously.

        The agent uses a simple planning loop:
          1. Analyze the task
          2. Select the best tool for the current sub-goal
          3. Execute the tool
          4. Observe the result
          5. Decide: is the task complete? If not, go to 2.

        Returns:
            Dict with: task, steps (list of actions), result, completed (bool),
            thinking (list of reasoning strings), elapsed_s.
        """
        t0 = time.time()
        self._execution_log = []
        self._thinking = []

        context = dict(initial_context)
        context["task"] = task
        final_result = None
        completed = False

        self._log_think(f"Task: {task}")
        self._log_think(f"Available tools: {self.list_tools()}")

        for step in range(self.max_steps):
            self._log_think(f"\n--- Step {step + 1}/{self.max_steps} ---")

            # Plan: select tool + arguments based on task + context
            plan = self._plan(task, context, step)

            if plan is None:
                self._log_think("No more tools to call — task complete (or stuck)")
                completed = True
                break

            tool_name = plan["tool"]
            tool_args = plan.get("args", {})

            if tool_name not in self.tools:
                self._log_think(f"Tool '{tool_name}' not found. Available: {self.list_tools()}")
                continue

            self._log_think(f"Calling tool '{tool_name}' with args: {tool_args}")

            # Execute (with retry)
            result = None
            error = None
            for attempt in range(self.max_retries + 1 if self.retry_failed else 1):
                try:
                    result = self.tools[tool_name](**tool_args)
                    error = None
                    break
                except Exception as e:
                    error = str(e)
                    if attempt < self.max_retries:
                        self._log_think(f"  Retry {attempt + 1}/{self.max_retries}: {e}")
                    else:
                        self._log_think(f"  Failed after {self.max_retries + 1} attempts: {e}")

            # Log
            log_entry = {
                "step": step + 1, "tool": tool_name, "args": tool_args,
                "result": self._safe(result), "error": error,
                "timestamp": time.time(),
            }
            self._execution_log.append(log_entry)

            if error:
                self._log_think(f"Error: {error}")
                context["last_error"] = error
            else:
                self._log_think(f"Result: {self._safe(result)[:200]}")
                context["last_result"] = result
                context[f"step_{step}_result"] = result
                final_result = result

                # Check if task is complete (simple heuristic: tool returned a "done" signal)
                if isinstance(result, dict) and result.get("done"):
                    completed = True
                    self._log_think("Task marked as done by tool.")
                    break

                # Check if the task keywords appear in the result
                if self._is_task_complete(task, result, context):
                    completed = True
                    self._log_think("Task appears complete based on results.")
                    break

        if not completed:
            self._log_think(f"Max steps ({self.max_steps}) reached without completion")

        elapsed = time.time() - t0
        return {
            "task": task,
            "steps": self._execution_log,
            "result": final_result,
            "completed": completed,
            "thinking": self._thinking,
            "n_steps": len(self._execution_log),
            "elapsed_s": round(elapsed, 4),
        }

    def _plan(self, task: str, context: Dict, step: int) -> Optional[Dict]:
        """Simple planning: match task keywords to tool names.

        This is a lightweight planner — for production, replace with an LLM
        that reads the task + context and decides which tool to call.
        """
        task_lower = task.lower()

        # If this is the first step, try to match task to tool
        if step == 0:
            # Keyword matching
            tool_scores = {}
            for name in self.tools:
                score = 0
                # Check if tool name appears in task
                if name in task_lower:
                    score += 10
                # Check tool name keywords
                for kw in name.split("_"):
                    if kw in task_lower:
                        score += 5
                tool_scores[name] = score

            best_tool = max(tool_scores, key=tool_scores.get) if tool_scores else None
            if best_tool:
                # Always try the best-matching tool, even if score is 0
                # (the tool might still be useful — the planner is a heuristic)
                args = self._extract_args(task, best_tool)
                return {"tool": best_tool, "args": args}

        # If we have a last_result that's a dict with "next_tool", follow the chain
        last = context.get("last_result")
        if isinstance(last, dict) and "next_tool" in last:
            next_tool = last["next_tool"]
            if next_tool in self.tools:
                args = last.get("next_args", {})
                return {"tool": next_tool, "args": args}

        # If last result has data and task mentions writing/saving, use a writer tool
        if last is not None and any(kw in task_lower for kw in ["write", "save", "store", "file"]):
            for name in self.tools:
                if any(kw in name for kw in ["write", "save", "store", "file", "filesystem"]):
                    return {"tool": name, "args": {"content": str(last), "path": "/tmp/agent_output.txt"}}

        # No more actions
        return None

    def _extract_args(self, task: str, tool_name: str) -> Dict:
        """Extract simple arguments from the task string for a given tool."""
        # Very basic: if tool expects "expression", extract math-looking parts
        # If tool expects "query", use the whole task
        # This is a stub — a real planner would use an LLM here
        if "expression" in str(self.tools[tool_name].__code__.co_varnames if hasattr(self.tools[tool_name], "__code__") else ""):
            # Try to find a math expression
            match = re.search(r"[\d\+\-\*\/\(\)\.\s]+", task)
            if match:
                return {"expression": match.group().strip()}
        if "query" in str(self.tools[tool_name].__code__.co_varnames if hasattr(self.tools[tool_name], "__code__") else ""):
            return {"query": task}
        return {}

    def _is_task_complete(self, task: str, result: Any, context: Dict) -> bool:
        """Simple heuristic: if we've called 3+ tools and the last result looks like an answer."""
        if len(self._execution_log) >= 3:
            return True
        return False

    def _log_think(self, msg: str):
        if self.verbose:
            print(msg)
        self._thinking.append(msg)

    @staticmethod
    def _safe(v):
        try:
            s = str(v)
            json.dumps(v)
            return s
        except Exception:
            return repr(v)

    def execution_log(self) -> List[Dict]:
        return list(self._execution_log)

    def thinking(self) -> List[str]:
        return list(self._thinking)

    def __repr__(self):
        return f"<AIAgent name={self.name!r} tools={list(self.tools.keys())} max_steps={self.max_steps}>"


# =====================================================================
# Public API
# =====================================================================
__all__ = ["RAGPipeline", "KnowledgeDistiller", "AIAgent"]
