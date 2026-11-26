"""
qai.agents -- agentic AI patterns and multi-agent orchestration (2025 era).

Six core patterns + an A2A (Agent-to-Agent) protocol implementation:

  1. ReAct           -- Reason + Act loop with tool calls (Yao 2022)
  2. Reflexion       -- self-critique + memory of past failures (Shinn 2023)
  3. PlanAndExecute  -- planner decomposes, executor runs sub-tasks
  4. TreeOfThoughts  -- ToT branching search (Yao 2023)
  5. SelfAsk         -- decompose complex Q into sub-questions
  6. MultiAgentOrchestrator -- N agents + a router that delegates
  + A2AProtocol      -- Google A2A-style agent-to-agent messaging
"""
from __future__ import annotations
import json
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ============================================================================
# Tool definition
# ============================================================================
class Tool:
    """A callable tool with a name, description, and signature."""

    def __init__(self, name: str, description: str, fn: Callable[..., Any],
                 args_schema: Optional[Dict[str, type]] = None):
        self.name = name
        self.description = description
        self.fn = fn
        self.args_schema = args_schema or {}

    def __call__(self, **kwargs) -> Any:
        # type-check args
        for arg, val in kwargs.items():
            if arg in self.args_schema and not isinstance(val, self.args_schema[arg]):
                try:
                    kwargs[arg] = self.args_schema[arg](val)
                except Exception:
                    pass
        return self.fn(**kwargs)

    def describe(self) -> str:
        args = ", ".join(self.args_schema.keys()) or "(no args)"
        return f"{self.name}({args}) - {self.description}"


# ============================================================================
# 1. ReAct -- Reason + Act (Yao et al. 2022)
# ============================================================================
class ReAct:
    """ReAct agent: alternates Thought / Action / Observation steps."""

    def __init__(self, llm: Callable[[str], str], tools: List[Tool],
                 max_steps: int = 10, verbose: bool = False):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.verbose = verbose
        self.history: List[Dict[str, str]] = []

    def _build_prompt(self, query: str) -> str:
        tool_descs = "\n".join(f"  - {t.describe()}" for t in self.tools.values())
        history = "\n".join(f"{h['role']}: {h['content']}" for h in self.history)
        return f"""Answer the question using the ReAct pattern.

Available tools:
{tool_descs}

History so far:
{history}

Question: {query}

Format your reply as either:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON args>
OR
Thought: <your reasoning>
Final Answer: <the answer>

Reply:"""

    def _parse_response(self, response: str) -> Dict[str, str]:
        lines = response.strip().split("\n")
        result: Dict[str, str] = {}
        for line in lines:
            if line.startswith("Thought:"):
                result["thought"] = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                result["action"] = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                result["action_input"] = line[len("Action Input:"):].strip()
            elif line.startswith("Final Answer:"):
                result["final_answer"] = line[len("Final Answer:"):].strip()
        return result

    def run(self, query: str) -> Dict[str, Any]:
        self.history = [{"role": "user", "content": query}]
        for step in range(self.max_steps):
            prompt = self._build_prompt(query)
            response = self.llm(prompt)
            parsed = self._parse_response(response)
            if self.verbose:
                print(f"[ReAct step {step}] {parsed}")
            if "final_answer" in parsed:
                return {"answer": parsed["final_answer"], "steps": self.history, "n_steps": step + 1}
            if "action" not in parsed:
                # LLM didn't follow the format; bail with last response
                return {"answer": response, "steps": self.history, "n_steps": step + 1}
            tool_name = parsed["action"]
            if tool_name not in self.tools:
                observation = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    args = json.loads(parsed.get("action_input", "{}"))
                    observation = str(self.tools[tool_name](**args))
                except Exception as e:
                    observation = f"Error: {e}"
            self.history.append({"role": "thought", "content": parsed.get("thought", "")})
            self.history.append({"role": "action", "content": tool_name})
            self.history.append({"role": "observation", "content": observation})
        return {"answer": "max_steps_reached", "steps": self.history, "n_steps": self.max_steps}


# ============================================================================
# 2. Reflexion -- self-critique + episodic memory (Shinn et al. 2023)
# ============================================================================
class Reflexion:
    """Reflexion agent: tries a task, reflects on failure, stores a
    self-reflection in memory, and retries."""

    def __init__(self, llm: Callable[[str], str], actor: Optional[ReAct] = None,
                 max_trials: int = 3):
        self.llm = llm
        self.actor = actor or ReAct(llm, tools=[], max_steps=5)
        self.max_trials = max_trials
        self.episodic_memory: List[str] = []

    def _reflect(self, query: str, attempt: str, feedback: str) -> str:
        prompt = f"""You tried to answer: {query}
Your attempt: {attempt}
Feedback: {feedback}

Write a short reflection (1-2 sentences) on what to do differently next time."""
        return self.llm(prompt)

    def run(self, query: str, evaluator: Optional[Callable[[str], Tuple[bool, str]]] = None) -> Dict[str, Any]:
        history = []
        for trial in range(self.max_trials):
            memory_str = "\n".join(f"- {m}" for m in self.episodic_memory)
            augmented = f"{query}\n\nPast reflections:\n{memory_str}" if memory_str else query
            attempt = self.actor.run(augmented)
            answer = attempt["answer"]
            if evaluator:
                success, feedback = evaluator(answer)
            else:
                # default: success if answer is non-trivial
                success = len(answer) > 10 and "max_steps" not in answer
                feedback = "" if success else "Answer was too short or invalid."
            history.append({"trial": trial, "answer": answer, "success": success,
                             "feedback": feedback})
            if success:
                return {"answer": answer, "history": history, "n_trials": trial + 1}
            reflection = self._reflect(query, answer, feedback)
            self.episodic_memory.append(reflection)
        return {"answer": history[-1]["answer"], "history": history,
                "n_trials": self.max_trials}


# ============================================================================
# 3. Plan-and-Execute (Wang et al. 2023)
# ============================================================================
class PlanAndExecute:
    """Decompose a task into sub-tasks, execute each, then synthesise."""

    def __init__(self, planner_llm: Callable[[str], str],
                 executor: Callable[[str], str],
                 synthesiser: Optional[Callable[[str, List[str]], str]] = None):
        self.planner = planner_llm
        self.executor = executor
        self.synthesiser = synthesiser

    @staticmethod
    def _default_planner(query: str) -> str:
        """Heuristic planner: split by question type."""
        if "compare" in query.lower():
            return "1. Identify the entities.\n2. List attributes.\n3. Compare side-by-side.\n4. Conclude."
        return "1. Gather relevant info.\n2. Analyse.\n3. Synthesise answer."

    def run(self, query: str) -> Dict[str, Any]:
        plan_text = self.planner(f"Decompose this task into numbered steps:\n{query}\n\nPlan:")
        if not plan_text or plan_text.startswith("[LLM"):
            plan_text = self._default_planner(query)
        # parse numbered steps
        steps = []
        for line in plan_text.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                steps.append(line.split(".", 1)[-1].strip())
        # execute each
        results = []
        for i, step in enumerate(steps):
            result = self.executor(f"Step {i+1}: {step}\nQuery: {query}")
            results.append(result)
        # synthesise
        if self.synthesiser:
            final = self.synthesiser(query, results)
        else:
            final = "\n".join(f"[{i+1}] {r}" for i, r in enumerate(results))
        return {"plan": steps, "results": results, "answer": final}


# ============================================================================
# 4. Tree of Thoughts (Yao et al. 2023)
# ============================================================================
class TreeOfThoughts:
    """ToT: explore multiple reasoning branches, evaluate, and pick best."""

    def __init__(self, llm: Callable[[str], str],
                 evaluator: Optional[Callable[[str, str], float]] = None,
                 branching: int = 3, depth: int = 3, beam_width: int = 2):
        self.llm = llm
        self.evaluator = evaluator
        self.branching = branching
        self.depth = depth
        self.beam_width = beam_width

    def _expand(self, query: str, current_thought: str, branch_idx: int) -> str:
        prompt = f"""Question: {query}
Current thought: {current_thought}

Generate thought branch #{branch_idx} (a different angle than previous branches):
"""
        return self.llm(prompt)

    def _evaluate(self, query: str, thought: str) -> float:
        if self.evaluator:
            return self.evaluator(query, thought)
        # heuristic: longer = more thorough (capped)
        return min(1.0, len(thought.split()) / 50.0)

    def run(self, query: str) -> Dict[str, Any]:
        # beam search over thoughts
        beam = [("", 0.0)]  # (thought, score)
        tree_log = []
        for depth in range(self.depth):
            candidates = []
            for thought, score in beam:
                for b in range(self.branching):
                    new_thought = self._expand(query, thought, b)
                    full_thought = (thought + " " + new_thought).strip()
                    new_score = self._evaluate(query, full_thought)
                    candidates.append((full_thought, new_score))
            # keep top beam_width
            candidates.sort(key=lambda x: -x[1])
            beam = candidates[:self.beam_width]
            tree_log.append({"depth": depth, "beam": beam})
        best_thought, best_score = beam[0]
        return {"answer": best_thought, "score": best_score, "tree": tree_log}


# ============================================================================
# 5. Self-Ask (Press et al. 2022)
# ============================================================================
class SelfAsk:
    """Self-Ask: decompose complex question into follow-up sub-questions."""

    def __init__(self, llm: Callable[[str], str],
                 retriever: Optional[Callable[[str], str]] = None):
        self.llm = llm
        self.retriever = retriever

    def run(self, query: str) -> Dict[str, Any]:
        prompt = f"""Question: {query}

Decompose this into follow-up questions, one per line, prefixed by "Follow up:".
End with "Final answer:" on its own line."""
        decomposition = self.llm(prompt)
        follow_ups = []
        for line in decomposition.split("\n"):
            if line.strip().lower().startswith("follow up:"):
                follow_ups.append(line.split(":", 1)[-1].strip())
        # answer each follow-up
        sub_answers = []
        for fq in follow_ups:
            if self.retriever:
                ctx = self.retriever(fq)
                ans = self.llm(f"Context: {ctx}\n\nAnswer this briefly: {fq}")
            else:
                ans = self.llm(f"Answer briefly: {fq}")
            sub_answers.append(ans)
        # synthesise
        synth = self.llm(f"Question: {query}\n\nSub-answers:\n" +
                         "\n".join(f"- {q}: {a}" for q, a in zip(follow_ups, sub_answers)) +
                         "\n\nFinal answer:")
        return {"follow_ups": follow_ups, "sub_answers": sub_answers, "answer": synth}


# ============================================================================
# 6. Multi-Agent Orchestrator (à la AutoGen / CrewAI)
# ============================================================================
class Agent:
    """A named agent with a role, tools, and an LLM."""

    def __init__(self, name: str, role: str, llm: Callable[[str], str],
                 tools: Optional[List[Tool]] = None,
                 system_prompt: str = ""):
        self.name = name
        self.role = role
        self.llm = llm
        self.tools = {t.name: t for t in (tools or [])}
        self.system_prompt = system_prompt or f"You are {name}, a {role}."

    def respond(self, message: str) -> str:
        prompt = f"{self.system_prompt}\n\nMessage: {message}\n\nResponse:"
        return self.llm(prompt)


class MultiAgentOrchestrator:
    """Routes messages to specialised agents based on the task.

    Modes:
      - 'sequential'  : pass through agents in order
      - 'router'      : pick best agent for the task
      - 'debate'      : all agents answer, then vote
    """

    def __init__(self, agents: List[Agent], mode: str = "router"):
        self.agents = {a.name: a for a in agents}
        self.mode = mode

    def _router_pick(self, query: str) -> str:
        # heuristic: pick agent whose role keywords match query best
        scores = {}
        for name, agent in self.agents.items():
            keywords = agent.role.lower().split()
            q = query.lower()
            scores[name] = sum(1 for kw in keywords if kw in q)
        # tiebreak: first agent
        best = max(scores.items(), key=lambda kv: kv[1])
        if best[1] == 0:
            return list(self.agents.keys())[0]
        return best[0]

    def run(self, query: str) -> Dict[str, Any]:
        if self.mode == "router":
            chosen = self._router_pick(query)
            answer = self.agents[chosen].respond(query)
            return {"mode": "router", "chosen_agent": chosen, "answer": answer}
        if self.mode == "sequential":
            current = query
            responses = []
            for agent in self.agents.values():
                current = agent.respond(current)
                responses.append({"agent": agent.name, "response": current})
            return {"mode": "sequential", "responses": responses, "answer": current}
        if self.mode == "debate":
            responses = {name: agent.respond(query) for name, agent in self.agents.items()}
            # vote: pick the longest response (heuristic)
            winner = max(responses.items(), key=lambda kv: len(kv[1]))
            return {"mode": "debate", "responses": responses, "answer": winner[1],
                    "winner": winner[0]}
        raise ValueError(f"Unknown mode '{self.mode}'")


# ============================================================================
# A2A Protocol (Google 2025) -- Agent-to-Agent messaging
# ============================================================================
class A2AMessage:
    """A single agent-to-agent message."""

    def __init__(self, sender: str, recipient: str, content: str,
                 message_type: str = "request",
                 reply_to: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.message_type = message_type  # request / response / broadcast
        self.reply_to = reply_to
        self.timestamp = time.time()
        self.payload: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "sender": self.sender, "recipient": self.recipient,
                "content": self.content, "type": self.message_type,
                "reply_to": self.reply_to, "timestamp": self.timestamp,
                "payload": self.payload}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "A2AMessage":
        msg = cls(d["sender"], d["recipient"], d["content"], d.get("type", "request"),
                  d.get("reply_to"))
        msg.id = d["id"]
        msg.timestamp = d["timestamp"]
        msg.payload = d.get("payload", {})
        return msg


class A2AProtocol:
    """A registry + message bus for agent-to-agent communication.

    Implements the Google A2A design points:
      - each agent has a unique id
      - messages have request/response semantics
      - broadcasts go to all agents
      - synchronous request/response with timeout
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.mailboxes: Dict[str, List[A2AMessage]] = defaultdict(list)
        self.message_log: List[A2AMessage] = []

    def register(self, agent: Agent) -> None:
        self.agents[agent.name] = agent
        self.mailboxes[agent.name] = []

    def send(self, msg: A2AMessage) -> None:
        if msg.recipient not in self.agents and msg.message_type != "broadcast":
            raise ValueError(f"Unknown recipient '{msg.recipient}'")
        self.message_log.append(msg)
        if msg.message_type == "broadcast":
            for name in self.agents:
                if name != msg.sender:
                    self.mailboxes[name].append(msg)
        else:
            self.mailboxes[msg.recipient].append(msg)

    def request(self, sender: str, recipient: str, content: str) -> str:
        """Synchronous request -> response from recipient agent."""
        if recipient not in self.agents:
            raise ValueError(f"Unknown recipient '{recipient}'")
        msg = A2AMessage(sender, recipient, content, "request")
        self.send(msg)
        # recipient processes
        agent = self.agents[recipient]
        response_content = agent.respond(content)
        response = A2AMessage(recipient, sender, response_content, "response",
                              reply_to=msg.id)
        self.send(response)
        return response_content

    def broadcast(self, sender: str, content: str) -> Dict[str, str]:
        """Send to all; collect responses."""
        msg = A2AMessage(sender, "ALL", content, "broadcast")
        self.send(msg)
        responses = {}
        for name, agent in self.agents.items():
            if name == sender:
                continue
            responses[name] = agent.respond(content)
        return responses

    def inbox(self, agent_name: str) -> List[A2AMessage]:
        return list(self.mailboxes.get(agent_name, []))

    def trace(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.message_log]


__all__ = [
    "Tool", "ReAct", "Reflexion", "PlanAndExecute", "TreeOfThoughts", "SelfAsk",
    "Agent", "MultiAgentOrchestrator", "A2AMessage", "A2AProtocol",
]
