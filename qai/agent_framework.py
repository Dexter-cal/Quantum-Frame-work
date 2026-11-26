"""
qai.agents — AI agent framework with tool use, multi-agent, and reasoning.

Provides:
  - Agent — single agent with tool use, memory, and reasoning loop
  - ReActAgent — Reasoning + Acting loop (Yao et al. 2022)
  - ToolUseAgent — function calling / tool use
  - MultiAgent — orchestrate multiple agents
  - PlannerAgent — task decomposition + planning
  - Tool — base class for tools (function calling)
  - Toolkit — collection of tools

Usage:
    import qai

    # Define tools
    @qai.Tool
    def calculator(expression: str) -> str:
        '''Evaluate a math expression.'''
        try:
            return str(eval(expression))
        except:
            return "Error"

    @qai.Tool
    def search(query: str) -> str:
        '''Search the web.'''
        return f"Results for: {query}"

    # Create agent
    agent = qai.ReActAgent(
        model=my_llm,
        tools=[calculator, search],
        max_steps=10,
    )
    result = agent.run("What is 25 * 17 + 100?")
    print(result)
"""
from __future__ import annotations
import re, json, time, os
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np


# =====================================================================
# Tool abstraction
# =====================================================================
@dataclass
class Tool:
    """A tool that an agent can call.

    Can be used as a decorator:
        @qai.Tool
        def my_tool(x: str) -> str:
            '''Does something.'''
            return x.upper()
    """
    func: Callable = None
    name: str = ""
    description: str = ""

    def __init__(self, func: Callable = None, name: str = "",
                 description: str = ""):
        if callable(func):
            self.func = func
            self.name = name or func.__name__
            self.description = description or func.__doc__ or ""
        else:
            self.func = None
            self.name = name
            self.description = description

    def __call__(self, *args, **kwargs):
        if self.func:
            return self.func(*args, **kwargs)
        raise NotImplementedError("Tool has no function")

    def schema(self) -> Dict:
        """Return a JSON-schema-like description for the LLM."""
        import inspect
        sig = inspect.signature(self.func) if self.func else {}
        params = {}
        if self.func:
            for pname, param in inspect.signature(self.func).parameters.items():
                params[pname] = {"type": "string", "description": ""}
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
        }


class Toolkit:
    """A collection of tools."""
    def __init__(self, tools: Optional[List[Tool]] = None):
        self.tools: Dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.add(t)

    def add(self, tool: Tool):
        if not isinstance(tool, Tool):
            tool = Tool(tool)
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def call(self, name: str, *args, **kwargs) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Error: tool '{name}' not found"
        try:
            return str(tool(*args, **kwargs))
        except Exception as e:
            return f"Error: {e}"

    def schemas(self) -> List[Dict]:
        return [t.schema() for t in self.tools.values()]

    def __len__(self):
        return len(self.tools)


# =====================================================================
# Agent memory
# =====================================================================
@dataclass
class AgentMemory:
    """Simple conversation memory for agents."""
    history: List[Dict[str, str]] = field(default_factory=list)
    max_messages: int = 50

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]

    def get_context(self, n: int = 10) -> str:
        """Get the last n messages as a context string."""
        recent = self.history[-n:]
        return "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    def clear(self):
        self.history = []


# =====================================================================
# Base Agent
# =====================================================================
class Agent:
    """Base agent with tool use and memory.

    Args:
        model: a qai Model (LLM) for reasoning
        tools: list of Tool objects
        system_prompt: system prompt for the agent
        max_steps: maximum reasoning steps
        memory: optional AgentMemory for conversation history
    """
    def __init__(self, model, tools: Optional[List[Tool]] = None,
                 system_prompt: str = "You are a helpful assistant.",
                 max_steps: int = 10, memory: Optional[AgentMemory] = None):
        self.model = model
        self.toolkit = Toolkit(tools or [])
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.memory = memory or AgentMemory()

    def _generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text from the model. Override for real LLM integration."""
        try:
            import qai
            tokens = qai.stream_generate(self.model, [ord(c) % 256 for c in prompt],
                                           max_new_tokens=max_new_tokens)
            return "".join(chr(t % 256) for t in tokens)
        except Exception:
            return "I cannot generate a response."

    def run(self, query: str) -> str:
        """Run the agent on a query. Override in subclasses."""
        raise NotImplementedError("Override run() in subclass")


# =====================================================================
# ReAct Agent (Reasoning + Acting)
# =====================================================================
class ReActAgent(Agent):
    """ReAct agent — alternates between reasoning and acting.

    Pattern:
        Thought: I need to calculate 25 * 17
        Action: calculator
        Action Input: 25 * 17
        Observation: 425
        Thought: Now I add 100
        Action: calculator
        Action Input: 425 + 100
        Observation: 525
        Thought: I have the answer
        Final Answer: 525
    """
    def __init__(self, model, tools=None, system_prompt: str = "",
                 max_steps: int = 10, memory=None):
        prompt = system_prompt or (
            "You are a reasoning agent. Use tools to answer questions.\n"
            "Format:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: <input for tool>\n"
            "Observation: <tool result>\n"
            "... (repeat as needed)\n"
            "Final Answer: <your answer>\n"
        )
        super().__init__(model, tools, prompt, max_steps, memory)

    def run(self, query: str) -> str:
        self.memory.add("user", query)
        context = f"{self.system_prompt}\n\nQuestion: {query}\n"

        for step in range(self.max_steps):
            # Generate the next thought/action
            response = self._generate(context, max_new_tokens=150)

            # Parse the response
            if "Final Answer:" in response:
                answer = response.split("Final Answer:")[-1].strip()
                self.memory.add("assistant", answer)
                return answer

            # Extract action and input
            action_match = re.search(r'Action:\s*(\w+)', response)
            input_match = re.search(r'Action Input:\s*(.+?)(?:\n|$)', response)
            if action_match and input_match:
                tool_name = action_match.group(1)
                tool_input = input_match.group(1).strip()
                # Call the tool
                observation = self.toolkit.call(tool_name, tool_input)
                context += f"\n{response}\nObservation: {observation}\n"
                self.memory.add("assistant", response)
                self.memory.add("tool", observation)
            else:
                # No action parsed — try to give a direct answer
                context += f"\n{response}\n"
                if "Thought:" in response and "Action:" not in response:
                    # Pure reasoning — continue
                    continue
                # Otherwise, return what we have
                self.memory.add("assistant", response)
                return response.strip()

        return "I could not solve this within the step limit."


# =====================================================================
# Tool-Use Agent (function calling)
# =====================================================================
class ToolUseAgent(Agent):
    """Agent that uses function-calling style tool use.

    Instead of ReAct's text-based parsing, this agent:
      1. Generates a JSON tool call
      2. Executes the tool
      3. Feeds the result back
      4. Repeats until done
    """
    def run(self, query: str) -> str:
        self.memory.add("user", query)
        tool_schemas = json.dumps(self.toolkit.schemas(), indent=2)
        context = (
            f"{self.system_prompt}\n\n"
            f"Available tools: {tool_schemas}\n\n"
            f"User: {query}\n"
        )
        for step in range(self.max_steps):
            response = self._generate(context, max_new_tokens=150)
            # Try to parse as JSON tool call
            try:
                # Find JSON in the response
                json_match = re.search(r'\{[^}]+\}', response)
                if json_match:
                    call = json.loads(json_match.group())
                    if "tool" in call and "input" in call:
                        result = self.toolkit.call(call["tool"], call["input"])
                        context += f"\nTool call: {call}\nResult: {result}\n"
                        self.memory.add("tool", result)
                        continue
            except (json.JSONDecodeError, KeyError):
                pass
            # No tool call — treat as final answer
            self.memory.add("assistant", response)
            return response.strip()
        return "I could not complete this task."


# =====================================================================
# Planner Agent (task decomposition)
# =====================================================================
class PlannerAgent(Agent):
    """Agent that decomposes a task into subtasks, then solves each.

    1. Plan: break the query into steps
    2. Execute each step (using tools or sub-agents)
    3. Synthesize the final answer
    """
    def run(self, query: str) -> str:
        self.memory.add("user", query)
        # Step 1: Generate a plan
        plan_prompt = (
            f"{self.system_prompt}\n\n"
            f"Task: {query}\n"
            f"Break this into numbered steps (1-5 steps max):\n"
        )
        plan = self._generate(plan_prompt, max_new_tokens=200)
        self.memory.add("assistant", f"Plan:\n{plan}")
        # Step 2: Execute each step
        results = []
        for line in plan.strip().split("\n"):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            step_result = self._generate(
                f"Execute this step: {line}\nContext: {self.memory.get_context(5)}",
                max_new_tokens=100)
            results.append(step_result)
        # Step 3: Synthesize
        synthesis = self._generate(
            f"Based on these results: {results}\n"
            f"Answer the original question: {query}",
            max_new_tokens=150)
        self.memory.add("assistant", synthesis)
        return synthesis.strip()


# =====================================================================
# Multi-Agent orchestration
# =====================================================================
class MultiAgent:
    """Orchestrate multiple agents to solve a task.

    Args:
        agents: dict of name → Agent
        coordinator: an agent that decides which sub-agent to call
        max_rounds: maximum coordination rounds

    Usage:
        multi = qai.MultiAgent(
            agents={
                "math": ReActAgent(model=m, tools=[calculator]),
                "search": ReActAgent(model=m, tools=[search]),
                "writer": Agent(model=m),
            },
            coordinator=Agent(model=m),
        )
        result = multi.run("Research the population of France and calculate density")
    """
    def __init__(self, agents: Dict[str, Agent],
                 coordinator: Optional[Agent] = None,
                 max_rounds: int = 5):
        self.agents = agents
        self.coordinator = coordinator or Agent(
            model=next(iter(agents.values())).model)
        self.max_rounds = max_rounds

    def run(self, task: str) -> str:
        """Run the multi-agent system on a task."""
        context = f"Task: {task}\nAvailable agents: {list(self.agents.keys())}\n"
        results = []
        for round_num in range(self.max_rounds):
            # Coordinator decides which agent to call
            decision = self.coordinator._generate(
                f"{context}\nWhich agent should handle this next? "
                f"Reply with just the agent name, or 'DONE' if finished.",
                max_new_tokens=20)
            agent_name = decision.strip().lower()
            if agent_name == "done" or agent_name not in self.agents:
                break
            # Call the selected agent
            sub_task = self.coordinator._generate(
                f"{context}\nWhat specific sub-task should '{agent_name}' do?",
                max_new_tokens=50)
            result = self.agents[agent_name].run(sub_task.strip())
            results.append({"agent": agent_name, "task": sub_task, "result": result})
            context += f"\n{agent_name} did: {result[:100]}\n"
        # Synthesize final answer
        final = self.coordinator._generate(
            f"Task: {task}\nResults: {results}\nSynthesize a final answer.",
            max_new_tokens=200)
        return final.strip()


__all__ = ["Tool", "Toolkit", "AgentMemory", "Agent", "ReActAgent",
           "ToolUseAgent", "PlannerAgent", "MultiAgent"]
