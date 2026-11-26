# Guide: Agents & MCP — Tool-Using AI

*Models that act: call tools, plan multi-step tasks, talk to each other, and expose themselves as tools.*

> Back to [Guides index](README.md) · Related: [MEMORY_RAG.md](MEMORY_RAG.md) · Full API: [User Manual](../USER_MANUAL.md)

---

## What you'll learn

1. Define tools (`@qai.Tool`)
2. Run ReAct agents and multi-agent teams
3. Test-time reasoning strategies (CoT, ToT, Self-Consistency)
4. Expose qai models as an MCP server (14 built-in tools)

---

## 1. Define a tool

```python
import qai

@qai.Tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@qai.Tool
def lookup_order(order_id: str) -> dict:
    """Fetch an order from the database."""
    return db.query("SELECT * FROM orders WHERE id = ?", order_id)
```

The docstring is the tool's description — the agent reads it to decide when to
call. Validate args yourself; agents get creative.

## 2. ReAct agent (reason + act loop)

```python
agent = qai.ReActAgent(model=my_llm, tools=[calculator, lookup_order])
result = agent.run("What is 25 * 17 + 100, and what's the status of order #4211?")
print(result.answer)
print(result.steps)      # the thought → action → observation trace
```

## 3. Multi-agent teams

```python
math_agent = qai.ReActAgent(model=m, tools=[calculator])
writer_agent = qai.Agent(model=m)

multi = qai.MultiAgent(agents={
    "math": math_agent,
    "writer": writer_agent,
})
result = multi.run("Compute the margin, then write a summary email.")
```

## 4. Autonomous agent (`qai.AIAgent`)

Beyond ReAct — task planning, tool selection, error recovery with retries,
execution audit log, and a visible thinking log:

```python
from qai.agent_framework import AIAgent

agent = AIAgent(model=m, tools=[...])
result = agent.execute("Research the three cheapest suppliers and draft an email")
print(result.plan)          # planned steps
print(result.audit_log)     # every action taken, timestamped
print(result.thinking)      # visible reasoning
```

## 5. Test-time reasoning (scale compute at inference)

```python
answer = qai.ChainOfThought(model=m).solve("What is 15% of 240?")
answer = qai.SelfConsistency(model=m, n_samples=5).solve(problem)
answer = qai.TreeOfThoughts(model=m, n_branches=3).solve(problem)
answer = qai.VerificationGuided(model=m, verifier=check_fn).solve(problem)
answer = qai.ReasoningChain(model=m).solve(problem)
```

| Strategy | How | Cost |
|---|---|---|
| ChainOfThought | generate reasoning steps, then answer | 1x |
| SelfConsistency | N samples, majority vote | N x |
| TreeOfThoughts | branch and backtrack | high |
| VerificationGuided | generate, verify, retry | variable |
| ReasoningChain | chained decomposition | medium |

---

## 6. MCP — Model Context Protocol server (`qai.mcp`)

**What:** expose qai capabilities to any MCP client (Claude Desktop, Cursor,
or your own) over JSON-RPC 2.0 — stdio or HTTP transport.

### 14 built-in tools

**10 reference tools** (mirroring the official MCP reference servers):

| Tool | What it does |
|---|---|
| `FilesystemTool` | sandboxed file read/write |
| `MemoryTool` | knowledge-graph memory |
| `GitTool` | read-only git operations |
| `FetchTool` | HTTP fetch + HTML → text |
| `TimeTool` | timezone conversion |
| `SqliteTool` | read-only SQL queries |
| `ExecTool` | sandboxed subprocess |
| `SequentialThinkingTool` | recorded reasoning chain |
| `CalculatorTool` | AST-validated math |
| `CodeEvalTool` | sandboxed Python with restricted builtins |

**4 qai-specific tools** (expose your qai assets to any MCP client):

| Tool | What it does |
|---|---|
| `QaiModelTool` | any trained qai Model becomes a callable tool |
| `QaiLogicTool` | `@qai.logic` functions become tools |
| `QaiMemoryTool` | bridge to qai.memory tiers |
| `QaiBundleTool` | load a `.qmodel` and predict |

### Run the server

```python
from qai.mcp import create_server

server = create_server(transport="stdio")    # for Claude/Cursor
server.start()
```

```bash
# stdio transport from the shell:
python -m qai.mcp --transport stdio
```

### Safety controls (qai additions beyond the reference servers)

```python
server.configure(
    permissions={"ExecTool": False, "FilesystemTool": True},  # per-tool allow/deny
    rate_limits={"FetchTool": 10},      # calls per minute per tool
    audit_log=True,                      # every call recorded: timestamp, args, result, elapsed
)
```

Argument schemas are JSON-schema validated before any tool runs.

---

## Cheat sheet

| Task | Call |
|---|---|
| Define a tool | `@qai.Tool` with docstring |
| Reason + act agent | `qai.ReActAgent(model, tools=[...])` |
| Team of agents | `qai.MultiAgent(agents={...})` |
| Autonomous executor | `qai.agent_framework.AIAgent` |
| Think harder | `qai.ChainOfThought / SelfConsistency / TreeOfThoughts` |
| MCP server | `qai.mcp.create_server(transport="stdio")` |
| Model as MCP tool | `QaiModelTool` |
| Kill-switch a tool | `permissions={"ExecTool": False}` |
| Rate-limit a tool | `rate_limits={"FetchTool": 10}` |
| Audit everything | `audit_log=True` |

---

## Common gotchas

- **Tool docstrings are prompts.** "Evaluate a math expression" beats "helper".
- **Always sandbox ExecTool / CodeEvalTool inputs.** They're subprocesses by design — treat user strings as hostile. See [SECURITY_GOVERNANCE.md](SECURITY_GOVERNANCE.md).
- **ReAct loops can loop.** Cap `max_steps`; the audit log shows you where it spun.
- **MCP stdio for IDEs, HTTP for testing** — the HTTP transport is a Flask app, convenient for curl debugging.

---

*Next: [MEMORY_RAG.md](MEMORY_RAG.md) — give your agent persistent memory.*
