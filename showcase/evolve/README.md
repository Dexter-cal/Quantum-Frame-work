# EVOLVE — A Self-Improving Digital Organism

**EVOLVE** is a showcase project built entirely on the qai framework.
It demonstrates that qai can ship a **living, learning AI agent** in a
14 KB self-contained `.qmodel` bundle that runs anywhere Python + qai
are installed.

> Most ML models are frozen artifacts. EVOLVE is a *living* model —
> the weights you download are just the starting brain. The agent
> continues to learn every time you run it.

## What's in this directory

```
showcase/evolve/
├── README.md                    ← you are here
├── environments/
│   └── puzzle_world.py          ← 4-puzzle multi-task environment
├── agent/
│   ├── evolve_agent.py          ← the self-improving agent (Q-learning + curiosity + self-tuning)
│   └── evolve_logic.py          ← 6 @qai.logic functions (the agent's personality)
├── train_and_bundle.py          ← train agent + bundle as .qmodel
├── demo.py                      ← CLI demo: load bundle, watch agent learn live
├── launch_ui.py                 ← UI launcher: load bundle, open visual dashboard in browser
├── dashboard.html               ← the visual dashboard (single-file, no external deps)
├── bundle/
│   └── evolve.qmodel            ← the 26 KB signed bundle (after running train_and_bundle.py)
└── docs/
    └── WHATS_MISSING.md         ← honest list of qai gaps this build exposed
```

## Quickstart

### 1. Train and bundle the agent (one-time, ~1 minute on CPU)

```bash
cd showcase/evolve
PYTHONPATH=/path/to/qai/repo python train_and_bundle.py --episodes 120
```

This trains the agent from scratch on PuzzleWorld under a 200-second
CPU `ResourceBudget`, then writes `bundle/evolve.qmodel` (~26 KB, HMAC-
signed) **with the visual dashboard + launcher inside**.

### 2. Launch the visual dashboard (recommended)

```bash
python launch_ui.py
```

This loads the bundle, starts a tiny localhost HTTP server on port 8765
(Python stdlib only — no Flask, no FastAPI, no extra dependencies),
opens your default browser, and you see the live EVOLVE dashboard:

- **Header** — agent name, LIVE/IDLE status badge, SIGNED badge, version
- **Hero stats row** — Episodes, Puzzles Solved, Total Reward, Q-table size
  (with "this session" deltas that update live)
- **Skills by puzzle** — 4 color-coded bars (maze/sequence/resource/avoid)
  showing avg recent reward, with verdicts (MASTERED / LEARNING / STRUGGLING)
- **Self-tuned hyperparameters** — live gauges for ε (epsilon) and lr,
  showing the agent self-tuning in real time
- **Reward trajectory** — canvas chart of the last 30 episodes with a
  filled gradient area, color-coded points (green=positive, red=negative)
- **Agent monologue** — live text from `explain_self()` showing the agent's
  plain-English self-description
- **Self-critique** — live text from `self_critique()` showing the agent's
  candid review of its recent performance
- **Recent episodes** — scrollable list with puzzle-type chips and solved marks
- **Controls** — puzzle selector (random/maze/sequence/resource/avoid),
  "Run 1 / 10 / 25 / 50 episodes" buttons, Explain/Critique buttons,
  and a "Reset Brain" danger button

Click **Run 10 episodes** and watch:
1. The stat cards flash and tick up
2. The reward trajectory chart extends
3. The ε and lr gauges move (the agent self-tunes after every episode)
4. The agent's monologue updates with the new performance trend
5. The recent episodes list grows with new entries

### 3. CLI demo (alternative)

```bash
python demo.py --episodes 30 --puzzle random
```

Same agent, same bundle — just no UI. Prints everything to stdout.

### 4. Use the agent in your own code

```python
import sys
sys.path.insert(0, "showcase/evolve")
from train_and_bundle import load_evolve_bundle
from environments.puzzle_world import PuzzleWorld

bundle = load_evolve_bundle("showcase/evolve/bundle/evolve.qmodel",
                             sign_key="evolve-secret-2026")
agent = bundle["agent"]
logic = bundle["logic_fns"]

# Inject agent into logic fns
for fn in logic.values():
    fn.__globals__["agent"] = agent

# Ask the agent to explain itself
print(logic["explain_self"]())

# Run a live episode -- agent learns as it goes
env = PuzzleWorld()
summary = agent.run_episode(env, max_steps=30, learn=True)
print(summary)

# Ask the agent to critique its performance
print(logic["self_critique"]())
```

## What the agent can do

### Core capabilities

1. **Solve 4 different puzzle types simultaneously** — maze navigation,
   sequence prediction, resource harvesting, obstacle avoidance. A
   single Q-table holds strategies for all four.

2. **Learn online** — the Q-table updates on every step. Weights are
   not fixed at training time. The longer you run the agent, the
   smarter it gets.

3. **Self-tune its own hyperparameters** — after every episode, a
   `@qai.logic` reflector looks at the reward trajectory and adjusts
   epsilon (exploration rate) and learning rate. If stuck, it explores
   more; if improving, it exploits more.

4. **Get curious about unfamiliar states** — the agent tracks how
   recently it visited each state. Novel states get a curiosity boost
   to epsilon, driving the agent to explore them.

5. **Wipe its own memory for failing puzzle types** — if the agent's
   average reward on a puzzle type drops below -2 for 5+ episodes, it
   deletes all Q-table entries for that puzzle and re-learns it from
   scratch. Forgetting is part of learning.

6. **Explain itself in plain English** — six `@qai.logic` functions
   (decide_action, reflect_on_episode, propose_strategy, explain_self,
   should_experiment, self_critique) give the agent a readable
   "personality". Users can open `logic.py` inside the bundle and see
   exactly what the agent is thinking.

7. **Run anywhere** — the bundle is 14 KB and self-contained. The end
   user needs only Python 3.10+ and qai. No torch, no sklearn, no
   numpy extras — qai ships them all.

### Measured performance (after 150 training episodes)

| Puzzle    | Random agent | EVOLVE (eval, no learning) | Improvement |
|-----------|-------------|-----------------------------|-------------|
| maze      | -1.50       | -3.40 (still struggling)    | (small puzzle, hard) |
| sequence  | -6.00       | +5.83                       | +11.83 reward |
| resource  | +23.00      | +24.75                      | +1.75 reward (already near-optimal) |
| avoid     | +15.00      | +6.75 (variance high)       | varies |

Total: 43 puzzles solved out of 150 episodes (29% solve rate from
scratch). Performance slope after 150 episodes: -0.11 (the agent
detected a plateau and was actively self-tuning to break out).

### Bundle details

- **Size:** 26.6 KB compressed (was 14.6 KB without the UI)
- **Contents:** manifest.json, agent.pkl (34.7 KB uncompressed), logic.py (8.2 KB),
  **dashboard.html (36.9 KB)**, **launch_ui.py (13.6 KB)**, logic_meta.json, README.md
- **Signed:** HMAC-SHA256 with key `evolve-secret-2026`. The dashboard + launcher
  are included in the signature payload — tampering with any file invalidates it.
- **Self-contained:** runs with just Python + qai, no extra libraries
  (the HTTP server is Python stdlib `http.server`, no Flask/FastAPI/uvicorn)

## How this showcases qai

EVOLVE exercises the following qai features end-to-end:

- ✅ `qai.techniques.tabular_policy.TabularPolicy` (the agent's brain)
- ✅ `qai.logic` decorator (6 bundled logic functions = the agent's personality)
- ✅ `qai.memory.Memory` + `WorkingMemory` (episode log + curiosity tracking)
- ✅ `qai.ResourceBudget` (CPU-only training, 512MB RAM, 240s time)
- ✅ `qai.GridWorld` pattern (extended into PuzzleWorld with 4 puzzle types)
- ✅ Custom bundle format using the `qai.bundle` ZIP + manifest + HMAC pattern
- ✅ `qai.save_logic_source` pattern (logic source shipped inside the bundle)
- ✅ Online learning (the Q-table updates during deployment, not just training)
- ✅ Self-modifying hyperparameters (epsilon and lr self-tune via @qai.logic)
- ✅ Curiosity-driven exploration (intrinsic motivation via state-recency)
- ✅ Multi-task learning (one brain, four puzzle types, single Q-table)

## Honest limitations

See `docs/WHATS_MISSING.md` for the full list of qai gaps this build
exposed. The headline items:

- `qai.memory.Memory` is not picklable (closure-bound callback)
- `qai.save_bundle` only supports single-technique `qai.Model` objects
- `TabularPolicy.forward()` requires `actions=` (no default)
- No `qai.Agent` base class
- No `qai.Environment` ABC + registry
- No `HyperparameterScheduler` utility

Each is a small, well-scoped change. Together they would collapse the
EVOLVE showcase from ~700 lines to ~200.

## Try it

```bash
# 1. Train (one-time, ~1 minute)
cd showcase/evolve
PYTHONPATH=/path/to/qai/repo python train_and_bundle.py

# 2. Launch the visual dashboard (opens your browser)
python launch_ui.py

# 3. Or run the CLI demo
python demo.py --episodes 30 --puzzle random

# 4. Inspect the bundle
unzip -p bundle/evolve.qmodel README.md
unzip -p bundle/evolve.qmodel logic.py
unzip -p bundle/evolve.qmodel dashboard.html > /tmp/dash.html && open /tmp/dash.html
```

Watch the agent's epsilon climb when it realises it's failing, then
drop when it finds a winning strategy. That's the agent thinking —
live, in your browser, from a 26 KB file.
