# EVOLVE Showcase — What's Missing in qai

Building EVOLVE end-to-end exposed real gaps in qai. This document is
the honest "what would have made this easier" list — features qai
doesn't have yet but should.

## Bugs found while building EVOLVE

### 1. `qai.memory.Memory` is not picklable

**Symptom:** `pickle.dumps(agent)` fails with
`AttributeError: Can't get local object 'Memory._setup_auto_promotion.<locals>.promote'`.

**Root cause:** `qai.memory.Memory._setup_auto_promotion` creates a
closure-bound `promote` function and stores it as an instance attribute
(`auto_promote_callback`). `pickle` cannot serialize closures.

**Why it matters:** Any agent that uses `qai.memory.Memory` cannot be
saved with the standard `qai.save_bundle` flow. Users have to choose
between "use memory" and "ship a bundle". This is exactly the
combination the spec promises (Section 41 + Section 76).

**Workaround used in EVOLVE:** I added a `to_pickleable()` method on
`EvolveAgent` that strips `self.memory` before pickling and rebuilds it
on load (`from_loaded`). The episode log is preserved (it's the actual
learned data); only the live memory object is reset.

**Proposed fix:** `qai.memory.Memory` should either (a) make
`auto_promote_callback` a class-level configurable function (not a
closure), or (b) implement `__getstate__` / `__setstate__` to skip the
callback on serialization and rebuild it on deserialization. Option (b)
is the smaller change.

### 2. `qai.save_bundle` only supports `qai.Model` objects, not composites

**Symptom:** `qai.save_bundle(my_agent, ...)` raises
`AttributeError: 'EvolveAgent' has no attribute 'technique'` inside
`_extract_weights`.

**Root cause:** `qai.bundle._extract_weights` does
`technique = getattr(model, "technique", None)` and bails if there's no
`technique` attribute. It assumes the object being bundled is always a
single `qai.Model` wrapping a single `Technique`.

**Why it matters:** Real agents are composites. EVOLVE has a
`TabularPolicy` brain + a `Memory` + a list of `@qai.logic` functions
+ a hyperparameter self-tuner. None of those fit the "single technique"
shape.

**Workaround used in EVOLVE:** I wrote a custom bundle format
(`save_evolve_bundle` / `load_evolve_bundle` in `train_and_bundle.py`)
that uses the same ZIP + manifest + HMAC pattern as `qai.bundle` but
stores the agent as a pickle instead of as a weights dict. This works
but means EVOLVE doesn't get to reuse `qai.bundle`'s tamper-detection
or `inspect_bundle` — those are coupled to the "single technique"
shape too.

**Proposed fix:** `qai.save_bundle` should accept any object that
either (a) is a `qai.Model`, OR (b) provides a `_qai_bundle_state()`
method returning a dict of picklable fields. The bundle format already
supports a "pickle_fallback" weight format — extend it to whole-object
fallback.

### 3. `qai.techniques.tabular_policy.TabularPolicy` cannot be evaluated without an `actions` list

**Symptom:** `model.predict(state)` raises
`ValueError: TabularPolicy.forward(state, actions=...) requires the action list`.

**Root cause:** `TabularPolicy.forward` has a hard requirement for
`actions=` because the Q-table doesn't store the action space — it
only stores `(state, action) -> Q` pairs. Without the action list,
there's no way to enumerate `argmax_a Q(s, a)`.

**Why it matters:** Any environment-agnostic deployment (REST serving,
CLI inference, embedded inference) breaks because the caller has to
somehow know the action space at every call site. For EVOLVE's 4
puzzle types, each has a different action space — the caller has to
dispatch on `state[0]` to figure out which action list to pass.

**Workaround used in EVOLVE:** `EvolveAgent.run_episode` reads
`env.ACTIONS` directly and passes it through. Works for training +
in-environment evaluation, but breaks if someone tries to call
`agent.brain.predict(state)` from outside the environment.

**Proposed fix:** `TabularPolicy.__init__` should accept an optional
`actions=` parameter that, if set, is stored and used as the default
in `forward()`. For multi-task agents, allow `actions` to be a dict
keyed by state-prefix. This is a 5-line change.

## Features that would have helped

### 4. No "agent base class"

qai has `qai.Model` (single technique) and `qai.AIAgent` (the
spec-section-87 agent), but nothing in between for "an agent that
wraps a `Technique` + adds a perception-action loop + memory +
self-reflection". I had to write `EvolveAgent` from scratch.

**Proposed addition:** `qai.Agent` — a base class with:
- `run_episode(env, learn=True)` 
- `reflect()` (overridable)
- `self_tune()` (overridable)
- `to_pickleable()` / `from_loaded()` (so it round-trips through bundles)
- A `memory: qai.memory.Memory` slot
- A `brain: qai.techniques.Technique` slot

This would let users build EVOLVE-like agents in 50 lines instead of 300.

### 5. No environment interface / registry

`qai.GridWorld` exists but is the only built-in environment. There's
no `qai.Environment` base class, no `register_environment()`, no
`qai.environments` sub-package. Every agent author has to re-invent
the `reset()` / `observe()` / `run(action)` / `ACTIONS` contract.

**Proposed addition:** `qai.Environment` ABC with `reset()`, `step(action)`,
`observe()`, `available_actions()`, `done`. Plus a small registry of
canonical environments: `GridWorld`, `Bandit`, `MountainCar` (small
discrete version), `PuzzleWorld` (this showcase!). Without this, every
RL example in qai has to start by writing its own environment.

### 6. No online-learning utility on top of `TabularPolicy`

`TabularPolicy.fit()` trains for N episodes then returns. There's no
`partial_fit()` or `update_step(state, action, reward, next_state)` —
the agent has to call `self.agent.update(...)` directly, reaching
through two layers of attribute access (`brain.agent.update`).

**Proposed addition:** A `qai.OnlineLearner` mixin that adds
`update_step(state, action, outcome)` to any technique. Plus a
`partial_fit(trajectory)` method for batch-online learning (replay a
stored trajectory).

### 7. `@qai.logic` functions can't easily reference the agent they belong to

The `@qai.logic` source is captured by `inspect.getsource()` and
re-executed in a fresh namespace on bundle load. If the function
references `agent` (a global), the loader has to manually inject
`agent` into the exec namespace. This works but is undocumented — I
had to read `qai.bundle.load_bundle` source to figure it out.

**Proposed addition:** `@qai.logic` should accept an optional
`bind_to="agent"` parameter that tells the bundle loader "this
function expects a global named `agent` to be injected; do it
automatically". Alternatively, `@qai.logic` functions could accept
`self` as the first arg (like methods) and the loader binds `self`
to the bundled object automatically.

### 8. No `ResourceBudget` integration with `qai.train` flows

`qai.ResourceBudget` exists and works, but `qai.build().train()` doesn't
check it. The user has to manually call `budget.can_load(model)` and
`budget.time_exceeded()` in their own training loop. EVOLVE does this
in `train_and_bundle.py`, but it should be a one-liner.

**Proposed addition:** `model.train(..., budget=qai.ResourceBudget(...))`.
When `budget` is passed, `train()` checks `budget.time_exceeded()` and
`budget.can_load(self)` after every epoch and stops early if needed.
Returns the partial-training state with a `budget_stopped=True` flag.

### 9. No "agent self-explanation" helper

EVOLVE's `explain_self()` and `self_critique()` are 50 lines of plain
Python. Every agent that wants to be inspectable has to write the same
boilerplate: pull stats from `self.episodes`, format as text, decide
verdict words.

**Proposed addition:** `qai.Agent.explain()` — a default method that
returns a formatted string of episode count, total reward, current
hyperparameters, performance slope, and skill-by-task breakdown. Users
override only the verdict words.

### 10. No "self-tuning hyperparameter" utility

EVOLVE's `_reflect_and_adapt()` is custom code, but the pattern
(epsilon-greedy + LR + self-tune based on reward slope) is universal
across RL agents. There's no `qai.HyperparameterScheduler` or similar.

**Proposed addition:** `qai.HyperparameterScheduler` — tracks a
rolling window of rewards, computes the slope, and applies user-
supplied rules to tune any set of hyperparameters. EVOLVE's
reflector would be a 10-line configuration of this utility instead
of a 60-line method.

### 11. Bundle README doesn't get rendered to the user on `load_bundle`

`qai.save_bundle` writes a `README.md` inside the .qmodel zip, but
`qai.load_bundle` doesn't print it or expose it. The user has to
manually `unzip` the bundle to read it.

**Proposed addition:** `bundle.readme` attribute on the `Bundle` object,
populated from `README.md` if present. Plus a `bundle.print_readme()`
convenience method.

### 12. No multi-puzzle / multi-task environment

This is a feature gap, not a bug. `qai.GridWorld` is a single puzzle.
There's no built-in multi-task environment where one agent has to
learn multiple skills simultaneously. This is exactly the kind of
"small, hard, demonstrably-learnable" benchmark that would showcase
qai's RL capabilities.

**Proposed addition:** Promote `PuzzleWorld` (from this showcase) into
`qai.environments.puzzle_world`. It's a 4-puzzle multi-task benchmark
that exercises Q-learning's ability to learn generalizable strategies.
4 puzzle types × ~200 distinct states × ~4 actions = a non-trivial
Q-table that visibly grows during training.

## Summary

EVOLVE works. The agent learns, self-tunes, and ships in a 14 KB bundle.
But building it required writing ~700 lines of code that should have
been ~200 if qai had:

1. A picklable `qai.memory.Memory`
2. A `qai.save_bundle` that handles composite agents
3. A `qai.Agent` base class
4. A `qai.Environment` ABC + registry
5. Online-learning utilities on `TabularPolicy`
6. Auto-injection of `agent` into `@qai.logic` exec namespaces
7. `ResourceBudget` integration with `model.train`
8. Default `explain()` / `self_critique()` on the agent base class
9. A `HyperparameterScheduler` utility
10. Bundle README exposed on the loaded `Bundle` object
11. A multi-task environment shipped with qai

Each of these is a small, well-scoped change. Together they would
collapse the EVOLVE showcase from ~700 lines to ~200 lines and make
the same pattern available to every qai user.
