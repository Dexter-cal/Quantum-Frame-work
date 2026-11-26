"""
qai.agents_base -- Base class for learning agents + replay buffer +
hyperparameter scheduler.

This module addresses three of the framework gaps identified in
showcase/evolve/docs/FRAMEWORK_CRITIQUE.md:

  1. No `qai.Agent` base class. Every agent author writes the same
     boilerplate (perception-action loop, memory slot, brain slot,
     explain(), to_pickleable(), from_loaded()). This module factors
     that out into AgentBase.

  2. No `qai.ReplayBuffer`. Experience replay is the foundation of
     modern RL since DQN (2013). This module provides a simple
     uniform-replay buffer plus a prioritized variant.

  3. No `qai.HyperparameterScheduler`. EVOLVE's _reflect_and_adapt()
     is custom code, but the pattern (rolling-window reward +
     self-tune based on slope) is universal across RL agents. This
     module provides a configurable scheduler.

Usage:

    class MyAgent(qai.AgentBase):
        def reflect(self, episode_summary):
            # Custom reflection logic
            ...
        def self_tune(self, episode_summary):
            # Custom hyperparameter tuning
            ...

    agent = MyAgent(brain=qai.build(technique="tabular_policy"),
                    memory=qai.Memory(),
                    name="my_agent")
    agent.run_episode(env, learn=True)
    print(agent.explain())
"""
from __future__ import annotations
import time, math, pickle, copy, random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np

# Make qai importable for isinstance checks
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# =====================================================================
# 1. ReplayBuffer -- experience replay for RL agents
# =====================================================================
class ReplayBuffer:
    """A uniform experience replay buffer (DQN-style).

    Stores (state, action, reward, next_state, done) tuples. Sample
    random mini-batches for off-policy training.

    Usage:
        buf = qai.ReplayBuffer(capacity=10000)
        buf.add(state, action, reward, next_state, done)
        if len(buf) >= 32:
            batch = buf.sample(32)
            # batch is a list of (s, a, r, s', done) tuples
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._rng = np.random.default_rng()

    def add(self, state, action, reward: float, next_state, done: bool) -> None:
        """Add a single transition to the buffer."""
        self._buffer.append((state, action, float(reward), next_state, bool(done)))

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample `batch_size` random transitions. Returns a list of tuples."""
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"[WHAT] Replay buffer has only {len(self._buffer)} items, "
                f"need {batch_size}\n"
                f"[FIX] Wait until len(buffer) >= batch_size before sampling"
            )
        idxs = self._rng.integers(0, len(self._buffer), size=batch_size)
        return [self._buffer[i] for i in idxs]

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def summary(self) -> Dict[str, Any]:
        return {"capacity": self.capacity, "size": len(self._buffer),
                "fill_ratio": len(self._buffer) / max(self.capacity, 1)}


class PrioritizedReplayBuffer(ReplayBuffer):
    """A prioritized experience replay buffer (PER, Schaul et al. 2015).

    Transitions are sampled proportional to their priority (typically
    |TD-error| + small epsilon). Important for agents that learn
    non-uniformly from past experience.

    Usage:
        buf = qai.PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
        idx = buf.add(state, action, reward, next_state, done)
        # ... after computing TD-error for this transition:
        buf.update_priority(idx, td_error=2.5)
        # Sample with importance-sampling weights:
        batch, weights, idxs = buf.sample(32)
    """
    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta: float = 0.4, epsilon: float = 1e-6):
        super().__init__(capacity=capacity)
        self.alpha = alpha  # prioritization strength (0=uniform, 1=full)
        self.beta = beta    # importance-sampling correction
        self.epsilon = epsilon
        self._priorities: deque = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done,
            priority: Optional[float] = None) -> int:
        """Add a transition. Returns its index (for later priority updates).

        Args:
            priority: optional initial priority. If None, uses the max
                      current priority (so new transitions are sampled
                      at least once).
        """
        idx = len(self._buffer)
        if priority is None:
            priority = max(self._priorities, default=1.0)
        self._buffer.append((state, action, float(reward), next_state, bool(done)))
        self._priorities.append(float(priority))
        return idx

    def update_priority(self, idx: int, td_error: float) -> None:
        """Update the priority of transition `idx` based on its TD-error."""
        if 0 <= idx < len(self._buffer):
            self._priorities[idx] = abs(float(td_error)) + self.epsilon

    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        """Sample a prioritized batch.

        Returns:
            (batch, weights, idxs) -- batch is a list of transitions,
            weights is an array of importance-sampling weights, idxs is
            an array of indices (for later priority updates).
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"[WHAT] Buffer has {len(self._buffer)} items, need {batch_size}"
            )
        priorities = np.array(self._priorities) ** self.alpha
        probs = priorities / priorities.sum()
        idxs = self._rng.choice(len(self._buffer), size=batch_size, p=probs)
        batch = [self._buffer[i] for i in idxs]
        # Importance-sampling weights
        weights = (len(self._buffer) * probs[idxs]) ** (-self.beta)
        weights = weights / weights.max()  # normalize
        return batch, weights, idxs

    def summary(self) -> Dict[str, Any]:
        s = super().summary()
        s.update({"alpha": self.alpha, "beta": self.beta,
                  "mean_priority": float(np.mean(self._priorities)) if self._priorities else 0.0})
        return s


# =====================================================================
# 2. HyperparameterScheduler -- rolling-window self-tuner
# =====================================================================
class HyperparameterScheduler:
    """Track a rolling window of rewards and apply user-supplied tuning rules.

    Solves the "every RL agent rewrites the same self-tuning code" problem.
    The user supplies:
      - hyperparameters: dict of {name: current_value}
      - rules: list of (condition_fn, action_fn) tuples

    After every episode, call `update(reward)`. The scheduler computes
    the rolling slope and triggers any rules whose condition is met.

    Usage:
        sched = qai.HyperparameterScheduler(
            params={"epsilon": 0.3, "lr": 0.1},
            window=20,
            rules=[
                # If slope is declining AND epsilon is low, raise epsilon
                (lambda s, p: s < -0.1 and p["epsilon"] < 0.5,
                 lambda p: {"epsilon": p["epsilon"] + 0.02}),
                # If slope is improving AND epsilon is high, lower it
                (lambda s, p: s > 0.1 and p["epsilon"] > 0.1,
                 lambda p: {"epsilon": p["epsilon"] - 0.02}),
            ],
        )
        # After every episode:
        sched.update(episode_reward)
        print(sched.params)  # current hyperparameter values
    """
    def __init__(self, params: Dict[str, float],
                 window: int = 20,
                 rules: Optional[List[Tuple[Callable, Callable]]] = None,
                 bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Args:
            params: dict of {name: initial_value} for each hyperparameter.
            window: rolling-window size for slope computation.
            rules: list of (condition_fn, action_fn) where:
                    - condition_fn(slope, params) -> bool
                    - action_fn(params) -> dict of {name: new_value} to merge
            bounds: optional dict of {name: (min, max)} to clip values.
        """
        self.params = dict(params)
        self.window = window
        self.rules = rules or []
        self.bounds = bounds or {}
        self._rewards: deque = deque(maxlen=window)
        self._slope: float = 0.0
        self._n_updates = 0
        self._n_rule_fires = 0

    def update(self, reward: float) -> Dict[str, Any]:
        """Record a new reward, recompute slope, fire any matching rules.

        Returns a dict describing what happened:
          - slope: the new rolling slope
          - rules_fired: list of rule indices that triggered
          - param_changes: dict of {name: (old, new)} for changed params
        """
        self._rewards.append(float(reward))
        self._n_updates += 1

        # Compute slope (linear regression on the rolling window)
        if len(self._rewards) >= 3:
            r = list(self._rewards)
            x = np.arange(len(r))
            try:
                self._slope = float(np.polyfit(x, r, 1)[0])
            except Exception:
                pass  # keep previous slope

        # Fire rules
        rules_fired = []
        param_changes = {}
        for i, (cond_fn, action_fn) in enumerate(self.rules):
            try:
                if cond_fn(self._slope, dict(self.params)):
                    new_vals = action_fn(dict(self.params))
                    for k, v in new_vals.items():
                        if k in self.params:
                            old = self.params[k]
                            # Apply bounds
                            if k in self.bounds:
                                lo, hi = self.bounds[k]
                                v = max(lo, min(hi, v))
                            self.params[k] = v
                            if v != old:
                                param_changes[k] = (old, v)
                    rules_fired.append(i)
                    self._n_rule_fires += 1
            except Exception:
                pass  # rule errors are non-fatal

        return {
            "slope": self._slope,
            "rules_fired": rules_fired,
            "param_changes": param_changes,
            "n_rewards": len(self._rewards),
        }

    @property
    def slope(self) -> float:
        return self._slope

    def summary(self) -> Dict[str, Any]:
        return {
            "params": dict(self.params),
            "slope": self._slope,
            "n_rewards": len(self._rewards),
            "n_updates": self._n_updates,
            "n_rule_fires": self._n_rule_fires,
            "window": self.window,
            "n_rules": len(self.rules),
        }

    def reset(self) -> None:
        """Clear the rolling window (but keep current params)."""
        self._rewards.clear()
        self._slope = 0.0


# =====================================================================
# 3. AgentBase -- the base class for learning agents
# =====================================================================
class AgentBase:
    """Base class for learning agents.

    Subclasses get for free:
      - self.brain: a qai.Model or qai.Technique (the agent's policy)
      - self.memory: a qai.Memory (three-tier memory system)
      - self.episodes: a log of episode summaries
      - self.replay_buffer: a qai.ReplayBuffer (off-policy training)
      - self.scheduler: a qai.HyperparameterScheduler (self-tuning)
      - run_episode(env, learn=True): the perception-action loop
      - reflect(episode_summary): override for custom reflection
      - self_tune(episode_summary): override for custom hyperparam tuning
      - explain(): default plain-English self-description
      - status(): snapshot of agent state
      - to_pickleable() / from_loaded(): for bundle round-trips

    Subclasses typically only need to override:
      - reflect(): what to do after each episode
      - self_tune(): how to adapt hyperparameters
      - explain(): custom narrative (optional)

    Usage:
        class MyAgent(qai.AgentBase):
            def reflect(self, summary):
                if summary["reward"] < 0:
                    print(f"  Bad episode on {summary['puzzle']}")

            def self_tune(self, summary):
                # Use the scheduler's slope to decide
                if self.scheduler.slope < -0.1:
                    self.scheduler.params["epsilon"] += 0.02

        agent = MyAgent(
            brain=qai.build(technique="tabular_policy"),
            memory=qai.Memory(),
            name="my_agent",
            hparams={"epsilon": 0.3, "lr": 0.1},
        )
        agent.run_episode(env, learn=True)
        print(agent.explain())
    """
    def __init__(self,
                 brain: Any = None,
                 memory: Any = None,
                 name: str = "agent",
                 hparams: Optional[Dict[str, float]] = None,
                 hparam_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 hparam_window: int = 20,
                 replay_capacity: int = 10000):
        """
        Args:
            brain: the agent's policy -- typically a qai.Model (from qai.build())
                   or a qai.Technique. Must support either fit(X, y) or
                   fit(environment=) for the chosen learning paradigm.
            memory: a qai.Memory instance (created with defaults if None).
            name: agent name (cosmetic).
            hparams: initial hyperparameters tracked by the scheduler.
            hparam_bounds: optional dict of {name: (min, max)} for clipping.
            hparam_window: rolling-window size for slope computation.
            replay_capacity: capacity of the experience replay buffer.
        """
        self.name = name
        self.brain = brain
        # Import qai lazily to avoid circular imports
        try:
            import qai
            self.memory = memory if memory is not None else qai.Memory()
        except ImportError:
            self.memory = memory

        self.episodes: List[Dict[str, Any]] = []
        self.recent_rewards: deque = deque(maxlen=hparam_window)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        # Hyperparameter scheduler (with no rules by default -- subclasses
        # add rules via self_tune() or by populating self.scheduler.rules)
        hparams = hparams or {}
        self.scheduler = HyperparameterScheduler(
            params=hparams,
            window=hparam_window,
            bounds=hparam_bounds or {},
        )

        # Cumulative stats
        self.total_reward = 0.0
        self.n_episodes_completed = 0
        self.n_successes = 0

    # ------------------------------------------------------------------
    # The perception-action-learning loop (subclasses can override)
    # ------------------------------------------------------------------
    def run_episode(self, env, max_steps: int = 100,
                    learn: bool = True,
                    **kwargs) -> Dict[str, Any]:
        """Run one episode in the environment.

        Args:
            env: an environment with reset() / step() / available_actions() / done().
            max_steps: hard cap on steps per episode.
            learn: if True, update the brain during the episode (online learning).
            **kwargs: passed to the technique's update method.

        Returns:
            Episode summary dict.
        """
        state = env.reset()
        total_reward = 0.0
        steps = 0
        success = False
        actions_taken = []

        for step in range(max_steps):
            available = env.available_actions()
            action = self._choose_action(state, available)
            outcome = env.step(action)
            next_state = outcome.get("next_state", state)
            reward = float(outcome.get("reward", 0.0))
            done = bool(outcome.get("done", False))

            # Store in replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)

            # Online learning
            if learn:
                self._learn_step(state, action, reward, next_state, done,
                                  available, **kwargs)

            actions_taken.append(action)
            total_reward += reward
            steps += 1
            state = next_state
            if done:
                if reward > 0:
                    success = True
                break

        # Episode summary
        summary = {
            "steps": steps,
            "total_reward": total_reward,
            "solved": success,
            "actions_taken": actions_taken,
            "final_state": state,
            "ts": time.time(),
        }
        self.episodes.append(summary)
        if len(self.episodes) > 1000:
            self.episodes = self.episodes[-1000:]
        self.recent_rewards.append(total_reward)
        self.total_reward += total_reward
        self.n_episodes_completed += 1
        if success:
            self.n_successes += 1

        # Update the hyperparameter scheduler
        self.scheduler.update(total_reward)

        # Memory: log to working tier
        if self.memory is not None and hasattr(self.memory, "working"):
            try:
                self.memory.working.add(
                    text=f"episode reward={total_reward:.2f} solved={success}",
                    importance=total_reward,
                    metadata=summary,
                )
            except Exception:
                pass

        # Subclass hooks
        self.reflect(summary)
        self.self_tune(summary)

        return summary

    # ------------------------------------------------------------------
    # Hooks for subclasses to override
    # ------------------------------------------------------------------
    def _choose_action(self, state, available_actions):
        """Pick an action. Override for custom policies (epsilon-greedy,
        softmax, etc.). Default: random."""
        if not available_actions:
            return None
        # If the brain has a choose_action method (TabularPolicy does), use it
        brain = self._get_brain_inner()
        if brain is not None and hasattr(brain, "choose_action"):
            eps = self.scheduler.params.get("epsilon", 0.0)
            if hasattr(brain, "agent"):
                # TabularPolicy path: it has brain.agent.choose_action(state, actions)
                return brain.agent.choose_action(state, available_actions)
            return brain.choose_action(state, available_actions)
        # Fallback: random
        idx = np.random.randint(len(available_actions))
        return available_actions[idx]

    def _learn_step(self, state, action, reward, next_state, done,
                    available_actions, **kwargs):
        """Online learning step. Override for custom update rules.
        Default: dispatch to the brain's update method if it has one."""
        brain = self._get_brain_inner()
        if brain is None:
            return
        # TabularPolicy path
        if hasattr(brain, "agent") and hasattr(brain.agent, "update"):
            outcome = {"reward": reward, "next_state": next_state, "done": done}
            try:
                brain.agent.update(state, action, outcome, available_actions)
            except Exception:
                pass

    def reflect(self, episode_summary: Dict[str, Any]) -> None:
        """Hook: called after every episode. Override for custom reflection.
        Default: no-op."""
        pass

    def self_tune(self, episode_summary: Dict[str, Any]) -> None:
        """Hook: called after every episode. Override for custom
        hyperparameter tuning. Default: no-op (the scheduler's rules
        handle it if any were registered)."""
        pass

    # ------------------------------------------------------------------
    # Introspection / explanation
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """Snapshot of the agent's current state."""
        return {
            "name": self.name,
            "episodes_completed": self.n_episodes_completed,
            "successes": self.n_successes,
            "total_reward": round(self.total_reward, 2),
            "hparams": dict(self.scheduler.params),
            "performance_slope": round(self.scheduler.slope, 4),
            "replay_buffer_size": len(self.replay_buffer),
            "n_recent_rewards": len(self.recent_rewards),
        }

    def explain(self) -> str:
        """Default plain-English self-description. Override for custom narrative."""
        s = self.status()
        slope = s["performance_slope"]
        if slope > 0.1:
            trend = "improving"
        elif slope < -0.1:
            trend = "declining"
        else:
            trend = "plateauing"
        lines = [
            f"=== {s['name']} ===",
            f"Episodes completed: {s['episodes_completed']}",
            f"Successes: {s['successes']}",
            f"Total reward: {s['total_reward']}",
            f"Performance trend: {slope:+.4f} ({trend})",
            f"Hyperparameters: {s['hparams']}",
            f"Replay buffer: {s['replay_buffer_size']} transitions",
        ]
        return "\n".join(lines)

    def recent_episodes(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.episodes[-n:]

    # ------------------------------------------------------------------
    # Pickle support (for bundling)
    # ------------------------------------------------------------------
    def to_pickleable(self) -> "AgentBase":
        """Return a copy of self that can be pickled.

        Default: try pickle.dumps(self); if it fails, strip the memory
        (which may have unpicklable closures) and try again. Subclasses
        with more unpicklable attrs should override.
        """
        try:
            pickle.dumps(self)
            return self
        except Exception:
            pass
        clone = copy.copy(self)
        clone.memory = None
        try:
            pickle.dumps(clone)
            return clone
        except Exception:
            # Last resort: rebuild a fresh agent and copy simple attrs
            fresh = type(self)(brain=self.brain, name=self.name)
            fresh.episodes = list(self.episodes)
            fresh.recent_rewards = self.recent_rewards
            fresh.total_reward = self.total_reward
            fresh.n_episodes_completed = self.n_episodes_completed
            fresh.n_successes = self.n_successes
            fresh.scheduler = self.scheduler
            fresh.replay_buffer = self.replay_buffer
            return fresh

    @classmethod
    def from_loaded(cls, loaded: "AgentBase") -> "AgentBase":
        """Restore an agent after pickle.load(). Rebuilds any state that
        was stripped in to_pickleable().

        Default: rebuild the qai.memory.Memory if it was stripped.
        Subclasses with more stripped state should override.
        """
        if getattr(loaded, "memory", None) is None:
            try:
                import qai
                loaded.memory = qai.Memory()
            except ImportError:
                pass
        return loaded

    # ------------------------------------------------------------------
    # Composite-bundle support (lets qai.save_bundle handle AgentBase
    # subclasses without users writing their own save/load code).
    # ------------------------------------------------------------------
    def _qai_bundle_state(self) -> Dict[str, Any]:
        """Return a picklable state dict for qai.save_bundle's composite path.

        Default: pickle the result of to_pickleable(). Subclasses with
        extra state can override to add fields. The state dict is stored
        as a single blob in the .qmodel file and restored via
        _qai_load_bundle_state().
        """
        return {"__agent_pickle__": pickle.dumps(self.to_pickleable())}

    def _qai_load_bundle_state(self, state: Dict[str, Any]) -> None:
        """Restore state saved by _qai_bundle_state().

        Default: unpickle the agent and copy its attributes onto self.
        Subclasses with custom restore logic can override.
        """
        if "__agent_pickle__" not in state:
            raise ValueError(
                "[WHAT] _qai_load_bundle_state: state dict has no '__agent_pickle__' key\n"
                "[WHY] The bundle was saved with a different _qai_bundle_state() implementation\n"
                "[FIX] Re-save the bundle, or override _qai_load_bundle_state on your subclass"
            )
        loaded = pickle.loads(state["__agent_pickle__"])
        # Restore via from_loaded() (rebuilds stripped state)
        loaded = type(self).from_loaded(loaded)
        # Copy attributes from loaded onto self
        for attr in vars(loaded):
            setattr(self, attr, getattr(loaded, attr))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_brain_inner(self):
        """Return the underlying technique (handles both qai.Model and
        raw Technique)."""
        if self.brain is None:
            return None
        # qai.Model wraps a technique
        if hasattr(self.brain, "technique"):
            return self.brain.technique
        return self.brain

    def __repr__(self):
        s = self.status()
        return f"<{type(self).__name__} name={s['name']!r} episodes={s['episodes_completed']} successes={s['successes']}>"


# =====================================================================
# Public API
# =====================================================================
__all__ = [
    "AgentBase",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "HyperparameterScheduler",
]
