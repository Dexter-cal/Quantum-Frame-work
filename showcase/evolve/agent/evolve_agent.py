"""
evolve_agent.py
===============
The EVOLVE agent -- a self-improving digital organism.

EVOLVE wraps qai's existing TabularPolicy (Q-learning) brain in a richer
agent loop that adds:

  1. ONLINE LEARNING -- the agent's Q-table keeps updating DURING
     deployment, not just at training time. The user can watch it get
     smarter as it solves puzzles.

  2. CURIOSITY-DRIVEN EXPLORATION -- the agent tracks how novel each
     state is (how recently it was visited). When in a stale state, it
     explores; when in a familiar state with a known good action, it
     exploits.

  3. SELF-MODIFYING HYPERPARAMETERS -- after every episode, the agent's
     reflector (@qai.logic) looks at the reward trajectory and adjusts
     epsilon (exploration rate) and learning_rate. If performance is
     plateauing, epsilon rises; if performance is improving fast, epsilon
     falls. THIS IS THE KEY "WEIGHTS AREN'T FIXED" FEATURE -- the agent
     literally rewrites its own learning rules.

  4. MEMORY -- uses qai.memory.Memory to log every episode: state,
     action, reward, outcome. The reflector queries this memory to
     detect patterns ("I keep failing the maze puzzle").

  5. PERSONALITY VIA @qai.logic -- the agent's behavior is encoded as
     @qai.logic functions: decide_action, reflect_on_episode,
     propose_strategy_change, explain_self. These get bundled with the
     model -- users can read them to see "what the agent is thinking".

The agent is initialised untrained (random Q-table). When the user
loads the bundled .qmodel and runs it, the agent starts learning FROM
SCRATCH on whatever puzzles they expose it to. The bundle is a SEED,
not a finished product. That's the revolutionary part.
"""
from __future__ import annotations
import sys, os
import time
import math
import json
import random
import pickle
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make qai importable -- the showcase lives inside the qai repo so we
# can do a relative-ish import.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import qai
from qai.techniques.tabular_policy import TabularPolicy


# =====================================================================
# EVOLVE agent
# =====================================================================
class EvolveAgent:
    """A self-improving digital organism.

    Constructor args:
        epsilon:       initial exploration rate (0..1). The agent will
                       self-tune this over time.
        learning_rate: initial Q-learning step size (0..1). Self-tuned.
        gamma:         discount factor for future rewards (fixed).
        memory_size:   how many recent episodes to remember.
        name:          the agent's name (cosmetic, used in bundles).

    Attributes the user can inspect after every interaction:
        agent.brain             -- the underlying TabularPolicy
        agent.episodes          -- list of episode summaries
        agent.current_epsilon   -- the (self-tuned) exploration rate
        agent.current_lr        -- the (self-tuned) learning rate
        agent.n_puzzles_solved  -- cumulative count of solved episodes
        agent.total_reward      -- cumulative reward across all episodes
        agent.skill_by_puzzle   -- {puzzle_type: avg_reward} running avg
    """

    def __init__(self,
                 epsilon: float = 0.3,
                 learning_rate: float = 0.1,
                 gamma: float = 0.95,
                 memory_size: int = 200,
                 name: str = "evolve"):
        self.name = name
        self.brain = TabularPolicy(gamma=gamma, epsilon=epsilon,
                                   learning_rate=learning_rate)
        self.current_epsilon = epsilon
        self.current_lr = learning_rate
        self.gamma = gamma

        # Episode log + working memory.
        self.episodes: List[Dict[str, Any]] = []
        self.recent_rewards: deque = deque(maxlen=20)  # last 20 episode rewards
        self.recent_by_puzzle: Dict[str, deque] = {}

        # Curiosity: how recently was each state visited?
        self.state_last_seen: Dict[Tuple, float] = {}
        self.n_state_visits: Dict[Tuple, int] = {}

        # Skill tracking.
        self.n_puzzles_solved = 0
        self.total_reward = 0.0
        self.skill_by_puzzle: Dict[str, float] = {}

        # qai memory system -- three tiers.
        self.memory = qai.Memory()

        # Hyperparameter self-tuning bounds.
        self.epsilon_min = 0.05
        self.epsilon_max = 0.7
        self.lr_min = 0.02
        self.lr_max = 0.4

        # The agent's "self-model" -- tracks its own performance trend.
        # Slope > 0 = improving, slope < 0 = declining.
        self.performance_slope = 0.0

        # A flag the @qai.logic reflector can set to force a strategy
        # reset (e.g. wipe Q-table for one puzzle type).
        self._reset_pending: Optional[str] = None

    # -----------------------------------------------------------------
    # Online perception-action-learning loop
    # -----------------------------------------------------------------
    def run_episode(self, env, max_steps: int = 30,
                    puzzle_idx: Optional[int] = None,
                    learn: bool = True) -> Dict[str, Any]:
        """Run a single episode in the environment.

        Args:
            env: a PuzzleWorld instance (or any env with reset/ACTIONS/run).
            max_steps: hard cap on steps per episode.
            puzzle_idx: optional puzzle-type index (else random).
            learn: if True, update the Q-table during the episode
                   (online learning -- this is the key feature).

        Returns:
            Episode summary dict.
        """
        state = env.reset(puzzle_idx=puzzle_idx)
        puzzle_name = env.current_puzzle_name()
        actions = list(env.ACTIONS)
        total_reward = 0.0
        steps = 0
        solved = False
        trajectory: List[Tuple] = []

        # Curiosity boost: if we haven't seen this state recently,
        # temporarily raise epsilon for this episode's first step.
        curiosity_boost = self._curiosity(state)

        for step in range(max_steps):
            # Mark state visit
            self.state_last_seen[state] = time.time()
            self.n_state_visits[state] = self.n_state_visits.get(state, 0) + 1

            # Choose action via Q-learning (with current epsilon + curiosity boost)
            self.brain.agent.epsilon = min(1.0, self.current_epsilon + curiosity_boost)
            self.brain.agent.lr = self.current_lr
            action = self.brain.agent.choose_action(state, actions)

            # Execute
            outcome = env.run(action)
            next_state = outcome["next_state"]
            reward = outcome["reward"]
            done = outcome["done"]

            # Online learning: update Q-table IN REAL TIME
            if learn:
                self.brain.agent.update(state, action, outcome, actions)

            # Log trajectory
            trajectory.append({
                "state": state, "action": action, "reward": reward,
                "next_state": next_state, "done": done,
            })

            total_reward += reward
            steps += 1
            state = next_state
            if done:
                # Positive terminal reward => "solved" the puzzle
                if reward > 0:
                    solved = True
                break

        # Episode summary
        summary = {
            "puzzle": puzzle_name,
            "steps": steps,
            "total_reward": total_reward,
            "solved": solved,
            "epsilon_used": self.brain.agent.epsilon,
            "lr_used": self.brain.agent.lr,
            "trajectory_length": len(trajectory),
        }
        self.episodes.append(summary)
        if len(self.episodes) > 1000:
            self.episodes = self.episodes[-1000:]
        self.recent_rewards.append(total_reward)
        self.recent_by_puzzle.setdefault(puzzle_name, deque(maxlen=20)).append(total_reward)
        self.total_reward += total_reward
        if solved:
            self.n_puzzles_solved += 1

        # Update skill-by-puzzle running average
        if puzzle_name in self.recent_by_puzzle:
            rewards = list(self.recent_by_puzzle[puzzle_name])
            self.skill_by_puzzle[puzzle_name] = float(np.mean(rewards))

        # Compute performance trend (slope of recent rewards)
        if len(self.recent_rewards) >= 5:
            r = list(self.recent_rewards)
            x = np.arange(len(r))
            try:
                slope = float(np.polyfit(x, r, 1)[0])
                self.performance_slope = slope
            except Exception:
                pass

        # Store episode in qai memory (working tier)
        self.memory.working.add(
            text=f"episode puzzle={puzzle_name} reward={total_reward:.2f} solved={solved}",
            importance=float(total_reward),
            metadata={
                "type": "episode",
                "puzzle": puzzle_name,
                "reward": total_reward,
                "solved": solved,
                "steps": steps,
                "ts": time.time(),
            },
        )

        # Self-reflection: tune hyperparameters
        self._reflect_and_adapt()

        return summary

    # -----------------------------------------------------------------
    # Curiosity: how novel is this state?
    # -----------------------------------------------------------------
    def _curiosity(self, state: Tuple) -> float:
        """Return a curiosity boost in [0, 0.3].

        States visited long ago (or never) get the max boost; states
        visited recently get ~0. This is the agent's intrinsic motivation
        to explore unfamiliar situations.
        """
        if state not in self.state_last_seen:
            return 0.3
        age_s = time.time() - self.state_last_seen[state]
        # Exponential decay: 1 minute => ~0.1 boost, 10 minutes => ~0
        boost = 0.3 * math.exp(-age_s / 60.0)
        return boost

    # -----------------------------------------------------------------
    # Self-reflection + hyperparameter tuning
    # -----------------------------------------------------------------
    def _reflect_and_adapt(self):
        """The agent's reflector: looks at recent performance and
        self-tunes epsilon and learning_rate.

        Rules (encoded in plain Python here, also re-encoded as
        @qai.logic in evolve_logic.py for the bundle):

          * If performance is declining (slope < -0.1) AND epsilon is
            low -> raise epsilon (we're stuck, explore more).
          * If performance is improving (slope > 0.1) AND epsilon is
            high -> lower epsilon (we're on a good track, exploit).
          * If performance is plateauing (|slope| < 0.05) AND we've
            done >= 10 episodes -> raise learning_rate (try to break
            out of the plateau faster).
          * If a specific puzzle is consistently failing (avg reward
            < -2 for >= 5 episodes) -> wipe that puzzle's Q-table
            entries (forced re-learning).
        """
        if len(self.recent_rewards) < 5:
            return  # not enough data to reflect

        slope = self.performance_slope

        # Rule 1: stuck -> explore more
        if slope < -0.1 and self.current_epsilon < 0.5:
            self.current_epsilon = min(self.epsilon_max,
                                       self.current_epsilon + 0.02)
        # Rule 2: improving -> exploit more
        elif slope > 0.1 and self.current_epsilon > 0.1:
            self.current_epsilon = max(self.epsilon_min,
                                       self.current_epsilon - 0.02)

        # Rule 3: plateau -> raise LR
        if abs(slope) < 0.05 and len(self.recent_rewards) >= 10:
            if self.current_lr < self.lr_max:
                self.current_lr = min(self.lr_max, self.current_lr + 0.01)
        # If improving strongly, can afford to slow down LR for stability
        elif slope > 0.3 and self.current_lr > self.lr_min:
            self.current_lr = max(self.lr_min, self.current_lr - 0.005)

        # Rule 4: wipe consistently-failing puzzle's Q-entries
        for puzzle_name, rewards in self.recent_by_puzzle.items():
            if len(rewards) >= 5:
                avg = float(np.mean(list(rewards)[-5:]))
                if avg < -2.0:
                    self._wipe_puzzle_from_qtable(puzzle_name)
                    # Reset that puzzle's history so we don't keep wiping
                    self.recent_by_puzzle[puzzle_name].clear()

    def _wipe_puzzle_from_qtable(self, puzzle_name: str):
        """Wipe all Q-table entries whose state starts with `puzzle_name`."""
        # Q-table keys are (state, action) tuples; state[0] is the puzzle type
        keys_to_remove = [k for k in self.brain.agent.q_table.keys()
                          if isinstance(k[0], tuple) and len(k[0]) > 0
                          and k[0][0] == puzzle_name]
        for k in keys_to_remove:
            del self.brain.agent.q_table[k]

    def to_pickleable(self) -> "EvolveAgent":
        """Return a copy of self that can be pickled.

        Previously this was a workaround for qai.memory.Memory not being
        picklable (closure-bound auto_promote_callback). That bug is now
        fixed in qai itself via __getstate__/__setstate__, so this method
        is a thin pass-through. Kept for backwards compatibility with
        code that calls it explicitly.
        """
        return self

    @classmethod
    def from_loaded(cls, pickled_agent: "EvolveAgent") -> "EvolveAgent":
        """Restore an agent loaded from a bundle. With the qai.memory
        picklability fix, this is now a no-op pass-through. Kept for
        backwards compatibility."""
        return pickled_agent

    # -----------------------------------------------------------------
    # Introspection / explanation
    # -----------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """Snapshot of the agent's current state for the user."""
        return {
            "name": self.name,
            "episodes_completed": len(self.episodes),
            "puzzles_solved": self.n_puzzles_solved,
            "total_reward": round(self.total_reward, 2),
            "current_epsilon": round(self.current_epsilon, 3),
            "current_lr": round(self.current_lr, 3),
            "performance_slope": round(self.performance_slope, 4),
            "q_table_size": len(self.brain.agent.q_table),
            "skill_by_puzzle": {k: round(v, 2) for k, v in self.skill_by_puzzle.items()},
            "states_seen": len(self.n_state_visits),
            "memory_items": len(self.memory.working.all_items()) if hasattr(self.memory.working, 'all_items') else 0,
        }

    def recent_episodes(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.episodes[-n:]

    def explain_self(self) -> str:
        """Plain-English self-description, derived from current state."""
        s = self.status()
        lines = [
            f"=== {s['name']} ===",
            f"Episodes completed: {s['episodes_completed']}",
            f"Puzzles solved:     {s['puzzles_solved']}",
            f"Total reward:       {s['total_reward']}",
            f"Current epsilon:    {s['current_epsilon']}  (exploration rate)",
            f"Current LR:         {s['current_lr']}  (learning rate)",
            f"Performance trend:  {s['performance_slope']:+.4f}  (positive = improving)",
            f"Q-table size:       {s['q_table_size']} entries",
            f"Distinct states:    {s['states_seen']}",
            "",
            "Skill by puzzle (avg recent reward):",
        ]
        for p, r in s["skill_by_puzzle"].items():
            lines.append(f"  {p:10s} {r:+.2f}")
        # Self-assessment
        if s["performance_slope"] > 0.1:
            lines.append("\nI'm improving -- I'll keep doing what's working.")
        elif s["performance_slope"] < -0.1:
            lines.append("\nI'm getting worse -- I'll explore more to find a better strategy.")
        else:
            lines.append("\nI'm plateauing -- I'll raise my learning rate to break out.")
        return "\n".join(lines)


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    # Add the environments path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from environments.puzzle_world import PuzzleWorld

    print("=== EVOLVE agent self-test ===")
    print("Training 60 episodes from scratch...\n")
    env = PuzzleWorld(seed=42)
    agent = EvolveAgent(epsilon=0.3, learning_rate=0.1)

    # Train: 60 random-puzzle episodes
    for i in range(60):
        agent.run_episode(env, max_steps=30, puzzle_idx=None, learn=True)

    print(agent.explain_self())
    print()
    print("--- last 5 episodes ---")
    for ep in agent.recent_episodes(5):
        print(f"  puzzle={ep['puzzle']:8s} steps={ep['steps']:2d} "
              f"reward={ep['total_reward']:+.2f} solved={ep['solved']}")

    print("\n--- now evaluate (no learning) on each puzzle type ---")
    for puzzle_name, idx in [("maze", 0), ("sequence", 1),
                              ("resource", 2), ("avoid", 3)]:
        # Run 5 evaluation episodes
        rewards = []
        for _ in range(5):
            ep = agent.run_episode(env, max_steps=30, puzzle_idx=idx, learn=False)
            rewards.append(ep["total_reward"])
        print(f"  {puzzle_name:10s} eval rewards: {[round(r,2) for r in rewards]}  "
              f"avg={np.mean(rewards):+.2f}")
