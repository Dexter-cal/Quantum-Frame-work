"""
evolve_logic.py
===============
The EVOLVE agent's behavior, encoded as @qai.logic functions.

These functions are bundled into the .qmodel file and re-executed when
the user loads the bundle. They give the agent its "personality" --
the rules by which it decides, reflects, and explains itself.

Each function's source code is human-readable in the bundle's logic.py
file. Users can open it and see exactly what the agent is "thinking".

The functions assume a global `agent` (an EvolveAgent instance) is
available in the execution namespace -- this is provided by the bundle
loader.
"""
from __future__ import annotations
import numpy as np
import qai


# =====================================================================
# 1. decide_action -- the agent's action-selection personality
# =====================================================================
@qai.logic(name="decide_action",
            max_seconds=2.0,
            description="Pick the next action: exploit known-good or explore?",
            tags=["decision", "action"],
            author="evolve",
            requires=["agent"])
def decide_action(state, actions):
    """Decide what action to take in `state` given available `actions`.

    The agent uses an epsilon-greedy policy with a curiosity boost: if
    the state is unfamiliar (rarely visited), it explores more; if the
    state is familiar and the agent has a known good action, it exploits.

    Returns the chosen action (a string or int from `actions`).
    """
    # Curiosity: how novel is this state?
    if state not in agent.state_last_seen:
        curiosity = 0.3  # never seen -> max curiosity
    else:
        import time
        age_s = time.time() - agent.state_last_seen[state]
        curiosity = 0.3 * np.exp(-age_s / 60.0)

    effective_epsilon = min(1.0, agent.current_epsilon + curiosity)
    if np.random.random() < effective_epsilon:
        # EXPLORE: pick a random action
        return actions[int(np.random.randint(len(actions)))]
    # EXPLOIT: pick the action with the highest Q-value
    q_values = [agent.brain.agent._get_q(state, a) for a in actions]
    return actions[int(np.argmax(q_values))]


# =====================================================================
# 2. reflect_on_episode -- the agent's self-reflection after each episode
# =====================================================================
@qai.logic(name="reflect_on_episode",
            max_seconds=5.0,
            description="After an episode, decide how to adapt hyperparameters",
            tags=["reflection", "self-tuning", "metacognition"],
            author="evolve",
            requires=["agent"])
def reflect_on_episode(episode_summary):
    """Look at the just-finished episode and decide how to adapt.

    Updates agent.current_epsilon and agent.current_lr in-place based on
    the agent's recent performance trajectory.

    Returns a dict describing the adaptation decision (for logging).
    """
    if len(agent.recent_rewards) < 5:
        return {"decision": "no_adaptation", "reason": "not enough history"}

    slope = agent.performance_slope
    decisions = []

    # Rule 1: stuck (declining) -> explore more
    if slope < -0.1 and agent.current_epsilon < 0.5:
        agent.current_epsilon = min(agent.epsilon_max,
                                    agent.current_epsilon + 0.02)
        decisions.append({"action": "raise_epsilon",
                          "new_value": agent.current_epsilon,
                          "reason": f"declining slope {slope:.3f}"})
    # Rule 2: improving -> exploit more
    elif slope > 0.1 and agent.current_epsilon > 0.1:
        agent.current_epsilon = max(agent.epsilon_min,
                                    agent.current_epsilon - 0.02)
        decisions.append({"action": "lower_epsilon",
                          "new_value": agent.current_epsilon,
                          "reason": f"improving slope {slope:.3f}"})

    # Rule 3: plateau -> raise LR
    if abs(slope) < 0.05 and len(agent.recent_rewards) >= 10:
        if agent.current_lr < agent.lr_max:
            agent.current_lr = min(agent.lr_max, agent.current_lr + 0.01)
            decisions.append({"action": "raise_lr",
                              "new_value": agent.current_lr,
                              "reason": "plateau"})
    # Rule 4: strong improvement -> slow down LR for stability
    elif slope > 0.3 and agent.current_lr > agent.lr_min:
        agent.current_lr = max(agent.lr_min, agent.current_lr - 0.005)
        decisions.append({"action": "lower_lr",
                          "new_value": agent.current_lr,
                          "reason": "strong improvement, consolidating"})

    if not decisions:
        return {"decision": "hold", "reason": "no rule triggered"}
    return {"decision": "adapted", "actions": decisions}


# =====================================================================
# 3. propose_strategy -- the agent's strategic-musing personality
# =====================================================================
@qai.logic(name="propose_strategy",
            max_seconds=3.0,
            description="Pick which puzzle to focus on next based on weakness",
            tags=["strategy", "planning"],
            author="evolve",
            requires=["agent"])
def propose_strategy():
    """Pick which puzzle type to focus on next.

    Strategy: the agent is most curious about the puzzle where its
    recent average reward is lowest (it has the most to learn), UNLESS
    one puzzle is actively declining (in which case it goes there to
    diagnose the regression).

    Returns the puzzle name to focus on.
    """
    if not agent.skill_by_puzzle:
        return None  # no data yet, let the caller pick randomly
    # Find the weakest puzzle
    weakest = min(agent.skill_by_puzzle.items(), key=lambda kv: kv[1])
    return weakest[0]


# =====================================================================
# 4. explain_self -- the agent's introspective narrator
# =====================================================================
@qai.logic(name="explain_self",
            max_seconds=2.0,
            description="Plain-English self-description for the user",
            tags=["explanation", "introspection"],
            author="evolve",
            requires=["agent"])
def explain_self():
    """Narrate the agent's current state in plain English.

    Returns a multi-line string the user can read to understand what
    the agent is feeling, how it's doing, and what it plans to do next.
    """
    s = agent.status()
    lines = [
        f"=== {s['name']} ===",
        f"I've completed {s['episodes_completed']} episodes and solved "
        f"{s['puzzles_solved']} puzzles.",
        f"My total reward is {s['total_reward']}.",
        f"My current exploration rate (epsilon) is {s['current_epsilon']}, "
        f"and my learning rate is {s['current_lr']}.",
        f"My performance trend is {s['performance_slope']:+.4f} "
        f"({'improving' if s['performance_slope'] > 0.1 else 'declining' if s['performance_slope'] < -0.1 else 'plateauing'}).",
        "",
        "My skill by puzzle (avg recent reward):",
    ]
    for p, r in s["skill_by_puzzle"].items():
        verdict = "mastered" if r > 10 else "learning" if r > 0 else "struggling"
        lines.append(f"  {p:10s} {r:+.2f}  ({verdict})")
    # Plan
    target = propose_strategy()
    if target is not None:
        lines.append("")
        lines.append(f"Next I want to focus on '{target}' -- that's where I have the most to learn.")
    return "\n".join(lines)


# =====================================================================
# 5. should_experiment -- the agent's curiosity gate
# =====================================================================
@qai.logic(name="should_experiment",
            max_seconds=1.0,
            description="Decide if the agent should try a random action right now",
            tags=["curiosity", "exploration"],
            author="evolve",
            requires=["agent"])
def should_experiment(state):
    """Return True if the agent should experiment (pick a random action)
    in the given state, based on curiosity.

    A state is 'worth experimenting in' if:
      - we've never seen it, OR
      - we haven't seen it in > 60 seconds, OR
      - we've seen it < 3 times total
    """
    if state not in agent.state_last_seen:
        return True
    import time
    age_s = time.time() - agent.state_last_seen[state]
    if age_s > 60.0:
        return True
    if agent.n_state_visits.get(state, 0) < 3:
        return True
    return False


# =====================================================================
# 6. self_critique -- the agent's inner critic
# =====================================================================
@qai.logic(name="self_critique",
            max_seconds=3.0,
            description="Critique recent performance and propose improvements",
            tags=["critique", "metacognition"],
            author="evolve",
            requires=["agent"])
def self_critique():
    """The agent reviews its last 10 episodes and writes a candid self-critique.

    Returns a string with: what went well, what went badly, and a
    concrete proposal for the next 10 episodes.
    """
    recent = agent.recent_episodes(10)
    if not recent:
        return "I haven't done anything yet -- no basis for self-critique."
    by_puzzle = {}
    for ep in recent:
        by_puzzle.setdefault(ep["puzzle"], []).append(ep)
    lines = ["=== Self-critique (last 10 episodes) ==="]
    for puzzle, eps in by_puzzle.items():
        rewards = [e["total_reward"] for e in eps]
        avg = float(np.mean(rewards))
        solved = sum(1 for e in eps if e["solved"])
        if avg > 5:
            verdict = "going well"
        elif avg > 0:
            verdict = "okay but room to improve"
        else:
            verdict = "going badly"
        lines.append(f"  {puzzle:10s} avg={avg:+.2f} solved={solved}/{len(eps)}  -- {verdict}")
    # Proposal
    target = propose_strategy()
    if target is not None:
        lines.append("")
        lines.append(f"Proposal: focus the next 10 episodes on '{target}' to lift my weakest skill.")
    return "\n".join(lines)
