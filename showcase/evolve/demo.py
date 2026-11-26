"""
demo.py
=======
End-user demonstration of the EVOLVE bundle.

This is what you run after downloading evolve.qmodel. It loads the
bundle, lets the agent explain itself, then runs a LIVE learning session
where you can watch the agent get smarter in real time.

Usage:
    python demo.py [--bundle evolve.qmodel] [--sign-key evolve-secret-2026]
                   [--episodes 30] [--puzzle random]

Requirements:
    - Python 3.10+
    - qai (pip install qai)

That's it. No torch, no sklearn, no numpy extras -- qai ships them all.
The bundle is fully self-contained.
"""
from __future__ import annotations
import sys, os, argparse, time

# Make qai importable (in case the user is running from the showcase dir).
# In a real deployment, qai would be `pip install`ed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import qai

# Local imports (the bundle loader + the environment).
sys.path.insert(0, os.path.dirname(__file__))
from train_and_bundle import load_evolve_bundle
from environments.puzzle_world import PuzzleWorld


# =====================================================================
# Pretty-printers
# =====================================================================
def banner(text: str, char: str = "="):
    line = char * 70
    print(f"\n{line}\n{text}\n{line}")


def print_episode_summary(summary: dict, idx: int):
    solved_marker = "✓" if summary["solved"] else " "
    puzzle = summary["puzzle"]
    reward = summary["total_reward"]
    steps = summary["steps"]
    print(f"  ep {idx:>3}  [{solved_marker}]  {puzzle:8s}  "
          f"reward={reward:+6.2f}  steps={steps:>2}")


# =====================================================================
# The demo flow
# =====================================================================
def run_demo(bundle_path: str, sign_key: str, n_episodes: int,
             puzzle_choice: str, watch_learning: bool = True):
    """Load the bundle and run a live demo."""

    banner(f"Loading EVOLVE bundle: {bundle_path}", "*")
    if not os.path.exists(bundle_path):
        print(f"ERROR: bundle not found at {bundle_path}")
        print("Run `python train_and_bundle.py` first to create it.")
        return

    bundle = load_evolve_bundle(bundle_path, sign_key=sign_key or None)
    agent = bundle["agent"]
    logic_fns = bundle["logic_fns"]
    manifest = bundle["manifest"]

    print(f"  format version: {manifest.get('format_version', '?')}")
    print(f"  agent size:     {manifest.get('agent_size_human', '?')}")
    print(f"  signed:         {manifest.get('signed', False)}")
    print(f"  logic functions: {manifest.get('logic_names', [])}")

    banner("STEP 1: The agent introduces itself")
    # Re-inject the agent into the logic functions' namespace
    for fn in logic_fns.values():
        # The logic fns reference a global `agent`; set it via the fn's globals
        try:
            fn.__globals__["agent"] = agent
        except Exception:
            pass
    print(logic_fns["explain_self"]())

    banner("STEP 2: The agent critiques its past performance")
    print(logic_fns["self_critique"]())

    banner("STEP 3: LIVE LEARNING SESSION")
    print(f"Running {n_episodes} new episodes. The agent's Q-table will")
    print("update IN REAL TIME -- you're watching it learn.\n")

    env = PuzzleWorld(seed=int(time.time()))  # different seed each run

    # Pick the puzzle for the demo
    puzzle_map = {"maze": 0, "sequence": 1, "resource": 2, "avoid": 3, "random": None}
    puzzle_idx = puzzle_map.get(puzzle_choice, None)

    print(f"{'ep':>4}  {'solved':>6}  {'puzzle':>8}  {'reward':>7}  "
          f"{'eps':>5}  {'lr':>5}  {'slope':>6}  {'total_reward':>12}")
    print("-" * 70)

    initial_solved = agent.n_puzzles_solved
    initial_total = agent.total_reward

    for i in range(n_episodes):
        # If puzzle is "random" the agent doesn't know which puzzle it'll get --
        # that's the harder, more realistic test.
        ep_puzzle = puzzle_idx if puzzle_idx is not None else None

        summary = agent.run_episode(env, max_steps=30, learn=True, puzzle_idx=ep_puzzle)
        s = agent.status()
        delta_solved = agent.n_puzzles_solved - initial_solved
        delta_total = agent.total_reward - initial_total
        print(f"{i+1:>4}  {delta_solved:>6}  {summary['puzzle']:>8}  "
              f"{summary['total_reward']:>+7.2f}  "
              f"{s['current_epsilon']:>5.3f}  {s['current_lr']:>5.3f}  "
              f"{s['performance_slope']:>+6.3f}  {delta_total:>+12.2f}")

        # The agent reflects after every episode (via the @qai.logic function)
        logic_fns["reflect_on_episode"](summary)

    banner("STEP 4: The agent's updated self-assessment")
    print(logic_fns["explain_self"]())

    banner("STEP 5: The agent's strategy proposal")
    target = logic_fns["propose_strategy"]()
    if target is not None:
        print(f"The agent proposes to focus on: '{target}'")
        print("(This is its weakest puzzle -- it wants to lift that skill next.)")
    else:
        print("The agent hasn't gathered enough data to propose a strategy yet.")

    banner("STEP 6: Final status")
    s = agent.status()
    print(f"Episodes completed: {s['episodes_completed']}")
    print(f"Puzzles solved:     {s['puzzles_solved']} "
          f"(+{s['puzzles_solved'] - initial_solved} during this demo)")
    print(f"Total reward:       {s['total_reward']} "
          f"(+{s['total_reward'] - initial_total:.2f} during this demo)")
    print(f"Final epsilon:      {s['current_epsilon']}")
    print(f"Final LR:           {s['current_lr']}")
    print(f"Performance trend:  {s['performance_slope']:+.4f}")
    print(f"Q-table size:       {s['q_table_size']}")

    banner("What just happened?", "*")
    print("""
You loaded a 14 KB .qmodel bundle and watched a self-improving digital
organism learn in real time. The agent you started with was a SEED --
it had a brain (Q-table) seeded with 150 episodes of prior experience,
but during this demo it:

  * Ran N new episodes against puzzles it had never seen in this exact
    configuration (random seed = current time).
  * Updated its Q-table on every step (online learning -- weights not
    fixed).
  * Self-tuned its exploration rate (epsilon) and learning rate (lr)
    based on its performance trajectory.
  * Got curious about unfamiliar states and explored them more.
  * Reflected on each episode via @qai.logic functions bundled inside
    the .qmodel file.
  * Proposed a strategy for what to focus on next.

The bundle you loaded contained:
  * manifest.json    -- metadata + HMAC signature
  * agent.pkl        -- the agent's brain state (Q-table, hyperparams)
  * logic.py         -- the agent's personality (6 @qai.logic functions)
  * logic_meta.json  -- metadata for each logic function
  * README.md        -- auto-generated human-readable description

No torch, no sklearn, no numpy extras were installed -- qai shipped
them all. The bundle is fully self-contained.

To run again with a fresh learning session:
    python demo.py --episodes 50 --puzzle random

To focus on one puzzle type:
    python demo.py --episodes 30 --puzzle maze
""")


def main():
    parser = argparse.ArgumentParser(description="Run the EVOLVE bundle demo")
    parser.add_argument("--bundle", default="bundle/evolve.qmodel",
                        help="Path to the .qmodel bundle")
    parser.add_argument("--sign-key", default="evolve-secret-2026",
                        help="HMAC signing key (must match training)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of live-learning episodes to run")
    parser.add_argument("--puzzle", default="random",
                        choices=["maze", "sequence", "resource", "avoid", "random"],
                        help="Which puzzle type to focus on (default: random)")
    args = parser.parse_args()

    bundle_path = args.bundle
    if not os.path.isabs(bundle_path):
        bundle_path = os.path.join(os.path.dirname(__file__), bundle_path)

    run_demo(bundle_path, args.sign_key, args.episodes, args.puzzle)


if __name__ == "__main__":
    main()
