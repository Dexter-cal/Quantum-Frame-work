"""
train_and_bundle.py
===================
Train the EVOLVE agent from scratch and bundle it as a self-contained
.qmodel file that runs with just Python + qai (no extra libraries).

Steps:
  1. Set up a CPU-only ResourceBudget (we have no GPU).
  2. Train the agent for N episodes against PuzzleWorld.
  3. Capture the agent's final state (Q-table, hyperparameters, memory).
  4. Bundle as evolve.qmodel with:
       - the agent's weights (Q-table) -- via the pickle_fallback path
       - the @qai.logic function source (so the agent's personality
         ships with it)
       - HMAC signing for supply-chain integrity
       - auto-generated README
  5. Verify the bundle: reload it, run an episode, confirm the agent
     still works.
  6. Print a final summary.

Usage:
    python train_and_bundle.py [--episodes 200] [--out evolve.qmodel]
"""
from __future__ import annotations
import sys, os, time, json, argparse, pickle, base64, zipfile, io
import numpy as np

# Make qai importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import qai

# Make the showcase modules importable
sys.path.insert(0, os.path.dirname(__file__))
from environments.puzzle_world import PuzzleWorld
from agent.evolve_agent import EvolveAgent
import agent.evolve_logic as evolve_logic


# =====================================================================
# 1. A custom bundle wrapper for the EvolveAgent
# =====================================================================
# qai.save_bundle works on qai.Model objects, but EvolveAgent is a richer
# composite (TabularPolicy brain + memory + logic). So we serialize it
# with a custom bundle format that STILL uses qai.bundle for the file
# structure (ZIP + manifest + README + logic source), but stores the
# agent as a pickle inside the bundle.

def save_evolve_bundle(agent: EvolveAgent, path: str,
                       logic_module, sign_key: str = None,
                       description: str = "",
                       dashboard_html: str = None,
                       launcher_source: str = None) -> str:
    """Save an EvolveAgent + its @qai.logic functions + the dashboard UI
    to a .qmodel file.

    The bundle contains:
      - manifest.json         metadata
      - agent.pkl             pickled EvolveAgent (Q-table, hyperparams, memory)
      - logic.py              source code of all @qai.logic functions
      - logic_meta.json       metadata for each logic function
      - dashboard.html        the visual UI (served by launch_ui.py)
      - launch_ui.py          the launcher script (so the bundle is fully self-contained)
      - README.md             auto-generated human-readable description
    """
    # Pickle the agent (Q-table is just a dict of tuples -> floats, so
    # this is safe; we strip the unpicklable qai.memory closure first).
    pickleable_agent = agent.to_pickleable()
    agent_bytes = pickle.dumps(pickleable_agent)

    # Capture the source of every @qai.logic function in the module
    import inspect, textwrap
    logic_sources = []
    logic_meta = []
    for name in dir(logic_module):
        obj = getattr(logic_module, name)
        if callable(obj) and hasattr(obj, "logic_name"):
            target = obj.__wrapped__ if hasattr(obj, "__wrapped__") else obj
            src = textwrap.dedent(inspect.getsource(target))
            logic_sources.append(src)
            logic_meta.append({
                "name": obj.logic_name,
                "metadata": obj.logic_metadata,
            })
    logic_source = "\n\n".join(logic_sources)

    # Build the manifest
    import hashlib, hmac
    manifest = {
        "format_version": "evolve-1.0",
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent_class": "EvolveAgent",
        "agent_size_bytes": len(agent_bytes),
        "agent_size_human": f"{len(agent_bytes) / 1024:.1f} KB",
        "logic_count": len(logic_meta),
        "logic_names": [m["name"] for m in logic_meta],
        "description": description,
        "signed": sign_key is not None,
        "agent_hash": hashlib.sha256(agent_bytes).hexdigest(),
        "logic_hash": hashlib.sha256(logic_source.encode("utf-8")).hexdigest(),
        "has_dashboard": dashboard_html is not None,
        "has_launcher": launcher_source is not None,
        "dashboard_hash": (hashlib.sha256(dashboard_html.encode("utf-8")).hexdigest()
                            if dashboard_html else None),
        "launcher_hash": (hashlib.sha256(launcher_source.encode("utf-8")).hexdigest()
                           if launcher_source else None),
        "signed_hash": None,
    }
    if sign_key is not None:
        # Include dashboard + launcher in the sign payload so any tampering
        # with the UI files invalidates the signature too.
        payload_parts = [manifest["agent_hash"], manifest["logic_hash"]]
        if manifest["dashboard_hash"]:
            payload_parts.append(manifest["dashboard_hash"])
        if manifest["launcher_hash"]:
            payload_parts.append(manifest["launcher_hash"])
        payload = "".join(payload_parts).encode("utf-8")
        manifest["signed_hash"] = hmac.new(sign_key.encode("utf-8"),
                                            payload, hashlib.sha256).hexdigest()

    # Write the bundle
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("agent.pkl", agent_bytes)
        zf.writestr("logic.py", logic_source)
        zf.writestr("logic_meta.json", json.dumps(logic_meta, indent=2))
        if dashboard_html:
            zf.writestr("dashboard.html", dashboard_html.encode("utf-8"))
        if launcher_source:
            zf.writestr("launch_ui.py", launcher_source.encode("utf-8"))
        zf.writestr("README.md", _generate_readme(manifest, agent))
    return path


def load_evolve_bundle(path: str, sign_key: str = None,
                       verify_signature: bool = True) -> dict:
    """Load an EVOLVE .qmodel bundle.

    Returns a dict with:
      - agent: reconstructed EvolveAgent
      - logic_fns: dict of {name: callable} (with `agent` injected)
      - manifest: the bundle manifest
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bundle not found: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))

        # Verify signature if requested
        if verify_signature and manifest.get("signed"):
            if sign_key is None:
                raise ValueError(
                    "[WHAT] Bundle is signed but no sign_key provided\n"
                    "[WHY] Signed bundles require the original signing key\n"
                    "[FIX] Pass sign_key='your_secret_key'"
                )
            import hashlib, hmac
            agent_bytes = zf.read("agent.pkl")
            logic_source = zf.read("logic.py").decode("utf-8")
            agent_hash = hashlib.sha256(agent_bytes).hexdigest()
            logic_hash = hashlib.sha256(logic_source.encode("utf-8")).hexdigest()
            payload_parts = [agent_hash, logic_hash]
            if "dashboard.html" in zf.namelist():
                dashboard_bytes = zf.read("dashboard.html")
                payload_parts.append(hashlib.sha256(dashboard_bytes).hexdigest())
            if "launch_ui.py" in zf.namelist():
                launcher_bytes = zf.read("launch_ui.py")
                payload_parts.append(hashlib.sha256(launcher_bytes).hexdigest())
            payload = "".join(payload_parts).encode("utf-8")
            expected = hmac.new(sign_key.encode("utf-8"),
                                 payload, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, manifest["signed_hash"]):
                raise ValueError(
                    "[WHAT] Bundle signature verification failed\n"
                    "[WHY] Bundle may have been tampered with, or wrong sign_key\n"
                    "[FIX] Use correct sign_key or reject this bundle"
                )

        # Reconstruct the agent (rebuild the qai.memory that was stripped)
        agent_bytes = zf.read("agent.pkl")
        agent = pickle.loads(agent_bytes)
        agent = EvolveAgent.from_loaded(agent)

        # Load the dashboard HTML if present
        dashboard_html = None
        if "dashboard.html" in zf.namelist():
            dashboard_html = zf.read("dashboard.html").decode("utf-8")

        # Load the launcher source if present (for inspection; the user
        # would normally use the on-disk launch_ui.py)
        launcher_source = None
        if "launch_ui.py" in zf.namelist():
            launcher_source = zf.read("launch_ui.py").decode("utf-8")

        # Reconstruct the @qai.logic functions
        logic_source = zf.read("logic.py").decode("utf-8")
        exec_namespace = {
            "qai": qai,
            "np": np,
            "agent": agent,  # inject the agent so the logic fns can use it
        }
        exec(logic_source, exec_namespace)
        logic_fns = {}
        for name, obj in exec_namespace.items():
            if name in ("qai", "np", "agent", "__builtins__"):
                continue
            if callable(obj) and hasattr(obj, "logic_name"):
                logic_fns[obj.logic_name] = obj

    return {
        "agent": agent,
        "logic_fns": logic_fns,
        "manifest": manifest,
        "path": path,
        "dashboard_html": dashboard_html,
        "launcher_source": launcher_source,
    }


def _generate_readme(manifest: dict, agent: EvolveAgent) -> str:
    """Auto-generate a human-readable README for the bundle."""
    s = agent.status()
    lines = [
        f"# EVOLVE -- {manifest.get('description', 'self-improving digital organism')}",
        "",
        f"- **Format version**: {manifest['format_version']}",
        f"- **Created**: {manifest.get('created_at_iso', 'unknown')}",
        f"- **Agent size**: {manifest.get('agent_size_human', '?')}",
        f"- **Logic functions**: {manifest.get('logic_count', 0)} ({', '.join(manifest.get('logic_names', []))})",
        f"- **Signed**: {'yes' if manifest.get('signed') else 'no'}",
        "",
        "## What is EVOLVE?",
        "",
        "EVOLVE is a self-improving digital organism. Unlike most ML models whose",
        "weights are frozen at training time, EVOLVE continues to learn while you",
        "use it. Every puzzle you expose it to, every action it takes, every reward",
        "it receives -- all of it updates its brain (a Q-learning Q-table) in real",
        "time. The bundle you've downloaded is a SEED, not a finished product.",
        "",
        "## Bundle contents",
        "",
        "1. `agent.pkl` -- the agent's current brain state:",
        f"   - {s['q_table_size']} Q-table entries",
        f"   - {s['episodes_completed']} episodes already experienced",
        f"   - {s['puzzles_solved']} puzzles solved",
        f"   - epsilon={s['current_epsilon']}, lr={s['current_lr']}",
        "",
        "2. `logic.py` -- the agent's personality, encoded as @qai.logic functions:",
        f"   {', '.join(manifest.get('logic_names', []))}",
        "",
        "3. `manifest.json` -- metadata + HMAC signature (if signed).",
        "",
        "## How to use",
        "",
        "```python",
        "import qai",
        "from evolve.bundle_loader import load_evolve_bundle  # or see demo script",
        "",
        'bundle = load_evolve_bundle("evolve.qmodel", sign_key="...")',
        "agent = bundle['agent']",
        "logic = bundle['logic_fns']",
        "",
        "# Ask the agent to explain itself",
        "print(logic['explain_self']())",
        "",
        "# Run a new episode -- the agent learns as it goes",
        "from evolve.environments.puzzle_world import PuzzleWorld",
        "env = PuzzleWorld()",
        "summary = agent.run_episode(env, max_steps=30, learn=True)",
        "print(summary)",
        "",
        "# See the agent's self-critique",
        "print(logic['self_critique']())",
        "```",
        "",
        "## What the agent can do",
        "",
        "- Solve 4 different puzzle types: maze, sequence, resource, avoid",
        "- Learn ONLINE -- its Q-table updates on every interaction",
        "- Self-tune its exploration rate (epsilon) and learning rate (lr)",
        "- Get curious about unfamiliar states and explore them",
        "- Remember past episodes via qai.memory",
        "- Explain itself, critique its performance, and propose strategies",
        "",
        "## Requirements",
        "",
        "Just Python 3.10+ and qai. No torch, no sklearn, no numpy extras --",
        "qai ships them all. The bundle is fully self-contained.",
        "",
        "---",
        "",
        f"Generated by EVOLVE train_and_bundle.py",
    ]
    return "\n".join(lines)


# =====================================================================
# 2. Training driver
# =====================================================================
def train_agent(n_episodes: int = 200, seed: int = 0,
                budget_ram_mb: float = 512,
                budget_time_s: float = 300) -> EvolveAgent:
    """Train an EvolveAgent under a CPU ResourceBudget.

    Uses qai.ResourceBudget to refuse to keep training if RAM or time
    runs out -- we have no GPU, so this is how we stay disciplined on
    a small machine.
    """
    budget = qai.ResourceBudget(ram_mb=budget_ram_mb, time_s=budget_time_s)
    print(f"Resource budget: {budget.summary()}")
    print()

    env = PuzzleWorld(seed=seed)
    agent = EvolveAgent(epsilon=0.3, learning_rate=0.1, name="evolve")

    # Register the agent with the budget manager
    load_info = budget.load(agent, "evolve_agent")
    if not load_info.get("loaded"):
        print(f"ERROR: agent didn't fit in budget: {load_info}")
        return agent

    print(f"Training {n_episodes} episodes (will stop early if budget exhausted)...")
    print(f"{'Ep':>4} {'eps':>6} {'lr':>6} {'slope':>7} {'solved':>7} "
          f"{'reward':>8}  skills-by-puzzle")
    print("-" * 80)

    start_time = time.time()
    last_print = 0
    for ep_idx in range(n_episodes):
        # Check budget
        if budget.time_exceeded():
            print(f"\n[BUDGET] Time budget exceeded at episode {ep_idx}, stopping early.")
            break
        if not budget.can_load(agent):
            print(f"\n[BUDGET] RAM budget exceeded at episode {ep_idx}, stopping early.")
            break

        agent.run_episode(env, max_steps=30, learn=True)

        # Progress prints at 0, 25, 50, 100, 150, 200, ...
        if ep_idx == 0 or (ep_idx + 1) % 25 == 0 or ep_idx == n_episodes - 1:
            s = agent.status()
            skills = ' '.join(f'{p[:3]}={r:+5.1f}' for p, r in s['skill_by_puzzle'].items())
            elapsed = time.time() - start_time
            print(f"{len(agent.episodes):>4} {s['current_epsilon']:>6.3f} "
                  f"{s['current_lr']:>6.3f} {s['performance_slope']:>+7.3f} "
                  f"{s['puzzles_solved']:>7} {s['total_reward']:>+8.1f}  {skills}"
                  f"  [{elapsed:.0f}s]")

    print()
    print(f"=== Training complete ({len(agent.episodes)} episodes, "
          f"{time.time() - start_time:.1f}s) ===")
    print(agent.explain_self())
    print()
    print(f"Final budget status: {budget.summary()}")
    return agent


# =====================================================================
# 3. Main entry point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train and bundle EVOLVE")
    parser.add_argument("--episodes", type=int, default=150,
                        help="Number of training episodes (default 150)")
    parser.add_argument("--out", type=str, default="evolve.qmodel",
                        help="Output bundle path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sign-key", type=str, default="evolve-secret-2026",
                        help="HMAC signing key (set to empty string to skip signing)")
    parser.add_argument("--time-budget", type=float, default=240.0,
                        help="Time budget in seconds")
    args = parser.parse_args()

    # 1. Train
    agent = train_agent(n_episodes=args.episodes,
                        seed=args.seed,
                        budget_time_s=args.time_budget)

    # 2. Inject the agent into the logic module so the logic functions
    #    have a live `agent` to call when bundling captures their source.
    evolve_logic.agent = agent

    # 3. Bundle
    out_path = os.path.join(os.path.dirname(__file__), "bundle", args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sign_key = args.sign_key if args.sign_key else None

    # 3a. Load the dashboard HTML + launcher source so they ship inside the bundle.
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    launcher_path = os.path.join(os.path.dirname(__file__), "launch_ui.py")
    dashboard_html = None
    launcher_source = None
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            dashboard_html = f.read()
        print(f"  loaded dashboard.html ({len(dashboard_html)} bytes)")
    if os.path.exists(launcher_path):
        with open(launcher_path, "r", encoding="utf-8") as f:
            launcher_source = f.read()
        print(f"  loaded launch_ui.py ({len(launcher_source)} bytes)")

    print(f"\nBundling to {out_path}...")
    save_evolve_bundle(
        agent, out_path,
        logic_module=evolve_logic,
        sign_key=sign_key,
        description="EVOLVE v1.0 -- a self-improving digital organism. "
                    "Trained on PuzzleWorld (maze/sequence/resource/avoid). "
                    "Continues to learn when deployed. Includes a visual "
                    "dashboard UI launchable in any browser.",
        dashboard_html=dashboard_html,
        launcher_source=launcher_source,
    )

    # 4. Verify
    print(f"\nVerifying bundle round-trip...")
    bundle = load_evolve_bundle(out_path, sign_key=sign_key)
    reconstructed = bundle["agent"]
    print(f"  reconstructed agent: {len(reconstructed.episodes)} episodes, "
          f"{reconstructed.n_puzzles_solved} solved")
    print(f"  logic functions reloaded: {list(bundle['logic_fns'].keys())}")
    print(f"  dashboard bundled: {bundle['dashboard_html'] is not None}")
    print(f"  launcher bundled:  {bundle['launcher_source'] is not None}")
    # Run a fresh episode with the reconstructed agent
    env = PuzzleWorld(seed=99)
    summary = reconstructed.run_episode(env, max_steps=30, learn=False)
    print(f"  test episode on reconstructed agent: {summary}")

    # 5. Print final bundle info
    print(f"\n=== Bundle ready ===")
    print(f"  path: {out_path}")
    print(f"  size: {os.path.getsize(out_path) / 1024:.1f} KB")
    print(f"  signed: {bundle['manifest'].get('signed', False)}")
    print(f"  logic: {bundle['manifest']['logic_names']}")
    print(f"  dashboard: {'yes' if bundle['manifest'].get('has_dashboard') else 'no'}")
    print(f"  launcher:  {'yes' if bundle['manifest'].get('has_launcher') else 'no'}")
    print(f"\nTo launch the visual UI:")
    print(f"  python launch_ui.py --bundle {out_path}")


if __name__ == "__main__":
    main()
