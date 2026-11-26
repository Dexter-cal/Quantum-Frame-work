"""
persistent_brain.py
===================
The persistent-memory layer for EVOLVE.

PROBLEM: the .qmodel bundle is a frozen snapshot. Every time the user
loads it, they get the agent as it was at training time. Any learning
that happens during a dashboard session is lost when the session ends.

SOLUTION: a sidecar file (`<bundle>.brain.json`) that lives next to
the .qmodel bundle. The launcher:

  1. On startup, checks for the sidecar. If it exists, loads the
     agent from the sidecar (the most-recent accumulated state)
     instead of from the frozen bundle.
  2. After every N episodes (default 5), auto-saves the agent to
     the sidecar.
  3. On shutdown (SIGINT/SIGTERM), saves one final time.
  4. Tracks session count + last-saved timestamp so the dashboard
     can show "knowledge accumulated across N sessions".

This maps cleanly onto qai's three-tier memory design:

  Tier              Where it lives        EVOLVE implementation
  ----------------  --------------------  -----------------------------------
  WeightsMemory     permanent (slow)     The .qmodel bundle (frozen at train)
  WorkingMemory     temporal (per-sess)  In-memory episode log (lost on close)
  PersistentBrain   permanent (live)     The .brain.json sidecar (THIS FILE)

The PersistentBrain is the "slowly-changing permanent memory" that
grows over time as the user interacts with the agent. It's the bridge
between the frozen bundle and the ephemeral working memory.

Usage:
    from persistent_brain import PersistentBrain

    brain = PersistentBrain(sidecar_path="bundle/evolve.qmodel.brain.json")

    # On startup:
    agent = brain.load_or_initialize(bundle_agent)  # use sidecar if present

    # After every N episodes:
    brain.checkpoint(agent, session_id=1)

    # On shutdown:
    brain.save(agent, session_id=1)

    # Status:
    info = brain.info()  # {sessions, last_saved, n_checkpoints, ...}

Reset to factory: just delete the sidecar file. The next launch will
re-initialize from the frozen bundle.
"""
from __future__ import annotations
import os, json, time, pickle, hashlib, threading
from typing import Any, Dict, Optional
import numpy as np

# Make qai importable
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import qai


class PersistentBrain:
    """A sidecar file that accumulates the agent's learned state across runs.

    The sidecar is a JSON file containing:
      - metadata: session_count, last_saved_iso, n_checkpoints,
        original_bundle_hash (so we can detect if the user replaced the
        bundle underneath us)
      - agent_pickle_b64: base64-encoded pickle of the agent

    On every save, we overwrite the sidecar atomically (write to a temp
    file, then rename) so a crash mid-save can't corrupt the brain.

    Thread-safe: all public methods acquire a lock.
    """

    def __init__(self, sidecar_path: str,
                 autosave_every_n: int = 5,
                 bundle_hash: Optional[str] = None):
        """
        Args:
            sidecar_path: Path to the .brain.json sidecar file.
                          Convention: <bundle_path> + ".brain.json".
            autosave_every_n: Auto-checkpoint after every N episodes
                              (set to 0 to disable auto-checkpointing).
            bundle_hash: Optional hash of the original bundle, stored
                          in the sidecar so we can detect bundle swaps.
        """
        self.sidecar_path = sidecar_path
        self.autosave_every_n = autosave_every_n
        self.bundle_hash = bundle_hash
        self._lock = threading.Lock()
        self._episodes_since_save = 0
        self._session_id = 0  # incremented on each load

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------
    def load_or_initialize(self, bundle_agent) -> Any:
        """Try to load the sidecar. If it exists, return the accumulated
        agent. Otherwise, return `bundle_agent` (the frozen bundle state)
        and write a fresh sidecar to start accumulating from there.

        Returns the agent the caller should use going forward.
        """
        with self._lock:
            if os.path.exists(self.sidecar_path):
                try:
                    agent = self._read_sidecar(bundle_agent)
                    self._session_id = (self._read_metadata().get("session_count", 0) + 1)
                    print(f"[PersistentBrain] loaded accumulated state "
                          f"(session #{self._session_id})")
                    return agent
                except Exception as e:
                    print(f"[PersistentBrain] WARNING: sidecar load failed ({e}); "
                          f"falling back to bundle")
            # No sidecar (or load failed) -- initialize from the bundle
            self._session_id = 1
            print(f"[PersistentBrain] no sidecar found; initializing from bundle "
                  f"(session #{self._session_id})")
            return bundle_agent

    def save(self, agent, force: bool = False) -> bool:
        """Save the agent to the sidecar.

        Args:
            agent: The agent to persist.
            force: If True, always save. If False, only save if there
                   have been `autosave_every_n` new episodes since the
                   last save.

        Returns True if saved, False if skipped.
        """
        with self._lock:
            if not force and self._episodes_since_save < self.autosave_every_n:
                return False
            self._write_sidecar(agent)
            self._episodes_since_save = 0
            return True

    def checkpoint(self, agent, session_id: Optional[int] = None) -> bool:
        """Record that one more episode has happened. Auto-save when the
        counter reaches `autosave_every_n`.

        Args:
            agent: The current agent.
            session_id: Optional session ID to record in the sidecar.

        Returns True if a save happened, False otherwise.
        """
        with self._lock:
            self._episodes_since_save += 1
            if self.autosave_every_n > 0 and self._episodes_since_save >= self.autosave_every_n:
                self._write_sidecar(agent, session_id=session_id)
                self._episodes_since_save = 0
                return True
            return False

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        """Return metadata about the persistent brain."""
        with self._lock:
            meta = self._read_metadata()
            return {
                "sidecar_path": self.sidecar_path,
                "sidecar_exists": os.path.exists(self.sidecar_path),
                "session_count": meta.get("session_count", 0),
                "current_session_id": self._session_id,
                "n_checkpoints": meta.get("n_checkpoints", 0),
                "last_saved_iso": meta.get("last_saved_iso", None),
                "last_saved_epoch": meta.get("last_saved_epoch", 0),
                "seconds_since_save": (time.time() - meta.get("last_saved_epoch", 0))
                                       if meta.get("last_saved_epoch") else None,
                "episodes_since_save": self._episodes_since_save,
                "original_bundle_hash": meta.get("original_bundle_hash", None),
                "current_bundle_hash": self.bundle_hash,
                "bundle_matches": (meta.get("original_bundle_hash") == self.bundle_hash
                                    if self.bundle_hash else None),
                "autosave_every_n": self.autosave_every_n,
            }

    def reset(self) -> bool:
        """Delete the sidecar file. Next load will re-initialize from
        the frozen bundle."""
        with self._lock:
            if os.path.exists(self.sidecar_path):
                os.remove(self.sidecar_path)
                self._episodes_since_save = 0
                self._session_id = 0
                return True
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _read_metadata(self) -> Dict[str, Any]:
        if not os.path.exists(self.sidecar_path):
            return {}
        try:
            with open(self.sidecar_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state.get("metadata", {})
        except Exception:
            return {}

    def _read_sidecar(self, fallback_agent):
        """Read the sidecar and reconstruct the agent.

        If the bundle hash doesn't match (the user replaced the bundle
        underneath us), warn but still load -- the agent's Q-table is
        technique-independent in shape.
        """
        with open(self.sidecar_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        meta = state.get("metadata", {})
        # Bundle-hash check (warn-only)
        if (self.bundle_hash
            and meta.get("original_bundle_hash")
            and meta.get("original_bundle_hash") != self.bundle_hash):
            print(f"[PersistentBrain] WARNING: bundle hash mismatch "
                  f"(sidecar was for a different bundle). Loading anyway.")
        # Reconstruct the agent
        agent_bytes = bytes.fromhex(state["agent_pickle_hex"])
        agent = pickle.loads(agent_bytes)
        # Rebuild the qai.memory that was stripped before pickling
        agent = type(fallback_agent).from_loaded(agent)
        return agent

    def _write_sidecar(self, agent, session_id: Optional[int] = None):
        """Write the sidecar atomically (temp file + rename)."""
        # Strip the unpicklable qai.memory closure before pickling
        pickleable = agent.to_pickleable() if hasattr(agent, "to_pickleable") else agent
        agent_bytes = pickle.dumps(pickleable)

        # Read existing metadata (to increment counters)
        existing_meta = self._read_metadata() if os.path.exists(self.sidecar_path) else {}
        n_checkpoints = existing_meta.get("n_checkpoints", 0) + 1
        session_count = existing_meta.get("session_count", 0)
        if session_id is not None and session_id > session_count:
            session_count = session_id

        meta = {
            "format_version": "evolve-brain-1.0",
            "session_count": session_count,
            "n_checkpoints": n_checkpoints,
            "last_saved_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_saved_epoch": time.time(),
            "original_bundle_hash": self.bundle_hash or existing_meta.get("original_bundle_hash"),
            "agent_class": type(agent).__name__,
            "agent_size_bytes": len(agent_bytes),
            "agent_episodes_completed": getattr(agent, "episodes", None)
                                          and len(agent.episodes) or 0,
            "agent_puzzles_solved": getattr(agent, "n_puzzles_solved", 0),
            "agent_q_table_size": (len(agent.brain.agent.q_table)
                                    if hasattr(agent, "brain") else 0),
        }

        state = {
            "metadata": meta,
            "agent_pickle_hex": agent_bytes.hex(),
        }

        # Atomic write: temp file + rename
        tmp_path = self.sidecar_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.sidecar_path)


# =====================================================================
# Convenience: compute a bundle hash for sidecar bookkeeping
# =====================================================================
def compute_bundle_hash(bundle_path: str) -> str:
    """SHA-256 of the .qmodel bundle file. Used as a fingerprint
    so the sidecar can detect if the user replaced the bundle."""
    h = hashlib.sha256()
    with open(bundle_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    import tempfile
    print("=== PersistentBrain self-test ===\n")

    # Make a fake agent class to test the round-trip
    sys.path.insert(0, os.path.dirname(__file__))
    from environments.puzzle_world import PuzzleWorld
    from agent.evolve_agent import EvolveAgent

    bundle_agent = EvolveAgent(epsilon=0.3, learning_rate=0.1, name="test")
    # Pre-train it a tiny bit
    env = PuzzleWorld(seed=0)
    for _ in range(3):
        bundle_agent.run_episode(env, max_steps=10, learn=True)
    print(f"bundle agent: {bundle_agent.status()['episodes_completed']} episodes, "
          f"{bundle_agent.status()['q_table_size']} Q-entries")

    sidecar = os.path.join(tempfile.mkdtemp(), "test.qmodel.brain.json")
    brain = PersistentBrain(sidecar, autosave_every_n=2, bundle_hash="fake-hash-123")

    print("\n--- Round 1: initialize from bundle ---")
    agent = brain.load_or_initialize(bundle_agent)
    print(f"  agent: {agent.status()['episodes_completed']} episodes, "
          f"{agent.status()['q_table_size']} Q-entries")
    # Run 5 more episodes
    for _ in range(5):
        agent.run_episode(env, max_steps=10, learn=True)
        brain.checkpoint(agent, session_id=1)
    brain.save(agent, force=True, ) if hasattr(brain.save, '__call__') else None
    # The above won't work since `save` signature is (agent, force). Fix:
    brain.save(agent, force=True)
    print(f"  after 5 episodes: {agent.status()['episodes_completed']} episodes, "
          f"{agent.status()['q_table_size']} Q-entries")
    print(f"  sidecar info: {brain.info()}")

    print("\n--- Round 2: load from sidecar (simulating user re-opening) ---")
    brain2 = PersistentBrain(sidecar, autosave_every_n=2, bundle_hash="fake-hash-123")
    agent2 = brain2.load_or_initialize(bundle_agent)
    print(f"  agent2: {agent2.status()['episodes_completed']} episodes, "
          f"{agent2.status()['q_table_size']} Q-entries")
    if agent2.status()['episodes_completed'] == 8:
        print("  >> PERSISTENCE WORKS: knowledge accumulated across runs!")
    else:
        print(f"  >> PERSISTENCE FAILED: expected 8 episodes, got {agent2.status()['episodes_completed']}")

    print("\n--- Round 3: reset to factory ---")
    brain2.reset()
    agent3 = PersistentBrain(sidecar).load_or_initialize(bundle_agent)
    print(f"  agent3 after reset: {agent3.status()['episodes_completed']} episodes (should be 3)")
    if agent3.status()['episodes_completed'] == 3:
        print("  >> RESET WORKS: sidecar deletion restores the bundle's frozen state")
