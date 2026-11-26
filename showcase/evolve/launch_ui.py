"""
launch_ui.py
============
Launch the EVOLVE dashboard in your default browser.

This script:
  1. Loads the evolve.qmodel bundle (with signature verification).
  2. Starts a tiny localhost HTTP server (Python stdlib only -- no
     Flask, no FastAPI, no extra dependencies).
  3. Serves the bundled dashboard.html at http://127.0.0.1:8765/
  4. Exposes a JSON API at /api/* for the dashboard to talk to.
  5. Opens your default browser to the dashboard URL.
  6. Runs until you press Ctrl+C (or 30 minutes of inactivity).

Usage:
    python launch_ui.py [--bundle evolve.qmodel] [--sign-key evolve-secret-2026]
                        [--port 8765] [--no-browser]

Requirements:
    - Python 3.10+
    - qai (pip install qai)
    - A modern browser (any)

That's it. No torch, no sklearn, no numpy extras, no Flask -- qai ships
them all, and the HTTP server is Python's stdlib http.server.
"""
from __future__ import annotations
import sys, os, time, json, argparse, threading, webbrowser, signal
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, Optional

# Make qai importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import qai

# Make the showcase modules importable
sys.path.insert(0, os.path.dirname(__file__))
from train_and_bundle import load_evolve_bundle
from environments.puzzle_world import PuzzleWorld
from persistent_brain import PersistentBrain, compute_bundle_hash


# =====================================================================
# Global state -- the loaded bundle + agent + environment + persistent brain
# =====================================================================
STATE: Dict[str, Any] = {
    "bundle": None,
    "agent": None,
    "logic_fns": {},
    "env": None,
    "brain": None,             # PersistentBrain sidecar manager
    "bundle_path": None,
    "bundle_hash": None,
    "last_activity": time.time(),
    "start_time": time.time(),
}
LOCK = threading.Lock()  # serialize API calls so the agent isn't trained concurrently


def load_bundle(bundle_path: str, sign_key: Optional[str]):
    """Load the bundle, set up the persistent brain, and inject the agent
    into the logic functions.

    The persistent brain takes precedence over the bundle's frozen agent:
    if a sidecar file exists next to the bundle, the agent is loaded from
    the sidecar (the most-recent accumulated state). Otherwise the
    bundle's frozen agent is used to seed a fresh sidecar.
    """
    bundle = load_evolve_bundle(bundle_path, sign_key=sign_key or None)
    bundle_agent = bundle["agent"]

    # Set up the persistent brain
    bundle_hash = compute_bundle_hash(bundle_path)
    sidecar_path = bundle_path + ".brain.json"
    brain = PersistentBrain(sidecar_path, autosave_every_n=5, bundle_hash=bundle_hash)

    # Load the accumulated agent (or initialize from the bundle if no sidecar)
    agent = brain.load_or_initialize(bundle_agent)

    # Inject agent into every logic function's globals so they can call `agent.xxx`
    for fn in bundle["logic_fns"].values():
        try:
            fn.__globals__["agent"] = agent
        except Exception:
            pass

    STATE["bundle"] = bundle
    STATE["agent"] = agent
    STATE["logic_fns"] = bundle["logic_fns"]
    STATE["env"] = PuzzleWorld(seed=int(time.time()))
    STATE["brain"] = brain
    STATE["bundle_path"] = bundle_path
    STATE["bundle_hash"] = bundle_hash
    STATE["last_activity"] = time.time()
    STATE["start_time"] = time.time()
    return bundle


# =====================================================================
# PuzzleWorld helper -- map puzzle name -> idx
# =====================================================================
PUZZLE_MAP = {"maze": 0, "sequence": 1, "resource": 2, "avoid": 3, "random": None}


# =====================================================================
# JSON API handlers
# =====================================================================
def handle_api_info(_) -> Dict[str, Any]:
    m = STATE["bundle"]["manifest"]
    return {
        "format_version": m.get("format_version", "?"),
        "agent_size_human": m.get("agent_size_human", "?"),
        "logic_count": m.get("logic_count", 0),
        "logic_names": m.get("logic_names", []),
        "signed": m.get("signed", False),
        "description": m.get("description", ""),
        "uptime_s": time.time() - STATE["start_time"],
    }


def handle_api_status(_) -> Dict[str, Any]:
    return STATE["agent"].status()


def handle_api_recent_episodes(query) -> Dict[str, Any]:
    n = int(query.get("n", ["20"])[0])
    return {"episodes": STATE["agent"].recent_episodes(n)}


def handle_api_explain(_) -> Dict[str, str]:
    fn = STATE["logic_fns"].get("explain_self")
    if fn is None:
        return {"text": "explain_self logic function not found in bundle."}
    return {"text": fn()}


def handle_api_critique(_) -> Dict[str, str]:
    fn = STATE["logic_fns"].get("self_critique")
    if fn is None:
        return {"text": "self_critique logic function not found in bundle."}
    return {"text": fn()}


def handle_api_run_episodes(body) -> Dict[str, Any]:
    n = int(body.get("n", 1))
    puzzle = body.get("puzzle", "random")
    puzzle_idx = PUZZLE_MAP.get(puzzle, None)

    summaries = []
    brain = STATE.get("brain")
    for _ in range(n):
        summary = STATE["agent"].run_episode(
            STATE["env"], max_steps=30, learn=True, puzzle_idx=puzzle_idx
        )
        # Reflect after every episode (via the @qai.logic function)
        reflect = STATE["logic_fns"].get("reflect_on_episode")
        if reflect:
            try:
                reflect(summary)
            except Exception as e:
                print(f"[warn] reflect_on_episode failed: {e}", file=sys.stderr)
        # Checkpoint to the persistent brain (auto-saves every N episodes)
        if brain is not None:
            brain.checkpoint(STATE["agent"], session_id=brain._session_id)
        summaries.append(summary)
    # Force a save at the end of every batch (so the user sees persistence immediately)
    if brain is not None:
        brain.save(STATE["agent"], force=True)
    return {
        "summaries": summaries,
        "n_run": len(summaries),
        "n_solved": sum(1 for s in summaries if s["solved"]),
        "total_reward": sum(s["total_reward"] for s in summaries),
        "persistence": brain.info() if brain else None,
    }


def handle_api_reset(body) -> Dict[str, Any]:
    """Reset the agent's brain in-memory: wipe the Q-table and history.

    NOTE: this does NOT delete the sidecar file. The next time the user
    launches the dashboard, the accumulated state will return. To truly
    reset to factory, use /api/reset_to_factory instead.
    """
    a = STATE["agent"]
    a.brain.agent.q_table.clear()
    a.brain.agent.cumulative_reward = 0.0
    a.brain.agent.episode_count = 0
    a.episodes = []
    a.recent_rewards.clear()
    a.recent_by_puzzle.clear()
    a.state_last_seen.clear()
    a.n_state_visits.clear()
    a.n_puzzles_solved = 0
    a.total_reward = 0.0
    a.skill_by_puzzle.clear()
    a.performance_slope = 0.0
    a.current_epsilon = 0.3
    a.current_lr = 0.1
    return {"reset": True, "status": a.status()}


def handle_api_persistence(_) -> Dict[str, Any]:
    """Return info about the persistent brain sidecar."""
    brain = STATE.get("brain")
    if brain is None:
        return {"enabled": False}
    return {"enabled": True, **brain.info()}


def handle_api_save(body) -> Dict[str, Any]:
    """Force-save the agent's current state to the sidecar."""
    brain = STATE.get("brain")
    if brain is None:
        return {"saved": False, "reason": "no persistent brain configured"}
    saved = brain.save(STATE["agent"], force=True)
    return {"saved": saved, "info": brain.info()}


def handle_api_reset_to_factory(body) -> Dict[str, Any]:
    """TRUE factory reset: delete the sidecar file AND reset the in-memory
    agent. The next launch will re-initialize from the frozen bundle."""
    brain = STATE.get("brain")
    if brain is None:
        return {"reset": False, "reason": "no persistent brain configured"}
    brain.reset()
    # Also reset the in-memory agent
    return handle_api_reset(body)


# API routing table
API_ROUTES = {
    "/api/info":              ("GET",  handle_api_info),
    "/api/status":            ("GET",  handle_api_status),
    "/api/recent_episodes":   ("GET",  handle_api_recent_episodes),
    "/api/explain":           ("GET",  handle_api_explain),
    "/api/critique":          ("GET",  handle_api_critique),
    "/api/persistence":       ("GET",  handle_api_persistence),
    "/api/run_episodes":      ("POST", handle_api_run_episodes),
    "/api/reset":             ("POST", handle_api_reset),
    "/api/save":              ("POST", handle_api_save),
    "/api/reset_to_factory":  ("POST", handle_api_reset_to_factory),
}


# =====================================================================
# HTTP handler
# =====================================================================
class DashboardHandler(BaseHTTPRequestHandler):
    # Quiet logging
    def log_message(self, fmt, *args):
        # Only log errors, not every request
        if "error" in fmt.lower() or "500" in str(args):
            super().log_message(fmt, *args)

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, code=200):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self):
        body = b"Not Found"
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        STATE["last_activity"] = time.time()
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Serve the dashboard at /
        if path == "/" or path == "/index.html":
            html = self._load_dashboard_html()
            if html is None:
                self._send_404()
                return
            self._send_html(html)
            return

        # API routes
        if path in API_ROUTES:
            method, handler = API_ROUTES[path]
            if method != "GET":
                self._send_json({"error": f"Method not allowed: {method}"}, 405)
                return
            try:
                with LOCK:
                    result = handler(query)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e), "type": type(e).__name__}, 500)
            return

        self._send_404()

    def do_POST(self):
        STATE["last_activity"] = time.time()
        parsed = urlparse(self.path)
        path = parsed.path

        if path in API_ROUTES:
            method, handler = API_ROUTES[path]
            if method != "POST":
                self._send_json({"error": f"Method not allowed: {method}"}, 405)
                return
            # Parse JSON body
            content_length = int(self.headers.get("Content-Length", 0))
            body = {}
            if content_length > 0:
                try:
                    raw = self.rfile.read(content_length)
                    body = json.loads(raw.decode("utf-8"))
                except Exception as e:
                    self._send_json({"error": f"Invalid JSON body: {e}"}, 400)
                    return
            try:
                with LOCK:
                    result = handler(body)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e), "type": type(e).__name__}, 500)
            return

        self._send_404()

    def _load_dashboard_html(self) -> Optional[str]:
        """Load dashboard.html. Try the bundle first, then the local file."""
        # 1. Try extracting from the bundle
        bundle = STATE.get("bundle")
        if bundle and "dashboard_html" in bundle:
            return bundle["dashboard_html"]
        # 2. Try the local file (development mode)
        local = os.path.join(os.path.dirname(__file__), "dashboard.html")
        if os.path.exists(local):
            with open(local, "r", encoding="utf-8") as f:
                return f.read()
        return None


# =====================================================================
# Auto-shutdown thread (30 min inactivity)
# =====================================================================
def auto_shutdown_watcher(server: HTTPServer, timeout_s: int = 1800):
    """Watch for inactivity; shut down the server after `timeout_s` of no requests."""
    while True:
        time.sleep(30)
        idle = time.time() - STATE["last_activity"]
        if idle > timeout_s:
            print(f"\n[auto-shutdown] {timeout_s}s of inactivity reached; shutting down.")
            server.shutdown()
            return


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Launch the EVOLVE dashboard")
    parser.add_argument("--bundle", default="bundle/evolve.qmodel",
                        help="Path to the .qmodel bundle")
    parser.add_argument("--sign-key", default="evolve-secret-2026",
                        help="HMAC signing key (must match training)")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to serve on (default 8765)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind (default 127.0.0.1 -- localhost only)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Auto-shutdown after N seconds of inactivity (default 1800)")
    args = parser.parse_args()

    bundle_path = args.bundle
    if not os.path.isabs(bundle_path):
        bundle_path = os.path.join(os.path.dirname(__file__), bundle_path)

    if not os.path.exists(bundle_path):
        print(f"ERROR: bundle not found at {bundle_path}")
        print("Run `python train_and_bundle.py` first to create it.")
        sys.exit(1)

    # 1. Load the bundle (this also sets up the persistent brain)
    print(f"Loading bundle: {bundle_path}")
    bundle = load_bundle(bundle_path, args.sign_key)
    agent = bundle["agent"]
    print(f"  agent: {agent.status()['episodes_completed']} episodes pre-loaded")
    print(f"  logic: {list(bundle['logic_fns'].keys())}")
    print(f"  signed: {bundle['manifest'].get('signed', False)}")
    brain_info = STATE["brain"].info()
    print(f"  persistent brain: session #{brain_info['current_session_id']}, "
          f"{brain_info['n_checkpoints']} prior checkpoints, "
          f"last saved {brain_info['last_saved_iso'] or 'never'}")
    print(f"  sidecar: {brain_info['sidecar_path']}")

    # 2. Start the HTTP server
    server = HTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"\nEVOLVE dashboard running at: {url}")
    print(f"  (auto-shutdown after {args.timeout}s of inactivity)")
    print(f"  (auto-saves the agent's brain every 5 episodes)")
    print(f"  press Ctrl+C to stop (will save the brain on exit)\n")

    # 3. Auto-shutdown watcher
    watcher = threading.Thread(
        target=auto_shutdown_watcher, args=(server, args.timeout), daemon=True
    )
    watcher.start()

    # 4. Open the browser
    if not args.no_browser:
        try:
            webbrowser.open(url)
            print(f"  opened browser -> {url}")
        except Exception as e:
            print(f"  (could not auto-open browser: {e}; please open {url} manually)")

    # 5. Serve until interrupted -- with a graceful-shutdown handler
    #    that force-saves the brain before exit.
    def graceful_shutdown(sig, frame):
        print("\nShutting down... saving brain...")
        try:
            brain = STATE.get("brain")
            if brain is not None and STATE.get("agent") is not None:
                brain.save(STATE["agent"], force=True)
                info = brain.info()
                print(f"  brain saved: {info['n_checkpoints']} checkpoints total, "
                      f"last saved {info['last_saved_iso']}")
        except Exception as e:
            print(f"  WARNING: brain save failed: {e}")
        server.shutdown()
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    main()
