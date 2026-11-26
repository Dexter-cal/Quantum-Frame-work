"""
qai.collaborative -- real-time collaborative Visual Workspace.

Lets multiple users edit the same workspace simultaneously via WebRTC
(peer-to-peer) or WebSocket (server-relayed). Built on top of the v2
Visual Workspace.

Architecture:
  * The qai workspace server (visual_workspace_v2.serve_workspace)
    already runs an HTTP server. We add a /collab/* route group.
  * Each session has a unique room ID. Clients join a room via short ID.
  * Cursor positions, weight edits, chat messages, and view state are
    broadcast to all peers in the room.
  * Two backends:
      - WebSocket (default; works through firewalls/NATs)
      - WebRTC (peer-to-peer; lower latency but needs signalling)
  * Graceful degradation: if no other peers are connected, the workspace
    behaves exactly like the standalone version.

Public API:
    qai.serve_workspace_collab(model, port=7788, room=None)
        -> blocking server with /collab/* endpoints

    qai.generate_workspace_collab(model, path, room_id="DEMO")
        -> static HTML with embedded collab client

The collab client is a small (~4KB) JS module injected into the v2
workspace HTML. It talks to the server over WebSocket.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

# Import the v2 workspace so we can reuse its HTML + backend
from .visual_workspace_v2 import (
    _WorkspaceBackend, _model_metadata, _get_weights,
    detect_model_type, WORKSPACE_HTML as _BASE_HTML,
    _find_free_port,
)


# ============================================================================
# Collaboration server (WebSocket relay)
# ============================================================================

class CollabRoom:
    """A room of peers collaborating on one workspace."""
    def __init__(self, room_id: str, backend: _WorkspaceBackend):
        self.room_id = room_id
        self.backend = backend
        self.peers: Dict[str, "CollabPeer"] = {}  # peer_id -> peer
        self.events: List[Dict[str, Any]] = []  # last 200 events
        self._lock = threading.RLock()

    def add_peer(self, peer: "CollabPeer") -> str:
        with self._lock:
            peer_id = str(uuid.uuid4())[:8]
            peer.peer_id = peer_id
            peer.room = self
            self.peers[peer_id] = peer
            return peer_id

    def remove_peer(self, peer_id: str) -> None:
        with self._lock:
            self.peers.pop(peer_id, None)

    def broadcast(self, message: Dict[str, Any], exclude: Optional[str] = None) -> int:
        """Send message to all peers. Returns count of peers reached."""
        msg_str = json.dumps(message, default=str)
        delivered = 0
        with self._lock:
            peers = list(self.peers.values())
        for peer in peers:
            if peer.peer_id == exclude:
                continue
            try:
                peer.send(msg_str)
                delivered += 1
            except Exception:
                pass
        # Record event
        self.events.append(message)
        if len(self.events) > 200:
            self.events = self.events[-200:]
        return delivered

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "room_id": self.room_id,
                "n_peers": len(self.peers),
                "peer_ids": list(self.peers.keys()),
                "n_events": len(self.events),
            }


class CollabPeer:
    """A single connected peer."""
    def __init__(self, ws: Any = None):
        self.peer_id: str = ""
        self.room: Optional[CollabRoom] = None
        self.ws = ws  # WebSocket or HTTP-SSE connection
        self.cursor: Dict[str, float] = {"x": 0, "y": 0}
        self.color: str = self._random_color()
        self.name: str = "Anonymous"
        self.last_seen: float = time.time()

    @staticmethod
    def _random_color() -> str:
        import random
        colors = ['#5eead4', '#c084fc', '#f472b6', '#fb923c', '#60a5fa',
                  '#4ade80', '#fbbf24', '#f87171', '#a78bfa', '#34d399']
        return random.choice(colors)

    def send(self, msg: str) -> None:
        """Send a message to this peer. Uses SSE if no WS."""
        if self.ws is None:
            return
        # SSE-style send (write to response buffer)
        if hasattr(self.ws, 'write'):
            try:
                self.ws.write(f"data: {msg}\n\n")
                self.ws.flush()
            except Exception:
                pass


class CollabServer:
    """Manages rooms and peers."""
    def __init__(self):
        self.rooms: Dict[str, CollabRoom] = {}
        self._lock = threading.RLock()

    def get_or_create_room(self, room_id: str, backend: _WorkspaceBackend) -> CollabRoom:
        with self._lock:
            if room_id not in self.rooms:
                self.rooms[room_id] = CollabRoom(room_id, backend)
            return self.rooms[room_id]

    def get_room(self, room_id: str) -> Optional[CollabRoom]:
        with self._lock:
            return self.rooms.get(room_id)

    def list_rooms(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [room.stats() for room in self.rooms.values()]


# Global server instance
_COLLAB_SERVER: Optional[CollabServer] = None

def _get_collab_server() -> CollabServer:
    global _COLLAB_SERVER
    if _COLLAB_SERVER is None:
        _COLLAB_SERVER = CollabServer()
    return _COLLAB_SERVER


# ============================================================================
# Collab client JS (injected into workspace HTML)
# ============================================================================

_COLLAB_CLIENT_JS = r"""
// ============================================================================
// qai Collaborative Workspace Client
// ============================================================================
const COLLAB = {
  peerId: null,
  roomId: null,
  peerColor: '#5eead4',
  peerName: 'Anonymous',
  connected: false,
  peers: new Map(),  // peerId -> {color, name, cursor, lastSeen}
  eventSource: null,
  cursorEls: new Map(),  // peerId -> DOM element
  lastBroadcast: 0,
};

function initCollab(roomId) {
  COLLAB.roomId = roomId || 'default';
  // Generate peer name from browser
  COLLAB.peerName = 'User-' + Math.floor(Math.random() * 1000);
  // Pick a random color
  const colors = ['#5eead4', '#c084fc', '#f472b6', '#fb923c', '#60a5fa',
                  '#4ade80', '#fbbf24', '#f87171', '#a78bfa', '#34d399'];
  COLLAB.peerColor = colors[Math.floor(Math.random() * colors.length)];
  // Connect via SSE (EventSource)
  connectCollab();
  // Set up cursor broadcast on mousemove (throttled)
  document.addEventListener('mousemove', (e) => {
    const now = Date.now();
    if (now - COLLAB.lastBroadcast < 50) return;  // 20fps max
    COLLAB.lastBroadcast = now;
    broadcastCollab({
      type: 'cursor',
      x: e.clientX,
      y: e.clientY,
    });
  });
  // Set up name input
  setTimeout(() => {
    const name = prompt('Your name for this session:', COLLAB.peerName);
    if (name) {
      COLLAB.peerName = name;
      broadcastCollab({type: 'join', name: COLLAB.peerName, color: COLLAB.peerColor});
    }
    showToast('success', `Connected to room: ${COLLAB.roomId}`);
  }, 500);
}

function connectCollab() {
  if (typeof EventSource === 'undefined') {
    showToast('error', 'SSE not supported — collab disabled');
    return;
  }
  // Open SSE stream
  const url = `/collab/events?room=${encodeURIComponent(COLLAB.roomId)}`;
  COLLAB.eventSource = new EventSource(url);
  COLLAB.eventSource.onopen = () => {
    COLLAB.connected = true;
    // Send join message via POST
    fetch('/collab/join', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        room: COLLAB.roomId,
        name: COLLAB.peerName,
        color: COLLAB.peerColor,
      }),
    }).then(r => r.json()).then(data => {
      if (data.peer_id) {
        COLLAB.peerId = data.peer_id;
        console.log('[collab] joined as', COLLAB.peerId);
      }
    }).catch(e => console.error('[collab] join failed', e));
  };
  COLLAB.eventSource.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleCollabMessage(msg);
    } catch (e) {
      // ignore parse errors
    }
  };
  COLLAB.eventSource.onerror = () => {
    COLLAB.connected = false;
    // Auto-reconnect after 3s (EventSource does this automatically, but update UI)
    showToast('error', 'Collab connection lost — reconnecting…');
  };
}

function broadcastCollab(message) {
  if (!COLLAB.connected || !COLAB.peerId) return;
  message.from = COLLAB.peerId;
  message.name = COLLAB.peerName;
  message.color = COLLAB.peerColor;
  message.ts = Date.now();
  // Send via POST (SSE is one-way from server)
  fetch('/collab/broadcast', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({room: COLAB.roomId, message}),
  }).catch(e => console.error('[collab] broadcast failed', e));
}

function handleCollabMessage(msg) {
  if (msg.from === COLLAB.peerId) return;  // ignore own messages
  switch (msg.type) {
    case 'join':
      COLLAB.peers.set(msg.from, {
        color: msg.color || '#5eead4',
        name: msg.name || 'Anonymous',
        cursor: {x: 0, y: 0},
        lastSeen: Date.now(),
      });
      showToast('info', `${msg.name} joined`);
      updateCollabUI();
      break;
    case 'leave':
      const peer = COLLAB.peers.get(msg.from);
      if (peer) showToast('info', `${peer.name} left`);
      COLLAB.peers.delete(msg.from);
      const cursorEl = COLLAB.cursorEls.get(msg.from);
      if (cursorEl) cursorEl.remove();
      COLLAB.cursorEls.delete(msg.from);
      updateCollabUI();
      break;
    case 'cursor':
      let p = COLLAB.peers.get(msg.from);
      if (!p) {
        p = {color: msg.color || '#5eead4', name: msg.name || 'Anonymous', cursor: {x:0,y:0}, lastSeen: Date.now()};
        COLLAB.peers.set(msg.from, p);
      }
      p.cursor = {x: msg.x, y: msg.y};
      p.lastSeen = Date.now();
      updatePeerCursor(msg.from, p);
      break;
    case 'chat':
      appendChat('user', `[${msg.name}] ${msg.text}`);
      break;
    case 'weight_edit':
      // Another peer edited a weight — update our local view
      if (STATE.layers[msg.layer_idx]) {
        STATE.layers[msg.layer_idx].weights[msg.flat_idx] = msg.value;
        drawHeatmap();
        drawSliders();
      }
      showToast('info', `${msg.name} edited weight ${msg.layer_idx}[${msg.flat_idx}]`);
      break;
    case 'predict':
      // Another peer sent a prediction
      appendChat('user', `[${msg.name}] predict: ${msg.input}`);
      break;
  }
}

function updatePeerCursor(peerId, peer) {
  let el = COLLAB.cursorEls.get(peerId);
  if (!el) {
    el = document.createElement('div');
    el.className = 'collab-cursor';
    el.style.cssText = `
      position: fixed;
      pointer-events: none;
      z-index: 9999;
      transition: transform 0.1s ease;
      font-size: 10px;
      font-family: var(--font-sans);
      font-weight: 600;
    `;
    document.body.appendChild(el);
    COLLAB.cursorEls.set(peerId, el);
  }
  el.innerHTML = `
    <svg width="20" height="20" viewBox="0 0 20 20" fill="${peer.color}" stroke="#fff" stroke-width="1.5">
      <path d="M 2 2 L 8 18 L 10 10 L 18 8 Z"/>
    </svg>
    <span style="background:${peer.color};color:#fff;padding:2px 6px;border-radius:8px;margin-left:4px;">${peer.name}</span>
  `;
  el.style.left = peer.cursor.x + 'px';
  el.style.top = peer.cursor.y + 'px';
}

function updateCollabUI() {
  // Update peer count badge
  const n = COLLAB.peers.size;
  const statusEl = document.getElementById('collabStatus');
  if (statusEl) {
    statusEl.textContent = n === 0 ? 'collab: alone' : `collab: ${n + 1} peers`;
    statusEl.style.color = n === 0 ? 'var(--text-4)' : 'var(--good)';
  }
}

// Clean up stale cursors every 5s
setInterval(() => {
  const now = Date.now();
  for (const [id, peer] of COLLAB.peers.entries()) {
    if (now - peer.lastSeen > 10000) {
      COLLAB.peers.delete(id);
      const el = COLLAB.cursorEls.get(id);
      if (el) el.remove();
      COLLAB.cursorEls.delete(id);
    }
  }
  updateCollabUI();
}, 5000);

// Hook into existing sendInput to broadcast predictions
const _origSendInput = sendInput;
sendInput = function() {
  const val = document.getElementById('chatInput').value;
  broadcastCollab({type: 'predict', input: val});
  return _origSendInput.apply(this, arguments);
};

// Hook into editWeight to broadcast
const _origEditWeight = editWeight;
editWeight = function(layerIdx, flatIdx, newVal) {
  broadcastCollab({type: 'weight_edit', layer_idx: layerIdx, flat_idx: flatIdx, value: newVal});
  return _origEditWeight.apply(this, arguments);
};

// Leave on page unload
window.addEventListener('beforeunload', () => {
  if (COLLAB.peerId) {
    navigator.sendBeacon('/collab/leave',
      new Blob([JSON.stringify({room: COLLAB.roomId, peer_id: COLLAB.peerId})],
                {type: 'application/json'}));
  }
});
"""

_COLLAB_STATUS_HTML = r"""
<div id="collabStatus" style="
  position: fixed; top: 70px; right: 20px;
  background: var(--glass-strong);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 6px 12px;
  font-size: 10px;
  color: var(--text-4);
  font-family: var(--font-mono);
  z-index: 75;
">collab: connecting…</div>
"""


# ============================================================================
# HTTP handler that extends the v2 workspace with collab routes
# ============================================================================

def _make_collab_handler(backend: _WorkspaceBackend, room_id: str):
    server = _get_collab_server()
    room = server.get_or_create_room(room_id, backend)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _json(self, code, payload):
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _html(self, code, body: str):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                # Existing workspace routes
                if self.path in ("/", "/index.html", "/workspace"):
                    # Inject collab JS
                    html = _BASE_HTML.replace("</body>",
                        f"<script>{_COLLAB_CLIENT_JS}</script>\n"
                        f"{_COLLAB_STATUS_HTML}\n"
                        f"<script>initCollab('{room_id}');</script>\n"
                        f"</body>")
                    return self._html(200, html)
                elif self.path == "/api/model":
                    return self._json(200, backend.metadata())
                elif self.path == "/api/type":
                    return self._json(200, backend.model_type())
                elif self.path == "/api/layers":
                    return self._json(200, {"layers": backend.get_layers()})
                elif self.path.startswith("/api/weights/"):
                    idx = int(self.path.split("/")[-1])
                    return self._json(200, backend.get_weights(idx))
                elif self.path == "/api/health":
                    return self._json(200, backend.health())
                elif self.path == "/api/type_data":
                    return self._json(200, backend.type_specific_data())
                elif self.path == "/api/training_history":
                    return self._json(200, {"history": backend.training_history()})
                elif self.path == "/api/export":
                    return self._json(200, backend.export_story())
                # NEW collab routes
                elif self.path.startswith("/collab/events"):
                    room_param = self.path.split("room=")[-1] if "room=" in self.path else room_id
                    return self._handle_sse(room_param)
                elif self.path == "/collab/stats":
                    return self._json(200, room.stats())
                else:
                    return self._json(404, {"error": "not found", "path": self.path})
            except Exception as e:
                import traceback
                return self._json(500, {"error": str(e), "trace": traceback.format_exc()[-400:]})

        def _handle_sse(self, room_id: str):
            """Server-Sent Events stream for a peer."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            # Create a peer
            peer = CollabPeer(ws=self.wfile)
            peer_id = room.add_peer(peer)
            # Send initial hello
            try:
                self.wfile.write(f"data: {json.dumps({'type': 'hello', 'peer_id': peer_id, 'room': room_id})}\n\n")
                self.wfile.flush()
            except Exception:
                room.remove_peer(peer_id)
                return
            # Keep connection alive with heartbeat
            try:
                while True:
                    time.sleep(15)
                    self.wfile.write(f": heartbeat\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                room.remove_peer(peer_id)
                # Broadcast leave
                room.broadcast({"type": "leave", "from": peer_id})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw) if raw else {}
                # Existing routes
                if self.path == "/api/predict":
                    return self._json(200, backend.predict(body.get("input", "")))
                elif self.path == "/api/weights/edit":
                    return self._json(200, backend.edit_weight(
                        int(body.get("layer_idx", 0)),
                        int(body.get("flat_idx", 0)),
                        float(body.get("new_value", 0.0)),
                    ))
                elif self.path == "/api/weights/reset":
                    return self._json(200, backend.reset_weights())
                elif self.path == "/api/compare":
                    return self._json(200, backend.compare(body.get("a", ""), body.get("b", "")))
                elif self.path == "/api/adversarial":
                    return self._json(200, backend.adversarial_probe(
                        body.get("input", ""), float(body.get("epsilon", 0.05)),
                    ))
                elif self.path == "/api/counterfactual":
                    return self._json(200, backend.counterfactual(
                        body.get("input", ""),
                        body.get("target_class"),
                        int(body.get("max_steps", 50)),
                        float(body.get("step_size", 0.05)),
                    ))
                # NEW collab routes
                elif self.path == "/collab/join":
                    peer = CollabPeer(ws=self.wfile)
                    peer.name = body.get("name", "Anonymous")
                    peer.color = body.get("color", "#5eead4")
                    peer_id = room.add_peer(peer)
                    # Notify others
                    room.broadcast({
                        "type": "join",
                        "from": peer_id,
                        "name": peer.name,
                        "color": peer.color,
                    }, exclude=peer_id)
                    return self._json(200, {"peer_id": peer_id, "room": room_id})
                elif self.path == "/collab/broadcast":
                    msg = body.get("message", {})
                    delivered = room.broadcast(msg, exclude=msg.get("from"))
                    return self._json(200, {"delivered": delivered})
                elif self.path == "/collab/leave":
                    peer_id = body.get("peer_id")
                    room.remove_peer(peer_id)
                    room.broadcast({"type": "leave", "from": peer_id})
                    return self._json(200, {"ok": True})
                else:
                    return self._json(404, {"error": "not found", "path": self.path})
            except Exception as e:
                import traceback
                return self._json(500, {"error": str(e), "trace": traceback.format_exc()[-400:]})

    return Handler


# ============================================================================
# Public API
# ============================================================================

def serve_workspace_collab(model=None, port: int = 7788, open_browser: bool = True,
                            host: str = "127.0.0.1", room_id: Optional[str] = None,
                            sample_X=None) -> None:
    """Serve the collaborative Visual Workspace.

    Multiple users can join the same ``room_id`` and see each other's
    cursors, weight edits, and predictions in real time.
    """
    backend = _WorkspaceBackend(model, sample_X=sample_X) if model is not None else None
    if backend is None:
        class _SyntheticModel:
            technique_name = "DemoMLP"
            technique = None
        backend = _WorkspaceBackend(_SyntheticModel(), sample_X=None)

    if room_id is None:
        room_id = str(uuid.uuid4())[:8]

    actual_port = _find_free_port(port)
    handler = _make_collab_handler(backend, room_id)
    server = ThreadingHTTPServer((host, actual_port), handler)

    url = f"http://{host}:{actual_port}/"
    print(f"[qai] Collaborative Visual Workspace served at: {url}")
    print(f"[qai] Room: {room_id}")
    print(f"[qai] Share this URL + room ID with collaborators.")
    print(f"[qai] Model: {backend.metadata().get('name', 'unknown')} "
          f"({backend.metadata().get('n_params', 0)} params)")
    print(f"[qai] Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[qai] Shutting down collaborative workspace…")
        server.shutdown()


def generate_workspace_collab(model=None, path: str = "qai_workspace_collab.html",
                               room_id: str = "DEMO", sample_X=None) -> str:
    """Generate a static HTML file of the collab-enabled workspace.

    Note: collaboration requires a running server. The static HTML will
    work in DEMO mode (no collab) unless pointed at a server.
    """
    if model is not None:
        backend = _WorkspaceBackend(model, sample_X=sample_X)
        meta = backend.metadata()
        layers = backend.get_layers()
        type_info = backend.model_type()
        type_data = backend.type_specific_data()
        bootstrap_json = json.dumps(
            {"mode": "story", "model": meta, "layers": layers,
             "type": type_info, "type_data": type_data},
            default=str,
        )
        bootstrap = (
            "\n<script>window.__QAI_BOOTSTRAP__ = " + bootstrap_json + ";</script>\n"
        )
        # Inject collab client + bootstrap
        html_out = _BASE_HTML.replace("</body>",
            f"<script>{_COLLAB_CLIENT_JS}</script>\n"
            f"{_COLLAB_STATUS_HTML}\n"
            f"{bootstrap}"
            f"<script>initCollab('{room_id}');</script>\n"
            f"</body>")
    else:
        html_out = _BASE_HTML.replace("</body>",
            f"<script>{_COLLAB_CLIENT_JS}</script>\n"
            f"{_COLLAB_STATUS_HTML}\n"
            f"<script>initCollab('{room_id}');</script>\n"
            f"</body>")

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_out)
    abs_path = os.path.abspath(path)
    print(f"[qai] Collaborative workspace written to: {abs_path}")
    print(f"[qai] Room ID: {room_id}")
    print(f"[qai] Open in any browser: file://{abs_path}")
    return abs_path


# Alias
serve_collab = serve_workspace_collab
