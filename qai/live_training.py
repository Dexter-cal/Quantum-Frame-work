"""
qai.live_training -- real-time training dashboard.

Opens a localhost web dashboard that shows training curves in real time
as the model trains. Stdlib-only (no Flask, no TensorBoard).

Usage:
    import qai
    import numpy as np

    X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)

    # Option A: one-shot helper (trains + shows dashboard + blocks)
    qai.live_training(
        qai.build(technique="neural_network", epochs=50),
        X, y,
        port=8765,
    )

    # Option B: as a callback (dashboard opens, training runs in the same process)
    from qai.live_training import LiveTrainingCallback
    cb = LiveTrainingCallback(port=8765)
    m = qai.build(technique="neural_network", epochs=50)
    m.train(X, y, callbacks=[cb])
    # dashboard stays open until you Ctrl+C

The dashboard shows:
  - Loss curve (live, updates every epoch)
  - Accuracy curve (if available)
  - Current epoch / total epochs
  - Time elapsed / ETA
  - A live console of training events
"""
from __future__ import annotations
import os, sys, time, json, threading, webbrowser, signal
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
import numpy as np

# Import the Callback base class
try:
    from .callbacks import Callback
except ImportError:
    # Fallback if callbacks module isn't available
    class Callback:  # type: ignore
        def on_epoch_end(self, *args, **kwargs): pass
        def on_train_end(self, *args, **kwargs): pass


# =====================================================================
# Live dashboard HTML
# =====================================================================
LIVE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>qai live training</title>
<style>
:root {
  --bg-0: #0a0e1a; --bg-1: #0f1424; --bg-2: #161c30; --bg-3: #1e2540;
  --border: #2a3354; --text-0: #e8ecf5; --text-1: #a0aac4; --text-2: #6b7494;
  --accent: #00ff9d; --accent-dim: #00b870; --warning: #ffb547; --error: #ff5470; --info: #5ec8ff;
  --mono: 'SF Mono', 'Monaco', 'Cascadia Code', monospace;
  --sans: -apple-system, system-ui, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--sans); background: var(--bg-0); color: var(--text-0); font-size: 14px;
       background-image: radial-gradient(circle at 20% 30%, rgba(0,255,157,0.04) 0%, transparent 40%); }
.app { max-width: 1100px; margin: 0 auto; padding: 24px; display: grid; gap: 16px; }
.header { padding: 20px 24px; background: linear-gradient(135deg, var(--bg-1), var(--bg-2));
          border: 1px solid var(--border); border-radius: 16px; display: flex; justify-content: space-between; }
.header h1 { font-family: var(--mono); font-size: 22px; letter-spacing: 2px; }
.header p { font-size: 12px; color: var(--text-2); margin-top: 2px; }
.badge { padding: 6px 12px; border-radius: 999px; background: var(--bg-3); border: 1px solid var(--border);
         color: var(--accent); font-family: var(--mono); font-size: 12px; }
.badge.idle { color: var(--text-2); }
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.stat { padding: 16px; background: linear-gradient(135deg, var(--bg-1), var(--bg-2));
        border: 1px solid var(--border); border-radius: 12px; }
.stat-label { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
              letter-spacing: 1px; color: var(--text-2); margin-bottom: 6px; }
.stat-value { font-family: var(--mono); font-size: 24px; font-weight: 700; }
.panel { padding: 20px; background: linear-gradient(135deg, var(--bg-1), var(--bg-2));
         border: 1px solid var(--border); border-radius: 14px; }
.panel-title { font-family: var(--mono); font-size: 12px; text-transform: uppercase;
               letter-spacing: 1.5px; color: var(--text-1); margin-bottom: 12px; }
canvas { display: block; width: 100%; height: 240px; }
.console { background: var(--bg-0); border: 1px solid var(--border); border-radius: 8px; padding: 12px;
           font-family: var(--mono); font-size: 11px; line-height: 1.6; color: var(--text-1);
           height: 180px; overflow-y: auto; }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div>
      <h1 id="title">live training</h1>
      <p id="subtitle">waiting for training events...</p>
    </div>
    <div><span class="badge idle" id="status-badge"><span id="status-text">IDLE</span></span></div>
  </div>

  <div class="stats-grid">
    <div class="stat"><div class="stat-label">EPOCH</div><div class="stat-value" id="stat-epoch">0 / 0</div></div>
    <div class="stat"><div class="stat-label">LOSS</div><div class="stat-value" id="stat-loss">—</div></div>
    <div class="stat"><div class="stat-label">ELAPSED</div><div class="stat-value" id="stat-elapsed">0s</div></div>
    <div class="stat"><div class="stat-label">ETA</div><div class="stat-value" id="stat-eta">—</div></div>
  </div>

  <div class="panel">
    <div class="panel-title">LOSS CURVE (live)</div>
    <canvas id="loss-chart"></canvas>
  </div>

  <div class="panel">
    <div class="panel-title">EVENT CONSOLE</div>
    <div class="console" id="console"></div>
  </div>
</div>
<script>
let lossHistory = [];
let startTime = null;
async function poll() {
  try {
    const r = await fetch('/api/state').then(r => r.json());
    document.getElementById('title').textContent = r.technique + ' live training';
    document.getElementById('subtitle').textContent = r.technique_class + ' · ' + (r.training ? 'training' : 'idle');
    document.getElementById('status-text').textContent = r.training ? 'TRAINING' : 'IDLE';
    document.getElementById('status-badge').className = 'badge ' + (r.training ? '' : 'idle');
    document.getElementById('stat-epoch').textContent = r.epoch + ' / ' + (r.total_epochs || '?');
    document.getElementById('stat-loss').textContent = r.last_loss !== null ? r.last_loss.toFixed(4) : '—';
    if (r.start_time && r.epoch > 0) {
      const elapsed = (Date.now() / 1000) - r.start_time;
      document.getElementById('stat-elapsed').textContent = elapsed.toFixed(1) + 's';
      if (r.epoch > 1 && r.total_epochs) {
        const eta = (elapsed / r.epoch) * (r.total_epochs - r.epoch);
        document.getElementById('stat-eta').textContent = eta.toFixed(1) + 's';
      }
    }
    lossHistory = r.loss_history || [];
    drawChart();
    const c = document.getElementById('console');
    c.innerHTML = (r.events || []).slice(-20).map(e => '<div>' + e + '</div>').join('');
    c.scrollTop = c.scrollHeight;
  } catch (e) { /* ignore */ }
}
function drawChart() {
  const canvas = document.getElementById('loss-chart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr; canvas.height = 240 * dpr;
  ctx.scale(dpr, dpr);
  const W = canvas.clientWidth, H = 240;
  ctx.clearRect(0, 0, W, H);
  if (lossHistory.length < 2) {
    ctx.fillStyle = 'rgba(160,170,196,0.5)'; ctx.font = '12px monospace'; ctx.textAlign = 'center';
    ctx.fillText('waiting for training data...', W/2, H/2); return;
  }
  const min = Math.min(...lossHistory), max = Math.max(...lossHistory);
  const range = max - min || 1;
  // grid
  ctx.strokeStyle = 'rgba(160,170,196,0.1)'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  // area
  ctx.beginPath();
  lossHistory.forEach((v, i) => {
    const x = (i / (lossHistory.length - 1)) * W;
    const y = H - ((v - min) / range) * H;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(0,255,157,0.3)'); grad.addColorStop(1, 'rgba(0,255,157,0)');
  ctx.fillStyle = grad; ctx.fill();
  // line
  ctx.strokeStyle = '#00ff9d'; ctx.lineWidth = 2; ctx.beginPath();
  lossHistory.forEach((v, i) => {
    const x = (i / (lossHistory.length - 1)) * W;
    const y = H - ((v - min) / range) * H;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // labels
  ctx.fillStyle = 'rgba(160,170,196,0.6)'; ctx.font = '10px monospace'; ctx.textAlign = 'left';
  ctx.fillText(max.toFixed(3), 4, 12); ctx.fillText(min.toFixed(3), 4, H - 4);
}
setInterval(poll, 500); poll();
</script>
</body>
</html>"""


# =====================================================================
# Server state + handlers
# =====================================================================
STATE: Dict[str, Any] = {
    "technique": "?",
    "technique_class": "?",
    "training": False,
    "epoch": 0,
    "total_epochs": 0,
    "last_loss": None,
    "loss_history": [],
    "events": [],
    "start_time": None,
}


def _set_state(**kwargs):
    STATE.update(kwargs)


def _add_event(msg: str):
    ts = time.strftime("%H:%M:%S")
    STATE["events"].append(f"[{ts}] {msg}")
    if len(STATE["events"]) > 100:
        STATE["events"] = STATE["events"][-100:]


def _add_loss(loss: float):
    STATE["loss_history"].append(float(loss))
    STATE["last_loss"] = float(loss)
    if len(STATE["loss_history"]) > 1000:
        STATE["loss_history"] = STATE["loss_history"][-1000:]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = LIVE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            body = json.dumps(STATE, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


class LiveTrainingCallback(Callback):
    """A callback that opens a live training dashboard.

    Usage:
        cb = LiveTrainingCallback(port=8765)
        model.train(X, y, callbacks=[cb])
    """
    def __init__(self, port: int = 8765, open_browser: bool = True):
        self.port = port
        self.open_browser = open_browser
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._started = False

    def _start_server(self):
        if self._started:
            return
        self._started = True
        self.server = HTTPServer(("127.0.0.1", self.port), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        url = f"http://127.0.0.1:{self.port}/"
        print(f"[qai.live_training] dashboard at {url}")
        if self.open_browser:
            try: webbrowser.open(url)
            except Exception: pass
        _add_event("dashboard started")

    def on_train_start(self, model, **kwargs):
        tech = getattr(model, "technique", None)
        _set_state(
            technique=getattr(tech, "name", "?"),
            technique_class=type(tech).__name__ if tech else "?",
            training=True,
            start_time=time.time(),
        )
        _add_event(f"training started: {STATE['technique']}")
        self._start_server()
        # Start a polling thread that watches the technique's loss history
        # (since not all techniques fire on_epoch_end). The thread exits when
        # training=False.
        self._poll_thread = threading.Thread(
            target=self._poll_loss_history, args=(model,), daemon=True
        )
        self._poll_thread.start()

    def _poll_loss_history(self, model):
        """Background poller: read the technique's loss history every 200ms
        and add new losses to the dashboard state.

        Note: due to Python's GIL, CPU-bound training (like a tight NumPy
        loop) won't yield to this thread until each epoch completes. For
        fast techniques, the dashboard will only show the final curve after
        training finishes. For slow techniques (large datasets, deep nets),
        the curve will update in real time as each epoch yields.
        """
        tech = getattr(model, "technique", None)
        last_count = 0
        while STATE.get("training", False):
            try:
                # Many qai techniques expose loss_history_ (NeuralNetwork,
                # CNN, Transformer, ...)
                hist = None
                for attr in ("loss_history_", "loss_history", "_loss_history"):
                    if tech is not None and hasattr(tech, attr):
                        h = getattr(tech, attr)
                        if isinstance(h, list) and h:
                            hist = h
                            break
                if hist is not None and len(hist) > last_count:
                    for i in range(last_count, len(hist)):
                        _add_loss(hist[i])
                        STATE["epoch"] = i + 1
                        _add_event(f"epoch {i+1}: loss={hist[i]:.4f}")
                    last_count = len(hist)
            except Exception:
                pass
            time.sleep(0.2)
        # Training is done -- do one final sweep to catch all losses
        try:
            tech = getattr(model, "technique", None)
            for attr in ("loss_history_", "loss_history", "_loss_history"):
                if tech is not None and hasattr(tech, attr):
                    h = getattr(tech, attr)
                    if isinstance(h, list) and h:
                        for i in range(last_count, len(h)):
                            _add_loss(h[i])
                            STATE["epoch"] = i + 1
                            _add_event(f"epoch {i+1}: loss={h[i]:.4f}")
                        break
        except Exception:
            pass

    def on_epoch_end(self, model, **kwargs):
        epoch = kwargs.get("epoch")
        loss = kwargs.get("loss")
        if epoch is not None:
            STATE["epoch"] = epoch
        if loss is not None:
            _add_loss(loss)
        if "total_epochs" in kwargs:
            STATE["total_epochs"] = kwargs["total_epochs"]
        _add_event(f"epoch {epoch or STATE['epoch']}: loss={loss}")

    def on_train_end(self, model, **kwargs):
        _set_state(training=False)
        _add_event("training complete")
        # Keep the server running so the user can inspect the final curve
        # (the daemon thread will exit when the main process does)

    def stop(self):
        """Stop the dashboard server."""
        if self.server:
            self.server.shutdown()
            self.server = None
            self._started = False


def live_training(model, X, y=None, port: int = 8765,
                  open_browser: bool = True, **train_kwargs) -> Any:
    """Train `model` on (X, y) with a live dashboard, blocking until done.

    Args:
        model: a qai.Model.
        X, y: training data.
        port: dashboard port.
        open_browser: auto-open the browser.
        **train_kwargs: passed to model.train().

    Returns:
        The trained model (for chaining).
    """
    cb = LiveTrainingCallback(port=port, open_browser=open_browser)
    callbacks = train_kwargs.pop("callbacks", []) or []
    callbacks.append(cb)
    model.train(X, y, callbacks=callbacks, **train_kwargs)
    # Block until the user Ctrl+Cs (so they can inspect the dashboard)
    print(f"\n[qai.live_training] training complete. Dashboard at http://127.0.0.1:{port}/")
    print("[qai.live_training] press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cb.stop()
    return model


__all__ = ["LiveTrainingCallback", "live_training"]
