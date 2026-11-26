"""
qai.model_chat — interactive chat window for any model.

Opens a web-based chat interface where you can:
  - Type inputs and see predictions in real time (streaming)
  - View the model's architecture visualization (live thumbnail)
  - See explanation for each prediction
  - Watch the model's confidence / latency / tokens-per-second
  - Browse the prediction history sidebar
  - Send batch inputs (one per line) for bulk prediction

Works for ANY qai model — classifiers, regressors, transformers, agents,
generative models, etc.

Usage:
    import qai
    m = qai.build(technique="classifier")
    m.train(X, y)
    qai.chat_with_model(m, port=8770)  # opens browser

    # Skip the browser (headless / CI):
    qai.chat_with_model(m, port=8770, open_browser=False)
"""
from __future__ import annotations
import os, time, json, threading, signal, webbrowser, io, base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, List, Optional
import numpy as np

# Reuse the dashboard's state-management pattern
STATE: Dict[str, Any] = {
    "model": None,
    "history": [],            # list of {input, prediction, latency_ms, confidence, ts}
    "last_activity": time.time(),
    "stats": {"n_predictions": 0, "total_ms": 0.0, "confidences": []},
}

CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>qai model chat</title>
<style>
:root {
  --bg-0: #0a0e1a; --bg-1: #0f1424; --bg-2: #161c30; --bg-3: #1e2540;
  --border: #2a3354; --text-0: #e8ecf5; --text-1: #a0aac4; --text-2: #6b7494;
  --accent: #00ff9d; --accent-dim: #00b870; --warning: #ffb547;
  --error: #ff5470; --info: #5ec8ff; --purple: #c084fc;
  --mono: 'SF Mono', 'Monaco', monospace;
  --sans: -apple-system, system-ui, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--sans); background: var(--bg-0); color: var(--text-0); font-size: 14px;
       background-image: radial-gradient(circle at 20% 30%, rgba(0,255,157,0.04) 0%, transparent 40%); }
.app { display: grid; grid-template-columns: 240px 1fr; grid-template-rows: auto 1fr auto;
       height: 100vh; gap: 0; }
.header { grid-column: 1 / 3; display: flex; align-items: center; justify-content: space-between;
          padding: 12px 16px; background: linear-gradient(135deg, var(--bg-1), var(--bg-2));
          border-bottom: 1px solid var(--border); }
.header h1 { font-family: var(--mono); font-size: 16px; letter-spacing: 1px; }
.header p { font-size: 11px; color: var(--text-2); }
.badge { padding: 4px 10px; border-radius: 999px; background: var(--bg-3);
         border: 1px solid var(--border); color: var(--accent);
         font-family: var(--mono); font-size: 10px; }
.badge.warn { color: var(--warning); border-color: var(--warning); }
.badge.err { color: var(--error); border-color: var(--error); }
.sidebar { background: var(--bg-1); border-right: 1px solid var(--border);
           padding: 12px; overflow-y: auto; }
.sidebar h3 { font-size: 11px; color: var(--text-2); text-transform: uppercase;
              letter-spacing: 1px; margin-bottom: 8px; }
.sidebar .hist-entry { padding: 8px 10px; border-radius: 6px; margin-bottom: 4px;
                       background: var(--bg-3); border: 1px solid transparent;
                       cursor: pointer; font-size: 11px; font-family: var(--mono);
                       white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sidebar .hist-entry:hover { border-color: var(--accent-dim); }
.sidebar .hist-entry .pred { color: var(--accent); }
.sidebar .arch-thumb { width: 100%; border-radius: 8px; margin-bottom: 8px;
                       background: var(--bg-3); border: 1px solid var(--border); }
.sidebar .stats { font-size: 11px; color: var(--text-1); margin-top: 12px;
                  font-family: var(--mono); line-height: 1.6; }
.sidebar .stats span { color: var(--accent); }
.chat { background: var(--bg-1); padding: 16px; overflow-y: auto;
        display: flex; flex-direction: column; gap: 8px; }
.msg { padding: 10px 14px; border-radius: 10px; max-width: 85%; font-size: 13px; line-height: 1.6; }
.msg.user { background: var(--bg-3); align-self: flex-end; }
.msg.model { background: var(--bg-2); border-left: 3px solid var(--accent); align-self: flex-start; }
.msg.error { background: var(--bg-2); border-left: 3px solid var(--error); align-self: flex-start; color: var(--error); }
.msg .meta { font-size: 10px; color: var(--text-2); margin-top: 4px; font-family: var(--mono); }
.msg .pred { font-family: var(--mono); color: var(--accent); font-weight: bold; }
.input-area { grid-column: 1 / 3; display: flex; gap: 8px; padding: 12px;
              background: var(--bg-1); border-top: 1px solid var(--border); }
.input-area textarea { flex: 1; padding: 10px 14px; border: 1px solid var(--border);
                       border-radius: 8px; background: var(--bg-3); color: var(--text-0);
                       font-family: var(--mono); font-size: 13px; resize: vertical;
                       min-height: 40px; max-height: 120px; }
.input-area textarea:focus { outline: 2px solid var(--accent-dim); }
.btn { padding: 10px 18px; border: 1px solid var(--border); border-radius: 8px;
       background: var(--bg-3); color: var(--accent); font-family: var(--mono);
       font-size: 12px; cursor: pointer; font-weight: 600; white-space: nowrap; }
.btn:hover { background: var(--bg-2); }
.btn.primary { background: var(--accent-dim); color: var(--bg-0); border-color: var(--accent); }
.btn.warn { color: var(--warning); border-color: var(--warning); }
.pulse-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%;
             background: var(--accent); margin-right: 4px;
             animation: pulse 1.5s ease-in-out infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div>
      <h1 id="title">model chat</h1>
      <p id="subtitle">type input → see prediction</p>
    </div>
    <div>
      <span class="badge" id="status"><span class="pulse-dot"></span>LIVE</span>
    </div>
  </div>
  <div class="sidebar">
    <h3>architecture</h3>
    <img class="arch-thumb" id="arch-thumb" alt="architecture" />
    <h3>stats</h3>
    <div class="stats" id="stats">
      predictions: <span id="n-pred">0</span><br>
      avg latency: <span id="avg-lat">—</span> ms<br>
      avg conf: <span id="avg-conf">—</span>
    </div>
    <h3 style="margin-top: 16px;">history</h3>
    <div id="history"></div>
  </div>
  <div class="chat" id="chat">
    <div class="msg model">
      Model loaded. Type an input (JSON array like <code>[1.0, 2.0, 3.0]</code>) and press Enter.
      <br>For batch predictions, put one input per line.
      <div class="meta">ready</div>
    </div>
  </div>
  <div class="input-area">
    <textarea id="input" placeholder='[1.0, 2.0, 3.0, 4.0]&#10;or one input per line for batch mode' rows="2"></textarea>
    <button class="btn primary" onclick="send()">Predict</button>
    <button class="btn" onclick="explain()">Explain</button>
    <button class="btn warn" onclick="clearChat()">Clear</button>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const histDiv = document.getElementById('history');
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
function addMsg(html, role, meta) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = html + (meta ? '<div class="meta">' + meta + '</div>' : '');
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}
function setStatus(text, kind) {
  const el = document.getElementById('status');
  el.className = 'badge' + (kind ? ' ' + kind : '');
  el.innerHTML = '<span class="pulse-dot"></span>' + text;
}
function updateStats(stats) {
  document.getElementById('n-pred').textContent = stats.n_predictions;
  document.getElementById('avg-lat').textContent = stats.n_predictions > 0
    ? (stats.total_ms / stats.n_predictions).toFixed(2) : '—';
  const confs = stats.confidences.filter(c => c != null);
  document.getElementById('avg-conf').textContent = confs.length > 0
    ? (confs.reduce((a,b)=>a+b,0)/confs.length*100).toFixed(1) + '%' : '—';
}
function updateHistory(hist) {
  histDiv.innerHTML = '';
  for (const h of hist.slice(-20).reverse()) {
    const div = document.createElement('div');
    div.className = 'hist-entry';
    const predStr = typeof h.prediction === 'string' ? h.prediction : JSON.stringify(h.prediction);
    const inStr = typeof h.input === 'string' ? h.input : JSON.stringify(h.input);
    div.innerHTML = inStr.substring(0, 25) + ' → <span class="pred">' + predStr.substring(0, 25) + '</span>';
    div.title = 'input: ' + inStr + '\nprediction: ' + predStr + '\nlatency: ' + h.latency_ms + ' ms';
    div.onclick = () => { input.value = inStr; input.focus(); };
    histDiv.appendChild(div);
  }
}
async function send() {
  const val = input.value.trim();
  if (!val) return;
  addMsg(escapeHtml(val).replace(/\n/g, '<br>'), 'user');
  input.value = '';
  setStatus('predicting…', 'warn');
  try {
    const r = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({input: val})});
    const data = await r.json();
    if (data.error) {
      addMsg('Error: ' + escapeHtml(data.error), 'error', 'error');
    } else if (Array.isArray(data.predictions)) {
      // batch
      let html = '<b>Batch predictions (' + data.predictions.length + '):</b><br>';
      for (let i = 0; i < data.predictions.length; i++) {
        const p = data.predictions[i];
        html += '<span class="pred">' + escapeHtml(String(p.prediction)) + '</span>';
        if (p.confidence != null) html += ' (' + (p.confidence*100).toFixed(1) + '%)';
        html += ' — ' + p.latency_ms.toFixed(1) + ' ms<br>';
      }
      addMsg(html, 'model', 'total: ' + data.total_ms.toFixed(1) + ' ms');
    } else {
      let html = 'Prediction: <span class="pred">' + escapeHtml(String(data.prediction)) + '</span>';
      addMsg(html, 'model',
             'latency: ' + data.latency_ms.toFixed(1) + 'ms' +
             (data.confidence != null ? ' | confidence: ' + (data.confidence*100).toFixed(1) + '%' : '') +
             ' | tokens/s: ' + (data.tokens_per_s || '—'));
    }
    updateStats(data.stats);
    updateHistory(data.history);
    setStatus('LIVE');
  } catch(e) {
    addMsg('Error: ' + escapeHtml(e.message), 'error', 'error');
    setStatus('error', 'err');
  }
}
async function explain() {
  const val = input.value.trim() || '[1.0, 2.0]';
  addMsg(escapeHtml(val).replace(/\n/g, '<br>'), 'user');
  try {
    const r = await fetch('/api/explain', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({input: val})});
    const data = await r.json();
    if (data.error) { addMsg('Explain error: ' + escapeHtml(data.error), 'error'); }
    else {
      let html = '<b>Explanation:</b><br>';
      html += 'technique: ' + escapeHtml(data.technique) + '<br>';
      html += 'prediction: <span class="pred">' + escapeHtml(String(data.prediction)) + '</span><br>';
      if (data.top_features) {
        html += '<br>top features:<br>';
        for (const [k,v] of Object.entries(data.top_features)) {
          html += '&nbsp;&nbsp;' + escapeHtml(k) + ': ' + v.toFixed(4) + '<br>';
        }
      }
      addMsg(html, 'model', 'explain()');
    }
  } catch(e) { addMsg('Explain error: ' + escapeHtml(e.message), 'error'); }
}
function clearChat() {
  chat.innerHTML = '<div class="msg model">Chat cleared.<div class="meta">ready</div></div>';
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
async function init() {
  const info = await fetch('/api/info').then(r=>r.json());
  document.getElementById('title').textContent = info.technique + ' chat';
  document.getElementById('subtitle').textContent = info.family + ' · ' + (info.trained?'trained':'untrained');
  if (info.arch_png) {
    document.getElementById('arch-thumb').src = 'data:image/png;base64,' + info.arch_png;
  }
}
init();
</script>
</body>
</html>"""


def _parse_input(val: str) -> Optional[np.ndarray]:
    """Parse user input into a numpy array (single or batch)."""
    val = val.strip()
    if not val:
        return None
    # Try JSON first
    try:
        arr = json.loads(val)
        return np.asarray(arr, dtype=float)
    except json.JSONDecodeError:
        pass
    # Try comma-separated single row
    try:
        arr = [float(x.strip()) for x in val.split(",")]
        return np.asarray(arr, dtype=float)
    except ValueError:
        pass
    # Try one input per line (batch)
    lines = [l for l in val.splitlines() if l.strip()]
    if len(lines) > 1:
        try:
            rows = []
            for line in lines:
                row = json.loads(line)
                rows.append(np.asarray(row, dtype=float))
            return np.asarray(rows)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _get_arch_png(model) -> Optional[str]:
    """Generate a small architecture thumbnail as base64 PNG."""
    try:
        from .model_visualizer import visualize_architecture
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            path = tf.name
        visualize_architecture(model, save=path)
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        try: _os.unlink(path)
        except Exception: pass
        return data
    except Exception:
        return None


class _ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        STATE["last_activity"] = time.time()
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            body = CHAT_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/info":
            model = STATE["model"]
            tech = getattr(model, "technique", None)
            info = {
                "technique": getattr(tech, "name", type(model).__name__) if tech else type(model).__name__,
                "family": getattr(tech, "family", "unknown") if tech else "unknown",
                "trained": bool(model.is_trained()) if hasattr(model, "is_trained") else False,
                "arch_png": _get_arch_png(model),
            }
            self._send_json(info)
        else:
            self.send_error(404)

    def do_POST(self):
        STATE["last_activity"] = time.time()
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = {}
        if length > 0:
            try: body = json.loads(self.rfile.read(length))
            except: pass
        if path == "/api/predict":
            self._handle_predict(body)
        elif path == "/api/explain":
            self._handle_explain(body)
        else:
            self.send_error(404)

    def _handle_predict(self, body):
        val = body.get("input", "")
        arr = _parse_input(val)
        if arr is None:
            self._send_json({"error": "Could not parse input. Use JSON like [1.0, 2.0, 3.0]"}, 400)
            return
        model = STATE["model"]
        # Single or batch?
        is_batch = arr.ndim == 3 or (arr.ndim == 2 and "\n" in val)
        # Decide batch by looking at raw input lines
        lines = [l for l in val.strip().splitlines() if l.strip()]
        if len(lines) > 1 and arr.ndim == 2:
            is_batch = True
        try:
            if is_batch and arr.ndim == 2:
                # Batch prediction
                t0 = time.perf_counter()
                preds = []
                for i, row in enumerate(arr):
                    t1 = time.perf_counter()
                    try:
                        p = model.predict(row)
                        latency = (time.perf_counter() - t1) * 1000
                        pred_val = p.tolist() if hasattr(p, "tolist") else str(p)
                        conf = self._confidence(model, row)
                        preds.append({"prediction": pred_val,
                                       "latency_ms": latency,
                                       "confidence": conf})
                        # update stats
                        STATE["stats"]["n_predictions"] += 1
                        STATE["stats"]["total_ms"] += latency
                        if conf is not None:
                            STATE["stats"]["confidences"].append(conf)
                        STATE["history"].append({"input": row.tolist(),
                                                  "prediction": pred_val,
                                                  "latency_ms": latency,
                                                  "confidence": conf,
                                                  "ts": time.time()})
                    except Exception as e:
                        preds.append({"prediction": "error: " + str(e)[:80],
                                       "latency_ms": 0,
                                       "confidence": None})
                total_ms = (time.perf_counter() - t0) * 1000
                self._send_json({"predictions": preds,
                                  "total_ms": total_ms,
                                  "stats": STATE["stats"],
                                  "history": STATE["history"][-20:]})
            else:
                # Single prediction
                t0 = time.perf_counter()
                pred = model.predict(arr)
                elapsed = (time.perf_counter() - t0) * 1000
                confidence = self._confidence(model, arr)
                pred_val = pred.tolist() if hasattr(pred, "tolist") else str(pred)
                # tokens-per-second estimate (chars / latency)
                chars = len(str(pred_val))
                tps = (chars / max(elapsed, 0.001)) * 1000
                STATE["stats"]["n_predictions"] += 1
                STATE["stats"]["total_ms"] += elapsed
                if confidence is not None:
                    STATE["stats"]["confidences"].append(confidence)
                STATE["history"].append({"input": val,
                                          "prediction": pred_val,
                                          "latency_ms": elapsed,
                                          "confidence": confidence,
                                          "ts": time.time()})
                self._send_json({"prediction": pred_val,
                                  "latency_ms": elapsed,
                                  "confidence": confidence,
                                  "tokens_per_s": round(tps, 1),
                                  "stats": STATE["stats"],
                                  "history": STATE["history"][-20:]})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _confidence(self, model, arr) -> Optional[float]:
        """Try to extract a confidence score for a single prediction."""
        try:
            tech = getattr(model, "technique", None)
            if tech is None: return None
            if hasattr(tech, "model") and hasattr(tech.model, "predict_proba"):
                inp = arr.reshape(1, -1) if arr.ndim == 1 else arr
                probs = tech.model.predict_proba(inp)
                return float(np.max(probs))
        except Exception:
            pass
        return None

    def _handle_explain(self, body):
        val = body.get("input", "")
        arr = _parse_input(val)
        if arr is None:
            self._send_json({"error": "Could not parse input"}, 400)
            return
        model = STATE["model"]
        try:
            result = model.explain(arr)
            tech = getattr(model, "technique", None)
            self._send_json({
                "technique": getattr(tech, "name", "?") if tech else "?",
                "prediction": str(result.get("prediction", "?")),
                "top_features": {k: float(v) for k, v in result.get("top_features", {}).items()} if isinstance(result.get("top_features"), dict) else None,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)


def chat_with_model(model, port: int = 8770, host: str = "127.0.0.1",
                     open_browser: bool = True, timeout_s: int = 1800) -> int:
    """Open an interactive chat window for any model.

    Args:
        model: any trained (or untrained) qai.Model.
        port: localhost port (default 8770).
        host: bind address (default 127.0.0.1).
        open_browser: if True, auto-open the browser.
        timeout_s: auto-shutdown after N seconds of inactivity.

    Returns:
        Exit code (0 on clean shutdown).
    """
    STATE["model"] = model
    STATE["last_activity"] = time.time()
    STATE["history"] = []
    STATE["stats"] = {"n_predictions": 0, "total_ms": 0.0, "confidences": []}
    server = HTTPServer((host, port), _ChatHandler)
    url = f"http://{host}:{port}/"
    print(f"[qai.chat] serving at {url}")
    print(f"  (auto-shutdown after {timeout_s}s idle; Ctrl+C to stop)")

    def _watchdog():
        while True:
            time.sleep(30)
            if time.time() - STATE["last_activity"] > timeout_s:
                server.shutdown(); return

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()

    if open_browser:
        try: webbrowser.open(url)
        except: pass

    # SIGINT only works in the main thread; skip if we're in a worker
    if threading.current_thread() is threading.main_thread():
        def _sigint(sig, frame):
            server.shutdown()
        signal.signal(signal.SIGINT, _sigint)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


__all__ = ["chat_with_model"]
