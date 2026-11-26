"""
qai.dashboard -- a generic web UI for any qai model.

Opens a localhost HTTP server (Python stdlib only -- no Flask, no
FastAPI) and serves a single-page dashboard that lets you:

  - See the model's metadata (technique, params, training history)
  - Run predictions from the browser (POST JSON, get prediction back)
  - View explanations (model.explain() output, rendered as a table)
  - Inspect the bundle (if the model was loaded from a .qmodel)
  - See training metrics (loss history, accuracy, etc.)

Usage:
    import qai
    model = qai.build(technique="regression")
    model.train(X, y)
    qai.dashboard(model, port=8765)  # opens browser

Works for any qai.Model AND any qai.AgentBase subclass. For agents,
the dashboard shows episode history + hyperparameter scheduler state
in addition to the standard model fields.
"""
from __future__ import annotations
import os, sys, time, json, threading, webbrowser, signal, argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, Optional
import numpy as np


# =====================================================================
# Dashboard HTML (single-file, no external deps, dark AI aesthetic)
# =====================================================================
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>qai dashboard</title>
<style>
:root {
  --bg-0: #0a0e1a; --bg-1: #0f1424; --bg-2: #161c30; --bg-3: #1e2540;
  --border: #2a3354; --text-0: #e8ecf5; --text-1: #a0aac4; --text-2: #6b7494;
  --accent: #00ff9d; --accent-dim: #00b870; --warning: #ffb547;
  --error: #ff5470; --info: #5ec8ff; --purple: #c084fc;
  --mono: 'SF Mono', 'Monaco', 'Cascadia Code', 'Consolas', monospace;
  --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--sans); background: var(--bg-0); color: var(--text-0);
       font-size: 14px; line-height: 1.5;
       background-image: radial-gradient(circle at 20% 30%, rgba(0,255,157,0.04) 0%, transparent 40%),
                         radial-gradient(circle at 80% 70%, rgba(192,132,252,0.04) 0%, transparent 40%); }
.app { max-width: 1200px; margin: 0 auto; padding: 24px; display: grid; gap: 16px; }
.header { display: flex; align-items: center; justify-content: space-between;
          padding: 20px 24px; background: linear-gradient(135deg, var(--bg-1), var(--bg-2));
          border: 1px solid var(--border); border-radius: 16px; position: relative; overflow: hidden; }
.header::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
                  background: linear-gradient(90deg, transparent, var(--accent), transparent); opacity: 0.5; }
.header h1 { font-family: var(--mono); font-size: 22px; font-weight: 700; letter-spacing: 2px; }
.header p { font-size: 12px; color: var(--text-2); letter-spacing: 0.5px; margin-top: 2px; }
.badge { padding: 6px 12px; border-radius: 999px; background: var(--bg-3); border: 1px solid var(--border);
         color: var(--text-1); font-family: var(--mono); font-size: 12px; }
.badge.live { border-color: var(--accent-dim); color: var(--accent); }
.panel { background: linear-gradient(135deg, var(--bg-1), var(--bg-2)); border: 1px solid var(--border);
         border-radius: 14px; padding: 20px; }
.panel-title { font-family: var(--mono); font-size: 12px; text-transform: uppercase;
               letter-spacing: 1.5px; color: var(--text-1); margin-bottom: 16px;
               display: flex; align-items: center; }
.panel-title::before { content: ''; display: inline-block; width: 4px; height: 14px;
                       background: var(--accent); margin-right: 10px; border-radius: 2px; }
.kv { display: grid; grid-template-columns: 200px 1fr; gap: 8px 16px; font-family: var(--mono); font-size: 13px; }
.kv .k { color: var(--text-2); }
.kv .v { color: var(--text-0); word-break: break-all; }
.input-row { display: flex; gap: 8px; margin-bottom: 12px; }
.input-row input { flex: 1; padding: 10px 14px; border: 1px solid var(--border); border-radius: 8px;
                   background: var(--bg-3); color: var(--text-0); font-family: var(--mono); font-size: 13px; }
.input-row input:focus { outline: 2px solid var(--accent-dim); outline-offset: 2px; }
.btn { padding: 10px 18px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-3);
       color: var(--text-0); font-family: var(--mono); font-size: 12px; font-weight: 600; cursor: pointer;
       text-transform: uppercase; letter-spacing: 0.5px; }
.btn:hover { border-color: var(--accent-dim); color: var(--accent); }
.btn.primary { background: linear-gradient(135deg, var(--accent-dim), var(--accent));
               border-color: var(--accent); color: var(--bg-0); }
.output { background: var(--bg-0); border: 1px solid var(--border); border-radius: 8px; padding: 14px;
          font-family: var(--mono); font-size: 12px; line-height: 1.7; white-space: pre-wrap;
          color: var(--text-0); min-height: 60px; }
.output.empty { color: var(--text-2); font-style: italic; }
table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; }
table td { padding: 6px 10px; border-bottom: 1px solid var(--bg-2); }
table td:first-child { color: var(--text-2); width: 40%; }
.explain-bar { display: inline-block; height: 14px; background: var(--accent-dim); vertical-align: middle;
               border-radius: 2px; margin-right: 8px; }
.footer { text-align: center; font-family: var(--mono); font-size: 10px; color: var(--text-2); padding: 12px; }
.toast { position: fixed; bottom: 24px; right: 24px; padding: 12px 18px; background: var(--bg-3);
         border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 8px;
         font-family: var(--mono); font-size: 12px; max-width: 360px;
         box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
.toast.error { border-left-color: var(--error); }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div>
      <h1 id="title">qai dashboard</h1>
      <p id="subtitle">loading model...</p>
    </div>
    <div>
      <span class="badge live"><span id="status">LIVE</span></span>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">MODEL METADATA</div>
    <div class="kv" id="metadata">loading...</div>
  </div>

  <div class="panel">
    <div class="panel-title">RUN PREDICTION</div>
    <div class="input-row">
      <input id="predict-input" placeholder='[1.0, 2.0, 3.0]  (JSON array)' />
      <button class="btn primary" onclick="runPredict()">Predict</button>
      <button class="btn" onclick="runExplain()">Explain</button>
    </div>
    <div class="output empty" id="predict-output">Run a prediction to see results here.</div>
  </div>

  <div class="panel">
    <div class="panel-title">EXPLAIN OUTPUT</div>
    <div class="output empty" id="explain-output">Click "Explain" to see model.explain() output.</div>
  </div>

  <div class="panel">
    <div class="panel-title">TRAINING HISTORY</div>
    <div class="output empty" id="history-output">No training history yet.</div>
  </div>

  <div class="footer">qai.dashboard · stdlib-only · auto-shuts down after 30 min idle</div>
</div>
<div id="toast" style="display:none" class="toast"></div>

<script>
async function api(path, body) {
  const opts = body !== undefined ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
function toast(msg, kind='info') {
  const el = document.getElementById('toast');
  el.textContent = msg; el.className = 'toast ' + kind; el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', 3000);
}
function fmt(v) {
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'object') return JSON.stringify(v, null, 2);
  return String(v);
}
async function init() {
  try {
    const info = await api('/api/info');
    document.getElementById('title').textContent = `${info.technique || 'model'} dashboard`;
    document.getElementById('subtitle').textContent = `qai · ${info.agent_class || 'Model'} · ${info.trained ? 'trained' : 'untrained'}`;
    const meta = await api('/api/metadata');
    const metaEl = document.getElementById('metadata');
    metaEl.innerHTML = Object.entries(meta).map(([k,v]) =>
      `<div class="k">${k}</div><div class="v">${fmt(v)}</div>`).join('');
    const hist = await api('/api/history');
    document.getElementById('history-output').textContent = JSON.stringify(hist, null, 2);
    document.getElementById('history-output').classList.remove('empty');
    toast('Dashboard ready.');
  } catch (e) { toast('Init failed: ' + e.message, 'error'); }
}
async function runPredict() {
  const inp = document.getElementById('predict-input').value;
  try {
    const input_data = JSON.parse(inp);
    const r = await api('/api/predict', {input: input_data});
    const el = document.getElementById('predict-output');
    el.textContent = `prediction: ${fmt(r.prediction)}`;
    el.classList.remove('empty');
  } catch (e) {
    toast('Predict failed: ' + e.message, 'error');
  }
}
async function runExplain() {
  const inp = document.getElementById('predict-input').value;
  try {
    const input_data = JSON.parse(inp);
    const r = await api('/api/explain', {input: input_data});
    const el = document.getElementById('explain-output');
    // Render the explanation as a table
    let html = '<table>';
    for (const [k, v] of Object.entries(r)) {
      if (k === 'top_features' && typeof v === 'object') {
        const maxAbs = Math.max(...Object.values(v).map(x => Math.abs(x)), 1);
        html += `<tr><td>${k}</td><td>` + Object.entries(v).map(([fk, fv]) => {
          const w = Math.abs(fv) / maxAbs * 100;
          const sign = fv >= 0 ? '+' : '';
          return `<div><span class="explain-bar" style="width:${w}px"></span>${fk}: ${sign}${fv.toFixed(4)}</div>`;
        }).join('') + '</td></tr>';
      } else if (typeof v === 'object') {
        html += `<tr><td>${k}</td><td><pre>${JSON.stringify(v, null, 2)}</pre></td></tr>`;
      } else {
        html += `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`;
      }
    }
    html += '</table>';
    el.innerHTML = html;
    el.classList.remove('empty');
  } catch (e) {
    toast('Explain failed: ' + e.message, 'error');
  }
}
init();
</script>
</body>
</html>"""


# =====================================================================
# State + handlers
# =====================================================================
STATE: Dict[str, Any] = {
    "model": None,
    "last_activity": time.time(),
    "start_time": time.time(),
}


def _model_info(model) -> Dict[str, Any]:
    """Extract lightweight metadata for the dashboard header."""
    info = {
        "technique": getattr(getattr(model, "technique", None), "name", None) or type(model).__name__,
        "agent_class": type(model).__name__,
        "trained": bool(getattr(model, "is_trained", lambda: False)()),
    }
    return info


def _model_metadata(model) -> Dict[str, Any]:
    """Extract richer metadata for the metadata panel."""
    out: Dict[str, Any] = {}
    out["class"] = type(model).__name__
    if hasattr(model, "technique_name"):
        out["technique_name"] = model.technique_name
    tech = getattr(model, "technique", None)
    if tech is not None:
        out["technique_class"] = type(tech).__name__
        out["family"] = getattr(tech, "family", "?")
        out["compatible_learning_techniques"] = getattr(tech, "compatible_learning_techniques", "?")
        # Common hyperparameters
        for attr in ("hidden_dim", "epochs", "lr", "n_estimators", "max_depth",
                     "n_clusters", "n_components", "k", "alpha", "kernel"):
            if hasattr(tech, attr):
                out[attr] = getattr(tech, attr)
    # AgentBase-specific
    if hasattr(model, "scheduler"):
        out["hparams"] = model.scheduler.params
        out["performance_slope"] = round(model.scheduler.slope, 4) if hasattr(model.scheduler, "slope") else None
        out["n_episodes"] = getattr(model, "n_episodes_completed", len(getattr(model, "episodes", [])))
        out["n_successes"] = getattr(model, "n_successes", 0)
        out["replay_buffer_size"] = len(model.replay_buffer) if hasattr(model, "replay_buffer") else 0
    # Training history
    if hasattr(model, "_history"):
        out["history_events"] = len(model._history)
    # Status method
    if hasattr(model, "status"):
        try:
            s = model.status()
            for k, v in s.items():
                out[f"status.{k}"] = v
        except Exception:
            pass
    return out


def handle_api_info(_) -> Dict[str, Any]:
    return _model_info(STATE["model"])


def handle_api_metadata(_) -> Dict[str, Any]:
    return _model_metadata(STATE["model"])


def handle_api_predict(body) -> Dict[str, Any]:
    input_data = body.get("input")
    if input_data is None:
        raise ValueError("missing 'input' field")
    arr = np.asarray(input_data, dtype=float)
    pred = STATE["model"].predict(arr)
    if hasattr(pred, "tolist"):
        pred = pred.tolist()
    return {"prediction": pred}


def handle_api_explain(body) -> Dict[str, Any]:
    input_data = body.get("input")
    if input_data is None:
        raise ValueError("missing 'input' field")
    arr = np.asarray(input_data, dtype=float)
    result = STATE["model"].explain(arr)
    # Convert numpy types for JSON
    def _conv(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, dict): return {k: _conv(vv) for k, vv in v.items()}
        if isinstance(v, list): return [_conv(x) for x in v]
        return v
    return {k: _conv(v) for k, v in result.items()}


def handle_api_history(_) -> Dict[str, Any]:
    model = STATE["model"]
    if hasattr(model, "_history"):
        return {"history": model._history[-20:]}
    if hasattr(model, "episodes"):
        return {"episodes": model.episodes[-20:]}
    return {"history": [], "note": "no training history available"}


API_ROUTES = {
    "/api/info":     ("GET",  handle_api_info),
    "/api/metadata": ("GET",  handle_api_metadata),
    "/api/predict":  ("POST", handle_api_predict),
    "/api/explain":  ("POST", handle_api_explain),
    "/api/history":  ("GET",  handle_api_history),
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass
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
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        STATE["last_activity"] = time.time()
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send_html(DASHBOARD_HTML); return
        if path in API_ROUTES and API_ROUTES[path][0] == "GET":
            try: self._send_json(API_ROUTES[path][1](parse_qs(urlparse(self.path).query)))
            except Exception as e: self._send_json({"error": str(e)}, 500)
            return
        self.send_error(404)
    def do_POST(self):
        STATE["last_activity"] = time.time()
        path = urlparse(self.path).path
        if path in API_ROUTES and API_ROUTES[path][0] == "POST":
            length = int(self.headers.get("Content-Length", 0))
            body = {}
            if length > 0:
                try: body = json.loads(self.rfile.read(length))
                except Exception as e:
                    self._send_json({"error": f"bad JSON: {e}"}, 400); return
            try: self._send_json(API_ROUTES[path][1](body))
            except Exception as e: self._send_json({"error": str(e)}, 500)
            return
        self.send_error(404)


def _auto_shutdown_watcher(server: HTTPServer, timeout_s: int = 1800):
    while True:
        time.sleep(30)
        if time.time() - STATE["last_activity"] > timeout_s:
            print(f"[dashboard] {timeout_s}s idle, shutting down.")
            server.shutdown(); return


def dashboard(model, port: int = 8765, host: str = "127.0.0.1",
              open_browser: bool = True, timeout_s: int = 1800,
              verbose: bool = True) -> int:
    """Open a web dashboard for `model` in your browser.

    Args:
        model: any qai.Model (from qai.build(...)) or qai.AgentBase subclass.
        port: localhost port to serve on (default 8765).
        host: bind address (default 127.0.0.1 -- localhost only).
        open_browser: if True, auto-open the user's default browser.
        timeout_s: auto-shutdown after this many seconds of inactivity.
        verbose: if True, print the URL + status.

    Returns:
        Exit code (0 on clean shutdown).
    """
    STATE["model"] = model
    STATE["last_activity"] = time.time()
    STATE["start_time"] = time.time()
    server = HTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    if verbose:
        info = _model_info(model)
        print(f"[qai.dashboard] serving {info['technique']} ({info['agent_class']}) at {url}")
        print(f"  (auto-shutdown after {timeout_s}s idle; Ctrl+C to stop)")
    watcher = threading.Thread(target=_auto_shutdown_watcher,
                                args=(server, timeout_s), daemon=True)
    watcher.start()
    if open_browser:
        try: webbrowser.open(url)
        except Exception: pass
    def _sigint(sig, frame):
        if verbose: print("\n[qai.dashboard] shutting down...")
        server.shutdown()
    signal.signal(signal.SIGINT, _sigint)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Open a qai dashboard for a .qmodel bundle")
    parser.add_argument("bundle", help="path to a .qmodel file")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sign-key", default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    import qai
    bundle = qai.load_bundle(args.bundle, sign_key=args.sign_key)
    return dashboard(bundle.model, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
