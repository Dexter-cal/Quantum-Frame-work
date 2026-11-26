"""
qai.webui -- interactive dashboard for browsing models/techniques/benchmarks.

A self-contained HTML+JS dashboard that doesn't require any front-end
framework. Generates a single-page app you can open in any browser.

Usage:
    qai.webui.serve_dashboard(port=8080)  # blocking; visit http://localhost:8080
    qai.webui.generate_static_html(path="dashboard.html")  # save and open later

The dashboard shows:
  - Technique catalog (filterable by family)
  - Creative ideas catalog (filterable by experimental file)
  - Live system snapshot (CPU/RAM/disk)
  - Model benchmark results (if you've run any)
  - Quickstart code samples
"""
from __future__ import annotations
import html
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Dashboard data collectors
# ============================================================================
def _collect_techniques() -> List[Dict[str, str]]:
    """Get all registered techniques with their families."""
    try:
        import qai
        return [{"name": name, "family": getattr(cls, "family", "unknown")}
                 for name, cls in qai.techniques._REGISTRY.items()]
    except Exception:
        return []


def _collect_creative_ideas() -> List[Dict[str, str]]:
    """Get all creative idea classes."""
    try:
        import qai
        return [{"name": n} for n in qai.creative.__all__ if not n.startswith("_")]
    except Exception:
        return []


def _collect_system_snapshot() -> Dict[str, Any]:
    """Get current system snapshot."""
    try:
        import qai
        return qai.system.system_snapshot()
    except Exception as e:
        return {"error": str(e)}


def _collect_quickstart() -> str:
    """Get the quickstart tutorial."""
    try:
        import qai
        return qai.get_quickstart()
    except Exception:
        return "Quickstart unavailable."


# ============================================================================
# HTML template
# ============================================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Quantum AI Framework — Dashboard</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; margin-bottom: 16px; }
h2 { color: #79c0ff; margin: 20px 0 10px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
        padding: 16px; margin-bottom: 12px; }
.stat { display: inline-block; margin-right: 24px; }
.stat .value { font-size: 28px; font-weight: bold; color: #58a6ff; }
.stat .label { font-size: 12px; color: #8b949e; text-transform: uppercase; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 8px; border-bottom: 1px solid #21262d; }
th { color: #8b949e; font-weight: 600; }
tr:hover { background: #1f2937; }
.badge { background: #238636; color: white; padding: 2px 8px; border-radius: 12px;
         font-size: 11px; display: inline-block; margin-right: 4px; }
.badge.family { background: #1f6feb; }
input.filter { width: 100%; padding: 8px; margin: 8px 0; background: #0d1117;
                border: 1px solid #30363d; color: #c9d1d9; border-radius: 4px; }
pre { background: #161b22; padding: 12px; border-radius: 6px; overflow-x: auto;
       font-size: 12px; color: #c9d1d9; }
code { background: #21262d; padding: 2px 6px; border-radius: 3px; font-size: 12px; }
.tab { display: inline-block; padding: 8px 16px; cursor: pointer; border: 1px solid #30363d;
        border-bottom: none; border-radius: 6px 6px 0 0; background: #161b22; }
.tab.active { background: #238636; color: white; }
.tab-content { display: none; }
.tab-content.active { display: block; }
.health-good { color: #3fb950; }
.health-warn { color: #d29922; }
.health-bad { color: #f85149; }
</style>
</head>
<body>
<h1>Quantum AI Framework — Dashboard</h1>

<div class="card">
  <div class="stat">
    <div class="value" id="n-techniques">__N_TECHNIQUES__</div>
    <div class="label">Techniques</div>
  </div>
  <div class="stat">
    <div class="value" id="n-creative">__N_CREATIVE__</div>
    <div class="label">Creative Ideas</div>
  </div>
  <div class="stat">
    <div class="value" id="cpu-pct">__CPU_PCT__%</div>
    <div class="label">CPU</div>
  </div>
  <div class="stat">
    <div class="value" id="ram-pct">__RAM_PCT__%</div>
    <div class="label">RAM</div>
  </div>
  <div class="stat">
    <div class="value" id="uptime">__UPTIME__</div>
    <div class="label">Uptime</div>
  </div>
</div>

<div class="tab active" onclick="showTab('techniques')">Techniques</div>
<div class="tab" onclick="showTab('creative')">Creative Ideas</div>
<div class="tab" onclick="showTab('system')">System</div>
<div class="tab" onclick="showTab('quickstart')">Quickstart</div>

<div id="techniques" class="tab-content active">
  <h2>Technique Catalog</h2>
  <input class="filter" type="text" placeholder="Filter techniques..." onkeyup="filterTable('tech-table', this.value)">
  <table id="tech-table">
    <thead><tr><th>Name</th><th>Family</th><th>Build Command</th></tr></thead>
    <tbody>__TECHNIQUE_ROWS__</tbody>
  </table>
</div>

<div id="creative" class="tab-content">
  <h2>Creative Ideas (75+ experimental modules)</h2>
  <input class="filter" type="text" placeholder="Filter ideas..." onkeyup="filterTable('creative-table', this.value)">
  <table id="creative-table">
    <thead><tr><th>Idea Name</th></tr></thead>
    <tbody>__CREATIVE_ROWS__</tbody>
  </table>
</div>

<div id="system" class="tab-content">
  <h2>System Snapshot</h2>
  <pre id="system-snapshot">__SYSTEM_SNAPSHOT__</pre>
</div>

<div id="quickstart" class="tab-content">
  <h2>Quickstart Tutorial</h2>
  <pre>__QUICKSTART__</pre>
</div>

<script>
function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(name).classList.add('active');
  event.target.classList.add('active');
}
function filterTable(tableId, query) {
  const table = document.getElementById(tableId);
  const rows = table.querySelectorAll('tbody tr');
  rows.forEach(row => {
    const text = row.textContent.toLowerCase();
    row.style.display = text.includes(query.toLowerCase()) ? '' : 'none';
  });
}
</script>
</body>
</html>
"""


def _build_technique_rows() -> str:
    techniques = _collect_techniques()
    rows = []
    for t in sorted(techniques, key=lambda x: x["name"]):
        rows.append(
            f"<tr><td><code>{html.escape(t['name'])}</code></td>"
            f"<td><span class='badge family'>{html.escape(t['family'])}</span></td>"
            f"<td><code>qai.build(technique='{html.escape(t['name'])}')</code></td></tr>"
        )
    return "\n".join(rows)


def _build_creative_rows() -> str:
    ideas = _collect_creative_ideas()
    rows = []
    for idea in sorted(ideas, key=lambda x: x["name"]):
        rows.append(f"<tr><td><code>{html.escape(idea['name'])}</code></td></tr>")
    return "\n".join(rows)


def _build_system_snapshot() -> str:
    snap = _collect_system_snapshot()
    return html.escape(json.dumps(snap, indent=2, default=str))


def generate_static_html(path: str = "qai_dashboard.html") -> str:
    """Generate a standalone HTML dashboard; save to `path`."""
    techniques = _collect_techniques()
    creative = _collect_creative_ideas()
    snap = _collect_system_snapshot()
    quickstart = _collect_quickstart()

    cpu_pct = snap.get("cpu_usage", {}).get("overall_pct", 0) if isinstance(snap, dict) else 0
    ram_pct = snap.get("ram", {}).get("used_pct", 0) if isinstance(snap, dict) else 0
    uptime = snap.get("uptime_human", "unknown") if isinstance(snap, dict) else "unknown"

    html_content = (DASHBOARD_HTML
                     .replace("__N_TECHNIQUES__", str(len(techniques)))
                     .replace("__N_CREATIVE__", str(len(creative)))
                     .replace("__CPU_PCT__", str(int(cpu_pct)))
                     .replace("__RAM_PCT__", str(int(ram_pct)))
                     .replace("__UPTIME__", str(uptime))
                     .replace("__TECHNIQUE_ROWS__", _build_technique_rows())
                     .replace("__CREATIVE_ROWS__", _build_creative_rows())
                     .replace("__SYSTEM_SNAPSHOT__", _build_system_snapshot())
                     .replace("__QUICKSTART__", html.escape(quickstart)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return path


# ============================================================================
# Live server
# ============================================================================
class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the live dashboard."""

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            # generate fresh dashboard (system stats update)
            html_content = _render_live()
            self.wfile.write(html_content.encode("utf-8"))
        elif self.path == "/api/snapshot":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_collect_system_snapshot(), default=str).encode())
        elif self.path == "/api/techniques":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_collect_techniques()).encode())
        elif self.path == "/api/creative":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_collect_creative_ideas()).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress logging


def _render_live() -> str:
    """Render the dashboard HTML with current data."""
    techniques = _collect_techniques()
    creative = _collect_creative_ideas()
    snap = _collect_system_snapshot()
    quickstart = _collect_quickstart()

    cpu_pct = snap.get("cpu_usage", {}).get("overall_pct", 0) if isinstance(snap, dict) else 0
    ram_pct = snap.get("ram", {}).get("used_pct", 0) if isinstance(snap, dict) else 0
    uptime = snap.get("uptime_human", "unknown") if isinstance(snap, dict) else "unknown"

    return (DASHBOARD_HTML
             .replace("__N_TECHNIQUES__", str(len(techniques)))
             .replace("__N_CREATIVE__", str(len(creative)))
             .replace("__CPU_PCT__", str(int(cpu_pct)))
             .replace("__RAM_PCT__", str(int(ram_pct)))
             .replace("__UPTIME__", str(uptime))
             .replace("__TECHNIQUE_ROWS__", _build_technique_rows())
             .replace("__CREATIVE_ROWS__", _build_creative_rows())
             .replace("__SYSTEM_SNAPSHOT__", _build_system_snapshot())
             .replace("__QUICKSTART__", html.escape(quickstart)))


def serve_dashboard(port: int = 8080, host: str = "0.0.0.0",
                     block: bool = True) -> Optional[HTTPServer]:
    """Serve the dashboard at http://host:port/.

    If block=True (default), runs forever. If block=False, starts in a
    background thread and returns the server.
    """
    server = HTTPServer((host, port), _DashboardHandler)
    print(f"qai dashboard: http://{host}:{port}")
    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard...")
            server.shutdown()
    else:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server
    return None


def _find_free_port(start: int = 8080, max_tries: int = 10) -> int:
    """Find a free port starting from `start`."""
    for port in range(start, start + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    return start


__all__ = [
    "generate_static_html", "serve_dashboard", "_find_free_port",
]
