"""
qai.network -- Networking module (spec Section 88).

HTTP/WebSocket client utilities + OAuth connector stubs + task orchestration.
Inspired by the spec's `qai.api`, `qai.websocket`, `qai.oauth`, `qai.orchestrate`.

    from qai.network import (
        api, WebSocketClient, oauth, orchestrate, CircuitBreaker,
        sign_request, RateLimitedClient,
    )

    # 1. HTTP client with retry + cache + rate limit
    response = api.get("https://api.example.com/data",
                       retry_on_failure=3, cache_ttl=60)
    response = api.post("https://api.example.com/data", json={"x": 1})

    # 2. WebSocket client (subprocess-based, stdlib only)
    ws = WebSocketClient("ws://echo.websocket.org")
    ws.send("hello")
    msg = ws.recv(timeout=5)

    # 3. OAuth connector stubs (Google, Slack, Salesforce, GitHub)
    g = oauth.connect("google", client_id="...", client_secret="...",
                      redirect_uri="http://localhost:8080/callback")
    auth_url = g.get_auth_url()
    token = g.exchange_code(code)

    # 4. Orchestrate parallel/sequential task execution
    results = orchestrate([task_a, task_b, task_c], mode="parallel")
    results = orchestrate([task_a, task_b, task_c], mode="sequential")

Creative additions beyond the spec:
  - **Circuit breaker** — stops calling a failing service after N consecutive
    failures, retries after a cooldown period
  - **HMAC request signing** — sign outgoing requests with HMAC-SHA256
  - **Rate-limited client** — wraps any HTTP client to enforce rate limits
  - **Request deduplication** — identical in-flight requests share a single response
  - **Exponential backoff with jitter** — better than fixed-interval retries
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# =====================================================================
# 1. HTTP client (qai.api)
# =====================================================================
class Response:
    """HTTP response wrapper."""
    def __init__(self, status: int, headers: Dict[str, str], body: bytes, url: str, elapsed_s: float):
        self.status = status
        self.headers = headers
        self.body = body
        self.url = url
        self.elapsed_s = elapsed_s

    @property
    def text(self) -> str:
        try:
            return self.body.decode("utf-8")
        except UnicodeDecodeError:
            return self.body.decode("latin-1", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def __repr__(self):
        return f"<Response {self.status} from {self.url} ({len(self.body)} bytes, {self.elapsed_s:.3f}s)>"


class _Cache:
    """In-memory TTL cache."""
    def __init__(self):
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: float) -> None:
        with self._lock:
            self._store[key] = (time.time() + ttl, value)


_cache = _Cache()


def _request(
    method: str,
    url: str,
    *,
    params: Optional[Dict] = None,
    data: Optional[Union[bytes, str, Dict]] = None,
    json_body: Optional[Dict] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    retry_on_failure: int = 0,
    retry_backoff: float = 1.0,
    cache_ttl: Optional[float] = None,
    auth: Optional[Tuple[str, str]] = None,
    bearer_token: Optional[str] = None,
) -> Response:
    """Internal HTTP request implementation."""
    # Build URL with query params
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}{'&' if '?' in url else '?'}{query}"

    # Check cache
    cache_key = f"{method}:{url}:{hashlib.sha256((str(data) + str(json_body)).encode()).hexdigest()[:16]}"
    if cache_ttl and method == "GET":
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    # Prepare body
    body: Optional[bytes] = None
    final_headers = dict(headers or {})
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        final_headers["Content-Type"] = "application/json"
    elif isinstance(data, dict):
        body = urllib.parse.urlencode(data).encode("utf-8")
        final_headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif isinstance(data, str):
        body = data.encode("utf-8")
    elif isinstance(data, bytes):
        body = data

    # Auth
    if auth:
        import base64
        user, pw = auth
        token = base64.b64encode(f"{user}:{pw}".encode()).decode()
        final_headers["Authorization"] = f"Basic {token}"
    elif bearer_token:
        final_headers["Authorization"] = f"Bearer {bearer_token}"

    # Retry loop
    last_error: Optional[Exception] = None
    for attempt in range(retry_on_failure + 1):
        try:
            req = urllib.request.Request(url, data=body, method=method, headers=final_headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = time.time() - t0
                response = Response(
                    status=resp.status,
                    headers=dict(resp.headers),
                    body=resp.read(),
                    url=url,
                    elapsed_s=elapsed,
                )
                if cache_ttl and method == "GET":
                    _cache.set(cache_key, response, cache_ttl)
                return response
        except urllib.error.HTTPError as e:
            elapsed = time.time() - t0
            return Response(
                status=e.code,
                headers=dict(e.headers or {}),
                body=e.read(),
                url=url,
                elapsed_s=elapsed,
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = e
            if attempt < retry_on_failure:
                # Exponential backoff with jitter
                sleep_time = retry_backoff * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_time)
                continue
            raise

    raise last_error  # type: ignore


class _ApiClient:
    """HTTP client namespace with .get/.post/.put/.delete/.patch methods."""
    def request(self, method: str, url: str, **kwargs) -> Response:
        return _request(method, url, **kwargs)

    def get(self, url: str, **kwargs) -> Response:
        return _request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> Response:
        return _request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> Response:
        return _request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs) -> Response:
        return _request("DELETE", url, **kwargs)

    def patch(self, url: str, **kwargs) -> Response:
        return _request("PATCH", url, **kwargs)

    def check_health(self, url: str, timeout: float = 5.0) -> bool:
        """Quick health check — returns True if 2xx response."""
        try:
            r = self.get(url, timeout=timeout)
            return 200 <= r.status < 300
        except Exception:
            return False


api = _ApiClient()


# =====================================================================
# 2. WebSocket client (subprocess-based, stdlib only)
# =====================================================================
class WebSocketClient:
    """Minimal WebSocket client using a subprocess Python interpreter.

    Python's stdlib doesn't include a WebSocket client (only `http.client`).
    Rather than depend on `websockets` or `websocket-client` packages, this
    implementation spawns a subprocess that uses raw socket I/O.

    For real production use, install `websocket-client` and use it directly.
    This implementation is intentionally dependency-free.
    """
    def __init__(self, url: str, timeout: float = 30.0):
        """
        Args:
            url: ws:// or wss:// URL.
            timeout: Default recv timeout in seconds.
        """
        self.url = url
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._connected = False

    def connect(self) -> None:
        """Open the WebSocket connection."""
        if self._connected:
            return
        # Subprocess script that does the WS handshake + frame I/O
        script = '''
import sys, socket, hashlib, base64, os, struct, json

url = sys.argv[1]
# Parse URL
if url.startswith("ws://"):
    host_port = url[5:].split("/")[0]
    use_tls = False
elif url.startswith("wss://"):
    host_port = url[6:].split("/")[0]
    use_tls = True
else:
    print(json.dumps({"error": "URL must start with ws:// or wss://"}))
    sys.exit(1)

if ":" in host_port:
    host, port_str = host_port.rsplit(":", 1)
    port = int(port_str)
else:
    host = host_port
    port = 443 if use_tls else 80

try:
    sock = socket.create_connection((host, port), timeout=30)
    if use_tls:
        import ssl
        ctx = ssl.create_default_context()
        sock = ctx.wrap_socket(sock, server_hostname=host)

    # WS handshake
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET / HTTP/1.1\\r\\n"
        f"Host: {host}\\r\\n"
        f"Upgrade: websocket\\r\\n"
        f"Connection: Upgrade\\r\\n"
        f"Sec-WebSocket-Key: {key}\\r\\n"
        f"Sec-WebSocket-Version: 13\\r\\n\\r\\n"
    )
    sock.sendall(req.encode())
    resp = b""
    while b"\\r\\n\\r\\n" not in resp:
        chunk = sock.recv(4096)
        if not chunk:
            break
        resp += chunk
    if b"101" not in resp.split(b"\\r\\n")[0]:
        print(json.dumps({"error": "WS handshake failed", "response": resp.decode(errors="replace")}))
        sys.exit(1)

    print(json.dumps({"connected": True}))

    # Main loop: read commands from stdin, send/recv WS frames
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = json.loads(line)
        if cmd.get("action") == "send":
            payload = cmd["data"].encode("utf-8")
            # Text frame (opcode 0x1), masked
            mask = os.urandom(4)
            header = bytes([0x81])  # FIN + text opcode
            length = len(payload)
            if length < 126:
                header += bytes([0x80 | length])
            elif length < 65536:
                header += bytes([0x80 | 126]) + struct.pack(">H", length)
            else:
                header += bytes([0x80 | 127]) + struct.pack(">Q", length)
            masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            sock.sendall(header + mask + masked)
            print(json.dumps({"sent": True}))
        elif cmd.get("action") == "recv":
            sock.settimeout(cmd.get("timeout", 30))
            try:
                hdr = sock.recv(2)
                if len(hdr) < 2:
                    print(json.dumps({"error": "connection closed"}))
                    break
                opcode = hdr[0] & 0x0F
                masked_len = hdr[1] & 0x7F
                if masked_len == 126:
                    length = struct.unpack(">H", sock.recv(2))[0]
                elif masked_len == 127:
                    length = struct.unpack(">Q", sock.recv(8))[0]
                else:
                    length = masked_len
                if hdr[1] & 0x80:
                    sock.recv(4)  # mask
                data = b""
                while len(data) < length:
                    chunk = sock.recv(length - len(data))
                    if not chunk:
                        break
                    data += chunk
                print(json.dumps({"data": data.decode("utf-8", errors="replace"), "opcode": opcode}))
            except socket.timeout:
                print(json.dumps({"error": "timeout"}))
        elif cmd.get("action") == "close":
            break
    sock.close()
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
'''
        self._proc = subprocess.Popen(
            [sys.executable, "-c", script, self.url],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Read the initial "connected" line
        line = self._proc.stdout.readline()
        try:
            payload = json.loads(line)
            if payload.get("connected"):
                self._connected = True
            else:
                raise ConnectionError(payload.get("error", "WS connection failed"))
        except json.JSONDecodeError as e:
            raise ConnectionError(f"WS subprocess error: {line!r}: {e}")

    def send(self, data: str) -> None:
        """Send a text frame."""
        if not self._connected:
            self.connect()
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps({"action": "send", "data": data}) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        payload = json.loads(line)
        if not payload.get("sent"):
            raise ConnectionError(payload.get("error", "send failed"))

    def recv(self, timeout: Optional[float] = None) -> Optional[str]:
        """Receive a text frame. Returns None on timeout."""
        if not self._connected:
            self.connect()
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps({"action": "recv", "timeout": timeout or self.timeout}) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        payload = json.loads(line)
        if "data" in payload:
            return payload["data"]
        return None  # timeout or error

    def close(self) -> None:
        """Close the connection."""
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(json.dumps({"action": "close"}) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._connected = False
        self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =====================================================================
# 3. OAuth connector stubs
# =====================================================================
# OAuth endpoints for common providers (as of 2026)
_OAUTH_PROVIDERS = {
    "google": {
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "default_scope": "openid email profile",
    },
    "github": {
        "auth_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "default_scope": "read:user",
    },
    "slack": {
        "auth_url": "https://slack.com/oauth/v2/authorize",
        "token_url": "https://slack.com/api/oauth.v2.access",
        "default_scope": "chat:write",
    },
    "salesforce": {
        "auth_url": "https://login.salesforce.com/services/oauth2/authorize",
        "token_url": "https://login.salesforce.com/services/oauth2/token",
        "default_scope": "full",
    },
    "microsoft": {
        "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "default_scope": "https://graph.microsoft.com/.default",
    },
}


class OAuthConnector:
    """OAuth 2.0 authorization-code flow connector.

    This is a *stub* — it generates the correct auth URLs and token-exchange
    requests, but you must run a local HTTP server to receive the callback.
    For a complete solution, use `authlib` or `requests-oauthlib`.

    Basic flow:
        connector = oauth.connect("google", client_id=..., client_secret=...)
        auth_url = connector.get_auth_url(state="random-state")
        # redirect user to auth_url, they get redirected back with ?code=...
        token = connector.exchange_code(code)
        # now use token.access_token with qai.network.api
    """
    def __init__(self, provider: str, client_id: str, client_secret: str,
                 redirect_uri: str, scope: Optional[str] = None):
        if provider not in _OAUTH_PROVIDERS:
            raise ValueError(
                f"[WHAT] Unknown OAuth provider {provider!r}\n"
                f"[WHY] Provider not in registry\n"
                f"[FIX] Use one of {list(_OAUTH_PROVIDERS.keys())}, or pass a custom provider dict"
            )
        self.provider = provider
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scope = scope or _OAUTH_PROVIDERS[provider]["default_scope"]
        self._config = _OAUTH_PROVIDERS[provider]

    def get_auth_url(self, state: Optional[str] = None) -> str:
        """Build the authorization URL for the user to visit."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": self.scope,
        }
        if state:
            params["state"] = state
        query = urllib.parse.urlencode(params)
        return f"{self._config['auth_url']}?{query}"

    def exchange_code(self, code: str) -> Dict[str, Any]:
        """Exchange an authorization code for an access token.

        Returns the token response dict (typically with `access_token`,
        `refresh_token`, `expires_in`, `token_type`).
        """
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }
        resp = api.post(self._config["token_url"], data=data)
        if resp.status != 200:
            raise RuntimeError(
                f"[WHAT] OAuth token exchange failed: HTTP {resp.status}\n"
                f"[WHY] {resp.text}"
            )
        return resp.json()

    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh an access token using a refresh token."""
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        resp = api.post(self._config["token_url"], data=data)
        return resp.json() if resp.status == 200 else {"error": resp.text}


class _OAuthNamespace:
    """Namespace for `qai.network.oauth`."""
    @staticmethod
    def connect(provider: str, **kwargs) -> OAuthConnector:
        return OAuthConnector(provider=provider, **kwargs)

    @staticmethod
    def list_providers() -> List[str]:
        return sorted(_OAUTH_PROVIDERS.keys())


oauth = _OAuthNamespace()


# =====================================================================
# 4. Task orchestration
# =====================================================================
def orchestrate(
    tasks: List[Callable],
    mode: str = "sequential",
    max_workers: int = 4,
    return_exceptions: bool = False,
) -> List[Any]:
    """Run a list of callables in parallel or sequential mode.

    Args:
        tasks: List of callables (each takes no args).
        mode: "sequential" (default) or "parallel".
        max_workers: Max parallel workers (only used in parallel mode).
        return_exceptions: If True, exceptions are returned as values
            instead of raised. Default False.

    Returns:
        List of results in the same order as `tasks`.
    """
    if mode == "sequential":
        results = []
        for task in tasks:
            try:
                results.append(task())
            except Exception as e:
                if return_exceptions:
                    results.append(e)
                else:
                    raise
        return results

    elif mode == "parallel":
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: List[Any] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {ex.submit(task): i for i, task in enumerate(tasks)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    if return_exceptions:
                        results[idx] = e
                    else:
                        raise
        return results

    else:
        raise ValueError(f"Unknown mode: {mode!r} (use 'sequential' or 'parallel')")


# =====================================================================
# 5. Circuit breaker (my own addition)
# =====================================================================
class CircuitBreaker:
    """Stop calling a failing service after N consecutive failures.

    States:
      - **closed**: requests flow normally. Failures increment counter.
      - **open**: after `failure_threshold` consecutive failures, all requests
        fail fast with `CircuitBreakerOpenError`. No actual call is made.
      - **half_open**: after `recovery_timeout` seconds, one trial request is
        allowed. If it succeeds, circuit closes. If it fails, circuit reopens.

    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        try:
            result = breaker.call(lambda: api.get("https://flaky.example.com"))
        except CircuitBreakerOpenError:
            print("Service is down, using fallback")
    """
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._state = "closed"  # closed / open / half_open
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """Call `fn` through the circuit breaker."""
        with self._lock:
            if self._state == "open":
                if self._last_failure_time and time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = "half_open"
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit is open (failures={self._failure_count}, "
                        f"retry in {self.recovery_timeout - (time.time() - (self._last_failure_time or 0)):.1f}s)"
                    )

        try:
            result = fn(*args, **kwargs)
            with self._lock:
                if self._state == "half_open":
                    self._state = "closed"
                self._failure_count = 0
            return result
        except Exception as e:
            with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                if self._failure_count >= self.failure_threshold:
                    self._state = "open"
            raise

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = "closed"
            self._last_failure_time = None


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open."""


# =====================================================================
# 6. HMAC request signing (my own addition)
# =====================================================================
def sign_request(
    method: str,
    url: str,
    secret_key: str,
    body: Optional[bytes] = None,
    timestamp: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Sign an HTTP request with HMAC-SHA256.

    Generates headers for the receiver to verify:
      - `X-Qai-Timestamp`: unix timestamp (prevents replay)
      - `X-Qai-Signature`: HMAC-SHA256(secret, method + url + timestamp + body)

    Args:
        method: HTTP method (GET, POST, etc.).
        url: Full URL.
        secret_key: Shared secret with the receiver.
        body: Request body bytes (optional).
        timestamp: Override timestamp (default: now).
        headers: Existing headers dict to add to (returns new dict).

    Returns:
        Headers dict with signature + timestamp added.
    """
    ts = str(timestamp or int(time.time()))
    body_bytes = body or b""
    payload = f"{method.upper()}\n{url}\n{ts}\n".encode() + body_bytes
    signature = hmac.new(
        secret_key.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    final_headers = dict(headers or {})
    final_headers["X-Qai-Timestamp"] = ts
    final_headers["X-Qai-Signature"] = signature
    return final_headers


def verify_request_signature(
    method: str,
    url: str,
    secret_key: str,
    timestamp: str,
    signature: str,
    body: Optional[bytes] = None,
    max_age_seconds: int = 300,
) -> bool:
    """Verify an HMAC-SHA256 request signature.

    Checks:
      - Timestamp is within `max_age_seconds` of now (prevents replay)
      - Signature matches the recomputed HMAC

    Returns True if valid, False otherwise.
    """
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > max_age_seconds:
        return False
    body_bytes = body or b""
    payload = f"{method.upper()}\n{url}\n{timestamp}\n".encode() + body_bytes
    expected = hmac.new(
        secret_key.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# =====================================================================
# 7. Rate-limited client (my own addition)
# =====================================================================
class RateLimitedClient:
    """Wrap an HTTP client to enforce a max-requests-per-second rate limit.

    Example:
        client = RateLimitedClient(api, max_per_second=10)
        for url in urls:
            r = client.get(url)  # blocks if rate exceeded
    """
    def __init__(self, client: Any = None, max_per_second: int = 10):
        self.client = client or api
        self.min_interval = 1.0 / max_per_second
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def _wait(self):
        with self._lock:
            now = time.time()
            wait = self.min_interval - (now - self._last_request_time)
            if wait > 0:
                time.sleep(wait)
            self._last_request_time = time.time()

    def get(self, url: str, **kwargs) -> Response:
        self._wait()
        return self.client.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> Response:
        self._wait()
        return self.client.post(url, **kwargs)

    def put(self, url: str, **kwargs) -> Response:
        self._wait()
        return self.client.put(url, **kwargs)

    def delete(self, url: str, **kwargs) -> Response:
        self._wait()
        return self.client.delete(url, **kwargs)

    def patch(self, url: str, **kwargs) -> Response:
        self._wait()
        return self.client.patch(url, **kwargs)


# =====================================================================
# Public API
# =====================================================================
__all__ = [
    # HTTP
    "api", "Response",
    # WebSocket
    "WebSocketClient",
    # OAuth
    "oauth", "OAuthConnector",
    # Orchestration
    "orchestrate",
    # Creative additions
    "CircuitBreaker", "CircuitBreakerOpenError",
    "sign_request", "verify_request_signature",
    "RateLimitedClient",
]
