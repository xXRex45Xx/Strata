#!/usr/bin/env python3
"""One endpoint in front of several Strata servers (one per GPU).

    python tools/dual_proxy.py --backend 127.0.0.1:8080:2 --backend 127.0.0.1:8081:1 [--listen-port 8090]

Point the client (ANTHROPIC_BASE_URL / an OpenAI base URL) at the proxy.  A request that starts a new conversation goes
to the backend with the least work for its weight (active requests / weight; the first listed wins a tie).  A conversation
that was seen before always goes back to the backend that served it and waits there if it is busy, because that server
keeps the conversation's prompt in its cache: another server would read the whole prompt again, which takes longer than
the wait on a long conversation.  --non-sticky lets a request move to an idle backend instead.  A backend that refuses
a connection is skipped for 10 s.  Responses are passed through as they arrive, so streaming works.
GET /_proxy/status shows the load.  Only the standard library is used.
"""
import argparse, hashlib, http.client, json, sys, threading, time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# headers that belong to one hop; date/server are written again by this server
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding",
       "upgrade", "content-length", "host", "date", "server"}
DOWN_FOR = 10.0


class Backend:
    def __init__(self, spec: str):
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(f"--backend HOST:PORT[:WEIGHT], not {spec!r}")
        self.host, self.port = parts[0], int(parts[1])
        self.weight = float(parts[2]) if len(parts) > 2 else 1.0
        self.active = 0
        self.served = 0
        self.down_until = 0.0

    @property
    def name(self) -> str:
        return f"{self.host}:{self.port}"


class Router:
    def __init__(self, backends: list, sticky: bool = True, map_size: int = 4096):
        self.backends, self.sticky, self.map_size = backends, sticky, map_size
        self.lock = threading.Lock()
        self.conv = OrderedDict()            # conversation key -> Backend, least recently used first

    def pick(self, key, exclude=()) -> "Backend | None":
        """The backend for this request, counted as one more active request there (release() undoes it)."""
        now = time.time()
        with self.lock:
            cand = [b for b in self.backends if b not in exclude]
            up = [b for b in cand if b.down_until <= now] or cand      # all marked down: try them anyway
            if not up:
                return None
            best = min(up, key=lambda b: (b.active / b.weight, self.backends.index(b)))
            chosen = best
            mapped = self.conv.get(key) if key is not None else None
            if mapped is not None and mapped in up:
                # stay on the server that holds this conversation's prompt, unless it is busy and another is idle
                if self.sticky or mapped.active == 0 or best.active > 0:
                    chosen = mapped
            chosen.active += 1
            if key is not None:
                self.conv[key] = chosen
                self.conv.move_to_end(key)
                while len(self.conv) > self.map_size:
                    self.conv.popitem(last=False)
            return chosen

    def release(self, b: Backend, ok: bool = True) -> None:
        with self.lock:
            b.active -= 1
            if ok:
                b.served += 1

    def mark_down(self, b: Backend) -> None:
        with self.lock:
            b.down_until = time.time() + DOWN_FOR

    def status(self) -> dict:
        with self.lock:
            return {"sticky": self.sticky, "conversations": len(self.conv),
                    "backends": [{"backend": b.name, "weight": b.weight, "active": b.active, "served": b.served,
                                  "down": b.down_until > time.time()} for b in self.backends]}


def conv_key(path: str, body: bytes):
    """A stable id for the conversation a chat request belongs to: its system prompt and first user message."""
    if not body or not path.split("?")[0].endswith(("/chat/completions", "/messages")):
        return None
    try:
        d = json.loads(body)
    except ValueError:
        return None
    msgs = d.get("messages") if isinstance(d, dict) else None
    if not isinstance(msgs, list):
        return None
    msgs = [m for m in msgs if isinstance(m, dict)]
    first = next((m for m in msgs if m.get("role") != "system"), None)
    system = next((m for m in msgs if m.get("role") == "system"), None)
    material = json.dumps([d.get("system"), system, first], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def make_handler(router: Router, read_timeout: float, verbose: bool):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if verbose:
                sys.stderr.write("proxy: " + (fmt % args) + "\n")

        def read_body(self) -> bytes:
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                out = b""
                while True:
                    n = int(self.rfile.readline().strip().split(b";")[0] or b"0", 16)
                    if n == 0:
                        while self.rfile.readline().strip():       # trailers
                            pass
                        return out
                    out += self.rfile.read(n)
                    self.rfile.readline()
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def reply(self, code: int, data: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def forward(self) -> None:
            self.close_connection = True
            if self.path.split("?")[0] == "/_proxy/status":
                return self.reply(200, json.dumps(router.status(), indent=1).encode())
            body = self.read_body()
            key = conv_key(self.path, body)
            tried = []
            while True:
                b = router.pick(key, tried)
                if b is None:
                    return self.reply(502, json.dumps({"error": "no backend answered"}).encode())
                try:
                    conn = http.client.HTTPConnection(b.host, b.port, timeout=10)
                    conn.connect()
                    conn.sock.settimeout(read_timeout)
                    headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
                    if body or self.command in ("POST", "PUT", "PATCH"):
                        headers["Content-Length"] = str(len(body))
                    conn.request(self.command, self.path, body, headers)
                    resp = conn.getresponse()
                    break
                except (OSError, http.client.HTTPException):
                    router.release(b, ok=False)
                    router.mark_down(b)
                    tried.append(b)
            if verbose:
                sys.stderr.write(f"proxy: {self.command} {self.path} key={(key or '-')[:8]} -> {b.name} "
                                 f"(active {b.active})\n")
            try:
                self.send_response(resp.status, resp.reason)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                length = resp.getheader("Content-Length")
                if length:
                    self.send_header("Content-Length", length)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except OSError:
                pass            # the client left (or the server stalled): closing the upstream connection cancels it
            finally:
                conn.close()
                router.release(b)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = forward

    return Handler


def serve(router: Router, host: str, port: int, read_timeout: float = 1800.0, verbose: bool = False) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(router, read_timeout, verbose))
    srv.daemon_threads = True
    srv.request_queue_size = 128
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", action="append", required=True, metavar="HOST:PORT[:WEIGHT]")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=8090)
    ap.add_argument("--non-sticky", action="store_true",
                    help="a conversation whose backend is busy moves to an idle one (its prompt is read again there)")
    ap.add_argument("--map-size", type=int, default=4096, help="conversations remembered")
    ap.add_argument("--read-timeout", type=float, default=1800.0, help="seconds a backend may stay silent")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    router = Router([Backend(s) for s in a.backend], not a.non_sticky, a.map_size)
    srv = serve(router, a.host, a.listen_port, a.read_timeout, a.verbose)
    print(f"proxy: http://{a.host}:{a.listen_port} -> " + ", ".join(f"{b.name} (weight {b.weight:g})" for b in router.backends),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
