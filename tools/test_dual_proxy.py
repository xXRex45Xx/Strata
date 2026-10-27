"""tools/dual_proxy.py against fake backends: routing, conversation affinity, streaming, failover."""
import http.client
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tools import dual_proxy


def fake_backend(hold: float = 0.6):
    """A server that answers a chat request in three chunks over `hold` seconds and says which port it is."""
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for i in range(3):
                self.wfile.write(f"data: {self.server.server_port} {i} {len(body)}\n\n".encode())
                self.wfile.flush()
                time.sleep(hold / 3)

        def do_GET(self):
            data = json.dumps({"port": self.server.server_port}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def chat(proxy_port: int, text: str, results: list = None, idx: int = 0, chunked: bool = False):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": text}]}).encode()
    c = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=30)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body if not chunked else iter([body[:7], body[7:]]),
              {"Content-Type": "application/json"}, encode_chunked=chunked)
    r = c.getresponse()
    first = r.read1(100)
    t_first = time.time() - t0
    rest = r.read()
    out = (first + rest).decode()
    port = int(out.split()[1])
    res = {"port": port, "first": t_first, "total": time.time() - t0, "body_len": int(out.split()[3])}
    if results is not None:
        results[idx] = res
    return res


class ProxyTest(unittest.TestCase):
    def setUp(self):
        self.a, self.b = fake_backend(), fake_backend()
        self.pa, self.pb = self.a.server_port, self.b.server_port
        self.router = dual_proxy.Router([dual_proxy.Backend(f"127.0.0.1:{self.pa}:2"),
                                         dual_proxy.Backend(f"127.0.0.1:{self.pb}:1")])
        self.proxy = dual_proxy.serve(self.router, "127.0.0.1", 0)
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()
        self.port = self.proxy.server_port

    def tearDown(self):
        for s in (self.proxy, self.a, self.b):
            s.shutdown()
            s.server_close()

    def parallel(self, texts):
        res = [None] * len(texts)
        ts = [threading.Thread(target=chat, args=(self.port, t, res, i)) for i, t in enumerate(texts)]
        for t in ts:
            t.start()
            time.sleep(0.1)                           # the first request is placed before the second arrives
        [t.join() for t in ts]
        return res

    def test_two_conversations_at_once_use_both_servers(self):
        r = self.parallel(["first conversation", "second conversation"])
        self.assertEqual({x["port"] for x in r}, {self.pa, self.pb})

    def test_idle_tie_goes_to_the_first_listed(self):
        self.assertEqual(chat(self.port, "alone")["port"], self.pa)

    def test_conversation_goes_back_to_its_server(self):
        self.parallel(["one", "two"])                 # one -> a, two -> b
        self.assertEqual(chat(self.port, "two")["port"], self.pb)
        self.assertEqual(chat(self.port, "one")["port"], self.pa)

    def test_non_sticky_hands_a_busy_conversation_to_an_idle_server(self):
        self.router.sticky = False
        chat(self.port, "mine")                       # mine -> a
        res = [None, None]
        t = threading.Thread(target=chat, args=(self.port, "other", res, 0))   # other -> a (tie), a is now busy
        t.start()
        time.sleep(0.2)
        res[1] = chat(self.port, "mine")              # a busy, b idle -> moves
        t.join()
        self.assertEqual(res[0]["port"], self.pa)
        self.assertEqual(res[1]["port"], self.pb)

    def test_default_waits_for_its_own_server(self):
        self.assertTrue(self.router.sticky)
        chat(self.port, "mine")
        t = threading.Thread(target=chat, args=(self.port, "other"))
        t.start()
        time.sleep(0.2)
        self.assertEqual(chat(self.port, "mine")["port"], self.pa)
        t.join()

    def test_response_streams_before_it_finishes(self):
        r = chat(self.port, "stream")
        self.assertLess(r["first"], r["total"] - 0.2)

    def test_chunked_request_body_arrives_whole(self):
        r = chat(self.port, "chunked", chunked=True)
        self.assertEqual(r["body_len"], len(json.dumps({"model": "x", "messages": [{"role": "user", "content": "chunked"}]})))

    def test_dead_backend_is_skipped(self):
        self.a.shutdown()
        self.a.server_close()
        for _ in range(3):
            self.assertEqual(chat(self.port, "x")["port"], self.pb)
        self.assertTrue(self.router.status()["backends"][0]["down"])

    def test_all_backends_down_is_a_502(self):
        for s in (self.a, self.b):
            s.shutdown()
            s.server_close()
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/v1/chat/completions", b"{}", {"Content-Type": "application/json"})
        self.assertEqual(c.getresponse().status, 502)

    def test_other_paths_are_forwarded_and_status_is_answered(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/v1/models")
        self.assertEqual(json.loads(c.getresponse().read())["port"], self.pa)
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/_proxy/status")
        st = json.loads(c.getresponse().read())
        self.assertEqual([b["active"] for b in st["backends"]], [0, 0])

    def test_conversation_key(self):
        k = lambda msgs, system=None: dual_proxy.conv_key(
            "/v1/messages", json.dumps({"system": system, "messages": msgs}).encode())
        u1 = {"role": "user", "content": "hello"}
        a1 = {"role": "assistant", "content": "hi"}
        u2 = {"role": "user", "content": "more"}
        self.assertEqual(k([u1]), k([u1, a1, u2]))             # later turns keep the key
        self.assertNotEqual(k([u1]), k([u2]))
        self.assertNotEqual(k([u1], "sys a"), k([u1], "sys b"))
        self.assertIsNone(dual_proxy.conv_key("/v1/models", b"{}"))
        self.assertIsNone(dual_proxy.conv_key("/v1/messages", b"not json"))


if __name__ == "__main__":
    unittest.main()
