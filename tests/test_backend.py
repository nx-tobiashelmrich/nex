"""OllamaBackend against a stub Ollama server on 127.0.0.1, and FakeBackend."""

import io
import json
import os
import socket
import threading
import time
import unittest
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from nex import backend as backend_module
from nex.backend import (
    DEFAULT_BACKEND_MODEL,
    DEFAULT_NUM_CTX,
    TOP_LOGPROBS,
    BackendError,
    ContextOverflowError,
    FakeBackend,
    ModelNotFoundError,
    NextToken,
    OllamaBackend,
    _error_detail,
    _is_loopback,
    _normalize_host,
)

MESSAGES = [
    {"role": "system", "content": "You are a decision model."},
    {"role": "user", "content": "STATE:\nI was charged twice.\n\nQUESTION:\nBilling?\n\nReply with only Yes or No."},
]
THINK_REFUSAL = '"m" does not support thinking'
ENV_VARS = ("OLLAMA_HOST", "NEX_BACKEND_MODEL", "NEX_NUM_CTX")
OVERFLOW_MESSAGE = "request (9768 tokens) exceeds the available context size (8192 tokens), try increasing it"
OVERFLOW_INNER = {
    "code": 400,
    "message": OVERFLOW_MESSAGE,
    "type": "exceed_context_size_error",
    "n_prompt_tokens": 9768,
    "n_ctx": 8192,
}
# Byte for byte what Ollama 0.35 sends. The error value is itself JSON text.
COMPACT = (",", ":")
OVERFLOW_BODY = json.dumps({"error": json.dumps({"error": OVERFLOW_INNER}, separators=COMPACT)}, separators=COMPACT)
OVERFLOW_ERROR = "prompt is 9768 tokens but the context window is 8192. Shorten the state or raise NEX_NUM_CTX."


def chat_response(top=None, prompt_eval_count=92, cached=48):
    """Shaped like a real Ollama 0.35 /api/chat reply to a num_predict 1 request."""
    if top is None:
        top = [("Yes", 0), ("yes", -20.883312225341797), (" Yes", -21.96424102783203), ("No", -22.5)]
    entries = [{"token": t, "logprob": lp, "bytes": list(t.encode())} for t, lp in top]
    return {
        "model": "qwen3:4b-instruct",
        "created_at": "2026-10-02T08:37:14.95813Z",
        "message": {"role": "assistant", "content": top[0][0]},
        "done": True,
        "done_reason": "length",
        "logprobs": [dict(entries[0], top_logprobs=entries)],
        "total_duration": 1153698416,
        "load_duration": 1070110333,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_cached_count": cached,
        "prompt_eval_duration": 71662000,
        "eval_count": 1,
        "eval_duration": 1000,
    }


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True
    # Do not wait for handlers that are still blocked when a test ends.
    block_on_close = False

    def handle_error(self, request, client_address):
        pass


class StubOllama:
    """A scripted Ollama on 127.0.0.1. ``handler(method, path, body)``
    returns ``(status, payload)`` and every request is recorded."""

    def __init__(self, handler):
        self.handler = handler
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.serve("GET")

            def do_POST(self):
                self.serve("POST")

            def serve(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                stub.requests.append(
                    {
                        "method": method,
                        "path": self.path,
                        "body": json.loads(raw) if raw else None,
                        "content_type": self.headers.get("Content-Type"),
                    }
                )
                status, payload = stub.handler(method, self.path, stub.requests[-1]["body"])
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = QuietServer(("127.0.0.1", 0), Handler)
        # A short poll interval keeps shutdown() from waiting 0.5 s per test.
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True)
        self.thread.start()
        self.host = f"http://127.0.0.1:{self.server.server_port}"

    @property
    def bodies(self):
        return [r["body"] for r in self.requests]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class RawServer:
    """Reads one request per connection on 127.0.0.1, writes ``reply`` as raw
    bytes and hangs up. Stands in for servers that do not speak HTTP."""

    def __init__(self, reply):
        self.reply = reply
        self.connections = 0
        self.stopped = threading.Event()
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.sock.settimeout(0.05)
        self.host = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while not self.stopped.is_set():
            try:
                conn, _ = self.sock.accept()
            except (TimeoutError, OSError):
                continue
            with conn, conn.makefile("rb") as f:
                conn.settimeout(5)
                length = 0
                for line in iter(f.readline, b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":")[1])
                f.read(length)
                self.connections += 1
                conn.sendall(self.reply)

    def close(self):
        self.stopped.set()
        self.sock.close()


def ok(method, path, body):
    return 200, chat_response()


class BackendTestCase(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ENV_VARS:
            os.environ.pop(name, None)
        # Only the backend's own reference to time is replaced, so retries do
        # not sleep and stub handlers can still sleep for real.
        clock = mock.patch.object(backend_module, "time")
        self.time = clock.start()
        self.addCleanup(clock.stop)

    def serve(self, handler):
        stub = StubOllama(handler)
        self.addCleanup(stub.close)
        return stub

    def backend(self, stub, **kwargs):
        kwargs.setdefault("model", "qwen3:4b-instruct")
        return OllamaBackend(host=stub.host, **kwargs)

    def sleeps(self):
        return [c.args[0] for c in self.time.sleep.call_args_list]


class RequestTest(BackendTestCase):
    def test_request_body(self):
        stub = self.serve(ok)
        self.backend(stub).next_token(MESSAGES)
        self.assertEqual(len(stub.requests), 1)
        req = stub.requests[0]
        self.assertEqual((req["method"], req["path"]), ("POST", "/api/chat"))
        self.assertEqual(req["content_type"], "application/json")
        body = req["body"]
        self.assertEqual(
            body,
            {
                "model": "qwen3:4b-instruct",
                "messages": MESSAGES,
                "stream": False,
                "logprobs": True,
                "top_logprobs": 20,
                "think": False,
                "keep_alive": "30m",
                "truncate": False,
                "options": {"num_predict": 1, "temperature": 0, "num_ctx": 8192},
            },
        )
        # JSON booleans, not 0 and 1.
        self.assertIs(body["stream"], False)
        self.assertIs(body["logprobs"], True)
        self.assertIs(body["think"], False)
        self.assertIs(body["truncate"], False)
        self.assertEqual(TOP_LOGPROBS, 20)

    def test_num_ctx_and_keep_alive_are_sent(self):
        stub = self.serve(ok)
        self.backend(stub, num_ctx=4096, keep_alive="5m").next_token(MESSAGES)
        self.assertEqual(stub.bodies[0]["options"]["num_ctx"], 4096)
        self.assertEqual(stub.bodies[0]["keep_alive"], "5m")

    def test_parses_realistic_response(self):
        stub = self.serve(ok)
        nt = self.backend(stub).next_token(MESSAGES)
        self.assertIsInstance(nt, NextToken)
        self.assertEqual([t for t, _ in nt.top_logprobs], ["Yes", "yes", " Yes", "No"])
        self.assertEqual(nt.top_logprobs[0], ("Yes", 0.0))
        self.assertIsInstance(nt.top_logprobs[0][1], float)
        self.assertAlmostEqual(nt.top_logprobs[1][1], -20.883312225341797)
        self.assertEqual(nt.prompt_tokens, 92)
        self.assertEqual(nt.cached_tokens, 48)

    def test_missing_token_counts_default_to_zero(self):
        def handler(method, path, body):
            data = chat_response()
            del data["prompt_eval_count"], data["prompt_eval_cached_count"]
            return 200, data

        nt = self.backend(self.serve(handler)).next_token(MESSAGES)
        self.assertEqual((nt.prompt_tokens, nt.cached_tokens), (0, 0))

    def test_missing_logprobs(self):
        variants = {
            "no key": lambda d: d.pop("logprobs"),
            "null": lambda d: d.update(logprobs=None),
            "empty list": lambda d: d.update(logprobs=[]),
            "empty top": lambda d: d["logprobs"][0].update(top_logprobs=[]),
            "no top key": lambda d: d["logprobs"][0].pop("top_logprobs"),
        }
        for name, mutate in variants.items():
            with self.subTest(name):
                def handler(method, path, body, mutate=mutate):
                    data = chat_response()
                    mutate(data)
                    return 200, data

                b = self.backend(self.serve(handler))
                with self.assertRaises(BackendError) as ctx:
                    b.next_token(MESSAGES)
                self.assertIn("no logprobs", str(ctx.exception))
                self.assertNotIsInstance(ctx.exception, ContextOverflowError)

    def test_reply_that_is_not_a_json_object(self):
        cases = {b"<html>hello</html>": "not JSON", b"": "not JSON", b"[1, 2]": "not a JSON object"}
        for raw, expected in cases.items():
            for call in ("next_token", "list_models"):
                with self.subTest(raw=raw, call=call):
                    stub = self.serve(lambda m, p, b, raw=raw: (200, raw))
                    b = self.backend(stub)
                    with self.assertRaises(BackendError) as ctx:
                        getattr(b, call)(*([MESSAGES] if call == "next_token" else []))
                    self.assertEqual(str(ctx.exception), f"Ollama at {stub.host} sent a reply that is {expected}")
                    self.assertIsNone(ctx.exception.__cause__)
                    self.assertEqual(len(stub.requests), 1)
        self.time.sleep.assert_not_called()


class ThinkTest(BackendTestCase):
    def refusing(self, method, path, body):
        if "think" in body:
            return 400, {"error": THINK_REFUSAL}
        return 200, chat_response()

    def test_refusal_retries_without_think(self):
        stub = self.serve(self.refusing)
        b = self.backend(stub)
        nt = b.next_token(MESSAGES)
        self.assertEqual(nt.top_logprobs[0], ("Yes", 0.0))
        self.assertEqual(len(stub.requests), 2)
        self.assertIs(stub.bodies[0]["think"], False)
        self.assertNotIn("think", stub.bodies[1])
        first = dict(stub.bodies[0])
        del first["think"]
        self.assertEqual(stub.bodies[1], first)
        self.time.sleep.assert_not_called()

    def test_later_calls_stop_sending_think(self):
        stub = self.serve(self.refusing)
        b = self.backend(stub)
        b.next_token(MESSAGES)
        b.next_token(MESSAGES)
        b.next_token(MESSAGES)
        self.assertEqual(len(stub.requests), 4)
        for body in stub.bodies[1:]:
            self.assertNotIn("think", body)

    def test_thinking_model_keeps_sending_think(self):
        stub = self.serve(ok)
        b = self.backend(stub)
        b.next_token(MESSAGES)
        b.next_token(MESSAGES)
        self.assertEqual([body.get("think") for body in stub.bodies], [False, False])

    def test_other_model_starts_with_think_again(self):
        stub = self.serve(self.refusing)
        b = self.backend(stub)
        b.next_token(MESSAGES)
        other = b.with_model("qwen3.5:9b")
        other.next_token(MESSAGES)
        self.assertEqual(stub.bodies[2]["model"], "qwen3.5:9b")
        self.assertIn("think", stub.bodies[2])

    def test_concurrent_refusals_all_retry(self):
        # Two first requests both carry think. The refusal for one arrives
        # after the other thread has already cleared the flag, and that
        # request still has to be retried rather than fail.
        arrived = threading.Barrier(2, timeout=5)

        def handler(method, path, body):
            if "think" in body:
                if arrived.wait() == 0:
                    time.sleep(0.3)
                return 400, {"error": THINK_REFUSAL}
            return 200, chat_response()

        stub = self.serve(handler)
        b = self.backend(stub)
        results, errors = [], []

        def call():
            try:
                results.append(b.next_token(MESSAGES))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(len(stub.requests), 4)
        self.assertEqual(sum("think" in body for body in stub.bodies), 2)

    def test_only_a_400_refusal_triggers_the_retry(self):
        cases = {
            "server error": (500, {"error": THINK_REFUSAL}),
            "other think error": (400, {"error": "invalid value for think"}),
            "other client error": (422, {"error": THINK_REFUSAL}),
        }
        for name, reply in cases.items():
            with self.subTest(name):
                stub = self.serve(lambda m, p, b, reply=reply: reply)
                b = self.backend(stub, retries=0)
                with self.assertRaises(BackendError) as ctx:
                    b.next_token(MESSAGES)
                self.assertEqual(ctx.exception.status, reply[0])
                self.assertEqual(len(stub.requests), 1)
                self.assertTrue(b._send_think)

    def test_refusal_of_a_request_without_think_is_not_retried(self):
        stub = self.serve(lambda m, p, b: (400, {"error": THINK_REFUSAL}))
        b = self.backend(stub)
        b._send_think = False
        with self.assertRaises(BackendError) as ctx:
            b.next_token(MESSAGES)
        self.assertEqual(str(ctx.exception), f"Ollama 400: {THINK_REFUSAL}")
        self.assertEqual(len(stub.requests), 1)

    def test_flag_is_kept_until_a_retry_succeeds(self):
        retry_reply = [(503, {"error": "busy"})]

        def handler(method, path, body):
            if "think" in body:
                return 400, {"error": THINK_REFUSAL}
            return retry_reply[0]

        stub = self.serve(handler)
        b = self.backend(stub, retries=0)
        with self.assertRaises(BackendError) as ctx:
            b.next_token(MESSAGES)
        self.assertEqual(ctx.exception.status, 503)
        self.assertTrue(b._send_think)

        retry_reply[0] = (200, chat_response())
        b.next_token(MESSAGES)
        self.assertFalse(b._send_think)
        b.next_token(MESSAGES)
        self.assertEqual(["think" in body for body in stub.bodies], [True, False, True, False, False])


class ErrorTest(BackendTestCase):
    def test_missing_model(self):
        # The first wording is from the Ollama docs, the second from a live 0.35 server.
        for message in ['model "nope" not found, try pulling it first', "model 'nope' not found"]:
            with self.subTest(message):
                stub = self.serve(lambda m, p, b, message=message: (404, {"error": message}))
                b = self.backend(stub, model="nope")
                with self.assertRaises(ModelNotFoundError) as ctx:
                    b.next_token(MESSAGES)
                self.assertIsInstance(ctx.exception, BackendError)
                self.assertEqual(str(ctx.exception), 'Ollama has no model "nope". Pull it with: ollama pull nope')
                self.assertEqual(ctx.exception.status, 404)
                self.assertEqual(len(stub.requests), 1)
        self.time.sleep.assert_not_called()

    def test_other_404_is_not_a_missing_model(self):
        stub = self.serve(lambda m, p, b: (404, b"404 page not found"))
        with self.assertRaises(BackendError) as ctx:
            self.backend(stub).next_token(MESSAGES)
        self.assertNotIsInstance(ctx.exception, ModelNotFoundError)
        self.assertEqual(str(ctx.exception), "Ollama 404: Not Found")
        self.assertEqual(len(stub.requests), 1)

    def test_other_400_is_not_a_think_retry(self):
        stub = self.serve(lambda m, p, b: (400, {"error": "invalid message format"}))
        b = self.backend(stub)
        with self.assertRaises(BackendError) as ctx:
            b.next_token(MESSAGES)
        self.assertEqual(str(ctx.exception), "Ollama 400: invalid message format")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(len(stub.requests), 1)
        self.time.sleep.assert_not_called()
        self.assertTrue(b._send_think)

    def test_server_errors_are_retried_then_raise(self):
        stub = self.serve(lambda m, p, b: (500, {"error": "llama runner process has terminated"}))
        b = self.backend(stub, retries=2)
        with self.assertRaises(BackendError) as ctx:
            b.next_token(MESSAGES)
        self.assertEqual(str(ctx.exception), "Ollama 500: llama runner process has terminated")
        self.assertEqual(ctx.exception.status, 500)
        self.assertEqual(len(stub.requests), 3)
        self.assertEqual(self.sleeps(), [0.5, 1.0])

    def test_nested_error_uses_the_inner_message(self):
        inner = json.dumps({"error": {"code": 500, "message": "failed to decode batch", "type": "server_error"}})
        stub = self.serve(lambda m, p, b: (500, {"error": inner}))
        with self.assertRaises(BackendError) as ctx:
            self.backend(stub, retries=0).next_token(MESSAGES)
        self.assertEqual(str(ctx.exception), "Ollama 500: failed to decode batch")

    def test_status_is_none_without_a_reply(self):
        e = BackendError("cannot reach Ollama")
        self.assertIsNone(e.status)
        self.assertEqual(BackendError("x", 503).status, 503)
        self.assertEqual(str(ContextOverflowError("too long", 400)), "too long")

    def test_server_error_then_success(self):
        replies = iter([(503, {"error": "busy"}), (502, b"<html>bad gateway</html>"), (200, chat_response())])
        stub = self.serve(lambda m, p, b: next(replies))
        nt = self.backend(stub, retries=2).next_token(MESSAGES)
        self.assertEqual(nt.prompt_tokens, 92)
        self.assertEqual(len(stub.requests), 3)
        self.assertEqual(self.sleeps(), [0.5, 1.0])

    def test_non_json_error_body(self):
        stub = self.serve(lambda m, p, b: (502, b"<html>bad gateway</html>"))
        with self.assertRaises(BackendError) as ctx:
            self.backend(stub, retries=0).next_token(MESSAGES)
        self.assertIn("Ollama 502", str(ctx.exception))
        self.assertEqual(len(stub.requests), 1)
        self.time.sleep.assert_not_called()

    def test_error_detail(self):
        reason = ("Internal Server Error", {})
        cases = [
            (b'{"error": "boom"}', ("boom", {})),
            (b"<html>oops</html>", reason),
            (b'["not", "an", "object"]', reason),
            (b"{}", reason),
            (b'{"error": null}', reason),
            (b"\xff\xfe\x00", reason),
            (b'{"error": "123"}', ("123", {})),
            (b'{"error": {"message": "inner", "type": "t"}}', ("inner", {"message": "inner", "type": "t"})),
            (OVERFLOW_BODY.encode(), (OVERFLOW_MESSAGE, OVERFLOW_INNER)),
            (json.dumps({"error": json.dumps({"error": "plain inner"})}).encode(), ("plain inner", {})),
            # Nested JSON without a message never reaches the user as raw JSON.
            (json.dumps({"error": json.dumps({"error": {"type": "x"}})}).encode(), reason),
            (json.dumps({"error": json.dumps([1, 2])}).encode(), reason),
        ]
        for raw, expected in cases:
            with self.subTest(raw):
                fp = io.BytesIO(raw)
                error = urllib.error.HTTPError("http://x/api/chat", 500, "Internal Server Error", {}, fp)
                self.assertEqual(_error_detail(error), expected)
                # The error holds the open response, so it must be closed.
                self.assertTrue(fp.closed)

    def test_connection_refused(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        b = OllamaBackend(model="m", host=f"127.0.0.1:{port}", retries=1)
        with self.assertRaises(BackendError) as ctx:
            b.next_token(MESSAGES)
        self.assertIn(f"cannot reach Ollama at http://127.0.0.1:{port}", str(ctx.exception))
        self.assertIsNone(ctx.exception.status)
        self.assertEqual(self.sleeps(), [0.5])

    def test_replies_that_are_not_http_are_retried(self):
        cases = {
            "not HTTP": (b"SSH-2.0-OpenSSH_9.6\r\n", "sent a broken HTTP reply (BadStatusLine)"),
            "hang up": (b"", "cannot reach Ollama at"),
            "cut off": (b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{"model"', "sent a broken HTTP reply"),
        }
        for name, (reply, expected) in cases.items():
            with self.subTest(name):
                self.time.reset_mock()
                server = RawServer(reply)
                self.addCleanup(server.close)
                b = OllamaBackend(model="m", host=server.host, retries=2)
                with self.assertRaises(BackendError) as ctx:
                    b.next_token(MESSAGES)
                self.assertIn(expected, str(ctx.exception))
                self.assertIn(server.host, str(ctx.exception))
                self.assertIsNone(ctx.exception.status)
                self.assertEqual(server.connections, 3)
                self.assertEqual(self.sleeps(), [0.5, 1.0])

    def test_timeout(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def handler(method, path, body):
            release.wait(5)
            return 200, chat_response()

        b = self.backend(self.serve(handler), timeout=0.2, retries=0)
        with self.assertRaises(BackendError) as ctx:
            b.next_token(MESSAGES)
        self.assertIn("cannot reach Ollama", str(ctx.exception))


class ContextWindowTest(BackendTestCase):
    def reply(self, prompt_eval_count):
        return self.serve(lambda m, p, b: (200, chat_response(prompt_eval_count=prompt_eval_count)))

    def test_overflow_raises(self):
        for count in [4095, 4096, 5000]:
            with self.subTest(count):
                b = self.backend(self.reply(count), num_ctx=4096)
                with self.assertRaises(ContextOverflowError) as ctx:
                    b.next_token(MESSAGES)
                self.assertIsInstance(ctx.exception, BackendError)
                self.assertIn("4096", str(ctx.exception))
                self.time.sleep.assert_not_called()

    def test_just_under_the_window(self):
        nt = self.backend(self.reply(4094), num_ctx=4096).next_token(MESSAGES)
        self.assertEqual(nt.prompt_tokens, 4094)

    def test_overflow_reply_raises(self):
        # With truncate false, Ollama answers an oversized prompt with a 400.
        stub = self.serve(lambda m, p, b: (400, OVERFLOW_BODY.encode()))
        b = self.backend(stub)
        with self.assertRaises(ContextOverflowError) as ctx:
            b.next_token(MESSAGES)
        self.assertEqual(str(ctx.exception), OVERFLOW_ERROR)
        self.assertEqual(ctx.exception.status, 400)
        self.assertIsNone(ctx.exception.__cause__)
        self.assertIs(stub.bodies[0]["truncate"], False)
        self.assertEqual(len(stub.requests), 1)
        self.time.sleep.assert_not_called()

    def test_overflow_reply_variants(self):
        no_counts = "prompt does not fit the 4096-token context window. Shorten the state or raise NEX_NUM_CTX."
        typed = json.dumps({"error": {"message": "too long", "type": "exceed_context_size_error"}})
        cases = {
            "plain text": ({"error": OVERFLOW_MESSAGE}, OVERFLOW_ERROR),
            "type only": ({"error": "exceed_context_size_error"}, no_counts),
            "no counts": ({"error": typed}, no_counts),
            "text without counts": ({"error": "the request exceeds the available context size"}, no_counts),
        }
        for name, (payload, expected) in cases.items():
            with self.subTest(name):
                stub = self.serve(lambda m, p, b, payload=payload: (400, payload))
                with self.assertRaises(ContextOverflowError) as ctx:
                    self.backend(stub, num_ctx=4096).next_token(MESSAGES)
                self.assertEqual(str(ctx.exception), expected)

    def test_overflow_after_think_refusal(self):
        def handler(method, path, body):
            if "think" in body:
                return 400, {"error": THINK_REFUSAL}
            return 400, OVERFLOW_BODY.encode()

        stub = self.serve(handler)
        b = self.backend(stub)
        with self.assertRaises(ContextOverflowError) as ctx:
            b.next_token(MESSAGES)
        self.assertEqual(str(ctx.exception), OVERFLOW_ERROR)
        self.assertEqual(len(stub.requests), 2)
        self.assertIs(stub.bodies[1]["truncate"], False)
        # The retry did not succeed, so think is still sent next time.
        self.assertTrue(b._send_think)


class ListModelsTest(BackendTestCase):
    def test_list_models(self):
        tags = {
            "models": [
                {"name": "qwen3.5:9b", "model": "qwen3.5:9b", "size": 6600000000, "digest": "abc"},
                {"name": "qwen3:4b-instruct", "model": "qwen3:4b-instruct", "size": 2500000000, "digest": "def"},
            ]
        }

        def handler(method, path, body):
            if (method, path) == ("GET", "/api/tags"):
                return 200, tags
            return 404, {"error": "not found"}

        stub = self.serve(handler)
        self.assertEqual(self.backend(stub).list_models(), ["qwen3.5:9b", "qwen3:4b-instruct"])
        self.assertEqual(stub.requests[0]["method"], "GET")
        self.assertIsNone(stub.requests[0]["body"])

    def test_no_models(self):
        stub = self.serve(lambda m, p, b: (200, {}))
        self.assertEqual(self.backend(stub).list_models(), [])


class ConfigTest(BackendTestCase):
    def test_normalize_host(self):
        cases = {
            "127.0.0.1:11434": "http://127.0.0.1:11434",
            "http://127.0.0.1:11434": "http://127.0.0.1:11434",
            "http://127.0.0.1:11434/": "http://127.0.0.1:11434",
            "  127.0.0.1:11434/ ": "http://127.0.0.1:11434",
            "0.0.0.0:11434": "http://127.0.0.1:11434",
            "http://0.0.0.0:11434/": "http://127.0.0.1:11434",
            "ollama.lan:8080": "http://ollama.lan:8080",
            "https://ollama.example.com/": "https://ollama.example.com",
            "[::1]:11500": "http://[::1]:11500",
            # Without a scheme or port the ollama CLI uses port 11434.
            "0.0.0.0": "http://127.0.0.1:11434",
            "localhost": "http://localhost:11434",
            "[::1]": "http://[::1]:11434",
            ":11434": "http://127.0.0.1:11434",
            "": "http://127.0.0.1:11434",
            # The ollama CLI strips quotes, as left by a quoted value in a .env file.
            '"127.0.0.1:11500"': "http://127.0.0.1:11500",
            " 'localhost' ": "http://localhost:11434",
            '""': "http://127.0.0.1:11434",
            # IPv6, bare or bracketed. "::" binds every address like 0.0.0.0.
            "::1": "http://[::1]:11434",
            "[::1]:11434": "http://[::1]:11434",
            "http://[::1]": "http://[::1]",
            "https://[::1]:8443/": "https://[::1]:8443",
            "::": "http://[::1]:11434",
            "[::]": "http://[::1]:11434",
            "[::]:11500": "http://[::1]:11500",
            # An empty or invalid port falls back to the default, like the ollama CLI.
            "localhost:": "http://localhost:11434",
            "localhost:abc": "http://localhost:11434",
            "localhost:99999": "http://localhost:11434",
            "localhost:" + "1" * 5000: "http://localhost:11434",
            "http://localhost:": "http://localhost",
            "http://localhost:abc": "http://localhost",
            "http://:11500": "http://127.0.0.1:11500",
            "HTTP://LocalHost:11434": "http://LocalHost:11434",
            # A path is kept, as the ollama CLI keeps it.
            "example.com/ollama/": "http://example.com:11434/ollama",
            # Userinfo, query, and fragment are dropped instead of being read
            # as the host or port.
            "https://user:pass@ollama.example.com": "https://ollama.example.com",
            "http://user:pass@ollama.example.com:11434/api": "http://ollama.example.com:11434/api",
            "user@[::1]:11500": "http://[::1]:11500",
            "http://host:11434?x=1": "http://host:11434",
            "host:11500/base?x=1#frag": "http://host:11500/base",
        }
        for raw, expected in cases.items():
            with self.subTest(raw):
                self.assertEqual(_normalize_host(raw), expected)
                # The result must survive urllib's own parsing.
                urllib.parse.urlsplit(expected).port
                self.assertEqual(OllamaBackend(host=raw).host, expected)

    def test_odd_hosts_do_not_crash(self):
        # The ollama CLI turns these into unusable addresses too. They have to
        # fail as a BackendError on use, not when the backend is built.
        for raw in ["::1:11500", "[::1", "fe80::1%en0", "tcp://x", "http://"]:
            with self.subTest(raw):
                self.assertIsInstance(_normalize_host(raw), str)
                OllamaBackend(host=raw)
        for raw in ["::1:11500", "tcp://x"]:
            with self.subTest(raw):
                b = OllamaBackend(model="m", host=raw, retries=2)
                with self.assertRaises(BackendError) as ctx:
                    b.next_token(MESSAGES)
                self.assertIn(f"cannot reach Ollama at {b.host}", str(ctx.exception))

    def test_loopback_hosts(self):
        cases = {
            "http://127.0.0.1:11434": True,
            "http://127.0.0.53:11434": True,
            "http://localhost:11434": True,
            "http://LOCALHOST:11434": True,
            "http://ollama.localhost:11434": True,
            "http://localhost.:11434": True,
            "http://[::1]:11434": True,
            "http://192.168.1.20:11434": False,
            "http://ollama.lan:11434": False,
            "http://localhost.example.com:11434": False,
            "https://ollama.example.com": False,
            "http://[::1:11500]:11434": False,
        }
        for url, expected in cases.items():
            with self.subTest(url):
                self.assertIs(_is_loopback(url), expected)

    def test_defaults(self):
        b = OllamaBackend()
        self.assertEqual(b.model, DEFAULT_BACKEND_MODEL)
        self.assertEqual(b.model, "qwen3.5:9b")
        self.assertEqual(b.host, "http://127.0.0.1:11434")
        self.assertEqual(b.num_ctx, DEFAULT_NUM_CTX)
        self.assertEqual(b.num_ctx, 8192)
        self.assertEqual((b.timeout, b.keep_alive, b.retries), (120.0, "30m", 2))

    def test_environment(self):
        os.environ["OLLAMA_HOST"] = "0.0.0.0:11500"
        os.environ["NEX_BACKEND_MODEL"] = "qwen3:4b-instruct"
        os.environ["NEX_NUM_CTX"] = "16384"
        b = OllamaBackend()
        self.assertEqual(b.host, "http://127.0.0.1:11500")
        self.assertEqual(b.model, "qwen3:4b-instruct")
        self.assertEqual(b.num_ctx, 16384)

    def test_arguments_beat_environment(self):
        os.environ["OLLAMA_HOST"] = "elsewhere:1"
        os.environ["NEX_BACKEND_MODEL"] = "env-model"
        os.environ["NEX_NUM_CTX"] = "1024"
        b = OllamaBackend(model="m", host="127.0.0.1:2", num_ctx=2048)
        self.assertEqual((b.model, b.host, b.num_ctx), ("m", "http://127.0.0.1:2", 2048))

    def test_with_model_keeps_settings(self):
        b = OllamaBackend(model="a", host="example.com:1234", num_ctx=2048, timeout=5.0, keep_alive="1m", retries=4)
        c = b.with_model("b")
        self.assertIsInstance(c, OllamaBackend)
        self.assertEqual(c.model, "b")
        self.assertEqual(
            (c.host, c.num_ctx, c.timeout, c.keep_alive, c.retries), ("http://example.com:1234", 2048, 5.0, "1m", 4)
        )
        self.assertEqual(b.model, "a")


class ProxyTest(BackendTestCase):
    """A recording stub plays the proxy that HTTP_PROXY points at."""

    def setUp(self):
        super().setUp()
        for name in list(os.environ):
            if name.lower().endswith("_proxy"):
                del os.environ[name]
        self.proxy = self.serve(ok)
        os.environ["HTTP_PROXY"] = self.proxy.host

    def test_loopback_hosts_skip_the_proxy(self):
        stub = self.serve(ok)
        b = self.backend(stub)
        b.next_token(MESSAGES)
        b.with_model("qwen3.5:9b").next_token(MESSAGES)
        b.list_models()
        self.assertEqual(len(stub.requests), 3)
        self.assertEqual(self.proxy.requests, [])

    def test_remote_hosts_use_the_proxy(self):
        b = OllamaBackend(model="m", host="ollama.invalid:11434")
        b.next_token(MESSAGES)
        b.with_model("other").next_token(MESSAGES)
        self.assertEqual([r["path"] for r in self.proxy.requests], ["http://ollama.invalid:11434/api/chat"] * 2)
        self.assertEqual([body["model"] for body in self.proxy.bodies], ["m", "other"])


class FakeBackendTest(unittest.TestCase):
    def test_wraps_pairs_and_records_calls(self):
        b = FakeBackend(lambda m: [("Yes", -0.1), ("No", -2.3)])
        nt = b.next_token(MESSAGES)
        self.assertEqual(nt.top_logprobs, [("Yes", -0.1), ("No", -2.3)])
        self.assertEqual(nt.prompt_tokens, sum(len(m["content"]) // 4 for m in MESSAGES))
        self.assertEqual(nt.cached_tokens, 0)
        self.assertEqual(b.calls, [MESSAGES])

    def test_accepts_generators(self):
        nt = FakeBackend(lambda m: ((t, -1.0) for t in "AB")).next_token(MESSAGES)
        self.assertEqual(nt.top_logprobs, [("A", -1.0), ("B", -1.0)])

    def test_passes_next_token_through(self):
        scripted = NextToken([("A", 0.0)], prompt_tokens=7, cached_tokens=3)
        self.assertIs(FakeBackend(lambda m: scripted).next_token(MESSAGES), scripted)

    def test_with_model(self):
        responder = lambda m: [("A", 0.0)]
        b = FakeBackend(responder)
        b.next_token(MESSAGES)
        c = b.with_model("other")
        self.assertEqual((b.model, c.model), ("fake", "other"))
        self.assertIs(c.responder, responder)
        self.assertEqual(c.calls, [])
        self.assertEqual(c.list_models(), ["other"])


if __name__ == "__main__":
    unittest.main()
