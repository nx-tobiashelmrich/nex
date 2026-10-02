"""Tests for the HTTP server, against a FakeBackend with scripted logprobs."""

import contextlib
import http.client
import io
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from nex import BackendError, Calibration, ContextOverflowError, FakeBackend, ModelNotFoundError, Nex
from nex.server import MAX_BODY, MAX_MODEL, NexHTTPServer, _is_local, make_server, serve

ROOT = Path(__file__).resolve().parent.parent
# Talk to the test server directly even when proxy variables are set.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

STATE = "Help! My payouts have been failing for 3 days."
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["Can wait", "Soon", "Right now"],
    },
    "upset": {"type": "noul", "instructions": "Is the customer upset?"},
}


def scripted(messages):
    """Next-token candidates chosen by question, with several spellings per
    label so the server answer also shows the folding."""
    text = messages[-1]["content"]
    if "Which team" in text:
        return [("B", math.log(0.5)), (" B", math.log(0.2)), ("A", math.log(0.2)), ("c", math.log(0.1))]
    if "How urgent" in text:
        return [("2", math.log(0.6)), ("1", math.log(0.3)), ("0", math.log(0.1))]
    if "upset" in text:
        return [("Yes", math.log(0.8)), (" no", math.log(0.2))]
    return [("A", 0.0)]


def raising(error):
    def responder(messages):
        raise error

    return responder


class BrokenListing(FakeBackend):
    def list_models(self):
        raise BackendError("cannot reach Ollama at http://127.0.0.1:1")


def call(url, method="GET", body=None, raw=None, headers=None):
    """Return ``(status, headers, parsed JSON body)`` for any status."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    # Added last so a test can override Content-Type or Host.
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    try:
        with OPENER.open(req, timeout=10) as resp:
            return resp.status, resp.headers, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        with e:
            return e.code, e.headers, json.loads(e.read())


class ServerTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("NEX_API_KEY", None)
        self.servers = []
        self.backend = FakeBackend(scripted)
        self.url = self.start(self.backend)

    def tearDown(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(5)

    def start(self, backend):
        server = make_server(Nex(backend, calibration=Calibration()), port=0, log_requests=False)
        # A short poll interval keeps shutdown() in tearDown fast.
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        host, port = server.server_address[:2]
        return f"http://{host}:{port}"

    def post(self, body, path="/v1/systemone", url=None, **kwargs):
        return call((url or self.url) + path, "POST", body, **kwargs)

    def assertError(self, result, status, kind, field=None):
        code, headers, body = result
        self.assertEqual(code, status, body)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(body["error"]["type"], kind)
        self.assertTrue(body["error"]["message"])
        if field is None:
            self.assertNotIn("field", body["error"])
        else:
            self.assertEqual(body["error"]["field"], field)
        return body

    def raw_exchange(self, data):
        """Send raw bytes, return ``(status, headers, body)`` of the reply."""
        host, port = self.servers[0][0].server_address[:2]
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(data)
            response = http.client.HTTPResponse(sock)
            response.begin()
            return response.status, response.headers, json.loads(response.read())

    # Routes

    def test_health(self):
        status, headers, body = call(self.url + "/health")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(body, {"status": "ok", "model": "nex-0.1.0+fake"})

    def test_system_one_answers_every_question(self):
        status, headers, body = self.post({"state": STATE, "model": "nex-latest", "questions": QUESTIONS})
        self.assertEqual(status, 200, body)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(body["model"], "nex-0.1.0+fake")
        self.assertEqual(set(body), {"model", "answers", "usage"})
        answers = body["answers"]

        department = answers["department"]
        self.assertEqual(department["type"], "choice")
        self.assertEqual(department["choice"], "technical")
        for option, p in {"billing": 0.2, "technical": 0.7, "sales": 0.1}.items():
            self.assertAlmostEqual(department["probabilities"][option], p, places=4)
        self.assertAlmostEqual(department["confidence"], 0.55, places=4)

        urgency = answers["urgency"]
        self.assertEqual(urgency["type"], "score")
        self.assertAlmostEqual(urgency["score"], 1.5, places=4)
        self.assertEqual(urgency["legend"], {"0": "Can wait", "1": "Soon", "2": "Right now"})
        for level, p in {"0": 0.1, "1": 0.3, "2": 0.6}.items():
            self.assertAlmostEqual(urgency["probabilities"][level], p, places=4)
        self.assertAlmostEqual(urgency["confidence"], 0.25, places=4)

        self.assertEqual(answers["upset"]["type"], "noul")
        self.assertAlmostEqual(answers["upset"]["noul"], 0.8, places=4)

        self.assertEqual(len(self.backend.calls), 3)
        expected_input = sum(len(m["content"]) // 4 for call_ in self.backend.calls for m in call_)
        self.assertEqual(body["usage"], {"input_tokens": expected_input, "output_tokens": 3})

    def test_model_is_optional_and_selects_backend(self):
        cases = {
            None: "nex-0.1.0+fake",
            "nex-latest": "nex-0.1.0+fake",
            "nex-0.1.0+fake": "nex-0.1.0+fake",
            "other-model": "nex-0.1.0+other-model",
            "nex-0.1.0+qwen3:4b-instruct": "nex-0.1.0+qwen3:4b-instruct",
        }
        for model, expected in cases.items():
            request = {"state": STATE, "questions": {"upset": QUESTIONS["upset"]}}
            if model is not None:
                request["model"] = model
            status, _, body = self.post(request)
            self.assertEqual(status, 200, body)
            self.assertEqual(body["model"], expected)

    def test_structured_state(self):
        for state in ({"ticket": {"text": STATE, "plan": "pro"}}, [STATE, "Still broken today."]):
            status, _, body = self.post({"state": state, "questions": {"upset": QUESTIONS["upset"]}})
            self.assertEqual(status, 200, body)
            self.assertAlmostEqual(body["answers"]["upset"]["noul"], 0.8, places=4)

    def test_debug_adds_diagnostics(self):
        status, _, body = self.post({"state": STATE, "questions": QUESTIONS}, path="/v1/systemone?debug=1")
        self.assertEqual(status, 200, body)
        diagnostics = body["diagnostics"]
        self.assertEqual(set(diagnostics), set(QUESTIONS))
        self.assertEqual(diagnostics["department"]["raw_probabilities"], [0.2, 0.7, 0.1])
        self.assertAlmostEqual(diagnostics["department"]["label_mass"], 1.0, places=4)
        self.assertEqual(diagnostics["upset"]["raw_probabilities"], [0.8, 0.2])
        for diag in diagnostics.values():
            self.assertEqual(
                set(diag), {"label_mass", "raw_probabilities", "prompt_tokens", "cached_tokens", "latency_ms"}
            )

        for path in ("/v1/systemone", "/v1/systemone?debug=0", "/v1/systemone?debug="):
            status, _, body = self.post({"state": STATE, "questions": QUESTIONS}, path=path)
            self.assertEqual(status, 200, body)
            self.assertNotIn("diagnostics", body)

    def test_models(self):
        status, headers, body = call(self.url + "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        models = body["models"]
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["name"], "nex-latest")
        self.assertEqual(models[0]["points_to"], "nex-0.1.0+fake")
        self.assertTrue(models[0]["description"])
        self.assertEqual(models[1]["name"], "nex-0.1.0+fake")
        self.assertEqual(models[1]["backend_model"], "fake")

    def test_models_survives_backend_failure(self):
        url = self.start(BrokenListing(scripted))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            status, _, body = call(url + "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual([m["name"] for m in body["models"]], ["nex-latest"])
        self.assertIn("could not list backend models", err.getvalue())

    def test_keep_alive_serves_several_requests_per_connection(self):
        host, port = self.servers[0][0].server_address[:2]
        conn = http.client.HTTPConnection(host, port, timeout=5)
        self.addCleanup(conn.close)
        payload = json.dumps({"state": STATE, "questions": QUESTIONS}).encode()
        first_socket = None
        for method, path, data, expected in [
            ("GET", "/health", None, 200),
            ("POST", "/v1/systemone", payload, 200),
            # An unused body on an error reply is drained, so the
            # connection stays usable.
            ("POST", "/v1/nope", payload, 404),
            ("GET", "/v1/systemone", None, 405),
            ("POST", "/v1/systemone", payload, 200),
        ]:
            conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            json.loads(response.read())
            self.assertEqual(response.status, expected, path)
            self.assertNotEqual(response.headers.get("Connection"), "close", path)
            first_socket = first_socket or conn.sock
            self.assertIs(conn.sock, first_socket, path)

    # Request errors

    def test_invalid_json_is_400(self):
        for raw in (b"{not json", b"", b"\xff\xfe\x00", b"[" * 100000):
            self.assertError(self.post(None, raw=raw), 400, "invalid_request")

    def test_non_object_body_is_400(self):
        for value in ([1, 2], "state", 3, None):
            self.assertError(self.post(None, raw=json.dumps(value).encode()), 400, "invalid_request")

    def test_unknown_top_level_field_is_422(self):
        body = {"state": STATE, "questions": QUESTIONS, "temperature": 0.2}
        error = self.assertError(self.post(body), 422, "validation_error", "temperature")
        self.assertIn("temperature", error["error"]["message"])
        self.assertEqual(self.backend.calls, [])

    def test_validation_errors_are_422_with_field(self):
        one_option = {"type": "choice", "instructions": "Pick", "criteria": {"only": None}}
        cases = [
            ({"questions": QUESTIONS}, "state"),
            ({"state": 42, "questions": QUESTIONS}, "state"),
            ({"state": STATE}, "questions"),
            ({"state": STATE, "questions": {}}, "questions"),
            ({"state": STATE, "questions": {"q": {"type": "rank", "instructions": "x"}}}, "questions.q.type"),
            ({"state": STATE, "questions": {"q": one_option}}, "questions.q.criteria"),
            ({"state": STATE, "model": 42, "questions": QUESTIONS}, "model"),
            ({"state": STATE, "model": " ", "questions": QUESTIONS}, "model"),
            ({"state": STATE, "model": "m" * (MAX_MODEL + 1), "questions": QUESTIONS}, "model"),
        ]
        for body, field in cases:
            with self.subTest(field=field, body=body):
                self.assertError(self.post(body), 422, "validation_error", field)
        self.assertEqual(self.backend.calls, [])

    def test_model_of_exactly_256_characters_is_accepted(self):
        model = "m" * MAX_MODEL
        status, _, body = self.post({"state": STATE, "model": model, "questions": {"upset": QUESTIONS["upset"]}})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["model"], f"nex-0.1.0+{model}")

    def test_model_not_found_is_422_on_model(self):
        missing = ModelNotFoundError('model "nope" not found, try pulling it first')
        url = self.start(FakeBackend(raising(missing)))
        result = self.post({"state": STATE, "model": "nope", "questions": QUESTIONS}, url=url)
        body = self.assertError(result, 422, "model_not_found", "model")
        self.assertIn("not found", body["error"]["message"])

    def test_context_overflow_is_422_on_state(self):
        url = self.start(FakeBackend(raising(ContextOverflowError("prompt reached the 8192-token context window"))))
        result = self.post({"state": STATE, "questions": QUESTIONS}, url=url)
        body = self.assertError(result, 422, "context_overflow", "state")
        self.assertIn("8192", body["error"]["message"])

    def test_backend_error_is_502(self):
        url = self.start(FakeBackend(raising(BackendError("cannot reach Ollama at http://127.0.0.1:1"))))
        body = self.assertError(self.post({"state": STATE, "questions": QUESTIONS}, url=url), 502, "backend_error")
        self.assertIn("cannot reach Ollama", body["error"]["message"])

    def test_unexpected_error_is_500_without_traceback(self):
        url = self.start(FakeBackend(raising(RuntimeError("secret internal detail"))))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            status, headers, body = self.post({"state": STATE, "questions": QUESTIONS}, url=url)
        self.assertEqual(status, 500)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(body["error"]["type"], "internal_error")
        text = json.dumps(body)
        self.assertNotIn("Traceback", text)
        self.assertNotIn("secret internal detail", text)
        self.assertIn("Traceback", err.getvalue())
        self.assertIn("secret internal detail", err.getvalue())

    def test_unknown_path_is_404(self):
        self.assertError(call(self.url + "/nope"), 404, "not_found")
        self.assertError(call(self.url + "/v1/nope"), 404, "not_found")
        self.assertError(self.post({"state": STATE}, path="/v1/systemtwo"), 404, "not_found")

    def test_wrong_method_is_405(self):
        for method, path, allowed in [
            ("GET", "/v1/systemone", "POST"),
            ("PUT", "/v1/systemone", "POST"),
            ("POST", "/v1/models", "GET"),
            ("DELETE", "/v1/models", "GET"),
            ("POST", "/health", "GET"),
        ]:
            with self.subTest(method=method, path=path):
                body = {} if method in ("POST", "PUT") else None
                result = call(self.url + path, method, body)
                self.assertError(result, 405, "method_not_allowed")
                self.assertEqual(result[1]["Allow"], allowed)

    def test_trailing_slash_is_accepted(self):
        status, _, body = call(self.url + "/health/")
        self.assertEqual(status, 200, body)

    def test_body_over_4mb_is_413(self):
        # Only the header is sent, so the server must refuse from the length
        # alone without waiting for the body.
        host, port = self.servers[0][0].server_address[:2]
        for extra in ({}, {"Expect": "100-continue"}):
            conn = http.client.HTTPConnection(host, port, timeout=5)
            self.addCleanup(conn.close)
            conn.putrequest("POST", "/v1/systemone")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(MAX_BODY + 1))
            for name, value in extra.items():
                conn.putheader(name, value)
            conn.endheaders()
            response = conn.getresponse()
            body = json.loads(response.read())
            self.assertEqual(response.status, 413, extra)
            self.assertEqual(response.headers["Content-Type"], "application/json")
            self.assertEqual(response.headers["Connection"], "close")
            self.assertEqual(body["error"]["type"], "payload_too_large")
        self.assertEqual(self.backend.calls, [])

    def test_body_of_exactly_4mb_is_accepted(self):
        payload = json.dumps({"state": STATE, "questions": {"upset": QUESTIONS["upset"]}}).encode()
        payload += b" " * (MAX_BODY - len(payload))
        status, _, body = self.post(None, raw=payload)
        self.assertEqual(status, 200, body)

    def test_bad_content_length_and_chunked_bodies_are_refused(self):
        host, port = self.servers[0][0].server_address[:2]
        for name, value, status in [("Content-Length", "abc", 400), ("Transfer-Encoding", "chunked", 411)]:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            self.addCleanup(conn.close)
            conn.putrequest("POST", "/v1/systemone")
            conn.putheader(name, value)
            conn.endheaders()
            response = conn.getresponse()
            body = json.loads(response.read())
            self.assertEqual(response.status, status)
            self.assertEqual(response.headers["Connection"], "close")
            self.assertEqual(body["error"]["type"], "invalid_request")

    def test_conflicting_content_lengths_are_400_and_close(self):
        # With the first value winning, the declared body would be parsed as
        # a second, smuggled request on the same connection.
        smuggled = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
        host, port = self.servers[0][0].server_address[:2]
        for line in (b"POST /v1/systemone HTTP/1.1", b"GET /health HTTP/1.1"):
            with self.subTest(line=line):
                request = (
                    line + b"\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 0\r\nContent-Length: %d\r\n\r\n" % len(smuggled)
                ) + smuggled
                with socket.create_connection((host, port), timeout=5) as sock:
                    sock.sendall(request)
                    received = b""
                    # The server must close, or this read times out.
                    while chunk := sock.recv(65536):
                        received += chunk
                head, _, payload = received.partition(b"\r\n\r\n")
                self.assertEqual(received.count(b"HTTP/1.1 "), 1, received)
                self.assertTrue(head.startswith(b"HTTP/1.1 400 "), head)
                self.assertIn(b"\r\nConnection: close", head)
                self.assertEqual(json.loads(payload)["error"]["type"], "invalid_request")
        self.assertEqual(self.backend.calls, [])

    def test_identical_duplicate_content_lengths_are_accepted(self):
        payload = json.dumps({"state": STATE, "questions": {"upset": QUESTIONS["upset"]}}).encode()
        request = (
            b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\nContent-Length: %d\r\n\r\n" % (len(payload), len(payload))
        ) + payload
        status, _, body = self.raw_exchange(request)
        self.assertEqual(status, 200, body)

    def test_errors_from_http_server_itself_are_json(self):
        status, headers, body = self.raw_exchange(b"BREW /health HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertEqual(status, 501)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(body["error"]["type"], "not_implemented")

        status, headers, body = self.raw_exchange(b"GET /health HTTP/1.1 extra\r\n\r\n")
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(body["error"]["type"], "invalid_request")

    # Host and Origin

    ROUTE_REQUESTS = [
        ("GET", "/health", None),
        ("GET", "/v1/models", None),
        ("POST", "/v1/systemone", {"state": STATE, "questions": {"upset": QUESTIONS["upset"]}}),
        ("GET", "/nope", None),
    ]

    def test_foreign_host_is_403_on_every_route(self):
        # A DNS rebinding page reaches 127.0.0.1 but its Host is still the
        # attacker's name.
        for method, path, body in self.ROUTE_REQUESTS:
            with self.subTest(path=path):
                result = call(self.url + path, method, body, headers={"Host": "evil.example:8787"})
                self.assertError(result, 403, "forbidden")
        self.assertEqual(self.backend.calls, [])

    def test_host_header_checks(self):
        allowed = [
            "localhost",
            "LOCALHOST:8787",
            "localhost.:8787",
            "app.localhost",
            "127.0.0.1",
            "127.5.6.7:80",
            "[::1]:8787",
        ]
        refused = [
            "evil.example",
            "localhost.evil.example",
            "127.0.0.1.nip.io",
            "evil.example@localhost",
            "localhost@evil.example",
            "::1",
            "[::1",
            "[::1]x",
            "",
        ]
        for value in allowed + refused:
            with self.subTest(host=value):
                result = call(self.url + "/health", headers={"Host": value})
                if value in allowed:
                    self.assertEqual(result[0], 200, result[2])
                else:
                    self.assertError(result, 403, "forbidden")

    def test_missing_host_is_accepted(self):
        # Browsers always send Host, so its absence is not a rebinding attempt.
        status, _, body = self.raw_exchange(b"GET /health HTTP/1.0\r\n\r\n")
        self.assertEqual(status, 200, body)

    def test_any_foreign_host_among_duplicates_is_403(self):
        request = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n"
        status, _, body = self.raw_exchange(request)
        self.assertEqual((status, body["error"]["type"]), (403, "forbidden"))

    def test_origin_header_checks(self):
        allowed = [
            "http://localhost:3000",
            "https://app.localhost",
            "http://127.0.0.1:5173",
            "http://[::1]:8080",
        ]
        refused = [
            "https://evil.example",
            "http://localhost.evil.example",
            "http://evil.example@localhost",
            "http://localhost/path",
            "null",
            "file://",
            "chrome-extension://abcdef",
            "localhost",
        ]
        request = {"state": STATE, "questions": {"upset": QUESTIONS["upset"]}}
        for value in allowed + refused:
            with self.subTest(origin=value):
                result = self.post(request, headers={"Origin": value})
                if value in allowed:
                    self.assertEqual(result[0], 200, result[2])
                else:
                    self.assertError(result, 403, "forbidden")

    def test_cross_origin_text_plain_post_is_403(self):
        # A browser sends this without a CORS preflight, so the Origin check
        # is what keeps another site from running questions.
        raw = json.dumps({"state": STATE, "questions": QUESTIONS}).encode()
        for method, path, _ in self.ROUTE_REQUESTS:
            with self.subTest(path=path):
                headers = {"Origin": "https://evil.example", "Content-Type": "text/plain"}
                result = call(self.url + path, method, raw=raw if method == "POST" else None, headers=headers)
                self.assertError(result, 403, "forbidden")
        self.assertEqual(self.backend.calls, [])

    def test_bound_host_name_is_accepted(self):
        server = self.servers[0][0]
        server.bound_names.add("nex.internal")
        status, _, body = call(self.url + "/health", headers={"Host": "nex.internal:8787"})
        self.assertEqual(status, 200, body)
        self.assertError(call(self.url + "/health", headers={"Host": "other.internal"}), 403, "forbidden")

    def test_ipv6_loopback_bind_is_protected(self):
        try:
            server = make_server(Nex(self.backend, calibration=Calibration()), "::1", 0, log_requests=False)
        except OSError:
            self.skipTest("no IPv6 loopback")
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        self.assertTrue(server.loopback)
        url = f"http://[::1]:{server.server_address[1]}"
        self.assertEqual(call(url + "/health")[0], 200)
        self.assertError(call(url + "/health", headers={"Host": "evil.example"}), 403, "forbidden")

    def test_non_loopback_bind_skips_host_and_origin_checks(self):
        # Binding 0.0.0.0 in a test would expose the port, so the flag the
        # server derives from its address is flipped instead. ServeTest
        # checks the flag for 0.0.0.0 and other addresses.
        self.servers[0][0].loopback = False
        headers = {"Host": "nex.example.com", "Origin": "https://dashboard.example.com"}
        for method, path, body in self.ROUTE_REQUESTS[:3]:
            with self.subTest(path=path):
                status, _, reply = call(self.url + path, method, body, headers=headers)
                self.assertEqual(status, 200, reply)

    def test_is_local(self):
        for name in ("localhost", "a.b.localhost", "127.0.0.1", "127.255.0.9", "::1"):
            self.assertTrue(_is_local(name), name)
        for name in ("", "0.0.0.0", "::", "192.168.1.10", "10.0.0.1", "example.com", "localhostx", "notlocalhost"):
            self.assertFalse(_is_local(name), name)

    # Auth

    def test_api_key_required_when_set(self):
        os.environ["NEX_API_KEY"] = "s3cret"
        request = {"state": STATE, "questions": {"upset": QUESTIONS["upset"]}}
        bad_headers = [
            {},
            {"Authorization": "Bearer wrong"},
            {"Authorization": "s3cret"},
            {"Authorization": "Basic s3cret"},
        ]
        for headers in bad_headers:
            with self.subTest(headers=headers):
                result = self.post(request, headers=headers)
                self.assertError(result, 401, "unauthorized")
                self.assertEqual(result[1]["WWW-Authenticate"], "Bearer")
        self.assertError(call(self.url + "/v1/models"), 401, "unauthorized")
        self.assertError(call(self.url + "/v1/nope"), 401, "unauthorized")
        self.assertEqual(self.backend.calls, [])

        for header in ("Bearer s3cret", "bearer s3cret"):
            status, _, body = self.post(request, headers={"Authorization": header})
            self.assertEqual(status, 200, body)
        status, _, _ = call(self.url + "/v1/models", headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(status, 200)
        # Health checks stay open so load balancers do not need the key.
        status, _, _ = call(self.url + "/health")
        self.assertEqual(status, 200)

    def test_any_authorization_accepted_when_key_unset(self):
        request = {"state": STATE, "questions": {"upset": QUESTIONS["upset"]}}
        for value in ("", "   "):
            os.environ["NEX_API_KEY"] = value
            for headers in ({}, {"Authorization": "Bearer jev_client_key"}, {"Authorization": "Basic eA=="}):
                with self.subTest(key=value, headers=headers):
                    status, _, body = self.post(request, headers=headers)
                    self.assertEqual(status, 200, body)
        os.environ.pop("NEX_API_KEY")
        status, _, body = self.post(request, headers={"Authorization": "Bearer anything"})
        self.assertEqual(status, 200, body)


class ServeTest(unittest.TestCase):
    """``serve`` runs in the foreground, so it is tested in a subprocess."""

    # A shell running the suite in the background starts children with
    # SIGINT ignored, so the child restores Python's Ctrl-C handler first.
    CHILD = (
        "import signal\n"
        "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
        "from nex import Calibration, FakeBackend, Nex\n"
        "from nex.server import serve\n"
        "serve(Nex(FakeBackend(lambda m: [('Yes', 0.0)]), calibration=Calibration()), '127.0.0.1', 0)\n"
    )

    def test_stops_cleanly_on_ctrl_c_and_sigterm(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum.name):
                env = {k: v for k, v in os.environ.items() if k != "NEX_API_KEY"}
                proc = subprocess.Popen(
                    [sys.executable, "-c", self.CHILD],
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                # Never hang the suite if the child fails to start.
                guard = threading.Timer(15, proc.kill)
                guard.start()
                try:
                    line = proc.stdout.readline()
                    match = re.search(r"http://\S+", line)
                    if match is None:
                        proc.kill()
                        self.fail(f"no URL in {line!r}, stderr: {proc.communicate()[1]}")
                    status, _, body = call(match.group(0) + "/health")
                    self.assertEqual((status, body["model"]), (200, "nex-0.1.0+fake"))
                    proc.send_signal(signum)
                    out, err = proc.communicate(timeout=10)
                finally:
                    guard.cancel()
                    if proc.poll() is None:
                        proc.kill()
                        proc.communicate()
                self.assertEqual(proc.returncode, 0, err)
                self.assertNotIn("Traceback", err)
                self.assertNotIn("warning", err)
                self.assertIn("stopped", out)

    @contextlib.contextmanager
    def unbound(self):
        """Let NexHTTPServer take any address without binding a socket, so
        the tests open no port on the network."""

        def bind(server):
            # server_address still holds the requested address here. A real
            # bind reports an empty host as 0.0.0.0.
            host, port = server.server_address[:2]
            server.server_address = (host or "0.0.0.0", port)

        with (
            mock.patch.object(NexHTTPServer, "server_bind", bind),
            mock.patch.object(NexHTTPServer, "server_activate", lambda server: None),
            mock.patch.object(NexHTTPServer, "serve_forever", side_effect=KeyboardInterrupt),
        ):
            yield

    def serve_stderr(self, host, key=None):
        """What ``serve`` on ``host`` writes to stderr before it stops."""
        nex = Nex(FakeBackend(scripted), calibration=Calibration())
        with (
            mock.patch.dict(os.environ),
            self.unbound(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            os.environ.pop("NEX_API_KEY", None)
            if key is not None:
                os.environ["NEX_API_KEY"] = key
            serve(nex, host, 8787)
        return err.getvalue()

    def test_loopback_flag_follows_bind_address(self):
        nex = Nex(FakeBackend(scripted), calibration=Calibration())
        cases = {
            "127.0.0.1": True,
            "127.0.0.2": True,
            "localhost": True,
            "::1": True,
            "0.0.0.0": False,
            "": False,
            "::": False,
            "192.168.1.10": False,
        }
        with self.unbound():
            for host, loopback in cases.items():
                server = NexHTTPServer((host, 8787), nex)
                server.server_close()
                self.assertEqual(server.loopback, loopback, host)

    def test_warns_on_non_loopback_bind_without_api_key(self):
        for host in ("0.0.0.0", "", "::", "192.168.1.10"):
            for key in (None, "", "   "):
                with self.subTest(host=host, key=key):
                    err = self.serve_stderr(host, key)
                    self.assertEqual(len(err.splitlines()), 1, err)
                    self.assertIn("warning", err)
                    self.assertIn("NEX_API_KEY", err)

    def test_no_warning_on_loopback_or_with_api_key(self):
        for host, key in [("127.0.0.1", None), ("localhost", None), ("::1", None), ("0.0.0.0", "s3cret")]:
            with self.subTest(host=host, key=key):
                self.assertEqual(self.serve_stderr(host, key), "")


if __name__ == "__main__":
    unittest.main()
