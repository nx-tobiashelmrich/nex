"""HTTP server that speaks the wire format of Jev's System One API.

``POST /v1/systemone`` takes ``{"state", "model", "questions"}`` and returns
typed answers in the format of docs/SPEC.md, which matches Jev's, so a client
written for Jev can point its base URL at Nex. ``GET /v1/models`` and ``GET /health``
report what is being served. Every reply, errors included, is JSON.

On a loopback address the server only answers requests addressed to
localhost and pages served from it, which blocks DNS rebinding and
cross-site requests from a browser.
"""

import hmac
import ipaddress
import json
import os
import signal
import socket
import socketserver
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .backend import BackendError, ContextOverflowError, ModelNotFoundError
from .primitives import ValidationError

MAX_BODY = 4 * 1024 * 1024
MAX_MODEL = 256
BODY_FIELDS = ("state", "model", "questions")
# Each known path accepts exactly one method.
ROUTES = {"/v1/systemone": "POST", "/v1/models": "GET", "/health": "GET"}
# Error types for statuses that http.server itself may send.
STATUS_TYPES = {400: "invalid_request", 413: "payload_too_large", 414: "invalid_request", 431: "invalid_request"}
TOO_LARGE = f"request body is over the {MAX_BODY // (1024 * 1024)} MB limit"


class ApiError(Exception):
    """An error reply: HTTP status plus a ``{"error": {...}}`` body."""

    def __init__(self, status, kind, message, field=None, headers=None):
        super().__init__(message)
        self.status = status
        self.body = _error(kind, message, field)
        self.headers = headers or {}


def _error(kind, message, field=None):
    error = {"type": kind, "message": message}
    if field is not None:
        error["field"] = field
    return {"error": error}


def _encode(payload):
    # NaN is not JSON. Failing here turns it into a 500 instead of a body
    # that clients cannot parse.
    return json.dumps(payload, allow_nan=False).encode()


def _printable(text):
    # Request paths are client input, so keep terminal escapes out of the log.
    return "".join(c if c.isprintable() else "?" for c in text)


def _is_local(name):
    """True for localhost, ``*.localhost``, and loopback IPs."""
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _bind_name(host):
    return host.strip().strip("[]").lower().rstrip(".")


def _host_name(value):
    """Hostname of a Host header value, lowercased and without port or
    brackets, or None when it does not parse."""
    value = value.strip().lower()
    if value.startswith("["):
        name, closed, rest = value[1:].partition("]")
        if not closed or (rest and not rest.startswith(":")):
            return None
    else:
        name = value.partition(":")[0]
    return name.rstrip(".") or None


def _origin_name(value):
    # A browser sends scheme://host[:port] or "null", never a path or
    # userinfo, so anything else is refused rather than parsed leniently.
    scheme, sep, rest = value.strip().partition("://")
    if not scheme or not sep or "/" in rest or "@" in rest:
        return None
    return _host_name(rest)


class NexRequestHandler(BaseHTTPRequestHandler):
    server_version = f"Nex/{__version__}"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    # Closes idle keep-alive connections and stalled uploads.
    timeout = 60
    # Headers and body go out in separate writes. With Nagle on, a keep-alive
    # client can wait 40 ms on a delayed ACK for every reply.
    disable_nagle_algorithm = True

    _started = None
    _body_read = False
    _close = False

    def handle_one_request(self):
        self._started = None
        self._body_read = False
        self._close = False
        super().handle_one_request()

    def do_GET(self):
        self._dispatch()

    do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = do_GET

    def _dispatch(self):
        self._started = time.perf_counter()
        url = urlsplit(self.path)
        path = url.path.rstrip("/") or "/"
        headers = {}
        try:
            if len(self._content_lengths()) > 1:
                # A proxy that picks another value would frame the next
                # request differently, so the connection cannot be reused.
                self._close = True
                raise ApiError(400, "invalid_request", "conflicting Content-Length headers")
            self._check_host()
            if path.startswith("/v1/"):
                self._check_auth()
            if path not in ROUTES:
                raise ApiError(404, "not_found", f"no route for {path}")
            if self.command != ROUTES[path]:
                allowed = ROUTES[path]
                raise ApiError(
                    405, "method_not_allowed", f"{path} only accepts {allowed}", headers={"Allow": allowed}
                )
            status, body = 200, _encode(self._route(path, parse_qs(url.query)))
        except ApiError as e:
            status, body, headers = e.status, _encode(e.body), e.headers
        except Exception:
            # The traceback goes to the operator, never to the client.
            sys.stderr.write(f"nex: unhandled error on {self.command} {_printable(self.path)}\n")
            traceback.print_exc(file=sys.stderr)
            status, body = 500, _encode(_error("internal_error", "internal server error, see the server log"))
        self._send(status, body, headers)

    def _route(self, path, query):
        if path == "/v1/systemone":
            return self._system_one(query)
        if path == "/v1/models":
            return self._models()
        return {"status": "ok", "model": self.server.nex.model_id}

    def _system_one(self, query):
        body = self._read_json()
        debug = query.get("debug", [""])[-1].lower() in ("1", "true")
        try:
            unknown = sorted(set(body) - set(BODY_FIELDS))
            if unknown:
                raise ValidationError(
                    unknown[0], f"unknown field(s) {', '.join(unknown)}. The body accepts state, model, and questions."
                )
            model = body.get("model")
            if model is not None and (not isinstance(model, str) or not model.strip()):
                raise ValidationError("model", "must be a non-empty string")
            if model is not None and len(model) > MAX_MODEL:
                raise ValidationError("model", f"must be at most {MAX_MODEL} characters")
            response = self.server.nex.system_one(body.get("state"), body.get("questions"), model)
        except ValidationError as e:
            raise ApiError(422, "validation_error", e.message, e.field) from None
        except ContextOverflowError as e:
            raise ApiError(422, "context_overflow", str(e), "state") from None
        except ModelNotFoundError as e:
            raise ApiError(422, "model_not_found", str(e), "model") from None
        except BackendError as e:
            raise ApiError(502, "backend_error", str(e)) from None
        return response.to_dict(include_diagnostics=debug)

    def _models(self):
        nex = self.server.nex
        models = [{"name": "nex-latest", "points_to": nex.model_id, "description": "Alias for the default Nex model."}]
        try:
            names = list(nex.backend.list_models())
        except Exception as e:
            # The alias is still useful when the backend cannot list its models.
            sys.stderr.write(f"nex: could not list backend models: {e}\n")
            names = []
        for name in names:
            models.append(
                {
                    "name": f"nex-{__version__}+{name}",
                    "backend_model": name,
                    "description": f"Nex {__version__} on backend model {name}.",
                }
            )
        return {"models": models}

    def _check_host(self):
        """On a loopback bind, refuse DNS rebinding and cross-site browser
        requests, the way Ollama does. A non-loopback bind was the
        operator's choice, so it is left open."""
        server = self.server
        if not server.loopback:
            return
        # Browsers always send Host. A request without one is not a
        # rebinding attempt, so only the values present are checked.
        for value in self.headers.get_all("Host") or []:
            name = _host_name(value)
            if name is None or not (_is_local(name) or name in server.bound_names):
                raise ApiError(
                    403,
                    "forbidden",
                    "Host header not allowed, a server bound to loopback only answers localhost, "
                    "loopback IPs, and its bound host",
                )
        # Simple cross-origin POSTs, text/plain for one, skip CORS preflight,
        # so the Origin is checked even though no CORS headers are sent.
        for value in self.headers.get_all("Origin") or []:
            name = _origin_name(value)
            if name is None or not _is_local(name):
                raise ApiError(
                    403,
                    "forbidden",
                    "cross-origin request refused, a server bound to loopback only answers pages "
                    "served from localhost or a loopback IP",
                )

    def _check_auth(self):
        key = (os.environ.get("NEX_API_KEY") or "").strip()
        if not key:
            return
        scheme, _, token = self.headers.get("Authorization", "").partition(" ")
        given = token.strip().encode("utf-8", "surrogateescape")
        if scheme.lower() != "bearer" or not hmac.compare_digest(given, key.encode("utf-8", "surrogateescape")):
            raise ApiError(
                401,
                "unauthorized",
                "missing or invalid API key, send Authorization: Bearer <NEX_API_KEY>",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def _read_json(self):
        if "Transfer-Encoding" in self.headers:
            raise ApiError(
                411, "invalid_request", "send the body with a Content-Length header, chunked bodies are not supported"
            )
        length = self._pending_body()
        if length is None:
            raise ApiError(400, "invalid_request", "invalid Content-Length header")
        if length > MAX_BODY:
            raise ApiError(413, "payload_too_large", TOO_LARGE)
        try:
            raw = self.rfile.read(length) if length else b""
        except OSError:
            self._close = True
            raise ApiError(400, "invalid_request", "could not read the request body") from None
        self._body_read = True
        if len(raw) < length:
            self._close = True
            raise ApiError(400, "invalid_request", "request body is shorter than its Content-Length")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ApiError(400, "invalid_request", f"request body is not valid JSON: {e}") from None
        except (ValueError, RecursionError):
            raise ApiError(400, "invalid_request", "request body is not valid JSON") from None
        if not isinstance(body, dict):
            raise ApiError(400, "invalid_request", "request body must be a JSON object")
        return body

    def _pending_body(self):
        """Bytes of request body not read yet, or None when the length is
        unknown or malformed."""
        if self._body_read:
            return 0
        if "Transfer-Encoding" in self.headers:
            return None
        lengths = self._content_lengths()
        if len(lengths) > 1:
            return None
        raw = next(iter(lengths), "") or "0"
        return int(raw) if raw.isascii() and raw.isdigit() else None

    def _content_lengths(self):
        # Identical duplicates collapse to one value and stay accepted.
        return {value.strip() for value in self.headers.get_all("Content-Length") or []}

    def _finish_body(self):
        # Reading an unused body keeps the connection usable for the next
        # request. A body too large or of unknown length closes it instead.
        if self._close:
            return
        length = self._pending_body()
        if length is None or length > MAX_BODY:
            self._close = True
        elif length:
            try:
                self.rfile.read(length)
                self._body_read = True
            except OSError:
                self._close = True

    def _send(self, status, body, headers):
        self._finish_body()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, value)
            if self._close:
                self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def send_error(self, code, message=None, explain=None):
        # http.server calls this for malformed requests and unknown methods,
        # and its own reply would be HTML.
        code = int(code)
        kind = STATUS_TYPES.get(code, "not_implemented" if code == 501 else "http_error")
        self._close = True
        self._send(code, _encode(_error(kind, message or self.responses.get(code, ("error",))[0])), {})

    def handle_expect_100(self):
        # Refuse an oversized upload before the client starts sending it.
        length = self._pending_body()
        if length is not None and length > MAX_BODY:
            self._close = True
            self._send(413, _encode(_error("payload_too_large", TOO_LARGE)), {})
            return False
        return super().handle_expect_100()

    def log_request(self, code="-", size="-"):
        if not self.server.log_requests:
            return
        elapsed = f" {(time.perf_counter() - self._started) * 1000:.0f}ms" if self._started else ""
        status = int(code) if isinstance(code, int) else code
        line = f"{self.address_string()} {self.command or '-'} {getattr(self, 'path', '-')} {status}{elapsed}"
        sys.stderr.write(_printable(line) + "\n")

    def log_message(self, format, *args):
        if self.server.log_requests:
            sys.stderr.write(_printable(f"{self.address_string()} {format % args}") + "\n")


class NexHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server holding the ``Nex`` instance it serves."""

    request_queue_size = 64

    def __init__(self, address, nex, log_requests=True):
        self.nex = nex
        self.log_requests = log_requests
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, NexRequestHandler)
        # Both the host as given and the address it resolved to count as the
        # bound host, so a name from /etc/hosts that points at 127.0.0.1
        # still passes the Host check.
        self.bound_names = {_bind_name(address[0]), _bind_name(self.server_address[0])} - {""}
        self.loopback = any(_is_local(name) for name in self.bound_names)

    def server_bind(self):
        # HTTPServer.server_bind resolves the host with getfqdn, which can
        # stall for seconds on a machine with slow reverse DNS. Nothing here
        # needs the name.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def make_server(nex, host="127.0.0.1", port=8787, log_requests=True):
    """Bind a server for ``nex`` without starting it. ``port=0`` picks a free
    port, read it back from ``server.server_address``."""
    return NexHTTPServer((host, port), nex, log_requests)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def serve(nex, host="127.0.0.1", port=8787):
    """Serve ``nex`` until Ctrl-C or SIGTERM, then close the socket."""
    server = make_server(nex, host, port)
    bound_host, bound_port = server.server_address[:2]
    shown = f"[{bound_host}]" if ":" in bound_host else bound_host
    print(f"Nex serving {nex.model_id} at http://{shown}:{bound_port} (Ctrl-C to stop)", flush=True)
    if not server.loopback and not (os.environ.get("NEX_API_KEY") or "").strip():
        print(
            f"nex: warning: {shown} is not a loopback address and NEX_API_KEY is unset, "
            "anyone who can reach it can use the API",
            file=sys.stderr,
            flush=True,
        )
    try:
        previous = signal.signal(signal.SIGTERM, _interrupt)
    except ValueError:
        # signal.signal only works in the main thread.
        previous = False
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if previous is not False:
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)
    print("Nex server stopped.", flush=True)
