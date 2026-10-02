"""Backends supply next-token log-probabilities for a chat prompt.

Nex needs exactly one thing from a language model: the distribution over the
first token of the reply. ``OllamaBackend`` gets it from a local Ollama server
in one forward pass (``num_predict: 1``). ``FakeBackend`` returns scripted
distributions for tests and offline demos.
"""

import http.client
import ipaddress
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

DEFAULT_BACKEND_MODEL = "qwen3.5:9b"
DEFAULT_NUM_CTX = 8192
# Ollama's ceiling for top_logprobs.
TOP_LOGPROBS = 20
# Ollama passes llama.cpp's overflow error through, as a type and as text.
OVERFLOW_TYPE = "exceed_context_size_error"
OVERFLOW_TEXT = re.compile(
    r"(?:request \((\d+) tokens\) )?exceeds the available context size(?: \((\d+) tokens\))?", re.IGNORECASE
)


class BackendError(RuntimeError):
    """The backend could not produce a distribution.

    ``status`` is the HTTP status of Ollama's reply, or None when there was no
    usable reply.
    """

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class ContextOverflowError(BackendError):
    """The prompt does not fit the backend's context window."""


class ModelNotFoundError(BackendError):
    """The backend does not have the requested model."""


@dataclass(frozen=True)
class NextToken:
    """Top candidates for the first reply token, as ``(token, logprob)``."""

    top_logprobs: list
    prompt_tokens: int = 0
    cached_tokens: int = 0


def _normalize_host(host):
    """Base URL for an ``OLLAMA_HOST`` value, read the way the ollama CLI reads it.

    Without a scheme the scheme is http and a missing or invalid port becomes
    11434. With a scheme, a missing or invalid port is left to the scheme.
    """
    host = host.strip().strip("\"'").strip()
    scheme, sep, rest = host.partition("://")
    if not sep:
        scheme, rest = "http", host
    # A query or fragment means nothing for an API base URL, and urllib never
    # sends userinfo, so both are dropped, as the ollama CLI drops them.
    rest = re.split(r"[?#]", rest, maxsplit=1)[0]
    hostport, _, path = rest.partition("/")
    hostport = hostport.rpartition("@")[2]
    name, port = _split_host_port(hostport)
    # The length check keeps int() away from absurdly long digit strings.
    if not (port.isascii() and port.isdigit() and len(port) <= 5 and int(port) <= 65535):
        port = "" if sep else "11434"
    name = _client_address(name)
    netloc = f"[{name}]" if ":" in name else name
    if port:
        netloc += ":" + port
    path = path.rstrip("/")
    return f"{scheme.lower()}://{netloc}" + (f"/{path}" if path else "")


def _split_host_port(hostport):
    """``(host, port)`` without brackets. A bare IPv6 address has no port."""
    if hostport.startswith("["):
        name, _, tail = hostport[1:].partition("]")
        return name, tail[1:] if tail.startswith(":") else ""
    if hostport.count(":") == 1:
        name, _, port = hostport.partition(":")
        return name, port
    return hostport, ""


def _client_address(name):
    """Loopback for an empty host or a bind-all address such as 0.0.0.0.

    OLLAMA_HOST is often the server's bind address, which is not a usable
    client target.
    """
    if not name:
        return "127.0.0.1"
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        return name
    if ip.is_unspecified:
        return "::1" if ip.version == 6 else "127.0.0.1"
    return name


def _is_loopback(url):
    try:
        name = (urllib.parse.urlsplit(url).hostname or "").rstrip(".")
    except ValueError:
        return False
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _opener(url):
    """URL opener that never sends loopback traffic through a proxy.

    urllib applies HTTP_PROXY even to 127.0.0.1, which breaks Nex behind a
    corporate proxy and hands every prompt to that proxy. Remote hosts keep
    the usual proxy settings.
    """
    if _is_loopback(url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


class OllamaBackend:
    """Reads next-token logprobs from Ollama's ``/api/chat``.

    ``num_ctx`` is sent with every request. Keep it fixed, because changing it
    makes Ollama reload the model. Some models silently truncate a prompt that
    exceeds it, so every request sends ``truncate: false`` and an overflow
    raises ``ContextOverflowError``. A prompt that still fills the window is
    treated as truncated too, for servers that ignore the flag.
    """

    def __init__(self, model=None, host=None, num_ctx=None, timeout=120.0, keep_alive="30m", retries=2):
        self.model = model or os.environ.get("NEX_BACKEND_MODEL", DEFAULT_BACKEND_MODEL)
        self.host = _normalize_host(host or os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434")
        self.num_ctx = int(num_ctx or os.environ.get("NEX_NUM_CTX", DEFAULT_NUM_CTX))
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.retries = retries
        self._opener = _opener(self.host)
        # Thinking models must be told not to think, or the first token is
        # "<think>". Models without a thinking mode may reject the flag, so it
        # is dropped once a request without it has worked.
        self._send_think = True

    def with_model(self, model):
        """Same server and settings, different model."""
        return OllamaBackend(model, self.host, self.num_ctx, self.timeout, self.keep_alive, self.retries)

    def next_token(self, messages):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "keep_alive": self.keep_alive,
            "truncate": False,
            "options": {"num_predict": 1, "temperature": 0, "num_ctx": self.num_ctx},
        }
        if self._send_think:
            body["think"] = False
        try:
            data = self._post("/api/chat", body)
        except BackendError as e:
            # Check this request's body, not the flag. A concurrent request may
            # have cleared the flag after this one was sent with think.
            if "think" not in body or not _refuses_think(e):
                raise
            body.pop("think")
            data = self._post("/api/chat", body)
            self._send_think = False

        positions = data.get("logprobs") or []
        if not positions or not positions[0].get("top_logprobs"):
            raise BackendError(
                f"Ollama returned no logprobs for {self.model!r}. Nex needs an Ollama version with logprobs support."
            )
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        if prompt_tokens >= self.num_ctx - 1:
            raise ContextOverflowError(
                f"prompt reached the {self.num_ctx}-token context window and was truncated. "
                "Shorten the state or raise NEX_NUM_CTX."
            )
        return NextToken(
            top_logprobs=[(t["token"], float(t["logprob"])) for t in positions[0]["top_logprobs"]],
            prompt_tokens=prompt_tokens,
            cached_tokens=int(data.get("prompt_eval_cached_count") or 0),
        )

    def list_models(self):
        """Names of the models the Ollama server has pulled."""
        return [m["name"] for m in self._request("GET", "/api/tags").get("models", [])]

    def _post(self, path, body):
        return self._request("POST", path, body)

    def _request(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.host + path, data=data, method=method, headers={"Content-Type": "application/json"}
                )
                with self._opener.open(req, timeout=self.timeout) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as e:
                error = self._http_error(e)
                # 4xx means the request itself is wrong, so retrying will not help.
                if e.code < 500:
                    raise error from None
                last_error = error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_error = BackendError(f"cannot reach Ollama at {self.host}: {getattr(e, 'reason', e)}")
            except http.client.HTTPException as e:
                # The reply was not HTTP, or it was cut off halfway.
                last_error = BackendError(f"Ollama at {self.host} sent a broken HTTP reply ({type(e).__name__})")
            except ValueError as e:
                # urllib rejects some hosts only on use, such as an unbracketed
                # IPv6 address with a port. Retrying cannot fix the address.
                raise BackendError(f"cannot reach Ollama at {self.host}: {e}") from None
            else:
                return self._decode(raw)
            if attempt < self.retries:
                time.sleep(0.5 * 2**attempt)
        raise last_error

    def _decode(self, raw):
        try:
            reply = json.loads(raw)
        except ValueError:
            raise BackendError(f"Ollama at {self.host} sent a reply that is not JSON") from None
        if not isinstance(reply, dict):
            raise BackendError(f"Ollama at {self.host} sent a reply that is not a JSON object")
        return reply

    def _http_error(self, error):
        """The ``BackendError`` for an HTTP error reply."""
        message, detail = _error_detail(error)
        status = error.code
        if status == 400 and _is_overflow(message, detail):
            return self._overflow_error(message, detail)
        if status == 404 and "model" in message.lower() and "not found" in message.lower():
            return ModelNotFoundError(
                f'Ollama has no model "{self.model}". Pull it with: ollama pull {self.model}', status
            )
        return BackendError(f"Ollama {status}: {message}", status)

    def _overflow_error(self, message, detail):
        prompt, window = detail.get("n_prompt_tokens"), detail.get("n_ctx")
        match = OVERFLOW_TEXT.search(message)
        if match:
            prompt, window = prompt or match.group(1), window or match.group(2)
        # Ollama may cap num_ctx below the requested size, so its own count wins.
        window = window or self.num_ctx
        if prompt:
            text = f"prompt is {prompt} tokens but the context window is {window}."
        else:
            text = f"prompt does not fit the {window}-token context window."
        return ContextOverflowError(text + " Shorten the state or raise NEX_NUM_CTX.", 400)


def _is_overflow(message, detail):
    return detail.get("type") == OVERFLOW_TYPE or OVERFLOW_TYPE in message or bool(OVERFLOW_TEXT.search(message))


def _refuses_think(error):
    return error.status == 400 and "does not support thinking" in str(error)


def _error_detail(error):
    """``(message, detail)`` for an HTTP error reply from Ollama.

    Ollama passes some llama.cpp errors through as JSON text inside
    ``error``. ``detail`` is that inner error object, or an empty dict.
    """
    try:
        payload = json.loads(error.read())
    except (ValueError, AttributeError, OSError, http.client.HTTPException):
        payload = None
    finally:
        # The error holds the open response. Without this, every retried 5xx
        # leaves a socket for the garbage collector.
        error.close()
    value = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(value, str):
        try:
            inner = json.loads(value)
        except ValueError:
            return value, {}
        if not isinstance(inner, (dict, list)):
            return value, {}
        value = inner
    if isinstance(value, dict):
        value = value.get("error", value)
    if isinstance(value, dict) and isinstance(value.get("message"), str):
        return value["message"], value
    if isinstance(value, str):
        return value, {}
    # Nested JSON without a usable message is not shown to the user.
    return str(error.reason), {}


class FakeBackend:
    """Scripted backend for tests and offline demos.

    ``responder(messages)`` returns a list of ``(token, logprob)`` pairs, or a
    ``NextToken``. Every call is recorded in ``calls``.
    """

    def __init__(self, responder, model="fake"):
        self.responder = responder
        self.model = model
        self.calls = []

    def with_model(self, model):
        return FakeBackend(self.responder, model)

    def next_token(self, messages):
        self.calls.append(messages)
        out = self.responder(messages)
        if isinstance(out, NextToken):
            return out
        return NextToken(top_logprobs=list(out), prompt_tokens=sum(len(m["content"]) // 4 for m in messages))

    def list_models(self):
        return [self.model]
