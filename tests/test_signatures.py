"""
Tests for the optional HTTP Message Signatures (RFC 9421) support,
including content digests (RFC 9530), sync/async sends, streaming body
digests, re-signing after redirects and response verification
diagnostics.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
import typing

import pytest

import httpx

# Shared key material / identity used across the tests.
KEY = b"super-secret-key"
KEYID = "key-1"
ALG = "hmac-sha256"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _sf(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def make_signer(**kwargs: typing.Any) -> httpx.MessageSigner:
    kwargs.setdefault(
        "key_resolver", httpx.StaticKeyResolver({KEYID: KEY})
    )
    return httpx.MessageSigner(**kwargs)


# ---------------------------------------------------------------------------
# Independent, test-local server-side implementation.
# This deliberately does not import httpx._signatures helpers, so the test
# verifies interoperability rather than internal round-tripping.
# ---------------------------------------------------------------------------


def _server_value(component: str, message: httpx.Request | httpx.Response) -> bytes:
    if component == "@method":
        return message.method.encode("ascii")
    if component == "@target-uri":
        return str(message.url).encode("ascii")
    if component == "@authority":
        return message.url.netloc
    if component == "@scheme":
        return message.url.scheme.encode("ascii")
    if component == "@path":
        return message.url.raw_path.split(b"?", 1)[0]
    if component == "@query":
        return b"?" + message.url.query
    if component == "@status":
        return str(message.status_code).encode("ascii")
    return ", ".join(message.headers.get_list(component)).encode("utf-8")


def _server_base(
    covered: list[str], message: httpx.Request | httpx.Response
) -> bytes:
    return b"\n".join(
        b'"' + c.encode("ascii") + b'": ' + _server_value(c, message)
        for c in covered
    )


def parse_signed_message(
    message: httpx.Request | httpx.Response,
) -> tuple[str, list[str], dict[str, str]]:
    signature_input = message.headers["Signature-Input"]
    label, rest = signature_input.split("=", 1)
    inner, _, params_str = rest[1:].partition(")")
    covered = [tok for tok in inner.replace('"', "").split()]
    params: dict[str, str] = {}
    for param in params_str.split(";"):
        if "=" in param:
            key, value = param.split("=", 1)
            params[key] = value.strip().strip('"')
    return label.strip(), covered, params


def server_verify_request(request: httpx.Request) -> list[str]:
    """Independently verify the request signature using the shared key."""
    label, covered, params = parse_signed_message(request)
    signature = base64.b64decode(
        request.headers["Signature"].split("=:", 1)[1].rstrip(":")
    )
    expected = hmac.new(KEY, _server_base(covered, request), hashlib.sha256).digest()
    assert hmac.compare_digest(expected, signature), "request signature mismatch"
    assert params["keyid"] == KEYID
    assert params["alg"] == ALG
    return covered


def signed_response(
    request: httpx.Request,
    *,
    body: bytes = b'{"ok": true}',
    status: int = 200,
    covered: list[str] | None = None,
    nonce: str | None = None,
    expires_at: int | None = None,
    created_at: int | None = None,
    tamper_signature: bool = False,
    bad_digest: bool = False,
) -> httpx.Response:
    """A test server that independently produces a signed response."""
    covered = covered or ["@status", "content-digest"]
    now = int(time.time())
    created_at = now if created_at is None else created_at

    digest = hashlib.sha256(body).digest()
    digest_header = f"sha-256=:{_b64(digest)}:"
    if bad_digest:
        digest_header = f"sha-256=:{_b64(b'')}:"

    def value(component: str) -> bytes:
        return _server_value(component, _ResponseView(status, digest_header))

    signature_base = b"\n".join(
        b'"' + c.encode("ascii") + b'": ' + value(c) for c in covered
    )
    parameters = (
        f";keyid={_sf(KEYID)};alg={_sf(ALG)};created={created_at}"
    )
    if expires_at is not None:
        parameters += f";expires={expires_at}"
    if nonce is not None:
        parameters += f";nonce={_sf(nonce)}"

    signature = hmac.new(KEY, signature_base, hashlib.sha256).digest()
    if tamper_signature:
        signature = signature[:-1] + bytes([signature[-1] ^ 0x01])

    headers = {
        "Content-Digest": digest_header,
        "Signature-Input": (
            "sig1=(" + " ".join(_sf(c) for c in covered) + ")" + parameters
        ),
        "Signature": f"sig1=:{_b64(signature)}:",
    }
    return httpx.Response(status_code=status, headers=headers, content=body)


class _ResponseView:
    """Minimal view exposing the bits '_server_value' needs for responses."""

    def __init__(self, status_code: int, digest_header: str) -> None:
        self.status_code = status_code
        self._headers = httpx.Headers({"content-digest": digest_header})

    @property
    def headers(self) -> httpx.Headers:
        return self._headers


# ---------------------------------------------------------------------------
# Basic request signing
# ---------------------------------------------------------------------------


def test_signs_request_with_configured_components() -> None:
    signer = make_signer(
        components=("@method", "@authority", "@path", "@query", "content-digest"),
        use_nonce=True,
        expires_in=300,
    )
    request = httpx.Request(
        "POST", "https://example.com/path?a=1", json={"hello": "world"}
    )
    signer.sign_request(request)

    assert "Signature" in request.headers
    signature_input = request.headers["Signature-Input"]
    assert signature_input.startswith(
        'sig1=("@method" "@authority" "@path" "@query" "content-digest")'
    )
    assert 'keyid="key-1"' in signature_input
    assert 'alg="hmac-sha256"' in signature_input
    assert "created=" in signature_input
    assert "expires=" in signature_input
    assert "nonce=" in signature_input
    assert "Content-Digest" in request.headers


def test_signature_is_independently_valid() -> None:
    signer = make_signer(
        components=(
            "@method",
            "@target-uri",
            "@authority",
            "@scheme",
            "@path",
            "@query",
            "x-custom",
            "content-digest",
        )
    )
    request = httpx.Request(
        "POST",
        "https://example.com/path?x=1",
        headers={"x-custom": "abc"},
        content=b"payload",
    )
    signer.sign_request(request)
    server_verify_request(request)


def test_get_request_without_body_skips_content_digest() -> None:
    signer = make_signer(
        components=("@method", "@path", "content-digest")
    )
    request = httpx.Request("GET", "https://example.org/path")
    signer.sign_request(request)
    assert "Content-Digest" not in request.headers
    _, covered, _ = parse_signed_message(request)
    assert "content-digest" not in covered


def test_event_hook_headers_are_covered() -> None:
    signer = make_signer(components=("@method", "@path", "x-added"))

    def add_header(request: httpx.Request) -> None:
        request.headers["x-added"] = "from-hook"

    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        server_verify_request(request)
        return httpx.Response(200)

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [add_header]},
        message_signature=signer,
    )
    client.get("https://example.org/")


def test_missing_keyid_raises_signing_error_with_request() -> None:
    signer = httpx.MessageSigner(key_resolver=httpx.KeyResolver())
    request = httpx.Request("GET", "https://example.org/")
    with pytest.raises(httpx.SignatureSigningError) as exc_info:
        signer.sign_request(request)
    assert exc_info.value.request is request


def test_unknown_keyid_raises_signing_error() -> None:
    signer = make_signer(keyid="nope")
    request = httpx.Request("GET", "https://example.org/")
    with pytest.raises(httpx.SignatureSigningError):
        signer.sign_request(request)


def test_static_resolver_bytes_shortcut() -> None:
    resolver = httpx.StaticKeyResolver(KEY)
    assert resolver.default_keyid == "default"
    request = httpx.Request("GET", "https://example.org/")
    signer = httpx.MessageSigner(key_resolver=resolver)
    signer.sign_request(request)
    assert "Signature" in request.headers


# ---------------------------------------------------------------------------
# Sync / async end-to-end sending
# ---------------------------------------------------------------------------


def test_sync_send_signed_request() -> None:
    signer = make_signer(
        components=("@method", "@authority", "@path", "content-digest")
    )
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        server_verify_request(request)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    response = client.post(
        "https://example.com/data", json={"name": "alice"}
    )
    assert response.status_code == 200
    assert seen["request"].content == b'{"name":"alice"}'


@pytest.mark.anyio
async def test_async_send_signed_request() -> None:
    signer = make_signer(
        components=("@method", "@authority", "@path", "content-digest")
    )
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        server_verify_request(request)
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), message_signature=signer
    ) as client:
        response = await client.post(
            "https://example.com/data", json={"name": "alice"}
        )
        assert response.status_code == 200
        assert seen["request"].content == b'{"name":"alice"}'


@pytest.mark.anyio
async def test_async_client_uses_async_key_resolver() -> None:
    class AsyncOnlyResolver(httpx.KeyResolver):
        default_keyid = KEYID

        def resolve_key(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            raise AssertionError("sync resolver should not be used")

        async def aresolve_key(
            self, keyid: str, request: httpx.Request, for_response: bool = False
        ) -> typing.Any:
            return KEY

    signer = httpx.MessageSigner(key_resolver=AsyncOnlyResolver())

    def handler(request: httpx.Request) -> httpx.Response:
        server_verify_request(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), message_signature=signer
    ) as client:
        response = await client.get("https://example.org/")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Streaming body digests
# ---------------------------------------------------------------------------


def test_streaming_body_digest_sync() -> None:
    signer = make_signer(
        components=("@method", "@path", "content-digest")
    )
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        covered = server_verify_request(request)
        assert covered == ["@method", "@path", "content-digest"]
        return httpx.Response(200)

    def body_chunks() -> typing.Iterator[bytes]:
        yield b"chunk-1|"
        yield b"chunk-2|"
        yield b"chunk-3"

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    response = client.post("https://example.org/upload", content=body_chunks())
    assert response.status_code == 200

    body = seen["request"].content
    assert body == b"chunk-1|chunk-2|chunk-3"
    expected = hashlib.sha256(body).digest()
    assert seen["request"].headers["Content-Digest"] == (
        f"sha-256=:{_b64(expected)}:"
    )


@pytest.mark.anyio
async def test_streaming_body_digest_async() -> None:
    signer = make_signer(
        components=("@method", "@path", "content-digest")
    )
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        server_verify_request(request)
        return httpx.Response(200)

    async def body_chunks() -> typing.AsyncIterator[bytes]:
        yield b"achunk-1|"
        yield b"achunk-2"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), message_signature=signer
    ) as client:
        response = await client.post(
            "https://example.org/upload", content=body_chunks()
        )
        assert response.status_code == 200

    body = seen["request"].content
    assert body == b"achunk-1|achunk-2"
    expected = hashlib.sha256(body).digest()
    assert seen["request"].headers["Content-Digest"] == (
        f"sha-256=:{_b64(expected)}:"
    )


# ---------------------------------------------------------------------------
# Re-signing on redirects
# ---------------------------------------------------------------------------


def test_redirect_is_re_signed_for_new_url() -> None:
    signer = make_signer(
        components=("@method", "@authority", "@path", "content-digest")
    )
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        server_verify_request(request)
        if len(calls) == 1:
            return httpx.Response(
                307, headers={"location": "https://example.com/new-place"}
            )
        return httpx.Response(200, text="done")

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    response = client.post(
        "https://example.com/start",
        content=b"redirect-body",
        follow_redirects=True,
    )
    assert response.status_code == 200

    assert len(calls) == 2
    hop1, hop2 = calls
    assert hop1.url.path == "/start"
    assert hop2.url.path == "/new-place"

    # Each hop must carry its own signature, bound to its own URL.
    assert hop1.headers["Signature"] != hop2.headers["Signature"]
    _, covered1, _ = parse_signed_message(hop1)
    _, covered2, _ = parse_signed_message(hop2)
    assert "@path" in covered1 and "@path" in covered2

    def _sent_signature(request: httpx.Request) -> bytes:
        return base64.b64decode(
            request.headers["Signature"].split("=:", 1)[1].rstrip(":")
        )

    # The hop-2 signature must be the HMAC of the base bound to /new-place.
    expected_hop2 = hmac.new(
        KEY, _server_base(covered2, hop2), hashlib.sha256
    ).digest()
    assert _sent_signature(hop2) == expected_hop2
    # And must not verify against the hop-1 base (URL /start).
    expected_hop1 = hmac.new(
        KEY, _server_base(covered1, hop1), hashlib.sha256
    ).digest()
    assert _sent_signature(hop2) != expected_hop1


def test_redirect_post_to_get_strips_body_digest() -> None:
    signer = make_signer(
        components=("@method", "@path", "content-digest")
    )
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        server_verify_request(request)
        if len(calls) == 1:
            return httpx.Response(
                302, headers={"location": "https://example.com/other"}
            )
        return httpx.Response(200)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    client.post(
        "https://example.com/from",
        content=b"body-bytes",
        follow_redirects=True,
    )

    hop2 = calls[1]
    assert hop2.method == "GET"
    assert "Content-Digest" not in hop2.headers
    _, covered, _ = parse_signed_message(hop2)
    assert "content-digest" not in covered


# ---------------------------------------------------------------------------
# Response signature verification
# ---------------------------------------------------------------------------


def test_verify_signed_response() -> None:
    signer = make_signer(verify_response=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return signed_response(request)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    response = client.get("https://example.org/")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_tampered_response_signature_rejected_and_released() -> None:
    signer = make_signer(verify_response=True)
    produced: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = signed_response(request, tamper_signature=True)
        produced.append(response)
        return response

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    with pytest.raises(httpx.InvalidSignature) as exc_info:
        client.get("https://example.org/")
    assert exc_info.value.request.url == "https://example.org/"
    # The connection/resource must be released on verification failure.
    assert produced[0].is_closed


def test_expired_signature_diagnostic() -> None:
    signer = make_signer(verify_response=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return signed_response(request, expires_at=int(time.time()) - 600)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    with pytest.raises(httpx.SignatureExpired) as exc_info:
        client.get("https://example.org/")
    assert exc_info.value.keyid == KEYID
    assert exc_info.value.expires is not None
    assert exc_info.value.now is not None
    assert exc_info.value.now > exc_info.value.expires


def test_not_yet_valid_signature_diagnostic() -> None:
    signer = make_signer(verify_response=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return signed_response(request, created_at=int(time.time()) + 600)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    with pytest.raises(httpx.SignatureNotYetValid) as exc_info:
        client.get("https://example.org/")
    assert exc_info.value.keyid == KEYID
    assert exc_info.value.created is not None


def test_max_age_diagnostic() -> None:
    signer = make_signer(verify_response=True, max_age=60)

    def handler(request: httpx.Request) -> httpx.Response:
        return signed_response(request, created_at=int(time.time()) - 300)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    with pytest.raises(httpx.SignatureExpired):
        client.get("https://example.org/")


def test_nonce_replay_diagnostic() -> None:
    signer = make_signer(verify_response=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return signed_response(request, nonce="fixed-nonce-value")

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    client.get("https://example.org/")
    with pytest.raises(httpx.DuplicateNonce) as exc_info:
        client.get("https://example.org/")
    assert exc_info.value.nonce == "fixed-nonce-value"
    assert exc_info.value.keyid == KEYID


def test_response_digest_mismatch_raises() -> None:
    signer = make_signer(verify_response=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return signed_response(request, bad_digest=True)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    with pytest.raises(httpx.ContentDigestError):
        client.get("https://example.org/")


def test_unsigned_response_accepted_by_default() -> None:
    signer = make_signer(verify_response=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="unsigned")

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    response = client.get("https://example.org/")
    assert response.text == "unsigned"


def test_unsigned_response_rejected_when_required() -> None:
    signer = make_signer(
        verify_response=True, require_response_signature=True
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="unsigned")

    client = httpx.Client(
        transport=httpx.MockTransport(handler), message_signature=signer
    )
    with pytest.raises(httpx.InvalidSignature):
        client.get("https://example.org/")


def test_streaming_response_digest_verified_while_consuming() -> None:
    signer = make_signer(verify_response=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return signed_response(request, body=b"streamed-body-bytes")

    # Build the response around an explicit stream so it is genuinely
    # unread when verification wraps it (as for a real network response).
    def streaming_handler(request: httpx.Request) -> httpx.Response:
        pre = signed_response(
            request, body=b"streamed-body-bytes"
        )
        return httpx.Response(
            status_code=200, headers=pre.headers, stream=pre.stream
        )

    client = httpx.Client(
        transport=httpx.MockTransport(streaming_handler),
        message_signature=signer,
    )
    with client.stream("GET", "https://example.org/") as response:
        chunks = list(response.iter_raw())
    assert b"".join(chunks) == b"streamed-body-bytes"


@pytest.mark.anyio
async def test_async_verify_signed_response() -> None:
    class AsyncOnlyResolver(httpx.KeyResolver):
        default_keyid = KEYID

        def resolve_key(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            raise AssertionError("sync resolver should not be used")

        async def aresolve_key(
            self, keyid: str, request: httpx.Request, for_response: bool = False
        ) -> typing.Any:
            return KEY

    signer = httpx.MessageSigner(
        key_resolver=AsyncOnlyResolver(), verify_response=True
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return signed_response(request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), message_signature=signer
    ) as client:
        response = await client.get("https://example.org/")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Disabled-by-default semantics
# ---------------------------------------------------------------------------


def test_disabled_client_sends_no_signature_headers() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert client._message_signature is None
    client.post("https://example.org/data", json={"a": 1})

    request = seen["request"]
    assert "Signature" not in request.headers
    assert "Signature-Input" not in request.headers
    assert "Content-Digest" not in request.headers


class _ObservingTransport(httpx.BaseTransport):
    """Records whether the request body had been eagerly read."""

    def __init__(self) -> None:
        self.read_before_transport: bool | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.read_before_transport = hasattr(request, "_content")
        b"".join(request.stream)  # Drain the body like a real transport.
        return httpx.Response(200)


def test_disabled_client_does_not_eagerly_read_streaming_body() -> None:
    transport = _ObservingTransport()
    client = httpx.Client(transport=transport)

    def chunks() -> typing.Iterator[bytes]:
        yield b"a"
        yield b"b"

    client.post("https://example.org/", content=chunks())
    assert transport.read_before_transport is False


def test_enabled_client_reads_streaming_body_for_digest() -> None:
    transport = _ObservingTransport()
    signer = make_signer(components=("@method", "content-digest"))
    client = httpx.Client(transport=transport, message_signature=signer)

    def chunks() -> typing.Iterator[bytes]:
        yield b"a"
        yield b"b"

    client.post("https://example.org/", content=chunks())
    assert transport.read_before_transport is True
