"""
Tests for the optional per-origin TLS policy resolver.

Covers CA / client-certificate / ALPN / hostname selection, pool key
equivalence, policy cache invalidation and generation boundaries, HTTP/2
multiplexing, HTTP proxy CONNECT tunnelling, sync and async concurrency,
resolution failure cleanup and cross-origin redirects.
"""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
import socket
import ssl
import tempfile
import threading
import time
import types
import typing

import anyio
import pytest
import trustme
from uvicorn.config import Config

import httpx
from tests.conftest import TestServer as _TestServer, serve_in_thread

if typing.TYPE_CHECKING:
    from httpx._transports.default import _AsyncTLSPolicies, _SyncTLSPolicies

# Uvicorn >= 0.53 implements HTTP/2 via `zttp`; the auto protocol performs
# ALPN dispatch between HTTP/2 and HTTP/1.1 on a single TLS port.
H2_HTTP_PROTOCOL = "uvicorn.protocols.http.auto_zttp_impl:AutoZttpProtocol"


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


Scope = dict[str, typing.Any]
Receive = typing.Callable[[], typing.Awaitable[typing.Any]]
Send = typing.Callable[[dict[str, typing.Any]], typing.Awaitable[None]]
ASGIApp = typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


async def hello_app(scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] != "http":
        return
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"Hello, world!"})


async def slow_app(scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] != "http":
        return
    await asyncio.sleep(0.5)
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"slow"})


def redirect_app(location: bytes) -> ASGIApp:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 301,
                "headers": [(b"location", location)],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    return app


class CertBundle:
    def __init__(self, *names: str) -> None:
        self.ca = trustme.CA()
        self.cert = self.ca.issue_cert(*names)

    def write(self, tmp_path: pathlib.Path, tag: str) -> tuple[str, str, str]:
        capath = tmp_path / f"ca-{tag}.pem"
        capath.write_bytes(self.ca.cert_pem.bytes())
        certpath = tmp_path / f"cert-{tag}.pem"
        certpath.write_bytes(self.cert.cert_chain_pems[0].bytes())
        keypath = tmp_path / f"key-{tag}.pem"
        keypath.write_bytes(self.cert.private_key_pem.bytes())
        return str(capath), str(certpath), str(keypath)

    def client_ssl_context(self) -> ssl.SSLContext:
        with tempfile.NamedTemporaryFile("wb", suffix=".pem", delete=False) as f:
            f.write(self.ca.cert_pem.bytes())
            cafile = f.name
        return ssl.create_default_context(cafile=cafile)


@contextlib.contextmanager
def https_server(
    tmp_path: pathlib.Path,
    bundle: CertBundle,
    tag: str,
    app: ASGIApp = hello_app,
    *,
    http: str = "h11",
    require_client_cert: bool = False,
) -> typing.Iterator[_TestServer]:
    capath, certpath, keypath = bundle.write(tmp_path, tag)
    port = free_port()
    kwargs: dict[str, typing.Any] = {}
    if require_client_cert:
        kwargs["ssl_ca_certs"] = capath
        kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
    config = Config(
        app=app,
        host="127.0.0.1",
        port=port,
        loop="asyncio",
        lifespan="off",
        http=http,
        ssl_certfile=certpath,
        ssl_keyfile=keypath,
        log_level="warning",
        # Do not let uvicorn reconfigure process-wide logging.
        log_config=None,
        **kwargs,
    )
    server = _TestServer(config=config)
    generator = serve_in_thread(server)
    started = next(generator)
    try:
        yield started
    finally:
        with contextlib.suppress(StopIteration):
            next(generator)


def policy_for(
    bundle: CertBundle,
    key: typing.Hashable,
    *,
    http2: bool | None = None,
    check_hostname: bool | None = None,
    alpn_protocols: typing.Sequence[str] | None = None,
) -> httpx.TLSPolicy:
    kwargs: dict[str, typing.Any] = {}
    if check_hostname is not None:
        kwargs["check_hostname"] = check_hostname
    return httpx.TLSPolicy.create(
        verify=bundle.client_ssl_context(),
        http2=http2,
        alpn_protocols=alpn_protocols,
        key=key,
        **kwargs,
    )


class MapResolver(httpx.TLSPolicyResolver):
    def __init__(
        self,
        policies: dict[httpx.Origin, httpx.TLSPolicy],
        *,
        fail: bool = False,
    ) -> None:
        super().__init__()
        self.policies = dict(policies)
        self.calls: list[httpx.Origin] = []
        self.fail = fail

    def resolve(self, origin: httpx.Origin) -> httpx.TLSPolicy:
        self.calls.append(origin)
        if self.fail or origin not in self.policies:
            raise RuntimeError(f"no policy configured for {origin}")
        return self.policies[origin]


class AsyncMapResolver(MapResolver):
    def __init__(
        self,
        policies: dict[httpx.Origin, httpx.TLSPolicy],
        *,
        fail: bool = False,
    ) -> None:
        super().__init__(policies, fail=fail)
        self.async_calls: list[httpx.Origin] = []

    async def aresolve(self, origin: httpx.Origin) -> httpx.TLSPolicy:
        self.async_calls.append(origin)
        return self.resolve(origin)


class StaticResolver(httpx.TLSPolicyResolver):
    def __init__(self, policy: httpx.TLSPolicy) -> None:
        super().__init__()
        self.policy = policy
        self.calls = 0

    def resolve(self, origin: httpx.Origin) -> httpx.TLSPolicy:
        self.calls += 1
        return self.policy


def _sync_policies(client: httpx.Client) -> _SyncTLSPolicies:
    transport = typing.cast(httpx.HTTPTransport, client._transport)
    policies = transport._tls
    assert policies is not None
    return policies


def _async_policies(client: httpx.AsyncClient) -> _AsyncTLSPolicies:
    transport = typing.cast(httpx.AsyncHTTPTransport, client._transport)
    policies = transport._tls
    assert policies is not None
    return policies


# ---------------------------------------------------------------------------
# Origin / policy value objects
# ---------------------------------------------------------------------------


def test_origin_from_url():
    assert httpx.Origin.from_url(httpx.URL("https://example.org/path")) == (
        httpx.Origin("https", "example.org", 443)
    )
    assert httpx.Origin.from_url(httpx.URL("http://example.org:8080")) == httpx.Origin(
        "http", "example.org", 8080
    )
    a = httpx.Origin("https", "example.org", 443)
    b = httpx.Origin("https", "example.org", 443)
    assert a == b and hash(a) == hash(b)
    assert a != httpx.Origin("https", "example.org", 444)
    assert str(a) == "https://example.org:443"


def test_policy_requires_ssl_context():
    with pytest.raises(TypeError):
        httpx.TLSPolicy("not-a-context")  # type: ignore[arg-type]


def test_policy_create_check_hostname():
    ctx = httpx.TLSPolicy.create(verify=False, check_hostname=False).ssl_context
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_NONE

    ctx = httpx.TLSPolicy.create(verify=False, check_hostname=True).ssl_context
    assert ctx.check_hostname is True
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_pinned_alpn_context_keeps_list():
    policy = httpx.TLSPolicy.create(verify=False, alpn_protocols=["h2"], key="pinned")
    ctx = policy.ssl_context
    # Simulate httpcore resetting the ALPN list on every handshake.
    ctx.set_alpn_protocols(["http/1.1"])
    # The pinned list survives.
    assert typing.cast(typing.Any, ctx)._pinned_alpn_protocols == ["h2"]


def test_pinned_alpn_requires_created_context():
    with pytest.raises(ValueError, match="alpn_protocols"):
        httpx.TLSPolicy.create(
            verify=ssl.create_default_context(), alpn_protocols=["h2"]
        )


# ---------------------------------------------------------------------------
# Resolution failure: identifiable exception before connecting
# ---------------------------------------------------------------------------


def test_resolution_failure_raises_before_connect():
    resolver = MapResolver({})  # nothing configured
    with httpx.Client(tls_policy=resolver) as client:
        with pytest.raises(httpx.TLSPolicyError) as exc_info:
            client.get("https://unknown-policy.example/")

    exc = exc_info.value
    assert isinstance(exc, httpx.TransportError)
    assert isinstance(exc.__cause__, RuntimeError)
    assert exc.request.url.host == "unknown-policy.example"
    # Nothing was ever connected: no pool was created.
    assert _sync_policies(client)._entries == {}


def test_invalid_resolver_result_raises_policy_error():
    class BadResolver(httpx.TLSPolicyResolver):
        def resolve(self, origin):
            return "not-a-policy"

    transport = httpx.HTTPTransport(tls_policy=BadResolver())
    request = httpx.Request("GET", "https://example.org/")
    with pytest.raises(httpx.TLSPolicyError, match="instead of a TLSPolicy"):
        transport.handle_request(request)


def test_resolution_failure_is_not_cached(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    origin = None
    with https_server(tmp_path, bundle, "recover") as server:
        url = str(server.url)
        origin = httpx.Origin.from_url(httpx.URL(url))
        resolver = MapResolver({origin: policy_for(bundle, "recover")}, fail=True)
        with httpx.Client(tls_policy=resolver) as client:
            with pytest.raises(httpx.TLSPolicyError):
                client.get(url)
            # The failed resolution is retried on the next request.
            resolver.fail = False
            response = client.get(url)
            assert response.status_code == 200
            assert response.text == "Hello, world!"


# ---------------------------------------------------------------------------
# CA / client certificate selection per origin
# ---------------------------------------------------------------------------


def test_policy_selects_ca_per_origin(tmp_path):
    bundle_a = CertBundle("localhost", "127.0.0.1")
    bundle_b = CertBundle("localhost", "127.0.0.1")
    with (
        https_server(tmp_path, bundle_a, "a") as server_a,
        https_server(tmp_path, bundle_b, "b") as server_b,
    ):
        url_a, url_b = str(server_a.url), str(server_b.url)
        policies = {
            httpx.Origin.from_url(httpx.URL(url_a)): policy_for(bundle_a, "a"),
            httpx.Origin.from_url(httpx.URL(url_b)): policy_for(bundle_b, "b"),
        }
        resolver = MapResolver(policies)
        with httpx.Client(tls_policy=resolver) as client:
            assert client.get(url_a).text == "Hello, world!"
            assert client.get(url_b).text == "Hello, world!"
        assert set(resolver.calls) == set(policies)


def test_wrong_ca_fails(tmp_path):
    bundle_server = CertBundle("localhost", "127.0.0.1")
    bundle_other = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle_server, "srv") as server:
        url = str(server.url)
        resolver = StaticResolver(policy_for(bundle_other, "wrong"))
        with httpx.Client(tls_policy=resolver) as client:
            with pytest.raises(httpx.ConnectError):
                client.get(url)


def test_policy_selects_client_certificate(tmp_path):
    # The server requires a client certificate signed by its own CA.
    server_bundle = CertBundle("localhost", "127.0.0.1")
    client_cert = server_bundle.ca.issue_cert("test-client")

    def context_with_client_cert() -> ssl.SSLContext:
        ctx = server_bundle.client_ssl_context()
        with tempfile.NamedTemporaryFile("wb", suffix=".pem", delete=False) as f:
            f.write(client_cert.cert_chain_pems[0].bytes())
            certfile = f.name
        with tempfile.NamedTemporaryFile("wb", suffix=".pem", delete=False) as f:
            f.write(client_cert.private_key_pem.bytes())
            keyfile = f.name
        ctx.load_cert_chain(certfile, keyfile)
        return ctx

    with https_server(
        tmp_path,
        server_bundle,
        "mtls",
        require_client_cert=True,
    ) as server:
        url = str(server.url)

        # A policy that presents a client certificate succeeds.
        good = httpx.TLSPolicy(context_with_client_cert(), key="with-cert")

        # A policy without a client certificate is rejected by the server.
        bad = httpx.TLSPolicy(server_bundle.client_ssl_context(), key="without-cert")

        with httpx.Client(tls_policy=StaticResolver(bad)) as client:
            # The missing client certificate is rejected during the TLS
            # handshake. Depending on timing the server sends a TLS alert
            # (ConnectError), resets the socket (ReadError) or closes the
            # connection (RemoteProtocolError).
            with pytest.raises(httpx.TransportError):
                client.get(url)

        with httpx.Client(tls_policy=StaticResolver(good)) as client:
            assert client.get(url).status_code == 200


def test_hostname_verification_policy(tmp_path):
    bundle = CertBundle("localhost")
    with https_server(tmp_path, bundle, "host") as server:
        # Connect via the IP address while the cert only names 'localhost'.
        ip_url = str(server.url).replace("localhost", "127.0.0.1")

        strict = httpx.TLSPolicy.create(
            verify=bundle.client_ssl_context(),
            key="strict",
        )
        no_check = httpx.TLSPolicy.create(
            verify=bundle.client_ssl_context(),
            check_hostname=False,
            key="no-host-check",
        )

        with httpx.Client(tls_policy=StaticResolver(strict)) as client:
            with pytest.raises(httpx.ConnectError):
                client.get(ip_url)

        with httpx.Client(tls_policy=StaticResolver(no_check)) as client:
            assert client.get(ip_url).status_code == 200


def test_plain_http_origin_uses_policy_transport(server):
    resolver = StaticResolver(httpx.TLSPolicy.create(verify=False, key="plain"))
    with httpx.Client(base_url=server.url, tls_policy=resolver) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert len(_sync_policies(client)._entries) == 1


# ---------------------------------------------------------------------------
# ALPN / HTTP/2
# ---------------------------------------------------------------------------


def test_alpn_selects_http2(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "h2", http=H2_HTTP_PROTOCOL) as server:
        url = str(server.url)

        h2_policy = httpx.TLSPolicy(bundle.client_ssl_context(), http2=True, key="h2")
        h1_policy = httpx.TLSPolicy(bundle.client_ssl_context(), http2=False, key="h1")

        with httpx.Client(tls_policy=StaticResolver(h2_policy)) as client:
            response = client.get(url)
            assert response.http_version == "HTTP/2"

        with httpx.Client(tls_policy=StaticResolver(h1_policy)) as client:
            response = client.get(url)
            assert response.http_version == "HTTP/1.1"


def test_pinned_alpn_forces_http1_on_http2_pool(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "pin", http=H2_HTTP_PROTOCOL) as server:
        # The pool allows HTTP/2, but the pinned ALPN list only offers
        # HTTP/1.1, so the handshake must negotiate HTTP/1.1. Verification is
        # disabled here so the factory-built pinned context is used.
        policy = httpx.TLSPolicy.create(
            verify=False,
            http2=True,
            alpn_protocols=["http/1.1"],
            key="pinned-h1",
        )
        with httpx.Client(tls_policy=StaticResolver(policy)) as client:
            response = client.get(str(server.url))
            assert response.http_version == "HTTP/1.1"


def test_http2_multiplexing_sync(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "multi", http=H2_HTTP_PROTOCOL) as server:
        url = str(server.url)
        policy = httpx.TLSPolicy(
            bundle.client_ssl_context(), http2=True, key="h2-multi"
        )
        with httpx.Client(tls_policy=StaticResolver(policy)) as client:
            threads = [
                threading.Thread(target=lambda: client.get(url)) for _ in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            entries = list(_sync_policies(client)._entries.values())
            assert len(entries) == 1
            connections = entries[0].pool.connections
            assert len(connections) == 1
            assert "HTTP/2" in repr(connections[0])


# ---------------------------------------------------------------------------
# Pool key equivalence & cache invalidation
# ---------------------------------------------------------------------------


def test_pool_key_equivalence(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    with (
        https_server(tmp_path, bundle, "eq1") as server_a,
        https_server(tmp_path, bundle, "eq2") as server_b,
    ):
        url_a, url_b = str(server_a.url), str(server_b.url)
        # Equivalent policies deliberately share one key.
        shared_ctx = bundle.client_ssl_context()
        resolver = MapResolver(
            {
                httpx.Origin.from_url(httpx.URL(url_a)): httpx.TLSPolicy(
                    shared_ctx, key="shared"
                ),
                httpx.Origin.from_url(httpx.URL(url_b)): httpx.TLSPolicy(
                    shared_ctx, key="shared"
                ),
            }
        )
        with httpx.Client(tls_policy=resolver) as client:
            client.get(url_a)
            client.get(url_b)
            assert len(_sync_policies(client)._entries) == 1

        # Different keys produce separate pools.
        resolver = MapResolver(
            {
                httpx.Origin.from_url(httpx.URL(url_a)): httpx.TLSPolicy(
                    bundle.client_ssl_context(), key="one"
                ),
                httpx.Origin.from_url(httpx.URL(url_b)): httpx.TLSPolicy(
                    bundle.client_ssl_context(), key="two"
                ),
            }
        )
        with httpx.Client(tls_policy=resolver) as client:
            client.get(url_a)
            client.get(url_b)
            assert len(_sync_policies(client)._entries) == 2


def test_invalidation_uses_new_pool_and_closes_old(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "inv") as server:
        url = str(server.url)
        origin = httpx.Origin.from_url(httpx.URL(url))
        resolver = MapResolver(
            {origin: httpx.TLSPolicy(bundle.client_ssl_context(), key="k")}
        )
        with httpx.Client(tls_policy=resolver) as client:
            state = _sync_policies(client)

            client.get(url)
            assert len(state._entries) == 1
            old_entry = list(state._entries.values())[0]
            assert len(old_entry.pool.connections) == 1
            old_connection = old_entry.pool.connections[0]

            # A new generation must not reuse the old pool/connection.
            resolver.invalidate()
            response = client.get(url)
            assert response.status_code == 200

            assert old_entry not in state._entries.values()
            assert len(state._entries) == 1
            new_entry = list(state._entries.values())[0]
            assert new_entry is not old_entry
            assert old_connection.is_closed()
            assert all(c.is_closed() for c in old_entry.pool.connections)


def test_invalidation_during_inflight_uses_generation_boundary():
    """Deterministic bookkeeping test with stubbed httpcore pools."""

    class FakeStream:
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self) -> typing.Iterator[bytes]:
            yield b""

        def close(self) -> None:
            self.closed = True

    class FakePool:
        def __init__(self) -> None:
            self.close_calls = 0
            self.requests = 0

        def handle_request(self, request: httpx.Request) -> types.SimpleNamespace:
            self.requests += 1
            return types.SimpleNamespace(
                status=200, headers=[], stream=FakeStream(), extensions={}
            )

        def close(self) -> None:
            self.close_calls += 1

    resolver = StaticResolver(
        httpx.TLSPolicy(ssl.create_default_context(), key="tenant")
    )
    transport = httpx.HTTPTransport(tls_policy=resolver)
    state = transport._tls
    assert state is not None
    state._build_pool = lambda *, ssl_context, http2: FakePool()

    request = httpx.Request("GET", "https://tenant.example/")

    # First request is in-flight (stream still open).
    first = transport.handle_request(request)
    assert len(state._entries) == 1
    old_entry = list(state._entries.values())[0]
    assert old_entry.in_flight == 1
    assert old_entry.pool.close_calls == 0

    # Policy update while that request is still open: the old pool must not
    # be forcibly closed yet, and new requests must land on a fresh pool.
    # Both generations coexist until the old request drains.
    resolver.invalidate()
    second = transport.handle_request(request)
    assert old_entry.pool.close_calls == 0
    assert old_entry.closing is True
    assert len(state._entries) == 2

    # Closing the old response stream is the generation boundary: the old
    # pool is closed exactly once, and no stale entry is left behind.
    first.close()
    assert old_entry.pool.close_calls == 1
    assert old_entry not in state._entries.values()
    assert len(state._entries) == 1

    second.close()
    assert len(state._entries) == 1
    transport.close()
    assert state._entries == {}


def test_close_closes_all_generation_pools() -> None:
    class FakeStream:
        def __iter__(self) -> typing.Iterator[bytes]:
            yield b""

        def close(self) -> None:
            pass

    class FakePool:
        def __init__(self) -> None:
            self.closed = False

        def handle_request(self, request: httpx.Request) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                status=200, headers=[], stream=FakeStream(), extensions={}
            )

        def close(self) -> None:
            self.closed = True

    resolver = StaticResolver(httpx.TLSPolicy(ssl.create_default_context(), key="a"))
    transport = httpx.HTTPTransport(tls_policy=resolver)
    state = transport._tls
    assert state is not None
    pools: list[FakePool] = []

    def build(*, ssl_context: ssl.SSLContext, http2: bool) -> FakePool:
        pool = FakePool()
        pools.append(pool)
        return pool

    state._build_pool = build
    transport.handle_request(httpx.Request("GET", "https://a.example/")).close()
    assert pools[0].closed is False

    # Invalidation with no in-flight requests closes the idle old pool
    # immediately; the next request gets a fresh pool.
    resolver.invalidate()
    resolver.policy = httpx.TLSPolicy(ssl.create_default_context(), key="b")
    transport.handle_request(httpx.Request("GET", "https://b.example/")).close()
    assert len(pools) == 2
    assert pools[0].closed is True
    assert pools[1].closed is False

    # Client close shuts down everything that remains.
    transport.close()
    assert all(p.closed for p in pools)
    assert state._entries == {}


# ---------------------------------------------------------------------------
# Redirects re-resolve the new origin at the same boundary
# ---------------------------------------------------------------------------


def test_redirect_re_resolves_new_origin(tmp_path):
    bundle_a = CertBundle("localhost", "127.0.0.1")
    bundle_b = CertBundle("localhost", "127.0.0.1")
    with (
        https_server(tmp_path, bundle_b, "rd-b") as server_b,
        https_server(
            tmp_path,
            bundle_a,
            "rd-a",
            redirect_app(str(server_b.url).encode()),
        ) as server_a,
    ):
        url_a, url_b = str(server_a.url), str(server_b.url)
        origin_a = httpx.Origin.from_url(httpx.URL(url_a))
        origin_b = httpx.Origin.from_url(httpx.URL(url_b))
        resolver = MapResolver(
            {
                origin_a: policy_for(bundle_a, "a"),
                origin_b: policy_for(bundle_b, "b"),
            }
        )
        with httpx.Client(tls_policy=resolver, follow_redirects=True) as client:
            response = client.get(url_a)
            assert response.status_code == 200
            assert response.text == "Hello, world!"

            # Both the redirect origin and the target origin were resolved,
            # and got separate pools (so no cross-origin/cert reuse).
            assert origin_a in resolver.calls
            assert origin_b in resolver.calls
            assert len(_sync_policies(client)._entries) == 2


# ---------------------------------------------------------------------------
# HTTP proxy CONNECT tunnelling
# ---------------------------------------------------------------------------


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    try:
        while True:
            data = source.recv(65536)
            if not data:
                break
            sink.sendall(data)
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            sink.shutdown(socket.SHUT_WR)


class ConnectProxy(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.port = self.listener.getsockname()[1]
        self.targets: list[tuple[str, int]] = []
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        self.listener.close()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            with conn:
                reader = conn.makefile("rb")
                request_line = reader.readline().decode()
                method, target, _ = request_line.split(" ", 2)
                assert method == "CONNECT"
                # Consume the remaining CONNECT headers using the same
                # buffered reader.
                while reader.readline() not in (b"\r\n", b"", b"\n"):
                    pass
                host, port = target.rsplit(":", 1)
                self.targets.append((host, int(port)))
                upstream = socket.create_connection((host, int(port)), timeout=5)
                with upstream:
                    conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                    t1 = threading.Thread(
                        target=_pipe, args=(conn, upstream), daemon=True
                    )
                    t2 = threading.Thread(
                        target=_pipe, args=(upstream, conn), daemon=True
                    )
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()
        except (OSError, ValueError):
            return


def test_proxy_connect_tunnel_uses_policy(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    proxy = ConnectProxy()
    proxy.start()
    try:
        with https_server(tmp_path, bundle, "proxy-tls") as server:
            url = str(server.url)
            origin = httpx.Origin.from_url(httpx.URL(url))
            resolver = MapResolver({origin: policy_for(bundle, "tunnelled")})
            with httpx.Client(
                proxy=f"http://127.0.0.1:{proxy.port}",
                tls_policy=resolver,
            ) as client:
                response = client.get(url)
                assert response.status_code == 200

                transport = typing.cast(
                    httpx.HTTPTransport, client._transport_for_url(httpx.URL(url))
                )
                assert transport._proxy is not None
                assert transport._tls is not None
                assert len(transport._tls._entries) == 1
                assert proxy.targets[-1] == ("127.0.0.1", origin.port)

            # Resolution failure happens before the proxy is contacted.
            resolver.fail = True
            with httpx.Client(
                proxy=f"http://127.0.0.1:{proxy.port}",
                tls_policy=resolver,
            ) as failing:
                with pytest.raises(httpx.TLSPolicyError):
                    failing.get(url)
                before = len(proxy.targets)
                with pytest.raises(httpx.TLSPolicyError):
                    failing.get(url)
                assert len(proxy.targets) == before
    finally:
        proxy.stop()


# ---------------------------------------------------------------------------
# Client wiring & compatibility
# ---------------------------------------------------------------------------


def test_tls_policy_rejected_with_custom_transport() -> None:
    unused_policy = httpx.TLSPolicy(ssl.create_default_context(), key="unused")

    with pytest.raises(ValueError, match="tls_policy"):
        httpx.Client(
            transport=httpx.HTTPTransport(),
            tls_policy=StaticResolver(unused_policy),
        )

    async def check_async() -> None:
        with pytest.raises(ValueError, match="tls_policy"):
            httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(),
                tls_policy=StaticResolver(unused_policy),
            )

    anyio.run(check_async)


def test_tls_policy_compatible_with_mounts(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    mounted = httpx.MockTransport(lambda request: httpx.Response(200, text="mounted"))
    with https_server(tmp_path, bundle, "mounts") as server:
        url = str(server.url)
        origin = httpx.Origin.from_url(httpx.URL(url))
        resolver = MapResolver({origin: policy_for(bundle, "m")})
        with httpx.Client(
            tls_policy=resolver,
            mounts={"all://mounted.example": mounted},
        ) as client:
            assert client.get(url).text == "Hello, world!"
            assert client.get("http://mounted.example/").text == "mounted"


def test_callable_resolver_supported(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "call") as server:
        url = str(server.url)
        origin = httpx.Origin.from_url(httpx.URL(url))
        policy = httpx.TLSPolicy(bundle.client_ssl_context(), key="callable")
        seen = []

        def resolver(seen_origin):
            seen.append(seen_origin)
            return policy

        with httpx.Client(tls_policy=resolver) as client:
            assert client.get(url).status_code == 200
            client.get(url)  # second call, same generation: cached
        assert seen == [origin]


def test_legacy_behaviour_unchanged():
    # No resolver: single eagerly constructed pool.
    transport = httpx.HTTPTransport()
    assert transport._tls is None
    assert transport._pool is not None
    transport.close()


def test_inflight_request_completes_on_old_generation(tmp_path):
    """End-to-end generation boundary against a real, slow server."""
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "slow-gen", slow_app) as server:
        url = str(server.url)
        origin = httpx.Origin.from_url(httpx.URL(url))
        resolver = MapResolver(
            {origin: httpx.TLSPolicy(bundle.client_ssl_context(), key="k")}
        )
        with httpx.Client(tls_policy=resolver) as client:
            state = _sync_policies(client)
            result: dict[str, object] = {}

            def slow_request() -> None:
                try:
                    result["old"] = client.get(url)
                except Exception as exc:  # pragma: no cover
                    result["old_error"] = exc

            thread = threading.Thread(target=slow_request)
            thread.start()

            # Wait until the first request has entered the old-generation
            # pool (connection in progress / response pending).
            while not state._entries:
                time.sleep(0.001)
            old_entry = list(state._entries.values())[0]
            while old_entry.in_flight == 0:
                time.sleep(0.001)

            # Invalidate while the request is still being served: new
            # requests go to a fresh generation; the old one drains.
            resolver.invalidate()
            new_response = client.get(url)
            assert new_response.status_code == 200

            thread.join()
            assert "old_error" not in result
            old_response = typing.cast(httpx.Response, result["old"])
            assert old_response.status_code == 200

            # The old pool was closed and only the current pool remains.
            assert old_entry not in state._entries.values()
            assert all(c.is_closed() for c in old_entry.pool.connections)
            assert len(state._entries) == 1


# ---------------------------------------------------------------------------
# Sync concurrency across invalidations
# ---------------------------------------------------------------------------


def test_sync_concurrent_requests_and_invalidations(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "conc") as server:
        url = str(server.url)
        origin = httpx.Origin.from_url(httpx.URL(url))
        resolver = MapResolver(
            {origin: httpx.TLSPolicy(bundle.client_ssl_context(), key="k")}
        )
        with httpx.Client(tls_policy=resolver) as client:
            errors: list[Exception] = []

            def worker(i: int) -> None:
                try:
                    if i % 3 == 0:
                        resolver.invalidate()
                    response = client.get(url)
                    assert response.status_code == 200
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == []
            # All superseded generations were cleaned up; only the current
            # pool remains.
            assert len(_sync_policies(client)._entries) == 1


# ---------------------------------------------------------------------------
# Async coverage
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_policy_per_origin(tmp_path):
    bundle_a = CertBundle("localhost", "127.0.0.1")
    bundle_b = CertBundle("localhost", "127.0.0.1")
    with (
        https_server(tmp_path, bundle_a, "aa") as server_a,
        https_server(tmp_path, bundle_b, "ab") as server_b,
    ):
        url_a, url_b = str(server_a.url), str(server_b.url)
        resolver = AsyncMapResolver(
            {
                httpx.Origin.from_url(httpx.URL(url_a)): policy_for(bundle_a, "a"),
                httpx.Origin.from_url(httpx.URL(url_b)): policy_for(bundle_b, "b"),
            }
        )
        async with httpx.AsyncClient(tls_policy=resolver) as client:
            assert (await client.get(url_a)).status_code == 200
            assert (await client.get(url_b)).status_code == 200
        assert len(resolver.async_calls) == 2


@pytest.mark.anyio
async def test_async_resolution_failure_before_connect() -> None:
    resolver = AsyncMapResolver({})
    async with httpx.AsyncClient(tls_policy=resolver) as client:
        with pytest.raises(httpx.TLSPolicyError):
            await client.get("https://unknown-policy.example/")
        assert _async_policies(client)._entries == {}


@pytest.mark.anyio
async def test_async_client_uses_sync_resolver(tmp_path: pathlib.Path) -> None:
    # A resolver that only implements the synchronous `resolve()` is driven
    # from a worker thread by the async transport.
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "async-sync-resolver") as server:
        url = str(server.url)
        resolver = MapResolver(
            {
                httpx.Origin.from_url(httpx.URL(url)): policy_for(
                    bundle, "sync-resolve"
                )
            }
        )
        async with httpx.AsyncClient(tls_policy=resolver) as client:
            response = await client.get(url)
            assert response.status_code == 200
        assert any(
            origin == httpx.Origin.from_url(httpx.URL(url))
            for origin in resolver.calls
        )


@pytest.mark.anyio
async def test_async_invalidation_closes_old_pool(tmp_path: pathlib.Path) -> None:
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "ai") as server:
        url = str(server.url)
        origin = httpx.Origin.from_url(httpx.URL(url))
        resolver = AsyncMapResolver(
            {origin: httpx.TLSPolicy(bundle.client_ssl_context(), key="k")}
        )
        async with httpx.AsyncClient(tls_policy=resolver) as client:
            state = _async_policies(client)
            await client.get(url)
            old_entry = list(state._entries.values())[0]
            resolver.invalidate()
            await client.get(url)
            assert old_entry not in state._entries.values()
            assert all(c.is_closed() for c in old_entry.pool.connections)
            assert len(state._entries) == 1


@pytest.mark.anyio
async def test_async_inflight_generation_boundary() -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self) -> typing.AsyncIterator[bytes]:
            async def gen() -> typing.AsyncIterator[bytes]:
                yield b""

            return gen()

        async def aclose(self) -> None:
            self.closed = True

    class FakePool:
        def __init__(self) -> None:
            self.close_calls = 0

        async def handle_async_request(
            self, request: httpx.Request
        ) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                status=200, headers=[], stream=FakeStream(), extensions={}
            )

        async def aclose(self) -> None:
            self.close_calls += 1

    resolver = StaticResolver(
        httpx.TLSPolicy(ssl.create_default_context(), key="tenant")
    )
    transport = httpx.AsyncHTTPTransport(tls_policy=resolver)
    state = transport._tls
    assert state is not None
    state._build_pool = lambda *, ssl_context, http2: FakePool()

    request = httpx.Request("GET", "https://tenant.example/")
    first = await transport.handle_async_request(request)
    old_entry = list(state._entries.values())[0]
    assert old_entry.in_flight == 1

    resolver.invalidate()
    second = await transport.handle_async_request(request)
    assert old_entry.pool.close_calls == 0
    assert old_entry.closing is True

    await first.aclose()
    assert old_entry.pool.close_calls == 1
    assert old_entry not in state._entries.values()
    await second.aclose()
    await transport.aclose()
    assert state._entries == {}


@pytest.mark.anyio
async def test_async_redirect_re_resolves_origin(tmp_path):
    bundle_a = CertBundle("localhost", "127.0.0.1")
    bundle_b = CertBundle("localhost", "127.0.0.1")
    with (
        https_server(tmp_path, bundle_b, "arb") as server_b,
        https_server(
            tmp_path,
            bundle_a,
            "ara",
            redirect_app(str(server_b.url).encode()),
        ) as server_a,
    ):
        url_a, url_b = str(server_a.url), str(server_b.url)
        resolver = AsyncMapResolver(
            {
                httpx.Origin.from_url(httpx.URL(url_a)): policy_for(bundle_a, "a"),
                httpx.Origin.from_url(httpx.URL(url_b)): policy_for(bundle_b, "b"),
            }
        )
        async with httpx.AsyncClient(
            tls_policy=resolver, follow_redirects=True
        ) as client:
            response = await client.get(url_a)
            assert response.status_code == 200


@pytest.mark.anyio
async def test_async_http2_multiplexing(tmp_path):
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "am", http=H2_HTTP_PROTOCOL) as server:
        url = str(server.url)
        policy = httpx.TLSPolicy(bundle.client_ssl_context(), http2=True, key="h2")
        async with httpx.AsyncClient(tls_policy=StaticResolver(policy)) as client:

            async def fetch() -> None:
                response = await client.get(url)
                assert response.http_version == "HTTP/2"

            async with anyio.create_task_group() as tg:
                for _ in range(6):
                    tg.start_soon(fetch)

            entries = list(_async_policies(client)._entries.values())
            assert len(entries) == 1
            assert len(entries[0].pool.connections) == 1
            assert "HTTP/2" in repr(entries[0].pool.connections[0])


@pytest.mark.anyio
async def test_async_concurrency_with_invalidations(
    tmp_path: pathlib.Path,
) -> None:
    bundle = CertBundle("localhost", "127.0.0.1")
    with https_server(tmp_path, bundle, "ac", slow_app) as server:
        url = str(server.url)
        origin = httpx.Origin.from_url(httpx.URL(url))
        resolver = AsyncMapResolver(
            {origin: httpx.TLSPolicy(bundle.client_ssl_context(), key="k")}
        )
        async with httpx.AsyncClient(tls_policy=resolver) as client:

            async def fetch(i: int) -> None:
                if i == 2:
                    resolver.invalidate()
                response = await client.get(url)
                assert response.status_code == 200

            async with anyio.create_task_group() as tg:
                for i in range(4):
                    tg.start_soon(fetch, i)

            # All generations but the last have been drained and removed.
            assert len(_async_policies(client)._entries) == 1
