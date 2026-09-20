"""
Custom transports, with nicely configured defaults.

The following additional keyword arguments are currently supported by httpcore...

* uds: str
* local_address: str
* retries: int

Example usages...

# Disable HTTP/2 on a single specific domain.
mounts = {
    "all://": httpx.HTTPTransport(http2=True),
    "all://*example.org": httpx.HTTPTransport()
}

# Using advanced httpcore configuration, with connection retries.
transport = httpx.HTTPTransport(retries=1)
client = httpx.Client(transport=transport)

# Using advanced httpcore configuration, with unix domain sockets.
transport = httpx.HTTPTransport(uds="socket.uds")
client = httpx.Client(transport=transport)
"""

from __future__ import annotations

import contextlib
import threading
import typing
from types import TracebackType

if typing.TYPE_CHECKING:
    import ssl  # pragma: no cover

    import httpx  # pragma: no cover

from .._config import DEFAULT_LIMITS, Limits, Proxy, create_ssl_context
from .._exceptions import (
    ConnectError,
    ConnectTimeout,
    LocalProtocolError,
    NetworkError,
    PoolTimeout,
    ProtocolError,
    ProxyError,
    ReadError,
    ReadTimeout,
    RemoteProtocolError,
    TLSPolicyError,
    TimeoutException,
    UnsupportedProtocol,
    WriteError,
    WriteTimeout,
)
from .._models import Request, Response
from .._tls import (
    Origin,
    TLSPolicy,
    TLSPolicyResolver,
    coerce_tls_policy_resolver,
)
from .._types import AsyncByteStream, CertTypes, ProxyTypes, SyncByteStream
from .._urls import URL
from .base import AsyncBaseTransport, BaseTransport

T = typing.TypeVar("T", bound="HTTPTransport")
A = typing.TypeVar("A", bound="AsyncHTTPTransport")

SOCKET_OPTION = typing.Union[
    typing.Tuple[int, int, int],
    typing.Tuple[int, int, typing.Union[bytes, bytearray]],
    typing.Tuple[int, int, None, int],
]

__all__ = ["AsyncHTTPTransport", "HTTPTransport"]

HTTPCORE_EXC_MAP: dict[type[Exception], type[httpx.HTTPError]] = {}


def _load_httpcore_exceptions() -> dict[type[Exception], type[httpx.HTTPError]]:
    import httpcore

    return {
        httpcore.TimeoutException: TimeoutException,
        httpcore.ConnectTimeout: ConnectTimeout,
        httpcore.ReadTimeout: ReadTimeout,
        httpcore.WriteTimeout: WriteTimeout,
        httpcore.PoolTimeout: PoolTimeout,
        httpcore.NetworkError: NetworkError,
        httpcore.ConnectError: ConnectError,
        httpcore.ReadError: ReadError,
        httpcore.WriteError: WriteError,
        httpcore.ProxyError: ProxyError,
        httpcore.UnsupportedProtocol: UnsupportedProtocol,
        httpcore.ProtocolError: ProtocolError,
        httpcore.LocalProtocolError: LocalProtocolError,
        httpcore.RemoteProtocolError: RemoteProtocolError,
    }


@contextlib.contextmanager
def map_httpcore_exceptions() -> typing.Iterator[None]:
    global HTTPCORE_EXC_MAP
    if len(HTTPCORE_EXC_MAP) == 0:
        HTTPCORE_EXC_MAP = _load_httpcore_exceptions()
    try:
        yield
    except Exception as exc:
        mapped_exc = None

        for from_exc, to_exc in HTTPCORE_EXC_MAP.items():
            if not isinstance(exc, from_exc):
                continue
            # We want to map to the most specific exception we can find.
            # Eg if `exc` is an `httpcore.ReadTimeout`, we want to map to
            # `httpx.ReadTimeout`, not just `httpx.TimeoutException`.
            if mapped_exc is None or issubclass(to_exc, mapped_exc):
                mapped_exc = to_exc

        if mapped_exc is None:  # pragma: no cover
            raise

        message = str(exc)
        raise mapped_exc(message) from exc


class ResponseStream(SyncByteStream):
    def __init__(self, httpcore_stream: typing.Iterable[bytes]) -> None:
        self._httpcore_stream = httpcore_stream

    def __iter__(self) -> typing.Iterator[bytes]:
        with map_httpcore_exceptions():
            for part in self._httpcore_stream:
                yield part

    def close(self) -> None:
        if hasattr(self._httpcore_stream, "close"):
            self._httpcore_stream.close()


class AsyncResponseStream(AsyncByteStream):
    def __init__(self, httpcore_stream: typing.AsyncIterable[bytes]) -> None:
        self._httpcore_stream = httpcore_stream

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        with map_httpcore_exceptions():
            async for part in self._httpcore_stream:
                yield part

    async def aclose(self) -> None:
        if hasattr(self._httpcore_stream, "aclose"):
            await self._httpcore_stream.aclose()


# Per-generation TLS policy state.
#
# A distinct httpcore pool is created for each `(generation, policy key,
# http2)` tuple, so connections can never be reused across different TLS
# policies. When the resolver generation changes, existing entries are marked
# as closing; idle entries are shut down immediately, while entries with
# in-flight requests (including requests mid-handshake, or with open response
# streams) are shut down the moment the last request drains. This enforces a
# single generation boundary and guarantees no old-policy connection is left
# behind.

PoolKey = typing.Tuple[int, typing.Hashable, bool]
BuildPool = typing.Callable[..., typing.Any]


class _SyncPoolEntry:
    def __init__(
        self, key: PoolKey, pool: typing.Any, generation: int
    ) -> None:
        self.key = key
        self.pool = pool
        self.generation = generation
        self.in_flight = 0
        self.closing = False


class _AsyncPoolEntry:
    def __init__(
        self, key: PoolKey, pool: typing.Any, generation: int
    ) -> None:
        self.key = key
        self.pool = pool
        self.generation = generation
        self.in_flight = 0
        self.closing = False


class _SyncTLSPolicies:
    def __init__(
        self,
        resolver: TLSPolicyResolver,
        *,
        default_http2: bool,
        build_pool: BuildPool,
    ) -> None:
        self._resolver = resolver
        self._default_http2 = default_http2
        self._build_pool = build_pool
        self._lock = threading.Lock()
        self._generation: int | None = None
        self._cache: dict[Origin, tuple[int, TLSPolicy]] = {}
        self._entries: dict[PoolKey, _SyncPoolEntry] = {}

    def _mark_generation(
        self, generation: int
    ) -> list[_SyncPoolEntry]:
        """Mark all entries as closing when the generation changes.

        Called while holding the lock. Returns the entries that have no
        in-flight requests and can be closed immediately.
        """
        if self._generation == generation:
            return []
        self._cache.clear()
        drained: list[_SyncPoolEntry] = []
        for entry in self._entries.values():
            entry.closing = True
            if entry.in_flight == 0:
                drained.append(entry)
        for entry in drained:
            self._entries.pop(entry.key, None)
        self._generation = generation
        return drained

    def resolve(self, origin: Origin) -> tuple[TLSPolicy, int]:
        """Resolve (possibly via cache) the policy to use for `origin`."""
        while True:
            generation = self._resolver.generation
            with self._lock:
                drained = self._mark_generation(generation)
                cached = self._cache.get(origin)
            # Close drained pools outside the lock to avoid lock ordering
            # issues against httpcore's internal pool locks.
            for entry in drained:
                entry.pool.close()

            if cached is not None:
                return cached[1], generation

            try:
                policy = self._resolver.resolve(origin)
            except TLSPolicyError:
                raise
            except Exception as exc:
                raise TLSPolicyError(
                    f"TLS policy resolver failed for origin {str(origin)}: "
                    f"{exc}",
                    origin=origin,
                ) from exc

            if not isinstance(policy, TLSPolicy):
                raise TLSPolicyError(
                    "TLS policy resolver returned a "
                    f"{type(policy).__name__!r} instead of a TLSPolicy "
                    f"for origin {str(origin)}.",
                    origin=origin,
                )

            with self._lock:
                if self._generation != generation:
                    # The resolver was invalidated while we were resolving.
                    # Discard the result and resolve against the new
                    # generation.
                    continue
                cached = self._cache.get(origin)
                if cached is not None:
                    policy = cached[1]
                else:
                    self._cache[origin] = (generation, policy)
            return policy, generation

    def acquire(
        self, policy: TLSPolicy, generation: int
    ) -> _SyncPoolEntry:
        http2 = self._default_http2 if policy.http2 is None else policy.http2
        identity = (
            policy.key
            if policy.key is not None
            else id(policy.ssl_context)
        )
        key = (generation, identity, http2)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                pool = self._build_pool(
                    ssl_context=policy.ssl_context, http2=http2
                )
                entry = _SyncPoolEntry(key=key, pool=pool, generation=generation)
                self._entries[key] = entry
            entry.in_flight += 1
        return entry

    def release(self, entry: _SyncPoolEntry) -> None:
        with self._lock:
            entry.in_flight -= 1
            should_close = entry.closing and entry.in_flight == 0
            if should_close:
                self._entries.pop(entry.key, None)
        if should_close:
            entry.pool.close()

    def close_all(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            self._cache.clear()
            for entry in entries:
                entry.closing = True
        for entry in entries:
            entry.pool.close()


class _AsyncTLSPolicies:
    def __init__(
        self,
        resolver: TLSPolicyResolver,
        *,
        default_http2: bool,
        build_pool: BuildPool,
    ) -> None:
        self._resolver = resolver
        self._default_http2 = default_http2
        self._build_pool = build_pool
        self._lock: typing.Any = None
        self._generation: int | None = None
        self._cache: dict[Origin, tuple[int, TLSPolicy]] = {}
        self._entries: dict[PoolKey, _AsyncPoolEntry] = {}

    async def _get_lock(self) -> typing.Any:
        # anyio locks must be created within an async context.
        if self._lock is None:
            import anyio

            self._lock = anyio.Lock()
        return self._lock

    async def _mark_generation(
        self, generation: int
    ) -> list[_AsyncPoolEntry]:
        if self._generation == generation:
            return []
        self._cache.clear()
        drained: list[_AsyncPoolEntry] = []
        for entry in self._entries.values():
            entry.closing = True
            if entry.in_flight == 0:
                drained.append(entry)
        for entry in drained:
            self._entries.pop(entry.key, None)
        self._generation = generation
        return drained

    async def resolve(self, origin: Origin) -> tuple[TLSPolicy, int]:
        while True:
            generation = self._resolver.generation
            lock = await self._get_lock()
            async with lock:
                drained = await self._mark_generation(generation)
                cached = self._cache.get(origin)
            for entry in drained:
                await entry.pool.aclose()

            if cached is not None:
                return cached[1], generation

            try:
                policy = await self._resolver.aresolve(origin)
            except TLSPolicyError:
                raise
            except Exception as exc:
                raise TLSPolicyError(
                    f"TLS policy resolver failed for origin {str(origin)}: "
                    f"{exc}",
                    origin=origin,
                ) from exc

            if not isinstance(policy, TLSPolicy):
                raise TLSPolicyError(
                    "TLS policy resolver returned a "
                    f"{type(policy).__name__!r} instead of a TLSPolicy "
                    f"for origin {str(origin)}.",
                    origin=origin,
                )

            async with lock:
                if self._generation != generation:
                    continue
                cached = self._cache.get(origin)
                if cached is not None:
                    policy = cached[1]
                else:
                    self._cache[origin] = (generation, policy)
            return policy, generation

    async def acquire(
        self, policy: TLSPolicy, generation: int
    ) -> _AsyncPoolEntry:
        http2 = self._default_http2 if policy.http2 is None else policy.http2
        identity = (
            policy.key
            if policy.key is not None
            else id(policy.ssl_context)
        )
        key = (generation, identity, http2)
        lock = await self._get_lock()
        async with lock:
            entry = self._entries.get(key)
            if entry is None:
                pool = self._build_pool(
                    ssl_context=policy.ssl_context, http2=http2
                )
                entry = _AsyncPoolEntry(key=key, pool=pool, generation=generation)
                self._entries[key] = entry
            entry.in_flight += 1
        return entry

    async def release(self, entry: _AsyncPoolEntry) -> None:
        lock = await self._get_lock()
        async with lock:
            entry.in_flight -= 1
            should_close = entry.closing and entry.in_flight == 0
            if should_close:
                self._entries.pop(entry.key, None)
        if should_close:
            await entry.pool.aclose()

    async def close_all(self) -> None:
        lock = await self._get_lock()
        async with lock:
            entries = list(self._entries.values())
            self._entries.clear()
            self._cache.clear()
            for entry in entries:
                entry.closing = True
        for entry in entries:
            await entry.pool.aclose()


class _PolicyResponseStream(ResponseStream):
    def __init__(
        self,
        httpcore_stream: typing.Iterable[bytes],
        entry: _SyncPoolEntry,
        policies: _SyncTLSPolicies,
    ) -> None:
        super().__init__(httpcore_stream)
        self._entry = entry
        self._policies = policies

    def close(self) -> None:
        try:
            if hasattr(self._httpcore_stream, "close"):
                self._httpcore_stream.close()
        finally:
            # The pool can only be closed once the response stream has been
            # fully released back to it.
            self._policies.release(self._entry)


class _AsyncPolicyResponseStream(AsyncResponseStream):
    def __init__(
        self,
        httpcore_stream: typing.AsyncIterable[bytes],
        entry: _AsyncPoolEntry,
        policies: _AsyncTLSPolicies,
    ) -> None:
        super().__init__(httpcore_stream)
        self._entry = entry
        self._policies = policies

    async def aclose(self) -> None:
        try:
            if hasattr(self._httpcore_stream, "aclose"):
                await self._httpcore_stream.aclose()
        finally:
            await self._policies.release(self._entry)


def _to_httpcore_request(request: Request) -> typing.Any:
    import httpcore

    return httpcore.Request(
        method=request.method,
        url=httpcore.URL(
            scheme=request.url.raw_scheme,
            host=request.url.raw_host,
            port=request.url.port,
            target=request.url.raw_path,
        ),
        headers=request.headers.raw,
        content=request.stream,
        extensions=request.extensions,
    )


def _to_httpx_response(
    resp: typing.Any, stream: typing.Any
) -> Response:
    return Response(
        status_code=resp.status,
        headers=resp.headers,
        stream=stream,
        extensions=resp.extensions,
    )


class HTTPTransport(BaseTransport):
    def __init__(
        self,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        proxy: ProxyTypes | None = None,
        uds: str | None = None,
        local_address: str | None = None,
        retries: int = 0,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
        tls_policy: TLSPolicyResolver
        | typing.Callable[[Origin], TLSPolicy]
        | None = None,
    ) -> None:
        proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy

        # Validate the proxy configuration eagerly in both legacy and
        # policy modes, so construction-time errors are identical regardless
        # of whether pools are created up-front or lazily.
        if proxy is not None and proxy.url.scheme in ("socks5", "socks5h"):
            try:
                import socksio  # noqa
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "Using SOCKS proxy, but the 'socksio' package is not "
                    "installed. Make sure to install httpx using "
                    "`pip install httpx[socks]`."
                ) from None
        elif proxy is not None and proxy.url.scheme not in (
            "http",
            "https",
        ):  # pragma: no cover
            raise ValueError(
                "Proxy protocol must be either 'http', 'https', 'socks5', or "
                f"'socks5h', but got {proxy.url.scheme!r}."
            )

        self._proxy = proxy
        self._limits = limits
        self._http1 = http1
        self._uds = uds
        self._local_address = local_address
        self._retries = retries
        self._socket_options = socket_options

        if tls_policy is None:
            # Legacy behaviour: a single SSL context and a single pool.
            ssl_context = create_ssl_context(
                verify=verify, cert=cert, trust_env=trust_env
            )
            self._pool = self._build_pool(
                ssl_context=ssl_context, http2=http2
            )
            self._tls: _SyncTLSPolicies | None = None
        else:
            # Dynamic behaviour: one pool per (generation, policy key), with
            # the SSL context and HTTP/2 choice supplied by the resolver.
            self._pool = None
            self._tls = _SyncTLSPolicies(
                coerce_tls_policy_resolver(tls_policy),
                default_http2=http2,
                build_pool=self._build_pool,
            )

    def _build_pool(
        self, *, ssl_context: ssl.SSLContext | None, http2: bool
    ) -> typing.Any:
        import httpcore

        if self._proxy is None:
            return httpcore.ConnectionPool(
                ssl_context=ssl_context,
                max_connections=self._limits.max_connections,
                max_keepalive_connections=self._limits.max_keepalive_connections,
                keepalive_expiry=self._limits.keepalive_expiry,
                http1=self._http1,
                http2=http2,
                uds=self._uds,
                local_address=self._local_address,
                retries=self._retries,
                socket_options=self._socket_options,
            )
        elif self._proxy.url.scheme in ("http", "https"):
            return httpcore.HTTPProxy(
                proxy_url=httpcore.URL(
                    scheme=self._proxy.url.raw_scheme,
                    host=self._proxy.url.raw_host,
                    port=self._proxy.url.port,
                    target=self._proxy.url.raw_path,
                ),
                proxy_auth=self._proxy.raw_auth,
                proxy_headers=self._proxy.headers.raw,
                ssl_context=ssl_context,
                proxy_ssl_context=self._proxy.ssl_context,
                max_connections=self._limits.max_connections,
                max_keepalive_connections=self._limits.max_keepalive_connections,
                keepalive_expiry=self._limits.keepalive_expiry,
                http1=self._http1,
                http2=http2,
                socket_options=self._socket_options,
            )
        else:
            return httpcore.SOCKSProxy(
                proxy_url=httpcore.URL(
                    scheme=self._proxy.url.raw_scheme,
                    host=self._proxy.url.raw_host,
                    port=self._proxy.url.port,
                    target=self._proxy.url.raw_path,
                ),
                proxy_auth=self._proxy.raw_auth,
                ssl_context=ssl_context,
                max_connections=self._limits.max_connections,
                max_keepalive_connections=self._limits.max_keepalive_connections,
                keepalive_expiry=self._limits.keepalive_expiry,
                http1=self._http1,
                http2=http2,
            )

    def __enter__(self: T) -> T:  # Use generics for subclass support.
        if self._tls is None:
            self._pool.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        if self._tls is None:
            with map_httpcore_exceptions():
                self._pool.__exit__(exc_type, exc_value, traceback)
        else:
            self.close()

    def handle_request(
        self,
        request: Request,
    ) -> Response:
        assert isinstance(request.stream, SyncByteStream)

        req = _to_httpcore_request(request)

        if self._tls is None:
            with map_httpcore_exceptions():
                resp = self._pool.handle_request(req)
            assert isinstance(resp.stream, typing.Iterable)
            return _to_httpx_response(resp, ResponseStream(resp.stream))

        # Resolve the TLS policy *before* any connection is attempted.
        origin = Origin.from_url(request.url)
        policy, generation = self._tls.resolve(origin)
        entry = self._tls.acquire(policy, generation)
        try:
            with map_httpcore_exceptions():
                resp = entry.pool.handle_request(req)
        except BaseException:
            self._tls.release(entry)
            raise

        assert isinstance(resp.stream, typing.Iterable)
        return _to_httpx_response(
            resp,
            _PolicyResponseStream(resp.stream, entry, self._tls),
        )

    def close(self) -> None:
        if self._tls is None:
            self._pool.close()
        else:
            self._tls.close_all()


class AsyncHTTPTransport(AsyncBaseTransport):
    def __init__(
        self,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        proxy: ProxyTypes | None = None,
        uds: str | None = None,
        local_address: str | None = None,
        retries: int = 0,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
        tls_policy: TLSPolicyResolver
        | typing.Callable[[Origin], TLSPolicy]
        | None = None,
    ) -> None:
        proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy

        if proxy is not None and proxy.url.scheme in ("socks5", "socks5h"):
            try:
                import socksio  # noqa
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "Using SOCKS proxy, but the 'socksio' package is not "
                    "installed. Make sure to install httpx using "
                    "`pip install httpx[socks]`."
                ) from None
        elif proxy is not None and proxy.url.scheme not in (
            "http",
            "https",
        ):  # pragma: no cover
            raise ValueError(
                "Proxy protocol must be either 'http', 'https', 'socks5', or "
                f"'socks5h', but got {proxy.url.scheme!r}."
            )

        self._proxy = proxy
        self._limits = limits
        self._http1 = http1
        self._uds = uds
        self._local_address = local_address
        self._retries = retries
        self._socket_options = socket_options

        if tls_policy is None:
            ssl_context = create_ssl_context(
                verify=verify, cert=cert, trust_env=trust_env
            )
            self._pool = self._build_pool(
                ssl_context=ssl_context, http2=http2
            )
            self._tls: _AsyncTLSPolicies | None = None
        else:
            self._pool = None
            self._tls = _AsyncTLSPolicies(
                coerce_tls_policy_resolver(tls_policy),
                default_http2=http2,
                build_pool=self._build_pool,
            )

    def _build_pool(
        self, *, ssl_context: ssl.SSLContext | None, http2: bool
    ) -> typing.Any:
        import httpcore

        if self._proxy is None:
            return httpcore.AsyncConnectionPool(
                ssl_context=ssl_context,
                max_connections=self._limits.max_connections,
                max_keepalive_connections=self._limits.max_keepalive_connections,
                keepalive_expiry=self._limits.keepalive_expiry,
                http1=self._http1,
                http2=http2,
                uds=self._uds,
                local_address=self._local_address,
                retries=self._retries,
                socket_options=self._socket_options,
            )
        elif self._proxy.url.scheme in ("http", "https"):
            return httpcore.AsyncHTTPProxy(
                proxy_url=httpcore.URL(
                    scheme=self._proxy.url.raw_scheme,
                    host=self._proxy.url.raw_host,
                    port=self._proxy.url.port,
                    target=self._proxy.url.raw_path,
                ),
                proxy_auth=self._proxy.raw_auth,
                proxy_headers=self._proxy.headers.raw,
                proxy_ssl_context=self._proxy.ssl_context,
                ssl_context=ssl_context,
                max_connections=self._limits.max_connections,
                max_keepalive_connections=self._limits.max_keepalive_connections,
                keepalive_expiry=self._limits.keepalive_expiry,
                http1=self._http1,
                http2=http2,
                socket_options=self._socket_options,
            )
        else:
            return httpcore.AsyncSOCKSProxy(
                proxy_url=httpcore.URL(
                    scheme=self._proxy.url.raw_scheme,
                    host=self._proxy.url.raw_host,
                    port=self._proxy.url.port,
                    target=self._proxy.url.raw_path,
                ),
                proxy_auth=self._proxy.raw_auth,
                ssl_context=ssl_context,
                max_connections=self._limits.max_connections,
                max_keepalive_connections=self._limits.max_keepalive_connections,
                keepalive_expiry=self._limits.keepalive_expiry,
                http1=self._http1,
                http2=http2,
            )

    async def __aenter__(self: A) -> A:  # Use generics for subclass support.
        if self._tls is None:
            await self._pool.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        if self._tls is None:
            with map_httpcore_exceptions():
                await self._pool.__aexit__(exc_type, exc_value, traceback)
        else:
            await self.aclose()

    async def handle_async_request(
        self,
        request: Request,
    ) -> Response:
        assert isinstance(request.stream, AsyncByteStream)

        req = _to_httpcore_request(request)

        if self._tls is None:
            with map_httpcore_exceptions():
                resp = await self._pool.handle_async_request(req)
            assert isinstance(resp.stream, typing.AsyncIterable)
            return _to_httpx_response(resp, AsyncResponseStream(resp.stream))

        origin = Origin.from_url(request.url)
        policy, generation = await self._tls.resolve(origin)
        entry = await self._tls.acquire(policy, generation)
        try:
            with map_httpcore_exceptions():
                resp = await entry.pool.handle_async_request(req)
        except BaseException:
            await self._tls.release(entry)
            raise

        assert isinstance(resp.stream, typing.AsyncIterable)
        return _to_httpx_response(
            resp,
            _AsyncPolicyResponseStream(resp.stream, entry, self._tls),
        )

    async def aclose(self) -> None:
        if self._tls is None:
            await self._pool.aclose()
        else:
            await self._tls.close_all()
