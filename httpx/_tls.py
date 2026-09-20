"""
Per-origin TLS policy resolution.

The optional TLS policy resolver allows multi-tenant clients to select the
CA bundle, client certificate, ALPN protocols and hostname verification
strategy dynamically, based on the target *origin* of each request.

Resolved policies are identified by an opaque, hashable `key`. Policies with
equal keys are considered equivalent and share a connection pool; a different
key (or a new resolver generation) produces a new pool, and the old pool -
including all of its connections - is closed once any in-flight requests have
drained.

* `Origin` - The target (scheme, host, port) of a request.
* `TLSPolicy` - A resolved SSL context, HTTP/2 preference and pool key.
* `TLSPolicyResolver` - Base class for implementing a resolver.
"""

from __future__ import annotations

import threading
import typing

from ._config import UNSET, UnsetType, create_ssl_context
from ._types import CertTypes
from ._urls import URL

if typing.TYPE_CHECKING:
    import ssl  # pragma: no cover

__all__ = [
    "Origin",
    "TLSPolicy",
    "TLSPolicyResolver",
]

_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


class Origin:
    """
    The target origin of a request, as a `(scheme, host, port)` triple.

    Origins are immutable and hashable, so they can be used as cache keys.
    """

    def __init__(self, scheme: str, host: str, port: int) -> None:
        self.scheme = scheme
        self.host = host
        self.port = port

    @classmethod
    def from_url(cls, url: URL) -> Origin:
        scheme = url.scheme
        host = url.host
        port = url.port
        if port is None:
            port = _DEFAULT_PORTS.get(scheme)
        if not host or port is None:
            from ._exceptions import InvalidURL

            raise InvalidURL(f"Cannot determine origin from URL {url!r}.")
        return cls(scheme=scheme, host=host, port=port)

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, Origin)
            and self.scheme == other.scheme
            and self.host == other.host
            and self.port == other.port
        )

    def __hash__(self) -> int:
        return hash((self.scheme, self.host, self.port))

    def __repr__(self) -> str:
        return f"Origin(scheme={self.scheme!r}, host={self.host!r}, port={self.port!r})"

    def __str__(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


class TLSPolicy:
    """
    A resolved TLS policy for a target origin.

    **Parameters:**

    * **ssl_context** - The SSL context to use for connections to origins
            matching this policy. The CA bundle, client certificate and
            hostname verification settings are all configured on it.
    * **http2** - Whether HTTP/2 (and the `h2` ALPN protocol) may be used.
            When `None`, the transport/client default is used.
    * **alpn_protocols** - The exact ALPN protocol list to offer during the
            TLS handshake. When `None`, httpcore derives the list from
            `http2`. Only configure this if you need to pin a non-default
            list.
    * **key** - An optional hashable identity for the policy. Policies that
            carry equal keys are considered equivalent and share a single
            connection pool. Use a stable key (e.g. tenant identifier plus
            credential version) to avoid unnecessary pools. When unset the
            SSL context's object identity is used.
    """

    def __init__(
        self,
        ssl_context: ssl.SSLContext,
        *,
        http2: bool | None = None,
        alpn_protocols: typing.Sequence[str] | None = None,
        key: typing.Hashable | None = None,
    ) -> None:
        import ssl

        if not isinstance(ssl_context, ssl.SSLContext):
            raise TypeError(
                "TLSPolicy.ssl_context must be an 'ssl.SSLContext' instance, "
                f"got {type(ssl_context).__name__!r}."
            )
        self.ssl_context = ssl_context
        self.http2 = http2
        self.alpn_protocols = None if alpn_protocols is None else tuple(alpn_protocols)
        self.key = key

    @classmethod
    def create(
        cls,
        *,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http2: bool | None = None,
        alpn_protocols: typing.Sequence[str] | None = None,
        check_hostname: bool | UnsetType = UNSET,
        key: typing.Hashable | None = None,
    ) -> TLSPolicy:
        """
        Build a policy from the same `verify` / `cert` / `trust_env`
        primitives that the client and transports accept.

        `check_hostname` may be set to `False` to keep verifying the server
        certificate chain while skipping the hostname match, or to `True` to
        force the hostname check on.
        """
        import ssl

        ssl_context = create_ssl_context(
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            alpn_protocols=alpn_protocols,
        )

        if not isinstance(check_hostname, UnsetType):
            if check_hostname is False:
                # Disable the hostname match while leaving certificate chain
                # verification (verify_mode) untouched.
                ssl_context.check_hostname = False
            elif check_hostname is True and not ssl_context.check_hostname:
                # Hostname verification requires certificate verification.
                if ssl_context.verify_mode == ssl.CERT_NONE:
                    ssl_context.verify_mode = ssl.CERT_REQUIRED
                ssl_context.check_hostname = True

        return cls(
            ssl_context,
            http2=http2,
            alpn_protocols=alpn_protocols,
            key=key,
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(http2={self.http2!r}, "
            f"alpn_protocols={self.alpn_protocols!r}, key={self.key!r})"
        )


class TLSPolicyResolver:
    """
    Base class for TLS policy resolvers.

    Subclasses must implement `resolve()`, which is called with the target
    `Origin` of every request before any connection is established, and must
    return a `TLSPolicy`.

    Async resolvers may additionally override `aresolve()`. The default
    implementation runs the synchronous `resolve()` in a worker thread.

    Call `invalidate()` after the configured policies have changed. This
    bumps the resolver generation; transports observe the new generation on
    their next request, clear the policy cache and shut down every pool (and
    connection) created under an older generation once in-flight requests
    have drained.
    """

    def __init__(self) -> None:
        self._generation = 0
        self._generation_lock = threading.Lock()

    def resolve(self, origin: Origin) -> TLSPolicy:
        raise NotImplementedError(
            "TLS policy resolvers must implement the 'resolve()' method."
        )

    async def aresolve(self, origin: Origin) -> TLSPolicy:
        import anyio

        return await anyio.to_thread.run_sync(self.resolve, origin)

    @property
    def generation(self) -> int:
        with self._generation_lock:
            return self._generation

    def invalidate(self) -> None:
        """
        Mark all previously resolved policies as stale.

        Subsequent requests resolve fresh policies and use new connection
        pools; pools from older generations are drained and closed.
        """
        with self._generation_lock:
            self._generation += 1


class _CallableTLSPolicyResolver(TLSPolicyResolver):
    """
    Adapter allowing a plain `Origin -> TLSPolicy` callable to be passed as a
    resolver. Such resolvers have a fixed generation, so use a
    `TLSPolicyResolver` subclass if cache invalidation is required.
    """

    def __init__(self, func: typing.Callable[[Origin], TLSPolicy]) -> None:
        super().__init__()
        self._func = func

    def resolve(self, origin: Origin) -> TLSPolicy:
        return self._func(origin)


TLSPolicyResolverTypes = typing.Union[
    "TLSPolicyResolver", typing.Callable[[Origin], TLSPolicy]
]


def coerce_tls_policy_resolver(
    resolver: TLSPolicyResolverTypes,
) -> TLSPolicyResolver:
    if isinstance(resolver, TLSPolicyResolver):
        return resolver
    if callable(resolver):
        return _CallableTLSPolicyResolver(resolver)
    raise TypeError(
        "tls_policy must be a 'TLSPolicyResolver' instance or a callable "
        f"accepting an Origin, got {type(resolver).__name__!r}."
    )
