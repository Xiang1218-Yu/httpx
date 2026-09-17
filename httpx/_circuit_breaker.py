"""
Client-level circuit breaker support.

The circuit breaker keeps independent failure statistics for each origin
(scheme + host + port). When an origin accumulates too many consecutive
failures the breaker opens, and any further request to that origin fails
fast, without touching the network. Once a cooldown period has elapsed a
single probe request is allowed through; depending on its outcome the
breaker either closes again (service recovered) or reopens (still down).

The implementation is shared between the synchronous and asynchronous
clients. All state transitions are guarded by a plain threading lock and
the critical sections never perform any I/O, so sharing a single breaker
between a `Client` and an `AsyncClient` is supported.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import typing

from ._exceptions import CircuitBreakerOpen, NetworkError, TimeoutException
from ._types import AsyncByteStream, SyncByteStream
from ._urls import URL

if typing.TYPE_CHECKING:
    from ._models import Request  # pragma: no cover

__all__ = ["CircuitBreaker", "CircuitState"]


logger = logging.getLogger("httpx")

FailureStatusCodes = typing.Union[typing.Iterable[int], typing.Callable[[int], bool]]
Origin = typing.Tuple[str, str, typing.Optional[int]]

# Status codes treated as failures unless overridden: every 5xx response.
DEFAULT_FAILURE_STATUS_CODES = range(500, 600)


class CircuitState(enum.Enum):
    """
    The state of a single origin's circuit breaker.

    * `CLOSED` - requests are allowed and failures are counted.
    * `OPEN` - requests fail fast until the recovery timeout elapses.
    * `HALF_OPEN` - a single probe request is in flight.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _OriginBreaker:
    __slots__ = ("state", "failure_count", "opened_at", "probe_in_flight")

    def __init__(self) -> None:
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.opened_at: float = 0.0
        self.probe_in_flight: bool = False


def _origin_key(url: URL) -> Origin:
    port = url.port
    if port is None:
        port = {"http": 80, "https": 443}.get(url.scheme)
    return (url.scheme, url.host, port)


def _origin_label(origin: Origin) -> str:
    scheme, host, port = origin
    if (
        port is None
        or (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


class CircuitBreaker:
    """
    An optional client-level circuit breaker policy.

    A single `CircuitBreaker` may be passed to a `Client` and/or an
    `AsyncClient`. Failures are tracked independently per origin
    (scheme/host/port).

    **Parameters:**

    * **failure_threshold** - The number of consecutive failures (connection
      errors, timeouts and configured failure status codes) after which the
      circuit opens and requests start failing fast. Defaults to 5.
    * **recovery_timeout** - Seconds to wait while the circuit is open before
      allowing a single probe request through. Defaults to 30 seconds.
    * **failure_status_codes** - Either an iterable of response status codes
      that should be treated as failures, or a callable taking a status code
      and returning a boolean. Defaults to all `5xx` status codes.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        failure_status_codes: FailureStatusCodes = DEFAULT_FAILURE_STATUS_CODES,
    ) -> None:
        if not isinstance(failure_threshold, int) or failure_threshold < 1:
            raise ValueError(
                "failure_threshold must be a positive integer, "
                f"got {failure_threshold!r}."
            )
        if recovery_timeout < 0:
            raise ValueError(
                f"recovery_timeout must be zero or positive, got {recovery_timeout!r}."
            )

        if callable(failure_status_codes):
            self._is_failure_status = failure_status_codes
        else:
            failure_codes = frozenset(failure_status_codes)
            self._is_failure_status = failure_codes.__contains__

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._lock = threading.Lock()
        self._origins: dict[Origin, _OriginBreaker] = {}

    def is_failure_status(self, status_code: int) -> bool:
        """
        Return `True` if the given response status code counts as a failure.
        """
        return self._is_failure_status(status_code)

    def state_for(self, url: URL | str | Request) -> CircuitState | None:
        """
        Return the current breaker state for the origin of the given URL
        or request, or `None` if no request has yet been made to it.
        """
        if hasattr(url, "url"):
            target_url = typing.cast("Request", url).url
        else:
            target_url = URL(url)
        key = _origin_key(target_url)
        with self._lock:
            origin = self._origins.get(key)
            return None if origin is None else origin.state

    def before_request(self, request: Request) -> None:
        """
        Gate a request attempt.

        Raises `CircuitBreakerOpen` (without contacting the transport) when
        the origin's circuit is open, or while a half-open probe is already
        in flight.
        """
        key = _origin_key(request.url)
        now = time.monotonic()
        with self._lock:
            origin = self._origins.get(key)
            if origin is None:
                # First request to this origin: nothing to gate against.
                self._origins[key] = _OriginBreaker()
                return

            if origin.state == CircuitState.OPEN:
                if now - origin.opened_at < self.recovery_timeout:
                    raise self._open_error(request, key)
                # Cooldown elapsed: allow exactly one probe request.
                origin.state = CircuitState.HALF_OPEN
                origin.probe_in_flight = True
                logger.debug(
                    "Circuit breaker for %s entering half-open state, "
                    "allowing a probe request.",
                    _origin_label(key),
                )
            elif origin.state == CircuitState.HALF_OPEN:
                if origin.probe_in_flight:
                    raise self._open_error(request, key)
                origin.probe_in_flight = True

    def record_success(self, request: Request) -> None:
        """
        Record a successful request outcome for the request's origin.
        """
        key = _origin_key(request.url)
        with self._lock:
            origin = self._origins.get(key)
            if origin is None:
                return

            if origin.state == CircuitState.HALF_OPEN:
                origin.state = CircuitState.CLOSED
                origin.failure_count = 0
                origin.probe_in_flight = False
                logger.debug(
                    "Circuit breaker for %s closed after a successful probe.",
                    _origin_label(key),
                )
            elif origin.state == CircuitState.CLOSED:
                origin.failure_count = 0
            # A late success from a request that was in flight when the
            # circuit opened does not provide fresh information, so OPEN
            # state is left untouched.

    def record_failure(self, request: Request) -> None:
        """
        Record a failed request outcome (connection error, timeout or
        configured failure status code) for the request's origin.
        """
        key = _origin_key(request.url)
        now = time.monotonic()
        with self._lock:
            origin = self._origins.get(key)
            if origin is None:
                origin = _OriginBreaker()
                self._origins[key] = origin

            if origin.state == CircuitState.OPEN:
                # Outcomes of stale, pre-trip in-flight requests are ignored.
                return

            if origin.state == CircuitState.HALF_OPEN:
                self._open(origin, now, key)
                origin.probe_in_flight = False
                return

            origin.failure_count += 1
            if origin.failure_count >= self.failure_threshold:
                self._open(origin, now, key)

    def record_inconclusive(self, request: Request) -> None:
        """
        Record that a request ended without a definite outcome (e.g. a
        streamed response closed before its body was consumed, or an
        exception outside the counted failure types).

        A half-open probe with an unknown outcome fails safe: the circuit
        reopens and a fresh probe is scheduled after the cooldown, so the
        breaker can never get stuck in the half-open state. Inconclusive
        outcomes in other states are ignored.
        """
        key = _origin_key(request.url)
        now = time.monotonic()
        with self._lock:
            origin = self._origins.get(key)
            if origin is None or origin.state != CircuitState.HALF_OPEN:
                return
            self._open(origin, now, key)
            origin.probe_in_flight = False

    def _open(self, origin: _OriginBreaker, now: float, key: Origin) -> None:
        origin.state = CircuitState.OPEN
        origin.opened_at = now
        origin.probe_in_flight = False
        logger.info(
            "Circuit breaker for %s opened after %d consecutive failure(s); "
            "requests will fail fast for %.1fs.",
            _origin_label(key),
            origin.failure_count or self.failure_threshold,
            self.recovery_timeout,
        )

    def _open_error(self, request: Request, key: Origin) -> CircuitBreakerOpen:
        return CircuitBreakerOpen(
            f"Circuit breaker for {_origin_label(key)} is open; "
            "request failed fast without being sent.",
            request=request,
        )


class _CircuitBreakerTrackedStream:
    """
    Shared bookkeeping for response streams.

    The breaker only learns the outcome of a request once its response body
    has been consumed:

    * a network error or timeout while reading the body is a failure;
    * a fully consumed body is a success, unless the response status code
      is configured as a failure;
    * closing the stream before the body has been consumed (the streaming
      case) leaves the breaker untouched - a streamed response is only
      counted when the request genuinely fails.
    """

    def __init__(
        self,
        *,
        breaker: CircuitBreaker,
        request: Request,
        status_code: int,
    ) -> None:
        self._breaker: CircuitBreaker = breaker
        self._request = request
        self._status_code = status_code
        self._reported = False
        # Streaming responses are opted out of status-code based accounting
        # by the client once it knows the response will be handed back
        # unread (see `Client.send(..., stream=True)`).
        self.count_status_failures = True

    def settle(self) -> None:
        """
        Report the outcome of a fully buffered response (one whose body
        will not be iterated, e.g. one returned by `MockTransport`).
        Idempotent.
        """
        self._report_completion()

    def dismiss(self) -> None:
        """
        Give up on outcome bookkeeping for a response whose stream will
        never be driven (e.g. a buffered response handed back in streaming
        mode). No failure or success is recorded, but an in-flight
        half-open probe is released.
        """
        if self._reported:
            return
        self._reported = True
        self._breaker.record_inconclusive(self._request)

    def _report_completion(self) -> None:
        if self._reported:
            return
        self._reported = True
        if self.count_status_failures and self._breaker.is_failure_status(
            self._status_code
        ):
            self._breaker.record_failure(self._request)
        else:
            self._breaker.record_success(self._request)

    def _report_failure(self) -> None:
        if self._reported:
            return
        self._reported = True
        self._breaker.record_failure(self._request)

    def _mark_settled(self) -> None:
        # The stream was closed before its body was exhausted; the request's
        # outcome is undetermined. No failure/success statistic is recorded,
        # but an in-flight half-open probe is released so the breaker cannot
        # get stuck waiting on it.
        if self._reported:
            return
        self._reported = True
        self._breaker.record_inconclusive(self._request)


# Transport-level failures accounted for while reading a response body:
# network/connection errors and timeouts. Failure status codes are handled
# separately once the body has been consumed.
_STREAM_FAILURES = (NetworkError, TimeoutException)


class _CircuitBreakerSyncStream(SyncByteStream, _CircuitBreakerTrackedStream):
    def __init__(
        self,
        stream: SyncByteStream,
        *,
        breaker: CircuitBreaker,
        request: Request,
        status_code: int,
    ) -> None:
        _CircuitBreakerTrackedStream.__init__(
            self,
            breaker=breaker,
            request=request,
            status_code=status_code,
        )
        self._stream = stream

    def __iter__(self) -> typing.Iterator[bytes]:
        try:
            for chunk in self._stream:
                yield chunk
        except _STREAM_FAILURES:
            self._report_failure()
            raise
        else:
            self._report_completion()

    def close(self) -> None:
        try:
            self._stream.close()
        except _STREAM_FAILURES:
            self._report_failure()
            raise
        else:
            self._mark_settled()


class _CircuitBreakerAsyncStream(AsyncByteStream, _CircuitBreakerTrackedStream):
    def __init__(
        self,
        stream: AsyncByteStream,
        *,
        breaker: CircuitBreaker,
        request: Request,
        status_code: int,
    ) -> None:
        _CircuitBreakerTrackedStream.__init__(
            self,
            breaker=breaker,
            request=request,
            status_code=status_code,
        )
        self._stream = stream

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                yield chunk
        except _STREAM_FAILURES:
            self._report_failure()
            raise
        else:
            self._report_completion()

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        except _STREAM_FAILURES:
            self._report_failure()
            raise
        else:
            self._mark_settled()
