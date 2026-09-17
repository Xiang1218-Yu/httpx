from __future__ import annotations

import time
import typing

import anyio
import pytest

import httpx
from tests.concurrency import sleep

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FailingSyncStream(httpx.SyncByteStream):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __iter__(self) -> typing.Iterator[bytes]:
        yield b"partial body"
        raise self._exc


class FailingAsyncStream(httpx.AsyncByteStream):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        yield b"partial body"
        raise self._exc


def build_sync_client(
    handler: typing.Callable[[httpx.Request], httpx.Response],
    **kwargs: typing.Any,
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)


def build_async_client(
    handler: typing.Callable[[httpx.Request], typing.Any],
    **kwargs: typing.Any,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        httpx.CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        httpx.CircuitBreaker(recovery_timeout=-1.0)
    with pytest.raises(TypeError):
        httpx.Client(circuit_breaker="boom")  # type: ignore[arg-type]


def test_default_configuration_counts_all_5xx() -> None:
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return httpx.Response(599)

    breaker = httpx.CircuitBreaker(failure_threshold=2, recovery_timeout=60)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        for _ in range(2):
            assert client.get("https://example.org/").status_code == 599
        with pytest.raises(httpx.CircuitBreakerOpen):
            client.get("https://example.org/")
    assert len(seen) == 2


def test_configurable_failure_status_codes_iterable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500 if request.url.path == "/500" else 503)

    breaker = httpx.CircuitBreaker(
        failure_threshold=1, recovery_timeout=60, failure_status_codes=[503]
    )
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        # 500 is not a configured failure status.
        assert client.get("https://example.org/500").status_code == 500
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED
        assert client.get("https://example.org/503").status_code == 503
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN
        with pytest.raises(httpx.CircuitBreakerOpen):
            client.get("https://example.org/503")


def test_configurable_failure_status_codes_callable() -> None:
    def only_503(status_code: int) -> bool:
        return status_code == 503

    breaker = httpx.CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=60,
        failure_status_codes=only_503,
    )
    assert breaker.is_failure_status(503)
    assert not breaker.is_failure_status(500)


def test_4xx_and_success_do_not_trip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    breaker = httpx.CircuitBreaker(failure_threshold=1)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        for _ in range(10):
            assert client.get("https://example.org/").status_code == 404
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED


def test_success_resets_consecutive_failures() -> None:
    statuses = iter([503, 503, 200, 503, 503])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses))

    breaker = httpx.CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        for _ in range(5):
            client.get("https://example.org/")
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Failure accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.WriteTimeout("write timed out"),
        httpx.PoolTimeout("pool timed out"),
        httpx.ReadError("connection reset"),
        httpx.WriteError("broken pipe"),
    ],
)
def test_transport_errors_trip_the_breaker(exc: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    breaker = httpx.CircuitBreaker(failure_threshold=2, recovery_timeout=60)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        for _ in range(2):
            with pytest.raises(httpx.TransportError):
                client.get("https://example.org/")
        with pytest.raises(httpx.CircuitBreakerOpen):
            client.get("https://example.org/")


def test_non_transport_errors_are_not_counted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.UnsupportedProtocol("nope")

    breaker = httpx.CircuitBreaker(failure_threshold=1)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        for _ in range(3):
            with pytest.raises(httpx.UnsupportedProtocol):
                client.get("ftp://example.org/")
        assert breaker.state_for("ftp://example.org/") == httpx.CircuitState.CLOSED


def test_fast_fail_does_not_contact_transport() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(503)

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        client.get("https://example.org/")
        with pytest.raises(httpx.CircuitBreakerOpen):
            client.get("https://example.org/")
    assert calls == ["https://example.org/"]


def test_circuit_breaker_open_is_a_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        client.get("https://example.org/")
        with pytest.raises(httpx.HTTPError) as exc_info:
            client.get("https://example.org/")
    assert isinstance(exc_info.value, httpx.CircuitBreakerOpen)
    assert exc_info.value.request.url == "https://example.org/"


# ---------------------------------------------------------------------------
# Per-origin isolation
# ---------------------------------------------------------------------------


def test_breakers_are_independent_per_origin() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503 if request.url.host == "down.example" else 200)

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        client.get("https://down.example/")
        assert breaker.state_for("https://down.example/") == httpx.CircuitState.OPEN
        with pytest.raises(httpx.CircuitBreakerOpen):
            client.get("https://down.example/")

        # A different host keeps working.
        assert client.get("https://up.example/").status_code == 200
        assert breaker.state_for("https://up.example/") == httpx.CircuitState.CLOSED

        # http vs https and distinct ports are distinct origins.
        assert breaker.state_for("http://down.example/") is None
        assert breaker.state_for("https://down.example:8443/") is None


# ---------------------------------------------------------------------------
# Open / half-open / closed state machine
# ---------------------------------------------------------------------------


def test_cooldown_allows_single_probe_which_recovers() -> None:
    status = {"code": 503}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status["code"])

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=0.02)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        client.get("https://example.org/")
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN
        with pytest.raises(httpx.CircuitBreakerOpen):
            client.get("https://example.org/")

        time.sleep(0.05)
        status["code"] = 200
        assert client.get("https://example.org/").status_code == 200
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED
        # Fully recovered: further requests pass.
        assert client.get("https://example.org/").status_code == 200


def test_failed_probe_reopens_the_circuit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=0.02)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        client.get("https://example.org/")
        time.sleep(0.05)
        assert client.get("https://example.org/").status_code == 503
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN
        with pytest.raises(httpx.CircuitBreakerOpen):
            client.get("https://example.org/")


def test_half_open_state_machine() -> None:
    breaker = httpx.CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
    request = httpx.Request("GET", "https://example.org/")

    breaker.before_request(request)
    breaker.record_failure(request)
    assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED
    breaker.record_failure(request)
    assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN

    # Cooldown elapsed: exactly one probe request is allowed through.
    breaker.before_request(request)
    assert breaker.state_for("https://example.org/") == httpx.CircuitState.HALF_OPEN
    with pytest.raises(httpx.CircuitBreakerOpen):
        breaker.before_request(request)

    # A successful probe closes the circuit again.
    breaker.record_success(request)
    assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED


def test_half_open_failure_reopens() -> None:
    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
    request = httpx.Request("GET", "https://example.org/")
    breaker.record_failure(request)
    breaker.before_request(request)
    assert breaker.state_for("https://example.org/") == httpx.CircuitState.HALF_OPEN
    breaker.record_failure(request)
    assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN


def test_inconclusive_probe_does_not_stick_half_open() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("malformed http")

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        # Trip the breaker first.
        breaker.record_failure(httpx.Request("GET", "https://example.org/"))
        # The probe raises an exception that is not counted as a failure;
        # the breaker reopens instead of wedging in half-open.
        with pytest.raises(httpx.RemoteProtocolError):
            client.get("https://example.org/")
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN
        # A fresh probe is allowed straight away (recovery_timeout is 0);
        # it, too, is inconclusive and fails the circuit back open.
        with pytest.raises(httpx.RemoteProtocolError):
            client.get("https://example.org/")
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN


def test_unread_streamed_probe_reopens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        breaker.record_failure(httpx.Request("GET", "https://example.org/"))
        # Probe as a streaming request that never reads the body: the
        # outcome is unknown, so the breaker fails safe back to OPEN.
        with client.stream("GET", "https://example.org/"):
            pass
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN


@pytest.mark.anyio
async def test_half_open_allows_only_one_concurrent_probe() -> None:
    started = anyio.Event()
    finish = anyio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await finish.wait()
        return httpx.Response(200)

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
    # Trip the circuit, using a request directly on the breaker.
    trip_request = httpx.Request("GET", "https://example.org/")
    breaker.record_failure(trip_request)

    probe_result: dict[str, httpx.Response] = {}

    async def probe() -> None:
        probe_result["response"] = await client.get("https://example.org/")

    async with build_async_client(handler, circuit_breaker=breaker) as client:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(probe)
            await started.wait()
            assert (
                breaker.state_for("https://example.org/")
                == httpx.CircuitState.HALF_OPEN
            )

            # While the probe is in flight, concurrent requests fail fast.
            with pytest.raises(httpx.CircuitBreakerOpen):
                await client.get("https://example.org/")

            finish.set()

    assert probe_result["response"].status_code == 200
    assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Streaming responses
# ---------------------------------------------------------------------------


def test_streamed_failure_status_is_not_counted_sync() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    breaker = httpx.CircuitBreaker(failure_threshold=1)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        for _ in range(5):
            with client.stream("GET", "https://example.org/") as response:
                response.read()
                assert response.status_code == 503
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED
    assert len(calls) == 5


def test_unread_stream_is_neither_failure_nor_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    breaker = httpx.CircuitBreaker(failure_threshold=2)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        # A streamed response closed without reading contributes nothing.
        with client.stream("GET", "https://example.org/"):
            pass
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED
        # Buffered failures are still counted normally.
        assert client.get("https://example.org/").status_code == 503
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED
        assert client.get("https://example.org/").status_code == 503
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN


def test_stream_body_failure_is_counted_sync() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=FailingSyncStream(httpx.ReadError("reset")))

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        with pytest.raises(httpx.ReadError):
            with client.stream("GET", "https://example.org/") as response:
                response.read()
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN
        with pytest.raises(httpx.CircuitBreakerOpen):
            client.get("https://example.org/")


def test_stream_connect_failure_is_counted_sync() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        with pytest.raises(httpx.ConnectError):
            with client.stream("GET", "https://example.org/"):
                pass
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN


@pytest.mark.anyio
async def test_streamed_failure_status_is_not_counted_async() -> None:
    calls: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    breaker = httpx.CircuitBreaker(failure_threshold=1)
    async with build_async_client(handler, circuit_breaker=breaker) as client:
        for _ in range(5):
            async with client.stream("GET", "https://example.org/") as response:
                await response.aread()
                assert response.status_code == 503
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED
    assert len(calls) == 5


@pytest.mark.anyio
async def test_stream_body_failure_is_counted_async() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=FailingAsyncStream(httpx.ReadTimeout("slow")))

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    async with build_async_client(handler, circuit_breaker=breaker) as client:
        with pytest.raises(httpx.ReadTimeout):
            async with client.stream("GET", "https://example.org/") as response:
                await response.aread()
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN
        with pytest.raises(httpx.CircuitBreakerOpen):
            await client.get("https://example.org/")


# ---------------------------------------------------------------------------
# Redirects, hooks and sharing between sync/async clients
# ---------------------------------------------------------------------------


def test_redirects_are_followed_unchanged() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(200, text="ok")

    breaker = httpx.CircuitBreaker(failure_threshold=1)
    with build_sync_client(
        handler, circuit_breaker=breaker, follow_redirects=True
    ) as client:
        response = client.get("https://example.org/redirect")
        assert response.status_code == 200
        assert response.text == "ok"
        assert [r.status_code for r in response.history] == [302]
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED


def test_redirect_status_is_not_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/elsewhere"})

    breaker = httpx.CircuitBreaker(failure_threshold=1)
    with build_sync_client(handler, circuit_breaker=breaker) as client:
        for _ in range(5):
            response = client.get("https://example.org/redirect")
            assert response.status_code == 302
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED


def test_redirect_to_open_origin_fails_fast() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "down.example":
            return httpx.Response(503)
        return httpx.Response(302, headers={"location": "https://down.example/final"})

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    with build_sync_client(
        handler, circuit_breaker=breaker, follow_redirects=True
    ) as client:
        # Trip the breaker on the redirect target.
        assert client.get("https://down.example/").status_code == 503
        with pytest.raises(httpx.CircuitBreakerOpen) as exc_info:
            client.get("https://example.org/redirect")
        assert exc_info.value.request.url.host == "down.example"


def test_event_hooks_fire_normally() -> None:
    request_hooks: list[str] = []
    response_hooks: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    hooks = {
        "request": [lambda request: request_hooks.append(str(request.url))],
        "response": [lambda response: response_hooks.append(response.status_code)],
    }
    with build_sync_client(
        handler, circuit_breaker=breaker, event_hooks=hooks
    ) as client:
        client.get("https://example.org/")
        # The request hook still fires for the attempt that fails fast;
        # the response hook does not (no response was received).
        request_hooks.clear()
        response_hooks.clear()
        with pytest.raises(httpx.CircuitBreakerOpen):
            client.get("https://example.org/")
        assert request_hooks == ["https://example.org/"]
        assert response_hooks == []


@pytest.mark.anyio
async def test_breaker_shared_between_sync_and_async_clients() -> None:
    def sync_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=60)

    with build_sync_client(sync_handler, circuit_breaker=breaker) as sync_client:
        with pytest.raises(httpx.ConnectError):
            sync_client.get("https://down.example/")
        assert breaker.state_for("https://down.example/") == httpx.CircuitState.OPEN

        async with build_async_client(
            async_handler, circuit_breaker=breaker
        ) as async_client:
            # The async client shares the same breaker state.
            with pytest.raises(httpx.CircuitBreakerOpen):
                await async_client.get("https://down.example/")
            # Other origins are unaffected for both clients.
            assert (await async_client.get("https://up.example/")).status_code == 200


# ---------------------------------------------------------------------------
# Async parity for the basic open/recover flow
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_open_and_probe_recovery() -> None:
    status = {"code": 503}

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status["code"])

    breaker = httpx.CircuitBreaker(failure_threshold=1, recovery_timeout=0.02)
    async with build_async_client(handler, circuit_breaker=breaker) as client:
        await client.get("https://example.org/")
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.OPEN
        with pytest.raises(httpx.CircuitBreakerOpen):
            await client.get("https://example.org/")

        await sleep(0.05)
        status["code"] = 200
        assert (await client.get("https://example.org/")).status_code == 200
        assert breaker.state_for("https://example.org/") == httpx.CircuitState.CLOSED


@pytest.mark.anyio
async def test_async_transport_errors_trip() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    breaker = httpx.CircuitBreaker(failure_threshold=2, recovery_timeout=60)
    async with build_async_client(handler, circuit_breaker=breaker) as client:
        for _ in range(2):
            with pytest.raises(httpx.ReadTimeout):
                await client.get("https://example.org/")
        with pytest.raises(httpx.CircuitBreakerOpen):
            await client.get("https://example.org/")
