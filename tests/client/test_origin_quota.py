from __future__ import annotations

import threading
import time
import typing

import anyio
import pytest

import httpx

ORIGIN = "https://example.com"
OTHER_ORIGIN = "https://other.example"


def wait_until(predicate: typing.Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out waiting for condition")


async def await_until(
    predicate: typing.Callable[[], bool], timeout: float = 5.0
) -> None:
    deadline = anyio.current_time() + timeout
    while True:
        if predicate():
            return
        if anyio.current_time() >= deadline:
            raise AssertionError("Timed out waiting for condition")
        await anyio.sleep(0.005)


def status_of(client: typing.Any, origin: str) -> typing.Callable[[], typing.Any]:
    """Return a live snapshot for an origin, or None before first use."""

    def get() -> typing.Any:
        try:
            return client.quota.status(origin)
        except KeyError:
            return None

    return get


def is_set(snapshot: typing.Any, *, occupied: int, queued: int) -> bool:
    return (
        snapshot is not None
        and snapshot.occupied == occupied
        and snapshot.queued == queued
    )


# ---------------------------------------------------------------------------
# Shared mock handlers
# ---------------------------------------------------------------------------


class BlockingSyncHandler:
    def __init__(self) -> None:
        self.gate = threading.Event()
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.gate.wait(10)
        finally:
            with self._lock:
                self.active -= 1
        return httpx.Response(200, json={"ok": True})


class BlockingAsyncHandler:
    def __init__(self) -> None:
        self.gate = anyio.Event()
        self.active = 0
        self.max_active = 0
        self._lock = anyio.Lock()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        async with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            await self.gate.wait()
        finally:
            async with self._lock:
                self.active -= 1
        return httpx.Response(200, json={"ok": True})


class ThreadWorkers:
    def __init__(self, count: int) -> None:
        self.results: list[typing.Any] = [None] * count
        self.errors: list[BaseException | None] = [None] * count
        self._threads: list[threading.Thread] = []

    def start(
        self,
        fn: typing.Callable[[int], typing.Any],
        stagger: float = 0.02,
    ) -> None:
        # Staggered startup keeps acquisition order == thread index, so that
        # index 0 deterministically occupies the first slot.
        for i in range(len(self.results)):
            def worker(i: int = i) -> None:
                try:
                    self.results[i] = fn(i)
                except BaseException as exc:  # report any failure to the test
                    self.errors[i] = exc

            thread = threading.Thread(target=worker)
            thread.start()
            self._threads.append(thread)
            time.sleep(stagger)

    def join(self, timeout: float = 10.0) -> None:
        for thread in self._threads:
            thread.join(timeout)
            assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_quota_config_validation() -> None:
    with pytest.raises(ValueError):
        httpx.OriginQuota(max_concurrency=0)
    with pytest.raises(ValueError):
        httpx.OriginQuota(max_concurrency=1, max_queue=-1)
    with pytest.raises(ValueError):
        httpx.OriginQuota(max_concurrency=1, queue_timeout=0)

    quota = httpx.OriginQuota(max_concurrency=2, max_queue=4, queue_timeout=3.5)
    assert (
        repr(quota)
        == "OriginQuota(max_concurrency=2, max_queue=4, queue_timeout=3.5)"
    )


def test_quota_disabled_by_default() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    assert client.quota is None
    client.close()


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


def test_sync_quota_caps_concurrency_and_queues() -> None:
    handler = BlockingSyncHandler()
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=2),
    )
    workers = ThreadWorkers(5)
    try:
        workers.start(lambda i: client.get(ORIGIN))

        get_status = status_of(client, ORIGIN)
        wait_until(lambda: is_set(get_status(), occupied=2, queued=3))
        assert get_status().origin == "https://example.com"
        assert handler.max_active == 2

        handler.gate.set()
        workers.join()

        assert all(result.status_code == 200 for result in workers.results)
        assert handler.max_active == 2
        assert get_status().occupied == 0
        assert get_status().queued == 0
        assert client.quota.statuses()
        # Default-port normalization.
        assert (
            client.quota.status("https://example.com:443").origin
            == "https://example.com"
        )
    finally:
        handler.gate.set()
        client.close()


def test_sync_quota_queue_full_is_rejected() -> None:
    handler = BlockingSyncHandler()
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1, max_queue=1),
    )
    workers = ThreadWorkers(2)
    try:
        workers.start(lambda i: client.get(ORIGIN))

        get_status = status_of(client, ORIGIN)
        wait_until(lambda: is_set(get_status(), occupied=1, queued=1))

        with pytest.raises(httpx.QuotaRejected) as raised:
            client.get(ORIGIN)
        assert raised.value.reason == httpx.QuotaRejected.REASON_QUEUE_FULL
        assert raised.value.origin == "https://example.com"
        assert raised.value.request.url.host == "example.com"
        assert get_status().rejection_reason == "queue_full"

        handler.gate.set()
        workers.join()
        assert workers.errors == [None, None]
    finally:
        handler.gate.set()
        client.close()


def test_sync_quota_queue_timeout() -> None:
    handler = BlockingSyncHandler()
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1, queue_timeout=0.1),
    )
    workers = ThreadWorkers(2)
    try:
        workers.start(lambda i: client.get(ORIGIN))

        get_status = status_of(client, ORIGIN)
        wait_until(lambda: is_set(get_status(), occupied=1, queued=1))

        wait_until(lambda: isinstance(workers.errors[1], httpx.QuotaRejected))
        assert workers.errors[1].reason == "queue_timeout"  # type: ignore[union-attr]

        handler.gate.set()
        workers.join()

        assert workers.results[0].status_code == 200
        assert get_status().occupied == 0
        assert get_status().queued == 0

        # The slot is usable again after the timeout.
        assert client.get(ORIGIN).status_code == 200
        assert get_status().occupied == 0
    finally:
        handler.gate.set()
        client.close()


def test_sync_quota_explicit_reject() -> None:
    handler = BlockingSyncHandler()
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    )
    workers = ThreadWorkers(3)
    try:
        workers.start(lambda i: client.get(ORIGIN))

        get_status = status_of(client, ORIGIN)
        wait_until(lambda: is_set(get_status(), occupied=1, queued=2))

        # Unknown origins are a no-op; in-flight requests keep running.
        assert client.quota.reject(OTHER_ORIGIN, reason="nope") == 0
        assert client.quota.reject(reason="drain") == 2

        wait_until(
            lambda: all(
                isinstance(exc, httpx.QuotaRejected) for exc in workers.errors[1:]
            )
        )
        assert workers.errors[1].reason == "drain"  # type: ignore[union-attr]
        assert workers.errors[2].reason == "drain"  # type: ignore[union-attr]
        assert workers.results[0] is None

        handler.gate.set()
        workers.join()
        assert workers.results[0].status_code == 200

        # Slots were not leaked.
        assert client.get(ORIGIN).status_code == 200
        assert get_status().occupied == 0
    finally:
        handler.gate.set()
        client.close()


def test_sync_close_rejects_waiting_requests() -> None:
    handler = BlockingSyncHandler()
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    )
    workers = ThreadWorkers(2)
    workers.start(lambda i: client.get(ORIGIN))

    get_status = status_of(client, ORIGIN)
    wait_until(lambda: is_set(get_status(), occupied=1, queued=1))

    client.close()

    wait_until(lambda: isinstance(workers.errors[1], httpx.QuotaRejected))
    assert workers.errors[1].reason == "client_closed"  # type: ignore[union-attr]

    handler.gate.set()
    workers.join()
    with pytest.raises(RuntimeError):
        client.get(ORIGIN)


def test_sync_streaming_response_holds_slot() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(b"hello"))

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    )
    try:
        request = client.build_request("GET", ORIGIN)
        response = client.send(request, stream=True)
        get_status = status_of(client, ORIGIN)
        assert get_status().occupied == 1
        assert response.read() == b"hello"
        response.close()
        assert get_status().occupied == 0
    finally:
        client.close()


def test_sync_redirect_counts_each_origin() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example":
            return httpx.Response(302, headers={"location": "http://b.example/done"})
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    )
    try:
        response = client.get("http://a.example/start")
        assert response.status_code == 200
        assert response.url.host == "b.example"

        origins = {status.origin for status in client.quota.statuses()}
        assert origins == {"http://a.example", "http://b.example"}
        assert all(status.occupied == 0 for status in client.quota.statuses())
    finally:
        client.close()


def test_sync_independent_origins() -> None:
    handler = BlockingSyncHandler()
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    )
    workers = ThreadWorkers(2)
    try:
        workers.start(lambda i: client.get(ORIGIN if i == 0 else OTHER_ORIGIN))

        get_a = status_of(client, ORIGIN)
        get_b = status_of(client, OTHER_ORIGIN)
        wait_until(
            lambda: is_set(get_a(), occupied=1, queued=0)
            and is_set(get_b(), occupied=1, queued=0)
        )
        assert handler.max_active == 2
        assert get_a().queued == 0 and get_b().queued == 0

        handler.gate.set()
        workers.join()
        assert workers.errors == [None, None]
    finally:
        handler.gate.set()
        client.close()


def test_sync_transport_error_releases_slot() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("boom")

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    )
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError):
                client.get(ORIGIN)
        status = client.quota.status(ORIGIN)
        assert status.occupied == 0
        assert status.queued == 0
    finally:
        client.close()


def test_sync_status_unknown_origin() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    )
    try:
        assert client.quota.statuses() == []
        with pytest.raises(KeyError):
            client.quota.status(ORIGIN)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


Outcome = tuple[str, bool, typing.Any]


def find_outcome(outcomes: list[Outcome], name: str) -> Outcome | None:
    return next((item for item in outcomes if item[0] == name), None)


class AsyncJobs:
    """
    Named task spawner reporting outcomes into a shared list. Tasks are
    spawned one at a time (staged), and the tests then wait on quota state
    before spawning the next role, making holder/waiter roles deterministic
    on both asyncio and trio.
    """

    def __init__(self, task_group: anyio.abc.TaskGroup) -> None:
        self._task_group = task_group
        self.outcomes: list[Outcome] = []

    async def spawn(
        self, name: str, factory: typing.Callable[[], typing.Any]
    ) -> None:
        async def runner() -> None:
            try:
                value = await factory()
                self.outcomes.append((name, True, value))
            except anyio.get_cancelled_exc_class() as exc:
                self.outcomes.append((name, False, exc))
            except BaseException as exc:
                self.outcomes.append((name, False, exc))

        self._task_group.start_soon(runner)
        await anyio.sleep(0)


@pytest.mark.anyio
async def test_async_quota_caps_concurrency_and_queues() -> None:
    handler = BlockingAsyncHandler()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=2),
    ) as client:
        get_status = status_of(client, ORIGIN)
        async with anyio.create_task_group() as task_group:
            jobs = AsyncJobs(task_group)
            await jobs.spawn("h1", lambda: client.get(ORIGIN))
            await jobs.spawn("h2", lambda: client.get(ORIGIN))
            await await_until(
                lambda: is_set(get_status(), occupied=2, queued=0)
            )
            for i in range(3):
                await jobs.spawn(f"w{i}", lambda: client.get(ORIGIN))
            await await_until(
                lambda: is_set(get_status(), occupied=2, queued=3)
            )
            assert handler.max_active == 2

            handler.gate.set()
            await await_until(lambda: len(jobs.outcomes) == 5)

        assert all(ok for _, ok, _ in jobs.outcomes)
        assert all(value.status_code == 200 for _, _, value in jobs.outcomes)
        assert get_status().occupied == 0
        assert get_status().queued == 0
        assert handler.max_active == 2


@pytest.mark.anyio
async def test_async_quota_queue_full_is_rejected() -> None:
    handler = BlockingAsyncHandler()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1, max_queue=1),
    ) as client:
        get_status = status_of(client, ORIGIN)
        async with anyio.create_task_group() as task_group:
            jobs = AsyncJobs(task_group)
            await jobs.spawn("holder", lambda: client.get(ORIGIN))
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=0)
            )
            await jobs.spawn("waiter", lambda: client.get(ORIGIN))
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=1)
            )

            with pytest.raises(httpx.QuotaRejected) as raised:
                await client.get(ORIGIN)
            assert raised.value.reason == "queue_full"
            assert raised.value.origin == "https://example.com"
            assert get_status().rejection_reason == "queue_full"

            handler.gate.set()
            await await_until(lambda: len(jobs.outcomes) == 2)

        assert all(ok for _, ok, _ in jobs.outcomes)


@pytest.mark.anyio
async def test_async_quota_queue_timeout() -> None:
    handler = BlockingAsyncHandler()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1, queue_timeout=0.1),
    ) as client:
        get_status = status_of(client, ORIGIN)
        async with anyio.create_task_group() as task_group:
            jobs = AsyncJobs(task_group)
            await jobs.spawn("holder", lambda: client.get(ORIGIN))
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=0)
            )
            await jobs.spawn("waiter", lambda: client.get(ORIGIN))
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=1)
            )

            # The queued request times out before the gate opens.
            await await_until(
                lambda: find_outcome(jobs.outcomes, "waiter") is not None
            )
            _, ok, value = find_outcome(jobs.outcomes, "waiter")  # type: ignore[misc]
            assert ok is False
            assert isinstance(value, httpx.QuotaRejected)
            assert value.reason == "queue_timeout"
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=0)
            )

            handler.gate.set()
            await await_until(
                lambda: find_outcome(jobs.outcomes, "holder") is not None
            )
            _, ok, value = find_outcome(jobs.outcomes, "holder")  # type: ignore[misc]
            assert ok is True
            assert value.status_code == 200

        assert get_status().occupied == 0
        assert get_status().queued == 0

        # The slot is usable again after the timeout.
        assert (await client.get(ORIGIN)).status_code == 200
        assert get_status().occupied == 0


@pytest.mark.anyio
async def test_async_quota_cancelled_while_waiting() -> None:
    handler = BlockingAsyncHandler()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    ) as client:
        get_status = status_of(client, ORIGIN)
        scope_box: dict[str, anyio.CancelScope] = {}

        async def holder() -> httpx.Response:
            return await client.get(ORIGIN)

        outcomes: list[Outcome] = []

        async def waiter() -> None:
            # The outcome must be observed *inside* the cancel scope: the
            # scope's own CancelledError is absorbed at the scope boundary.
            with anyio.CancelScope() as scope:
                scope_box["waiter"] = scope
                try:
                    response = await client.get(ORIGIN)
                except anyio.get_cancelled_exc_class() as exc:
                    outcomes.append(("waiter", False, exc))
                    raise
                except BaseException as exc:
                    outcomes.append(("waiter", False, exc))
                    raise
                else:
                    outcomes.append(("waiter", True, response))

        async with anyio.create_task_group() as task_group:
            jobs = AsyncJobs(task_group)
            await jobs.spawn("holder", holder)
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=0)
            )
            task_group.start_soon(waiter)
            await anyio.sleep(0)
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=1)
            )

            scope_box["waiter"].cancel()
            await await_until(
                lambda: find_outcome(outcomes, "waiter") is not None
            )
            _, ok, value = find_outcome(outcomes, "waiter")  # type: ignore[misc]
            assert ok is False
            assert isinstance(value, anyio.get_cancelled_exc_class())
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=0)
            )

            # A new request can queue normally; cancellation leaked nothing.
            await jobs.spawn("next", lambda: client.get(ORIGIN))
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=1)
            )
            handler.gate.set()
            await await_until(
                lambda: find_outcome(jobs.outcomes, "holder") is not None
                and find_outcome(jobs.outcomes, "next") is not None
            )

        assert all(item[1] for item in jobs.outcomes)
        assert all(item[2].status_code == 200 for item in jobs.outcomes)
        assert get_status().occupied == 0


@pytest.mark.anyio
async def test_async_quota_explicit_reject() -> None:
    handler = BlockingAsyncHandler()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    ) as client:
        get_status = status_of(client, ORIGIN)
        async with anyio.create_task_group() as task_group:
            jobs = AsyncJobs(task_group)
            await jobs.spawn("holder", lambda: client.get(ORIGIN))
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=0)
            )
            await jobs.spawn("w1", lambda: client.get(ORIGIN))
            await jobs.spawn("w2", lambda: client.get(ORIGIN))
            await await_until(
                lambda: is_set(get_status(), occupied=1, queued=2)
            )

            assert await client.quota.reject(OTHER_ORIGIN, reason="nope") == 0
            assert await client.quota.reject(reason="drain") == 2

            await await_until(lambda: len(jobs.outcomes) == 2)
            assert all(
                not ok
                and isinstance(value, httpx.QuotaRejected)
                and value.reason == "drain"
                for _, ok, value in jobs.outcomes
            )

            handler.gate.set()
            await await_until(lambda: len(jobs.outcomes) == 3)
            _, ok, value = find_outcome(jobs.outcomes, "holder")  # type: ignore[misc]
            assert ok is True
            assert value.status_code == 200

        assert get_status().occupied == 0
        assert (await client.get(ORIGIN)).status_code == 200


@pytest.mark.anyio
async def test_async_aclose_rejects_waiting_requests() -> None:
    handler = BlockingAsyncHandler()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    )
    get_status = status_of(client, ORIGIN)
    async with anyio.create_task_group() as task_group:
        jobs = AsyncJobs(task_group)
        await jobs.spawn("holder", lambda: client.get(ORIGIN))
        await await_until(lambda: is_set(get_status(), occupied=1, queued=0))
        await jobs.spawn("waiter", lambda: client.get(ORIGIN))
        await await_until(lambda: is_set(get_status(), occupied=1, queued=1))

        await client.aclose()

        await await_until(lambda: len(jobs.outcomes) == 1)
        _, ok, value = jobs.outcomes[0]
        assert ok is False
        assert isinstance(value, httpx.QuotaRejected)
        assert value.reason == "client_closed"

        handler.gate.set()
        await await_until(lambda: len(jobs.outcomes) == 2)

    _, ok, value = find_outcome(jobs.outcomes, "holder")  # type: ignore[misc]
    assert ok is True
    with pytest.raises(RuntimeError):
        await client.get(ORIGIN)


@pytest.mark.anyio
async def test_async_streaming_response_holds_slot() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(b"hello"))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    ) as client:
        get_status = status_of(client, ORIGIN)
        request = client.build_request("GET", ORIGIN)
        response = await client.send(request, stream=True)
        assert get_status().occupied == 1
        await response.aclose()
        assert get_status().occupied == 0


@pytest.mark.anyio
async def test_async_redirect_counts_each_origin() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example":
            return httpx.Response(302, headers={"location": "http://b.example/done"})
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    ) as client:
        response = await client.get("http://a.example/start")
        assert response.status_code == 200
        assert response.url.host == "b.example"

        origins = {item.origin for item in client.quota.statuses()}
        assert origins == {"http://a.example", "http://b.example"}
        assert all(item.occupied == 0 for item in client.quota.statuses())


@pytest.mark.anyio
async def test_async_independent_origins() -> None:
    handler = BlockingAsyncHandler()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    ) as client:
        get_a = status_of(client, ORIGIN)
        get_b = status_of(client, OTHER_ORIGIN)
        async with anyio.create_task_group() as task_group:
            jobs = AsyncJobs(task_group)
            await jobs.spawn("a", lambda: client.get(ORIGIN))
            await jobs.spawn("b", lambda: client.get(OTHER_ORIGIN))
            await await_until(
                lambda: is_set(get_a(), occupied=1, queued=0)
                and is_set(get_b(), occupied=1, queued=0)
            )
            assert handler.max_active == 2
            handler.gate.set()
            await await_until(lambda: len(jobs.outcomes) == 2)

        assert all(ok for _, ok, _ in jobs.outcomes)


@pytest.mark.anyio
async def test_async_transport_error_releases_slot() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("boom")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        origin_quota=httpx.OriginQuota(max_concurrency=1),
    ) as client:
        get_status = status_of(client, ORIGIN)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await client.get(ORIGIN)
        assert get_status().occupied == 0
        assert get_status().queued == 0


# ---------------------------------------------------------------------------
# Live-server integration (real connection pool, no mocks)
# ---------------------------------------------------------------------------


def test_sync_quota_live_server(server) -> None:
    with httpx.Client(
        origin_quota=httpx.OriginQuota(max_concurrency=2)
    ) as client:
        response = client.get(server.url)
        assert response.status_code == 200
        status = client.quota.status(str(server.url.copy_with(path="/")))
        assert status.occupied == 0
        assert status.queued == 0


@pytest.mark.anyio
async def test_async_quota_live_server(server) -> None:
    async with httpx.AsyncClient(
        origin_quota=httpx.OriginQuota(max_concurrency=2)
    ) as client:
        async with anyio.create_task_group() as task_group:
            jobs = AsyncJobs(task_group)
            for i in range(4):
                await jobs.spawn(str(i), lambda: client.get(server.url))
            await await_until(lambda: len(jobs.outcomes) == 4, timeout=15)

        assert all(ok for _, ok, _ in jobs.outcomes)
        assert all(value.status_code == 200 for _, _, value in jobs.outcomes)
        status = client.quota.status(str(server.url.copy_with(path="/")))
        assert status.occupied == 0
        assert status.queued == 0


@pytest.mark.anyio
async def test_async_quota_live_server_http2_stack(server) -> None:
    # Exercises the quota in front of the HTTP/2-enabled transport stack;
    # connection pooling/multiplexing remains the transport's responsibility.
    async with httpx.AsyncClient(
        http2=True, origin_quota=httpx.OriginQuota(max_concurrency=1)
    ) as client:
        first = await client.get(server.url)
        assert first.status_code == 200
        second = await client.get(server.url)
        assert second.status_code == 200

        status = client.quota.status(str(server.url.copy_with(path="/")))
        assert status.occupied == 0
        assert status.queued == 0
