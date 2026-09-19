"""
Per-origin concurrency quotas.

The quota sits at the client boundary, in front of the connection pool.
Each request must first acquire a slot for the origin it is being sent
to. When every slot for an origin is occupied, additional requests wait
in a bounded FIFO queue. Waiting requests may time out, be cancelled, be
explicitly rejected, or be rejected when the client is closed.

The quota is deliberately independent of the transport layer, so
connection pooling, HTTP/2 multiplexing and proxy handling keep their
existing behaviour.
"""

from __future__ import annotations

import threading
import typing
from collections import deque

import anyio

from ._exceptions import QuotaRejected
from ._types import AsyncByteStream, SyncByteStream
from ._urls import URL

__all__ = [
    "AsyncOriginQuotaLimiter",
    "OriginQuota",
    "QuotaStatus",
    "SyncOriginQuotaLimiter",
]

REASON_QUEUE_FULL = QuotaRejected.REASON_QUEUE_FULL
REASON_QUEUE_TIMEOUT = QuotaRejected.REASON_QUEUE_TIMEOUT
REASON_CLIENT_CLOSED = QuotaRejected.REASON_CLIENT_CLOSED
REASON_REJECTED = QuotaRejected.REASON_REJECTED


class Origin(typing.NamedTuple):
    scheme: str
    host: str
    port: int

    def __str__(self) -> str:
        default_port = 443 if self.scheme == "https" else 80
        port = "" if self.port == default_port else f":{self.port}"
        return f"{self.scheme}://{self.host}{port}"


def origin_from_url(url: URL) -> Origin:
    port = url.port
    if port is None:
        port = 443 if url.scheme == "https" else 80
    return Origin(scheme=url.scheme, host=url.host, port=port)


class OriginQuota:
    """
    Configuration for per-origin concurrency quotas.

    Requests beyond `max_concurrency` for a given origin wait in a FIFO
    queue, instead of all being dispatched onto the connection pool and
    the server at once.

    **Parameters:**

    * **max_concurrency** - The maximum number of in-flight requests
            allowed per origin at any given time.
    * **max_queue** - The maximum number of requests that may wait for a
            slot per origin. Requests beyond this are rejected immediately
            with reason `"queue_full"`. Set to `None` for an unbounded queue.
    * **queue_timeout** - The maximum number of seconds a request may wait
            in the queue before it is rejected with reason
            `"queue_timeout"`. Set to `None` to wait indefinitely (waiting
            requests may still be explicitly rejected or rejected on close).
    """

    def __init__(
        self,
        *,
        max_concurrency: int,
        max_queue: int | None = None,
        queue_timeout: float | None = None,
    ) -> None:
        if not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer.")
        if max_queue is not None and (
            not isinstance(max_queue, int) or max_queue < 0
        ):
            raise ValueError("max_queue must be a non-negative integer or None.")
        if queue_timeout is not None and queue_timeout <= 0:
            raise ValueError("queue_timeout must be a positive number or None.")

        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.max_concurrency == other.max_concurrency
            and self.max_queue == other.max_queue
            and self.queue_timeout == other.queue_timeout
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(max_concurrency={self.max_concurrency}, "
            f"max_queue={self.max_queue}, queue_timeout={self.queue_timeout})"
        )


class QuotaStatus:
    """
    A point-in-time snapshot of a single per-origin quota item.

    **Attributes:**

    * **origin** - The textual origin (`scheme://host:port`).
    * **max_concurrency** - Configured in-flight slot count for the origin.
    * **occupied** - Number of requests currently holding a slot.
    * **queued** - Number of requests currently waiting for a slot.
    * **max_queue** - Configured maximum queue depth (may be `None`).
    * **queue_timeout** - Configured maximum queue wait in seconds (may be `None`).
    * **rejection_reason** - Reason for the most recent rejection against
            this origin, or `None` if none has occurred.
    """

    __slots__ = (
        "max_concurrency",
        "max_queue",
        "occupied",
        "origin",
        "queued",
        "queue_timeout",
        "rejection_reason",
    )

    def __init__(
        self,
        *,
        origin: str,
        max_concurrency: int,
        occupied: int,
        queued: int,
        max_queue: int | None,
        queue_timeout: float | None,
        rejection_reason: str | None,
    ) -> None:
        self.origin = origin
        self.max_concurrency = max_concurrency
        self.occupied = occupied
        self.queued = queued
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.rejection_reason = rejection_reason

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and all(
                getattr(self, name) == getattr(other, name)
                for name in self.__slots__
            )
        )

    def __repr__(self) -> str:
        return (
            f"QuotaStatus(origin={self.origin!r}, "
            f"max_concurrency={self.max_concurrency}, occupied={self.occupied}, "
            f"queued={self.queued}, max_queue={self.max_queue}, "
            f"queue_timeout={self.queue_timeout}, "
            f"rejection_reason={self.rejection_reason!r})"
        )


class QuotaToken:
    """Represents a single acquired per-origin quota slot."""

    __slots__ = ("origin", "released")

    def __init__(self, origin: Origin) -> None:
        self.origin = origin
        self.released = False


class _Slot:
    __slots__ = ("occupied", "waiters", "last_reason")

    def __init__(self) -> None:
        self.occupied: int = 0
        self.waiters: deque[typing.Any] = deque()
        self.last_reason: str | None = None


OriginTypes = typing.Union[URL, str, Origin]


def _rejection_error(origin: Origin, reason: str) -> QuotaRejected:
    return QuotaRejected(
        "Request rejected by the per-origin concurrency quota "
        f"for {origin} (reason={reason!r}).",
        origin=str(origin),
        reason=reason,
    )


class BaseOriginQuotaLimiter:
    def __init__(self, quota: OriginQuota) -> None:
        self._config = quota
        self._slots: dict[Origin, _Slot] = {}
        self._closed = False

    # -- helpers -----------------------------------------------------------

    def _coerce_origin(self, origin: OriginTypes) -> Origin:
        if isinstance(origin, Origin):
            return origin
        if isinstance(origin, str):
            origin = URL(origin)
        return origin_from_url(origin)

    def _slot_for(self, origin: Origin) -> _Slot:
        slot = self._slots.get(origin)
        if slot is None:
            slot = _Slot()
            self._slots[origin] = slot
        return slot

    def _hand_off_or_release(self, origin: Origin, slot: _Slot) -> None:
        """
        Transfer the returned slot to the next waiting request, if any,
        otherwise mark it as free. Must be called while holding the lock.
        """
        while slot.waiters:
            waiter = slot.waiters.popleft()
            if waiter.reason is None:
                waiter.event.set()
                return
        slot.occupied -= 1

    def _reject_waiters(
        self, origin: Origin | None, reason: str
    ) -> int:
        """
        Wake every waiting request with a rejection. Must be called while
        holding the lock. Returns the number of rejected waiters.
        """
        if origin is None:
            target_slots = list(self._slots.values())
        else:
            slot = self._slots.get(origin)
            target_slots = [slot] if slot is not None else []

        count = 0
        for slot in target_slots:
            while slot.waiters:
                waiter = slot.waiters.popleft()
                if waiter.reason is None:
                    waiter.reason = reason
                    waiter.event.set()
                    count += 1
            slot.last_reason = reason
        return count

    def _statuses_locked(self) -> list[QuotaStatus]:
        return [
            QuotaStatus(
                origin=str(origin),
                max_concurrency=self._config.max_concurrency,
                occupied=slot.occupied,
                queued=len(slot.waiters),
                max_queue=self._config.max_queue,
                queue_timeout=self._config.queue_timeout,
                rejection_reason=slot.last_reason,
            )
            for origin, slot in self._slots.items()
        ]

    # -- status introspection ---------------------------------------------

    def statuses(self) -> list[QuotaStatus]:
        """
        Return a snapshot of every origin the client has dispatched a
        request to, including its current occupancy, queue depth and the
        most recent rejection reason.
        """
        raise NotImplementedError  # pragma: no cover

    def status(self, origin: OriginTypes) -> QuotaStatus:
        """
        Return the current status for a single origin.

        Raises `KeyError` if no request has ever targeted the origin.
        """
        raise NotImplementedError  # pragma: no cover


class AsyncOriginQuotaLimiter(BaseOriginQuotaLimiter):
    """
    Async per-origin concurrency gate, used by `AsyncClient`.

    Obtain an instance through `client.quota`.
    """

    def __init__(self, quota: OriginQuota) -> None:
        super().__init__(quota)
        # Lazily instantiated so that constructing the client does not bind
        # any asyncio event loop.
        self._lock: anyio.Lock | None = None

    def _get_lock(self) -> anyio.Lock:
        if self._lock is None:
            self._lock = anyio.Lock()
        return self._lock

    async def acquire(self, origin: OriginTypes) -> QuotaToken:
        """
        Acquire a slot for the origin, queueing until one is available.

        Raises `QuotaRejected` if the request cannot or can no longer be
        admitted.
        """
        origin_key = self._coerce_origin(origin)
        lock = self._get_lock()
        async with lock:
            if self._closed:
                slot = self._slot_for(origin_key)
                slot.last_reason = REASON_CLIENT_CLOSED
                raise _rejection_error(origin_key, REASON_CLIENT_CLOSED)

            slot = self._slot_for(origin_key)
            if slot.occupied < self._config.max_concurrency and not slot.waiters:
                slot.occupied += 1
                return QuotaToken(origin_key)

            if (
                self._config.max_queue is not None
                and len(slot.waiters) >= self._config.max_queue
            ):
                slot.last_reason = REASON_QUEUE_FULL
                raise _rejection_error(origin_key, REASON_QUEUE_FULL)

            waiter = _AsyncWaiter()
            slot.waiters.append(waiter)

        timeout = self._config.queue_timeout
        try:
            if timeout is None:
                await waiter.event.wait()
            else:
                with anyio.fail_after(timeout):
                    await waiter.event.wait()
        except anyio.get_cancelled_exc_class():
            # The waiting task was cancelled. Either we are still queued,
            # or a slot was handed to us just before cancellation - in which
            # case we must hand it straight back on. The cleanup is shielded
            # so the pending cancellation cannot interrupt it.
            with anyio.CancelScope(shield=True):
                async with lock:
                    if waiter in slot.waiters:
                        slot.waiters.remove(waiter)
                    elif waiter.reason is None:
                        self._hand_off_or_release(origin_key, slot)
            raise
        except TimeoutError:
            async with lock:
                if waiter in slot.waiters:
                    slot.waiters.remove(waiter)
                    slot.last_reason = REASON_QUEUE_TIMEOUT
                elif waiter.reason is not None:
                    raise _rejection_error(origin_key, waiter.reason) from None
                else:
                    # A slot was handed over at the exact deadline. Honor it.
                    return QuotaToken(origin_key)
            raise _rejection_error(origin_key, REASON_QUEUE_TIMEOUT) from None

        async with lock:
            if waiter.reason is not None:
                raise _rejection_error(origin_key, waiter.reason)
            # The waiter was dequeued by a slot handoff.
            return QuotaToken(origin_key)

    async def release(self, token: QuotaToken) -> None:
        """Return a previously acquired slot, waking the next waiter."""
        if token.released:
            return
        lock = self._get_lock()
        async with lock:
            if token.released:
                return
            token.released = True
            slot = self._slots.get(token.origin)
            if slot is not None:
                self._hand_off_or_release(token.origin, slot)

    async def reject(
        self,
        origin: OriginTypes | None = None,
        *,
        reason: str = REASON_REJECTED,
    ) -> int:
        """
        Reject requests that are currently waiting.

        When `origin` is given only requests queued for that origin are
        rejected, otherwise waiters for every origin are rejected. In-flight
        requests are not affected. Returns the number of waiters rejected.
        """
        origin_key = None if origin is None else self._coerce_origin(origin)
        lock = self._get_lock()
        async with lock:
            return self._reject_waiters(origin_key, reason)

    async def aclose(self) -> None:
        """
        Close the limiter, rejecting every waiting request with reason
        `"client_closed"`. In-flight requests may still release their slot.
        """
        lock = self._get_lock()
        async with lock:
            self._closed = True
            self._reject_waiters(None, REASON_CLIENT_CLOSED)

    def statuses(self) -> list[QuotaStatus]:
        # Called synchronously from within the event loop's thread; no await
        # point may mutate the structure while this method runs.
        return self._statuses_locked()

    def status(self, origin: OriginTypes) -> QuotaStatus:
        origin_key = self._coerce_origin(origin)
        slot = self._slots[origin_key]
        return QuotaStatus(
            origin=str(origin_key),
            max_concurrency=self._config.max_concurrency,
            occupied=slot.occupied,
            queued=len(slot.waiters),
            max_queue=self._config.max_queue,
            queue_timeout=self._config.queue_timeout,
            rejection_reason=slot.last_reason,
        )


class SyncOriginQuotaLimiter(BaseOriginQuotaLimiter):
    """
    Blocking, thread-safe per-origin concurrency gate, used by `Client`.

    Obtain an instance through `client.quota`.
    """

    def __init__(self, quota: OriginQuota) -> None:
        super().__init__(quota)
        self._lock = threading.Lock()

    def acquire(self, origin: OriginTypes) -> QuotaToken:
        """
        Acquire a slot for the origin, blocking in the queue until one is
        available.

        Raises `QuotaRejected` if the request cannot or can no longer be
        admitted.
        """
        origin_key = self._coerce_origin(origin)
        with self._lock:
            if self._closed:
                slot = self._slot_for(origin_key)
                slot.last_reason = REASON_CLIENT_CLOSED
                raise _rejection_error(origin_key, REASON_CLIENT_CLOSED)

            slot = self._slot_for(origin_key)
            if slot.occupied < self._config.max_concurrency and not slot.waiters:
                slot.occupied += 1
                return QuotaToken(origin_key)

            if (
                self._config.max_queue is not None
                and len(slot.waiters) >= self._config.max_queue
            ):
                slot.last_reason = REASON_QUEUE_FULL
                raise _rejection_error(origin_key, REASON_QUEUE_FULL)

            waiter = _SyncWaiter()
            slot.waiters.append(waiter)

        timeout = self._config.queue_timeout
        try:
            signaled = waiter.event.wait(timeout)
        except BaseException:
            # Interrupted while waiting (e.g. KeyboardInterrupt). Drop out of
            # the queue, returning any slot that had just been handed over.
            with self._lock:
                if waiter in slot.waiters:
                    slot.waiters.remove(waiter)
                elif waiter.reason is None:
                    self._hand_off_or_release(origin_key, slot)
            raise

        with self._lock:
            if waiter.reason is not None:
                raise _rejection_error(origin_key, waiter.reason)
            if not signaled:
                if waiter in slot.waiters:
                    slot.waiters.remove(waiter)
                    slot.last_reason = REASON_QUEUE_TIMEOUT
                    raise _rejection_error(origin_key, REASON_QUEUE_TIMEOUT)
                # A slot handoff raced the deadline; it is now ours.
            return QuotaToken(origin_key)

    def release(self, token: QuotaToken) -> None:
        """Return a previously acquired slot, waking the next waiter."""
        if token.released:
            return
        with self._lock:
            if token.released:
                return
            token.released = True
            slot = self._slots.get(token.origin)
            if slot is not None:
                self._hand_off_or_release(token.origin, slot)

    def reject(
        self,
        origin: OriginTypes | None = None,
        *,
        reason: str = REASON_REJECTED,
    ) -> int:
        """
        Reject requests that are currently waiting.

        When `origin` is given only requests queued for that origin are
        rejected, otherwise waiters for every origin are rejected. In-flight
        requests are not affected. Returns the number of waiters rejected.
        """
        origin_key = None if origin is None else self._coerce_origin(origin)
        with self._lock:
            return self._reject_waiters(origin_key, reason)

    def close(self) -> None:
        """
        Close the limiter, rejecting every waiting request with reason
        `"client_closed"`. In-flight requests may still release their slot.
        """
        with self._lock:
            self._closed = True
            self._reject_waiters(None, REASON_CLIENT_CLOSED)

    def statuses(self) -> list[QuotaStatus]:
        with self._lock:
            return self._statuses_locked()

    def status(self, origin: OriginTypes) -> QuotaStatus:
        origin_key = self._coerce_origin(origin)
        with self._lock:
            slot = self._slots[origin_key]
            return QuotaStatus(
                origin=str(origin_key),
                max_concurrency=self._config.max_concurrency,
                occupied=slot.occupied,
                queued=len(slot.waiters),
                max_queue=self._config.max_queue,
                queue_timeout=self._config.queue_timeout,
                rejection_reason=slot.last_reason,
            )


class _AsyncWaiter:
    __slots__ = ("event", "reason")

    def __init__(self) -> None:
        self.event = anyio.Event()
        self.reason: str | None = None


class _SyncWaiter:
    __slots__ = ("event", "reason")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.reason: str | None = None


class QuotaReleaseAsyncStream(AsyncByteStream):
    """
    Wraps a response stream so that the acquired quota slot is released
    once the response is closed.
    """

    def __init__(
        self,
        stream: AsyncByteStream,
        limiter: AsyncOriginQuotaLimiter,
        token: QuotaToken,
    ) -> None:
        self._stream = stream
        self._limiter = limiter
        self._token = token

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        finally:
            await self._limiter.release(self._token)


class QuotaReleaseSyncStream(SyncByteStream):
    """
    Wraps a response stream so that the acquired quota slot is released
    once the response is closed.
    """

    def __init__(
        self,
        stream: SyncByteStream,
        limiter: SyncOriginQuotaLimiter,
        token: QuotaToken,
    ) -> None:
        self._stream = stream
        self._limiter = limiter
        self._token = token

    def __iter__(self) -> typing.Iterator[bytes]:
        for chunk in self._stream:
            yield chunk

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._limiter.release(self._token)
