import threading
from http.cookiejar import Cookie, CookieJar

import anyio
import pytest

import httpx


def get_and_set_cookies(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/echo_cookies":
        data = {"cookies": request.headers.get("cookie")}
        return httpx.Response(200, json=data)
    elif request.url.path == "/set_cookie":
        return httpx.Response(200, headers={"set-cookie": "example-name=example-value"})
    else:
        raise NotImplementedError()  # pragma: no cover


def test_set_cookie() -> None:
    """
    Send a request including a cookie.
    """
    url = "http://example.org/echo_cookies"
    cookies = {"example-name": "example-value"}

    client = httpx.Client(
        cookies=cookies, transport=httpx.MockTransport(get_and_set_cookies)
    )
    response = client.get(url)

    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_set_per_request_cookie_is_deprecated() -> None:
    """
    Sending a request including a per-request cookie is deprecated.
    """
    url = "http://example.org/echo_cookies"
    cookies = {"example-name": "example-value"}

    client = httpx.Client(transport=httpx.MockTransport(get_and_set_cookies))
    with pytest.warns(DeprecationWarning):
        response = client.get(url, cookies=cookies)

    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_set_cookie_with_cookiejar() -> None:
    """
    Send a request including a cookie, using a `CookieJar` instance.
    """

    url = "http://example.org/echo_cookies"
    cookies = CookieJar()
    cookie = Cookie(
        version=0,
        name="example-name",
        value="example-value",
        port=None,
        port_specified=False,
        domain="",
        domain_specified=False,
        domain_initial_dot=False,
        path="/",
        path_specified=True,
        secure=False,
        expires=None,
        discard=True,
        comment=None,
        comment_url=None,
        rest={"HttpOnly": ""},
        rfc2109=False,
    )
    cookies.set_cookie(cookie)

    client = httpx.Client(
        cookies=cookies, transport=httpx.MockTransport(get_and_set_cookies)
    )
    response = client.get(url)

    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_setting_client_cookies_to_cookiejar() -> None:
    """
    Send a request including a cookie, using a `CookieJar` instance.
    """

    url = "http://example.org/echo_cookies"
    cookies = CookieJar()
    cookie = Cookie(
        version=0,
        name="example-name",
        value="example-value",
        port=None,
        port_specified=False,
        domain="",
        domain_specified=False,
        domain_initial_dot=False,
        path="/",
        path_specified=True,
        secure=False,
        expires=None,
        discard=True,
        comment=None,
        comment_url=None,
        rest={"HttpOnly": ""},
        rfc2109=False,
    )
    cookies.set_cookie(cookie)

    client = httpx.Client(
        cookies=cookies, transport=httpx.MockTransport(get_and_set_cookies)
    )
    response = client.get(url)

    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_set_cookie_with_cookies_model() -> None:
    """
    Send a request including a cookie, using a `Cookies` instance.
    """

    url = "http://example.org/echo_cookies"
    cookies = httpx.Cookies()
    cookies["example-name"] = "example-value"

    client = httpx.Client(transport=httpx.MockTransport(get_and_set_cookies))
    client.cookies = cookies
    response = client.get(url)

    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_get_cookie() -> None:
    url = "http://example.org/set_cookie"

    client = httpx.Client(transport=httpx.MockTransport(get_and_set_cookies))
    response = client.get(url)

    assert response.status_code == 200
    assert response.cookies["example-name"] == "example-value"
    assert client.cookies["example-name"] == "example-value"


def test_cookie_persistence() -> None:
    """
    Ensure that Client instances persist cookies between requests.
    """
    client = httpx.Client(transport=httpx.MockTransport(get_and_set_cookies))

    response = client.get("http://example.org/echo_cookies")
    assert response.status_code == 200
    assert response.json() == {"cookies": None}

    response = client.get("http://example.org/set_cookie")
    assert response.status_code == 200
    assert response.cookies["example-name"] == "example-value"
    assert client.cookies["example-name"] == "example-value"

    response = client.get("http://example.org/echo_cookies")
    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def multi_handler(request: httpx.Request) -> httpx.Response:
    """
    Shared mock handler exercising the various cookie mechanics.
    """
    path = request.url.path

    if path.endswith("/echo_cookies"):
        return httpx.Response(200, json={"cookies": request.headers.get("cookie")})

    if path == "/login":
        value = request.url.params["session"]
        return httpx.Response(
            200, headers={"set-cookie": f"session={value}; Path=/"}
        )

    if path == "/prelogin":
        value = request.url.params["value"]
        return httpx.Response(
            200,
            headers={
                "set-cookie": f"pre={value}; Domain=.example.org; Path=/"
            },
        )

    if path == "/set_domain":
        return httpx.Response(
            200,
            headers={
                "set-cookie": "token=scoped; Domain=example.org; Path=/api"
            },
        )

    if path == "/set_persistent":
        return httpx.Response(
            200,
            headers={
                "set-cookie": (
                    "temp=v; Path=/; Expires=Wed, 01 Jan 2099 00:00:00 GMT"
                )
            },
        )

    if path == "/expire_temp":
        return httpx.Response(
            200,
            headers={
                "set-cookie": (
                    "temp=v; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT"
                )
            },
        )

    if path == "/redirect_login":
        return httpx.Response(
            302,
            headers=[
                ("set-cookie", "sid=chain; Path=/"),
                ("location", "/echo_cookies"),
            ],
        )

    if path == "/cross_redirect":
        return httpx.Response(
            302, headers={"location": "http://other.example.org/echo_cookies"}
        )

    if path == "/boom":
        raise httpx.ConnectError("boom", request=request)

    raise NotImplementedError(path)  # pragma: no cover


def test_cookie_partition_basic_isolation() -> None:
    """
    Cookies set within a partition are invisible to other partitions and
    to the unpartitioned shared jar.
    """
    client = httpx.Client(
        cookie_partitions=True, transport=httpx.MockTransport(multi_handler)
    )

    client.get(
        "http://example.org/login",
        params={"session": "A"},
        cookie_context="tenant-a",
    )

    assert "tenant-a" in client.cookie_partitions
    assert client.cookie_partitions["tenant-a"]["session"] == "A"

    # A second, unused partition starts from a completely empty jar.
    response_b = client.get(
        "http://example.org/echo_cookies", cookie_context="tenant-b"
    )
    assert response_b.json()["cookies"] is None

    # The owning partition continues to select its own cookie.
    response_a = client.get(
        "http://example.org/echo_cookies", cookie_context="tenant-a"
    )
    assert response_a.json()["cookies"] == "session=A"

    # The shared jar never sees a partition's Set-Cookie updates.
    assert not client.cookies
    response_default = client.get("http://example.org/echo_cookies")
    assert response_default.json()["cookies"] is None

    client.close()


def test_unpartitioned_request_still_shares_jar_when_enabled() -> None:
    """
    With partitioning enabled, requests without a context still behave
    exactly as before, sharing the client cookie jar.
    """
    client = httpx.Client(
        cookie_partitions=True, transport=httpx.MockTransport(multi_handler)
    )

    client.get("http://example.org/login", params={"session": "shared"})

    response = client.get("http://example.org/echo_cookies")
    assert response.json()["cookies"] == "session=shared"

    # The shared jar cookie is not offered inside a partition.
    response_partition = client.get(
        "http://example.org/echo_cookies", cookie_context="tenant-a"
    )
    assert response_partition.json()["cookies"] is None
    assert client.cookie_partitions["tenant-a"].get("session") is None

    client.close()


def test_cookie_partition_set_cookie_domain_and_path() -> None:
    """
    Domain and path attributes of Set-Cookie are honoured, and remain
    scoped to the partition they were written into.
    """
    client = httpx.Client(
        cookie_partitions=True, transport=httpx.MockTransport(multi_handler)
    )

    client.get("http://example.org/set_domain", cookie_context="tenant-a")

    # Matching domain + path -> cookie is selected.
    response = client.get(
        "http://example.org/api/echo_cookies", cookie_context="tenant-a"
    )
    assert response.json()["cookies"] == "token=scoped"

    # Same domain, non-matching path -> cookie is not selected.
    response = client.get(
        "http://example.org/echo_cookies", cookie_context="tenant-a"
    )
    assert response.json()["cookies"] is None

    # Matching URL in a different partition and in the shared jar -> nothing.
    response = client.get(
        "http://example.org/api/echo_cookies", cookie_context="tenant-b"
    )
    assert response.json()["cookies"] is None
    response = client.get("http://example.org/api/echo_cookies")
    assert response.json()["cookies"] is None

    client.close()


def test_cookie_partition_expired_cookie_is_dropped() -> None:
    """
    A Set-Cookie that is already expired never reaches the partition jar.
    """
    client = httpx.Client(
        cookie_partitions=True, transport=httpx.MockTransport(multi_handler)
    )

    client.get("http://example.org/expire_temp", cookie_context="tenant-a")

    assert client.cookie_partitions["tenant-a"].get("temp") is None
    response = client.get(
        "http://example.org/echo_cookies", cookie_context="tenant-a"
    )
    assert response.json()["cookies"] is None

    client.close()


def test_cookie_partition_expiry_update_removes_cookie() -> None:
    """
    A later Set-Cookie carrying an expired date removes the previously
    stored cookie within the same partition.
    """
    client = httpx.Client(
        cookie_partitions=True, transport=httpx.MockTransport(multi_handler)
    )

    client.get("http://example.org/set_persistent", cookie_context="tenant-a")
    assert client.cookie_partitions["tenant-a"]["temp"] == "v"

    client.get("http://example.org/expire_temp", cookie_context="tenant-a")
    assert client.cookie_partitions["tenant-a"].get("temp") is None

    response = client.get(
        "http://example.org/echo_cookies", cookie_context="tenant-a"
    )
    assert response.json()["cookies"] is None

    # The expiry update does not leak across partitions.
    client.get("http://example.org/set_persistent", cookie_context="tenant-b")
    assert client.cookie_partitions["tenant-b"]["temp"] == "v"

    client.close()


def test_cookie_partition_redirect_chain_inherits_context() -> None:
    """
    Redirected requests inherit the pinned partition: cookies already in
    that partition are carried, and Set-Cookie on an earlier hop is seen
    by the next hop.
    """
    client = httpx.Client(
        cookie_partitions=True,
        follow_redirects=True,
        transport=httpx.MockTransport(multi_handler),
    )

    # Tenant A already has a parent-domain cookie on record.
    client.get(
        "http://example.org/prelogin",
        params={"value": "Aval"},
        cookie_context="tenant-a",
    )

    response = client.get(
        "http://example.org/redirect_login", cookie_context="tenant-a"
    )
    header = response.json()["cookies"]
    sent_cookies = {
        item.split("=")[0]: item.split("=", 1)[1]
        for item in header.split("; ")
    }
    assert sent_cookies["pre"] == "Aval"
    assert sent_cookies["sid"] == "chain"

    # A fresh partition follows the same chain: it receives the chain's
    # own Set-Cookie, but never tenant A's pre-existing cookie.
    response_b = client.get(
        "http://example.org/redirect_login", cookie_context="tenant-b"
    )
    header_b = response_b.json()["cookies"]
    assert "sid=chain" in header_b
    assert "pre=Aval" not in header_b

    client.close()


def test_cookie_partition_cross_origin_redirect() -> None:
    """
    When a chain crosses origins the context stays pinned to the same
    partition; it never falls back to another partition or the shared jar.
    """
    client = httpx.Client(
        cookie_partitions=True,
        follow_redirects=True,
        transport=httpx.MockTransport(multi_handler),
    )

    # Tenant A establishes a cookie matching the parent domain.
    client.get(
        "http://api.example.org/prelogin",
        params={"value": "Aval"},
        cookie_context="tenant-a",
    )

    response_a = client.get(
        "http://api.example.org/cross_redirect", cookie_context="tenant-a"
    )
    assert response_a.request.url.host == "other.example.org"
    assert response_a.json()["cookies"] == "pre=Aval"

    # Tenant B's cross-origin chain has no access to tenant A's cookie.
    response_b = client.get(
        "http://api.example.org/cross_redirect", cookie_context="tenant-b"
    )
    assert response_b.request.url.host == "other.example.org"
    assert response_b.json()["cookies"] is None

    # An unpartitioned chain reads from the empty shared jar.
    response_default = client.get("http://api.example.org/cross_redirect")
    assert response_default.json()["cookies"] is None

    client.close()


def test_cookie_partition_explicit_cookies_priority() -> None:
    """
    Explicit per-request cookies override same-named partition cookies
    for that request, but are not persisted back into the partition.
    """
    client = httpx.Client(
        cookie_partitions=True, transport=httpx.MockTransport(multi_handler)
    )

    client.get(
        "http://example.org/login",
        params={"session": "old"},
        cookie_context="tenant-a",
    )

    with pytest.warns(DeprecationWarning):
        response = client.get(
            "http://example.org/echo_cookies",
            cookie_context="tenant-a",
            cookies={"session": "new", "extra": "1"},
        )

    header = response.json()["cookies"]
    assert "session=new" in header
    assert "session=old" not in header
    assert "extra=1" in header

    # The persistent partition jar keeps the old value and never stores
    # a request-only cookie.
    assert client.cookie_partitions["tenant-a"]["session"] == "old"
    assert client.cookie_partitions["tenant-a"].get("extra") is None

    client.close()


def test_build_request_pins_cookie_context() -> None:
    """
    build_request records the context on the request, so a later send --
    including redirect/auth re-requests -- stays on the same partition.
    """
    client = httpx.Client(
        cookie_partitions=True, transport=httpx.MockTransport(multi_handler)
    )

    request = client.build_request(
        "GET", "http://example.org/echo_cookies", cookie_context="tenant-a"
    )
    assert request.extensions["cookie_context"] == "tenant-a"

    with pytest.raises(TypeError):
        client.build_request(
            "GET", "http://example.org/", cookie_context=123  # type: ignore[arg-type]
        )

    plain_client = httpx.Client(transport=httpx.MockTransport(multi_handler))

    with pytest.raises(RuntimeError):
        plain_client.build_request(
            "GET", "http://example.org/", cookie_context="tenant-a"
        )

    # Sending a raw request tagged with a context is also rejected when
    # partitioning was not enabled.
    raw_request = httpx.Request(
        "GET",
        "http://example.org/",
        extensions={"cookie_context": "tenant-a"},
    )
    with pytest.raises(RuntimeError):
        plain_client.send(raw_request)

    plain_client.close()
    client.close()


def test_cookie_partitions_manager_api() -> None:
    """
    The partition manager creates jars lazily, returns independent
    snapshots, and drops cookies irrecoverably on delete/clear.
    """
    partitions = httpx.CookiePartitions()

    assert len(partitions) == 0
    assert "a" not in partitions

    jar = partitions.get("a")
    jar.set("k", "v")
    assert "a" in partitions
    assert partitions["a"].get("k") == "v"
    assert list(partitions) == ["a"]

    # Mutating a snapshot never affects the live partition.
    snapshot = partitions.snapshot("a")
    snapshot.set("k", "other")
    snapshot.set("only", "snap")
    assert partitions["a"].get("k") == "v"
    assert partitions["a"].get("only") is None

    # Snapshotting an unknown identifier creates that partition.
    partitions.snapshot("b")
    assert "b" in partitions

    partitions.delete("a")
    assert "a" not in partitions
    with pytest.raises(KeyError):
        partitions["a"]

    # Deleting a missing partition is a no-op.
    partitions.delete("missing")

    partitions.clear()
    assert len(partitions) == 0
    assert "CookiePartitions" in repr(partitions)


def test_concurrent_threads_use_isolated_cookie_partitions() -> None:
    """
    A sync client shared across threads keeps each context's cookies
    isolated, including concurrent access to one shared partition.
    """
    client = httpx.Client(
        cookie_partitions=True, transport=httpx.MockTransport(multi_handler)
    )

    contexts = [f"tenant-{i}" for i in range(5)]
    for context in contexts:
        client.get(
            "http://example.org/login",
            params={"session": context},
            cookie_context=context,
        )
    client.get(
        "http://example.org/login",
        params={"session": "shared"},
        cookie_context="shared",
    )

    barrier = threading.Barrier(len(contexts) + 3)
    errors: list[BaseException] = []

    def isolated_worker(context: str) -> None:
        try:
            barrier.wait()
            for _ in range(30):
                response = client.get(
                    "http://example.org/echo_cookies", cookie_context=context
                )
                assert response.json()["cookies"] == f"session={context}"
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def shared_worker() -> None:
        try:
            barrier.wait()
            for _ in range(30):
                response = client.get(
                    "http://example.org/echo_cookies", cookie_context="shared"
                )
                assert response.json()["cookies"] == "session=shared"
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=isolated_worker, args=(context,))
        for context in contexts
    ]
    threads.extend(threading.Thread(target=shared_worker) for _ in range(3))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    client.close()


def test_failure_and_partition_deletion_do_not_leak_cookies() -> None:
    """
    A failed request leaves no phantom cookies behind; deleting a
    partition makes its cookies unavailable to every subsequent request,
    even one reusing the same identifier.
    """
    client = httpx.Client(
        cookie_partitions=True, transport=httpx.MockTransport(multi_handler)
    )

    client.get(
        "http://example.org/login",
        params={"session": "A"},
        cookie_context="tenant-a",
    )

    with pytest.raises(httpx.ConnectError):
        client.get("http://example.org/boom", cookie_context="tenant-a")

    # Existing semantics: a transport failure does not wipe the jar.
    assert client.cookie_partitions["tenant-a"]["session"] == "A"

    # Explicit deletion drops the partition irrecoverably.
    client.cookie_partitions.delete("tenant-a")
    assert "tenant-a" not in client.cookie_partitions

    response = client.get(
        "http://example.org/echo_cookies", cookie_context="tenant-a"
    )
    assert response.json()["cookies"] is None
    assert client.cookie_partitions["tenant-a"].get("session") is None

    client.close()


def test_closed_client_cookie_partition_semantics() -> None:
    """
    Closing a client does not clear partition cookies (mirroring the
    shared jar), but no further requests can be sent.
    """
    client = httpx.Client(
        cookie_partitions=True, transport=httpx.MockTransport(multi_handler)
    )

    client.get(
        "http://example.org/login",
        params={"session": "A"},
        cookie_context="tenant-a",
    )
    client.close()

    assert client.is_closed
    assert client.cookie_partitions["tenant-a"]["session"] == "A"

    with pytest.raises(RuntimeError):
        client.get(
            "http://example.org/echo_cookies", cookie_context="tenant-a"
        )


@pytest.mark.anyio
async def test_async_concurrent_cookie_partitions() -> None:
    """
    Concurrent async requests for different tenants cannot exchange
    cookies, while multiple tasks sharing one context see a consistent
    partition.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(0)
        if request.url.path == "/login":
            await anyio.sleep(0)
            value = request.url.params["session"]
            return httpx.Response(
                200, headers={"set-cookie": f"session={value}; Path=/"}
            )
        if request.url.path == "/echo_cookies":
            await anyio.sleep(0)
            return httpx.Response(
                200, json={"cookies": request.headers.get("cookie")}
            )
        raise NotImplementedError(request.url.path)  # pragma: no cover

    async with httpx.AsyncClient(
        cookie_partitions=True, transport=httpx.MockTransport(handler)
    ) as client:

        async def login(context: str) -> None:
            await anyio.sleep(0)
            await client.get(
                "http://example.org/login",
                params={"session": context},
                cookie_context=context,
            )

        async with anyio.create_task_group() as task_group:
            for context in ("tenant-a", "tenant-b", "tenant-c"):
                task_group.start_soon(login, context)

        async def echo(context: str) -> None:
            for _ in range(10):
                await anyio.sleep(0)
                response = await client.get(
                    "http://example.org/echo_cookies",
                    cookie_context=context,
                )
                assert response.json()["cookies"] == f"session={context}"

        async with anyio.create_task_group() as task_group:
            for context in ("tenant-a", "tenant-b", "tenant-c"):
                task_group.start_soon(echo, context)
                task_group.start_soon(echo, context)

        # A context-less request must continue to use the empty shared jar.
        response = await client.get("http://example.org/echo_cookies")
        assert response.json()["cookies"] is None


@pytest.mark.anyio
async def test_async_redirect_chain_pins_context_snapshot() -> None:
    """
    Throughout an async redirect chain the context snapshot is fixed:
    every hop carries the same identifier and reads only its partition.
    """
    hops: list[tuple[str, str | None, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(0)
        hops.append(
            (
                request.url.path,
                request.extensions.get("cookie_context"),
                request.headers.get("cookie"),
            )
        )
        if request.url.path == "/redirect_login":
            return httpx.Response(
                302,
                headers=[
                    ("set-cookie", "sid=chain; Path=/"),
                    ("location", "/echo_cookies"),
                ],
            )
        if request.url.path == "/echo_cookies":
            return httpx.Response(
                200, json={"cookies": request.headers.get("cookie")}
            )
        raise NotImplementedError(request.url.path)  # pragma: no cover

    async with httpx.AsyncClient(
        cookie_partitions=True,
        follow_redirects=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        response = await client.get(
            "http://example.org/redirect_login", cookie_context="tenant-a"
        )

        assert response.json()["cookies"] == "sid=chain"

        # Both hops are pinned to the same context; the first hop has no
        # cookie while the second inherits the cookie set on hop one.
        assert hops[0][0] == "/redirect_login"
        assert hops[0][1] == "tenant-a"
        assert hops[0][2] is None
        assert hops[1][0] == "/echo_cookies"
        assert hops[1][1] == "tenant-a"
        assert hops[1][2] == "sid=chain"

        # A different partition never observes tenant A's chain cookies
        # before its own first hop, and never on any other chain.
        response_b = await client.get(
            "http://example.org/echo_cookies", cookie_context="tenant-b"
        )
        assert response_b.json()["cookies"] is None
