"""
Optional HTTP Message Signatures support (RFC 9421), including content
digests (RFC 9530).

This module is self-contained and is only exercised when a client is
configured with a `MessageSigner`. When message signatures are not
enabled, request authentication, headers, streaming bodies, redirects
and client/stream close semantics are unchanged.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
import typing

from ._content import ByteStream
from ._exceptions import RequestError
from ._types import AsyncByteStream, SyncByteStream
from ._urls import URL

if typing.TYPE_CHECKING:
    from ._models import Request, Response

__all__ = [
    "ContentDigestError",
    "DuplicateNonce",
    "InvalidSignature",
    "KeyResolver",
    "MessageSigner",
    "SignatureError",
    "SignatureExpired",
    "SignatureNotYetValid",
    "SignatureSigningError",
    "StaticKeyResolver",
]


# Exceptions
#
# These all subclass 'RequestError' so that the 'request_context()' context
# manager associates them with the request being signed / verified, and so
# that callers can handle them with the standard 'httpx' error hierarchy.


class SignatureError(RequestError):
    """
    Base class for HTTP Message Signature (RFC 9421) failures.
    """


class SignatureSigningError(SignatureError):
    """
    The client failed to produce a signature for an outgoing request.
    """


class InvalidSignature(SignatureError):
    """
    A response's signature could not be verified.
    """


class SignatureExpired(SignatureError):
    """
    A signature is past its `expires` time (or the configured maximum age).

    The diagnostic attributes (`keyid`, `created`, `expires`, `now`) allow
    callers to determine why the signature was rejected.
    """

    def __init__(
        self,
        message: str,
        *,
        request: Request | None = None,
        keyid: str | None = None,
        created: int | None = None,
        expires: int | None = None,
        now: int | None = None,
    ) -> None:
        super().__init__(message, request=request)
        self.keyid = keyid
        self.created = created
        self.expires = expires
        self.now = now


class SignatureNotYetValid(SignatureError):
    """
    A signature has a `created` time too far in the future.
    """

    def __init__(
        self,
        message: str,
        *,
        request: Request | None = None,
        keyid: str | None = None,
        created: int | None = None,
        now: int | None = None,
    ) -> None:
        super().__init__(message, request=request)
        self.keyid = keyid
        self.created = created
        self.now = now


class DuplicateNonce(SignatureError):
    """
    A signature nonce has already been seen (replay attempt).
    """

    def __init__(
        self,
        message: str,
        *,
        request: Request | None = None,
        keyid: str | None = None,
        nonce: str | None = None,
    ) -> None:
        super().__init__(message, request=request)
        self.keyid = keyid
        self.nonce = nonce


class ContentDigestError(SignatureError):
    """
    A 'Content-Digest' (RFC 9530) did not match the received content.
    """


# Cryptographic algorithms


def _hmac_sign(key: typing.Any, data: bytes, digest: str) -> bytes:
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(f"{digest} requires a bytes key, got {type(key)!r}")
    return hmac.new(bytes(key), data, digest).digest()


def _hmac_verify(
    key: typing.Any, signature: bytes, data: bytes, digest: str
) -> None:
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(f"{digest} requires a bytes key, got {type(key)!r}")
    expected = hmac.new(bytes(key), data, digest).digest()
    if not hmac.compare_digest(expected, signature):
        raise InvalidSignature("The message signature is invalid.")


def _load_cryptography() -> typing.Any:
    try:
        import cryptography  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Using asymmetric message signature algorithms requires the "
            "'cryptography' package. Install it with "
            "`pip install cryptography`."
        ) from exc
    return cryptography


def _asym_sign(key: typing.Any, data: bytes, *, hash: str, padding: str) -> bytes:
    hasher = _crypto_hasher(hash)
    if padding == "pkcs1v15":
        from cryptography.hazmat.primitives.asymmetric import padding as _padding

        return typing.cast(
            bytes, key.sign(data, _padding.PKCS1v15(), hasher)
        )
    elif padding == "pss":
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding as _padding

        return typing.cast(
            bytes,
            key.sign(
                data,
                # RFC 9421 uses a salt length equal to the digest length.
                _padding.PSS(
                    mgf=_padding.MGF1(hashes.SHA512()),
                    salt_length=hasher.digest_size,
                ),
                hasher,
            ),
        )
    elif padding == "ecdsa":
        from cryptography.hazmat.primitives.asymmetric.ec import ECDSA

        return typing.cast(bytes, key.sign(data, ECDSA(hasher)))
    elif padding == "ed25519":
        return typing.cast(bytes, key.sign(data))
    raise ValueError(f"Unknown padding {padding!r}")  # pragma: no cover


def _asym_verify(
    key: typing.Any,
    signature: bytes,
    data: bytes,
    *,
    hash: str,
    padding: str,
) -> None:
    _load_cryptography()
    from cryptography.exceptions import InvalidSignature as _InvalidSignature

    def _raise() -> None:
        raise InvalidSignature("The message signature is invalid.")

    try:
        if padding == "pkcs1v15":
            from cryptography.hazmat.primitives.asymmetric import padding as _padding

            key.verify(
                signature, data, _padding.PKCS1v15(), _crypto_hasher(hash)
            )
        elif padding == "pss":
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding as _padding

            key.verify(
                signature,
                data,
                _padding.PSS(
                    mgf=_padding.MGF1(hashes.SHA512()),
                    salt_length=_padding.PSS.AUTO,
                ),
                _crypto_hasher(hash),
            )
        elif padding == "ecdsa":
            from cryptography.hazmat.primitives.asymmetric import ec

            key.verify(signature, data, ec.ECDSA(_crypto_hasher(hash)))
        elif padding == "ed25519":
            key.verify(signature, data)
        else:  # pragma: no cover
            raise ValueError(f"Unknown padding {padding!r}")
    except _InvalidSignature:
        _raise()


def _sign_with_algorithm(alg: str, key: typing.Any, data: bytes) -> bytes:
    if alg == "hmac-sha256":
        return _hmac_sign(key, data, "sha256")
    elif alg == "hmac-sha384":
        return _hmac_sign(key, data, "sha384")
    elif alg == "hmac-sha512":
        return _hmac_sign(key, data, "sha512")
    elif alg == "rsa-v1_5-sha256":
        return _asym_sign(key, data, hash="sha256", padding="pkcs1v15")
    elif alg == "rsa-pss-sha512":
        return _asym_sign(key, data, hash="sha512", padding="pss")
    elif alg == "ecdsa-p256-sha256":
        return _asym_sign(key, data, hash="sha256", padding="ecdsa")
    elif alg == "ecdsa-p384-sha384":
        return _asym_sign(key, data, hash="sha384", padding="ecdsa")
    elif alg == "ed25519":
        return _asym_sign(key, data, hash="sha512", padding="ed25519")
    raise SignatureSigningError(f"Unsupported signature algorithm {alg!r}.")


def _verify_with_algorithm(
    alg: str, key: typing.Any, signature: bytes, data: bytes
) -> None:
    if alg == "hmac-sha256":
        _hmac_verify(key, signature, data, "sha256")
    elif alg == "hmac-sha384":
        _hmac_verify(key, signature, data, "sha384")
    elif alg == "hmac-sha512":
        _hmac_verify(key, signature, data, "sha512")
    elif alg == "rsa-v1_5-sha256":
        _asym_verify(key, signature, data, hash="sha256", padding="pkcs1v15")
    elif alg == "rsa-pss-sha512":
        _asym_verify(key, signature, data, hash="sha512", padding="pss")
    elif alg == "ecdsa-p256-sha256":
        _asym_verify(key, signature, data, hash="sha256", padding="ecdsa")
    elif alg == "ecdsa-p384-sha384":
        _asym_verify(key, signature, data, hash="sha384", padding="ecdsa")
    elif alg == "ed25519":
        _asym_verify(key, signature, data, hash="sha512", padding="ed25519")
    else:
        raise InvalidSignature(f"Unsupported signature algorithm {alg!r}.")


# Content digest algorithms (RFC 9530)

_DIGEST_ALGORITHMS: dict[str, typing.Callable[[], typing.Any]] = {
    "sha-256": hashlib.sha256,
    "sha-384": hashlib.sha384,
    "sha-512": hashlib.sha512,
}

# Key resolution


class KeyResolver:
    """
    Base class for resolving signature key material from a `keyid`.

    Subclass this and override `.resolve_key()` (and optionally
    `.aresolve_key()` for async I/O) to return the key used to sign
    outgoing requests and verify incoming responses.

    For symmetric algorithms such as `hmac-sha256` the key is returned as
    `bytes`. For asymmetric algorithms return a `cryptography` key object.
    """

    #: Used when the signer itself does not pin a `keyid`.
    default_keyid: str | None = None

    def resolve_key(
        self,
        keyid: str,
        request: Request,
        *,
        for_response: bool = False,
    ) -> typing.Any:
        raise NotImplementedError(  # pragma: no cover
            "KeyResolver subclasses must implement the 'resolve_key' method."
        )

    async def aresolve_key(
        self,
        keyid: str,
        request: Request,
        *,
        for_response: bool = False,
    ) -> typing.Any:
        # Key resolution is usually in-memory; subclasses doing async I/O
        # should override this method.
        return self.resolve_key(keyid, request, for_response=for_response)


class StaticKeyResolver(KeyResolver):
    """
    A simple key resolver backed by a static mapping of `keyid` to key.
    """

    def __init__(
        self,
        keys: typing.Mapping[str, typing.Any] | typing.Any,
        *,
        default_keyid: str | None = None,
    ) -> None:
        if isinstance(keys, (bytes, bytearray)):
            keys = {"default": bytes(keys)}
            default_keyid = "default"
        self._keys = dict(keys)
        if default_keyid is None and len(self._keys) == 1:
            default_keyid = next(iter(self._keys))
        self.default_keyid = default_keyid

    def resolve_key(
        self,
        keyid: str,
        request: Request,
        *,
        for_response: bool = False,
    ) -> typing.Any:
        try:
            return self._keys[keyid]
        except KeyError:
            raise SignatureSigningError(
                f"No key found for keyid={keyid!r}.", request=request
            ) from None


# Structured Fields (RFC 8941) helpers


def _sf_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# Signature base construction (RFC 9421, section 2.5)


def _raw_path(url: URL) -> bytes:
    # 'URL.raw_path' includes the query; split it off without decoding.
    raw_path = url.raw_path
    return raw_path.split(b"?", 1)[0]


def _crypto_hasher(name: str) -> typing.Any:
    _load_cryptography()
    from cryptography.hazmat.primitives import hashes

    if name == "sha256":
        return hashes.SHA256()
    elif name == "sha384":
        return hashes.SHA384()
    return hashes.SHA512()


def _hash_content(algo: str, data: bytes) -> bytes:
    return hashlib.new(algo, data).digest()


def _request_component_value(component: str, request: Request) -> bytes:
    if component == "@method":
        return request.method.encode("ascii")
    elif component == "@target-uri":
        return str(request.url).encode("ascii")
    elif component == "@authority":
        return request.url.netloc
    elif component == "@scheme":
        return request.url.scheme.encode("ascii")
    elif component == "@path":
        return _raw_path(request.url)
    elif component == "@query":
        return b"?" + request.url.query
    elif component.startswith("@"):
        raise SignatureSigningError(
            f"Unsupported signature component {component!r}.", request=request
        )
    return _header_value(request, component)


def _response_component_value(component: str, response: Response) -> bytes:
    if component == "@status":
        return str(response.status_code).encode("ascii")
    elif component.startswith("@"):
        raise InvalidSignature(
            f"Unsupported response signature component {component!r}.",
            request=response.request,
        )
    return _header_value(response, component)


def _header_value(message: Request | Response, name: str) -> bytes:
    values = message.headers.get_list(name)
    if not values:
        if isinstance(message, Request) or getattr(message, "_request", None) is None:
            request = message if isinstance(message, Request) else None
            raise SignatureSigningError(
                f"Cannot sign missing header {name!r}.", request=request
            )
        raise InvalidSignature(
            f"Signed header {name!r} is missing.", request=message.request
        )
    # Multiple field values are combined with ", " (RFC 9421 2.1.3).
    return ", ".join(values).encode("utf-8")


def _build_signature_base(
    covered: typing.Sequence[str],
    values: typing.Callable[[str], bytes],
) -> bytes:
    lines: list[bytes] = []
    for component in covered:
        lines.append(
            b'"' + component.encode("ascii") + b'": ' + values(component)
        )
    return b"\n".join(lines)


# Body digesting


def _format_content_digest(algo: str, digest: bytes) -> str:
    return f"{algo}=:{_b64(digest)}:"


def _digest_body_sync(request: Request, algo: str) -> bytes:
    """
    Consume a (possibly chunked) request stream, incrementally computing
    its content digest, and buffer the bytes so the request remains
    replayable (redirects/auth re-sends).
    """
    hasher = _DIGEST_ALGORITHMS[algo]()
    buffer = bytearray()
    assert isinstance(request.stream, SyncByteStream)
    for chunk in request.stream:
        hasher.update(chunk)
        buffer.extend(chunk)
    body = bytes(buffer)
    request._content = body
    request.stream = ByteStream(body)
    return body


async def _adigest_body(request: Request, algo: str) -> bytes:
    hasher = _DIGEST_ALGORITHMS[algo]()
    buffer = bytearray()
    assert isinstance(request.stream, AsyncByteStream)
    async for chunk in request.stream:
        hasher.update(chunk)
        buffer.extend(chunk)
    body = bytes(buffer)
    request._content = body
    request.stream = ByteStream(body)
    return body


# Response stream digest verification wrappers


class _DigestVerifySyncStream(SyncByteStream):
    def __init__(
        self,
        stream: SyncByteStream,
        algo: str,
        expected: bytes,
        request: Request,
    ) -> None:
        self._stream = stream
        self._algo = algo
        self._expected = expected
        self._request = request

    def __iter__(self) -> typing.Iterator[bytes]:
        hasher = _DIGEST_ALGORITHMS[self._algo]()
        for chunk in self._stream:
            hasher.update(chunk)
            yield chunk
        if not hmac.compare_digest(hasher.digest(), self._expected):
            raise ContentDigestError(
                "Response Content-Digest does not match the received body.",
                request=self._request,
            )

    def close(self) -> None:
        self._stream.close()


class _DigestVerifyAsyncStream(AsyncByteStream):
    def __init__(
        self,
        stream: AsyncByteStream,
        algo: str,
        expected: bytes,
        request: Request,
    ) -> None:
        self._stream = stream
        self._algo = algo
        self._expected = expected
        self._request = request

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        hasher = _DIGEST_ALGORITHMS[self._algo]()
        async for chunk in self._stream:
            hasher.update(chunk)
            yield chunk
        if not hmac.compare_digest(hasher.digest(), self._expected):
            raise ContentDigestError(
                "Response Content-Digest does not match the received body.",
                request=self._request,
            )

    async def aclose(self) -> None:
        await self._stream.aclose()


# Parsing of response signature headers


def _parse_signature_input(value: str) -> tuple[str, list[str], dict[str, str]]:
    """
    Parse a single 'Signature-Input' member:

        sig1=("@method" "@path");keyid="k";created=123;nonce="n"

    Returns (label, covered_components, parameters).
    """
    if "=" not in value:
        raise InvalidSignature("Malformed 'Signature-Input' header.")
    label, rest = value.split("=", 1)
    label = label.strip()
    rest = rest.strip()
    if not rest.startswith("(") or ")" not in rest:
        raise InvalidSignature("Malformed 'Signature-Input' inner list.")
    inner, _, params_str = rest[1:].partition(")")

    covered: list[str] = []
    token = ""
    in_quotes = False
    for char in inner:
        if char == '"':
            in_quotes = not in_quotes
            continue
        if not in_quotes and char.isspace():
            if token:
                covered.append(token)
                token = ""
            continue
        token += char
    if token:
        covered.append(token)

    params = _parse_signature_params(params_str)
    return label, covered, params


def _parse_signature_params(params_str: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for part in params_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        params[key.strip()] = value
    return params


def _parse_signature_header(value: str, label: str) -> bytes:
    """
    Extract the base64 signature for `label` from a 'Signature' header.
    """
    prefix = label + "=:"
    start = value.find(prefix)
    if start < 0:
        raise InvalidSignature(
            f"No 'Signature' value found for label {label!r}."
        )
    start += len(prefix)
    end = value.find(":", start)
    if end < 0:
        raise InvalidSignature("Malformed 'Signature' header.")
    try:
        return base64.b64decode(value[start:end], validate=True)
    except Exception as exc:
        raise InvalidSignature("Malformed signature encoding.") from exc


def _parse_content_digest(value: str) -> dict[str, bytes]:
    digests: dict[str, bytes] = {}
    for part in value.split(","):
        part = part.strip()
        if not part or "=:" not in part:
            continue
        algo, _, b64value = part.partition("=:")
        b64value = b64value[:-1] if b64value.endswith(":") else b64value
        try:
            digests[algo.strip()] = base64.b64decode(b64value, validate=True)
        except Exception:
            continue
    return digests


# The signer


class MessageSigner:
    """
    Signs outgoing requests using HTTP Message Signatures (RFC 9421) and
    optionally verifies signatures on incoming responses.

    Usage:

    ```python
    signer = httpx.MessageSigner(
        key_resolver=httpx.StaticKeyResolver({"key-1": b"shared-secret"}),
        components=("@method", "@authority", "@path", "content-digest"),
    )
    with httpx.Client(message_signature=signer) as client:
        client.post("https://example.com/data", json={"hello": "world"})
    ```

    **Parameters:**

    * **key_resolver** - A `KeyResolver` (or any object with compatible
      `resolve_key`/`aresolve_key` methods) used to locate key material.
    * **keyid** - The keyid to sign with. If unset the resolver's
      `default_keyid` is used.
    * **alg** - Signature algorithm. Defaults to `hmac-sha256`.
    * **components** - Components covered by the signature. RFC 9421
      derived components (`@method`, `@target-uri`, `@authority`,
      `@scheme`, `@path`, `@query`) and HTTP header names (including
      `content-digest`) are supported.
    * **digest_algorithm** - Content digest algorithm (RFC 9530) used when
      `content-digest` is a covered component. Defaults to `sha-256`.
    * **include_created** - Include a `created` parameter. Default `True`.
    * **expires_in** - Optional lifetime in seconds, producing an `expires`
      parameter.
    * **use_nonce** - Include a unique per-signature `nonce`.
    * **label** - Signature label. Defaults to `sig1`.
    * **verify_response** - Verify response signatures when present.
      Default `False`.
    * **require_response_signature** - Reject responses without a signature.
    * **max_skew** - Allowed clock skew in seconds when checking `created`.
    * **max_age** - Optional maximum accepted signature age in seconds.
    * **replay_protection** - Remember nonces and reject replays.
    """

    def __init__(
        self,
        key_resolver: KeyResolver,
        *,
        keyid: str | None = None,
        alg: str = "hmac-sha256",
        components: typing.Sequence[str] = (
            "@method",
            "@authority",
            "@path",
        ),
        digest_algorithm: str = "sha-256",
        include_created: bool = True,
        expires_in: int | None = None,
        use_nonce: bool = False,
        label: str = "sig1",
        verify_response: bool = False,
        require_response_signature: bool = False,
        max_skew: int = 30,
        max_age: int | None = None,
        replay_protection: bool = True,
    ) -> None:
        if digest_algorithm not in _DIGEST_ALGORITHMS:
            raise ValueError(
                f"Unsupported digest algorithm {digest_algorithm!r}."
            )
        self.key_resolver = key_resolver
        self.keyid = keyid
        self.alg = alg
        self.components = tuple(components)
        self.digest_algorithm = digest_algorithm
        self.include_created = include_created
        self.expires_in = expires_in
        self.use_nonce = use_nonce
        self.label = label
        # Note: stored under a private name so it does not shadow the
        # public 'verify_response()' method.
        self.verify_responses = verify_response
        self.require_response_signature = require_response_signature
        self.max_skew = max_skew
        self.max_age = max_age
        self.replay_protection = replay_protection
        self._seen_nonces: set[tuple[str, str]] = set()

    # -- signing --

    def _resolve_keyid(self, request: Request) -> str:
        keyid = self.keyid or getattr(
            self.key_resolver, "default_keyid", None
        )
        if not keyid:
            raise SignatureSigningError(
                "No keyid configured and the resolver has no default keyid.",
                request=request,
            )
        return keyid

    def sign_request(self, request: Request) -> None:
        """
        Sign an outgoing request in place. Any pre-existing signature
        headers (e.g. carried over from a previous redirect hop) are
        replaced with a fresh signature bound to the current URL.
        """
        self._strip_signature_headers(request)

        keyid = self._resolve_keyid(request)
        covered = list(self.components)

        if "content-digest" in covered:
            has_body = self._digest_sync(request)
            if not has_body:
                covered.remove("content-digest")

        created = int(time.time()) if self.include_created else None
        expires = (
            created + self.expires_in
            if self.expires_in is not None and created is not None
            else None
        )
        nonce = (
            base64.b64encode(secrets.token_bytes(16)).decode("ascii")
            if self.use_nonce
            else None
        )

        try:
            key = self.key_resolver.resolve_key(keyid, request)
        except SignatureError:
            raise
        except Exception as exc:
            raise SignatureSigningError(
                f"Failed to resolve key for keyid={keyid!r}: {exc}",
                request=request,
            ) from exc

        signature_base = _build_signature_base(
            covered,
            lambda component: _request_component_value(component, request),
        )

        try:
            signature = _sign_with_algorithm(self.alg, key, signature_base)
        except SignatureError:
            raise
        except Exception as exc:
            raise SignatureSigningError(
                f"Failed to sign request: {exc}", request=request
            ) from exc

        params = self._signature_params(
            keyid=keyid, created=created, expires=expires, nonce=nonce
        )
        request.headers["Signature-Input"] = (
            f"{self.label}=" + _serialize_inner_list(covered, params)
        )
        request.headers["Signature"] = f"{self.label}=:{_b64(signature)}:"

    async def asign_request(self, request: Request) -> None:
        """
        Asynchronous counterpart to `.sign_request()`.
        """
        self._strip_signature_headers(request)

        keyid = self._resolve_keyid(request)
        covered = list(self.components)

        if "content-digest" in covered:
            has_body = await self._adigest(request)
            if not has_body:
                covered.remove("content-digest")

        created = int(time.time()) if self.include_created else None
        expires = (
            created + self.expires_in
            if self.expires_in is not None and created is not None
            else None
        )
        nonce = (
            base64.b64encode(secrets.token_bytes(16)).decode("ascii")
            if self.use_nonce
            else None
        )

        try:
            key = await self.key_resolver.aresolve_key(keyid, request)
        except SignatureError:
            raise
        except Exception as exc:
            raise SignatureSigningError(
                f"Failed to resolve key for keyid={keyid!r}: {exc}",
                request=request,
            ) from exc

        signature_base = _build_signature_base(
            covered,
            lambda component: _request_component_value(component, request),
        )

        try:
            signature = _sign_with_algorithm(self.alg, key, signature_base)
        except SignatureError:
            raise
        except Exception as exc:
            raise SignatureSigningError(
                f"Failed to sign request: {exc}", request=request
            ) from exc

        params = self._signature_params(
            keyid=keyid, created=created, expires=expires, nonce=nonce
        )
        request.headers["Signature-Input"] = (
            f"{self.label}=" + _serialize_inner_list(covered, params)
        )
        request.headers["Signature"] = f"{self.label}=:{_b64(signature)}:"

    def _digest_sync(self, request: Request) -> bool:
        if hasattr(request, "_content"):
            body = request.content
            if body:
                digest = _DIGEST_ALGORITHMS[self.digest_algorithm]()
                digest.update(body)
                request.headers["Content-Digest"] = _format_content_digest(
                    self.digest_algorithm, digest.digest()
                )
                return True
            return False
        body = _digest_body_sync(request, self.digest_algorithm)
        if body:
            digest = _DIGEST_ALGORITHMS[self.digest_algorithm]()
            digest.update(body)
            request.headers["Content-Digest"] = _format_content_digest(
                self.digest_algorithm, digest.digest()
            )
            return True
        return False

    async def _adigest(self, request: Request) -> bool:
        if hasattr(request, "_content"):
            body = request.content
            if body:
                digest = _DIGEST_ALGORITHMS[self.digest_algorithm]()
                digest.update(body)
                request.headers["Content-Digest"] = _format_content_digest(
                    self.digest_algorithm, digest.digest()
                )
                return True
            return False
        body = await _adigest_body(request, self.digest_algorithm)
        if body:
            digest = _DIGEST_ALGORITHMS[self.digest_algorithm]()
            digest.update(body)
            request.headers["Content-Digest"] = _format_content_digest(
                self.digest_algorithm, digest.digest()
            )
            return True
        return False

    def _signature_params(
        self,
        *,
        keyid: str,
        created: int | None,
        expires: int | None,
        nonce: str | None,
    ) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = [("keyid", _sf_string(keyid))]
        if self.alg is not None:
            params.append(("alg", _sf_string(self.alg)))
        if created is not None:
            params.append(("created", str(created)))
        if expires is not None:
            params.append(("expires", str(expires)))
        if nonce is not None:
            params.append(("nonce", _sf_string(nonce)))
        return params

    @staticmethod
    def _strip_signature_headers(request: Request) -> None:
        request.headers.pop("Signature", None)
        request.headers.pop("Signature-Input", None)
        request.headers.pop("Content-Digest", None)

    # -- verification --

    def verify_response(self, response: Response) -> None:
        """
        Verify a response's message signature, if present and verification
        is enabled. Wraps the response stream so that a signed
        `Content-Digest` is checked incrementally as the body is consumed.
        """
        covered, params, signature, keyid = self._prepare_verify(response)
        if covered is None:
            return

        try:
            key = self.key_resolver.resolve_key(
                keyid, response.request, for_response=True
            )
        except SignatureError:
            raise
        except Exception as exc:
            raise InvalidSignature(
                f"Failed to resolve key for keyid={keyid!r}: {exc}",
                request=response.request,
            ) from exc

        self._finish_verify(response, covered, params, signature, keyid, key)

    async def averify_response(self, response: Response) -> None:
        """
        Asynchronous counterpart to `.verify_response()`.
        """
        covered, params, signature, keyid = self._prepare_verify(response)
        if covered is None:
            return

        try:
            key = await self.key_resolver.aresolve_key(
                keyid, response.request, for_response=True
            )
        except SignatureError:
            raise
        except Exception as exc:
            raise InvalidSignature(
                f"Failed to resolve key for keyid={keyid!r}: {exc}",
                request=response.request,
            ) from exc

        self._finish_verify(response, covered, params, signature, keyid, key)

    def _prepare_verify(
        self, response: Response
    ) -> tuple[list[str] | None, dict[str, str], bytes, str]:
        if not self.verify_responses:
            return None, {}, b"", ""

        signature_input = response.headers.get("Signature-Input")
        if signature_input is None:
            if self.require_response_signature:
                raise InvalidSignature(
                    "Response is missing a 'Signature-Input' header.",
                    request=response.request,
                )
            return None, {}, b"", ""

        label, covered, params = _parse_signature_input(signature_input)

        signature_header = response.headers.get("Signature")
        if signature_header is None:
            raise InvalidSignature(
                "Response is missing a 'Signature' header.",
                request=response.request,
            )
        signature = _parse_signature_header(signature_header, label)

        keyid = params.get("keyid", "")
        self._check_timestamps(params, keyid, response.request)
        self._check_nonce(params, keyid, response.request)

        return covered, params, signature, keyid

    def _finish_verify(
        self,
        response: Response,
        covered: list[str],
        params: dict[str, str],
        signature: bytes,
        keyid: str,
        key: typing.Any,
    ) -> None:
        alg = params.get("alg", self.alg)
        signature_base = _build_signature_base(
            covered,
            lambda component: _response_component_value(component, response),
        )
        _verify_with_algorithm(alg, key, signature, signature_base)

        self._wrap_response_digest(response)

    def _wrap_response_digest(self, response: Response) -> None:
        content_digest = response.headers.get("Content-Digest")
        if content_digest is None:
            return
        digests = _parse_content_digest(content_digest)
        algo: str | None = None
        for candidate in digests:
            if candidate in _DIGEST_ALGORITHMS:
                algo = candidate
                break
        if algo is None:
            raise InvalidSignature(
                "No supported algorithm in 'Content-Digest' header.",
                request=response.request,
            )
        request = response.request

        # If the body has already been buffered (e.g. test transports, or
        # response constructed with 'content=...') the stream wrappers below
        # would never run, so verify the buffered bytes immediately.
        if hasattr(response, "_content"):
            actual = _hash_content(algo, response._content)
            if not hmac.compare_digest(actual, digests[algo]):
                raise ContentDigestError(
                    "Response Content-Digest does not match the received body.",
                    request=request,
                )
            return

        if isinstance(response.stream, SyncByteStream):
            response.stream = _DigestVerifySyncStream(
                response.stream, algo, digests[algo], request
            )
        else:
            response.stream = _DigestVerifyAsyncStream(
                response.stream, algo, digests[algo], request
            )

    def _check_timestamps(
        self, params: dict[str, str], keyid: str, request: Request
    ) -> None:
        now = int(time.time())
        created_raw = params.get("created")
        expires_raw = params.get("expires")
        created = int(created_raw) if created_raw is not None else None
        expires = int(expires_raw) if expires_raw is not None else None

        if created is not None and created - now > self.max_skew:
            raise SignatureNotYetValid(
                f"Signature from keyid={keyid!r} is not valid yet: "
                f"created={created}, now={now}.",
                request=request,
                keyid=keyid,
                created=created,
                now=now,
            )

        if expires is not None and now - expires > self.max_skew:
            raise SignatureExpired(
                f"Signature from keyid={keyid!r} has expired: "
                f"expires={expires}, now={now}.",
                request=request,
                keyid=keyid,
                created=created,
                expires=expires,
                now=now,
            )

        if self.max_age is not None and created is not None:
            age = now - created
            if age > self.max_age + self.max_skew:
                raise SignatureExpired(
                    f"Signature from keyid={keyid!r} exceeds max age: "
                    f"created={created}, age={age}s, max_age={self.max_age}s.",
                    request=request,
                    keyid=keyid,
                    created=created,
                    expires=expires,
                    now=now,
                )

    def _check_nonce(
        self, params: dict[str, str], keyid: str, request: Request
    ) -> None:
        nonce = params.get("nonce")
        if nonce is None:
            return
        if self.replay_protection:
            marker = (keyid, nonce)
            if marker in self._seen_nonces:
                raise DuplicateNonce(
                    f"Signature nonce {nonce!r} for keyid={keyid!r} has "
                    "already been seen.",
                    request=request,
                    keyid=keyid,
                    nonce=nonce,
                )
            self._seen_nonces.add(marker)


def _serialize_inner_list(
    covered: typing.Sequence[str], params: typing.Sequence[tuple[str, str]]
) -> str:
    value = "(" + " ".join(_sf_string(c) for c in covered) + ")"
    for key, val in params:
        value += f";{key}={val}"
    return value
