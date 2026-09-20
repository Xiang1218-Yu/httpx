When making a request over HTTPS, HTTPX needs to verify the identity of the requested host. To do this, it uses a bundle of SSL certificates (a.k.a. CA bundle) delivered by a trusted certificate authority (CA).

### Enabling and disabling verification

By default httpx will verify HTTPS connections, and raise an error for invalid SSL cases...

```pycon
>>> httpx.get("https://expired.badssl.com/")
httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:997)
```

You can disable SSL verification completely and allow insecure requests...

```pycon
>>> httpx.get("https://expired.badssl.com/", verify=False)
<Response [200 OK]>
```

### Configuring client instances

If you're using a `Client()` instance you should pass any `verify=<...>` configuration when instantiating the client.

By default the [certifi CA bundle](https://certifiio.readthedocs.io/en/latest/) is used for SSL verification.

For more complex configurations you can pass an [SSL Context](https://docs.python.org/3/library/ssl.html) instance...

```python
import certifi
import httpx
import ssl

# This SSL context is equivalent to the default `verify=True`.
ctx = ssl.create_default_context(cafile=certifi.where())
client = httpx.Client(verify=ctx)
```

Using [the `truststore` package](https://truststore.readthedocs.io/) to support system certificate stores...

```python
import ssl
import truststore
import httpx

# Use system certificate stores.
ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
client = httpx.Client(verify=ctx)
```

Loding an alternative certificate verification store using [the standard SSL context API](https://docs.python.org/3/library/ssl.html)...

```python
import httpx
import ssl

# Use an explicitly configured certificate store.
ctx = ssl.create_default_context(cafile="path/to/certs.pem")  # Either cafile or capath.
client = httpx.Client(verify=ctx)
```

### Client side certificates

Client side certificates allow a remote server to verify the client. They tend to be used within private organizations to authenticate requests to remote servers.

You can specify client-side certificates, using the [`.load_cert_chain()`](https://docs.python.org/3/library/ssl.html#ssl.SSLContext.load_cert_chain) API...

```python
ctx = ssl.create_default_context()
ctx.load_cert_chain(certfile="path/to/client.pem")  # Optionally also keyfile or password.
client = httpx.Client(verify=ctx)
```

### Working with `SSL_CERT_FILE` and `SSL_CERT_DIR`

`httpx` does respect the `SSL_CERT_FILE` and `SSL_CERT_DIR` environment variables by default. For details, refer to [the section on the environment variables page](../environment_variables.md#ssl_cert_file).

### Per-origin TLS policies

For multi-tenant clients that need to select the CA bundle, client certificate, ALPN protocols, or hostname verification strategy dynamically, pass a TLS policy resolver to the client.

A resolver subclasses `httpx.TLSPolicyResolver` and implements `resolve(origin)`, which is called with the target `httpx.Origin` of each request, *before* any connection is made. It returns an `httpx.TLSPolicy` carrying the `SSLContext`, an optional `http2` flag (which drives ALPN negotiation), and a hashable `key` that identifies equivalent policies...

```python
import httpx
import ssl


class TenantTLSPolicyResolver(httpx.TLSPolicyResolver):
    def resolve(self, origin: httpx.Origin) -> httpx.TLSPolicy:
        tenant = self._lookup_tenant(origin.host)
        ctx = ssl.create_default_context(cafile=tenant.ca_bundle)
        ctx.load_cert_chain(certfile=tenant.client_cert)
        # Equal keys share a connection pool; a different key (or a new
        # resolver generation) creates a fresh pool.
        return httpx.TLSPolicy(ctx, http2=False, key=(tenant.id, tenant.version))

    def _lookup_tenant(self, host: str):
        ...


client = httpx.Client(tls_policy=TenantTLSPolicyResolver())
```

You can also construct a policy from the same `verify`, `cert`, `trust_env`, `alpn_protocols` and `check_hostname` primitives that the client accepts, using `httpx.TLSPolicy.create(...)`.

Connections are pooled per `(resolver generation, policy key)`, so a connection negotiated for one policy is never reused for a different one. When configured policies change, call `resolver.invalidate()` to bump the generation: subsequent requests resolve fresh policies and use new pools, while pools from older generations are closed as soon as any in-flight requests have drained, including requests that were mid-handshake. Redirected requests resolve the policy for the new origin on every hop.

If resolution fails, an `httpx.TLSPolicyError` is raised before the connection is attempted. The resolver works for both direct connections and HTTP/SOCKS proxies, for `Client` and `AsyncClient` (async resolvers may additionally implement `aresolve(origin)`), and cannot be combined with a custom `transport=` instance — pass `tls_policy` to an `httpx.HTTPTransport` instead.

### Making HTTPS requests to a local server

When making requests to local servers, such as a development server running on `localhost`, you will typically be using unencrypted HTTP connections.

If you do need to make HTTPS connections to a local server, for example to test an HTTPS-only service, you will need to create and use your own certificates. Here's one way to do it...

1. Use [trustme](https://github.com/python-trio/trustme) to generate a pair of server key/cert files, and a client cert file.
2. Pass the server key/cert files when starting your local server. (This depends on the particular web server you're using. For example, [Uvicorn](https://www.uvicorn.org) provides the `--ssl-keyfile` and `--ssl-certfile` options.)
3. Configure `httpx` to use the certificates stored in `client.pem`.

```python
ctx = ssl.create_default_context(cafile="client.pem")
client = httpx.Client(verify=ctx)
```
