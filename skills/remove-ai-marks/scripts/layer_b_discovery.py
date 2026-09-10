#!/usr/bin/env python3
"""Layer B endpoint policy and model discovery.

A front end that lets an operator type a base URL needs two things the rewrite
path already owns internally: is this endpoint allowed to receive my document,
and which models does it serve.  Both are exposed here so no caller has to
reimplement them — and, critically, so no caller reaches for ``urllib``
directly.  Every request goes through ``layer_b_http``, inheriting its
same-origin redirect opener, header validation, and bounded JSON reader.

Nothing here sends document content.  Discovery is a read of the endpoint's own
model list; the text only ever leaves via ``rewrite_text.rewrite``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layer_b_http import LayerBHTTPError, get_json
from rewrite_text import LOOPBACK_HOSTS

#: Discovery is a liveness check, not a generation: fail fast rather than hang
#: a UI thread behind a model-load timeout.
DEFAULT_PROBE_TIMEOUT = 5.0

#: Route each live backend serves its model list on.
MODEL_ROUTES = {
    "ollama": "/api/tags",
    "openai-compatible": "/v1/models",
}


class EndpointPolicyError(ValueError):
    """The endpoint is not allowed to receive content under the current policy."""


@dataclass(frozen=True, slots=True)
class EndpointPolicy:
    """What sending to this endpoint would mean, decided before anything is sent."""

    base_url: str
    scheme_ok: bool
    host: str
    loopback: bool
    allowed: bool
    #: Verbatim operator-facing warning when this endpoint is off-machine.
    warning: str | None = None
    reason: str | None = None


def classify_endpoint(base_url: str | None, *, allow_remote: bool = False) -> EndpointPolicy:
    """Decide whether *base_url* may receive document content.

    Mirrors ``rewrite_text._check_remote``'s default-deny rule — loopback is
    free, anything else needs an explicit opt-in, non-http(s) is always
    refused — but returns the verdict instead of exiting, so a UI can render it
    before the operator commits.
    """
    if not base_url:
        return EndpointPolicy(
            base_url="",
            scheme_ok=False,
            host="",
            loopback=False,
            allowed=False,
            reason="no base URL set",
        )
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return EndpointPolicy(
            base_url=base_url,
            scheme_ok=False,
            host="",
            loopback=False,
            allowed=False,
            reason="unparseable base URL",
        )
    if parsed.scheme not in ("http", "https"):
        return EndpointPolicy(
            base_url=base_url,
            scheme_ok=False,
            host=parsed.hostname or "",
            loopback=False,
            allowed=False,
            reason=f"base URL must be http(s), got scheme '{parsed.scheme}'",
        )
    host = parsed.hostname or ""
    if host in LOOPBACK_HOSTS:
        return EndpointPolicy(
            base_url=base_url,
            scheme_ok=True,
            host=host,
            loopback=True,
            allowed=True,
        )
    warning = (
        f"warning: rewrite base URL host is '{host}' (not localhost); "
        "content will leave this machine"
    )
    if not allow_remote:
        return EndpointPolicy(
            base_url=base_url,
            scheme_ok=True,
            host=host,
            loopback=False,
            allowed=False,
            warning=warning,
            reason=(
                f"base URL host is not loopback ('{host}'); refusing to send content "
                "off-machine without an explicit opt-in"
            ),
        )
    return EndpointPolicy(
        base_url=base_url,
        scheme_ok=True,
        host=host,
        loopback=False,
        allowed=True,
        warning=warning,
    )


@dataclass(frozen=True, slots=True)
class BackendProbe:
    """Result of asking an endpoint what it serves."""

    backend: str
    base_url: str
    reachable: bool
    models: tuple[str, ...] = ()
    error: str | None = None
    policy: EndpointPolicy | None = field(default=None, repr=False)

    @property
    def summary(self) -> str:
        if not self.reachable:
            return self.error or "unreachable"
        if not self.models:
            return "reachable, no models listed"
        return f"reachable, {len(self.models)} model(s)"


def _models_from_payload(backend: str, payload: dict) -> tuple[str, ...]:
    """Pull model names out of either provider's list shape, defensively.

    The payload is operator-supplied remote data: every level is checked rather
    than indexed, and anything unexpected yields no models instead of raising.
    """
    names: list[str] = []
    if backend == "ollama":
        entries = payload.get("models")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    name = entry.get("name") or entry.get("model")
                    if isinstance(name, str) and name:
                        names.append(name)
    else:
        entries = payload.get("data")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    name = entry.get("id")
                    if isinstance(name, str) and name:
                        names.append(name)
    # Stable, de-duplicated, and bounded: a hostile or misconfigured endpoint
    # must not be able to flood a picker with unbounded entries.
    seen: dict[str, None] = {}
    for name in names:
        seen.setdefault(name, None)
    return tuple(sorted(seen))[:500]


def probe_backend(
    backend: str,
    base_url: str | None,
    *,
    api_key: str | None = None,
    allow_remote: bool = False,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> BackendProbe:
    """Ask *base_url* which models it serves. Never raises.

    Refuses before any request when the endpoint policy would refuse the
    rewrite itself, so discovery can never be the thing that reaches a host the
    rewrite path would have denied.
    """
    policy = classify_endpoint(base_url, allow_remote=allow_remote)
    if not policy.allowed:
        return BackendProbe(
            backend=backend,
            base_url=base_url or "",
            reachable=False,
            error=policy.reason,
            policy=policy,
        )
    route = MODEL_ROUTES.get(backend)
    if route is None:
        return BackendProbe(
            backend=backend,
            base_url=policy.base_url,
            reachable=False,
            error=f"{backend} has no model-discovery route",
            policy=policy,
        )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        payload = get_json(policy.base_url, route, headers=headers, timeout=timeout)
    except LayerBHTTPError as error:
        return BackendProbe(
            backend=backend,
            base_url=policy.base_url,
            reachable=False,
            error=str(error),
            policy=policy,
        )
    except Exception as error:  # discovery must never take down the caller
        return BackendProbe(
            backend=backend,
            base_url=policy.base_url,
            reachable=False,
            error=f"{type(error).__name__}: {error}",
            policy=policy,
        )
    return BackendProbe(
        backend=backend,
        base_url=policy.base_url,
        reachable=True,
        models=_models_from_payload(backend, payload),
        policy=policy,
    )


def layer_b_status(
    backend: str | None,
    base_url: str | None,
    *,
    allow_remote: bool = False,
) -> dict[str, object]:
    """Configuration-only Layer B summary for capability reporting.

    Deliberately does not touch the network: ``/capabilities`` is polled and
    must not turn into an outbound request per call.
    """
    policy = classify_endpoint(base_url, allow_remote=allow_remote)
    return {
        "backend": backend,
        "configured": bool(backend and backend in MODEL_ROUTES and base_url),
        "base_url_host": policy.host or None,
        "loopback": policy.loopback,
        "endpoint_allowed": policy.allowed,
    }
