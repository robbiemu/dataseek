"""Sync LLM rate limiter with manual (DSL) and real (HTTP header) modes.

Two limiting modes share one limiter:

- ``manual``: the operator feeds a rate limit via a compact string DSL
  (``"30m #burst, 500h #steady, 10000d"``) to match a perceived provider quota
  at any granularity. Multiple limits compose; the tightest active bound wins.
- ``real``: follow official HTTP rate-limit headers (IETF ``RateLimit``/
  ``RateLimit-Policy``, ``RateLimit-Limit/Remaining/Reset``, legacy
  ``X-RateLimit-*``, and ``Retry-After``) to pace proactively.

Both modes do pre-call pacing (sleep before sending if exhausted) and the
limiter exposes :meth:`RateLimiter.apply_retry_after` for reactive 429 backoff.

Internal timing uses :func:`time.monotonic` throughout; wall-clock header
values are converted to monotonic deadlines at the parse boundary so NTP
adjustments cannot move the floor mid-wait.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Unit -> seconds. The DSL is count-plus-window: ``30m`` means 30 requests per
# minute, not 30 minutes.
_UNIT_SECONDS: dict[str, float] = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}

# A wall-clock epoch reset value above this threshold is treated as an absolute
# Unix timestamp; below it, as a relative number of seconds. The boundary is
# well below any plausible Unix epoch (1e9 ~= year 2001) and well above any
# plausible relative reset (provider reset windows are at most a few hours).
_ABSOLUTE_RESET_THRESHOLD = 1_000_000_000

# Provider prefix -> the env var litellm's dispatch chain reads for the API
# key. Mirrors litellm/main.py:_complete_openrouter (OPENROUTER_API_KEY) and
# _complete_custom_openai (OPENAI_API_KEY). Used to compute a non-secret
# credential identity for the limiter's scope key.
_PROVIDER_ENV_KEY: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
    "azure": "AZURE_API_KEY",
}

# A 429 response header value that looks like "32/32" (Nvidia's non-standard
# limit string). Best-effort: parse remaining as (used, limit) -> remaining =
# limit - used. Low confidence; the limiter logs and never over-throttles on a
# misparse.
_NVIDIA_LIMIT_RE = re.compile(r"^\s*(?P<used>\d+)\s*/\s*(?P<limit>\d+)\s*$")


# --------------------------------------------------------------------------- #
# DSL parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Limit:
    """One manual rate limit: ``count`` requests per ``window_seconds``.

    ``role`` is a freeform operator tag (``burst``, ``steady``, ``free-tier``)
    used only for logging which bound gated a call.
    """

    count: int
    window_seconds: float
    role: str | None = None


# One DSL token: ``<count><unit>`` with an optional ``#role`` tag. Unit is a
# single letter (s/m/h/d); role allows word chars and dashes (``#free-tier``).
_LIMIT_TOKEN_RE = re.compile(r"^(?P<count>\d+)(?P<unit>[smhd])(?:\s+#(?P<role>[\w-]+))?$")


def parse_limits_dsl(s: str) -> list[Limit]:
    """Parse the manual rate-limit DSL into a list of :class:`Limit`.

    Grammar: ``"<count><unit> [#role][, <count><unit> [#role]]..."`` where unit
    is ``s``/``m``/``h``/``d``. ``30m`` means 30 requests per minute (count
    plus window, not a duration). Multiple limits compose; the tightest active
    bound wins at runtime.

    Raises :class:`ValueError` naming the offending token on any parse failure
    so misconfigurations surface loudly at config load, not silently at runtime.
    """
    if not s or not s.strip():
        raise ValueError("rate-limit DSL is empty")
    limits: list[Limit] = []
    for raw in s.split(","):
        token = raw.strip()
        if not token:
            raise ValueError(f"rate-limit DSL has an empty token in: {s!r}")
        m = _LIMIT_TOKEN_RE.match(token)
        if not m:
            raise ValueError(
                f"rate-limit DSL token {token!r} is malformed; "
                "expected '<count><unit> [#role]' with unit in s/m/h/d "
                "(e.g. '30m #burst', '500h', '10000d')"
            )
        count = int(m.group("count"))
        unit = m.group("unit")
        if count <= 0:
            raise ValueError(f"rate-limit DSL token {token!r}: count must be positive")
        limits.append(
            Limit(
                count=count,
                window_seconds=_UNIT_SECONDS[unit],
                role=m.group("role"),
            )
        )
    return limits


# --------------------------------------------------------------------------- #
# HTTP rate-limit header parsing (shared with the async search limiter)
# --------------------------------------------------------------------------- #


@dataclass
class RateLimitHeaders:
    """Parsed rate-limit headers from one provider response.

    ``reset_at`` is an absolute :func:`time.monotonic` deadline (not wall
    clock). ``retry_after`` is seconds (from the ``Retry-After`` header, or
    ``None`` if absent).
    """

    limit: int | None = None
    remaining: int | None = None
    reset_at: float | None = None
    retry_after: float | None = None
    policy: str | None = None
    # The header family that was parsed, for observability/debugging.
    source: str | None = None


def _parse_reset_to_monotonic(value: str) -> float | None:
    """Normalize a reset header value to an absolute monotonic deadline.

    Providers emit reset as either an absolute Unix epoch (seconds) or a
    relative number of seconds. A value above :data:`_ABSOLUTE_RESET_THRESHOLD`
    is treated as absolute epoch and converted to a monotonic deadline; a
    smaller value is treated as relative seconds from now.
    """
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    if n >= _ABSOLUTE_RESET_THRESHOLD:
        # Absolute wall-clock epoch -> monotonic deadline. The conversion is
        # robust to wall/monotonic skew at this instant; subsequent waiting is
        # monotonic and immune to NTP drift.
        return time.monotonic() + (n - time.time())
    return time.monotonic() + n


def _first_header(headers: dict[str, Any], *names: str) -> str | None:
    """Case-insensitive lookup returning the first present header value."""
    lower = {k.lower(): v for k, v in headers.items()}
    for name in names:
        v = lower.get(name.lower())
        if v is not None:
            return str(v)
    return None


def _parse_comma_first_int(value: str | None) -> int | None:
    """Parse the first comma-separated integer from a header value.

    Some providers (e.g. Brave) return ``"0, 1995"``; we take the first field.
    """
    if value is None:
        return None
    first = str(value).split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def parse_rate_limit_headers(headers: dict[str, Any]) -> RateLimitHeaders | None:
    """Parse HTTP rate-limit headers from a provider response.

    Handles, in priority order (newest standard first, legacy de-facto last):

    1. ``Retry-After`` (RFC 7231 §7.1.3) — seconds or HTTP-date.
    2. ``RateLimit`` + ``RateLimit-Policy`` (IETF draft-ietf-httpapi-ratelimit-headers-11)
       — single structured header, e.g. ``RateLimit: limit=32, remaining=28``.
    3. ``RateLimit-Limit`` / ``RateLimit-Remaining`` / ``RateLimit-Reset``
       (earlier IETF draft, separate fields, no ``X-`` prefix).
    4. ``X-RateLimit-Limit`` / ``X-RateLimit-Remaining`` / ``X-RateLimit-Reset``
       (de-facto legacy; what OpenAI and OpenRouter actually emit today).
    5. Best-effort Nvidia ``"32/32"`` limit string (non-standard, low confidence).

    Returns ``None`` if no recognized rate-limit header is present.
    """
    if not headers:
        return None

    out = RateLimitHeaders()

    # Retry-After (may appear on 429/503 or proactively on 200).
    retry_after_raw = _first_header(headers, "retry-after")
    if retry_after_raw is not None:
        try:
            out.retry_after = float(retry_after_raw)
        except ValueError:
            # HTTP-date form (RFC 7231); parse to seconds from now.
            try:
                from email.utils import parsedate_to_datetime

                dt = parsedate_to_datetime(retry_after_raw)
                if dt is not None:
                    out.retry_after = max(0.0, dt.timestamp() - time.time())
            except (TypeError, ValueError):
                pass

    # IETF draft-11 single structured RateLimit header.
    ratelimit = _first_header(headers, "ratelimit")
    if ratelimit:
        # e.g. "limit=32, remaining=28; q=1" — parse limit/remaining key=value
        # pairs from the first semicolon-delimited segment.
        primary = ratelimit.split(";")[0]
        for pair in primary.split(","):
            kv = pair.strip().split("=", 1)
            if len(kv) != 2:
                continue
            k, v = kv[0].strip(), kv[1].strip()
            if k == "limit":
                with suppress(ValueError):
                    out.limit = int(v)
            elif k == "remaining":
                with suppress(ValueError):
                    out.remaining = int(v)
        out.policy = _first_header(headers, "ratelimit-policy")
        if out.limit is not None or out.remaining is not None:
            out.source = "ietf-ratelimit"
            # draft-11 reset lives in the policy/quota params; fall through to
            # the separate reset headers below if present.

    # Earlier IETF draft: RateLimit-Limit/Remaining/Reset (no X- prefix).
    if out.limit is None and out.remaining is None:
        rl_limit = _parse_comma_first_int(_first_header(headers, "ratelimit-limit"))
        rl_remaining = _parse_comma_first_int(_first_header(headers, "ratelimit-remaining"))
        if rl_limit is not None or rl_remaining is not None:
            out.limit = rl_limit
            out.remaining = rl_remaining
            reset_raw = _first_header(headers, "ratelimit-reset")
            if reset_raw is not None:
                out.reset_at = _parse_reset_to_monotonic(reset_raw)
            out.source = "ietf-ratelimit-fields"

    # Legacy de-facto: X-RateLimit-Limit/Remaining/Reset.
    if out.limit is None and out.remaining is None:
        xl_limit = _parse_comma_first_int(_first_header(headers, "x-ratelimit-limit"))
        xl_remaining = _parse_comma_first_int(_first_header(headers, "x-ratelimit-remaining"))
        if xl_limit is not None or xl_remaining is not None:
            out.limit = xl_limit
            out.remaining = xl_remaining
            reset_raw = _first_header(headers, "x-ratelimit-reset")
            if reset_raw is not None:
                out.reset_at = _parse_reset_to_monotonic(reset_raw)
            out.source = "x-ratelimit-legacy"

    # Reset may have been carried by the legacy/ietf fields even when limit/
    # remaining came from the structured header. Parse it if still missing.
    if out.reset_at is None:
        for name in ("ratelimit-reset", "x-ratelimit-reset"):
            reset_raw = _first_header(headers, name)
            if reset_raw is not None:
                out.reset_at = _parse_reset_to_monotonic(reset_raw)
                break

    # Best-effort Nvidia "32/32" limit string (non-standard). Only consulted
    # when nothing else was found, and logged at DEBUG so a misparse never
    # silently over-throttles.
    if out.limit is None and out.remaining is None and out.retry_after is None:
        for v in headers.values():
            m = _NVIDIA_LIMIT_RE.match(str(v))
            if m:
                used = int(m.group("used"))
                limit = int(m.group("limit"))
                out.limit = limit
                out.remaining = max(0, limit - used)
                out.source = "nvidia-limit-string"
                logger.debug(
                    "parsed non-standard Nvidia limit string %r -> "
                    "limit=%d remaining=%d (low confidence)",
                    str(v),
                    limit,
                    out.remaining,
                )
                break

    if (
        out.limit is None
        and out.remaining is None
        and out.reset_at is None
        and out.retry_after is None
    ):
        return None
    return out


# --------------------------------------------------------------------------- #
# Scope-key resolution
# --------------------------------------------------------------------------- #


def _credential_identity(provider: str, api_base: str | None, api_key: str | None) -> str:
    """Compute a non-secret identity for the credential litellm will use.

    Prefer the env-var *name* (non-secret, stable) over the key value. An
    explicit ``api_key`` yields a truncated hash (the value is already in
    memory in ``create_llm``, so no extra secret exposure). A keyless local
    server falls back to its hostname.
    """
    if api_key:
        import hashlib

        return f"explicit:{hashlib.sha256(api_key.encode()).hexdigest()[:12]}"
    env_var = _PROVIDER_ENV_KEY.get(provider)
    if env_var:
        return f"env:{env_var}"
    if api_base:
        host = urlparse(api_base).hostname or "unknown"
        return f"none:{host}"
    return "none:default"


def resolve_scope_key(
    model: str,
    api_base: str | None,
    api_key: str | None,
    scope: str,
) -> str:
    """Build the limiter's scope key from the resolved model identity.

    - ``scope="model"`` (default): ``provider | api_base | model | credential``
    - ``scope="provider"``: ``provider | api_base | credential``

    Two nodes pointing at the same model+api_base share a quota under model
    scope (implicit, no config). Provider scope shares across models under one
    provider+credential — the right choice when the provider enforces a per-key
    cross-model limit (the OpenRouter/Nvidia free-tier case).
    """
    # litellm.get_llm_provider reliably returns the provider prefix string;
    # the legacy litellm.get_api_key() and OpenrouterConfig.get_api_key() are
    # NOT used (both give wrong/None answers for openrouter).
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    try:
        import litellm

        _, provider, _, _ = litellm.get_llm_provider(model, api_base=api_base)
    except Exception:
        pass  # fall back to the prefix split above
    cred = _credential_identity(provider, api_base, api_key)
    base = f"{provider} | {api_base or 'default'} | {cred}"
    if scope == "provider":
        return base
    return f"{base} | {model}"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RateLimitConfig:
    """Resolved rate-limit configuration for one endpoint.

    ``mode="off"`` is the default and disables limiting entirely (current
    behavior for any endpoint without a ``rate_limit`` block).
    """

    mode: str = "off"  # "manual" | "real" | "off"
    limits: tuple[Limit, ...] = ()
    scope: str = "model"  # "model" | "provider"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RateLimitConfig:
        if not d:
            return cls()
        mode = str(d.get("mode", "off")).lower()
        if mode not in {"manual", "real", "off"}:
            raise ValueError(f"rate_limit.mode {mode!r} must be one of: manual, real, off")
        scope = str(d.get("scope", "model")).lower()
        if scope not in {"model", "provider"}:
            raise ValueError(f"rate_limit.scope {scope!r} must be one of: model, provider")
        limits: tuple[Limit, ...] = ()
        if mode == "manual":
            raw = d.get("limits")
            if not raw:
                raise ValueError(
                    "rate_limit.mode=manual requires a 'limits' DSL string "
                    '(e.g. "30m #burst, 500h #steady")'
                )
            limits = tuple(parse_limits_dsl(str(raw)))
        elif mode == "real":
            # Optional fallback ceiling when the server sends no headers.
            raw = d.get("limits")
            if raw:
                limits = tuple(parse_limits_dsl(str(raw)))
        return cls(mode=mode, limits=limits, scope=scope)

    @property
    def active(self) -> bool:
        return self.mode != "off"


# --------------------------------------------------------------------------- #
# The limiter
# --------------------------------------------------------------------------- #


@dataclass
class _KeyState:
    """Per-scope-key limiter state."""

    # Manual mode: one deque of admit timestamps per Limit (matched by index
    # into RateLimitConfig.limits). Each deque holds monotonic timestamps of
    # calls admitted within that limit's window.
    manual_windows: list[deque[float]] = field(default_factory=list)
    # Real mode: last-known header-derived remaining count + reset deadline.
    remaining: int | None = None
    reset_at: float | None = None
    # One-shot reactive cooldown set by apply_retry_after on a 429.
    cooldown_until: float | None = None


class RateLimiter:
    """Sync, thread-safe LLM rate limiter.

    One process-wide instance (:data:`RATE_LIMITER`) holds per-scope-key state.
    :meth:`acquire` blocks before a call until pacing allows it; the wrapper
    around ``invoke`` calls it. :meth:`update_from_response` feeds header state
    back for real mode. :meth:`apply_retry_after` sets a one-shot cooldown on a
    429 so the next ``max_retries`` attempt doesn't immediately re-trip.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, _KeyState] = {}

    def _state(self, key: str, n_limits: int) -> _KeyState:
        st = self._states.get(key)
        if st is None:
            st = _KeyState(manual_windows=[deque() for _ in range(n_limits)])
            self._states[key] = st
        # Grow the manual windows list if the config grew since first use.
        while len(st.manual_windows) < n_limits:
            st.manual_windows.append(deque())
        return st

    def acquire(self, key: str, config: RateLimitConfig) -> None:
        """Block until a call may proceed under ``config``.

        Manual mode enforces every limit in the list; the call waits until the
        **maximum** deadline across all exhausted limits (not the minimum), so a
        burst cap and a steady cap can't leak past the steady bound. The lock is
        released before sleeping and reacquired to re-check, so other threads
        aren't blocked while this one waits. After admission, a timestamp is
        appended to every active limit's window atomically.
        """
        if not config.active:
            return

        while True:
            with self._lock:
                st = self._state(key, len(config.limits))
                now = time.monotonic()

                # Reactive cooldown (from a prior 429's Retry-After).
                if st.cooldown_until is not None and now < st.cooldown_until:
                    wait = st.cooldown_until - now
                else:
                    st.cooldown_until = None
                    wait = self._compute_wait(st, config, now)

                if wait <= 0:
                    # Admitted: record a timestamp in every active window.
                    for window in st.manual_windows:
                        window.append(now)
                    return
                # Release the lock for the duration of the sleep so other keys
                # aren't blocked. Re-check on wake in case state moved.

            logger.debug(
                "rate_limit key=%s mode=%s sleeping %.3fs",
                key,
                config.mode,
                wait,
            )
            time.sleep(wait)

    def _compute_wait(self, st: _KeyState, config: RateLimitConfig, now: float) -> float:
        """Return seconds to wait, or 0 if the call may proceed now."""
        wait = 0.0

        if config.mode == "manual":
            # Enforce every limit; gate on the maximum deadline across all
            # exhausted limits so the tightest bound wins.
            for limit, window in zip(config.limits, st.manual_windows, strict=False):
                # Evict timestamps outside this limit's window.
                cutoff = now - limit.window_seconds
                while window and window[0] <= cutoff:
                    window.popleft()
                if len(window) >= limit.count:
                    # Time until the oldest in-window call ages out.
                    need = window[0] + limit.window_seconds - now
                    if need > wait:
                        wait = need
                        logger.debug(
                            "rate_limit bound %r (count=%d window=%.0fs) " "exhausted; wait=%.3fs",
                            limit.role or "(none)",
                            limit.count,
                            limit.window_seconds,
                            wait,
                        )

        elif config.mode == "real":
            # Header-driven: if remaining is known and exhausted, wait for the
            # reset deadline. Fall back to any configured manual ceiling when
            # the server has not yet sent headers.
            if (
                st.remaining is not None
                and st.remaining <= 0
                and st.reset_at is not None
                and st.reset_at > now
            ):
                wait = max(wait, st.reset_at - now)
            for limit, window in zip(config.limits, st.manual_windows, strict=False):
                cutoff = now - limit.window_seconds
                while window and window[0] <= cutoff:
                    window.popleft()
                if len(window) >= limit.count:
                    need = window[0] + limit.window_seconds - now
                    if need > wait:
                        wait = need

        return wait

    def update_from_response(self, key: str, headers: dict[str, Any]) -> None:
        """Feed response headers into real-mode state for ``key``."""
        parsed = parse_rate_limit_headers(headers)
        if parsed is None:
            return
        with self._lock:
            st = self._state(key, 0)
            if parsed.remaining is not None:
                st.remaining = parsed.remaining
            if parsed.reset_at is not None:
                st.reset_at = parsed.reset_at
            logger.debug(
                "rate_limit key=%s updated from %s: remaining=%s reset_at=%.2fs",
                key,
                parsed.source,
                parsed.remaining,
                (parsed.reset_at - time.monotonic()) if parsed.reset_at else None,
            )

    def apply_retry_after(self, key: str, headers: dict[str, Any] | None) -> float:
        """Set a one-shot cooldown from a 429's ``Retry-After``.

        Returns the seconds to sleep (0 if no Retry-After was present). The
        cooldown gates the next :meth:`acquire`; after it elapses it is cleared.
        Called by the invoke wrapper when it catches a 429-ish exception, so
        litellm's own ``max_retries`` retry honors the server's backoff rather
        than immediately re-tripping.
        """
        if not headers:
            return 0.0
        parsed = parse_rate_limit_headers(headers)
        if parsed is None or parsed.retry_after is None:
            return 0.0
        with self._lock:
            st = self._state(key, 0)
            st.cooldown_until = time.monotonic() + parsed.retry_after
        logger.debug(
            "rate_limit key=%s cooldown %.2fs (Retry-After)",
            key,
            parsed.retry_after,
        )
        return parsed.retry_after


# Process-wide singleton. The litellm callback and the invoke wrapper both
# reference this instance; tests may construct their own RateLimiter for
# isolation and pass it to the wiring explicitly.
RATE_LIMITER = RateLimiter()
