"""litellm success/failure callback that feeds response headers to the limiter.

Registered lazily and idempotently from :func:`create_llm` when an endpoint's
``rate_limit.mode != "off"``. The callback is a **no-op unless dataseek
metadata is present** on the call, so registering it globally is safe even in
tests that bypass dataseek's wiring: calls without the
``dataseek_rate_limit_key`` metadata tag are ignored.

The callback handles the **success** path (reading rate-limit headers from
successful responses to drive real-mode pacing). The **failure** path (429
Retry-After) is handled by the invoke wrapper in :mod:`utils`, not here:
litellm's ``log_failure_event`` fires only *after* retry exhaustion (verified
against litellm 1.91.0), which is too late to gate the next retry attempt. The
wrapper catches 429-ish exceptions, calls ``apply_retry_after``, and re-raises
without touching ``max_retries`` semantics.
"""

from __future__ import annotations

import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from seek.components.search_graph.rate_limiter import RATE_LIMITER

logger = logging.getLogger(__name__)

# Metadata keys stashed on each litellm call (via .bind(metadata=...) in
# create_llm) so the callback can recover the limiter's scope key. Verified
# to survive into kwargs["litellm_params"]["metadata"] on litellm 1.91.0.
_KEY_FIELD = "dataseek_rate_limit_key"
_MODE_FIELD = "dataseek_rate_limit_mode"

_callbacks_registered = False


class _LLMRateLogger(CustomLogger):
    """Feeds response headers into :data:`RATE_LIMITER` for real-mode pacing.

    No-ops when dataseek metadata is absent (non-dataseek calls, test mocks).
    Only updates state for ``mode="real"`` endpoints — manual-mode pacing is
    entirely driven by the configured DSL, so header updates are irrelevant.
    """

    @staticmethod
    def _extract_key(kwargs: dict[str, Any]) -> str | None:
        meta = kwargs.get("litellm_params", {}).get("metadata") or {}
        if not isinstance(meta, dict):
            return None
        return meta.get(_KEY_FIELD)

    @staticmethod
    def _extract_mode(kwargs: dict[str, Any]) -> str | None:
        meta = kwargs.get("litellm_params", {}).get("metadata") or {}
        if not isinstance(meta, dict):
            return None
        return meta.get(_MODE_FIELD)

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,  # noqa: ARG002 (CustomLogger signature)
        end_time: float,  # noqa: ARG002 (CustomLogger signature)
    ) -> None:
        key = self._extract_key(kwargs)
        if key is None:
            return  # not a dataseek rate-limited call
        mode = self._extract_mode(kwargs)
        if mode != "real":
            return  # manual mode: pacing is config-driven, headers irrelevant
        headers = _extract_response_headers(response_obj)
        if headers:
            RATE_LIMITER.update_from_response(key, headers)

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,  # noqa: ARG002
        end_time: float,  # noqa: ARG002
    ) -> None:
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    # log_failure_event is intentionally NOT used for Retry-After. Verified:
    # litellm fires it once after retry exhaustion, too late to gate the next
    # attempt. The invoke wrapper in utils.py handles 429 Retry-After by
    # catching the exception, calling RATE_LIMITER.apply_retry_after, and
    # re-raising without altering max_retries semantics.


def _extract_response_headers(response_obj: Any) -> dict[str, Any]:
    """Pull the raw header dict off a litellm ModelResponse.

    litellm populates ``_response_headers`` (raw httpx headers) and
    ``_hidden_params["additional_headers"]`` (normalized) on every successful
    call including streams. langchain-litellm discards both, so the callback —
    which fires before langchain strips them — is the only place to read them.
    """
    headers: dict[str, Any] = {}
    raw = getattr(response_obj, "_response_headers", None)
    if isinstance(raw, dict):
        headers.update(raw)
    hidden = getattr(response_obj, "_hidden_params", None)
    if isinstance(hidden, dict):
        extra = hidden.get("additional_headers")
        if isinstance(extra, dict):
            headers.update(extra)
    return headers


def register_llm_rate_callbacks() -> None:
    """Register the rate-limit callback with litellm, once per process.

    Called lazily from :func:`create_llm` when an endpoint's
    ``rate_limit.mode != "off"``. Idempotent: a module-level flag guards
    against re-registration across ``create_llm`` calls. The callback no-ops
    on calls lacking dataseek metadata, so presence in
    ``litellm.success_callback`` is safe for non-dataseek traffic.
    """
    global _callbacks_registered
    if _callbacks_registered:
        return
    import litellm

    logger_instance = _LLMRateLogger()
    succ = litellm.success_callback or []
    if not any(isinstance(c, _LLMRateLogger) for c in succ):
        succ.append(logger_instance)
        litellm.success_callback = succ
    _callbacks_registered = True
    logger.debug("registered litellm rate-limit success callback")
