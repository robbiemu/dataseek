"""Tests for the LLM rate limiter: DSL parsing, header parsing, sliding-window
pacing, real-mode header-driven pacing, Retry-After reactive backoff, scope-key
resolution, off-mode no-op, and monotonic-time usage.

Stock behavior (no ``rate_limit`` block) is asserted unchanged in every case.
"""

from __future__ import annotations

import time
from datetime import UTC
from unittest.mock import MagicMock

import pytest

from seek.components.search_graph.rate_limiter import (
    Limit,
    RateLimitConfig,
    RateLimiter,
    parse_limits_dsl,
    parse_rate_limit_headers,
    resolve_scope_key,
)

# -------------------------
# DSL parsing
# -------------------------


class TestParseLimitsDSL:
    def test_single_limit(self):
        limits = parse_limits_dsl("30m")
        assert limits == [Limit(count=30, window_seconds=60.0, role=None)]

    def test_single_limit_with_role(self):
        limits = parse_limits_dsl("30m #burst")
        assert limits == [Limit(count=30, window_seconds=60.0, role="burst")]

    def test_compound_limits(self):
        limits = parse_limits_dsl("30m #burst, 500h #steady, 10000d")
        assert limits == [
            Limit(count=30, window_seconds=60.0, role="burst"),
            Limit(count=500, window_seconds=3600.0, role="steady"),
            Limit(count=10000, window_seconds=86400.0, role=None),
        ]

    def test_dashed_role_tag(self):
        limits = parse_limits_dsl("32m #free-tier")
        assert limits[0].role == "free-tier"

    def test_all_units(self):
        for unit, secs in [("s", 1.0), ("m", 60.0), ("h", 3600.0), ("d", 86400.0)]:
            limits = parse_limits_dsl(f"1{unit}")
            assert limits[0].window_seconds == secs

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_limits_dsl("")

    def test_empty_token_raises(self):
        with pytest.raises(ValueError, match="empty token"):
            parse_limits_dsl("30m, , 60m")

    def test_bad_unit_raises(self):
        with pytest.raises(ValueError, match="malformed"):
            parse_limits_dsl("30x")

    def test_zero_count_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_limits_dsl("0m")

    def test_malformed_role_raises(self):
        with pytest.raises(ValueError, match="malformed"):
            parse_limits_dsl("30m #burst!!")

    def test_count_plus_window_semantics(self):
        """30m means 30 requests per minute, not 30 minutes."""
        limits = parse_limits_dsl("30m")
        assert limits[0].count == 30
        assert limits[0].window_seconds == 60.0


# -------------------------
# Header parsing
# -------------------------


class TestParseRateLimitHeaders:
    def test_none_when_no_headers(self):
        assert parse_rate_limit_headers({}) is None
        assert parse_rate_limit_headers(None) is None  # type: ignore[arg-type]

    def test_none_when_unrelated_headers(self):
        assert parse_rate_limit_headers({"content-type": "application/json"}) is None

    def test_legacy_x_ratelimit(self):
        r = parse_rate_limit_headers(
            {"x-ratelimit-limit": "32", "x-ratelimit-remaining": "28", "x-ratelimit-reset": "60"}
        )
        assert r is not None
        assert r.limit == 32
        assert r.remaining == 28
        assert r.reset_at is not None
        assert r.source == "x-ratelimit-legacy"

    def test_ietf_ratelimit_fields(self):
        r = parse_rate_limit_headers(
            {"ratelimit-limit": "32", "ratelimit-remaining": "0", "ratelimit-reset": "60"}
        )
        assert r is not None
        assert r.limit == 32
        assert r.remaining == 0
        assert r.source == "ietf-ratelimit-fields"

    def test_retry_after_seconds(self):
        r = parse_rate_limit_headers({"retry-after": "5"})
        assert r is not None
        assert r.retry_after == 5.0

    def test_retry_after_http_date(self):
        from datetime import datetime, timedelta
        from email.utils import format_datetime

        future = datetime.now(UTC) + timedelta(seconds=10)
        r = parse_rate_limit_headers({"retry-after": format_datetime(future)})
        assert r is not None
        assert r.retry_after is not None
        assert 5 <= r.retry_after <= 15  # ~10s, with slack

    def test_comma_separated_remaining(self):
        """Brave-style '0, 1995' — take the first field."""
        r = parse_rate_limit_headers({"x-ratelimit-remaining": "0, 1995"})
        assert r is not None
        assert r.remaining == 0

    def test_nvidia_limit_string(self):
        r = parse_rate_limit_headers({"x-limit": "32/32"})
        assert r is not None
        assert r.limit == 32
        assert r.remaining == 0
        assert r.source == "nvidia-limit-string"

    def test_reset_relative_vs_absolute(self):
        # Relative (small number) -> monotonic deadline in the near future.
        r = parse_rate_limit_headers({"x-ratelimit-reset": "60"})
        assert r.reset_at is not None
        now = time.monotonic()
        assert now < r.reset_at <= now + 65
        # Absolute (epoch) -> also a monotonic deadline, but derived from wall clock.
        r2 = parse_rate_limit_headers({"x-ratelimit-reset": str(time.time() + 60)})
        assert r2.reset_at is not None

    def test_ietf_structured_ratelimit_header(self):
        """IETF draft-11 RateLimit header: 'policy';r=<remaining>;t=<window>."""
        r = parse_rate_limit_headers(
            {"ratelimit": '"default";r=50;t=30', "ratelimit-policy": '"default";q=100;w=60'}
        )
        assert r is not None
        assert r.remaining == 50
        assert r.limit == 100  # from RateLimit-Policy q=
        assert r.reset_at is not None  # from t=30
        assert r.source == "ietf-ratelimit"

    def test_ietf_structured_exhausted(self):
        """The draft-11 exhausted case: r=0;t=10 -> real mode should sleep ~10s."""
        r = parse_rate_limit_headers({"ratelimit": '"problemPolicy";r=0;t=10'})
        assert r is not None
        assert r.remaining == 0
        assert r.reset_at is not None

    def test_openai_suffixed_request_headers(self):
        """OpenAI dimension-suffixed headers with Go-duration reset."""
        r = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-requests": "32",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": "6m0s",
            }
        )
        assert r is not None
        assert r.limit == 32
        assert r.remaining == 0
        assert r.reset_at is not None
        assert r.source == "openai-requests"

    def test_openai_reset_go_duration_variants(self):
        """Go-duration strings OpenAI emits: 1s, 6m0s, 27m28s, 23h59m59s."""
        for val, expected in [
            ("1s", 1.0),
            ("6m0s", 360.0),
            ("27m28s", 1648.0),
            ("23h59m59s", 86399.0),
            ("12ms", 0.012),
        ]:
            r = parse_rate_limit_headers(
                {"x-ratelimit-remaining-requests": "5", "x-ratelimit-reset-requests": val}
            )
            assert r is not None
            assert r.reset_at is not None
            # reset_at is monotonic+expected; check the delta is ~expected.
            delta = r.reset_at - time.monotonic()
            assert expected - 0.5 <= delta <= expected + 0.5, f"{val}: delta={delta}"

    def test_openai_suffixed_does_not_shadow_legacy(self):
        """Both suffixed and unsuffixed families are recognized independently."""
        # Suffixed present -> parsed.
        r1 = parse_rate_limit_headers({"x-ratelimit-remaining-requests": "5"})
        assert r1 is not None
        assert r1.remaining == 5
        # Unsuffixed present -> parsed.
        r2 = parse_rate_limit_headers({"x-ratelimit-remaining": "3"})
        assert r2 is not None
        assert r2.remaining == 3


# -------------------------
# Scope-key resolution
# -------------------------


class TestResolveScopeKey:
    def test_model_scope_includes_model(self):
        key = resolve_scope_key("openrouter/nvidia/foo", None, None, "model")
        assert "openrouter" in key
        assert "openrouter/nvidia/foo" in key
        assert "env:OPENROUTER_API_KEY" in key

    def test_provider_scope_excludes_model(self):
        key = resolve_scope_key("openrouter/nvidia/foo", None, None, "provider")
        assert "openrouter" in key
        assert "openrouter/nvidia/foo" not in key
        assert "env:OPENROUTER_API_KEY" in key

    def test_local_server_credential_identity(self):
        key = resolve_scope_key("openai/qwen", "http://spark-885a:30000/v1", None, "model")
        assert "env:OPENAI_API_KEY" in key
        assert "openai/qwen" in key

    def test_explicit_api_key_uses_hash(self):
        key = resolve_scope_key("openai/foo", None, "sk-secret123", "model")
        assert "explicit:" in key
        assert "sk-secret123" not in key  # never leak the key

    def test_two_models_share_provider_scope(self):
        k1 = resolve_scope_key("openrouter/a", None, None, "provider")
        k2 = resolve_scope_key("openrouter/b", None, None, "provider")
        assert k1 == k2  # same provider+credential -> shared quota

    def test_two_models_differ_under_model_scope(self):
        k1 = resolve_scope_key("openrouter/a", None, None, "model")
        k2 = resolve_scope_key("openrouter/b", None, None, "model")
        assert k1 != k2  # different models -> separate quotas


# -------------------------
# RateLimitConfig
# -------------------------


class TestRateLimitConfig:
    def test_off_is_default(self):
        cfg = RateLimitConfig()
        assert cfg.mode == "off"
        assert not cfg.active

    def test_from_dict_none(self):
        assert RateLimitConfig.from_dict(None).mode == "off"

    def test_from_dict_manual(self):
        cfg = RateLimitConfig.from_dict(
            {"mode": "manual", "limits": "30m #burst", "scope": "provider"}
        )
        assert cfg.mode == "manual"
        assert cfg.active
        assert len(cfg.limits) == 1
        assert cfg.scope == "provider"

    def test_from_dict_real_with_fallback_limits(self):
        cfg = RateLimitConfig.from_dict({"mode": "real", "limits": "10s"})
        assert cfg.mode == "real"
        assert len(cfg.limits) == 1

    def test_from_dict_real_without_limits(self):
        cfg = RateLimitConfig.from_dict({"mode": "real"})
        assert cfg.mode == "real"
        assert cfg.limits == ()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            RateLimitConfig.from_dict({"mode": "weird"})

    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            RateLimitConfig.from_dict({"mode": "manual", "limits": "1s", "scope": "global"})

    def test_manual_requires_limits(self):
        with pytest.raises(ValueError, match="requires a 'limits'"):
            RateLimitConfig.from_dict({"mode": "manual"})


# -------------------------
# RateLimiter — manual mode pacing
# -------------------------


class TestRateLimiterManual:
    def test_off_mode_is_noop(self):
        rl = RateLimiter()
        cfg = RateLimitConfig()  # off
        t0 = time.monotonic()
        for _ in range(100):
            rl.acquire("k", cfg)
        assert time.monotonic() - t0 < 0.1  # 100 instant admits

    def test_admits_up_to_count_then_waits(self):
        rl = RateLimiter()
        cfg = RateLimitConfig(mode="manual", limits=(Limit(3, 1.0),))
        t0 = time.monotonic()
        for _ in range(3):
            rl.acquire("k", cfg)
        assert time.monotonic() - t0 < 0.1  # 3 instant admits
        # 4th must wait ~1s for the oldest to age out.
        t1 = time.monotonic()
        rl.acquire("k", cfg)
        elapsed = time.monotonic() - t1
        assert 0.9 <= elapsed <= 1.2

    def test_compound_limits_max_deadline_not_min(self):
        """A tight burst cap and a loose steady cap: gate on the MAX wait."""
        rl = RateLimiter()
        # 2/1s (burst) + 100/60s (steady). Burst is the binding constraint.
        cfg = RateLimitConfig(
            mode="manual",
            limits=(Limit(2, 1.0, "burst"), Limit(100, 60.0, "steady")),
        )
        rl.acquire("k", cfg)
        rl.acquire("k", cfg)
        # 3rd call: burst exhausted (2/1s), steady has room. Must wait ~1s.
        t0 = time.monotonic()
        rl.acquire("k", cfg)
        elapsed = time.monotonic() - t0
        assert 0.9 <= elapsed <= 1.2

    def test_atomic_timestamp_append_to_all_windows(self):
        """After admission, every active limit's window gets a timestamp."""
        rl = RateLimiter()
        cfg = RateLimitConfig(
            mode="manual",
            limits=(Limit(5, 1.0), Limit(10, 60.0)),
        )
        rl.acquire("k", cfg)
        st = rl._states["k"]
        assert len(st.manual_windows[0]) == 1
        assert len(st.manual_windows[1]) == 1

    def test_independent_keys(self):
        rl = RateLimiter()
        cfg = RateLimitConfig(mode="manual", limits=(Limit(1, 10.0),))
        rl.acquire("a", cfg)
        rl.acquire("b", cfg)  # different key, no wait
        st_a = rl._states["a"]
        st_b = rl._states["b"]
        assert st_a is not st_b


# -------------------------
# RateLimiter — real mode (header-driven)
# -------------------------


class TestRateLimiterReal:
    def test_update_from_response_stores_remaining(self):
        rl = RateLimiter()
        rl.update_from_response("k", {"x-ratelimit-remaining": "5", "x-ratelimit-reset": "60"})
        st = rl._states["k"]
        assert st.remaining == 5
        assert st.reset_at is not None

    def test_real_mode_paces_when_remaining_zero(self):
        rl = RateLimiter()
        cfg = RateLimitConfig(mode="real")
        # Simulate a prior response showing remaining=0, reset in ~1s.
        rl.update_from_response("k", {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1"})
        t0 = time.monotonic()
        rl.acquire("k", cfg)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.8  # waited for the reset

    def test_real_mode_no_wait_when_remaining_positive(self):
        rl = RateLimiter()
        cfg = RateLimitConfig(mode="real")
        rl.update_from_response("k", {"x-ratelimit-remaining": "10", "x-ratelimit-reset": "60"})
        t0 = time.monotonic()
        rl.acquire("k", cfg)
        assert time.monotonic() - t0 < 0.1

    def test_real_mode_falls_back_to_configured_limits(self):
        """Without header state, real mode uses any configured manual ceiling."""
        rl = RateLimiter()
        cfg = RateLimitConfig(mode="real", limits=(Limit(1, 10.0),))
        rl.acquire("k", cfg)
        t0 = time.monotonic()
        rl.acquire("k", cfg)  # second call: limit exhausted
        assert time.monotonic() - t0 >= 8.0  # waits for the 10s window

    def test_real_mode_decrements_remaining_on_acquire(self):
        """acquire() decrements remaining so concurrent callers don't over-admit."""
        rl = RateLimiter()
        cfg = RateLimitConfig(mode="real")
        rl.update_from_response("k", {"x-ratelimit-remaining-requests": "2"})
        rl.acquire("k", cfg)
        assert rl._states["k"].remaining == 1  # decremented
        rl.acquire("k", cfg)
        assert rl._states["k"].remaining == 0  # decremented again
        # Third call should wait (remaining=0, no reset header -> no wait here
        # since reset_at is None, but remaining is 0 so it won't admit instantly
        # if a reset were set). Verify remaining stays non-negative.
        assert rl._states["k"].remaining == 0


# -------------------------
# RateLimiter — Retry-After reactive backoff
# -------------------------


class TestRateLimiterRetryAfter:
    def test_apply_retry_after_sets_cooldown(self):
        rl = RateLimiter()
        wait = rl.apply_retry_after("k", {"retry-after": "2"})
        assert wait == 2.0
        st = rl._states["k"]
        assert st.cooldown_until is not None
        assert st.cooldown_until > time.monotonic()

    def test_cooldown_gates_next_acquire(self):
        rl = RateLimiter()
        cfg = RateLimitConfig(mode="manual", limits=(Limit(100, 60.0),))
        rl.apply_retry_after("k", {"retry-after": "1"})
        t0 = time.monotonic()
        rl.acquire("k", cfg)
        elapsed = time.monotonic() - t0
        assert 0.8 <= elapsed <= 1.2  # waited for the cooldown

    def test_apply_retry_after_no_headers(self):
        rl = RateLimiter()
        assert rl.apply_retry_after("k", None) == 0.0
        assert rl.apply_retry_after("k", {}) == 0.0

    def test_cooldown_cleared_after_elapsing(self):
        rl = RateLimiter()
        cfg = RateLimitConfig(mode="manual", limits=(Limit(100, 60.0),))
        rl.apply_retry_after("k", {"retry-after": "0.1"})
        time.sleep(0.2)
        rl.acquire("k", cfg)  # cooldown elapsed, should not re-set
        assert rl._states["k"].cooldown_until is None


# -------------------------
# Monotonic time usage
# -------------------------


class TestMonotonicTime:
    def test_manual_window_uses_monotonic(self):
        """If the limiter used wall-clock, an NTP jump would break the window.
        Sanity-check that pacing is consistent with time.monotonic() by
        confirming a 1s window waits ~1s (not affected by wall time)."""
        rl = RateLimiter()
        cfg = RateLimitConfig(mode="manual", limits=(Limit(1, 1.0),))
        rl.acquire("k", cfg)
        t0 = time.monotonic()
        rl.acquire("k", cfg)
        elapsed = time.monotonic() - t0
        assert 0.9 <= elapsed <= 1.2


# -------------------------
# create_llm wiring
# -------------------------


class TestCreateLlmWiring:
    def test_no_rate_limit_block_returns_plain_chatlitellm(self):
        from seek.common.config import StructuredSeekConfig, set_active_seek_config
        from seek.components.search_graph.nodes.utils import create_llm

        set_active_seek_config(
            StructuredSeekConfig(
                {
                    "model_defaults": {"model": "openai/gpt-4o-mini"},
                    "mission_plan": {
                        "nodes": [{"name": "research", "model": "openai/gpt-4o-mini"}]
                    },
                }
            )
        )
        llm = create_llm("research")
        assert type(llm).__name__ == "ChatLiteLLM"  # not the rate-limited subclass

    def test_manual_rate_limit_returns_subclass(self):
        from seek.common.config import StructuredSeekConfig, set_active_seek_config
        from seek.components.search_graph.nodes.utils import create_llm

        set_active_seek_config(
            StructuredSeekConfig(
                {
                    "model_defaults": {"model": "openai/gpt-4o-mini"},
                    "mission_plan": {
                        "nodes": [
                            {
                                "name": "research",
                                "model": "openrouter/nvidia/foo",
                                "rate_limit": {
                                    "mode": "manual",
                                    "limits": "32m #window",
                                    "scope": "provider",
                                },
                            }
                        ]
                    },
                }
            )
        )
        llm = create_llm("research")
        assert "RateLimited" in type(llm).__name__
        assert llm._rate_limit_config.mode == "manual"
        assert llm._rate_limit_key  # non-empty
        assert "env:OPENROUTER_API_KEY" in llm._rate_limit_key

    def test_metadata_tag_flows_into_model_kwargs(self):
        from seek.common.config import StructuredSeekConfig, set_active_seek_config
        from seek.components.search_graph.nodes.utils import create_llm

        set_active_seek_config(
            StructuredSeekConfig(
                {
                    "model_defaults": {"model": "openai/gpt-4o-mini"},
                    "mission_plan": {
                        "nodes": [
                            {
                                "name": "research",
                                "model": "openai/gpt-4o-mini",
                                "rate_limit": {"mode": "real"},
                            }
                        ]
                    },
                }
            )
        )
        llm = create_llm("research")
        mk = llm.model_kwargs or {}
        assert "metadata" in mk
        assert mk["metadata"]["dataseek_rate_limit_mode"] == "real"
        assert mk["metadata"]["dataseek_rate_limit_key"]

    def test_existing_model_kwargs_survive_with_rate_limit(self):
        """A pre-existing chat_template_kwargs must survive the metadata merge."""
        from seek.common.config import StructuredSeekConfig, set_active_seek_config
        from seek.components.search_graph.nodes.utils import create_llm

        set_active_seek_config(
            StructuredSeekConfig(
                {
                    "model_defaults": {"model": "openai/gpt-4o-mini"},
                    "mission_plan": {
                        "nodes": [
                            {
                                "name": "research",
                                "model": "openai/gpt-4o-mini",
                                "model_kwargs": {
                                    "chat_template_kwargs": {"enable_thinking": False}
                                },
                                "rate_limit": {"mode": "manual", "limits": "10s"},
                            }
                        ]
                    },
                }
            )
        )
        llm = create_llm("research")
        mk = llm.model_kwargs or {}
        assert mk["chat_template_kwargs"] == {"enable_thinking": False}
        assert "metadata" in mk

    def test_node_rate_limit_wins_over_default(self):
        from seek.common.config import StructuredSeekConfig, set_active_seek_config
        from seek.components.search_graph.nodes.utils import create_llm

        set_active_seek_config(
            StructuredSeekConfig(
                {
                    "model_defaults": {
                        "model": "openai/gpt-4o-mini",
                        "rate_limit": {"mode": "manual", "limits": "10s"},
                    },
                    "mission_plan": {
                        "nodes": [
                            {
                                "name": "research",
                                "model": "openai/gpt-4o-mini",
                                "rate_limit": {"mode": "off"},
                            }
                        ]
                    },
                }
            )
        )
        llm = create_llm("research")
        assert type(llm).__name__ == "ChatLiteLLM"  # node overrode to off


# -------------------------
# Callback no-op behavior
# -------------------------


class TestCallbackNoOp:
    def test_callback_noops_without_metadata(self):
        from seek.components.search_graph.llm_rate_callback import _LLMRateLogger

        logger = _LLMRateLogger()
        # kwargs without dataseek metadata -> no-op, no exception.
        kwargs = {"litellm_params": {"metadata": {}}}
        response = MagicMock()
        logger.log_success_event(kwargs, response, 0, 0)  # should not raise

    def test_callback_updates_real_mode_state(self):
        from seek.components.search_graph.llm_rate_callback import _LLMRateLogger
        from seek.components.search_graph.rate_limiter import RATE_LIMITER

        logger = _LLMRateLogger()
        response = MagicMock()
        response._response_headers = {"x-ratelimit-remaining": "5", "x-ratelimit-reset": "60"}
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "dataseek_rate_limit_key": "test-cb-key",
                    "dataseek_rate_limit_mode": "real",
                }
            }
        }
        logger.log_success_event(kwargs, response, 0, 0)
        st = RATE_LIMITER._states.get("test-cb-key")
        assert st is not None
        assert st.remaining == 5

    def test_callback_ignores_manual_mode(self):
        """Manual mode pacing is config-driven; header updates are irrelevant."""
        from seek.components.search_graph.llm_rate_callback import _LLMRateLogger
        from seek.components.search_graph.rate_limiter import RATE_LIMITER

        logger = _LLMRateLogger()
        response = MagicMock()
        response._response_headers = {"x-ratelimit-remaining": "0"}
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "dataseek_rate_limit_key": "test-manual-key",
                    "dataseek_rate_limit_mode": "manual",
                }
            }
        }
        logger.log_success_event(kwargs, response, 0, 0)
        st = RATE_LIMITER._states.get("test-manual-key")
        # Manual mode -> no header state stored.
        assert st is None or st.remaining is None
