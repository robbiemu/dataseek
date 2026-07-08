import logging
import os
import re
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_litellm import ChatLiteLLM

from seek.common.config import get_active_seek_config
from seek.components.search_graph.rate_limiter import (
    RATE_LIMITER,
    RateLimitConfig,
    resolve_scope_key,
)
from seek.components.tool_manager.tools import get_tools_for_role


def _deep_merge_kwargs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dicts so nested keys (e.g. chat_template_kwargs)
    combine rather than replacing wholesale."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_kwargs(result[key], value)
        else:
            result[key] = value
    return result


logger = logging.getLogger(__name__)

# Diagnostics for list-shaped message.content from reasoning models. Off by
# default; set DATASEEK_INSPECT_CONTENT=1 to log the content shape of every
# assistant message returned by create_llm, with full detail when the content
# is list-shaped (the anomaly that breaks node JSON parsing and server history
# validation). Captures both what langchain hands dataseek and, via a litellm
# logging callback, the raw stream shape that produced it.
_INSPECT_CONTENT = os.getenv("DATASEEK_INSPECT_CONTENT", "").lower() in {"1", "true", "yes"}


def _summarize_content(content: Any) -> str:
    """Compact, non-spammy description of a message's content for diagnostics."""
    if isinstance(content, str):
        return f"str(len={len(content)}, head={content[:80]!r})"
    if isinstance(content, list):
        # Describe the shape without dumping the whole list (which can be huge
        # for thinking-block streams)
        parts = []
        for item in content[:5]:
            if isinstance(item, dict):
                parts.append(f"dict(keys={sorted(item.keys())})")
            else:
                parts.append(f"{type(item).__name__}({item!r:.40})")
        more = f", +{len(content) - 5} more" if len(content) > 5 else ""
        return f"list(len={len(content)}, [{', '.join(parts)}{more}])"
    return f"{type(content).__name__}({content!r:.80})"


def _inspect_message(role: str, message: BaseMessage) -> None:
    """Log the content shape of an assistant message (env-gated)."""
    if not _INSPECT_CONTENT or not isinstance(message, AIMessage):
        return
    is_list = isinstance(message.content, list)
    level = logging.WARNING if is_list else logging.DEBUG
    logger.log(
        level,
        "create_llm(%s) -> AIMessage.content is %s; additional_kwargs=%s; " "usage_metadata=%s",
        role,
        "LIST-SHAPED" if is_list else "str",
        sorted(message.additional_kwargs.keys()),
        getattr(message, "usage_metadata", None),
    )
    if is_list:
        logger.warning(
            "list content detail for role=%s: %s", role, _summarize_content(message.content)
        )
        if message.additional_kwargs.get("reasoning_content"):
            rc = message.additional_kwargs["reasoning_content"]
            logger.warning(
                "reasoning_content present (len=%d, head=%r)", len(str(rc)), str(rc)[:120]
            )


def _wrap_llm_for_inspection(role: str, llm: ChatLiteLLM) -> ChatLiteLLM:
    """Wrap a ChatLiteLLM so every invoke logs the returned message's content shape.

    Only active when DATASEEK_INSPECT_CONTENT is set; otherwise returns the llm
    unchanged with zero overhead.
    """
    if not _INSPECT_CONTENT:
        return llm

    original_invoke = llm.invoke

    def inspected_invoke(*args: Any, **kwargs: Any) -> Any:
        result = original_invoke(*args, **kwargs)
        if isinstance(result, BaseMessage):
            _inspect_message(role, result)
        return result

    # pydantic models reject attribute assignment; skip wrapping on
    # ChatLiteLLM (subclass-based instrumentation would go here if needed).
    return llm


class _RateLimitedChatLiteLLM(ChatLiteLLM):
    """ChatLiteLLM subclass that gates each call through the rate limiter.

    Overrides ``_generate``/``_agenerate`` (the LangChain-idiomatic seam) so
    ``.bind_tools()`` and all inherited behavior is preserved. Each call:
    1. ``RATE_LIMITER.acquire(key, config)`` — pre-call pacing (manual or real).
    2. Delegate to ``super()._generate`` (the normal path).
    3. On a 429-ish exception, call ``apply_retry_after`` to set a one-shot
       cooldown so the next ``max_retries`` attempt honors the server's
       ``Retry-After``, then re-raise without altering retry semantics.

    The limiter config + scope key are stashed on the instance at construction
    (private attributes, the pydantic-v2 pattern for non-validated fields).
    """

    _rate_limit_config: RateLimitConfig | None = None
    _rate_limit_key: str | None = None

    def _configure_rate_limit(self, config: RateLimitConfig, key: str) -> "_RateLimitedChatLiteLLM":
        """Stash the limiter config + resolved scope key on this instance."""
        object.__setattr__(self, "_rate_limit_config", config)
        object.__setattr__(self, "_rate_limit_key", key)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        cfg = self._rate_limit_config
        key = self._rate_limit_key
        if cfg is not None and key is not None and cfg.active:
            RATE_LIMITER.acquire(key, cfg)
        try:
            return super()._generate(messages, stop, run_manager, stream, **kwargs)
        except Exception as exc:
            _handle_rate_limit_exception(exc, key)
            raise

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        cfg = self._rate_limit_config
        key = self._rate_limit_key
        if cfg is not None and key is not None and cfg.active:
            RATE_LIMITER.acquire(key, cfg)
        try:
            return await super()._agenerate(messages, stop, run_manager, stream, **kwargs)
        except Exception as exc:
            _handle_rate_limit_exception(exc, key)
            raise


def _handle_rate_limit_exception(exc: Exception, key: str | None) -> None:
    """On a 429-ish exception, set a one-shot cooldown from Retry-After.

    litellm populates ``exception.headers`` on RateLimitError (verified in
    1.91.0). The cooldown gates the next acquire so litellm's own
    ``max_retries`` retry doesn't immediately re-trip the rate limit. Re-raises
    nothing — the caller re-raises so retry semantics are untouched.
    """
    if key is None:
        return
    headers = getattr(exc, "headers", None)
    if not isinstance(headers, dict):
        # Some exception types nest the response; try a couple of common attrs.
        resp = getattr(exc, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", None)
    if isinstance(headers, dict):
        RATE_LIMITER.apply_retry_after(key, headers)


def create_llm(role: str) -> ChatLiteLLM:
    """Creates a configured ChatLiteLLM instance for a given agent role."""
    # Load the seek config
    seek_config = get_active_seek_config()

    # Get model defaults from seek config
    model_defaults = seek_config.get("model_defaults", {})
    default_model = model_defaults.get("model", "openai/gpt-5.4-mini")
    default_temperature = model_defaults.get("temperature", 0.1)
    # max_tokens is opt-in: omitted by default so the server/model decides the
    # output budget. A hardcoded floor truncates reasoning models mid-thought
    # (the answer never gets emitted because the budget runs out during the
    # reasoning phase). Set model_defaults.max_tokens or a per-node max_tokens
    # when you need an explicit cap.
    default_max_tokens = model_defaults.get("max_tokens")
    default_top_p = model_defaults.get("top_p")
    # api_base lets a role target a custom OpenAI-compatible server (e.g. a
    # local sglang/Spark box) instead of the provider's default endpoint.
    default_api_base = model_defaults.get("api_base")
    # max_retries tunes litellm/langchain transport-layer retries on transient
    # transport/5xx/429 errors. Left unset → ChatLiteLLM's own default applies.
    default_max_retries = model_defaults.get("max_retries")
    # Streaming defaults on. Long local-server generations with streaming=False
    # hold an idle socket until the full response is ready; an idle-read timeout
    # (or proxy/OS keepalive) then resets the connection mid-generation,
    # surfacing as InternalServerError/Connection error. Streaming keeps bytes
    # flowing during generation so the connection is never treated as idle.
    default_streaming = model_defaults.get("streaming", True)
    # model_kwargs flows verbatim into the litellm completion payload. Used for
    # provider-specific params that don't have a direct field on ChatLiteLLM,
    # e.g. chat_template_kwargs: {enable_thinking: false} for reasoning models
    # (Qwen3) that otherwise emit text in reasoning_content and leave content
    # empty. A per-node model_kwargs deep-merges over model_defaults.
    default_model_kwargs = model_defaults.get("model_kwargs", {})
    # rate_limit: optional per-endpoint LLM rate limiting (manual DSL or real
    # HTTP-header-driven). Node-level wins over model_defaults; an absent block
    # means mode="off" (no limiting, the current behavior). See
    # seek/components/search_graph/rate_limiter.py for the DSL grammar.
    default_rate_limit = model_defaults.get("rate_limit")

    # Try to find node-specific config in mission plan
    node_config = None
    mission_plan = seek_config.get("mission_plan")
    if mission_plan and isinstance(mission_plan, dict):
        nodes = mission_plan.get("nodes", [])
        if isinstance(nodes, list):
            # Find the node config that matches the role
            for node in nodes:
                if isinstance(node, dict) and node.get("name") == role:
                    node_config = node
                    break

    # Use node-specific config if available, otherwise fall back to defaults
    if node_config:
        model = node_config.get("model", default_model)
        temperature = node_config.get("temperature", default_temperature)
        max_tokens = node_config.get("max_tokens", default_max_tokens)
        top_p = node_config.get("top_p", default_top_p)
        api_base = node_config.get("api_base", default_api_base)
        max_retries = node_config.get("max_retries", default_max_retries)
        streaming = node_config.get("streaming", default_streaming)
        rate_limit_raw = node_config.get("rate_limit", default_rate_limit)
        # Node-level model_kwargs deep-merges over model_defaults' model_kwargs
        # so a node override for one nested key (e.g. one chat_template_kwargs
        # setting) doesn't discard sibling defaults.
        node_model_kwargs = node_config.get("model_kwargs", {})
        model_kwargs = _deep_merge_kwargs(default_model_kwargs, node_model_kwargs)
    else:
        # Fallback to default values from seek config
        model = default_model
        temperature = default_temperature
        max_tokens = default_max_tokens
        top_p = default_top_p
        api_base = default_api_base
        max_retries = default_max_retries
        streaming = default_streaming
        rate_limit_raw = default_rate_limit
        model_kwargs = dict(default_model_kwargs)

    # Only pass top_p when explicitly configured. Some providers (e.g. greedy
    # sampling on certain models) reject the parameter entirely, and omitting it
    # preserves the stock behavior for missions that do not set it.
    #
    # top_p is routed through model_kwargs rather than as a direct field:
    # ChatLiteLLM stores a top_p field but does not forward it into the litellm
    # completion payload, whereas model_kwargs is always passed through. This
    # matters for greedy models (e.g. Leanstral) that 400 without top_p=1.
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    # Only pass max_tokens when explicitly configured. Left unset, the
    # server/model decides the output budget — important for reasoning models,
    # which can spend a large budget in the reasoning phase before emitting the
    # answer; a hardcoded cap truncates them mid-thought.
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    # top_p is routed through model_kwargs: ChatLiteLLM stores a top_p field but
    # does not forward it into the litellm completion payload, whereas
    # model_kwargs is always passed through. Set it into the (possibly
    # config-provided) model_kwargs dict rather than overwriting the dict, so
    # provider-specific keys like chat_template_kwargs survive alongside it.
    if top_p is not None:
        model_kwargs = {**model_kwargs, "top_p": top_p}
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    # Route to a custom OpenAI-compatible endpoint when configured. This is what
    # lets roles target local servers (e.g. Spark/sglang) rather than the cloud.
    if api_base:
        kwargs["api_base"] = api_base
    # Only override the langchain retry count when explicitly configured; left
    # unset, ChatLiteLLM applies its own default. This is the primary mechanism
    # for absorbing transient transport/5xx/429 errors (e.g. a briefly-full
    # local server queue) at the transport layer where reconnects are clean.
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    # Streaming is always passed (default True) so that long generations
    # against local servers keep the connection alive, as described above.
    kwargs["streaming"] = streaming
    # Resolve the rate-limit config (parsed per call, not cached by role — the
    # parser cost is negligible and caching by role breaks if two same-role
    # nodes resolve different models/api_bases/overrides).
    rl_config = RateLimitConfig.from_dict(rate_limit_raw)
    if rl_config.active:
        # Lazy-register the litellm success callback (idempotent; no-ops on
        # calls without dataseek metadata, so safe for non-dataseek traffic).
        from seek.components.search_graph.llm_rate_callback import (
            register_llm_rate_callbacks,
        )

        register_llm_rate_callbacks()
        # Resolve the scope key from the resolved model identity (provider +
        # api_base + credential + model). Non-secret credential identity via
        # env-var-name table; see resolve_scope_key in rate_limiter.py.
        rl_key = resolve_scope_key(
            model=model, api_base=api_base, api_key=None, scope=rl_config.scope
        )
        # Tag each call with the key + mode so the litellm success callback can
        # recover them (verified to survive into kwargs["litellm_params"]
        # ["metadata"] on litellm 1.91.0). model_kwargs flows through
        # ChatLiteLLM's {**params, **kwargs} spread into completion(), so
        # stashing metadata here tags every call from this LLM instance.
        tagged_kwargs = _deep_merge_kwargs(
            kwargs.get("model_kwargs", {}),
            {
                "metadata": {
                    "dataseek_rate_limit_key": rl_key,
                    "dataseek_rate_limit_mode": rl_config.mode,
                }
            },
        )
        kwargs["model_kwargs"] = tagged_kwargs
        llm: ChatLiteLLM = _RateLimitedChatLiteLLM(**kwargs)._configure_rate_limit(
            rl_config, rl_key
        )
    else:
        llm = ChatLiteLLM(**kwargs)
    return _wrap_llm_for_inspection(role, llm)


def create_agent_runnable(
    llm: ChatLiteLLM,
    system_prompt: str,
    role: str,
    mission_config: dict[str, Any] | None = None,
    tools: list[Any] | None = None,
) -> Runnable:
    """Factory to create a new agent node's runnable.

    When ``tools`` is provided (e.g. the ToolManager-prepared, configured
    instances from the graph), those exact instances are bound to the model.
    Otherwise, falls back to ``get_tools_for_role`` which freshly instantiates
    plugins without mission config or setup() — use the explicit path when the
    model and ToolNode must share the same instances.
    """
    # Load the seek config to get the use_robots setting
    seek_config = get_active_seek_config()
    seek_config.get("use_robots", True)

    bound_tools = tools if tools is not None else get_tools_for_role(role, mission_config)
    # Escape curly braces to avoid ChatPromptTemplate treating literals as variables
    safe_system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", safe_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    if bound_tools:
        # Force a provider-compatible tool_choice
        return prompt | llm.bind_tools(bound_tools, tool_choice="auto")
    return prompt | llm


def normalize_url(url: str) -> str:
    """Simple URL normalization to avoid variants."""
    try:
        # Convert to lowercase and strip trailing slash
        normalized = url.lower().rstrip("/")
        # Basic protocol normalization
        if normalized.startswith("http://"):
            normalized = normalized.replace("http://", "https://", 1)
        return normalized
    except Exception:
        return url


def get_characteristic_context(task: dict, mission_config: dict) -> str | None:
    """
    Finds the definitional context for a characteristic from the mission config.
    """
    if not task or not mission_config:
        return None

    characteristic_name = task.get("characteristic")
    if not characteristic_name:
        return None

    # Accept either a full mission plan (with 'missions') or a single-mission dict (with 'goals')
    if "missions" in mission_config and isinstance(mission_config["missions"], list):
        search_space = mission_config["missions"]
    else:
        search_space = [mission_config]

    for mission in search_space:
        for goal in mission.get("goals", []):
            if goal.get("characteristic") == characteristic_name:
                return goal.get("context")

    return None  # Return None if no matching characteristic is found


def get_default_strategy_block(characteristic: str) -> str:
    """
    Get a generic, fallback strategy block for a given characteristic.

    This function is used when a specific strategy_block is not provided
    in the mission's prompts.yaml file. It provides a basic, unopinionated
    instruction to the agent.
    """
    # The new implementation is a simple, formatted string that works for any characteristic.
    # This removes the hardcoded, mission-specific logic.
    return (
        f"**Strategic Focus for {characteristic}:**\n"
        f"Your goal is to find source documents whose writing style and structure "
        f"make them exceptionally good sources for extracting factual claims that "
        f"exemplify the principle of '{characteristic}'. Look for documents that are "
        f"naturally rich in sentences that a downstream agent could easily turn into "
        f"high-quality examples with this desired characteristic."
    )


def strip_reasoning_block(content: str, tags: list[str] | None = None) -> str:
    """
    Removes a reasoning block from the beginning of a string if present.

    This function can strip blocks denoted by various tags like <think>,
    <scratchpad>, <reasoning>, etc.

    Args:
        content: The input string.
        tags: A list of tag names to look for. Defaults to a standard list.

    Returns:
        The string with the initial reasoning block removed.
    """
    if tags is None:
        tags = [
            "think",
            "thinking",
            "thought",
            "scratchpad",
            "reasoning",
            "plan",
            "reflection",
            "rationale",
        ]

    # Create a regex 'or' condition by joining the tags with '|'
    # This will match any of the words in the list.
    tag_pattern = "|".join(tags)

    # The main pattern now uses the tag_pattern.
    # - <({tag_pattern})>: Captures the specific tag found (e.g., "scratchpad").
    # - <\/\1>: The backreference \1 ensures the closing tag matches the opening one.
    pattern = rf"^\s*<({tag_pattern})>(.*?)<\/\1>\s*"

    return re.sub(pattern, "", content, count=1, flags=re.DOTALL | re.IGNORECASE)
