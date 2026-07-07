import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_litellm import ChatLiteLLM

from seek.common.config import get_active_seek_config
from seek.components.tool_manager.tools import get_tools_for_role


def create_llm(role: str) -> ChatLiteLLM:
    """Creates a configured ChatLiteLLM instance for a given agent role."""
    # Load the seek config
    seek_config = get_active_seek_config()

    # Get model defaults from seek config
    model_defaults = seek_config.get("model_defaults", {})
    default_model = model_defaults.get("model", "openai/gpt-5.4-mini")
    default_temperature = model_defaults.get("temperature", 0.1)
    default_max_tokens = model_defaults.get("max_tokens", 2000)
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
    else:
        # Fallback to default values from seek config
        model = default_model
        temperature = default_temperature
        max_tokens = default_max_tokens
        top_p = default_top_p
        api_base = default_api_base
        max_retries = default_max_retries
        streaming = default_streaming

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
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        kwargs["model_kwargs"] = {"top_p": top_p}
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
    return ChatLiteLLM(**kwargs)


def create_agent_runnable(
    llm: ChatLiteLLM,
    system_prompt: str,
    role: str,
    mission_config: dict[str, Any] | None = None,
) -> Runnable:
    """Factory to create a new agent node's runnable."""
    # Load the seek config to get the use_robots setting
    seek_config = get_active_seek_config()
    seek_config.get("use_robots", True)

    tools = get_tools_for_role(role, mission_config)
    # Escape curly braces to avoid ChatPromptTemplate treating literals as variables
    safe_system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", safe_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    if tools:
        # Force a provider-compatible tool_choice
        return prompt | llm.bind_tools(tools, tool_choice="auto")
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
