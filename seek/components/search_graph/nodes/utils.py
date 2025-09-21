import re

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
    default_model = model_defaults.get("model", "openai/gpt-5-mini")
    default_temperature = model_defaults.get("temperature", 0.1)
    default_max_tokens = model_defaults.get("max_tokens", 2000)

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
    else:
        # Fallback to default values from seek config
        model = default_model
        temperature = default_temperature
        max_tokens = default_max_tokens

    return ChatLiteLLM(model=model, temperature=temperature, max_tokens=max_tokens)


def create_agent_runnable(llm: ChatLiteLLM, system_prompt: str, role: str) -> Runnable:
    """Factory to create a new agent node's runnable."""
    # Load the seek config to get the use_robots setting
    seek_config = get_active_seek_config()
    seek_config.get("use_robots", True)

    tools = get_tools_for_role(role)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    if tools:
        return prompt | llm.bind_tools(tools)
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

    # Search through all missions and goals to find the matching context
    for mission in mission_config.get("missions", []):
        for goal in mission.get("goals", []):
            if goal.get("characteristic") == characteristic_name:
                return goal.get("context")  # Return the context string

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
