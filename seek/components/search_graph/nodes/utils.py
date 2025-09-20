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
    use_robots = seek_config.get("use_robots", True)

    tools = get_tools_for_role(role, use_robots=use_robots)
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


def get_claimify_strategy_block(characteristic: str) -> str:
    """Get the strategic focus block for Claimify characteristics in data prospecting."""
    strategies = {
        "Decontextualization": """
**Strategic Focus for Decontextualization:**
Look for formal, encyclopedic, reference-style text that presents facts in a neutral, standalone manner. Ideal sources include:
- Academic papers with clear factual statements
- Technical documentation with precise specifications
- News articles with objective reporting style
- Reference materials like encyclopedias or handbooks

Avoid sources with:
- Heavy contextual dependencies ("as mentioned above", "this approach")
- Conversational or informal tone
- Opinion pieces or subjective commentary

The best documents will have sentences that can be extracted and understood independently, without needing surrounding context.""",
        "Coverage": """
**Strategic Focus for Coverage:**
Seek data-dense, comprehensive sources that thoroughly cover their subject matter with factual breadth. Ideal sources include:
- Comprehensive reports or surveys
- Statistical summaries and data compilations
- Complete technical specifications
- Thorough news coverage of events
- Academic literature reviews

Avoid sources with:
- Narrow, single-topic focus
- Sparse factual content
- Heavily theoretical or abstract content

The best documents will be rich repositories of diverse, verifiable facts that demonstrate comprehensive coverage of their domain.""",
        "Entailment": """
**Strategic Focus for Entailment:**
Target sources with clear, logical, unambiguous sentence structures that support straightforward factual claims. Ideal sources include:
- Technical manuals with step-by-step processes
- Scientific papers with clear methodology sections
- News reports with direct factual statements
- Educational materials with explicit explanations
- Legal or regulatory documents with precise language

Avoid sources with:
- Complex, multi-clause sentences
- Ambiguous or vague language
- Heavy use of metaphors or figurative language
- Speculative or hypothetical statements

The best documents will have simple, direct sentences where the logical relationship between premise and conclusion is crystal clear.""",
    }
    return strategies.get(
        characteristic,
        f"Look for sources that demonstrate clear {characteristic} characteristics in their writing style and structure.",
    )
