from langchain_core.messages import AIMessage

from seek.common.config import get_prompt
from seek.components.mission_runner.state import DataSeekState

from .utils import (
    create_agent_runnable,
    create_llm,
    get_default_strategy_block,
    strip_reasoning_block,
)


def synthetic_node(state: DataSeekState) -> dict:
    """The synthetic node, responsible for generating synthetic data when syntethic results are preferable or when research fails."""
    llm = create_llm("synthetic")

    print("🎨 SYNTHETIC NODE: Starting synthetic data generation")
    print(f"   State messages: {len(state.get('messages', []))}")
    print(f"   Decision history: {state.get('decision_history', [])[-3:]}")

    current_task = state.get("current_task")
    strategy_block = state.get("strategy_block", "")

    if not current_task:
        raise ValueError("FATAL: No current_task found in state. The agent cannot proceed without a task.")

    characteristic = current_task.get("characteristic")
    if not characteristic:
        raise ValueError(f"FATAL: The current task is missing a 'characteristic'. Task: {current_task}")

    topic = current_task.get("topic", "general domain")
    print(f"   🎯 Task selected: characteristic={characteristic} topic={topic}")

    if not strategy_block:
        print(
            f"   ⚠️  No strategy block found in state. Using built-in fallback for '{characteristic}'."
        )
        strategy_block = get_default_strategy_block(characteristic)

    tpl = get_prompt("synthetic", "base_prompt")
    system_prompt = tpl.format(
        characteristic=characteristic, topic=topic, strategy_block=strategy_block
    )

    agent_runnable = create_agent_runnable(llm, system_prompt, "synthetic")
    result = agent_runnable.invoke({"messages": state["messages"]})

    # Process LLM output to match research node format for downstream compatibility
    report_content = strip_reasoning_block(result.content)

    print(f"🎨 SYNTHETIC NODE: Generated response of {len(report_content)} characters")

    # Log the generated content in the same format as research node for TUI display
    # This creates a message that the TUI can parse and extract sample content from
    display_message = AIMessage(
        content=(
            "      📝 --- START LLM RESPONSE ---"
            f"🎨 SYNTHETIC GENERATION: Created synthetic content for {characteristic} - {topic}\n\n"
            f"## Generated Content (Synthetic)\n\n{report_content}\n\n"
            "      📝 ---  END LLM RESPONSE  ---"
        )
    )

    # The synthetic node bypasses fitness and routes directly to archive.
    # We must populate the state in the same way the supervisor would when routing
    # to the fitness node, so the archive node can find the content.
    return {
        "messages": [display_message],  # Use display message for TUI logging
        "research_findings": [report_content],  # Pass content to the archive node
        "current_sample_provenance": "synthetic",  # Explicitly set provenance
    }
