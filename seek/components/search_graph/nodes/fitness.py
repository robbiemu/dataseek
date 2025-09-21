import json

import json_repair
from langchain_core.messages import AIMessage

from seek.common.config import get_prompt
from seek.common.models import FitnessReport
from seek.components.mission_runner.state import DataSeekState

from .utils import (
    create_agent_runnable,
    create_llm,
    get_default_strategy_block,
    strip_reasoning_block,
)


def fitness_node(state: "DataSeekState") -> dict:
    """The fitness node, responsible for evaluating content and producing a structured report."""
    llm = create_llm("fitness")

    # --- START: PROVENANCE-AWARE LOGIC (Part 3) ---
    # 1. Retrieve the provenance flag from the state
    provenance = state.get("current_sample_provenance", "unknown")

    # 2. Construct dynamic guidance based on provenance
    provenance_guidance = ""
    if provenance == "researched":
        provenance_guidance = (
            "**Provenance Note:** The source of this document has been programmatically verified."
        )
    elif provenance == "synthetic":
        provenance_guidance = "\n**Provenance Note:** The indicated source for this document could not be programmatically verified, suggesting the content may be synthetic. Please evaluate its content and style on its own merits against the mission rubric, focusing on its potential value as a training example rather than its real-world origin."
    # --- END: PROVENANCE-AWARE LOGIC ---

    # Get the current mission context (characteristic, topic) from the task queue.
    current_task = state.get("current_task")
    strategy_block = state.get("strategy_block", "")

    if not current_task:
        characteristic = "the target"
        topic = "the correct"
        strategy_block = "No specific strategy defined."
    else:
        characteristic = current_task.get("characteristic", "the target")
        topic = current_task.get("topic", "the correct")

    if not strategy_block:
        print(
            f"   ⚠️  No strategy block found in state. Using built-in fallback for '{characteristic}'."
        )
        strategy_block = get_default_strategy_block(characteristic)

    research_findings = state.get("research_findings", [])
    # Escape braces in research findings to prevent template errors
    escaped_research_findings = str(research_findings).replace("{", "{{").replace("}", "}}")
    fitness_schema = json.dumps(FitnessReport.model_json_schema(), indent=2)

    # Escape curly braces in the JSON schema to prevent f-string template conflicts
    escaped_fitness_schema = fitness_schema.replace("{", "{{").replace("}", "}}")

    # --- START: REVISED PROMPT AND RUNNABLE CONSTRUCTION ---

    # 3. Define the system prompt via prompts configuration
    tpl = get_prompt("fitness", "base_prompt")
    system_prompt = tpl.format(
        characteristic=characteristic,
        topic=topic,
        strategy_block=strategy_block,
        provenance_guidance=provenance_guidance,
        research_findings=escaped_research_findings,
        fitness_schema=escaped_fitness_schema,
    )

    agent_runnable = create_agent_runnable(llm, system_prompt, "fitness")
    raw_result = agent_runnable.invoke({"messages": state["messages"]})

    try:
        dethought = strip_reasoning_block(raw_result.content)
        repaired_data = json_repair.loads(dethought)
        report = FitnessReport.model_validate(repaired_data)
    except Exception as parse_error:
        print(f"⚠️ Fitness Node: JSON parsing failed: {parse_error}")
        print(f"   Raw content: '{raw_result.content}'")
        report = FitnessReport(
            passed=False,
            reason="The quality inspector LLM failed to produce a valid structured evaluation. The source document could not be reliably assessed.",
        )

    # --- END: REVISED PROMPT AND RUNNABLE CONSTRUCTION ---

    # The rest of the function remains the same, but is now more robust.
    if not isinstance(report, FitnessReport):
        print(
            "❌ Fitness Node: LLM failed to return a valid FitnessReport object. Treating as REJECTED."
        )
        report = FitnessReport(
            passed=False,
            reason="The quality inspector LLM failed to produce a valid structured evaluation. The source document could not be reliably assessed.",
        )

    status = "APPROVED" if report.passed else "REJECTED"
    log_message_content = (
        f"**Inspection Report**\n- **Status:** {status}\n- **Reason:** {report.reason}"
    )

    return {
        "messages": [AIMessage(content=log_message_content)],
        "fitness_report": report,
        "current_sample_provenance": provenance,  # Pass through provenance from state
        "research_findings": state.get("research_findings", []),  # Pass through content for archive
    }
