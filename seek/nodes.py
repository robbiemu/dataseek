# nodes.py
"""
Agent nodes for the Data Seek graph.
"""

import json
import json_repair
import yaml
from datetime import datetime
import time
from typing import Dict, List, Optional

import random

from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage

from .tools import (
    get_tools_for_role,
    write_file,
    _truncate_response_for_role,
)
from .utils import (
    get_characteristic_context,
    get_claimify_strategy_block,
    append_to_pedigree,
    strip_reasoning_block,
)

from .state import DataSeekState
from .models import FitnessReport
from .config import get_active_seek_config
from pydantic import BaseModel, Field


# Define the minimum step cost for a successful research cycle
MIN_STEPS_FOR_SUCCESSFUL_RESEARCH = (
    10  # 9 allows a full completion, 10 allows a synthetic fallback for failed research
)


class SupervisorDecision(BaseModel):
    """Defines the structured decision output for the supervisor LLM."""

    next_agent: str = Field(
        ...,
        description="The name of the next agent to route to (e.g., 'research', 'fitness').",
    )
    new_task: Optional[Dict] = Field(
        None,
        description="Optional: A new task to assign if the supervisor decides to switch focus.",
    )


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


def create_agent_runnable(llm: ChatLiteLLM, system_prompt: str, role: str):
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


def _get_next_task_from_progress(
    progress: Dict, exclude: Optional[List[tuple[str, str]]] = None
) -> Optional[Dict]:
    """Finds the next uncompleted characteristic/topic pair."""
    if not progress:
        return None

    mission_name = list(progress.keys())[0]
    eligible_tasks = []
    for char, char_data in progress[mission_name].items():
        if char_data["collected"] < char_data["target"]:
            for topic, topic_data in char_data["topics"].items():
                if topic_data["collected"] < topic_data["target"]:
                    if not exclude or (char, topic) not in exclude:
                        eligible_tasks.append({"characteristic": char, "topic": topic})

    if not eligible_tasks:
        return None

    return random.choice(eligible_tasks)


def supervisor_node(state: DataSeekState) -> Dict:
    """The supervisor node, now driven by a progress tracker."""
    llm = create_llm("supervisor")

    # Track recursion step
    step_count = state.get("step_count", 0)
    max_recursion_steps = state.get("max_recursion_steps", 30)  # Default value

    # Check if research is off-limits due to insufficient remaining steps
    steps_remaining = max_recursion_steps - step_count
    research_is_off_limits = steps_remaining < MIN_STEPS_FOR_SUCCESSFUL_RESEARCH

    # Initialize session tool/domain blocklist
    session_tool_domain_blocklist = state.get("session_tool_domain_blocklist", [])

    # Output recursion step information
    print(f"🔄 Supervisor: Recursion step {step_count}/{max_recursion_steps}")

    # --- CACHED-ONLY MODE CHECK ---
    # If we're in cached-only mode, deterministically route to research without LLM consultation
    if state.get("cached_only_mode"):
        print(
            "🔁 Supervisor: In cached-only mode. Checking quotas and routing deterministically."
        )

        current_task = state.get("current_task")
        progress = state.get("progress", {})

        # Verify quotas again
        if not needs_more_for_current_pair(progress, current_task, pending_increment=0):
            print(
                "   📊 Supervisor: Quota satisfied for cached mode. Setting exhausted flag."
            )
            next_agent = "end"
            steps_to_add = calculate_predictive_steps(next_agent)
            next_step_count = step_count + steps_to_add
            return {
                "next_agent": next_agent,
                "progress": progress,
                "current_task": current_task,
                "strategy_block": state.get("strategy_block", ""),
                "cached_exhausted": True,
                "next_cycle_cached_reuse": {"active": False},
                "decision_history": state.get("decision_history", []) + ["end"],
                "excluded_urls": state.get("excluded_urls", []),
                "tool_execution_failures": state.get("tool_execution_failures", 0),
                "research_attempts": state.get("research_attempts", 0),
                "consecutive_failures": state.get("consecutive_failures", 0),
                "last_action_status": "success",
                "last_action_agent": "supervisor",
                "synthetic_samples_generated": state.get(
                    "synthetic_samples_generated", 0
                ),
                "research_samples_generated": state.get(
                    "research_samples_generated", 0
                ),
                "synthetic_budget": state.get("synthetic_budget", 0.2),
                "fitness_report": None,
                "step_count": next_step_count,  # Increment step count for next iteration
            }

        # Check if we have any allowed URLs left
        allowed = set(state.get("allowed_url_whitelist", [])) - set(
            state.get("excluded_urls", [])
        )
        if not allowed:
            print(
                "   🚫 Supervisor: No allowed URLs remaining in cached mode. Setting exhausted flag."
            )
            next_agent = "end"
            steps_to_add = calculate_predictive_steps(next_agent)
            next_step_count = step_count + steps_to_add
            return {
                "next_agent": next_agent,
                "progress": progress,
                "current_task": current_task,
                "strategy_block": state.get("strategy_block", ""),
                "cached_exhausted": True,
                "next_cycle_cached_reuse": {"active": False},
                "decision_history": state.get("decision_history", []) + ["end"],
                "excluded_urls": state.get("excluded_urls", []),
                "tool_execution_failures": state.get("tool_execution_failures", 0),
                "research_attempts": state.get("research_attempts", 0),
                "consecutive_failures": state.get("consecutive_failures", 0),
                "last_action_status": "success",
                "last_action_agent": "supervisor",
                "synthetic_samples_generated": state.get(
                    "synthetic_samples_generated", 0
                ),
                "research_samples_generated": state.get(
                    "research_samples_generated", 0
                ),
                "synthetic_budget": state.get("synthetic_budget", 0.2),
                "fitness_report": None,
                "step_count": next_step_count,  # Increment step count for next iteration
            }

        # Deterministically route to research for cached-only processing
        print(
            f"   🎯 Supervisor: Routing to research in cached-only mode with {len(allowed)} allowed URLs."
        )
        next_agent = "research"
        steps_to_add = calculate_predictive_steps(next_agent)
        next_step_count = step_count + steps_to_add
        return {
            "next_agent": next_agent,
            "progress": progress,
            "current_task": current_task,
            "strategy_block": state.get("strategy_block", ""),
            "decision_history": state.get("decision_history", []) + ["research"],
            "excluded_urls": state.get("excluded_urls", []),
            "tool_execution_failures": state.get("tool_execution_failures", 0),
            "research_attempts": state.get("research_attempts", 0) + 1,
            "consecutive_failures": state.get("consecutive_failures", 0),
            "last_action_status": "success",
            "last_action_agent": "supervisor",
            "synthetic_samples_generated": state.get("synthetic_samples_generated", 0),
            "research_samples_generated": state.get("research_samples_generated", 0),
            "synthetic_budget": state.get("synthetic_budget", 0.2),
            "fitness_report": None,
            # Preserve cached-only mode state
            "cached_only_mode": True,
            "no_search_tools": True,
            "allowed_url_whitelist": list(allowed),
            "cached_exhausted": False,
            "next_cycle_cached_reuse": state.get("next_cycle_cached_reuse"),
            "step_count": next_step_count,  # Increment step count for next iteration
        }

    # --- 1. Load State ---
    progress = state.get("progress", {})
    _current_mission = state.get("current_mission", "production_corpus")
    current_task = state.get("current_task")

    # --- 3. Select Next Task (if necessary) ---
    # We only select a new task if there's no current task.
    if not current_task:
        next_task = _get_next_task_from_progress(progress)
    else:
        next_task = current_task

    # --- 4. Decide What to Do ---
    if not next_task:
        # All tasks are complete.
        print("🎉 Supervisor: All tasks in the mission are complete!")
        next_agent = "end"
        steps_to_add = calculate_predictive_steps(next_agent)
        next_step_count = step_count + steps_to_add
        return {
            "next_agent": next_agent,
            "progress": progress,
            "strategy_block": "",
            "step_count": next_step_count,
            "session_tool_domain_blocklist": session_tool_domain_blocklist,
        }

    # We have a task. Now, we determine the strategy block for it.
    characteristic = next_task.get("characteristic", "Verifiability")
    topic = next_task.get("topic", "general domain")
    try:
        with open("settings/mission_config.yaml", "r") as f:
            content = f.read()
            if content.startswith("#"):
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1 :]
            mission_config = yaml.safe_load(content)
        characteristic_context = get_characteristic_context(next_task, mission_config)
    except Exception:
        characteristic_context = None

    if characteristic_context:
        print(
            f"   ✅ Supervisor: Using dynamic context for '{characteristic}' from mission plan."
        )
        strategy_block = (
            f"**Strategic Focus for '{characteristic}':**\n{characteristic_context}"
        )
    else:
        print(
            f"   ⚠️  Supervisor: Could not find dynamic context for '{characteristic}'. Using built-in fallback."
        )
        strategy_block = get_claimify_strategy_block(characteristic)

    decision_history = state.get("decision_history", [])
    tool_execution_failures = state.get("tool_execution_failures", 0)
    research_attempts = state.get("research_attempts", 0)
    consecutive_failures = state.get("consecutive_failures", 0)
    last_action_status = state.get("last_action_status", "success")
    last_action_agent = state.get("last_action_agent", "")

    # If the last action was a successful archive, evaluate for cached reuse.
    if last_action_agent == "archive":
        print(
            "✅ Supervisor: Detected a successful archival. Evaluating for cached reuse."
        )

        # 1) Update exclusions
        research_findings = state.get("research_findings", [])
        used_urls = extract_used_urls_from_findings(research_findings)
        current_excluded = state.get("excluded_urls", [])
        # Normalize and deduplicate URLs
        all_excluded = list(
            set([normalize_url(url) for url in current_excluded + used_urls])
        )

        print(
            f"   📝 Supervisor: Adding {len(used_urls)} URLs to exclusion list. Total excluded: {len(all_excluded)}"
        )

        # 2) Check provenance and quota
        provenance = state.get("current_sample_provenance", "unknown")
        if provenance != "researched":
            print(
                f"   ⚠️  Supervisor: Sample provenance '{provenance}' != 'researched'. Skipping cached reuse."
            )
            next_agent = "end"
            steps_to_add = calculate_predictive_steps(next_agent)
            next_step_count = step_count + steps_to_add
            return {
                "next_agent": next_agent,
                "progress": state.get("progress", {}),
                "current_task": state.get("current_task"),
                "strategy_block": state.get("strategy_block", ""),
                "decision_history": state.get("decision_history", []) + ["end"],
                "excluded_urls": all_excluded,
                "next_cycle_cached_reuse": {"active": False},
                "tool_execution_failures": state.get("tool_execution_failures", 0),
                "research_attempts": state.get("research_attempts", 0),
                "consecutive_failures": state.get("consecutive_failures", 0),
                "last_action_status": "success",
                "last_action_agent": "supervisor",
                "synthetic_samples_generated": state.get(
                    "synthetic_samples_generated", 0
                ),
                "research_samples_generated": state.get(
                    "research_samples_generated", 0
                ),
                "synthetic_budget": state.get("synthetic_budget", 0.2),
                "fitness_report": None,
                "step_count": next_step_count,  # Increment step count for next iteration
            }

        # Check if we need more samples for this characteristic/topic pair
        if not needs_more_for_current_pair(
            state.get("progress", {}), next_task, pending_increment=1
        ):
            print(
                f"   📊 Supervisor: Quota satisfied for {next_task.get('characteristic', 'N/A')}/{next_task.get('topic', 'N/A')}. Ending cycle."
            )
            next_agent = "end"
            steps_to_add = calculate_predictive_steps(next_agent)
            next_step_count = step_count + steps_to_add
            return {
                "next_agent": next_agent,
                "progress": state.get("progress", {}),
                "current_task": state.get("current_task"),
                "strategy_block": state.get("strategy_block", ""),
                "decision_history": state.get("decision_history", []) + ["end"],
                "excluded_urls": all_excluded,
                "next_cycle_cached_reuse": {"active": False},
                "tool_execution_failures": state.get("tool_execution_failures", 0),
                "research_attempts": state.get("research_attempts", 0),
                "consecutive_failures": state.get("consecutive_failures", 0),
                "last_action_status": "success",
                "last_action_agent": "supervisor",
                "synthetic_samples_generated": state.get(
                    "synthetic_samples_generated", 0
                ),
                "research_samples_generated": state.get(
                    "research_samples_generated", 0
                ),
                "synthetic_budget": state.get("synthetic_budget", 0.2),
                "fitness_report": None,
                "step_count": next_step_count,  # Increment step count for next iteration
            }

        # 3) Build candidate cache index
        cache_index = index_research_cache(state.get("research_session_cache", []))
        # Filter to exclude already used URLs
        excluded_set = set(all_excluded)
        available_candidates = [
            entry for entry in cache_index if entry["url"] not in excluded_set
        ]

        print(
            f"   🗂️  Supervisor: Found {len(available_candidates)} unused cached sources"
        )

        if not available_candidates:
            print("   📭 Supervisor: No unused cached sources available. Ending cycle.")
            next_agent = "end"
            steps_to_add = calculate_predictive_steps(next_agent)
            next_step_count = step_count + steps_to_add
            return {
                "next_agent": next_agent,
                "progress": state.get("progress", {}),
                "current_task": state.get("current_task"),
                "strategy_block": state.get("strategy_block", ""),
                "decision_history": state.get("decision_history", []) + ["end"],
                "excluded_urls": all_excluded,
                "next_cycle_cached_reuse": {"active": False},
                "tool_execution_failures": state.get("tool_execution_failures", 0),
                "research_attempts": state.get("research_attempts", 0),
                "consecutive_failures": state.get("consecutive_failures", 0),
                "last_action_status": "success",
                "last_action_agent": "supervisor",
                "synthetic_samples_generated": state.get(
                    "synthetic_samples_generated", 0
                ),
                "research_samples_generated": state.get(
                    "research_samples_generated", 0
                ),
                "synthetic_budget": state.get("synthetic_budget", 0.2),
                "fitness_report": None,
                "step_count": next_step_count,  # Increment step count for next iteration
            }

        # 4) Let LLM select cached sources
        print(
            f"   🤖 Supervisor: Asking LLM to evaluate {len(available_candidates)} cached sources..."
        )

        # Get synthetic budget info for LLM context
        synthetic_samples_generated = state.get("synthetic_samples_generated", 0)
        research_samples_generated = state.get("research_samples_generated", 0)
        total_samples_generated = (
            synthetic_samples_generated + research_samples_generated
        )
        synthetic_budget = state.get("synthetic_budget", 0.2)

        # Calculate total target from progress
        total_samples_target = 0
        progress = state.get("progress", {})
        if progress:
            mission_name = list(progress.keys())[0]
            for char_data in progress[mission_name].values():
                total_samples_target += char_data["target"]

        decision, selected_urls, rationale = llm_select_cached_sources(
            llm=llm,
            characteristic=characteristic,
            topic=topic,
            strategy_block=strategy_block,
            excluded_urls=all_excluded,
            cache_index=available_candidates,
            synthetic_samples_generated=synthetic_samples_generated,
            total_samples_generated=total_samples_generated,
            synthetic_budget=synthetic_budget,
            total_samples_target=total_samples_target,
        )

        print(
            f"   🎯 Supervisor: LLM decision='{decision}', selected={len(selected_urls)} URLs"
        )
        print(f"   💭 Supervisor: Rationale: {rationale}")

        if decision == "reuse_cached" and selected_urls:
            # Final validation: ensure selected URLs are not in excluded list
            final_selected = [
                url for url in selected_urls if normalize_url(url) not in excluded_set
            ]

            if final_selected:
                # Build carryover plan
                carryover_plan = {
                    "active": True,
                    "allowed_url_whitelist": final_selected,
                    "current_task": next_task,
                    "research_session_cache": state.get("research_session_cache", []),
                    "rationale": rationale,
                }

                print(
                    f"   ✅ Supervisor: Scheduling cached reuse with {len(final_selected)} sources for next cycle"
                )
                next_agent = "end"
                steps_to_add = calculate_predictive_steps(next_agent)
                next_step_count = step_count + steps_to_add
                return {
                    "next_agent": next_agent,
                    "progress": state.get("progress", {}),
                    "current_task": state.get("current_task"),
                    "strategy_block": state.get("strategy_block", ""),
                    "decision_history": state.get("decision_history", []) + ["end"],
                    "excluded_urls": all_excluded,
                    "next_cycle_cached_reuse": carryover_plan,
                    "tool_execution_failures": state.get("tool_execution_failures", 0),
                    "research_attempts": state.get("research_attempts", 0),
                    "consecutive_failures": state.get("consecutive_failures", 0),
                    "last_action_status": "success",
                    "last_action_agent": "supervisor",
                    "synthetic_samples_generated": state.get(
                        "synthetic_samples_generated", 0
                    ),
                    "research_samples_generated": state.get(
                        "research_samples_generated", 0
                    ),
                    "synthetic_budget": state.get("synthetic_budget", 0.2),
                    "fitness_report": None,
                    "step_count": next_step_count,  # Increment step count for next iteration
                    "session_tool_domain_blocklist": session_tool_domain_blocklist,  # Pass through blocklist
                }

        # Default: no cached reuse
        print("   📋 Supervisor: No cached reuse planned. Ending cycle normally.")
        next_agent = "end"
        steps_to_add = calculate_predictive_steps(next_agent)
        next_step_count = step_count + steps_to_add
        return {
            "next_agent": next_agent,
            "progress": state.get("progress", {}),
            "current_task": state.get("current_task"),
            "strategy_block": state.get("strategy_block", ""),
            "decision_history": state.get("decision_history", []) + ["end"],
            "excluded_urls": all_excluded,
            "next_cycle_cached_reuse": {"active": False},
            "tool_execution_failures": state.get("tool_execution_failures", 0),
            "research_attempts": state.get("research_attempts", 0),
            "consecutive_failures": state.get("consecutive_failures", 0),
            "last_action_status": "success",
            "last_action_agent": "supervisor",
            "synthetic_samples_generated": state.get("synthetic_samples_generated", 0),
            "research_samples_generated": state.get("research_samples_generated", 0),
            "synthetic_budget": state.get("synthetic_budget", 0.2),
            "fitness_report": None,
            "step_count": next_step_count,  # Increment step count for next iteration
        }

    synthetic_samples_generated = state.get("synthetic_samples_generated", 0)
    research_samples_generated = state.get("research_samples_generated", 0)
    synthetic_budget = state.get("synthetic_budget", 0.2)
    total_samples_generated = synthetic_samples_generated + research_samples_generated

    messages = state.get("messages", [])

    # Check if we have a new research report first (this takes priority over old fitness reports)
    last_message_content = str(messages[-1].content) if messages else ""
    has_new_research_report = (
        decision_history
        and decision_history[-1] == "research"
        and "# Data Prospecting Report" in last_message_content
    )

    # Only check for old fitness reports if we don't have a new research report
    fitness_report = (
        state.get("fitness_report") if not has_new_research_report else None
    )

    if fitness_report:
        if fitness_report.passed:
            print(
                "✅ Supervisor: Detected PASSED fitness report in state. Deterministically routing to 'archive'."
            )
            # Clear the fitness report from state
            state_dict = dict(state)
            state_dict["fitness_report"] = None

            next_agent = "archive"
            steps_to_add = calculate_predictive_steps(next_agent)
            next_step_count = step_count + steps_to_add
            return {
                "next_agent": next_agent,
                "progress": progress,
                "current_task": next_task,
                "strategy_block": strategy_block,
                "fitness_report": None,
                "decision_history": decision_history + ["archive"],
                "tool_execution_failures": tool_execution_failures,
                "research_attempts": research_attempts,
                "consecutive_failures": 0,
                "last_action_status": "success",
                "last_action_agent": "supervisor",
                "synthetic_samples_generated": synthetic_samples_generated,
                "research_samples_generated": research_samples_generated,
                "synthetic_budget": synthetic_budget,
                "research_findings": state.get("research_findings", []),
                "current_sample_provenance": state.get(
                    "current_sample_provenance", "unknown"
                ),  # Pass through provenance
                "step_count": next_step_count,  # Increment step count for next iteration
                "session_tool_domain_blocklist": [],  # Reset blocklist after successful archive
            }
        else:
            print(
                "❌ Supervisor: Detected FAILED fitness report in state. Proposing a new task to the supervisor."
            )
            # Clear the fitness report from state
            state_dict = dict(state)
            state_dict["fitness_report"] = None

            # --- NEW: Log the failure ---
            task_history = state.get("task_history", [])
            current_task = state.get("current_task")
            if current_task:
                task_history.append(
                    (
                        current_task.get("characteristic"),
                        current_task.get("topic"),
                        fitness_report.reason,
                    )
                )

            # --- NEW: Give the supervisor the option to change tasks ---
            excluded_tasks = [(t[0], t[1]) for t in task_history]
            alt_task = _get_next_task_from_progress(progress, exclude=excluded_tasks)

            if alt_task and alt_task != next_task:
                alt_characteristic = alt_task.get("characteristic", "N/A")
                alt_topic = alt_task.get("topic", "N/A")
                synthetic_budget = state.get("synthetic_budget", 0.2)
                total_samples_target = 0
                if progress:
                    mission_name = list(progress.keys())[0]
                    for char_data in progress[mission_name].values():
                        total_samples_target += char_data["target"]
                max_synthetic_samples = int(total_samples_target * synthetic_budget)
                synthetic_samples_generated = state.get(
                    "synthetic_samples_generated", 0
                )

                last_action_analysis = f"""**3. Last Action Analysis:** FAILURE
   - **Agent:** fitness
   - **Reason:** The agent rejected the previous submission: {fitness_report.reason}
   - **Guidance:** You need to decide the next action based on the complete history you see. It may be that the current task is difficult to research, and we could more easily make progress on a different task. You have three options:
     1. Delegate to `research` to retry the current task (`{next_task["characteristic"]}` / `{next_task["topic"]}`).
     2. Delegate to 'synthetic' to complete the task (we have only {max_synthetic_samples - synthetic_samples_generated} of {max_synthetic_samples} submissions remaining that should ideally be synthetic)
     3. Switch to a different, uncompleted task, such as (`{alt_characteristic}` / `{alt_topic}`), by setting the `new_task` field in your response. If you switch, the researcher's memory will be cleared."""
            else:
                last_action_analysis = f"""**3. Last Action Analysis:** FAILURE
   - **Agent:** fitness
   - **Reason:** The agent rejected the previous submission: {fitness_report.reason}
   - **Guidance:** The current task is the only one remaining. You must delegate to `research` to retry it, but consider suggesting a new strategy."""

            # We will now fall through to the standard supervisor prompt, but with the special failure guidance.

    last_message_content = str(messages[-1].content) if messages else ""

    # --- SENTINEL AND WHITELIST ENFORCEMENT ---
    # Check for "No More Cached Data" sentinel from research
    if (
        decision_history
        and decision_history[-1] == "research"
        and (
            "# No More Cached Data" in last_message_content
            or "NO_MORE_CACHED_DATA" in last_message_content
        )
    ):
        print(
            "📋 Supervisor: Research signaled 'No More Cached Data'. Setting exhausted flag and ending cycle."
        )
        next_agent = "end"
        steps_to_add = calculate_predictive_steps(next_agent)
        next_step_count = step_count + steps_to_add
        return {
            "next_agent": next_agent,
            "progress": progress,
            "current_task": next_task,
            "strategy_block": strategy_block,
            "decision_history": decision_history + ["end"],
            "excluded_urls": state.get("excluded_urls", []),
            "cached_exhausted": True,
            "next_cycle_cached_reuse": {"active": False},
            "tool_execution_failures": tool_execution_failures,
            "research_attempts": research_attempts,
            "consecutive_failures": consecutive_failures,
            "last_action_status": "success",
            "last_action_agent": "supervisor",
            "synthetic_samples_generated": synthetic_samples_generated,
            "research_samples_generated": research_samples_generated,
            "synthetic_budget": synthetic_budget,
            "fitness_report": None,
            "step_count": next_step_count,  # Increment step count for next iteration
        }

    # Check for Data Prospecting Report and enforce whitelist in cached-only mode
    if (
        decision_history
        and decision_history[-1] == "research"
        and "# Data Prospecting Report" in last_message_content
    ):
        # If we're in cached-only mode, enforce whitelist before routing to fitness
        is_cached_mode = state.get("cached_only_mode") or state.get("no_search_tools")
        if is_cached_mode:
            print(
                "🛂 Supervisor: In cached-only mode. Enforcing URL whitelist before routing to fitness."
            )

            # Extract URL from the report
            report_url = None
            for line in last_message_content.split("\n"):
                if "**Source URL**:" in line:
                    try:
                        parts = line.split("`")
                        if len(parts) > 1:
                            report_url = parts[1].strip()
                    except Exception:
                        pass
                    break

            if report_url:
                # Check if URL is in whitelist and not in excluded list
                allowed = set(state.get("allowed_url_whitelist", [])) - set(
                    state.get("excluded_urls", [])
                )
                normalized_report_url = normalize_url(report_url)

                if normalized_report_url not in {normalize_url(url) for url in allowed}:
                    print(
                        f"⚠️  Supervisor: URL '{report_url}' not in allowed whitelist. Treating as violation."
                    )
                    print(f"   Allowed URLs: {list(allowed)}")
                    next_agent = "end"
                    steps_to_add = calculate_predictive_steps(next_agent)
                    next_step_count = step_count + steps_to_add
                    return {
                        "next_agent": next_agent,
                        "progress": progress,
                        "current_task": next_task,
                        "strategy_block": strategy_block,
                        "decision_history": decision_history + ["end"],
                        "excluded_urls": state.get("excluded_urls", []),
                        "cached_exhausted": True,
                        "next_cycle_cached_reuse": {"active": False},
                        "tool_execution_failures": tool_execution_failures,
                        "research_attempts": research_attempts,
                        "consecutive_failures": consecutive_failures,
                        "last_action_status": "success",
                        "last_action_agent": "supervisor",
                        "synthetic_samples_generated": synthetic_samples_generated,
                        "research_samples_generated": research_samples_generated,
                        "synthetic_budget": synthetic_budget,
                        "fitness_report": None,
                        "step_count": next_step_count,  # Increment step count for next iteration
                    }
                else:
                    print(
                        f"✅ Supervisor: URL '{report_url}' is in whitelist. Proceeding to fitness."
                    )

        # Continue with normal Data Prospecting Report processing
        print(
            "✅ Supervisor: Detected a completed 'Data Prospecting Report'. Deterministically routing to 'fitness'."
        )

        # --- FIX: Extract content for the fitness node ---
        # Default values
        source_url = None
        provenance = "synthetic"
        research_findings = []

        # Safely extract the report content
        report_content = strip_reasoning_block(last_message_content)

        # --- FIX: Handle cache references ---
        research_cache = state.get("research_session_cache", [])
        if "[CACHE_REFERENCE:" in report_content and research_cache:
            try:
                # Extract call_id from the report
                call_id = report_content.split("[CACHE_REFERENCE:")[1].split("]")[0]

                # Find the corresponding evidence in the cache
                for evidence in research_cache:
                    if evidence.get("call_id") == call_id:
                        # Replace the token with the actual tool output
                        tool_output = evidence.get("output", "")
                        report_content = str(tool_output)
                        print(
                            f"✅ Supervisor: Resolved cache reference for call_id '{call_id}'."
                        )
                        break
            except IndexError:
                print("⚠️ Supervisor: Malformed cache reference found.")
                pass  # Keep the original report_content if reference is malformed

        research_findings.append(report_content)

        # Try to parse the URL from the report
        for line in report_content.split("\n"):
            if "**Source URL**:" in line:
                try:
                    source_url = line.split("`")
                    if len(source_url) > 1:
                        source_url = source_url[1]
                    else:
                        source_url = None
                except IndexError:
                    pass  # Keep source_url as None
                break

        # Verify provenance if a URL was found
        research_cache = state.get("research_session_cache", [])
        if source_url and research_cache:
            for evidence in research_cache:
                is_valid_evidence = (
                    isinstance(evidence, dict)
                    and "args" in evidence
                    and isinstance(evidence["args"], dict)
                )
                if is_valid_evidence:
                    args = evidence["args"]
                    found_url = (
                        args.get("url") or args.get("base_url") or args.get("start_url")
                    )
                    if found_url == source_url:
                        output = evidence.get("output", {})
                        if isinstance(output, dict) and output.get("status") == "ok":
                            provenance = "researched"
                            print(
                                f"✅ Supervisor: Provenance VERIFIED as 'researched' for URL: {source_url}"
                            )
                            break  # Found definitive proof
            if provenance == "synthetic":
                print(
                    f"⚠️  Supervisor: Provenance could NOT be verified for URL: {source_url}. Defaulting to 'synthetic'."
                )
        else:
            print(
                "ℹ️ Supervisor: No source URL or research cache found, defaulting provenance to 'synthetic'."
            )

        next_agent = "fitness"
        steps_to_add = calculate_predictive_steps(next_agent)
        next_step_count = step_count + steps_to_add
        return {
            "next_agent": next_agent,
            "decision_history": decision_history + ["fitness"],
            "consecutive_failures": 0,
            "last_action_status": "success",
            "last_action_agent": "supervisor",
            "current_sample_provenance": provenance,  # Pass through provenance from state
            "progress": progress,
            "current_task": next_task,
            "strategy_block": strategy_block,
            "tool_execution_failures": tool_execution_failures,
            "research_attempts": research_attempts,
            "synthetic_samples_generated": synthetic_samples_generated,
            "research_samples_generated": research_samples_generated,
            "synthetic_budget": synthetic_budget,
            # Pass the extracted content to the next node
            "research_findings": research_findings,
            # Clear any old fitness report when routing to fitness for a new report
            "fitness_report": None,
            "step_count": next_step_count,  # Increment step count for next iteration
        }

    research_detail = (
        "\n- `research`: Finds source documents from the web."
        if not research_is_off_limits
        else ""
    )

    base_prompt = f"""You are the supervisor of a team of Data Prospecting agents. Your role is to analyze the current mission status and decide which agent should act next.
    
    Available Agents:
    - `fitness`: Evaluates the quality of a source document.
    - `archive`: Saves an approved document.
    - `synthetic`: Generates a document from scratch.
    - `end`: Finishes the mission.{research_detail}"""

    characteristic = next_task.get("characteristic", "N/A")
    topic = next_task.get("topic", "N/A")
    current_task_str = f"   - Find sources for the characteristic '{characteristic}' in the topic '{topic}'."

    total_samples_target = 0
    if progress:
        mission_name = list(progress.keys())[0]
        for char_data in progress[mission_name].values():
            total_samples_target += char_data["target"]

    max_synthetic_samples = int(total_samples_target * synthetic_budget)
    remaining_synthetic_budget = max_synthetic_samples - synthetic_samples_generated
    remaining_total_work = total_samples_target - total_samples_generated
    current_synthetic_pct = (
        (synthetic_samples_generated / total_samples_generated)
        if total_samples_generated > 0
        else 0.0
    )
    research_success_rate = (
        "high"
        if consecutive_failures < 2 and research_attempts < 3
        else "moderate"
        if consecutive_failures < 4
        else "low"
    )

    # More integrated status summary
    progress_pct = (
        (total_samples_generated / total_samples_target) * 100
        if total_samples_target > 0
        else 0
    )
    mission_status = f"{total_samples_generated}/{total_samples_target} samples ({progress_pct:.0f}%) | {current_synthetic_pct:.0%} synthetic | {research_success_rate} research success"

    research_detail = (
        "\n**CRITICAL ALERT:** There are not enough steps remaining to complete a full research cycle. You **MUST NOT** route to `research`. Your only productive options are `synthetic` (if the budget allows) or `end`.\n"
        if research_is_off_limits
        else ""
    )

    strategic_guidance = f"""
**4. Strategic Reasoning Guidance:**
Your primary goal is generating high-quality samples efficiently. Consider both task requirements and resource management:

**Mission Progress:**
- **Generated**: {total_samples_generated}/{total_samples_target} samples ({progress_pct:.0f}% complete)
- **Breakdown**: {synthetic_samples_generated} synthetic, {total_samples_generated - synthetic_samples_generated} research-based
- **Current rate**: {current_synthetic_pct:.0%} synthetic (target: ~{synthetic_budget:.0%})

**Resource Status:**
- **Synthetic budget**: {remaining_synthetic_budget}/{max_synthetic_samples} remaining (target: {synthetic_budget:.0%}, current: {current_synthetic_pct:.0%})
- **Research success**: {research_success_rate} (based on recent attempts)
- **Remaining work**: {remaining_total_work} samples needed
{research_detail}
**Decision Framework:**
1. **Primary consideration**: Which approach is most likely to succeed for this specific characteristic/topic?
2. **Secondary consideration**: Balance toward the synthetic target while ensuring quality
3. **Budget awareness**: Synthetic budget is both a target (aim for ~{synthetic_budget:.0%}) and ceiling (don't significantly exceed it)

**Practical guidance**:
- **Under synthetic target** ({current_synthetic_pct:.0%} < {synthetic_budget:.0%}): Prefer `synthetic` when research is challenging or synthetic would be high-quality
- **Near synthetic target** ({current_synthetic_pct:.0%} ~= {synthetic_budget:.0%}): Either approach fine--choose based on success likelihood
- **Over synthetic target** ({current_synthetic_pct:.0%} > {synthetic_budget:.0%}): Prefer `research` unless it's repeatedly failing
- **Research struggling**: Don't let budget prevent synthetic generation if research keeps failing
- **Mission end**: If approaching completion, balance final ratio toward target if possible"""

    last_action_analysis = ""
    last_message_content = (
        str(messages[-1].content) if messages else "No recent messages."
    )

    if last_action_status == "failure":
        failure_agent = last_action_agent if last_action_agent else "unknown"
        if failure_agent == "research":
            last_action_analysis = f"""**3. Last Action Analysis:** FAILURE
   - **Agent:** research
   - **Reason:** The agent failed to produce a valid Data Prospecting Report.
   - **Context**: {research_attempts} research attempts, {consecutive_failures} consecutive failures
   - **Guidance:** Research is struggling for this characteristic/topic. Consider:
     1. Try `research` again with a different approach (if this seems like a temporary issue)
     2. Switch to `synthetic` (especially if research has failed {research_attempts}+ times or this characteristic/topic is inherently difficult to research)
     3. The decision should prioritize success over synthetic budget--better to generate a useful synthetic sample than fail repeatedly"""
        elif failure_agent == "archive":
            error_snippet = last_message_content.replace("\n", " ").strip()[:150]
            last_action_analysis = f"""**3. Last Action Analysis:** FAILURE
   - **Agent:** archive
   - **Reason:** A tool error occurred during file saving: '{error_snippet}...'
   - **Guidance:** This is a system error. You should probably `end` the mission so a human can investigate. Retrying is unlikely to succeed."""
        else:
            last_action_analysis = f"""**3. Last Action Analysis:** FAILURE
   - **Agent:** {failure_agent}
   - **Reason:** The last action resulted in an error.
   - **Guidance:** Analyze the mission history and decide the best recovery path."""
    else:
        success_agent = last_action_agent if last_action_agent else "initial_start"
        last_action_analysis = f"""**3. Last Action Analysis:** SUCCESS
   - **Agent:** {success_agent}
   - **Guidance:** The previous task was completed. It is time to start the next task. Analyze the full mission context and decide on the next agent."""

    mission_context = f"""
---
**MISSION CONTEXT**

**1. Current Task:**
{current_task_str}

**2. Mission Status:**
   - **Progress**: {mission_status}
   - **Decision History (Last 5):** {decision_history[-5:]}
   - **Recent Failures:** {consecutive_failures} consecutive

{last_action_analysis}

{strategic_guidance}
---
Your response MUST be a JSON object matching the required schema, with a single key "next_agent" and the name of the agent as the value."""

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", base_prompt),
            ("human", mission_context),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    agent = agent_prompt | llm
    raw_result = agent.invoke({"messages": messages})

    try:
        dethought = strip_reasoning_block(raw_result.content)
        repaired_data = json_repair.loads(dethought)
        decision = SupervisorDecision.model_validate(repaired_data)
        next_agent = decision.next_agent.lower()

        # Override research choice if it's off-limits due to insufficient steps
        if research_is_off_limits and next_agent == "research":
            print(
                "   ⚠️ Supervisor: LLM chose 'research' despite limit. Overriding to 'end'."
            )
            # You could also add logic here to try 'synthetic' if it's cheaper and budget allows.
            next_agent = "end"

        # --- NEW: Handle task switching ---
        if decision.new_task:
            print(f"✅ Supervisor: Switching to new task: {decision.new_task}")
            next_task = decision.new_task
            # Recalculate strategy block for the new task
            characteristic = next_task.get("characteristic", "Verifiability")
            try:
                with open("settings/mission_config.yaml", "r") as f:
                    content = f.read()
                    if content.startswith("#"):
                        first_newline = content.find("\n")
                        if first_newline != -1:
                            content = content[first_newline + 1 :]
                    mission_config = yaml.safe_load(content)
                characteristic_context = get_characteristic_context(
                    next_task, mission_config
                )
            except Exception:
                characteristic_context = None

            if characteristic_context:
                strategy_block = f"**Strategic Focus for '{characteristic}':**\n{characteristic_context}"
            else:
                strategy_block = get_claimify_strategy_block(characteristic)

            # Clear messages for the researcher
            messages = []

            # Reset the session tool/domain blocklist for the new task
            session_tool_domain_blocklist = []
    except Exception as parse_error:
        print(f"⚠️ Supervisor: JSON parsing failed: {parse_error}")
        print(f"   Raw content: '{raw_result.content}'")
        next_agent = "research"

    # Calculate predictive steps for the LLM decision case
    steps_to_add = calculate_predictive_steps(next_agent)
    next_step_count = step_count + steps_to_add
    return {
        "next_agent": next_agent,
        "progress": progress,
        "current_task": next_task,
        "strategy_block": strategy_block,
        "decision_history": decision_history + [next_agent],
        "tool_execution_failures": tool_execution_failures,
        "research_attempts": research_attempts + (1 if next_agent == "research" else 0),
        "consecutive_failures": consecutive_failures,
        "last_action_status": "success",
        "last_action_agent": "supervisor",
        "synthetic_samples_generated": synthetic_samples_generated,
        "research_samples_generated": research_samples_generated,
        "synthetic_budget": synthetic_budget,
        # Clear any fitness report when falling through to standard logic
        "fitness_report": None,
        "step_count": next_step_count,  # Increment step count for next iteration
        "session_tool_domain_blocklist": session_tool_domain_blocklist,  # Pass through blocklist
    }


def research_node(state: "DataSeekState") -> Dict:
    """
    Verifiable Research Workflow:
    - Dynamically scoped ReAct loop (discover -> extract -> synthesize)
    - Cache ALL successful tool outputs as evidence in research_session_cache
    - Return a single 'Data Prospecting Report' as the final submission
    - Do NOT mutate research_findings here (Supervisor handles parsing)
    """
    # --- Setup ---
    llm = create_llm("research")

    print("🔍 RESEARCH NODE (Verifiable Research Workflow)")
    print(f"   Incoming messages: {len(state.get('messages', []))}")
    print(f"   LLM model: {getattr(llm, 'model', 'unknown')}")

    # --- Read session tool/domain blocklist ---
    session_tool_domain_blocklist = state.get("session_tool_domain_blocklist", [])
    if session_tool_domain_blocklist:
        print(
            f"   🚫 Session blocklist contains {len(session_tool_domain_blocklist)} entries:"
        )
        for tool_name, domain in session_tool_domain_blocklist:
            print(f"      - {tool_name} blocked for domain {domain}")

    # --- Extract the latest human question ---
    user_question = None
    for msg in reversed(state.get("messages", [])):
        # LangChain HumanMessage or any object with type == "human"
        if getattr(msg, "content", None):
            if getattr(msg, "type", None) == "human" or "HumanMessage" in str(
                type(msg)
            ):
                user_question = msg.content
                break

    if not user_question:
        print("   ❗ No user question found; returning request for clarification.")
        no_question_msg = AIMessage(
            content="No clear research question found in conversation history. Please provide a specific question for me to research."
        )
        return {
            "messages": [no_question_msg],
            # Return unchanged cache if any exists
            "research_session_cache": state.get("research_session_cache", []),
            # Return unchanged blocklist
            "session_tool_domain_blocklist": session_tool_domain_blocklist,
        }

    # --- Evidence cache (auditable log of work for Supervisor) ---
    session_cache = list(
        state.get("research_session_cache", [])
    )  # make a copy we can append to
    print(f"   Session cache (pre-run): {len(session_cache)} items")

    # --- Tools & Mission Context ---
    # Load the seek config to get the use_robots setting
    seek_config = get_active_seek_config()
    use_robots = seek_config.get("use_robots", True)

    all_research_tools = get_tools_for_role("research", use_robots=use_robots)
    print(f"   Tools available (global): {[t.name for t in all_research_tools]}")

    current_task = state.get("current_task")
    strategy_block = state.get("strategy_block", "")
    if current_task:
        characteristic = current_task.get("characteristic", "Verifiability")
        topic = current_task.get("topic", "general domain")
        print(f"   🎯 Task selected: characteristic={characteristic} topic={topic}")
    else:
        characteristic = "Verifiability"
        topic = "general domain"
        print("   🎯 No specific task queued; using default mission focus.")

    if not strategy_block:
        print(
            f"   ⚠️  No strategy block found in state. Using built-in fallback for '{characteristic}'."
        )
        strategy_block = get_claimify_strategy_block(characteristic)

    # --- CACHED-ONLY MODE CHECK ---
    cached_only = state.get("cached_only_mode") or state.get("no_search_tools")
    allowed_urls = list(
        set(state.get("allowed_url_whitelist", []))
        - set(state.get("excluded_urls", []))
    )

    if cached_only:
        print(
            f"   🔁 Research: Operating in cached-only mode with {len(allowed_urls)} allowed URLs"
        )

        # If no allowed URLs remain, emit sentinel
        if not allowed_urls:
            print(
                "   📭 Research: No allowed URLs remaining. Emitting 'No More Cached Data' sentinel."
            )
            sentinel_msg = AIMessage(content="# No More Cached Data")
            return {
                "messages": [sentinel_msg],
                "research_session_cache": session_cache,
            }

        # Build condensed cache index for allowed URLs only
        cache_index = index_research_cache(session_cache)
        allowed_cache_entries = [
            entry
            for entry in cache_index
            if entry["url"] in {normalize_url(url) for url in allowed_urls}
        ]

        print(
            f"   🗂️  Research: Found {len(allowed_cache_entries)} cached entries for allowed URLs"
        )

        # Build cached entries description for LLM
        cache_descriptions = []
        for i, entry in enumerate(allowed_cache_entries, 1):
            desc = f"{i}. **{entry['url']}** ({entry['source_type']})\n   Content: {entry['content_excerpt'][:200]}..."
            cache_descriptions.append(desc)

        cache_context = (
            "\n\n".join(cache_descriptions)
            if cache_descriptions
            else "No cached entries available."
        )

    # --- System prompt (mission-specific) ---
    if cached_only:
        # Cached-only mode prompt
        system_prompt = f"""You are a Data Prospector operating in CACHED-ONLY MODE. You have already gathered cached data and must now select from pre-retrieved sources.

Your Mission: Select and process one source from the previously cached data for the characteristic **"{characteristic}"** in the topic **"{topic}"**.

---
{strategy_block}
---

### CACHED-ONLY CONSTRAINTS

**CRITICAL RESTRICTIONS:**
- You may NOT use search tools (`web_search`, `arxiv_search`, `wikipedia_search`)
- You may ONLY select from the allowed URLs listed below
- You may reference cached data using `[CACHE_REFERENCE: call_id]` tokens
- You may use `url_to_markdown` tool ONLY for URLs in the allowed list

### ALLOWED URLs:
{chr(10).join(f"- {url}" for url in allowed_urls)}

### CACHED DATA AVAILABLE:
{cache_context}

### REQUIRED OUTPUT:

You must produce either:

1. **A complete Data Prospecting Report** (if you find a suitable cached source):

# Data Prospecting Report

**Target Characteristic**: `{characteristic}`
**Search Domain**: `{topic}`

**Source URL**: `[Must be one of the allowed URLs above]`
**Source Title**: `"[Title from cached data or URL]"`

---

## Justification for Selection

* **Alignment with `{characteristic}`**: [Explain why this cached source aligns with the characteristic]
* **Potential for High Yield**: [Explain why this will yield good training examples]

---

## Retrieved Content (Markdown)

`[Use [CACHE_REFERENCE: call_id] or paste curated content from allowed URL]`

2. **OR, if no cached sources are suitable:**

# No More Cached Data

If none of the cached sources are appropriate for this characteristic/topic combination.

**Remember:** Only select URLs from the allowed list. Any other URL will be rejected.
"""
    else:
        # Normal mode prompt
        system_prompt = f"""You are a Data Prospector, a specialist in identifying high-quality raw text for data extraction pipelines. You operate using a ReAct (Reasoning and Acting) methodology.

Your Mission: Your goal is to find and retrieve a source document from the **{topic}** domain whose writing style and structure make it an exceptionally good source for extracting factual claims that exemplify the principle of **"{characteristic}"**.

You are not extracting the final claims. You are finding the *ore*. You must find a document that is naturally rich in sentences that a downstream agent could easily turn into high-quality claims with the desired characteristic.

---
{strategy_block}
---

### ReAct Process & Tool Usage

Your workflow is a two-step process: **Discover, then Extract.**

1.  **REASON:** Based on your strategic focus, formulate a search plan.
2.  **ACT (Discover):** Use the search tools (`web_search`, `arxiv_search`, `wikipedia_search`) to find promising URLs. The output of these tools is just a list of links and snippets; it is **not** the final document.
3.  **OBSERVE:** Analyze the search results. Identify the single most promising URL that is likely to contain the full source document.
4.  **ACT (Extract):** Use the `url_to_markdown` or `documentation_crawler` tool on that single URL to retrieve the **full text** of the candidate document.
5.  **REPEAT:** If the extracted document is low-quality or irrelevant, discard it and refine your search. Continue until you find one high-quality source document that is a strong match.

### Content Curation (Expert Refinement)

Your goal is to maximize the signal-to-noise ratio for the next agent.

- **To submit a specific, high-value excerpt:** If a specific section of a document is highly relevant, you should extract and submit ONLY that section in the `Retrieved Content (Markdown)` block. You may use ellipses `(...)` on their own line to indicate where you have removed irrelevant surrounding text.
- **To submit the entire document:** If the whole document is a good fit, you do NOT need to copy its contents. Instead, use a special token to reference the cached tool output. Find the `call_id` from the successful tool call in your history and place it in the report like this:

`[CACHE_REFERENCE: call_...]`

This tells the supervisor to fetch the full content from the cache, saving context space.

When you have successfully found and extracted a suitable source, you MUST output a single, structured 'Data Prospecting Report' exactly in the format below--no extra commentary. Your response must start with `# Data Prospecting Report`.

# Data Prospecting Report

**Target Characteristic**: `{characteristic}`
**Search Domain**: `{topic}`

**Source URL**: `[The specific URL of the retrieved content]`
**Source Title**: `"[The title of the web page or document]"`

---

## Justification for Selection

* **Alignment with `{characteristic}`**: [Explain in detail *why* the writing style, sentence structure, and overall format of this document make it an excellent source for extracting claims that will have the property of `{characteristic}`. Refer back to your strategic focus in your reasoning.]
* **Potential for High Yield**: [Briefly explain why you believe this document will provide a large number of usable examples for the downstream agents.]

---

## Retrieved Content (Markdown)

`[Either paste the curated excerpt OR provide the [CACHE_REFERENCE: ...] token here.]`
"""

    # --- Base prompt template (system is dynamic per-iteration) ---
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt_for_iteration}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # --- Seed conversation for the agent (internal to this node) ---
    react_messages = []
    if state.get("messages"):
        react_messages.extend(state["messages"])

    seek_config = get_active_seek_config()

    # Get max_iterations from seek config, with fallback to default
    max_iterations = 8  # Default value
    if seek_config and "nodes" in seek_config and "research" in seek_config["nodes"]:
        max_iterations = int(
            seek_config["nodes"]["research"].get("max_iterations", max_iterations)
        )

    print(f"   🔄 ReAct loop starting (max_iterations={max_iterations})")

    final_report_msg = None
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"   ▶ Iteration {iteration}/{max_iterations}")

        # Context tracking for research progress
        history_char_count = sum(len(str(m.content)) for m in react_messages)
        print(
            f"   📊 CONTEXT: {len(react_messages)} messages, ~{history_char_count} chars"
        )

        # Dynamic scoping + warnings
        warning_message = ""
        current_tools = all_research_tools

        # In cached-only mode, filter out search tools from the start
        if cached_only:
            current_tools = [
                t
                for t in all_research_tools
                if t.name not in ["web_search", "arxiv_search", "wikipedia_search"]
            ]
            if iteration == 1:
                warning_message = (
                    "\n\n**CACHED-ONLY MODE:** Search tools are disabled. You must work with cached data only. "
                    "Select from allowed URLs or emit 'No More Cached Data' if none are suitable."
                )
            elif iteration == max_iterations:
                current_tools = []
                warning_message = (
                    "\n\n**FINAL ITERATION (CACHED MODE):** All tools disabled. "
                    "Produce either a 'Data Prospecting Report' or 'No More Cached Data' now."
                )
        else:
            # Normal mode dynamic scoping
            if iteration == max_iterations - 2:
                warning_message = (
                    "\n\n**SYSTEM WARNING:** You have 3 iterations remaining."
                    "Begin concluding discovery and prepare to select a final document for extraction. This is the last turn that you may use the search tools (`web_search`, `arxiv_search`, `wikipedia_search`)."
                )
            elif iteration == max_iterations - 1:
                # Disable discovery; force extraction
                current_tools = [
                    t
                    for t in all_research_tools
                    if t.name not in ["web_search", "arxiv_search", "wikipedia_search"]
                ]
                warning_message = (
                    "\n\n**SYSTEM WARNING:** You have 2 iterations remaining. "
                    "The search tools (`web_search`, `arxiv_search`, `wikipedia_search`) have been disabled. Analyze current leads and extract a full document. This is the last turn that you may use the tools you are being provided."
                )
            elif iteration == max_iterations:
                # Disable all tools; force synthesis
                current_tools = []
                warning_message = (
                    "\n\n**SYSTEM WARNING:** FINAL iteration. All tools disabled. "
                    "Write the final 'Data Prospecting Report' now."
                )

        system_prompt_for_iteration = system_prompt + warning_message

        # Debug final iteration (uncomment for detailed debugging)
        # if iteration == max_iterations:
        #     print("      ❗ FINAL PROMPT: The following system prompt is being sent to the LLM for its last chance.")
        #     print("      " + "-" * 20)
        #     print(f"      {system_prompt_for_iteration[-500:]}")  # Print last 500 chars
        #     print("      " + "-" * 20)

        # Bind tools as scoped this iteration
        llm_with_tools = llm.bind_tools(current_tools) if current_tools else llm
        print(
            f"      Tools this iteration: {[t.name for t in current_tools] if current_tools else '[]'}"
        )

        # Build runnable and invoke
        react_agent = (
            prompt_template.partial(
                system_prompt_for_iteration=system_prompt_for_iteration
            )
            | llm_with_tools
        )

        try:
            result = react_agent.invoke({"messages": react_messages})

            # Debug LLM responses (uncomment for detailed debugging)
            # print("      📝 --- START RAW LLM RESPONSE ---")
            # print(f"{getattr(result, 'content', '[NO CONTENT]').strip()}")
            # print("      📝 ---  END RAW LLM RESPONSE  ---")

            # RECOVERY: Handle empty responses on final iteration with sleep-and-retry
            raw_content = getattr(result, "content", "")
            if iteration == max_iterations and (
                not raw_content or not raw_content.strip()
            ):
                print(
                    "      🚨 CRITICAL: Empty response on final iteration - attempting retry after brief pause"
                )

                try:
                    print("      ⏳ Waiting 3 seconds for model to stabilize...")
                    import time

                    time.sleep(3)

                    print("      🔄 Retrying final iteration with same prompt...")
                    # Use the exact same agent and prompt - just retry
                    retry_result = react_agent.invoke({"messages": react_messages})
                    retry_content = getattr(retry_result, "content", "")

                    print("      📝 --- RETRY RESPONSE ---")
                    print(f"{retry_content.strip()}")
                    print("      📝 --- END RETRY ---")

                    # If retry produced content, use it
                    if retry_content and retry_content.strip():
                        result.content = retry_content
                        print("      ✅ Retry successful - using response")
                    else:
                        print("      ⚠️ Retry also failed - will use fallback")

                except Exception as retry_error:
                    print(f"      ❌ Retry attempt failed: {retry_error}")
                    # Continue with the empty result, fallback will handle it

            react_messages.append(result)

            # Successful termination: exact string check per spec
            print(
                f"      🕵️‍♀️ Checking for final report in content: {getattr(result, 'content', '')[:100]}..."
            )
            if (
                getattr(result, "content", "")
                and "# Data Prospecting Report" in result.content
            ):
                print("      🏁 Final submission detected ('Data Prospecting Report').")
                final_report_msg = result
                break

            # Process tool calls (cache ALL successful outputs as evidence)
            tool_calls = getattr(result, "tool_calls", None)
            if tool_calls:
                print(f"      🔧 Tool calls: {len(tool_calls)}")
                for idx, tool_call in enumerate(tool_calls):
                    try:
                        # Normalize tool call access (dict or object)
                        tool_name = (
                            tool_call.get("name")
                            if isinstance(tool_call, dict)
                            else getattr(tool_call, "name", None)
                        )
                        tool_args = (
                            tool_call.get("args", {})
                            if isinstance(tool_call, dict)
                            else getattr(tool_call, "args", {}) or {}
                        )
                        tool_id = (
                            tool_call.get("id")
                            if isinstance(tool_call, dict)
                            else getattr(tool_call, "id", f"call_{iteration}_{idx}")
                        )

                        matching_tool = next(
                            (t for t in current_tools if t.name == tool_name), None
                        )
                        if not matching_tool:
                            print(
                                f"         ⚠️ Tool '{tool_name}' not found in current scope; skipping."
                            )
                            continue

                        print(
                            f"         ▶ Executing {tool_name} with args: {str(tool_args)[:200]} ..."
                        )

                        tool_result = matching_tool.invoke(tool_args)

                        # Debug: Print tool result details
                        if isinstance(tool_result, dict):
                            status = tool_result.get("status")
                            if status == "error":
                                error_detail = tool_result.get("error", "")
                                print(
                                    f"         📊 Tool result status: {status} {error_detail}"
                                )
                            else:
                                print(f"         📊 Tool result status: {status}")

                            if tool_name in [
                                "url_to_markdown",
                                "documentation_crawler",
                            ]:
                                content_key = (
                                    "markdown"
                                    if tool_name == "url_to_markdown"
                                    else "full_markdown"
                                )
                                content_data = tool_result.get(content_key)
                                data_type = type(content_data).__name__
                                data_len = len(content_data) if content_data else 0
                                print(
                                    f"         📊 Tool result data: <class '{data_type}'> - {data_len} chars"
                                )
                            else:
                                results_data = tool_result.get("results")
                                data_type = type(results_data).__name__
                                num_items = (
                                    len(results_data)
                                    if results_data
                                    else "None/Negative"
                                )
                                print(
                                    f"         📊 Tool result data: <class '{data_type}'> - {num_items} items"
                                )
                        else:
                            print(
                                f"         📊 Tool result: {type(tool_result)} - {str(tool_result)[:200]}"
                            )

                        # Validate search tool results to ensure URLs are accessible
                        if tool_name in [
                            "web_search",
                            "arxiv_search",
                            "wikipedia_search",
                        ] and isinstance(tool_result, dict):
                            if tool_name == "web_search":
                                # Only proceed with validation if there are results left
                                if tool_result.get(
                                    "status"
                                ) == "ok" and tool_result.get("results"):
                                    from .seek_utils import _validate_search_results

                                    print(
                                        f"         🔍 Validating {tool_name} results..."
                                    )
                                    validation_result = _validate_search_results(
                                        tool_result["results"],
                                        tool_name,
                                        tool_args,
                                        matching_tool,
                                        session_tool_domain_blocklist,  # Pass the blocklist
                                    )

                                    tool_result["results"] = validation_result[
                                        "results"
                                    ]
                                    tool_result["validation_info"] = validation_result
                                    print(
                                        f"         ✅ {tool_name} results validated ({len(validation_result['results'])} results)"
                                    )

                                    # Log retry information if performed
                                    if validation_result.get("retry_performed"):
                                        if validation_result.get("retry_successful"):
                                            print(
                                                f"         🔄 {tool_name}: Auto-retry successful"
                                            )
                                        else:
                                            print(
                                                f"         🔄 {tool_name}: Auto-retry attempted but unsuccessful"
                                            )
                                else:
                                    # Handle error or genuinely empty results
                                    if tool_result.get("status") == "error":
                                        error_msg = tool_result.get(
                                            "error", "Unknown error"
                                        )
                                        print(
                                            f"         ❌ {tool_name} tool failed: {error_msg}"
                                        )
                                    else:
                                        print(
                                            f"         ⚠️  No results returned by {tool_name} tool"
                                        )

                                    # Create a validation result indicating no results to maintain structure
                                    tool_result["validation_info"] = {
                                        "results": [],
                                        "validation_performed": True,
                                        "needs_retry": False,
                                        "filtered_count": 0,
                                        "retry_performed": False,
                                    }
                            else:
                                # For non-web_search tools, proceed with normal validation
                                if tool_result.get(
                                    "status"
                                ) == "ok" and tool_result.get("results"):
                                    from .seek_utils import _validate_search_results

                                    print(
                                        f"         🔍 Validating {tool_name} results..."
                                    )
                                    validation_result = _validate_search_results(
                                        tool_result["results"],
                                        tool_name,
                                        tool_args,
                                        matching_tool,
                                        session_tool_domain_blocklist,  # Pass the blocklist
                                    )

                                    tool_result["results"] = validation_result[
                                        "results"
                                    ]
                                    tool_result["validation_info"] = validation_result
                                    print(
                                        f"         ✅ {tool_name} results validated ({len(validation_result['results'])} results)"
                                    )

                                    # Log retry information if performed
                                    if validation_result.get("retry_performed"):
                                        if validation_result.get("retry_successful"):
                                            print(
                                                f"         🔄 {tool_name}: Auto-retry successful"
                                            )
                                        else:
                                            print(
                                                f"         🔄 {tool_name}: Auto-retry attempted but unsuccessful"
                                            )

                        # Post-process results based on tool + role limits
                        if isinstance(tool_result, dict):
                            tool_result = _truncate_response_for_role(
                                tool_result, "research", tool_name=tool_name
                            )

                        # Append ToolMessage for agent observation
                        react_messages.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_id,
                            )
                        )

                        # === EVIDENCE CACHING (unconditional for successful calls) ===
                        session_cache.append(
                            {
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "iteration": iteration,
                                "call_id": tool_id,
                                "tool": tool_name,
                                "args": tool_args,  # original arguments
                                "output": tool_result,  # full raw output (JSON/dict/string)
                                "user_question": user_question,
                            }
                        )
                        print("         ✅ Evidence cached.")

                    except Exception as tool_error:
                        print(f"         ❌ Tool execution error: {tool_error}")
                        # Provide an observation to the agent but DO NOT cache as successful evidence
                        react_messages.append(
                            ToolMessage(
                                content=f"Tool '{tool_name}' failed: {tool_error}",
                                tool_call_id=tool_id
                                if "tool_id" in locals()
                                else f"error_{iteration}_{idx}",
                            )
                        )

                        # === UPDATE BLOCKLIST FOR TOOL/DOMAIN FAILURES ===
                        # Extract domain from URL args and add to blocklist
                        if tool_name and tool_args and isinstance(tool_args, dict):
                            url = (
                                tool_args.get("url")
                                or tool_args.get("base_url")
                                or tool_args.get("start_url")
                            )
                            if url:
                                try:
                                    from urllib.parse import urlparse

                                    domain = urlparse(url).netloc
                                    if domain:
                                        # Check if this combination is already blocked
                                        block_entry = (tool_name, domain)
                                        if (
                                            block_entry
                                            not in session_tool_domain_blocklist
                                        ):
                                            session_tool_domain_blocklist.append(
                                                block_entry
                                            )
                                            print(
                                                f"         🚫 Added to blocklist: {tool_name} for domain {domain}"
                                            )
                                        else:
                                            print(
                                                f"         ℹ️  Already blocked: {tool_name} for domain {domain}"
                                            )
                                except Exception as parse_error:
                                    print(
                                        f"         ⚠️  Could not parse domain from URL {url}: {parse_error}"
                                    )
                # Let the loop continue so the model can reason over observations
                continue

            else:  # This else block is the key change
                # No tools were called and no final report yet; continue  iterations
                print("      ℹ️ No tool calls this step; continuing.")
                continue

        except Exception as iter_error:
            print(f"      ❌ Iteration error: {iter_error}")
            # Continue to fallback after loop

    print(f"   ✅ Loop ended after {iteration} iterations")

    # --- Fallback: always return a 'Data Prospecting Report' (even if empty) ---
    if not final_report_msg:
        # [EXISTING CODE] print("   ⚠️ No final report produced; generating a minimal report to satisfy contract.")

        # Research failure analysis - useful for diagnosing issues
        print("   ⚠️ Research failed to produce final report after all iterations")
        print("   Last 3 messages:")
        for msg in react_messages[-3:]:
            print(
                f"      - [{getattr(msg, 'type', 'UNKNOWN').upper()}]: {str(getattr(msg, 'content', ''))[:100]}..."
            )

        # Produce an honest, minimal report so Supervisor can proceed
        fallback_report = f"""# Data Prospecting Report

**Target Characteristic**: `{characteristic}`
**Search Domain**: `{topic}`

**Source URL**: `None`
**Source Title**: `"No qualifying source selected"`

---

## Justification for Selection

* **Alignment with `{characteristic}`**: The agent did not produce a final selection within the allotted iterations.
* **Potential for High Yield**: Unable to assess due to missing final selection.

---

## Retrieved Content (Markdown)

`No extracted content. See research_session_cache for all gathered evidence.`
"""
        final_report_msg = AIMessage(content=fallback_report)
    elif not final_report_msg.content.startswith("# Data Prospecting Report"):
        final_report_msg.content = (
            "# Data Prospecting Report\n\n" + final_report_msg.content
        )

    # --- Return only the final submission + full evidence cache ---
    print(
        f"   🧾 Returning final submission + evidence (cache size: {len(session_cache)})"
    )
    return {
        "messages": [final_report_msg],  # Append the final Data Prospecting Report ONLY
        "research_session_cache": session_cache,  # Full, updated evidence cache (no clearing)
        "session_tool_domain_blocklist": session_tool_domain_blocklist,  # Updated blocklist
    }


def archive_node(state: "DataSeekState") -> Dict:
    """
    The archive node, responsible for saving data and updating the audit trail
    using a procedural approach.
    """
    # Load seek config instead of main config for writer paths
    seek_config = get_active_seek_config()
    llm = create_llm("archive")

    # --- FIX: Get content from research_findings, not messages ---
    provenance = state.get("current_sample_provenance", "synthetic")
    print(f"   🏷️  Archive: Received provenance '{provenance}' from state")
    messages = state.get("messages", [])
    research_findings = state.get("research_findings", [])

    # Robustly find the document content
    document_content = None
    if research_findings:
        # Content is now a list of strings, so we join them.
        document_content = "\n\n---\n\n".join(research_findings)
        print("   ✅ Archive: Found content in 'research_findings'.")
    else:
        # Fallback for older states or different paths
        for msg in reversed(messages):
            if hasattr(msg, "content") and "# Data Prospecting Report" in msg.content:
                document_content = msg.content
                print("   ⚠️  Archive: Found content via message search fallback.")
                break

    if not document_content:
        error_message = AIMessage(
            content="Archive Error: Could not find a 'Data Prospecting Report' in the conversation history to save."
        )
        return {"messages": messages + [error_message]}

    # Get task details for naming
    current_task = state.get("current_task")
    characteristic = "unknown"
    topic = "unknown"
    if current_task:
        characteristic = (
            current_task.get("characteristic", "unknown").lower().replace(" ", "_")
        )
        topic = current_task.get("topic", "unknown").lower().replace(" ", "_")

    # --- FIX: Remove JSON and ask for the raw markdown string directly ---
    system_prompt = f"""You are the Library Cataloger in the Claimify data pipeline.

Your task is to generate a concise pedigree catalog entry in Markdown format based on the provided document.

The entry MUST include:
- The sample's provenance: **'{provenance}'**
- The source URL from the document.
- The target characteristic from the document.

Respond ONLY with the Markdown content for the pedigree entry. Do NOT include any other text, greetings, or JSON formatting.

**Example Response:**
### YYYY-MM-DD -- Sample Archived
- **Source Type:** {provenance}
- **Source URL:** [source url]
- **Target Characteristic:** {characteristic}
"""
    agent_runnable = create_agent_runnable(llm, system_prompt, "archive")
    llm_result = agent_runnable.invoke({"messages": messages})

    # The LLM's raw output is now our entry markdown. No parsing needed.
    entry_markdown = llm_result.content

    # --- Procedural control flow ---

    # Generate a unique timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")

    # Construct the deterministic filepath using mission state
    filename = f"{characteristic}_{topic}_{timestamp}.md"
    samples_path = state.get("samples_path", "examples/data/datasets/tier1")
    filepath = f"{samples_path}/{filename}"

    # Now use this 'filepath' variable in the write_file tool.
    write_result = write_file.invoke(
        {"filepath": filepath, "content": document_content}
    )

    if write_result.get("status") == "ok":
        run_id = state.get("run_id")
        # Use pedigree path from mission state
        pedigree_path = state.get("pedigree_path") or "examples/PEDIGREE.md"
        append_to_pedigree(
            pedigree_path=pedigree_path,
            entry_markdown=entry_markdown,
            run_id=run_id,
        )
    else:
        error_message = AIMessage(
            content=f"Archive Error: Failed to write file to '{filepath}'. Error: {write_result.get('error')}"
        )
        return {"messages": messages + [error_message]}

    # --- Progress counters ---
    is_synthetic_sample = provenance == "synthetic"
    synthetic_samples_generated = state.get("synthetic_samples_generated", 0)
    research_samples_generated = state.get("research_samples_generated", 0)

    if is_synthetic_sample:
        synthetic_samples_generated += 1
    else:
        research_samples_generated += 1

    samples_generated = state.get("samples_generated", 0) + 1
    total_target = state.get("total_samples_target", 1200)
    total_samples = synthetic_samples_generated + research_samples_generated
    synthetic_pct = (
        (synthetic_samples_generated / total_samples) if total_samples > 0 else 0.0
    )
    progress_pct = (samples_generated / total_target) * 100 if total_target > 0 else 0
    source_type = "synthetic" if is_synthetic_sample else "research"

    print(
        f"   📊 Sample #{samples_generated} archived ({progress_pct:.1f}% complete) - Source: {source_type}"
    )
    print(
        f"   📈 Updated ratios - Research: {research_samples_generated}, Synthetic: {synthetic_samples_generated} ({synthetic_pct:.1%})"
    )

    confirmation_message = AIMessage(
        content=f"Successfully archived document to '{filepath}' and updated the pedigree."
    )
    return {
        "messages": messages + [confirmation_message],
        "samples_generated": samples_generated,
        "synthetic_samples_generated": synthetic_samples_generated,
        "research_samples_generated": research_samples_generated,
        "last_action_agent": "archive",
    }


def fitness_node(state: "DataSeekState") -> Dict:
    """The fitness node, responsible for evaluating content and producing a structured report."""
    llm = create_llm("fitness")

    # --- START: PROVENANCE-AWARE LOGIC (Part 3) ---
    # 1. Retrieve the provenance flag from the state
    provenance = state.get("current_sample_provenance", "unknown")

    # 2. Construct dynamic guidance based on provenance
    provenance_guidance = ""
    if provenance == "researched":
        provenance_guidance = "**Provenance Note:** The source of this document has been programmatically verified."
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
        strategy_block = get_claimify_strategy_block(characteristic)

    research_findings = state.get("research_findings", [])
    # Escape braces in research findings to prevent template errors
    escaped_research_findings = (
        str(research_findings).replace("{", "{{").replace("}", "}}")
    )
    fitness_schema = json.dumps(FitnessReport.model_json_schema(), indent=2)

    # Escape curly braces in the JSON schema to prevent f-string template conflicts
    escaped_fitness_schema = fitness_schema.replace("{", "{{").replace("}", "}}")

    # --- START: REVISED PROMPT AND RUNNABLE CONSTRUCTION ---

    # 3. Define the system prompt with all dynamic content properly escaped
    system_prompt = f"""You are a Quality Inspector in the Claimify data pipeline. Your role is to evaluate whether a 'book' (a source document) found by the Research Agent is a high-quality source for our mission.

Your job is to inspect the **entire book** and decide if it's worth keeping. A downstream process, the 'Copy Editor', will later extract the specific 'quotes' (claims). You are NOT extracting quotes, only approving the source.

---
**Current Mission Context**
- **Target Characteristic:** {characteristic}
- **Search Domain:** {topic}
---
**Quality Standards for this Mission**

To be approved, the document's writing style and structure must align with the strategic focus for '{characteristic}'. Here is your rubric:

---
{strategy_block}
---

**Your Task**

{provenance_guidance}

The Research Agent has returned the following document(s):
{escaped_research_findings}

Evaluate the retrieved content against the mission. Is this document a goldmine for the Copy Editor, or a waste of time?

**CRITICAL: Your response must be ONLY a valid JSON object with no additional text, explanations, or formatting. Do not include any text before or after the JSON. Start your response directly with the opening brace {{ and end with the closing brace }}.

**JSON Schema to follow:**
```json
{escaped_fitness_schema}
```"""

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
        "research_findings": state.get(
            "research_findings", []
        ),  # Pass through content for archive
    }


def synthetic_node(state: DataSeekState) -> Dict:
    """The synthetic node, responsible for generating synthetic data when syntethic results are preferable or when research fails."""
    llm = create_llm("synthetic")

    print("🎨 SYNTHETIC NODE: Starting synthetic data generation")
    print(f"   State messages: {len(state.get('messages', []))}")
    print(f"   Decision history: {state.get('decision_history', [])[-3:]}")

    current_task = state.get("current_task")
    strategy_block = state.get("strategy_block", "")
    if current_task:
        characteristic = current_task.get("characteristic", "Verifiability")
        topic = current_task.get("topic", "general domain")
        print(f"   🎯 Task selected: characteristic={characteristic} topic={topic}")
    else:
        characteristic = "Verifiability"
        topic = "general domain"
        print("   🎯 No specific task queued; using default mission focus.")

    if not strategy_block:
        print(
            f"   ⚠️  No strategy block found in state. Using built-in fallback for '{characteristic}'."
        )
        strategy_block = get_claimify_strategy_block(characteristic)

    system_prompt = f"""You are a Synthetic Book Author in the Claimify data pipeline. Your role is to create high-quality synthetic books (source documents) when the Librarian has failed to find suitable real books from the internet.

When the Librarian has been unable to find good books for a specific characteristic and domain, you step in to author a synthetic book that would be perfect for the Copy Editor to extract quotes from.

Analyze the conversation history to understand:
1. What Claimify characteristic was being targeted ({characteristic})
2. What topic domain was being searched ({topic})
3. Why the real book search failed

**Your synthetic book should be crafted to maximize signal-to-noise ratio for quote extraction:**

{strategy_block}

**Output Format (mimic the Librarian's Data Prospecting Report):**

# Data Prospecting Report

**Target Characteristic**: `{characteristic}`
**Search Domain**: `{topic}`

**Source URL**: `https://synthetic-library.generated/[relevant-path]`
**Source Title**: `"[Title of your synthetic book]"`

---

## Justification for Selection

* **Alignment with `{characteristic}`**: [Why this synthetic book is perfect for the characteristic]
* **Potential for High Yield**: [Why the Copy Editor will find many excellent quotes here]

---

## Retrieved Content (Markdown)

`[Your substantial, realistic synthetic book content - rich with extractable quotes]`

Focus on creating a book that will be a goldmine for the Copy Editor to extract high-quality sentences from."""

    agent_runnable = create_agent_runnable(llm, system_prompt, "synthetic")
    result = agent_runnable.invoke({"messages": state["messages"]})

    # --- FIX: Align output with the new data flow ---
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


def get_node_config(node_name: str) -> Optional[Dict]:
    """Retrieve the configuration for a specific node by name."""
    # Load the seek config instead of using the old config.seek_agent
    seek_config = get_active_seek_config()

    if (
        seek_config
        and "mission_plan" in seek_config
        and "nodes" in seek_config["mission_plan"]
    ):
        # Find the node config that matches the node_name
        for node in seek_config["mission_plan"]["nodes"]:
            if node.get("name") == node_name:
                return node
    return None


def calculate_predictive_steps(next_agent: str) -> int:
    """Calculate the number of steps that will be taken in the upcoming path.

    Based on the graph structure:
    - research path takes 3 steps (supervisor -> research -> research_tools -> supervisor)
    - fitness path takes 2 steps (supervisor -> fitness -> supervisor)
    - archive path takes 3 steps (supervisor -> archive -> archive_tools -> supervisor)
    - synthetic path takes 4 steps (supervisor -> synthetic -> archive -> archive_tools -> supervisor)
    - end path takes 0 steps

    Returns:
        int: Number of steps to add to the step count
    """
    steps_map = {"research": 3, "fitness": 2, "archive": 3, "synthetic": 4, "end": 0}
    return steps_map.get(next_agent, 0)


# === Supervisor Helper Functions for Cached Reuse ===


def needs_more_for_current_pair(
    progress: Dict, current_task: Optional[Dict], pending_increment: int = 0
) -> bool:
    """Compute if we need more samples for the current characteristic/topic pair."""
    if not progress or not current_task:
        return False

    mission_name = list(progress.keys())[0] if progress else None
    if not mission_name:
        return False

    characteristic = current_task.get("characteristic")
    topic = current_task.get("topic")

    if not characteristic or not topic:
        return False

    char_data = progress.get(mission_name, {}).get(characteristic, {})
    topic_data = char_data.get("topics", {}).get(topic, {})

    collected = topic_data.get("collected", 0) + pending_increment
    target = topic_data.get("target", 0)

    return collected < target


def extract_used_urls_from_findings(research_findings: List) -> List[str]:
    """Parse Source URLs from Data Prospecting Reports in research findings."""
    urls = []

    if not research_findings:
        return urls

    for finding in research_findings:
        finding_str = str(finding)
        lines = finding_str.split("\n")

        for line in lines:
            if "**Source URL**:" in line:
                try:
                    # Extract URL from markdown format: **Source URL**: `url`
                    parts = line.split("`")
                    if len(parts) > 1:
                        url = parts[1].strip()
                        if url and url not in urls:
                            urls.append(url)
                except Exception:
                    continue

    return urls


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


def index_research_cache(cache: List[Dict]) -> List[Dict]:
    """Index research cache entries by URL with metadata."""
    indexed = []
    seen_urls = set()

    for entry in cache or []:
        if not isinstance(entry, dict):
            continue

        args = entry.get("args", {})
        if not isinstance(args, dict):
            continue

        # Extract URL from args
        url = args.get("url") or args.get("base_url") or args.get("start_url")
        if not url:
            continue

        # Normalize URL
        normalized_url = normalize_url(url)
        if normalized_url in seen_urls:
            continue  # Skip duplicates, prefer first occurrence
        seen_urls.add(normalized_url)

        # Determine status
        output = entry.get("output")
        ok_status = False
        content_excerpt = ""
        content_length = 0

        if isinstance(output, dict):
            ok_status = output.get("status") == "ok"
            if ok_status and "content" in output:
                content = str(output["content"])
                content_length = len(content)
                content_excerpt = content[:800] + ("..." if len(content) > 800 else "")
        elif output:
            ok_status = True
            content = str(output)
            content_length = len(content)
            content_excerpt = content[:800] + ("..." if len(content) > 800 else "")

        # Extract source type from tool name
        tool_name = entry.get("tool", "unknown")
        source_type = {
            "web_search": "web",
            "arxiv_search": "arxiv",
            "wikipedia_search": "wikipedia",
            "url_to_markdown": "web",
            "documentation_crawler": "web",
        }.get(tool_name, "unknown")

        indexed.append(
            {
                "call_id": entry.get("call_id"),
                "url": normalized_url,
                "source_type": source_type,
                "ok_status": ok_status,
                "content_excerpt": content_excerpt,
                "content_length": content_length,
            }
        )

    return indexed


def llm_select_cached_sources(
    llm: ChatLiteLLM,
    characteristic: str,
    topic: str,
    strategy_block: str,
    excluded_urls: List[str],
    cache_index: List[Dict],
    max_candidates: int = 15,
    synthetic_samples_generated: int = 0,
    total_samples_generated: int = 0,
    synthetic_budget: float = 0.2,
    total_samples_target: int = 0,
) -> tuple[str, List[str], str]:
    """Have LLM select cached sources for reuse.

    Returns:
        (decision, selected_urls, rationale)
    """
    # Filter cache to remove excluded URLs and limit candidates
    excluded_set = {normalize_url(url) for url in excluded_urls}
    available_candidates = [
        entry
        for entry in cache_index
        if entry["url"] not in excluded_set and entry["ok_status"]
    ][:max_candidates]

    if not available_candidates:
        return "stop", [], "No unused cached sources available."

    # Prepare candidate descriptions for LLM
    candidates_desc = []
    for i, entry in enumerate(available_candidates, 1):
        desc = f"{i}. URL: {entry['url']}\n   Source: {entry['source_type']}\n   Content: {entry['content_excerpt'][:200]}..."
        candidates_desc.append(desc)

    candidates_text = "\n\n".join(candidates_desc)

    # Calculate synthetic budget metrics
    max_synthetic_samples = (
        int(total_samples_target * synthetic_budget) if total_samples_target > 0 else 0
    )
    current_synthetic_pct = (
        (synthetic_samples_generated / total_samples_generated)
        if total_samples_generated > 0
        else 0.0
    )
    remaining_synthetic_budget = max_synthetic_samples - synthetic_samples_generated
    remaining_total_needed = total_samples_target - total_samples_generated

    system_prompt = f"""You are the supervisor selecting whether any unused cached sources can produce another high-quality sample for characteristic '{characteristic}' in topic '{topic}'.

Your task: Evaluate the available cached sources and decide if any could yield additional samples.

---
{strategy_block}
---

**Mission Context & Synthetic Budget:**
- **Current Status**: Generated {total_samples_generated} samples so far (Research: {total_samples_generated - synthetic_samples_generated}, Synthetic: {synthetic_samples_generated})
- **Mission Target**: {total_samples_target} total samples
- **Synthetic Budget**: {synthetic_budget:.0%} ({max_synthetic_samples} samples max)
- **Current Synthetic Rate**: {current_synthetic_pct:.1%}
- **Remaining Synthetic Budget**: {remaining_synthetic_budget} synthetic samples available
- **Remaining Work**: {remaining_total_needed} more samples needed to complete mission

**Available Cached Sources:**
{candidates_text}

**Decision Guidance:**
Consider both cached source quality AND synthetic budget balance (target: {synthetic_budget:.0%}, current: {current_synthetic_pct:.1%}):
- **Under target** ({current_synthetic_pct:.1%} < {synthetic_budget:.0%}): Prefer "stop" (synthetic) unless cached sources are excellent
- **Near target** ({current_synthetic_pct:.1%} ~= {synthetic_budget:.0%}): Choose based on cached source quality
- **Over target** ({current_synthetic_pct:.1%} > {synthetic_budget:.0%}): Prefer "reuse_cached" unless cached sources are very poor
- **Quality priority**: Don't sacrifice quality for budget--a good sample is better than hitting exact ratios

**Constraints:**
- Only select from the sources listed above
- Do NOT select any URL that was already used: {excluded_urls}
- Consider synthetic budget in your decision rationale

**Response Format (JSON only):**
{{
    "decision": "reuse_cached" or "stop",
    "selected_urls": ["list", "of", "selected", "urls"],
    "rationale": "Brief explanation considering both source quality and synthetic budget"
}}

Respond ONLY with the JSON object, no other text."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
        ]
    )

    agent = prompt | llm

    try:
        result = agent.invoke({})
        dethought = strip_reasoning_block(result.content)
        repaired_data = json_repair.loads(dethought)

        decision = repaired_data.get("decision", "stop")
        selected_urls = repaired_data.get("selected_urls", [])
        rationale = repaired_data.get("rationale", "LLM selection completed")

        # Validate decision
        if decision not in ["reuse_cached", "stop"]:
            decision = "stop"

        # Filter out any excluded URLs that might have slipped through
        if isinstance(selected_urls, list):
            selected_urls = [
                url for url in selected_urls if normalize_url(url) not in excluded_set
            ]
        else:
            selected_urls = []

        return decision, selected_urls, rationale

    except Exception as e:
        print(f"⚠️ llm_select_cached_sources: Parsing failed: {e}")
        return "stop", [], "LLM selection failed"
