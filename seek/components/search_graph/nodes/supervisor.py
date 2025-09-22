import secrets
from typing import Any

import json_repair
import yaml
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel, Field

from seek.common.config import get_active_seek_config, get_prompt
from seek.components.mission_runner.state import DataSeekState

from .utils import (
    create_llm,
    get_characteristic_context,
    get_default_strategy_block,
    normalize_url,
    strip_reasoning_block,
)

# Minimum steps required for a complete research cycle
MIN_STEPS_FOR_SUCCESSFUL_RESEARCH = (
    10  # Ensures adequate steps for research completion or synthetic fallback
)


class SupervisorDecision(BaseModel):
    """Defines the structured decision output for the supervisor LLM."""

    next_agent: str = Field(
        ...,
        description="The name of the next agent to route to (e.g., 'research', 'fitness').",
    )
    new_task: dict | None = Field(
        None,
        description="Optional: A new task to assign if the supervisor decides to switch focus.",
    )


class CacheSelectionDecision(BaseModel):
    """Structured output for cache selection decision."""

    decision: str
    selected_urls: list[str] = []
    rationale: str | None = None


def _get_next_task_from_progress(
    progress: dict, exclude: list[tuple[str, str]] | None = None
) -> dict | None:
    """Finds the next uncompleted characteristic/topic pair."""
    if not progress:
        return None

    mission_name = list(progress.keys())[0]
    eligible_tasks = []
    for char, char_data in progress[mission_name].items():
        for topic, topic_data in char_data.get("topics", {}).items():
            if (
                char_data["collected"] < char_data["target"]
                and topic_data["collected"] < topic_data["target"]
                and (not exclude or (char, topic) not in exclude)
            ):
                eligible_tasks.append({"characteristic": char, "topic": topic})

    if not eligible_tasks:
        return None

    # Select next task using secure random selection
    return secrets.choice(eligible_tasks)


def supervisor_node(state: DataSeekState) -> dict:
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
        print("🔁 Supervisor: In cached-only mode. Checking quotas and routing deterministically.")

        current_task = state.get("current_task")
        progress = state.get("progress", {})

        # Verify quotas again
        if not needs_more_for_current_pair(progress, current_task, pending_increment=0):
            print("   📊 Supervisor: Quota satisfied for cached mode. Setting exhausted flag.")
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
                "synthetic_samples_generated": state.get("synthetic_samples_generated", 0),
                "research_samples_generated": state.get("research_samples_generated", 0),
                "synthetic_budget": state.get("synthetic_budget", 0.2),
                "fitness_report": None,
                "step_count": next_step_count,  # Increment step count for next iteration
            }

        # Check if we have any allowed URLs left
        allowed = set(state.get("allowed_url_whitelist", [])) - set(state.get("excluded_urls", []))
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
                "synthetic_samples_generated": state.get("synthetic_samples_generated", 0),
                "research_samples_generated": state.get("research_samples_generated", 0),
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
    next_task = current_task if current_task else _get_next_task_from_progress(progress)

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
    characteristic = next_task["characteristic"]
    topic = next_task.get("topic", "general domain")
    # Resolve mission config: prefer value from state, else try known file paths
    mission_config_data = state.get("mission_config")
    if not mission_config_data:
        for path in ("config/mission_config.yaml", "settings/mission_config.yaml"):
            try:
                with open(path) as f:
                    content = f.read()
                    if content.startswith("#"):
                        first_newline = content.find("\n")
                        if first_newline != -1:
                            content = content[first_newline + 1 :]
                    mission_config_data = yaml.safe_load(content)
                    break
            except Exception:
                mission_config_data = None
    try:
        characteristic_context = get_characteristic_context(next_task, mission_config_data or {})
    except Exception:
        characteristic_context = None

    if characteristic_context:
        print(f"   ✅ Supervisor: Using dynamic context for '{characteristic}' from mission plan.")
        strategy_block = f"**Strategic Focus for '{characteristic}':**\n{characteristic_context}"
    else:
        print(
            f"   ⚠️  Supervisor: Could not find dynamic context for '{characteristic}'. Using built-in fallback."
        )
        strategy_block = get_default_strategy_block(characteristic)

    decision_history = state.get("decision_history", [])
    tool_execution_failures = state.get("tool_execution_failures", 0)
    research_attempts = state.get("research_attempts", 0)
    consecutive_failures = state.get("consecutive_failures", 0)
    last_action_status = state.get("last_action_status", "success")
    last_action_agent = state.get("last_action_agent", "")

    # If the last action was a successful archive, evaluate for cached reuse.
    if last_action_agent == "archive":
        print("✅ Supervisor: Detected a successful archival. Evaluating for cached reuse.")

        # 1) Update exclusions
        research_findings = state.get("research_findings", [])
        used_urls = extract_used_urls_from_findings(research_findings)
        current_excluded = state.get("excluded_urls", [])
        # Normalize and deduplicate URLs
        all_excluded = list({normalize_url(url) for url in current_excluded + used_urls})

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
                "synthetic_samples_generated": state.get("synthetic_samples_generated", 0),
                "research_samples_generated": state.get("research_samples_generated", 0),
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
                "synthetic_samples_generated": state.get("synthetic_samples_generated", 0),
                "research_samples_generated": state.get("research_samples_generated", 0),
                "synthetic_budget": state.get("synthetic_budget", 0.2),
                "fitness_report": None,
                "step_count": next_step_count,  # Increment step count for next iteration
            }

        # 3) Build candidate cache index
        cache_index = index_research_cache(state.get("research_session_cache", []))
        # Filter to exclude already used URLs
        excluded_set = set(all_excluded)
        available_candidates = [entry for entry in cache_index if entry["url"] not in excluded_set]

        print(f"   🗂️  Supervisor: Found {len(available_candidates)} unused cached sources")

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
                "synthetic_samples_generated": state.get("synthetic_samples_generated", 0),
                "research_samples_generated": state.get("research_samples_generated", 0),
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
        total_samples_generated = synthetic_samples_generated + research_samples_generated
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

        print(f"   🎯 Supervisor: LLM decision='{decision}', selected={len(selected_urls)} URLs")
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
                    "synthetic_samples_generated": state.get("synthetic_samples_generated", 0),
                    "research_samples_generated": state.get("research_samples_generated", 0),
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
    fitness_report = state.get("fitness_report") if not has_new_research_report else None

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

            # Log the failure in task history for analysis
            task_history: list[tuple[str, str, str]] = state.get("task_history", [])
            current_task = state.get("current_task")
            if current_task:
                char = str(current_task.get("characteristic", ""))
                topic = str(current_task.get("topic", ""))
                reason = str(fitness_report.reason)
                task_history.append((char, topic, reason))

            # Provide failure context to supervisor for decision-making
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
                synthetic_samples_generated = state.get("synthetic_samples_generated", 0)

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
                    except Exception as e:
                        print(
                            f"   ⚠️  Supervisor: Failed to extract Source URL from line: {line!r} ({e})"
                        )
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
        findings_for_next_node: list[str] = []

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
                        print(f"✅ Supervisor: Resolved cache reference for call_id '{call_id}'.")
                        break
            except IndexError:
                print("⚠️ Supervisor: Malformed cache reference found.")
                pass

        findings_for_next_node.append(report_content)

        # Try to parse the URL from the report
        for line in report_content.split("\n"):
            if "**Source URL**:" in line:
                try:
                    source_url_parts = line.split("`")
                    source_url = source_url_parts[1] if len(source_url_parts) > 1 else None
                    if source_url:
                        # Strip surrounding quotes and whitespace
                        source_url = source_url.strip().strip('"').strip("'")
                except IndexError:
                    pass
                break

        # Verify provenance if a URL was found
        if source_url and research_cache:
            for evidence in research_cache:
                is_valid_evidence = (
                    isinstance(evidence, dict)
                    and "args" in evidence
                    and isinstance(evidence["args"], dict)
                )
                if is_valid_evidence:
                    args = evidence["args"]
                    found_url = args.get("url") or args.get("base_url") or args.get("start_url")
                    # Normalize and compare without surrounding quotes
                    if found_url:
                        n_found = normalize_url(found_url)
                        n_report = normalize_url(source_url)
                        if n_found == n_report:
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
            "research_findings": findings_for_next_node,
            # Clear any old fitness report when routing to fitness for a new report
            "fitness_report": None,
            "step_count": next_step_count,  # Increment step count for next iteration
        }

    research_detail = (
        "\n- `research`: Finds source documents from the web." if not research_is_off_limits else ""
    )

    base_prompt_template = get_prompt("supervisor", "base_prompt")
    base_prompt = base_prompt_template.format(research_detail=research_detail)

    characteristic = next_task.get("characteristic", "N/A")
    topic = next_task.get("topic", "N/A")
    current_task_str = (
        f"   - Find sources for the characteristic '{characteristic}' in the topic '{topic}'."
    )

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
        else "moderate" if consecutive_failures < 4 else "low"
    )

    # More integrated status summary
    progress_pct = (
        (total_samples_generated / total_samples_target) * 100 if total_samples_target > 0 else 0
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
    last_message_content = str(messages[-1].content) if messages else "No recent messages."

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

    mission_context_template = get_prompt("supervisor", "mission_context")
    mission_context = mission_context_template.format(
        current_task_str=current_task_str,
        mission_status=mission_status,
        decision_history=str(decision_history[-5:]),
        consecutive_failures=consecutive_failures,
        last_action_analysis=last_action_analysis,
        strategic_guidance=strategic_guidance,
    )

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", base_prompt),
            ("human", mission_context),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Prefer structured output when supported
    next_agent = "research"
    decision_obj: SupervisorDecision | None = None

    try:
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(SupervisorDecision)
            agent = agent_prompt | structured_llm
            maybe = agent.invoke({"messages": messages})
            if isinstance(maybe, SupervisorDecision):
                decision_obj = maybe
                next_agent = decision_obj.next_agent.lower()
        if decision_obj is None:
            # Fallback to previous parsing path
            agent = agent_prompt | llm
            raw_result = agent.invoke({"messages": messages})
            content_value = raw_result.content
            dethought = strip_reasoning_block(
                content_value if isinstance(content_value, str) else str(content_value)
            )
            repaired_data = json_repair.loads(dethought)
            decision_obj = SupervisorDecision.model_validate(repaired_data)
            next_agent = decision_obj.next_agent.lower()

        # Override research choice if it's off-limits due to insufficient steps
        if research_is_off_limits and next_agent == "research":
            print("   ⚠️ Supervisor: LLM chose 'research' despite limit. Overriding to 'end'.")
            # You could also add logic here to try 'synthetic' if it's cheaper and budget allows.
            next_agent = "end"

        # Handle task switching based on failure patterns
        if decision_obj and decision_obj.new_task:
            print(f"✅ Supervisor: Switching to new task: {decision_obj.new_task}")
            next_task = decision_obj.new_task
            # Recalculate strategy block for the new task
            characteristic = next_task["characteristic"]
            # Resolve mission config as above
            mission_config_data = state.get("mission_config")
            if not mission_config_data:
                for path in ("config/mission_config.yaml", "settings/mission_config.yaml"):
                    try:
                        with open(path) as f:
                            content = f.read()
                            if content.startswith("#"):
                                first_newline = content.find("\n")
                                if first_newline != -1:
                                    content = content[first_newline + 1 :]
                            mission_config_data = yaml.safe_load(content)
                            break
                    except Exception:
                        mission_config_data = None

            try:
                characteristic_context = get_characteristic_context(
                    next_task, mission_config_data or {}
                )
            except Exception:
                characteristic_context = None

            if characteristic_context:
                strategy_block = (
                    f"**Strategic Focus for '{characteristic}':**\n{characteristic_context}"
                )
            else:
                strategy_block = get_default_strategy_block(characteristic)

            # Clear messages for the researcher
            messages = []

            # Reset the session tool/domain blocklist for the new task
            session_tool_domain_blocklist = []
        # Prevent consecutive 'fitness' unless driven by a new research report path
        if decision_history and decision_history[-1] == "fitness" and next_agent == "fitness":
            print(
                "   ⚠️ Supervisor: Preventing consecutive 'fitness' decision. Overriding to 'research'."
            )
            next_agent = "research"

    except Exception as parse_error:
        print(f"⚠️ Supervisor: JSON parsing failed: {parse_error}")
        try:
            print(f"   Raw content: '{raw_result.content}'")  # type: ignore[name-defined]
        except Exception:
            pass
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


def get_node_config(node_name: str) -> dict | None:
    """Retrieve the configuration for a specific node by name."""
    # Load the seek config instead of using the old config.seek_agent
    seek_config = get_active_seek_config()

    if seek_config and "mission_plan" in seek_config and "nodes" in seek_config["mission_plan"]:
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
    progress: dict, current_task: dict | None, pending_increment: int = 0
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


def extract_used_urls_from_findings(research_findings: list) -> list[str]:
    """Parse Source URLs from Data Prospecting Reports in research findings."""
    urls: list[str] = []

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
                except Exception as e:
                    print(f"   ⚠️  URL parse error in research findings line: {line!r} ({e})")
                    continue

    return urls


def index_research_cache(cache: list[dict]) -> list[dict]:
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
    excluded_urls: list[str],
    cache_index: list[dict],
    max_candidates: int = 15,
    synthetic_samples_generated: int = 0,
    total_samples_generated: int = 0,
    synthetic_budget: float = 0.2,
    total_samples_target: int = 0,
) -> tuple[str, list[str], str]:
    """Have LLM select cached sources for reuse.

    Returns:
        (decision, selected_urls, rationale)
    """
    # Filter cache to remove excluded URLs and limit candidates
    excluded_set = {normalize_url(url) for url in excluded_urls}
    available_candidates = [
        entry for entry in cache_index if entry["url"] not in excluded_set and entry["ok_status"]
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

    tpl = get_prompt("supervisor_cache_selection", "base_prompt")
    system_prompt = tpl.format(
        characteristic=characteristic,
        topic=topic,
        strategy_block=strategy_block,
        total_samples_generated=total_samples_generated,
        research_samples_generated=total_samples_generated - synthetic_samples_generated,
        synthetic_samples_generated=synthetic_samples_generated,
        total_samples_target=total_samples_target,
        synthetic_budget=f"{synthetic_budget:.0%}",
        max_synthetic_samples=max_synthetic_samples,
        current_synthetic_pct=f"{current_synthetic_pct:.1%}",
        remaining_synthetic_budget=remaining_synthetic_budget,
        remaining_total_needed=remaining_total_needed,
        candidates_text=candidates_text,
        excluded_urls=list(excluded_urls),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
        ]
    )

    # Prefer structured output if supported
    try:
        decision = "stop"
        selected_urls: list[str] = []
        rationale = ""
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(CacheSelectionDecision)
            agent = prompt | structured_llm
            maybe = agent.invoke({})
            if isinstance(maybe, CacheSelectionDecision):
                decision = maybe.decision
                selected_urls = maybe.selected_urls or []
                rationale = maybe.rationale or ""
            else:
                # Fallback to unstructured parsing
                agent = prompt | llm
                result = agent.invoke({})
                content_val = result.content
                dethought = strip_reasoning_block(
                    content_val if isinstance(content_val, str) else str(content_val)
                )
                data_any = json_repair.loads(dethought)
                repaired_data: dict[str, Any] = data_any if isinstance(data_any, dict) else {}
                decision = repaired_data.get("decision", "stop")
                selected_urls = repaired_data.get("selected_urls", [])
                rationale = repaired_data.get("rationale", "LLM selection completed")
        else:
            agent = prompt | llm
            result = agent.invoke({})
            content_val = result.content
            dethought = strip_reasoning_block(
                content_val if isinstance(content_val, str) else str(content_val)
            )
            data_any = json_repair.loads(dethought)
            repaired_data: dict[str, Any] = data_any if isinstance(data_any, dict) else {}
            decision = repaired_data.get("decision", "stop")
            selected_urls = repaired_data.get("selected_urls", [])
            rationale = repaired_data.get("rationale", "LLM selection completed")

        # Validate decision
        if decision not in ["reuse_cached", "stop"]:
            decision = "stop"

        # Filter out any excluded URLs that might have slipped through
        if isinstance(selected_urls, list):
            selected_urls = [url for url in selected_urls if normalize_url(url) not in excluded_set]
        else:
            selected_urls = []

        return decision, selected_urls, rationale

    except Exception as e:
        print(f"⚠️ llm_select_cached_sources: Parsing failed: {e}")
        return "stop", [], "LLM selection failed"
