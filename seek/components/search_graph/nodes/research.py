import logging
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from seek.common.config import get_active_seek_config, get_prompt
from seek.components.mission_runner.state import DataSeekState
from seek.components.tool_manager.clients import HTTP_CLIENT, RATE_MANAGER
from seek.components.tool_manager.tools import get_tools_for_role
from seek.components.tool_manager.utils import (
    _run_async_safely,
    _safe_request_get,
    _truncate_response_for_role,
)

from .supervisor import index_research_cache
from .utils import create_llm, get_default_strategy_block, normalize_url

logger = logging.getLogger(__name__)

# Constants
DEFAULT_PREFETCH_LIMIT = 3
DEFAULT_MIN_RESULTS = 3
DEFAULT_MAX_RETRIES = 2
DEFAULT_RESULT_MULTIPLIER = 3
URL_VALIDATION_TIMEOUT = 10
RATE_LIMIT_RPS = 2.0


def research_node(state: "DataSeekState") -> dict:
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
        print(f"   🚫 Session blocklist contains {len(session_tool_domain_blocklist)} entries:")
        for blocked_tool, domain in session_tool_domain_blocklist:
            print(f"      - {blocked_tool} blocked for domain {domain}")

    # --- Extract the latest human question ---
    user_question = None
    for msg in reversed(state.get("messages", [])):
        # LangChain HumanMessage or any object with type == "human"
        if (
            getattr(msg, "content", None)
            and getattr(msg, "type", None) == "human"
            or "HumanMessage" in str(type(msg))
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
    session_cache = list(state.get("research_session_cache", []))  # make a copy we can append to
    print(f"   Session cache (pre-run): {len(session_cache)} items")

    # --- Tools & Mission Context ---
    # Load the seek config to get the use_robots setting
    seek_config = get_active_seek_config()
    use_robots = seek_config.get("use_robots", True)

    all_research_tools = get_tools_for_role("research")
    print(f"   Tools available (global): {[t.name for t in all_research_tools]}")

    # Honor mission_config.tool_configs roles: only enable tools explicitly listed for 'research'.
    # Also auto-include corresponding *_get_content tools when their search counterpart is listed.
    mission_cfg = state.get("mission_config", {}) or {}
    tool_cfgs = mission_cfg.get("tool_configs", {}) if isinstance(mission_cfg, dict) else {}
    allowed_tool_names: set[str] = set()
    has_tool_configs = isinstance(tool_cfgs, dict) and len(tool_cfgs) > 0
    try:
        for name, cfg in (tool_cfgs or {}).items():
            if not isinstance(cfg, dict):
                continue
            roles = cfg.get("roles", [])
            if isinstance(roles, list) and any(str(r).lower() == "research" for r in roles):
                allowed_tool_names.add(str(name))
        # Implicit pairs for content retrieval when search tools are allowed
        if "arxiv_search" in allowed_tool_names:
            allowed_tool_names.add("arxiv_get_content")
        if "wikipedia_search" in allowed_tool_names:
            allowed_tool_names.add("wikipedia_get_content")
    except Exception as e:
        print(f"   ⚠️  Research: Could not parse mission tool_configs for filtering ({e})")
        allowed_tool_names = set()

    if allowed_tool_names:
        filtered_tools = [
            t for t in all_research_tools if getattr(t, "name", "") in allowed_tool_names
        ]
        print(
            f"   🔧 Filtering tools per mission_config: allowed={sorted(allowed_tool_names)}, effective={[t.name for t in filtered_tools]}"
        )
        all_research_tools = filtered_tools
    elif has_tool_configs:
        # tool_configs present but none enabled for research -> disable tools
        print(
            "   🔧 Research: mission_config.tool_configs present but none enabled for 'research'; disabling tools"
        )
        all_research_tools = []
    else:
        # No explicit tool list provided for research; retain defaults
        names = [getattr(t, "name", "") for t in all_research_tools]
        print(f"   🔧 No mission-config tool filter for research; using defaults: {names}")

    current_task = state.get("current_task")
    strategy_block = state.get("strategy_block", "")

    if not current_task:
        # A current_task is essential for the agent to function.
        # If it's missing, it indicates a problem in the mission's state management.
        # It's better to fail fast than to proceed with incorrect or default values.
        raise ValueError(
            "FATAL: No current_task found in state. The agent cannot proceed without a task."
        )

    # If we have a current_task, we can safely extract its properties.
    characteristic = current_task.get("characteristic")
    topic = current_task.get("topic", "general domain")  # A default for topic is acceptable.

    if not characteristic:
        # A characteristic is also essential.
        raise ValueError(
            f"FATAL: The current task is missing a 'characteristic'. Task: {current_task}"
        )

    print(f"   🎯 Task selected: characteristic={characteristic} topic={topic}")

    if not strategy_block:
        print(
            f"   ⚠️  No strategy block found in state. Using built-in fallback for '{characteristic}'."
        )
        strategy_block = get_default_strategy_block(characteristic)

    # --- CACHED-ONLY MODE CHECK ---
    cached_only = state.get("cached_only_mode") or state.get("no_search_tools")
    allowed_urls = list(
        set(state.get("allowed_url_whitelist", [])) - set(state.get("excluded_urls", []))
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

        print(f"   🗂️  Research: Found {len(allowed_cache_entries)} cached entries for allowed URLs")

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
    # Base role prompt (shared)
    base_tpl = get_prompt("research", "base_prompt")
    try:
        base_prompt = base_tpl.format(
            characteristic=characteristic, topic=topic, strategy_block=strategy_block
        )
    except Exception:
        # Tolerate base prompts without placeholders
        base_prompt = base_tpl

    if cached_only:
        # Cached-only mode prompt template
        tpl = get_prompt("research", "cached_only_prompt")
        allowed_urls_list = "\n".join(f"- {url}" for url in allowed_urls)
        specific_prompt = tpl.format(
            characteristic=characteristic,
            topic=topic,
            strategy_block=strategy_block,
            allowed_urls_list=allowed_urls_list,
            cache_context=cache_context,
        )
    else:
        # Normal mode prompt template
        tpl = get_prompt("research", "normal_prompt")
        specific_prompt = tpl.format(
            characteristic=characteristic,
            topic=topic,
            strategy_block=strategy_block,
        )

    system_prompt = f"{base_prompt}\n\n{specific_prompt}" if base_prompt else specific_prompt

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
        max_iterations = int(seek_config["nodes"]["research"].get("max_iterations", max_iterations))

    print(f"   🔄 ReAct loop starting (max_iterations={max_iterations})")

    final_report_msg = None
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"   ▶ Iteration {iteration}/{max_iterations}")

        # Context tracking for research progress
        history_char_count = sum(len(str(m.content)) for m in react_messages)
        print(f"   📊 CONTEXT: {len(react_messages)} messages, ~{history_char_count} chars")

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

        # Optional detailed debugging for final iteration
        # if iteration == max_iterations:
        #     print("      ❗ FINAL PROMPT: The following system prompt is being sent to the LLM for its last chance.")
        #     print("      " + "-" * 20)
        #     print(f"      {system_prompt_for_iteration[-500:]}")  # Print last 500 chars
        #     print("      " + "-" * 20)

        # Bind tools as scoped this iteration
        llm_with_tools = llm.bind_tools(current_tools, tool_choice="auto") if current_tools else llm
        print(
            f"      Tools this iteration: {[t.name for t in current_tools] if current_tools else '[]'}"
        )

        # Build runnable and invoke
        react_agent = (
            prompt_template.partial(system_prompt_for_iteration=system_prompt_for_iteration)
            | llm_with_tools
        )

        try:
            result = react_agent.invoke({"messages": react_messages})

            # Optional detailed debugging of LLM responses
            # print("      📝 --- START RAW LLM RESPONSE ---")
            # print(f"{getattr(result, 'content', '[NO CONTENT]').strip()}")
            # print("      📝 ---  END RAW LLM RESPONSE  ---")

            # Handle empty responses on final iteration with retry mechanism
            raw_content = getattr(result, "content", "")
            if iteration == max_iterations and (not raw_content or not raw_content.strip()):
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
            if getattr(result, "content", "") and "# Data Prospecting Report" in result.content:
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
                        tool_call_name: str | None = (
                            tool_call.get("name")
                            if isinstance(tool_call, dict)
                            else getattr(tool_call, "name", None)
                        )
                        tool_call_args = (
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
                            (t for t in current_tools if t.name == tool_call_name), None
                        )
                        if not matching_tool:
                            print(
                                f"         ⚠️ Tool '{tool_call_name}' not found in current scope; skipping."
                            )
                            continue

                        print(
                            f"         ▶ Executing {tool_call_name} with args: {str(tool_call_args)[:200]} ..."
                        )

                        # Normalize tool name for downstream logging/handling
                        tool_name: str = tool_call_name or matching_tool.name

                        tool_result = matching_tool.invoke(tool_call_args)

                        # Log tool execution results
                        if isinstance(tool_result, dict):
                            status = tool_result.get("status")
                            if status == "error":
                                error_detail = tool_result.get("error", "")
                                print(f"         📊 Tool result status: {status} {error_detail}")
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
                                num_items = len(results_data) if results_data else "None/Negative"
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
                            print(f"         🔍 Validating {tool_name} results...")
                            validation_result = _validate_search_results(
                                tool_result["results"],
                                tool_name,
                                tool_call_args,
                                matching_tool,
                                session_tool_domain_blocklist,  # Pass the blocklist
                            )

                            tool_result["results"] = validation_result["results"]
                            tool_result["validation_info"] = validation_result
                            print(
                                f"         ✅ {tool_name} results validated ({len(validation_result['results'])} results)"
                            )

                            # Log retry information if performed
                            if validation_result.get("retry_performed"):
                                if validation_result.get("retry_successful"):
                                    print(f"         🔄 {tool_name}: Auto-retry successful")
                                else:
                                    print(
                                        f"         🔄 {tool_name}: Auto-retry attempted but unsuccessful"
                                    )
                            else:
                                # Handle error or genuinely empty results
                                if tool_result.get("status") == "error":
                                    error_msg = tool_result.get("error", "Unknown error")
                                    print(f"         ❌ {tool_name} tool failed: {error_msg}")
                                else:
                                    print(f"         ⚠️  No results returned by {tool_name} tool")

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
                            if tool_result.get("status") == "ok" and tool_result.get("results"):
                                print(f"         🔍 Validating {tool_name} results...")
                                validation_result = _validate_search_results(
                                    tool_result["results"],
                                    tool_name,
                                    tool_call_args,
                                    matching_tool,
                                    session_tool_domain_blocklist,  # Pass the blocklist
                                )

                                tool_result["results"] = validation_result["results"]
                                tool_result["validation_info"] = validation_result
                                print(
                                    f"         ✅ {tool_name} results validated ({len(validation_result['results'])} results)"
                                )

                                # Log retry information if performed
                                if validation_result.get("retry_performed"):
                                    if validation_result.get("retry_successful"):
                                        print(f"         🔄 {tool_name}: Auto-retry successful")
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
                                "tool": tool_call_name,
                                "args": tool_call_args,  # original arguments
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
                                content=f"Tool '{tool_call_name}' failed: {tool_error}",
                                tool_call_id=(
                                    tool_id if "tool_id" in locals() else f"error_{iteration}_{idx}"
                                ),
                            )
                        )

                        # === UPDATE BLOCKLIST FOR TOOL/DOMAIN FAILURES ===
                        # Extract domain from URL args and add to blocklist
                        if tool_name and tool_call_args and isinstance(tool_call_args, dict):
                            url = (
                                tool_call_args.get("url")
                                or tool_call_args.get("base_url")
                                or tool_call_args.get("start_url")
                            )
                            if url:
                                try:
                                    from urllib.parse import urlparse

                                    domain = urlparse(url).netloc
                                    if domain:
                                        # Check if this combination is already blocked
                                        block_entry = (tool_name, domain)
                                        if block_entry not in session_tool_domain_blocklist:
                                            session_tool_domain_blocklist.append(block_entry)
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
    elif not (str(final_report_msg.content).startswith("# Data Prospecting Report")):
        final_report_msg.content = "# Data Prospecting Report\n\n" + str(final_report_msg.content)

    # --- Return only the final submission + full evidence cache ---
    print(f"   🧾 Returning final submission + evidence (cache size: {len(session_cache)})")
    return {
        "messages": [final_report_msg],  # Append the final Data Prospecting Report ONLY
        "research_session_cache": session_cache,  # Full, updated evidence cache (no clearing)
        "session_tool_domain_blocklist": session_tool_domain_blocklist,  # Updated blocklist
    }


def find_url_field(results: list[dict[str, Any]]) -> str | None:
    """Find the URL field name in search results."""
    if not results:
        return None

    first = results[0]
    if not isinstance(first, dict):
        return None

    for url_field in ["url", "link"]:
        if url_field in first:
            return url_field

    return None


class SearchResultsValidator:
    """Handles validation and filtering of search results."""

    def __init__(self, tool_name: str, config: Any | None = None) -> None:
        self.tool_name = tool_name
        self.config = config or get_active_seek_config()
        self._load_tool_config()

    def _load_tool_config(self) -> None:
        """Load configuration settings for the tool."""
        # Set defaults
        self.prefetch_enabled = False
        self.prefetch_limit = DEFAULT_PREFETCH_LIMIT
        self.validate_urls = True
        self.retry_on_failure = True
        self.max_retries = DEFAULT_MAX_RETRIES

        # Load structured seek config to get tool configuration
        seek_config = get_active_seek_config()
        tool_config = seek_config.get_tool_config(self.tool_name)
        if tool_config:
            self.prefetch_enabled = tool_config.pre_fetch_pages
            self.prefetch_limit = tool_config.pre_fetch_limit
            self.validate_urls = tool_config.validate_urls
            self.retry_on_failure = tool_config.retry_on_failure
            self.max_retries = tool_config.max_retries

    def validate_search_results(
        self,
        results: list[dict[str, Any]],
        tool_args: dict[str, Any] | None = None,
        matching_tool: Any = None,
        session_tool_domain_blocklist: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Validate search results by filtering blocked domains and checking URL accessibility.

        Args:
            results: List of search results to validate
            tool_args: Arguments used for the original tool call
            matching_tool: Tool instance for retry attempts
            session_tool_domain_blocklist: List of blocked (tool_name, domain) tuples

        Returns:
            Dictionary containing validated results and metadata
        """
        if not self.prefetch_enabled or not self.validate_urls:
            return self._create_response(results, validation_performed=False)

        # Filter blocked domains
        filtered_results, blocked_count = self._filter_blocked_domains(
            results, session_tool_domain_blocklist
        )

        # Expand search if insufficient results after filtering
        if (
            self.tool_name == "web_search"
            and len(filtered_results) < DEFAULT_MIN_RESULTS
            and results
            and matching_tool
            and tool_args
        ):
            filtered_results = self._expand_search_results(
                filtered_results,
                tool_args,
                matching_tool,
                session_tool_domain_blocklist,
            )

        # Validate URLs
        validation_batch = filtered_results[: self.prefetch_limit]
        logger.info(f"Validating {len(validation_batch)} results for {self.tool_name}")

        validated_results = self._validate_url_batch(validation_batch)
        accessible_results = [r for r in validated_results if r.get("status") == "accessible"]

        # Retry if no accessible results found
        if (
            self.retry_on_failure
            and not accessible_results
            and results
            and matching_tool
            and tool_args
        ):
            retry_results = self._perform_retry(
                validated_results,
                tool_args,
                matching_tool,
            )
            if retry_results:
                return retry_results

        filtered_count = len([r for r in validated_results if r.get("status") == "inaccessible"])

        return self._create_response(
            validated_results,
            validation_performed=True,
            filtered_count=filtered_count,
            original_count=len(validation_batch),
        )

    def _filter_blocked_domains(
        self,
        results: list[dict[str, Any]],
        blocklist: list[tuple[str, str]] | None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Filter out results from blocked domains."""
        if not results or not blocklist:
            return results, 0

        # Only filter for tools that can have domains blocked
        url_field = find_url_field(results)
        if not url_field:
            return results, 0

        filtered_results = []
        blocked_count = 0

        for result in results:
            if not isinstance(result, dict) or url_field not in result:
                filtered_results.append(result)
                continue

            try:
                domain = urlparse(result[url_field]).netloc
                if (self.tool_name, domain) in blocklist:
                    blocked_count += 1
                    logger.info(f"Filtered blocked result from {domain}")
                else:
                    filtered_results.append(result)
            except Exception:
                # Include result if domain parsing fails
                filtered_results.append(result)

        if blocked_count > 0:
            logger.info(f"Filtered {blocked_count} blocked results for {self.tool_name}")

        return filtered_results, blocked_count

    def _expand_search_results(
        self,
        filtered_results: list[dict[str, Any]],
        tool_args: dict[str, Any],
        matching_tool: Any,
        blocklist: list[tuple[str, str]] | None,
    ) -> list[dict[str, Any]]:
        """Expand search results when insufficient results remain after filtering."""
        logger.info(f"Expanding search: only {len(filtered_results)} results after filtering")

        expanded_queries = self._generate_expanded_queries(tool_args.get("query", ""))
        expanded_results: list[dict[str, Any]] = []

        for i, expanded_query in enumerate(expanded_queries):
            if len(expanded_results) >= DEFAULT_MIN_RESULTS:
                break

            logger.info(f"Trying expanded query {i + 1}: {expanded_query}")

            try:
                expanded_tool_args = tool_args.copy()
                expanded_tool_args["query"] = expanded_query

                expanded_tool_result = matching_tool.invoke(expanded_tool_args)

                if isinstance(expanded_tool_result, dict) and expanded_tool_result.get("results"):
                    new_results, _ = self._filter_blocked_domains(
                        expanded_tool_result["results"], blocklist
                    )
                    expanded_results.extend(new_results)
                    logger.info(f"Expanded query yielded {len(new_results)} new results")

            except Exception as e:
                logger.warning(f"Expanded query failed: {e}")

        # Combine and deduplicate results
        combined_results = filtered_results + expanded_results
        unique_results = self._deduplicate_results(combined_results)

        # Limit to reasonable number
        final_results = unique_results[:10]
        logger.info(f"Final result set: {len(final_results)} results")

        return final_results

    def _generate_expanded_queries(self, original_query: str) -> list[str]:
        """Generate expanded search queries for retry attempts."""
        return [
            f"{original_query} site:*",
            f"{original_query} *",
            f"related:{original_query}",
        ]

    def _deduplicate_results(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove duplicate results based on URL."""
        seen_urls: set[str] = set()
        unique_results = []

        for result in results:
            if isinstance(result, dict):
                url_field = find_url_field([result])
                if url_field and url_field in result:
                    url = result[url_field]
                    if url not in seen_urls:
                        seen_urls.add(url)
                        unique_results.append(result)
                else:
                    unique_results.append(result)
            else:
                unique_results.append(result)

        return unique_results

    def _validate_url_batch(
        self,
        results_batch: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate a batch of search results by checking URL accessibility."""
        validated_results = []

        for result in results_batch:
            if not isinstance(result, dict):
                validated_results.append({"content": result, "status": "accessible"})
                continue

            url = self._extract_url_from_result(result)

            if url:
                status, error = self._check_url_accessibility(url)
                result_copy = result.copy()
                result_copy["status"] = status
                if error:
                    result_copy["error"] = error
                validated_results.append(result_copy)
            else:
                # Handle results without URLs - assume accessible
                result_copy = result.copy()
                result_copy["status"] = "accessible"
                validated_results.append(result_copy)

        return validated_results

    def _check_url_accessibility(self, url: str) -> tuple[str, str | None]:
        """Check if a URL is accessible."""
        try:
            response = self._safe_request_head(url)
            if response.status_code == 200:
                return "accessible", None
            else:
                logger.warning(f"URL {url} inaccessible (status {response.status_code})")
                return "inaccessible", f"HTTP {response.status_code}"

        except Exception:
            # Fallback to GET request
            try:
                response = _safe_request_get(url, timeout_s=URL_VALIDATION_TIMEOUT, max_retries=1)
                if response.status_code == 200:
                    return "accessible", None
                else:
                    logger.warning(f"URL {url} inaccessible (status {response.status_code})")
                    return "inaccessible", f"HTTP {response.status_code}"

            except Exception as e:
                logger.warning(f"URL {url} inaccessible ({type(e).__name__})")
                return "inaccessible", f"{type(e).__name__}: {str(e)}"

    def _safe_request_head(self, url: str) -> httpx.Response:
        """Perform a rate-limited HEAD request."""
        domain = urlparse(url).netloc

        for attempt in range(self.max_retries + 1):
            try:

                async def do_request() -> httpx.Response:
                    async with RATE_MANAGER.acquire(
                        f"domain:{domain}", requests_per_second=RATE_LIMIT_RPS
                    ):
                        return await HTTP_CLIENT.head(
                            url,
                            timeout=URL_VALIDATION_TIMEOUT,
                            headers={"User-Agent": "DataSeek/1.0"},
                        )

                return _run_async_safely(do_request())

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 405:
                    raise  # Method not allowed - fallback to GET
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2**attempt))
                    continue
                raise

            except Exception:
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2**attempt))
                    continue
                raise

        # This should never be reached, but mypy requires it
        raise RuntimeError("Failed to perform HEAD request after all retries")

    def _perform_retry(
        self,
        failed_results: list[dict[str, Any]],
        tool_args: dict[str, Any],
        matching_tool: Any,
    ) -> dict[str, Any] | None:
        """Perform retry with expanded parameters when all initial results fail."""
        logger.info("All initial results inaccessible, attempting retry with expanded results")

        bad_urls: set[str] = {
            r.get("url")  # type: ignore[misc]
            for r in failed_results
            if r.get("status") == "inaccessible" and r.get("url") is not None
        }

        for attempt in range(self.max_retries):
            try:
                expanded_args = self._create_expanded_args(tool_args)
                logger.info(f"Retry attempt {attempt + 1}/{self.max_retries} with: {expanded_args}")

                expanded_result = matching_tool.invoke(expanded_args)

                if isinstance(expanded_result, dict) and expanded_result.get("status") == "ok":
                    expanded_results = expanded_result.get("results", [])
                    fresh_results = self._filter_fresh_results(expanded_results, bad_urls)

                    if fresh_results:
                        validation_limit = min(self.prefetch_limit * 2, len(fresh_results))
                        validated_fresh = self._validate_url_batch(fresh_results[:validation_limit])

                        accessible_fresh = [
                            r for r in validated_fresh if r.get("status") == "accessible"
                        ]

                        if accessible_fresh:
                            logger.info(
                                f"Retry successful: Found {len(accessible_fresh)} accessible results"
                            )
                            return self._create_response(
                                validated_fresh,
                                validation_performed=True,
                                filtered_count=len(validated_fresh) - len(accessible_fresh),
                                original_count=len(failed_results),
                                retry_performed=True,
                                retry_successful=True,
                                bad_urls_excluded=list(bad_urls),
                            )

                        # Add newly discovered bad URLs
                        bad_urls.update(
                            r.get("url")  # type: ignore[misc]
                            for r in validated_fresh
                            if r.get("status") == "inaccessible" and r.get("url") is not None
                        )

            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")

        logger.error("All retry attempts failed")
        return None

    def _create_expanded_args(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        """Create expanded arguments for retry attempts."""
        expanded_args = tool_args.copy()

        # Increase result count if possible
        for count_field in ["num_results", "count", "limit"]:
            if count_field in expanded_args:
                expanded_args[count_field] *= DEFAULT_RESULT_MULTIPLIER
                break

        return expanded_args

    def _filter_fresh_results(
        self,
        results: list[dict[str, Any]],
        bad_urls: set[str],
    ) -> list[dict[str, Any]]:
        """Filter out results with URLs we already know are bad."""
        fresh_results = []

        for result in results:
            if isinstance(result, dict):
                url = self._extract_url_from_result(result)
                if url and url not in bad_urls or not url:
                    fresh_results.append(result)
            else:
                fresh_results.append(result)

        return fresh_results

    def _extract_url_from_result(self, result: dict[str, Any]) -> str | None:
        """Extract URL from a search result."""
        url_field = find_url_field([result])
        return result[url_field] if url_field else None

    def _create_response(
        self,
        results: list[dict[str, Any]],
        validation_performed: bool = False,
        needs_retry: bool = False,
        filtered_count: int = 0,
        original_count: int | None = None,
        retry_performed: bool = False,
        retry_successful: bool = False,
        bad_urls_excluded: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a standardized response dictionary."""
        response = {
            "results": results,
            "validation_performed": validation_performed,
            "needs_retry": needs_retry,
            "filtered_count": filtered_count,
            "retry_performed": retry_performed,
        }

        if original_count is not None:
            response["original_count"] = original_count

        if retry_performed:
            response["retry_successful"] = retry_successful

        if bad_urls_excluded:
            response["bad_urls_excluded"] = bad_urls_excluded

        return response


def _validate_search_results(
    results: list[dict[str, Any]],
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
    matching_tool: Any = None,
    session_tool_domain_blocklist: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Legacy function wrapper for backward compatibility.

    Args:
        results: List of search results to validate
        tool_name: Name of the search tool
        tool_args: Arguments passed to the search tool
        matching_tool: Tool instance for retry operations
        session_tool_domain_blocklist: List of blocked (tool_name, domain) tuples

    Returns:
        Dictionary containing validated results and metadata
    """
    validator = SearchResultsValidator(tool_name)
    return validator.validate_search_results(
        results, tool_args, matching_tool, session_tool_domain_blocklist
    )
