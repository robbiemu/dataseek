# state.py
"""
Defines the overall state for the Data Seek agent graph.
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from .models import FitnessReport


class DataSeekState(TypedDict):
    strategy_block: str
    """
    Represents the state of the Data Seek graph.

    Attributes:
        messages: The conversation history, managed by `add_messages`.
        run_id: The unique identifier for the current agent run (thread_id).
        progress: A dictionary tracking the mission progress.
        current_task: The current task being processed by the agent.
        research_findings: A list of dictionaries containing research results.
        research_session_cache: A list of raw results accumulated during current research session.
        pedigree_path: The file path for the audit trail, loaded from config.
        next_agent: The name of the next agent to be invoked by the supervisor.
        decision_history: A list of recent supervisor decisions to prevent loops.
        tool_execution_failures: A counter for consecutive tool execution failures.
        research_attempts: A counter for research attempts on current question.
        samples_generated: A counter for tracking completed samples.
        total_samples_target: The total number of samples to generate.
        current_mission: The current mission being processed.
        synthetic_samples_generated: A counter for synthetic samples created.
        research_samples_generated: A counter for research-based samples created.
        consecutive_failures: A counter for consecutive agent failures.
        last_action_status: The status of the last agent action (success/failure).
        last_action_agent: The name of the last agent that took action.
        synthetic_budget: The maximum allowed percentage of synthetic samples (0.0-1.0).
        fitness_report: A structured report from the FitnessAgent evaluating a source document.
        current_sample_provenance: The provenance type of the current sample ('researched', 'synthetic', 'unknown').
        task_history: A list of (characteristic, topic, failure_reason) tuples.
        excluded_urls: A list of URLs that have already been used for samples.
        cached_only_mode: Flag indicating if the agent should only use cached data.
        no_search_tools: Flag indicating if search tools are disabled.
        allowed_url_whitelist: List of URLs that are allowed to be used in cached-only mode.
        cached_exhausted: Flag indicating if all cached data has been exhausted.
        next_cycle_cached_reuse: Plan for next cycle's cached reuse behavior.
        step_count: Counter for tracking recursion steps in the agent graph.
        max_recursion_steps: Maximum allowed recursion steps for the agent graph.
        session_tool_domain_blocklist: List of (tool_name, domain) tuples that have failed in the current session.
    """
    messages: Annotated[list, add_messages]
    run_id: str
    progress: dict
    current_task: dict | None
    research_findings: list[dict]
    research_session_cache: list[dict]
    pedigree_path: str
    next_agent: str
    decision_history: list[str]
    tool_execution_failures: int
    research_attempts: int
    samples_generated: int
    total_samples_target: int
    current_mission: str
    synthetic_samples_generated: int
    research_samples_generated: int
    consecutive_failures: int
    last_action_status: str
    last_action_agent: str
    synthetic_budget: float
    fitness_report: FitnessReport | None
    current_sample_provenance: str | None
    task_history: list[tuple[str, str, str]]
    excluded_urls: list[str]
    cached_only_mode: bool
    no_search_tools: bool
    allowed_url_whitelist: list[str]
    cached_exhausted: bool
    next_cycle_cached_reuse: dict[str, Any] | None
    step_count: int
    max_recursion_steps: int
    session_tool_domain_blocklist: list[tuple[str, str]]
