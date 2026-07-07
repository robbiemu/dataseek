# graph.py
"""
Builds the LangGraph application graph for the Data Seek agent.
"""

import asyncio
from functools import partial
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from seek.components.mission_runner.state import DataSeekState
from seek.components.tool_manager.tool_manager import ToolManager

from .nodes import (
    archive_node,
    fitness_node,
    research_node,
    supervisor_node,
    synthetic_node,
)


def build_graph(checkpointer: SqliteSaver, mission_config: dict[str, Any]) -> Any:
    """
    Builds and compiles the multi-agent graph with a supervisor.

    Args:
        checkpointer: A LangGraph checkpointer instance for persisting state.
        seek_config: The seek configuration dictionary.
        mission_config: The mission configuration dictionary.

    Returns:
        A compiled LangGraph app.
    """
    workflow = StateGraph(DataSeekState)

    # --- Prepare toolsets (needed before node registration) ---
    tool_manager = ToolManager()
    toolsets = asyncio.run(tool_manager.get_toolsets_for_mission(mission_config))

    # Thread the configured, setup()-initialized fitness toolset into fitness_node
    # so the model and ToolNode share the same instances. Without this, the model
    # would be bound against freshly instantiated unconfigured plugins while
    # ToolNode executes the configured ones — a dangerous split.
    fitness_toolset = toolsets.get("fitness", [])

    # --- Define Agent Nodes ---
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("archive", archive_node)
    # Always pass the prepared fitness toolset (even when empty) so fitness_node
    # knows tools were explicitly resolved (None = "not supplied", [] = "supplied
    # but none survived config validation"). Without this, an empty toolset falls
    # back to get_tools_for_role which creates fresh unconfigured plugins.
    workflow.add_node("fitness", partial(fitness_node, configured_tools=fitness_toolset))
    workflow.add_node("synthetic", synthetic_node)  # Handles synthetic data generation

    research_tools_node = ToolNode(toolsets.get("research", []))
    workflow.add_node("research_tools", research_tools_node)

    archive_tools_node = ToolNode(toolsets.get("archive", []))
    workflow.add_node("archive_tools", archive_tools_node)

    # Fitness tools (from the same configured toolset) are only present for
    # missions that map plugins to "fitness". When absent, stock behavior is
    # preserved: fitness flows directly back to the supervisor.
    has_fitness_tools = bool(fitness_toolset)
    if has_fitness_tools:
        fitness_tools_node = ToolNode(fitness_toolset)
        workflow.add_node("fitness_tools", fitness_tools_node)

    # --- Wire the Graph ---
    workflow.set_entry_point("supervisor")

    # Routes to the appropriate agent based on supervisor decision
    def supervisor_router(state: DataSeekState) -> str:
        """Route based on supervisor decision with debugging."""
        # Track recursion step (for TUI display only)
        step_count = state.get("step_count", 0)
        max_recursion_steps = state.get("max_recursion_steps", 25)  # Default value

        # Output recursion step information for TUI
        print(f"🔄 Supervisor: Recursion step {step_count}/{max_recursion_steps}")

        next_agent = state.get("next_agent")
        decision_history = state.get("decision_history", [])
        print(f"🔀 Graph Router: next_agent = '{next_agent}' (type: {type(next_agent)})")
        print(f"🔀 Graph Router: decision_history = {decision_history[-5:]}")
        print(f"🔀 Graph Router: messages count = {len(state.get('messages', []))}")

        if next_agent == "end":
            print("🏁 Graph Router: Routing to END")
            return "end"
        elif next_agent == "research":
            print("🔍 Graph Router: Routing to research")
            return "research"
        elif next_agent == "archive":
            print("📁 Graph Router: Routing to archive")
            return "archive"
        elif next_agent == "fitness":
            print("💪 Graph Router: Routing to fitness")
            return "fitness"
        elif next_agent == "synthetic":
            print("🎨 Graph Router: Routing to synthetic")
            return "synthetic"
        else:
            print(f"⚠️  Graph Router: Unknown next_agent '{next_agent}', defaulting to END")
            return "end"

    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "research": "research",
            "archive": "archive",
            "fitness": "fitness",
            "synthetic": "synthetic",  # Route for synthetic data generation
            "end": END,
        },
    )

    # Connect agents to their tools and back to supervisor
    workflow.add_edge("research", "research_tools")
    workflow.add_edge("research_tools", "supervisor")

    workflow.add_edge("archive", "archive_tools")
    workflow.add_edge("archive_tools", "supervisor")

    # When the fitness node has tools, it forms a ReAct loop:
    # fitness → fitness_tools → fitness (model consumes tool results and either
    # calls more tools or produces its final report). The transition from
    # fitness is conditional: if the model emitted tool_calls, route to
    # fitness_tools for execution; otherwise route to supervisor (the model
    # returned its final report). Without tools, fitness flows straight back.
    if has_fitness_tools:
        workflow.add_edge("fitness_tools", "fitness")

        def fitness_router(state: DataSeekState) -> str:
            """Route to fitness_tools if the model emitted tool calls; else supervisor."""
            last = state["messages"][-1]
            return "fitness_tools" if getattr(last, "tool_calls", None) else "supervisor"

        workflow.add_conditional_edges(
            "fitness",
            fitness_router,
            {"fitness_tools": "fitness_tools", "supervisor": "supervisor"},
        )
    else:
        workflow.add_edge("fitness", "supervisor")

    # Synthetic data flows directly to archive
    workflow.add_edge("synthetic", "archive")

    # Compile the graph with the provided checkpointer
    return workflow.compile(checkpointer=checkpointer)
