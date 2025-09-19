# graph.py
"""
Builds the LangGraph application graph for the Data Seek agent.
"""

from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from seek.components.mission_runner.state import DataSeekState
from seek.components.tool_manager.tools import get_tools_for_role

from .nodes import (
    archive_node,
    fitness_node,
    research_node,
    supervisor_node,
    synthetic_node,
)


def build_graph(checkpointer: SqliteSaver, seek_config: dict[str, Any]) -> Any:
    """
    Builds and compiles the multi-agent graph with a supervisor.

    Args:
        checkpointer: A LangGraph checkpointer instance for persisting state.
        seek_config: The seek configuration dictionary.

    Returns:
        A compiled LangGraph app.
    """
    workflow = StateGraph(DataSeekState)

    # --- Define Agent Nodes ---
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("archive", archive_node)
    workflow.add_node("fitness", fitness_node)
    workflow.add_node("synthetic", synthetic_node)  # Add synthetic node

    # --- Define Role-Specific Tool Nodes ---
    use_robots = seek_config.get("use_robots", True)

    research_tools = get_tools_for_role("research", use_robots=use_robots)
    archive_tools = get_tools_for_role("archive", use_robots=use_robots)

    workflow.add_node("research_tools", ToolNode(research_tools))
    workflow.add_node("archive_tools", ToolNode(archive_tools))

    # --- Wire the Graph ---
    workflow.set_entry_point("supervisor")

    # The supervisor decides which agent to run next.
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
            "synthetic": "synthetic",  # Add synthetic route
            "end": END,
        },
    )

    # Agent nodes route to their tool nodes, which then route back to the supervisor.
    workflow.add_edge("research", "research_tools")
    workflow.add_edge("research_tools", "supervisor")

    workflow.add_edge("archive", "archive_tools")
    workflow.add_edge("archive_tools", "supervisor")

    # The fitness agent has no tools, so it routes directly back to the supervisor.
    workflow.add_edge("fitness", "supervisor")

    # The synthetic agent routes directly to archive (bypassing fitness check)
    # as per the "Unhappy Path" design: synthetic data is assumed to be correct by design
    workflow.add_edge("synthetic", "archive")

    # Compile the graph with the provided checkpointer
    return workflow.compile(checkpointer=checkpointer)
