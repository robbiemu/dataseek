# Search Graph

## Purpose
Defines and builds the multi-agent LangGraph workflow that powers the Data Seek system. Establishes the supervisor architecture with specialized agent nodes (research, archive, fitness, synthetic) that collaborate on data generation tasks. Manages agent routing, tool binding, and state transitions according to configured mission plans. Provides the core execution engine that coordinates complex agent interactions.

**Why this might change**: Graph modifications could introduce new agent types (e.g., validation, refinement agents), alternative routing strategies (e.g., parallel execution paths), or enhanced supervisor logic. Changes are triggered by new research workflows, quality assurance requirements, or performance optimizations through parallel processing.

## Key Interfaces
- `seek.components.search_graph.graph.build_graph()`: Factory function for compiled LangGraph workflows
- Agent node functions:
  - `supervisor_node()`: Decision-making coordinator for agent routing
  - `research_node()`: Information discovery and evidence collection
  - `fitness_node()`: Content quality evaluation and structured reporting
  - `archive_node()`: Data persistence and audit trail management
  - `synthetic_node()`: Artificial content generation
- Node helper utilities: `create_llm()`, `create_agent_runnable()`

**When to extend**: New nodes are created for specialized tasks, and routing edges are modified to implement new workflow patterns or decision criteria.

## Dependencies
- `langgraph.graph.StateGraph`: Core graph construction and compilation
- `langgraph.prebuilt.ToolNode`: Tool integration for agent nodes
- `seek.components.tool_manager.tools`: Agent tool bindings
- `seek.common.config`: Configuration-driven LLM and tool setup
- State management: `seek.components.mission_runner.state.DataSeekState`

**Dependency changes**: Graph construction may need updates when migrating to new LangGraph versions or integrating alternative agent frameworks.

## Configuration
Uses node-specific configurations from `seek.common.config` and mission plans:
- Agent model settings (temperature, max_tokens per node type)
- Tool availability per role
- Recursion limits and graph depth constraints
- Role-specific parameters (e.g., research max_iterations)

Configuration updates are required when adding new node types, adjusting decision thresholds, or supporting configurable routing rules.

## Minimal Usage
- `from seek.components.search_graph.graph import build_graph` then compile with a `SqliteSaver` checkpointer.
- Graph automatically configures agents based on active seek and mission configurations.
