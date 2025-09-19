# Search Graph

- Purpose: Builds the LangGraph workflow and defines agent nodes (supervisor, research, archive, fitness, synthetic).
- Key interfaces: `seek.components.search_graph.graph.build_graph`, nodes in `seek.components.search_graph.nodes`.
- Dependencies: Tools from `seek.components.tool_manager.tools`, state from `seek.components.mission_runner.state`.
- Configuration: Uses `seek.common.config` via node helpers to set models, temperatures, and limits.
- Minimal usage:
  - `from seek.components.search_graph.graph import build_graph` then compile with a `SqliteSaver` checkpointer.
