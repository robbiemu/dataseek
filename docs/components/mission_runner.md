# Mission Runner

- Purpose: Orchestrates mission execution cycles over the LangGraph app, provides progress reporting and mission control.
- Key interfaces: `seek.components.mission_runner.mission_runner.MissionRunner`, `seek.components.mission_runner.state.DataSeekState`.
- Dependencies: `seek.components.search_graph.graph` (compiled app), `langgraph.checkpoint.sqlite.SqliteSaver`.
- Configuration: Reads effective settings via `seek.common.config` (e.g., recursion limits via CLI or config).
- Minimal usage:
  - Construct `MissionRunner(checkpointer, app, mission_config, seek_config)` and call `run_mission()`.
