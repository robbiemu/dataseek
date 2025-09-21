# Mission Runner

## Purpose
The Mission Runner orchestrates mission execution cycles over the LangGraph application, managing state transitions and persistence. It handles mission progression through multiple sample generation cycles, coordinates agent routing decisions, and ensures resumable operations via checkpointing. This component is responsible for tracking progress across different characteristics and topics, managing synthetic versus research data budgets, and maintaining mission-level state that persists across process restarts.

**Why this might change**: Modifications could be needed to support new mission types, additional progress tracking metrics (e.g., quality scores), different checkpointing strategies (like remote storage), or enhanced error handling for longer-running missions. Changes might also occur when introducing parallel mission execution or advanced retry mechanisms.

## Key Interfaces
- `seek.components.mission_runner.mission_runner.MissionRunner`: Main class that manages mission execution lifecycle
- `seek.components.mission_runner.state.DataSeekState`: Defines the complete graph state with all required fields for agent nodes
- `seek.components.mission_runner.mission_runner.MissionStateManager`: Handles persistent storage of mission state using SQLite

**When to extend**: New interfaces might be added when introducing mission branching logic, hierarchical mission structures, or external state synchronization.

## Dependencies
- `seek.components.search_graph.graph` (compiled app for agent execution)
- `langgraph.checkpoint.sqlite.SqliteSaver` (core state persistence)
- `seek.common.config` (configuration management)
- Internal state manager for mission-specific persistence

**Dependency changes**: Watch for updates to LangGraph checkpointing APIs or configuration formats that could require interface adaptations.

## Configuration
Reads effective settings via `seek.common.config` including:
- `recursion_per_sample`: Maximum agent steps per sample cycle
- Mission-specific parameters (target sizes, goals, output paths)
- Tool limits and provider settings
- Robots.txt compliance and search constraints

Configuration updates are needed when adding new mission parameters (e.g., quality thresholds) or supporting alternative persistence backends.

## Minimal Usage
- Construct `MissionRunner(checkpointer, app, mission_config, seek_config)` and call `run_mission()`.
- Supports optional resume from mission ID for interrupted executions.
