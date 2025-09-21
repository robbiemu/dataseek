# TUI (Terminal User Interface)

## Purpose
Provides an interactive textual interface for monitoring and controlling Data Seek missions. Displays real-time progress, conversation logs, mission status, and statistics in a terminal environment. Enables users to select missions, view detailed logs, and observe agent decision-making processes. Supports theme switching and responsive layouts for different terminal sizes.

**Why this might change**: Interface expansions might include new visualization modes (e.g., charts for progress analytics), interaction features (e.g., agent result inspection, manual overrides), or accessibility improvements. Changes could also address new mission types with specialized display requirements or integration with external monitoring systems.

## Key Interfaces
- `seek.components.tui.dataseek_tui.DataSeekTUI`: Main application class managing interface lifecycle
- `seek.components.tui.components.conversation_panel.ConversationPanel`: Real-time agent message display
- `seek.components.tui.components.mission_panel.MissionPanel`: Mission configuration and status visualization
- `seek.components.tui.components.progress_panel.ProgressPanel`: Sample generation progress tracking
- `seek.components.tui.components.stats_header.StatsHeader`: Aggregate statistics display
- `seek.components.tui.agent_output_parser.AgentOutputParser`: Parses agent logs for UI events

**When to extend**: New components are added for specialized views (e.g., performance metrics), and existing panels are modified for enhanced data presentation or user interaction.

## Dependencies
- `textual`: Core TUI framework for interface management
- `seek.common.config`: Active configuration loading
- `seek.components.tui.agent_process_manager.AgentProcessManager`: Subprocess agent execution
- Theme detection: `darkdetect` (optional, falls back gracefully)
- Configuration containers: `typer` for CLI argument handling

**Dependency changes**: Updates may be required when adopting newer Textual versions or integrating alternative TUI frameworks.

## Configuration
Loads and applies settings from `seek.common.config.load_seek_config()`:
- Mission selection and plan paths
- Robots.txt compliance settings
- Recursion limits and agent parameters
- Theme preferences (auto-detected if `darkdetect` available)
- Logging configurations

Configuration changes occur when adding new display options, interaction modes, or integrating with external configuration sources.

## Minimal Usage
- `python -m seek.components.tui.dataseek_tui` to launch, or import `DataSeekTUI` and call `.run()`.
- Automatically detects available missions and provides interactive selection if none specified at startup.
