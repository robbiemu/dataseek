from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .agent_output_parser import (
    ErrorMessage,
    NewMessage,
    ProgressUpdate,
    RecursionStepUpdate,
    SyntheticSampleUpdate,
)

if TYPE_CHECKING:
    from .dataseek_tui import DataSeekTUI


async def _run_agent(app: DataSeekTUI) -> None:
    """Run the Data Seek Agent as a subprocess and parse its output."""
    app.stats.started_at = datetime.now()
    if app.agent_process_manager is None:
        app.debug_log("Agent process manager not initialized")
        return
    try:
        process = await app.agent_process_manager.start()
        if process.stdout is None:
            app.debug_log("Agent process has no stdout")
            return
        while True:
            line_bytes = await process.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8").strip()
            if line:
                # Log to file if log file is specified
                if app.log_handle:
                    try:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        app.log_handle.write(f"[{timestamp}] {line}\n")
                        app.log_handle.flush()
                    except Exception as e:
                        app.debug_log(f"Failed writing log line: {e}")

                # Parse the line and handle events
                events = list(app.agent_output_parser.parse_line(line))
                if events:
                    try:
                        with open(app._debug_log_path, "a") as f:
                            f.write(f"GENERATED {len(events)} EVENTS for line: {line[:100]}...\n")
                    except Exception as e:
                        app.debug_log(f"Failed writing debug events: {e}")
                for event in events:
                    _handle_agent_event(app, event)
    except Exception as e:
        app.conversation.add_message("error", f"Agent error: {str(e)}")
    finally:
        app.conversation.add_message("info", "Agent process completed")
        # Start a 5-second timer to auto-close the TUI
        app.set_timer(5.0, app.action_quit)


def _handle_agent_event(app: DataSeekTUI, event: Any) -> None:
    """Handle events from the agent output parser."""
    # Log event handling for debugging purposes
    app.debug_log(f"HANDLING EVENT: {type(event).__name__} - {event}")

    if isinstance(event, ProgressUpdate):
        new_stats = app.stats.copy()
        new_stats.completed = event.completed
        new_stats.target = event.target
        app.stats = new_stats

        # Update mission status when progress is made
        if event.completed > 0:
            app.mission_status = f"Generating... ({event.completed}/{event.target})"

        app.debug_log(f"PROGRESS UPDATE: {event.completed}/{event.target}")
    elif isinstance(event, SyntheticSampleUpdate):
        # Update synthetic sample count
        new_stats = app.stats.copy()
        new_stats.synthetic_completed = event.count
        app.stats = new_stats
        app.debug_log(f"SYNTHETIC SAMPLE UPDATE: {event.count}")
    elif isinstance(event, RecursionStepUpdate):
        # Update recursion step information
        new_stats = app.stats.copy()
        new_stats.current_recursion_step = event.current_step
        new_stats.total_recursion_steps = event.total_steps
        app.stats = new_stats
        app.debug_log(f"RECURSION STEP UPDATE: {event.current_step}/{event.total_steps}")
    elif isinstance(event, NewMessage):
        app.debug_log(f"NEW MESSAGE EVENT: {event.role} -> {event.content[:100]}...")
        app.conversation.add_message(event.role, event.content)

        # Update mission status based on message content
        # Check for Graph Router patterns
        if "Graph Router: Routing to" in event.content:
            route_match = re.search(r"Graph Router: Routing to (\w+)", event.content)
            if route_match:
                route_name = route_match.group(1)
                if route_name.lower() == "end":
                    app.mission_status = "Sample Completed"
                elif route_name.lower() == "archive":
                    app.mission_status = "Archiving Sample..."
                elif route_name.lower() == "fitness":
                    app.mission_status = "Checking Fitness..."
                elif route_name.lower() == "synthetic":
                    app.mission_status = "Generating Synthetic..."
                else:
                    app.mission_status = f"Routing to {route_name}..."

        # Check for node execution patterns like "🔍 RESEARCH NODE" or "🎨 SYNTHETIC NODE"
        elif ("NODE" in event.content and "🔍" in event.content) or (
            "SYNTHETIC NODE" in event.content and "🎨" in event.content
        ):
            if "🎨 SYNTHETIC NODE" in event.content:
                app.mission_status = "Generating Synthetic Content..."
            else:
                node_match = re.search(r"🔍\s+(\w+)\s+NODE", event.content)
                if node_match:
                    node_name = node_match.group(1)
                    app.mission_status = f"Working on {node_name}..."

        # Check for agent starting work (move from Initializing)
        elif (
            "▶ Iteration" in event.content
            or "🔧 Tool calls:" in event.content
            or "📊 CONTEXT:" in event.content
        ) and app.mission_status == "Initializing...":
            app.mission_status = "Working..."
            # Extract current recursion step if available
            iteration_match = re.search(r"▶ Iteration (\d+)/(\d+)", event.content)
            if iteration_match:
                current_step = int(iteration_match.group(1))
                total_steps = int(iteration_match.group(2))
                new_stats = app.stats.copy()
                new_stats.current_recursion_step = current_step
                new_stats.total_recursion_steps = total_steps
                app.stats = new_stats

        # Check for routing to END (fallback pattern)
        elif "Routing to END" in event.content or "Decided on 'end'" in event.content:
            app.mission_status = "Sample Completed"

        # Check for sample archival and add to recent samples
        elif "sample #" in event.content.lower() and "archived" in event.content.lower():
            sample_match = re.search(r"#(\d+)", event.content)
            if sample_match:
                sample_num = int(sample_match.group(1))

                # Extract completion percentage if available
                pct_match = re.search(r"\((\d+\.?\d*)% complete\)", event.content)
                completion_pct = pct_match.group(1) if pct_match else "?"

                # Extract source type (provenance) if available
                source_match = re.search(r"Source: (\w+)", event.content)
                source_type = source_match.group(1) if source_match else "unknown"

                # Try to extract sample excerpt from the most recent content
                sample_excerpt = app._extract_recent_sample_excerpt()

                if sample_excerpt:
                    description = (
                        f"{sample_excerpt} ({completion_pct}% complete) - Source: {source_type}"
                    )
                else:
                    description = f"Archived ({completion_pct}% complete) - Source: {source_type}"

                app.progress_panel.add_sample(sample_num, description)
    elif isinstance(event, ErrorMessage):
        app.debug_log(f"ERROR MESSAGE EVENT: {event.message[:100]}...")
        app.conversation.add_message("error", event.message)
        # Keep a copy for the error modal even if conversation scrolls
        try:
            app.error_messages.append(event.message)
            # Cap error list growth
            if len(app.error_messages) > 200:
                app.error_messages = app.error_messages[-200:]
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            app.debug_log(f"Could not append to error_messages due to: {e}")
        new_stats = app.stats.copy()
        new_stats.errors += 1
        app.stats = new_stats
