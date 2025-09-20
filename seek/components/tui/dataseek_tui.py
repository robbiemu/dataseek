#!/usr/bin/env python3
"""
Data Seek Agent TUI - A modern terminal interface for sample generation.
"""
from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from typing import Any

import typer
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header

from seek.common.config import load_seek_config, set_active_seek_config
from seek.components.tui.agent_handler import _run_agent
from seek.components.tui.agent_output_parser import AgentOutputParser
from seek.components.tui.agent_process_manager import AgentProcessManager
from seek.components.tui.components.conversation_panel import ConversationPanel
from seek.components.tui.components.mission_panel import MissionPanel
from seek.components.tui.components.mission_selector import MissionSelector
from seek.components.tui.components.progress_panel import ProgressPanel
from seek.components.tui.components.stats_header import StatsHeader
from seek.components.tui.state import TUIState
from seek.components.tui.theme_manager import (
    DARKDETECT_AVAILABLE,
    action_toggle_theme,
    apply_footer_styles,
    apply_theme_styling,
    detect_system_theme,
    sync_with_system_theme,
)
from seek.components.tui.utils import get_mission_details_from_file


class DataSeekTUI(App):
    """Main TUI application."""

    CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.tcss")

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("e", "show_error_modal", "Errors"),
        Binding("t", "toggle_theme", "Theme"),
        Binding("f10", "debug_scrollbars", "Debug Scrollbars"),
        Binding("f12", "show_dom", "DOM"),
    ]

    def __init__(
        self,
        mission_plan_path: str = "config/mission_config.yaml",
        log_file: str | None = None,
        debug: bool = False,
        seek_config_path: str | None = None,
        use_robots: bool = True,
        mission_name: str | None = None,
    ):
        super().__init__()
        self.mission_plan_path = mission_plan_path
        self.log_file = log_file
        self.debug_enabled = debug
        self.seek_config_path = seek_config_path
        self.use_robots = use_robots
        self.mission_name = mission_name
        self._debug_log_path = os.path.join(tempfile.gettempdir(), "tui_debug.log")
        if debug:
            print(f"Writing debug log to {self._debug_log_path}")

        self.tui_state = TUIState()
        self.agent_process_manager: AgentProcessManager | None = None
        self.agent_output_parser = AgentOutputParser()
        self.log_handle: io.TextIOWrapper | None = None

        # System theme detection
        self.system_theme_sync_enabled = DARKDETECT_AVAILABLE
        self.last_detected_theme: str | None = None
        # When True, a user manually toggled theme; don't auto-revert
        self.user_theme_override = False
        self.dark: bool = False  # Initialize dark theme to False
        self.show_tree = False

    def debug_log(self, message: str) -> None:
        """Log debug message if debug mode is enabled."""
        if self.debug_enabled:
            try:
                with open(self._debug_log_path, "a") as f:
                    f.write(f"{message}\n")
            except Exception as _e:
                # Ignore debug logging errors
                pass  # nosec B110 # - debug logging failure is non-fatal

    def _load_mission_config(self, mission_name: str) -> dict:
        """Load mission-specific configuration from the mission config file."""
        try:
            import yaml

            with open(self.mission_plan_path) as f:
                content = f.read()
                # Remove comment header if it exists
                if content.startswith("#"):
                    first_newline = content.find("\n")
                    if first_newline != -1:
                        content = content[first_newline + 1 :]

                mission_plan = yaml.safe_load(content)
                if not mission_plan or "missions" not in mission_plan:
                    return {}

                # Find the specific mission
                for mission in mission_plan.get("missions", []):
                    if mission.get("name") == mission_name:
                        return mission

                return {}
        except Exception as e:
            self.debug_log(f"Failed to load mission config for {mission_name}: {e}")
            return {}

    def compose(self) -> ComposeResult:
        # Initialize with system theme if available, otherwise default to dark
        if self.system_theme_sync_enabled:
            system_theme = detect_system_theme(self)
            if system_theme is not None:
                self.dark = system_theme == "dark"
                self.last_detected_theme = system_theme
                self.debug_log(f"Initialized with system theme: {system_theme}")
            else:
                self.dark = True  # Default to dark if detection fails
                self.debug_log("System theme detection failed, defaulting to dark")
        else:
            self.dark = True  # Default to dark if darkdetect not available
            self.debug_log("darkdetect not available, defaulting to dark theme")

        apply_footer_styles(self)
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        """Start the mission selector when the app mounts."""
        mission_details = get_mission_details_from_file(self.mission_plan_path)
        if not mission_details or not mission_details["mission_names"]:
            self.exit(message=f"❌ No missions found in {self.mission_plan_path}")
            return

        if self.mission_name:
            if self.mission_name not in mission_details["mission_names"]:
                self.exit(
                    message=f"❌ Mission '{self.mission_name}' not found in {self.mission_plan_path}"
                )
                return
            self.on_mission_selected(self.mission_name)
        else:
            self.push_screen(
                MissionSelector(mission_details["mission_names"], self.dark),
                self.on_mission_selected,
            )
        apply_footer_styles(self)

    def on_mission_selected(self, mission_name: str | None) -> None:
        """Called when a mission is selected from the modal."""
        if not mission_name:
            self.exit(message="❌ No mission selected")
            return

        # Open log file if specified
        if self.log_file:
            try:
                self.log_handle = open(self.log_file, "a")
                self.log_handle.write(
                    f"\n=== Data Seek TUI Session Started at {datetime.now().isoformat()} ===\n"
                )
                self.log_handle.flush()
            except Exception as _e:
                # Log error but continue
                self.debug_log(f"Failed flushing log header: {_e}")

        mission_details = get_mission_details_from_file(self.mission_plan_path)
        if mission_details is None:
            raise ValueError(f"Failed to load mission details from {self.mission_plan_path}")
        total_samples_target = mission_details["mission_targets"].get(mission_name, 1200)
        new_stats = self.tui_state.stats
        new_stats.target = total_samples_target
        self.tui_state.stats = new_stats

        seek_config = load_seek_config(self.seek_config_path, use_robots=self.use_robots)
        # Set active config for TUI context as well (non-subprocess usage)
        try:
            set_active_seek_config(seek_config)
        except Exception as e:
            self.debug_log(f"Failed to set active seek config: {e}")
        recursion_limit = seek_config.get("recursion_per_sample", 27)

        # Get synthetic budget and target size from mission config (not seek config)
        mission_config = self._load_mission_config(mission_name)
        synthetic_budget = mission_config.get("synthetic_budget", 1.0)
        target_size = mission_config.get("target_size", 1)

        # Debug log the mission config values
        self.debug_log(f"Mission config loaded for '{mission_name}':")
        self.debug_log(f"  synthetic_budget: {synthetic_budget}")
        self.debug_log(f"  target_size: {target_size}")
        self.debug_log(f"  total_samples_target: {total_samples_target}")

        self.agent_process_manager = AgentProcessManager(
            mission_name=mission_name,
            recursion_limit=recursion_limit,
            use_robots=self.use_robots,
            seek_config_path=self.seek_config_path,
            mission_plan_path=self.mission_plan_path,
        )

        self.stats_header = StatsHeader(self.tui_state)
        # Initialize stats with synthetic budget and target size
        new_stats = self.tui_state.stats
        new_stats.synthetic_budget = synthetic_budget
        new_stats.target_size = target_size
        new_stats.total_recursion_steps = recursion_limit
        self.tui_state.stats = new_stats

        self.progress_panel = ProgressPanel(self.tui_state)
        self.mission_panel = MissionPanel(
            self.tui_state,
            self.mission_plan_path,
            mission_name,
            total_samples_target,
            seek_config.to_dict(),
        )
        self.conversation = ConversationPanel(debug=self.debug_enabled)

        # Mount the main UI components
        self.debug_log("MOUNTING COMPONENTS:")
        self.debug_log(f"  - stats_header: {self.stats_header}")
        self.debug_log(f"  - progress_panel: {self.progress_panel}")
        self.debug_log(f"  - mission_panel: {self.mission_panel}")
        self.debug_log(f"  - conversation: {self.conversation}")

        self.mount(
            Container(
                self.stats_header,
                Container(
                    self.progress_panel,
                    self.mission_panel,
                    id="main-container",
                ),
                self.conversation,
            )
        )

        # Apply initial theme styling to all components
        apply_theme_styling(self)

        # Also apply footer styling with a small delay to ensure footer keys are rendered
        def _delayed_footer_styling() -> None:
            self.debug_log("Applying delayed footer key styling after UI load")
            apply_footer_styles(self)

        self.set_timer(0.5, _delayed_footer_styling)

        # Start periodic system theme checking
        if self.system_theme_sync_enabled:

            def _check_system_theme() -> None:
                sync_with_system_theme(self)
                # Schedule next check in 3 seconds
                self.set_timer(3.0, _check_system_theme)

            # Start the first check after 1 second to let everything initialize
            self.set_timer(1.0, _check_system_theme)
            self.debug_log("Started periodic system theme checking (every 3 seconds)")

        # Add initial messages to the conversation
        if self.log_file:
            self.conversation.add_message("info", f"Logging to: {self.log_file}")
        self.conversation.add_message("info", "Starting Data Seek Agent...")
        self.conversation.add_message("info", f"Mission: {mission_name}")
        self.conversation.add_message("info", f"Target: {total_samples_target} samples")

        self.run_worker(_run_agent(self), name="agent")

    def action_toggle_theme(self) -> None:
        """An action to toggle between dark and light themes (manual override)."""
        action_toggle_theme(self)

    def action_quit(self) -> Any:
        """Quit the application."""
        if self.agent_process_manager:
            self.agent_process_manager.terminate()

        # Close log file if open
        if self.log_handle:
            try:
                self.log_handle.write(
                    f"\n=== Data Seek TUI Session Ended at {datetime.now().isoformat()} ===\n"
                )
                self.log_handle.close()
            except Exception as e:
                self.debug_log(f"Failed closing log handle: {e}")

        return self.exit()

    def action_show_dom(self) -> None:
        """Show the DOM tree in the console."""
        if hasattr(self, "log"):
            self.log(self.tree)  # This will print the tree to the dev console
        else:
            print(self.tree)  # Fallback to print

    def action_show_error_modal(self) -> None:
        """Show the error display modal."""
        from seek.components.tui.components.error_modal import ErrorModal

        error_messages = [
            msg["content"] for msg in self.conversation.messages if msg["role"] == "error"
        ]
        if not error_messages:
            error_messages = ["No errors recorded yet."]
        self.push_screen(ErrorModal(error_messages=error_messages))

    def debug_all_scrollbars(self) -> None:
        """Find and inspect actual scrollbar objects."""
        print("=== SCROLLBAR DEBUG START ===")

        try:
            conversation = self.query_one("#conversation")

            # Access the actual scrollbar objects
            v_scrollbar = conversation.vertical_scrollbar
            h_scrollbar = conversation.horizontal_scrollbar

            print(f"Vertical scrollbar: {v_scrollbar}")
            print(f"Vertical scrollbar type: {type(v_scrollbar).__name__}")
            print(f"Vertical scrollbar classes: {getattr(v_scrollbar, 'classes', 'none')}")

            print(f"Horizontal scrollbar: {h_scrollbar}")
            print(f"Horizontal scrollbar type: {type(h_scrollbar).__name__}")
            print(f"Horizontal scrollbar classes: {getattr(h_scrollbar, 'classes', 'none')}")

            # Try to access scrollbar styles
            if v_scrollbar:
                print(f"V-scrollbar styles: {getattr(v_scrollbar, 'styles', 'none')}")
                print(
                    f"V-scrollbar background: {getattr(v_scrollbar.styles, 'background', 'none')}"
                )

        except Exception as e:
            print(f"Debug failed: {e}")
            import traceback

            traceback.print_exc()

        print("=== SCROLLBAR DEBUG END ===")

    def _auto_close_after_completion(self) -> None:
        """Auto-close the TUI after agent process completion with a countdown."""
        self.conversation.add_message(
            "info", "TUI will close in 5 seconds... (Press 'q' to exit immediately)"
        )
        # Set another timer to actually close
        self.set_timer(5.0, self.action_quit)

    def _extract_recent_sample_excerpt(self) -> str | None:
        """Extract a content excerpt from the most recently archived sample.

        Scans the entire conversation history to find the most recent
        '## Retrieved Content (Markdown)' or '## Generated Content (Synthetic)' section
        and extracts content between that marker and the end boundary.

        Supports both boundary patterns:
        - '📝 ---  END RAW LLM RESPONSE  ---'
        - '📝 ---  END LLM RESPONSE  ---'
        """
        try:
            # Scan the entire conversation history (not just recent messages)
            messages = self.conversation.messages
            self.debug_log(
                f"EXCERPT EXTRACTION: Scanning {len(messages)} total messages for content boundaries"
            )

            # Find the most recent content marker (either retrieved or synthetic)
            content_start = -1
            content_message_idx = -1
            full_content = ""
            content_type = "unknown"

            # Search backwards through all messages to find the latest content
            for i in range(len(messages) - 1, -1, -1):
                message = messages[i]
                content = message.get("content", "")

                # Check for retrieved content first
                if "## Retrieved Content (Markdown)" in content:
                    content_start = content.find("## Retrieved Content (Markdown)")
                    content_message_idx = i
                    full_content = content
                    content_type = "retrieved"
                    self.debug_log(
                        f"FOUND Retrieved Content marker in message {i} at position {content_start}"
                    )
                    break
                # Also check for synthetic content
                elif "## Generated Content (Synthetic)" in content:
                    content_start = content.find("## Generated Content (Synthetic)")
                    content_message_idx = i
                    full_content = content
                    content_type = "synthetic"
                    self.debug_log(
                        f"FOUND Generated Content marker in message {i} at position {content_start}"
                    )
                    break

            if content_start == -1:
                self.debug_log("NO Content marker found in conversation history")
                return None

            # Extract content from the marker onwards (handle both types)
            if content_type == "retrieved":
                marker_text = "## Retrieved Content (Markdown)"
            else:  # synthetic
                marker_text = "## Generated Content (Synthetic)"

            content_after_marker = full_content[content_start + len(marker_text) :]

            # Look for the end boundary in the same message or subsequent messages
            # Support both "END RAW LLM RESPONSE" and "END LLM RESPONSE" patterns
            end_boundary_raw = "📝 ---  END RAW LLM RESPONSE  ---"
            end_boundary_simple = "📝 ---  END LLM RESPONSE  ---"
            sample_content = content_after_marker
            end_boundary = None

            # Check if either end boundary is in the same message
            if end_boundary_raw in content_after_marker:
                end_boundary = end_boundary_raw
                sample_content = content_after_marker.split(end_boundary)[0]
                self.debug_log(
                    f"FOUND RAW end boundary in same message, extracted {len(sample_content)} chars"
                )
            elif end_boundary_simple in content_after_marker:
                end_boundary = end_boundary_simple
                sample_content = content_after_marker.split(end_boundary)[0]
                self.debug_log(
                    f"FOUND simple end boundary in same message, extracted {len(sample_content)} chars"
                )
            elif end_boundary is None:
                # Search subsequent messages for either end boundary
                self.debug_log("End boundary not in same message, searching subsequent messages")
                accumulated_content = [content_after_marker]

                for i in range(content_message_idx + 1, len(messages)):
                    next_message = messages[i]
                    next_content = next_message.get("content", "")

                    if end_boundary_raw in next_content:
                        end_boundary = end_boundary_raw
                        content_before_boundary = next_content.split(end_boundary)[0]
                        accumulated_content.append(content_before_boundary)
                        self.debug_log(
                            f"FOUND RAW end boundary in message {i}, total accumulated content"
                        )
                        break
                    elif end_boundary_simple in next_content:
                        end_boundary = end_boundary_simple
                        content_before_boundary = next_content.split(end_boundary)[0]
                        accumulated_content.append(content_before_boundary)
                        self.debug_log(
                            f"FOUND simple end boundary in message {i}, total accumulated content"
                        )
                        break
                    else:
                        # Add the entire message content to our sample
                        accumulated_content.append(next_content)
                        self.debug_log(f"Adding full message {i} to sample content")

                sample_content = "\n".join(accumulated_content)

            # Clean up and extract meaningful excerpt from the sample content
            sample_content = sample_content.strip()
            self.debug_log(f"RAW SAMPLE CONTENT length: {len(sample_content)} chars")

            if not sample_content:
                self.debug_log("Empty sample content after extraction")
                return None

            # Handle cache reference tokens
            if "[CACHE_REFERENCE:" in sample_content:
                self.debug_log("Sample contains cache reference")
                # Try to extract meaningful content around the cache reference
                lines = sample_content.split("\n")
                meaningful_lines = []

                for line in lines:
                    line = line.strip()
                    if (
                        line
                        and not line.startswith("[CACHE_REFERENCE")
                        and not line.startswith("`")
                        and not line.startswith("#")
                        and len(line) > 20
                    ):
                        meaningful_lines.append(line)
                        if len(meaningful_lines) >= 2:
                            break

                if meaningful_lines:
                    excerpt = " ".join(meaningful_lines)
                    if len(excerpt) > 120:
                        excerpt = excerpt[:120] + "..."
                    self.debug_log(f"EXTRACTED cache excerpt: {excerpt[:50]}...")
                    return excerpt
                else:
                    return f"{content_type.title()} sample (from cache)"

            # Extract meaningful content lines
            lines = sample_content.split("\n")
            clean_lines = []

            for line in lines:
                line = line.strip()
                # Skip markdown formatting, empty lines, timestamps, and other noise
                if (
                    line
                    and not line.startswith("`")
                    and not line.startswith("#")
                    and not line.startswith("[")
                    and not line.startswith("---")
                    and not line.startswith("**")
                    and not line.startswith("*")
                    and not line.startswith("|")  # Skip table formatting
                    and len(line) > 25
                ):  # Only meaningful content
                    clean_lines.append(line)
                    if len(clean_lines) >= 3:  # Get first 3 meaningful lines
                        break

            if clean_lines:
                excerpt = " ".join(clean_lines)
                # Truncate to reasonable display length
                if len(excerpt) > 120:
                    excerpt = excerpt[:120] + "..."
                self.debug_log(f"FINAL EXCERPT: {excerpt[:50]}...")
                return excerpt
            else:
                self.debug_log("No clean meaningful lines found in sample content")
                return "Sample content (formatting only)"

        except Exception as e:
            self.debug_log(f"Error extracting sample excerpt: {e}")
            import traceback

            self.debug_log(f"Exception traceback: {traceback.format_exc()}")
            return None


cli_app = typer.Typer()


@cli_app.command()
def generate(
    mission_name: str | None = typer.Option(
        None, "--mission", "-m", help="The name of the mission to run."
    ),
    mission_plan: str = typer.Option(
        "config/mission_config.yaml",
        "--mission-plan",
        help="Path to the mission plan YAML file.",
    ),
    log: str | None = typer.Option(None, "--log", "-l", help="Path to write a detailed log file."),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug logging to a temporary file."
    ),
    config: str | None = typer.Option(
        None, "--config", "-c", help="Path to an alternative seek_config.yaml file."
    ),
    robots: bool = typer.Option(
        True, "--robots/--no-robots", help="Enable or disable robots.txt compliance."
    ),
    tui: bool = typer.Option(False, "--tui", "-t", help="Run with Terminal User Interface"),
    recursion_limit: int = typer.Option(
        30, "--recursion-limit", help="Maximum recursion steps for the agent."
    ),
) -> None:
    """Start the Data Seek Agent TUI for sample generation."""
    # Validate mission plan path, not the mission name
    if mission_plan and not os.path.exists(mission_plan):
        typer.echo(f"❌ Mission plan file not found: {mission_plan}", err=True)
        raise typer.Exit(1)

    # Handle --no-robots flag
    use_robots = robots

    app = DataSeekTUI(
        mission_plan_path=mission_plan,
        log_file=log,
        debug=debug,
        seek_config_path=config,
        use_robots=use_robots,
        mission_name=mission_name,
    )
    app.run()


def main() -> None:
    cli_app()


if __name__ == "__main__":
    main()
