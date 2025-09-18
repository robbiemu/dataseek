#!/usr/bin/env python3
"""
Data Seek Agent TUI - A modern terminal interface for sample generation.
"""

import os
from datetime import datetime
from typing import Optional
import io

try:
    import darkdetect

    DARKDETECT_AVAILABLE = True
except ImportError:
    DARKDETECT_AVAILABLE = False

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Header, Footer
import typer

from seek.tui.components.stats_header import StatsHeader, GenerationStats
from seek.tui.components.progress_panel import ProgressPanel
from seek.tui.components.mission_panel import MissionPanel
from seek.tui.components.conversation_panel import ConversationPanel
from seek.tui.components.mission_selector import MissionSelector
from seek.tui.agent_process_manager import AgentProcessManager
from seek.tui.agent_output_parser import (
    AgentOutputParser,
    ProgressUpdate,
    NewMessage,
    ErrorMessage,
    SyntheticSampleUpdate,
    RecursionStepUpdate,
)
from ..config import load_seek_config, set_active_seek_config
from ..seek_utils import get_mission_details_from_file


class DataSeekTUI(App):
    """Main TUI application."""

    CSS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "styles.tcss"
    )

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
        log_file: Optional[str] = None,
        debug: bool = False,
        seek_config_path: Optional[str] = None,
        use_robots: bool = True,
        mission_name: Optional[str] = None,
    ):
        super().__init__()
        self.mission_plan_path = mission_plan_path
        self.log_file = log_file
        self.debug_enabled = debug
        self.seek_config_path = seek_config_path
        self.use_robots = use_robots
        self.mission_name = mission_name
        if debug:
            print("Writing debug log to /tmp/tui_debug.log")
        self.stats = GenerationStats()
        self.agent_process_manager: Optional[AgentProcessManager] = None
        self.agent_output_parser = AgentOutputParser()
        self.log_handle: Optional[io.TextIOWrapper] = None

        # System theme detection
        self.system_theme_sync_enabled = DARKDETECT_AVAILABLE
        self.last_detected_theme = None
        # When True, a user manually toggled theme; don't auto-revert
        self.user_theme_override = False
        self.show_tree = False

    def debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug_enabled:
            try:
                with open("/tmp/tui_debug.log", "a") as f:
                    f.write(f"{message}\n")
            except Exception:
                pass  # Ignore debug logging errors

    def detect_system_theme(self) -> Optional[str]:
        """Detect the current system theme (dark/light)."""
        if not DARKDETECT_AVAILABLE:
            return None

        try:
            # darkdetect.isDark() returns True for dark mode, False for light mode, None if unknown
            is_dark = darkdetect.isDark()
            if is_dark is None:
                return None
            return "dark" if is_dark else "light"
        except Exception as e:
            self.debug_log(f"Failed to detect system theme: {e}")
            return None

    def sync_with_system_theme(self) -> None:
        """Sync the app theme with the system theme if it has changed."""
        if not self.system_theme_sync_enabled:
            return

        current_system_theme = self.detect_system_theme()
        if current_system_theme is None:
            return

        # Only change if system theme is different from what we detected last time
        if current_system_theme != self.last_detected_theme:
            # If the user manually toggled, clear the override only when the
            # system theme actually changes. This avoids immediate auto-revert.
            if self.user_theme_override:
                self.user_theme_override = False
            self.last_detected_theme = current_system_theme

            # Convert system theme to our dark boolean
            system_wants_dark = current_system_theme == "dark"

            # Only apply if different from current app theme
            if system_wants_dark != self.dark:
                self.dark = system_wants_dark
                theme_name = "dark" if self.dark else "light"

                # Apply theme styling
                self.apply_theme_styling()

                # Show user feedback
                if hasattr(self, "conversation"):
                    self.conversation.add_message(
                        "info", f"🌓 Auto-synced to system {theme_name} theme"
                    )

                self.debug_log(f"Auto-synced theme to system: {theme_name}")

    def _load_mission_config(self, mission_name: str) -> dict:
        """Load mission-specific configuration from the mission config file."""
        try:
            import yaml

            with open(self.mission_plan_path, "r") as f:
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
            system_theme = self.detect_system_theme()
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

        self.apply_footer_styles()
        yield Header()
        yield Footer()

    def on_mount(self):
        """Start the mission selector when the app mounts."""
        mission_details = get_mission_details_from_file(self.mission_plan_path)
        if not mission_details or not mission_details["mission_names"]:
            self.exit(message=f"❌ No missions found in {self.mission_plan_path}")
            return

        if self.mission_name:
            if self.mission_name not in mission_details["mission_names"]:
                self.exit(message=f"❌ Mission '{self.mission_name}' not found in {self.mission_plan_path}")
                return
            self.on_mission_selected(self.mission_name)
        else:
            self.push_screen(
                MissionSelector(mission_details["mission_names"], self.dark),
                self.on_mission_selected,
            )
        self.apply_footer_styles()

    def apply_footer_styles(self) -> None:
        """Applies custom styles to the footer keys based on the current theme."""

        def _apply_styles():
            try:
                footer = self.query_one(Footer)
                key_color = "#58a6ff" if self.dark else "#0969da"

                # Try multiple selectors for footer keys
                selectors_tried = []
                keys_found = 0

                for selector in [".footer-key", "FooterKey", "Key", ".key"]:
                    try:
                        keys = footer.query(selector)
                        selectors_tried.append(f"{selector}:{len(keys)}")
                        if keys:
                            keys_found += len(keys)
                            for key in keys:
                                key.styles.color = key_color
                    except Exception as e:
                        selectors_tried.append(f"{selector}:error({e})")

                # Also try CSS class approach for footer keys
                theme_class = "dark-theme" if self.dark else "light-theme"
                opposite_class = "light-theme" if self.dark else "dark-theme"

                try:
                    for selector in ["FooterKey", ".footer-key"]:
                        for key in footer.query(selector):
                            key.add_class(theme_class)
                            key.remove_class(opposite_class)
                except:
                    pass

                self.debug_log(
                    f"Footer key styling - selectors tried: {selectors_tried}, keys found: {keys_found}, color: {key_color}"
                )

            except Exception as e:
                self.debug_log(f"Footer styling failed: {e}")

        # Schedule the styling to happen after the footer is fully rendered
        self.call_after_refresh(_apply_styles)

    def apply_theme_styling(self) -> None:
        """Apply theme styling to all elements programmatically."""

        def _apply_theme_styles():
            try:
                # Define colors for current theme
                if self.dark:
                    bg_color = "#0d1117"
                    text_color = "#c9d1d9"
                    header_bg = "#21262d"
                    container_bg = "#161b22"
                    conversation_bg = "#1b1f27"
                    progress_bg = "#1b1f27"
                    mission_bg = "#1f2230"
                    title_color = "#58a6ff"
                    progress_text_color = "#58a6ff"
                else:
                    bg_color = "#f6f8fa"
                    text_color = "#24292f"
                    header_bg = "#eaeef2"
                    container_bg = "#ffffff"
                    conversation_bg = "#f0f2f6"
                    progress_bg = "#f0f2f6"
                    mission_bg = "#f6f8fa"
                    title_color = "#0969da"
                    progress_text_color = "#0969da"

                theme_name = "dark" if self.dark else "light"
                self.debug_log(f"Applying {theme_name} theme styling...")

                # Apply screen/app background if possible
                try:
                    self.screen.styles.background = bg_color
                    self.screen.styles.color = text_color
                except:
                    pass

                # Apply built-in Header widget styling with multiple approaches
                try:
                    from textual.widgets import Header

                    header_widget = self.query_one(Header)

                    # Try direct styling
                    header_widget.styles.background = header_bg
                    header_widget.styles.color = text_color

                    # Try forcing a refresh
                    header_widget.refresh()

                    # Try setting CSS variables if they exist
                    try:
                        header_widget.styles.update(
                            background=header_bg, color=text_color
                        )
                    except:
                        pass

                    # Try accessing sub-components of the header
                    try:
                        for child in header_widget.children:
                            child.styles.background = header_bg
                            child.styles.color = text_color
                    except:
                        pass

                    # Try adding/removing CSS classes for theme control
                    try:
                        if self.dark:
                            header_widget.add_class("dark-theme")
                            header_widget.remove_class("light-theme")
                        else:
                            header_widget.add_class("light-theme")
                            header_widget.remove_class("dark-theme")
                    except:
                        pass

                    self.debug_log(
                        f"Applied Header widget styling: bg={header_bg}, color={text_color}"
                    )
                except Exception as e:
                    self.debug_log(f"Header widget styling failed: {e}")

                # Apply stats header styling (custom widget)
                try:
                    # Try both the widget reference and ID selector
                    if hasattr(self, "stats_header"):
                        # Apply background and text color
                        self.stats_header.styles.background = header_bg
                        self.stats_header.styles.color = text_color
                        # Force refresh to pick up the new colors
                        self.stats_header.refresh()
                        self.debug_log(f"Applied stats_header reference styling")

                    # Also try targeting by ID
                    try:
                        stats_header = self.query_one("#stats-header")
                        stats_header.styles.background = header_bg
                        stats_header.styles.color = text_color
                        stats_header.refresh()
                        self.debug_log(f"Applied stats_header ID styling")
                    except:
                        pass

                    # Try adding CSS classes for theme control
                    try:
                        if hasattr(self, "stats_header"):
                            if self.dark:
                                self.stats_header.add_class("dark-theme")
                                self.stats_header.remove_class("light-theme")
                            else:
                                self.stats_header.add_class("light-theme")
                                self.stats_header.remove_class("dark-theme")
                    except:
                        pass

                except Exception as e:
                    self.debug_log(f"Stats header styling failed: {e}")

                # Apply main container styling
                try:
                    container = self.query_one("#main-container")
                    container.styles.background = container_bg
                    container.styles.color = text_color
                except:
                    pass

                # Apply progress panel styling
                if hasattr(self, "progress_panel"):
                    try:
                        self.progress_panel.styles.background = progress_bg
                        self.progress_panel.styles.color = text_color
                    except:
                        pass

                # Apply mission panel styling
                if hasattr(self, "mission_panel"):
                    try:
                        self.mission_panel.styles.background = mission_bg
                        self.mission_panel.styles.color = text_color
                    except:
                        pass

                # Apply conversation styling
                if hasattr(self, "conversation"):
                    try:
                        self.conversation.styles.background = conversation_bg
                        self.conversation.styles.color = text_color
                    except:
                        pass

                # Apply title styling
                for title in self.query(".panel-title"):
                    try:
                        title.styles.color = title_color
                    except:
                        pass

                # Apply progress text styling
                try:
                    progress = self.query_one("#main-progress")
                    progress.styles.color = progress_text_color
                except:
                    pass

                # Apply CSS classes to all elements for theme control
                try:
                    # Apply theme class to all major elements
                    theme_class = "dark-theme" if self.dark else "light-theme"
                    opposite_class = "light-theme" if self.dark else "dark-theme"

                    # List of all elements to apply theme classes to
                    element_selectors = [
                        "#progress-panel",
                        "#main-progress",
                        "#recent-samples",
                        "#mission-panel",
                        "#mission-info",
                        "#conversation",
                        "#main-container",
                    ]

                    for selector in element_selectors:
                        try:
                            element = self.query_one(selector)
                            element.add_class(theme_class)
                            element.remove_class(opposite_class)
                        except:
                            pass

                    # Apply theme class to all panel titles
                    for title in self.query(".panel-title"):
                        try:
                            title.add_class(theme_class)
                            title.remove_class(opposite_class)
                        except:
                            pass

                    self.debug_log(f"Applied CSS theme classes: {theme_class}")

                except Exception as e:
                    self.debug_log(f"CSS class application failed: {e}")

                # Apply scrollbar styling with multiple approaches
                try:
                    theme_class = "dark-theme" if self.dark else "light-theme"
                    opposite_class = "light-theme" if self.dark else "dark-theme"

                    # Try multiple scrollbar element selectors
                    scrollbar_selectors = [
                        ".scrollbar",
                        "Scrollbar",
                        ".vertical-scrollbar",
                        ".horizontal-scrollbar",
                    ]

                    thumb_selectors = [".scrollbar-thumb", "ScrollbarThumb", ".thumb"]

                    # Apply to all scrollbar elements in the app
                    for selector in scrollbar_selectors:
                        try:
                            for scrollbar in self.query(selector):
                                scrollbar.add_class(theme_class)
                                scrollbar.remove_class(opposite_class)
                        except:
                            pass

                    # Apply to all scrollbar thumb elements
                    for selector in thumb_selectors:
                        try:
                            for thumb in self.query(selector):
                                thumb.add_class(theme_class)
                                thumb.remove_class(opposite_class)
                        except:
                            pass

                    # Debug: Log what scrollbar elements we can find
                    found_elements = []
                    for selector in scrollbar_selectors + thumb_selectors:
                        try:
                            elements = self.query(selector)
                            if elements:
                                found_elements.append(
                                    f"{selector}: {len(elements)} elements"
                                )
                        except:
                            pass

                    if found_elements:
                        self.debug_log(
                            f"Found scrollbar elements: {', '.join(found_elements)}"
                        )
                    else:
                        self.debug_log("No scrollbar elements found with any selector")

                        # Try to find ALL elements and log them
                        all_elements = []
                        try:
                            for element in self.query("*"):
                                if hasattr(element, "id") and element.id:
                                    all_elements.append(element.id)
                                elif hasattr(element, "__class__"):
                                    class_name = element.__class__.__name__
                                    if "scroll" in class_name.lower():
                                        all_elements.append(class_name)

                            if all_elements:
                                self.debug_log(
                                    f"All elements with IDs or scroll-related: {', '.join(set(all_elements))}"
                                )
                        except:
                            pass

                except Exception as e:
                    self.debug_log(f"Scrollbar styling failed: {e}")

            except Exception as e:
                self.debug_log(f"Theme styling error: {e}")

        # Apply built-in Footer widget styling
        def _apply_footer_widget_styles():
            try:
                from textual.widgets import Footer

                footer_widget = self.query_one(Footer)
                # Use the same color variables from the parent scope
                current_footer_bg = "#21262d" if self.dark else "#eaeef2"
                current_footer_text = "#c9d1d9" if self.dark else "#24292f"
                footer_widget.styles.background = current_footer_bg
                footer_widget.styles.color = current_footer_text
                self.debug_log(
                    f"Applied Footer widget styling: bg={current_footer_bg}, color={current_footer_text}"
                )
            except Exception as e:
                self.debug_log(f"Footer widget styling failed: {e}")

        self.call_after_refresh(_apply_footer_widget_styles)

        # Apply footer key styles
        self.apply_footer_styles()

        # Apply main theme styles
        self.call_after_refresh(_apply_theme_styles)

    def action_toggle_theme(self) -> None:
        """An action to toggle between dark and light themes (manual override)."""
        # Toggle theme manually
        self.dark = not self.dark
        theme_name = "dark" if self.dark else "light"

        # Mark a manual override and align last_detected_theme to current
        # system theme so periodic checks don't immediately revert it.
        if self.system_theme_sync_enabled:
            self.user_theme_override = True
            current_system = self.detect_system_theme()
            if current_system is not None:
                self.last_detected_theme = current_system

        # Apply all theme styling programmatically
        self.apply_theme_styling()

        # Show user feedback
        if hasattr(self, "conversation"):
            sync_status = (
                " (manual override)" if self.system_theme_sync_enabled else ""
            )
            self.conversation.add_message(
                "info", f"🎨 Switched to {theme_name} theme{sync_status}"
            )

        self.debug_log(f"Theme manually toggled to: {theme_name}")

    def action_debug_scrollbars(self):
        """Action to debug scrollbars."""
        self.debug_all_scrollbars()

    def on_mission_selected(self, mission_name: str):
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
                pass

        mission_details = get_mission_details_from_file(self.mission_plan_path)
        total_samples_target = mission_details["mission_targets"].get(
            mission_name, 1200
        )
        self.stats.target = total_samples_target

        seek_config = load_seek_config(
            self.seek_config_path, use_robots=self.use_robots
        )
        # Set active config for TUI context as well (non-subprocess usage)
        try:
            set_active_seek_config(seek_config)
        except Exception:
            pass
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

        self.stats_header = StatsHeader()
        # Initialize stats with synthetic budget and target size
        self.stats.synthetic_budget = synthetic_budget
        self.stats.target_size = target_size
        self.stats.total_recursion_steps = recursion_limit

        self.progress_panel = ProgressPanel()
        self.mission_panel = MissionPanel(
            self.mission_plan_path,
            mission_name,
            total_samples_target,
            seek_config,
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
        self.apply_theme_styling()

        # Also apply footer styling with a small delay to ensure footer keys are rendered
        def _delayed_footer_styling():
            self.debug_log("Applying delayed footer key styling after UI load")
            self.apply_footer_styles()

        self.set_timer(0.5, _delayed_footer_styling)

        # Start periodic system theme checking
        if self.system_theme_sync_enabled:

            def _check_system_theme():
                self.sync_with_system_theme()
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

        self.run_worker(self._run_agent(), name="agent")

    async def _run_agent(self):
        """Run the Data Seek Agent as a subprocess and parse its output."""
        self.stats.started_at = datetime.now()
        try:
            process = await self.agent_process_manager.start()
            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if line:
                    # Log to file if log file is specified
                    if self.log_handle:
                        try:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            self.log_handle.write(f"[{timestamp}] {line}\n")
                            self.log_handle.flush()
                        except Exception:
                            pass  # Continue if log writing fails

                    # Parse the line and handle events
                    events = list(self.agent_output_parser.parse_line(line))
                    if events:
                        with open("/tmp/tui_debug.log", "a") as f:
                            f.write(
                                f"GENERATED {len(events)} EVENTS for line: {line[:100]}...\n"
                            )
                    for event in events:
                        self._handle_agent_event(event)
        except Exception as e:
            self.conversation.add_message("error", f"Agent error: {str(e)}")
        finally:
            self.conversation.add_message("info", "Agent process completed")
            # Start a 5-second timer to auto-close the TUI
            self.set_timer(5.0, self._auto_close_after_completion)

    def _handle_agent_event(self, event):
        """Handle events from the agent output parser."""
        # Debug: Log all events being handled
        self.debug_log(f"HANDLING EVENT: {type(event).__name__} - {event}")

        if isinstance(event, ProgressUpdate):
            self.stats.completed = event.completed
            self.stats.target = event.target
            self.stats_header.update_stats(self.stats)
            self.progress_panel.update_progress(self.stats)

            # Update mission status when progress is made
            if event.completed > 0:
                self.mission_panel.update_status(
                    f"Generating... ({event.completed}/{event.target})"
                )

            self.debug_log(f"PROGRESS UPDATE: {event.completed}/{event.target}")
        elif isinstance(event, SyntheticSampleUpdate):
            # Update synthetic sample count
            self.stats.synthetic_completed = event.count
            self.stats_header.update_stats(self.stats)
            self.debug_log(f"SYNTHETIC SAMPLE UPDATE: {event.count}")
        elif isinstance(event, RecursionStepUpdate):
            # Update recursion step information
            self.stats.current_recursion_step = event.current_step
            self.stats.total_recursion_steps = event.total_steps
            self.stats_header.update_stats(self.stats)
            self.debug_log(
                f"RECURSION STEP UPDATE: {event.current_step}/{event.total_steps}"
            )
        elif isinstance(event, NewMessage):
            self.debug_log(
                f"NEW MESSAGE EVENT: {event.role} -> {event.content[:100]}..."
            )
            self.conversation.add_message(event.role, event.content)

            # Update mission status based on message content
            import re

            # Check for Graph Router patterns
            if "Graph Router: Routing to" in event.content:
                route_match = re.search(
                    r"Graph Router: Routing to (\w+)", event.content
                )
                if route_match:
                    route_name = route_match.group(1)
                    if route_name.lower() == "end":
                        self.mission_panel.update_status("Sample Completed")
                    elif route_name.lower() == "archive":
                        self.mission_panel.update_status("Archiving Sample...")
                    elif route_name.lower() == "fitness":
                        self.mission_panel.update_status("Checking Fitness...")
                    elif route_name.lower() == "synthetic":
                        self.mission_panel.update_status("Generating Synthetic...")
                    else:
                        self.mission_panel.update_status(f"Routing to {route_name}...")

            # Check for node execution patterns like "🔍 RESEARCH NODE" or "🎨 SYNTHETIC NODE"
            elif ("NODE" in event.content and "🔍" in event.content) or (
                "SYNTHETIC NODE" in event.content and "🎨" in event.content
            ):
                if "🎨 SYNTHETIC NODE" in event.content:
                    self.mission_panel.update_status("Generating Synthetic Content...")
                else:
                    node_match = re.search(r"🔍\s+(\w+)\s+NODE", event.content)
                    if node_match:
                        node_name = node_match.group(1)
                        self.mission_panel.update_status(f"Working on {node_name}...")

            # Check for agent starting work (move from Initializing)
            elif (
                "▶ Iteration" in event.content
                or "🔧 Tool calls:" in event.content
                or "📊 CONTEXT:" in event.content
            ) and self.mission_panel.current_status == "Initializing...":
                self.mission_panel.update_status("Working...")
                # Extract current recursion step if available
                import re

                iteration_match = re.search(r"▶ Iteration (\d+)/(\d+)", event.content)
                if iteration_match:
                    current_step = int(iteration_match.group(1))
                    total_steps = int(iteration_match.group(2))
                    self.stats.current_recursion_step = current_step
                    self.stats.total_recursion_steps = total_steps
                    self.stats_header.update_stats(self.stats)

            # Check for routing to END (fallback pattern)
            elif (
                "Routing to END" in event.content or "Decided on 'end'" in event.content
            ):
                self.mission_panel.update_status("Sample Completed")

            # Check for sample archival and add to recent samples
            elif (
                "sample #" in event.content.lower()
                and "archived" in event.content.lower()
            ):
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
                    sample_excerpt = self._extract_recent_sample_excerpt()

                    if sample_excerpt:
                        description = f"{sample_excerpt} ({completion_pct}% complete) - Source: {source_type}"
                    else:
                        description = f"Archived ({completion_pct}% complete) - Source: {source_type}"

                    self.progress_panel.add_sample(sample_num, description)
        elif isinstance(event, ErrorMessage):
            self.debug_log(f"ERROR MESSAGE EVENT: {event.message[:100]}...")
            self.conversation.add_message("error", event.message)
            self.stats.errors += 1
            self.stats_header.update_stats(self.stats)

    def action_quit(self):
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
            except Exception:
                pass  # Ignore errors during cleanup

        self.exit()

    def action_show_dom(self):
        """Show the DOM tree in the console."""
        if hasattr(self, "log"):
            self.log(self.tree)  # This will print the tree to the dev console
        else:
            print(self.tree)  # Fallback to print

    def action_show_error_modal(self) -> None:
        """Show the error display modal."""
        from seek.tui.components.error_modal import ErrorModal

        error_messages = [
            msg["content"]
            for msg in self.conversation.messages
            if msg["role"] == "error"
        ]
        if not error_messages:
            error_messages = ["No errors recorded yet."]
        self.push_screen(ErrorModal(error_messages=error_messages))

    def debug_all_scrollbars(self):
        """Find and inspect actual scrollbar objects."""
        print("=== SCROLLBAR DEBUG START ===")

        try:
            conversation = self.query_one("#conversation")

            # Access the actual scrollbar objects
            v_scrollbar = conversation.vertical_scrollbar
            h_scrollbar = conversation.horizontal_scrollbar

            print(f"Vertical scrollbar: {v_scrollbar}")
            print(f"Vertical scrollbar type: {type(v_scrollbar).__name__}")
            print(
                f"Vertical scrollbar classes: {getattr(v_scrollbar, 'classes', 'none')}"
            )

            print(f"Horizontal scrollbar: {h_scrollbar}")
            print(f"Horizontal scrollbar type: {type(h_scrollbar).__name__}")
            print(
                f"Horizontal scrollbar classes: {getattr(h_scrollbar, 'classes', 'none')}"
            )

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

    def _auto_close_after_completion(self):
        """Auto-close the TUI after agent process completion with a countdown."""
        self.conversation.add_message(
            "info", "TUI will close in 5 seconds... (Press 'q' to exit immediately)"
        )
        # Set another timer to actually close
        self.set_timer(5.0, self.action_quit)

    def _extract_recent_sample_excerpt(self) -> str:
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
                self.debug_log(
                    "End boundary not in same message, searching subsequent messages"
                )
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
    mission: Optional[str] = typer.Option(None, "--mission", "-m", help="The name of the mission to run."),
    log: Optional[str] = typer.Option(
        None, "--log", help="Log file to save terminal output for debugging"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug logging to /tmp/tui_debug.log"
    ),
    config: str = typer.Option(
        "config/seek_config.yaml",
        "--config",
        "-c",
        help="Path to the agent configuration file.",
    ),
    mission_config: str = typer.Option(
        "config/mission_config.yaml",
        "--mission-config",
        help="Path to the mission configuration file.",
    ),
    no_robots: bool = typer.Option(
        False, "--no-robots", help="Ignore robots.txt rules"
    ),
):
    """Start the Data Seek Agent TUI for sample generation."""
    if not os.path.exists(mission_config):
        typer.echo(f"❌ Mission file not found: {mission_config}", err=True)
        raise typer.Exit(1)

    # Handle --no-robots flag
    use_robots = not no_robots

    app = DataSeekTUI(
        mission_plan_path=mission_config,
        log_file=log,
        debug=debug,
        seek_config_path=config,
        use_robots=use_robots,
        mission_name=mission,
    )
    app.run()


def main():
    cli_app()

if __name__ == "__main__":
    main()
