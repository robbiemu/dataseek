import os
import tempfile
from datetime import datetime
from typing import Any

from textual.widgets import Log


class ConversationPanel(Log):
    """Bottom panel showing agent conversation with auto-scroll."""

    def __init__(self, debug: bool = False) -> None:
        super().__init__(auto_scroll=True, id="conversation")
        self.debug = debug
        self.messages: list[dict[str, Any]] = []  # Store messages for excerpt extraction
        # Add initial placeholder
        self.write("🤖 Data Seek Agent TUI")
        self.write("Waiting for agent to start...")
        self._initialized = False

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation log with role-based coloring."""
        # Store the message for later extraction
        self.messages.append({"role": role, "content": content, "timestamp": datetime.now()})

        # Keep only the last 50 messages to avoid memory issues
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

        # Clear placeholder on first real message
        if not self._initialized:
            self.clear()
            self._initialized = True

        colors = {
            "system": "bright_black",
            "user": "cyan",
            "assistant": "green",
            "tool": "yellow",
            "debug": "magenta",
            "info": "blue",
            "error": "red",
        }
        _color = colors.get(role, "white")
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Use plain text without markup to avoid rendering issues
        role_str = f"{role:>9}"
        message_line = f"{timestamp} {role_str}: {content}"

        # Log write attempts for debugging when enabled
        if self.debug:
            try:
                debug_path = os.path.join(tempfile.gettempdir(), "tui_debug.log")
                with open(debug_path, "a") as f:
                    f.write(f"TUI WRITE ATTEMPT: {role} -> {message_line}\n")
            except Exception:
                pass  # nosec B110 # - debug logging failure is non-fatal

        # Use write() with explicit newline to ensure each message is on its own line
        try:
            self.write(message_line + "\n")
            # Force refresh/update after writing
            self.refresh()
            if self.debug:
                try:
                    debug_path = os.path.join(tempfile.gettempdir(), "tui_debug.log")
                    with open(debug_path, "a") as f:
                        f.write(f"TUI WRITE SUCCESS: {role}\n")
                except OSError:
                    pass
        except Exception as e:
            # Debug: write to a file if widget writing fails
            if self.debug:
                self._write_debug_log(e, role, content)

    def _write_debug_log(self, e: Exception, role: str, content: str) -> None:
        """A best-effort, non-critical debug logger."""
        try:
            debug_path = os.path.join(tempfile.gettempdir(), "tui_debug.log")
            with open(debug_path, "a") as f:
                f.write(f"ERROR writing message: {e}\n")
                f.write(f"Role: {role}, Content: {content[:100]}...\n")
        except OSError:
            # This best-effort logging can fail silently.
            pass
