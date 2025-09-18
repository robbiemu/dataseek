from datetime import datetime

from textual.widgets import Log


class ConversationPanel(Log):
    """Bottom panel showing agent conversation with auto-scroll."""

    def __init__(self, debug: bool = False):
        super().__init__(auto_scroll=True, id="conversation")
        self.debug = debug
        self.messages = []  # Store messages for excerpt extraction
        # Add initial placeholder
        self.write("🤖 Data Seek Agent TUI")
        self.write("Waiting for agent to start...")
        self._initialized = False

    def add_message(self, role: str, content: str):
        """Add a message to the conversation log with role-based coloring."""
        # Store the message for later extraction
        self.messages.append(
            {"role": role, "content": content, "timestamp": datetime.now()}
        )

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

        # Debug: Log what we're trying to write if debug enabled
        if self.debug:
            try:
                with open("/tmp/tui_debug.log", "a") as f:
                    f.write(f"TUI WRITE ATTEMPT: {role} -> {message_line}\n")
            except Exception:
                pass

        # Use write() with explicit newline to ensure each message is on its own line
        try:
            self.write(message_line + "\n")
            # Force refresh/update after writing
            self.refresh()
            if self.debug:
                try:
                    with open("/tmp/tui_debug.log", "a") as f:
                        f.write(f"TUI WRITE SUCCESS: {role}\n")
                except Exception:
                    pass
        except Exception as e:
            # Debug: write to a file if widget writing fails
            if self.debug:
                try:
                    with open("/tmp/tui_debug.log", "a") as f:
                        f.write(f"ERROR writing message: {e}\n")
                        f.write(f"Role: {role}, Content: {content[:100]}...\n")
                except Exception:
                    pass
