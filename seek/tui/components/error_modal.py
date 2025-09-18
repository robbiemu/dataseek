from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, RichLog


class ErrorModal(ModalScreen):
    """Modal screen for displaying error messages."""

    BINDINGS = [
        Binding("escape", "close_modal", "Close", show=False),
        Binding("e", "close_modal", "Close", show=False),
    ]

    def __init__(self, error_messages: list[str], **kwargs):
        super().__init__(**kwargs)
        self.error_messages = error_messages

    def compose(self) -> ComposeResult:
        yield Container(
            RichLog(id="error-log", wrap=True),
            Button("Close", variant="primary", id="close-errors"),
            id="error-modal-container",
        )

    def on_mount(self) -> None:
        error_log = self.query_one(RichLog)
        error_log.write("## Error Log")
        for msg in self.error_messages:
            error_log.write(msg)

        # Apply theme styling to match the parent app
        self.apply_theme_styling()

    def apply_theme_styling(self) -> None:
        """Apply theme styling to the error modal based on the parent app's theme."""
        try:
            # Get the theme from the parent app
            parent_app = self.app
            is_dark = getattr(parent_app, "dark", True)

            # Define colors for current theme
            if is_dark:
                container_bg = "#1b1f27"
                text_color = "#f85149"
                border_color = "#58a6ff"
                log_bg = "#161b22"
                log_text = "#c9d1d9"
                button_bg = "#2d333b"
                button_text = "#f85149"
            else:
                container_bg = "#f6f8fa"
                text_color = "#cf222e"
                border_color = "#0969da"
                log_bg = "#ffffff"
                log_text = "#24292f"
                button_bg = "#eaeef2"
                button_text = "#cf222e"

            # Apply styling to the modal screen background
            try:
                # Apply theme class to the modal screen
                theme_class = "dark-theme" if is_dark else "light-theme"
                opposite_class = "light-theme" if is_dark else "dark-theme"

                self.add_class(theme_class)
                self.remove_class(opposite_class)

                # Also try direct styling as backup
                if is_dark:
                    screen_bg = "rgba(0, 0, 0, 0.5)"  # Dark semi-transparent
                else:
                    screen_bg = "rgba(200, 200, 200, 0.5)"  # Light semi-transparent

                self.styles.background = screen_bg
            except Exception:
                pass  # nosec B110 # - styling failure is non-fatal

            # Apply styling to modal components
            try:
                container = self.query_one("#error-modal-container")
                container.styles.background = container_bg
                container.styles.color = text_color
                container.styles.border = ("solid", border_color)
            except Exception:
                pass  # nosec B110 # - styling failure is non-fatal

            try:
                error_log = self.query_one("#error-log")
                error_log.styles.background = log_bg
                error_log.styles.color = log_text
            except Exception:
                pass  # nosec B110 # - styling failure is non-fatal

            try:
                close_button = self.query_one("#close-errors")
                close_button.styles.background = button_bg
                close_button.styles.color = button_text
            except Exception:
                pass  # nosec B110 # - styling failure is non-fatal

        except Exception:
            # Silently continue if styling fails
            pass  # nosec B110 # - non-critical visual styling failure

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-errors":
            self.dismiss()

    def action_close_modal(self) -> None:
        """Close the modal screen."""
        self.dismiss()
