from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class MissionSelector(ModalScreen[str]):
    """A modal screen to select a mission."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("f12", "show_dom", "DOM"),
    ]

    def __init__(self, missions: list[str], is_dark: bool):
        super().__init__()
        self.missions = missions
        # Store the theme state passed from the main app
        self._is_dark = is_dark

    def compose(self) -> ComposeResult:
        yield Static("🚀 Please select a mission to run:", id="mission-title")
        for i, mission in enumerate(self.missions):
            yield Button(
                f"{i + 1}. {mission}",
                id=mission,
                variant="primary",
                classes="mission-button",
            )

    def on_mount(self) -> None:
        """
        Applies the correct theme class to the modal and its children
        when the screen is mounted. This is a robust way to ensure
        modals match the app's theme.
        """
        theme_class = "dark-theme" if self._is_dark else "light-theme"
        opposite_class = "light-theme" if self._is_dark else "dark-theme"

        # Apply classes to the modal screen itself and the title
        self.add_class(theme_class)
        self.remove_class(opposite_class)

        title = self.query_one("#mission-title", Static)
        title.add_class(theme_class)
        title.remove_class(opposite_class)

        # Apply classes to all buttons
        for button in self.query(Button):
            button.add_class(theme_class)
            button.remove_class(opposite_class)

    def action_quit(self) -> None:
        self.app.action_quit()

    def action_show_dom(self) -> None:
        self.app.action_show_dom()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)
