from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class MissionPanel(Static):
    """Right panel with mission status and details."""

    def __init__(
        self,
        mission_path: str,
        mission_name: str,
        total_samples_target: int,
        config: dict[str, Any],
    ):
        super().__init__(id="mission-panel")
        self.mission_path = mission_path
        self.mission_name = mission_name
        self.total_samples_target = total_samples_target
        self.config = config
        self.mission_info_widget: Static | None = None

    def on_mount(self) -> None:
        """Start watching for mission status changes when the component is mounted."""
        self.watch(self.app, "mission_status", self.on_status_change)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🎯 Mission Status", classes="panel-title")
            self.mission_info_widget = Static(self._get_mission_info(), id="mission-info")
            if self.mission_info_widget is not None:
                yield self.mission_info_widget

    def _get_mission_info(self) -> str:
        """Get the mission info text."""
        mission_plan = self.config.get("mission_plan", {})
        nodes = mission_plan.get("nodes", [])
        model = nodes[0].get("model", "unknown") if nodes else "unknown"
        provider = (self.config.get("web_search") or {}).get("provider") or self.config.get(
            "search_provider", "unknown"
        )
        mission_status = getattr(self.app, "mission_status", "unknown")
        return (
            f"Mission: {self.mission_name}\n"
            f"Target: {self.total_samples_target} samples\n"
            f"Model: {model}\n"
            f"Provider: {provider}\n"
            f"Status: {mission_status}"
        )

    def on_status_change(self, new_status: str) -> None:
        """Update the mission status displayed in the panel."""
        if self.mission_info_widget:
            self.mission_info_widget.update(self._get_mission_info())
