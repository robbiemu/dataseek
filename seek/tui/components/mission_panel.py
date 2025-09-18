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
        self.current_status = "Initializing..."
        self.mission_info_widget = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🎯 Mission Status", classes="panel-title")
            # Extract values from config dict
            mission_plan = self.config.get("mission_plan", {})
            nodes = mission_plan.get("nodes", [])
            model = nodes[0].get("model", "unknown") if nodes else "unknown"
            provider = (self.config.get("web_search") or {}).get("provider") or self.config.get(
                "search_provider", "unknown"
            )

            mission_info = (
                f"Mission: {self.mission_name}\n"
                f"Target: {self.total_samples_target} samples\n"
                f"Model: {model}\n"
                f"Provider: {provider}\n"
                f"Status: {self.current_status}"
            )
            self.mission_info_widget = Static(mission_info, id="mission-info")
            yield self.mission_info_widget

    def update_status(self, new_status: str):
        """Update the mission status displayed in the panel."""
        self.current_status = new_status
        if self.mission_info_widget:
            # Extract values from config dict
            mission_plan = self.config.get("mission_plan", {})
            nodes = mission_plan.get("nodes", [])
            model = nodes[0].get("model", "unknown") if nodes else "unknown"
            provider = (self.config.get("web_search") or {}).get("provider") or self.config.get(
                "search_provider", "unknown"
            )

            mission_info = (
                f"Mission: {self.mission_name}\n"
                f"Target: {self.total_samples_target} samples\n"
                f"Model: {model}\n"
                f"Provider: {provider}\n"
                f"Status: {self.current_status}"
            )
            self.mission_info_widget.update(mission_info)
