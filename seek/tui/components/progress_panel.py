from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from .stats_header import GenerationStats


class ProgressPanel(Static):
    """Left panel with progress bar and recent samples."""

    def __init__(self):
        super().__init__(id="progress-panel")
        self.progress_display = None
        self.recent_samples_widget = None
        self.recent_samples = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("📈 Generation Progress", classes="panel-title")
            self.progress_display = Static("0%", id="main-progress")
            yield self.progress_display
            self.recent_samples_widget = Static(
                "Recent samples will appear here...", id="recent-samples"
            )
            yield self.recent_samples_widget

    def update_progress(self, stats: GenerationStats):
        if self.progress_display:
            progress_pct = int(100 * stats.completed / max(1, stats.target))
            # Create a simple progress bar using characters
            bar_width = 40
            filled_width = int(bar_width * progress_pct / 100)
            bar = "█" * filled_width + "░" * (bar_width - filled_width)
            self.progress_display.update(f"{progress_pct}% [{bar}]")

    def add_sample(self, sample_number: int, description: str):
        """Add a recent sample to the display."""
        # Format the sample entry with proper wrapping for long descriptions
        if len(description) > 80:
            # For long excerpts, show on multiple lines with proper indentation
            sample_entry = f"#{sample_number}:\n  {description}"
        else:
            # For short descriptions, keep on one line
            sample_entry = f"#{sample_number}: {description}"

        # Keep more recent samples (increased from 3 to 10)
        self.recent_samples.append(sample_entry)
        if len(self.recent_samples) > 10:
            self.recent_samples = self.recent_samples[-10:]

        # Update the widget
        if self.recent_samples_widget:
            if self.recent_samples:
                content = "\n\n".join(
                    self.recent_samples
                )  # Extra spacing for readability
            else:
                content = "Recent samples will appear here..."
            self.recent_samples_widget.update(content)
