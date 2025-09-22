from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from rich.text import Text
from textual.widgets import Static


@dataclass
class GenerationStats:
    """Statistics about the sample generation process."""

    target: int = 1200
    completed: int = 0
    synthetic_completed: int = 0  # Count of synthetic samples generated
    errors: int = 0
    started_at: datetime | None = None
    updated_at: datetime | None = None
    agent_activities: int = 0  # Count of agent actions/responses
    current_recursion_step: int = 0  # Current recursion step
    total_recursion_steps: int = 30  # Total recursion steps
    synthetic_budget: float = 1.0  # Synthetic budget ratio (from mission config)
    target_size: int = 1  # Target size per goal (from mission config)

    def copy(self) -> "GenerationStats":
        """Return a shallow copy of this GenerationStats dataclass.

        This provides a convenient `.copy()` method to match existing callers
        in the TUI code which expect a `copy()` method on the stats object.
        """
        # Use dataclasses.replace to produce a new instance with the same field values
        return replace(self)

    @property
    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0
        return (self.updated_at or datetime.now()).timestamp() - self.started_at.timestamp()

    @property
    def samples_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0
        return self.completed / self.elapsed_seconds

    @property
    def eta_seconds(self) -> float:
        if self.samples_per_second <= 0:
            return 0
        return (self.target - self.completed) / self.samples_per_second

    @property
    def eta_human(self) -> str:
        if self.completed >= self.target:
            return "Done"
        if self.samples_per_second <= 0 or self.elapsed_seconds < 60:
            return "..."

        eta_delta = timedelta(seconds=int(self.eta_seconds))
        if eta_delta.total_seconds() < 60:
            return f"{int(eta_delta.total_seconds())}s"
        elif eta_delta.total_seconds() < 3600:
            return f"{int(eta_delta.total_seconds() / 60)}m"
        else:
            hours = int(eta_delta.total_seconds() / 3600)
            minutes = int((eta_delta.total_seconds() % 3600) / 60)
            return f"{hours}h {minutes}m"

    @property
    def elapsed_human(self) -> str:
        if self.elapsed_seconds < 60:
            return f"{int(self.elapsed_seconds)}s"
        elif self.elapsed_seconds < 3600:
            return f"{int(self.elapsed_seconds / 60)}m"
        else:
            hours = int(self.elapsed_seconds / 3600)
            minutes = int((self.elapsed_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    @property
    def synthetic_percentage(self) -> float:
        """Calculate the percentage of generated samples that are synthetic."""
        if self.completed <= 0:
            return 0.0
        return (self.synthetic_completed / self.completed) * 100

    @property
    def max_synthetic_allowed(self) -> int:
        """Calculate the maximum synthetic samples allowed based on target size and synthetic budget."""
        return int(self.target * self.synthetic_budget)


class StatsHeader(Static):
    """Header showing generation statistics."""

    def __init__(self) -> None:
        super().__init__(id="stats-header", markup=True)  # Enable Rich markup

    def on_mount(self) -> None:
        """Start watching for stats changes when the component is mounted."""
        self.watch(self.app, "stats", self.on_stats_change)

    def on_stats_change(self, stats: GenerationStats) -> None:
        """Update the header when the stats object changes."""
        int(100 * stats.completed / max(1, stats.target))
        synthetic_pct = stats.synthetic_percentage

        # Determine color for synthetic percentage
        synth_color = ""
        synth_color_end = ""

        # Calculate target percentage and lambda threshold
        target_synthetic_pct = stats.synthetic_budget * 100  # e.g., 20% for 0.2 budget
        lambda_factor = 0.8
        lower_threshold_pct = target_synthetic_pct * lambda_factor  # e.g., 16% for 20% target

        # Color coding logic:
        # - At 0/n progress: Default color (no coloring)
        # - Red if synthetic samples exceed max allowed OR impossible to achieve target
        # - Orange if percentage is above target OR below lambda threshold (but not at 0%)
        # - Default color when within acceptable range
        remaining_samples = stats.target - stats.completed
        max_possible_synthetic = stats.synthetic_completed + remaining_samples

        if (
            stats.synthetic_completed > stats.max_synthetic_allowed
            or max_possible_synthetic < stats.max_synthetic_allowed
        ):
            # Red if exceeded budget or impossible to achieve minimum target
            synth_color = "[red]"
            synth_color_end = "[/red]"
        elif stats.completed > 0 and (
            synthetic_pct > target_synthetic_pct or synthetic_pct < lower_threshold_pct
        ):
            # Orange if above target or below lambda threshold (but only when completed > 0)
            synth_color = "[orange1]"
            synth_color_end = "[/orange1]"

        # Format synthetic progress with color using Rich Text objects
        synthetic_text = Text(f"{synthetic_pct:.0f}% synth")
        if synth_color == "[red]":
            synthetic_text.stylize("red")
        elif synth_color == "[orange1]":
            synthetic_text.stylize("orange1")

        # Build the full status line using Rich Text
        status_text = Text()
        status_text.append(f"📊 Progress: {stats.completed}/{stats.target} (")
        status_text.append(synthetic_text)
        status_text.append(") | ")
        status_text.append(f"⏱️  Elapsed: {stats.elapsed_human} | ")
        status_text.append(f"🎯 ETA: {stats.eta_human} | ")
        status_text.append(
            f"🤖 Activity: {stats.current_recursion_step}/{stats.total_recursion_steps} | "
        )
        status_text.append(f"❌ Errors: {stats.errors}")

        # Maintain markup string version for debugging
        synthetic_display = f"{synth_color}{synthetic_pct:.0f}% synth{synth_color_end}"
        status_line = (
            f"📊 Progress: {stats.completed}/{stats.target} ({synthetic_display}) | "
            f"⏱️  Elapsed: {stats.elapsed_human} | "
            f"🎯 ETA: {stats.eta_human} | "
            f"🤖 Activity: {stats.current_recursion_step}/{stats.total_recursion_steps} | "
            f"❌ Errors: {stats.errors}"
        )

        # Log stats header updates for debugging
        try:
            import os
            import tempfile

            debug_path = os.path.join(tempfile.gettempdir(), "tui_debug.log")
            with open(debug_path, "a") as f:
                f.write(f"STATS HEADER UPDATE: {status_line}\n")
                f.write(
                    f"  Synthetic %: {synthetic_pct:.1f}% (target: {target_synthetic_pct}%), Color: {synth_color or 'default'}\n"
                )
        except Exception:
            pass  # nosec B110 # - debug logging failure is non-fatal

        # Use Rich Text object for proper color rendering
        self.update(status_text)
