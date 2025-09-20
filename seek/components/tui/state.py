from textual.reactive import reactive

from .components.stats_header import GenerationStats


class TUIState:
    """A centralized, observable store for TUI state."""

    stats = reactive(GenerationStats())
    mission_status = reactive("Initializing...")
    # Add other reactive state variables here as needed
    # For example:
    # error_messages = reactive([])
