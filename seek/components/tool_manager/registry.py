# A simple dictionary to act as our central registry.
from seek.components.tool_manager.plugin_base import BaseTool

PLUGIN_REGISTRY: dict[str, type[BaseTool]] = {}


def register_plugin(cls: type[BaseTool]) -> type[BaseTool]:
    """A decorator to register a tool plugin."""
    plugin_name = cls.name.lower()
    if plugin_name in PLUGIN_REGISTRY:
        # Handle potential name collisions
        raise ValueError(f"Plugin with name '{plugin_name}' is already registered.")
    PLUGIN_REGISTRY[plugin_name] = cls
    return cls
