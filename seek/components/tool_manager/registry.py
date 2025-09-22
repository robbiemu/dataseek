# A simple dictionary to act as our central registry.
from seek.components.tool_manager.plugin_base import BaseTool

PLUGIN_REGISTRY: dict[str, type[BaseTool]] = {}


def _camel_to_snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def register_plugin(cls: type[BaseTool]) -> type[BaseTool]:
    """A decorator to register a tool plugin.

    Uses a defensive lookup for the plugin name to avoid issues with
    Pydantic/LangChain model field descriptors at class-level.
    """
    # Prefer explicit class attribute if present and a string
    explicit_name = getattr(cls, "name", None)
    if isinstance(explicit_name, str) and explicit_name:
        plugin_name = explicit_name.lower()
    else:
        # Next, attempt to use the module (file) name if available
        module_name = getattr(cls, "__module__", "") or ""
        module_basename = module_name.rsplit(".", 1)[-1]
        try:
            import sys as _sys

            mod = _sys.modules.get(module_name)
            mod_file = getattr(mod, "__file__", "") if mod is not None else ""
        except Exception:
            mod_file = ""

        # Prefer module filename when the class is defined in a plugin module
        if mod_file and "plugins" in mod_file and module_basename:
            plugin_name = module_basename.lower()
        else:
            # Fallback to a snake_case of the class name
            plugin_name = _camel_to_snake(cls.__name__)

    if plugin_name in PLUGIN_REGISTRY:
        # Handle potential name collisions
        raise ValueError(f"Plugin with name '{plugin_name}' is already registered.")
    PLUGIN_REGISTRY[plugin_name] = cls
    return cls
