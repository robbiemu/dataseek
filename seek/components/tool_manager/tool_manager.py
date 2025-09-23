import importlib.util
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from seek.components.tool_manager.plugin_base import BaseTool
from seek.components.tool_manager.registry import PLUGIN_REGISTRY


class ToolManager:
    """Manages discovery, loading, and lifecycle of tools."""

    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self._load_plugins()

    def _load_plugins(self) -> None:
        """Dynamically discovers and imports plugins."""
        if not self.plugin_dir.is_dir():
            print(f"Warning: Plugin directory '{self.plugin_dir}' not found.")
            return

        for file_path in self.plugin_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                    print(f"Successfully loaded plugin: {module_name}")
                except Exception as e:
                    print(f"Error loading plugin {module_name}: {e}")

    async def get_toolsets_for_mission(
        self, mission_config: dict[str, Any]
    ) -> dict[str, list[BaseTool]]:
        """
        Instantiates and prepares toolsets for a given mission, indexed by role.
        """
        toolsets: dict[str, list[BaseTool]] = {}
        tool_configs = mission_config.get("tool_configs", {})

        for tool_name, config_data in tool_configs.items():
            plugin_name = tool_name.lower()
            if plugin_name not in PLUGIN_REGISTRY:
                print(f"Warning: Tool '{plugin_name}' is not registered.")
                continue

            # Work on a shallow copy to avoid mutating the mission_config
            cfg = dict(config_data)

            if "roles" not in cfg:
                raise ValueError(f"Tool '{plugin_name}' in mission config must have a 'roles' key.")
            roles = cfg.pop("roles")

            # No special injections currently needed for built-in plugins.
            tool_class = PLUGIN_REGISTRY[plugin_name]
            validated_config: BaseModel | None = None

            config_attr = getattr(tool_class, "ConfigSchema", None)
            if config_attr is not None:
                try:
                    validated_config = config_attr(**cfg)
                except ValidationError as e:
                    print(f"Error validating config for tool '{plugin_name}': {e}")
                    continue

            try:
                tool_instance = tool_class(
                    name=tool_class.name if hasattr(tool_class, "name") else tool_name,
                    description=(
                        tool_class.description if hasattr(tool_class, "description") else "Tool"
                    ),
                    config=validated_config,
                )
                await tool_instance.setup()
                for role in roles:
                    if role not in toolsets:
                        toolsets[role] = []
                    toolsets[role].append(tool_instance)
                print(f"Successfully instantiated tool: {plugin_name} for roles: {roles}")
            except Exception as e:
                print(f"Error instantiating tool '{plugin_name}': {e}")

        return toolsets

    async def teardown_tools(self, toolsets: dict[str, list[BaseTool]]) -> None:
        """Calls the teardown method on a dictionary of tools."""
        # De-duplicate by object identity to avoid pydantic hash issues
        seen: set[int] = set()
        unique_tools: list[BaseTool] = []
        for tools in toolsets.values():
            for tool in tools:
                oid = id(tool)
                if oid in seen:
                    continue
                seen.add(oid)
                unique_tools.append(tool)

        for tool in unique_tools:
            try:
                await tool.teardown()
            except Exception as e:
                print(f"Error during teardown for tool '{tool.name}': {e}")
