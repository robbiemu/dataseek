import tempfile
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pydantic import BaseModel

from seek.components.tool_manager.plugin_base import BaseSearchTool
from seek.components.tool_manager.registry import PLUGIN_REGISTRY, register_plugin
from seek.components.tool_manager.tool_manager import ToolManager


@pytest.fixture
def plugin_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def clear_plugin_registry():
    PLUGIN_REGISTRY.clear()


class TestPluginSystem:
    def test_tool_manager_discovery(self, plugin_dir: Path, capsys):
        """Test that the ToolManager correctly discovers and loads plugins."""
        # Create a valid plugin file
        valid_plugin_content = """
from seek.components.tool_manager.plugin_base import BaseTool
from typing import ClassVar
from seek.components.tool_manager.registry import register_plugin

@register_plugin
class ValidPlugin(BaseTool):
    name: ClassVar[str] = "valid_plugin"
    description: ClassVar[str] = "valid plugin tool"
"""
        (plugin_dir / "valid_plugin.py").write_text(valid_plugin_content)

        # Create a plugin with a syntax error
        invalid_plugin_content = """
from seek.components.tool_manager.plugin_base import BaseTool
from seek.components.tool_manager.registry import register_plugin

@register_plugin
class InvalidPlugin(BaseTool):
    name = "invalid_plugin"
    some_syntax_error
"""
        (plugin_dir / "invalid_plugin.py").write_text(invalid_plugin_content)

        # Create a file that is not a plugin
        (plugin_dir / "not_a_plugin.txt").write_text("hello")

        # Instantiate the ToolManager
        ToolManager(plugin_dir=str(plugin_dir))

        # Check that the valid plugin is registered
        assert "valid_plugin" in PLUGIN_REGISTRY

        # Check that the invalid plugin is not registered
        assert "invalid_plugin" not in PLUGIN_REGISTRY

        # Check the output for warnings
        captured = capsys.readouterr()
        assert "Successfully loaded plugin: valid_plugin" in captured.out
        assert "Error loading plugin invalid_plugin" in captured.out

    def test_get_toolsets_for_mission(self, plugin_dir: Path):
        """Test that get_toolsets_for_mission correctly prepares toolsets."""
        import asyncio

        # Create a dummy plugin
        @register_plugin
        class MyTool(BaseSearchTool):
            name: ClassVar[str] = "my_tool"
            description: ClassVar[str] = "dummy search tool"

            async def search(self, _query: str, _num_results: int = 5) -> list[dict[str, Any]]:
                return [{"result": "foo"}]

        # Valid mission config
        mission_config = {"tool_configs": {"my_tool": {"roles": ["research"]}}}

        tm = ToolManager(plugin_dir=str(plugin_dir))
        toolsets = asyncio.run(tm.get_toolsets_for_mission(mission_config))

        assert "research" in toolsets
        assert len(toolsets["research"]) == 1
        assert isinstance(toolsets["research"][0], MyTool)

        # Test missing roles
        mission_config_no_roles = {"tool_configs": {"my_tool": {}}}
        with pytest.raises(ValueError, match="must have a 'roles' key"):
            asyncio.run(tm.get_toolsets_for_mission(mission_config_no_roles))

        # Test unregistered tool
        mission_config_unregistered = {
            "tool_configs": {"unregistered_tool": {"roles": ["research"]}}
        }
        toolsets_unregistered = asyncio.run(
            tm.get_toolsets_for_mission(mission_config_unregistered)
        )
        assert not toolsets_unregistered

    def test_config_validation_and_teardown(self, plugin_dir: Path):
        """Validate ConfigSchema enforcement and teardown lifecycle handling."""
        import asyncio

        setup_calls: dict[str, int] = {"count": 0}
        teardown_calls: dict[str, int] = {"count": 0}

        class MyCfg(BaseModel):  # type: ignore[name-defined]
            threshold: int

        @register_plugin
        class CfgTool(BaseSearchTool):
            name: ClassVar[str] = "cfg_tool"
            description: ClassVar[str] = "cfg search tool"
            ConfigSchema: ClassVar[type[BaseModel]] = MyCfg

            async def setup(self) -> None:
                setup_calls["count"] += 1

            async def teardown(self) -> None:
                teardown_calls["count"] += 1

            async def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
                config_cast = cast(MyCfg, self.config)
                return [{"ok": True, "q": query, "n": num_results, "t": config_cast.threshold}]

        tm = ToolManager(plugin_dir=str(plugin_dir))

        # Valid config -> tool is instantiated and indexed under both roles
        mission_config_ok = {
            "tool_configs": {"cfg_tool": {"roles": ["research", "archive"], "threshold": 7}}
        }
        toolsets_ok = asyncio.run(tm.get_toolsets_for_mission(mission_config_ok))
        assert "research" in toolsets_ok
        assert "archive" in toolsets_ok
        assert any(isinstance(t, CfgTool) for t in toolsets_ok["research"])  # type: ignore[name-defined]
        assert any(isinstance(t, CfgTool) for t in toolsets_ok["archive"])  # type: ignore[name-defined]
        # Setup called exactly once even if tool is bound to multiple roles
        assert setup_calls["count"] == 1

        # Teardown de-duplicates instances across roles
        asyncio.run(tm.teardown_tools(toolsets_ok))
        assert teardown_calls["count"] == 1

        # Invalid config -> validation error causes skip (no tool instantiated)
        mission_config_bad = {
            "tool_configs": {"cfg_tool": {"roles": ["research"], "threshold": "bad"}}
        }
        toolsets_bad = asyncio.run(tm.get_toolsets_for_mission(mission_config_bad))
        assert toolsets_bad == {}
