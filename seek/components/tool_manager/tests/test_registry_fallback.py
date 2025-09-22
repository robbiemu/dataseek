from typing import ClassVar

from seek.components.tool_manager.plugin_base import BaseTool
from seek.components.tool_manager.registry import PLUGIN_REGISTRY, register_plugin


def test_register_plugin_fallback_to_snake_case_when_name_missing():
    # Ensure a clean registry for this test
    PLUGIN_REGISTRY.clear()

    @register_plugin
    class MySearchTool(BaseTool):
        # Intentionally omit 'name' to force fallback
        # Avoid Pydantic override errors by annotating as ClassVar
        description: ClassVar[str] = "fallback case"

    assert "my_search_tool" in PLUGIN_REGISTRY
    assert PLUGIN_REGISTRY["my_search_tool"].__name__ == "MySearchTool"


def test_register_plugin_ignores_non_string_name_and_falls_back():
    # Ensure a clean registry for this test
    PLUGIN_REGISTRY.clear()

    @register_plugin
    class ToolWithWeirdName(BaseTool):
        # Simulate a non-string class attribute; registry should fall back
        name: ClassVar[object] = object()  # non-string, annotated
        description: ClassVar[str] = "non-string name"

    assert "tool_with_weird_name" in PLUGIN_REGISTRY
    assert PLUGIN_REGISTRY["tool_with_weird_name"].__name__ == "ToolWithWeirdName"
