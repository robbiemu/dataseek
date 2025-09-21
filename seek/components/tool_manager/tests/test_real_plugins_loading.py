from seek.components.tool_manager.registry import PLUGIN_REGISTRY
from seek.components.tool_manager.tool_manager import ToolManager


def test_loading_builtin_plugins_does_not_error_and_registers(capsys):
    # Reset registry then load from the real plugins dir
    PLUGIN_REGISTRY.clear()

    ToolManager(plugin_dir="plugins")
    captured = capsys.readouterr()

    # No generic load errors should be printed for the built-in plugins
    assert "Error loading plugin" not in captured.out

    # Verify expected plugins are registered
    for name in ("arxiv_search", "wikipedia_search", "url_to_markdown", "file_saver"):
        assert name in PLUGIN_REGISTRY


def test_toolsets_build_with_builtin_plugins():
    # Build toolsets directly from a minimal mission config using builtin plugins
    PLUGIN_REGISTRY.clear()
    tm = ToolManager(plugin_dir="plugins")

    mission_config = {
        "tool_configs": {
            "arxiv_search": {"roles": ["research"], "max_docs": 2},
            "wikipedia_search": {"roles": ["research"], "max_docs": 2},
            "url_to_markdown": {"roles": ["research"], "timeout": 5},
            "file_saver": {"roles": ["archive"], "output_path": "datasets/tmp"},
        }
    }

    import asyncio

    toolsets = asyncio.run(tm.get_toolsets_for_mission(mission_config))

    assert "research" in toolsets
    assert len(toolsets["research"]) >= 3
    assert "archive" in toolsets
    assert len(toolsets["archive"]) >= 1
