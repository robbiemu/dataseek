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
    for name in ("arxiv_search", "wikipedia_search", "url_to_markdown"):
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
        }
    }

    import asyncio

    toolsets = asyncio.run(tm.get_toolsets_for_mission(mission_config))

    assert "research" in toolsets
    assert len(toolsets["research"]) >= 3
    # No archive plugins are required; archive node writes procedurally
    assert "archive" not in toolsets or len(toolsets["archive"]) == 0


def test_no_archive_plugins_needed_for_procedural_archive():
    # Reset registry then load from the real plugins dir
    PLUGIN_REGISTRY.clear()
    tm = ToolManager(plugin_dir="plugins")

    mission_config = {
        "output_paths": {"samples_path": "examples/datasets/mac_ai_corpus/samples"},
        "tool_configs": {
            # Only research plugins configured
            "arxiv_search": {"roles": ["research"], "max_docs": 2},
        },
    }

    import asyncio

    toolsets = asyncio.run(tm.get_toolsets_for_mission(mission_config))

    # Archive toolset may be absent or empty; that's acceptable
    assert "archive" not in toolsets or len(toolsets["archive"]) == 0
