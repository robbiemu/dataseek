"""Tests for the Tier B Lean proof-trace integration (challenges C1-C4).

Covers: plugin-tool unification, prompt externalization, the LeanTraceState
subclass + fitness tool-node gating, the top_p passthrough fix, and the
provenance guard. Stock behavior is asserted to be unchanged in every case.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from seek.common.config import (
    StructuredSeekConfig,
    get_prompt,
    set_prompts_config,
)
from seek.components.mission_runner.state import DataSeekState, LeanTraceState
from seek.components.search_graph.nodes.utils import create_llm
from seek.components.tool_manager.registry import PLUGIN_REGISTRY, register_plugin
from seek.components.tool_manager.tools import get_plugin_tools_for_role, get_tools_for_role

# -------------------------
# C1: plugin-tool unification
# -------------------------


def test_get_tools_for_role_stock_unchanged_without_mission_config():
    """Built-in research tools are returned unchanged when no mission_config is given."""
    tools = get_tools_for_role("research")
    names = [t.name for t in tools]
    assert "web_search" in names
    assert "url_to_markdown" in names


def test_get_plugin_tools_for_role_empty_when_no_config():
    assert get_plugin_tools_for_role("research", None) == []
    assert get_plugin_tools_for_role("research", {}) == []
    assert get_plugin_tools_for_role("research", {"tool_configs": {}}) == []


def _register_dummy_lean_check_plugin() -> None:
    """Register a minimal plugin named 'lean_check' into PLUGIN_REGISTRY.

    The class name snake-cases to 'lean_check' (the registry's fallback
    resolution path), since class-level Pydantic field descriptors prevent
    the explicit name attribute from being read at decoration time.
    """
    from seek.components.tool_manager.plugin_base import BaseUtilityTool

    @register_plugin
    class LeanCheck(BaseUtilityTool):
        name: str = "lean_check"
        description: str = "Runs PyPantograph to verify a Lean proof trace."

        async def execute(self, **kwargs: Any) -> dict[str, Any]:
            return {"status": "ok"}

    return None


def test_get_plugin_tools_for_role_instantiates_mapped_plugin():
    """A plugin registered and mapped via tool_configs[tool].roles is returned for that role."""
    saved = dict(PLUGIN_REGISTRY)
    PLUGIN_REGISTRY.clear()
    try:
        _register_dummy_lean_check_plugin()
        mission_config = {
            "tool_configs": {
                "lean_check": {"roles": ["fitness"]},
                "write_file": {"roles": ["archive"]},
            }
        }

        fitness_plugins = get_plugin_tools_for_role("fitness", mission_config)
        archive_plugins = get_plugin_tools_for_role("archive", mission_config)
        research_plugins = get_plugin_tools_for_role("research", mission_config)

        assert [t.name for t in fitness_plugins] == ["lean_check"]
        assert archive_plugins == []  # write_file is a builtin, not in PLUGIN_REGISTRY
        assert research_plugins == []  # lean_check is not mapped to research
    finally:
        PLUGIN_REGISTRY.clear()
        PLUGIN_REGISTRY.update(saved)


def test_get_tools_for_role_merges_builtin_and_plugin():
    """get_tools_for_role returns builtin tools plus mapped plugins."""
    saved = dict(PLUGIN_REGISTRY)
    PLUGIN_REGISTRY.clear()
    try:
        _register_dummy_lean_check_plugin()
        mission_config = {"tool_configs": {"lean_check": {"roles": ["fitness"]}}}
        tools = get_tools_for_role("fitness", mission_config)
        names = [t.name for t in tools]
        assert "lean_check" in names
    finally:
        PLUGIN_REGISTRY.clear()
        PLUGIN_REGISTRY.update(saved)


# -------------------------
# C2: prompt externalization
# -------------------------


def test_set_prompts_config_override_takes_effect(tmp_path, monkeypatch):
    """An override path set via set_prompts_config replaces the bundled prompts."""
    import seek.common.config as cfg

    monkeypatch.setattr(cfg, "_prompts_config", None)
    monkeypatch.setattr(cfg, "_prompts_config_path", None)

    override = tmp_path / "prompts.yaml"
    override.write_text(
        yaml.dump({"archive": {"base_prompt": "CUSTOM: {provenance} {characteristic}"}})
    )

    set_prompts_config(str(override))
    assert get_prompt("archive", "base_prompt").startswith("CUSTOM:")


def test_set_prompts_config_reset_restores_default(tmp_path, monkeypatch):
    """Resetting to None falls back to the bundled config."""
    import seek.common.config as cfg

    monkeypatch.setattr(cfg, "_prompts_config", None)

    override = tmp_path / "prompts.yaml"
    override.write_text(yaml.dump({"archive": {"base_prompt": "CUSTOM"}}))
    set_prompts_config(str(override))
    assert get_prompt("archive", "base_prompt") == "CUSTOM"

    set_prompts_config(None)
    bundled = get_prompt("archive", "base_prompt")
    assert bundled != "CUSTOM"
    assert "Library Cataloger" in bundled


# -------------------------
# C3: LeanTraceState + fitness tool-node gating
# -------------------------


def test_lean_trace_state_is_superset_of_data_seek_state():
    """LeanTraceState must add the Lean fields while inheriting all stock fields."""
    stock_keys = set(DataSeekState.__annotations__.keys())
    lean_keys = set(LeanTraceState.__annotations__.keys())
    new_keys = lean_keys - stock_keys
    assert new_keys == {
        "theorem",
        "proof_state",
        "trace",
        "verification",
        "category",
        "provenance",
    }
    # All stock keys are still present
    assert stock_keys.issubset(lean_keys)


def test_build_graph_has_no_fitness_tools_node_for_stock_mission():
    """A mission with no fitness plugins keeps the stock fitness → supervisor edge."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    from seek.components.search_graph.graph import build_graph

    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer, mission_config={})
        node_names = set(app.get_graph().nodes.keys())
        assert "fitness_tools" not in node_names
        assert "fitness" in node_names


def test_build_graph_adds_fitness_tools_node_when_plugins_mapped():
    """A mission with fitness plugins gets a fitness_tools ToolNode in the graph."""
    saved = dict(PLUGIN_REGISTRY)
    PLUGIN_REGISTRY.clear()
    try:
        _register_dummy_lean_check_plugin()

        from langgraph.checkpoint.sqlite import SqliteSaver

        from seek.components.search_graph.graph import build_graph

        mission_config = {"tool_configs": {"lean_check": {"roles": ["fitness"]}}}
        with SqliteSaver.from_conn_string(":memory:") as checkpointer:
            app = build_graph(checkpointer=checkpointer, mission_config=mission_config)
            node_names = set(app.get_graph().nodes.keys())
            assert "fitness_tools" in node_names
    finally:
        PLUGIN_REGISTRY.clear()
        PLUGIN_REGISTRY.update(saved)


# -------------------------
# C4: top_p passthrough + provenance guard
# -------------------------


def _set_config(monkeypatch, config: dict) -> None:
    import seek.common.config as cfg

    monkeypatch.setattr(cfg, "_active_seek_config", StructuredSeekConfig(config))


def test_create_llm_omits_top_p_when_unset(monkeypatch):
    """When top_p is not configured, it is not forwarded (stock behavior)."""
    _set_config(
        monkeypatch,
        {"model_defaults": {"model": "x", "temperature": 0.1, "max_tokens": 100}},
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("supervisor")
        kwargs = mock_llm.call_args.kwargs
        assert "top_p" not in kwargs
        assert "model_kwargs" not in kwargs


def test_create_llm_forwards_top_p_via_model_kwargs(monkeypatch):
    """top_p is delivered through model_kwargs so it reaches the provider payload."""
    _set_config(
        monkeypatch,
        {
            "model_defaults": {
                "model": "mistral/labs-leanstral-1-5",
                "temperature": 0,
                "max_tokens": 64,
                "top_p": 1.0,
            }
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("supervisor")
        kwargs = mock_llm.call_args.kwargs
        assert kwargs["model_kwargs"] == {"top_p": 1.0}
        assert kwargs["model"] == "mistral/labs-leanstral-1-5"


def test_create_llm_node_config_overrides_default_top_p(monkeypatch):
    """Node-specific top_p wins over model_defaults."""
    _set_config(
        monkeypatch,
        {
            "model_defaults": {"model": "x", "temperature": 0.1, "max_tokens": 100, "top_p": 0.9},
            "mission_plan": {"nodes": [{"name": "fitness", "model": "leanstral", "top_p": 1.0}]},
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("fitness")
        kwargs = mock_llm.call_args.kwargs
        assert kwargs["model"] == "leanstral"
        assert kwargs["model_kwargs"] == {"top_p": 1.0}


def test_archive_provenance_guard_blocks_synthetic_under_zero_budget(monkeypatch):
    """A synthetic sample reaching archive under synthetic_budget=0 raises."""
    _set_config(
        monkeypatch,
        {
            "model_defaults": {"model": "x", "temperature": 0.1, "max_tokens": 100},
            "synthetic_budget": 0,
        },
    )
    from seek.components.search_graph.nodes.archive import archive_node

    state = {
        "messages": [],
        "current_sample_provenance": "synthetic",
        "research_findings": ["# Data Prospecting Report\n..."],
    }
    with pytest.raises(AssertionError, match="Provenance guard"):
        archive_node(state)


def test_archive_provenance_guard_passes_researched_under_zero_budget(monkeypatch):
    """A researched sample under budget=0 passes the guard (fails downstream, not at guard)."""
    _set_config(
        monkeypatch,
        {
            "model_defaults": {"model": "x", "temperature": 0.1, "max_tokens": 100},
            "synthetic_budget": 0,
        },
    )
    from seek.components.search_graph.nodes.archive import archive_node

    state = {
        "messages": [],
        "current_sample_provenance": "researched",
        "research_findings": ["# Data Prospecting Report\n..."],
        "mission_config": {"output_paths": {"base_path": "/tmp/dataseek_test"}},
    }
    # Guard must not raise; downstream LLM call is expected to fail without a real model.
    with patch("seek.components.search_graph.nodes.archive.create_llm") as mock_llm:
        mock_llm.return_value = MagicMock()
        with patch("seek.components.search_graph.nodes.archive.create_agent_runnable") as mock_ar:
            mock_runnable = MagicMock()
            mock_runnable.invoke.return_value = MagicMock(content="entry")
            mock_ar.return_value = mock_runnable
            with patch("seek.components.search_graph.nodes.archive.write_file") as mock_wf:
                mock_wf.invoke.return_value = {"status": "ok", "bytes_written": 1}
                with patch(
                    "seek.components.search_graph.nodes.archive.append_to_pedigree"
                ) as mock_ap:
                    mock_ap.return_value = {"status": "ok"}
                    result = archive_node(state)
                    # Guard passed, archive completed
                    assert "samples_generated" in result


def test_archive_provenance_guard_inactive_when_budget_nonzero(monkeypatch):
    """When synthetic_budget > 0, synthetic samples are archived normally."""
    _set_config(
        monkeypatch,
        {
            "model_defaults": {"model": "x", "temperature": 0.1, "max_tokens": 100},
            "synthetic_budget": 0.5,
        },
    )
    from seek.components.search_graph.nodes.archive import archive_node

    state = {
        "messages": [],
        "current_sample_provenance": "synthetic",
        "research_findings": ["# Data Prospecting Report\n..."],
        "mission_config": {"output_paths": {"base_path": "/tmp/dataseek_test"}},
    }
    with patch("seek.components.search_graph.nodes.archive.create_llm") as mock_llm:
        mock_llm.return_value = MagicMock()
        with patch("seek.components.search_graph.nodes.archive.create_agent_runnable") as mock_ar:
            mock_runnable = MagicMock()
            mock_runnable.invoke.return_value = MagicMock(content="entry")
            mock_ar.return_value = mock_runnable
            with patch("seek.components.search_graph.nodes.archive.write_file") as mock_wf:
                mock_wf.invoke.return_value = {"status": "ok", "bytes_written": 1}
                with patch(
                    "seek.components.search_graph.nodes.archive.append_to_pedigree"
                ) as mock_ap:
                    mock_ap.return_value = {"status": "ok"}
                    # Should NOT raise
                    result = archive_node(state)
                    assert result["samples_generated"] >= 1
