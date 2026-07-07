"""Tests for the plugin/tool, prompt, fitness, and resilience changes.

Covers: plugin-tool unification, prompt externalization, fitness tool-node
gating, the top_p passthrough fix, the provenance guard, and the
endpoint-failure resilience (max_retries config + fitness
degrade-on-transient-error). Stock behavior is asserted to be unchanged
in every case.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import litellm
import pytest
import yaml

from seek.common.config import (
    StructuredSeekConfig,
    get_prompt,
    set_prompts_config,
)
from seek.common.models import FitnessReport
from seek.components.search_graph.nodes.fitness import fitness_node
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
    """An override replaces the roles it specifies; unmentioned roles keep bundled.

    Per-role replace (not deep per-key merge): specifying ``research`` in the
    override replaces the whole research role wholesale — sibling keys not
    re-set in the override are gone for that role (not leaked from the base).
    Roles not mentioned in the override keep their bundled prompts entirely.
    This matters because nodes read multiple keys per role together (research
    concatenates base_prompt + normal_prompt); deep-merge would leak the
    bundled normal_prompt into a custom research role.
    """
    import seek.common.config as cfg

    monkeypatch.setattr(cfg, "_prompts_config", None)
    monkeypatch.setattr(cfg, "_prompts_config_path", None)

    override = tmp_path / "prompts.yaml"
    # Override ONLY archive, with only base_prompt (no other keys)
    override.write_text(
        yaml.dump({"archive": {"base_prompt": "CUSTOM: {provenance} {characteristic}"}})
    )

    set_prompts_config(str(override))
    # The overridden role takes the new value
    assert get_prompt("archive", "base_prompt").startswith("CUSTOM:")
    # Roles NOT in the override keep their bundled prompts (the bug this guards
    # against: previously a partial override emptied every other role)
    assert get_prompt("fitness", "base_prompt"), "fitness prompt must not be empty"
    assert get_prompt("supervisor", "base_prompt"), "supervisor prompt must not be empty"
    assert get_prompt("research", "base_prompt"), "research prompt must not be empty"


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


def test_bundled_prompts_resolve_from_package_not_cwd(tmp_path, monkeypatch):
    """The bundled config/prompts.yaml is found even when CWD has no config/.

    Regression guard: when seek is invoked from a downstream repo CWD (so
    ./plugins is globbed but ./config does not exist), the bundled prompts must
    still resolve — via a package-relative path, not a CWD-relative one. Without
    this, omitted roles get empty base prompts because there's no bundled file
    to fall back onto.
    """
    import seek.common.config as cfg

    monkeypatch.setattr(cfg, "_prompts_config", None)
    monkeypatch.setattr(cfg, "_prompts_config_path", None)

    # Run from a CWD with no config/ dir
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "config").exists()

    research = get_prompt("research", "base_prompt")
    fitness = get_prompt("fitness", "base_prompt")
    assert research, "research prompt must resolve from package, not CWD"
    assert fitness, "fitness prompt must resolve from package, not CWD"


def test_bundled_seek_config_resolves_from_package_not_cwd(tmp_path, monkeypatch):
    """The bundled config/seek_config.yaml is found even when CWD has no config/."""
    from seek.common.config import load_seek_config

    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "config").exists()

    sc = load_seek_config()
    assert sc.get("model_defaults", {}).get(
        "model"
    ), "seek config must resolve from package, not CWD"


# -------------------------
# C3: fitness tool-node gating
# -------------------------


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


# -------------------------
# Endpoint-failure resilience (max_retries config + fitness degrade guard)
# -------------------------


def _fitness_state(provenance: str = "researched") -> dict:
    """Minimal state that drives fitness_node down the no-tools fallback path."""
    return {
        "messages": [MagicMock()],
        "current_sample_provenance": provenance,
        "research_findings": ["# Data Prospecting Report\nsample content"],
        "current_task": {
            "characteristic": "test_characteristic",
            "topic": "test_topic",
        },
        "strategy_block": "",
        "mission_config": {},
    }


def _patch_fitness_llm(monkeypatch):
    """Patch fitness.create_llm + create_agent_runnable to isolate invoke behavior."""
    monkeypatch.setattr(
        "seek.components.search_graph.nodes.fitness.create_llm", lambda _role: MagicMock()
    )
    return monkeypatch


def test_fitness_node_degrades_on_connection_error(monkeypatch):
    """A transport connection error degrades to a REJECTED report, not a raise."""
    _patch_fitness_llm(monkeypatch)
    err = litellm.APIConnectionError(message="Connection error.", llm_provider="openai", model="x")
    with patch("seek.components.search_graph.nodes.fitness.create_agent_runnable") as mock_ar:
        mock_ar.return_value.invoke.side_effect = err
        result = fitness_node(_fitness_state())

    report = result["fitness_report"]
    assert isinstance(report, FitnessReport)
    assert report.passed is False
    assert "endpoint unavailable" in report.reason.lower()


def test_fitness_node_degrades_on_503_queue_full(monkeypatch):
    """A 503 queue-full (surfaced as InternalServerError) degrades to REJECTED."""
    _patch_fitness_llm(monkeypatch)
    err = litellm.InternalServerError(
        message="request queue is full", llm_provider="openai", model="x"
    )
    with patch("seek.components.search_graph.nodes.fitness.create_agent_runnable") as mock_ar:
        mock_ar.return_value.invoke.side_effect = err
        result = fitness_node(_fitness_state())

    report = result["fitness_report"]
    assert isinstance(report, FitnessReport)
    assert report.passed is False
    assert "endpoint unavailable" in report.reason.lower()


def test_fitness_node_does_not_swallow_4xx_errors(monkeypatch):
    """4xx errors (auth/validation/config) must still propagate, not be masked as REJECTED."""
    _patch_fitness_llm(monkeypatch)
    err = litellm.BadRequestError(message="invalid model", model="x", llm_provider="openai")
    with patch("seek.components.search_graph.nodes.fitness.create_agent_runnable") as mock_ar:
        mock_ar.return_value.invoke.side_effect = err
        with pytest.raises(litellm.BadRequestError):
            fitness_node(_fitness_state())


def test_create_llm_passes_max_retries(monkeypatch):
    """max_retries from config flows through to ChatLiteLLM (default + node override)."""
    _set_config(
        monkeypatch,
        {
            "model_defaults": {
                "model": "x",
                "temperature": 0.1,
                "max_tokens": 100,
                "max_retries": 3,
            }
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("supervisor")
        assert mock_llm.call_args.kwargs["max_retries"] == 3

    # Node-level override wins
    _set_config(
        monkeypatch,
        {
            "model_defaults": {
                "model": "x",
                "temperature": 0.1,
                "max_tokens": 100,
                "max_retries": 3,
            },
            "mission_plan": {"nodes": [{"name": "fitness", "model": "f", "max_retries": 5}]},
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("fitness")
        assert mock_llm.call_args.kwargs["max_retries"] == 5


def test_create_llm_defaults_streaming_true(monkeypatch):
    """With no streaming key in config, create_llm defaults streaming to True.

    Streaming is the root-cause fix for idle-socket resets on long local-server
    generations, so the safe default is on; cloud endpoints accept it too.
    """
    _set_config(
        monkeypatch,
        {
            "model_defaults": {
                "model": "x",
                "temperature": 0.1,
                "max_tokens": 100,
            }
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("supervisor")
        assert mock_llm.call_args.kwargs["streaming"] is True


def test_create_llm_streaming_passthrough_and_override(monkeypatch):
    """streaming is configurable from model_defaults and overridable per node."""
    # model_defaults.streaming=False disables it
    _set_config(
        monkeypatch,
        {
            "model_defaults": {
                "model": "x",
                "temperature": 0.1,
                "max_tokens": 100,
                "streaming": False,
            }
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("supervisor")
        assert mock_llm.call_args.kwargs["streaming"] is False

    # node-level override wins over model_defaults
    _set_config(
        monkeypatch,
        {
            "model_defaults": {
                "model": "x",
                "temperature": 0.1,
                "max_tokens": 100,
                "streaming": False,
            },
            "mission_plan": {"nodes": [{"name": "fitness", "model": "f", "streaming": True}]},
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("fitness")
        assert mock_llm.call_args.kwargs["streaming"] is True


def test_create_llm_merges_model_kwargs_without_clobbering_top_p(monkeypatch):
    """A config model_kwargs block reaches the server and coexists with top_p.

    Reasoning models (e.g. Qwen3) emit text in reasoning_content and leave
    content empty unless chat_template_kwargs.enable_thinking=false is sent.
    That key flows through model_kwargs. Previously create_llm overwrote
    model_kwargs with only top_p, so the block never reached the server.
    """
    _set_config(
        monkeypatch,
        {
            "model_defaults": {
                "model": "x",
                "temperature": 0.1,
                "max_tokens": 100,
                "top_p": 1.0,
                "model_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
            }
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("supervisor")
        assert mock_llm.call_args.kwargs["model_kwargs"] == {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_p": 1.0,
        }


def test_create_llm_node_model_kwargs_deep_merges(monkeypatch):
    """Node-level model_kwargs merges over model_defaults' (node keys win)."""
    _set_config(
        monkeypatch,
        {
            "model_defaults": {
                "model": "x",
                "model_kwargs": {"a": 1, "b": 2},
            },
            "mission_plan": {
                "nodes": [{"name": "fitness", "model": "f", "model_kwargs": {"b": 99, "c": 3}}]
            },
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("fitness")
        assert mock_llm.call_args.kwargs["model_kwargs"] == {"a": 1, "b": 99, "c": 3}


def test_create_llm_omits_max_tokens_when_not_configured(monkeypatch):
    """max_tokens is opt-in: omitted entirely when not in config.

    A hardcoded cap truncates reasoning models mid-thought (the answer never
    gets emitted because the budget runs out during the reasoning phase).
    Left unset, the server/model decides the output budget.
    """
    _set_config(
        monkeypatch,
        {
            "model_defaults": {
                "model": "x",
                "temperature": 0.1,
                # no max_tokens
            }
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("supervisor")
        assert "max_tokens" not in mock_llm.call_args.kwargs


def test_create_llm_passes_max_tokens_when_configured(monkeypatch):
    """When max_tokens IS set (global or per-node), it flows through."""
    # From model_defaults
    _set_config(
        monkeypatch,
        {"model_defaults": {"model": "x", "temperature": 0.1, "max_tokens": 2000}},
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("supervisor")
        assert mock_llm.call_args.kwargs["max_tokens"] == 2000

    # Per-node override
    _set_config(
        monkeypatch,
        {
            "model_defaults": {"model": "x", "temperature": 0.1},
            "mission_plan": {"nodes": [{"name": "fitness", "model": "f", "max_tokens": 4096}]},
        },
    )
    with patch("seek.components.search_graph.nodes.utils.ChatLiteLLM") as mock_llm:
        create_llm("fitness")
        assert mock_llm.call_args.kwargs["max_tokens"] == 4096
