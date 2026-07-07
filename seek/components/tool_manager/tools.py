"Tool discovery and role-based tool scoping for the Data Seek Agent."

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool

from .file_tools import write_file
from .registry import PLUGIN_REGISTRY
from .search_tools import (
    arxiv_get_content,
    arxiv_search,
    create_web_search_tool,
    wikipedia_get_content,
    wikipedia_search,
)
from .web_tools import _HAVE_LIBCRAWLER, url_to_markdown

if _HAVE_LIBCRAWLER:
    from .web_tools import documentation_crawler

# -------------------------
# Tool discovery / role scoping
# -------------------------


def get_available_tools() -> list[Callable]:
    """Return actual tool callables available in this environment."""
    # Create the web_search tool with the configured provider
    web_search = create_web_search_tool()
    core_tools = [
        url_to_markdown,
        write_file,
        web_search,
        arxiv_search,
        arxiv_get_content,
        wikipedia_search,
        wikipedia_get_content,
    ]
    optional = [documentation_crawler] if _HAVE_LIBCRAWLER else []
    return core_tools + optional


def get_plugin_tools_for_role(
    role: str, mission_config: dict[str, Any] | None = None
) -> list[BaseTool]:
    """Instantiate PLUGIN_REGISTRY tools mapped to a role via mission_config.tool_configs.

    Returns an empty list when no plugins are registered or mapped, so stock
    missions are unaffected.
    """
    if not mission_config:
        return []

    tool_configs = mission_config.get("tool_configs")
    if not isinstance(tool_configs, dict) or not tool_configs:
        return []

    role_lower = (role or "").lower()
    tools: list[BaseTool] = []

    for tool_name, cfg in tool_configs.items():
        if not isinstance(cfg, dict):
            continue
        roles = cfg.get("roles", [])
        if not isinstance(roles, list):
            continue
        if role_lower not in [str(r).lower() for r in roles]:
            continue

        plugin_name = str(tool_name).lower()
        tool_class = PLUGIN_REGISTRY.get(plugin_name)
        if tool_class is None:
            continue

        try:
            name_attr = getattr(tool_class, "name", None) or str(tool_name)
            desc_attr = getattr(tool_class, "description", None) or "Tool"
            tools.append(tool_class(name=str(name_attr), description=str(desc_attr)))
        except Exception as exc:
            print(f"Warning: could not instantiate plugin '{plugin_name}': {exc}")

    return tools


def get_tools_for_role(role: str, mission_config: dict[str, Any] | None = None) -> list[BaseTool]:
    """Return tools intended for a specific role.

    Built-in tools are always available per the role mapping. When
    mission_config is provided, PLUGIN_REGISTRY tools mapped to the role via
    tool_configs[tool].roles are appended, making @register_plugin live for
    node agents.
    """
    role = (role or "").lower()

    # Create the web_search tool with the configured provider
    web_search = create_web_search_tool()

    # Build tool registry, guarding optional crawler tool
    all_tools = {
        "web_search": web_search,
        "arxiv_search": arxiv_search,
        "arxiv_get_content": arxiv_get_content,
        "wikipedia_search": wikipedia_search,
        "wikipedia_get_content": wikipedia_get_content,
        "url_to_markdown": url_to_markdown,
        "write_file": write_file,
    }
    if _HAVE_LIBCRAWLER:
        all_tools["documentation_crawler"] = documentation_crawler  # type: ignore[name-defined]

    # Define which tools are available for each role
    role_mapping = {
        "research": [
            "web_search",
            "arxiv_search",
            "arxiv_get_content",
            "wikipedia_search",
            "wikipedia_get_content",
            "url_to_markdown",
        ],
        "archive": ["write_file"],
        "supervisor": [],
        "fitness": [],
        "synthetic": [],
    }

    if _HAVE_LIBCRAWLER:
        role_mapping["research"].append("documentation_crawler")

    tool_names_for_role = role_mapping.get(role, [])
    # When a mission explicitly maps tools to this role via tool_configs, the
    # mission becomes authoritative: only keep the hardcoded builtins it also
    # lists. This lets a mission drop the default web/search plugins (e.g. a
    # mathlib mission that only wants its own Lean plugins). Missions that do
    # not configure tools for a role keep the full stock builtin set.
    if mission_config:
        tool_configs = mission_config.get("tool_configs")
        if isinstance(tool_configs, dict):
            role_lower = (role or "").lower()
            mapped = {
                name.lower()
                for name, cfg in tool_configs.items()
                if isinstance(cfg, dict)
                and role_lower in [str(r).lower() for r in cfg.get("roles", [])]
            }
            if mapped:
                tool_names_for_role = [n for n in tool_names_for_role if n in mapped]
    builtin_tools = [
        all_tools[tool_name] for tool_name in tool_names_for_role if tool_name in all_tools
    ]
    plugin_tools = get_plugin_tools_for_role(role, mission_config)
    return builtin_tools + plugin_tools
