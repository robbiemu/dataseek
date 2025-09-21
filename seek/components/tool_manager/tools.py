"Tool discovery and role-based tool scoping for the Data Seek Agent."

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import BaseTool

from .file_tools import write_file
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


def get_tools_for_role(role: str) -> list[BaseTool]:
    """Return tools intended for a specific role."""
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
    return [all_tools[tool_name] for tool_name in tool_names_for_role if tool_name in all_tools]
