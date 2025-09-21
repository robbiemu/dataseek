# Tool Manager

## Purpose
Implements production-ready tools for data acquisition and processing used by agents during research and synthesis phases. Provides a unified interface for various data sources including web search, academic papers, wiki content, and local file operations. Handles rate limiting, robots.txt compliance, and content formatting to ensure reliable and ethical data collection.

**Why this might change**: Updates are likely when integrating new data sources (e.g., social media APIs, proprietary databases), adding content processing features (e.g., image analysis, multimedia extraction), or implementing advanced filtering (e.g., content quality scoring). Changes may also address new compliance requirements or performance optimizations for high-rate data collection.

## Plugin System

The plugin system enables extensible tools via Python modules in the `plugins/` directory. Tools are discovered dynamically, registered with `@register_plugin`, and must inherit from `BaseTool`, `BaseSearchTool`, or `BaseUtilityTool` in `plugin_base.py`.

Key components:
- `registry.py`: Central `PLUGIN_REGISTRY` dictionary and registration decorator.
- `tool_manager.py`: Loads plugins from `plugins/*.py`, validates configs, instantiates tools, and manages lifecycle via `get_toolsets_for_mission` and `teardown_tools`.
- `plugin_base.py`: Abstract base classes with optional `ConfigSchema`, `setup()`, and `teardown()` hooks.

Plugins support role-based assignment in mission configs and integrate with LangGraph's tool calling.

For implementation details, see the [Plugin Tutorial](https://github.com/blob/main/docs/guides/plugin_tutorial.md).

**Why this might change**: Enhancements may include plugin versioning, dependency injection, or hot-reloading for development.

**When to extend**: Create new plugins for custom data sources; extend base classes for tool subtypes.


## Dependencies
- `seek.components.search_graph.litesearch`: Async rate-limited search proxy implementation
- `langchain_communityutilities`: Search provider wrappers (Google, Bing, Tavily, etc.)
- HTTP clients (`httpx`): For direct API interactions
- `seek.common.config.get_active_seek_config()`: Runtime configuration access

**Dependency changes**: Monitor for updates to external API libraries or rate limiting requirements that could necessitate wrapper updates.

## Configuration
Uses `seek.common.config.get_active_seek_config()` to access:
- `web_search.provider`: Search backend (google, brave, duckduckgo, etc.)
- `web_search.requests_per_second`: Rate limiting per provider
- `web_search.max_results`: Result count limits
- `max_results`: Global truncation settings
- `use_robots`: Robots.txt compliance setting

Configuration files should be updated when adding new providers, adjusting rate limits, or modifying content processing parameters.

## Minimal Usage
- `from seek.components.tool_manager.tools import get_tools_for_role` and bind to agents based on role.
- Tools are automatically configured based on active seek config; no manual initialization required for standard usage.
