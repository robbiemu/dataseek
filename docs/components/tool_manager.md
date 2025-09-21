# Tool Manager

## Purpose
Implements production-ready tools for data acquisition and processing used by agents during research and synthesis phases. Provides a unified interface for various data sources including web search, academic papers, wiki content, and local file operations. Handles rate limiting, robots.txt compliance, and content formatting to ensure reliable and ethical data collection.

**Why this might change**: Updates are likely when integrating new data sources (e.g., social media APIs, proprietary databases), adding content processing features (e.g., image analysis, multimedia extraction), or implementing advanced filtering (e.g., content quality scoring). Changes may also address new compliance requirements or performance optimizations for high-rate data collection.

## Key Interfaces
- `seek.components.tool_manager.tools.get_tools_for_role()`: Factory function returning role-specific tool sets
- Individual tool functions:
  - `create_web_search_tool()`: Configurable web search with provider abstraction
  - `url_to_markdown()`: Content extraction and HTML-to-markdown conversion
  - `write_file()`: Secure file writing with metadata tracking
  - Search-specific tools: `arxiv_search()`, `wikipedia_search()`, etc.
- `seek.components.tool_manager.tools.SearchProviderProxy`: Async proxy for various search backends

**When to extend**: New tools are added for novel data types, and existing tools are modified to support additional formatting options or enhanced error handling.

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
