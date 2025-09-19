# Tool Manager

- Purpose: Implements production-ready tools and HTTP helpers used by agents (search, content retrieval, markdown conversion, file I/O).
- Key interfaces: `seek.components.tool_manager.tools` (e.g., `get_tools_for_role`, `create_web_search_tool`, `write_file`).
- Dependencies: Async search proxy in `seek.components.search_graph.litesearch`, HTTP clients (`httpx`).
- Configuration: `seek.common.config.get_active_seek_config()` provides search provider, RPS, and tool limits.
- Minimal usage:
  - `from seek.components.tool_manager.tools import get_tools_for_role` and bind to agents based on role.
