# DataSeek Overview

Purpose and scope
- DataSeek is a standalone data prospecting agent that discovers, extracts, and archives verifiable sources. It is structured as discrete components under `seek/`.

Key interfaces and modules
- CLI entrypoints:
  - `main.py` (root): Typer app (`seek.main`) exposing `run()` and orchestration.
  - `dataseek-tui`: Textual TUI (`seek.components.tui.dataseek_tui:main`).
- Graph builder: `seek/components/search_graph/graph.py` — assembles LangGraph nodes and tool nodes.
- Nodes: `seek/components/search_graph/nodes.py` — supervisor, research, fitness, synthetic, archive.
- Tools: `seek/components/tool_manager/tools.py` — production tools (search, content, URL→Markdown, write_file).
- Config: `seek/common/config.py` — `load_seek_config()`, `set_active_seek_config()`, `get_active_seek_config()`.
- State: `seek/components/mission_runner/state.py` — `DataSeekState` schema used across nodes.

Dependencies and integration points
- LangChain/LangGraph for orchestration.
- Optional crawler: `libcrawler` (off by default). Enable via extras: `pip install -e .[crawler]`.
- Observability (optional): configure via LangSmith/LangChain env vars; see `setup_observability()` in `seek/main.py`.

Configuration
- Primary runtime config: `config/seek_config.yaml` (model defaults, search provider, writer, etc.).
- Mission plan (targets and quotas): `settings/mission_config.yaml`.
- Config precedence: env > `.env` > files. Use `seek.common.config` helpers.

Minimal usage examples
- CLI
  - New mission: `dataseek --mission <name> --config config/seek_config.yaml --mission-config settings/mission_config.yaml`
  - Resume: `dataseek --resume-from <mission_id>`
- TUI
  - `dataseek-tui settings/mission_config.yaml`
- Library usage
  - Build and run a graph with a `SqliteSaver`: see `tests/seek/test_integration.py` for a minimal programmatic example.

Notes
- Tests block outbound network by default via `tests/conftest.py`. Patch tool APIs or set `ALLOW_NET_TESTS=1` to allow net in explicit tests.
