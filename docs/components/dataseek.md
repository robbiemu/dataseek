# DataSeek Component

Purpose and scope
- DataSeek is a standalone data prospecting agent that discovers, extracts, and archives verifiable sources. It originated inside a larger system but is now structured as an independent component under `seek/`.

Key interfaces and APIs
- CLI entrypoint: `main.py` at the repository root (Typer app) exposes `run()`.
- Graph builder: `seek/graph.py` assembles the LangGraph nodes and tool nodes.
- Nodes: `seek/nodes.py` contains supervisor, research, fitness, synthetic, and archive nodes.
- Tools: `seek/tools.py` provides production-ready tools (web search, content fetch, URL-to-Markdown, etc.).
- Config: `seek/config.py` provides a `StructuredSeekConfig`-style API via `get_active_seek_config()`/`set_active_seek_config()` and `load_seek_config()` to read `config/seek_config.yaml`.
- State: `seek/state.py` defines the `DataSeekState` schema used across nodes.

Dependencies and integration points
- LangChain/LangGraph: for LLM orchestration and graph execution.
- Optional crawler: `libcrawler` (off by default). Enable via extras: `pip install -e .[crawler]`.
- Observability (optional): LangSmith environment variables; see `setup_observability()` in `seek/main.py`.

Configuration requirements
- Primary runtime config: `config/seek_config.yaml` (model defaults, mission plan references, web search provider, etc.).
- Mission plan (targets and quotas): `settings/mission_config.yaml` (referenced by CLI options).
- Configuration loading priority follows session env > `.env` > config files. Use `load_seek_config()` and `set_active_seek_config()`.

Minimal usage examples
- CLI
  - New mission: `python main.py --mission <name> --config config/seek_config.yaml --mission-config settings/mission_config.yaml`
  - Resume: `python main.py --resume-from <mission_id>`
- Library usage
  - Build and run a graph with an existing SqliteSaver, then stream: see `tests/seek/test_integration.py` for a minimal programmatic example.

Notes
- Networking in tests is blocked by default via `tests/conftest.py`. Patch tool layer APIs or set `ALLOW_NET_TESTS=1` to allow external requests in explicitly marked tests.
