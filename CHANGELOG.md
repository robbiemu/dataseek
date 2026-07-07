# Changelog

All notable changes to DataSeek are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-07

First release with support for local, OpenAI-compatible model servers (sglang, Spark)
and reasoning models (Qwen3). Adds the missing extension points a downstream Lean
proof-trace pipeline needs to drive dataseek as an engine rather than a fork.

### Added

- **Local model server support.** Roles can target a custom OpenAI-compatible endpoint
  (`api_base`) instead of the cloud provider. `seek_config.yaml` now accepts `api_base`
  per role or globally in `model_defaults`.
- **Streaming completions (default on).** Long generations against local servers no longer
  drop with `Connection error` — `create_llm` now streams by default so bytes flow during
  generation and idle-read timeouts never fire. Configurable: `model_defaults.streaming`
  and per-node override.
- **`model_kwargs` passthrough to the server.** Provider-specific params
  (`chat_template_kwargs`, `repetition_penalty`, `top_k`, etc.) now reach the server.
  This unblocks **reasoning models** (e.g. Qwen3) — set
  `model_kwargs: {chat_template_kwargs: {enable_thinking: false}}` and the model populates
  `content` directly instead of leaving it empty in `reasoning_content`.
- **Externalizable prompts.** New `--prompts` CLI flag and `set_prompts_config()` mirror
  `--config`/`set_active_seek_config`, so a mission can supply its own prompt templates
  without editing the bundled `config/prompts.yaml`.
- **Unified tool system.** `@register_plugin` is now live for node agents:
  `get_tools_for_role` merges `PLUGIN_REGISTRY` tools mapped via
  `mission_config.tool_configs[tool].roles`. A mission can map its own plugins to any role,
  and (when it does) the mission becomes authoritative for that role's builtin tools.
- **Fitness node runs tools.** When tools are mapped to the `fitness` role, the node is
  tool-driven (`create_agent_runnable`) and a `fitness_tools` ToolNode is added to the graph.
  Stock missions with no fitness tools keep the exact prior topology.
- **Provenance guard.** A hard assertion in `archive_node` rejects any `synthetic` sample
  when `synthetic_budget == 0`, converting the supervisor's prompt-directed gating into a
  structural guarantee of real-data-only archives.
- **`--prompts` is accepted by `scripts/check_prompts.py`** (already was; documented now).
- **Docs:** `docs/components/litellm-patch-compatibility.md` — a source-diff review of every
  custom LiteLLM touch point across the 1.77.3 → 1.91.0 bump.

### Changed

- **Default models refreshed** to the current OpenAI lineup:
  `supervisor`/`research`/`fitness` → `openai/gpt-5.5`;
  `archive`/`synthetic` + `model_defaults` → `openai/gpt-5.4-mini` (current mini tier).
- **`top_p` is now configurable** per role and from `model_defaults`, and is routed through
  `model_kwargs` (the direct `ChatLiteLLM.top_p` field is silently dropped by langchain).
  Unblocks greedy models (e.g. Leanstral at `top_p: 1.0`).
- **`max_retries` is now configurable** (default 3). Tunes LiteLLM's transport-layer retry
  of transient transport/5xx/429 errors at the layer where reconnects are clean.
- **`max_tokens` is now opt-in** (no default). Previously the repo defaulted to `2000`
  (and per-node `65536`); both truncate reasoning models, which spend the budget in the
  reasoning phase before the answer is emitted. Left unset, the server/model decides the
  output budget. Set `model_defaults.max_tokens` or a per-node `max_tokens` when you need
  an explicit cap.
- **LiteLLM bumped `1.77.1` → `1.83.14+`** (resolved to 1.91.0). Resolves CVE-2026-42271
  (RCE, CISA KEV-listed), CVE-2026-42208 (SQLi), CVE-2026-47101 (privilege escalation),
  and clears the malicious 1.82.7/1.82.8 supply-chain range. All custom LiteLLM patches
  verified to land identically on the new version.

### Fixed

- **Resilience to transient LLM endpoint failures.** A dropped connection or 503 in
  `fitness_node` no longer aborts the whole sample cycle. Transport/5xx/429 errors
  (`APIConnectionError`, `InternalServerError`, `RateLimitError`) are caught at the node
  boundary and degrade to a deterministic REJECTED `FitnessReport` so the graph continues
  to archive. 4xx errors (auth/validation/config) still propagate, so genuine bugs aren't
  masked as quality rejects.
- **MyPy 2.x compatibility** in `supervisor.py` (a `dict | None` narrowing fix).
- **LangGraph 1.x compatibility** in `test_seek.py` (use a real `InMemorySaver` instead
  of a `MagicMock` checkpointer).
- **Black 26.x compatibility** (reformatted two files for the new string/paren hugging rules).

### Removed

- **`LeanTraceState`** subclass — dead code (the graph uses `DataSeekState`; no production
  code read or wrote the subclass or its six fields). Removed to avoid implying a contract
  the framework doesn't enforce.

### Compatibility

All changes are backward-compatible. Stock cloud missions that succeed on the first try
behave identically: `streaming` only matters on failure paths, the resilience guard only
fires on transport errors, `model_kwargs`/`top_p`/`max_retries` are only passed when
configured, and the fitness tool-node only appears when a mission maps fitness tools.

## [0.2.2] - 2025-09

Initial public release: a LangGraph supervisor-routed, multi-agent data-prospecting
framework with an externalizable config system, plugin registry, role-based model routing
via LiteLLM, SQLite checkpointing, and a respected synthetic-content budget.
