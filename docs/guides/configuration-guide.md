# DataSeek Configuration Guide

## Overview

DataSeek uses a modular configuration system with three primary files, each serving a distinct purpose. This separation ensures clarity, flexibility, and maintainability. Configurations follow a hierarchy: environment variables override `.env` overrides file defaults.

- **seek_config.yaml**: Global runtime environment and tool behaviors (e.g., API keys, rate limits, default models).
- **mission_config.yaml**: Mission-specific goals, quotas, and overrides for tools/models.
- **prompts.yaml**: Customizable prompt templates for agent nodes.

All files are located in the `config/` directory relative to the project root (`dataseek/`).

## seek_config.yaml: Global Runtime Configuration

This file defines the baseline environment for all missions. It includes tool defaults, observability, and node-level settings. Changes here apply globally unless overridden in `mission_config.yaml`.

### Structure and Key Sections

```yaml
# config/seek_config.yaml
model_defaults:
  model: "openai/gpt-5-mini"  # Default LLM model for all nodes
  temperature: 1
  max_tokens: 2000

web_search:
  provider: "duckduckgo/search"  # Default: "duckduckgo/search" (fallback to root search_provider)
  respect_robots: true           # Respect robots.txt (global use_robots overrides possible)
  requests_per_second: 2.0       # RPS for non-header-based providers
  max_results: 10                # Canonical limit (maps to provider-specific params)

recursion_per_sample: 35        # Max iterations per sample (allows retries)
initial_prompt: |               # Starting agent prompt
  Begin the data prospecting mission based on your configured mission plan.

observability:                  # LangSmith integration
  tracing: true                 # Enable tracing (use env vars for API key)
  endpoint: "https://api.smith.langchain.com"
  project: "dataseek"

nodes:                          # Node-specific defaults (overridable per mission)
  research:
    max_iterations: 7           # ReAct loop iterations

checkpointer_path: ".checkpointer.sqlite"  # State persistence
```

### Tool Configuration

Tools like web_search can be tuned here for global behavior:

- `provider`: Search backend (e.g., "brave/search", "tavily/search", "arxiv/search").
- `respect_robots`: Per-tool robots.txt compliance.
- `requests_per_second`: Throttling for fixed-rate providers (header-based use response limits).
- `max_results`: Result cap (legacy keys like `num_results` supported for compatibility).

Advanced tool options (prefetching, validation) are set in `mission_config.yaml` for mission-specific overrides.

### Observability

Prefer environment variables for sensitive data:

```bash
export LANGCHAIN_API_KEY="your-key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="dataseek"
```

Config file alternatives (avoid for keys):

```yaml
observability:
  api_key: "your-key"  # Not recommended; use env vars
  tracing: true
```

## mission_config.yaml: Mission-Specific Configuration

This file defines goals and quotas for individual missions. It overrides `seek_config.yaml` settings where needed (e.g., tool prefetching).

### Structure and Key Sections

```yaml
# config/mission_config.yaml
missions:
  - name: "production_corpus"
    target_size: 150              # Samples per goal
    synthetic_budget: 0.2         # Max synthetic proportion (0.0-1.0)
    
    # Mission-specific overrides
    output_paths:
      samples_path: "examples/data/datasets/tier1"
      audit_trail_path: "examples/PEDIGREE.md"
    
    # Tool overrides (e.g., enable prefetching)
    tools:
      web_search:
        pre_fetch_pages: true
        pre_fetch_limit: 5
        validate_urls: true
        retry_on_failure: true
        max_retries: 2
      arxiv_search:
        pre_fetch_pages: true
        pre_fetch_limit: 3
        validate_urls: true
        retry_on_failure: true
        max_retries: 2
    
    # Node overrides (e.g., higher tokens for research)
    nodes:
      research:
        max_tokens: 65536
        summary_char_limit: 6000  # Per-entry cap for search results
        result_char_limit: 65536  # Cap for fetched markdown
    
    goals:                         # Per-characteristic targets
      - characteristic: "Evidence-Based Reasoning"
        topics: ["clinical psychology", "data analysis"]
      - characteristic: "Controlled Experimental Design"
        topics: ["machine learning ethics", "scientific methods"]
```

### Overrides

- **Tools**: Mission-specific tuning (e.g., prefetching for robustness).
- **Nodes**: Adjust models, tokens, or limits (e.g., `summary_char_limit` for search truncation).
- **Output Paths**: Per-mission storage (overrides global if set).

## prompts.yaml: Agent Prompt Templates

Customizes prompts for each node (supervisor, research, etc.). Defaults are predefined; override for fine-tuning.

### Structure

```yaml
# config/prompts.yaml
supervisor:
  base_prompt: >
    You are the supervisor...  # Full prompt template

research:
  base_prompt: >                 # Normal mode prompt
    You are a Data Prospector...
  cached_only_prompt: >          # Fallback for cached data
    You are operating in CACHED-ONLY MODE...

# Other nodes: archive, fitness, synthetic, etc.
```

Edit templates to adjust agent behavior without code changes.

## Result Size Limits

Post-tool truncation ensures efficient processing:

- **Search Tools** (web_search, arxiv_search, etc.): Per-entry limit via `summary_char_limit` (default derived from `max_tokens`).
- **Fetch Tools** (url_to_markdown, documentation_crawler): Overall markdown cap via `result_char_limit` (default: 65,536 chars).

Truncation appends notes (e.g., "[Entry truncated]"). Configure per node in `mission_config.yaml`.

## Loading and CLI Usage

Configs load via `seek.common.config` helpers. CLI flags:

```bash
dataseek --mission config/mission_config.yaml --config config/seek_config.yaml
dataseek-tui config/mission_config.yaml  # Uses default seek_config.yaml
```

Precedence: CLI > env > `.env` > files. For custom prompts, set `PROMPTS_PATH` env var.

## Migration Notes

- From legacy: Flatten `seek_agent:` wrappers into root keys.
- Paths: All relative to `dataseek/` root (e.g., `examples/data/datasets/tier1`).
- Testing: Verify with `dataseek --mission <file> --dry-run` (if implemented).

For advanced overrides and examples, see [Data Seek Agent Guide](data-seek-agent.md).