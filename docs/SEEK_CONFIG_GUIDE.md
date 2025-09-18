# Seek Agent Mission Configuration Guide

## Overview

The Data Seek Agent configuration has been separated from the main `dataseek` configuration to provide better separation of concerns. This allows you to:

1. **Use custom configurations** without modifying the main config
2. **Specify configuration files** via command-line flags
3. **Have proper path control** for where samples are generated

## Changes Made

### 1. New Mission Configuration File

Created `seek_config.yaml` in the project root containing all mission parameters and seek settings:

```yaml
# Mission Configuration for Data Seek Agent
web_search:
  provider: "brave/search"
  respect_robots: true
  requests_per_second: 2.0
recursion_per_sample: 30
initial_prompt: |
  Begin the data prospecting mission based on your configured mission plan.

nodes:
  research:
    max_iterations: 7

mission_plan:
  goal: "Find reliable, verifiable data sources..."
  nodes:
    - name: "supervisor"
      model: "openai/gpt-5-mini"
      temperature: 1
      max_tokens: 65536
    # ... other nodes

writer:
  samples_path: "examples/data/datasets/tier1"
  tier2_path: "examples/data/datasets/tier2"
  audit_trail_path: "examples/PEDIGREE.md"

checkpointer_path: ".checkpointer.sqlite"
```

### 2. New Configuration Loader

Created `dataseek/seek/config.py` with a simple `load_seek_config()` function that:
- Defaults to `seek_config.yaml` in the current directory
- Accepts custom config file paths
- Returns sensible defaults if the config file is missing

### 3. Command-Line Flag Support

All seek commands now support `--config` (`-c`) flag:

```bash
# CLI usage
dataseek --mission research_dataset --config my_seek_config.yaml

# TUI usage  
dataseek-tui --config my_seek_config.yaml

# Step 1 script usage
./scripts/run_seek_step1.sh --mission research_dataset --config my_seek_config.yaml
```

### 4. Fixed Path Discrepancy

**Previously**: Samples were going to `output/approved_books/` (hardcoded) while the config specified `examples/data/datasets/tier1/`

**Now**: The archive node properly reads the `writer.samples_path` from the seek config and saves samples there.

## Web Search Configuration

You can configure the web search provider and behavior via the `web_search` section:

```yaml
web_search:
  provider: "duckduckgo/search"    # or brave/search, google/search, tavily/search, etc.
  respect_robots: true              # overrides global use_robots for web_search
  requests_per_second: 2.0          # RPS for providers without header-based rate limits
  max_results: 10                   # Canonical result cap (proxy maps per provider)
```

Notes:
- `provider` falls back to the legacy root `search_provider` if `web_search.provider` is not set.
- `respect_robots` affects only the web_search tool; other tools keep using the global `use_robots`.
- `requests_per_second` controls throttling for non-header-based providers (e.g., DuckDuckGo, Wikipedia wrappers). For header-based providers (e.g., Brave), rate limiting uses response headers.
- `max_results` is the canonical field. The proxy maps it internally to provider-specific params (e.g., `num_results`, `count`, `num_web_results`). Older keys like `num_results`, `count`, and `limit` are accepted for backward compatibility but should be migrated.

### Result Size Limits (Post-Tool)

The agent enforces result size limits after each tool call, based on tool category:

- Search-like tools (`web_search`, `arxiv_search`, `wikipedia_search`, `arxiv_get_content`, `wikipedia_get_content`):
  - Apply `summary_char_limit` per entry in `results`.
- Fetch-like tools (`url_to_markdown`, `documentation_crawler`):
  - Apply `result_char_limit` to the resulting markdown (`markdown` or `full_markdown`).

Configure these in the research node:

```yaml
mission_plan:
  nodes:
    - name: "research"
      model: "openai/gpt-5-mini"
      temperature: 1
      max_tokens: 65536
      summary_char_limit: 6000     # Optional per-entry cap for search-like tools (chars)
      result_char_limit: 65536     # Hard cap for fetch-like tool markdown (chars)
```

If `summary_char_limit` is omitted, the system derives a default assuming up to ~5 sources per cycle and reserving overhead:

- Default: `max(2000, int(max_tokens * 4 * 0.8 / 5))`
- This roughly allocates 80% of the research role’s character budget across 5 sources and keeps 20% for prompts and reasoning.

`result_char_limit` defaults to `65536` if not specified.

## Usage Examples

### Basic Usage (Default Config)
```bash
# Uses seek_config.yaml in current directory
dataseek --mission research_dataset
```

### Custom Configuration
```bash
# Uses a custom config file
dataseek --mission research_dataset --config /path/to/my_config.yaml
```

### Step 1 Script with Custom Config
```bash
./scripts/run_seek_step1.sh --mission research_dataset --config my_custom_mission.yaml
```

### TUI with Custom Config
```bash
dataseek-tui --config my_config.yaml --log seek.log
```

## Configuration Structure

The seek config file supports these main sections:

- **`search_provider`**: Search service to use (e.g., "brave/search", "duckduckgo/search")
- **`recursion_per_sample`**: Maximum iterations per sample generation cycle
- **`initial_prompt`**: Starting prompt for the agent
- **`observability`**: LangSmith configuration for tracing and monitoring
  - **`api_key`**: LangSmith API key (recommended to use environment variables instead)
  - **`tracing`**: Enable/disable tracing (boolean)
  - **`endpoint`**: Custom LangSmith endpoint for self-hosted deployments
  - **`project`**: Project name for organizing traces
- **`nodes.research.max_iterations`**: ReAct loop iterations for research
- **`mission_plan`**: LLM model configurations for each agent node
- **`writer`**: Output path configurations
  - `samples_path`: Where raw samples are saved
  - `tier2_path`: Where curated samples go
  - `audit_trail_path`: Where PEDIGREE.md is written
- **`checkpointer_path`**: SQLite database for agent state persistence

## Migration from Old Config

If you have existing `seek_agent` configuration in `dataseek/settings/config.yaml`, you can:

1. **Copy the configuration** to a new `seek_config.yaml` file
2. **Flatten the structure** (remove the `seek_agent:` wrapper)
3. **Use the --config flag** to specify your new file

Example migration:
```yaml
# Old: dataseek/settings/config.yaml
seek_agent:
  writer:
    samples_path: "examples/data/datasets/tier1"
  # ... other settings

# New: seek_config.yaml  
writer:
  samples_path: "examples/data/datasets/tier1"
# ... other settings
```

## Benefits

1. **Clean Separation**: Seek configuration is independent from main claimify config
2. **Flexible Deployment**: Use different configs for different environments
3. **Correct Paths**: Samples now go to the configured paths, not hardcoded ones
4. **Better Defaults**: Sensible fallbacks if config file is missing
5. **CLI Integration**: All tools support the --config flag consistently

## LangSmith Observability Configuration

The seek agent supports LangSmith tracing for monitoring and debugging your missions. You can configure LangSmith in two ways:

### Environment Variables (Recommended)

Set these environment variables for LangSmith configuration:

```bash
# Authentication (required for tracing)
export LANGCHAIN_API_KEY="your-langsmith-api-key"
# or
export LANGSMITH_API_KEY="your-langsmith-api-key"

# Enable tracing (optional, enabled by default when API key is present)
export LANGCHAIN_TRACING_V2="true"
# or
export LANGSMITH_TRACING="true"

# Custom endpoint for self-hosted deployments (optional)
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# or
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"

# Project name for organizing traces (optional)
export LANGCHAIN_PROJECT="dataseek"
# or
export LANGSMITH_PROJECT="dataseek"
```

### Configuration File

Alternatively, you can configure LangSmith in your `seek_config.yaml`:

```yaml
observability:
  # api_key: "your-langsmith-api-key"  # Not recommended in config files
  tracing: true
  endpoint: "https://api.smith.langchain.com"
  project: "dataseek"
```

Note: It's recommended to use environment variables for sensitive information like API keys rather than including them in configuration files.

When LangSmith is properly configured, you'll see tracing information in your LangSmith dashboard, which can help you monitor and debug your seek missions.

## Testing the Changes

To verify everything works:

To verify everything works:

1. **Test with default config**:
   ```bash
   dataseek --mission production_corpus
   # Should use seek_config.yaml and save to examples/data/datasets/tier1/
   ```

2. **Test with custom config**:
   ```bash
   cp seek_config.yaml my_config.yaml
   # Edit my_config.yaml to change tier1_path
   dataseek --mission production_corpus --config my_config.yaml
   # Should save to your custom path
   ```

3. **Verify paths**: Check that samples appear in the configured `samples_path`, not in `output/approved_books/`
