# DataSeek

DataSeek is a versatile, extensible framework for autonomous data collection and prospecting. It uses AI agents to discover, validate, and organize data from sources like academic papers, Wikipedia, and web searches, targeting configurable data quality characteristics.

![dataseek_manga_v2](https://github.com/user-attachments/assets/52527858-6764-415f-87b3-148f06dad23c)


## Overview

DataSeek is a powerful tool for automated data collection and prospecting. It uses advanced AI techniques to search, validate, and organize data from various sources including academic papers, Wikipedia, and web searches.

## Features

- Automated data prospecting from multiple sources
- Configurable mission plans for targeted data collection
- Support for academic papers (arXiv), Wikipedia, and web searches
- Built-in validation and filtering mechanisms
- Command-line interface for easy execution

## Quickstart

1. **Clone and set up the project**:
   ```bash
   git clone <repository-url>
   cd dataseek
   ```

2. **Create and activate a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

4. **Review and customize configuration files**:
   - Agent configuration: `config/seek_config.yaml` (see [Configuration Guide](docs/guides/configuration-guide.md))
   - Mission configuration: `config/mission_config.yaml` (see [Mission Configuration Guide](docs/guides/data-seek-agent.md))

5. **Run the default mac_ai_corpus_v1 mission**:
   ```bash
   dataseek --mission mac_ai_corpus_v1
   ```
   
   This collects 50 samples per characteristic into `examples/datasets/mac_ai_corpus/samples/` with audit trails in `examples/datasets/mac_ai_corpus/PEDIGREE.md`.

   _note: this can be run with the TUI for a more elegant experience:_
   ```bash
   dataseek-tui --log tui.log
   ```

6. **Monitor progress** through the terminal interface or check the output directories:
   - `examples/datasets/mac_ai_corpus/samples/` - Raw data samples
   - `examples/datasets/mac_ai_corpus/PEDIGREE.md` - Audit trail of the data collection process

## Documentation

### Overview
- [DataSeek Overview](docs/dataseek.md): High-level project description and purpose.

### Components
- [Mission Runner](docs/components/mission_runner.md): Manages the execution and state of data collection missions.
- [Search Graph](docs/components/search_graph.md): Defines the AI agent workflow for data prospecting and validation.
- [Tool Manager](docs/components/tool_manager.md): Handles tool registration, execution, and integration for agents.
- [TUI (Terminal User Interface)](docs/components/tui.md): Provides an interactive terminal interface for monitoring missions.

### Guides
- [Data Seek Agent Guide](docs/guides/data-seek-agent.md): Instructions for curating datasets using the agent.
- [Configuration Guide](docs/guides/configuration-guide.md): Details setup and customization of agent and mission configurations.
- [Prompting Guide](docs/guides/prompting-guide.md): Explains prompt assembly for different agent nodes.
- [Tools Guide](docs/guides/tools-guide.md): Documentation on available tools and their usage.

### Tutorials
- [From Idea to Dataset](docs/tutorials/from-idea-to-dataset.tutorial.md): Step-by-step guide to configuring your first mission.
- [Plugin System Tutorial](docs/tutorials/plugin.tutorial.md): How to create and integrate custom plugins.


## Installation

To set up the development environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

After installation, you can run DataSeek in two ways:

### Command-Line Interface
```bash
dataseek --mission mac_ai_corpus_v1
```

### Terminal User Interface (TUI)
```bash
dataseek-tui config/mission_config.yaml
```

You can also use Python module syntax:

```bash
python -m seek.main --mission research_dataset
```

To use a custom agent configuration file:

```bash
dataseek --mission research_dataset --config config/my_seek_config.yaml
```

## Configuration

DataSeek uses two types of configuration files:

### 1. Agent Configuration (`config/seek_config.yaml`)

This file configures the overall behavior of the DataSeek agent, including model settings, search parameters, and output paths. 

See [Agent Configuration Guide](docs/guides/seek_config_guide.md) for detailed documentation.

### 2. Mission Configuration (`config/mission_config.yaml`)

This file defines specific missions with their goals and parameters, including target sizes, synthetic budgets, and topic lists.

See [Mission Configuration Guide](docs/guides/data-seek-agent.md) for detailed documentation.

## Development

To set up the development environment with test dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

### Developer Setup (uv)

- Create venv + install dev deps
  ```bash
  uv venv && source .venv/bin/activate
  uv pip install -e .[dev]
  ```

- Optional: install pre-commit hooks
  ```bash
  pip install pre-commit
  pre-commit install
  ```

- Without activating the venv, you can use `uv run` to execute commands:
  ```bash
  pytest -q
  ruff check --fix .
  black .
  mypy seek
  bandit -c pyproject.toml -r seek
  ```

### Quality Checks

- Format (Black)
  ```bash
  black .
  ```

- Lint + import sort + modernize (Ruff)
  ```bash
  ruff check --fix .
  ```

- Type check (MyPy)
  ```bash
  mypy seek plugins --exclude tests
  ```

- Security scan (Bandit)
  ```bash
  bandit -c pyproject.toml -r seek
  ```

- Tests with coverage
  ```bash
  pytest -q --cov=seek
  ```

### Scripts

- [check_prompts.py](scripts/check_prompts.py): Verifies prompt templates in `config/prompts.yaml` against code references (missing, unused, placeholder mismatches).
- [dup.awk](scripts/dup.awk): Identifies groups of adjacent duplicate lines in log files by extracting core messages.
- [local_ci_runner.py](scripts/local_ci_runner.py): Generates shell scripts to execute specific CI jobs locally (e.g., `uv run python scripts/local_ci_runner.py quality-checks`).

## Testing

Run tests with:

```bash
pytest
```

Test discovery looks in `tests/` and component-local `seek/components/**/tests/` as configured in `pyproject.toml`.

### Repository Layout

Core modules live under `seek/`:
- Common utilities: `seek/common/` (config, models, utils)
- Components: `seek/components/`
  - Mission Runner: `seek/components/mission_runner/`
  - Search Graph: `seek/components/search_graph/`
  - Tool Manager: `seek/components/tool_manager/`
  - TUI: `seek/components/tui/`

Note on LiteLLM/Ollama
- Some Ollama models (e.g., gpt-oss-20b) may not work reliably with LiteLLM’s default transformations.
- We apply a small runtime shim in `seek/components/patch.py` and import it early in `seek/__init__.py` to stabilize tool-call handling.
- This patch is intentionally not part of Tool Manager because it affects LLM call plumbing across components.
