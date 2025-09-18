# DataSeek

DataSeek is a standalone data prospecting agent for finding reliable, verifiable data sources.

## Overview

DataSeek is a powerful tool for automated data collection and prospecting. It uses advanced AI techniques to search, validate, and organize data from various sources including academic papers, Wikipedia, and web searches.

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
   - Agent configuration: `config/seek_config.yaml` (see [Agent Configuration Guide](SEEK_CONFIG_GUIDE.md))
   - Mission configuration: `settings/mission_config.yaml` (see [Mission Configuration Guide](docs/guides/data-seek-agent.md))

5. **Run a data prospecting mission**:
   ```bash
   dataseek --mission research_dataset
   ```

6. **Monitor progress** through the terminal interface or check the output directories:
   - `examples/data/datasets/tier1` - Raw data samples
   - `examples/data/datasets/tier2` - Curated data samples
   - `examples/PEDIGREE.md` - Audit trail of the data collection process

## Features

- Automated data prospecting from multiple sources
- Configurable mission plans for targeted data collection
- Support for academic papers (arXiv), Wikipedia, and web searches
- Built-in validation and filtering mechanisms
- Command-line interface for easy execution

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
dataseek --mission research_dataset
```

### Terminal User Interface (TUI)
```bash
dataseek-tui settings/mission_config.yaml
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

See [Agent Configuration Guide](SEEK_CONFIG_GUIDE.md) for detailed documentation.

### 2. Mission Configuration (`settings/mission_config.yaml`)

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
  mypy seek
  ```

- Security scan (Bandit)
  ```bash
  bandit -c pyproject.toml -r seek
  ```

- Tests with coverage
  ```bash
  pytest -q --cov=seek
  ```

Note on tests and networking: tests block outbound network connections by default. If you intentionally need network for an explicit integration test, set `ALLOW_NET_TESTS=1` in the environment.

## Testing

Run tests with:

```bash
pytest
```
