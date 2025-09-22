# Plugin System Tutorial

## Introduction

The DataSeek plugin system allows extending the tool manager with custom tools for data acquisition and processing. Plugins are Python modules in the `plugins/` directory that inherit from base tool classes and are automatically discovered and loaded by the `ToolManager`.

This tutorial guides you through creating a simple plugin that searches a fictional API for weather data.

## Prerequisites

  - Basic Python knowledge
  - Familiarity with Pydantic for configuration (optional)
  - Access to the DataSeek codebase

## Step 1: Define the Tool Class

Create a new file in the `plugins/` directory, e.g., `weather_search.py`. Inherit from `BaseSearchTool` for search-type tools or `BaseUtilityTool` for others.

```python
from typing import Any, ClassVar
from pydantic import BaseModel, Field

from seek.components.tool_manager.plugin_base import BaseSearchTool
from seek.components.tool_manager.registry import register_plugin

class WeatherConfig(BaseModel):
    api_key: str = Field(..., description="API key for the weather service")
    location: str = Field(default="global", description="Default location to query")

@register_plugin
class WeatherSearchTool(BaseSearchTool):
    name: str = "weather_search"
    description: str = "Searches for weather data using a fictional API."
    ConfigSchema: ClassVar[type[BaseModel]] = WeatherConfig

    async def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        # Implement your search logic here
        # Example: Use self.config to access API key and location
        if not self.config:
            return [{"error": "No configuration provided"}]
        
        # Placeholder implementation
        return [
            {
                "query": query,
                "results": [{"location": "Example", "temperature": "20C"} for _ in range(num_results)],
                "provider": "weather/api",
                "status": "ok",
            }
        ]
```

## Step 2: Optional Lifecycle Hooks

Override `setup` and `teardown` for initialization and cleanup.

```python
async def setup(self) -> None:
    # Initialize any resources, e.g., authenticate with API
    print(f"Setting up {self.name} with location {self.config.location}")

async def teardown(self) -> None:
    # Clean up resources
    print(f"Tearing down {self.name}")
```

## Step 3: Configuration in Mission YAML

In your mission configuration file (e.g., `config/mission_config.yaml`), add the tool under `tool_configs`.

```yaml
tool_configs:
  weather_search:
    api_key: "your_api_key_here"
    location: "New York"
    roles: ["researcher", "synthesizer"]
```

## Step 4: Testing the Plugin

1.  Place `weather_search.py` in `plugins/`.
2.  Run the mission with the updated config.
3.  The `ToolManager` will load and instantiate the tool for specified roles.
4.  Bind tools to agents in your mission setup.

## Advanced Topics

  - **Error Handling**: Raise informative exceptions in `search` or `execute`; they will be captured and returned in results.
  - **Dependencies**: Add required packages to `plugins/requirements.txt` if not already in the main `pyproject.toml`.
  - **Integration**: For web APIs, use the provided `HTTP_CLIENT` and `RATE_MANAGER` from `seek.components.tool_manager.clients`.
  - **Validation**: Use Pydantic's `ConfigSchema` for runtime config validation.

For full API details, see the [Tool Manager Component Documentation](https://www.google.com/search?q=../components/tool_manager.md).

**Note**: Plugins are loaded dynamically at runtime. Ensure no name collisions in `PLUGIN_REGISTRY`.

## Validating Prompts with check\_prompts.py

When extending DataSeek with plugins, you should use the `scripts/check_prompts.py` script to verify prompt templates in `config/prompts.yaml` against your code references.

### How to Execute

Run the script from the project root:

```bash
uv run python scripts/check_prompts.py --prompts config/prompts.yaml
```

You can optionally specify additional source directories to scan, such as your plugins folder:

```bash
uv run python scripts/check_prompts.py --prompts config/prompts.yaml --src seek/components/search_graph/nodes --src plugins
```

### Expected Output and How to Read It

The script performs three main checks:

1.  **Missing Prompts**: Checks that every `get_prompt(role, key)` call in the code corresponds to a prompt in the YAML file.
2.  **Unused Prompts**: Flags prompts in the YAML file that are not referenced in the code (unless you use the `--allow-extra` flag).
3.  **Placeholder Validation**: Verifies that `{variable}` placeholders in a template match what the code expects to provide.

A successful run will show a confirmation message and exit with code 0. If failures are found, the script will list the specific errors (e.g., missing prompts, unused prompts, or placeholder mismatches) and exit with code 1. You should fix the issues in your code or `prompts.yaml` and re-run the script.