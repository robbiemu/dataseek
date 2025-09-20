"""
Configuration loader for the Data Seek Agent.
Handles loading mission-specific configuration from separate config files.
"""

import logging
from collections.abc import Iterator
from typing import Any, Optional

import yaml

from .models import SeekAgentMissionPlanToolConfig


def merge_configs(default: dict, override: dict) -> dict:
    """Recursively merge two dictionaries."""
    for key, value in override.items():
        if isinstance(value, dict) and key in default and isinstance(default[key], dict):
            default[key] = merge_configs(default[key], value)
        else:
            default[key] = value
    return default


logger = logging.getLogger(__name__)

# Global variable to store the use_robots setting
_global_use_robots = True

# Active configuration instance set at application startup
_active_seek_config: Optional["StructuredSeekConfig"] = None

# Prompts configuration
_prompts_config: dict | None = None


def set_global_use_robots(use_robots: bool) -> None:
    """Set the global use_robots setting."""
    global _global_use_robots
    _global_use_robots = use_robots


def get_global_use_robots() -> bool:
    """Get the global use_robots setting."""
    global _global_use_robots
    return _global_use_robots


def set_active_seek_config(config: "StructuredSeekConfig") -> None:
    """Set the process-wide active seek configuration instance."""
    global _active_seek_config
    _active_seek_config = config


def get_active_seek_config() -> "StructuredSeekConfig":
    """Get the active seek configuration, loading defaults if not yet set."""
    global _active_seek_config
    if _active_seek_config is None:
        # Fall back to loading with current global use_robots
        _active_seek_config = load_seek_config(use_robots=get_global_use_robots())
    return _active_seek_config


def load_prompts_config(config_path: str = "config/prompts.yaml") -> dict:
    """Load prompts configuration from a YAML file."""
    global _prompts_config
    if _prompts_config is None:
        try:
            with open(config_path) as f:
                _prompts_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.error(f"Prompts configuration file not found: {config_path}")
            _prompts_config = {}
    return _prompts_config


def get_prompt(agent_name: str, prompt_type: str = "base_prompt") -> str:
    """Get a prompt template for a specific agent."""
    prompts_config = load_prompts_config()
    agent_config = prompts_config.get(agent_name, {})
    return agent_config.get(prompt_type, "")


class StructuredSeekConfig:
    """Structured configuration wrapper that provides object-oriented access to seek config."""

    def __init__(self, config_dict: dict[str, Any]):
        self._raw_config = config_dict

        # Extract mission plan tools if they exist
        self._tools = {}
        mission_plan = self._raw_config.get("mission_plan", {})
        if isinstance(mission_plan, dict) and "tools" in mission_plan:
            tools_config = mission_plan["tools"]
            if isinstance(tools_config, dict):
                for tool_name, tool_config in tools_config.items():
                    if isinstance(tool_config, dict):
                        self._tools[tool_name] = SeekAgentMissionPlanToolConfig(**tool_config)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the raw config dictionary."""
        return self._raw_config.get(name)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to config values."""
        return self._raw_config.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the config."""
        return key in self._raw_config

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access to config values with default."""
        return self._raw_config.get(key, default)

    def keys(self) -> Any:
        """Return the keys of the config dictionary."""
        return self._raw_config.keys()

    def values(self) -> Any:
        """Return the values of the config dictionary."""
        return self._raw_config.values()

    def items(self) -> Any:
        """Return the items of the config dictionary."""
        return self._raw_config.items()

    def __iter__(self) -> Iterator[str]:
        """Make the config iterable like a dictionary."""
        return iter(self._raw_config)

    def get_tool_config(self, tool_name: str) -> SeekAgentMissionPlanToolConfig | None:
        """Retrieve the configuration for a specific tool by name."""
        return self._tools.get(tool_name)

    def to_dict(self) -> dict[str, Any]:
        """Return the raw configuration dictionary."""
        return self._raw_config.copy()


def load_seek_config(
    config_path: str | None = None, use_robots: bool | None = None
) -> StructuredSeekConfig:
    """
    Load mission configuration for the seek agent from a separate config file.

    Args:
        config_path: Optional path to a config file to override defaults.
        use_robots: Whether to respect robots.txt rules. If None, uses global setting.

    Returns:
        StructuredSeekConfig object containing the mission configuration.
    """
    default_config_path = "config/seek_config.yaml"

    # Load default config
    try:
        with open(default_config_path) as f:
            config_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"Default configuration file not found: {default_config_path}")
        config_data = {}

    # Load override config if provided
    if config_path:
        try:
            with open(config_path) as f:
                override_config = yaml.safe_load(f) or {}
            config_data = merge_configs(config_data, override_config)
        except FileNotFoundError:
            logger.error(f"Override configuration file not found: {config_path}")

    # Use global setting if use_robots is not explicitly provided
    if use_robots is None:
        use_robots = get_global_use_robots()

    # Add use_robots to the config
    config_data["use_robots"] = use_robots

    logger.debug("Loaded seek configuration.")
    return StructuredSeekConfig(config_data)
