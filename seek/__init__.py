# Apply LiteLLM patches for Ollama compatibility
# This must happen before any other imports to ensure patches are applied
from .common import defensive_model_adapter  # noqa: F401 # for side effects
from .components import patch  # noqa: F401 # imported for side effects

# Export key classes
from .components.mission_runner.mission_runner import MissionRunner

__all__ = ["MissionRunner"]
