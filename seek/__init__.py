# Apply LiteLLM patches for Ollama compatibility
# This must happen before any other imports to ensure patches are applied
try:
    from .components import patch  # noqa: F401 # imported for side effects
except ImportError as e:
    print(f"⚠️  Warning: Could not import LiteLLM patches: {e}")
    print("Some Ollama functionality may be limited.")

# Export key classes
from .components.mission_runner.mission_runner import MissionRunner

__all__ = ["MissionRunner"]
