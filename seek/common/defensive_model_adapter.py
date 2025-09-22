"""
Permanent defensive adapters for LLM model interactions.

This module houses model-agnostic functions that provide resilience against common
LLM API unpredictability, such as malformed JSON responses and invalid provider
arguments. These functions are designed to be reusable across the codebase and
adhere to the project's highest code quality standards.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def fix_malformed_json_arguments(args_str: str) -> str:
    """
    Attempt to fix malformed JSON using the json-repair library.
    """
    # Only operate on strings; passthrough non-strings unchanged
    if not isinstance(args_str, str):
        return args_str

    # Treat empty/whitespace-only strings as an empty JSON object
    if args_str.strip() == "":
        return "{}"

    try:
        # Use json-repair library to fix malformed JSON
        import json_repair

        repaired = json_repair.repair_json(args_str)

        # Validate the repaired JSON can be parsed
        json.loads(repaired)
        return repaired

    except Exception as e:
        print(f"🔧 json-repair failed: {e}, falling back to empty dict")
        return "{}"


def sanitize_provider_kwargs(**kwargs: Any) -> dict:
    """
    Sanitize provider-specific kwargs to ensure compatibility across models.

    Specifically, maps tool_choice='any' to 'auto' for providers that reject the former.
    """
    kwargs_copy = dict(kwargs)  # Avoid mutating original

    try:
        if kwargs_copy.get("tool_choice") == "any":
            kwargs_copy["tool_choice"] = "auto"
        # Some integrations pass tool_choice via extra_body
        extra_body = kwargs_copy.get("extra_body")
        if isinstance(extra_body, dict) and extra_body.get("tool_choice") == "any":
            extra_body["tool_choice"] = "auto"
            kwargs_copy["extra_body"] = extra_body
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        logging.warning(
            f"Could not sanitize 'extra_body' due to a caught exception: {e}. "
            "This may happen with immutable or custom dictionary-like objects."
        )
        pass

    return kwargs_copy
