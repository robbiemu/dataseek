import json

from seek.common.defensive_model_adapter import (
    fix_malformed_json_arguments,
    sanitize_provider_kwargs,
)


class TestDefensiveModelAdapter:
    def test_fix_malformed_json_arguments(self):
        # Test with valid JSON
        valid_json = '{"key": "value"}'
        repaired = fix_malformed_json_arguments(valid_json)
        assert json.loads(repaired) == {"key": "value"}

        # Test with malformed JSON (missing quote)
        malformed = '{"key": value}'
        repaired = fix_malformed_json_arguments(malformed)
        # json-repair should fix it to '{"key": "value"}' or similar
        assert json.loads(repaired) == {"key": "value"}  # Assuming repair adds quote

        # Test with empty string
        assert fix_malformed_json_arguments("") == "{}"

        # Test with non-string
        assert fix_malformed_json_arguments(123) == 123

    def test_sanitize_provider_kwargs(self):
        # Test tool_choice sanitization
        kwargs = {"tool_choice": "any"}
        sanitized = sanitize_provider_kwargs(**kwargs)
        assert sanitized["tool_choice"] == "auto"

        # Test extra_body sanitization
        kwargs = {"extra_body": {"tool_choice": "any"}}
        sanitized = sanitize_provider_kwargs(**kwargs)
        assert sanitized["extra_body"]["tool_choice"] == "auto"

        # Test unchanged kwargs
        kwargs = {"temperature": 0.1}
        sanitized = sanitize_provider_kwargs(**kwargs)
        assert sanitized["temperature"] == 0.1
        assert len(sanitized) == 1  # No extra keys added
