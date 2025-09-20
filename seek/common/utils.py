"""
Simple utilities for deduplicating and ranking research results within a single session.

This replaces the complex ResearchCache approach with simpler functions that operate
on the research_session_cache in the state.
"""

import re


def strip_reasoning_block(content: str, tags: list[str] | None = None) -> str:
    """
    Removes a reasoning block from the beginning of a string if present.

    This function can strip blocks denoted by various tags like <think>,
    <scratchpad>, <reasoning>, etc.

    Args:
        content: The input string.
        tags: A list of tag names to look for. Defaults to a standard list.

    Returns:
        The string with the initial reasoning block removed.
    """
    if tags is None:
        tags = [
            "think",
            "thinking",
            "thought",
            "scratchpad",
            "reasoning",
            "plan",
            "reflection",
            "rationale",
        ]

    # Create a regex 'or' condition by joining the tags with '|'
    # This will match any of the words in the list.
    tag_pattern = "|".join(tags)

    # The main pattern now uses the tag_pattern.
    # - <({tag_pattern})>: Captures the specific tag found (e.g., "scratchpad").
    # - <\/\1>: The backreference \1 ensures the closing tag matches the opening one.
    pattern = rf"^\s*<({tag_pattern})>(.*?)<\/\1>\s*"

    return re.sub(pattern, "", content, count=1, flags=re.DOTALL | re.IGNORECASE)
