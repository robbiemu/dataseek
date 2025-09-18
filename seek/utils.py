"""
Simple utilities for deduplicating and ranking research results within a single session.

This replaces the complex ResearchCache approach with simpler functions that operate
on the research_session_cache in the state.
"""

import os
import re
import time
from typing import Any


def strip_reasoning_block(content: str, tags: list[str] = None) -> str:
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


def get_characteristic_context(task: dict, mission_config: dict) -> str | None:
    """
    Finds the definitional context for a characteristic from the mission config.
    """
    if not task or not mission_config:
        return None

    characteristic_name = task.get("characteristic")
    if not characteristic_name:
        return None

    # Search through all missions and goals to find the matching context
    for mission in mission_config.get("missions", []):
        for goal in mission.get("goals", []):
            if goal.get("characteristic") == characteristic_name:
                return goal.get("context")  # Return the context string

    return None  # Return None if no matching characteristic is found


def get_claimify_strategy_block(characteristic: str) -> str:
    """Get the strategic focus block for Claimify characteristics in data prospecting."""
    strategies = {
        "Decontextualization": """
**Strategic Focus for Decontextualization:**
Look for formal, encyclopedic, reference-style text that presents facts in a neutral, standalone manner. Ideal sources include:
- Academic papers with clear factual statements
- Technical documentation with precise specifications
- News articles with objective reporting style
- Reference materials like encyclopedias or handbooks

Avoid sources with:
- Heavy contextual dependencies ("as mentioned above", "this approach")
- Conversational or informal tone
- Opinion pieces or subjective commentary

The best documents will have sentences that can be extracted and understood independently, without needing surrounding context.""",
        "Coverage": """
**Strategic Focus for Coverage:**
Seek data-dense, comprehensive sources that thoroughly cover their subject matter with factual breadth. Ideal sources include:
- Comprehensive reports or surveys
- Statistical summaries and data compilations
- Complete technical specifications
- Thorough news coverage of events
- Academic literature reviews

Avoid sources with:
- Narrow, single-topic focus
- Sparse factual content
- Heavily theoretical or abstract content

The best documents will be rich repositories of diverse, verifiable facts that demonstrate comprehensive coverage of their domain.""",
        "Entailment": """
**Strategic Focus for Entailment:**
Target sources with clear, logical, unambiguous sentence structures that support straightforward factual claims. Ideal sources include:
- Technical manuals with step-by-step processes
- Scientific papers with clear methodology sections
- News reports with direct factual statements
- Educational materials with explicit explanations
- Legal or regulatory documents with precise language

Avoid sources with:
- Complex, multi-clause sentences
- Ambiguous or vague language
- Heavy use of metaphors or figurative language
- Speculative or hypothetical statements

The best documents will have simple, direct sentences where the logical relationship between premise and conclusion is crystal clear.""",
    }
    return strategies.get(
        characteristic,
        f"Look for sources that demonstrate clear {characteristic} characteristics in their writing style and structure.",
    )


def append_to_pedigree(
    pedigree_path: str, entry_markdown: str, run_id: str | None = None
) -> dict[str, Any]:
    """
    Appends a markdown entry to the pedigree file in a standardized block with a timestamp.

    Args:
        pedigree_path: The full path to the pedigree file.
        entry_markdown: The markdown content to append to the file.
        run_id: An optional unique identifier for the run.

    Returns:
        A dictionary containing the status of the operation.
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    header = f"---\nrun_id: {run_id or 'N/A'}\ntimestamp: {timestamp}\n---\n"
    final_entry = header + entry_markdown + "\n"
    try:
        # Ensure the directory exists before attempting to write the file
        directory = os.path.dirname(pedigree_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(pedigree_path, "a", encoding="utf-8") as f:
            f.write(final_entry)

        return {
            "pedigree_path": pedigree_path,
            "status": "ok",
            "entry_snippet": final_entry[:200],
        }
    except Exception as e:
        return {
            "pedigree_path": pedigree_path,
            "status": "error",
            "entry_snippet": None,
            "error": f"{type(e).__name__}: {str(e)}",
        }
