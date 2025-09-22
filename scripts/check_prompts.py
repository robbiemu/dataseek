#!/usr/bin/env python3
"""
Prompt template verifier for DataSeek.

What it does:
- Scans the codebase for get_prompt("role", "key") calls to derive the set of required
  prompt templates.
- Loads config/prompts.yaml and verifies:
  * All required templates are present
  * No unused templates exist (defined but never referenced in code)
  * Each template's placeholders match the expected variables for that template

Usage:
  scripts/check_prompts.py [--prompts config/prompts.yaml] [--src seek/components/search_graph/nodes]

Exit codes:
  0 on success, non-zero if any issues are found.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from collections.abc import Iterable
from string import Formatter

import yaml


def find_required_prompts(src_dirs: list[pathlib.Path]) -> set[tuple[str, str]]:
    required: set[tuple[str, str]] = set()
    for src_dir in src_dirs:
        for path in src_dir.rglob("*.py"):
            try:
                node = ast.parse(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name != "get_prompt":
                    continue
                # Expect two positional string args: role, key
                if len(call.args) >= 2:
                    a0, a1 = call.args[0], call.args[1]
                    if isinstance(a0, ast.Constant) and isinstance(a1, ast.Constant):
                        if isinstance(a0.value, str) and isinstance(a1.value, str):
                            required.add((a0.value, a1.value))
    return required


def load_prompts(path: pathlib.Path) -> dict[str, dict[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, str]] = {}
    for role, mapping in data.items():
        if isinstance(mapping, dict):
            out[role] = {k: v for k, v in mapping.items() if isinstance(v, str)}
    return out


EXPECTED_PLACEHOLDERS: dict[tuple[str, str], set[str]] = {
    ("research", "normal_prompt"): {"characteristic", "topic", "strategy_block"},
    (
        "research",
        "cached_only_prompt",
    ): {"characteristic", "topic", "strategy_block", "allowed_urls_list", "cache_context"},
    (
        "fitness",
        "base_prompt",
    ): {
        "characteristic",
        "topic",
        "strategy_block",
        "provenance_guidance",
        "research_findings",
        "fitness_schema",
    },
    ("synthetic", "base_prompt"): {"characteristic", "topic", "strategy_block"},
    ("archive", "base_prompt"): {"provenance", "characteristic"},
    ("supervisor", "base_prompt"): {"research_detail"},
    (
        "supervisor",
        "mission_context",
    ): {
        "current_task_str",
        "mission_status",
        "decision_history",
        "consecutive_failures",
        "last_action_analysis",
        "strategic_guidance",
    },
    (
        "supervisor_cache_selection",
        "base_prompt",
    ): {
        "characteristic",
        "topic",
        "strategy_block",
        "total_samples_generated",
        "research_samples_generated",
        "synthetic_samples_generated",
        "total_samples_target",
        "synthetic_budget",
        "max_synthetic_samples",
        "current_synthetic_pct",
        "remaining_synthetic_budget",
        "remaining_total_needed",
        "candidates_text",
        "excluded_urls",
    },
}


def iter_placeholders(template: str) -> Iterable[str]:
    # Extract placeholders from a python str.format template
    for _literal_text, field_name, _format_spec, _conversion in Formatter().parse(template):
        if field_name:
            # Skip escaped braces ("{{" or "}}") which appear with None field_name
            yield field_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        default="config/prompts.yaml",
        help="Path to prompts.yaml",
    )
    parser.add_argument(
        "--src",
        action="append",
        default=["seek/components/search_graph/nodes"],
        help="Source directory to scan for get_prompt() calls (can repeat)",
    )
    parser.add_argument(
        "--allow-extra",
        action="store_true",
        default=False,
        help="Do not fail on prompts defined in YAML but unused in code (default: False)",
    )
    args = parser.parse_args()

    prompts_path = pathlib.Path(args.prompts)
    src_dirs = [pathlib.Path(p) for p in args.src]

    missing: list[str] = []
    extra: list[str] = []
    placeholder_errors: list[str] = []

    required = find_required_prompts(src_dirs)
    defined = load_prompts(prompts_path)

    defined_pairs = {(role, key) for role, mapping in defined.items() for key in mapping}

    for role, key in sorted(required):
        if role not in defined or key not in defined[role]:
            missing.append(f"Missing prompt: {role}.{key}")

    if not args.allow_extra:
        for role, key in sorted(defined_pairs - required):
            extra.append(f"Unused prompt: {role}.{key}")

    # Placeholder checks only for required templates we can parse
    for role, key in sorted(required):
        tpl = defined.get(role, {}).get(key)
        if not tpl:
            continue
        placeholders = set(iter_placeholders(tpl))
        expected = EXPECTED_PLACEHOLDERS.get((role, key))
        if expected is None:
            # Unknown template; skip strict check but warn if there are placeholders
            continue
        unexpected = placeholders - expected
        missing_fields = expected - placeholders
        if unexpected:
            placeholder_errors.append(
                f"Unexpected placeholders in {role}.{key}: {sorted(unexpected)}"
            )
        if missing_fields:
            placeholder_errors.append(
                f"Missing placeholders in {role}.{key}: expected {sorted(missing_fields)}"
            )

    issues = missing + extra + placeholder_errors
    if issues:
        print("Prompt verification failed:\n")
        for msg in issues:
            print(f" - {msg}")
        print(
            "\nTip: Ensure config/prompts.yaml covers all get_prompt(role,key) references, "
            "and that each template includes only the placeholders the node supplies."
        )
        return 1
    else:
        print("All prompts verified. ✔")
        print(f"Required templates: {len(required)}; Defined templates: {len(defined_pairs)}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
