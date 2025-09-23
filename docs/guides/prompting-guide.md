# Prompting Guide

This guide explains how the DataSeek agent assembles prompts for each node, the
template variables that appear in `config/prompts.yaml`, and how they map to the
mission state.

## Where prompts live

- Active prompts: `config/prompts.yaml`
- Golden example packs: `examples/claimify/prompts.yaml` (compatible superset)

`scripts/check_prompts.py` verifies that every `get_prompt("role","key")` call
in the agent code has a matching template and that required placeholders are
present.

Run:

```
python scripts/check_prompts.py --prompts config/prompts.yaml
```

## Common assembly rules

- All system prompts pass through a safety step that escapes curly braces before
  building a `ChatPromptTemplate`. This prevents literal braces from being
  misinterpreted as template variables.
- When a node requires structured output, we prefer
  `llm.with_structured_output(...)` and fall back to JSON repair + validation.

## Supervisor

- `supervisor.base_prompt`
  - Variables: `{research_detail}`
  - Provided by the node based on whether research is allowed in the remaining
    steps.

- `supervisor.mission_context`
  - Variables: `{current_task_str}`, `{mission_status}`, `{decision_history}`,
    `{consecutive_failures}`, `{last_action_analysis}`, `{strategic_guidance}`
  - Derived from mission state; describes recent behavior and guidance.

- `supervisor_cache_selection.base_prompt`
  - Variables: `{characteristic}`, `{topic}`, `{strategy_block}`,
    `{total_samples_generated}`, `{research_samples_generated}`,
    `{synthetic_samples_generated}`, `{total_samples_target}`,
    `{synthetic_budget}`, `{max_synthetic_samples}`, `{current_synthetic_pct}`,
    `{remaining_synthetic_budget}`, `{remaining_total_needed}`,
    `{candidates_text}`, `{excluded_urls}`
  - Used when selecting cached sources for the next cycle.

## Research

- Assembly: `research.base_prompt + research.normal_prompt` (or
  `cached_only_prompt` when search tools are disabled)

- `research.base_prompt`
  - Variables: `{characteristic}`, `{topic}`

- `research.normal_prompt`
  - Variables: `{characteristic}`, `{topic}`, `{strategy_block}`
  - Specifies the “Data Prospecting Report” format.

- `research.cached_only_prompt`
  - Variables: `{characteristic}`, `{topic}`, `{strategy_block}`,
    `{allowed_urls_list}`, `{cache_context}`
  - Enforces whitelist + cache usage only; may produce “No More Cached Data”.

## Fitness

- `fitness.base_prompt`
  - Variables: `{characteristic}`, `{topic}`, `{strategy_block}`,
    `{provenance_guidance}`, `{research_findings}`, `{fitness_schema}`
  - Requires JSON-only output that validates as `FitnessReport`.

## Synthetic

- `synthetic.base_prompt`
  - Variables: `{characteristic}`, `{topic}`, `{strategy_block}`
  - Generates content in the same general report style used by research, then
    routes directly to archive.

## Archive

- `archive.base_prompt`
  - Variables: `{provenance}`, `{characteristic}`
  - Produces a short pedigree entry in Markdown.

## Output path rules

- The archive node writes files using:
  - `mission_config.output_paths.base_path` as a base directory.
  - `output_paths.samples_path` and `output_paths.audit_trail_path` are resolved
    relative to that base directory when they are not absolute paths.
  - Defaults (when not specified): `samples/` and `PEDIGREE.md`.

Example:

```
output_paths:
  base_path: "datasets/mac_ai_corpus"
  samples_path: "samples"
  audit_trail_path: "PEDIGREE.md"
```

Result:

- Samples written to `datasets/mac_ai_corpus/samples/`.
- Pedigree written to `datasets/mac_ai_corpus/PEDIGREE.md`.
