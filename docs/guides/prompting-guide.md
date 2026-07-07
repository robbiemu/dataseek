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

## Overriding prompts (`--prompts`)

Pass `--prompts path/to/your_prompts.yaml` to supply your own prompt file. The
override is layered over the bundled `config/prompts.yaml` with **per-role
replace** semantics — not a deep per-key merge:

- **A role you specify in the override replaces that role's prompt set
  wholesale.** If the bundled `research:` has `base_prompt`, `normal_prompt`,
  and `cached_only_prompt`, and your override specifies `research:` with only
  `base_prompt`, then `research.normal_prompt` and `research.cached_only_prompt`
  are **empty** for your run — they do not leak in from the bundled file.
- **A role you omit from the override keeps its bundled prompts entirely.**
  If your override has only a `research:` block, then `fitness`, `archive`,
  `supervisor`, and `synthetic` use their bundled prompts unchanged.

This is deliberate. Prompt roles are cohesive units: nodes read multiple keys
per role together (for example, `research` concatenates `base_prompt` +
`normal_prompt`). Deep-merging one key would leak the bundled sibling keys into
a role you intend to fully replace — so a custom-tools mission overriding
`research.base_prompt` would still inherit a bundled `research.normal_prompt`
instructing the model to use web tools that aren't bound. Per-role replace
avoids that.

**Practical consequence:** when you override a role, you must re-specify every
key that role's node reads (see the per-node sections below for which keys each
node uses). For `research`, that's `base_prompt`, `normal_prompt`, and
`cached_only_prompt`. For roles with only `base_prompt` (`fitness`, `archive`,
`synthetic`), specifying the role with just `base_prompt` is a full replacement.

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
