# LiteLLM 1.77.3 → 1.91.0 Patch Compatibility Review

**Task:** Verify that dataseek's custom LiteLLM patches land identically after the security-driven bump from 1.77.3 → 1.91.0 (14 minor versions).
**Method:** Source diff of old vs new LiteLLM wheels for every patched symbol, plus runtime verification that each patch installs and flows end-to-end under the new version.
**Verdict:** ✅ **All patches land identically. No code changes required.** One pre-existing latent issue surfaced (unrelated to the version bump) — documented below.

---

## Custom LiteLLM touch points (inventory)

All custom code lives in two files, loaded for side effects at startup via `seek/__init__.py`:

| # | File:line | Target symbol | Patch | Guard |
|---|---|---|---|---|
| 1 | `patch.py:368` | `litellm.litellm_core_utils.prompt_templates.factory.ollama_pt` | Replace with `_patched_ollama_pt` | **Bare** |
| 2 | `patch.py:372` | `litellm.llms.ollama.completion.transformation.ollama_pt` | Replace with `_patched_ollama_pt` | **Bare** |
| 3 | `patch.py:376-384` | `litellm.main.ollama_pt` | Replace if present | Defensive (`hasattr` + try/except) |
| 4 | `patch.py:386-391` | `litellm.ollama_pt` (root) | Replace if present | Defensive (`hasattr` + try/except) |
| 5 | `patch.py:394-411` | `sys.modules` sweep | Late-patch any loaded module exposing `ollama_pt` | Defensive (try/except) |
| 6 | `patch.py:414-466` | `__builtins__.__import__` + `sys.meta_path` | Import hook to catch future `ollama_pt` imports | Defensive (try/except) |
| 7 | `patch.py:469-624` | `litellm.completion` | Wrap: detect dropped tool calls/content for Ollama, fall back to direct HTTP | Defensive (try/except, falls back to original) |

Supporting helpers in `seek/common/defensive_model_adapter.py`: `fix_malformed_json_arguments`, `sanitize_provider_kwargs` (`tool_choice: "any" → "auto"`). No LiteLLM imports there.

---

## Old vs new source comparison

### 1. `ollama_pt` — the bare-patched function (touch points #1, #2)

| Aspect | 1.77.3 (old) | 1.91.0 (new) | Compatible? |
|---|---|---|---|
| File path | `litellm/litellm_core_utils/prompt_templates/factory.py` | same | ✅ |
| Definition | `def ollama_pt(model: str, messages: list)` at line 191 | `def ollama_pt(model: str, messages: list)` at line 199 | ✅ signature identical |
| Imported by transformation | `from ...factory import ollama_pt` (line 15), called as `ollama_pt(model=model, messages=messages)` (line 403) | `from ...factory import ollama_pt` (line 16), called as `ollama_pt(model=model, messages=messages)` (line 366) | ✅ call site unchanged |
| Transformation file path | `litellm/llms/ollama/completion/transformation.py` | same | ✅ |

**Our replacement signature:** `_patched_ollama_pt(model, messages, roles=None, model_kwargs=None)`. The two extra optional params are never passed by the caller (which uses `model=`/`messages=` keywords), so they default safely. **Compatible.**

The bug the original patch fixes (IndexError when the last message is from the assistant) is still present in upstream 1.91.0's `ollama_pt` — the patch is still needed.

### 2. `litellm.completion` — the wrapper (touch point #7)

| Aspect | 1.77.3 (old) | 1.91.0 (new) | Compatible? |
|---|---|---|---|
| Definition location | `main.py:888` | `main.py:4705` | ✅ (moved, still exists) |
| Core signature params | `model, messages, timeout, temperature, top_p, n, stream, ..., tools, tool_choice, ...` | identical core params | ✅ |
| New optional params | — | added `verbosity`, `reasoning_effort` gained `"xhigh"` | ✅ additive; wrapper passes `**kwargs` through |
| `ModelResponse.choices[0].message.tool_calls` | present | present (runtime-verified) | ✅ |
| `ModelResponse.choices[0].message.content` | present | present (runtime-verified) | ✅ |

**The wrapper's assumptions all hold:** the response shape it introspects (`choices[0].message.tool_calls`, `.content`) is intact; `sanitize_provider_kwargs` operates on `tool_choice`/`extra_body` which remain valid kwargs. **Compatible.**

### 3. Optional symbols (touch points #3, #4)

`litellm.main.ollama_pt` and `litellm.ollama_pt` (root) are **absent in both versions** — the defensive `hasattr` checks correctly no-op. **Compatible (no-op as designed).**

---

## Runtime verification (under litellm 1.91.0)

```
✅ Patched litellm.litellm_core_utils.prompt_templates.factory.ollama_pt
✅ Patched litellm.llms.ollama.completion.transformation.ollama_pt
✅ Patched litellm.main.ollama_pt  (no-op, absent)
✅ Patched litellm.ollama_pt        (no-op, absent)
✅ Applied comprehensive LiteLLM patch for Ollama tool calls
```

- All 4 bare patches resolve (no ImportError/AttributeError).
- After patching: `factory.ollama_pt is _patched_ollama_pt` ✓, `transformation.ollama_pt is _patched_ollama_pt` ✓, `litellm.completion.__name__ == 'patched_litellm_completion'` ✓.
- End-to-end completion call through the wrapper (stubbed Ollama server): returned a valid `ModelResponse` with `choices[0].message` containing both `.content` and `.tool_calls`; `tool_choice: "any" → "auto"` rewrite applied. The wrapper's "suspicious response" detection and direct-call fallback both exercised without error.
- Full test suite: **69/69 pass**.

---

## One pre-existing latent issue (surfaced, not introduced)

> `⚠️ Could not install import hook: 'dict' object has no attribute '__import__'`

**Cause:** `patch.py:432` does `__builtins__["__import__"] = patched_import`, which assumes `__builtins__` is the `builtins` **module**. In some execution contexts (e.g. `python -c "..."`), CPython exposes `__builtins__` as a **dict** instead, so the assignment fails.

**Is this a 1.91.0 regression?** No. It is pure Python semantics — `__builtins__`'s type depends on execution context, not on the LiteLLM version. It behaves identically under 1.77.3. It is caught by the surrounding `try/except` and logged, so it is non-fatal.

**Does it matter?** No. The import hook (touch point #6) is a redundant safety net: its job is to late-patch `ollama_pt` in modules imported *after* the patch runs. But the only runtime caller of `ollama_pt` is `litellm.llms.ollama.completion.transformation`, which is already patched at import time by the direct patches (#1, #2) and the `sys.modules` sweep (#5). Runtime-verified: `transformation.ollama_pt is _patched_ollama_pt` holds even when the import hook fails to install.

**Recommendation (optional, separate from this review):** the `__builtins__` access is the most invasive part of the patch (it replaces the global import builtin process-wide). Since it is both failing in some contexts and redundant, a future cleanup could remove touch point #6 entirely and rely on #1, #2, and #5. Not blocking for this bump.

---

## Conclusion

The LiteLLM 1.77.3 → 1.91.0 bump is **patch-compatible**: every custom touch point resolves at the same paths with the same signatures, the response shapes the wrapper depends on are intact, and end-to-end runtime behavior is preserved. No code changes to `seek/components/patch.py` or `seek/common/defensive_model_adapter.py` are required.

The patches remain as brittle to *future* LiteLLM refactors as they were before (the two bare patches at `patch.py:31-32, 368, 372` would crash if LiteLLM renames `litellm_core_utils` or moves `ollama_pt`), but that is a pre-existing fragility, not something this bump introduced or worsened.
