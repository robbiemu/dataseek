import os
import time
from typing import Any

from langchain_core.messages import AIMessage

from seek.common.config import get_active_seek_config, get_prompt
from seek.components.mission_runner.state import DataSeekState
from seek.components.tool_manager.tools import write_file

from .utils import create_agent_runnable, create_llm
import os


def archive_node(state: "DataSeekState") -> dict:
    """
    The archive node, responsible for saving data and updating the audit trail
    using a procedural approach.
    """
    # Load seek config instead of main config for writer paths
    get_active_seek_config()
    llm = create_llm("archive")

    # Extract content from research findings
    provenance = state.get("current_sample_provenance", "synthetic")
    print(f"   🏷️  Archive: Received provenance '{provenance}' from state")
    messages = state.get("messages", [])
    research_findings_any = state.get("research_findings", [])

    # Robustly find the document content
    document_content = None
    if research_findings_any:
        # Content is now a list of strings, so we join them.
        rf_list: list[str] = (
            [str(x) for x in research_findings_any]
            if isinstance(research_findings_any, list)
            else []
        )
        document_content = "\n\n---\n\n".join(rf_list)
        print("   ✅ Archive: Found content in 'research_findings'.")
    else:
        # Fallback for older states or different paths
        for msg in reversed(messages):
            if hasattr(msg, "content") and "# Data Prospecting Report" in msg.content:
                document_content = msg.content
                print("   ⚠️  Archive: Found content via message search fallback.")
                break

    if not document_content:
        error_message = AIMessage(
            content="Archive Error: Could not find a 'Data Prospecting Report' in the conversation history to save."
        )
        return {"messages": messages + [error_message]}

    # Get task details for naming
    current_task = state.get("current_task")
    characteristic = "unknown"
    topic = "unknown"
    if current_task:
        characteristic = current_task.get("characteristic", "unknown").lower().replace(" ", "_")
        topic = current_task.get("topic", "unknown").lower().replace(" ", "_")

    # Request raw markdown string directly from LLM
    tpl = get_prompt("archive", "base_prompt")
    system_prompt = tpl.format(provenance=provenance, characteristic=characteristic)
    agent_runnable = create_agent_runnable(llm, system_prompt, "archive")
    llm_result = agent_runnable.invoke({"messages": messages})

    # The LLM's raw output is now our entry markdown. No parsing needed.
    entry_markdown = llm_result.content

    # --- Procedural control flow ---

    # Resolve base output directory from mission_config.tool_configs.file_saver.output_path
    mission_cfg = state.get("mission_config", {}) or {}
    tool_cfgs = mission_cfg.get("tool_configs", {}) if isinstance(mission_cfg, dict) else {}
    file_saver_cfg = tool_cfgs.get("file_saver", {}) if isinstance(tool_cfgs, dict) else {}
    base_output_dir = file_saver_cfg.get("output_path") or ""
    # Get paths from state (may be relative)
    samples_path_rel = state.get("samples_path") or "samples"
    pedigree_rel = state.get("pedigree_path") or "PEDIGREE.md"

    # Generate a unique timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")

    # Construct the deterministic filepath using mission state
    filename = f"{characteristic}_{topic}_{timestamp}.md"
    if os.path.isabs(samples_path_rel) or not base_output_dir:
        samples_dir = samples_path_rel
    else:
        samples_dir = os.path.join(base_output_dir, samples_path_rel)
    filepath = f"{samples_dir}/{filename}"
    abs_filepath = filepath

    # Now use this 'filepath' variable in the write_file tool.
    write_result = write_file.invoke({"filepath": filepath, "content": document_content})

    if write_result.get("status") == "ok":
        print(f"📄 Archive: Wrote sample to '{abs_filepath}' ({write_result.get('bytes_written', 0)} bytes)")
        run_id = state.get("run_id")
        # Use pedigree path from mission state (resolve relative to base)
        if os.path.isabs(pedigree_rel) or not base_output_dir:
            pedigree_path = pedigree_rel
        else:
            pedigree_path = os.path.join(base_output_dir, pedigree_rel)
        append_to_pedigree(
            pedigree_path=pedigree_path,
            entry_markdown=entry_markdown,
            run_id=run_id,
        )
    else:
        err = write_result.get('error')
        print(f"❌ Archive: Failed to write sample to '{filepath}': {err}")
        error_message = AIMessage(
            content=f"Archive Error: Failed to write file to '{filepath}'. Error: {err}"
        )
        return {"messages": messages + [error_message]}

    # --- Progress counters ---
    is_synthetic_sample = provenance == "synthetic"
    synthetic_samples_generated = state.get("synthetic_samples_generated", 0)
    research_samples_generated = state.get("research_samples_generated", 0)

    if is_synthetic_sample:
        synthetic_samples_generated += 1
    else:
        research_samples_generated += 1

    samples_generated = state.get("samples_generated", 0) + 1
    total_target = state.get("total_samples_target", 1200)
    total_samples = synthetic_samples_generated + research_samples_generated
    synthetic_pct = (synthetic_samples_generated / total_samples) if total_samples > 0 else 0.0
    progress_pct = (samples_generated / total_target) * 100 if total_target > 0 else 0
    source_type = "synthetic" if is_synthetic_sample else "research"

    print(
        f"   📊 Sample #{samples_generated} archived ({progress_pct:.1f}% complete) - Source: {source_type}"
    )
    print(
        f"   📈 Updated ratios - Research: {research_samples_generated}, Synthetic: {synthetic_samples_generated} ({synthetic_pct:.1%})"
    )

    confirmation_message = AIMessage(
        content=f"Successfully archived document to '{filepath}' and updated the pedigree."
    )
    return {
        "messages": messages + [confirmation_message],
        "samples_generated": samples_generated,
        "synthetic_samples_generated": synthetic_samples_generated,
        "research_samples_generated": research_samples_generated,
        "last_action_agent": "archive",
    }


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
