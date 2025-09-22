from pathlib import Path

from langchain_core.messages import AIMessage

from seek.components.search_graph.nodes.archive import archive_node


def test_archive_node_writes_sample_and_pedigree(tmp_path, monkeypatch):
    # Arrange state with temp output paths
    samples_dir = tmp_path / "datasets" / "mac_ai_corpus" / "samples"
    pedigree_path = tmp_path / "datasets" / "mac_ai_corpus" / "PEDIGREE.md"

    state = {
        "current_sample_provenance": "researched",
        "research_findings": ["# Data Prospecting Report\n\nSome content"],
        "messages": [],
        "current_task": {"characteristic": "Controlled Experimental Design", "topic": "AI"},
        "samples_path": str(samples_dir),
        "pedigree_path": str(pedigree_path),
    }

    # Stub LLM path used to generate the pedigree entry
    class Dummy:
        def invoke(self, _):
            return AIMessage(
                content="### 2024-01-01 -- Sample Archived\n- **Source Type:** researched\n- **Target Characteristic:** controlled_experimental_design"
            )

    monkeypatch.setattr(
        "seek.components.search_graph.nodes.archive.create_agent_runnable",
        lambda *_args, **_kwargs: Dummy(),
    )

    # Act
    result = archive_node(state)

    # Assert
    assert result["samples_generated"] == 1
    # One markdown file should exist under samples_dir
    assert samples_dir.exists()
    md_files = list(samples_dir.glob("*.md"))
    assert md_files, "Expected a markdown sample file to be written"
    # Pedigree file should exist and contain entry
    assert Path(pedigree_path).exists()
    content = Path(pedigree_path).read_text(encoding="utf-8")
    assert "Sample Archived" in content
