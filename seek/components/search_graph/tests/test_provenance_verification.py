from langchain_core.messages import AIMessage

from seek.components.search_graph.nodes.supervisor import supervisor_node


def test_supervisor_verifies_provenance_from_cache(monkeypatch):
    # Build a minimal state to trigger the deterministic 'Data Prospecting Report' path
    report_url = "https://arxiv.org/pdf/0708.0375"
    report_message = AIMessage(
        content=(
            "# Data Prospecting Report\n\n"
            "**Target Characteristic**: `Controlled Experimental Design`\n"
            f'**Source URL**: `"{report_url}"`\n'
            "\n## Retrieved Content (Markdown)\nLorem ipsum."
        )
    )

    # Research cache contains an entry with matching URL and ok status
    research_cache = [
        {
            "call_id": "call_1",
            "tool": "url_to_markdown",
            "args": {"url": report_url},
            "output": {"status": "ok", "markdown": "..."},
        }
    ]

    state = {
        "messages": [report_message],
        "decision_history": ["research"],
        "progress": {"test_mission": {}},
        "current_task": {"characteristic": "Controlled Experimental Design", "topic": "AI"},
        "research_session_cache": research_cache,
    }

    # Avoid creating real LLM
    class Dummy:
        pass

    monkeypatch.setattr(
        "seek.components.search_graph.nodes.supervisor.create_llm", lambda _role: Dummy()
    )

    result = supervisor_node(state)
    assert result["next_agent"] == "fitness"
    assert result["current_sample_provenance"] == "researched"
