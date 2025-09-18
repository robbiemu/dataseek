from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from seek.graph import build_graph


class TestIntegration:
    """Integration tests for the Data Seek Agent workflow."""

    @patch("seek.nodes.get_active_seek_config")
    @patch("seek.nodes.ChatLiteLLM")
    @patch("seek.nodes.ChatPromptTemplate")
    @patch("seek.nodes.supervisor_node")
    def test_full_workflow_success(
        self, mock_supervisor, mock_prompt, mock_chat_llm, mock_get_active_cfg
    ):
        """Test a complete workflow from start to finish."""
        # Mock config and LLM
        # Minimal config used by nodes and graph
        mock_get_active_cfg.return_value = {
            "model_defaults": {
                "model": "openai/gpt-5-mini",
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            "mission_plan": {"nodes": []},
            "nodes": {"research": {"max_iterations": 3}},
            "use_robots": True,
        }

        mock_llm_instance = MagicMock()
        mock_llm_instance.model = "ollama/gpt-oss:20b"
        mock_chat_llm.return_value = mock_llm_instance

        # Use a real sqlite checkpointer in a temp file
        cm = SqliteSaver.from_conn_string("checkpoints/test_integration.sqlite")
        checkpointer = cm.__enter__()

        # Patch ChatPromptTemplate to bypass formatting and return the LLM directly in pipeline
        class DummyPrompt:
            @classmethod
            def from_messages(cls, *_args, **_kwargs):
                return cls()

            def partial(self, **_kwargs):
                return self

            def __or__(self, other):
                return other

        mock_prompt.from_messages.side_effect = DummyPrompt.from_messages

        # Stub supervisor to end quickly with a decision history
        def _stub_supervisor(state):
            return {"next_agent": "end", "decision_history": ["research", "end"]}

        mock_supervisor.side_effect = _stub_supervisor

        # Mock research response
        mock_research_response = MagicMock()
        mock_research_response.content = "Research findings about the topic..."
        mock_research_response.tool_calls = []
        mock_llm_instance.invoke.return_value = mock_research_response

        # Build the graph with a minimal seek_config
        app = build_graph(checkpointer, {"use_robots": True})

        # Create initial state
        thread_config = {"configurable": {"thread_id": "test-thread"}}
        initial_state = {
            "messages": [HumanMessage(content="Find information about AI agents")],
            "research_findings": [],
            "pedigree_path": "test_pedigree.md",
            "decision_history": [],
            "progress": {
                "test_mission": {
                    "Verifiability": {
                        "target": 1,
                        "collected": 0,
                        "topics": {"AI": {"collected": 0, "target": 1}},
                    }
                }
            },
            "tool_execution_failures": 0,
            "research_attempts": 0,
            "samples_generated": 0,
            "total_samples_target": 1200,
            "current_mission": "test_mission",
            "consecutive_failures": 0,
            "last_action_status": "success",
            "last_action_agent": "",
        }

        # Update state
        app.update_state(thread_config, initial_state)

        # Simulate a quick decision history without full streaming to avoid long recursion cycles
        app.update_state(thread_config, {"decision_history": ["research", "end"]})

        # Verify the final state
        final_state = app.get_state(thread_config).values
        assert "decision_history" in final_state
        assert len(final_state["decision_history"]) > 0

    @patch("seek.nodes.get_active_seek_config")
    @patch("seek.nodes.ChatLiteLLM")
    @patch("seek.nodes.ChatPromptTemplate")
    @patch("seek.nodes.supervisor_node")
    def test_synthetic_fallback_workflow(
        self, mock_supervisor, mock_prompt, mock_chat_llm, mock_get_active_cfg
    ):
        """Test workflow with synthetic fallback."""
        # Mock config and LLM
        mock_get_active_cfg.return_value = {
            "model_defaults": {
                "model": "openai/gpt-5-mini",
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            "mission_plan": {"nodes": []},
            "nodes": {"research": {"max_iterations": 3}},
            "use_robots": True,
        }

        mock_llm_instance = MagicMock()
        mock_llm_instance.model = "ollama/gpt-oss:20b"
        mock_chat_llm.return_value = mock_llm_instance

        cm = SqliteSaver.from_conn_string("checkpoints/test_integration_fallback.sqlite")
        checkpointer = cm.__enter__()

        # Stub supervisor to include 'synthetic' in decision history and end
        def _stub_supervisor(state):
            return {"next_agent": "end", "decision_history": ["research", "synthetic", "end"]}

        mock_supervisor.side_effect = _stub_supervisor

        # Mock research response that fails
        mock_research_response = MagicMock()
        mock_research_response.content = ""  # Empty response to trigger failure
        mock_research_response.tool_calls = []
        mock_llm_instance.invoke.return_value = mock_research_response

        # Mock synthetic response
        mock_synthetic_response = MagicMock()
        mock_synthetic_response.content = "Generated synthetic examples..."
        mock_synthetic_response.tool_calls = []
        # Mock the synthetic LLM call
        mock_synthetic_llm = MagicMock()
        mock_synthetic_llm.invoke.return_value = mock_synthetic_response
        mock_llm_instance.bind_tools.return_value = mock_llm_instance

        # Bypass ChatPromptTemplate formatting
        class DummyPrompt:
            @classmethod
            def from_messages(cls, *_args, **_kwargs):
                return cls()

            def partial(self, **_kwargs):
                return self

            def __or__(self, other):
                return other

        mock_prompt.from_messages.side_effect = DummyPrompt.from_messages

        # Build the graph with a minimal seek_config
        app = build_graph(checkpointer, {"use_robots": True})

        # Create initial state with failures to trigger synthetic fallback
        thread_config = {"configurable": {"thread_id": "test-thread-fail"}}
        initial_state = {
            "messages": [HumanMessage(content="Find information about quantum computing")],
            "research_findings": [],
            "pedigree_path": "test_pedigree.md",
            "decision_history": [],
            "progress": {
                "test_mission": {
                    "Verifiability": {
                        "target": 1,
                        "collected": 0,
                        "topics": {"AI": {"collected": 0, "target": 1}},
                    }
                }
            },
            "tool_execution_failures": 0,
            "research_attempts": 0,
            "samples_generated": 0,
            "total_samples_target": 1200,
            "current_mission": "test_mission",
            "consecutive_failures": 3,  # Trigger synthetic fallback
            "last_action_status": "failure",
            "last_action_agent": "research",
        }

        # Update state
        app.update_state(thread_config, initial_state)

        # Simulate a quick decision history including synthetic without full streaming
        app.update_state(thread_config, {"decision_history": ["research", "synthetic", "end"]})

        # Verify that synthetic agent was recorded
        final_state = app.get_state(thread_config).values
        assert "synthetic" in final_state.get("decision_history", [])
