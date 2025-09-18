import pytest
import uuid
from unittest.mock import patch, MagicMock
from seek.graph import build_graph
from seek.state import DataSeekState
from langchain_core.messages import HumanMessage

class TestIntegration:
    """Integration tests for the Data Seek Agent workflow."""
    
    @patch('seek.graph.SqliteSaver')
    @patch('seek.nodes.load_claimify_config')
    @patch('seek.nodes.ChatLiteLLM')
    def test_full_workflow_success(self, mock_chat_llm, mock_load_config, mock_sqlite_saver):
        """Test a complete workflow from start to finish."""
        # Mock config and LLM
        mock_config = MagicMock()
        mock_config.temperature = 0.1
        mock_config.max_tokens = 2000
        mock_config.seek_agent = None
        mock_load_config.return_value = mock_config
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.model = "ollama/gpt-oss:20b"
        mock_chat_llm.return_value = mock_llm_instance
        
        # Mock checkpointer
        mock_checkpointer = MagicMock()
        mock_checkpoint_tuple = MagicMock()
        mock_checkpoint_tuple.checkpoint = {
            "v": 4,
            "id": str(uuid.uuid4()),
            "ts": "2025-09-06T19:40:00Z",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
        }
        mock_checkpointer.get_tuple.return_value = mock_checkpoint_tuple
        mock_sqlite_saver.from_conn_string.return_value = mock_checkpointer
        
        # Mock supervisor decisions
        mock_structured_llm = MagicMock()
        mock_decision1 = MagicMock()
        mock_decision1.next_agent = "research"
        mock_decision2 = MagicMock()
        mock_decision2.next_agent = "end"
        mock_structured_llm.invoke.side_effect = [mock_decision1, mock_decision2]
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm
        
        # Mock research response
        mock_research_response = MagicMock()
        mock_research_response.content = "Research findings about the topic..."
        mock_research_response.tool_calls = []
        mock_llm_instance.invoke.return_value = mock_research_response
        
        # Build the graph
        app = build_graph(mock_checkpointer)
        
        # Create initial state
        thread_config = {"configurable": {"thread_id": "test-thread"}}
        initial_state = {
            "messages": [HumanMessage(content="Find information about AI agents")],
            "research_findings": [],
            "pedigree_path": "test_pedigree.md",
            "decision_history": [],
            "tool_execution_failures": 0,
            "research_attempts": 0,
            "samples_generated": 0,
            "total_samples_target": 1200,
            "current_mission": "test_mission",
            "consecutive_failures": 0,
            "last_action_status": "success",
            "last_action_agent": ""
        }
        
        # Update state
        app.update_state(thread_config, initial_state)
        
        # Run the workflow
        inputs = {"messages": [HumanMessage(content="Find information about AI agents")]}
        events = list(app.stream(inputs, config=thread_config))
        
        # Verify we got events
        assert len(events) > 0
        
        # Verify the final state
        final_state = app.get_state(thread_config).values
        assert "decision_history" in final_state
        assert len(final_state["decision_history"]) > 0
    
    @patch('seek.graph.SqliteSaver')
    @patch('seek.nodes.load_claimify_config')
    @patch('seek.nodes.ChatLiteLLM')
    def test_synthetic_fallback_workflow(self, mock_chat_llm, mock_load_config, mock_sqlite_saver):
        """Test workflow with synthetic fallback."""
        # Mock config and LLM
        mock_config = MagicMock()
        mock_config.temperature = 0.1
        mock_config.max_tokens = 2000
        mock_config.seek_agent = None
        mock_load_config.return_value = mock_config
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.model = "ollama/gpt-oss:20b"
        mock_chat_llm.return_value = mock_llm_instance
        
        # Mock checkpointer
        mock_checkpointer = MagicMock()
        mock_checkpoint_tuple = MagicMock()
        mock_checkpoint_tuple.checkpoint = {
            "v": 4,
            "id": str(uuid.uuid4()),
            "ts": "2025-09-06T19:40:00Z",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
        }
        mock_checkpointer.get_tuple.return_value = mock_checkpoint_tuple
        mock_sqlite_saver.from_conn_string.return_value = mock_checkpointer
        
        # Mock supervisor decisions - go to synthetic after research fails
        mock_structured_llm = MagicMock()
        mock_decision1 = MagicMock()
        mock_decision1.next_agent = "research"
        mock_decision2 = MagicMock()
        mock_decision2.next_agent = "synthetic"
        mock_decision3 = MagicMock()
        mock_decision3.next_agent = "end"
        mock_structured_llm.invoke.side_effect = [mock_decision1, mock_decision2, mock_decision3]
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm
        
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
        
        # Build the graph
        app = build_graph(mock_checkpointer)
        
        # Create initial state with failures to trigger synthetic fallback
        thread_config = {"configurable": {"thread_id": "test-thread-fail"}}
        initial_state = {
            "messages": [HumanMessage(content="Find information about quantum computing")],
            "research_findings": [],
            "pedigree_path": "test_pedigree.md",
            "decision_history": [],
            "tool_execution_failures": 0,
            "research_attempts": 0,
            "samples_generated": 0,
            "total_samples_target": 1200,
            "current_mission": "test_mission",
            "consecutive_failures": 3,  # Trigger synthetic fallback
            "last_action_status": "failure",
            "last_action_agent": "research"
        }
        
        # Update state
        app.update_state(thread_config, initial_state)
        
        # Run the workflow
        inputs = {"messages": [HumanMessage(content="Find information about quantum computing")]}
        events = list(app.stream(inputs, config=thread_config))
        
        # Verify we got events
        assert len(events) > 0
        
        # Verify that synthetic agent was called
        final_state = app.get_state(thread_config).values
        assert "synthetic" in final_state.get("decision_history", [])