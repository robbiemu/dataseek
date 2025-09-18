import pytest
from unittest.mock import patch, MagicMock
from seek.nodes import (
    supervisor_node,
    research_node,
    archive_node,
    fitness_node,
    synthetic_node
)
from seek.state import DataSeekState

class TestNodes:
    """Test suite for the Data Seek Agent nodes."""

    def create_test_state(self, **kwargs):
        """Create a default test state, allowing overrides."""
        state = {
            "messages": [],
            "progress": {},
            "current_task": None,
            "research_findings": [],
            "pedigree_path": "test_pedigree.md",
            "decision_history": [],
            "tool_execution_failures": 0,
            "research_attempts": 0,
            "samples_generated": 0,
            "total_samples_target": 10,
            "current_mission": "test_mission",
            "synthetic_samples_generated": 0,
            "research_samples_generated": 0,
            "consecutive_failures": 0,
            "last_action_status": "success",
            "last_action_agent": "",
            "synthetic_budget": 0.2,
        }
        state.update(kwargs)
        return state

    @patch("seek.nodes._get_next_task_from_progress")
    @patch("seek.nodes.load_claimify_config")
    @patch("seek.nodes.ChatLiteLLM")
    def test_supervisor_node_research_decision(
        self,
        mock_chat_llm,
        mock_load_config,
        mock_get_next_task,
    ):
        """Test supervisor node making a research decision when a task is available."""
        # Mock config and LLM
        mock_config = MagicMock()
        mock_config.seek_agent.mission_plan.get_node_config.return_value = None
        mock_load_config.return_value = mock_config
        mock_chat_llm.return_value = MagicMock()

        # Mock the progress tracker to return a task
        mock_get_next_task.return_value = {
            "characteristic": "Verifiability",
            "topic": "AI",
        }

        # Mock the supervisor's LLM to make a predictable decision
        mock_decision = MagicMock()
        mock_decision.next_agent = "research"
        mock_decision.new_task = None
        mock_chat_llm.return_value.invoke.return_value.content = '{"next_agent": "research"}'
        
        # Create an empty state, so the supervisor has to find a new task
        state = self.create_test_state(current_task=None, progress={"test_mission": {}})
        
        # Call the node
        result = supervisor_node(state)

        # Verify the result
        assert result["next_agent"] == "research"
        assert result["current_task"] is not None
        assert result["current_task"]["topic"] == "AI"

    @patch("seek.nodes._get_next_task_from_progress")
    @patch("seek.nodes.load_claimify_config")
    @patch("seek.nodes.ChatLiteLLM")
    def test_supervisor_node_end_decision(
        self,
        mock_chat_llm,
        mock_load_config,
        mock_get_next_task,
    ):
        """Test supervisor deciding to end when no tasks are left."""
        # Mock config and LLM
        mock_load_config.return_value = MagicMock()
        mock_chat_llm.return_value = MagicMock()

        # Mock the progress tracker to return no tasks
        mock_get_next_task.return_value = None
        
        # Create an empty state
        state = self.create_test_state(current_task=None, progress={"test_mission": {}})
        
        # Call the node
        result = supervisor_node(state)
        
        # Verify the result
        assert result["next_agent"] == "end"
    
    @patch('seek.nodes.load_claimify_config')
    @patch('seek.nodes.ChatLiteLLM')
    @patch('seek.nodes.get_tools_for_role')
    def test_research_node_success(self, mock_get_tools, mock_chat_llm, mock_load_config):
        """Test research node successful operation."""
        from langchain_core.messages import HumanMessage, AIMessage
        from seek.models import SeekAgentConfig, SeekAgentMissionPlanNodeConfig, SeekAgentNodesConfig, SeekAgentResearchNodeConfig

        # Mock config and LLM
        mock_config = MagicMock()
        mock_node_config = SeekAgentMissionPlanNodeConfig(name="research", model="test-model", temperature=0.1, max_tokens=100)
        mock_research_config = SeekAgentResearchNodeConfig(max_iterations=3)
        mock_config.seek_agent = SeekAgentConfig(
            mission_plan=MagicMock(),
            nodes=SeekAgentNodesConfig(research=mock_research_config)
        )
        mock_config.seek_agent.mission_plan.get_node_config.return_value = mock_node_config
        mock_load_config.return_value = mock_config
        
        mock_llm_instance = MagicMock()
        mock_chat_llm.return_value = mock_llm_instance
        mock_get_tools.return_value = []
        
        state = self.create_test_state(messages=[HumanMessage(content="Find info on X")])
        
        # Mock LLM to return a final report immediately
        final_report = AIMessage(content="# Data Prospecting Report\nSuccess")
        mock_llm_instance.invoke.return_value = final_report
        
        result = research_node(state)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "# Data Prospecting Report\nSuccess"
        assert "research_session_cache" in result

    @patch('seek.nodes.load_claimify_config')
    @patch('seek.nodes.ChatLiteLLM')
    def test_fitness_node_evaluation(self, mock_chat_llm, mock_load_config):
        """Test fitness node evaluation."""
        from langchain_core.messages import AIMessage
        # Mock config and LLM
        mock_load_config.return_value = MagicMock()
        mock_llm_instance = MagicMock()
        mock_chat_llm.return_value = mock_llm_instance
        
        # Mock LLM to return a valid JSON for a passing report
        mock_llm_instance.invoke.return_value = AIMessage(
            content='{"passed": true, "reason": "The document is well-structured."}'
        )
        
        state = self.create_test_state(research_findings=["Some great content."])
        
        result = fitness_node(state)
        
        assert "fitness_report" in result
        assert result["fitness_report"].passed is True
        assert "well-structured" in result["fitness_report"].reason

    @patch('seek.nodes.load_claimify_config')
    @patch('seek.nodes.ChatLiteLLM')
    def test_synthetic_node_generation(self, mock_chat_llm, mock_load_config):
        """Test synthetic node generation."""
        from langchain_core.messages import AIMessage
        # Mock config and LLM
        mock_load_config.return_value = MagicMock()
        mock_llm_instance = MagicMock()
        mock_chat_llm.return_value = mock_llm_instance
        
        # Mock LLM to return a report
        mock_response = AIMessage(content="# Data Prospecting Report\nThis is a synthetic document.")
        mock_llm_instance.invoke.return_value = mock_response
        
        state = self.create_test_state()
        
        result = synthetic_node(state)
        
        assert "research_findings" in result
        assert len(result["research_findings"]) == 1
        assert "synthetic document" in result["research_findings"][0]
        assert result["current_sample_provenance"] == "synthetic"

    @patch('seek.nodes.load_claimify_config')
    @patch('seek.nodes.create_llm')
    @patch('seek.nodes.write_file')
    @patch('seek.nodes.append_to_pedigree')
    @patch('time.strftime')
    def test_archive_node_filepath_generation(self, mock_strftime, mock_append_to_pedigree, mock_write_file, mock_create_llm, mock_load_config):
        """Test that the archive_node generates a deterministic filepath."""
        # Arrange
        mock_load_config.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_llm.invoke.return_value.content = "### 2025-09-04 — Sample Archived..."

        mock_strftime.return_value = "20250904123456"
        mock_write_file.invoke.return_value = {"status": "ok"}

        state = self.create_test_state(
            current_task={"characteristic": "Test Characteristic", "topic": "Test Topic"},
            research_findings=["# Data Prospecting Report\nThis is a test report."]
        )

        # Act
        result = archive_node(state)

        # Assert
        expected_filename = "test_characteristic_test_topic_20250904123456.md"
        expected_filepath = f"output/approved_books/{expected_filename}"
        
        mock_write_file.invoke.assert_called_once_with({
            "filepath": expected_filepath,
            "content": "# Data Prospecting Report\nThis is a test report."
        })
        mock_append_to_pedigree.assert_called_once()
        assert "Successfully archived document" in result["messages"][-1].content
