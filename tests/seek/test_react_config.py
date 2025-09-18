"""
Test for configurable ReAct max_iterations functionality.
"""

import pytest
from unittest.mock import Mock, patch
from seek.models import (
    SeekAgentConfig,
    SeekAgentNodesConfig,
    SeekAgentResearchNodeConfig,
    SeekAgentMissionPlanConfig,
    SeekAgentWriterConfig,
)
from seek.nodes import research_node
from seek.state import DataSeekState
from langchain_core.messages import HumanMessage, AIMessage


def test_research_node_max_iterations_default():
    """Test that the default max_iterations for the research node is 7."""
    config = SeekAgentNodesConfig()
    assert config.research.max_iterations == 7


def test_research_node_max_iterations_custom():
    """Test that custom max_iterations for the research node is respected."""
    config = SeekAgentNodesConfig(
        research=SeekAgentResearchNodeConfig(max_iterations=5)
    )
    assert config.research.max_iterations == 5


def test_research_node_max_iterations_validation():
    """Test that research node max_iterations validation works."""
    # Valid range
    SeekAgentResearchNodeConfig(max_iterations=1)
    SeekAgentResearchNodeConfig(max_iterations=50)

    # Invalid values should raise validation error
    with pytest.raises(Exception):
        SeekAgentResearchNodeConfig(max_iterations=0)
    with pytest.raises(Exception):
        SeekAgentResearchNodeConfig(max_iterations=51)


@patch("seek.nodes.get_active_seek_config")
@patch("seek.nodes.ChatLiteLLM")
@patch("seek.nodes.get_tools_for_role")
def test_research_node_uses_config_max_iterations(
    mock_get_tools, mock_chat_llm, mock_get_active_seek_config
):
    """Test that research_node uses configured max_iterations."""
    # --- Arrange ---
    # Set up custom config with max_iterations = 5
    mock_research_config = SeekAgentResearchNodeConfig(max_iterations=5)
    mock_seek_config = SeekAgentConfig(
        mission_plan=SeekAgentMissionPlanConfig(goal="Test Goal", nodes=[], tools={}),
        writer=SeekAgentWriterConfig(),
        nodes=SeekAgentNodesConfig(research=mock_research_config),
    )
    mock_get_active_seek_config.return_value = mock_seek_config

    # Mock LLM and tools
    mock_llm_instance = Mock()
    mock_chat_llm.return_value = mock_llm_instance
    mock_get_tools.return_value = []

    # Mock LLM to return a final report on the last iteration
    final_report = AIMessage(content="# Data Prospecting Report\nSuccess")
    # Simulate the loop by having invoke return non-reports until the last call
    mock_llm_instance.invoke.side_effect = [AIMessage(content="Thinking...")] * 4 + [
        final_report
    ]

    # Create state
    state = DataSeekState(messages=[HumanMessage(content="Find stuff")])

    # --- Act ---
    research_node(state)

    # --- Assert ---
    # The research_node's internal loop should run `max_iterations` times.
    # So, the LLM should be invoked 5 times.
    assert mock_llm_instance.invoke.call_count == 5
