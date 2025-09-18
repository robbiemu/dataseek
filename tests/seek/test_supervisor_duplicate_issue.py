"""
Test file to reproduce the supervisor duplicate agent calls issue.

This test demonstrates the problem where the supervisor calls the same agent
consecutively (e.g., ['research', 'fitness', 'fitness']) without any failure
or meaningful change in state.
"""
import pytest
from unittest.mock import Mock, MagicMock
from seek.nodes import supervisor_node
from seek.state import DataSeekState
from langchain_core.messages import HumanMessage, AIMessage


def test_supervisor_prevents_duplicate_consecutive_agent_calls():
    """
    Test that demonstrates the current bug: supervisor allows consecutive calls 
    to the same agent even when there's no failure or meaningful state change.
    
    Expected behavior: ['research', 'fitness', 'research'] or similar fallback
    Current buggy behavior: ['research', 'fitness', 'fitness']
    """
    
    # Mock state that simulates the scenario from the log
    state = DataSeekState(
        messages=[
            HumanMessage(content="Please help me research and collect high-quality examples..."),
            AIMessage(content="Research completed after 3 iterations. Found 3 research findings..."),
            AIMessage(content="**Inspection Report**\n# | Source (query) | Assessment...")  # Fitness report
        ],
        research_findings=[
            {"source": "web_search", "content": "news article lists...", "query": "test1"},
            {"source": "web_search", "content": "Reuters article...", "query": "test2"},
            {"source": "web_search", "content": "Reuters NIST 800-171...", "query": "test3"}
        ],
        decision_history=["research", "fitness"],  # Already called research, then fitness
        tool_execution_failures=0,
        research_attempts=1,
        consecutive_failures=0,
        last_action_status="success",  # Previous fitness call was successful
        last_action_agent="supervisor",
        task_queue=[
            {"type": "research", "characteristic": "Verifiability", "topic": "news reports"},
            {"type": "research", "characteristic": "Verifiability", "topic": "scientific abstracts"},
        ],
        synthetic_samples_generated=0,
        research_samples_generated=0,
        samples_generated=0,
        total_samples_target=1200,
        synthetic_budget=0.2
    )
    
    # Mock the LLM to return fitness again (simulating the bug scenario)
    import unittest.mock
    
    mock_llm = Mock()
    mock_structured_output = Mock()
    mock_decision = Mock()
    mock_decision.next_agent = "fitness"  # LLM decides on fitness again
    
    mock_structured_output.invoke.return_value = mock_decision
    mock_llm.with_structured_output.return_value = mock_structured_output
    
    # Mock the create_llm function to return our mock
    with unittest.mock.patch("seek.nodes.create_llm") as mock_create_llm_func, \
         unittest.mock.patch("seek.nodes.load_claimify_config") as mock_config_func:
        
        mock_create_llm_func.return_value = mock_llm
        mock_config_func.return_value = Mock()
        
        # Call supervisor_node
        result = supervisor_node(state)
        
        # This test currently FAILS because the supervisor allows consecutive fitness calls
        # Once we fix the issue, this assertion should pass
        
        # Check that we don't get consecutive duplicate agent calls
        decision_history = result["decision_history"]
        
        # The bug: decision_history becomes ['research', 'fitness', 'fitness']
        # The fix should make it ['research', 'fitness', 'research'] or another valid path
        
        # For now, we expect this test to fail, demonstrating the bug
        last_two_decisions = decision_history[-2:]
        
        # This assertion will fail with current code, proving the bug exists
        assert not (len(last_two_decisions) == 2 and last_two_decisions[0] == last_two_decisions[1]), \
            f"Supervisor allowed consecutive duplicate calls: {decision_history}. " \
            f"Last two decisions: {last_two_decisions}"
        
        # Additional check: next_agent should not be the same as the last successful agent
        # when there's no failure or meaningful state change
        if result["last_action_status"] == "success" and result["consecutive_failures"] == 0:
            last_decision = decision_history[-2] if len(decision_history) >= 2 else None
            current_decision = result["next_agent"]
            
            assert current_decision != last_decision, \
                f"Supervisor chose same agent '{current_decision}' consecutively without failure. " \
                f"Decision history: {decision_history}"


def test_supervisor_allows_retry_only_with_valid_reason():
    """
    Test that supervisor should allow retries only when there's a valid reason:
    - Tool failure
    - Input state has meaningfully changed  
    - Configured retry policy allows it
    """
    
    # This test will be implemented after we add the retry logic
    # For now, it's a placeholder to show the expected behavior
    pass


if __name__ == "__main__":
    # Run this test to see the current failure
    test_supervisor_prevents_duplicate_consecutive_agent_calls()
