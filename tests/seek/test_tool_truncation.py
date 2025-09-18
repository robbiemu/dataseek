#!/usr/bin/env python3
"""
Test script to verify that the tool response truncation works correctly.
"""

import sys
import os
import unittest
from unittest.mock import patch

# Add the project root to the path so we can import the seek module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from seek.tools import get_tools_for_role, _truncate_response_for_role


class TestToolTruncation(unittest.TestCase):
    
    def test_truncate_response_for_role_no_truncation(self):
        """Test that responses within limits are not truncated."""
        response = {
            "results": "This is a short response that should not be modified.",
            "status": "ok"
        }
        
        # Test with a large max_tokens limit
        with patch('seek.tools.load_claimify_config') as mock_config:
            # Mock config to return a large max_tokens
            mock_config.return_value.seek_agent.mission_plan.get_node_config.return_value.max_tokens = 1000
            
            truncated_response = _truncate_response_for_role(response, "research")
            
            # Should be unchanged
            self.assertEqual(truncated_response["results"], "This is a short response that should not be modified.")
            # Should not contain truncation message
            self.assertNotIn("Response truncated", truncated_response["results"])
    
    def test_truncate_response_for_role_with_truncation(self):
        """Test that responses exceeding limits are truncated."""
        # Create a long response
        long_response = "A" * 1000  # 1000 characters
        
        response = {
            "results": long_response,
            "status": "ok"
        }
        
        # Test with a small max_tokens limit
        with patch('seek.tools.load_claimify_config') as mock_config:
            # Mock config to return a small max_tokens
            mock_config.return_value.seek_agent.mission_plan.get_node_config.return_value.max_tokens = 10  # 40 characters
            
            truncated_response = _truncate_response_for_role(response, "research")
            
            # Should be truncated
            self.assertIn("Response truncated", truncated_response["results"])
            self.assertLess(len(truncated_response["results"]), len(long_response))
    
    def test_get_tools_for_role_research(self):
        """Test that research role tools are properly wrapped."""
        tools = get_tools_for_role("research")
        
        # Should have at least the basic tools
        self.assertGreater(len(tools), 0)
        
        # Check that tools have names
        for tool in tools:
            self.assertTrue(hasattr(tool, 'name') or hasattr(tool, '__name__'))


if __name__ == "__main__":
    unittest.main()