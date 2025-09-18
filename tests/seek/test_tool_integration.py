#!/usr/bin/env python3
"""
Test script to verify that the new Arxiv and Wikipedia content tools are properly integrated with the role-based system.
"""

import sys
import os

# Add the project root to the path so we can import the seek module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_research_role_tools():
    """Test that the research role has access to the new tools."""
    print("Testing research role tools...")
    try:
        from seek.tools import get_tools_for_role
        
        # Get tools for the research role
        research_tools = get_tools_for_role("research")
        
        # Extract tool names
        tool_names = [tool.name for tool in research_tools]
        print(f"Research role tools: {tool_names}")
        
        # Check that our new tools are included
        expected_tools = ["arxiv_get_content", "wikipedia_get_content"]
        for tool_name in expected_tools:
            if tool_name in tool_names:
                print(f"✅ {tool_name} is available for research role")
            else:
                print(f"❌ {tool_name} is NOT available for research role")
                
    except Exception as e:
        print(f"❌ Error testing research role tools: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_tool_availability():
    """Test that all tools are available through get_available_tools."""
    print("Testing tool availability...")
    try:
        from seek.tools import get_available_tools
        
        # Get all available tools
        all_tools = get_available_tools()
        
        # Extract tool names
        tool_names = [tool.name for tool in all_tools]
        print(f"Available tools: {tool_names}")
        
        # Check that our new tools are included
        expected_tools = ["arxiv_get_content", "wikipedia_get_content"]
        for tool_name in expected_tools:
            if tool_name in tool_names:
                print(f"✅ {tool_name} is available")
            else:
                print(f"❌ {tool_name} is NOT available")
                
    except Exception as e:
        print(f"❌ Error testing tool availability: {e}")
        import traceback
        traceback.print_exc()
    print()


if __name__ == "__main__":
    test_research_role_tools()
    test_tool_availability()