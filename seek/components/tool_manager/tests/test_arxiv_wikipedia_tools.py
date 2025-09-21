#!/usr/bin/env python3
"""
Test script to verify that the Arxiv and Wikipedia search tools work correctly.
"""

import os
import sys

# Add the project root to the path so we can import the seek module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_arxiv_search():
    """Test the Arxiv search tool."""
    print("Testing Arxiv search tool...")
    try:
        from seek.components.tool_manager.tools import arxiv_search

        # Use .invoke to avoid deprecated __call__
        result = arxiv_search.invoke({"query": "machine learning"})
        print(f"Arxiv search result: {result}")
        if result.get("status") == "ok":
            print("✅ Arxiv search tool works correctly!")
        else:
            print(f"❌ Arxiv search tool failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Error testing Arxiv search tool: {e}")
        import traceback

        traceback.print_exc()
    print()


def test_wikipedia_search():
    """Test the Wikipedia search tool."""
    print("Testing Wikipedia search tool...")
    try:
        from seek.components.tool_manager.tools import wikipedia_search

        # Use .invoke to avoid deprecated __call__
        result = wikipedia_search.invoke({"query": "artificial intelligence"})
        print(f"Wikipedia search result: {result}")
        if result.get("status") == "ok":
            print("✅ Wikipedia search tool works correctly!")
        else:
            print(f"❌ Wikipedia search tool failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Error testing Wikipedia search tool: {e}")
        import traceback

        traceback.print_exc()
    print()


if __name__ == "__main__":
    test_arxiv_search()
    test_wikipedia_search()
