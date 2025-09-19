#!/usr/bin/env python3
"""
Test script to verify that the new Arxiv and Wikipedia content tools work correctly.
"""

import os
import sys

# Add the project root to the path so we can import the seek module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_arxiv_get_content():
    """Test the Arxiv get content tool."""
    print("Testing Arxiv get content tool...")
    try:
        from seek.components.tool_manager.tools import arxiv_get_content

        result = arxiv_get_content("machine learning")
        print(f"Arxiv get content result: {result}")
        if result.get("status") == "ok":
            print("✅ Arxiv get content tool works correctly!")
        else:
            print(f"❌ Arxiv get content tool failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Error testing Arxiv get content tool: {e}")
        import traceback

        traceback.print_exc()
    print()


def test_wikipedia_get_content():
    """Test the Wikipedia get content tool."""
    print("Testing Wikipedia get content tool...")
    try:
        from seek.components.tool_manager.tools import wikipedia_get_content

        result = wikipedia_get_content("artificial intelligence")
        print(f"Wikipedia get content result: {result}")
        if result.get("status") == "ok":
            print("✅ Wikipedia get content tool works correctly!")
        else:
            print(f"❌ Wikipedia get content tool failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Error testing Wikipedia get content tool: {e}")
        import traceback

        traceback.print_exc()
    print()


if __name__ == "__main__":
    test_arxiv_get_content()
    test_wikipedia_get_content()
