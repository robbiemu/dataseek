#!/usr/bin/env python3
"""
Test script to demonstrate the difference between search and content tools for Arxiv and Wikipedia.
"""

import os
import sys

# Add the project root to the path so we can import the seek module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_arxiv_search_vs_content():
    """Test the difference between Arxiv search and content tools."""
    print("Testing Arxiv search vs content tools...")
    try:
        from seek.components.tool_manager.tools import arxiv_get_content, arxiv_search

        # Test search tool
        print("\n--- Arxiv Search Tool ---")
        search_result = arxiv_search("machine learning")
        print(f"Search status: {search_result.get('status')}")
        if search_result.get("status") == "ok":
            print(f"Search results type: {type(search_result.get('results'))}")
            print(f"Search results length: {len(str(search_result.get('results', '')))} characters")
            # Show first 500 characters of search results
            print(f"Search results preview: {str(search_result.get('results', ''))[:500]}...")

        # Test content tool
        print("\n--- Arxiv Content Tool ---")
        content_result = arxiv_get_content("machine learning")
        print(f"Content status: {content_result.get('status')}")
        if content_result.get("status") == "ok":
            results = content_result.get("results", [])
            print(f"Number of documents retrieved: {len(results)}")
            if results:
                first_doc = results[0]
                print(f"First document title: {first_doc.get('title', 'N/A')}")
                print(f"First document authors: {first_doc.get('authors', 'N/A')}")
                print(
                    f"First document content length: {len(first_doc.get('content', ''))} characters"
                )
                # Show first 500 characters of content
                print(
                    f"First document content preview: {str(first_doc.get('content', ''))[:500]}..."
                )
        elif content_result.get("status") == "error":
            print(f"Content tool error: {content_result.get('error')}")

    except Exception as e:
        print(f"❌ Error testing Arxiv tools: {e}")
        import traceback

        traceback.print_exc()
    print()


def test_wikipedia_search_vs_content():
    """Test the difference between Wikipedia search and content tools."""
    print("Testing Wikipedia search vs content tools...")
    try:
        from seek.components.tool_manager.tools import wikipedia_get_content, wikipedia_search

        # Test search tool
        print("\n--- Wikipedia Search Tool ---")
        search_result = wikipedia_search("artificial intelligence")
        print(f"Search status: {search_result.get('status')}")
        if search_result.get("status") == "ok":
            print(f"Search results type: {type(search_result.get('results'))}")
            print(f"Search results length: {len(str(search_result.get('results', '')))} characters")
            # Show first 500 characters of search results
            print(f"Search results preview: {str(search_result.get('results', ''))[:500]}...")

        # Test content tool
        print("\n--- Wikipedia Content Tool ---")
        content_result = wikipedia_get_content("artificial intelligence")
        print(f"Content status: {content_result.get('status')}")
        if content_result.get("status") == "ok":
            results = content_result.get("results", [])
            print(f"Number of documents retrieved: {len(results)}")
            if results:
                first_doc = results[0]
                print(f"First document title: {first_doc.get('title', 'N/A')}")
                print(
                    f"First document content length: {len(first_doc.get('content', ''))} characters"
                )
                # Show first 500 characters of content
                print(
                    f"First document content preview: {str(first_doc.get('content', ''))[:500]}..."
                )

    except Exception as e:
        print(f"❌ Error testing Wikipedia tools: {e}")
        import traceback

        traceback.print_exc()
    print()


if __name__ == "__main__":
    test_arxiv_search_vs_content()
    test_wikipedia_search_vs_content()
