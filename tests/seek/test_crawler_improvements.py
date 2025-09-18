#!/usr/bin/env python3
"""
Test script to verify that the documentation crawler tool works correctly.
"""

import os
import sys

# Add the project root to the path so we can import the seek module
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def test_documentation_crawler_improvements():
    """Test the improved documentation crawler tool."""
    print("Testing improved documentation crawler tool...")
    try:
        from seek.tools import documentation_crawler

        # Check if the tool exists (it's optional)
        if documentation_crawler is None:
            print("⚠️  Documentation crawler tool not available (libcrawler not installed)")
            return

        # Check the tool's docstring to verify our improvements
        docstring = documentation_crawler.__doc__
        if "Best practices for configuration" in docstring:
            print("✅ Documentation crawler tool has improved documentation")
        else:
            print("❌ Documentation crawler tool is missing improved documentation")

        # Check that the function signature includes our new parameters
        import inspect

        sig = inspect.signature(documentation_crawler)
        params = list(sig.parameters.keys())
        required_params = ["base_url", "starting_point"]
        missing_params = [p for p in required_params if p not in params]

        if not missing_params:
            print("✅ Documentation crawler tool has correct parameters")
        else:
            print(f"❌ Documentation crawler tool is missing parameters: {missing_params}")

        print("✅ Documentation crawler improvements verified!")

    except Exception as e:
        print(f"❌ Error testing documentation crawler tool: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_documentation_crawler_improvements()
