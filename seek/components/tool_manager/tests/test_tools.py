from unittest.mock import MagicMock, patch

import pytest

from seek.components.tool_manager.search_tools import create_web_search_tool
from seek.components.tool_manager.tools import get_tools_for_role
from seek.components.tool_manager.utils import _safe_request_get
from seek.components.tool_manager.web_tools import url_to_markdown

# Don't create the web_search tool at module level since it causes issues with mocking


class TestTools:
    """Test suite for the Data Seek Agent tools."""

    def test_get_tools_for_role(self):
        """Test that tools are correctly assigned to roles."""
        # Test research role
        research_tools = get_tools_for_role("research")
        tool_names = [tool.name for tool in research_tools]
        assert "web_search" in tool_names
        assert "url_to_markdown" in tool_names
        # Verify the tools are in the role bindings
        assert "arxiv_search" in tool_names
        assert "arxiv_get_content" in tool_names
        assert "wikipedia_search" in tool_names
        assert "wikipedia_get_content" in tool_names

        # Test synthetic role
        # we have no tools to test

        # Test archive role
        archive_tools = get_tools_for_role("archive")
        tool_names = [tool.name for tool in archive_tools]
        assert "write_file" in tool_names

    @patch("seek.components.tool_manager.search_tools.get_active_seek_config")
    @patch("seek.components.tool_manager.search_tools.SearchProviderProxy.run")
    def test_web_search_success(self, mock_proxy_run, mock_get_cfg):
        """Test successful web search."""
        # Force a provider that doesn't require extra packages
        mock_get_cfg.return_value = {
            "web_search": {"provider": "wikipedia/search"},
            "use_robots": True,
        }
        # Create the web search tool inside the test
        web_search = create_web_search_tool()

        # Mock the async result from the proxy
        async def _fake_run(_query: str, **kwargs):  # type: ignore[no-redef]
            return "Test search results"

        mock_proxy_run.side_effect = _fake_run

        # Call the tool with .invoke to avoid deprecation
        result = web_search.invoke({"query": "test query"})

        # Verify the result
        assert result["status"] == "ok"
        assert result["query"] == "test query"
        # Results are normalized to a list
        assert result["results"] == ["Test search results"]
        # The provider should reflect our mocked provider
        assert result["provider"] == "wikipedia/search"

    @patch("seek.components.tool_manager.search_tools.get_active_seek_config")
    @patch("seek.components.tool_manager.search_tools.SearchProviderProxy.run")
    def test_web_search_failure(self, mock_proxy_run, mock_get_cfg):
        """Test failed web search."""
        mock_get_cfg.return_value = {
            "web_search": {"provider": "wikipedia/search"},
            "use_robots": True,
        }
        # Create the web search tool inside the test
        web_search = create_web_search_tool()

        # Mock the async result to raise an exception
        async def _fake_run_err(_query: str, **kwargs):  # type: ignore[no-redef]
            raise Exception("Search failed")

        mock_proxy_run.side_effect = _fake_run_err

        # Call the tool with .invoke to avoid deprecation
        result = web_search.invoke({"query": "test query"})

        # Verify the result
        assert result["status"] == "error"
        assert result["query"] == "test query"
        assert result["results"] is None
        assert "Search failed" in result["error"]

    @patch("seek.components.tool_manager.web_tools._safe_request_get")
    def test_url_to_markdown_success(self, mock_safe_request):
        """Test successful URL to markdown conversion."""
        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.text = (
            "<html><head><title>Test Title</title></head><body><p>Test content</p></body></html>"
        )
        mock_safe_request.return_value = mock_response

        # Call the tool with explicit .invoke to avoid __call__ deprecation
        result = url_to_markdown.invoke({"url": "http://example.com"})

        # Verify the result
        assert result["status"] == "ok"
        assert result["url"] == "http://example.com"
        assert "Test Title" in result["markdown"]
        assert "Test content" in result["markdown"]
        assert result["title"] == "Test Title"

    @patch("seek.components.tool_manager.web_tools._safe_request_get")
    def test_url_to_markdown_failure(self, mock_safe_request):
        """Test failed URL to markdown conversion."""
        # Mock the HTTP request to raise an exception
        mock_safe_request.side_effect = Exception("Network error")

        # Call the tool with explicit .invoke to avoid __call__ deprecation
        result = url_to_markdown.invoke({"url": "http://example.com"})

        # Verify the result
        assert result["status"] == "error"
        assert result["url"] == "http://example.com"
        assert result["markdown"] == ""
        assert result["title"] == ""
        assert "Network error" in result["error"]

    @patch("seek.components.tool_manager.clients.SYNC_HTTP_CLIENT.get")
    def test_safe_request_get_success(self, mock_sync_get):
        """Test successful safe request."""
        # Mock the sync http response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_sync_get.return_value = mock_response

        # Call the function
        result = _safe_request_get("http://example.com")

        # Verify the result
        assert result == mock_response

    @patch("seek.components.tool_manager.clients.SYNC_HTTP_CLIENT.get")
    def test_safe_request_get_failure(self, mock_sync_get):
        """Test failed safe request."""
        # Mock the sync request to raise a specific exception
        mock_sync_get.side_effect = RuntimeError("Network error")

        # Call the function and expect it to raise
        with pytest.raises(RuntimeError, match="Network error"):
            _safe_request_get("http://example.com", max_retries=0)
