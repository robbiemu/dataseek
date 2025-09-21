from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pydantic
from langchain_core.tools import tool

from seek.common.config import get_active_seek_config
from seek.components.search_graph.litesearch import SearchProviderProxy

from .clients import HTTP_CLIENT, RATE_MANAGER
from .utils import _find_url_field, _run_async_safely


class WebSearchInput(pydantic.BaseModel):
    """Input schema for the web_search tool."""

    query: str = pydantic.Field(description="The search query to execute.")


def create_web_search_tool() -> Any:
    """
    Factory function that creates a web_search tool with the configured provider hard-coded.
    This ensures that the search provider is determined by the configuration, not by the agent.

    Args:
        use_robots: Whether to respect robots.txt rules.
    """
    # Use the active seek configuration established at startup
    config = get_active_seek_config()
    ws_cfg = (config.get("web_search") or {}) if isinstance(config.get("web_search"), dict) else {}
    # Backward compat: fall back to root search_provider if present
    search_provider = ws_cfg.get(
        "provider",
        config.get("search_provider", "duckduckgo/search"),
    )
    ws_rps = ws_cfg.get("requests_per_second") or ws_cfg.get("rps")
    # Canonical config key: max_results (backward-compat: accept older names if present)
    ws_max_results = ws_cfg.get("max_results") if isinstance(ws_cfg, dict) else None
    if ws_max_results is None and isinstance(ws_cfg, dict):
        # Backward compatibility for older configs
        for legacy_key in ("num_results", "count", "limit"):
            if legacy_key in ws_cfg:
                ws_max_results = ws_cfg.get(legacy_key)
                break
    # Respect robots only if the overall config allows it; CLI flag overrides per-tool setting
    # When overall use_robots is False (e.g., --no-robots), disable robots filtering entirely
    respect_robots = bool(config.get("use_robots", True)) and bool(
        ws_cfg.get("respect_robots", True)
    )

    @tool("web_search", args_schema=WebSearchInput)
    def web_search(query: str) -> dict[str, Any]:
        """
        Performs a web search using the configured provider and returns the results.
        This tool is rate-limited to avoid API abuse.

        Note: This function is synchronous to work with LangGraph's ToolNode.
        It uses asyncio.run() internally to handle the async operations.
        """
        proxy = SearchProviderProxy(
            provider=search_provider,
            rate_limit_manager=RATE_MANAGER,
            http_client=HTTP_CLIENT,
        )
        try:
            # Run the async operation safely with improved event loop management
            # Pass configured RPS to the proxy (if provided)
            run_kwargs = {}
            if ws_rps is not None:
                run_kwargs["requests_per_second"] = ws_rps
            # Pass canonical max_results; proxy maps to provider-specific params
            if ws_max_results is not None:
                run_kwargs["max_results"] = ws_max_results
            results = _run_async_safely(proxy.run(query, **run_kwargs))

            # Normalize results to a list form for downstream processing
            # Some providers (e.g., DuckDuckGo) return a single string
            if isinstance(results, str):
                normalized_results = [results]
            elif isinstance(results, list):
                normalized_results = results
            else:
                # Keep as-is but wrap in a list to maintain consistency
                normalized_results = [results]

            # Post-processing: check robots.txt if enabled
            # Only attempt robots checks for structured results with URLs
            if respect_robots and normalized_results and isinstance(normalized_results[0], dict):
                filtered_results = []
                key = _find_url_field(normalized_results)
                for result in normalized_results:
                    url = result.get(key) if isinstance(result, dict) else None
                    if not url:
                        filtered_results.append(result)
                        continue

                    try:
                        parsed_url = urlparse(url)
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        robots_url = urljoin(base_url, "/robots.txt")

                        rp = RobotFileParser()
                        rp.set_url(robots_url)
                        rp.read()

                        if rp.can_fetch("DataSeek/1.0", url):
                            filtered_results.append(result)
                        else:
                            print(f"      ⚠️ Filtering out {url} due to robots.txt")
                    except Exception as e:
                        # If we can't check robots.txt, assume it's okay to proceed
                        print(f"      ⚠️ Could not check robots.txt for {url}: {e}, keeping result")
                        filtered_results.append(result)
                normalized_results = filtered_results

            return {
                "query": query,
                "results": normalized_results,
                "provider": search_provider,
                "status": "ok",
            }
        except Exception as e:
            return {
                "query": query,
                "results": None,
                "provider": search_provider,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)}",
            }

    return web_search


class ArxivSearchInput(pydantic.BaseModel):
    """Input schema for the arxiv_search tool."""

    query: str = pydantic.Field(description="The search query to execute on Arxiv.")


@tool("arxiv_search", args_schema=ArxivSearchInput)
def arxiv_search(query: str) -> dict[str, Any]:
    """
    Performs a search on Arxiv and returns the results.
    This tool is rate-limited to avoid API abuse.

    Note: This function is synchronous to work with LangGraph's ToolNode.
    It uses asyncio.run() internally to handle the async operations.
    """
    proxy = SearchProviderProxy(
        provider="arxiv/search",
        rate_limit_manager=RATE_MANAGER,
        http_client=HTTP_CLIENT,
    )
    try:
        # Run the async operation safely with improved event loop management
        results = _run_async_safely(proxy.run(query))

        return {
            "query": query,
            "results": results,
            "provider": "arxiv/search",
            "status": "ok",
        }
    except Exception as e:
        return {
            "query": query,
            "results": None,
            "provider": "arxiv/search",
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }


class ArxivGetContentInput(pydantic.BaseModel):
    """Input schema for the arxiv_get_content tool."""

    query: str = pydantic.Field(description="The search query to get content for from Arxiv.")


@tool("arxiv_get_content", args_schema=ArxivGetContentInput)
def arxiv_get_content(query: str) -> dict[str, Any]:
    """
    Retrieves detailed content from Arxiv based on a search query.
    This tool is rate-limited to avoid API abuse.

    Note: This function is synchronous to work with LangGraph's ToolNode.
    It uses asyncio.run() internally to handle the async operations.
    """
    try:
        # Using LangChain's ArxivAPIWrapper for detailed content retrieval
        from langchain_community.utilities import ArxivAPIWrapper

        arxiv_wrapper = ArxivAPIWrapper(
            top_k_results=5,
            ARXIV_MAX_QUERY_LENGTH=300,
            load_max_docs=5,
            load_all_available_meta=False,
            arxiv_search=None,
            arxiv_exceptions=None,
        )

        # Get detailed documents
        docs = arxiv_wrapper.load(query)

        # Format the results for consistency
        formatted_results = []
        for doc in docs:
            formatted_results.append(
                {
                    "title": doc.metadata.get("Title", "N/A"),
                    "authors": doc.metadata.get("Authors", "N/A"),
                    "published": doc.metadata.get("Published", "N/A"),
                    "summary": doc.metadata.get("Summary", "N/A"),
                    "content": doc.page_content if doc.page_content else "N/A",
                }
            )

        return {
            "query": query,
            "results": formatted_results,
            "provider": "arxiv/get_content",
            "status": "ok",
        }
    except Exception as e:
        return {
            "query": query,
            "results": None,
            "provider": "arxiv/get_content",
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }


class WikipediaSearchInput(pydantic.BaseModel):
    """Input schema for the wikipedia_search tool."""

    query: str = pydantic.Field(description="The search query to execute on Wikipedia.")


@tool("wikipedia_search", args_schema=WikipediaSearchInput)
def wikipedia_search(query: str) -> dict[str, Any]:
    """
    Performs a search on Wikipedia and returns the results.
    This tool is rate-limited to avoid API abuse.

    Note: This function is synchronous to work with LangGraph's ToolNode.
    It uses asyncio.run() internally to handle the async operations.
    """
    proxy = SearchProviderProxy(
        provider="wikipedia/search",
        rate_limit_manager=RATE_MANAGER,
        http_client=HTTP_CLIENT,
    )
    try:
        # Run the async operation safely with improved event loop management
        results = _run_async_safely(proxy.run(query))

        return {
            "query": query,
            "results": results,
            "provider": "wikipedia/search",
            "status": "ok",
        }
    except Exception as e:
        return {
            "query": query,
            "results": None,
            "provider": "wikipedia/search",
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }


class WikipediaGetContentInput(pydantic.BaseModel):
    """Input schema for the wikipedia_get_content tool."""

    query: str = pydantic.Field(description="The search query to get content for from Wikipedia.")


@tool("wikipedia_get_content", args_schema=WikipediaGetContentInput)
def wikipedia_get_content(query: str) -> dict[str, Any]:
    """
    Retrieves detailed content from Wikipedia based on a search query.
    This tool is rate-limited to avoid API abuse.

    Note: This function is synchronous to work with LangGraph's ToolNode.
    It uses asyncio.run() internally to handle the async operations.
    """
    try:
        # Using LangChain's WikipediaAPIWrapper for detailed content retrieval
        from langchain_community.utilities import WikipediaAPIWrapper

        wikipedia_wrapper = WikipediaAPIWrapper(
            top_k_results=3,
            lang="en",
            load_all_available_meta=False,
            wiki_client=None,
        )

        # Get detailed documents
        docs = wikipedia_wrapper.load(query)

        # Format the results for consistency
        formatted_results = []
        for doc in docs:
            formatted_results.append(
                {
                    "title": doc.metadata.get("title", "N/A"),
                    "summary": doc.metadata.get("summary", "N/A"),
                    "content": doc.page_content if doc.page_content else "N/A",
                }
            )

        return {
            "query": query,
            "results": formatted_results,
            "provider": "wikipedia/get_content",
            "status": "ok",
        }
    except Exception as e:
        return {
            "query": query,
            "results": None,
            "provider": "wikipedia/get_content",
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }
