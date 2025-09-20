"Production-ready tool implementations for the Data Seek Agent."

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import pydantic
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool

from seek.common.config import get_active_seek_config
from seek.components.search_graph.litesearch import AsyncRateLimitManager, SearchProviderProxy

TOKEN_CHARACTER_RATIO = 3.5


# Avoid circular import with seek_utils by defining a minimal local helper
def _find_url_field(results: list[dict[str, Any]]) -> str | None:
    if not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None
    for key in ("url", "link"):
        if key in first:
            return key
    return None


# Optional import for deep crawling
try:
    from libcrawler.libcrawler import crawl_and_convert

    _HAVE_LIBCRAWLER = True
except ImportError:
    crawl_and_convert = None
    _HAVE_LIBCRAWLER = False

# -------------------------
# Globals for tool clients
# -------------------------

# Use a single async rate manager and http client for async search providers
RATE_MANAGER = AsyncRateLimitManager()
HTTP_CLIENT = httpx.AsyncClient()

# Dedicated synchronous client for page fetches (markdown, pre-scan, etc.)
SYNC_HTTP_CLIENT = httpx.Client(follow_redirects=True)


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Ensure we have a running event loop, create one if needed."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        # No event loop running or it's closed, create a new one
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        except Exception:
            # If we can't create a new loop, we'll handle this in the calling code
            raise


def _run_async_safely(coro: Any) -> Any:
    """Run async code safely, handling event loop issues."""
    try:
        # First try to get current loop
        _loop = asyncio.get_running_loop()
        # We're in an event loop, run in a thread with a new loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=60)
    except RuntimeError:
        # No event loop running, safe to use asyncio.run
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Create a new event loop and try again
                _ensure_event_loop()
                return asyncio.run(coro)
            raise


# -------------------------
# Pydantic schemas & tools
# -------------------------


class WebSearchInput(pydantic.BaseModel):
    """Input schema for the web_search tool."""

    query: str = pydantic.Field(description="The search query to execute.")


def create_web_search_tool(use_robots: bool = True) -> Any:
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
            if respect_robots and normalized_results:
                # Only attempt robots checks for structured results with URLs
                if isinstance(normalized_results[0], dict):
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
                            print(
                                f"      ⚠️ Could not check robots.txt for {url}: {e}, keeping result"
                            )
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


# -------------------------
# Utility helpers
# -------------------------


def _truncate_response_for_role(
    response: dict[str, Any], role: str, tool_name: str | None = None
) -> dict[str, Any]:
    """
    Truncate tool responses based on the max_tokens setting for the given role.

    Args:
        response: The tool response to potentially truncate
        role: The role of the agent calling the tool

    Returns:
        The response, possibly with truncated results
    """
    # If tool_name is provided, apply category-specific caps
    if tool_name:
        try:
            search_like = {
                "web_search",
                "arxiv_search",
                "wikipedia_search",
                "arxiv_get_content",
                "wikipedia_get_content",
            }
            fetch_like = {"url_to_markdown", "documentation_crawler"}

            summary_char_limit, result_char_limit = _get_research_limits_from_config()

            if tool_name in search_like:
                items = response.get("results")
                if isinstance(items, list):
                    new_items: list[str | dict[str, Any]] = []
                    for item in items:
                        if isinstance(item, str):
                            if len(item) > summary_char_limit:
                                new_items.append(item[:summary_char_limit] + "\n[Entry truncated]")
                            else:
                                new_items.append(item)
                        elif isinstance(item, dict):
                            truncated = dict(item)
                            for k in (
                                "summary",
                                "content",
                                "snippet",
                                "description",
                                "text",
                            ):
                                v = truncated.get(k)
                                if isinstance(v, str) and len(v) > summary_char_limit:
                                    truncated[k] = v[:summary_char_limit] + "\n[Entry truncated]"
                            new_items.append(truncated)
                        else:
                            new_items.append(item)
                    response["results"] = new_items

            if tool_name in fetch_like:
                if "markdown" in response and isinstance(response["markdown"], str):
                    if len(response["markdown"]) > result_char_limit:
                        response["markdown"] = (
                            response["markdown"][:result_char_limit]
                            + "\n\n[Markdown truncated due to size]"
                        )
                if "full_markdown" in response and isinstance(response["full_markdown"], str):
                    if len(response["full_markdown"]) > result_char_limit:
                        response["full_markdown"] = (
                            response["full_markdown"][:result_char_limit]
                            + "\n\n[Content truncated due to size]"
                        )
        except Exception:
            # Fail open; don't break tool chain on truncation errors
            return response
        return response

    # Backward-compatible behavior: token-estimated truncation when no tool context
    config = get_active_seek_config()
    max_tokens = None
    try:
        for node in (config.get("mission_plan", {}) or {}).get("nodes", []) or []:
            if isinstance(node, dict) and node.get("name") == role:
                max_tokens = node.get("max_tokens")
                break
    except Exception:
        max_tokens = None
    if max_tokens is None:
        max_tokens = ((config.get("nodes", {}) or {}).get(role, {}) or {}).get("max_tokens")
    if max_tokens is None:
        max_tokens = 2000

    text_fields = ["results", "markdown", "content", "text", "output"]
    for field in text_fields:
        if field in response and isinstance(response[field], str):
            max_chars = int(max_tokens * TOKEN_CHARACTER_RATIO)
            if len(response[field]) > max_chars:
                truncated = response[field][:max_chars]
                response[field] = (
                    truncated
                    + f"\n\n[Response truncated to {max_chars} characters due to token limits for role '{role}']"
                )

    return response


def _get_research_limits_from_config() -> tuple[int, int]:
    """Fetch (summary_char_limit, result_char_limit) for the research node.

    Returns:
        A tuple of (summary_char_limit, result_char_limit)
    """
    cfg = get_active_seek_config()
    summary_char_limit = None
    result_char_limit = None

    try:
        for node in (cfg.get("mission_plan", {}) or {}).get("nodes", []) or []:
            if isinstance(node, dict) and node.get("name") == "research":
                summary_char_limit = node.get("summary_char_limit")
                result_char_limit = node.get("result_char_limit")
                break
    except Exception:
        summary_char_limit = None
        result_char_limit = None

    if summary_char_limit is None:
        summary_char_limit = ((cfg.get("nodes", {}) or {}).get("research", {}) or {}).get(
            "summary_char_limit"
        )
    if result_char_limit is None:
        result_char_limit = ((cfg.get("nodes", {}) or {}).get("research", {}) or {}).get(
            "result_char_limit"
        )

    # Derive a reasonable default for summary_char_limit if not present
    if summary_char_limit is None:
        # Use mission plan research max_tokens, else 2000
        role_max_tokens = None
        try:
            for node in (cfg.get("mission_plan", {}) or {}).get("nodes", []) or []:
                if isinstance(node, dict) and node.get("name") == "research":
                    role_max_tokens = node.get("max_tokens")
                    break
        except Exception:
            role_max_tokens = None
        if role_max_tokens is None:
            role_max_tokens = ((cfg.get("nodes", {}) or {}).get("research", {}) or {}).get(
                "max_tokens"
            )
        if role_max_tokens is None:
            role_max_tokens = 2000
        # Default ~80% of 4 chars/token budget across 5 sources
        summary_char_limit = max(2000, int((int(role_max_tokens) * 4) * 0.8 / 5))

    if result_char_limit is None:
        result_char_limit = 65536

    return int(summary_char_limit), int(result_char_limit)


# removed: apply_tool_result_limits (logic merged into _truncate_response_for_role)


def _safe_request_get(
    url: str, timeout_s: int = 15, max_retries: int = 2, backoff: float = 1.0
) -> httpx.Response:
    """Synchronous httpx.get with retries/backoff.

    Avoids mixing async clients/locks across event loops, which caused
    'cannot reuse already awaited coroutine' during retries.
    """
    from httpx import HTTPStatusError

    last_exc: HTTPStatusError | Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = SYNC_HTTP_CLIENT.get(
                url, timeout=timeout_s, headers={"User-Agent": "DataSeek/1.0"}
            )
            resp.raise_for_status()
            return resp
        except HTTPStatusError as e:
            last_exc = e
            if e.response.status_code == 429:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                        print(
                            f"      ⏳ Rate limited (429) for {url}, waiting {wait_time}s as instructed"
                        )
                        time.sleep(min(wait_time, 60))
                        continue
                    except ValueError:
                        pass
                if attempt < max_retries:
                    rate_limit_backoff = backoff * (3**attempt)
                    print(
                        f"      ⏳ Rate limited (429) for {url}, backing off {rate_limit_backoff:.1f}s"
                    )
                    time.sleep(rate_limit_backoff)
                    continue
                else:
                    print(
                        f"      ❌ Rate limit exceeded for {url} after {max_retries + 1} attempts"
                    )
                    raise
            elif 400 <= e.response.status_code < 500:
                raise
            elif e.response.status_code >= 500:
                if attempt < max_retries:
                    server_error_backoff = backoff * (2**attempt)
                    print(
                        f"      ⚠️ Server error {e.response.status_code} for {url}, retrying in {server_error_backoff:.1f}s"
                    )
                    time.sleep(server_error_backoff)
                    continue
                else:
                    print(
                        f"      ❌ Server error persisted for {url} after {max_retries + 1} attempts"
                    )
                    raise
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                regular_backoff = backoff * (2**attempt)
                print(
                    f"      ⚠️ {type(e).__name__} for {url}: {str(e)}, retrying in {regular_backoff:.1f}s"
                )
                time.sleep(regular_backoff)
            else:
                print(
                    f"      ❌ {type(e).__name__} for {url} after {max_retries + 1} attempts: {str(e)}"
                )
                raise

    if last_exc is not None:
        raise last_exc  # pragma: no cover
    else:
        raise Exception("Unknown error occurred")  # pragma: no cover


def _extract_main_text_and_title(html: str, css_selector: str | None = None) -> dict[str, str]:
    """Extract title and main textual content from HTML."""
    soup = BeautifulSoup(html, "html5lib")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
        tag.decompose()
    content_text = soup.get_text(" ", strip=True)
    content_text = re.sub(r" \s+\n", "\n", content_text)
    content_text = re.sub(r"[ \t]{2,}", " ", content_text).strip()
    return {"title": title, "text": content_text}


def _to_markdown_simple(
    title: str, text: str, url: str | None = None, add_front_matter: bool = True
) -> str:
    """Produce a simple Markdown representation."""
    parts = []
    if title:
        parts.append(f"# {title}\n")
    if add_front_matter:
        fm = {
            "source_url": url or "",
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        parts.append(f"\n{fm}\n\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for p in paragraphs:
        parts.append(p + "\n")
    return "\n".join(parts).strip()


def _sha256_of_file(path: str) -> str:
    """Computes the SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------
# Pydantic schemas & tools
# -------------------------


class UrlToMarkdownInput(pydantic.BaseModel):
    """Input schema for url_to_markdown tool."""

    url: str = pydantic.Field(
        description="Fully qualified URL to fetch (e.g., https://example.com/article)."
    )
    css_selector: str | None = pydantic.Field(
        default=None, description="Optional CSS selector to isolate the main content."
    )
    timeout_s: int = pydantic.Field(default=15, description="HTTP timeout in seconds.")
    max_retries: int = pydantic.Field(default=2, description="Network retry attempts.")
    add_front_matter: bool = pydantic.Field(
        default=True,
        description="If true, include minimal front-matter metadata in the returned Markdown.",
    )
    user_agent: str = pydantic.Field(
        default="DataSeek/1.0", description="User-Agent string to use for the request."
    )


@tool("url_to_markdown", args_schema=UrlToMarkdownInput)
def url_to_markdown(
    url: str,
    css_selector: str | None = None,
    timeout_s: int = 15,
    max_retries: int = 2,
    add_front_matter: bool = True,
    user_agent: str = "DataSeek/1.0",
) -> dict[str, Any]:
    """
    Fetch a single web page, extract the main textual content and title, and return a Markdown string plus metadata.
    This tool respects robots.txt by default.
    Returns a dictionary with status, URL, markdown, title, and a text snippet.
    """
    # Load config and determine if we should check robots.txt
    config = get_active_seek_config()
    check_robots = config.get("use_robots", True)

    try:
        # Respect robots.txt if enabled
        if check_robots:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_url = urljoin(base_url, "/robots.txt")

            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                if not rp.can_fetch(user_agent, url):
                    return {
                        "url": url,
                        "markdown": "",
                        "title": "",
                        "text_snippet": "",
                        "status": "error",
                        "error": f"Request to {url} is disallowed by robots.txt.",
                    }
            except Exception as e:
                # It's often fine to proceed if robots.txt is unavailable or malformed
                print(f"      ⚠️ Could not fetch or parse robots.txt at {robots_url}: {e}")

        resp = _safe_request_get(url, timeout_s=timeout_s, max_retries=max_retries)
        html = resp.text
        extracted = _extract_main_text_and_title(html, css_selector=css_selector)

        # Prefer rich HTML->Markdown via attachments if available
        markdown = None
        try:
            from attachments import attach, present

            # Write HTML to a temporary file to preserve structure for conversion
            with tempfile.NamedTemporaryFile(
                "w", suffix=".html", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(html)
                temp_path = tf.name
            try:
                md = attach(temp_path) | present.markdown
                if isinstance(md, str) and md.strip():
                    # Optionally add minimal front matter and title if missing
                    parts_md = []
                    if extracted.get("title") and not md.lstrip().startswith("# "):
                        parts_md.append(f"# {extracted['title']}\n")
                    if add_front_matter:
                        fm = {
                            "source_url": url or "",
                            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        parts_md.append(f"\n{fm}\n\n")
                    parts_md.append(md)
                    markdown = "\n".join(parts_md).strip()
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    # Best-effort cleanup; ignore failure
                    pass  # nosec B110 # - best-effort temp cleanup only
        except Exception:
            markdown = None

        if not markdown:
            markdown = _to_markdown_simple(
                extracted["title"],
                extracted["text"],
                url=url,
                add_front_matter=add_front_matter,
            )

        snippet = extracted["text"][:500].strip()
        return {
            "url": url,
            "markdown": markdown,
            "title": extracted["title"],
            "text_snippet": snippet,
            "status": "ok",
        }
    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}"
        if isinstance(e, httpx.HTTPStatusError):
            if 400 <= e.response.status_code < 500:
                error_message = f"{e.response.status_code} - for {url}, not retrying"
        return {
            "url": url,
            "markdown": "",
            "title": "",
            "text_snippet": "",
            "status": "error",
            "error": error_message,
        }


class CrawlInput(pydantic.BaseModel):
    """Input schema for documentation_crawler tool."""

    base_url: str = pydantic.Field(
        description="The base URL of the documentation site (e.g., https://example.com)."
    )
    starting_point: str = pydantic.Field(
        description="The starting path (e.g., /docs/ or /en/latest/)."
    )
    max_depth: int = pydantic.Field(
        default=3, description="Max crawl depth to avoid runaway crawls."
    )
    allowed_paths: list[str] | None = pydantic.Field(
        default=None,
        description="Optional list of URL paths to include during crawling.",
    )
    ignore_paths: list[str] | None = pydantic.Field(
        default=None, description="Optional list of URL paths to skip during crawling."
    )
    timeout_s: int = pydantic.Field(
        default=30, description="Timeout for network ops during heuristic scanning."
    )
    similarity_threshold: float = pydantic.Field(
        default=0.7, description="Duplicate-similarity threshold for libcrawler."
    )


if _HAVE_LIBCRAWLER:

    @tool("documentation_crawler", args_schema=CrawlInput)
    def documentation_crawler(
        base_url: str,
        starting_point: str,
        max_depth: int = 3,
        allowed_paths: list[str] | None = None,
        ignore_paths: list[str] | None = None,
        timeout_s: int = 30,
        similarity_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """
        Deep-crawl a documentation site using Playwright for JS rendering and return a structured result.

        Best practices for configuration:
        1. It's highly recommended to first download and examine the starting page using the url_to_markdown tool
           to understand the site structure before running this crawler.
        2. Use the insights from the starting page to set appropriate allowed_paths and ignore_paths.
        3. Start with a shallow max_depth (1-2) and increase gradually to avoid crawling too much content.

        The tool will attempt to infer sensible allowed/ignored paths if none are provided, but manual
        configuration typically yields better results.
        """
        base_url_clean = base_url.rstrip("/")
        start_url = urljoin(base_url_clean + "/", starting_point.lstrip("/"))

        # Enhanced pre-scan with better path analysis
        try:
            config = get_active_seek_config()
            resp = _safe_request_get(start_url, timeout_s=timeout_s, max_retries=1)
            soup = BeautifulSoup(resp.text, "html5lib")
            hrefs = {urljoin(start_url, a.get("href", "")) for a in soup.find_all("a", href=True)}

            # Extract page title for better context
            title_tag = soup.find("title")
            page_title = title_tag.get_text(strip=True) if title_tag else "Unknown"
        except Exception as e:
            return {
                "base_url": base_url,
                "start_url": start_url,
                "pages": {},
                "status": "error",
                "error": f"Pre-scan failed: {type(e).__name__}: {str(e)}",
            }

        # Improved path inference with more sophisticated analysis
        inferred_allowed = allowed_paths or []
        if not inferred_allowed:
            common_prefixes: dict[str, int] = {}
            path_frequencies: dict[int, int] = {}

            for href in hrefs:
                if href.startswith(base_url_clean):
                    path = urlparse(href).path
                    # Count path segments to identify common structures
                    segments = [seg for seg in path.strip("/").split("/") if seg]
                    if segments:
                        # Track common first segments (e.g., /docs, /api, /guides)
                        first_segment = "/" + segments[0] if segments else "/"
                        common_prefixes[first_segment] = common_prefixes.get(first_segment, 0) + 1

                        # Track path depth frequencies
                        depth = len(segments)
                        path_frequencies[depth] = path_frequencies.get(depth, 0) + 1

            # Sort by frequency and take top 3 paths
            if common_prefixes:
                inferred_allowed = [
                    p
                    for p, count in sorted(
                        common_prefixes.items(), key=lambda item: item[1], reverse=True
                    )[:3]
                ]
            else:
                inferred_allowed = [starting_point]

        # Enhanced ignore patterns with more comprehensive defaults
        inferred_ignore = ignore_paths or []
        common_ignores = [
            "/login",
            "/signup",
            "/search",
            "/admin",
            "/dashboard",
            "/account",
            "/profile",
            "/settings",
            "/download",
            "/assets",
            "/static",
            "/images",
            "/css",
            "/js",
            "/fonts",
        ]
        for ignore in common_ignores:
            if ignore not in inferred_ignore:
                inferred_ignore.append(ignore)

        temp_dir = tempfile.mkdtemp(prefix="data_seek_crawl_")
        output_file = os.path.join(temp_dir, "crawled_docs.md")

        try:
            # Pass additional metadata to libcrawler for better processing
            asyncio.run(
                crawl_and_convert(
                    start_url=start_url,
                    base_url=base_url_clean,
                    output_filename=output_file,
                    allowed_paths=inferred_allowed,
                    ignore_paths=inferred_ignore,
                    similarity_threshold=similarity_threshold,
                    max_depth=max_depth,
                    respect_robots=config.get("use_robots", True),
                )
            )
            if os.path.exists(output_file):
                with open(output_file, encoding="utf-8") as f:
                    md_all = f.read()
                summary = md_all[:1000] + ("..." if len(md_all) > 1000 else "")
                pedigree_entry = f"""### {time.strftime("%Y-%m-%d")} — Documentation Crawl: {base_url}
- **Start URL:** `{start_url}`
- **Page Title:** `{page_title}`
- **Max Depth:** `{max_depth}`
- **Similarity Threshold:** `{similarity_threshold}`
- **Allowed Paths (Inferred):** `{json.dumps(inferred_allowed)}`
- **Ignored Paths (Inferred):** `{json.dumps(inferred_ignore)}`"""
                return {
                    "base_url": base_url,
                    "start_url": start_url,
                    "page_title": page_title,
                    "full_markdown": md_all,
                    "summary": summary,
                    "pedigree_entry": pedigree_entry,
                    "status": "ok",
                }
            else:
                return {
                    "base_url": base_url,
                    "start_url": start_url,
                    "pages": {},
                    "status": "error",
                    "error": "libcrawler completed but output file was not created",
                }
        except Exception as e:
            return {
                "base_url": base_url,
                "start_url": start_url,
                "pages": {},
                "status": "error",
                "error": f"crawl_and_convert failed: {type(e).__name__}: {str(e)}",
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

else:
    documentation_crawler_func: Callable[..., Any] | None = None


class WriteFileInput(pydantic.BaseModel):
    """Input schema for write_file tool."""

    filepath: str = pydantic.Field(description="Full path (directory + filename) to write.")
    content: str = pydantic.Field(description="Text content to write.")


@tool("write_file", args_schema=WriteFileInput)
def write_file(filepath: str, content: str) -> dict[str, Any]:
    """Writes text content to a file, creating directories if needed. Returns metadata including a sha256 checksum."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        bytes_written = len(content.encode("utf-8"))
        sha = _sha256_of_file(filepath)
        return {
            "filepath": filepath,
            "bytes_written": bytes_written,
            "sha256": sha,
            "status": "ok",
        }
    except Exception as e:
        return {
            "filepath": filepath,
            "bytes_written": 0,
            "sha256": None,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }


# -------------------------
# Fallback sample generation
# -------------------------

# A proper synthetic agent would need to be context-aware and generate targeted examples, a tool may not be useful for this.

# -------------------------
# Tool discovery / role scoping
# -------------------------


def get_available_tools() -> list[Callable]:
    """Return actual tool callables available in this environment."""
    # Create the web_search tool with the configured provider
    web_search = create_web_search_tool()
    core_tools = [
        url_to_markdown,
        write_file,
        web_search,
        arxiv_search,
        arxiv_get_content,
        wikipedia_search,
        wikipedia_get_content,
    ]
    optional = [documentation_crawler] if _HAVE_LIBCRAWLER else []
    return core_tools + optional


def get_tools_for_role(role: str, use_robots: bool = True) -> list[BaseTool]:
    """Return tools intended for a specific role."""
    role = (role or "").lower()

    # Create the web_search tool with the configured provider
    web_search = create_web_search_tool(use_robots=use_robots)

    # Build tool registry, guarding optional crawler tool
    all_tools = {
        "web_search": web_search,
        "arxiv_search": arxiv_search,
        "arxiv_get_content": arxiv_get_content,
        "wikipedia_search": wikipedia_search,
        "wikipedia_get_content": wikipedia_get_content,
        "url_to_markdown": url_to_markdown,
        "write_file": write_file,
    }
    if _HAVE_LIBCRAWLER:
        all_tools["documentation_crawler"] = documentation_crawler  # type: ignore[name-defined]

    # Define which tools are available for each role
    role_mapping = {
        "research": [
            "web_search",
            "arxiv_search",
            "arxiv_get_content",
            "wikipedia_search",
            "wikipedia_get_content",
            "url_to_markdown",
        ],
        "archive": ["write_file"],
        "supervisor": [],
        "fitness": [],
        "synthetic": [],
    }

    if _HAVE_LIBCRAWLER:
        role_mapping["research"].append("documentation_crawler")

    tool_names_for_role = role_mapping.get(role, [])
    return [all_tools[tool_name] for tool_name in tool_names_for_role if tool_name in all_tools]
