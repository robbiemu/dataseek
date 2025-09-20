import asyncio
import hashlib
import os
import re
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup

from seek.common.config import get_active_seek_config

from .clients import SYNC_HTTP_CLIENT

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
