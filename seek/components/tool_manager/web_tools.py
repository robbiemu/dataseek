import asyncio
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pydantic
from langchain_core.tools import tool

from seek.common.config import get_active_seek_config

from .utils import (
    _extract_main_text_and_title,
    _safe_request_get,
    _to_markdown_simple,
)

# Optional import for deep crawling
try:
    from libcrawler.libcrawler import crawl_and_convert

    _HAVE_LIBCRAWLER = True
except ImportError:
    crawl_and_convert = None
    _HAVE_LIBCRAWLER = False


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
        import httpx

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
        from bs4 import BeautifulSoup, Tag

        base_url_clean = base_url.rstrip("/")
        start_url = urljoin(base_url_clean + "/", starting_point.lstrip("/"))

        # Enhanced pre-scan with better path analysis
        try:
            config = get_active_seek_config()
            resp = _safe_request_get(start_url, timeout_s=timeout_s, max_retries=1)
            soup = BeautifulSoup(resp.text, "html5lib")
            # Use type narrowing to ensure we only process Tag elements
            links = [a for a in soup.find_all("a", href=True) if isinstance(a, Tag)]
            hrefs: set[str] = set()
            for link in links:
                href = link.get("href", "")
                if isinstance(href, str) and href:
                    joined_url = urljoin(start_url, href)
                    if isinstance(joined_url, str):
                        hrefs.add(joined_url)

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
                if isinstance(href, str) and href.startswith(base_url_clean):
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
