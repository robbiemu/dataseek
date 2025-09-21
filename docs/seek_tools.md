# Seek Tools Documentation

## Overview

The Data Seek Agent includes several specialized tools for finding and retrieving information from various sources. These tools are designed to be robust and efficient, with built-in validation and retry mechanisms to handle common network issues.

## Arxiv Search Tool

The `arxiv_search` tool allows you to search for academic papers on Arxiv.org. It's a free tool that doesn't require API keys.

### Usage
```python
from seek.components.tool_manager.tools import arxiv_search

result = arxiv_search("machine learning")
```

## Wikipedia Search Tool

The `wikipedia_search` tool allows you to search for information on Wikipedia. It's a free tool that doesn't require API keys.

### Usage
```python
from seek.components.tool_manager.tools import wikipedia_search

result = wikipedia_search("artificial intelligence")
```

## Documentation Crawler Tool

The `documentation_crawler` tool performs deep crawling of documentation sites using Playwright for JavaScript rendering.

### Best Practices

1. **Examine the starting page first**: Before running the crawler, use the `url_to_markdown` tool to download and examine the starting page. This will help you understand the site structure and configure appropriate paths.

2. **Configure paths carefully**: Based on your examination of the starting page, set appropriate `allowed_paths` and `ignore_paths` parameters to focus the crawl on relevant content and avoid irrelevant sections.

3. **Start shallow**: Begin with a shallow `max_depth` (1-2) and increase gradually to avoid crawling too much content.

### Usage
```python
from seek.components.tool_manager.tools import documentation_crawler

result = documentation_crawler(
    base_url="https://example.com",
    starting_point="/docs/",
    max_depth=2,
    allowed_paths=["/docs/guides", "/docs/api"],
    ignore_paths=["/docs/admin", "/docs/private"]
)
```

## URL to Markdown Conversion

The `url_to_markdown` tool converts a single web page into Markdown.

- When the `attachments` package is available, it uses the pipeline `attach(file_or_url) | present.markdown` for high-fidelity HTML→Markdown conversion.
- If `attachments` is unavailable or conversion fails, it falls back to a simple, robust extractor that strips scripts/styles and emits minimal Markdown.

Robots.txt is respected by default based on the global `use_robots` setting in the seek config.

## Tool Configuration and Validation

All search tools support advanced configuration options for prefetching and validation to make the research process more robust:

### Prefetching and Validation Features

1. **URL Validation**: Tools can automatically validate that URLs are accessible before returning results
2. **Automatic Retry**: When all results are inaccessible, tools can automatically retry with expanded result sets
3. **Smart Filtering**: Previously validated bad URLs are excluded from retry attempts to avoid redundant checks
4. **Efficient Checking**: Uses HEAD requests for initial validation to minimize bandwidth usage

### Configuration

Tool behavior can be configured in the mission plan YAML file:

```yaml
missions:
  - name: "production_corpus"
    tools:
      web_search:
        pre_fetch_pages: true
        pre_fetch_limit: 5
        validate_urls: true
        retry_on_failure: true
        max_retries: 2
```

See the [Data Seek Agent Guide](../guides/data-seek-agent.md#65-advanced-configuration-tool-prefetching-and-validation) for complete documentation of tool configuration options.

## Result Size Limits (Post-Tool)

The agent enforces post-tool truncation for efficiency. Search tools truncate per entry to `summary_char_limit`; fetch tools cap markdown to `result_char_limit`. Defaults derive from node `max_tokens`.

For full details, see [Configuration Guide - Result Size Limits](guides/configuration-guide.md#result-size-limits).
