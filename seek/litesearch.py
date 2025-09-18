import asyncio
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import httpx
from langchain_community.tools import DuckDuckGoSearchRun

# Import all the necessary wrappers from the LangChain ecosystem
from langchain_community.utilities import (
    ArxivAPIWrapper,
    # BraveSearchWrapper,
    BingSearchAPIWrapper,
    PubMedAPIWrapper,
    SerpAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_community.utilities.you import YouSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_tavily import TavilySearch


class AsyncRateLimitManager:
    """
    Manages rate limits for different API providers in an async environment.

    It starts with a conservative default (1 req/sec) for header-based
    providers and switches to dynamic header-based limiting once the first
    API response is received.
    """

    def __init__(self):
        self._providers: dict[str, dict[str, Any]] = {}

    def _get_provider_state(self, provider: str) -> dict[str, Any]:
        """Initializes and/or returns the state for a given provider."""
        if provider not in self._providers:
            self._providers[provider] = {
                "lock": asyncio.Lock(),
                "initialized": False,  # Starts as uninitialized
                "remaining": 1,
                "reset_time": 0,
                "request_timestamps": deque(),
            }
        return self._providers[provider]

    @asynccontextmanager
    async def acquire(
        self,
        provider: str,
        requests_per_second: float | None = None,
    ):
        """
        An async context manager to acquire a slot for an API call.
        Waits if the rate limit has been reached.
        """
        state = self._get_provider_state(provider)
        async with state["lock"]:
            # Case 1: Provider uses a fixed, explicit time-based limit.
            if requests_per_second:
                now = time.time()
                while state["request_timestamps"] and state["request_timestamps"][0] <= now - 1.0:
                    state["request_timestamps"].popleft()

                if len(state["request_timestamps"]) >= requests_per_second:
                    time_to_wait = 1.0 - (now - state["request_timestamps"][0])
                    await asyncio.sleep(time_to_wait)
                state["request_timestamps"].append(time.time())

            # Case 2: Provider is header-based.
            else:
                # Subcase 2a: Not initialized. Use a safe default of 1 req/sec.
                if not state["initialized"]:
                    now = time.time()
                    while (
                        state["request_timestamps"] and state["request_timestamps"][0] <= now - 1.0
                    ):
                        state["request_timestamps"].popleft()

                    if len(state["request_timestamps"]) >= 1:  # Conservative 1 req/sec
                        time_to_wait = 1.0 - (now - state["request_timestamps"][0])
                        await asyncio.sleep(time_to_wait)
                    state["request_timestamps"].append(time.time())

                # Subcase 2b: Initialized. Use dynamic header values.
                else:
                    if state["remaining"] <= 0:
                        current_time = time.time()
                        sleep_duration = state["reset_time"] - current_time
                        if sleep_duration > 0:
                            await asyncio.sleep(sleep_duration)
                    state["remaining"] -= 1
        try:
            yield
        finally:
            pass

    def update_from_headers(self, provider: str, headers: dict[str, Any]):
        """Updates the rate limit state from API response headers."""
        state = self._get_provider_state(provider)
        headers = {k.lower(): v for k, v in headers.items()}

        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")

        if remaining is not None:
            # Handle comma-separated values (Brave API returns "0, 1995" format)
            remaining_str = str(remaining).split(",")[0].strip()
            try:
                state["remaining"] = int(remaining_str)
                state["initialized"] = True  # Mark as initialized
            except ValueError:
                # If parsing fails, keep using default conservative approach
                pass

        if reset is not None:
            # Handle comma-separated values (Brave API returns "1, 2369545" format)
            reset_str = str(reset).split(",")[0].strip()
            try:
                state["reset_time"] = int(reset_str)
            except ValueError:
                # If parsing fails, keep using default conservative approach
                pass


class SearchProviderProxy:
    """
    An async, rate-limited proxy for various search providers.

    It uses a RateLimitManager to avoid 429 errors and can be configured
    to use different backends like "brave/search".
    """

    def __init__(
        self,
        provider: str,
        rate_limit_manager: AsyncRateLimitManager,
        http_client: httpx.AsyncClient,
    ):
        """
        Initializes the proxy.

        Args:
            provider (str): Identifier like "brave/search".
            rate_limit_manager: An instance of AsyncRateLimitManager.
            http_client: An instance of httpx.AsyncClient for making API calls.
        """
        self.provider = provider
        self.rate_limit_manager = rate_limit_manager
        self.http_client = http_client
        self.client, self.is_custom = self._get_client()

    def _get_client(self) -> tuple[Any, bool]:
        """
        Maps the provider string to a handler.

        Returns a tuple of (handler, is_custom_implementation).
        'is_custom' is True if we are making the HTTP call directly,
        False if we are using a standard LangChain wrapper.
        """
        # Providers for which we need custom logic to get headers
        if self.provider == "brave/search":
            return self._run_brave_async, True

        # Mapping for standard LangChain wrappers
        provider_map = {
            "tavily/search": TavilySearch,
            "bing/search": BingSearchAPIWrapper,
            "serpapi/search": SerpAPIWrapper,
            "you/search": YouSearchAPIWrapper,
            "pubmed/search": PubMedAPIWrapper,
            "wikipedia/search": WikipediaAPIWrapper,
            "arxiv/search": ArxivAPIWrapper,
        }

        # Special handling for providers that need additional configuration
        if self.provider == "google/search":
            # Google Search needs API key and CSE ID
            api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
            cse_id = os.getenv("CSE_ID")
            if not api_key:
                raise ValueError(
                    "GOOGLE_SEARCH_API_KEY environment variable not set for Google Search provider"
                )
            if not cse_id:
                raise ValueError("CSE_ID environment variable not set for Google Search provider")
            return GoogleSearchAPIWrapper(google_api_key=api_key, google_cse_id=cse_id), False
        elif self.provider == "duckduckgo/search":
            return DuckDuckGoSearchRun(), False

        if self.provider in provider_map:
            client_class = provider_map[self.provider]
            return client_class(), False
        else:
            raise ValueError(f"Unsupported provider: '{self.provider}'.")

    async def _run_brave_async(self, query: str, **kwargs: Any) -> str:
        """Custom implementation for Brave Search to handle rate limits."""
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            raise ValueError("BRAVE_SEARCH_API_KEY environment variable not set.")

        params = {"q": query, **kwargs}
        headers = {"X-Subscription-Token": api_key}

        response = await self.http_client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        self.rate_limit_manager.update_from_headers(self.provider, response.headers)

        # Process and return the result similarly to the LC wrapper
        data = response.json()
        if not data.get("web") or not data["web"].get("results"):
            return "No good search results found."

        # Return the original results format - let the validation handle normalization
        return data["web"]["results"]

    async def run(self, query: str, **kwargs: Any) -> Any:
        """
        Runs a search query using the configured provider, respecting rate limits.

        Args:
            query (str): The search query.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Provider-specific results. For Google Search this returns a
            structured list of result dicts (via `results()`), while providers
            that only expose `run()` may return a formatted string.
        """
        # Allow caller to override RPS; default to 2.0 for non-custom providers
        requests_per_second = kwargs.pop("requests_per_second", None)
        if requests_per_second is None:
            requests_per_second = 2.0 if not self.is_custom else None

        async with self.rate_limit_manager.acquire(
            self.provider, requests_per_second=requests_per_second
        ):
            if self.is_custom:
                # Custom clients are already async and handle headers
                return await self.client(query, **kwargs)
            else:
                # Map a canonical max_results to provider-specific params/methods
                max_results = kwargs.pop("max_results", None)

                # Google: structured list via results(query, num_results)
                if self.provider == "google/search" and hasattr(self.client, "results"):
                    num_results = max_results if max_results is not None else 10
                    return await asyncio.to_thread(self.client.results, query, num_results)

                # Bing: structured list via results(query, num_results=)
                if self.provider == "bing/search" and hasattr(self.client, "results"):
                    num_results = max_results if max_results is not None else 10
                    return await asyncio.to_thread(self.client.results, query, num_results)

                # SerpAPI: structured via results(query, num=)
                if self.provider == "serpapi/search" and hasattr(self.client, "results"):
                    num = max_results if max_results is not None else 10
                    # Pass as positional to avoid kwarg mismatches across versions
                    return await asyncio.to_thread(self.client.results, query, num)

                # Tavily: prefer search(query, max_results=)
                if self.provider == "tavily/search" and hasattr(self.client, "search"):
                    kwargs_local = {}
                    if max_results is not None:
                        kwargs_local["max_results"] = max_results
                    return await asyncio.to_thread(self.client.search, query, **kwargs_local)

                # You.com: run(query, num_web_results=)
                if self.provider == "you/search" and hasattr(self.client, "run"):
                    if max_results is not None:
                        kwargs["num_web_results"] = max_results
                    return await asyncio.to_thread(self.client.run, query, **kwargs)

                # Brave custom: map to 'count'
                if self.provider == "brave/search" and self.is_custom:
                    if max_results is not None:
                        kwargs["count"] = max_results
                    return await self.client(query, **kwargs)

                # Default path: run() with any remaining kwargs
                if not hasattr(self.client, "run"):
                    raise NotImplementedError(f"Client for '{self.provider}' has no 'run' method.")
                return await asyncio.to_thread(self.client.run, query, **kwargs)


# --- Example Usage ---


async def main():
    """
    Demonstrates how to use the async, rate-limited SearchProviderProxy.
    """
    rate_manager = AsyncRateLimitManager()
    async with httpx.AsyncClient() as http_client:
        print("--- Testing DuckDuckGo (fixed 2 req/sec limit) ---")
        try:
            ddg_proxy = SearchProviderProxy("duckduckgo/search", rate_manager, http_client)
            tasks = [
                ddg_proxy.run("Benefits of serverless computing?"),
                ddg_proxy.run("What is WebAssembly?"),
                ddg_proxy.run("Latest AI news"),
            ]
            results = await asyncio.gather(*tasks)
            for i, res in enumerate(results):
                print(f"Result {i + 1}: " + res[:100] + "...")
        except Exception as e:
            print(f"Error with DuckDuckGo: {e}")

        print("\n" + "=" * 50 + "\n")

        print("--- Testing Brave Search (starts with 1 req/sec, then uses headers) ---")
        if os.getenv("BRAVE_SEARCH_API_KEY"):
            try:
                brave_proxy = SearchProviderProxy("brave/search", rate_manager, http_client)
                # These three will be spaced out by the default 1 req/sec limit
                tasks = [
                    brave_proxy.run("What is LangGraph?"),
                    brave_proxy.run("Key features of Rust programming language?"),
                    brave_proxy.run("Async programming in Python"),
                ]
                results = await asyncio.gather(*tasks)
                for i, res in enumerate(results):
                    print(f"Result {i + 1}: " + res[:100] + "...")
            except Exception as e:
                print(f"Error with Brave Search: {e}")
        else:
            print("BRAVE_SEARCH_API_KEY env var not set. Skipping Brave Search test.")


if __name__ == "__main__":
    asyncio.run(main())
