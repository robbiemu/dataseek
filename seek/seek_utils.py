"""
Search results validation and URL accessibility checking module.

This module provides functionality to validate search results by checking URL accessibility,
filtering blocked domains, and handling retry logic for failed searches.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import logging
import time
from urllib.parse import urlparse

import httpx
import yaml

from .config import get_active_seek_config
from .tools import _safe_request_get, _run_async_safely, RATE_MANAGER, HTTP_CLIENT

logger = logging.getLogger(__name__)

# Constants
DEFAULT_PREFETCH_LIMIT = 3
DEFAULT_MIN_RESULTS = 3
DEFAULT_MAX_RETRIES = 2
DEFAULT_RESULT_MULTIPLIER = 3
URL_VALIDATION_TIMEOUT = 10
RATE_LIMIT_RPS = 2.0


def find_url_field(results: List[Dict[str, Any]]) -> Optional[str]:
    """Find the URL field name in search results."""
    if not results:
        return None

    first = results[0]
    if not isinstance(first, dict):
        return None

    for url_field in ["url", "link"]:
        if url_field in first:
            return url_field

    return None


class SearchResultsValidator:
    """Handles validation and filtering of search results."""

    def __init__(self, tool_name: str, config=None):
        self.tool_name = tool_name
        self.config = config or get_active_seek_config()
        self._load_tool_config()

    def _load_tool_config(self):
        """Load configuration settings for the tool."""
        # Set defaults
        self.prefetch_enabled = False
        self.prefetch_limit = DEFAULT_PREFETCH_LIMIT
        self.validate_urls = True
        self.retry_on_failure = True
        self.max_retries = DEFAULT_MAX_RETRIES

        # Load structured seek config to get tool configuration
        seek_config = get_active_seek_config()
        tool_config = seek_config.get_tool_config(self.tool_name)
        if tool_config:
            self.prefetch_enabled = tool_config.pre_fetch_pages
            self.prefetch_limit = tool_config.pre_fetch_limit
            self.validate_urls = tool_config.validate_urls
            self.retry_on_failure = tool_config.retry_on_failure
            self.max_retries = tool_config.max_retries

    def validate_search_results(
        self,
        results: List[Dict[str, Any]],
        tool_args: Optional[Dict[str, Any]] = None,
        matching_tool=None,
        session_tool_domain_blocklist: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate search results by filtering blocked domains and checking URL accessibility.

        Args:
            results: List of search results to validate
            tool_args: Arguments used for the original tool call
            matching_tool: Tool instance for retry attempts
            session_tool_domain_blocklist: List of blocked (tool_name, domain) tuples

        Returns:
            Dictionary containing validated results and metadata
        """
        if not self.prefetch_enabled or not self.validate_urls:
            return self._create_response(results, validation_performed=False)

        # Filter blocked domains
        filtered_results, blocked_count = self._filter_blocked_domains(
            results, session_tool_domain_blocklist
        )

        # Expand search if insufficient results after filtering
        if (
            self.tool_name == "web_search"
            and len(filtered_results) < DEFAULT_MIN_RESULTS
            and results
            and matching_tool
            and tool_args
        ):
            filtered_results = self._expand_search_results(
                filtered_results,
                tool_args,
                matching_tool,
                session_tool_domain_blocklist,
            )

        # Validate URLs
        validation_batch = filtered_results[: self.prefetch_limit]
        logger.info(f"Validating {len(validation_batch)} results for {self.tool_name}")

        validated_results = self._validate_url_batch(validation_batch)
        accessible_results = [
            r for r in validated_results if r.get("status") == "accessible"
        ]

        # Retry if no accessible results found
        if (
            self.retry_on_failure
            and not accessible_results
            and results
            and matching_tool
            and tool_args
        ):
            retry_results = self._perform_retry(
                validated_results,
                tool_args,
                matching_tool,
            )
            if retry_results:
                return retry_results

        filtered_count = len(
            [r for r in validated_results if r.get("status") == "inaccessible"]
        )

        return self._create_response(
            validated_results,
            validation_performed=True,
            filtered_count=filtered_count,
            original_count=len(validation_batch),
        )

    def _filter_blocked_domains(
        self,
        results: List[Dict[str, Any]],
        blocklist: Optional[List[Tuple[str, str]]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Filter out results from blocked domains."""
        if not results or not blocklist:
            return results, 0

        # Only filter for tools that can have domains blocked
        url_field = find_url_field(results)
        if not url_field:
            return results, 0

        filtered_results = []
        blocked_count = 0

        for result in results:
            if not isinstance(result, dict) or url_field not in result:
                filtered_results.append(result)
                continue

            try:
                domain = urlparse(result[url_field]).netloc
                if (self.tool_name, domain) in blocklist:
                    blocked_count += 1
                    logger.info(f"Filtered blocked result from {domain}")
                else:
                    filtered_results.append(result)
            except Exception:
                # Include result if domain parsing fails
                filtered_results.append(result)

        if blocked_count > 0:
            logger.info(
                f"Filtered {blocked_count} blocked results for {self.tool_name}"
            )

        return filtered_results, blocked_count

    def _expand_search_results(
        self,
        filtered_results: List[Dict[str, Any]],
        tool_args: Dict[str, Any],
        matching_tool,
        blocklist: Optional[List[Tuple[str, str]]],
    ) -> List[Dict[str, Any]]:
        """Expand search results when insufficient results remain after filtering."""
        logger.info(
            f"Expanding search: only {len(filtered_results)} results after filtering"
        )

        expanded_queries = self._generate_expanded_queries(tool_args.get("query", ""))
        expanded_results = []

        for i, expanded_query in enumerate(expanded_queries):
            if len(expanded_results) >= DEFAULT_MIN_RESULTS:
                break

            logger.info(f"Trying expanded query {i + 1}: {expanded_query}")

            try:
                expanded_tool_args = tool_args.copy()
                expanded_tool_args["query"] = expanded_query

                expanded_tool_result = matching_tool.invoke(expanded_tool_args)

                if isinstance(expanded_tool_result, dict) and expanded_tool_result.get(
                    "results"
                ):
                    new_results, _ = self._filter_blocked_domains(
                        expanded_tool_result["results"], blocklist
                    )
                    expanded_results.extend(new_results)
                    logger.info(
                        f"Expanded query yielded {len(new_results)} new results"
                    )

            except Exception as e:
                logger.warning(f"Expanded query failed: {e}")

        # Combine and deduplicate results
        combined_results = filtered_results + expanded_results
        unique_results = self._deduplicate_results(combined_results)

        # Limit to reasonable number
        final_results = unique_results[:10]
        logger.info(f"Final result set: {len(final_results)} results")

        return final_results

    def _generate_expanded_queries(self, original_query: str) -> List[str]:
        """Generate expanded search queries for retry attempts."""
        return [
            f"{original_query} site:*",
            f"{original_query} *",
            f"related:{original_query}",
        ]

    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL."""
        seen_urls: Set[str] = set()
        unique_results = []

        for result in results:
            if isinstance(result, dict):
                url_field = find_url_field([result])
                if url_field and url_field in result:
                    url = result[url_field]
                    if url not in seen_urls:
                        seen_urls.add(url)
                        unique_results.append(result)
                else:
                    unique_results.append(result)
            else:
                unique_results.append(result)

        return unique_results

    def _validate_url_batch(
        self,
        results_batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Validate a batch of search results by checking URL accessibility."""
        validated_results = []

        for result in results_batch:
            if not isinstance(result, dict):
                validated_results.append({"content": result, "status": "accessible"})
                continue

            url = self._extract_url_from_result(result)

            if url:
                status, error = self._check_url_accessibility(url)
                result_copy = result.copy()
                result_copy["status"] = status
                if error:
                    result_copy["error"] = error
                validated_results.append(result_copy)
            else:
                # Handle results without URLs - assume accessible
                result_copy = result.copy()
                result_copy["status"] = "accessible"
                validated_results.append(result_copy)

        return validated_results

    def _check_url_accessibility(self, url: str) -> Tuple[str, Optional[str]]:
        """Check if a URL is accessible."""
        try:
            response = self._safe_request_head(url)
            if response.status_code == 200:
                return "accessible", None
            else:
                logger.warning(
                    f"URL {url} inaccessible (status {response.status_code})"
                )
                return "inaccessible", f"HTTP {response.status_code}"

        except Exception:
            # Fallback to GET request
            try:
                response = _safe_request_get(
                    url, timeout_s=URL_VALIDATION_TIMEOUT, max_retries=1
                )
                if response.status_code == 200:
                    return "accessible", None
                else:
                    logger.warning(
                        f"URL {url} inaccessible (status {response.status_code})"
                    )
                    return "inaccessible", f"HTTP {response.status_code}"

            except Exception as e:
                logger.warning(f"URL {url} inaccessible ({type(e).__name__})")
                return "inaccessible", f"{type(e).__name__}: {str(e)}"

    def _safe_request_head(self, url: str) -> httpx.Response:
        """Perform a rate-limited HEAD request."""
        domain = urlparse(url).netloc

        for attempt in range(self.max_retries + 1):
            try:

                async def do_request():
                    async with RATE_MANAGER.acquire(
                        f"domain:{domain}", requests_per_second=RATE_LIMIT_RPS
                    ):
                        return await HTTP_CLIENT.head(
                            url,
                            timeout=URL_VALIDATION_TIMEOUT,
                            headers={"User-Agent": "DataSeek/1.0"},
                        )

                return _run_async_safely(do_request())

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 405:
                    raise  # Method not allowed - fallback to GET
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2**attempt))
                    continue
                raise

            except Exception as _e:
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2**attempt))
                    continue
                raise

    def _perform_retry(
        self,
        failed_results: List[Dict[str, Any]],
        tool_args: Dict[str, Any],
        matching_tool,
    ) -> Optional[Dict[str, Any]]:
        """Perform retry with expanded parameters when all initial results fail."""
        logger.info(
            "All initial results inaccessible, attempting retry with expanded results"
        )

        bad_urls = {
            r.get("url")
            for r in failed_results
            if r.get("status") == "inaccessible" and r.get("url")
        }

        for attempt in range(self.max_retries):
            try:
                expanded_args = self._create_expanded_args(tool_args)
                logger.info(
                    f"Retry attempt {attempt + 1}/{self.max_retries} with: {expanded_args}"
                )

                expanded_result = matching_tool.invoke(expanded_args)

                if (
                    isinstance(expanded_result, dict)
                    and expanded_result.get("status") == "ok"
                ):
                    expanded_results = expanded_result.get("results", [])
                    fresh_results = self._filter_fresh_results(
                        expanded_results, bad_urls
                    )

                    if fresh_results:
                        validation_limit = min(
                            self.prefetch_limit * 2, len(fresh_results)
                        )
                        validated_fresh = self._validate_url_batch(
                            fresh_results[:validation_limit]
                        )

                        accessible_fresh = [
                            r
                            for r in validated_fresh
                            if r.get("status") == "accessible"
                        ]

                        if accessible_fresh:
                            logger.info(
                                f"Retry successful: Found {len(accessible_fresh)} accessible results"
                            )
                            return self._create_response(
                                validated_fresh,
                                validation_performed=True,
                                filtered_count=len(validated_fresh)
                                - len(accessible_fresh),
                                original_count=len(failed_results),
                                retry_performed=True,
                                retry_successful=True,
                                bad_urls_excluded=list(bad_urls),
                            )

                        # Add newly discovered bad URLs
                        bad_urls.update(
                            r.get("url")
                            for r in validated_fresh
                            if r.get("status") == "inaccessible" and r.get("url")
                        )

            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")

        logger.error("All retry attempts failed")
        return None

    def _create_expanded_args(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Create expanded arguments for retry attempts."""
        expanded_args = tool_args.copy()

        # Increase result count if possible
        for count_field in ["num_results", "count", "limit"]:
            if count_field in expanded_args:
                expanded_args[count_field] *= DEFAULT_RESULT_MULTIPLIER
                break

        return expanded_args

    def _filter_fresh_results(
        self,
        results: List[Dict[str, Any]],
        bad_urls: Set[str],
    ) -> List[Dict[str, Any]]:
        """Filter out results with URLs we already know are bad."""
        fresh_results = []

        for result in results:
            if isinstance(result, dict):
                url = self._extract_url_from_result(result)
                if url and url not in bad_urls:
                    fresh_results.append(result)
                elif not url:
                    fresh_results.append(result)
            else:
                fresh_results.append(result)

        return fresh_results

    def _extract_url_from_result(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract URL from a search result."""
        url_field = find_url_field([result])
        return result[url_field] if url_field else None

    def _create_response(
        self,
        results: List[Dict[str, Any]],
        validation_performed: bool = False,
        needs_retry: bool = False,
        filtered_count: int = 0,
        original_count: Optional[int] = None,
        retry_performed: bool = False,
        retry_successful: bool = False,
        bad_urls_excluded: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a standardized response dictionary."""
        response = {
            "results": results,
            "validation_performed": validation_performed,
            "needs_retry": needs_retry,
            "filtered_count": filtered_count,
            "retry_performed": retry_performed,
        }

        if original_count is not None:
            response["original_count"] = original_count

        if retry_performed:
            response["retry_successful"] = retry_successful

        if bad_urls_excluded:
            response["bad_urls_excluded"] = bad_urls_excluded

        return response


def _validate_search_results(
    results: List[Dict[str, Any]],
    tool_name: str,
    tool_args: Dict[str, Any] = None,
    matching_tool=None,
    session_tool_domain_blocklist: List[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """
    Legacy function wrapper for backward compatibility.

    Args:
        results: List of search results to validate
        tool_name: Name of the search tool
        tool_args: Arguments passed to the search tool
        matching_tool: Tool instance for retry operations
        session_tool_domain_blocklist: List of blocked (tool_name, domain) tuples

    Returns:
        Dictionary containing validated results and metadata
    """
    validator = SearchResultsValidator(tool_name)
    return validator.validate_search_results(
        results, tool_args, matching_tool, session_tool_domain_blocklist
    )


class MissionDetailsParser:
    """Parser for mission configuration files."""

    @staticmethod
    def get_mission_details_from_file(
        mission_plan_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse mission_config.yaml to extract mission names and target sizes.

        Args:
            mission_plan_path: Path to the mission configuration file

        Returns:
            Dictionary with mission names and targets, or None if parsing fails
        """
        try:
            with open(mission_plan_path, "r", encoding="utf-8") as f:
                content = f.read()

                # Remove comment header if present
                if content.startswith("#"):
                    first_newline = content.find("\n")
                    if first_newline != -1:
                        content = content[first_newline + 1 :]

                mission_plan = yaml.safe_load(content)

                if not mission_plan or "missions" not in mission_plan:
                    logger.warning(f"No missions found in {mission_plan_path}")
                    return None

                return MissionDetailsParser._extract_mission_info(mission_plan)

        except FileNotFoundError:
            logger.error(f"Mission plan file not found: {mission_plan_path}")
            return None
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {mission_plan_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading {mission_plan_path}: {e}")
            return None

    @staticmethod
    def _extract_mission_info(mission_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Extract mission information from parsed YAML."""
        mission_names = []
        mission_targets = {}

        for mission in mission_plan.get("missions", []):
            name = mission.get("name")
            if not name:
                logger.warning("Found mission without name, skipping")
                continue

            mission_names.append(name)

            # Calculate total samples
            target_size = mission.get("target_size", 0)
            goals = mission.get("goals", [])
            total_samples = target_size * len(goals)
            mission_targets[name] = total_samples

        return {"mission_names": mission_names, "mission_targets": mission_targets}


# Legacy function for backward compatibility
def get_mission_details_from_file(mission_plan_path: str) -> Optional[Dict[str, Any]]:
    """Legacy wrapper for mission details parsing."""
    return MissionDetailsParser.get_mission_details_from_file(mission_plan_path)
