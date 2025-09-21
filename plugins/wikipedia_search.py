from typing import Any, ClassVar, cast

from pydantic import BaseModel, Field

from seek.components.search_graph.litesearch import SearchProviderProxy
from seek.components.tool_manager.clients import HTTP_CLIENT, RATE_MANAGER
from seek.components.tool_manager.plugin_base import BaseSearchTool
from seek.components.tool_manager.registry import register_plugin


class WikipediaConfig(BaseModel):
    max_docs: int = Field(default=5, description="Maximum documents to retrieve.")


@register_plugin
class WikipediaSearchTool(BaseSearchTool):
    name: str = "wikipedia_search"
    description: str = "Searches Wikipedia and returns results with metadata."
    ConfigSchema: ClassVar[type[BaseModel]] = WikipediaConfig

    async def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        proxy = SearchProviderProxy(
            provider="wikipedia/search",
            rate_limit_manager=RATE_MANAGER,
            http_client=HTTP_CLIENT,
        )
        try:
            if self.config:
                config_cast = cast(WikipediaConfig, self.config)
                max_to_fetch = config_cast.max_docs
            else:
                max_to_fetch = num_results
            results = await proxy.run(query, max_results=max_to_fetch)

            return [
                {
                    "query": query,
                    "results": results,
                    "provider": "wikipedia/search",
                    "status": "ok",
                }
            ]
        except Exception as e:
            return [
                {
                    "query": query,
                    "results": None,
                    "provider": "wikipedia/search",
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)}",
                }
            ]
