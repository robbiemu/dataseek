from abc import abstractmethod
from typing import Any, ClassVar

from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel


class BaseTool(LangChainBaseTool):
    """Base for tool plugins compatible with LangGraph's ToolNode."""

    # Optional metadata (class-level, not Pydantic fields)
    version: ClassVar[str] = "1.0.0"
    author: ClassVar[str] = "Unknown"

    # Optional configuration schema (class-level hint for plugins)
    ConfigSchema: ClassVar[type[BaseModel] | None] = None
    # Instance configuration (kept as field to allow passing at init time)
    config: BaseModel | None = None

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Bridge sync invocation to the async implementation.

        MissionRunner drives the graph synchronously (app.stream), so ToolNode
        calls tool.invoke() which hits this method. We bridge to _arun() via
        asyncio.run() so async-only plugins work in the sync graph path.
        """
        import asyncio

        return asyncio.run(self._arun(*args, **kwargs))

    # Optional lifecycle hooks
    async def setup(self) -> None:
        """Called once when the tool is loaded for a mission."""
        pass

    async def teardown(self) -> None:
        """Called once when the mission is complete."""
        pass


class BaseSearchTool(BaseTool):
    """Abstract base class for a search-type tool."""

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        pass

    async def _arun(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        return await self.search(query, num_results)


class BaseUtilityTool(BaseTool):
    """Abstract base class for a utility-type tool."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        pass

    async def _arun(self, **kwargs: Any) -> dict[str, Any]:
        return await self.execute(**kwargs)
