from typing import Any, ClassVar, cast

from pydantic import BaseModel, Field

from seek.components.tool_manager.plugin_base import BaseUtilityTool
from seek.components.tool_manager.registry import register_plugin
from seek.components.tool_manager.web_tools import url_to_markdown as url_to_markdown_func


class UrlToMarkdownConfig(BaseModel):
    timeout: int = Field(default=15, description="HTTP timeout in seconds.")
    max_retries: int = Field(default=2, description="Network retry attempts.")
    add_front_matter: bool = Field(
        default=True,
        description="If true, include minimal front-matter metadata in the returned Markdown.",
    )
    user_agent: str = Field(
        default="DataSeek/1.0", description="User-Agent string to use for the request."
    )


@register_plugin
class UrlToMarkdownTool(BaseUtilityTool):
    name: str = "url_to_markdown"
    description: str = "Fetches a URL and converts main content to Markdown with metadata."
    ConfigSchema: ClassVar[type[BaseModel]] = UrlToMarkdownConfig

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        url: str = kwargs["url"]
        if self.config:
            config_cast = cast(UrlToMarkdownConfig, self.config)
            timeout = config_cast.timeout
            max_retries = config_cast.max_retries
            add_front_matter = config_cast.add_front_matter
            user_agent = config_cast.user_agent
        else:
            config_default = UrlToMarkdownConfig()
            timeout = config_default.timeout
            max_retries = config_default.max_retries
            add_front_matter = config_default.add_front_matter
            user_agent = config_default.user_agent
        tool_input = {
            "url": url,
            "timeout_s": timeout,
            "max_retries": max_retries,
            "add_front_matter": add_front_matter,
            "user_agent": user_agent,
        }
        return url_to_markdown_func.invoke(tool_input)
