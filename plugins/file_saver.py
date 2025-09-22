import os
from typing import Any, ClassVar, cast

from pydantic import BaseModel, Field

from seek.components.tool_manager.file_tools import write_file as write_file_tool
from seek.components.tool_manager.plugin_base import BaseUtilityTool
from seek.components.tool_manager.registry import register_plugin


class FileSaverConfig(BaseModel):
    output_path: str = Field(
        default="datasets/samples", description="Default directory to save files."
    )


@register_plugin
class FileSaverTool(BaseUtilityTool):
    name: str = "file_saver"
    description: str = "Writes text content to a file, creating directories if needed."
    ConfigSchema: ClassVar[type[BaseModel]] = FileSaverConfig

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        filepath: str = kwargs["filepath"]
        content: str = kwargs["content"]
        if self.config:
            config_cast = cast(FileSaverConfig, self.config)
            output_path = config_cast.output_path
        else:
            config_default = FileSaverConfig()
            output_path = config_default.output_path
        final_path = filepath
        if not os.path.isabs(filepath):
            final_path = os.path.join(output_path, filepath)

        tool_input = {"filepath": final_path, "content": content}
        return write_file_tool.invoke(tool_input)
