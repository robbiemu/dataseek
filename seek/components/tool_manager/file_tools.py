import os
from typing import Any

import pydantic
from langchain_core.tools import tool

from .utils import _sha256_of_file


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
