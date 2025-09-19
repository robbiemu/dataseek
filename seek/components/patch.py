"""
LiteLLM/Ollama patch shims (component-level).

Why this exists
- Some Ollama models (notably gpt-oss-20b) do not interoperate reliably with
  LiteLLM’s default transformation pipeline. Symptoms include dropped tool calls
  and empty content when using tool-enabled prompts. This file installs a
  targeted runtime patch to make those code paths robust in our environment.

Placement rationale
- This is not a tool definition; it affects model invocation semantics used by
  multiple components. We keep it at `seek/components/patch.py` so it’s applied
  early via `seek/__init__.py` but remains separate from the tool manager.

Observability
- We prefer LangChain/LangGraph’s tracing (LANGCHAIN_TRACING_V2) over LiteLLM’s
  logger to avoid event-loop pitfalls. This file prints only if
  VERBOSE_PATCHING=true.
"""

# Import the modules we need to patch
import json
import os
from typing import Any

import litellm
import litellm.litellm_core_utils.prompt_templates.factory
import litellm.llms.ollama.completion.transformation
from langchain_core.messages import AIMessage

# Check if verbose patching logs are enabled
VERBOSE_PATCHING = os.getenv("VERBOSE_PATCHING", "false").lower() == "true"


def _patch_log(message: str) -> None:
    """Log patching messages only if verbose mode is enabled."""
    if VERBOSE_PATCHING:
        print(message)


def _patched_ollama_pt(
    model: str, messages: list, roles: Any | None = None, model_kwargs: Any | None = None
) -> Any:
    """
    Fully patched version of litellm's ollama_pt function.

    This is a complete replacement of the original buggy function to avoid
    the IndexError when accessing messages[msg_i] after incrementing msg_i
    in a loop where the last message is from the assistant.

    Args:
        model (str): The model name
        messages (list): List of message dictionaries
        roles (dict, optional): Role mappings (unused in this implementation)
        model_kwargs (dict, optional): Additional model arguments (unused in this implementation)

    Returns:
        dict: A dictionary with 'prompt' and 'images' keys
    """

    # Handle the common case where we just want to format messages with roles
    # This is a simplified version that should work for most cases
    # and avoids the complex logic that has the bug

    prompt = ""
    images = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle image content if present
        if isinstance(content, list):
            text_content = ""
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content += item.get("text", "")
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", "")
                        if isinstance(image_url, dict):
                            images.append(image_url.get("url", ""))
                        else:
                            images.append(image_url)
            content = text_content

        # Format the prompt based on role
        if role == "system":
            prompt += f"<system>\n{content}\n</system>\n"
        elif role in ["user", "tool", "function"]:
            prompt += f"<user>\n{content}\n</user>\n"
        elif role == "assistant":
            prompt += f"<assistant>\n{content}\n"
            # Handle tool calls if present
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                prompt += "<tool_code>\n"
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict) and "function" in tool_call:
                        function = tool_call["function"]
                        name = function.get("name", "")
                        args = function.get("arguments", "")
                        prompt += f"{name}({args})\n"
                prompt += "</tool_code>\n"
            prompt += "\n"

    # Return the structured response expected by the caller
    response_dict = {
        "prompt": prompt,
        "images": images,
    }

    return response_dict


def _fix_malformed_json_arguments(args_str: str) -> str:
    """
    Attempt to fix malformed JSON using the json-repair library.
    """
    if not args_str or not isinstance(args_str, str):
        return args_str

    try:
        # Use json-repair library to fix malformed JSON
        import json_repair

        repaired = json_repair.repair_json(args_str)

        # Validate the repaired JSON can be parsed
        json.loads(repaired)
        return repaired

    except Exception as e:
        print(f"🔧 json-repair failed: {e}, falling back to empty dict")
        return "{}"


def _direct_ollama_call_with_tools(
    messages: list,
    model: str,
    tools: Any | None = None,
    temperature: float = 0.1,
    timeout: Any | None = None,
    **kwargs: Any,
) -> "AIMessage":
    """
    Direct call to Ollama API that properly handles tools.

    When LiteLLM fails to preserve tool calls, this function bypasses
    LiteLLM entirely and calls Ollama directly, then formats the response
    to be compatible with LangChain.

    Args:
        timeout: Optional timeout in seconds. If None, will use config timeout or default to 120.
    """
    try:
        import json

        import requests
        from langchain_core.messages import AIMessage

        # Convert LangChain messages to Ollama format
        ollama_messages = []
        for msg in messages:
            if hasattr(msg, "content"):
                # Handle different message formats
                if hasattr(msg, "type"):
                    # LangChain message objects
                    role_map = {"human": "user", "ai": "assistant", "system": "system"}
                    role = role_map.get(msg.type, "user")
                elif hasattr(msg, "role"):
                    # Already in dict format
                    role = msg.role
                else:
                    role = "user"  # Default to user

                content = msg.content if hasattr(msg, "content") else str(msg)

                ollama_messages.append({"role": role, "content": content})
            elif isinstance(msg, dict):
                # Handle dict messages directly
                ollama_messages.append(
                    {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                )

        # print(f"🔄 Converted {len(messages)} messages to {len(ollama_messages)} Ollama messages")

        # Prepare Ollama API payload
        payload = {
            "model": model.replace("ollama/", ""),  # Remove ollama/ prefix
            "messages": ollama_messages,
            "stream": False,
            "temperature": temperature,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        # Validate tools payload to prevent malformed JSON errors
        if tools:
            try:
                # Test if tools can be serialized to JSON properly
                json.dumps(payload)
            except (TypeError, ValueError) as e:
                print(f"🔧 Invalid payload JSON, attempting to clean: {e}")
                # Try to clean up any malformed JSON in tool definitions
                if "tools" in payload and isinstance(payload["tools"], (list, tuple)):
                    cleaned_tools = []
                    for tool in payload["tools"]:
                        try:
                            # Test if this tool can be serialized
                            json.dumps(tool)
                            cleaned_tools.append(tool)
                        except Exception:
                            print(f"⚠️ Skipping malformed tool: {tool}")
                    payload["tools"] = cleaned_tools

        # Determine timeout - use provided timeout, config timeout, or default to 120 seconds
        if timeout is None:
            try:
                from seek.common.config import get_active_seek_config

                config = get_active_seek_config()
                timeout = config.get("timeout_seconds", 120)
                print(f"🕐 Using config timeout: {timeout}s")
            except Exception:
                timeout = 120  # Default to 2 minutes for large models
                print(f"🕐 Using default timeout: {timeout}s (config unavailable)")

        # Make direct call to Ollama with retry logic
        # print(f"📞 Making direct Ollama call with payload: {json.dumps(payload, indent=2)}")
        print(f"🔄 Making direct Ollama call with {timeout}s timeout...")

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"🔄 Retry attempt {attempt}/{max_retries} for Ollama call...")

                response = requests.post(
                    "http://localhost:11434/api/chat", json=payload, timeout=timeout
                )
                break  # Success, exit retry loop

            except requests.exceptions.Timeout as _e:
                print(
                    f"⏰ Ollama call timed out after {timeout}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                if attempt < max_retries:
                    # On timeout, try with a simplified payload for retry
                    if tools and len(tools) > 1:
                        print("🔧 Simplifying payload for retry - reducing tool complexity")
                        # Keep only the first tool to reduce complexity
                        payload["tools"] = tools[:1]
                    continue
                else:
                    # Final timeout - return a graceful error
                    print(f"❌ Final timeout after {max_retries + 1} attempts")
                    return AIMessage(
                        content=f"Request timed out after {timeout}s. The model may be busy or the request too complex. Please try again with a simpler query."
                    )

            except requests.exceptions.JSONDecodeError as e:
                print(f"❌ JSON encoding error in request: {e}")
                # Fallback with minimal payload
                minimal_payload = {
                    "model": model.replace("ollama/", ""),
                    "messages": ollama_messages,
                    "stream": False,
                    "temperature": temperature,
                }
                try:
                    response = requests.post(
                        "http://localhost:11434/api/chat",
                        json=minimal_payload,
                        timeout=timeout,
                    )
                    break  # Success with minimal payload
                except requests.exceptions.Timeout:
                    print(
                        f"⏰ Minimal payload also timed out (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    if attempt >= max_retries:
                        return AIMessage(
                            content="Request failed due to timeout and JSON encoding issues. Please try again."
                        )
                    continue

            except requests.exceptions.ConnectionError as e:
                print(f"🔌 Connection error to Ollama: {e}")
                if attempt >= max_retries:
                    return AIMessage(
                        content="Could not connect to Ollama. Please ensure Ollama is running on localhost:11434."
                    )
                print("⏳ Waiting 2 seconds before retry...")
                import time

                time.sleep(2)
                continue

        if response.status_code == 200:
            result = response.json()
            # print(f"📥 Ollama response: {json.dumps(result, indent=2)[:500]}...")

            # Extract message content and tool calls
            message_data = result.get("message", {})
            content = message_data.get("content", "")
            tool_calls = message_data.get("tool_calls", [])

            # DEBUG: Show what we actually extracted
            print("🔍 DEBUG: Direct Ollama extracted:")
            print(f"  - content length: {len(content) if content else 0}")
            print(
                f"  - content preview: '{content[:200] if content else '[EMPTY]'}{'...' if content and len(content) > 200 else ''}"
            )
            print(f"  - tool_calls: {len(tool_calls) if tool_calls else 0} calls")

            # Create AIMessage with proper tool calls
            if tool_calls:
                # Convert Ollama tool calls to LangChain format
                formatted_tool_calls = []
                for i, tc in enumerate(tool_calls):
                    if isinstance(tc, dict) and "function" in tc:
                        function = tc["function"]

                        # Get and fix arguments if they're malformed JSON
                        args = function.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                # Try to parse as JSON first
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                # If that fails, try to fix common malformations
                                print(f"🔧 Attempting to fix malformed JSON: {args[:100]}...")
                                fixed_args = _fix_malformed_json_arguments(args)
                                try:
                                    args = json.loads(fixed_args)
                                    print("✅ Successfully fixed malformed JSON")
                                except json.JSONDecodeError as e:
                                    print(f"❌ Could not fix malformed JSON: {e}")
                                    # Fallback to empty dict
                                    args = {}

                        formatted_tc = {
                            "name": function.get("name", ""),
                            "args": args,
                            "id": f"call_{i}",
                        }
                        formatted_tool_calls.append(formatted_tc)
                    else:
                        print(f"⚠️ Unexpected tool call format: {tc}")

                # Create AIMessage with tool calls
                ai_message = AIMessage(content=content, tool_calls=formatted_tool_calls)
            else:
                # Create regular AIMessage
                ai_message = AIMessage(content=content)

            return ai_message

        else:
            print(f"❌ Direct Ollama call failed: {response.status_code} - {response.text}")
            return AIMessage(content="Direct Ollama call failed. Please try again.")

    except Exception as e:
        print(f"❌ Direct Ollama call error: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return AIMessage(content=f"Direct Ollama call failed with error: {str(e)}")


# Apply the monkeypatch for the prompt template bug to ALL locations
# Patch all known locations where ollama_pt might be called

# 1. Main factory location (where the error occurs)
litellm.litellm_core_utils.prompt_templates.factory.ollama_pt = _patched_ollama_pt  # type: ignore
_patch_log("✅ Patched litellm.litellm_core_utils.prompt_templates.factory.ollama_pt")

# 2. Transformation module location
litellm.llms.ollama.completion.transformation.ollama_pt = _patched_ollama_pt  # type: ignore
_patch_log("✅ Patched litellm.llms.ollama.completion.transformation.ollama_pt")

# 3. Main litellm module
try:
    import litellm.main

    if hasattr(litellm.main, "ollama_pt"):
        litellm.main.ollama_pt = _patched_ollama_pt  # type: ignore
        _patch_log("✅ Patched litellm.main.ollama_pt")
except Exception as e:
    _patch_log(f"⚠️  Could not patch litellm.main.ollama_pt: {e}")

# 4. Root litellm module
try:
    if hasattr(litellm, "ollama_pt"):
        litellm.ollama_pt = _patched_ollama_pt  # type: ignore
        _patch_log("✅ Patched litellm.ollama_pt")
except Exception as e:
    _patch_log(f"⚠️  Could not patch litellm.ollama_pt: {e}")

# 5. More aggressive patching - ensure all imported modules use our version
try:
    import sys

    # Patch all modules that might have imported ollama_pt
    for module_name, module in sys.modules.items():
        if module and hasattr(module, "ollama_pt"):
            module.ollama_pt = _patched_ollama_pt  # type: ignore
            _patch_log(f"✅ Patched {module_name}.ollama_pt")

    # Special handling for direct function references that might have been imported
    # Force reload of critical modules with our patch in place
    if "litellm.litellm_core_utils.prompt_templates.factory" in sys.modules:
        factory_module = sys.modules["litellm.litellm_core_utils.prompt_templates.factory"]
        factory_module.ollama_pt = _patched_ollama_pt  # type: ignore
        _patch_log("✅ Force-patched factory module")

except Exception as e:
    _patch_log(f"⚠️  Could not apply comprehensive module patch: {e}")

# 6. Install an import hook to catch future imports of ollama_pt
try:
    import sys

    class OllamaPtImportHook:
        """Import hook that patches ollama_pt in modules as they're loaded."""

        def find_spec(self, fullname: str, path: Any, target: Any | None = None) -> Any:
            return None  # We don't want to handle loading

        def find_module(self, fullname: str, path: Any | None = None) -> Any:
            return None  # We don't want to handle loading

    # Install our hook
    ollama_hook = OllamaPtImportHook()
    if ollama_hook not in sys.meta_path:
        sys.meta_path.insert(0, ollama_hook)

    # Monkey patch the import mechanism to catch ollama_pt imports
    original_import = __builtins__.__import__

    def patched_import(
        name: str,
        globals: Any | None = None,
        locals: Any | None = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        module = original_import(name, globals, locals, fromlist, level)

        # If this module has ollama_pt, patch it
        if hasattr(module, "ollama_pt"):
            module.ollama_pt = _patched_ollama_pt  # type: ignore
            _patch_log(f"✅ Late-patched {name}.ollama_pt during import")

        # Also check submodules if fromlist is specified
        if fromlist:
            for attr in fromlist:
                try:
                    submodule = getattr(module, attr, None)
                    if submodule and hasattr(submodule, "ollama_pt"):
                        submodule.ollama_pt = _patched_ollama_pt
                        _patch_log(f"✅ Late-patched {name}.{attr}.ollama_pt during import")
                except Exception as _e:
                    # Ignore attribute access errors during late patching
                    pass  # nosec B110 # - late import patch is best-effort only

        return module

    __builtins__["__import__"] = patched_import
    _patch_log("✅ Installed import hook for ollama_pt patching")

except Exception as e:
    _patch_log(f"⚠️  Could not install import hook: {e}")

# More comprehensive patch to handle tool calls in Ollama responses
try:
    # First, let's patch the main LiteLLM completion function to preserve tool calls
    original_litellm_completion = litellm.completion

    def patched_litellm_completion(*args: Any, **kwargs: Any) -> Any:
        """Patched LiteLLM completion that properly handles Ollama tool calls."""

        # Store original kwargs for debugging
        model = kwargs.get("model", args[0] if args else "")

        try:
            # Call the original completion function
            response = original_litellm_completion(*args, **kwargs)

            # Only process Ollama responses
            if "ollama" in str(model).lower():
                # Check if we have tool calls that might have been lost
                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    message = choice.message

                    # If we have empty content but tools were provided, this might be a tool calling scenario
                    has_tools = kwargs.get("tools") or kwargs.get("functions")

                    # Check if message has tool_calls attribute and if it's empty
                    has_tool_calls_attr = hasattr(message, "tool_calls")
                    tool_calls_empty = not has_tool_calls_attr or not message.tool_calls

                    # Debug info (reduced verbosity)
                    # print(f"🔍 Checking Ollama response: has_tools={bool(has_tools)}, content={bool(message.content)}, has_tool_calls_attr={has_tool_calls_attr}, tool_calls_empty={tool_calls_empty}")

                    # Check for lost tool calls OR lost content (both indicate LiteLLM parsing issues)
                    content_missing = not message.content

                    # Detect suspicious responses:
                    # 1. When tools are expected but missing OR content is missing with tools
                    # 2. When content is missing even without tools (final iteration case)
                    suspicious_response = (
                        has_tools and (tool_calls_empty or content_missing)
                    ) or (  # Original condition
                        content_missing
                    )  # Also trigger if content is missing, regardless of tools

                    if suspicious_response:
                        # This looks like Ollama returned content/tool calls but LiteLLM didn't preserve them
                        # Let's make a direct call to get the proper response
                        print(
                            f"🔧 LiteLLM lost tool calls for {model}, making direct Ollama call..."
                        )

                        # Use our direct Ollama function
                        messages = kwargs.get("messages", [])
                        tools = kwargs.get("tools", [])
                        temperature = kwargs.get("temperature", 0.1)

                        # Get timeout from config for consistency
                        try:
                            from seek.common.config import get_active_seek_config

                            config = get_active_seek_config()
                            timeout = config.get("timeout_seconds", 120)
                        except Exception:
                            timeout = 120  # Default timeout

                        direct_result = _direct_ollama_call_with_tools(
                            messages=messages,
                            model=model,
                            tools=tools,
                            temperature=temperature,
                            timeout=timeout,
                        )

                        # Always preserve content from direct call, regardless of tool calls
                        message.content = direct_result.content or ""

                        if hasattr(direct_result, "tool_calls") and direct_result.tool_calls:
                            # Convert the direct result back to LiteLLM format

                            # DEBUG: Show what tool calls we're trying to restore
                            print(
                                f"🔧 DEBUG: Attempting to restore {len(direct_result.tool_calls)} tool calls:"
                            )
                            for i, tc in enumerate(direct_result.tool_calls):
                                print(
                                    f"  [{i}] name='{tc.get('name', 'MISSING')}', args='{tc.get('args', 'MISSING')}', id='{tc.get('id', 'MISSING')}'"
                                )

                            # Create proper LiteLLM tool calls format, but validate them first
                            litellm_tool_calls: list[dict[str, Any]] = []
                            valid_tool_calls = []
                            invalid_content_parts = []

                            for tc in direct_result.tool_calls:
                                tool_name = tc.get("name", "")
                                tool_args = tc.get("args", {})

                                # Validate tool call - must have non-empty name and valid args
                                if tool_name and tool_name.strip():
                                    litellm_tc = {
                                        "id": tc.get("id", f"call_{len(litellm_tool_calls)}"),
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": (
                                                json.dumps(tool_args)
                                                if isinstance(tool_args, dict)
                                                else str(tool_args)
                                            ),
                                        },
                                    }
                                    litellm_tool_calls.append(litellm_tc)
                                    valid_tool_calls.append(tc)
                                    print(f"  ✅ Valid tool call: {tool_name}")
                                else:
                                    # Invalid tool call - treat as content
                                    invalid_part = f"<Attempted tool call: name='{tool_name}', args='{tool_args}'>"
                                    invalid_content_parts.append(invalid_part)
                                    print(
                                        "  ❌ Invalid tool call (empty name): treating as content"
                                    )

                            if litellm_tool_calls:
                                # Update the response with valid tool calls
                                message.tool_calls = litellm_tool_calls
                                print(
                                    f"✅ Restored {len(litellm_tool_calls)} valid tool calls to LiteLLM response"
                                )

                            # If we had invalid tool calls, append them to content
                            if invalid_content_parts:
                                current_content = message.content or ""
                                additional_content = "\n".join(invalid_content_parts)
                                message.content = (
                                    f"{current_content}\n{additional_content}"
                                    if current_content
                                    else additional_content
                                )
                                print(
                                    f"📝 Added {len(invalid_content_parts)} invalid tool calls as content"
                                )
                        else:
                            print("✅ Restored content from direct Ollama call (no tool calls)")

            return response

        except Exception as e:
            print(f"⚠️  LiteLLM completion patch error: {e}")
            # Fall back to original
            return original_litellm_completion(*args, **kwargs)

    # Apply the patch
    litellm.completion = patched_litellm_completion
    _patch_log("✅ Applied comprehensive LiteLLM patch for Ollama tool calls")

except Exception as e:
    _patch_log(f"⚠️  Could not apply LiteLLM completion patch: {e}")
    import traceback

    _patch_log(f"Traceback: {traceback.format_exc()}")
