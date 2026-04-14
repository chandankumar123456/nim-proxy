"""
adapters/nvidia.py
===================
NVIDIA NIM adapter — Fallback provider.

NVIDIA NIM exposes an OpenAI-compatible API.  This adapter handles:

  Anthropic request  →  OpenAI request  →  NVIDIA NIM
  NVIDIA NIM response  →  OpenAI response  →  Anthropic response

Tool call safety
----------------
Anthropic tool definitions cannot be reliably mapped to the NVIDIA API.
Per spec: if tools are present in the request when routed to NVIDIA,
we convert them to a plain-text representation and inject them into the
system prompt.  No structured tool schema is forwarded to avoid schema breaks.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from adapters.base import AdapterError, BaseAdapter, RateLimitError, TimeoutError as AdapterTimeout
from config import config
from utils.models import get_provider_model
from utils.tokens import estimate_request_tokens, truncate_messages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request conversion: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def _system_to_str(system: Any) -> str:
    """Flatten an Anthropic system prompt (string or list) to a plain string."""
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: List[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(system)


def _tools_to_system_text(tools: List[Dict[str, Any]]) -> str:
    """Convert Anthropic tool definitions to a plain-text instruction block."""
    lines = ["You have access to the following tools (respond in plain text when calling them):"]
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        schema = tool.get("input_schema", {})
        lines.append(f"\n- {name}: {desc}")
        if schema:
            lines.append(f"  Parameters: {json.dumps(schema, indent=2)}")
    return "\n".join(lines)


def _content_block_to_str(content: Any) -> str:
    """Flatten Anthropic content (string or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    parts.append(
                        f"[Tool call: {block.get('name', '')} with input: {json.dumps(block.get('input', {}))}]"
                    )
                elif btype == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, str):
                        parts.append(f"[Tool result: {inner}]")
                    elif isinstance(inner, list):
                        for ib in inner:
                            if isinstance(ib, dict) and ib.get("type") == "text":
                                parts.append(f"[Tool result: {ib.get('text', '')}]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _anthropic_to_openai_messages(
    messages: List[Dict[str, Any]],
    system_text: str,
) -> List[Dict[str, str]]:
    """Convert Anthropic messages array to OpenAI messages array."""
    openai_msgs: List[Dict[str, str]] = []

    if system_text:
        openai_msgs.append({"role": "system", "content": system_text})

    for msg in messages:
        role = msg.get("role", "user")
        content_str = _content_block_to_str(msg.get("content", ""))
        # Anthropic uses "user" / "assistant"; OpenAI also accepts these roles
        openai_msgs.append({"role": role, "content": content_str})

    return openai_msgs


def anthropic_to_openai_request(
    body: Dict[str, Any],
    requested_model: str,
    nvidia_model: str,
) -> Dict[str, Any]:
    """Build an OpenAI-format request dict from an Anthropic request."""
    system_parts: List[str] = []
    raw_system = body.get("system")
    if raw_system:
        system_parts.append(_system_to_str(raw_system))

    # Inline tools as text (tool call safety rule)
    tools = body.get("tools")
    if tools:
        system_parts.append(_tools_to_system_text(tools))

    system_text = "\n\n".join(system_parts)

    messages = body.get("messages", [])
    openai_messages = _anthropic_to_openai_messages(messages, system_text)

    req: Dict[str, Any] = {
        "model": nvidia_model,
        "messages": openai_messages,
        "max_tokens": body.get("max_tokens", 1024),
    }

    if body.get("temperature") is not None:
        req["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        req["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        req["stop"] = body["stop_sequences"]
    if body.get("stream"):
        req["stream"] = True

    return req


# ---------------------------------------------------------------------------
# Response conversion: OpenAI → Anthropic
# ---------------------------------------------------------------------------

def openai_to_anthropic_response(
    openai_resp: Dict[str, Any],
    requested_model: str,
) -> Dict[str, Any]:
    """Convert an OpenAI non-streaming response to Anthropic format (pre-normalizer)."""
    choices = openai_resp.get("choices", [])
    if not choices:
        raise AdapterError("NVIDIA returned no choices", retryable=True)

    choice = choices[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    # Map OpenAI finish_reason → Anthropic stop_reason
    _stop_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "end_turn",
    }
    stop_reason = _stop_map.get(finish_reason, "end_turn")

    content_text = message.get("content", "")
    content_blocks = [{"type": "text", "text": content_text or ""}]

    # Usage
    raw_usage = openai_resp.get("usage", {})
    usage = {
        "input_tokens": raw_usage.get("prompt_tokens", 0),
        "output_tokens": raw_usage.get("completion_tokens", 0),
    }

    return {
        "id": openai_resp.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": requested_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# NVIDIA Adapter
# ---------------------------------------------------------------------------

class NvidiaAdapter(BaseAdapter):
    """Sends requests to NVIDIA NIM (OpenAI-compatible endpoint)."""

    provider_name = "nvidia"

    def _httpx_timeouts(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=config.nvidia.timeout_connect,
            read=config.nvidia.timeout_response,
            write=config.nvidia.timeout_response,
            pool=config.nvidia.timeout_total,
        )

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {config.nvidia.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_http_error(self, resp: httpx.Response) -> None:
        if resp.status_code == 429:
            raise RateLimitError("NVIDIA rate limit exceeded (HTTP 429)")
        if resp.status_code >= 400:
            try:
                detail = resp.json()
                err_msg = detail.get("error", {}).get("message", resp.text) if isinstance(detail, dict) else resp.text
            except Exception:
                err_msg = resp.text
            retryable = resp.status_code >= 500
            raise AdapterError(
                f"NVIDIA HTTP {resp.status_code}: {err_msg}",
                status_code=resp.status_code,
                retryable=retryable,
            )

    def _build_request(self, body: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
        nvidia_model = get_provider_model(requested_model, "nvidia")
        messages = body.get("messages", [])
        system = body.get("system")

        # Truncate if necessary
        max_output = body.get("max_tokens", 1024)
        try:
            messages = truncate_messages(
                messages, system,
                max_tokens=config.nvidia.max_context_tokens,
                max_output_tokens=max_output,
            )
        except ValueError as exc:
            raise AdapterError(str(exc), status_code=400, retryable=False) from exc

        patched_body = dict(body)
        patched_body["messages"] = messages
        return anthropic_to_openai_request(patched_body, requested_model, nvidia_model)

    async def send(self, request_body: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
        req = self._build_request(request_body, requested_model)
        req.pop("stream", None)  # Ensure non-streaming
        url = f"{config.nvidia.base_url.rstrip('/')}/chat/completions"

        logger.info("NvidiaAdapter.send → %s model=%s", url, req.get("model"))

        try:
            async with httpx.AsyncClient(timeout=self._httpx_timeouts()) as client:
                resp = await client.post(url, json=req, headers=self._auth_headers())
        except httpx.TimeoutException as exc:
            raise AdapterTimeout(f"NVIDIA timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise AdapterError(f"NVIDIA connection error: {exc}", retryable=True) from exc

        self._handle_http_error(resp)

        try:
            openai_resp = resp.json()
        except Exception as exc:
            raise AdapterError(f"NVIDIA returned invalid JSON: {exc}", retryable=True) from exc

        return openai_to_anthropic_response(openai_resp, requested_model)

    async def send_stream(
        self, request_body: Dict[str, Any], requested_model: str
    ) -> AsyncIterator[bytes]:
        req = self._build_request(request_body, requested_model)
        req["stream"] = True
        url = f"{config.nvidia.base_url.rstrip('/')}/chat/completions"

        logger.info("NvidiaAdapter.send_stream → %s model=%s", url, req.get("model"))

        try:
            async with httpx.AsyncClient(timeout=self._httpx_timeouts()) as client:
                async with client.stream("POST", url, json=req, headers=self._auth_headers()) as resp:
                    self._handle_http_error(resp)
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
        except httpx.TimeoutException as exc:
            raise AdapterTimeout(f"NVIDIA stream timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise AdapterError(f"NVIDIA stream connection error: {exc}", retryable=True) from exc
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(f"NVIDIA stream unexpected error: {exc}", retryable=True) from exc
