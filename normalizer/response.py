"""
normalizer/response.py
=======================
Ensures every outbound response strictly matches the Anthropic Messages API
response schema.  Any provider-specific quirks are fixed here.

Non-streaming response schema enforced:
  {
    "id": str,
    "type": "message",
    "role": "assistant",
    "content": [...],           # list of content blocks, never empty
    "model": str,               # echoes the requested model name
    "stop_reason": str,         # always present
    "stop_sequence": null | str,
    "usage": {
      "input_tokens": int,
      "output_tokens": int
    }
  }

Valid stop_reason values: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Map provider finish_reason → Anthropic stop_reason
_FINISH_REASON_MAP: Dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
    "null": "end_turn",
}

VALID_STOP_REASONS = {"end_turn", "max_tokens", "stop_sequence", "tool_use"}


def _normalize_stop_reason(raw: Optional[str]) -> str:
    if not raw:
        return "end_turn"
    if raw in VALID_STOP_REASONS:
        return raw
    return _FINISH_REASON_MAP.get(raw, "end_turn")


def _ensure_content_blocks(content: Any) -> List[Dict[str, Any]]:
    """Convert any content representation to a list of valid content blocks."""
    if isinstance(content, list) and content:
        # Ensure each block has at least a type field
        normalized: List[Dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and "type" in block:
                normalized.append(block)
            elif isinstance(block, str):
                normalized.append({"type": "text", "text": block})
        return normalized if normalized else [{"type": "text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    # Fallback: empty text block (provider returned nothing useful)
    logger.warning("normalizer: content is empty or unrecognised type %s", type(content).__name__)
    return [{"type": "text", "text": ""}]


def _ensure_usage(usage: Any, input_tokens: int = 0, output_tokens: int = 0) -> Dict[str, int]:
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("input_tokens", input_tokens)),
            "output_tokens": int(usage.get("output_tokens", output_tokens)),
        }
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def normalize_response(
    raw: Dict[str, Any],
    requested_model: str,
    estimated_input_tokens: int = 0,
) -> Dict[str, Any]:
    """
    Accept a raw provider response dict and return a fully-compliant Anthropic
    response dict.

    Parameters
    ----------
    raw:
        The provider's response (may be Anthropic-native or already converted).
    requested_model:
        The model name the client originally sent — echoed back as ``model``.
    estimated_input_tokens:
        Fallback if the provider did not return usage.
    """
    msg_id = raw.get("id") or f"msg_{uuid.uuid4().hex[:24]}"
    content = _ensure_content_blocks(raw.get("content"))
    stop_reason = _normalize_stop_reason(raw.get("stop_reason") or raw.get("finish_reason"))

    # Estimate output tokens from content if usage is missing
    raw_usage = raw.get("usage") or {}
    estimated_output = sum(
        len(b.get("text", "")) // 4
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )
    usage = _ensure_usage(raw_usage, estimated_input_tokens, estimated_output)

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": requested_model,
        "stop_reason": stop_reason,
        "stop_sequence": raw.get("stop_sequence"),
        "usage": usage,
    }
