"""
streaming/engine.py
====================
Converts provider-specific streaming responses into strict Anthropic SSE events.

Anthropic SSE event sequence (MANDATORY ORDER):
  1. message_start
  2. content_block_start  (index 0)
  3. ping                 (optional, but Anthropic emits it)
  4. content_block_delta  (repeated, one per token/chunk)
  5. content_block_stop   (index 0)
  6. message_delta        (stop_reason + usage)
  7. message_stop

For tool_use blocks the same skeleton applies with different block types.

Two conversion paths
--------------------
* Ollama   : already emits Anthropic SSE — forward events verbatim after
             light validation.
* NVIDIA   : emits OpenAI-style SSE ("data: {...choices[0].delta...}")
             — convert each chunk to Anthropic SSE.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: Any) -> str:
    """Format a single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _ping() -> str:
    return "event: ping\ndata: {\"type\": \"ping\"}\n\n"


# ---------------------------------------------------------------------------
# Anthropic-native stream (Ollama)
# ---------------------------------------------------------------------------

async def stream_anthropic_passthrough(
    raw_stream: AsyncIterator[bytes],
    requested_model: str,
    request_id: str,
) -> AsyncIterator[str]:
    """
    Forward an already-Anthropic SSE stream (Ollama) verbatim.

    We lightly validate each event and patch ``model`` / ``id`` fields so
    the echoed model name is always what the client sent, not the internal
    provider model.
    """
    message_stop_sent = False
    # Track whether we have sent message_start so we can inject one if missing
    message_start_sent = False
    buffer = ""

    async for chunk in raw_stream:
        if not chunk:
            continue
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            event_type: Optional[str] = None
            data_str: Optional[str] = None
            for line in frame.splitlines():
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_str = line[len("data:"):].strip()

            if not event_type or not data_str:
                continue

            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                logger.warning("stream_passthrough: bad JSON in frame, skipping")
                continue

            # Patch model / id in message_start
            if event_type == "message_start":
                msg = payload.get("message", {})
                msg["model"] = requested_model
                # Use first 24 hex chars of the UUID (always 32 hex chars after removing hyphens)
                msg.setdefault("id", f"msg_{request_id.replace('-', '')[:24]}")
                payload["message"] = msg
                message_start_sent = True

            if event_type == "message_stop":
                message_stop_sent = True

            yield _sse(event_type, payload)

    # Guarantee message_stop is always the last event
    if not message_stop_sent:
        yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# OpenAI → Anthropic stream conversion (NVIDIA)
# ---------------------------------------------------------------------------

async def stream_openai_to_anthropic(
    raw_stream: AsyncIterator[bytes],
    requested_model: str,
    request_id: str,
    estimated_input_tokens: int = 0,
) -> AsyncIterator[str]:
    """
    Convert an OpenAI-compatible SSE stream (NVIDIA) into Anthropic SSE events.

    OpenAI chunk format:
      data: {"id": "...", "choices": [{"delta": {"role"?: "assistant", "content"?: "..."}, "finish_reason": null|"stop"}], ...}
    """
    # Use first 24 hex chars of the UUID (always 32 hex chars after removing hyphens)
    msg_id = f"msg_{request_id.replace('-', '')[:24]}"
    block_index = 0
    output_tokens = 0
    stop_reason = "end_turn"
    finish_reason_seen = False
    buffer = ""

    # --- 1. message_start --------------------------------------------------
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": requested_model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": estimated_input_tokens, "output_tokens": 0},
        },
    })

    # --- 2. content_block_start (text) ------------------------------------
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": block_index,
        "content_block": {"type": "text", "text": ""},
    })

    yield _ping()

    # --- 3. content_block_delta events ------------------------------------
    async for raw_chunk in raw_stream:
        if not raw_chunk:
            continue
        buffer += raw_chunk.decode("utf-8", errors="replace")

        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            for line in frame.splitlines():
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    finish_reason_seen = True
                    continue
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("stream_nvidia: bad JSON chunk, skipping")
                    continue

                choices: List[Dict[str, Any]] = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                fr = choice.get("finish_reason")

                if fr and fr not in (None, "null"):
                    # Map OpenAI finish_reason → Anthropic stop_reason
                    _map = {
                        "stop": "end_turn",
                        "length": "max_tokens",
                        "tool_calls": "tool_use",
                        "function_call": "tool_use",
                    }
                    stop_reason = _map.get(fr, "end_turn")
                    finish_reason_seen = True

                # Extract text content
                text = delta.get("content") or ""
                if text:
                    output_tokens += max(1, len(text) // 4)
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })

                # Handle tool_calls in delta (NVIDIA may return these)
                tool_calls: List[Dict] = delta.get("tool_calls") or []
                for tc in tool_calls:
                    # Emit tool call fragments as text (NVIDIA tool safety rule)
                    fn = tc.get("function", {})
                    tc_text = fn.get("arguments") or fn.get("name") or ""
                    if tc_text:
                        output_tokens += max(1, len(tc_text) // 4)
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "text_delta", "text": tc_text},
                        })

    # --- 4. content_block_stop --------------------------------------------
    yield _sse("content_block_stop", {
        "type": "content_block_stop",
        "index": block_index,
    })

    # --- 5. message_delta -------------------------------------------------
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": None,
        },
        "usage": {"output_tokens": output_tokens},
    })

    # --- 6. message_stop --------------------------------------------------
    yield _sse("message_stop", {"type": "message_stop"})
