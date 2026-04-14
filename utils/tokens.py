"""
utils/tokens.py
================
Token-counting and context-truncation utilities.

Because we cannot run model-specific tokenisers at runtime without heavy
dependencies, we use character-based estimation (≈ 4 chars per token for
English text). This is "good enough" for routing decisions and guardrails.

Truncation strategy (per spec):
  1. Always preserve the system prompt.
  2. Always preserve the most-recent messages.
  3. Drop oldest messages first until within budget.
  4. Never produce an invalid message list.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # Conservative estimate


def estimate_tokens(text: str) -> int:
    """Estimate token count from a string."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def _content_to_str(content: Any) -> str:
    """Flatten message content (string or block list) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                # text block
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # tool_result block
                elif block.get("type") == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, str):
                        parts.append(inner)
                    elif isinstance(inner, list):
                        for ib in inner:
                            if isinstance(ib, dict) and ib.get("type") == "text":
                                parts.append(ib.get("text", ""))
                # tool_use: include function name + args
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool_use:{block.get('name','')}]")
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


def estimate_request_tokens(messages: List[Dict[str, Any]], system: Optional[Any] = None) -> int:
    """Estimate the total tokens for a list of messages + optional system."""
    total = 0
    if system:
        total += estimate_tokens(_content_to_str(system))
    for msg in messages:
        total += estimate_tokens(_content_to_str(msg.get("content", "")))
        total += 4  # role + overhead per message
    return total


def truncate_messages(
    messages: List[Dict[str, Any]],
    system: Optional[Any],
    max_tokens: int,
    max_output_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Truncate messages so that input_tokens + max_output_tokens <= max_tokens.

    Returns a (possibly shortened) copy of ``messages``. Never mutates the
    original list. Raises ``ValueError`` if even a single message cannot fit.
    """
    budget = max_tokens - max_output_tokens
    if budget <= 0:
        raise ValueError(f"max_tokens ({max_tokens}) too small for max_output_tokens ({max_output_tokens})")

    system_tokens = estimate_tokens(_content_to_str(system)) if system else 0
    if system_tokens >= budget:
        raise ValueError("System prompt alone exceeds token budget")

    available = budget - system_tokens

    # Work from newest → oldest, accumulate until we exceed the budget
    selected: List[Dict[str, Any]] = []
    used = 0
    for msg in reversed(messages):
        t = estimate_tokens(_content_to_str(msg.get("content", ""))) + 4
        if used + t > available:
            if not selected:
                # Even the newest single message doesn't fit — hard failure
                raise ValueError(
                    f"Even the newest message ({t} tokens) exceeds the available budget ({available} tokens)"
                )
            # Subsequent messages don't fit — stop here
            break
        selected.append(msg)
        used += t

    if not selected:
        raise ValueError("No messages fit within the token budget")

    # selected is newest-first; reverse to restore chronological order
    result = list(reversed(selected))
    if len(result) < len(messages):
        logger.warning(
            "Truncated messages from %d to %d to fit token budget (%d tokens available)",
            len(messages),
            len(result),
            available,
        )
    return result
