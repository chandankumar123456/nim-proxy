"""
router/routing.py
==================
Deterministic routing-decision logic.

Rules (in priority order)
--------------------------
1. If a specific provider has been requested (via state / retry), honour it.
2. If total prompt length >= LONG_PROMPT_THRESHOLD → NVIDIA.
3. If number of messages >= MANY_MESSAGES_THRESHOLD → NVIDIA.
4. If request contains keywords associated with deep reasoning → NVIDIA.
5. Default → Ollama.

Same input always produces the same decision — no randomness.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from config import config

logger = logging.getLogger(__name__)

# Words/phrases that hint at complex reasoning tasks better handled by NVIDIA
_REASONING_KEYWORDS = re.compile(
    r"\b("
    r"analys[ei]s|analyze|analyse|"
    r"architect|design|plan|planning|"
    r"debug|diagnos[ei]s|investigate|"
    r"explain in detail|step.by.step|"
    r"complex|complicated|"
    r"refactor large|rewrite entire|"
    r"comprehensive|exhaustive|"
    r"compare and contrast|"
    r"pros and cons"
    r")\b",
    re.IGNORECASE,
)


def _total_prompt_chars(messages: List[Dict[str, Any]], system: Optional[Any]) -> int:
    """Sum of character lengths across all message contents + system prompt."""
    total = 0
    if system:
        if isinstance(system, str):
            total += len(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict):
                    total += len(block.get("text", ""))
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(block.get("text", "") or "")
    return total


def _has_reasoning_keywords(messages: List[Dict[str, Any]], system: Optional[Any]) -> bool:
    """Return True if any message or system prompt contains reasoning-heavy phrases."""
    texts: List[str] = []
    if isinstance(system, str):
        texts.append(system)
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
    combined = " ".join(texts)
    return bool(_REASONING_KEYWORDS.search(combined))


def decide_provider(
    messages: List[Dict[str, Any]],
    system: Optional[Any],
    providers_already_used: List[str],
) -> str:
    """
    Return ``"ollama"`` or ``"nvidia"``.

    Parameters
    ----------
    messages:
        The messages array from the Anthropic request.
    system:
        The system prompt (string or list of blocks), or None.
    providers_already_used:
        Providers already tried in this request cycle (from RequestState).
    """
    # If we already tried a provider, use the other one
    if "ollama" in providers_already_used and "nvidia" not in providers_already_used:
        logger.debug("Routing: Ollama already tried → NVIDIA fallback")
        return "nvidia"
    if "nvidia" in providers_already_used and "ollama" not in providers_already_used:
        logger.debug("Routing: NVIDIA already tried → Ollama fallback")
        return "ollama"

    # Fresh request — apply threshold rules
    total_chars = _total_prompt_chars(messages, system)
    msg_count = len(messages)

    if total_chars >= config.router.long_prompt_threshold:
        logger.debug(
            "Routing: long prompt (%d chars >= threshold %d) → NVIDIA",
            total_chars,
            config.router.long_prompt_threshold,
        )
        return "nvidia"

    if msg_count >= config.router.many_messages_threshold:
        logger.debug(
            "Routing: many messages (%d >= threshold %d) → NVIDIA",
            msg_count,
            config.router.many_messages_threshold,
        )
        return "nvidia"

    if _has_reasoning_keywords(messages, system):
        logger.debug("Routing: reasoning keywords detected → NVIDIA")
        return "nvidia"

    logger.debug("Routing: short/simple request → Ollama (default)")
    return "ollama"
