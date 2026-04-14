"""
utils/models.py
================
Model-name mapping layer.

Claude Code sends a single ``model`` string (e.g. "claude-3-5-sonnet-20241022").
The router must map that name to a provider-specific model identifier
*without* exposing the provider name externally.

Rules
-----
* Mapping is deterministic (same input → same output).
* Configured via environment variables; falls back to defaults.
* Provider model names are NEVER returned to the caller — the original
  requested model string is always echoed back in the response.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Default model mappings: <incoming_model_prefix> → (ollama_model, nvidia_model)
# ---------------------------------------------------------------------------
# The key is matched as a prefix of the incoming model name (case-insensitive).
# Order matters: first match wins.

DEFAULT_MAPPINGS: Dict[str, Tuple[str, str]] = {
    # Any Claude model → configured defaults
    "claude": (
        os.getenv("OLLAMA_MODEL", "llama3.2"),
        os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
    ),
    # Explicit pass-through for common aliases used in testing
    "ollama": (
        os.getenv("OLLAMA_MODEL", "llama3.2"),
        os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
    ),
    "llama": (
        os.getenv("OLLAMA_MODEL", "llama3.2"),
        os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
    ),
}


def get_provider_model(requested_model: str, provider: str) -> str:
    """
    Return the provider-specific model name for *requested_model*.

    Parameters
    ----------
    requested_model:
        The model name as sent by Claude Code.
    provider:
        ``"ollama"`` or ``"nvidia"``.

    Returns
    -------
    str
        Provider-specific model identifier. Falls back to the configured
        environment variable default if no prefix matches.
    """
    lower = requested_model.lower()
    for prefix, (ollama_model, nvidia_model) in DEFAULT_MAPPINGS.items():
        if lower.startswith(prefix):
            return ollama_model if provider == "ollama" else nvidia_model

    # No mapping found — use the environment defaults
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3.2")
    return os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
