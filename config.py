"""
config.py
=========
Central configuration loaded from environment variables (or a .env file).
All tunables live here so that no magic numbers appear elsewhere in the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env if present (python-dotenv is optional; skip silently if absent)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _str(name: str, default: str) -> str:
    return os.getenv(name, default)


# ---------------------------------------------------------------------------
# Provider: Ollama
# ---------------------------------------------------------------------------

@dataclass
class OllamaConfig:
    base_url: str = field(default_factory=lambda: _str("OLLAMA_BASE_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: _str("OLLAMA_MODEL", "llama3.2"))
    timeout_connect: float = field(default_factory=lambda: _float("OLLAMA_TIMEOUT_CONNECT", 10.0))
    timeout_response: float = field(default_factory=lambda: _float("OLLAMA_TIMEOUT_RESPONSE", 30.0))
    timeout_stream_idle: float = field(default_factory=lambda: _float("OLLAMA_TIMEOUT_STREAM_IDLE", 60.0))
    timeout_total: float = field(default_factory=lambda: _float("OLLAMA_TIMEOUT_TOTAL", 300.0))
    max_context_tokens: int = field(default_factory=lambda: _int("OLLAMA_MAX_TOKENS", 8192))


# ---------------------------------------------------------------------------
# Provider: NVIDIA NIM
# ---------------------------------------------------------------------------

@dataclass
class NvidiaConfig:
    api_key: str = field(default_factory=lambda: _str("NVIDIA_API_KEY", ""))
    base_url: str = field(default_factory=lambda: _str("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    model: str = field(default_factory=lambda: _str("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"))
    timeout_connect: float = field(default_factory=lambda: _float("NVIDIA_TIMEOUT_CONNECT", 10.0))
    timeout_response: float = field(default_factory=lambda: _float("NVIDIA_TIMEOUT_RESPONSE", 30.0))
    timeout_stream_idle: float = field(default_factory=lambda: _float("NVIDIA_TIMEOUT_STREAM_IDLE", 60.0))
    timeout_total: float = field(default_factory=lambda: _float("NVIDIA_TIMEOUT_TOTAL", 300.0))
    max_context_tokens: int = field(default_factory=lambda: _int("NVIDIA_MAX_TOKENS", 32768))
    rate_limit_cooldown: float = field(default_factory=lambda: _float("NVIDIA_RATE_LIMIT_COOLDOWN", 60.0))


# ---------------------------------------------------------------------------
# Router settings
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    host: str = field(default_factory=lambda: _str("ROUTER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("ROUTER_PORT", 8080))
    max_concurrent_requests: int = field(default_factory=lambda: _int("MAX_CONCURRENT_REQUESTS", 50))
    max_queue_size: int = field(default_factory=lambda: _int("MAX_QUEUE_SIZE", 100))
    max_retries: int = field(default_factory=lambda: _int("MAX_RETRIES", 2))
    # Routing thresholds
    long_prompt_threshold: int = field(default_factory=lambda: _int("LONG_PROMPT_THRESHOLD", 2000))
    many_messages_threshold: int = field(default_factory=lambda: _int("MANY_MESSAGES_THRESHOLD", 10))


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    nvidia: NvidiaConfig = field(default_factory=NvidiaConfig)
    router: RouterConfig = field(default_factory=RouterConfig)


# Singleton instance — imported everywhere
config = Config()
