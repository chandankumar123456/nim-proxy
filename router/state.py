"""
router/state.py
================
Per-request state.  Isolated, never shared across requests.

Fields
------
request_id      : UUID4 string, unique per request, used for logging/tracing.
retry_count     : Number of retries attempted so far (max = config.router.max_retries).
providers_used  : Ordered list of providers already tried in this request cycle.
current_provider: The provider currently in use ("ollama" | "nvidia").
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RequestState:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retry_count: int = 0
    providers_used: List[str] = field(default_factory=list)
    current_provider: str = "ollama"
    failure_reasons: List[str] = field(default_factory=list)

    # --- helpers -----------------------------------------------------------

    def mark_provider_used(self, provider: str) -> None:
        self.current_provider = provider
        if provider not in self.providers_used:
            self.providers_used.append(provider)

    def can_retry(self, max_retries: int) -> bool:
        """True if we have not exhausted retry budget AND not tried all providers."""
        return self.retry_count < max_retries and len(self.providers_used) < 2

    def next_provider(self) -> Optional[str]:
        """
        Return the provider to use for the next attempt, or None if exhausted.
        Priority: ollama → nvidia → ollama (final fallback handled by core).
        """
        if "ollama" not in self.providers_used:
            return "ollama"
        if "nvidia" not in self.providers_used:
            return "nvidia"
        return None

    def record_failure(self, reason: str) -> None:
        self.failure_reasons.append(reason)
        self.retry_count += 1
