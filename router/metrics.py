"""
router/metrics.py
==================
Minimal in-process observability counters.

All counters are process-local integers (no external dependencies).
Exposed via GET /metrics in the FastAPI app.

Counters
--------
total_requests        : All requests received since startup.
active_requests       : Currently in-flight requests.
provider_usage        : Dict[provider, count] — how often each was used.
failure_count         : Total requests that ended in an error.
retry_count           : Total retry attempts across all requests.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Metrics:
    total_requests: int = 0
    active_requests: int = 0
    failure_count: int = 0
    retry_count: int = 0
    provider_usage: Dict[str, int] = field(default_factory=lambda: {"ollama": 0, "nvidia": 0})
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    async def request_started(self) -> None:
        async with self._lock:
            self.total_requests += 1
            self.active_requests += 1

    async def request_finished(self) -> None:
        async with self._lock:
            self.active_requests = max(0, self.active_requests - 1)

    async def record_provider_use(self, provider: str) -> None:
        async with self._lock:
            self.provider_usage[provider] = self.provider_usage.get(provider, 0) + 1

    async def record_failure(self) -> None:
        async with self._lock:
            self.failure_count += 1

    async def record_retry(self) -> None:
        async with self._lock:
            self.retry_count += 1

    def snapshot(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "failure_count": self.failure_count,
            "retry_count": self.retry_count,
            "provider_usage": dict(self.provider_usage),
        }


# Singleton instance shared across the app
metrics = Metrics()
