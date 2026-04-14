"""
adapters/base.py
=================
Abstract base class that every provider adapter must implement.

Design contract
---------------
* ``send`` handles non-streaming requests.
* ``send_stream`` handles streaming requests; yields raw bytes from the
  provider so the streaming engine can convert them.
* Both methods raise ``AdapterError`` on failure — callers must not handle
  provider-specific exceptions.
* Adapters are **stateless** per call; no cross-request state is stored.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional


class AdapterError(Exception):
    """Raised when a provider adapter encounters an unrecoverable error."""

    def __init__(self, message: str, status_code: int = 500, retryable: bool = True) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class RateLimitError(AdapterError):
    """Provider returned HTTP 429 — rate-limited."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, status_code=429, retryable=False)


class TimeoutError(AdapterError):  # noqa: A001 — intentional shadow
    """Provider did not respond within the configured timeout."""

    def __init__(self, message: str = "Provider timed out") -> None:
        super().__init__(message, status_code=504, retryable=True)


class BaseAdapter(ABC):
    """Abstract provider adapter interface."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'ollama', 'nvidia')."""

    @abstractmethod
    async def send(
        self,
        request_body: Dict[str, Any],
        requested_model: str,
    ) -> Dict[str, Any]:
        """
        Send a non-streaming request.

        Parameters
        ----------
        request_body:
            The full validated Anthropic request dict.
        requested_model:
            Original model name from the client (for echoing in the response).

        Returns
        -------
        dict
            Provider response converted to Anthropic format (pre-normalizer).

        Raises
        ------
        AdapterError / subclass
        """

    @abstractmethod
    async def send_stream(
        self,
        request_body: Dict[str, Any],
        requested_model: str,
    ) -> AsyncIterator[bytes]:
        """
        Send a streaming request; yield raw response bytes from the provider.

        The streaming engine in ``streaming/engine.py`` handles SSE conversion.

        Raises
        ------
        AdapterError / subclass
        """
