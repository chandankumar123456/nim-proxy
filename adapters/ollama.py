"""
adapters/ollama.py
===================
Ollama adapter — Primary provider.

Ollama exposes an Anthropic-compatible endpoint at ``/v1/messages`` so
request/response conversion is minimal.  The main tasks are:

* Replace the client-facing model name with the Ollama-specific model.
* Enforce timeout settings.
* Handle Ollama-specific error codes.
* For streaming: forward raw SSE bytes to the streaming engine.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict

import httpx

from adapters.base import AdapterError, BaseAdapter, RateLimitError, TimeoutError as AdapterTimeout
from config import config
from utils.models import get_provider_model
from utils.tokens import estimate_request_tokens, truncate_messages

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseAdapter):
    """Sends requests to Ollama's Anthropic-compatible /v1/messages endpoint."""

    provider_name = "ollama"

    def _build_request(self, body: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
        """Prepare the Ollama request body."""
        ollama_model = get_provider_model(requested_model, "ollama")
        messages = body.get("messages", [])
        system = body.get("system")

        # Truncate messages if they exceed Ollama's context window
        max_output = body.get("max_tokens", 1024)
        try:
            messages = truncate_messages(
                messages, system,
                max_tokens=config.ollama.max_context_tokens,
                max_output_tokens=max_output,
            )
        except ValueError as exc:
            raise AdapterError(str(exc), status_code=400, retryable=False) from exc

        req = {
            "model": ollama_model,
            "max_tokens": max_output,
            "messages": messages,
        }

        # Optional fields — only include if present in original request
        for field_name in ("system", "temperature", "top_p", "stop_sequences", "tools", "tool_choice", "stream"):
            if field_name in body and body[field_name] is not None:
                req[field_name] = body[field_name]

        return req

    def _httpx_timeouts(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=config.ollama.timeout_connect,
            read=config.ollama.timeout_response,
            write=config.ollama.timeout_response,
            pool=config.ollama.timeout_total,
        )

    def _handle_http_error(self, resp: httpx.Response) -> None:
        if resp.status_code == 429:
            raise RateLimitError("Ollama rate limit exceeded")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            retryable = resp.status_code >= 500
            raise AdapterError(
                f"Ollama HTTP {resp.status_code}: {detail}",
                status_code=resp.status_code,
                retryable=retryable,
            )

    async def send(self, request_body: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
        req = self._build_request(request_body, requested_model)
        req["stream"] = False
        url = f"{config.ollama.base_url.rstrip('/')}/v1/messages"

        logger.info("OllamaAdapter.send → %s model=%s", url, req["model"])

        try:
            async with httpx.AsyncClient(timeout=self._httpx_timeouts()) as client:
                resp = await client.post(url, json=req, headers={"Content-Type": "application/json"})
        except httpx.TimeoutException as exc:
            raise AdapterTimeout(f"Ollama timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise AdapterError(f"Ollama connection error: {exc}", retryable=True) from exc

        self._handle_http_error(resp)

        try:
            return resp.json()
        except Exception as exc:
            raise AdapterError(f"Ollama returned invalid JSON: {exc}", retryable=True) from exc

    async def send_stream(
        self, request_body: Dict[str, Any], requested_model: str
    ) -> AsyncIterator[bytes]:
        req = self._build_request(request_body, requested_model)
        req["stream"] = True
        url = f"{config.ollama.base_url.rstrip('/')}/v1/messages"

        logger.info("OllamaAdapter.send_stream → %s model=%s", url, req["model"])

        try:
            async with httpx.AsyncClient(timeout=self._httpx_timeouts()) as client:
                async with client.stream("POST", url, json=req, headers={"Content-Type": "application/json"}) as resp:
                    self._handle_http_error(resp)
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
        except httpx.TimeoutException as exc:
            raise AdapterTimeout(f"Ollama stream timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise AdapterError(f"Ollama stream connection error: {exc}", retryable=True) from exc
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(f"Ollama stream unexpected error: {exc}", retryable=True) from exc
