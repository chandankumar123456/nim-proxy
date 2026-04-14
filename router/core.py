"""
router/core.py
===============
RouterCore — The central orchestrator for every /v1/messages request.

Responsibilities
----------------
1. Validate the request schema.
2. Initialize per-request state (RequestState).
3. Decide the initial provider via routing.py.
4. Invoke the correct adapter (Ollama or NVIDIA).
5. Handle failures: retry with the alternate provider.
6. Apply the response normalizer.
7. Log every decision for traceability.
8. Enforce loop-prevention (max retries, both-providers-used guard).

Concurrency control lives in main.py (semaphore + queue).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

from adapters.base import AdapterError, RateLimitError, TimeoutError as AdapterTimeout
from adapters.nvidia import NvidiaAdapter
from adapters.ollama import OllamaAdapter
from config import config
from normalizer.response import normalize_response
from router.metrics import metrics
from router.routing import decide_provider
from router.state import RequestState
from streaming.engine import stream_anthropic_passthrough, stream_openai_to_anthropic
from utils.errors import api_error, overloaded_error
from utils.tokens import estimate_request_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "ollama": OllamaAdapter(),
    "nvidia": NvidiaAdapter(),
}


def _get_adapter(provider: str):
    adapter = _ADAPTERS.get(provider)
    if adapter is None:
        raise RuntimeError(f"Unknown provider: {provider!r}")
    return adapter


# ---------------------------------------------------------------------------
# Rate-limit cooldown state (per provider)
# ---------------------------------------------------------------------------

_rate_limit_until: Dict[str, float] = {}


def _is_rate_limited(provider: str) -> bool:
    until = _rate_limit_until.get(provider, 0.0)
    return time.monotonic() < until


def _set_rate_limited(provider: str, cooldown: float) -> None:
    _rate_limit_until[provider] = time.monotonic() + cooldown
    logger.warning("Provider %s rate-limited for %.0f seconds", provider, cooldown)


# ---------------------------------------------------------------------------
# RouterCore
# ---------------------------------------------------------------------------

class RouterCore:
    """Stateless per-call orchestrator (state is stored in RequestState)."""

    async def handle(
        self,
        request_body: Dict[str, Any],
        requested_model: str,
        stream: bool,
    ) -> Any:
        """
        Entry point for non-streaming requests.

        Returns a normalized Anthropic response dict, or an error dict.
        """
        state = RequestState()
        messages = request_body.get("messages", [])
        system = request_body.get("system")
        estimated_input = estimate_request_tokens(messages, system)

        await metrics.request_started()
        try:
            return await self._execute(state, request_body, requested_model, estimated_input)
        finally:
            await metrics.request_finished()

    async def handle_stream(
        self,
        request_body: Dict[str, Any],
        requested_model: str,
    ) -> AsyncIterator[str]:
        """
        Entry point for streaming requests.

        Yields SSE-formatted strings.  The FastAPI StreamingResponse wraps this.
        """
        state = RequestState()
        messages = request_body.get("messages", [])
        system = request_body.get("system")
        estimated_input = estimate_request_tokens(messages, system)

        await metrics.request_started()
        try:
            async for chunk in self._execute_stream(state, request_body, requested_model, estimated_input):
                yield chunk
        finally:
            await metrics.request_finished()

    # ------------------------------------------------------------------
    # Internal: non-streaming execution with retry logic
    # ------------------------------------------------------------------

    async def _execute(
        self,
        state: RequestState,
        body: Dict[str, Any],
        requested_model: str,
        estimated_input: int,
    ) -> Dict[str, Any]:
        """Attempt the request, retrying with the alternate provider on failure."""
        last_error: Optional[str] = None

        while True:
            # Guard: stop if all retry budget is exhausted
            if not state.can_retry(config.router.max_retries):
                logger.error(
                    "[%s] Exhausted all retries. providers_used=%s failures=%s",
                    state.request_id, state.providers_used, state.failure_reasons,
                )
                break

            # Choose provider
            provider = decide_provider(
                body.get("messages", []),
                body.get("system"),
                state.providers_used,
            )

            # Skip rate-limited providers and loop back to re-evaluate
            if _is_rate_limited(provider):
                logger.info(
                    "[%s] Provider %s is rate-limited, skipping",
                    state.request_id, provider,
                )
                state.mark_provider_used(provider)
                state.record_failure(f"{provider} rate-limited")
                last_error = f"{provider} rate-limited"
                await metrics.record_retry()
                continue

            state.mark_provider_used(provider)
            await metrics.record_provider_use(provider)

            logger.info(
                "[%s] Attempt %d — provider=%s",
                state.request_id, state.retry_count + 1, provider,
            )

            try:
                adapter = _get_adapter(provider)
                raw = await adapter.send(body, requested_model)

                # Validate response is not empty
                content = raw.get("content")
                if not content or (isinstance(content, list) and not any(
                    b.get("text") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )):
                    logger.warning("[%s] Empty response from %s, retrying", state.request_id, provider)
                    state.record_failure(f"{provider} returned empty content")
                    await metrics.record_retry()
                    if not state.can_retry(config.router.max_retries):
                        # Return the (possibly empty) normalised response
                        return normalize_response(raw, requested_model, estimated_input)
                    continue

                result = normalize_response(raw, requested_model, estimated_input)
                logger.info("[%s] Success via %s", state.request_id, provider)
                return result

            except RateLimitError as exc:
                logger.warning("[%s] Rate limit on %s: %s", state.request_id, provider, exc)
                if provider == "nvidia":
                    _set_rate_limited("nvidia", config.nvidia.rate_limit_cooldown)
                state.record_failure(str(exc))
                last_error = str(exc)
                await metrics.record_retry()

            except (AdapterTimeout, AdapterError) as exc:
                logger.warning("[%s] Error on %s: %s", state.request_id, provider, exc)
                state.record_failure(str(exc))
                last_error = str(exc)
                if not exc.retryable:
                    break
                await metrics.record_retry()

            except Exception as exc:
                logger.exception("[%s] Unexpected error on %s", state.request_id, provider)
                state.record_failure("internal error")
                last_error = "internal error"
                await metrics.record_retry()

        # All retries exhausted — return Anthropic-format error
        await metrics.record_failure()
        return api_error(f"All providers failed. Last error: {last_error or 'unknown'}")

    # ------------------------------------------------------------------
    # Internal: streaming execution with retry logic
    # ------------------------------------------------------------------

    async def _execute_stream(
        self,
        state: RequestState,
        body: Dict[str, Any],
        requested_model: str,
        estimated_input: int,
    ) -> AsyncIterator[str]:
        """Attempt the streaming request, retrying on failure."""
        last_error: Optional[str] = None

        while True:
            # Guard: stop if all retry budget is exhausted
            if not state.can_retry(config.router.max_retries):
                break

            provider = decide_provider(
                body.get("messages", []),
                body.get("system"),
                state.providers_used,
            )

            # Skip rate-limited providers and loop back to re-evaluate
            if _is_rate_limited(provider):
                logger.info("[%s] Stream: provider %s rate-limited, skipping", state.request_id, provider)
                state.mark_provider_used(provider)
                state.record_failure(f"{provider} rate-limited")
                last_error = f"{provider} rate-limited"
                await metrics.record_retry()
                continue

            state.mark_provider_used(provider)
            await metrics.record_provider_use(provider)

            logger.info("[%s] Stream attempt %d — provider=%s", state.request_id, state.retry_count + 1, provider)

            try:
                adapter = _get_adapter(provider)
                raw_stream = adapter.send_stream(body, requested_model)

                if provider == "ollama":
                    async for chunk in stream_anthropic_passthrough(
                        raw_stream, requested_model, state.request_id
                    ):
                        yield chunk
                else:
                    async for chunk in stream_openai_to_anthropic(
                        raw_stream, requested_model, state.request_id, estimated_input
                    ):
                        yield chunk

                logger.info("[%s] Stream complete via %s", state.request_id, provider)
                return  # Success

            except RateLimitError as exc:
                logger.warning("[%s] Stream rate limit on %s: %s", state.request_id, provider, exc)
                if provider == "nvidia":
                    _set_rate_limited("nvidia", config.nvidia.rate_limit_cooldown)
                state.record_failure(str(exc))
                last_error = str(exc)
                await metrics.record_retry()

            except (AdapterTimeout, AdapterError) as exc:
                logger.warning("[%s] Stream error on %s: %s", state.request_id, provider, exc)
                state.record_failure(str(exc))
                last_error = str(exc)
                if not exc.retryable:
                    break
                await metrics.record_retry()

            except Exception as exc:
                logger.exception("[%s] Stream unexpected error on %s", state.request_id, provider)
                state.record_failure("internal error")
                last_error = "internal error"
                await metrics.record_retry()

        # All retries exhausted — yield an Anthropic-format error as a final SSE event
        await metrics.record_failure()
        import json
        error_payload = api_error(f"All providers failed. Last error: {last_error or 'unknown'}")
        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
