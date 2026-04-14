"""
main.py
========
FastAPI application — single entry point for the Anthropic-compatible LLM router.

Endpoints
---------
POST /v1/messages     — Anthropic Messages API (streaming + non-streaming)
GET  /health          — Liveness probe
GET  /metrics         — Observability counters
GET  /v1/models       — Minimal model listing (for Claude Code compatibility)

Concurrency controls
--------------------
* asyncio.Semaphore limits concurrent in-flight requests.
* Requests beyond MAX_CONCURRENT_REQUESTS are queued (up to MAX_QUEUE_SIZE).
* Requests beyond the queue limit are rejected with HTTP 529.
* Client disconnects are detected and in-flight processing is cancelled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from config import config
from router.core import RouterCore
from router.metrics import metrics
from utils.errors import invalid_request, overloaded_error, api_error

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency control
# ---------------------------------------------------------------------------

_semaphore: asyncio.Semaphore
_queue_count: int = 0
_queue_lock: asyncio.Lock


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _semaphore, _queue_lock
    _semaphore = asyncio.Semaphore(config.router.max_concurrent_requests)
    _queue_lock = asyncio.Lock()
    logger.info(
        "NIM-Proxy router starting on %s:%d "
        "(max_concurrent=%d, max_queue=%d)",
        config.router.host,
        config.router.port,
        config.router.max_concurrent_requests,
        config.router.max_queue_size,
    )
    yield
    logger.info("NIM-Proxy router shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NIM-Proxy: Anthropic-Compatible LLM Router",
    description="Routes Claude Code requests to Ollama (primary) or NVIDIA NIM (fallback)",
    version="1.0.0",
    lifespan=lifespan,
)

_router_core = RouterCore()


# ---------------------------------------------------------------------------
# Request/Response schemas (Anthropic API)
# ---------------------------------------------------------------------------

class TextContentBlock(BaseModel):
    type: str = "text"
    text: str = ""


class ImageSource(BaseModel):
    type: str
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None


class ImageContentBlock(BaseModel):
    type: str = "image"
    source: ImageSource


class ToolUseContentBlock(BaseModel):
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: Dict[str, Any] = Field(default_factory=dict)


class ToolResultContentBlock(BaseModel):
    type: str = "tool_result"
    tool_use_id: str = ""
    content: Union[str, List[Dict[str, Any]], None] = None


ContentBlock = Union[
    TextContentBlock,
    ImageContentBlock,
    ToolUseContentBlock,
    ToolResultContentBlock,
    Dict[str, Any],
]

MessageContent = Union[str, List[ContentBlock]]


class Message(BaseModel):
    role: str
    content: MessageContent

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {v!r}. Must be 'user' or 'assistant'")
        return v


class SystemBlock(BaseModel):
    type: str = "text"
    text: str = ""


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int = Field(gt=0, le=32768)
    messages: List[Message] = Field(min_length=1)
    system: Optional[Union[str, List[SystemBlock]]] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Concurrency helpers
# ---------------------------------------------------------------------------

async def _acquire_slot() -> bool:
    """
    Try to acquire a processing slot.

    Returns True if acquired, False if the system is overloaded (both
    semaphore and queue are full).
    """
    global _queue_count

    if _semaphore.locked():
        async with _queue_lock:
            if _queue_count >= config.router.max_queue_size:
                return False
            _queue_count += 1

        try:
            await _semaphore.acquire()
        finally:
            async with _queue_lock:
                _queue_count = max(0, _queue_count - 1)
    else:
        await _semaphore.acquire()

    return True


def _release_slot() -> None:
    _semaphore.release()


# ---------------------------------------------------------------------------
# /v1/messages endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def messages(request: Request):
    # Parse and validate
    try:
        raw_body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content=invalid_request("Request body must be valid JSON"))

    try:
        parsed = MessagesRequest.model_validate(raw_body)
    except ValidationError as exc:
        # Extract structured field errors — no raw exception string reaches the response
        error_parts = [
            f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=400,
            content=invalid_request("Validation failed: " + "; ".join(error_parts[:5])),
        )
    except Exception:
        logger.warning("Unexpected error during request validation", exc_info=True)
        return JSONResponse(status_code=400, content=invalid_request("Invalid request format"))

    # Concurrency control
    acquired = await _acquire_slot()
    if not acquired:
        logger.warning("System overloaded — rejecting request")
        return JSONResponse(status_code=529, content=overloaded_error())

    # Convert validated model back to dict for adapters (preserve all fields)
    request_dict = parsed.model_dump(exclude_none=True)

    try:
        if parsed.stream:
            # Streaming response — detect client disconnect and cancel
            async def _stream_with_disconnect() -> AsyncIterator[str]:
                try:
                    async for chunk in _router_core.handle_stream(request_dict, parsed.model):
                        if await request.is_disconnected():
                            logger.info("Client disconnected — terminating stream")
                            return
                        yield chunk
                finally:
                    _release_slot()

            return StreamingResponse(
                _stream_with_disconnect(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming response
            try:
                result = await _router_core.handle(request_dict, parsed.model, stream=False)
            finally:
                _release_slot()

            # If result is an error dict, return appropriate HTTP status
            if result.get("type") == "error":
                error_type = result.get("error", {}).get("type", "api_error")
                status_map = {
                    "invalid_request_error": 400,
                    "authentication_error": 401,
                    "permission_error": 403,
                    "not_found_error": 404,
                    "rate_limit_error": 429,
                    "overloaded_error": 529,
                    "api_error": 500,
                }
                status = status_map.get(error_type, 500)
                return JSONResponse(status_code=status, content=result)

            return JSONResponse(content=result)

    except Exception:
        _release_slot()
        logger.exception("Unhandled error in /v1/messages")
        return JSONResponse(status_code=500, content=api_error("An internal error occurred. Please try again."))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_requests": metrics.active_requests,
        "total_requests": metrics.total_requests,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()


# ---------------------------------------------------------------------------
# Model listing (Claude Code probes this endpoint on startup)
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": config.ollama.model,
                "object": "model",
                "created": 1700000000,
                "owned_by": "ollama",
            },
            {
                "id": config.nvidia.model,
                "object": "model",
                "created": 1700000000,
                "owned_by": "nvidia",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.router.host,
        port=config.router.port,
        reload=False,
        log_level="info",
    )
