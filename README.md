# NIM-Proxy: Anthropic-Compatible LLM Router

A **production-grade, protocol-preserving router** that lets **Claude Code** (or any Anthropic API client) use **Ollama** and **NVIDIA NIM** as backend providers — without any modification to the client.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Component Breakdown](#4-component-breakdown)
5. [Request Flow](#5-request-flow)
6. [Routing Logic](#6-routing-logic)
7. [Streaming Implementation](#7-streaming-implementation)
8. [API Contract](#8-api-contract)
9. [Error Handling & Reliability](#9-error-handling--reliability)
10. [Production Safeguards](#10-production-safeguards)
11. [Setup & Configuration](#11-setup--configuration)
12. [Running the Server](#12-running-the-server)
13. [Claude Code Integration](#13-claude-code-integration)
14. [Example Requests](#14-example-requests)
15. [Testing](#15-testing)
16. [Environment Variables Reference](#16-environment-variables-reference)

---

## 1. Overview

### The Problem

Claude Code communicates **exclusively** via the Anthropic Messages API.
It cannot speak to OpenAI-compatible endpoints. NVIDIA NIM — which offers
powerful free-tier LLM access — uses the OpenAI API format.

### The Solution

**NIM-Proxy** sits between Claude Code and the providers:

```
Claude Code
    │  POST /v1/messages (Anthropic API)
    ▼
NIM-Proxy  ◄─────────────────── single public interface
    │
    ├── Ollama           (primary — Anthropic-compatible, local)
    └── NVIDIA NIM       (fallback — OpenAI-compatible, cloud)
```

Claude Code never learns which provider handled its request. Every response
is guaranteed to match the Anthropic API schema exactly.

### Key Properties

| Property | Guarantee |
|----------|-----------|
| API compatibility | 100% Anthropic Messages API |
| Streaming | Correct SSE event sequence, zero buffering |
| Routing | Deterministic — same input → same provider |
| Failures | Automatic fallback, never crashes |
| Schema | Zero deviation from Anthropic contract |

---

## 2. Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Claude Code                              │
│              ANTHROPIC_BASE_URL=http://localhost:8080            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ POST /v1/messages
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        NIM-Proxy Router                          │
│                                                                  │
│   ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐  │
│   │  Validation  │───▶│ Router Core   │───▶│   Normalizer    │  │
│   │  (Pydantic)  │    │  + Retry Mgr  │    │ (Schema Guard)  │  │
│   └──────────────┘    └───────┬───────┘    └─────────────────┘  │
│                               │                                  │
│              ┌────────────────┼────────────────┐                │
│              ▼                                 ▼                │
│   ┌──────────────────┐            ┌──────────────────────────┐  │
│   │  Ollama Adapter  │            │    NVIDIA NIM Adapter    │  │
│   │  (Primary)       │            │    (Fallback)            │  │
│   │                  │            │                          │  │
│   │  /v1/messages    │            │  Anthropic → OpenAI      │  │
│   │  (Anthropic API) │            │  /v1/chat/completions    │  │
│   │                  │            │  OpenAI → Anthropic      │  │
│   └────────┬─────────┘            └───────────┬──────────────┘  │
│            │                                  │                  │
│            └────────────┬─────────────────────┘                 │
│                         ▼                                        │
│            ┌────────────────────────┐                           │
│            │   Streaming Engine     │                           │
│            │  (SSE Conversion)      │                           │
│            │  · Ollama passthrough  │                           │
│            │  · NVIDIA→Anthropic    │                           │
│            └────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
Request arrives
    │
    ├─ Pydantic schema validation
    ├─ Concurrency semaphore (MAX_CONCURRENT_REQUESTS)
    │
    ▼
RequestState created (request_id, retry_count=0, providers_used=[])
    │
    ▼
decide_provider() → "ollama" | "nvidia"
    │
    ├─ [ollama] ──▶ OllamaAdapter.send() / send_stream()
    │                     │ failure?
    │                     └──▶ state.record_failure()
    │                              │
    │                              ▼
    └─ [nvidia] ──▶ NvidiaAdapter.send() / send_stream()
                          │
                          ▼
                   normalize_response()
                          │
                          ▼
                   JSONResponse / StreamingResponse
```

---

## 3. Directory Structure

```
nim-proxy/
│
├── main.py                  # FastAPI application entry point
├── config.py                # All configuration (env vars + defaults)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
│
├── router/                  # Core orchestration
│   ├── core.py              # RouterCore — request lifecycle
│   ├── routing.py           # Deterministic routing decision
│   ├── state.py             # Per-request state (RequestState)
│   ├── metrics.py           # In-process observability counters
│   └── README.md
│
├── adapters/                # Provider-specific adapters
│   ├── base.py              # Abstract BaseAdapter + exception types
│   ├── ollama.py            # Ollama adapter (primary)
│   ├── nvidia.py            # NVIDIA NIM adapter (fallback)
│   └── README.md
│
├── streaming/               # SSE streaming engine
│   ├── engine.py            # Anthropic SSE passthrough + OpenAI→Anthropic conversion
│   └── README.md
│
├── normalizer/              # Response schema enforcement
│   ├── response.py          # normalize_response()
│   └── README.md
│
├── utils/                   # Shared stateless helpers
│   ├── errors.py            # Anthropic error envelope constructors
│   ├── tokens.py            # Token estimation + context truncation
│   ├── models.py            # Model name mapping
│   └── README.md
│
└── tests/                   # Unit test suite (44 tests, no live providers needed)
    ├── test_routing.py
    ├── test_normalizer.py
    ├── test_tokens.py
    ├── test_state.py
    ├── test_nvidia_adapter.py
    ├── test_models.py
    └── README.md
```

---

## 4. Component Breakdown

### A. Router Core (`router/core.py`)

The central orchestrator. Responsibilities:

- Create `RequestState` with a unique `request_id` (UUID4).
- Call `decide_provider()` for routing decisions.
- Skip rate-limited providers (NVIDIA cooldown).
- Call the appropriate adapter.
- Validate response (non-empty content check).
- Execute retry logic with alternate provider.
- Apply `normalize_response()` before returning.
- Log every decision with `request_id` for traceability.

### B. Ollama Adapter (`adapters/ollama.py`)

- Target: `{OLLAMA_BASE_URL}/v1/messages` (Anthropic-native endpoint)
- Substitutes internal Ollama model name.
- Applies context truncation if request exceeds `OLLAMA_MAX_TOKENS`.
- Enforces per-level timeouts via `httpx.Timeout`.

### C. NVIDIA Adapter (`adapters/nvidia.py`)

Full bidirectional protocol conversion:

**Anthropic → OpenAI:**
- `system` (string/blocks) → `messages[0]` with `role: "system"`.
- `tools` → serialised to plain-text system prompt (**tool call safety**).
- `messages[].content` (blocks) → flattened string.
- `stop_sequences` → `stop`.

**OpenAI → Anthropic:**
- `choices[0].message.content` → `content[0].text`.
- `finish_reason: "stop"` → `stop_reason: "end_turn"`.
- `finish_reason: "length"` → `stop_reason: "max_tokens"`.
- `usage.prompt_tokens` → `usage.input_tokens`.

### D. Streaming Engine (`streaming/engine.py`)

Two code paths:

| Path | Provider | Strategy |
|------|----------|----------|
| `stream_anthropic_passthrough` | Ollama | Forward Anthropic SSE verbatim (patch model/id) |
| `stream_openai_to_anthropic` | NVIDIA | Construct full Anthropic SSE event sequence from OpenAI chunks |

Both paths guarantee `message_stop` is always the final event.

### E. Response Normalizer (`normalizer/response.py`)

Last-mile schema enforcement:
- Generates missing `id`.
- Wraps string `content` in text blocks.
- Fills empty `content` with a text block.
- Maps all `finish_reason`/`stop_reason` variants to Anthropic values.
- Estimates `usage` if provider omitted it.
- Always echoes `requested_model` (never the internal provider model).

### F. Request State (`router/state.py`)

Per-request, never shared:
- `request_id`: UUID4 string
- `retry_count`: 0 → max 2
- `providers_used`: `["ollama"]` → `["ollama", "nvidia"]`
- Loop prevention: stops when `retry_count >= 2` OR both providers tried.

### G. Metrics (`router/metrics.py`)

Async-safe in-process counters exposed at `GET /metrics`.

---

## 5. Request Flow

### Non-Streaming

```
POST /v1/messages
    │
    ├── [1] JSON parse + Pydantic validation
    ├── [2] Semaphore acquire (concurrency control)
    ├── [3] RequestState created
    ├── [4] decide_provider() → "ollama"
    ├── [5] OllamaAdapter.send(body)
    │         ├── truncate_messages() if needed
    │         ├── httpx POST to Ollama /v1/messages
    │         └── returns raw Anthropic response
    ├── [6] normalize_response()
    ├── [7] Semaphore release
    └── [8] JSONResponse → client
```

### Streaming

```
POST /v1/messages  (stream=true)
    │
    ├── [1–4] Same as non-streaming
    ├── [5] OllamaAdapter.send_stream(body)
    │         └── yields raw bytes from Ollama SSE stream
    ├── [6] stream_anthropic_passthrough()
    │         └── yields "event: ...\ndata: ...\n\n" strings
    └── [7] StreamingResponse → client (token by token)
```

### Failure + Fallback

```
[5] OllamaAdapter.send() → AdapterTimeout
    │
    ├── state.record_failure("timeout")
    ├── metrics.record_retry()
    ├── state.can_retry()? → True
    ├── decide_provider(providers_used=["ollama"]) → "nvidia"
    ├── NvidiaAdapter.send(body)
    │         ├── anthropic_to_openai_request()
    │         ├── httpx POST to NVIDIA /v1/chat/completions
    │         └── openai_to_anthropic_response()
    └── normalize_response() → return
```

---

## 6. Routing Logic

Routing is **deterministic** — identical inputs always produce the same provider decision.

### Decision Tree

```
IF providers_already_used contains "ollama" AND NOT "nvidia":
    → NVIDIA (fallback path)

IF providers_already_used contains "nvidia" AND NOT "ollama":
    → Ollama (fallback path)

ELSE (fresh request):
    total_chars = sum of all message + system content lengths

    IF total_chars >= LONG_PROMPT_THRESHOLD (default: 2000):
        → NVIDIA

    IF len(messages) >= MANY_MESSAGES_THRESHOLD (default: 10):
        → NVIDIA

    IF any message/system contains reasoning keywords:
        → NVIDIA

    DEFAULT:
        → Ollama
```

### Reasoning Keywords (partial list)

`analysis`, `analyze`, `architect`, `design`, `plan`, `debug`, `diagnose`,
`explain in detail`, `step-by-step`, `comprehensive`, `compare and contrast`,
`pros and cons`, `complex`, `investigate`, `refactor large`, `rewrite entire`

### Retry Policy

```
Attempt 1: Primary provider (Ollama default)
    │ failure
    ▼
Attempt 2: Alternate provider (NVIDIA)
    │ failure
    ▼
Return Anthropic error response

STOP when: retry_count >= 2  OR  both providers tried
```

---

## 7. Streaming Implementation

### Anthropic SSE Contract (enforced)

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message",
       "role":"assistant","content":[],"model":"...","stop_reason":null,
       "usage":{"input_tokens":N,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,
       "content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta      ← repeated per token
data: {"type":"content_block_delta","index":0,
       "delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta",
       "delta":{"stop_reason":"end_turn","stop_sequence":null},
       "usage":{"output_tokens":M}}

event: message_stop
data: {"type":"message_stop"}
```

### Ollama Stream (passthrough)

Ollama natively emits Anthropic SSE. The engine:
1. Parses raw bytes into `event:` / `data:` frames.
2. Patches `model` and `id` to match the requested model.
3. Forwards each frame immediately (zero buffering).
4. Injects `message_stop` if Ollama omits it.

### NVIDIA Stream (conversion)

NVIDIA emits OpenAI-style chunks:
```
data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
```

The engine:
1. Emits `message_start` + `content_block_start` + `ping` **before** reading the stream.
2. For each chunk: extracts `delta.content`, emits `content_block_delta`.
3. Tracks `finish_reason` → maps to `stop_reason`.
4. After stream: emits `content_block_stop` + `message_delta` + `message_stop`.

### Client Disconnect Handling

The FastAPI streaming response checks `await request.is_disconnected()` before each chunk. If the client disconnects, the generator stops immediately — no orphan provider requests.

---

## 8. API Contract

### Endpoint

```
POST /v1/messages
```

### Request Body

```json
{
  "model":           "claude-3-5-sonnet-20241022",
  "max_tokens":      1024,
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "system":          "You are a helpful assistant.",
  "temperature":     0.7,
  "top_p":           1.0,
  "stop_sequences":  [],
  "stream":          false,
  "tools":           [],
  "tool_choice":     null
}
```

### Non-Streaming Response

```json
{
  "id":           "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type":         "message",
  "role":         "assistant",
  "content":      [{"type": "text", "text": "Hello! How can I help you today?"}],
  "model":        "claude-3-5-sonnet-20241022",
  "stop_reason":  "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens":  25,
    "output_tokens": 11
  }
}
```

### Error Response

```json
{
  "type": "error",
  "error": {
    "type":    "api_error",
    "message": "All providers failed. Last error: Ollama timed out"
  }
}
```

### Additional Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Observability counters |
| `GET` | `/v1/models` | Model listing (Claude Code probe) |

---

## 9. Error Handling & Reliability

### Provider Error Matrix

| Error | Provider | Action |
|-------|----------|--------|
| Timeout | Any | Retry with alternate provider |
| HTTP 429 (rate limit) | NVIDIA | Set cooldown, switch to Ollama |
| HTTP 5xx | Any | Retry with alternate provider |
| Empty response | Any | Retry with alternate provider |
| Invalid JSON | Any | Retry with alternate provider |
| HTTP 400 | Any | Non-retryable, return error |

### NVIDIA Rate Limit Cooldown

When NVIDIA returns HTTP 429:
1. `_rate_limit_until["nvidia"]` is set to `now + NVIDIA_RATE_LIMIT_COOLDOWN`.
2. All subsequent requests skip NVIDIA until the cooldown expires.
3. Traffic falls through to Ollama automatically.

### Timeout Levels

| Level | Configured via | Description |
|-------|---------------|-------------|
| Connection | `*_TIMEOUT_CONNECT` | TCP handshake timeout |
| Response | `*_TIMEOUT_RESPONSE` | First byte timeout |
| Stream idle | `*_TIMEOUT_STREAM_IDLE` | Max gap between stream chunks |
| Total | `*_TIMEOUT_TOTAL` | Absolute request lifetime |

---

## 10. Production Safeguards

### Concurrency Control

```python
MAX_CONCURRENT_REQUESTS = 50   # Simultaneous in-flight requests
MAX_QUEUE_SIZE          = 100  # Queued requests waiting for a slot
```

- Requests beyond `MAX_CONCURRENT_REQUESTS` are queued.
- Requests beyond `MAX_CONCURRENT_REQUESTS + MAX_QUEUE_SIZE` are rejected with HTTP 529.
- Every in-flight request holds a semaphore slot that is released on completion, error, or cancellation.

### Context/Token Management

- Each provider has a configured `*_MAX_TOKENS` limit.
- If a request would exceed the limit, oldest messages are dropped until the request fits.
- System prompts are always preserved.
- If truncation is impossible (even one message is too large), a valid Anthropic error is returned.

### Resource Cleanup

- All `httpx.AsyncClient` instances are created with `async with` — closed after each request.
- Streaming generators terminate cleanly on disconnect or error.
- No global mutable state except the metrics counters and rate-limit timestamps.

### Observability

`GET /metrics` returns:

```json
{
  "total_requests":   1024,
  "active_requests":  3,
  "failure_count":    12,
  "retry_count":      8,
  "provider_usage": {
    "ollama":  980,
    "nvidia":  44
  }
}
```

---

## 11. Setup & Configuration

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally
- NVIDIA NIM API key (optional, only needed for fallback)

### Install

```bash
git clone https://github.com/chandankumar123456/nim-proxy
cd nim-proxy

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env with your values
```

Minimum required configuration:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Only needed if NVIDIA fallback is desired:
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_MODEL=meta/llama-3.1-70b-instruct
```

---

## 12. Running the Server

### Start Ollama

```bash
ollama serve
ollama pull llama3.2   # or your chosen model
```

### Start NIM-Proxy

```bash
# Direct (development)
python main.py

# Production (via uvicorn)
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1
```

> **Note:** Use `--workers 1` with the default asyncio semaphore. For multi-process deployments, use a shared concurrency mechanism (Redis semaphore or similar).

### Verify

```bash
curl http://localhost:8080/health
# {"status":"ok","active_requests":0,"total_requests":0}
```

---

## 13. Claude Code Integration

Set these environment variables **before** launching Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-not-checked
```

Or in a `.env` file in your project:

```env
ANTHROPIC_BASE_URL=http://localhost:8080
ANTHROPIC_API_KEY=dummy-not-checked
```

Claude Code will then send all its Anthropic API requests to NIM-Proxy,
which transparently routes them to Ollama or NVIDIA NIM.

---

## 14. Example Requests

### Non-Streaming

```bash
curl -s -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Write a Python hello world."}]
  }' | python3 -m json.tool
```

### Streaming

```bash
curl -s -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 256,
    "stream": true,
    "messages": [{"role": "user", "content": "Count to 5."}]
  }'
```

### With System Prompt

```bash
curl -s -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 512,
    "system": "You are a senior Python engineer.",
    "messages": [
      {"role": "user", "content": "Explain list comprehensions."}
    ]
  }'
```

### Metrics

```bash
curl http://localhost:8080/metrics
```

---

## 15. Testing

```bash
# Install pytest
pip install pytest

# Run all 44 unit tests (no live providers needed)
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_routing.py -v
python -m pytest tests/test_normalizer.py -v
```

All tests pass without any running Ollama or NVIDIA instance.

---

## 16. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_HOST` | `0.0.0.0` | Server bind address |
| `ROUTER_PORT` | `8080` | Server port |
| `MAX_CONCURRENT_REQUESTS` | `50` | Max simultaneous in-flight requests |
| `MAX_QUEUE_SIZE` | `100` | Max queued requests |
| `MAX_RETRIES` | `2` | Max retry attempts per request |
| `LONG_PROMPT_THRESHOLD` | `2000` | Char count that triggers NVIDIA routing |
| `MANY_MESSAGES_THRESHOLD` | `10` | Message count that triggers NVIDIA routing |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_TIMEOUT_CONNECT` | `10` | TCP connect timeout (s) |
| `OLLAMA_TIMEOUT_RESPONSE` | `30` | First byte timeout (s) |
| `OLLAMA_TIMEOUT_STREAM_IDLE` | `60` | Stream idle timeout (s) |
| `OLLAMA_TIMEOUT_TOTAL` | `300` | Total request timeout (s) |
| `OLLAMA_MAX_TOKENS` | `8192` | Max context tokens for Ollama |
| `NVIDIA_API_KEY` | _(required for NVIDIA)_ | NVIDIA NIM API key |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA endpoint |
| `NVIDIA_MODEL` | `meta/llama-3.1-70b-instruct` | NVIDIA model name |
| `NVIDIA_TIMEOUT_CONNECT` | `10` | TCP connect timeout (s) |
| `NVIDIA_TIMEOUT_RESPONSE` | `30` | First byte timeout (s) |
| `NVIDIA_TIMEOUT_STREAM_IDLE` | `60` | Stream idle timeout (s) |
| `NVIDIA_TIMEOUT_TOTAL` | `300` | Total request timeout (s) |
| `NVIDIA_MAX_TOKENS` | `32768` | Max context tokens for NVIDIA |
| `NVIDIA_RATE_LIMIT_COOLDOWN` | `60` | Cooldown seconds after 429 |

---

## License

This project is provided as-is for educational and integration purposes.
