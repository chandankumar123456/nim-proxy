# adapters — Provider Adapter Layer

Each adapter isolates a single provider. **No provider logic leaks outside
its adapter file.** Adapters never communicate with each other.

---

## Architecture Diagram

```
RouterCore
    │
    ├── OllamaAdapter          adapters/ollama.py
    │       │
    │       └── Ollama /v1/messages   (Anthropic-compatible API)
    │
    └── NvidiaAdapter          adapters/nvidia.py
            │
            ├── Convert: Anthropic request → OpenAI request
            ├── NVIDIA NIM /v1/chat/completions   (OpenAI-compatible API)
            └── Convert: OpenAI response → Anthropic response
```

---

## Modules

### `base.py` — Abstract Base Adapter

```python
class BaseAdapter(ABC):
    provider_name: str                 # "ollama" | "nvidia"

    async def send(body, requested_model) -> Dict:
        """Non-streaming request → Anthropic response dict."""

    async def send_stream(body, requested_model) -> AsyncIterator[bytes]:
        """Streaming request → raw bytes from provider."""
```

**Exception hierarchy:**
```
AdapterError(Exception)
├── RateLimitError   # HTTP 429 — not retryable
└── TimeoutError     # Connection/read timeout — retryable
```

All provider exceptions MUST be caught within the adapter and re-raised as
`AdapterError` subclasses. The router core only handles `AdapterError`.

---

### `ollama.py` — Ollama Adapter (Primary)

Ollama exposes an **Anthropic-compatible** `/v1/messages` endpoint, so
this adapter requires minimal transformation.

**Responsibilities:**
- Replace client model name with `OLLAMA_MODEL`.
- Apply context truncation (oldest messages dropped if over `OLLAMA_MAX_TOKENS`).
- Configure per-request httpx timeouts.
- Translate HTTP error codes to `AdapterError` subclasses.
- For streaming: yield raw SSE bytes to the streaming engine.

**Request flow:**
```
Anthropic body
    │ swap model name
    │ truncate if needed
    ▼
POST {OLLAMA_BASE_URL}/v1/messages
    ▼
Raw Anthropic response (passes through normalizer)
```

**Configuration:**

| Env var | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Internal model name |
| `OLLAMA_TIMEOUT_CONNECT` | `10` | TCP connect timeout (s) |
| `OLLAMA_TIMEOUT_RESPONSE` | `30` | First byte timeout (s) |
| `OLLAMA_TIMEOUT_STREAM_IDLE` | `60` | Stream idle timeout (s) |
| `OLLAMA_TIMEOUT_TOTAL` | `300` | Total request timeout (s) |
| `OLLAMA_MAX_TOKENS` | `8192` | Max context tokens |

---

### `nvidia.py` — NVIDIA NIM Adapter (Fallback)

NVIDIA NIM uses the **OpenAI-compatible** API. Full bidirectional conversion
is required.

**Anthropic → OpenAI conversion:**

| Anthropic field | OpenAI field | Notes |
|-----------------|--------------|-------|
| `messages[].content` (blocks) | `messages[].content` (string) | Flattened |
| `system` (string/list) | `messages[0]` with `role: "system"` | Prepended |
| `tools` | Injected as system text | Tool call safety rule |
| `stop_sequences` | `stop` | Direct mapping |
| `max_tokens` | `max_tokens` | Direct mapping |
| `temperature` | `temperature` | Direct mapping |
| `top_p` | `top_p` | Direct mapping |

**OpenAI → Anthropic conversion:**

| OpenAI field | Anthropic field | Notes |
|--------------|-----------------|-------|
| `choices[0].message.content` | `content[0].text` | Wrapped in text block |
| `choices[0].finish_reason: "stop"` | `stop_reason: "end_turn"` | Mapped |
| `choices[0].finish_reason: "length"` | `stop_reason: "max_tokens"` | Mapped |
| `usage.prompt_tokens` | `usage.input_tokens` | Direct |
| `usage.completion_tokens` | `usage.output_tokens` | Direct |

**Tool call safety rule:**
Anthropic tool definitions (`tools` array) are **not** forwarded to NVIDIA as
structured JSON schemas. Instead they are serialised as a plain-text block
appended to the system prompt. This prevents schema validation failures on
the NVIDIA side while preserving the tool instructions for the model.

**Configuration:**

| Env var | Default | Purpose |
|---------|---------|---------|
| `NVIDIA_API_KEY` | (required) | Bearer token |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM endpoint |
| `NVIDIA_MODEL` | `meta/llama-3.1-70b-instruct` | Internal model |
| `NVIDIA_TIMEOUT_CONNECT` | `10` | TCP connect timeout (s) |
| `NVIDIA_TIMEOUT_RESPONSE` | `30` | First byte timeout (s) |
| `NVIDIA_TIMEOUT_STREAM_IDLE` | `60` | Stream idle timeout (s) |
| `NVIDIA_TIMEOUT_TOTAL` | `300` | Total request timeout (s) |
| `NVIDIA_MAX_TOKENS` | `32768` | Max context tokens |
| `NVIDIA_RATE_LIMIT_COOLDOWN` | `60` | Seconds to skip NVIDIA after 429 |

---

## Adding a New Provider

1. Create `adapters/<provider>.py`.
2. Implement `BaseAdapter` (both `send` and `send_stream`).
3. Register in `router/core.py` → `_ADAPTERS` dict.
4. Add routing logic in `router/routing.py`.
5. Add streaming conversion path in `streaming/engine.py` if needed.
