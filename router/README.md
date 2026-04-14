# router — Core Orchestration Components

This package contains the central orchestration logic for the NIM-Proxy router.
It is responsible for routing decisions, per-request state, metrics, and the
end-to-end request lifecycle.

---

## Architecture Diagram

```
FastAPI /v1/messages
        │
        ▼
  RouterCore.handle()          ← main entry point
        │
        ├── RequestState       ← per-request isolation
        ├── decide_provider()  ← deterministic routing
        │
        ├── OllamaAdapter.send()    ─┐
        │   (primary)               │  on failure → retry with
        └── NvidiaAdapter.send()   ─┘  alternate provider
                │
                ▼
        normalize_response()   ← strict Anthropic schema
                │
                ▼
        return to caller
```

---

## Modules

### `state.py` — Per-Request State

Holds all mutable state for a single request lifecycle. **Never shared across requests.**

```python
@dataclass
class RequestState:
    request_id: str          # UUID4, unique per request
    retry_count: int         # 0 to MAX_RETRIES
    providers_used: List[str]  # ["ollama"] or ["ollama", "nvidia"]
    current_provider: str    # "ollama" | "nvidia"
    failure_reasons: List[str]
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `mark_provider_used(provider)` | Records provider attempt |
| `can_retry(max_retries)` | True if retries remain AND not all providers tried |
| `next_provider()` | Returns next provider to try, or `None` |
| `record_failure(reason)` | Increments `retry_count`, appends reason |

---

### `routing.py` — Routing Decision Logic

Deterministic provider selection. **Same input always produces same output.**

**Decision tree:**
```
IF retry (provider already tried):
    → use the OTHER provider

ELSE (fresh request):
    IF total_chars >= LONG_PROMPT_THRESHOLD (default 2000): → NVIDIA
    IF message_count >= MANY_MESSAGES_THRESHOLD (default 10): → NVIDIA
    IF reasoning keywords detected: → NVIDIA
    DEFAULT: → Ollama
```

**Reasoning keywords detected (partial list):**
`analysis`, `analyze`, `architect`, `design`, `plan`, `debug`, `diagnose`,
`explain in detail`, `step-by-step`, `comprehensive`, `compare and contrast`, etc.

**Key function:**
```python
decide_provider(
    messages: List[Dict],
    system: Optional[Any],
    providers_already_used: List[str],
) -> str  # "ollama" | "nvidia"
```

---

### `metrics.py` — In-Process Observability

Thread-safe async counters for the `/metrics` endpoint.

| Counter | Description |
|---------|-------------|
| `total_requests` | Total requests received since startup |
| `active_requests` | Currently in-flight |
| `failure_count` | Requests that ended in an error |
| `retry_count` | Total retry attempts |
| `provider_usage` | `{"ollama": N, "nvidia": M}` |

---

### `core.py` — RouterCore Orchestrator

The central execution engine. Stateless per call (state lives in `RequestState`).

**Non-streaming flow (`handle`):**
1. Create `RequestState` (unique `request_id`).
2. Call `decide_provider()`.
3. Skip rate-limited providers.
4. Call adapter (`OllamaAdapter` or `NvidiaAdapter`).
5. Validate response (non-empty check).
6. On failure: record, retry with alternate provider.
7. After max retries: return Anthropic error.
8. On success: run `normalize_response()` → return.

**Streaming flow (`handle_stream`):**
Same retry logic, but yields SSE strings via the streaming engine.

**Rate-limit cooldown:**
When NVIDIA returns HTTP 429, `_rate_limit_until["nvidia"]` is set to
`now + NVIDIA_RATE_LIMIT_COOLDOWN`. Requests skip NVIDIA until the cooldown expires.

**Retry policy:**
```
Attempt 1: primary provider (e.g. Ollama)
Attempt 2: alternate provider (NVIDIA)
→ STOP if both tried OR retry_count >= MAX_RETRIES
```

---

## Loop Prevention

`RequestState.can_retry()` returns `False` when **either**:
- `retry_count >= max_retries`, **or**
- `len(providers_used) >= 2` (both providers tried)

This guarantees at most **2 attempts** regardless of failure patterns.
