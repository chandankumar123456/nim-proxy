# utils — Shared Helper Utilities

This package contains stateless helper modules used across the entire router.
No component in this package has knowledge of providers or HTTP.

---

## Modules

### `errors.py` — Anthropic Error Formatting

Produces valid Anthropic-format error envelopes for every failure path.

**Anthropic error structure:**
```json
{
  "type": "error",
  "error": {
    "type": "<error_type>",
    "message": "<human-readable message>"
  }
}
```

| Helper | HTTP Status | `error.type` |
|--------|-------------|--------------|
| `invalid_request()` | 400 | `invalid_request_error` |
| `auth_error()` | 401 | `authentication_error` |
| `rate_limit_error()` | 429 | `rate_limit_error` |
| `overloaded_error()` | 529 | `overloaded_error` |
| `api_error()` | 500 | `api_error` |
| `from_http_status(code, msg)` | varies | mapped |

---

### `tokens.py` — Token Estimation and Context Truncation

**Purpose:** Estimate token counts (without a real tokeniser) and enforce
provider context limits.

**Estimation strategy:** `len(text) // 4` characters per token (conservative
English-text estimate). Always returns ≥ 1 for non-empty input.

**Truncation strategy (per spec):**
1. Preserve the system prompt (always kept).
2. Preserve the **newest** messages.
3. Drop the **oldest** messages first.
4. Raise `ValueError` if even the newest single message cannot fit.

**Key functions:**

| Function | Description |
|----------|-------------|
| `estimate_tokens(text)` | Estimate tokens for a string |
| `estimate_request_tokens(messages, system)` | Total token estimate for a request |
| `truncate_messages(messages, system, max_tokens, max_output_tokens)` | Return truncated message list |

---

### `models.py` — Model Name Mapping

Maps the client-facing model name (e.g. `claude-3-5-sonnet-20241022`) to a
provider-specific model identifier.

**Rules:**
- Mapping is prefix-based (case-insensitive).
- `claude-*` → configured `OLLAMA_MODEL` / `NVIDIA_MODEL`.
- Unknown models fall back to env-var defaults.
- Provider model names are **never** returned to the caller — the original
  model name is always echoed back in responses.

**Key function:**
```python
get_provider_model(requested_model: str, provider: str) -> str
```

---

## Design Constraints

- No cross-module imports within `utils/`.
- No provider knowledge.
- Fully unit-testable without a running server.
