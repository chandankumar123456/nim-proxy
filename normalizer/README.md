# normalizer — Response Normalizer

Ensures every outbound response strictly matches the Anthropic Messages API
response schema. Acts as the last transformation step before returning to
the client.

---

## Why This Layer Exists

Provider responses can have:
- Missing `stop_reason` fields
- `finish_reason` in OpenAI format instead of Anthropic's `stop_reason`
- Empty or absent `content` arrays
- Missing `id` fields
- Absent or partial `usage` statistics

The normalizer **fixes all of these** so the client always receives a
100% spec-compliant response.

---

## Enforced Anthropic Response Schema

```json
{
  "id":           "msg_...",
  "type":         "message",
  "role":         "assistant",
  "content":      [{"type": "text", "text": "..."}],
  "model":        "<requested_model>",
  "stop_reason":  "end_turn | max_tokens | stop_sequence | tool_use",
  "stop_sequence": null,
  "usage": {
    "input_tokens":  42,
    "output_tokens": 17
  }
}
```

---

## `response.py` — normalize_response

```python
def normalize_response(
    raw: Dict[str, Any],
    requested_model: str,
    estimated_input_tokens: int = 0,
) -> Dict[str, Any]:
```

**Transformations applied:**

| Issue | Fix |
|-------|-----|
| Missing `id` | Generate `msg_<uuid24>` |
| `content` is string | Wrap in `[{"type": "text", "text": ...}]` |
| `content` is empty list | Insert `[{"type": "text", "text": ""}]` |
| Content blocks without `type` | Skip invalid blocks |
| Missing `stop_reason` | Default to `"end_turn"` |
| OpenAI `finish_reason: "stop"` | Map to `"end_turn"` |
| OpenAI `finish_reason: "length"` | Map to `"max_tokens"` |
| `model` in response | Always overwritten with `requested_model` |
| Missing `usage` | Estimate from content char count |
| Partial `usage` | Fill missing fields with estimates |

### stop_reason mapping

| Provider value | Normalised to |
|----------------|---------------|
| `"end_turn"` | `"end_turn"` (pass-through) |
| `"stop"` | `"end_turn"` |
| `"max_tokens"` | `"max_tokens"` (pass-through) |
| `"length"` | `"max_tokens"` |
| `"tool_use"` | `"tool_use"` (pass-through) |
| `"tool_calls"` | `"tool_use"` |
| `"function_call"` | `"tool_use"` |
| anything else | `"end_turn"` (safe default) |

---

## Design Constraints

- This module has **no network I/O**.
- It **never mutates** the input `raw` dict.
- It always returns a **complete, valid** Anthropic response.
- It is the **only place** where response schema is enforced.
