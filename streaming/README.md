# streaming — SSE Streaming Engine

This package handles conversion of provider-specific streaming responses
into strict Anthropic Server-Sent Events (SSE).

---

## Mandatory Anthropic SSE Event Sequence

```
event: message_start
data: {"type": "message_start", "message": {"id": "...", "type": "message",
       "role": "assistant", "content": [], "model": "...",
       "stop_reason": null, "usage": {"input_tokens": N, "output_tokens": 0}}}

event: content_block_start
data: {"type": "content_block_start", "index": 0,
       "content_block": {"type": "text", "text": ""}}

event: ping
data: {"type": "ping"}

event: content_block_delta        ← repeated per token/chunk
data: {"type": "content_block_delta", "index": 0,
       "delta": {"type": "text_delta", "text": "..."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta",
       "delta": {"stop_reason": "end_turn", "stop_sequence": null},
       "usage": {"output_tokens": N}}

event: message_stop
data: {"type": "message_stop"}
```

**Constraints:**
- Events MUST be emitted in the order above.
- No event may be skipped, reordered, or merged.
- Tokens MUST stream incrementally (no full-response buffering).
- The sequence MUST always end with `message_stop`.

---

## `engine.py` — Streaming Engine

Two conversion paths:

### Path 1: `stream_anthropic_passthrough` (Ollama)

Ollama already emits Anthropic SSE. This function:
1. Parses raw bytes into SSE frames.
2. Patches `model` and `id` fields (so the echoed model matches the client's request).
3. Yields validated SSE strings.
4. Guarantees `message_stop` is the last event (injects one if Ollama omits it).

```
Ollama SSE bytes
    │ parse frames
    │ patch model/id
    │ validate order
    ▼
Anthropic SSE strings  →  client
```

### Path 2: `stream_openai_to_anthropic` (NVIDIA)

NVIDIA emits OpenAI-style SSE chunks. This function:
1. Emits `message_start` immediately.
2. Emits `content_block_start` for index 0.
3. Emits a `ping`.
4. For each OpenAI chunk: emits a `content_block_delta`.
5. Tracks `finish_reason` → maps to `stop_reason`.
6. After stream ends: emits `content_block_stop`, `message_delta`, `message_stop`.

```
NVIDIA OpenAI SSE bytes
    │ parse data: {...} frames
    │ extract choices[0].delta.content
    ▼
content_block_delta events  (one per chunk)
    │
    ▼
Anthropic SSE strings  →  client
```

**Tool call handling in NVIDIA stream:**
If NVIDIA returns `delta.tool_calls[].function.arguments`, the fragment is
emitted as a `text_delta` (plain text). No structured `tool_use` blocks are
generated in streaming mode (tool call safety rule).

---

## Error Safety

If an error occurs mid-stream:
- The generator raises the exception.
- The router core catches it and terminates the stream.
- An `event: error` SSE frame with a valid Anthropic error body is yielded.
- The client receives a clean termination.

---

## Performance Notes

- No full-response buffering. Chunks are forwarded as soon as they arrive.
- Per-frame JSON parsing is O(chunk_size), not O(response_size).
- `asyncio` async generators are used throughout for non-blocking I/O.
