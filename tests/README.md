# tests — Test Suite

Unit tests for the NIM-Proxy router. All tests run without any live provider
connections (no Ollama or NVIDIA required).

---

## Running Tests

```bash
# Install dependencies
pip install -r requirements.txt pytest

# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_routing.py -v
```

---

## Test Files

| File | What is tested |
|------|----------------|
| `test_routing.py` | Routing decision logic (determinism, thresholds, keywords) |
| `test_normalizer.py` | Response normalizer (schema enforcement, field defaults) |
| `test_tokens.py` | Token estimation, request estimation, truncation strategy |
| `test_state.py` | Per-request state (retry logic, provider tracking, loop prevention) |
| `test_nvidia_adapter.py` | NVIDIA adapter conversion (Anthropic↔OpenAI, tool safety) |
| `test_models.py` | Model name mapping (prefix matching, case-insensitivity) |

---

## Coverage Summary

| Component | Coverage |
|-----------|----------|
| Routing decision | Short/long/many-messages/keywords/retry-path |
| Response normalizer | All field defaults, finish_reason mapping |
| Token utilities | Estimation, truncation, budget edge cases |
| Request state | All state transitions and loop guards |
| NVIDIA conversion | Request conversion, response conversion, tool injection |
| Model mapping | Prefix matching, fallback, case-insensitivity |

---

## Design Notes

- Tests are **pure unit tests** — no mocking of HTTP or providers needed.
- The streaming engine and router core require live adapters; they are tested
  via integration paths (see manual testing in the root README).
