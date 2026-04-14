"""
tests/test_normalizer.py
=========================
Unit tests for the response normalizer.
"""
import pytest

from normalizer.response import normalize_response


def test_basic_response_normalized():
    raw = {
        "id": "msg_abc",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    result = normalize_response(raw, "claude-3-5-sonnet-20241022")
    assert result["type"] == "message"
    assert result["role"] == "assistant"
    assert result["model"] == "claude-3-5-sonnet-20241022"
    assert result["stop_reason"] == "end_turn"
    assert isinstance(result["content"], list)
    assert len(result["content"]) > 0
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 5


def test_missing_stop_reason_defaults_to_end_turn():
    raw = {
        "id": "msg_abc",
        "content": [{"type": "text", "text": "Hi"}],
    }
    result = normalize_response(raw, "claude-test")
    assert result["stop_reason"] == "end_turn"


def test_openai_finish_reason_mapped():
    raw = {
        "content": [{"type": "text", "text": "Done"}],
        "stop_reason": "stop",  # OpenAI style
    }
    result = normalize_response(raw, "test-model")
    assert result["stop_reason"] == "end_turn"


def test_string_content_wrapped_in_block():
    raw = {
        "content": "Plain text response",
        "stop_reason": "end_turn",
    }
    result = normalize_response(raw, "test-model")
    assert isinstance(result["content"], list)
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "Plain text response"


def test_empty_content_gets_fallback_block():
    raw = {"content": [], "stop_reason": "end_turn"}
    result = normalize_response(raw, "test-model")
    assert isinstance(result["content"], list)
    assert len(result["content"]) > 0


def test_missing_id_gets_generated():
    raw = {"content": [{"type": "text", "text": "Hi"}]}
    result = normalize_response(raw, "test-model")
    assert result["id"].startswith("msg_")


def test_model_echoed_back():
    raw = {"content": [{"type": "text", "text": "Hi"}], "model": "internal-model-name"}
    result = normalize_response(raw, "claude-3-opus")
    assert result["model"] == "claude-3-opus"  # Always echoes requested model


def test_usage_estimated_when_missing():
    raw = {"content": [{"type": "text", "text": "A" * 100}]}
    result = normalize_response(raw, "test-model", estimated_input_tokens=50)
    assert result["usage"]["input_tokens"] == 50
    assert result["usage"]["output_tokens"] >= 0
