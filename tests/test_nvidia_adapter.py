"""
tests/test_nvidia_adapter.py
=============================
Unit tests for the NVIDIA adapter's request/response conversion logic.
"""
import pytest

from adapters.nvidia import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    _anthropic_to_openai_messages,
    _content_block_to_str,
    _system_to_str,
    _tools_to_system_text,
)


def test_system_to_str_string():
    assert _system_to_str("You are helpful") == "You are helpful"


def test_system_to_str_list():
    system = [{"type": "text", "text": "Block 1"}, {"type": "text", "text": "Block 2"}]
    result = _system_to_str(system)
    assert "Block 1" in result
    assert "Block 2" in result


def test_system_to_str_none():
    assert _system_to_str(None) == ""


def test_content_block_to_str_string():
    assert _content_block_to_str("Hello world") == "Hello world"


def test_content_block_to_str_list():
    blocks = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
    result = _content_block_to_str(blocks)
    assert "Hello" in result
    assert "World" in result


def test_content_block_tool_use():
    blocks = [{"type": "tool_use", "name": "bash", "input": {"cmd": "ls"}}]
    result = _content_block_to_str(blocks)
    assert "bash" in result


def test_anthropic_to_openai_messages_with_system():
    messages = [{"role": "user", "content": "Hello"}]
    result = _anthropic_to_openai_messages(messages, "You are helpful")
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"


def test_anthropic_to_openai_request_basic():
    body = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    result = anthropic_to_openai_request(body, "claude-3-5-sonnet", "meta/llama-3.1-70b-instruct")
    assert result["model"] == "meta/llama-3.1-70b-instruct"
    assert result["max_tokens"] == 1024
    assert isinstance(result["messages"], list)


def test_anthropic_to_openai_request_with_tools_injected_as_text():
    body = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [{"name": "bash", "description": "Run shell command", "input_schema": {}}],
    }
    result = anthropic_to_openai_request(body, "claude-3-5-sonnet", "nvidia-model")
    # Tools should be injected into the system message, not as structured tool defs
    assert "tools" not in result
    system_msg = next((m for m in result["messages"] if m["role"] == "system"), None)
    assert system_msg is not None
    assert "bash" in system_msg["content"]


def test_openai_to_anthropic_response_basic():
    openai_resp = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    result = openai_to_anthropic_response(openai_resp, "claude-3-5-sonnet")
    assert result["type"] == "message"
    assert result["role"] == "assistant"
    assert result["model"] == "claude-3-5-sonnet"
    assert result["stop_reason"] == "end_turn"
    assert result["content"][0]["text"] == "Hello there!"
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 5


def test_openai_to_anthropic_response_length_stop_reason():
    openai_resp = {
        "choices": [{"message": {"content": "..."}, "finish_reason": "length"}],
        "usage": {},
    }
    result = openai_to_anthropic_response(openai_resp, "test-model")
    assert result["stop_reason"] == "max_tokens"


def test_openai_to_anthropic_response_no_choices_raises():
    from adapters.base import AdapterError
    with pytest.raises(AdapterError):
        openai_to_anthropic_response({"choices": []}, "test-model")
