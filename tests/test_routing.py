"""
tests/test_routing.py
======================
Unit tests for the routing decision logic.
"""
import pytest

from router.routing import decide_provider


def _short_messages(n: int = 1, text: str = "Hi"):
    return [{"role": "user", "content": text}] * n


def test_short_prompt_routes_to_ollama():
    messages = _short_messages(1, "Fix this function.")
    assert decide_provider(messages, None, []) == "ollama"


def test_long_prompt_routes_to_nvidia():
    long_text = "A" * 3000
    messages = [{"role": "user", "content": long_text}]
    assert decide_provider(messages, None, []) == "nvidia"


def test_many_messages_routes_to_nvidia():
    messages = _short_messages(10, "msg")
    assert decide_provider(messages, None, []) == "nvidia"


def test_reasoning_keyword_routes_to_nvidia():
    messages = [{"role": "user", "content": "Please analyze this code and provide a comprehensive step-by-step explanation."}]
    assert decide_provider(messages, None, []) == "nvidia"


def test_ollama_tried_routes_to_nvidia():
    messages = _short_messages(1, "Hello")
    assert decide_provider(messages, None, ["ollama"]) == "nvidia"


def test_nvidia_tried_routes_to_ollama():
    messages = _short_messages(1, "Hello")
    assert decide_provider(messages, None, ["nvidia"]) == "ollama"


def test_system_prompt_length_counted():
    short_messages = [{"role": "user", "content": "hi"}]
    long_system = "A" * 2500
    assert decide_provider(short_messages, long_system, []) == "nvidia"


def test_same_input_deterministic():
    messages = [{"role": "user", "content": "What is 2+2?"}]
    result1 = decide_provider(messages, None, [])
    result2 = decide_provider(messages, None, [])
    assert result1 == result2
