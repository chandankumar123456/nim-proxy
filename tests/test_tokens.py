"""
tests/test_tokens.py
=====================
Unit tests for token estimation and message truncation utilities.
"""
import pytest

from utils.tokens import estimate_tokens, estimate_request_tokens, truncate_messages


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_basic():
    # 40 chars → 10 tokens
    assert estimate_tokens("A" * 40) == 10


def test_estimate_request_tokens():
    messages = [
        {"role": "user", "content": "A" * 400},
        {"role": "assistant", "content": "B" * 400},
    ]
    tokens = estimate_request_tokens(messages, system="C" * 400)
    # ~100 + 4 + 100 + 4 + 100 = 308
    assert tokens > 250


def test_truncate_messages_within_budget():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]
    result = truncate_messages(messages, None, max_tokens=1000, max_output_tokens=100)
    assert result == messages  # No truncation needed


def test_truncate_messages_drops_oldest():
    # Each message ~250 tokens (1000 chars / 4)
    messages = [
        {"role": "user", "content": "A" * 1000},  # Message 0 (oldest)
        {"role": "user", "content": "B" * 1000},  # Message 1
        {"role": "user", "content": "C" * 100},   # Message 2 (newest, small)
    ]
    # Budget: 400 tokens - 50 output = 350 for input
    # Message 2 ≈ 25 tokens, fits; Message 1 ≈ 250 tokens, might fit; Message 0 is oldest
    result = truncate_messages(messages, None, max_tokens=400, max_output_tokens=50)
    assert len(result) < 3
    # Newest message must be preserved
    assert result[-1]["content"] == "C" * 100


def test_truncate_messages_raises_if_impossible():
    messages = [{"role": "user", "content": "A" * 10000}]
    with pytest.raises(ValueError):
        truncate_messages(messages, None, max_tokens=10, max_output_tokens=5)
