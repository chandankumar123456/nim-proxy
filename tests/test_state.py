"""
tests/test_state.py
====================
Unit tests for per-request state management.
"""
import pytest

from router.state import RequestState


def test_initial_state():
    s = RequestState()
    assert s.retry_count == 0
    assert s.providers_used == []
    assert s.current_provider == "ollama"
    assert s.request_id  # UUID string


def test_mark_provider_used():
    s = RequestState()
    s.mark_provider_used("ollama")
    assert "ollama" in s.providers_used
    assert s.current_provider == "ollama"
    # Should not duplicate
    s.mark_provider_used("ollama")
    assert s.providers_used.count("ollama") == 1


def test_can_retry():
    s = RequestState()
    s.mark_provider_used("ollama")
    assert s.can_retry(max_retries=2)
    s.record_failure("timeout")
    assert s.can_retry(max_retries=2)
    s.mark_provider_used("nvidia")
    # Both providers used → no more retries
    assert not s.can_retry(max_retries=2)


def test_can_retry_max_retries_exhausted():
    s = RequestState()
    s.record_failure("err1")
    s.record_failure("err2")
    assert not s.can_retry(max_retries=2)


def test_next_provider_sequence():
    s = RequestState()
    # Start: no providers used → should suggest ollama
    assert s.next_provider() == "ollama"
    s.mark_provider_used("ollama")
    # Ollama used → suggest nvidia
    assert s.next_provider() == "nvidia"
    s.mark_provider_used("nvidia")
    # Both used → no more providers
    assert s.next_provider() is None


def test_record_failure():
    s = RequestState()
    s.record_failure("timeout")
    assert s.retry_count == 1
    assert "timeout" in s.failure_reasons
