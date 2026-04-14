"""
tests/test_models.py
=====================
Unit tests for the model mapping utility.
"""
import pytest

from utils.models import get_provider_model


def test_claude_model_maps_to_ollama():
    result = get_provider_model("claude-3-5-sonnet-20241022", "ollama")
    assert isinstance(result, str)
    assert len(result) > 0


def test_claude_model_maps_to_nvidia():
    result = get_provider_model("claude-3-5-sonnet-20241022", "nvidia")
    assert isinstance(result, str)
    assert len(result) > 0


def test_unknown_model_falls_back_to_defaults():
    result_ollama = get_provider_model("unknown-model-xyz", "ollama")
    result_nvidia = get_provider_model("unknown-model-xyz", "nvidia")
    assert isinstance(result_ollama, str)
    assert isinstance(result_nvidia, str)


def test_case_insensitive_mapping():
    r1 = get_provider_model("Claude-3-Opus", "ollama")
    r2 = get_provider_model("claude-3-opus", "ollama")
    assert r1 == r2
