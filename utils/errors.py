"""
utils/errors.py
================
Helpers for producing valid Anthropic-format error responses.

Anthropic error structure:
  {
    "type": "error",
    "error": {
      "type": "<error_type>",
      "message": "<human readable message>"
    }
  }

Standard error types:
  invalid_request_error  — 400
  authentication_error   — 401
  permission_error       — 403
  not_found_error        — 404
  rate_limit_error       — 429
  api_error              — 500
  overloaded_error       — 529
"""

from __future__ import annotations

from typing import Any, Dict


def anthropic_error(error_type: str, message: str) -> Dict[str, Any]:
    """Return a dict that matches the Anthropic error envelope."""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


# Convenience constructors -------------------------------------------------

def invalid_request(message: str) -> Dict[str, Any]:
    return anthropic_error("invalid_request_error", message)


def auth_error(message: str = "Invalid API key") -> Dict[str, Any]:
    return anthropic_error("authentication_error", message)


def rate_limit_error(message: str = "Rate limit exceeded") -> Dict[str, Any]:
    return anthropic_error("rate_limit_error", message)


def overloaded_error(message: str = "System overloaded, please retry later") -> Dict[str, Any]:
    return anthropic_error("overloaded_error", message)


def api_error(message: str = "Internal server error") -> Dict[str, Any]:
    return anthropic_error("api_error", message)


# HTTP status codes that map to Anthropic error types ----------------------

HTTP_TO_ANTHROPIC_ERROR: Dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "api_error",
    529: "overloaded_error",
}


def from_http_status(status_code: int, message: str) -> Dict[str, Any]:
    error_type = HTTP_TO_ANTHROPIC_ERROR.get(status_code, "api_error")
    return anthropic_error(error_type, message)
