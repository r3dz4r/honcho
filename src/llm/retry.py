"""Retry classification shared by all LLM call paths."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_PERMANENT_BUDGET_STATUS_CODES = frozenset({402})


def _http_status_code(error: BaseException) -> int | None:
    """Extract a provider HTTP status without depending on one SDK's error type."""
    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def is_retryable_llm_error(error: BaseException) -> bool:
    """Return whether a failed provider call may safely consume a retry attempt.

    Provider SDKs expose HTTP status inconsistently, so this intentionally uses
    duck typing rather than importing provider-specific error classes. A 402 is
    a permanent billing/credit failure (including OpenRouter credit exhaustion)
    and must surface immediately; other errors retain the established retry
    behavior and fallback semantics.
    """
    if isinstance(error, asyncio.CancelledError):
        return False

    status_code = _http_status_code(error)
    if status_code in _PERMANENT_BUDGET_STATUS_CODES:
        logger.warning(
            "Permanent LLM budget failure (HTTP %s); failing fast without retry",
            status_code,
        )
        return False
    return True


__all__ = ["is_retryable_llm_error"]
