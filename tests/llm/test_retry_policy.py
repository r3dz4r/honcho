"""Regression coverage for LLM retry classification."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.config import ModelConfig
from src.llm.api import honcho_llm_call
from src.llm.retry import is_retryable_llm_error
from src.llm.runtime import AttemptPlan
from src.llm.types import HonchoLLMCallResponse


class ProviderError(Exception):
    def __init__(self, status_code: int, message: str = "provider error") -> None:
        super().__init__(message)
        self.status_code = status_code


def test_openrouter_credit_exhaustion_is_not_retryable() -> None:
    assert not is_retryable_llm_error(
        ProviderError(402, "OpenRouter credit balance is too low")
    )


def test_transient_provider_failures_remain_retryable() -> None:
    assert is_retryable_llm_error(TimeoutError("timed out"))
    assert is_retryable_llm_error(ProviderError(429, "rate limited"))


def _attempt_plan() -> AttemptPlan:
    config = ModelConfig(model="test-model", transport="openai")
    return AttemptPlan(
        provider="openai",
        model=config.model,
        client=object(),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=config,
        attempt=1,
        retry_attempts=2,
        is_fallback=False,
    )


@pytest.mark.asyncio
async def test_budget_exhaustion_fails_once_without_consuming_retry_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider_error = ProviderError(402, "OpenRouter credit balance is too low")
    call = AsyncMock(side_effect=provider_error)

    with (
        patch("src.llm.api.plan_attempt", return_value=_attempt_plan()),
        patch("src.llm.api.honcho_llm_call_inner", new=call),
        pytest.raises(ProviderError, match="credit balance"),
    ):
        await honcho_llm_call(
            model_config=ModelConfig(model="test-model", transport="openai"),
            prompt="hello",
            max_tokens=32,
            retry_attempts=2,
        )

    assert call.await_count == 1
    assert "Permanent LLM budget failure (HTTP 402)" in caplog.text


@pytest.mark.asyncio
async def test_transient_failure_retries_within_configured_budget() -> None:
    call = AsyncMock(
        side_effect=[
            TimeoutError("timed out"),
            HonchoLLMCallResponse(content="ok", output_tokens=0, finish_reasons=[]),
        ]
    )

    with (
        patch("src.llm.api.plan_attempt", return_value=_attempt_plan()),
        patch("src.llm.api.honcho_llm_call_inner", new=call),
        patch("src.llm.api.wait_exponential", return_value=lambda _: 0),
    ):
        response = await honcho_llm_call(
            model_config=ModelConfig(model="test-model", transport="openai"),
            prompt="hello",
            max_tokens=32,
            retry_attempts=2,
        )

    assert response.content == "ok"
    assert call.await_count == 2
