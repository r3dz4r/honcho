"""Regression coverage for LLM retry classification."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.config import ModelConfig, settings
from src.llm import executor
from src.llm.api import honcho_llm_call
from src.llm.retry import is_retryable_llm_error
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
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


def test_cancellation_is_not_retryable() -> None:
    assert not is_retryable_llm_error(asyncio.CancelledError())


@pytest.mark.asyncio
async def test_cancellation_propagates_without_another_provider_attempt() -> None:
    call = AsyncMock(side_effect=asyncio.CancelledError())

    with (
        patch("src.llm.api.plan_attempt", return_value=_attempt_plan()),
        patch("src.llm.api.honcho_llm_call_inner", new=call),
        pytest.raises(asyncio.CancelledError),
    ):
        await honcho_llm_call(
            model_config=ModelConfig(model="test-model", transport="openai"),
            prompt="hello",
            max_tokens=32,
            retry_attempts=2,
        )

    assert call.await_count == 1


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


@pytest.mark.asyncio
async def test_provider_call_times_out_within_configured_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled provider attempt must not await indefinitely."""

    async def stall(*_args: object, **_kwargs: object) -> None:
        await asyncio.Event().wait()

    call = AsyncMock(side_effect=stall)
    monkeypatch.setattr(settings.LLM, "REQUEST_TIMEOUT_SECONDS", 0.01)

    with (
        patch.object(executor, "backend_for_provider", return_value=object()),
        patch.object(executor, "execute_completion", new=call),
        pytest.raises(TimeoutError),
    ):
        await executor.honcho_llm_call_inner(
            "openai", "test-model", "hello", 32, client_override=object()
        )

    assert call.await_count == 1


@pytest.mark.asyncio
async def test_tool_loop_provider_call_uses_central_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stall(*_args: object, **_kwargs: object) -> None:
        await asyncio.Event().wait()

    call = AsyncMock(side_effect=stall)
    monkeypatch.setattr(settings.LLM, "REQUEST_TIMEOUT_SECONDS", 0.01)

    with (
        patch.object(executor, "backend_for_provider", return_value=object()),
        patch.object(executor, "execute_completion", new=call),
        pytest.raises(TimeoutError),
    ):
        await execute_tool_loop(
            prompt="hello",
            max_tokens=32,
            messages=None,
            tools=[{"type": "function", "function": {"name": "noop"}}],
            tool_choice=None,
            tool_executor=AsyncMock(),
            max_tool_iterations=1,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=None,
            get_attempt_plan=_attempt_plan,
            before_retry_callback=lambda _state: None,
        )

    assert call.await_count == 1


@pytest.mark.asyncio
async def test_stalled_stream_drain_uses_total_provider_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stalled_stream() -> asyncio.AsyncIterator[object]:
        await asyncio.Event().wait()
        yield object()

    async def start_stream(*_args: object, **_kwargs: object) -> object:
        return stalled_stream()

    monkeypatch.setattr(settings.LLM, "REQUEST_TIMEOUT_SECONDS", 0.01)

    with (
        patch.object(executor, "backend_for_provider", return_value=object()),
        patch.object(executor, "execute_stream", new=start_stream),
        pytest.raises(TimeoutError),
    ):
        stream = await executor.honcho_llm_call_inner(
            "openai", "test-model", "hello", 32, stream=True, client_override=object()
        )
        await anext(stream)
