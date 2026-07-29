"""Fast standalone test for the embedding validator retry/recovery path.

This test does NOT use the heavy project conftest — it imports only the
validator and the function under test, then drives the retry path with a
mocked engine. Runs in <100ms; no test DB needed.

Regression scope: covers the bug class we hit on 2026-07-29 when the
deriver crashed because the DB wasn't up yet at boot time. The
"always-raises" case was already tested; this covers the
"intermittent-failure-then-success" recovery path that the existing
test_fixture does not exercise.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError

from src.startup.embedding_validator import (
    StartupValidationError,
    validate_embedding_schema,
)


def _op_error(msg: str = "DB unreachable") -> OperationalError:
    return OperationalError("SELECT 1", {}, Exception(msg))


async def test_validator_recovers_when_introspection_succeeds_after_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two transient DB failures followed by a success should NOT raise.

    The validator's contract is "fail closed after N failures" — but if
    introspection recovers within the retry budget, it should accept
    the recovered state. This case was not covered by the existing
    always-raise test and was the silent failure mode we hit when the
    deriver was racing the DB at boot.
    """
    call_count = 0

    async def fail_twice_then_succeed(
        _engine: object, _schema: str
    ) -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise _op_error()
        return {"documents": 1536, "message_embeddings": 1536}

    monkeypatch.setattr(
        "src.startup.embedding_validator._introspect_pgvector_dims_once",
        fail_twice_then_succeed,
    )
    # Make backoff effectively instant.
    monkeypatch.setattr("src.startup.embedding_validator._RETRY_BACKOFF_SECONDS", 0.0)
    # Bypass the dim assertion so the test only exercises the retry/recovery path.
    monkeypatch.setattr(
        "src.startup.embedding_validator._assert_pgvector_dims_match",
        lambda *_a, **_kw: None,
    )

    await validate_embedding_schema(engine=AsyncMock())

    assert call_count == 3, "should have called the int exactly 3 times (2 fail + 1 ok)"


async def test_validator_logs_each_retry_with_attempt_number(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every retry attempt should be logged with the attempt number so
    operators can see the deriver is recovering vs. stuck.
    """
    call_count = 0

    async def fail_twice_then_succeed(
        _engine: object, _schema: str
    ) -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise _op_error()
        return {"documents": 1536, "message_embeddings": 1536}

    monkeypatch.setattr(
        "src.startup.embedding_validator._introspect_pgvector_dims_once",
        fail_twice_then_succeed,
    )
    monkeypatch.setattr("src.startup.embedding_validator._RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(
        "src.startup.embedding_validator._assert_pgvector_dims_match",
        lambda *_a, **_kw: None,
    )

    import logging
    with caplog.at_level(logging.INFO, logger="src.startup.embedding_validator"):
        await validate_embedding_schema(engine=AsyncMock())

    retry_logs = [r for r in caplog.records if "retry" in r.message.lower()]
    assert len(retry_logs) >= 2, (
        f"expected at least 2 retry log lines, got {len(retry_logs)}: "
        f"{[r.message for r in caplog.records]}"
    )


async def test_validator_uses_correct_max_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry budget must be EXACTLY 3 attempts (not 2, not 4).

    Regression: a change to the retry-budget constant must not silently
    weaken the fail-closed guarantee.
    """
    from src.startup import embedding_validator as ev_module

    # Pull the source constant. If this changes, this test fails loudly.
    assert ev_module._RETRY_ATTEMPTS == 3, (
        f"retry budget changed from 3 to {ev_module._RETRY_ATTEMPTS}; "
        "update this test and reconsider the fail-closed guarantee."
    )
