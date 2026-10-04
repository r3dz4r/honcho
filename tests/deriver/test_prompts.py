import json
from unittest.mock import patch

import pytest

from src.deriver.prompts import (
    estimate_deriver_prompt_tokens,
    estimate_minimal_deriver_prompt_tokens,
    minimal_deriver_prompt,
)
from src.utils.representation import PromptRepresentation


def test_minimal_deriver_prompt_includes_custom_instructions_when_present() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        custom_instructions="Prefer concrete timeline facts.",
    )

    assert "CUSTOM INSTRUCTIONS:" in prompt
    assert "Prefer concrete timeline facts." in prompt


def test_minimal_deriver_prompt_omits_custom_instructions_when_absent() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        custom_instructions=None,
    )

    assert "CUSTOM INSTRUCTIONS:" not in prompt


def test_estimate_deriver_prompt_tokens_increases_with_custom_instructions() -> None:
    base_tokens = estimate_minimal_deriver_prompt_tokens()
    custom_tokens = estimate_deriver_prompt_tokens(
        "Prefer explicit facts with absolute dates and keep the subject precise."
    )

    assert custom_tokens > base_tokens


def test_estimate_deriver_prompt_tokens_propagates_token_estimation_errors() -> None:
    estimate_minimal_deriver_prompt_tokens.cache_clear()

    with patch(
        "src.deriver.prompts.estimate_tokens",
        side_effect=RuntimeError("tokenizer unavailable"),
    ):
        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            estimate_deriver_prompt_tokens(None)

        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            estimate_deriver_prompt_tokens("Prefer concrete facts.")


def _cache_stable_prefix(prompt: str) -> str:
    """Return the prompt slice that must stay byte-identical across peers.

    Providers cache a byte-identical prompt prefix across requests, so nothing
    in this slice may depend on the peer id.
    """
    # The rules mention `Target peer:` in backticks; the section header is the
    # only bare occurrence. Anchor on the bare line so we slice the real section.
    marker = "\nTarget peer:\n"
    assert marker in prompt, "deriver prompt lost its Target peer section"
    return prompt.split(marker, 1)[0]


def test_deriver_prompt_cache_stable_prefix_is_peer_independent() -> None:
    """The cache-stable prefix must be byte-identical across peer ids.

    If a peer id leaks anywhere before the `Target peer:` section, every peer
    renders a different prefix and the provider's prompt cache misses for all
    of them, so memory-formation cost scales with the number of peers.
    """
    alice_prompt = minimal_deriver_prompt(
        peer_id="alice", messages="", custom_instructions=None
    )
    bob_prompt = minimal_deriver_prompt(
        peer_id="bob", messages="", custom_instructions=None
    )

    alice_prefix = _cache_stable_prefix(alice_prompt)
    bob_prefix = _cache_stable_prefix(bob_prompt)

    # The cacheable prefix is identical no matter who the target peer is.
    assert alice_prefix == bob_prefix

    # No peer id appears before the marker -- peer identity lives only after it.
    for prefix in (alice_prefix, bob_prefix):
        assert "alice" not in prefix
        assert "bob" not in prefix

    assert "Target peer:\nalice\n" in alice_prompt
    assert "Target peer:\nbob\n" in bob_prompt


def test_minimal_deriver_prompt_carries_no_example_facts() -> None:
    """Few-shot facts left the scaffold; the model must read the batch instead.

    The worked examples taught concrete facts (including the example target
    peer's id) that the deriver then echoed. Those facts are gone from both the
    rendered prompt and the structured-output schema, while the
    fabricated-example guard survives.
    """
    prompt = minimal_deriver_prompt(peer_id="", messages="", custom_instructions=None)
    schema = json.dumps(PromptRepresentation.model_json_schema())

    assert "→" not in prompt
    assert "fabricated illustrations" in prompt
    for legacy in ("dog", "NYC", "25 years", "six years", "alice", "Rover", "Ann "):
        assert legacy not in prompt
        assert legacy not in schema
