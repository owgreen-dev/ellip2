"""Tests for ellip2.llm.bedrock_client (T-019).

Synthetic / offline / mocked (SIGN-101): a STUB bedrock-runtime client returns a
canned Converse payload. No real boto3 client is constructed, no network, no AWS
credentials. The load-bearing properties: response parsing into
``{typology, confidence, rationale, evidence}`` and client injectability.
"""

from __future__ import annotations

from typing import Any

import pytest

import ellip2.llm.bedrock_client as bc
from ellip2.llm.bedrock_client import (
    BedrockConfig,
    BedrockTypologyClient,
    TypologyResult,
    build_converse_request,
    extract_text,
    parse_converse_response,
)


def _converse_payload(text: str) -> dict[str, Any]:
    """A minimal Bedrock Converse response wrapping ``text``."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 20},
    }


_VERDICT_JSON = (
    '{"typology": "peeling_chain", "confidence": 0.82, '
    '"rationale": "Sequential fan-out with shrinking transfers toward an exchange.", '
    '"evidence": ["high in_out_ratio at relays", "monotone decreasing edge weights"]}'
)


class _StubClient:
    """A stand-in bedrock-runtime client recording the request it receives."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.payload


def test_parse_converse_response_extracts_fields() -> None:
    result = parse_converse_response(_converse_payload(_VERDICT_JSON))
    assert isinstance(result, TypologyResult)
    assert result.typology == "peeling_chain"
    assert result.confidence == 0.82
    assert result.rationale.startswith("Sequential fan-out")
    assert result.evidence == (
        "high in_out_ratio at relays",
        "monotone decreasing edge weights",
    )


def test_classify_uses_injected_client_and_parses() -> None:
    stub = _StubClient(_converse_payload(_VERDICT_JSON))
    client = BedrockTypologyClient(client=stub, config=BedrockConfig(model_id="m-123"))
    result = client.classify({"subgraph_id": 7, "pu_score": 0.9})

    assert result.typology == "peeling_chain"
    # The injected stub was the one called, exactly once, with our model id.
    assert len(stub.calls) == 1
    assert stub.calls[0]["modelId"] == "m-123"
    assert stub.calls[0]["messages"][0]["role"] == "user"


def test_no_default_boto3_client_constructed(monkeypatch: pytest.MonkeyPatch) -> None:
    # If a default client is ever built, this blows up — proving the injected
    # stub path never touches boto3 / AWS.
    def _boom(_config: BedrockConfig) -> Any:
        raise AssertionError("build_default_client must not be called under test")

    monkeypatch.setattr(bc, "build_default_client", _boom)

    stub = _StubClient(_converse_payload(_VERDICT_JSON))
    client = BedrockTypologyClient(client=stub)
    # Accessing .client returns the injected stub without constructing one.
    assert client.client is stub
    result = client.classify("serialized-subgraph-string")
    assert result.typology == "peeling_chain"


def test_classify_accepts_json_string_passthrough() -> None:
    stub = _StubClient(_converse_payload(_VERDICT_JSON))
    client = BedrockTypologyClient(client=stub)
    client.classify('{"already":"serialized"}')
    sent = stub.calls[0]["messages"][0]["content"][0]["text"]
    assert sent == '{"already":"serialized"}'


def test_build_converse_request_shape() -> None:
    req = build_converse_request("hello", BedrockConfig(model_id="x", max_tokens=256))
    assert req["modelId"] == "x"
    assert req["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    assert req["system"][0]["text"]  # non-empty system steer
    assert req["inferenceConfig"]["maxTokens"] == 256
    assert req["inferenceConfig"]["temperature"] == 0.0


def test_parse_tolerates_markdown_fence() -> None:
    fenced = f"```json\n{_VERDICT_JSON}\n```"
    result = parse_converse_response(_converse_payload(fenced))
    assert result.typology == "peeling_chain"


def test_parse_tolerates_surrounding_prose() -> None:
    text = f"Here is my analysis:\n{_VERDICT_JSON}\nLet me know if you need more."
    result = parse_converse_response(_converse_payload(text))
    assert result.confidence == 0.82


def test_confidence_clamped_to_unit_interval() -> None:
    payload = _converse_payload(
        '{"typology": "consolidation", "confidence": 1.7, '
        '"rationale": "r", "evidence": []}'
    )
    assert parse_converse_response(payload).confidence == 1.0

    payload_neg = _converse_payload(
        '{"typology": "consolidation", "confidence": -0.5, '
        '"rationale": "r", "evidence": []}'
    )
    assert parse_converse_response(payload_neg).confidence == 0.0


def test_evidence_string_normalised_to_tuple() -> None:
    payload = _converse_payload(
        '{"typology": "layering_smurfing", "confidence": 0.5, '
        '"rationale": "r", "evidence": "single signal"}'
    )
    assert parse_converse_response(payload).evidence == ("single signal",)


def test_missing_required_key_raises() -> None:
    payload = _converse_payload(
        '{"typology": "nested_service", "confidence": 0.5, "rationale": "r"}'
    )
    with pytest.raises(ValueError, match="missing keys"):
        parse_converse_response(payload)


def test_non_json_text_raises() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_converse_response(_converse_payload("not json at all"))


def test_malformed_response_raises() -> None:
    with pytest.raises(ValueError, match="malformed Converse"):
        extract_text({"output": {"message": {}}})


def test_no_text_block_raises() -> None:
    payload = {"output": {"message": {"content": [{"toolUse": {}}]}}}
    with pytest.raises(ValueError, match="no text block"):
        extract_text(payload)
