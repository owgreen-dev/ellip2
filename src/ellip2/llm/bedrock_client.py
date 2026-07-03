"""Stage 4 (LLM layer) — a thin wrapper over Amazon Bedrock's Converse API.

plan.md §8: each Stage 3 candidate (serialized by
:mod:`ellip2.llm.serialize_subgraph`) is handed to a foundation model that
classifies its money-laundering *typology* (peeling chain, nested service,
layering / smurfing, consolidation) and returns a structured verdict. This
module owns the I/O boundary to Bedrock and the parsing of its response into a
small, typed result — :class:`TypologyResult` ``{typology, confidence,
rationale, evidence}``.

Two design constraints (SIGN-101 — tests are offline / CPU / mocked):

* **The boto3 client is injectable.** :class:`BedrockTypologyClient` accepts an
  already-constructed ``bedrock-runtime`` client, so a test can pass a stub that
  returns a canned Converse payload. A real client is built *lazily* and only
  when none was injected, so importing this module — and exercising it with a
  stub — never touches the network, AWS credentials, or even ``boto3`` itself.
* **Parsing is a pure function.** :func:`parse_converse_response` turns a Bedrock
  Converse response dict into a :class:`TypologyResult` with no I/O, so the
  whole response-handling path is unit-testable from a literal payload.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Default Bedrock model id (a configurable string — never invoked in tests). Uses the
#: cross-region *inference profile* id: the bare foundation-model id
#: ("anthropic.claude-sonnet-4-6") rejects on-demand Converse and demands a profile.
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
#: Default AWS region for the bedrock-runtime client.
DEFAULT_REGION = "us-east-1"
#: The four AML typologies the classifier is asked to choose from (plan.md §8).
TYPOLOGIES = (
    "peeling_chain",
    "nested_service",
    "layering_smurfing",
    "consolidation",
)

#: The system prompt steering the model to emit a strict JSON verdict.
DEFAULT_SYSTEM_PROMPT = (
    "You are a blockchain anti-money-laundering analyst. You are given a compact "
    "JSON description of a suspicious Bitcoin transaction subgraph (nodes with "
    "heuristic roles and binned features, directed edges, an exit path to a "
    "heuristic licit endpoint, structural statistics, and a PU suspicion score). "
    "Classify its laundering typology as one of: "
    f"{', '.join(TYPOLOGIES)}. "
    "Respond with ONLY a JSON object with keys: "
    '"typology" (one of the listed values), '
    '"confidence" (a number in [0, 1]), '
    '"rationale" (a short string), and '
    '"evidence" (a JSON array of short strings citing structural signals). '
    "Do not include any prose outside the JSON object."
)

#: Required keys in the model's JSON verdict.
_REQUIRED_KEYS = ("typology", "confidence", "rationale", "evidence")


@dataclass(frozen=True)
class BedrockConfig:
    """Configuration for the Converse request.

    Attributes:
        model_id: the Bedrock model id (a string; never invoked under test).
        region: AWS region used only when a default client must be built.
        max_tokens: ``maxTokens`` inference cap.
        temperature: sampling temperature (low for reproducible verdicts).
        system_prompt: the system steer prepended to the request.
    """

    model_id: str = DEFAULT_MODEL_ID
    region: str = DEFAULT_REGION
    max_tokens: int = 1024
    temperature: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class TypologyResult:
    """A parsed Converse verdict.

    Attributes:
        typology: the chosen typology label.
        confidence: model confidence, clamped to ``[0, 1]``.
        rationale: the model's free-text justification.
        evidence: structural signals the model cited.
    """

    typology: str
    confidence: float
    rationale: str
    evidence: tuple[str, ...] = field(default_factory=tuple)


def build_converse_request(
    prompt: str, config: BedrockConfig | None = None
) -> dict[str, Any]:
    """Build the keyword arguments for a ``bedrock-runtime`` ``converse`` call.

    Pure function: returns the request dict (``modelId`` / ``messages`` /
    ``system`` / ``inferenceConfig``) without performing any I/O.
    """
    cfg = config if config is not None else BedrockConfig()
    return {
        "modelId": cfg.model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "system": [{"text": cfg.system_prompt}],
        "inferenceConfig": {
            "maxTokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        },
    }


def extract_text(response: Mapping[str, Any]) -> str:
    """Concatenate the text blocks of a Bedrock Converse response.

    A Converse response has the shape
    ``{"output": {"message": {"content": [{"text": ...}, ...]}}}``; non-text
    blocks (e.g. tool use) are ignored.

    Raises:
        ValueError: if the response is missing the ``output/message/content``
            path or contains no text block.
    """
    try:
        content = response["output"]["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed Converse response: {exc}") from exc
    if not isinstance(content, Sequence):
        raise ValueError("Converse response content is not a sequence")
    texts = [
        block["text"]
        for block in content
        if isinstance(block, Mapping) and "text" in block
    ]
    if not texts:
        raise ValueError("Converse response contained no text block")
    return "".join(str(t) for t in texts)


def _strip_json_fence(text: str) -> str:
    """Return the JSON object substring, tolerating ```json fences / prose."""
    s = text.strip()
    if s.startswith("```"):
        # Drop the opening fence line (``` or ```json) and any trailing fence.
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[: -len("```")]
        s = s.strip()
    # Fall back to the outermost {...} span if there is surrounding prose.
    if not s.startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    return s


def _coerce_confidence(value: Any) -> float:
    """Parse ``confidence`` to a float clamped to ``[0, 1]``."""
    try:
        conf = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"confidence is not a number: {value!r}") from exc
    return max(0.0, min(1.0, conf))


def _coerce_evidence(value: Any) -> tuple[str, ...]:
    """Normalise ``evidence`` to a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def parse_converse_response(response: Mapping[str, Any]) -> TypologyResult:
    """Parse a Bedrock Converse response into a :class:`TypologyResult`.

    Pure function (no I/O): extracts the assistant text, decodes the JSON verdict
    (tolerating markdown fences / surrounding prose), and validates the required
    keys.

    Raises:
        ValueError: if the response is malformed, the text is not valid JSON, or
            a required key (``typology`` / ``confidence`` / ``rationale`` /
            ``evidence``) is missing.
    """
    text = extract_text(response)
    try:
        verdict = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Converse text is not valid JSON: {exc}") from exc
    if not isinstance(verdict, Mapping):
        raise ValueError("Converse JSON is not an object")
    missing = [k for k in _REQUIRED_KEYS if k not in verdict]
    if missing:
        raise ValueError(f"Converse verdict missing keys: {missing}")
    return TypologyResult(
        typology=str(verdict["typology"]),
        confidence=_coerce_confidence(verdict["confidence"]),
        rationale=str(verdict["rationale"]),
        evidence=_coerce_evidence(verdict["evidence"]),
    )


def build_default_client(config: BedrockConfig) -> Any:
    """Construct a real ``bedrock-runtime`` boto3 client.

    Imported lazily so that injecting a stub client never imports ``boto3`` or
    requires AWS credentials (SIGN-101). Not exercised in the test suite.
    """
    import boto3  # noqa: PLC0415 — lazy import keeps the offline path clean

    return boto3.client("bedrock-runtime", region_name=config.region)


class BedrockTypologyClient:
    """Classify a serialized subgraph's AML typology via Bedrock Converse.

    The ``bedrock-runtime`` client is *injectable*: pass one to ``client`` to use
    a stub (tests) or a pre-configured real client. When ``None``, a real client
    is built lazily on first use — so constructing this object, and calling
    :meth:`classify` with an injected stub, performs no network or AWS work.
    """

    def __init__(
        self,
        client: Any | None = None,
        config: BedrockConfig | None = None,
    ) -> None:
        self._client = client
        self.config = config if config is not None else BedrockConfig()

    @property
    def client(self) -> Any:
        """The bedrock-runtime client, building a real one lazily if needed."""
        if self._client is None:
            self._client = build_default_client(self.config)
        return self._client

    def build_prompt(self, subgraph: str | Mapping[str, Any]) -> str:
        """Render the serialized subgraph as the user-message prompt text."""
        if isinstance(subgraph, str):
            return subgraph
        return json.dumps(subgraph, sort_keys=True, separators=(",", ":"))

    def classify(self, subgraph: str | Mapping[str, Any]) -> TypologyResult:
        """Send the serialized subgraph to Converse and parse the verdict.

        Args:
            subgraph: a serialized subgraph — either the compact JSON string from
                :func:`ellip2.llm.serialize_subgraph.serialize_subgraph_json` or
                the JSON-able dict from
                :func:`ellip2.llm.serialize_subgraph.serialize_subgraph`.

        Returns:
            The parsed :class:`TypologyResult`.
        """
        request = build_converse_request(self.build_prompt(subgraph), self.config)
        response = self.client.converse(**request)
        return parse_converse_response(response)
