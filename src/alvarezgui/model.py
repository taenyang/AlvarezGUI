from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from alvarezgui.action import validate_action


class InferenceRequest(BaseModel):
    request_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    image_uri: str = Field(min_length=1)
    ui_tree_uri: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=4_096)
    timeout_ms: int = Field(default=30_000, ge=1, le=300_000)


class InferenceResponse(BaseModel):
    request_id: str
    action: dict[str, Any]
    raw_output: str
    model_version: str
    latency_ms: int = Field(ge=0)
    output_format: Literal["action-v1", "sokoban-plan-v1"] = "action-v1"
    # Mean per-token entropy (nats) of the distribution each output token was sampled from.
    entropy: float | None = Field(default=None, ge=0)


class ModelError(Exception):
    pass


class ModelUnavailableError(ModelError):
    pass


class ModelVersionNotFoundError(ModelError):
    pass


class InferenceTimeoutError(ModelError):
    pass


class InvalidActionOutputError(ModelError):
    def __init__(self, message: str, *, raw_output: str | None = None,
                 output_format: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.output_format = output_format


class ModelService(Protocol):
    def infer(self, request: InferenceRequest) -> InferenceResponse: ...


@dataclass(slots=True)
class FakeModelService:
    """Deterministic model boundary used until MiniCPM-V is deployed."""

    supported_versions: set[str] | None = None

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        started = time.monotonic()
        if self.supported_versions is not None and request.model_version not in self.supported_versions:
            raise ModelVersionNotFoundError(request.model_version)
        action = {"type": "terminate", "reason": "fake_model"}
        try:
            action = validate_action(action)
        except ValueError as exc:
            raise InvalidActionOutputError(str(exc)) from exc
        return InferenceResponse(
            request_id=request.request_id,
            action=action,
            raw_output='{"type":"terminate","reason":"fake_model"}',
            model_version=request.model_version,
            latency_ms=max(0, round((time.monotonic() - started) * 1000)),
        )
