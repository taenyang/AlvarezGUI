from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from alvarezgui.model import (
    InferenceRequest,
    InferenceResponse,
    InvalidActionOutputError,
    InferenceTimeoutError,
    ModelService,
    ModelUnavailableError,
    ModelVersionNotFoundError,
)


class HttpModelService(ModelService):
    """HTTP client for the independently deployed MiniCPM-V service."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        payload = json.dumps(request.model_dump(mode="json")).encode()
        http_request = Request(
            f"{self.base_url}/internal/v1/model/infer",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=min(self.timeout, request.timeout_ms / 1000)) as response:
                body = json.loads(response.read())
        except HTTPError as exc:
            if exc.code == 404:
                raise ModelVersionNotFoundError(request.model_version) from exc
            if exc.code == 504:
                raise InferenceTimeoutError(request.request_id) from exc
            if exc.code >= 500:
                raise ModelUnavailableError(request.model_version) from exc
            raise InvalidActionOutputError(f"model service returned HTTP {exc.code}") from exc
        except TimeoutError as exc:
            raise InferenceTimeoutError(request.request_id) from exc
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError(request.model_version) from exc
        try:
            return InferenceResponse.model_validate(body)
        except ValueError as exc:
            raise InvalidActionOutputError("model service response is invalid") from exc
