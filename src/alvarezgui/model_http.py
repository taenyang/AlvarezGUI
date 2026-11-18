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
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raw_output = output_format = None
            if detail:
                try:
                    body = json.loads(detail)
                    if isinstance(body, dict):
                        detail = str(body.get("detail", detail))
                        raw_output = body.get("raw_output")
                        output_format = body.get("output_format")
                except json.JSONDecodeError:
                    pass
            suffix = f": {detail[:300]}" if detail else ""
            raise InvalidActionOutputError(
                f"model service returned HTTP {exc.code}{suffix}",
                raw_output=raw_output[:16000] if isinstance(raw_output, str) else None,
                output_format=output_format if isinstance(output_format, str) else None,
            ) from exc
        except TimeoutError as exc:
            raise InferenceTimeoutError(request.request_id) from exc
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError(request.model_version) from exc
        try:
            return InferenceResponse.model_validate(body)
        except ValueError as exc:
            raise InvalidActionOutputError("model service response is invalid") from exc
