from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from alvarezgui.action import validate_action
from alvarezgui.model import (
    InferenceRequest,
    InferenceResponse,
    InvalidActionOutputError,
    ModelVersionNotFoundError,
)


class MiniCPMDependencyError(RuntimeError):
    pass


def _read_image(uri: str, timeout: float) -> Any:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        payload = Path(unquote(parsed.path)).read_bytes()
    elif parsed.scheme in {"http", "https"}:
        request = Request(uri, headers={"Accept": "image/png,image/*"})
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    elif parsed.scheme == "data":
        header, encoded = uri.split(",", 1)
        if ";base64" not in header:
            raise ValueError("data image URI must use base64")
        payload = base64.b64decode(encoded, validate=True)
    else:
        raise ValueError("image_uri must use file, http, https, or data scheme")
    from PIL import Image

    return Image.open(BytesIO(payload)).convert("RGB")


def _parse_action(raw_output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates = [raw_output.strip()]
    candidates.extend(
        raw_output[index:]
        for index, char in enumerate(raw_output)
        if char == "{"
    )
    for candidate in candidates:
        try:
            value, end = decoder.raw_decode(candidate)
            if candidate[end:].strip():
                continue
            return validate_action(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    raise InvalidActionOutputError("MiniCPM output did not contain a valid Android action")


@dataclass(slots=True)
class MiniCPMService:
    model: Any
    processor: Any
    model_version: str
    generation_lock: threading.Lock
    torch: Any

    @classmethod
    def load(cls, model_path: str | Path, model_version: str) -> "MiniCPMService":
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise MiniCPMDependencyError(
                "real MiniCPM serving requires torch, torchvision and transformers>=5.18"
            ) from exc
        processor = AutoProcessor.from_pretrained(str(model_path))
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        return cls(model, processor, model_version, threading.Lock(), torch)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if request.model_version != self.model_version:
            raise ModelVersionNotFoundError(request.model_version)
        started = time.monotonic()
        image = _read_image(request.image_uri, request.timeout_ms / 1000)
        instruction = (
            f"{request.instruction}\n"
            "Return exactly one JSON object and nothing else. Use exactly one of these schemas: "
            '{"type":"tap","x":integer,"y":integer}, '
            '{"type":"swipe","x1":integer,"y1":integer,"x2":integer,"y2":integer,"duration_ms":integer}, '
            '{"type":"text","text":"..."}, {"type":"back"}, '
            '{"type":"wait","duration_ms":integer}, or {"type":"terminate","reason":"..."}. '
            "Never output a key named action, an action list, parentheses, Markdown, or explanation."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        with self.generation_lock:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                downsample_mode="16x",
                max_slice_nums=36,
                enable_thinking=False,
            ).to(self.model.device)
            generation_args: dict[str, Any] = {
                "max_new_tokens": request.max_tokens,
                "downsample_mode": "16x",
                "do_sample": request.temperature > 0,
            }
            if request.temperature > 0:
                generation_args["temperature"] = request.temperature
            with self.torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **generation_args)
            trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_output = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        action = _parse_action(raw_output)
        return InferenceResponse(
            request_id=request.request_id,
            action=action,
            raw_output=raw_output,
            model_version=self.model_version,
            latency_ms=max(0, round((time.monotonic() - started) * 1000)),
        )


class _Handler(BaseHTTPRequestHandler):
    server: "_ModelServer"

    def log_message(self, *_: object) -> None:
        return

    def _reply(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path != "/internal/v1/model/infer":
            self._reply(404, {"detail": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_048_576:
                raise ValueError("invalid request body")
            request = InferenceRequest.model_validate(json.loads(self.rfile.read(length)))
            response = self.server.service.infer(request)
            self._reply(200, response.model_dump(mode="json"))
        except ModelVersionNotFoundError:
            self._reply(404, {"detail": "model version not found"})
        except InvalidActionOutputError as exc:
            self._reply(422, {"detail": str(exc)})
        except TimeoutError:
            self._reply(504, {"detail": "inference timeout"})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._reply(400, {"detail": str(exc)})
        except Exception as exc:
            self._reply(503, {"detail": f"model unavailable: {exc}"})


class _ModelServer(ThreadingHTTPServer):
    service: MiniCPMService


def serve(model_path: str, model_version: str, host: str, port: int) -> None:
    service = MiniCPMService.load(model_path, model_version)
    server = _ModelServer((host, port), _Handler)
    server.service = service
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-version", default="minicpm-v-4.7-1b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    serve(args.model_path, args.model_version, args.host, args.port)


if __name__ == "__main__":
    main()
