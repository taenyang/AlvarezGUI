from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import shutil
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
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


def service_instruction(instruction: str) -> str:
    """The exact user prompt the policy sees; training rows must reuse it verbatim."""
    return (
        f"{instruction}\n"
        "Return exactly one JSON object and nothing else. Use exactly one of these schemas: "
        '{"type":"tap","x":integer,"y":integer}, '
        '{"type":"swipe","x1":integer,"y1":integer,"x2":integer,"y2":integer,"duration_ms":integer}, '
        '{"type":"text","text":"..."}, {"type":"back"}, '
        '{"type":"wait","duration_ms":integer}, or {"type":"terminate","reason":"..."}. '
        "Never output a key named action, an action list, parentheses, Markdown, or explanation."
    )


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


Resolver = Callable[[str], str]
Loader = Callable[[str], tuple[Any, Any, Any]]


def load_pretrained(model_path: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise MiniCPMDependencyError(
            "real MiniCPM serving requires torch, torchvision and transformers>=5.18"
        ) from exc
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, processor, torch


def control_plane_resolver(base_url: str, password: str, timeout: float = 5.0) -> Resolver:
    """Resolve model versions through the control plane registry (file:// checkpoints on this host)."""

    def resolve(version: str) -> str:
        request = Request(
            f"{base_url.rstrip('/')}/internal/v1/model-versions/{quote(version, safe='')}",
            headers={"Authorization": f"Bearer {password}"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                row = json.loads(response.read())
        except HTTPError as exc:
            if exc.code == 404:
                raise ModelVersionNotFoundError(version) from exc
            raise
        uri = row.get("checkpoint_uri") or ""
        if not uri.startswith("file://"):
            raise ModelVersionNotFoundError(f"{version} has no local checkpoint")
        return unquote(urlparse(uri).path)

    return resolve


class MiniCPMService:
    """Serves several checkpoints, loading on demand and evicting the least recently used."""

    def __init__(
        self,
        resolver: Resolver,
        max_loaded: int = 2,
        loader: Loader = load_pretrained,
        checkpoint_root: Path | None = None,
    ) -> None:
        if max_loaded < 1:
            raise ValueError("max_loaded must be at least 1")
        self.resolver = resolver
        self.max_loaded = max_loaded
        self.loader = loader
        self.checkpoint_root = checkpoint_root
        self.models: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self.disabled_versions: set[str] = set()
        self.torch: Any = None
        self.generation_lock = threading.Lock()
        self.waiting = 0
        self.requests = 0
        self.rejected = 0
        self.latencies: deque[int] = deque(maxlen=200)

    @classmethod
    def load(cls, model_path: str | Path, model_version: str) -> "MiniCPMService":
        service = cls(_static_resolver({model_version: str(model_path)}), max_loaded=1)
        with service.generation_lock:
            service._model(model_version)
        return service

    def _model(self, version: str) -> tuple[Any, Any]:
        """Caller must hold generation_lock."""
        if version in self.disabled_versions:
            raise ModelVersionNotFoundError(f"model version is offline: {version}")
        if version in self.models:
            self.models.move_to_end(version)
            return self.models[version]
        path = self.resolver(version)
        while len(self.models) >= self.max_loaded:
            self._evict()
        model, processor, self.torch = self.loader(path)
        self.models[version] = (model, processor)
        return model, processor

    def _clear(self, version: str) -> None:
        self.models.pop(version)
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _evict(self) -> None:
        self._clear(next(iter(self.models)))

    def load(self, version: str) -> None:
        with self.generation_lock:
            self.disabled_versions.discard(version)
            self._model(version)

    def unload(self, version: str) -> bool:
        with self.generation_lock:
            self.disabled_versions.add(version)
            if version not in self.models:
                return False
            self._clear(version)
            return True

    def delete(self, version: str) -> bool:
        with self.generation_lock:
            if self.checkpoint_root is None:
                raise ValueError("checkpoint cleanup root is not configured")
            self.disabled_versions.add(version)
            if version in self.models:
                self._clear(version)
            path = _managed_checkpoint_path(self.resolver(version), self.checkpoint_root)
            if not path.exists():
                return False
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True

    def unload_all(self) -> list[str]:
        with self.generation_lock:
            versions = list(self.models)
            while self.models:
                self._evict()
        return versions

    def health(self) -> dict[str, Any]:
        latencies = sorted(self.latencies)
        return {
            "loaded_versions": list(self.models),
            "offline_versions": sorted(self.disabled_versions),
            "max_loaded": self.max_loaded,
            "waiting_requests": self.waiting,
            "requests": self.requests,
            "rejected_outputs": self.rejected,
            "latency_p50_ms": latencies[len(latencies) // 2] if latencies else None,
            "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
        }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        started = time.monotonic()
        image = _read_image(request.image_uri, request.timeout_ms / 1000)
        instruction = service_instruction(request.instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        self.waiting += 1
        with self.generation_lock:
            self.waiting -= 1
            model, processor = self._model(request.model_version)
            raw_output = self._generate(model, processor, messages, request)
        self.requests += 1
        try:
            action = _parse_action(raw_output)
        except InvalidActionOutputError:
            self.rejected += 1
            raise
        latency_ms = max(0, round((time.monotonic() - started) * 1000))
        self.latencies.append(latency_ms)
        return InferenceResponse(
            request_id=request.request_id,
            action=action,
            raw_output=raw_output,
            model_version=request.model_version,
            latency_ms=latency_ms,
        )

    def _generate(self, model: Any, processor: Any, messages: list[dict[str, Any]], request: InferenceRequest) -> str:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            downsample_mode="16x",
            max_slice_nums=36,
            enable_thinking=False,
        ).to(model.device)
        generation_args: dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            "downsample_mode": "16x",
            "do_sample": request.temperature > 0,
        }
        if request.temperature > 0:
            generation_args["temperature"] = request.temperature
            if request.seed is not None:
                self.torch.manual_seed(request.seed)
        with self.torch.inference_mode():
            generated_ids = model.generate(**inputs, **generation_args)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def _managed_checkpoint_path(path: str, root: Path) -> Path:
    target = Path(path).expanduser()
    cleanup_root = root.expanduser().resolve(strict=False)
    resolved = target.resolve(strict=False)
    if resolved == cleanup_root or cleanup_root not in resolved.parents:
        raise ValueError("checkpoint is outside the cleanup root")
    current = target
    while current != current.parent and current != cleanup_root:
        if current.is_symlink():
            raise ValueError("checkpoint path contains a symlink")
        current = current.parent
    if target.is_symlink():
        raise ValueError("checkpoint path is a symlink")
    return target


def _static_resolver(paths: dict[str, str]) -> Resolver:
    def resolve(version: str) -> str:
        if version not in paths:
            raise ModelVersionNotFoundError(version)
        return paths[version]

    return resolve


def _chain(*resolvers: Resolver) -> Resolver:
    def resolve(version: str) -> str:
        for resolver in resolvers[:-1]:
            try:
                return resolver(version)
            except ModelVersionNotFoundError:
                continue
        return resolvers[-1](version)

    return resolve


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

    def do_GET(self) -> None:
        if self.path != "/health":
            self._reply(404, {"detail": "not found"})
            return
        self._reply(200, self.server.service.health())

    def do_POST(self) -> None:
        if self.path in {
            "/internal/v1/model/load",
            "/internal/v1/model/unload",
            "/internal/v1/model/delete",
        }:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_048_576:
                    raise ValueError("invalid request body")
                body = json.loads(self.rfile.read(length))
                version = body.get("model_version") if isinstance(body, dict) else None
                if self.path.endswith("/load"):
                    if not isinstance(version, str) or not version:
                        raise ValueError("model_version is required")
                    self.server.service.load(version)
                    self._reply(200, {"loaded": version, **self.server.service.health()})
                elif self.path.endswith("/delete"):
                    if not isinstance(version, str) or not version:
                        raise ValueError("model_version is required")
                    deleted = self.server.service.delete(version)
                    self._reply(200, {
                        "deleted": version,
                        "checkpoint_deleted": deleted,
                        **self.server.service.health(),
                    })
                else:
                    unloaded = self.server.service.unload(version) if version else self.server.service.unload_all()
                    self._reply(200, {"unloaded": [version] if unloaded else [], **self.server.service.health()})
            except ModelVersionNotFoundError:
                self._reply(404, {"detail": "model version not found"})
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._reply(400, {"detail": str(exc)})
            except Exception as exc:
                self._reply(503, {"detail": f"model unavailable: {exc}"})
            return
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


def serve(service: MiniCPMService, host: str, port: int) -> None:
    server = _ModelServer((host, port), _Handler)
    server.service = service
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=[], metavar="VERSION=PATH",
                        help="statically known checkpoint; may repeat")
    parser.add_argument("--control-plane-url", help="resolve other versions via the registry; "
                        "password comes from CONSOLE_PASSWORD")
    parser.add_argument("--max-loaded", type=int, default=2)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    static = dict(entry.split("=", 1) for entry in args.model)
    resolvers = [_static_resolver(static)]
    if args.control_plane_url:
        password = os.environ.get("CONSOLE_PASSWORD")
        if not password:
            raise SystemExit("CONSOLE_PASSWORD is required with --control-plane-url")
        resolvers.append(control_plane_resolver(args.control_plane_url, password))
    serve(
        MiniCPMService(
            _chain(*resolvers),
            max_loaded=args.max_loaded,
            checkpoint_root=args.checkpoint_root,
        ),
        args.host,
        args.port,
    )


if __name__ == "__main__":
    main()
