from __future__ import annotations

import argparse
import base64
import gc
import hmac
import json
import math
import os
import re
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

from alvarezgui.action import MODEL_SCREEN_HEIGHT, MODEL_SCREEN_WIDTH, validate_action
from alvarezgui.model_families import model_family
from alvarezgui.sokoban import ACTIONS
from alvarezgui.policy_output import OUTPUT_FORMATS, parse_policy_output, policy_instruction
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

    image = Image.open(BytesIO(payload)).convert("RGB")
    if image.size != (MODEL_SCREEN_WIDTH, MODEL_SCREEN_HEIGHT):
        image = image.resize((MODEL_SCREEN_WIDTH, MODEL_SCREEN_HEIGHT))
    return image


def service_instruction(instruction: str) -> str:
    """The exact user prompt the policy sees; training rows must reuse it verbatim."""
    return policy_instruction(instruction)


def token_entropies(torch: Any, logits: Any) -> Any:
    """Entropy in nats of each row's next-token distribution; filtered (-inf) tokens contribute 0."""
    return torch.special.entr(torch.softmax(logits.float(), dim=-1)).sum(dim=-1)


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
            # Same rule as training rows (parse_policy_output): "360" or 360.0 is not an integer coordinate,
            # so a rollout never executes an output the trainer would reject.
            if isinstance(value, dict) and any(
                    key in value and type(value[key]) is not int
                    for key in ("x", "y", "x1", "y1", "x2", "y2", "duration_ms")):
                continue
            return validate_action(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    raise InvalidActionOutputError(
        "model output did not contain a valid Android action; "
        f"raw_output_preview={raw_output[:160]!r}"
    )


Resolver = Callable[[str], str]
Loader = Callable[[str], tuple[Any, Any, Any]]


def _has_routed_direction_head(config: dict[str, Any]) -> bool:
    return bool(config.get("direction") and config.get("visual_views") == "task-routed"
                and config.get("head_views", {}).get("direction") == "board-crop")


def load_pretrained(model_path: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise MiniCPMDependencyError(
            "real model serving requires torch, torchvision and transformers>=5.18"
        ) from exc
    checkpoint = Path(model_path)
    policy_path = checkpoint / "policy_config.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.is_file() else {}
    perception_config_path = checkpoint / "perception_config.json"
    perception_config = (
        json.loads(perception_config_path.read_text(encoding="utf-8"))
        if perception_config_path.is_file() else {}
    )
    perception = policy.get("inference_mode") == "perception_direction" or (
        not policy and _has_routed_direction_head(perception_config)
    )
    adapter = checkpoint / "adapter_config.json"
    base_path = model_path
    if perception:
        base_path = policy.get("base_model") or perception_config.get("base_model")
        if not isinstance(base_path, str) or not Path(base_path).is_dir():
            raise ValueError("perception checkpoint requires its existing local base model")
    elif adapter.is_file():
        from peft import PeftConfig, PeftModel

        base_path = PeftConfig.from_pretrained(model_path).base_model_name_or_path
        if not base_path or not Path(base_path).is_dir():
            raise ValueError("adapter requires its existing local base model")
    family = model_family(AutoConfig.from_pretrained(base_path))  # Reject unknown architectures before loading.
    processor = AutoProcessor.from_pretrained(base_path if perception else model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        base_path,
        torch_dtype="auto",
        device_map="auto",
    )
    if family.prepare_model is not None:
        family.prepare_model(model)
    if adapter.is_file():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_path, is_trainable=False,
                                          autocast_adapter_dtype=True)
        # BF16 merging can change outputs: opt in only checkpoints validated for serving.
        if str(Path(model_path).resolve()) in os.environ.get("ALVAREZ_MERGE_ADAPTER_PATHS", "").split(os.pathsep):
            model = model.merge_and_unload(safe_merge=True)
    model.eval()
    return model, processor, torch


def merge_adapter(checkpoint: Path, output: Path) -> None:
    """Save a LoRA checkpoint merged into its base model as full weights at ``output``.

    Loads like ``load_pretrained`` (base dtype, FP32 adapter) but on CPU, so a serving GPU is not needed.
    BF16 rounding of the merged update can change outputs (it did for tiny4): compare the merged model
    with the adapter before training or serving from it.
    """
    from peft import PeftConfig, PeftModel
    from transformers import AutoConfig, AutoModelForImageTextToText

    base_path = PeftConfig.from_pretrained(str(checkpoint)).base_model_name_or_path
    if not base_path or not Path(base_path).is_dir():
        raise ValueError("adapter requires its existing local base model")
    model_family(AutoConfig.from_pretrained(base_path))  # Reject unknown architectures before loading.
    model = AutoModelForImageTextToText.from_pretrained(base_path, torch_dtype="auto")
    model = PeftModel.from_pretrained(model, str(checkpoint), autocast_adapter_dtype=True)
    model = model.merge_and_unload(safe_merge=True)
    pending = output.with_name(output.name + ".tmp")
    shutil.rmtree(pending, ignore_errors=True)
    # Processor, chat template and policy_config (output format, visual views) carry over byte for byte.
    shutil.copytree(checkpoint, pending, ignore=shutil.ignore_patterns("adapter_*", "README.md", "metrics.json"))
    model.save_pretrained(pending, safe_serialization=True)
    policy_path = pending / "policy_config.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.is_file() else {}
    policy_path.write_text(json.dumps({**policy, "finetuning_method": "full", "merged_from": str(checkpoint)},
                                      indent=2), encoding="utf-8")
    pending.rename(output)


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
        self.policy_configs: dict[str, dict[str, Any]] = {}
        self.perception_heads: dict[str, Any] = {}
        self.disabled_versions: set[str] = set()
        self.torch: Any = None
        self.generation_lock = threading.Lock()
        self.merge_lock = threading.Lock()
        self.waiting = 0
        self.requests = 0
        self.rejected = 0
        self.latencies: deque[int] = deque(maxlen=200)

    @classmethod
    def load_checkpoint(cls, model_path: str | Path, model_version: str) -> "MiniCPMService":
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
        checkpoint = Path(path)
        policy_path = checkpoint / "policy_config.json"
        policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
        output_format = policy.get("output_format", "action-v1")
        if output_format not in OUTPUT_FORMATS:
            raise ValueError("unsupported checkpoint output format")
        perception_config_path = checkpoint / "perception_config.json"
        perception_config = (
            json.loads(perception_config_path.read_text(encoding="utf-8"))
            if perception_config_path.is_file() else {}
        )
        inference_mode = policy.get("inference_mode")
        if inference_mode is None and not policy and _has_routed_direction_head(perception_config):
            inference_mode = "perception_direction"
        if inference_mode not in (None, "perception_direction"):
            raise ValueError("unsupported checkpoint inference mode")
        if inference_mode == "perception_direction" and output_format != "action-v1":
            raise ValueError("perception direction checkpoints must use action-v1")
        visual_views = policy.get("visual_views", "full")
        if visual_views not in ("full", "full+board-crop"):
            raise ValueError("unsupported checkpoint visual views")
        max_tokens = policy.get("max_tokens", 512 if output_format == "sokoban-plan-v1" else 128)
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("invalid checkpoint token budget")
        while len(self.models) >= self.max_loaded:
            self._evict()
        model, processor, self.torch = self.loader(path)
        if inference_mode == "perception_direction":
            perception_config_path = Path(path) / "perception_config.json"
            perception_config = json.loads(perception_config_path.read_text(encoding="utf-8"))
            if not _has_routed_direction_head(perception_config):
                raise ValueError("perception checkpoint must contain a task-routed direction head")
            heads = self._load_perception_heads(path, self.torch)
            if "direction" not in heads:
                raise ValueError("perception checkpoint has no direction head")
            heads.to(model.device)
            self.perception_heads[version] = heads
        self.models[version] = (model, processor)
        self.policy_configs[version] = {
            "output_format": output_format,
            "max_tokens": max_tokens,
            "inference_mode": inference_mode,
            "visual_views": visual_views,
        }
        return model, processor

    @staticmethod
    def _load_perception_heads(path: str, torch: Any) -> Any:
        from alvarezgui.minicpm_training import load_perception_head

        return load_perception_head(path, torch)

    def _clear(self, version: str) -> None:
        self.models.pop(version)
        self.policy_configs.pop(version, None)
        self.perception_heads.pop(version, None)
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

    def merge(self, version: str, target: str) -> str:
        """Merge ``version``'s adapter into ``<checkpoint_root>/merged/<target>``; serving is not blocked."""
        if self.checkpoint_root is None:
            raise ValueError("checkpoint root is not configured")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", target):
            raise ValueError("invalid target_version")
        source = Path(self.resolver(version))
        if not (source / "adapter_config.json").is_file():
            raise ValueError(f"{version} is not a LoRA adapter checkpoint")
        output = self.checkpoint_root / "merged" / target
        # One merge at a time, so a leftover .tmp can only come from an interrupted earlier merge.
        with self.merge_lock:
            if output.exists():
                raise ValueError(f"{output} already exists")
            output.parent.mkdir(parents=True, exist_ok=True)
            merge_adapter(source, output)
        return str(output)

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
        self.waiting += 1
        with self.generation_lock:
            self.waiting -= 1
            model, processor = self._model(request.model_version)
            policy = self.policy_configs[request.model_version]
            output_format = policy["output_format"]
            entropy = None
            if policy["inference_mode"] == "perception_direction":
                direction = self._predict_direction(request.model_version, model, processor, image, request)
                raw_output = json.dumps(ACTIONS[direction], separators=(",", ":"))
            else:
                images = [image]
                if policy["visual_views"] == "full+board-crop":
                    from alvarezgui.minicpm_training import _board_crop_image

                    images.append(_board_crop_image(image))
                content = [{"type": "image", "image": value} for value in images]
                content.append({"type": "text", "text": policy_instruction(request.instruction, output_format)})
                messages = [{"role": "user", "content": content}]
                if request.max_tokens is None:
                    request = request.model_copy(update={"max_tokens": policy["max_tokens"]})
                raw_output, entropy = self._generate(model, processor, messages, request)
        self.requests += 1
        try:
            action = (_parse_action(raw_output) if output_format == "action-v1"
                      else parse_policy_output(raw_output, output_format))
        except (InvalidActionOutputError, ValueError, TypeError) as exc:
            self.rejected += 1
            raise InvalidActionOutputError(
                f"invalid {output_format} output; raw_output_preview={raw_output[:160]!r}",
                raw_output=raw_output[:16000], output_format=output_format,
            ) from exc
        latency_ms = max(0, round((time.monotonic() - started) * 1000))
        self.latencies.append(latency_ms)
        return InferenceResponse(
            request_id=request.request_id,
            action=action,
            raw_output=raw_output,
            model_version=request.model_version,
            latency_ms=latency_ms,
            output_format=output_format,
            entropy=entropy,
        )

    def _predict_direction(
        self,
        version: str,
        model: Any,
        processor: Any,
        image: Any,
        request: InferenceRequest,
    ) -> str:
        from alvarezgui.minicpm_training import DIRECTIONS, _perception_head_features

        row = {
            "image": "",
            "conversations": [
                {"role": "user", "content": policy_instruction(request.instruction, "action-v1")},
                {"role": "assistant", "content": ""},
            ],
        }
        with self.torch.inference_mode():
            features = _perception_head_features(
                model,
                processor,
                row,
                self.torch,
                model.device,
                ("direction",),
                image=image,
                multiview=True,
                task_views=True,
            )
            prediction = self.perception_heads[version]["direction"](features["direction"])
        return DIRECTIONS[int(prediction.argmax(dim=-1).item())]

    def _generate(self, model: Any, processor: Any, messages: list[dict[str, Any]],
                  request: InferenceRequest) -> tuple[str, float | None]:
        """Returns the decoded output and its mean per-token sampling entropy (nats)."""
        family = model_family(model.config)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **family.chat_template_kwargs,
        ).to(model.device)
        generation_args: dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            **family.generate_kwargs,
            "do_sample": request.temperature > 0,
        }
        if request.temperature > 0:
            generation_args["temperature"] = request.temperature
            if request.seed is not None:
                self.torch.manual_seed(request.seed)
        with self.torch.inference_mode():
            generated = model.generate(**inputs, **generation_args, output_scores=True, return_dict_in_generate=True)
            # Scores are what each token was drawn from: after temperature, top-k/top-p and other processors.
            # Per step, so a 512-token plan never materializes a (tokens, vocab) softmax.
            entropy = (float(sum(token_entropies(self.torch, step).sum() for step in generated.scores))
                       / len(generated.scores) if generated.scores else None)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated.sequences)
        ]
        text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text, entropy if entropy is not None and math.isfinite(entropy) else None


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

    def _authorized(self) -> bool:
        token = self.server.token
        if token and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
            self._reply(401, {"detail": "authentication required"})
            return False
        return True

    def do_GET(self) -> None:
        if not self._authorized():
            return
        if self.path != "/health":
            self._reply(404, {"detail": "not found"})
            return
        self._reply(200, self.server.service.health())

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path in {
            "/internal/v1/model/load",
            "/internal/v1/model/unload",
            "/internal/v1/model/delete",
            "/internal/v1/model/merge",
        }:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_048_576:
                    raise ValueError("invalid request body")
                body = json.loads(self.rfile.read(length))
                version = body.get("model_version") if isinstance(body, dict) else None
                if self.path.endswith("/merge"):
                    target = body.get("target_version") if isinstance(body, dict) else None
                    if not isinstance(version, str) or not version or not isinstance(target, str):
                        raise ValueError("model_version and target_version are required")
                    self._reply(200, {"checkpoint": self.server.service.merge(version, target)})
                elif self.path.endswith("/load"):
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
            self._reply(422, {"detail": str(exc), "raw_output": exc.raw_output,
                              "output_format": exc.output_format})
        except TimeoutError:
            self._reply(504, {"detail": "inference timeout"})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._reply(400, {"detail": str(exc)})
        except Exception as exc:
            self._reply(503, {"detail": f"model unavailable: {exc}"})


class _ModelServer(ThreadingHTTPServer):
    service: MiniCPMService
    token: str | None = None
    # socketserver's default backlog of 5 overflows under 16+ concurrent clients (sglang adapter); with
    # SYN cookies the client's connect succeeds and its first send gets "Connection reset by peer".
    request_queue_size = 256


def serve(service: MiniCPMService, host: str, port: int, *, token: str | None = None,
          ready_file: Path | None = None) -> None:
    """``token`` guards every POST (needed off loopback); ``ready_file`` receives the bound port."""
    server = _ModelServer((host, port), _Handler)
    server.service = service
    server.token = token
    if ready_file is not None:
        pending = ready_file.with_name(ready_file.name + ".tmp")
        pending.write_text(json.dumps({"port": server.server_address[1]}))
        pending.replace(ready_file)
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
    parser.add_argument("--port", type=int, default=18081, help="0 picks a free port; see --ready-file")
    parser.add_argument("--ready-file", type=Path, help="write {\"port\": N} once listening")
    args = parser.parse_args()
    token = os.environ.get("MODEL_SERVICE_TOKEN") or None
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
        raise SystemExit("MODEL_SERVICE_TOKEN is required when listening beyond loopback")
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
        token=token,
        ready_file=args.ready_file,
    )


if __name__ == "__main__":
    main()
