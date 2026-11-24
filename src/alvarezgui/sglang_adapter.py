"""H800 rollout and evaluation services on sglang: the ``/internal/v1/model/infer`` protocol of
``MiniCPMService`` in front of supervised sglang servers, so requests are batched continuously. See docs/h800-sglang-rollout.md and
docs/h800-sglang-evaluation.md.

Runs with the sglang image's Python; needs only pydantic and PIL besides the standard library.
"""

from __future__ import annotations

import base64
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from alvarezgui.h800_launcher import H800_PYTHON
from alvarezgui.minicpm_service import MiniCPMService, _parse_action, _read_image, _static_resolver, serve
from alvarezgui.model import InferenceRequest, InferenceResponse, InvalidActionOutputError, ModelVersionNotFoundError
from alvarezgui.model_families import MINICPM_V_4_7
from alvarezgui.policy_output import parse_policy_output, policy_instruction

TOP_LOGPROBS = 20  # OpenAI-compatible maximum.
HF_MEMORY = 0.05  # Share of an 80 GB H800 a 1B HF model takes (~4 GB).
MIN_SGLANG_MEMORY = 0.1


def topk_entropy(top_logprobs: list[list[dict[str, Any]]]) -> float | None:
    """Mean over tokens of -sum p log p across each token's top-k alternatives (nats).

    ponytail: a top-k lower bound on the exact entropy MiniCPMService reports; not comparable across the switch.
    """
    values = [-sum(math.exp(item["logprob"]) * item["logprob"] for item in token if math.isfinite(item["logprob"]))
              for token in top_logprobs if token]
    if not values:
        return None
    mean = sum(values) / len(values)
    return max(0.0, mean) if math.isfinite(mean) else None


def _data_uri(image: Any) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class SglangService:
    def __init__(self, version: str, path: str, sglang_url: str) -> None:
        self.version, self.path, self.url = version, path, sglang_url.rstrip("/")
        policy_path = Path(path) / "policy_config.json"
        policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
        self.output_format = policy.get("output_format", "action-v1")
        self.visual_views = policy.get("visual_views", "full")
        self.max_tokens = policy.get("max_tokens", 512 if self.output_format == "sokoban-plan-v1" else 128)
        self.requests = self.rejected = 0
        self.latencies: deque[int] = deque(maxlen=200)

    def health(self) -> dict[str, Any]:
        latencies = sorted(self.latencies)
        return {"loaded_versions": [self.version], "backend": "sglang", "requests": self.requests,
                "rejected_outputs": self.rejected,
                "latency_p50_ms": latencies[len(latencies) // 2] if latencies else None}

    def body(self, request: InferenceRequest) -> dict[str, Any]:
        image = _read_image(request.image_uri, request.timeout_ms / 1000)
        images = [image]
        if self.visual_views == "full+board-crop":
            from alvarezgui.minicpm_training import _board_crop_image

            images.append(_board_crop_image(image))
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": _data_uri(value)}} for value in images]
        content.append({"type": "text", "text": policy_instruction(request.instruction, self.output_format)})
        body = {"model": "default", "messages": [{"role": "user", "content": content}],
                "temperature": request.temperature, "max_tokens": request.max_tokens or self.max_tokens,
                "logprobs": True, "top_logprobs": TOP_LOGPROBS,
                "chat_template_kwargs": MINICPM_V_4_7.chat_template_kwargs}
        if request.seed is not None and request.temperature > 0:
            body["seed"] = request.seed
        return body

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        started = time.monotonic()
        if request.model_version != self.version:
            raise ModelVersionNotFoundError(request.model_version)
        http = Request(f"{self.url}/v1/chat/completions", data=json.dumps(self.body(request)).encode(),
                       headers={"Content-Type": "application/json"}, method="POST")
        remaining = request.timeout_ms / 1000 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError
        with urlopen(http, timeout=remaining) as response:
            choice = json.loads(response.read())["choices"][0]
        raw_output = choice["message"]["content"] or ""
        tokens = (choice.get("logprobs") or {}).get("content") or []
        entropy = topk_entropy([token.get("top_logprobs") or [] for token in tokens])
        self.requests += 1
        try:
            action = (_parse_action(raw_output) if self.output_format == "action-v1"
                      else parse_policy_output(raw_output, self.output_format))
        except (InvalidActionOutputError, ValueError, TypeError) as exc:
            self.rejected += 1
            raise InvalidActionOutputError(
                f"invalid {self.output_format} output; raw_output_preview={raw_output[:160]!r}",
                raw_output=raw_output[:16000], output_format=self.output_format,
            ) from exc
        latency_ms = max(0, round((time.monotonic() - started) * 1000))
        self.latencies.append(latency_ms)
        return InferenceResponse(request_id=request.request_id, action=action, raw_output=raw_output,
                                 model_version=self.version, latency_ms=latency_ms,
                                 output_format=self.output_format, entropy=entropy)


def _free_port() -> int:
    # Bind-and-release races other host-network processes, as do sglang's own internal ports; _start retries.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_ready(url: str, process: subprocess.Popen, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sglang exited with status {process.returncode} before it was ready")
        try:
            with urlopen(f"{url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f"sglang was not ready after {timeout:.0f} s")


def _start(path: str, log_path: Path, current: list[subprocess.Popen], attempts: int = 3,
           memory: float = 0.5, extra: tuple[str, ...] = ()) -> str:
    """Launch sglang until it answers, appending it to ``current`` (so SIGTERM can stop it while starting);
    an exit before ready (e.g. EADDRINUSE on a host-network port) retries."""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent / "sglang_patch")}
    for attempt in range(1, attempts + 1):
        port = _free_port()
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [H800_PYTHON, "-m", "sglang.launch_server", "--model-path", path, "--host", "127.0.0.1",
                 "--port", str(port), "--trust-remote-code", "--mem-fraction-static", f"{memory:.3f}", *extra],
                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        current.append(process)
        url = f"http://127.0.0.1:{port}"
        try:
            _wait_ready(url, process)
            return url
        except RuntimeError:
            if process.poll() is None or attempt == attempts:
                raise  # Hung (timeout) or out of attempts.
            current.remove(process)
            print(f"sglang attempt {attempt} exited with status {process.returncode}; retrying", flush=True)
    raise AssertionError("unreachable")


class SwapPaused(RuntimeError):
    """Raised by ``infer`` while weights are being swapped; the handler answers 503 and consumers retry."""


class ResidentService:
    """A run's resident rollout service (docs/h800-resident-pods.md): one sglang server for every round.

    ``swap()`` loads the next round's weights in place with sglang's ``/update_weights_from_disk`` and only
    then answers to the new policy version, so a round's requests are never served by another round's weights.
    """

    def __init__(self, service: SglangService) -> None:
        self.service = service
        self.lock = threading.Lock()
        self.swapping = False

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        with self.lock:
            if self.swapping:
                raise SwapPaused(f"loading the weights for the next round; {request.model_version} is not ready")
            service = self.service
        return service.infer(request)

    def health(self) -> dict[str, Any]:
        return {**self.service.health(), "swapping": self.swapping}

    def swap(self, version: str, path: str, *, timeout: float = 600) -> None:
        with self.lock:
            self.swapping = True
        try:
            http = Request(f"{self.service.url}/update_weights_from_disk",
                           data=json.dumps({"model_path": path}).encode(),
                           headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(http, timeout=timeout) as response:
                    result = json.loads(response.read())
            except HTTPError as exc:
                result = {"success": False, "message": exc.read().decode(errors="replace")[:500]}
            if not result.get("success"):
                raise RuntimeError(f"sglang could not load {path}: {result.get('message')}")
            service = SglangService(version, path, self.service.url)
            with self.lock:
                self.service = service
        finally:
            with self.lock:
                self.swapping = False


def watch_swaps(work: Path, resident: ResidentService, poll: float = 2.0) -> None:
    """Apply ``next.json`` (written by the launcher) and publish the served policy in ``current.json``.

    A failed swap writes FAILED and stops the pod: serving the previous round's weights would be wrong.
    """
    while True:
        target = work / "next.json"
        if target.exists():
            wanted = json.loads(target.read_text())
            if wanted["version"] != resident.service.version:
                started = time.monotonic()
                try:
                    resident.swap(wanted["version"], wanted["path"])
                except Exception:
                    (work / "FAILED").write_text(traceback.format_exc())
                    os._exit(1)
                print(f"swapped to {wanted['version']} from {wanted['path']} in "
                      f"{time.monotonic() - started:.1f} s", flush=True)
            current = {"version": resident.service.version, "path": resident.service.path}
            if not (work / "current.json").exists() or json.loads((work / "current.json").read_text()) != current:
                pending = work / "current.json.tmp"
                pending.write_text(json.dumps(current))
                pending.replace(work / "current.json")
        time.sleep(poll)


class Dispatcher:
    """One endpoint for an evaluation's models: each request goes to its ``model_version``'s service."""

    def __init__(self, services: dict[str, Any]) -> None:
        self.services = services

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if request.model_version not in self.services:
            raise ModelVersionNotFoundError(request.model_version)
        return self.services[request.model_version].infer(request)

    def health(self) -> dict[str, Any]:
        return {"loaded_versions": sorted(self.services),
                "models": {version: service.health() for version, service in self.services.items()}}


def _backends(spec: dict[str, Any], log: Path) -> dict[str, dict[str, str]]:
    """Normalize spec models to {version: {path, backend}}, merging LoRA adapters that sglang will serve."""
    models: dict[str, dict[str, str]] = {}
    for version, entry in spec["models"].items():
        entry = {"path": entry, "backend": "sglang"} if isinstance(entry, str) else dict(entry)
        merged = entry.get("merged_copy")
        if merged:
            from alvarezgui.minicpm_service import merge_adapter

            try:
                if not Path(merged).is_dir():  # A replacement pod reuses an earlier pod's merge.
                    Path(merged).parent.mkdir(parents=True, exist_ok=True)
                    merge_adapter(Path(entry["path"]), Path(merged))
                entry["path"] = merged
            except Exception:
                with log.open("a") as file:
                    file.write(f"merging {version} failed; serving the adapter on HF\n{traceback.format_exc()}")
                entry["backend"] = "hf"
        models[version] = entry
    return models


def run(work: Path) -> None:
    """Start one sglang server per sglang model (one after another), load HF models, then serve them all
    on a free host port behind the pod's token. Any sglang exit fails the whole pod."""
    spec = json.loads((work / "spec.json").read_text())
    resident = spec.get("resident", False)
    if resident and (work / "current.json").exists():
        # A restarted container resumes the round it was serving, not round 0's weights.
        served = json.loads((work / "current.json").read_text())
        spec["models"] = {served["version"]: served["path"]}
    current: list[subprocess.Popen] = []

    def stop(*_: object) -> None:
        for process in current:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(143)

    signal.signal(signal.SIGTERM, stop)
    services: dict[str, Any] = {}
    try:
        models = _backends(spec, work / "run.log")
        sglang = {version: entry["path"] for version, entry in models.items() if entry["backend"] == "sglang"}
        hf = {version: entry["path"] for version, entry in models.items() if entry["backend"] != "sglang"}
        if sglang:
            memory = (spec.get("sglang_memory", 0.5) - HF_MEMORY * len(hf)) / len(sglang)
            if memory < MIN_SGLANG_MEMORY:
                raise RuntimeError(f"{len(sglang)} sglang + {len(hf)} HF models leave {memory:.3f} of the GPU "
                                   f"per sglang server (minimum {MIN_SGLANG_MEMORY})")
        for version, path in sglang.items():
            # Serial: concurrent CUDA graph captures peak together and run out of memory.
            url = _start(path, work / "sglang.log", current, memory=memory,
                         extra=tuple(spec.get("sglang_args", ())))  # e.g. --enable-deterministic-inference
            services[version] = SglangService(version, path, url)
            print(f"serving {version} from {path} via sglang at {url}", flush=True)
        if hf:
            service = MiniCPMService(_static_resolver(hf), max_loaded=len(hf))
            for version in hf:
                service.load(version)  # No trial should wait for a checkpoint load.
                services[version] = service
            print(f"serving {sorted(hf)} via HF", flush=True)
    except Exception:
        (work / "FAILED").write_text(traceback.format_exc() + "\n(see sglang.log)\n")
        stop()

    def watch(process: subprocess.Popen) -> None:
        status = process.wait()
        (work / "FAILED").write_text(f"sglang exited with status {status}; see sglang.log\n")
        os._exit(1)

    for process in current:
        threading.Thread(target=watch, args=(process,), daemon=True).start()
    service = next(iter(services.values())) if len(services) == 1 else Dispatcher(services)
    if resident:
        if not isinstance(service, SglangService):
            (work / "FAILED").write_text("a resident rollout service needs exactly one sglang model\n")
            stop()
        service = ResidentService(service)
        threading.Thread(target=watch_swaps, args=(work, service), daemon=True).start()
    serve(service, "0.0.0.0", 0,  # type: ignore[arg-type]
          token=(work / "token").read_text().strip(), ready_file=work / "ready.json")
