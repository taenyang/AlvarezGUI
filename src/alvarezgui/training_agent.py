"""GPU-host training agent: claims runs from the control plane, trains, and reports progress over HTTP.

Only needs the standard library plus the MiniCPM training dependencies; it never talks to PostgreSQL.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

log = logging.getLogger("alvarezgui.training_agent")
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED"}
SCORED_OUTCOMES = {"success", "unsuccessful", "invalid_output", "max_steps"}


class RunCanceled(Exception):
    pass


class ControlPlane:
    def __init__(self, base_url: str, password: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.password = password
        self.timeout = timeout

    def call(self, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None) -> Any:
        payload = None if body is None else json.dumps(body).encode()
        request = Request(
            self.base_url + path,
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {self.password}",
                "Content-Type": "application/json",
                **(headers or {}),
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            data = response.read()
        return json.loads(data) if response.headers.get("Content-Type", "").startswith("application/json") else data

    def download(self, path: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.call("GET", path))


def gpu_snapshot() -> list[dict[str, Any]]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in output.strip().splitlines():
        index, name, util, used, total, temp = (part.strip() for part in line.split(","))
        gpus.append({
            "index": int(index),
            "name": name,
            "utilization_percent": float(util),
            "memory_used_mib": float(used),
            "memory_total_mib": float(total),
            "temperature_c": float(temp),
        })
    return gpus


def episode_advantages(rollouts: list[dict[str, Any]]) -> dict[str, float]:
    """Group-normalized episode reward; infrastructure errors are excluded from the baseline."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for rollout in rollouts:
        if rollout["outcome"] in SCORED_OUTCOMES:
            groups.setdefault(rollout["group_id"], []).append(rollout)
    advantages: dict[str, float] = {}
    for members in groups.values():
        rewards = [member["reward_total"] for member in members]
        mean = statistics.fmean(rewards)
        scale = statistics.pstdev(rewards) or 1.0
        for member in members:
            advantages[member["episode_id"]] = (member["reward_total"] - mean) / scale
    return advantages


def rollout_metrics(rollouts: list[dict[str, Any]]) -> dict[str, float]:
    scored = [r for r in rollouts if r["outcome"] in SCORED_OUTCOMES]
    if not scored:
        return {"rollout/scored": 0.0}
    groups: dict[str, list[float]] = {}
    for rollout in scored:
        groups.setdefault(rollout["group_id"], []).append(rollout["reward_total"])
    return {
        "rollout/scored": float(len(scored)),
        "rollout/reward_mean": statistics.fmean(r["reward_total"] for r in scored),
        "rollout/reward_std": statistics.pstdev(r["reward_total"] for r in scored),
        "rollout/success_rate": sum(r["outcome"] == "success" for r in scored) / len(scored),
        "rollout/invalid_output_rate": sum(r["outcome"] == "invalid_output" for r in scored) / len(scored),
        "rollout/mean_steps": statistics.fmean(r["steps"] for r in scored),
        "rollout/zero_variance_groups": float(sum(len(set(v)) <= 1 for v in groups.values())),
        "rollout/errors": float(sum(r["outcome"] == "error" for r in rollouts)),
    }


class TrainingAgent:
    def __init__(
        self,
        control_plane: ControlPlane,
        *,
        node_id: str,
        checkpoint_root: Path,
        work_root: Path,
        model_service_url: str | None,
        poll_seconds: float = 5.0,
        rollout_timeout_seconds: float = 3 * 3600,
    ) -> None:
        self.cp = control_plane
        self.node_id = node_id
        self.checkpoint_root = checkpoint_root
        self.work_root = work_root
        self.model_service_url = model_service_url
        self.poll_seconds = poll_seconds
        self.rollout_timeout_seconds = rollout_timeout_seconds
        self.active_run: str | None = None
        self.run_started = 0.0

    # --- background reporting ------------------------------------------------

    def start_heartbeat(self) -> None:
        def loop() -> None:
            while True:
                gpus = gpu_snapshot()
                try:
                    self.cp.call("POST", f"/internal/v1/nodes/{self.node_id}/heartbeat", {
                        "role": "gpu",
                        "address": socket.gethostname(),
                        "capacity": {"gpus": len(gpus), "training_runs": 1},
                        "resources": {"gpus": gpus, "active_run": self.active_run},
                        "heartbeat_interval_seconds": 10,
                    })
                    run_id = self.active_run
                    if run_id and gpus:
                        step = int(time.monotonic() - self.run_started)
                        self.cp.call("POST", f"/internal/v1/training/runs/{run_id}/metrics", {"points": [
                            {"name": "gpu/utilization_percent", "step": step, "value": gpus[0]["utilization_percent"]},
                            {"name": "gpu/memory_used_mib", "step": step, "value": gpus[0]["memory_used_mib"]},
                        ]})
                except Exception as exc:
                    log.warning("heartbeat failed: %s", exc)
                time.sleep(10)

        threading.Thread(target=loop, name="heartbeat", daemon=True).start()

    def log(self, run_id: str, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        log.info("%s %s", run_id, message)
        try:
            self.cp.call("POST", f"/internal/v1/training/runs/{run_id}/logs", {"lines": [line]})
        except Exception as exc:
            log.warning("log upload failed: %s", exc)

    def metrics(self, run_id: str, step: int, values: dict[str, float]) -> None:
        points = [
            {"name": name, "step": step, "value": float(value)}
            for name, value in values.items()
            if value is not None and math.isfinite(float(value))
        ]
        if not points:
            return
        status = self.cp.call("POST", f"/internal/v1/training/runs/{run_id}/metrics", {"points": points})["status"]
        if status == "CANCELED":
            raise RunCanceled(run_id)

    # --- main loop -------------------------------------------------------------

    def run_forever(self) -> None:
        self.start_heartbeat()
        while True:
            try:
                if not self.run_once():
                    time.sleep(self.poll_seconds)
            except Exception:
                log.exception("agent loop failed")
                time.sleep(self.poll_seconds)

    def run_once(self) -> bool:
        run = self.cp.call("POST", "/internal/v1/training/claim")["run"]
        if run is None:
            return False
        run_id = run["run_id"]
        self.active_run, self.run_started = run_id, time.monotonic()
        try:
            self.log(run_id, f"claimed {run['algorithm']} run from base {run['base_model']}")
            result = self.execute(run)
            self.finish(run_id, True, result["metrics"], result["checkpoint_uri"])
            self.log(run_id, "finished")
        except RunCanceled:
            self.log(run_id, "canceled; stopped without registering a checkpoint")
        except Exception as exc:
            log.exception("run %s failed", run_id)
            self.log(run_id, f"failed: {exc}")
            self.finish(run_id, False, {"error": str(exc)[:500]}, None)
        finally:
            self.active_run = None
        return True

    def finish(self, run_id: str, succeeded: bool, metrics: dict[str, Any], checkpoint_uri: str | None) -> None:
        self.cp.call("POST", f"/internal/v1/training/runs/{run_id}/finish", {
            "succeeded": succeeded, "metrics": metrics, "checkpoint_uri": checkpoint_uri,
        })

    def execute(self, run: dict[str, Any]) -> dict[str, Any]:
        from alvarezgui.minicpm_training import load_dataset

        config = run["config"]
        base_path = _local_path(run.get("base_model_checkpoint_uri"), "base model")
        run_dir = self.checkpoint_root / run["run_id"]
        policy_path = Path(base_path) / "policy_config.json"
        policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
        if run["algorithm"] != "sft" and (
            (Path(base_path) / "adapter_config.json").is_file()
            or policy.get("output_format", "action-v1") != "action-v1"
        ):
            raise ValueError("adapter and structured checkpoints currently require SFT")
        if run["algorithm"] == "gigpo":
            self.log(run["run_id"], "GIGPO compatibility mode: GRPO episode advantages and clipped policy loss; "
                     "no step-level state grouping")
        if run["algorithm"] == "sft" or (run["algorithm"] in {"grpo", "gigpo"} and run.get("dataset_version")):
            rows = load_dataset(_local_path(run.get("dataset_source"), "dataset"))
            self.log(run["run_id"], f"offline {run['algorithm']} on {len(rows)} rows")
            output_dir = run_dir / "final"
            self.unload_serving()
            try:
                metrics = self.train(run, base_path, rows, output_dir, step_offset=0)
            finally:
                self.release_training_memory()
            version = self.register(run, output_dir, metrics, "final")
        elif run["algorithm"] in {"grpo", "gigpo"}:
            version, output_dir, metrics = self.online_grpo(run, base_path, run_dir)
        else:
            raise RuntimeError(f"algorithm {run['algorithm']} is not supported by this agent")
        if config.get("eval_trials") and config.get("tasks"):
            evaluation = self.cp.call(
                "POST",
                "/api/v1/evaluations",
                {
                    "model_versions": [run["base_model"], version],
                    "tasks": config["tasks"],
                    "trials_per_task": config["eval_trials"],
                    "max_steps": config["max_steps"],
                    "temperature": 0.0,
                    "seed": config.get("seed", 0),
                    "run_id": run["run_id"],
                    "runtime_backend": run.get("runtime_backend", "firecracker"),
                },
                {"Idempotency-Key": f"{run['run_id']}:auto-eval"},
            )
            self.log(run["run_id"], f"queued evaluation {evaluation['eval_id']} of {run['base_model']} vs {version}")
        return {"metrics": {**metrics, "model_version": version}, "checkpoint_uri": output_dir.as_uri()}

    def online_grpo(self, run: dict[str, Any], base_path: str, run_dir: Path) -> tuple[str, Path, dict[str, Any]]:
        config = run["config"]
        policy_version, policy_path = run["base_model"], base_path
        step_offset = 0
        metrics: dict[str, Any] = {}
        output_dir = Path(policy_path)
        for round_number in range(config["rounds"]):
            self.log(run["run_id"], f"round {round_number}: {run['algorithm']} sampling with {policy_version}")
            rollouts = self.collect(run, round_number, policy_version)
            summary = rollout_metrics(rollouts)
            self.metrics(run["run_id"], round_number, summary)
            rows = self.rows(run, round_number, rollouts)
            if not rows:
                self.log(run["run_id"], f"round {round_number}: no group had reward variance; policy unchanged")
                metrics = {**summary, "skipped_rounds": metrics.get("skipped_rounds", 0) + 1}
                continue
            self.unload_serving()
            output_dir = run_dir / f"round-{round_number}"
            try:
                trained_metrics = self.train(run, policy_path, rows, output_dir, step_offset)
            finally:
                self.release_training_memory()
            metrics = {**summary, **trained_metrics}
            step_offset += metrics["step_count"]
            policy_version = self.register(run, output_dir, metrics, f"r{round_number}")
            policy_path = str(output_dir)
        if policy_version == run["base_model"]:
            raise RuntimeError("no round produced a policy update (all groups had zero reward variance)")
        return policy_version, output_dir, metrics

    def collect(self, run: dict[str, Any], round_number: int, policy_version: str) -> list[dict[str, Any]]:
        config = run["config"]
        groups = [
            {"group_id": f"r{round_number}-t{index}", "task": task}
            for index, task in enumerate(config["tasks"])
        ]
        self.cp.call("POST", f"/internal/v1/training/runs/{run['run_id']}/rollouts", {
            "round": round_number,
            "policy_version": policy_version,
            "group_size": config["group_size"],
            "max_steps": config["max_steps"],
            "temperature": config["temperature"],
            "seed": config.get("seed", 0) + 1000 * round_number,
            "groups": groups,
        })
        deadline = time.monotonic() + self.rollout_timeout_seconds
        while True:
            state = self.cp.call("GET", f"/internal/v1/training/runs/{run['run_id']}/rollouts?round={round_number}")
            if state["status"] == "CANCELED":
                raise RunCanceled(run["run_id"])
            rollouts = state["tasks"]
            if all(r["outcome"] not in {"queued", "running"} for r in rollouts):
                return rollouts
            if time.monotonic() > deadline:
                raise RuntimeError(f"round {round_number} rollouts did not finish in time")
            time.sleep(self.poll_seconds)

    def rows(self, run: dict[str, Any], round_number: int, rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from alvarezgui.minicpm_service import service_instruction

        advantages = episode_advantages(rollouts)
        image_dir = self.work_root / run["run_id"] / f"round-{round_number}"
        rows = []
        for rollout in rollouts:
            advantage = advantages.get(rollout["episode_id"], 0.0)
            if advantage == 0.0:
                continue
            for step in rollout["step_records"]:
                output = step.get("model_output")
                if not output:
                    continue
                image = image_dir / f"{rollout['episode_id']}-{step['step_id']}.png"
                self.cp.download(
                    f"/api/v1/episodes/{rollout['episode_id']}/steps/{step['step_id']}/screenshot", image
                )
                rows.append({
                    "id": f"{rollout['episode_id']}-step-{step['step_id']}",
                    "episode_id": rollout["episode_id"],
                    "group_id": rollout["group_id"],
                    "step_id": step["step_id"],
                    "image": str(image),
                    "conversations": [
                        {"role": "user", "content": service_instruction(rollout["instruction"])},
                        {"role": "assistant", "content": output["raw_output"]},
                    ],
                    "reward": {"total": rollout["reward_total"]},
                    "advantage": advantage,
                    "done": step["done"],
                })
        self.log(run["run_id"], f"round {round_number}: {len(rows)} training rows from {len(rollouts)} rollouts")
        return rows

    def train(self, run: dict[str, Any], base_path: str, rows: list[dict[str, Any]], output_dir: Path,
              step_offset: int) -> dict[str, Any]:
        from alvarezgui.minicpm_training import train

        config = run["config"]
        return train(
            run["algorithm"],
            base_path,
            rows,
            output_dir,
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            gradient_accumulation=config["gradient_accumulation"],
            finetuning_method=config.get("finetuning_method", "full"),
            train_merger_mlp=config.get("train_merger_mlp", False),
            output_format=config.get("output_format", "action-v1"),
            seed=config.get("seed", 0),
            clip_epsilon=config["clip_epsilon"],
            group_size=config["group_size"],
            compute_old_logprob=True,
            on_step=lambda step, values: self.metrics(run["run_id"], step_offset + step, values),
        )

    def register(self, run: dict[str, Any], output_dir: Path, metrics: dict[str, Any], suffix: str) -> str:
        if not any((output_dir / name).is_file() for name in ("config.json", "adapter_config.json")):
            raise RuntimeError(f"training did not produce a reloadable checkpoint in {output_dir}")
        model_name = run.get("config", {}).get("model_name") or run["run_id"]
        version = f"{model_name}-{suffix}"
        try:
            self.cp.call("POST", "/internal/v1/model-versions", {
                "version": version,
                "base_model": run["base_model"],
                "checkpoint_uri": output_dir.as_uri(),
                "status": "REGISTERED",
                "metrics": {key: value for key, value in metrics.items() if isinstance(value, (int, float, str))},
            })
        except HTTPError as exc:
            if exc.code != 409:
                raise
        self.log(run["run_id"], f"registered {version} -> {output_dir}")
        return version

    def release_training_memory(self) -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_serving(self) -> None:
        """Free GPU memory held by the co-located model service before a full-parameter update."""
        if not self.model_service_url:
            return
        try:
            with urlopen(Request(self.model_service_url.rstrip("/") + "/internal/v1/model/unload",
                                 data=b"{}", method="POST"), timeout=30):
                pass
        except OSError as exc:
            log.warning("could not unload model service: %s", exc)


def _local_path(uri: str | None, what: str) -> str:
    if not uri:
        raise RuntimeError(f"{what} has no checkpoint/source path")
    parsed = urlparse(uri)
    path = unquote(parsed.path) if parsed.scheme == "file" else uri
    if parsed.scheme not in {"", "file"} or not Path(path).exists():
        raise RuntimeError(f"{what} path is not available on this GPU host: {uri}")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-plane-url", required=True)
    parser.add_argument("--node-id", default=f"gpu-{socket.gethostname()}")
    parser.add_argument("--checkpoint-root", default="/home/alvarez-checkpoints")
    parser.add_argument("--work-root", default="/home/alvarez-training-data/runs")
    parser.add_argument("--model-service-url", default="http://127.0.0.1:18081")
    args = parser.parse_args()
    password = os.environ.get("CONSOLE_PASSWORD")
    if not password:
        raise SystemExit("CONSOLE_PASSWORD must be set")
    TrainingAgent(
        ControlPlane(args.control_plane_url, password),
        node_id=args.node_id,
        checkpoint_root=Path(args.checkpoint_root),
        work_root=Path(args.work_root),
        model_service_url=args.model_service_url,
    ).run_forever()


if __name__ == "__main__":
    main()
