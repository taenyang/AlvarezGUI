"""H800 training launcher for the control-plane host, which mounts the same CFS path as the H800 pods.

``serve`` claims ``accelerator=h800`` runs, stages the base model and dataset on CFS, starts one
Kubernetes Deployment per run (copied from a template), relays CFS metrics/logs to the control plane,
then copies the checkpoint to the 4090 serving host, registers it and deletes the Deployment.
Online GRPO repeats this per round on H800 only: a serving Deployment answers that round's rollouts
with the current policy, then a fresh Deployment trains the next policy. Checkpoints are still copied
to the 4090 serving host so evaluations there can load them.
``pod`` is the in-pod entry point. CFS holds all run state, so a restarted launcher resumes its runs.
H800 evaluations (``accelerator=h800``) get one Deployment that preloads their models and serves them
on a free port of the pod's host network behind a per-evaluation token; the launcher publishes that
endpoint to the control plane while it answers, and deletes the Deployment once no trial is left.
Only the standard library is needed outside the pod.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from alvarezgui.training_agent import ControlPlane, TrainingAgent, rollout_metrics, train_options

log = logging.getLogger("alvarezgui.h800_launcher")
CFS_ROOT = Path("/ainf-cfs/yangtianze")
SERVING_HOST = "root@10.157.155.124"
# Same content on the 4090 serving host and on CFS.
PATH_MAP = (("/home/alvarez-models", "alvarez-models"), ("/home/alvarez-datasets", "alvarez-data"),
            ("/home/alvarez-checkpoints", "alvarez-checkpoints"))
# Every H800 pod (training, rollout, evaluation) runs this image's Python: torch 2.14, transformers 5.19,
# peft and accelerate preinstalled, so only the source snapshot goes on PYTHONPATH.
H800_IMAGE = "iregistry.baidu-int.com/sglang-public/sglang_v0.5.23:base"
H800_PYTHON = "/opt/sglang/bin/python3"
# Some H800 nodes still run driver 535 (CUDA 12.2); the image's CUDA 13 forward-compat libraries let its
# cu130 torch run there. Only below 580: the compat package must not shadow a newer driver.
H800_COMPAT = """if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 580 ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH"
fi
"""
# kernels/: flash-linear-attention 0.6.0 and causal-conv1d 1.7.0 (built for this image's torch) give MiniCPM-V 4.7's
# linear-attention layers their fast path in transformers; sglang has its own kernels and does not get it.
POD_ENV = """export PYTHONPATH="{src}:$root/alvarez-runtime/kernels"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$root/alvarez-runtime/cache/huggingface" TORCH_HOME="$root/alvarez-runtime/cache/torch"
""" + H800_COMPAT
POD_SCRIPT = """#!/bin/bash
run={run}
root={root}
""" + POD_ENV + """export TRITON_CACHE_DIR="$root/alvarez-runtime/cache/triton" TMPDIR="$run/tmp"
mkdir -p "$TMPDIR"
if [ ! -f "$run/COMPLETE" ] && [ ! -f "$run/FAILED" ]; then
    printf '\\npod %s started: %s\\n' "$(hostname)" "$(date -Is)" >> "$run/run.log"
    # Forward SIGTERM so a deleted/preempted pod's trainer stops before its replacement resumes.
    """ + H800_PYTHON + """ -m alvarezgui.h800_launcher pod "$run" >> "$run/run.log" 2>&1 &
    trainer=$!
    trap 'kill -TERM "$trainer"' TERM
    wait "$trainer" || status=$?
    trap - TERM
    wait "$trainer" 2>/dev/null
    # ponytail: killed by a signal (OOM, preemption) -> restart the container and resume; no retry cap.
    if [ "${{status:-0}}" -ge 128 ]; then exit "$status"; fi
    if [ -n "${{status:-}}" ] && [ ! -f "$run/FAILED" ]; then
        echo "trainer exited with status $status" > "$run/FAILED"
    fi
fi
exec sleep infinity
"""
SERVE_POD_SCRIPT = """#!/bin/bash
work={work}
root={root}
""" + POD_ENV + """export TRITON_CACHE_DIR="$root/alvarez-runtime/cache/triton" TMPDIR="$work/tmp"
mkdir -p "$TMPDIR"
rm -f "$work/ready.json"  # A replacement pod announces its own port.
if [ ! -f "$work/FAILED" ]; then
    printf '\\npod %s started: %s\\n' "$(hostname)" "$(date -Is)" >> "$work/run.log"
    """ + H800_PYTHON + """ -m alvarezgui.h800_launcher serve-pod "$work" >> "$work/run.log" 2>&1 &
    server=$!
    trap 'kill -TERM "$server"' TERM
    wait "$server" || status=$?
    trap - TERM
    if [ "${{status:-0}}" -ge 128 ]; then exit "$status"; fi
    [ -f "$work/FAILED" ] || echo "model service exited with status ${{status:-0}}" > "$work/FAILED"
fi
exec sleep infinity
"""
# Resident online-RL training worker (docs/h800-resident-pods.md): one process trains every round of a run.
# Not a signal exit -> FAILED is written per round by the worker, so the container itself never ends.
WORKER_POD_SCRIPT = """#!/bin/bash
run={run}
root={root}
""" + POD_ENV + """export TRITON_CACHE_DIR="$root/alvarez-runtime/cache/triton"
printf '\\nresident worker pod %s started: %s\\n' "$(hostname)" "$(date -Is)" >> "$run/worker.log"
""" + H800_PYTHON + """ -m alvarezgui.train_worker "$run" >> "$run/worker.log" 2>&1 &
worker=$!
trap 'kill -TERM "$worker"' TERM
wait "$worker" || status=$?
trap - TERM
# Any exit (crash, kill) restarts the container: the worker resumes the round that has no COMPLETE yet.
exit "${{status:-1}}"
"""
# Restarts in one round beyond this fail the run instead of looping on a broken pod.
MAX_CONTAINER_RESTARTS = 5
# Container-local heartbeats (never on CFS). Startup covers image pull, imports and the first model load.
WORKER_PROBES = {
    "startupProbe": {"exec": {"command": ["test", "-f", "/tmp/alive"]},
                     "periodSeconds": 10, "failureThreshold": 60},
    "livenessProbe": {"exec": {"command": ["bash", "-c", "[ $(( $(date +%s) - $(stat -c %Y /tmp/alive) )) -lt 60 ]"]},
                      "periodSeconds": 10, "failureThreshold": 3},
}
# The adapter writes its port to ready.json and checks the token on every request, /health included,
# so the probe reads both from the pod's work directory. CFS is read here; a hung mount fails the probe.
ROLLOUT_PROBE_COMMAND = (
    'p=$(sed -n \'s/.*"port": *\\([0-9]*\\).*/\\1/p\' {work}/ready.json) && '
    'curl -sf -m 5 -H "Authorization: Bearer $(cat {work}/token)" "http://127.0.0.1:$p/health" >/dev/null'
)


def rollout_probes(work: Path) -> dict[str, Any]:
    command = ["bash", "-c", ROLLOUT_PROBE_COMMAND.format(work=work)]
    return {"startupProbe": {"exec": {"command": command}, "periodSeconds": 10, "failureThreshold": 90},
            "livenessProbe": {"exec": {"command": command}, "periodSeconds": 10, "failureThreshold": 3,
                              "timeoutSeconds": 10}}


# Rollout services for full MiniCPM-V 4.7 weights run sglang (docs/h800-sglang-rollout.md).
SGLANG_POD_SCRIPT = """#!/bin/bash
work={work}
root={root}
export PYTHONPATH="{src}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Without this torch starts one thread per visible CPU and image preprocessing on a busy node takes ~1.4 s.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HOME="$root/alvarez-runtime/cache/huggingface" TRITON_CACHE_DIR="$root/alvarez-runtime/cache/triton-sglang"
export TMPDIR="$work/tmp"
""" + H800_COMPAT + """mkdir -p "$TMPDIR"
rm -f "$work/ready.json"
if [ ! -f "$work/FAILED" ]; then
    printf '\\npod %s started: %s\\n' "$(hostname)" "$(date -Is)" >> "$work/run.log"
    """ + H800_PYTHON + """ -m alvarezgui.h800_launcher sglang-adapter "$work" >> "$work/run.log" 2>&1 &
    server=$!
    trap 'kill -TERM "$server"' TERM
    wait "$server" || status=$?
    trap - TERM
    if [ "${{status:-0}}" -ge 128 ]; then exit "$status"; fi
    [ -f "$work/FAILED" ] || echo "sglang adapter exited with status ${{status:-0}}" > "$work/FAILED"
fi
exec sleep infinity
"""


def rollout_weights(policy_path: str) -> str:
    """Where the training pod leaves a LoRA round merged into full weights for sglang."""
    return policy_path.rstrip("/") + "-merged"


def sglang_servable(path: str | Path) -> bool:
    """Full MiniCPM-V 4.7 weights with a plain generation policy; everything else stays on HF serving."""
    checkpoint = Path(path)
    if (checkpoint / "adapter_config.json").is_file() or not (checkpoint / "config.json").is_file():
        return False
    if json.loads((checkpoint / "config.json").read_text()).get("model_type") != "minicpmv4_7":
        return False
    return _plain_policy(checkpoint)


def sglang_mergeable(path: str | Path) -> bool:
    """A LoRA adapter over servable MiniCPM-V 4.7 weights: merged in the pod, then served by sglang."""
    checkpoint = Path(path)
    adapter = checkpoint / "adapter_config.json"
    if not adapter.is_file():
        return False
    base = json.loads(adapter.read_text()).get("base_model_name_or_path")
    return bool(base) and sglang_servable(base) and _plain_policy(checkpoint)


def _plain_policy(checkpoint: Path) -> bool:
    policy_path = checkpoint / "policy_config.json"
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
    return (policy.get("inference_mode") is None and not (checkpoint / "perception_config.json").is_file()
            and policy.get("visual_views", "full") in ("full", "full+board-crop"))


def is_online(run: dict[str, Any]) -> bool:
    """GRPO/GIGPO without a dataset samples rollouts with each round's policy."""
    return run["algorithm"] != "sft" and not run.get("dataset_version")


def to_cfs(path: str, root: Path = CFS_ROOT) -> Path:
    if Path(path).is_relative_to(root):
        return Path(path)
    for serving, cfs in PATH_MAP:
        if Path(path).is_relative_to(serving):
            return root / cfs / Path(path).relative_to(serving)
    raise ValueError(f"no CFS mapping for {path}")


def from_cfs(path: str, root: Path = CFS_ROOT) -> str:
    for serving, cfs in PATH_MAP:
        if Path(path).is_relative_to(root / cfs):
            return str(Path(serving) / Path(path).relative_to(root / cfs))
    return path


def cfs_checkpoint(uri: str | None, root: Path = CFS_ROOT) -> Path | None:
    """CFS original of a launcher-published ``/home/alvarez-checkpoints/<run_id>/{final,round-N,epoch-N}``."""
    path = Path(unquote(urlparse(uri).path)) if uri else None
    if path is None or path.parent.parent != Path("/home/alvarez-checkpoints") or not (
            path.name == "final" or re.fullmatch(r"(round|epoch)-[0-9]+", path.name)):
        return None
    run_id = path.parent.name
    if not (root / "alvarez-runs" / run_id / "run.json").is_file():
        return None
    original = root / "alvarez-checkpoints" / run_id / path.name
    return None if original.is_symlink() or original.parent.is_symlink() else original


def cfs_staged_copy(uri: str | None, root: Path = CFS_ROOT) -> Path | None:
    """CFS copy that stage_model() made of a serving-host path, recognised by its sync marker."""
    path = unquote(urlparse(uri).path) if uri else None
    try:
        target = to_cfs(path, root) if path else None
    except ValueError:
        return None
    if target is None or target.is_symlink() or target == Path(path):
        return None
    marker = target / ".alvarez-synced"
    return target if marker.is_file() and marker.read_text() == path else None


def sha256_tree(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                result[str(path.relative_to(directory))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def deployment_manifest(template: dict[str, Any], name: str, run_id: str, script: Path,
                        probes: dict[str, Any] | None = None) -> dict[str, Any]:
    """Copy the template Deployment under a new name and selector, running ``script`` on H800_IMAGE."""
    manifest = copy.deepcopy(template)
    meta = manifest["metadata"]
    labels = {key: value for key, value in meta.get("labels", {}).items() if key not in {"gpu-be-label", "manager"}}
    labels.update({"gpu-be-label": name, "alvarez-run": run_id})
    manifest["metadata"] = {
        "name": name, "namespace": meta["namespace"], "labels": labels,
        "annotations": {key: value for key, value in meta.get("annotations", {}).items() if "queue-name" in key},
    }
    manifest.pop("status", None)
    spec = manifest["spec"]
    spec["replicas"] = 1
    spec["selector"] = {"matchLabels": {"gpu-be-label": name}}
    spec["template"]["metadata"]["labels"] = {"gpu-be-label": name, "alvarez-run": run_id}
    spec["template"]["spec"]["containers"][0]["command"] = ["/bin/bash", str(script)]
    spec["template"]["spec"]["containers"][0]["image"] = H800_IMAGE
    spec["template"]["spec"]["containers"][0].update(probes or {})
    return manifest


class H800Launcher:
    def __init__(self, agent: TrainingAgent, *, kubectl: list[str], template: str, max_runs: int = 4,
                 root: Path = CFS_ROOT, serving_host: str = SERVING_HOST, node_id: str = "h800-k8s",
                 source_dir: Path = Path(__file__).resolve().parent, poll_seconds: float = 15.0,
                 max_serving: int = 2) -> None:
        self.agent, self.cp = agent, agent.cp
        self.kubectl, self.template_name, self.max_runs = kubectl, template, max_runs
        self.root, self.runs_root = root, root / "alvarez-runs"
        self.serving_root, self.max_serving = root / "alvarez-serving", max_serving
        # max_* are the ceilings; the console picks values within them, applied from each heartbeat reply.
        self.run_limit, self.serving_limit = max_runs, max_serving
        self.serving_host, self.node_id, self.source_dir = serving_host, node_id, source_dir
        self.poll_seconds = poll_seconds
        self._template: dict[str, Any] | None = None
        self.copy_threads: dict[str, CopyThread] = {}

    # --- commands ---------------------------------------------------------------

    def command(self, args: list[str], stdin: str | None = None) -> str:
        result = subprocess.run(args, capture_output=True, text=True, input=stdin, timeout=3600)
        if result.returncode:
            raise RuntimeError(f"{' '.join(args[:3])} failed: {result.stderr.strip()[-500:]}")
        return result.stdout

    def kube(self, *args: str, stdin: str | None = None) -> str:
        return self.command([*self.kubectl, *args], stdin)

    def remote(self, script: str, stdin: str | None = None) -> str:
        return self.command(["ssh", "-o", "BatchMode=yes", self.serving_host, script], stdin)

    def template(self) -> dict[str, Any]:
        if self._template is None:
            self._template = json.loads(self.kube("get", "deployment", self.template_name, "-o", "json"))
        return self._template

    @staticmethod
    def deployment_name(run_id: str, work: str = "") -> str:
        """One name per pod workload; GRPO rounds get their own, so a terminating one never blocks the next."""
        suffix = f"-{work.removeprefix('round-')}" if work.startswith("round-") else ""
        return f"alvarez-train-{run_id.removeprefix('run_')[:12]}{suffix}"

    def delete_deployment(self, run_id: str) -> None:
        self.kube("delete", "deployment", "-l", f"alvarez-run={run_id}", "--ignore-not-found", "--wait=false")
        if (self.runs_root / run_id / "serve" / "merged-copies.json").exists():
            self.delete_merged_copies(self.runs_root / run_id)

    def apply_deployment(self, run_id: str, work: Path) -> str:
        name = self.deployment_name(run_id, work.name)
        self.kube("apply", "-f", "-", stdin=json.dumps(
            deployment_manifest(self.template(), name, run_id, work / "pod.sh")))
        return name

    def write_pod(self, work: Path, run_dir: Path, spec: dict[str, Any]) -> None:
        (work / "spec.json").write_text(json.dumps(spec, default=str, indent=2))
        (work / "pod.sh").write_text(POD_SCRIPT.format(run=work, root=self.root, src=run_dir / "src"))

    # --- loop -------------------------------------------------------------------

    def active_dirs(self) -> list[Path]:
        if not self.runs_root.is_dir():
            return []
        return sorted(path for path in self.runs_root.iterdir()
                      if (path / "run.json").is_file() and not (path / "DONE").exists())

    def serve_forever(self) -> None:
        while True:
            try:
                self.tick()
            except Exception:
                log.exception("launcher tick failed")
            time.sleep(self.poll_seconds)

    def tick(self) -> None:
        for run_dir in self.active_dirs():
            try:
                self.advance(run_dir)
            except Exception:
                log.exception("run %s: advance failed; retrying next tick", run_dir.name)
        active = len(self.active_dirs())
        while active < self.run_limit:
            run = self.cp.call("POST", "/internal/v1/training/claim?accelerator=h800")["run"]
            if run is None:
                break
            self.start(run)
            active += 1
        try:
            self.tick_evaluations()
        except Exception:
            log.exception("evaluation serving tick failed")
        active_ids = [path.name for path in self.active_dirs()]
        reply = self.cp.call("POST", f"/internal/v1/nodes/{self.node_id}/heartbeat", {
            "role": "gpu", "address": f"k8s:{self.template_name}",
            "capacity": {"training_runs": self.max_runs, "max_training_runs": self.max_runs,
                         "evaluations": self.max_serving, "max_evaluations": self.max_serving},
            "resources": {"accelerator": "h800", "active_runs": active_ids,
                          "active_evaluations": [path.name for path in self.serving_dirs()]},
            "heartbeat_interval_seconds": max(10, int(self.poll_seconds)),
        })
        capacity = reply.get("capacity", {}) if isinstance(reply, dict) else {}
        # Lowering never stops running pods; it only holds back new ones.
        self.run_limit = capacity.get("training_runs", self.max_runs)
        self.serving_limit = capacity.get("evaluations", self.max_serving)

    # --- staging ----------------------------------------------------------------

    def start(self, run: dict[str, Any]) -> None:
        run_id = run["run_id"]
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps(run, default=str, indent=2))
        self.agent.log(run_id, f"claimed by H800 launcher; staging inputs on CFS {run_dir}")
        try:
            if is_online(run):
                self.stage_online(run, run_dir)
                return
            self.stage(run, run_dir)
            name = self.apply_deployment(run_id, run_dir)
        except Exception as exc:
            log.exception("run %s: staging failed", run_id)
            self.fail(run_id, run_dir, f"H800 staging failed: {exc}")
            return
        self.agent.log(run_id, f"created deployment {name}; waiting for an H800 pod")

    def stage(self, run: dict[str, Any], run_dir: Path) -> None:
        base = self.stage_model(_uri_path(run.get("base_model_checkpoint_uri"), "base model"))
        source = _uri_path(run.get("dataset_source"), "dataset")
        rows = [json.loads(line) for line in self.remote(f"cat {_quote(source)}").splitlines() if line.strip()]
        missing = [row["image"] for row in rows if not to_cfs(row["image"], self.root).is_file()]
        self.sync_files(missing)
        for row in rows:
            row["image"] = str(to_cfs(row["image"], self.root))
        (run_dir / "train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        spec = {
            "run": run, "base_model": str(base), "dataset": str(run_dir / "train.jsonl"),
            "dataset_source": source, "missing_images_synced": len(missing),
            "output_dir": str(self.root / "alvarez-checkpoints" / run["run_id"] / "final"),
            "source_sha256": self.snapshot_source(run_dir),
            "save_every": run["config"].get("checkpoint_every"),
        }
        if run["algorithm"] == "sft":  # The trainer only resumes SFT from epoch state.
            spec["checkpoint_dir"] = str(run_dir / "resume")
        self.write_pod(run_dir, run_dir, spec)
        self.agent.log(run["run_id"], f"staged {len(rows)} rows ({len(missing)} images copied) and base {base}")

    def snapshot_source(self, run_dir: Path) -> dict[str, str]:
        """Copy the trainer source so every pod of this run and later audits use exactly this code."""
        shutil.copytree(self.source_dir, run_dir / "src" / "alvarezgui", ignore=shutil.ignore_patterns("__pycache__"))
        return sha256_tree(run_dir / "src")

    def stage_online(self, run: dict[str, Any], run_dir: Path) -> None:
        base = self.stage_model(_uri_path(run.get("base_model_checkpoint_uri"), "base model"))
        policy = base / "policy_config.json"
        # A LoRA base continues training its adapter; the trainer rejects it for full fine-tuning.
        if policy.is_file() and json.loads(policy.read_text()).get("output_format", "action-v1") != "action-v1":
            raise ValueError("structured checkpoints currently require SFT")
        state = {"round": 0, "policy_version": run["base_model"], "policy_path": str(base), "step_offset": 0,
                 "metrics": {}, "source_sha256": self.snapshot_source(run_dir), "resident": True}
        (run_dir / "online.json").write_text(json.dumps(state, indent=2))
        self.agent.log(run["run_id"], f"online {run['algorithm']}: {run['config']['rounds']} rounds, "
                                      f"sampling and training on H800; base {base}")

    def stage_model(self, path: str) -> Path:
        target = to_cfs(path, self.root)
        marker = target / ".alvarez-synced"
        if not marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            self.command(["rsync", "-a", "--exclude", ".alvarez-synced", f"{self.serving_host}:{path}/", f"{target}/"])
            marker.write_text(path)
        adapter = target / "adapter_config.json"
        if adapter.is_file():
            config = json.loads(adapter.read_text())
            base = str(self.stage_model(config["base_model_name_or_path"]))
            if base != config["base_model_name_or_path"]:
                adapter.write_text(json.dumps({**config, "base_model_name_or_path": base}, indent=2))
        return target

    def sync_files(self, paths: list[str]) -> None:
        for serving, cfs in PATH_MAP:
            relative = [str(Path(path).relative_to(serving)) for path in paths if Path(path).is_relative_to(serving)]
            if relative:
                self.command(["rsync", "-a", "--files-from=-", f"{self.serving_host}:{serving}/",
                              f"{self.root / cfs}/"], "\n".join(relative) + "\n")

    # --- progress ---------------------------------------------------------------

    def advance(self, run_dir: Path) -> None:
        run = json.loads((run_dir / "run.json").read_text())
        run_id = run["run_id"]
        status = self.cp.call("GET", f"/api/v1/training/runs/{run_id}")["status"]
        if status != "RUNNING":
            self.delete_deployment(run_id)
            (run_dir / "DONE").write_text(f"control plane status {status}\n")
            log.info("run %s: %s; deployment deleted", run_id, status)
            return
        if (run_dir / "online.json").exists():
            self.advance_online(run, run_dir)
        elif self.trained(run_id, run_dir, run_dir):
            output = Path(json.loads((run_dir / "spec.json").read_text())["output_dir"])
            metrics = json.loads((output / "metrics.json").read_text())
            self.agent.register_epochs(run, output.parent, lambda path: self.copy_to_serving(run_id, path))
            uri = self.copy_to_serving(run_id, output)
            version = self.agent.register(run, output, metrics, "final", checkpoint_uri=uri)
            self.succeed(run, run_dir, version, metrics, output, uri)

    def trained(self, run_id: str, run_dir: Path, work: Path) -> bool:
        """Relay one pod workload; True once it completed and its Deployment is deleted."""
        self.relay(run_id, work)
        if (work / "COMPLETE").exists():
            self.delete_deployment(run_id)
            return True
        if (work / "FAILED").exists():
            self.fail(run_id, run_dir, "H800 training failed: " + (work / "FAILED").read_text()[-1500:])
        elif not self.kube("get", "deployment", "-l", f"alvarez-run={run_id}", "-o", "name").strip():
            # Lost (deleted externally or the launcher died mid-apply): recreate; training resumes from CFS.
            self.agent.log(run_id, f"recreated missing deployment {self.apply_deployment(run_id, work)}")
        return False

    # --- background checkpoint copies (resident runs) ----------------------------------

    @staticmethod
    def pending_copies(run_dir: Path) -> list[dict[str, Any]]:
        path = run_dir / "copies.json"
        return json.loads(path.read_text()) if path.exists() else []

    def queue_copy(self, run_dir: Path, entry: dict[str, Any]) -> None:
        pending = self.pending_copies(run_dir)
        if all(item["round"] != entry["round"] for item in pending):
            (run_dir / "copies.json").write_text(json.dumps(pending + [entry], indent=2))

    def publish_copies(self, run: dict[str, Any], run_dir: Path) -> bool:
        """Copy finished rounds to the 4090 in a background thread, then register them in order.

        False only when the run failed: a copy whose SHA-256 check fails fails the run, as it did inline.
        """
        run_id = run["run_id"]
        thread = self.copy_threads.get(run_id)
        if thread is not None and thread.is_alive():
            return True
        if thread is not None:
            del self.copy_threads[run_id]
            if thread.error is not None:
                self.fail(run_id, run_dir, f"H800 checkpoint copy failed: {thread.error}")
                return False
            entry = self.pending_copies(run_dir)[0]
            self.agent.register(run, Path(entry["output"]), entry["metrics"], f"r{entry['round']}",
                                checkpoint_uri=thread.uri)
            (run_dir / "copies.json").write_text(json.dumps(self.pending_copies(run_dir)[1:], indent=2))
        pending = self.pending_copies(run_dir)
        if pending:
            thread = CopyThread(self, run_id, Path(pending[0]["output"]))
            self.copy_threads[run_id] = thread
            thread.start()
        return True

    def worker_name(self, run_id: str) -> str:
        return self.deployment_name(run_id) + "-w"

    def ensure_worker(self, run_id: str, run_dir: Path) -> None:
        """Keep the run's resident training worker deployed (created on first use, recreated if lost)."""
        name = self.worker_name(run_id)
        if self.kube("get", "deployment", name, "--ignore-not-found", "-o", "name").strip():
            return
        script = run_dir / "worker.sh"
        script.write_text(WORKER_POD_SCRIPT.format(run=run_dir, root=self.root, src=run_dir / "src"))
        self.kube("apply", "-f", "-", stdin=json.dumps(
            deployment_manifest(self.template(), name, run_id, script, WORKER_PROBES)))
        self.agent.log(run_id, f"created resident training worker {name}")

    def restarts(self, name: str) -> int:
        """Container restarts of the deployment's live pod (kubelet's liveness/crash restarts)."""
        pods = json.loads(self.kube("get", "pods", "-l", f"gpu-be-label={name}", "-o", "json"))["items"]
        return max((status.get("restartCount", 0) for pod in pods if not pod["metadata"].get("deletionTimestamp")
                    for status in pod["status"].get("containerStatuses", [])), default=0)

    def check_restarts(self, run_id: str, run_dir: Path, work: Path, name: str, what: str) -> bool:
        """False (and the run failed) once a pod restarted too often within this round."""
        work.mkdir(parents=True, exist_ok=True)  # A round's rollout service runs before its spec exists.
        marker = work / f"restarts-{what.replace(' ', '-')}.json"
        count = self.restarts(name)
        seen = json.loads(marker.read_text()) if marker.exists() else {"start": count}
        if count < seen["start"]:  # A new pod (recreated deployment) counts from zero again.
            seen = {"start": count}
        marker.write_text(json.dumps(seen))
        if count - seen["start"] > MAX_CONTAINER_RESTARTS:
            self.fail(run_id, run_dir, f"H800 {what} restarted {count - seen['start']} times in {work.name}; "
                                       f"see {work}/run.log and {run_dir}/worker.log")
            return False
        return True

    def trained_resident(self, run_id: str, run_dir: Path, work: Path) -> bool:
        """Like ``trained`` for the resident worker, which stays up for the next round."""
        self.relay(run_id, work)
        if (work / "COMPLETE").exists():
            return True
        if (work / "FAILED").exists():
            self.fail(run_id, run_dir, "H800 training failed: " + (work / "FAILED").read_text()[-1500:])
            return False
        self.ensure_worker(run_id, run_dir)
        self.check_restarts(run_id, run_dir, work, self.worker_name(run_id), "training worker")
        return False

    def advance_online(self, run: dict[str, Any], run_dir: Path) -> None:
        """One step of a per-round state machine kept in online.json, so a restarted launcher resumes."""
        run_id, config = run["run_id"], run["config"]
        state_path = run_dir / "online.json"
        state = json.loads(state_path.read_text())
        number = state["round"]
        if state.get("resident"):
            if not self.publish_copies(run, run_dir):
                return
            if number >= config["rounds"] and self.pending_copies(run_dir):
                return  # The run succeeds only once every round is on the 4090 and registered.
        if number >= config["rounds"]:
            if state["policy_version"] == run["base_model"]:
                self.fail(run_id, run_dir, "no round produced a policy update (all groups had zero reward variance)")
            else:
                self.succeed(run, run_dir, state["policy_version"], state["metrics"],
                             Path(state["policy_path"]), state["checkpoint_uri"])
            return
        work = run_dir / f"round-{number}"
        if not (work / "spec.json").exists():
            if not (work / "REQUESTED").exists():
                if not self.rollout_service_ready(run, state, work):
                    return
                self.agent.log(run_id, f"round {number}: {run['algorithm']} sampling with {state['policy_version']}")
                self.agent.request_rollouts(run, number, state["policy_version"])
                (work / "REQUESTED").write_text(state["policy_version"])
                return
            if not self.rollout_service_ready(run, state, work):
                return  # Queued and running rollouts wait for the replacement pod.
            rollouts = self.agent.finished_rollouts(run, number)
            if rollouts is None:
                return
            self.stop_rollout_service(run_id, work)
            summary = rollout_metrics(rollouts)
            self.agent.metrics(run_id, number, summary)
            rows = self.agent.rows(run, number, rollouts)  # Screenshots land in this round's CFS directory.
            if not rows:
                self.agent.log(run_id, f"round {number}: no group had reward variance; policy unchanged")
                skipped = state["metrics"].get("skipped_rounds", 0) + 1
                state.update(round=number + 1, metrics={**summary, "skipped_rounds": skipped})
                state_path.write_text(json.dumps(state, indent=2))
                return
            (work / "train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            (work / "rollout-summary.json").write_text(json.dumps(summary))
            self.write_pod(work, run_dir, {
                "run": run, "base_model": state["policy_path"], "dataset": str(work / "train.jsonl"),
                "output_dir": str(self.root / "alvarez-checkpoints" / run_id / work.name),
                "step_offset": state["step_offset"],
                "sampling_temperature": run["config"].get("temperature"),
                # The next round's sglang rollout service needs full weights; the last round has none.
                "merge_for_rollout": number + 1 < config["rounds"],
            })
            if state.get("resident"):
                self.agent.log(run_id, f"round {number}: queued for the resident trainer")
            else:
                self.agent.log(run_id, f"round {number}: created deployment {self.apply_deployment(run_id, work)}")
            return
        if not (self.trained_resident(run_id, run_dir, work) if state.get("resident")
                else self.trained(run_id, run_dir, work)):
            return
        output = Path(json.loads((work / "spec.json").read_text())["output_dir"])
        metrics = {**json.loads((work / "rollout-summary.json").read_text()),
                   **json.loads((output / "metrics.json").read_text())}
        if state.get("resident"):
            # The 4090 copy is only for serving and evaluation there; the next round samples from CFS.
            # Registration waits for the verified copy (see publish_copies), so the version name is known now.
            uri = Path(from_cfs(str(output), self.root)).as_uri()
            version = f"{run['config'].get('model_name') or run_id}-r{number}"
            self.queue_copy(run_dir, {"round": number, "output": str(output), "metrics": metrics})
            self.publish_copies(run, run_dir)  # Start it now rather than on the next tick.
        else:
            uri = self.copy_to_serving(run_id, output)
            version = self.agent.register(run, output, metrics, f"r{number}", checkpoint_uri=uri)
        state.update(round=number + 1, policy_version=version, policy_path=str(output), checkpoint_uri=uri,
                     step_offset=state["step_offset"] + metrics["step_count"], metrics=metrics)
        state_path.write_text(json.dumps(state, indent=2))

    # --- H800 rollout serving -------------------------------------------------------
    # One Deployment per round serves that round's policy; it shares the run's alvarez-run label, so
    # fail() and cancellation remove it with the training pod.

    def rollout_service_name(self, run_id: str, work: Path) -> str:
        return self.deployment_name(run_id, work.name).replace("alvarez-train-", "alvarez-roll-", 1)

    def rollout_weights_for(self, state: dict[str, Any]) -> tuple[str, str | None]:
        """(weights to serve, merged copy to delete afterwards or None) for the round's policy."""
        merged = rollout_weights(state["policy_path"])
        if Path(merged).is_dir() and sglang_servable(merged):
            return merged, merged
        return state["policy_path"], None

    def rollout_service_ready(self, run: dict[str, Any], state: dict[str, Any], work: Path) -> bool:
        """Start, watch and publish this round's H800 policy service; True while it answers."""
        if state.get("resident"):
            weights, _ = self.rollout_weights_for(state)
            if sglang_servable(weights):
                return self.resident_rollout_ready(run, state, work)
            # ponytail: e.g. a LoRA round whose merge failed; that round falls back to a per-round HF pod.
            self.stop_resident_rollout(run["run_id"], work.parent)
        run_id, serve = run["run_id"], work / "serve"
        name = self.rollout_service_name(run_id, work)
        if not (serve / "spec.json").exists():
            serve.mkdir(parents=True, exist_ok=True)
            token = serve / "token"
            token.touch(mode=0o600)
            token.write_text(secrets.token_urlsafe(32))
            merged = rollout_weights(state["policy_path"])
            weights = merged if Path(merged).is_dir() else state["policy_path"]
            backend = "sglang" if sglang_servable(weights) else "hf"
            script = SGLANG_POD_SCRIPT if backend == "sglang" else SERVE_POD_SCRIPT
            if backend == "hf":
                weights = state["policy_path"]
            (serve / "pod.sh").write_text(script.format(work=serve, root=self.root, src=run_dir_src(work)))
            spec = {"run_id": run_id, "backend": backend, "models": {state["policy_version"]: weights}}
            if weights == merged:
                spec["merged_copy"] = merged  # The only path stop_rollout_service may delete.
            (serve / "spec.json").write_text(json.dumps(spec, indent=2))
            self.apply_rollout_service(run_id, name, serve)
            self.agent.log(run_id, f"{work.name}: created {backend} rollout service {name} for {state['policy_version']}")
            return False
        if (serve / "FAILED").exists():
            self.fail(run_id, work.parent, "H800 rollout service failed: " + (serve / "FAILED").read_text()[-1500:])
            return False
        if not self.kube("get", "deployment", name, "--ignore-not-found", "-o", "name").strip():
            self.agent.log(run_id, f"{work.name}: recreated missing rollout service {name}")
            self.apply_rollout_service(run_id, name, serve)
        token = (serve / "token").read_text().strip()
        endpoint = self.serving_endpoint(f"gpu-be-label={name}", serve)
        target = endpoint if endpoint is not None and self.healthy(endpoint, token) else None
        # ponytail: published every tick (15 s); compare with the stored value if this gets noisy.
        self.cp.call("PUT", f"/internal/v1/training/runs/{run_id}/model-endpoint",
                     {"endpoint": target, "token": token if target else None})
        return target is not None

    # --- resident rollout service (docs/h800-resident-pods.md) ---------------------------
    # One sglang Deployment per run under <run>/serve; each round's policy is swapped in through next.json.

    def resident_rollout_name(self, run_id: str) -> str:
        return self.deployment_name(run_id).replace("alvarez-train-", "alvarez-roll-", 1)

    def resident_rollout_ready(self, run: dict[str, Any], state: dict[str, Any], work: Path) -> bool:
        run_id, run_dir = run["run_id"], work.parent
        serve, name = run_dir / "serve", self.resident_rollout_name(run_id)
        weights, merged = self.rollout_weights_for(state)
        wanted = {"version": state["policy_version"], "path": weights}
        if not (serve / "spec.json").exists():
            serve.mkdir(parents=True, exist_ok=True)
            token = serve / "token"
            token.touch(mode=0o600)
            token.write_text(secrets.token_urlsafe(32))
            (serve / "pod.sh").write_text(SGLANG_POD_SCRIPT.format(work=serve, root=self.root, src=run_dir / "src"))
            (serve / "next.json").write_text(json.dumps(wanted))
            (serve / "spec.json").write_text(json.dumps(
                {"run_id": run_id, "backend": "sglang", "resident": True, "models": {wanted["version"]: weights}},
                indent=2))
            self.apply_resident_rollout(run_id, name, serve)
            self.agent.log(run_id, f"{work.name}: created resident sglang rollout service {name} for {wanted['version']}")
            return False
        if (serve / "FAILED").exists():
            self.fail(run_id, run_dir, "H800 rollout service failed: " + (serve / "FAILED").read_text()[-1500:])
            return False
        if not self.kube("get", "deployment", name, "--ignore-not-found", "-o", "name").strip():
            self.agent.log(run_id, f"{work.name}: recreated missing resident rollout service {name}")
            self.apply_resident_rollout(run_id, name, serve)
        if not self.check_restarts(run_id, run_dir, work, name, "rollout service"):
            return False
        if not (serve / "next.json").exists() or json.loads((serve / "next.json").read_text()) != wanted:
            pending = serve / "next.json.tmp"
            pending.write_text(json.dumps(wanted))
            pending.replace(serve / "next.json")
            self.agent.log(run_id, f"{work.name}: swapping the resident rollout service to {wanted['version']}")
        copies = self.merged_copies(run_dir)
        if merged and merged not in copies:
            (serve / "merged-copies.json").write_text(json.dumps(copies + [merged]))
        current = json.loads((serve / "current.json").read_text()) if (serve / "current.json").exists() else None
        if current == wanted:
            # A restarted container reloads current.json's path, so only copies it has swapped off may go.
            self.delete_merged_copies(run_dir, keep=current["path"])
        token = (serve / "token").read_text().strip()
        endpoint = self.serving_endpoint(f"gpu-be-label={name}", serve)
        # Publish only once it serves this round's policy: rollouts must never sample the previous one.
        target = (endpoint if current == wanted and endpoint is not None and self.healthy(endpoint, token)
                  else None)
        self.cp.call("PUT", f"/internal/v1/training/runs/{run_id}/model-endpoint",
                     {"endpoint": target, "token": token if target else None})
        return target is not None

    @staticmethod
    def merged_copies(run_dir: Path) -> list[str]:
        path = run_dir / "serve" / "merged-copies.json"
        return json.loads(path.read_text()) if path.exists() else []

    def delete_merged_copies(self, run_dir: Path, keep: str | None = None) -> None:
        """Delete merged copies the resident service made for earlier rounds (only ever these recorded paths)."""
        copies = self.merged_copies(run_dir)
        for path in copies:
            if path != keep:
                shutil.rmtree(path, ignore_errors=True)
        if copies:
            (run_dir / "serve" / "merged-copies.json").write_text(json.dumps([path for path in copies if path == keep]))

    def apply_resident_rollout(self, run_id: str, name: str, serve: Path) -> None:
        (serve / "ready.json").unlink(missing_ok=True)  # The new pod announces its own port.
        self.kube("apply", "-f", "-", stdin=json.dumps(deployment_manifest(
            self.template(), name, run_id, serve / "pod.sh", rollout_probes(serve))))

    def stop_resident_rollout(self, run_id: str, run_dir: Path) -> None:
        """Remove the run's resident rollout service (fallback rounds start fresh from round state)."""
        serve = run_dir / "serve"
        if not (serve / "spec.json").exists():
            return
        self.kube("delete", "deployment", self.resident_rollout_name(run_id), "--ignore-not-found", "--wait=false")
        for name in ("spec.json", "next.json", "current.json", "ready.json", "token"):
            (serve / name).unlink(missing_ok=True)

    def apply_rollout_service(self, run_id: str, name: str, serve: Path) -> None:
        self.kube("apply", "-f", "-", stdin=json.dumps(deployment_manifest(self.template(), name, run_id, serve / "pod.sh")))

    def stop_rollout_service(self, run_id: str, work: Path) -> None:
        name = self.rollout_service_name(run_id, work)
        self.cp.call("PUT", f"/internal/v1/training/runs/{run_id}/model-endpoint", {"endpoint": None, "token": None})
        if not (work / "serve" / "spec.json").exists():
            return  # The run's resident service: it stays up, withdrawn, until the next round's swap.
        self.kube("delete", "deployment", name, "--ignore-not-found", "--wait=false")
        (work / "serve" / "token").unlink(missing_ok=True)
        # The merged copy (3 GB) only fed this round's sglang rollouts; the adapter stays the checkpoint.
        # Never match served paths by name: a registered base model can itself be called "*-merged".
        merged = json.loads((work / "serve" / "spec.json").read_text()).get("merged_copy")
        if merged:
            shutil.rmtree(merged, ignore_errors=True)

    def relay(self, run_id: str, work: Path) -> None:
        offsets_path = work / "relay-offsets.json"
        offsets = json.loads(offsets_path.read_text()) if offsets_path.exists() else {}
        metrics_path, log_path = work / "metrics.jsonl", work / "run.log"
        if metrics_path.exists():
            data, offset = _read_complete_lines(metrics_path, offsets.get("metrics", 0))
            points = [point for line in data for point in json.loads(line)]
            for start in range(0, len(points), 1000):
                self.cp.call("POST", f"/internal/v1/training/runs/{run_id}/metrics", {"points": points[start:start + 1000]})
            offsets["metrics"] = offset
        if log_path.exists():
            data, offset = _read_complete_lines(log_path, offsets.get("log", 0))
            lines = [line[:2000] for line in data if line.strip() and "Kwargs passed to `processor" not in line]
            for start in range(0, len(lines), 500):
                self.cp.call("POST", f"/internal/v1/training/runs/{run_id}/logs", {"lines": lines[start:start + 500]})
            offsets["log"] = offset
        offsets_path.write_text(json.dumps(offsets))

    def copy_to_serving(self, run_id: str, output: Path) -> str:
        """Copy a CFS checkpoint to the same path under /home on the serving host; returns its file URI."""
        target = from_cfs(str(output), self.root)
        hashes = {name: digest for name, digest in sha256_tree(output).items() if name != "adapter_config.json"}
        self.remote(f"mkdir -p {_quote(target)}")
        self.command(["rsync", "-a", "--delete", f"{output}/", f"{self.serving_host}:{target}/"])
        remote_hashes = dict(line.split("  ", 1)[::-1] for line in self.remote(
            f"cd {_quote(target)} && find . -type f ! -name adapter_config.json -printf '%P\\0' | xargs -0 -r sha256sum"
        ).splitlines())
        if remote_hashes != hashes:
            raise RuntimeError("checkpoint copy on the serving host does not match CFS")
        adapter = output / "adapter_config.json"
        if adapter.is_file():
            config = json.loads(adapter.read_text())
            config["base_model_name_or_path"] = from_cfs(config["base_model_name_or_path"], self.root)
            self.remote(f"cat > {_quote(target)}/adapter_config.json", json.dumps(config, indent=2))
        self.agent.log(run_id, f"copied checkpoint to {self.serving_host}:{target}; {len(hashes)} files SHA-256 match")
        return Path(target).as_uri()

    def succeed(self, run: dict[str, Any], run_dir: Path, version: str, metrics: dict[str, Any],
                output: Path, uri: str) -> None:
        run_id = run["run_id"]
        self.delete_deployment(run_id)  # Resident pods outlive the last round; free their GPUs first.
        self.agent.publish(run, version)
        self.agent.finish(run_id, True, {**metrics, "model_version": version, "accelerator": "h800",
                                         "cfs_checkpoint": str(output)}, uri)
        (run_dir / "DONE").write_text("succeeded\n")
        # Resume state only matters for an unfinished run; it can be as large as the model.
        shutil.rmtree(run_dir / "resume", ignore_errors=True)
        self.agent.log(run_id, "finished on H800; deployment deleted")

    def fail(self, run_id: str, run_dir: Path, message: str) -> None:
        self.delete_deployment(run_id)
        self.agent.log(run_id, message)
        self.agent.finish(run_id, False, {"error": message[:500], "accelerator": "h800"}, None)
        (run_dir / "DONE").write_text("failed\n")

    # --- H800 evaluation serving ------------------------------------------------------
    # Deployments are labelled alvarez-run=<eval_id>, so delete_deployment() serves both kinds.

    def serving_dirs(self) -> list[Path]:
        if not self.serving_root.is_dir():
            return []
        return sorted(path for path in self.serving_root.iterdir()
                      if (path / "spec.json").is_file() and not (path / "DONE").exists())

    def tick_evaluations(self) -> None:
        active = {row["eval_id"]: row for row in self.cp.call("GET", "/internal/v1/evaluations/h800")}
        for work in self.serving_dirs():
            try:
                if work.name in active:
                    self.advance_serving(active[work.name], work)
                else:
                    self.stop_serving(work.name, work, "no trial left")
            except Exception:
                log.exception("evaluation %s: serving step failed; retrying next tick", work.name)
        for eval_id, evaluation in active.items():
            work = self.serving_root / eval_id
            if work.is_dir() and not (work / "spec.json").exists() and not (work / "DONE").exists():
                shutil.rmtree(work)  # The launcher stopped while staging; start over.
            # ponytail: one pod per evaluation, capped by serving_limit; share pods if evaluations pile up.
            if not work.exists() and len(self.serving_dirs()) < self.serving_limit:
                self.start_serving(evaluation)

    def apply_serving(self, eval_id: str, work: Path) -> str:
        name = f"alvarez-eval-{eval_id.removeprefix('eval_')[:12]}"
        self.kube("apply", "-f", "-", stdin=json.dumps(deployment_manifest(self.template(), name, eval_id, work / "pod.sh")))
        return name

    def start_serving(self, evaluation: dict[str, Any]) -> None:
        eval_id = evaluation["eval_id"]
        work = self.serving_root / eval_id
        work.mkdir(parents=True)
        try:
            # Per model: sglang for full MiniCPM-V 4.7 weights or a LoRA over them (merged in the pod), else HF.
            # See docs/h800-sglang-evaluation.md.
            models: dict[str, dict[str, str]] = {}
            for version in evaluation["model_versions"]:
                path = self.serving_model(version)
                entry = {"path": str(path), "backend": "sglang" if sglang_servable(path) else "hf"}
                if sglang_mergeable(path):
                    entry = {"path": str(path), "backend": "sglang",
                             "merged_copy": str(work / "merged" / quote(version, safe=""))}
                models[version] = entry
            sglang = any(entry["backend"] == "sglang" for entry in models.values())
            token = work / "token"
            token.touch(mode=0o600)
            token.write_text(secrets.token_urlsafe(32))
            self.snapshot_source(work)
            script = SGLANG_POD_SCRIPT if sglang else SERVE_POD_SCRIPT
            (work / "pod.sh").write_text(script.format(work=work, root=self.root, src=work / "src"))
            # Written last: a directory without spec.json is an interrupted start (see tick_evaluations).
            (work / "spec.json").write_text(json.dumps(
                {"eval_id": eval_id, "models": models, "sglang_memory": 0.8,
                 # Greedy reruns agree 97.9% by default, 100% with this at 11.4 req/s vs 13.9 (sglang-eval-parity-20261124).
                 "sglang_args": ["--enable-deterministic-inference"]}, indent=2))
            name = self.apply_serving(eval_id, work)
        except Exception as exc:
            log.exception("evaluation %s: staging failed", eval_id)
            self.serving_failed(eval_id, work, f"H800 model staging failed: {exc}")
            return
        log.info("evaluation %s: created deployment %s for %s", eval_id, name,
                 {version: entry["backend"] for version, entry in models.items()})

    def serving_model(self, version: str) -> Path:
        """CFS path of a registered model: the launcher's own original, else a copy from the 4090."""
        uri = self.cp.call("GET", f"/internal/v1/model-versions/{quote(version, safe='')}")["checkpoint_uri"]
        original = cfs_checkpoint(uri, self.root)
        if original is not None and original.is_dir():
            return original
        path = _uri_path(uri, version)
        target = to_cfs(path, self.root)
        if target.is_dir() and any(target.iterdir()) and not (target / ".alvarez-synced").exists():
            raise ValueError(f"{target} holds other data; refusing to stage {version} over it")
        return self.stage_model(path)

    def advance_serving(self, evaluation: dict[str, Any], work: Path) -> None:
        eval_id = evaluation["eval_id"]
        if (work / "FAILED").exists():
            self.serving_failed(eval_id, work, "H800 model service failed: " + (work / "FAILED").read_text()[-1500:])
            return
        if not self.kube("get", "deployment", "-l", f"alvarez-run={eval_id}", "-o", "name").strip():
            log.info("evaluation %s: recreated missing deployment %s", eval_id, self.apply_serving(eval_id, work))
            return
        token = (work / "token").read_text().strip()
        endpoint = self.serving_endpoint(f"alvarez-run={eval_id}", work)
        target = endpoint if endpoint is not None and self.healthy(endpoint, token) else None
        if target != evaluation["model_endpoint"]:
            # ponytail: trials already waiting on a pod that just died fail; queued ones wait for the next pod.
            self.cp.call("PUT", f"/internal/v1/evaluations/{eval_id}/model-endpoint",
                         {"endpoint": target, "token": token if target else None})
            log.info("evaluation %s: model endpoint %s", eval_id, target or "withdrawn")

    def serving_endpoint(self, selector: str, work: Path) -> str | None:
        ready = work / "ready.json"
        if not ready.exists():
            return None
        pods = json.loads(self.kube("get", "pods", "-l", selector, "-o", "json"))["items"]
        addresses = [pod["status"].get("podIP") for pod in pods
                     if pod["status"].get("phase") == "Running" and not pod["metadata"].get("deletionTimestamp")]
        if len(addresses) != 1 or not addresses[0]:
            return None
        try:  # A replacement pod's start removes ready.json; it may go between the check above and here.
            return f"http://{addresses[0]}:{json.loads(ready.read_text())['port']}"
        except FileNotFoundError:
            return None

    def healthy(self, endpoint: str, token: str) -> bool:
        request = Request(f"{endpoint}/health", headers={"Authorization": f"Bearer {token}"})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status == 200
        except OSError:
            return False

    def serving_failed(self, eval_id: str, work: Path, message: str) -> None:
        log.error("evaluation %s: %s", eval_id, message)
        self._ignore_missing("POST", f"/internal/v1/evaluations/{eval_id}/serving-failed", {"message": message[:2000]})
        self.stop_serving(eval_id, work, "failed: " + message[:500])

    def stop_serving(self, eval_id: str, work: Path, reason: str) -> None:
        self.delete_deployment(eval_id)
        self._ignore_missing("PUT", f"/internal/v1/evaluations/{eval_id}/model-endpoint",
                             {"endpoint": None, "token": None})
        (work / "token").unlink(missing_ok=True)
        spec_path = work / "spec.json"
        # Only the merged copies spec.json records; registered checkpoints are never touched.
        for entry in (json.loads(spec_path.read_text())["models"].values() if spec_path.exists() else ()):
            if isinstance(entry, dict) and entry.get("merged_copy"):
                for path in (entry["merged_copy"], entry["merged_copy"] + ".tmp"):  # .tmp: an interrupted merge.
                    shutil.rmtree(path, ignore_errors=True)
        (work / "DONE").write_text(reason + "\n")
        log.info("evaluation %s: serving stopped (%s); deployment deleted", eval_id, reason)

    def _ignore_missing(self, method: str, path: str, body: dict[str, Any]) -> None:
        try:
            self.cp.call(method, path, body)
        except HTTPError as exc:
            if exc.code != 404:  # The evaluation was deleted meanwhile.
                raise


class CopyThread(threading.Thread):
    """One ``copy_to_serving`` off the launcher's tick; the tick reads ``uri`` / ``error`` once it ends."""

    def __init__(self, launcher: "H800Launcher", run_id: str, output: Path) -> None:
        super().__init__(daemon=True, name=f"copy-{run_id}")
        self.launcher, self.run_id, self.output = launcher, run_id, output
        self.uri: str | None = None
        self.error: str | None = None

    def run(self) -> None:
        try:
            self.uri = self.launcher.copy_to_serving(self.run_id, self.output)
        except Exception as exc:
            log.exception("run %s: copying %s to the serving host failed", self.run_id, self.output)
            self.error = str(exc)[:500]


def run_dir_src(work: Path) -> Path:
    """The run's source snapshot, for a pod started from one of its round directories."""
    return work.parent / "src"


def _read_complete_lines(path: Path, offset: int) -> tuple[list[str], int]:
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read()
    end = data.rfind(b"\n") + 1
    return data[:end].decode("utf-8", "replace").splitlines(), offset + end


def _uri_path(uri: str | None, what: str) -> str:
    if not uri:
        raise RuntimeError(f"{what} has no path")
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise RuntimeError(f"{what} is not a file path: {uri}")
    return unquote(parsed.path) if parsed.scheme else uri


def _quote(path: str) -> str:
    return "'" + path.replace("'", "'\\''") + "'"


# --- inside the H800 pod ----------------------------------------------------------


def run_pod(run_dir: Path) -> None:
    from alvarezgui.minicpm_training import load_dataset, train

    spec = json.loads((run_dir / "spec.json").read_text())
    run = spec["run"]
    metrics_path = run_dir / "metrics.jsonl"
    offset = spec.get("step_offset", 0)

    def on_step(step: int, values: dict[str, float]) -> None:
        points = [{"name": name, "step": offset + step, "value": float(value)} for name, value in values.items()
                  if value is not None and float(value) == float(value) and abs(float(value)) != float("inf")]
        if points:
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(points) + "\n")

    try:
        rows = load_dataset(spec["dataset"])
        print(f"training {run['run_id']}: {len(rows)} rows from {spec['dataset']}", flush=True)
        metrics = train(run["algorithm"], spec["base_model"], rows, spec["output_dir"],
                        **train_options(run["config"]), checkpoint_dir=spec.get("checkpoint_dir"),
                        save_every=spec.get("save_every"),
                        sampling_temperature=spec.get("sampling_temperature"),
                        compute_old_logprob=run["algorithm"] != "sft", on_step=on_step)
    except Exception:
        (run_dir / "FAILED").write_text(traceback.format_exc())
        raise
    output = Path(spec["output_dir"])
    if spec.get("merge_for_rollout") and (output / "adapter_config.json").is_file():
        from alvarezgui.minicpm_service import merge_adapter

        merged = Path(rollout_weights(str(output)))
        shutil.rmtree(merged, ignore_errors=True)  # A preempted earlier attempt.
        try:
            merge_adapter(output, merged)
        except Exception:
            # ponytail: the next rollout falls back to HF serving of the adapter instead of failing the run.
            traceback.print_exc()
        else:
            print(f"merged adapter for sglang rollout: {merged}", flush=True)
    (run_dir / "COMPLETE").write_text(json.dumps(metrics))
    print(f"completed: {json.dumps(metrics)}", flush=True)


def serve_pod(work: Path) -> None:
    """Preload an evaluation's models, then serve them on a free host-network port behind its token."""
    from alvarezgui.minicpm_service import MiniCPMService, _static_resolver, serve

    spec = json.loads((work / "spec.json").read_text())
    paths = {version: entry if isinstance(entry, str) else entry["path"] for version, entry in spec["models"].items()}
    service = MiniCPMService(_static_resolver(paths), max_loaded=len(paths))
    try:
        for version in spec["models"]:
            service.load(version)  # No trial should wait for a checkpoint load.
    except Exception:
        (work / "FAILED").write_text(traceback.format_exc())
        raise
    print(f"serving {sorted(spec['models'])}", flush=True)
    serve(service, "0.0.0.0", 0, token=(work / "token").read_text().strip(), ready_file=work / "ready.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--control-plane-url", required=True)
    serve.add_argument("--kubeconfig", required=True)
    serve.add_argument("--template-deployment", required=True)
    serve.add_argument("--max-runs", type=int, default=4, help="ceiling for concurrent H800 training pods (console adjusts within)")
    serve.add_argument("--serving-host", default=SERVING_HOST)
    serve.add_argument("--max-serving", type=int, default=2, help="ceiling for concurrent H800 evaluation pods (console adjusts within)")
    pod = commands.add_parser("pod")
    pod.add_argument("run_dir", type=Path)
    serve_pod_parser = commands.add_parser("serve-pod")
    serve_pod_parser.add_argument("work", type=Path)
    commands.add_parser("sglang-adapter").add_argument("work", type=Path)
    args = parser.parse_args()
    if args.command == "sglang-adapter":
        from alvarezgui.sglang_adapter import run

        run(args.work)
        return
    if args.command == "pod":
        run_pod(args.run_dir)
        return
    if args.command == "serve-pod":
        serve_pod(args.work)
        return
    password = os.environ.get("CONSOLE_PASSWORD")
    if not password:
        raise SystemExit("CONSOLE_PASSWORD must be set")
    cp = ControlPlane(args.control_plane_url, password)
    agent = TrainingAgent(cp, node_id="h800-k8s", checkpoint_root=CFS_ROOT / "alvarez-checkpoints",
                          work_root=CFS_ROOT / "alvarez-runs", model_service_url=None)
    H800Launcher(agent, kubectl=["kubectl", "--kubeconfig", args.kubeconfig], template=args.template_deployment,
                 max_runs=args.max_runs, serving_host=args.serving_host, max_serving=args.max_serving).serve_forever()


if __name__ == "__main__":
    main()
