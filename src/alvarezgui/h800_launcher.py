"""H800 training launcher for the control-plane host, which mounts the same CFS path as the H800 pods.

``serve`` claims ``accelerator=h800`` runs, stages the base model and dataset on CFS, starts one
Kubernetes Deployment per run (copied from a template), relays CFS metrics/logs to the control plane,
then copies the checkpoint to the 4090 serving host, registers it and deletes the Deployment.
Online GRPO repeats this per round: rollouts are sampled through the 4090 serving host with the last
published policy, then a fresh Deployment trains the next policy on H800.
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
POD_ENV = """export PYTHONPATH="{src}:$root/alvarez-runtime/deps"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$root/alvarez-runtime/cache/huggingface" TORCH_HOME="$root/alvarez-runtime/cache/torch"
"""
POD_SCRIPT = """#!/bin/bash
run={run}
root={root}
""" + POD_ENV + """export TRITON_CACHE_DIR="$root/alvarez-runtime/cache/triton" TMPDIR="$run/tmp"
mkdir -p "$TMPDIR"
if [ ! -f "$run/COMPLETE" ] && [ ! -f "$run/FAILED" ]; then
    printf '\\npod %s started: %s\\n' "$(hostname)" "$(date -Is)" >> "$run/run.log"
    # Forward SIGTERM so a deleted/preempted pod's trainer stops before its replacement resumes.
    /opt/nemo_rl_venv/bin/python -m alvarezgui.h800_launcher pod "$run" >> "$run/run.log" 2>&1 &
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
    /opt/nemo_rl_venv/bin/python -m alvarezgui.h800_launcher serve-pod "$work" >> "$work/run.log" 2>&1 &
    server=$!
    trap 'kill -TERM "$server"' TERM
    wait "$server" || status=$?
    trap - TERM
    if [ "${{status:-0}}" -ge 128 ]; then exit "$status"; fi
    [ -f "$work/FAILED" ] || echo "model service exited with status ${{status:-0}}" > "$work/FAILED"
fi
exec sleep infinity
"""


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


def deployment_manifest(template: dict[str, Any], name: str, run_id: str, script: Path) -> dict[str, Any]:
    """Copy the template Deployment under a new name and selector, running ``script`` with one replica."""
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
        if (base / "adapter_config.json").is_file() or (
                policy.is_file() and json.loads(policy.read_text()).get("output_format", "action-v1") != "action-v1"):
            raise ValueError("adapter and structured checkpoints currently require SFT")
        state = {"round": 0, "policy_version": run["base_model"], "policy_path": str(base), "step_offset": 0,
                 "metrics": {}, "source_sha256": self.snapshot_source(run_dir)}
        (run_dir / "online.json").write_text(json.dumps(state, indent=2))
        self.agent.log(run["run_id"], f"online {run['algorithm']}: {run['config']['rounds']} rounds, "
                                      f"sampling on the 4090 serving host, training on H800; base {base}")

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

    def advance_online(self, run: dict[str, Any], run_dir: Path) -> None:
        """One step of a per-round state machine kept in online.json, so a restarted launcher resumes."""
        run_id, config = run["run_id"], run["config"]
        state_path = run_dir / "online.json"
        state = json.loads(state_path.read_text())
        number = state["round"]
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
                self.agent.log(run_id, f"round {number}: {run['algorithm']} sampling with {state['policy_version']}")
                self.agent.request_rollouts(run, number, state["policy_version"])
                work.mkdir(parents=True, exist_ok=True)
                (work / "REQUESTED").write_text(state["policy_version"])
                return
            rollouts = self.agent.finished_rollouts(run, number)
            if rollouts is None:
                return
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
            })
            self.agent.log(run_id, f"round {number}: created deployment {self.apply_deployment(run_id, work)}")
            return
        if not self.trained(run_id, run_dir, work):
            return
        output = Path(json.loads((work / "spec.json").read_text())["output_dir"])
        metrics = {**json.loads((work / "rollout-summary.json").read_text()),
                   **json.loads((output / "metrics.json").read_text())}
        uri = self.copy_to_serving(run_id, output)
        version = self.agent.register(run, output, metrics, f"r{number}", checkpoint_uri=uri)
        state.update(round=number + 1, policy_version=version, policy_path=str(output), checkpoint_uri=uri,
                     step_offset=state["step_offset"] + metrics["step_count"], metrics=metrics)
        state_path.write_text(json.dumps(state, indent=2))

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
            models = {version: str(self.serving_model(version)) for version in evaluation["model_versions"]}
            token = work / "token"
            token.touch(mode=0o600)
            token.write_text(secrets.token_urlsafe(32))
            self.snapshot_source(work)
            (work / "pod.sh").write_text(SERVE_POD_SCRIPT.format(work=work, root=self.root, src=work / "src"))
            # Written last: a directory without spec.json is an interrupted start (see tick_evaluations).
            (work / "spec.json").write_text(json.dumps({"eval_id": eval_id, "models": models}, indent=2))
            name = self.apply_serving(eval_id, work)
        except Exception as exc:
            log.exception("evaluation %s: staging failed", eval_id)
            self.serving_failed(eval_id, work, f"H800 model staging failed: {exc}")
            return
        log.info("evaluation %s: created deployment %s for %s", eval_id, name, sorted(models))

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
        endpoint = self.serving_endpoint(eval_id, work)
        target = endpoint if endpoint is not None and self.healthy(endpoint, token) else None
        if target != evaluation["model_endpoint"]:
            # ponytail: trials already waiting on a pod that just died fail; queued ones wait for the next pod.
            self.cp.call("PUT", f"/internal/v1/evaluations/{eval_id}/model-endpoint",
                         {"endpoint": target, "token": token if target else None})
            log.info("evaluation %s: model endpoint %s", eval_id, target or "withdrawn")

    def serving_endpoint(self, eval_id: str, work: Path) -> str | None:
        ready = work / "ready.json"
        if not ready.exists():
            return None
        pods = json.loads(self.kube("get", "pods", "-l", f"alvarez-run={eval_id}", "-o", "json"))["items"]
        addresses = [pod["status"].get("podIP") for pod in pods
                     if pod["status"].get("phase") == "Running" and not pod["metadata"].get("deletionTimestamp")]
        if len(addresses) != 1 or not addresses[0]:
            return None
        return f"http://{addresses[0]}:{json.loads(ready.read_text())['port']}"

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
        (work / "DONE").write_text(reason + "\n")
        log.info("evaluation %s: serving stopped (%s); deployment deleted", eval_id, reason)

    def _ignore_missing(self, method: str, path: str, body: dict[str, Any]) -> None:
        try:
            self.cp.call(method, path, body)
        except HTTPError as exc:
            if exc.code != 404:  # The evaluation was deleted meanwhile.
                raise


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
    (run_dir / "COMPLETE").write_text(json.dumps(metrics))
    print(f"completed: {json.dumps(metrics)}", flush=True)


def serve_pod(work: Path) -> None:
    """Preload an evaluation's models, then serve them on a free host-network port behind its token."""
    from alvarezgui.minicpm_service import MiniCPMService, _static_resolver, serve

    spec = json.loads((work / "spec.json").read_text())
    service = MiniCPMService(_static_resolver(spec["models"]), max_loaded=len(spec["models"]))
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
    args = parser.parse_args()
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
