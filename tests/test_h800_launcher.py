import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from alvarezgui.h800_launcher import (
    WORKER_PROBES,
    H800Launcher,
    _read_complete_lines,
    cfs_checkpoint,
    cfs_staged_copy,
    deployment_manifest,
    from_cfs,
    sha256_tree,
    to_cfs,
)
from alvarezgui.training_agent import epoch_checkpoints

TEMPLATE = {
    "apiVersion": "apps/v1", "kind": "Deployment",
    "metadata": {"name": "template", "namespace": "ns", "uid": "x",
                 "labels": {"gpu-be-label": "template", "manager": "m", "volcano.sh/queue-name": "q"},
                 "annotations": {"volcano.sh/queue-name": "q", "kubectl.kubernetes.io/last-applied-configuration": "{}"}},
    "spec": {"replicas": 0, "selector": {"matchLabels": {"gpu-be-label": "template"}},
             "template": {"metadata": {"labels": {"gpu-be-label": "template"}},
                          "spec": {"containers": [{"name": "c", "command": ["sleep", "infinity"]}]}}},
    "status": {"replicas": 0},
}


class FakeLauncher(H800Launcher):
    """Records kubectl calls; ``remote``/``command`` act on a local "serving host" directory."""

    def __init__(self, root: Path, serving: Path, cp: MagicMock) -> None:
        agent = MagicMock()
        agent.cp = cp
        agent.register.return_value = "run_abc-final"
        super().__init__(agent, kubectl=["kubectl"], template="template", root=root, serving_host="gpu")
        self.serving, self.deployments = serving, {}

    def kube(self, *args: str, stdin: str | None = None) -> str:
        if args[0] == "get" and args[1] == "deployment" and args[2] == "template":
            return json.dumps(TEMPLATE)
        if args[0] == "apply":
            manifest = json.loads(stdin)
            self.deployments[manifest["metadata"]["name"]] = manifest
            return ""
        if args[0] == "delete" and args[2] != "-l":  # By name.
            self.deployments.pop(args[2], None)
            return ""
        if args[0] == "delete":
            run_id = args[3].split("=", 1)[1]
            self.deployments = {name: m for name, m in self.deployments.items()
                                if m["metadata"]["labels"]["alvarez-run"] != run_id}
            return ""
        if args[0] == "get" and args[1] == "pods":
            key, value = args[3].split("=", 1)
            return json.dumps({"items": [
                {"metadata": {}, "status": {"phase": "Running", "podIP": "10.0.0.5"}}
                for m in self.deployments.values() if m["spec"]["template"]["metadata"]["labels"][key] == value]})
        if args[0] == "get" and args[2] != "-l":  # By name.
            return f"deployment/{args[2]}\n" if args[2] in self.deployments else ""
        if args[0] == "get":
            run_id = args[3].split("=", 1)[1]
            return "".join(f"deployment/{name}\n" for name, m in self.deployments.items()
                           if m["metadata"]["labels"]["alvarez-run"] == run_id)
        raise AssertionError(args)

    def local(self, path: str) -> Path:
        return self.serving / path.lstrip("/")

    def remote(self, script: str, stdin: str | None = None) -> str:
        if script.startswith("cat > "):
            self.local(script.split("'")[1] + "/adapter_config.json").write_text(stdin)
            return ""
        if script.startswith("cat "):
            return self.local(script.split("'")[1]).read_text()
        if script.startswith("mkdir -p "):
            self.local(script.split("'")[1]).mkdir(parents=True, exist_ok=True)
            return ""
        target = self.local(script.split("'")[1])
        return "".join(f"{digest}  {name}\n" for name, digest in sha256_tree(target).items()
                       if name != "adapter_config.json")

    def command(self, args: list[str], stdin: str | None = None) -> str:
        assert args[0] == "rsync", args
        source, target = args[-2], args[-1]
        if source.startswith("gpu:"):
            source = str(self.local(source[4:]))
        if target.startswith("gpu:"):
            target = str(self.local(target[4:]))
        names = stdin.split() if stdin else [str(p.relative_to(source)) for p in Path(source).rglob("*") if p.is_file()]
        for name in names:
            if name == ".alvarez-synced":
                continue
            Path(target, name).parent.mkdir(parents=True, exist_ok=True)
            Path(target, name).write_bytes(Path(source, name).read_bytes())
        return ""


class H800LauncherTest(unittest.TestCase):
    def test_path_mapping_round_trips(self) -> None:
        root = Path("/cfs")
        self.assertEqual(to_cfs("/home/alvarez-models/M", root), Path("/cfs/alvarez-models/M"))
        self.assertEqual(to_cfs("/home/alvarez-datasets/d/a.png", root), Path("/cfs/alvarez-data/d/a.png"))
        self.assertEqual(to_cfs("/cfs/alvarez-models/M", root), Path("/cfs/alvarez-models/M"))
        self.assertEqual(from_cfs("/cfs/alvarez-checkpoints/run_x/final", root), "/home/alvarez-checkpoints/run_x/final")
        with self.assertRaises(ValueError):
            to_cfs("/etc/passwd", root)

    def test_manifest_copies_template_under_new_name(self) -> None:
        manifest = deployment_manifest(TEMPLATE, "alvarez-train-abc", "run_abc", Path("/cfs/pod.sh"))
        self.assertEqual(manifest["metadata"], {
            "name": "alvarez-train-abc", "namespace": "ns",
            "labels": {"volcano.sh/queue-name": "q", "gpu-be-label": "alvarez-train-abc", "alvarez-run": "run_abc"},
            "annotations": {"volcano.sh/queue-name": "q"},
        })
        self.assertEqual(manifest["spec"]["replicas"], 1)
        self.assertEqual(manifest["spec"]["selector"], {"matchLabels": {"gpu-be-label": "alvarez-train-abc"}})
        self.assertEqual(manifest["spec"]["template"]["spec"]["containers"][0]["command"], ["/bin/bash", "/cfs/pod.sh"])
        self.assertNotIn("status", manifest)
        self.assertEqual(TEMPLATE["spec"]["replicas"], 0)

    def test_partial_lines_wait_for_newline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "log")
            path.write_bytes(b"one\ntw")
            self.assertEqual(_read_complete_lines(path, 0), (["one"], 4))
            path.write_bytes(b"one\ntwo\n")
            self.assertEqual(_read_complete_lines(path, 4), (["two"], 8))

    def test_run_lifecycle_stage_relay_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, serving = Path(directory, "cfs"), Path(directory, "gpu")
            image = serving / "home/alvarez-datasets/d/images/0.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            (serving / "home/alvarez-datasets/d/train.jsonl").write_text(json.dumps({"image": "/home/alvarez-datasets/d/images/0.png"}) + "\n")
            model = serving / "home/alvarez-models/M"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}")
            cp = MagicMock()
            claims = iter([{"run": {"run_id": "run_abc", "algorithm": "sft", "base_model": "base",
                                    "config": {"checkpoint_every": 2},
                                    "base_model_checkpoint_uri": "file:///home/alvarez-models/M",
                                    "dataset_source": "file:///home/alvarez-datasets/d/train.jsonl"}}])
            cp.call.side_effect = lambda method, path, *a, **k: (
                next(claims, {"run": None}) if path.startswith("/internal/v1/training/claim")
                else [] if path == "/internal/v1/evaluations/h800"
                else {"status": "RUNNING"} if method == "GET" else {})
            launcher = FakeLauncher(root, serving, cp)
            launcher.tick()

            run_dir = root / "alvarez-runs/run_abc"
            self.assertEqual(list(launcher.deployments), ["alvarez-train-abc"])
            spec = json.loads((run_dir / "spec.json").read_text())
            self.assertEqual(spec["base_model"], str(root / "alvarez-models/M"))
            self.assertEqual(spec["save_every"], 2)
            row = json.loads((run_dir / "train.jsonl").read_text())
            self.assertEqual(Path(row["image"]).read_bytes(), b"png")
            self.assertTrue((run_dir / "src/alvarezgui/h800_launcher.py").is_file())
            self.assertIn(str(run_dir), (run_dir / "pod.sh").read_text())

            # A lost deployment is recreated; training resumes from CFS state.
            launcher.deployments.clear()
            launcher.tick()
            self.assertEqual(list(launcher.deployments), ["alvarez-train-abc"])

            (run_dir / "metrics.jsonl").write_text(json.dumps([{"name": "train/loss", "step": 1, "value": 0.5}]) + "\n")
            (run_dir / "run.log").write_text("step 1\n[transformers] Kwargs passed to `processor.__call__`\n")
            output = Path(spec["output_dir"])
            output.mkdir(parents=True)
            (output / "adapter_model.safetensors").write_bytes(b"weights")
            (output / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(root / "alvarez-models/M")}))
            (output / "metrics.json").write_text(json.dumps({"loss": 0.1}))
            (run_dir / "resume").mkdir()
            (output.parent / "epoch-2").mkdir()
            (output.parent / "epoch-2/adapter_model.safetensors").write_bytes(b"epoch-2")
            launcher.agent.register_epochs.side_effect = lambda run, directory, copy: [
                copy(path) for path in epoch_checkpoints(directory)]
            (run_dir / "COMPLETE").write_text("{}")
            launcher.tick()
            self.assertEqual((serving / "home/alvarez-checkpoints/run_abc/epoch-2/adapter_model.safetensors").read_bytes(),
                             b"epoch-2")
            launcher.agent.publish.assert_called_once()

            self.assertEqual(launcher.deployments, {})
            target = serving / "home/alvarez-checkpoints/run_abc/final"
            self.assertEqual((target / "adapter_model.safetensors").read_bytes(), b"weights")
            self.assertEqual(json.loads((target / "adapter_config.json").read_text()),
                             {"base_model_name_or_path": "/home/alvarez-models/M"})
            uri = "file:///home/alvarez-checkpoints/run_abc/final"
            self.assertEqual(launcher.agent.register.call_args.kwargs["checkpoint_uri"], uri)
            launcher.agent.finish.assert_called_once()
            self.assertEqual(launcher.agent.finish.call_args.args[:2], ("run_abc", True))
            self.assertTrue((run_dir / "DONE").exists())
            self.assertFalse((run_dir / "resume").exists())
            posted = {call.args[1]: call.args[2] for call in cp.call.call_args_list if len(call.args) > 2}
            self.assertEqual(posted["/internal/v1/training/runs/run_abc/metrics"]["points"][0]["value"], 0.5)
            self.assertEqual(posted["/internal/v1/training/runs/run_abc/logs"], {"lines": ["step 1"]})
            self.assertEqual(cfs_checkpoint(uri, root), root / "alvarez-checkpoints/run_abc/final")
            self.assertEqual(cfs_checkpoint("file:///home/alvarez-checkpoints/run_abc/round-1", root),
                             root / "alvarez-checkpoints/run_abc/round-1")
            self.assertEqual(cfs_checkpoint("file:///home/alvarez-checkpoints/run_abc/epoch-2", root),
                             root / "alvarez-checkpoints/run_abc/epoch-2")
            self.assertIsNone(cfs_checkpoint("file:///home/alvarez-checkpoints/run_abc/epoch-x", root))
            self.assertIsNone(cfs_checkpoint("file:///home/alvarez-checkpoints/other/final", root))

    def test_online_grpo_trains_each_round_on_h800_and_samples_with_published_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, serving = Path(directory, "cfs"), Path(directory, "gpu")
            model = serving / "home/alvarez-models/M"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}")
            run = {"run_id": "run_abc", "algorithm": "grpo", "base_model": "base", "dataset_version": None,
                   "base_model_checkpoint_uri": "file:///home/alvarez-models/M",
                   "config": {"rounds": 3, "temperature": 0.7, "tasks": [{"task_type": "sokoban", "app_image": "a"}]}}
            claims, calls = iter([{"run": run}]), []

            def call(method, path, body=None, *args, **kwargs):
                calls.append((method, path, body))
                return (next(claims, {"run": None}) if path.startswith("/internal/v1/training/claim")
                        else [] if path == "/internal/v1/evaluations/h800"
                        else {"status": "RUNNING"} if method == "GET" else {})

            cp = MagicMock()
            cp.call.side_effect = call
            launcher = FakeLauncher(root, serving, cp)
            agent = launcher.agent
            agent.register.side_effect = lambda run, output, metrics, suffix, checkpoint_uri: f"abc-{suffix}"
            variance = {0: True, 1: False, 2: True}  # Round 1 has no reward variance and is skipped.
            agent.finished_rollouts.side_effect = lambda run, number: [
                {"outcome": "success" if variance[number] else "max_steps", "group_id": "g", "reward_total": 1.0,
                 "steps": 1, "episode_id": "e"}]
            agent.rows.side_effect = lambda run, number, rollouts: [{"id": number}] if variance[number] else []
            run_dir = root / "alvarez-runs/run_abc"

            def finish_pod(number: int) -> None:
                work = run_dir / f"round-{number}"
                spec = json.loads((work / "spec.json").read_text())
                output = Path(spec["output_dir"])
                output.mkdir(parents=True)
                (output / "model.safetensors").write_bytes(f"round-{number}".encode())
                (output / "metrics.json").write_text(json.dumps({"loss": 0.1, "step_count": 5}))
                (work / "COMPLETE").write_text("{}")

            launcher.healthy = MagicMock(return_value=True)
            endpoints = lambda: [body for method, path, body in calls if path == "/internal/v1/training/runs/run_abc/model-endpoint"]

            def serve_round(number: int, policy: str) -> None:
                """Start the round's H800 rollout service, let it answer, then sample with it."""
                launcher.tick()
                serve = run_dir / f"round-{number}/serve"
                self.assertEqual(json.loads((serve / "spec.json").read_text())["models"],
                                 {policy: str(launcher.root / ("alvarez-models/M" if number == 0 else "alvarez-checkpoints/run_abc/round-0"))})
                self.assertEqual(list(launcher.deployments), [f"alvarez-roll-abc-{number}"])
                self.assertIn("h800_launcher serve-pod", (serve / "pod.sh").read_text())
                agent.request_rollouts.reset_mock()
                launcher.tick()  # Pod not listening yet: nothing is sampled.
                agent.request_rollouts.assert_not_called()
                self.assertEqual(endpoints()[-1], {"endpoint": None, "token": None})
                (serve / "ready.json").write_text(json.dumps({"port": 4321}))
                launcher.tick()
                token = (serve / "token").read_text()
                self.assertEqual(endpoints()[-1], {"endpoint": "http://10.0.0.5:4321", "token": token})
                self.assertEqual(agent.request_rollouts.call_args.args[1:], (number, policy))

            launcher.tick()  # Claim and stage.
            # Runs staged before resident pods keep the per-round pods they were started with.
            state_path = run_dir / "online.json"
            state_path.write_text(json.dumps({**json.loads(state_path.read_text()), "resident": False}))
            serve_round(0, "base")
            launcher.tick()  # Rollouts done: stop the rollout service, stage round 0 training pod.
            self.assertEqual(endpoints()[-1], {"endpoint": None, "token": None})
            spec = json.loads((run_dir / "round-0/spec.json").read_text())
            self.assertEqual((spec["base_model"], spec["step_offset"]), (str(root / "alvarez-models/M"), 0))
            self.assertEqual(spec["sampling_temperature"], 0.7)
            self.assertEqual(list(launcher.deployments), ["alvarez-train-abc-0"])
            self.assertIn(str(run_dir / "src"), (run_dir / "round-0/pod.sh").read_text())
            finish_pod(0)
            launcher.tick()  # Publish round 0 to the serving host.
            self.assertEqual(launcher.deployments, {})
            self.assertEqual((serving / "home/alvarez-checkpoints/run_abc/round-0/model.safetensors").read_bytes(),
                             b"round-0")
            serve_round(1, "abc-r0")  # Round 1 sampled with the round-0 policy...
            launcher.tick()  # ...and skipped.
            self.assertEqual(launcher.deployments, {})
            serve_round(2, "abc-r0")
            launcher.tick()
            spec = json.loads((run_dir / "round-2/spec.json").read_text())
            self.assertEqual((spec["base_model"], spec["step_offset"]),
                             (str(root / "alvarez-checkpoints/run_abc/round-0"), 5))
            self.assertEqual(list(launcher.deployments), ["alvarez-train-abc-2"])
            finish_pod(2)
            launcher.tick()
            agent.finish.assert_not_called()
            launcher.tick()  # All rounds done.
            self.assertEqual(agent.finish.call_args.args[:2], ("run_abc", True))
            self.assertEqual(agent.finish.call_args.args[2]["model_version"], "abc-r2")
            self.assertEqual(agent.finish.call_args.args[3], "file:///home/alvarez-checkpoints/run_abc/round-2")
            agent.publish.assert_called_once_with(run | {"run_id": "run_abc"}, "abc-r2")
            self.assertTrue((run_dir / "DONE").exists())

    def test_resident_online_run_keeps_one_worker_and_one_rollout_service(self) -> None:
        """docs/h800-resident-pods.md: one worker and one sglang service for the whole run, weights swapped
        per round, endpoint published only for the round's own policy, 4090 copies off the critical path."""
        with tempfile.TemporaryDirectory() as directory:
            root, serving = Path(directory, "cfs"), Path(directory, "gpu")
            model = serving / "home/alvarez-models/M"
            model.mkdir(parents=True)
            (model / "config.json").write_text(json.dumps({"model_type": "minicpmv4_7"}))
            run = {"run_id": "run_abc", "algorithm": "grpo", "base_model": "base", "dataset_version": None,
                   "base_model_checkpoint_uri": "file:///home/alvarez-models/M",
                   "config": {"rounds": 2, "temperature": 1.0, "tasks": [{"task_type": "sokoban", "app_image": "a"}]}}
            claims, calls = iter([{"run": run}]), []

            def call(method, path, body=None, *args, **kwargs):
                calls.append((method, path, body))
                return (next(claims, {"run": None}) if path.startswith("/internal/v1/training/claim")
                        else [] if path == "/internal/v1/evaluations/h800"
                        else {"status": "RUNNING"} if method == "GET" else {})

            cp = MagicMock()
            cp.call.side_effect = call
            launcher = FakeLauncher(root, serving, cp)
            launcher.healthy = MagicMock(return_value=True)
            agent = launcher.agent
            agent.register.side_effect = lambda run, output, metrics, suffix, checkpoint_uri: f"run_abc-{suffix}"
            agent.finished_rollouts.side_effect = lambda run, number: [
                {"outcome": "success", "group_id": "g", "reward_total": 1.0, "steps": 1, "episode_id": "e"}]
            agent.rows.side_effect = lambda run, number, rollouts: [{"id": number}]
            run_dir = root / "alvarez-runs/run_abc"
            serve = run_dir / "serve"
            endpoints = lambda: [body for method, path, body in calls
                                 if path == "/internal/v1/training/runs/run_abc/model-endpoint"]
            copied = []
            launcher.copy_to_serving = lambda run_id, output: copied.append(output.name) or f"file:///home/{output.name}"

            def swapped(version: str, path: Path) -> None:
                """What the adapter does on next.json: serve ``version`` and say so in current.json."""
                self.assertEqual(json.loads((serve / "next.json").read_text()), {"version": version, "path": str(path)})
                (serve / "current.json").write_text(json.dumps({"version": version, "path": str(path)}))

            def finish_round(number: int, merged: bool) -> Path:
                spec = json.loads((run_dir / f"round-{number}/spec.json").read_text())
                output = Path(spec["output_dir"])
                output.mkdir(parents=True)
                (output / "adapter_config.json").write_text("{}")
                (output / "metrics.json").write_text(json.dumps({"loss": 0.1, "step_count": 5}))
                if merged:  # The worker's merge for the next round's sglang rollouts.
                    copy = Path(str(output) + "-merged")
                    copy.mkdir()
                    (copy / "config.json").write_text(json.dumps({"model_type": "minicpmv4_7"}))
                (run_dir / f"round-{number}/COMPLETE").write_text("{}")
                return output

            launcher.tick()  # Claim and stage.
            self.assertTrue(json.loads((run_dir / "online.json").read_text())["resident"])
            launcher.tick()  # The resident rollout service for round 0.
            self.assertEqual(list(launcher.deployments), ["alvarez-roll-abc"])
            container = launcher.deployments["alvarez-roll-abc"]["spec"]["template"]["spec"]["containers"][0]
            self.assertIn("livenessProbe", container)
            self.assertIn(str(serve / "ready.json"), container["livenessProbe"]["exec"]["command"][2])
            self.assertTrue(json.loads((serve / "spec.json").read_text())["resident"])
            base = root / "alvarez-models/M"
            (serve / "ready.json").write_text(json.dumps({"port": 4321}))
            launcher.tick()  # Answering, but not yet on round 0's policy: nothing is published or sampled.
            agent.request_rollouts.assert_not_called()
            self.assertEqual(endpoints()[-1], {"endpoint": None, "token": None})
            swapped("base", base)
            launcher.tick()
            self.assertEqual(endpoints()[-1]["endpoint"], "http://10.0.0.5:4321")
            self.assertEqual(agent.request_rollouts.call_args.args[1:], (0, "base"))
            launcher.tick()  # Rollouts done: withdraw (the service stays), queue round 0 for the worker.
            self.assertEqual(endpoints()[-1], {"endpoint": None, "token": None})
            self.assertEqual(set(launcher.deployments), {"alvarez-roll-abc"})
            launcher.tick()  # The worker appears with its probes.
            self.assertEqual(set(launcher.deployments), {"alvarez-roll-abc", "alvarez-train-abc-w"})
            worker = launcher.deployments["alvarez-train-abc-w"]["spec"]["template"]["spec"]["containers"][0]
            self.assertEqual(worker["livenessProbe"], WORKER_PROBES["livenessProbe"])
            self.assertIn("alvarezgui.train_worker", (run_dir / "worker.sh").read_text())
            del launcher.deployments["alvarez-train-abc-w"]
            launcher.tick()  # Lost worker: recreated.
            self.assertIn("alvarez-train-abc-w", launcher.deployments)
            output0 = finish_round(0, merged=True)
            launcher.tick()  # Round 0 done: copy in the background, swap the service to round 0's merge.
            launcher.copy_threads["run_abc"].join()
            self.assertEqual(copied, ["round-0"])
            launcher.tick()  # Copy verified: register round 0. Not yet swapped: nothing published.
            self.assertEqual(agent.register.call_args.args[3], "r0")
            self.assertEqual(agent.register.call_args.kwargs["checkpoint_uri"], "file:///home/round-0")
            merged0 = Path(str(output0) + "-merged")
            swapped("run_abc-r0", merged0)
            launcher.tick()
            self.assertEqual(agent.request_rollouts.call_args.args[1:], (1, "run_abc-r0"))
            self.assertEqual(set(launcher.deployments), {"alvarez-roll-abc", "alvarez-train-abc-w"})
            self.assertTrue(merged0.is_dir())  # In use by the service.
            launcher.tick()  # Round 1 rollouts done.
            finish_round(1, merged=False)  # The last round is not merged.
            launcher.tick()
            launcher.copy_threads["run_abc"].join()
            agent.finish.assert_not_called()  # Round 1 is not on the 4090 yet.
            launcher.tick()  # Register round 1; all rounds on the 4090: succeed and delete both deployments.
            self.assertEqual(agent.finish.call_args.args[:2], ("run_abc", True))
            self.assertEqual(agent.finish.call_args.args[2]["model_version"], "run_abc-r1")
            self.assertEqual(launcher.deployments, {})
            self.assertFalse(merged0.exists())  # The service's merged copies go with it.
            self.assertTrue(output0.exists())  # The checkpoint itself stays.

    def test_resident_pod_restart_cap_fails_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeLauncher(Path(directory, "cfs"), Path(directory, "gpu"), MagicMock())
            launcher.fail = MagicMock()
            work = Path(directory, "round-3")
            work.mkdir()
            for count, ok in ((2, True), (7, True), (8, False)):
                launcher.restarts = MagicMock(return_value=count)
                self.assertEqual(launcher.check_restarts("run_abc", work.parent, work, "w", "training worker"), ok)
            self.assertIn("restarted 6 times in round-3", launcher.fail.call_args.args[2])
            (work / "restarts-training-worker.json").unlink()
            launcher.restarts = MagicMock(return_value=1)  # A recreated pod counts from zero again.
            self.assertTrue(launcher.check_restarts("run_abc", work.parent, work, "w", "training worker"))

    def test_canceled_run_deletes_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "alvarez-runs/run_abc"
            run_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(json.dumps({"run_id": "run_abc"}))
            cp = MagicMock()
            cp.call.side_effect = lambda method, path, *a, **k: (
                {"run": None} if "claim" in path else [] if path == "/internal/v1/evaluations/h800"
                else {"status": "CANCELED"} if method == "GET" else {})
            launcher = FakeLauncher(root, root, cp)
            launcher.deployments["alvarez-train-abc"] = {"metadata": {"labels": {"alvarez-run": "run_abc"}}}
            launcher.tick()
            self.assertEqual(launcher.deployments, {})
            self.assertTrue((run_dir / "DONE").exists())
            launcher.agent.finish.assert_not_called()

    def test_h800_evaluation_serving_publishes_endpoint_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, serving = Path(directory, "cfs"), Path(directory, "gpu")
            # A checkpoint the launcher produced (CFS original) and one only present on the 4090.
            (root / "alvarez-runs/run_x").mkdir(parents=True)
            (root / "alvarez-runs/run_x/run.json").write_text("{}")
            (root / "alvarez-runs/run_x/DONE").write_text("succeeded\n")
            (root / "alvarez-checkpoints/run_x/final").mkdir(parents=True)
            base = serving / "home/alvarez-models/M"
            base.mkdir(parents=True)
            (base / "config.json").write_text("{}")
            candidate = serving / "home/alvarez-checkpoints/cand"
            candidate.mkdir(parents=True)
            (candidate / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "/home/alvarez-models/M"}))
            uris = {"trained": "file:///home/alvarez-checkpoints/run_x/final",
                    "cand": "file:///home/alvarez-checkpoints/cand", "remote": "s3://bucket/model"}
            active, calls = [{"eval_id": "eval_1", "model_versions": ["trained", "cand"], "model_endpoint": None}], []

            def call(method, path, body=None, *args, **kwargs):
                calls.append((method, path, body))
                if path == "/internal/v1/evaluations/h800":
                    return active
                if path.startswith("/internal/v1/model-versions/"):
                    return {"checkpoint_uri": uris[path.rsplit("/", 1)[1]]}
                return {"run": None} if "claim" in path else {}

            cp = MagicMock()
            cp.call.side_effect = call
            launcher = FakeLauncher(root, serving, cp)
            launcher.healthy = MagicMock(return_value=True)
            published = lambda: [body for method, path, body in calls if path.endswith("/model-endpoint")]
            launcher.tick()
            work = root / "alvarez-serving/eval_1"
            spec = json.loads((work / "spec.json").read_text())
            self.assertEqual(spec["models"], {
                "trained": {"path": str(root / "alvarez-checkpoints/run_x/final"), "backend": "hf"},
                "cand": {"path": str(root / "alvarez-checkpoints/cand"), "backend": "hf"}})
            staged = json.loads((root / "alvarez-checkpoints/cand/adapter_config.json").read_text())
            self.assertEqual(staged["base_model_name_or_path"], str(root / "alvarez-models/M"))
            self.assertEqual((work / "token").stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(launcher.deployments), ["alvarez-eval-1"])
            self.assertEqual(launcher.deployments["alvarez-eval-1"]["metadata"]["labels"]["alvarez-run"], "eval_1")
            self.assertIn("h800_launcher serve-pod", (work / "pod.sh").read_text())
            self.assertEqual(published(), [])  # Not listening yet.

            (work / "ready.json").write_text(json.dumps({"port": 4321}))
            launcher.tick()
            token = (work / "token").read_text()
            self.assertEqual(published(), [{"endpoint": "http://10.0.0.5:4321", "token": token}])
            launcher.healthy.assert_called_with("http://10.0.0.5:4321", token)
            active[0]["model_endpoint"] = "http://10.0.0.5:4321"
            launcher.tick()
            self.assertEqual(len(published()), 1)  # Unchanged endpoint is not republished.
            launcher.healthy.return_value = False
            launcher.tick()
            self.assertEqual(published()[-1], {"endpoint": None, "token": None})
            active[0]["model_endpoint"] = None

            launcher.deployments.clear()  # Deleted externally: recreated.
            launcher.tick()
            self.assertEqual(list(launcher.deployments), ["alvarez-eval-1"])

            active.clear()  # Every trial finished or was canceled.
            launcher.tick()
            self.assertEqual(launcher.deployments, {})
            self.assertTrue((work / "DONE").exists())
            self.assertFalse((work / "token").exists())
            launcher.tick()
            self.assertEqual(launcher.deployments, {})

            # Pod failure and staging failure both cancel the evaluation's trials and remove the pod.
            active.append({"eval_id": "eval_2", "model_versions": ["trained"], "model_endpoint": None})
            launcher.tick()
            (root / "alvarez-serving/eval_2/FAILED").write_text("CUDA error")
            launcher.tick()
            failed = [body for method, path, body in calls if path == "/internal/v1/evaluations/eval_2/serving-failed"]
            self.assertIn("CUDA error", failed[0]["message"])
            self.assertEqual(launcher.deployments, {})
            active[:] = [{"eval_id": "eval_3", "model_versions": ["remote"], "model_endpoint": None}]
            launcher.tick()
            failed = [body for method, path, body in calls if path == "/internal/v1/evaluations/eval_3/serving-failed"]
            self.assertIn("staging failed", failed[0]["message"])
            self.assertEqual(launcher.deployments, {})
            self.assertTrue((root / "alvarez-serving/eval_3/DONE").exists())
            # Model deletion removes the staged copy (it carries the sync marker), not unrelated data.
            self.assertEqual(cfs_staged_copy(uris["cand"], root), root / "alvarez-checkpoints/cand")
            self.assertIsNone(cfs_staged_copy(uris["trained"], root))
            self.assertIsNone(cfs_staged_copy(uris["remote"], root))
            self.assertIsNone(cfs_staged_copy("file:///etc/passwd", root))

    def test_console_limits_from_heartbeat_reply_gate_new_pods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "cfs")
            (root / "alvarez-runs/run_x").mkdir(parents=True)
            (root / "alvarez-runs/run_x/run.json").write_text("{}")
            paths = []

            def call(method, path, body=None, *args, **kwargs):
                paths.append(path)
                if path.endswith("/heartbeat"):
                    self.assertEqual((body["capacity"]["max_training_runs"], body["capacity"]["max_evaluations"]), (4, 2))
                    return {"capacity": {"training_runs": 1, "evaluations": 1}}
                return [] if path == "/internal/v1/evaluations/h800" else {"run": None}

            cp = MagicMock()
            cp.call.side_effect = call
            launcher = FakeLauncher(root, Path(directory, "gpu"), cp)
            launcher.advance = MagicMock()
            launcher.tick()
            self.assertTrue(any("claim" in path for path in paths))  # 1 active < ceiling 4.
            self.assertEqual((launcher.run_limit, launcher.serving_limit), (1, 1))
            paths.clear()
            launcher.tick()
            self.assertFalse(any("claim" in path for path in paths))  # 1 active, console limit 1.


if __name__ == "__main__":
    unittest.main()
