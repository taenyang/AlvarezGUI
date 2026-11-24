import base64
import functools
import importlib.metadata
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from alvarezgui.h800_launcher import H800_IMAGE, deployment_manifest, sglang_servable
from alvarezgui.model import InferenceRequest, InvalidActionOutputError
from alvarezgui.sglang_adapter import SglangService, topk_entropy



class FakeSglang(BaseHTTPRequestHandler):
    reply = ""
    bodies: list = []

    def log_message(self, *_):
        return

    def do_POST(self):
        FakeSglang.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        top = [{"token": "a", "logprob": -0.6931471805599453}, {"token": "b", "logprob": -0.6931471805599453}]
        payload = json.dumps({"choices": [{"message": {"content": FakeSglang.reply},
                                           "logprobs": {"content": [{"top_logprobs": top}]}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class SglangAdapterTest(unittest.TestCase):
    def test_patch_installs_only_on_probed_version(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/alvarezgui/sglang_patch"))
        try:
            for version, installed in (("0.5.24", False), ("0.5.23", True)):
                sys.modules.pop("sitecustomize", None)
                before = len(sys.meta_path)
                with patch.object(importlib.metadata, "version", return_value=version):
                    import sitecustomize  # noqa: F401
                self.assertEqual(len(sys.meta_path) - before, int(installed))
                if installed:
                    sys.meta_path.pop(0)
        finally:
            sys.path.pop(0)
            sys.modules.pop("sitecustomize", None)

    def test_topk_entropy(self):
        half = {"logprob": -0.6931471805599453}
        self.assertAlmostEqual(topk_entropy([[half, half], [{"logprob": 0.0}]]), 0.6931471805599453 / 2)
        self.assertIsNone(topk_entropy([]))

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "needs Pillow (run on the 4090 model-env)")
    def test_infer_converts_request_and_rejects_invalid_output(self):
        from PIL import Image

        buffer = BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, format="PNG")
        png = base64.b64encode(buffer.getvalue()).decode()
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSglang)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "policy_config.json").write_text(json.dumps({"visual_views": "full+board-crop"}))
                service = SglangService("v", directory, f"http://127.0.0.1:{server.server_address[1]}")
                request = InferenceRequest(request_id="r", model_version="v", instruction="solve",
                                           image_uri="data:image/png;base64," + png, temperature=1.0, seed=7)
                FakeSglang.reply = '{"type": "swipe", "x1": 1, "y1": 2, "x2": 3, "y2": 4, "duration_ms": 200}'
                with patch("alvarezgui.minicpm_training._board_crop_image", side_effect=lambda image: image):
                    response = service.infer(request)
                body = FakeSglang.bodies[-1]
                self.assertEqual(response.action["type"], "swipe")
                self.assertAlmostEqual(response.entropy, 0.6931471805599453)
                self.assertEqual([part["type"] for part in body["messages"][0]["content"]],
                                 ["image_url", "image_url", "text"])
                self.assertEqual((body["temperature"], body["seed"], body["max_tokens"]), (1.0, 7, 128))
                self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
                FakeSglang.reply = "up"
                with patch("alvarezgui.minicpm_training._board_crop_image", side_effect=lambda image: image), \
                        self.assertRaises(InvalidActionOutputError) as error:
                    service.infer(request)
                self.assertEqual((error.exception.raw_output, error.exception.output_format), ("up", "action-v1"))
        finally:
            server.shutdown()

    def test_servable_and_manifest_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertFalse(sglang_servable(path))
            (path / "config.json").write_text(json.dumps({"model_type": "minicpmv4_7"}))
            self.assertTrue(sglang_servable(path))
            (path / "adapter_config.json").write_text("{}")
            self.assertFalse(sglang_servable(path))
        template = {"metadata": {"namespace": "ns"},
                    "spec": {"template": {"metadata": {}, "spec": {"containers": [{"image": "nemo"}]}}}}
        manifest = deployment_manifest(template, "n", "r", Path("/p"))
        self.assertEqual(manifest["spec"]["template"]["spec"]["containers"][0]["image"], H800_IMAGE)



class RolloutBackendTest(unittest.TestCase):
    def test_minicpm_policy_gets_sglang_pod_and_merged_weights(self):
        from unittest.mock import MagicMock

        from tests.test_h800_launcher import FakeLauncher

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "cfs")
            policy = root / "alvarez-checkpoints/run_abc/round-0"
            merged = Path(str(policy) + "-merged")
            for path in (policy, merged):
                path.mkdir(parents=True)
            (policy / "adapter_config.json").write_text("{}")
            (merged / "config.json").write_text(json.dumps({"model_type": "minicpmv4_7"}))
            launcher = FakeLauncher(root, Path(directory, "gpu"), MagicMock())
            work = root / "alvarez-runs/run_abc/round-1"
            state = {"policy_version": "abc-r0", "policy_path": str(policy)}
            launcher.rollout_service_ready({"run_id": "run_abc"}, state, work)
            spec = json.loads((work / "serve/spec.json").read_text())
            self.assertEqual((spec["backend"], spec["models"]), ("sglang", {"abc-r0": str(merged)}))
            self.assertIn("sglang-adapter", (work / "serve/pod.sh").read_text())
            container = launcher.deployments["alvarez-roll-abc-1"]["spec"]["template"]["spec"]["containers"][0]
            self.assertEqual(container["image"], H800_IMAGE)
            launcher.stop_rollout_service("run_abc", work)
            self.assertFalse(merged.exists())  # Only rollout input; the adapter checkpoint stays.
            self.assertTrue((policy / "adapter_config.json").exists())


class MergeForRolloutTest(unittest.TestCase):
    def test_lora_round_is_merged_beside_its_checkpoint(self):
        import types

        from alvarezgui.h800_launcher import run_pod

        with tempfile.TemporaryDirectory() as directory:
            run_dir, output = Path(directory, "round-0"), Path(directory, "ckpt/round-0")
            run_dir.mkdir()
            spec = {"run": {"run_id": "r", "algorithm": "grpo", "config": {}}, "base_model": "b", "dataset": "d",
                    "output_dir": str(output), "merge_for_rollout": True}
            (run_dir / "spec.json").write_text(json.dumps(spec))

            def train(*_, **__):
                output.mkdir(parents=True)
                (output / "adapter_config.json").write_text("{}")
                return {"step_count": 1}

            merged = []
            training = types.SimpleNamespace(load_dataset=lambda _: [], train=train)
            with patch.dict(sys.modules, {"alvarezgui.minicpm_training": training}), \
                    patch("alvarezgui.minicpm_service.merge_adapter", lambda src, dst: merged.append((src, dst))), \
                    patch("alvarezgui.h800_launcher.train_options", lambda config: {}):
                run_pod(run_dir)
            self.assertEqual(merged, [(output, Path(str(output) + "-merged"))])
            self.assertTrue((run_dir / "COMPLETE").exists())


class StartRetryTest(unittest.TestCase):
    def test_exit_before_ready_retries_on_a_new_port(self):
        from alvarezgui import sglang_adapter

        class Process:
            def __init__(self, returncode):
                self.returncode, self.pid = returncode, 0

            def poll(self):
                return self.returncode

        launched = []
        processes = iter([Process(1), Process(None)])

        def popen(args, **_):
            launched.append(args[args.index("--port") + 1])
            return next(processes)

        def wait_ready(url, process, **_):
            if process.poll() is not None:
                raise RuntimeError("exited")

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sglang_adapter.subprocess, "Popen", popen), \
                patch.object(sglang_adapter, "_wait_ready", wait_ready), \
                patch.object(sglang_adapter, "_free_port", side_effect=[1111, 2222]):
            current = []
            url = sglang_adapter._start("/model", Path(directory, "sglang.log"), current)
        self.assertEqual((launched, url, current[0].returncode), (["1111", "2222"], "http://127.0.0.1:2222", None))

    def test_live_server_that_lost_its_port_is_killed_and_retried(self):
        """eval_423508c1: uvicorn fails to bind but the scheduler keeps the process (and GPU memory) alive."""
        from alvarezgui import sglang_adapter

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory, "sglang.log")
            log_path.write_bytes(b"earlier attempt: address already in use\n")  # Not this process's output.
            launched, killed = [], []

            class Process:
                def __init__(self, port):
                    self.port, self.pid, self.returncode = port, 4242, None

                def poll(self):
                    return self.returncode

                def wait(self, timeout=None):
                    self.returncode = -15
                    return self.returncode

            def popen(args, stdout, **_):
                port = args[args.index("--port") + 1]
                launched.append(port)
                if port == "1111":  # This attempt's server loses the race for its port.
                    stdout.write(b"ERROR: [Errno 98] error while attempting to bind on address "
                                 b"('127.0.0.1', 1111): address already in use\n")
                    stdout.flush()
                return Process(port)

            answered = []

            def urlopen(url, timeout):
                if ":2222/" not in url:
                    raise OSError("refused")
                answered.append(url)
                return type("R", (), {"status": 200, "__enter__": lambda self: self, "__exit__": lambda self, *a: None})()

            # A regression (bind failure not noticed) fails within 5 s instead of waiting out the 900 s default.
            wait_ready = functools.partial(sglang_adapter._wait_ready, timeout=5)
            with patch.object(sglang_adapter.subprocess, "Popen", popen), \
                    patch.object(sglang_adapter, "_wait_ready", wait_ready), \
                    patch.object(sglang_adapter, "urlopen", urlopen), \
                    patch.object(sglang_adapter, "_free_port", side_effect=[1111, 2222]), \
                    patch.object(sglang_adapter.os, "killpg", lambda pid, sig: killed.append((pid, sig))), \
                    patch.object(sglang_adapter.time, "sleep", lambda _: None):
                current = []
                url = sglang_adapter._start("/model", log_path, current)
            self.assertEqual((launched, url), (["1111", "2222"], "http://127.0.0.1:2222"))
            self.assertEqual(killed, [(4242, sglang_adapter.signal.SIGTERM)])  # Group killed, not left holding GPU memory.
            self.assertEqual([process.port for process in current], ["2222"])

    def test_run_serves_adapter_once_sglang_is_ready(self):
        from alvarezgui import sglang_adapter

        class Process:
            pid = 0

            def wait(self):
                threading.Event().wait()  # Never exits during the test.

        def start(path, log_path, current, memory, extra):
            current.append(Process())
            return "http://127.0.0.1:2222"

        served = []
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "spec.json").write_text(json.dumps({"models": {"v": str(work)}}))
            (work / "token").write_text("secret")
            with patch.object(sglang_adapter, "_start", start), \
                    patch.object(sglang_adapter, "serve", lambda service, *args, **kwargs: served.append((service, kwargs))), \
                    patch.object(sglang_adapter.signal, "signal"):
                sglang_adapter.run(work)
        service, kwargs = served[0]
        self.assertEqual((service.url, service.version, kwargs["token"]), ("http://127.0.0.1:2222", "v", "secret"))

    def test_stopping_round_zero_keeps_a_base_model_named_merged(self):
        from unittest.mock import MagicMock

        from tests.test_h800_launcher import FakeLauncher

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "cfs")
            base = root / "alvarez-checkpoints/merged/sokoban-sd1-merged"  # Same shape as the real base model.
            base.mkdir(parents=True)
            (base / "config.json").write_text(json.dumps({"model_type": "minicpmv4_7"}))
            launcher = FakeLauncher(root, Path(directory, "gpu"), MagicMock())
            work = root / "alvarez-runs/run_abc/round-0"
            launcher.rollout_service_ready({"run_id": "run_abc"},
                                           {"policy_version": "sokoban-sd1-merged", "policy_path": str(base)}, work)
            self.assertEqual(json.loads((work / "serve/spec.json").read_text())["backend"], "sglang")
            launcher.stop_rollout_service("run_abc", work)
            self.assertTrue((base / "config.json").exists())


class EvaluationServingTest(unittest.TestCase):
    def test_backend_per_model_merged_copy_and_cleanup(self):
        from unittest.mock import MagicMock

        from tests.test_h800_launcher import FakeLauncher

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "cfs")
            full, lora, other = (root / "alvarez-checkpoints" / name for name in ("full", "lora", "qwen"))
            for path in (full, lora, other):
                path.mkdir(parents=True)
            (full / "config.json").write_text(json.dumps({"model_type": "minicpmv4_7"}))
            (lora / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(full)}))
            (other / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
            launcher = FakeLauncher(root, Path(directory, "gpu"), MagicMock())
            launcher.serving_model = lambda version: root / "alvarez-checkpoints" / version
            launcher.start_serving({"eval_id": "eval_1", "model_versions": ["full", "lora", "qwen"]})
            work = root / "alvarez-serving/eval_1"
            # One pod per model, each with its own spec, script, token and deployment.
            specs = [json.loads((work / f"models/{index}/spec.json").read_text()) for index in range(3)]
            self.assertEqual([spec["model_version"] for spec in specs], ["full", "lora", "qwen"])
            self.assertTrue(all(spec["sglang_args"] == ["--enable-deterministic-inference"] for spec in specs))
            models = {spec["model_version"]: spec["models"][spec["model_version"]] for spec in specs}
            self.assertEqual({version: entry["backend"] for version, entry in models.items()},
                             {"full": "sglang", "lora": "sglang", "qwen": "hf"})
            self.assertEqual(models["lora"]["merged_copy"], str(work / "merged/lora"))
            self.assertIn("sglang-adapter", (work / "models/0/pod.sh").read_text())
            self.assertIn("serve-pod", (work / "models/2/pod.sh").read_text())  # HF model, HF pod.
            self.assertEqual(sorted(launcher.deployments),
                             ["alvarez-eval-1-m0", "alvarez-eval-1-m1", "alvarez-eval-1-m2"])
            tokens = {(work / f"models/{index}/token").read_text() for index in range(3)}
            self.assertEqual(len(tokens), 3)
            (work / "merged/lora").mkdir(parents=True)
            launcher.stop_serving("eval_1", work, "done")
            self.assertFalse((work / "merged/lora").exists())
            self.assertTrue((full / "config.json").exists() and (lora / "adapter_config.json").exists())

    def test_run_routes_by_version_and_checks_memory(self):
        from alvarezgui import sglang_adapter

        class Process:
            pid = 0

            def wait(self):
                threading.Event().wait()

        class FakeHF:
            def __init__(self, *_, **__):
                self.loaded = []

            def load(self, version):
                self.loaded.append(version)

        starts, served = [], []

        def start(path, log_path, current, memory, extra):
            starts.append((path, memory))
            current.append(Process())
            return f"http://127.0.0.1:{len(starts)}"

        def merge(src, dst):
            dst.mkdir(parents=True)

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "token").write_text("secret")
            spec = {"models": {"a": {"path": str(work), "backend": "sglang"},
                               "b": {"path": "/adapter", "backend": "sglang", "merged_copy": str(work / "merged/b")},
                               "h": {"path": "/hf", "backend": "hf"}}, "sglang_memory": 0.8}
            (work / "spec.json").write_text(json.dumps(spec))
            with patch.object(sglang_adapter, "_start", start), \
                    patch.object(sglang_adapter, "MiniCPMService", FakeHF), \
                    patch("alvarezgui.minicpm_service.merge_adapter", merge), \
                    patch.object(sglang_adapter, "serve", lambda service, *a, **k: served.append(service)), \
                    patch.object(sglang_adapter.signal, "signal"):
                sglang_adapter.run(work)
                dispatcher = served[0]
                self.assertEqual([path for path, _ in starts], [str(work), str(work / "merged/b")])
                self.assertAlmostEqual(starts[0][1], (0.8 - sglang_adapter.HF_MEMORY) / 2)
                self.assertEqual(dispatcher.services["h"].loaded, ["h"])
                self.assertEqual(dispatcher.services["b"].url, "http://127.0.0.1:2")
                with self.assertRaises(sglang_adapter.ModelVersionNotFoundError):
                    dispatcher.infer(InferenceRequest(request_id="r", model_version="x", instruction="i",
                                                      image_uri="data:,"))

                # Too little memory per sglang server: FAILED, nothing started.
                spec["models"] = {str(i): {"path": "/m", "backend": "sglang"} for i in range(5)}
                spec["models"]["h"] = {"path": "/hf", "backend": "hf"}
                spec["sglang_memory"] = 0.5
                (work / "spec.json").write_text(json.dumps(spec))
                starts.clear()
                with self.assertRaises(SystemExit):
                    sglang_adapter.run(work)
                self.assertEqual(starts, [])
                self.assertIn("minimum", (work / "FAILED").read_text())


if __name__ == "__main__":
    unittest.main()


class FakeSwapSglang(BaseHTTPRequestHandler):
    loaded: list = []
    fail = False

    def log_message(self, *_):
        return

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ok = not FakeSwapSglang.fail
        if ok:
            FakeSwapSglang.loaded.append(body["model_path"])
        payload = json.dumps({"success": ok, "message": "ok" if ok else "bad weights"}).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ResidentSwapTest(unittest.TestCase):
    def test_swap_switches_version_only_after_sglang_loaded_it(self):
        from alvarezgui.sglang_adapter import ResidentService, SwapPaused

        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSwapSglang)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        FakeSwapSglang.loaded, FakeSwapSglang.fail = [], False
        try:
            with tempfile.TemporaryDirectory() as directory:
                url = f"http://127.0.0.1:{server.server_address[1]}"
                resident = ResidentService(SglangService("r0", directory, url))
                request = lambda version: InferenceRequest(
                    request_id="q", model_version=version, image_uri="file:///x.png", instruction="go")
                calls = []
                resident.service.infer = lambda r: calls.append(r.model_version)
                resident.infer(request("r0"))
                resident.swapping = True
                with self.assertRaises(SwapPaused):  # 503 to the consumer, which retries.
                    resident.infer(request("r0"))
                resident.swapping = False
                resident.swap("r1", directory)
                self.assertEqual((FakeSwapSglang.loaded, resident.service.version), ([directory], "r1"))
                self.assertFalse(resident.health()["swapping"])
                FakeSwapSglang.fail = True
                with self.assertRaisesRegex(RuntimeError, "bad weights"):
                    resident.swap("r2", directory)
                self.assertEqual(resident.service.version, "r1")  # Never claims weights it did not load.
                self.assertFalse(resident.swapping)
                self.assertEqual(calls, ["r0"])
        finally:
            server.shutdown()

    def test_restarted_resident_pod_resumes_the_served_round(self):
        from alvarezgui import sglang_adapter

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "spec.json").write_text(json.dumps({"resident": True, "models": {"base": "/m/base"}}))
            (work / "current.json").write_text(json.dumps({"version": "r3", "path": "/m/r3-merged"}))
            (work / "token").write_text("t")
            started = []

            def start(path, *_args, **_kwargs):
                started.append(path)
                raise RuntimeError("stop here")

            with patch.object(sglang_adapter, "_start", start), \
                    patch.object(sglang_adapter.sys, "exit", side_effect=SystemExit):
                with self.assertRaises(SystemExit):
                    sglang_adapter.run(work)
            self.assertEqual(started, ["/m/r3-merged"])


class FreePortTest(unittest.TestCase):
    def test_random_port_outside_reserved_and_below_sglang_derived_range(self):
        """H800 nodes: bind(0) kept returning 18001-18015 (eval_423508c1); reserved ports must be skipped."""
        from alvarezgui import sglang_adapter

        reserved = "2375,6676-18000,21270-21271,23456,29500,30000-32768,34595,35000-61000"
        with patch.object(Path, "read_text", return_value=reserved):
            ports = {sglang_adapter._free_port() for _ in range(100)}
            self.assertTrue({21270, 23456, 29500}.issubset(sglang_adapter._reserved_ports()))
        self.assertGreater(len(ports), 50)  # Spread out, not the same few ports every time.
        self.assertTrue(all(20000 <= port <= 29000 and port not in (21270, 21271, 23456, 29500) for port in ports))
        self.assertTrue(all(port + 233 + 6 < 30000 for port in ports))
