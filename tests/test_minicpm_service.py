import itertools
import json
import math
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from alvarezgui.consumer import _model_image_uri
from alvarezgui.minicpm_service import MiniCPMService, _parse_action
from alvarezgui.model import InvalidActionOutputError
from alvarezgui.sokoban import ACTIONS


class MiniCPMServiceTest(unittest.TestCase):
    def test_adapter_merge_requires_explicit_validated_checkpoint_path(self) -> None:
        from alvarezgui.minicpm_service import load_pretrained

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "adapter_config.json").write_text("{}")
            for enabled in (False, True):
                with self.subTest(enabled=enabled):
                    adapter, merged, processor, torch = Mock(), Mock(), Mock(), Mock()
                    adapter.merge_and_unload.return_value = merged
                    peft = SimpleNamespace(
                        PeftConfig=Mock(), PeftModel=Mock(),
                    )
                    peft.PeftConfig.from_pretrained.return_value.base_model_name_or_path = directory
                    peft.PeftModel.from_pretrained.return_value = adapter
                    transformers = SimpleNamespace(AutoModelForImageTextToText=Mock(), AutoProcessor=Mock(),
                                                   AutoConfig=Mock())
                    transformers.AutoConfig.from_pretrained.return_value = SimpleNamespace(model_type="minicpmv4_7")
                    transformers.AutoProcessor.from_pretrained.return_value = processor
                    allowed = str(checkpoint.resolve()) if enabled else str(checkpoint / "other")
                    with patch.dict("sys.modules", {"torch": torch, "transformers": transformers, "peft": peft}), patch.dict(
                        "os.environ", {"ALVAREZ_MERGE_ADAPTER_PATHS": allowed}
                    ):
                        model, loaded_processor, loaded_torch = load_pretrained(directory)
                    self.assertIs(model, merged if enabled else adapter)
                    self.assertIs(loaded_processor, processor)
                    self.assertIs(loaded_torch, torch)
                    model.eval.assert_called_once_with()
                    if enabled:
                        adapter.merge_and_unload.assert_called_once_with(safe_merge=True)
                    else:
                        adapter.merge_and_unload.assert_not_called()
                    self.assertEqual((checkpoint / "adapter_config.json").read_text(), "{}")

    def test_merge_writes_full_checkpoint_with_adapter_metadata_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, base = Path(directory, "checkpoints"), Path(directory, "base")
            adapter, full = root / "run" / "final", root / "full"
            for path in (adapter, full, base):
                path.mkdir(parents=True)
            for name in ("adapter_config.json", "adapter_model.safetensors", "README.md", "metrics.json",
                         "tokenizer.json", "chat_template.jinja"):
                (adapter / name).write_text(name)
            (adapter / "policy_config.json").write_text('{"visual_views": "full+board-crop", "finetuning_method": "lora"}')
            service = MiniCPMService(lambda version: str({"adapter-v1": adapter, "full-v1": full}[version]),
                                     checkpoint_root=root)
            for version, target, message in (("adapter-v1", "../x", "invalid target"),
                                             ("full-v1", "x", "not a LoRA adapter")):
                with self.subTest(target), self.assertRaisesRegex(ValueError, message):
                    service.merge(version, target)

            merged = Mock()
            merged.save_pretrained.side_effect = lambda path, **_: Path(path, "model.safetensors").write_text("full")
            peft = SimpleNamespace(PeftConfig=Mock(), PeftModel=Mock())
            peft.PeftConfig.from_pretrained.return_value.base_model_name_or_path = str(base)
            peft.PeftModel.from_pretrained.return_value.merge_and_unload.return_value = merged
            transformers = SimpleNamespace(AutoModelForImageTextToText=Mock(), AutoConfig=Mock())
            transformers.AutoConfig.from_pretrained.return_value = SimpleNamespace(model_type="minicpmv4_7")
            with patch.dict("sys.modules", {"peft": peft, "transformers": transformers}):
                output = Path(service.merge("adapter-v1", "adapter-v1-merged"))
                with self.assertRaisesRegex(ValueError, "already exists"):
                    service.merge("adapter-v1", "adapter-v1-merged")
            self.assertEqual(output, root / "merged" / "adapter-v1-merged")
            peft.PeftModel.from_pretrained.return_value.merge_and_unload.assert_called_once_with(safe_merge=True)
            self.assertEqual(sorted(path.name for path in output.iterdir()),
                             ["chat_template.jinja", "model.safetensors", "policy_config.json", "tokenizer.json"])
            self.assertEqual(json.loads((output / "policy_config.json").read_text()),
                             {"visual_views": "full+board-crop", "finetuning_method": "full", "merged_from": str(adapter)})
            self.assertFalse(output.with_name(output.name + ".tmp").exists())
            self.assertEqual(json.loads((adapter / "policy_config.json").read_text())["finetuning_method"], "lora")

    def test_token_entropy_matches_distribution_and_sampling_filters(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")
        from alvarezgui.minicpm_service import token_entropies

        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, -float("inf"), -float("inf"), 10.0],
                               [100.0, 0.0, 0.0, 0.0]])
        entropy = token_entropies(torch, logits)
        # Uniform over 4 -> ln 4; top-k/top-p filtered to 2 equal tokens -> ln 2; near-greedy -> ~0.
        torch.testing.assert_close(entropy[:2], torch.tensor([math.log(4), math.log(2)]))
        self.assertLess(entropy[2].item(), 1e-6)
        self.assertTrue(torch.isfinite(entropy).all())

        request_kwargs = {"request_id": "r", "model_version": "v", "instruction": "i", "image_uri": "data:,",
                          "max_tokens": 8, "temperature": 1.0}
        from alvarezgui.model import InferenceRequest

        service = MiniCPMService(lambda _: "unused")
        service.torch = torch

        class Inputs(dict):
            input_ids = [[1, 2]]

            def to(self, _device):
                return self

        processor = Mock()
        processor.apply_chat_template.return_value = Inputs()
        processor.batch_decode.return_value = ['{"type":"back"}']
        model = Mock(config=SimpleNamespace(model_type="qwen3_vl"))
        model.generate.return_value = SimpleNamespace(sequences=[[1, 2, 3, 4]], scores=(logits[:1], logits[1:2]))
        text, mean = service._generate(model, processor, [], InferenceRequest(**request_kwargs))
        self.assertEqual(text, '{"type":"back"}')
        self.assertAlmostEqual(mean, (math.log(4) + math.log(2)) / 2, places=5)

    def test_local_file_image_becomes_data_uri_for_remote_service(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "screen.png"
            image.write_bytes(b"png-bytes")
            uri = _model_image_uri(image.as_uri())
        self.assertEqual(uri, "data:image/png;base64,cG5nLWJ5dGVz")

    def test_parse_action_from_plain_or_explanatory_output(self) -> None:
        self.assertEqual(
            _parse_action('{"type":"tap","x":10,"y":20}'),
            {"type": "tap", "x": 10, "y": 20},
        )
        self.assertEqual(
            _parse_action('Next action: {"type":"back"}'),
            {"type": "back"},
        )

    def test_parse_action_rejects_non_action_json(self) -> None:
        with self.assertRaises(InvalidActionOutputError) as caught:
            _parse_action('{"command":"shell","value":"id"}')
        self.assertIn('raw_output_preview=\'{"command":"shell","value":"id"}\'', str(caught.exception))
        with self.assertRaises(InvalidActionOutputError):
            _parse_action('{"type":"back"} trailing output')

    def test_parse_action_rejects_coordinates_outside_the_model_screen(self) -> None:
        # Seen in run_f630a6de round 8: accepted by serving, then rejected by the trainer's dataset check.
        for raw in ('{"type":"tap","x":836,"y":360}', '{"type":"tap","x":10,"y":1280}',
                    '{"type":"swipe","x1":0,"y1":0,"x2":720,"y2":0}'):
            with self.subTest(raw=raw), self.assertRaises(InvalidActionOutputError):
                _parse_action(raw)
        self.assertEqual(_parse_action('{"type":"tap","x":719,"y":1279}'), {"type": "tap", "x": 719, "y": 1279})

    def test_h800_trial_waits_for_a_replacement_model_service(self) -> None:
        from alvarezgui.consumer import Consumer
        from alvarezgui.model import ModelUnavailableError

        consumer = Consumer.__new__(Consumer)
        consumer.lease_ttl = None
        consumer.runtime = Mock()
        consumer.runtime.execute.return_value = {"reward": {}, "done": True}
        context = {"status": "RUNNING_ENV", "accelerator": "h800", "current_step": 0, "max_steps": 10}
        consumer._store_call = lambda method, *args: context if method == "get_execution_context" else True
        consumer._next_action = Mock(side_effect=[ModelUnavailableError("gone"), ModelUnavailableError("gone"),
                                                  ({"type": "terminate"}, None)])
        consumer._final_observation = consumer._upload_observation = consumer._destroy_if_lease_owner = Mock()
        with patch("alvarezgui.consumer.time.sleep") as sleep:
            consumer._run_episode("t", "e", "s", "l")
        self.assertEqual((sleep.call_count, consumer._next_action.call_count), (2, 3))

        consumer._next_action = Mock(side_effect=ModelUnavailableError("gone"))
        with patch("alvarezgui.consumer.time.sleep"), \
                patch("alvarezgui.consumer.time.monotonic", side_effect=itertools.count(0, 500)), \
                self.assertRaisesRegex(Exception, "unavailable for 900s"):
            consumer._run_episode("t", "e", "s", "l")
        context["accelerator"] = "4090"
        with self.assertRaises(ModelUnavailableError):
            consumer._run_episode("t", "e", "s", "l")

    def test_replayed_invalid_outputs_remain_rejected_with_evidence(self) -> None:
        for raw in (
            '{\n"type":"tap","x":544, "y":980}\n}',
            '{\n"type":"swipe","x1":176, "y1":980, "x2":544, "y2":980, "duration_ms":0}',
        ):
            with self.subTest(raw=raw), self.assertRaises(InvalidActionOutputError) as caught:
                _parse_action(raw)
            self.assertIn(repr(raw), str(caught.exception))

    def test_checkpoint_protocol_controls_prompt_budget_and_strict_parsing(self) -> None:
        from alvarezgui.model import InferenceRequest
        from alvarezgui.policy_output import make_plan

        board = ["########", "#      #", "# @$.  #", "#      #", "#      #", "########"]
        plan = make_plan(board, "right")
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "policy_config.json").write_text(json.dumps({
                "output_format": "sokoban-plan-v1", "max_tokens": 512,
            }))
            service = MiniCPMService(lambda _: directory, loader=lambda _: (object(), object(), None))
            request = InferenceRequest(request_id="r", model_version="plan", instruction="solve", image_uri="data:,")
            with patch("alvarezgui.minicpm_service._read_image", return_value=object()), patch.object(
                service, "_generate", return_value=(json.dumps(plan), 0.25)
            ) as generate:
                response = service.infer(request)
                self.assertEqual(response.action, plan["action"])
                self.assertEqual(response.output_format, "sokoban-plan-v1")
                self.assertEqual(response.entropy, 0.25)
                self.assertEqual(generate.call_args.args[3].max_tokens, 512)
                self.assertIn("Output format: sokoban-plan-v1", generate.call_args.args[2][0]["content"][1]["text"])
                self.assertIsNone(request.max_tokens)
                service.infer(request.model_copy(update={"max_tokens": 64}))
                self.assertEqual(generate.call_args.args[3].max_tokens, 64)
                # A valid nested action must not rescue an invalid structured response.
                generate.return_value = (json.dumps({"action": plan["action"]}), 0.25)
                with self.assertRaises(InvalidActionOutputError) as error:
                    service.infer(request)
                self.assertEqual(error.exception.raw_output, generate.return_value[0])
                self.assertEqual(error.exception.output_format, "sokoban-plan-v1")
            service.unload("plan")
            self.assertNotIn("plan", service.policy_configs)

    def test_multiview_checkpoint_supplies_training_views_in_order(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        from alvarezgui.minicpm_training import _board_crop_image
        from alvarezgui.model import InferenceRequest

        image = Image.new("RGB", (720, 1280), "black")
        image.paste("red", (180, 270, 360, 650))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "policy_config.json").write_text(json.dumps({
                "visual_views": "full+board-crop",
            }))
            service = MiniCPMService(lambda _: directory, loader=lambda _: (object(), object(), None))
            with patch("alvarezgui.minicpm_service._read_image", return_value=image), patch.object(
                service, "_generate", return_value=(json.dumps(ACTIONS["left"]), None)
            ) as generate:
                service.infer(InferenceRequest(
                    request_id="r", model_version="multiview", instruction="solve", image_uri="data:,"
                ))
            content = generate.call_args.args[2][0]["content"]
            self.assertEqual([part["type"] for part in content], ["image", "image", "text"])
            self.assertIs(content[0]["image"], image)
            self.assertEqual(content[1]["image"].tobytes(), _board_crop_image(image).tobytes())

    def test_perception_direction_serializes_fixed_sokoban_action(self) -> None:
        from alvarezgui.model import InferenceRequest

        class FakeHeads:
            def __contains__(self, name):
                return name == "direction"

            def __getitem__(self, name):
                return Mock()

            def to(self, _device):
                return self

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "policy_config.json").write_text(json.dumps({
                "output_format": "action-v1",
                "max_tokens": 1,
                "inference_mode": "perception_direction",
            }))
            (checkpoint / "perception_config.json").write_text(json.dumps({
                "direction": True,
                "visual_views": "task-routed",
                "head_views": {"direction": "board-crop"},
            }))
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            model = SimpleNamespace(device="cpu")
            service = MiniCPMService(
                lambda _: str(checkpoint),
                loader=lambda _: (model, object(), torch),
            )
            request = InferenceRequest(
                request_id="r", model_version="perception", instruction="solve", image_uri="data:,"
            )
            with patch.object(service, "_load_perception_heads", return_value=FakeHeads()), patch(
                "alvarezgui.minicpm_service._read_image", return_value=object()
            ), patch.object(service, "_predict_direction", return_value="left") as predict, patch.object(
                service, "_generate", side_effect=AssertionError("decoder must not run")
            ):
                response = service.infer(request)
            self.assertEqual(response.action, ACTIONS["left"])
            self.assertEqual(response.raw_output, '{"type":"tap","x":176,"y":980}')
            self.assertEqual(response.output_format, "action-v1")
            predict.assert_called_once()
            service.unload("perception")
            self.assertNotIn("perception", service.perception_heads)

    def test_http_rejection_preserves_structured_output_evidence(self) -> None:
        from alvarezgui.model import InferenceRequest
        from alvarezgui.model_http import HttpModelService

        body = {"detail": "invalid plan", "raw_output": '{"board":', "output_format": "sokoban-plan-v1"}
        failure = HTTPError("http://model", 422, "invalid", {}, BytesIO(json.dumps(body).encode()))
        with patch("alvarezgui.model_http.urlopen", side_effect=failure):
            with self.assertRaises(InvalidActionOutputError) as error:
                HttpModelService("http://model").infer(InferenceRequest(
                    request_id="r", model_version="v", instruction="solve", image_uri="data:,"))
        self.assertEqual(error.exception.raw_output, body["raw_output"])
        self.assertEqual(error.exception.output_format, body["output_format"])

    def test_token_guards_every_request_and_ready_file_reports_bound_port(self) -> None:
        import threading
        import time
        from urllib.request import Request, urlopen

        from alvarezgui.minicpm_service import _ModelServer, serve

        service = Mock()
        service.health.return_value = {"loaded_versions": ["v"]}
        servers = []
        real_init = _ModelServer.__init__

        def remember(self, *args, **kwargs):
            real_init(self, *args, **kwargs)
            servers.append(self)

        with tempfile.TemporaryDirectory() as directory, patch.object(_ModelServer, "__init__", remember):
            ready = Path(directory, "ready.json")
            thread = threading.Thread(target=serve, args=(service, "127.0.0.1", 0),
                                      kwargs={"token": "secret-token", "ready_file": ready}, daemon=True)
            thread.start()
            for _ in range(100):
                if ready.exists():
                    break
                time.sleep(0.05)
            port = json.loads(ready.read_text())["port"]
            self.assertGreater(port, 0)
            base = f"http://127.0.0.1:{port}"
            try:
                for request in (Request(f"{base}/health"),
                                Request(f"{base}/health", headers={"Authorization": "Bearer wrong"}),
                                Request(f"{base}/internal/v1/model/infer", data=b"{}", method="POST")):
                    with self.subTest(url=request.full_url), self.assertRaises(HTTPError) as rejected:
                        urlopen(request, timeout=5)
                    self.assertEqual(rejected.exception.code, 401)
                service.infer.assert_not_called()
                with urlopen(Request(f"{base}/health", headers={"Authorization": "Bearer secret-token"}),
                             timeout=5) as response:
                    self.assertEqual(json.loads(response.read()), {"loaded_versions": ["v"]})
            finally:
                servers[0].shutdown()
                thread.join(timeout=5)

    def test_http_client_sends_bearer_token_only_when_configured(self) -> None:
        from alvarezgui.model import InferenceRequest
        from alvarezgui.model_http import HttpModelService

        reply = Mock()
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        reply.read.return_value = json.dumps({"request_id": "r", "action": {"type": "back"}, "raw_output": "x",
                                              "model_version": "v", "latency_ms": 1}).encode()
        request = InferenceRequest(request_id="r", model_version="v", instruction="i", image_uri="data:,")
        for token, expected in ((None, None), ("abc", "Bearer abc")):
            with self.subTest(token=token), patch("alvarezgui.model_http.urlopen", return_value=reply) as opened:
                HttpModelService("http://model", token=token).infer(request)
                self.assertEqual(opened.call_args.args[0].get_header("Authorization"), expected)

    def test_generation_uses_model_family_processor_options(self) -> None:
        from alvarezgui.model import InferenceRequest

        request = InferenceRequest(request_id="r", model_version="v", instruction="i", image_uri="data:,", max_tokens=8)
        service = MiniCPMService(lambda _: "unused")
        from contextlib import nullcontext

        service.torch = SimpleNamespace(inference_mode=nullcontext)
        for model_type, template, generate in (
            ("minicpmv4_7", {"downsample_mode": "16x", "max_slice_nums": 36, "enable_thinking": False},
             {"downsample_mode": "16x"}),
            ("qwen3_vl", {}, {}),
        ):
            with self.subTest(model_type=model_type):
                class Inputs(dict):
                    input_ids = [[1, 2]]

                    def to(self, _device):
                        return self

                inputs = Inputs(input_ids=[[1, 2]])
                processor = Mock()
                processor.apply_chat_template.return_value = inputs
                processor.batch_decode.return_value = ['{"type":"back"}']
                model = Mock(config=SimpleNamespace(model_type=model_type))
                model.generate.return_value = SimpleNamespace(sequences=[[1, 2, 3]], scores=())
                self.assertEqual(service._generate(model, processor, [], request), ('{"type":"back"}', None))
                kwargs = processor.apply_chat_template.call_args.kwargs
                self.assertEqual({k: v for k, v in kwargs.items()
                                  if k not in {"tokenize", "add_generation_prompt", "return_dict", "return_tensors"}},
                                 template)
                self.assertEqual(model.generate.call_args.kwargs,
                                 {"max_new_tokens": 8, "do_sample": False, **generate, **inputs,
                                  "output_scores": True, "return_dict_in_generate": True})
                self.assertEqual(processor.batch_decode.call_args.args[0], [[3]])
        with self.assertRaisesRegex(ValueError, "unsupported model architecture"):
            service._generate(Mock(config=SimpleNamespace(model_type="llava")), Mock(), [], request)

    def test_load_checkpoint_factory_and_instance_load_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            with patch.object(MiniCPMService, "_model", return_value=(object(), object())) as load:
                service = MiniCPMService.load_checkpoint(checkpoint, "model-v1")
            load.assert_called_once_with("model-v1")
            self.assertEqual(service.max_loaded, 1)

    def test_delete_removes_checkpoint_and_unloads_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model-v1"
            checkpoint.mkdir()
            service = MiniCPMService(
                lambda version: str(checkpoint),
                loader=lambda path: (object(), object(), None),
                checkpoint_root=root,
            )
            service.load("model-v1")

            self.assertTrue(service.delete("model-v1"))
            self.assertFalse(checkpoint.exists())
            self.assertNotIn("model-v1", service.models)
            self.assertIn("model-v1", service.disabled_versions)

    def test_delete_rejects_unmanaged_and_symlinked_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            service = MiniCPMService(
                lambda version: str(outside),
                checkpoint_root=root / "managed",
            )
            with self.assertRaisesRegex(ValueError, "outside"):
                service.delete("outside-v1")

            managed = root / "managed"
            managed.mkdir()
            target = managed / "real-v1"
            target.mkdir()
            link = managed / "link-v1"
            link.symlink_to(target, target_is_directory=True)
            service.resolver = lambda version: str(link)
            with self.assertRaisesRegex(ValueError, "symlink"):
                service.delete("link-v1")

    def test_delete_requires_checkpoint_root(self) -> None:
        service = MiniCPMService(lambda version: "/tmp/model-v1")
        with self.assertRaisesRegex(ValueError, "root is not configured"):
            service.delete("model-v1")


if __name__ == "__main__":
    unittest.main()
