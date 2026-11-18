import json
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
                    transformers = SimpleNamespace(AutoModelForImageTextToText=Mock(), AutoProcessor=Mock())
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
                service, "_generate", return_value=json.dumps(plan)
            ) as generate:
                response = service.infer(request)
                self.assertEqual(response.action, plan["action"])
                self.assertEqual(response.output_format, "sokoban-plan-v1")
                self.assertEqual(generate.call_args.args[3].max_tokens, 512)
                self.assertIn("Output format: sokoban-plan-v1", generate.call_args.args[2][0]["content"][1]["text"])
                self.assertIsNone(request.max_tokens)
                service.infer(request.model_copy(update={"max_tokens": 64}))
                self.assertEqual(generate.call_args.args[3].max_tokens, 64)
                # A valid nested action must not rescue an invalid structured response.
                generate.return_value = json.dumps({"action": plan["action"]})
                with self.assertRaises(InvalidActionOutputError) as error:
                    service.infer(request)
                self.assertEqual(error.exception.raw_output, generate.return_value)
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
                service, "_generate", return_value=json.dumps(ACTIONS["left"])
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
