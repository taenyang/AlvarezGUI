import json
import math
import statistics
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import unittest
from pathlib import Path

from alvarezgui.model_families import MINICPM_V_4_7, QWEN3_VL
from alvarezgui.minicpm_training import (
    DatasetValidationError, _adapter_base_model, _collate_sft, _direction_head, _direction_targets,
    _language_attention_targets, _lora_model, _masked_batch, _messages, _perception_features,
    _perception_head_features, _pool_spatial, _save_sft_epoch, _sft_checkpoints,
    _sft_dataset_digest, _structured_targets, _top_layer_modules,
    _visual_features, group_advantages,
    load_dataset, load_perception_head, main,
    train, train_perception,
)
from alvarezgui.policy_output import make_plan, policy_instruction


class MiniCPMTrainingTest(unittest.TestCase):
    def _dataset(self, directory: str, rows: list[dict]) -> Path:
        path = Path(directory) / "data.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return path

    def _row(self, image: Path, reward: float = 1.0) -> dict:
        return {
            "id": "episode-1/step-0",
            "episode_id": "episode-1",
            "group_id": "task-1",
            "step_id": 0,
            "image": str(image),
            "conversations": [
                {"role": "user", "content": "<image>\nDo the next action."},
                {"role": "assistant", "content": '{"type":"back"}'},
            ],
            "reward": {"total": reward},
            "done": True,
        }

    def test_load_dataset_validates_image_and_safe_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "screen.png"
            image.write_bytes(b"png")
            rows = load_dataset(self._dataset(directory, [self._row(image)]))
        self.assertEqual(rows[0]["id"], "episode-1/step-0")

    def test_load_dataset_rejects_missing_image_or_unsafe_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = self._row(Path(directory) / "missing.png")
            with self.assertRaises(DatasetValidationError):
                load_dataset(self._dataset(directory, [missing]))
            image = Path(directory) / "screen.png"
            image.write_bytes(b"png")
            unsafe = self._row(image)
            unsafe["conversations"][-1]["content"] = '{"type":"shell","command":"id"}'
            with self.assertRaises(DatasetValidationError):
                load_dataset(self._dataset(directory, [unsafe]))

    def _plan_row(self, image: Path) -> dict:
        row = self._row(image)
        row.update(output_format="sokoban-plan-v1", instruction="Solve Sokoban.")
        row["conversations"][-2]["content"] = policy_instruction(row["instruction"], row["output_format"])
        board = ["########", "#      #", "# @$ . #", "#      #", "#      #", "########"]
        row["conversations"][-1]["content"] = json.dumps(make_plan(board, "right"))
        return row

    def test_protocol_validation_accepts_legacy_and_rejects_mismatched_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory, "screen.png")
            image.write_bytes(b"png")
            legacy, plan = self._row(image), self._plan_row(image)
            rows = load_dataset(self._dataset(directory, [legacy, plan]))
            self.assertNotIn("output_format", rows[0])
            self.assertEqual(rows[1]["output_format"], "sokoban-plan-v1")
            for mutation in ("format", "instruction", "prompt", "completion", "missing_instruction", "legacy_prompt"):
                row = self._plan_row(image)
                if mutation == "format":
                    row["output_format"] = "unknown"
                elif mutation == "instruction":
                    row["instruction"] = "Different instruction"
                elif mutation == "prompt":
                    row["conversations"][-2]["content"] = policy_instruction(row["instruction"])
                elif mutation == "completion":
                    row["conversations"][-1]["content"] = '{"type":"back"}'
                elif mutation == "missing_instruction":
                    del row["instruction"]
                else:
                    row["output_format"] = "action-v1"
                    del row["instruction"]
                    row["conversations"][-1]["content"] = '{"type":"back"}'
                with self.subTest(mutation=mutation), self.assertRaises(DatasetValidationError):
                    load_dataset(self._dataset(directory, [row]))
            for completion in ('prefix {"type":"back"}', '{"type":"back","type":"wait"}', '{"type":"back"} trailing'):
                row = self._row(image)
                row["conversations"][-1]["content"] = completion
                with self.subTest(completion=completion), self.assertRaises(DatasetValidationError):
                    load_dataset(self._dataset(directory, [row]))

    def test_unsupported_training_and_protocol_mismatch_fail_before_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch("alvarezgui.minicpm_training._imports") as imports:
            cases = [
                ("gigpo", {"output_format": "sokoban-plan-v1"}),
                ("sft", {"finetuning_method": "other"}),
                ("sft", {"output_format": "other"}),
                ("sft", {"output_format": "sokoban-plan-v1"}),
                ("sft", {"seed": -1}),
                *[("sft", {"batch_size": size}) for size in (0, -1, True, 1.5, "2")],
                ("sft", {"cache_preprocessed": "false"}),
                ("sft", {"gradient_checkpointing": "true"}),
                ("sft", {"shuffle": "true"}),
                *[(algorithm, option) for algorithm in ("grpo", "gigpo")
                  for option in ({"shuffle": True},)],
                ("sft", {"train_top_layers": -1}),
                ("sft", {"train_top_layers": True}),
                ("sft", {"train_top_layers": 1.5}),
                ("sft", {"train_top_layers": "2"}),
                ("sft", {"train_top_layers": 2}),
                ("grpo", {"finetuning_method": "lora", "train_top_layers": 2}),
                ("gigpo", {"finetuning_method": "lora", "train_top_layers": 2}),
                *[("sft", {"finetuning_method": "lora", "train_top_layers": 2, "top_layer_learning_rate": rate})
                  for rate in (0, -1, float("nan"), float("inf"), True, "invalid")],
            ]
            for algorithm, kwargs in cases:
                with self.subTest(algorithm=algorithm, kwargs=kwargs), self.assertRaises(ValueError):
                    train(algorithm, "unused-model", [self._row(Path("unused"))], directory, **kwargs)
            plan = self._plan_row(Path("unused"))
            with self.assertRaises(DatasetValidationError):
                train("sft", "unused-model", [plan], directory)
            plan["conversations"][-2]["content"] += " Extra incompatible instruction."
            with self.assertRaises(DatasetValidationError):
                train("sft", "unused-model", [plan], directory, output_format="sokoban-plan-v1")
            Path(directory, "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "base_model_name_or_path": "original-base",
            }))
            self.assertEqual(_adapter_base_model(directory), "original-base")
            with self.assertRaisesRegex(ValueError, "existing adapter"):
                train("sft", directory, [self._row(Path("unused"))], directory)
            imports.assert_not_called()

    def test_masking_preserves_plan_tokens_and_masks_prompt_and_padding(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch; runs on CPU")

        class Batch(dict):
            @property
            def input_ids(self):
                return self["input_ids"]

        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        for output_format in ("action-v1", "sokoban-plan-v1"):
            batch = Batch(input_ids=torch.tensor([[1, 2, 3, 4, 5, 0]]), tgt_sizes=torch.tensor([1]))
            prompt = Batch(input_ids=torch.tensor([[1, 2, 3]]))
            with self.subTest(output_format=output_format), patch(
                "alvarezgui.minicpm_training._prepare", side_effect=[(batch, 6), (prompt, 3)]
            ):
                result, labels = _masked_batch(processor, {"output_format": output_format}, torch, family=MINICPM_V_4_7)
            self.assertEqual(labels.tolist(), [[-100, -100, -100, 4, 5, -100]])
            self.assertEqual(result.input_ids.tolist(), [[1, 2, 3, 4, 5, 0]])
            self.assertIn("target_sizes", result)
            self.assertNotIn("tgt_sizes", result)
        batch = Batch(input_ids=torch.tensor([[1, 2, 3, 4]]))
        with patch("alvarezgui.minicpm_training._prepare",
                   side_effect=[(batch, 4), (Batch(input_ids=torch.tensor([[1, 9]])), 2)]), \
                self.assertRaisesRegex(ValueError, "not a prefix"):
            _masked_batch(processor, {}, torch, family=QWEN3_VL)

    def test_sft_collation_keeps_image_order_prompt_masks_and_padding(self) -> None:
        try:
            import torch
            from transformers.feature_extraction_utils import BatchFeature
        except ImportError:
            self.skipTest("requires torch and transformers; runs on CPU")
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=99))
        samples = []
        for index, (length, prompt_length, slices) in enumerate(((7, 4, 2), (5, 2, 1))):
            ids = torch.arange(length).unsqueeze(0)
            batch = BatchFeature({
                "input_ids": ids, "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": torch.zeros_like(ids),
                "pixel_values": torch.full((1, 3, 14, slices * 14), float(index + 1)),
                "tgt_sizes": torch.full((slices, 2), index + 1, dtype=torch.int32),
            })
            prompt = BatchFeature({"input_ids": ids[:, :prompt_length]})
            with patch("alvarezgui.minicpm_training._prepare", side_effect=[(batch, length), (prompt, prompt_length)]):
                samples.append(_masked_batch(processor, {}, torch, family=MINICPM_V_4_7))
        batch, labels = _collate_sft(samples, processor, torch, family=MINICPM_V_4_7)
        self.assertEqual(batch["input_ids"].tolist(), [list(range(7)), [0, 1, 2, 3, 4, 99, 99]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1] * 7, [1] * 5 + [0, 0]])
        self.assertEqual(labels.tolist(), [[-100] * 4 + [4, 5, 6], [-100, -100, 2, 3, 4, -100, -100]])
        self.assertEqual(batch["target_sizes"].tolist(), [[1, 1], [1, 1], [2, 2]])
        self.assertEqual(batch["pixel_values"].shape, (1, 3, 14, 42))
        torch.testing.assert_close(batch["pixel_values"][..., :28], samples[0][0]["pixel_values"])
        torch.testing.assert_close(batch["pixel_values"][..., 28:], samples[1][0]["pixel_values"])
        batch["pixel_values"].zero_()
        labels.zero_()
        self.assertTrue(samples[0][0]["pixel_values"].eq(1).all())
        self.assertEqual(samples[1][1].tolist(), [[-100, -100, 2, 3, 4]])
        samples[0][0]["unexpected_images"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "processor keys"):
            _collate_sft(samples, processor, torch, family=MINICPM_V_4_7)

        # Qwen3-VL flattens every image's patches along dim 0 with one grid_thw row per image.
        qwen = []
        for index, (length, images) in enumerate(((6, 2), (4, 1))):
            ids = torch.arange(length).unsqueeze(0)
            qwen.append((BatchFeature({
                "input_ids": ids, "attention_mask": torch.ones_like(ids), "mm_token_type_ids": torch.zeros_like(ids),
                "pixel_values": torch.full((images * 4, 8), float(index + 1)),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * images) * (index + 1),
            }), ids.clone()))
        batch, labels = _collate_sft(qwen, processor, torch, family=QWEN3_VL)
        self.assertEqual(batch["pixel_values"].shape, (12, 8))
        self.assertEqual(batch["pixel_values"][:, 0].tolist(), [1.0] * 8 + [2.0] * 4)
        self.assertEqual(batch["image_grid_thw"].tolist(), [[1, 2, 2], [1, 2, 2], [2, 4, 4]])
        self.assertEqual(batch["input_ids"].tolist(), [list(range(6)), [0, 1, 2, 3, 99, 99]])
        self.assertEqual(labels.tolist(), [list(range(6)), [0, 1, 2, 3, -100, -100]])
        with self.assertRaisesRegex(ValueError, "Qwen3-VL SFT processor keys"):
            _collate_sft(samples[1:], processor, torch, family=QWEN3_VL)

    def test_visual_features_pool_direct_encoder_output(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")

        class TinyVision(torch.nn.Module):
            def get_vision_embedding(self, pixel_values, image_bound=None, tgt_sizes=None):
                return pixel_values

        batch = {
            "input_ids": torch.zeros((1, 3), dtype=torch.long),
            "pixel_values": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        }
        features = _visual_features(TinyVision(), batch, torch)
        torch.testing.assert_close(features, torch.tensor([[2.0, 3.0]]))

    def test_spatial_pooling_keeps_encoder_slices_and_token_bins(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")

        slices = [torch.arange(8, dtype=torch.float32).reshape(4, 2),
                  torch.arange(8, dtype=torch.float32).reshape(4, 2) + 10]
        features = _pool_spatial(slices, torch, batch_size=1, tokens=2)
        self.assertEqual(features.shape, (1, 8))
        torch.testing.assert_close(
            features,
            torch.tensor([[1.0, 2.0, 5.0, 6.0, 11.0, 12.0, 15.0, 16.0]]),
        )

    def test_messages_keep_both_images_before_policy_text(self) -> None:
        messages = _messages({"conversations": [
            {"role": "user", "content": "Choose an action."},
            {"role": "assistant", "content": "{}"},
        ]}, (object(), object()))
        self.assertEqual(len(messages[0]["content"]), 3)
        self.assertEqual(messages[0]["content"][-1], {"type": "text", "text": "Choose an action."})

    def test_perception_multiview_concatenates_full_and_board_crop(self) -> None:
        try:
            import torch
            from PIL import Image
        except ImportError:
            self.skipTest("requires torch and pillow")

        row = {"image": "/tmp/screen.png", "conversations": []}
        images = []

        def prepare(_processor, _row, *, family, prompt_only, image):
            images.append(image)
            return {"input_ids": torch.zeros((1, 1), dtype=torch.long)}, 1

        with patch("alvarezgui.minicpm_training._read_image", return_value=Image.new("RGB", (720, 1280))), \
                patch("alvarezgui.minicpm_training._prepare", side_effect=prepare), \
                patch("alvarezgui.minicpm_training._visual_features", side_effect=(
                    torch.ones((1, 4)), torch.ones((1, 4)) * 2
                )):
            features = _perception_features(Mock(), Mock(), row, torch, torch.device("cpu"), multiview=True)

        self.assertEqual(features.shape, (1, 8))
        self.assertEqual([image.size for image in images], [(720, 1280), (720, 1280)])
        torch.testing.assert_close(features, torch.tensor([[1., 1., 1., 1., 2., 2., 2., 2.]]))

        images.clear()
        with patch("alvarezgui.minicpm_training._read_image", return_value=Image.new("RGB", (720, 1280))), \
                patch("alvarezgui.minicpm_training._prepare", side_effect=prepare), \
                patch("alvarezgui.minicpm_training._visual_features", return_value=torch.ones((1, 4)) * 2) as encode:
            routed = _perception_head_features(
                Mock(), Mock(), row, torch, torch.device("cpu"), iter(("direction",)), task_views=True,
            )
        self.assertEqual(len(images), 1)
        encode.assert_called_once()
        torch.testing.assert_close(routed["direction"], torch.ones((1, 4)) * 2)

    def test_task_views_route_state_heads_to_expected_features(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")

        full = torch.ones((1, 2))
        crop = torch.ones((1, 3))
        with patch(
            "alvarezgui.minicpm_training._perception_feature_views",
            return_value={"full": full, "board-crop": crop},
        ):
            features = _perception_head_features(
                Mock(), Mock(), {}, torch, torch.device("cpu"),
                ("player", "box", "goal", "wall", "direction"), task_views=True,
            )
        self.assertIs(features["player"], full)
        self.assertIs(features["box"], full)
        self.assertIs(features["goal"], full)
        self.assertIs(features["wall"], crop)
        self.assertIs(features["direction"], crop)

    def test_direction_mlp_head_has_four_outputs(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")
        head = _direction_head(torch, 5, mlp=True)
        self.assertEqual(head[-1].out_features, 4)
        self.assertEqual(head[0].in_features, 5)

    def test_direction_targets_accept_all_optimal_moves(self) -> None:
        self.assertEqual(_direction_targets({"direction": "right", "optimal_directions": ["up", "right"]}), (0, 3))
        with self.assertRaises(DatasetValidationError):
            _direction_targets({"direction": "left", "optimal_directions": ["up", "right"]})

    def test_structured_targets_encode_goal_and_wall_layout(self) -> None:
        player, box, goal, walls = _structured_targets({
            "board": ["########", "#      #", "# @$ . #", "#      #", "#      #", "########"],
        })
        self.assertEqual((player, box, goal), (18, 19, 21))
        self.assertEqual(sum(walls), 24)

    def test_perception_head_checkpoint_round_trips(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            heads = torch.nn.ModuleDict({
                "player": torch.nn.Linear(2, 48), "box": torch.nn.Linear(2, 48),
            })
            (output / "perception_config.json").write_text(json.dumps({
                "feature_dim": 2, "classes": 48,
            }))
            torch.save(heads.state_dict(), output / "perception_head.pt")
            restored = load_perception_head(output, torch)
            for name, parameter in heads.state_dict().items():
                torch.testing.assert_close(parameter, restored.state_dict()[name])

    def test_routed_perception_checkpoint_restores_head_dimensions(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            dimensions = {"player": 2, "box": 2, "goal": 2, "wall": 3, "direction": 3}
            heads = torch.nn.ModuleDict({
                "player": torch.nn.Linear(2, 48), "box": torch.nn.Linear(2, 48),
                "goal": torch.nn.Linear(2, 48), "wall": torch.nn.Linear(3, 48),
                "direction": torch.nn.Linear(3, 4),
            })
            (output / "perception_config.json").write_text(json.dumps({
                "feature_dim": 2, "feature_dims": dimensions, "classes": 48,
                "structured_state": True, "direction": True,
            }))
            torch.save(heads.state_dict(), output / "perception_head.pt")
            restored = load_perception_head(output, torch)
            self.assertEqual(restored["player"].in_features, 2)
            self.assertEqual(restored["wall"].in_features, 3)
            self.assertEqual(restored["direction"].in_features, 3)

    def test_perception_training_freezes_backbone_and_writes_metrics(self) -> None:
        try:
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate")

        class TinyVision(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def get_vision_embedding(self, pixel_values, image_bound=None, tgt_sizes=None):
                return pixel_values

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory, "screen.png")
            image.write_bytes(b"png")
            row = self._row(image)
            row["board"] = ["########", "#      #", "# @$ . #", "#      #", "#      #", "########"]
            row["direction"] = "right"
            row["feature"] = 1
            processor = Mock()
            model = TinyVision()

            def prepared(_processor, row, family, prompt_only=False):
                feature = torch.tensor([[[float(row["feature"]), 1.0]]], dtype=torch.bfloat16)
                return {"input_ids": torch.zeros((1, 1), dtype=torch.long), "pixel_values": feature}, 1

            imports = (
                torch,
                lambda **kwargs: Accelerator(cpu=True, **kwargs),
                SimpleNamespace(from_pretrained=Mock(return_value=model)),
                SimpleNamespace(from_pretrained=Mock(return_value=processor)),
            )
            output = Path(directory, "perception")
            with patch("alvarezgui.minicpm_training._imports", return_value=imports), patch(
                "alvarezgui.minicpm_training._prepare", side_effect=prepared
            ):
                metrics = train_perception(
                    "unused-model", [row], output, epochs=1, direction=True, structured_state=True
                )
            self.assertEqual(metrics["algorithm"], "perception")
            self.assertEqual(metrics["feature_dim"], 32)
            self.assertIn(metrics["both_correct"], (0, 1))
            self.assertIn(metrics["direction_accuracy"], (0.0, 1.0))
            self.assertIn(metrics["goal_accuracy"], (0.0, 1.0))
            self.assertIn(metrics["wall_exact_accuracy"], (0.0, 1.0))
            self.assertTrue((output / "perception_head.pt").is_file())
            self.assertTrue((output / "perception_predictions.jsonl").is_file())
            restored = load_perception_head(output, torch)
            self.assertEqual(restored["player"].out_features, 48)
            self.assertEqual(restored["goal"].out_features, 48)
            self.assertEqual(restored["wall"].out_features, 48)
            self.assertEqual(restored["direction"].out_features, 4)
            self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_group_advantages_are_normalized_per_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "screen.png"
            image.write_bytes(b"png")
            rows = [self._row(image, 0.0), self._row(image, 2.0)]
            rows[1]["id"] = "episode-1/step-1"
        self.assertEqual(group_advantages(rows), [-1.0, 1.0])

    def test_policy_algorithms_reject_incomplete_group_before_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "screen.png"
            image.write_bytes(b"png")
            for algorithm in ("grpo", "gigpo"):
                row = self._row(image)
                row["old_logprob"] = -1.0
                with self.assertRaises(DatasetValidationError):
                    train(algorithm, "missing-model", [row], Path(directory) / algorithm)

    def test_real_peft_fp32_update_checkpoint_and_resume_on_cpu(self) -> None:
        try:
            import copy
            import torch
            import peft
            from accelerate import Accelerator
            from transformers import PretrainedConfig
        except ImportError:
            self.skipTest("requires torch, accelerate, transformers and peft; runs on CPU without downloads")

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.name_or_path = "tiny-base"
                self.is_gradient_checkpointing = False
                self.input_grad_calls = 0
                self.config = PretrainedConfig()
                self.config.model_type = "minicpmv4_7"
                self.config.text_config = PretrainedConfig()
                self.config.text_config.model_type = "qwen3_5_text"
                self.config.text_config.num_hidden_layers = 2
                self.config.text_config.layer_types = ["linear_attention", "full_attention"]
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
                linear, full = self.model.language_model.layers
                linear.linear_attn = torch.nn.ModuleDict({
                    name: torch.nn.Linear(2, 2, bias=False)
                    for name in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj")
                })
                full.self_attn = torch.nn.ModuleDict({
                    name: torch.nn.Linear(2, 2, bias=False) for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                })
                self.model.vision_tower = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(2, 2)})
                self.model.language_model.mlp = torch.nn.Linear(2, 2)

            def gradient_checkpointing_enable(self):
                self.is_gradient_checkpointing = True

            def enable_input_require_grads(self):
                self.input_grad_calls += 1

            def forward(self, features, use_cache=False):
                linear, full = self.model.language_model.layers
                logits = sum(layer(features) for layer in linear.linear_attn.values())
                logits = logits + sum(layer(features) for layer in full.self_attn.values())
                return SimpleNamespace(logits=logits)

        torch.manual_seed(7)
        pristine = TinyModel().to(torch.bfloat16)
        original_weights = {name: tensor.clone() for name, tensor in pristine.state_dict().items()}
        targets = _language_attention_targets(pristine, torch)
        self.assertEqual(len(targets), 9)
        self.assertTrue(all("language_model.layers." in name for name in targets))
        self.assertFalse(any("vision" in name or "mlp" in name for name in targets))
        bad_model = copy.deepcopy(pristine)
        bad_model.config.model_type = "unsupported"
        with self.assertRaisesRegex(ValueError, "unsupported model architecture"):
            _language_attention_targets(bad_model, torch)
        # Qwen3-VL: only text self-attention, never the vision blocks, MLP or mergers.
        qwen = torch.nn.Module()
        qwen.config = SimpleNamespace(model_type="qwen3_vl", text_config=SimpleNamespace(num_hidden_layers=2))
        qwen.model = torch.nn.Module()
        qwen.model.language_model = torch.nn.Module()
        qwen.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        for layer in qwen.model.language_model.layers:
            layer.self_attn = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in ("q_proj", "k_proj", "v_proj", "o_proj")})
            layer.mlp = torch.nn.ModuleDict({"gate_proj": torch.nn.Linear(2, 2)})
        qwen.model.visual = torch.nn.Module()
        qwen.model.visual.blocks = torch.nn.ModuleList([torch.nn.ModuleDict({"qkv": torch.nn.Linear(2, 2)})])
        qwen.model.visual.merger = torch.nn.ModuleDict({"linear_fc1": torch.nn.Linear(2, 2)})
        self.assertEqual(_language_attention_targets(qwen, torch), [
            f"model.language_model.layers.{index}.self_attn.{name}"
            for index in range(2) for name in ("q_proj", "k_proj", "v_proj", "o_proj")])
        self.assertEqual(_top_layer_modules(qwen.config, 1), ["model.language_model.layers.1"])
        del qwen.model.language_model.layers[1].self_attn
        with self.assertRaisesRegex(ValueError, "Qwen3-VL language attention"):
            _language_attention_targets(qwen, torch)
        features = torch.tensor([[[1., -1.], [.5, .5]]], dtype=torch.bfloat16)
        labels = torch.tensor([[-100, 0]])
        saved_models = []
        initial_adapters = []
        real_lora_model = _lora_model

        def lora_model(*args, **kwargs):
            model = real_lora_model(*args, **kwargs)
            initial_adapters.append({name: value.detach().clone() for name, value in model.named_parameters()
                                     if value.requires_grad})
            saved_models.append(model)
            return model

        with tempfile.TemporaryDirectory() as directory:
            base_path = str(Path(directory, "base"))
            pristine.name_or_path = base_path
            pristine.config.save_pretrained(base_path)
            processor = Mock()
            imports = (
                torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                SimpleNamespace(from_pretrained=Mock(side_effect=lambda *args, **kwargs: copy.deepcopy(pristine))),
                SimpleNamespace(from_pretrained=Mock(return_value=processor)),
            )
            first_output = Path(directory, "adapter")
            runs = ((base_path, first_output, True), (str(first_output), Path(directory, "resumed"), True),
                    (base_path, Path(directory, "repeated"), True),
                    (base_path, Path(directory, "no-checkpointing"), False))
            for source, output, checkpointing in runs:
                rows = [self._plan_row(Path("unused")) for _ in range(3)]
                with patch("alvarezgui.minicpm_training._imports", return_value=imports), patch(
                    "alvarezgui.minicpm_training._masked_batch", return_value=({"features": features}, labels)
                ), patch("alvarezgui.minicpm_training._lora_model", side_effect=lora_model):
                    metrics = train("sft", source, rows, output, learning_rate=.01, gradient_accumulation=2,
                                    finetuning_method="lora", output_format="sokoban-plan-v1",
                                    gradient_checkpointing=checkpointing)
                model = saved_models[-1]
                self.assertEqual(model.is_gradient_checkpointing, checkpointing)
                self.assertEqual(model.input_grad_calls, int(checkpointing))
                self.assertEqual(metrics["update_count"], 2)
                self.assertEqual(metrics["seed"], 20261117)
                trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
                self.assertTrue(trainable)
                self.assertTrue(all(value.dtype == torch.float32 for value in trainable.values()))
                self.assertTrue(any(not torch.equal(value, initial_adapters[-1][name])
                                    for name, value in trainable.items()))
                for name, value in model.named_parameters():
                    if not value.requires_grad:
                        self.assertEqual(value.dtype, torch.bfloat16)
                        base_name = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
                        self.assertTrue(torch.equal(value, original_weights[base_name]), name)
                self.assertTrue(Path(output, "adapter_config.json").is_file())
                self.assertTrue(Path(output, "adapter_model.safetensors").is_file())
                self.assertFalse(Path(output, "model.safetensors").exists())
                self.assertEqual(_adapter_base_model(output), base_path)
                processor.save_pretrained.assert_called_with(output)
                self.assertEqual(json.loads(Path(output, "policy_config.json").read_text()), {
                    "output_format": "sokoban-plan-v1", "max_tokens": 512,
                    "finetuning_method": "lora", "train_merger_mlp": False, "visual_views": "full", "seed": 20261117,
                })
                restored = real_lora_model(copy.deepcopy(pristine), torch, str(output))
                with torch.no_grad():
                    torch.testing.assert_close(restored(features=features).logits, model(features=features).logits)
                self.assertTrue(all(parameter.dtype == torch.float32 for parameter in restored.parameters()
                                    if parameter.requires_grad))
            for name, value in saved_models[0].named_parameters():
                torch.testing.assert_close(value, dict(saved_models[2].named_parameters())[name], rtol=0, atol=0)
            adapter_config_path = first_output / "adapter_config.json"
            adapter_config = json.loads(adapter_config_path.read_text())
            adapter_config["target_modules"].append("model.vision_tower.q_proj")
            adapter_config_path.write_text(json.dumps(adapter_config))
            with self.assertRaisesRegex(ValueError, "supported LoRA recipe"):
                real_lora_model(copy.deepcopy(pristine), torch, str(first_output))

            # GRPO rounds: the first trains a new FP32 adapter, the next continues from it; base stays frozen.
            source = base_path
            for round_number in range(2):
                output = Path(directory, f"grpo-round-{round_number}")
                rows = [{"id": index, "group_id": "g", "reward": {"total": float(index)}, "advantage": index - .5,
                         "conversations": self._row(Path("unused"))["conversations"]} for index in range(2)]
                with patch("alvarezgui.minicpm_training._imports", return_value=imports), patch(
                    "alvarezgui.minicpm_training._masked_batch", return_value=({"features": features}, labels)
                ), patch("alvarezgui.minicpm_training._lora_model", side_effect=lora_model):
                    metrics = train("grpo", source, rows, output, learning_rate=.01, finetuning_method="lora",
                                    compute_old_logprob=True)
                model = saved_models[-1]
                self.assertEqual(metrics["update_count"], 2)
                self.assertEqual(metrics["finetuning_method"], "lora")
                self.assertTrue(all("old_logprob" in row for row in rows))
                trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
                self.assertTrue(all(value.dtype == torch.float32 for value in trainable.values()))
                self.assertTrue(any(not torch.equal(value, initial_adapters[-1][name])
                                    for name, value in trainable.items()))
                for name, value in model.named_parameters():
                    if not value.requires_grad:
                        base_name = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
                        self.assertTrue(torch.equal(value, original_weights[base_name]), name)
                self.assertEqual(_adapter_base_model(output), base_path)
                self.assertFalse(Path(output, "model.safetensors").exists())
                if round_number:  # Continued from the previous round's adapter, not a fresh one.
                    previous = dict(saved_models[-2].named_parameters())
                    for name, value in initial_adapters[-1].items():
                        torch.testing.assert_close(value, previous[name].detach(), rtol=0, atol=0)
                source = str(output)
            with self.assertRaisesRegex(ValueError, "existing adapter requires finetuning_method='lora'"):
                train("grpo", source, rows, Path(directory, "full-on-adapter"), compute_old_logprob=True)

    def test_top_layer_selection_and_bounds_before_loading_weights(self) -> None:
        config = SimpleNamespace(model_type="minicpmv4_7", text_config=SimpleNamespace(
            model_type="qwen3_5_text", num_hidden_layers=24,
        ))
        self.assertEqual(_top_layer_modules(config, 2), [
            "model.language_model.layers.22", "model.language_model.layers.23",
        ])
        self.assertEqual(_top_layer_modules(None, 0), [])
        model_class, processor_class = Mock(), Mock()
        with patch("alvarezgui.minicpm_training._imports", return_value=(
            Mock(), Mock(return_value=Mock()), model_class, processor_class,
        )), patch.dict("sys.modules", {
            "accelerate.utils": SimpleNamespace(set_seed=Mock()),
            "transformers": SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config))),
        }):
            for count in (24, 25):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "at least one language layer"):
                    train("sft", "unused-model", [self._row(Path("unused"))], "unused-output",
                          finetuning_method="lora", train_top_layers=count)
        model_class.from_pretrained.assert_not_called()
        processor_class.from_pretrained.assert_not_called()

    def test_top_layer_cli_options_and_restrictions(self) -> None:
        arguments = ["minicpm_training", "--algorithm", "sft", "--base-model", "base",
                     "--dataset", "data.jsonl", "--output-dir", "output", "--finetuning-method", "lora",
                     "--train-top-layers", "2", "--top-layer-learning-rate", "0.000002"]
        with patch("sys.argv", arguments), patch("alvarezgui.minicpm_training.load_dataset", return_value=[]) as load, patch(
            "alvarezgui.minicpm_training.train", return_value={}
        ) as trainer, patch("builtins.print"):
            main()
            self.assertEqual(trainer.call_args.kwargs["train_top_layers"], 2)
            self.assertEqual(trainer.call_args.kwargs["top_layer_learning_rate"], 2e-6)
            for algorithm, method, count in (("sft", "full", "2"), ("sft", "lora", "-1"),
                                             ("grpo", "lora", "2"), ("gigpo", "lora", "2"),
                                             ("perception", "lora", "2")):
                arguments[2], arguments[10], arguments[12] = algorithm, method, count
                with self.subTest(algorithm=algorithm, method=method, count=count), patch("sys.stderr"), self.assertRaises(SystemExit):
                    main()
            load.assert_called_once()
            trainer.assert_called_once()

    def test_real_peft_top_layers_groups_inference_reload_and_epoch_resume(self) -> None:
        try:
            import copy
            import torch
            from accelerate import Accelerator
            from peft import PeftModel
            from transformers import PretrainedConfig
        except ImportError:
            self.skipTest("requires torch, accelerate, transformers and peft; runs on CPU without downloads")

        class TinyLayer(torch.nn.Module):
            def __init__(self, attention):
                super().__init__()
                self.attention = attention
                setattr(self, attention, torch.nn.ModuleDict({"q_proj": torch.nn.Linear(2, 2)}))
                self.mlp = torch.nn.Linear(2, 2)
                self.norm = torch.nn.LayerNorm(2)

            def forward(self, features):
                return getattr(self, self.attention)["q_proj"](features) + self.mlp(self.norm(features))

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = PretrainedConfig()
                self.config.model_type = "minicpmv4_7"
                self.config.text_config = PretrainedConfig()
                self.config.text_config.model_type = "qwen3_5_text"
                self.config.text_config.num_hidden_layers = 4
                self.config.text_config.layer_types = ["linear_attention", "full_attention"] * 2
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.ModuleList([
                    TinyLayer("linear_attn"), TinyLayer("self_attn"), TinyLayer("linear_attn"), TinyLayer("self_attn"),
                ])
                self.model.merger = torch.nn.Module()
                self.model.merger.mlp = torch.nn.Sequential(torch.nn.Linear(2, 2))
                self.model.vision_tower = torch.nn.Linear(2, 2)

            def forward(self, features, use_cache=False):
                features = self.model.merger.mlp(features)
                for layer in self.model.language_model.layers:
                    features = layer(features)
                return SimpleNamespace(logits=features)

        torch.manual_seed(19)
        pristine = TinyModel().to(torch.bfloat16)
        features = torch.tensor([[[1., -.5], [-.5, 1.]]], dtype=torch.bfloat16)
        labels = torch.tensor([[-100, 0]])
        real_adamw = torch.optim.AdamW
        with tempfile.TemporaryDirectory() as directory:
            base_path = str(Path(directory, "base"))
            pristine.name_or_path = base_path
            pristine.config.save_pretrained(base_path)
            image = Path(directory, "screen.png")
            image.write_bytes(b"image")
            rows = [self._row(image) for _ in range(3)]
            imports = (
                torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                SimpleNamespace(from_pretrained=Mock(side_effect=lambda *args, **kwargs: copy.deepcopy(pristine))),
                SimpleNamespace(from_pretrained=Mock(return_value=Mock())),
            )
            for merger in (False, True):
                with self.subTest(merger=merger):
                    models, optimizers = [], []
                    root = Path(directory, f"states-{merger}")
                    top_modules = _top_layer_modules(pristine.config, 2)

                    def lora_model(*args, **kwargs):
                        model = _lora_model(*args, **kwargs)
                        models.append(model)
                        return model

                    def adamw(*args, **kwargs):
                        optimizer = real_adamw(*args, **kwargs)
                        optimizers.append(optimizer)
                        return optimizer

                    def run(name, epochs, checkpoints=None):
                        output = Path(directory, f"{name}-{merger}")
                        with patch("alvarezgui.minicpm_training._imports", return_value=imports), patch(
                            "transformers.AutoConfig.from_pretrained", return_value=pristine.config
                        ), patch("alvarezgui.minicpm_training._masked_batch", return_value=({"features": features}, labels)), patch(
                            "alvarezgui.minicpm_training._lora_model", side_effect=lora_model
                        ), patch.object(torch.optim, "AdamW", side_effect=adamw), patch("builtins.print"):
                            metrics = train("sft", base_path, rows, output, epochs=epochs, gradient_accumulation=2,
                                            finetuning_method="lora", train_top_layers=2, train_merger_mlp=merger,
                                            checkpoint_dir=checkpoints)
                        return output, metrics

                    output, expected_metrics = run("uninterrupted-top", 2)
                    model, optimizer = models[-1], optimizers[-1]
                    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
                    self.assertTrue(all(parameter.dtype == torch.float32 for parameter in trainable.values()))
                    top = {name: parameter for name, parameter in trainable.items()
                           if any(f".{module}.modules_to_save." in name for module in top_modules)}
                    self.assertEqual(sum(p.numel() for p in top.values()),
                                     sum(p.numel() for layer in pristine.model.language_model.layers[2:] for p in layer.parameters()))
                    self.assertTrue(any(".mlp." in name for name in top))
                    self.assertTrue(any(".norm." in name for name in top))
                    self.assertFalse(any(".lora_" in name for name in top))
                    self.assertTrue(all(name in top or ".lora_" in name or (merger and ".merger.mlp.0.modules_to_save." in name)
                                        for name in trainable))
                    self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-5, 1e-6])
                    self.assertEqual({id(p) for p in optimizer.param_groups[1]["params"]}, {id(p) for p in top.values()})
                    self.assertEqual({id(p) for group in optimizer.param_groups for p in group["params"]},
                                     {id(p) for p in trainable.values()})
                    for module in top_modules:
                        original = pristine.get_submodule(module)
                        saved = model.get_submodule(f"base_model.model.{module}").modules_to_save["default"]
                        self.assertTrue(any(not torch.equal(p.float(), dict(saved.named_parameters())[name])
                                            for name, p in original.named_parameters()))
                    config = json.loads((output / "adapter_config.json").read_text())
                    self.assertEqual(set(config["modules_to_save"]), set(top_modules + (["model.merger.mlp.0"] if merger else [])))
                    self.assertFalse(any(target.startswith(f"{module}.") for module in top_modules for target in config["target_modules"]))
                    policy = json.loads((output / "policy_config.json").read_text())
                    for values in (policy, expected_metrics):
                        self.assertEqual(values["train_top_layers"], 2)
                        self.assertEqual(values["top_layer_learning_rate"], 1e-6)
                    restored = PeftModel.from_pretrained(copy.deepcopy(pristine), str(output),
                                                        is_trainable=False, autocast_adapter_dtype=True)
                    model.eval()
                    restored.eval()
                    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                        torch.testing.assert_close(restored(features=features).logits, model(features=features).logits,
                                                   rtol=.02, atol=.02)
                    trainable_restored = _lora_model(copy.deepcopy(pristine), torch, str(output),
                                                    train_merger_mlp=merger, train_top_layers=2)
                    self.assertTrue(all(p.dtype == torch.float32 for p in trainable_restored.parameters() if p.requires_grad))
                    for name, parameter in trainable.items():
                        torch.testing.assert_close(dict(trainable_restored.named_parameters())[name], parameter, rtol=0, atol=0)
                    with self.assertRaisesRegex(ValueError, "supported LoRA recipe"):
                        _lora_model(copy.deepcopy(pristine), torch, str(output), train_merger_mlp=merger, train_top_layers=1)
                    config["target_modules"].append("model.language_model.layers.3.self_attn.q_proj")
                    (output / "adapter_config.json").write_text(json.dumps(config))
                    with self.assertRaisesRegex(ValueError, "supported LoRA recipe"):
                        _lora_model(copy.deepcopy(pristine), torch, str(output), train_merger_mlp=merger, train_top_layers=2)
                    run("first-top", 1, root)
                    _, metrics = run("resumed-top", 2, root)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(models[-1].state_dict()[name], value, rtol=0, atol=0)
                    for key in ("loss", "step_count", "update_count"):
                        self.assertEqual(metrics[key], expected_metrics[key])
                    expected_state, state = optimizer.state_dict(), optimizers[-1].state_dict()
                    self.assertEqual(state["param_groups"], expected_state["param_groups"])
                    for index, values in expected_state["state"].items():
                        for key, value in values.items():
                            torch.testing.assert_close(state["state"][index][key], value, rtol=0, atol=0)
                    saved_recipe = json.loads((root / "epoch-00000002/training_state.json").read_text())["recipe"]
                    self.assertEqual(saved_recipe["train_top_layers"], 2)
                    self.assertEqual(saved_recipe["top_layer_learning_rate"], 1e-6)

    def test_sft_acceleration_cli_defaults_options_and_restrictions(self) -> None:
        arguments = ["minicpm_training", "--algorithm", "sft", "--base-model", "base",
                     "--dataset", "data.jsonl", "--output-dir", "output"]
        with patch("alvarezgui.minicpm_training.load_dataset", return_value=[]), patch(
            "alvarezgui.minicpm_training.train", return_value={}
        ) as trainer, patch("builtins.print"):
            for options, expected in (([], (1, False, True, False)),
                                      (["--batch-size", "4", "--cache-preprocessed", "--no-gradient-checkpointing",
                                        "--shuffle"], (4, True, False, True))):
                with patch("sys.argv", arguments + options):
                    main()
                self.assertEqual(tuple(trainer.call_args.kwargs[key] for key in (
                    "batch_size", "cache_preprocessed", "gradient_checkpointing", "shuffle",
                )), expected)
            rejected = [(algorithm, ["--shuffle"]) for algorithm in ("grpo", "gigpo", "perception")]
            rejected += [("perception", option) for option in (["--batch-size", "2"], ["--cache-preprocessed"])]
            for algorithm, option in rejected:
                with self.subTest(algorithm=algorithm, option=option), patch(
                    "sys.argv", arguments[:2] + [algorithm] + arguments[3:] + option
                ), patch("sys.stderr"), self.assertRaises(SystemExit):
                    main()
            # Policy algorithms take micro-batches and the preprocessing cache.
            with patch("sys.argv", arguments[:2] + ["grpo"] + arguments[3:] + ["--batch-size", "4", "--cache-preprocessed"]):
                main()
            self.assertEqual((trainer.call_args.kwargs["batch_size"], trainer.call_args.kwargs["cache_preprocessed"]), (4, True))
            with patch("sys.argv", arguments + ["--batch-size", "0"]), patch("sys.stderr"), self.assertRaises(SystemExit):
                main()
            self.assertEqual(trainer.call_count, 3)

    def test_cli_passes_checkpoint_dir_and_rejects_non_sft(self) -> None:
        arguments = ["minicpm_training", "--algorithm", "sft", "--base-model", "base",
                     "--dataset", "data.jsonl", "--output-dir", "output", "--checkpoint-dir", "states"]
        with patch("sys.argv", arguments), patch("alvarezgui.minicpm_training.load_dataset", return_value=[]) as load, patch(
            "alvarezgui.minicpm_training.train", return_value={}
        ) as trainer, patch("builtins.print"):
            main()
            self.assertEqual(trainer.call_args.kwargs["checkpoint_dir"], "states")
            for algorithm in ("grpo", "gigpo", "perception"):
                arguments[2] = algorithm
                with self.subTest(algorithm=algorithm), patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
            load.assert_called_once()
            trainer.assert_called_once()

    def test_sft_epoch_publication_ignores_partial_saves_and_retains_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def save_state(path):
                pending = Path(path)
                self.assertTrue(pending.name.endswith(".incomplete"))
                self.assertFalse((pending / "stale").exists())
                pending.mkdir()
                (pending / "model.safetensors").write_bytes(b"state")

            accelerator = Mock(save_state=Mock(side_effect=save_state))
            state = {"recipe": {}, "completed_epochs": 1, "losses": [.5], "updates": 1}
            for epoch in (1, 2):
                state["completed_epochs"] = epoch
                _save_sft_epoch(accelerator, root, state)
            previous = _sft_checkpoints(root)
            state["completed_epochs"] = 3

            def interrupted(path):
                save_state(path)
                (Path(path) / "stale").touch()
                raise RuntimeError("preempted while saving")

            accelerator.save_state.side_effect = interrupted
            with self.assertRaisesRegex(RuntimeError, "preempted"):
                _save_sft_epoch(accelerator, root, state)
            self.assertEqual(_sft_checkpoints(root), previous)
            self.assertFalse((root / "epoch-00000003").exists())
            accelerator.save_state.side_effect = save_state
            _save_sft_epoch(accelerator, root, state)
            self.assertEqual([path.name for path in _sft_checkpoints(root)], ["epoch-00000002", "epoch-00000003"])
            self.assertEqual(json.loads((root / "epoch-00000003/training_state.json").read_text()), state)
            self.assertFalse(list(root.glob(".*.incomplete")))

    def test_sft_resume_validates_recipe_rows_and_image_contents_before_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory, "screen.png")
            image.write_bytes(b"original image")
            rows = [self._row(image)]
            recipe = {
                "version": 1, "base_model": "unused-model", "adapter_base": None,
                "learning_rate": 1e-5, "gradient_accumulation": 1, "finetuning_method": "full",
                "output_format": "action-v1", "seed": 20261117, "multiview": False,
                "train_merger_mlp": False, "mixed_precision": "bf16",
                "dataset_sha256": _sft_dataset_digest(rows),
            }
            root = Path(directory, "states")
            completed = root / "epoch-00000002"
            completed.mkdir(parents=True)
            (completed / "training_state.json").write_text(json.dumps({
                "recipe": recipe, "completed_epochs": 2, "losses": [.5, .4], "updates": 2,
            }))
            cases = [
                {"learning_rate": .01}, {"gradient_accumulation": 2}, {"seed": 42},
                {"finetuning_method": "lora"}, {"multiview": True},
                {"batch_size": 2}, {"cache_preprocessed": True}, {"gradient_checkpointing": False},
            ]
            with patch("alvarezgui.minicpm_training._imports") as imports:
                for kwargs in cases:
                    with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "recipe or dataset"):
                        train("sft", "unused-model", rows, directory, epochs=2, checkpoint_dir=root, **kwargs)
                with self.assertRaisesRegex(ValueError, "recipe or dataset"):
                    train("sft", "different-model", rows, directory, epochs=2, checkpoint_dir=root)
                rows[0]["conversations"][-1]["content"] = '{"type":"wait"}'
                with self.assertRaisesRegex(ValueError, "recipe or dataset"):
                    train("sft", "unused-model", rows, directory, epochs=2, checkpoint_dir=root)
                rows = [self._row(image)]
                image.write_bytes(b"changed image")
                with self.assertRaisesRegex(ValueError, "recipe or dataset"):
                    train("sft", "unused-model", rows, directory, epochs=2, checkpoint_dir=root)
                image.write_bytes(b"original image")
                with self.assertRaisesRegex(ValueError, "exceeds requested epochs"):
                    train("sft", "unused-model", rows, directory, epochs=1, checkpoint_dir=root)
                for algorithm in ("grpo", "gigpo"):
                    with self.assertRaisesRegex(ValueError, "only for SFT"):
                        train(algorithm, "unused-model", rows, directory, checkpoint_dir=root)
                imports.assert_not_called()

                manifest = completed / "training_state.json"
                original_state = manifest.read_text()
                top_state = json.loads(original_state)
                top_state["recipe"].update(finetuning_method="lora", train_top_layers=2, top_layer_learning_rate=1e-6)
                manifest.write_text(json.dumps(top_state))
                for options in ({"train_top_layers": 1}, {"train_top_layers": 0},
                                {"train_top_layers": 2, "top_layer_learning_rate": 2e-6}):
                    with self.subTest(options=options), self.assertRaisesRegex(ValueError, "recipe or dataset"):
                        train("sft", "unused-model", rows, directory, epochs=2, checkpoint_dir=root,
                              finetuning_method="lora", **options)
                imports.assert_not_called()
                manifest.write_text(original_state)

                # A completed run must restore the latest state and export without any new steps.
                older = root / "epoch-00000001"
                older.mkdir()  # Missing metadata here also proves that only the latest is read.
                (root / ".epoch-00000003.incomplete").mkdir()
                torch, model, processor = Mock(), Mock(), Mock()
                model.config.model_type = "minicpmv4_7"
                model.parameters.return_value = []
                torch.utils.data.DataLoader.return_value = [0]
                accelerator = Mock(num_processes=1, is_main_process=True)
                accelerator.prepare.side_effect = lambda *objects: objects
                imports.return_value = (
                    torch, Mock(return_value=accelerator),
                    SimpleNamespace(from_pretrained=Mock(return_value=model)),
                    SimpleNamespace(from_pretrained=Mock(return_value=processor)),
                )
                on_step = Mock()
                with patch.dict("sys.modules", {"accelerate.utils": SimpleNamespace(set_seed=Mock())}), patch(
                    "alvarezgui.minicpm_training._checkpoint"
                ) as export, patch("builtins.print") as progress:
                    metrics = train("sft", "unused-model", rows, directory, epochs=2,
                                    checkpoint_dir=root, on_step=on_step)
                progress.assert_called_once()
                self.assertEqual(progress.call_args.kwargs, {"flush": True})
                self.assertEqual(json.loads(progress.call_args.args[0]), {
                    "event": "sft_resume", "checkpoint": str(completed), "completed_epochs": 2,
                    "step": 2, "total": 2, "updates": 2,
                })
                accelerator.load_state.assert_called_once_with(str(completed))
                accelerator.save_state.assert_not_called()
                on_step.assert_not_called()
                export.assert_called_once()
                self.assertEqual(metrics["step_count"], 2)
                self.assertEqual(metrics["update_count"], 2)
                self.assertEqual(metrics["completed_epochs"], 2)
                self.assertEqual(metrics["loss"], .45)

                accelerator.num_processes = 2
                with patch.dict("sys.modules", {"accelerate.utils": SimpleNamespace(set_seed=Mock())}):
                    with self.assertRaisesRegex(ValueError, "single Accelerate process"):
                        train("sft", "unused-model", rows, directory, epochs=2, checkpoint_dir=root)

    def test_sft_epoch_resume_matches_uninterrupted_model_optimizer_rng_and_metrics(self) -> None:
        try:
            import random
            import numpy as np
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate; runs on CPU without model downloads")

        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def forward(self, features, use_cache=False):
                score = self.weight * features * (torch.rand(()) + random.random() + np.random.random())
                return SimpleNamespace(logits=torch.stack((score, -score)).repeat(1, 2, 1))

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory, "screen.png")
            image.write_bytes(b"image")
            rows = [self._row(image) for _ in range(13)]
            root = Path(directory, "states")
            optimizers = []
            adamw = torch.optim.AdamW

            def optimizer(*args, **kwargs):
                result = adamw(*args, **kwargs)
                optimizers.append(result)
                return result

            def run(name, checkpoints=None, callback=None):
                model = TinyModel()
                output = Path(directory, name)
                output.mkdir(exist_ok=True)
                with patch("alvarezgui.minicpm_training._imports", return_value=(
                    torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                    SimpleNamespace(from_pretrained=Mock(return_value=model)),
                    SimpleNamespace(from_pretrained=Mock(return_value=Mock())),
                )), patch("alvarezgui.minicpm_training._masked_batch", return_value=(
                    {"features": torch.tensor(1.)}, torch.tensor([[-100, 0]])
                )), patch("alvarezgui.minicpm_training._checkpoint"), patch.object(
                    torch.optim, "AdamW", side_effect=optimizer
                ):
                    metrics = train("sft", "unused-model", rows, output, epochs=4, learning_rate=.01,
                                    gradient_accumulation=2, checkpoint_dir=checkpoints, on_step=callback)
                return model, metrics, (torch.rand(()), random.random(), np.random.random())

            with patch("builtins.print") as progress:
                expected_model, expected_metrics, expected_rng = run("uninterrupted")
            progress.assert_not_called()

            def preempt(step, _values):
                if step == 15:
                    raise RuntimeError("preempted during second epoch")

            with self.assertRaisesRegex(RuntimeError, "preempted"):
                run("interrupted", root, preempt)
            self.assertEqual([p.name for p in _sft_checkpoints(root)], ["epoch-00000001"])
            (root / ".epoch-99999999.incomplete").mkdir()
            steps = []
            with patch("builtins.print") as progress:
                model, metrics, rng = run("resumed", root, lambda step, _values: steps.append(step))
            events = [json.loads(call.args[0]) for call in progress.call_args_list]
            self.assertTrue(all(call.kwargs == {"flush": True} for call in progress.call_args_list))
            self.assertEqual([event["event"] for event in events], [
                "sft_resume", "sft_checkpoint", "sft_checkpoint", "sft_progress", "sft_checkpoint",
            ])
            self.assertEqual([event["step"] for event in events], [13, 26, 39, 50, 52])
            self.assertTrue(all(event["total"] == 52 for event in events))
            self.assertEqual(events[3]["updates"], 26)
            self.assertTrue(math.isfinite(events[3]["loss"]))
            self.assertEqual(steps, list(range(14, 53)))
            torch.testing.assert_close(model.weight, expected_model.weight, rtol=0, atol=0)
            for key in ("loss", "step_count", "update_count"):
                self.assertEqual(metrics[key], expected_metrics[key])
            self.assertEqual(metrics["update_count"], 28)  # Includes the partial group in each epoch.
            for actual, expected in zip(rng, expected_rng):
                self.assertEqual(actual, expected)
            expected_state, actual_state = optimizers[0].state_dict(), optimizers[-1].state_dict()
            self.assertEqual(actual_state["param_groups"], expected_state["param_groups"])
            for key, value in expected_state["state"][0].items():
                torch.testing.assert_close(actual_state["state"][0][key], value, rtol=0, atol=0)
            self.assertEqual([p.name for p in _sft_checkpoints(root)], ["epoch-00000003", "epoch-00000004"])
            steps.clear()
            _, finished_metrics, _ = run("already-finished", root, lambda step, _values: steps.append(step))
            self.assertEqual(steps, [])
            self.assertEqual(finished_metrics["loss"], expected_metrics["loss"])

    def test_save_every_publishes_intermediate_epoch_models(self) -> None:
        try:
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate; runs on CPU without model downloads")

        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def forward(self, features, use_cache=False):
                score = self.weight * features
                return SimpleNamespace(logits=torch.stack((score, -score)).repeat(1, 2, 1))

        def save(model, processor, accelerator, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "weight.txt").write_text(str(float(model.weight)))

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory, "screen.png")
            image.write_bytes(b"image")
            output = Path(directory, "run", "final")
            (output.parent / ".epoch-2.incomplete").mkdir(parents=True)  # Left by an interrupted save.
            with patch("alvarezgui.minicpm_training._imports", return_value=(
                torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                SimpleNamespace(from_pretrained=Mock(return_value=TinyModel())),
                SimpleNamespace(from_pretrained=Mock(return_value=Mock())),
            )), patch("alvarezgui.minicpm_training._masked_batch", return_value=(
                {"features": torch.tensor(1.)}, torch.tensor([[-100, 0]])
            )), patch("alvarezgui.minicpm_training._checkpoint", side_effect=save):
                train("sft", "unused-model", [self._row(image) for _ in range(3)], output, epochs=5,
                      learning_rate=.01, save_every=2)
            self.assertEqual(sorted(p.name for p in output.parent.iterdir()), ["epoch-2", "epoch-4", "final"])
            weights = [float((output.parent / name / "weight.txt").read_text()) for name in ("epoch-2", "epoch-4", "final")]
            self.assertTrue(weights[0] != weights[1] != weights[2])
            metrics = json.loads((output.parent / "epoch-4/metrics.json").read_text())
            self.assertEqual((metrics["epoch"], metrics["step_count"]), (4, 12))
            self.assertEqual(json.loads((output.parent / "epoch-2/policy_config.json").read_text()),
                             json.loads((output / "policy_config.json").read_text()))
            with self.assertRaisesRegex(ValueError, "save_every"):
                train("sft", "unused-model", [self._row(image)], output, save_every=0)

    def test_sft_microbatch_cache_recipe_resume_matches_model_optimizer_and_rng(self) -> None:
        try:
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate; runs on CPU")

        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.1))

            def forward(self, input_ids, use_cache=False, **_kwargs):
                scores = self.weight * input_ids.float() * torch.rand_like(input_ids.float())
                return SimpleNamespace(logits=torch.stack((scores, -scores), dim=-1))

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory, "image.png")
            image.write_bytes(b"image")
            rows = [dict(self._row(image), id=index) for index in range(7)]
            root = Path(directory, "states")
            optimizers = []
            adamw = torch.optim.AdamW

            def optimizer(*args, **kwargs):
                result = adamw(*args, **kwargs)
                optimizers.append(result)
                return result

            def masked(_processor, row, _torch, **_kwargs):
                ids = torch.ones((1, 3 + row["id"] % 3), dtype=torch.long)
                labels = torch.zeros_like(ids)
                labels[:, :2] = -100
                return {"input_ids": ids, "attention_mask": ids.clone(), "mm_token_type_ids": ids.clone(),
                        "pixel_values": torch.ones((1, 3, 14, 14)),
                        "target_sizes": torch.ones((1, 2), dtype=torch.int32)}, labels

            def run(name, epochs, checkpoints=None):
                output = Path(directory, name)
                output.mkdir()
                model = TinyModel()
                prepared = Mock(side_effect=masked)
                with patch("alvarezgui.minicpm_training._imports", return_value=(
                    torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                    SimpleNamespace(from_pretrained=Mock(return_value=model)),
                    SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(
                        tokenizer=SimpleNamespace(pad_token_id=0)
                    ))),
                )), patch("alvarezgui.minicpm_training._masked_batch", prepared), patch(
                    "alvarezgui.minicpm_training._checkpoint"
                ), patch.object(torch.optim, "AdamW", side_effect=optimizer), patch("builtins.print"):
                    metrics = train("sft", "unused-model", rows, output, epochs=epochs,
                                    gradient_accumulation=2, batch_size=2, cache_preprocessed=True,
                                    gradient_checkpointing=False, checkpoint_dir=checkpoints)
                self.assertEqual(prepared.call_count, len(rows))
                return model.weight.detach().clone(), metrics, torch.rand(()), optimizers[-1].state_dict()

            expected = run("uninterrupted", 3)
            run("first", 1, root)
            manifest = json.loads((root / "epoch-00000001/training_state.json").read_text())
            self.assertEqual(len(manifest["losses"]), 7)
            self.assertEqual(manifest["updates"], 2)
            self.assertEqual(manifest["recipe"]["batch_size"], 2)
            self.assertTrue(manifest["recipe"]["cache_preprocessed"])
            self.assertFalse(manifest["recipe"]["gradient_checkpointing"])
            with patch("alvarezgui.minicpm_training._imports") as imports:
                for mismatch in ({"batch_size": 4, "gradient_accumulation": 1}, {"cache_preprocessed": False},
                                 {"gradient_checkpointing": True}):
                    options = dict(batch_size=2, gradient_accumulation=2, cache_preprocessed=True,
                                   gradient_checkpointing=False)
                    options.update(mismatch)
                    with self.subTest(mismatch=mismatch), self.assertRaisesRegex(ValueError, "recipe or dataset"):
                        train("sft", "unused-model", rows, directory, epochs=3, checkpoint_dir=root, **options)
                imports.assert_not_called()
            actual = run("resumed", 3, root)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
            for key in ("loss", "step_count", "update_count"):
                self.assertEqual(actual[1][key], expected[1][key])
            self.assertEqual(actual[3]["param_groups"], expected[3]["param_groups"])
            for key, value in expected[3]["state"][0].items():
                torch.testing.assert_close(actual[3]["state"][0][key], value, rtol=0, atol=0)

    def test_sft_shuffle_is_a_seeded_per_epoch_permutation_that_resumes(self) -> None:
        try:
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate; runs on CPU")

        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.1))
                self.seen = []

            def forward(self, input_ids, use_cache=False, **_kwargs):
                self.seen.append(int(input_ids[0, 0]) - 1)
                scores = self.weight * input_ids.float()
                return SimpleNamespace(logits=torch.stack((scores, -scores), dim=-1))

        def masked(_processor, row, _torch, **_kwargs):
            ids = torch.full((1, 4), row["id"] + 1, dtype=torch.long)
            labels = torch.zeros_like(ids)
            labels[:, :2] = -100
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, labels

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory, "image.png")
            image.write_bytes(b"image")
            rows = [dict(self._row(image), id=index) for index in range(7)]
            root = Path(directory, "states")

            def run(name, epochs, shuffle=True, seed=1, checkpoints=None):
                output = Path(directory, name)
                output.mkdir()
                model = TinyModel()
                with patch("alvarezgui.minicpm_training._imports", return_value=(
                    torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                    SimpleNamespace(from_pretrained=Mock(return_value=model)),
                    SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace())),
                )), patch("alvarezgui.minicpm_training._masked_batch", side_effect=masked), patch(
                    "alvarezgui.minicpm_training._checkpoint"
                ), patch("builtins.print"):
                    metrics = train("sft", "unused-model", rows, output, epochs=epochs, seed=seed,
                                    gradient_accumulation=2, shuffle=shuffle, checkpoint_dir=checkpoints)
                self.assertEqual(metrics["shuffle"], shuffle)
                return [model.seen[start:start + len(rows)] for start in range(0, len(model.seen), len(rows))]

            self.assertEqual(run("plain", 2, shuffle=False), [list(range(7))] * 2)
            epochs = run("shuffled", 3)
            self.assertTrue(all(sorted(order) == list(range(7)) for order in epochs))
            self.assertEqual(len({tuple(order) for order in epochs}), 3)
            self.assertEqual(run("repeat", 3), epochs)
            self.assertNotEqual(run("other-seed", 3, seed=2), epochs)
            self.assertEqual(run("first", 1, checkpoints=root), epochs[:1])
            with patch("alvarezgui.minicpm_training._imports") as imports, \
                    self.assertRaisesRegex(ValueError, "recipe or dataset"):
                train("sft", "unused-model", rows, directory, epochs=3, seed=1, gradient_accumulation=2,
                      checkpoint_dir=root)
            imports.assert_not_called()
            self.assertEqual(run("resumed", 3, checkpoints=root), epochs[1:])

    def test_sft_microbatches_match_sample_mean_gradients_updates_and_cache(self) -> None:
        try:
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate; runs on CPU")
        test = self

        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def __init__(self, count):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.15))
                self.count = count
                self.seen = []

            def forward(self, input_ids, attention_mask, mm_token_type_ids, pixel_values, target_sizes, use_cache=False):
                markers = pixel_values[0, 0, 0, ::14]
                offset = 0
                image_features = []
                for ids, modalities in zip(input_ids, mm_token_type_ids):
                    slices = int(modalities.sum())
                    marker = int(ids[0])
                    test.assertTrue(markers[offset:offset + slices].eq(marker).all())
                    self.seen.append(marker - 1)
                    image_features.append(markers[offset:offset + slices].mean())
                    offset += slices
                test.assertEqual(offset, len(target_sizes))
                image_features = torch.stack(image_features)[:, None] / self.count
                features = image_features + torch.arange(input_ids.shape[1])[None, :] / 10
                score = self.weight * features
                # A cached processor result must survive models which mutate their inputs.
                pixel_values.zero_()
                input_ids.zero_()
                return SimpleNamespace(logits=torch.stack((score, -score), dim=-1))

        for count, epochs in ((333, 1), (7, 2), (6, 2), (5, 2)):
            baseline = None
            for batch_size, accumulation in ((1, 4), (2, 2), (4, 1)):
                for cache in (False, True):
                    with self.subTest(count=count, batch_size=batch_size, cache=cache), tempfile.TemporaryDirectory() as directory:
                        rows = [dict(self._row(Path("unused")), id=index) for index in range(count)]
                        model = TinyModel(count)
                        gradients = []
                        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
                        real_step = optimizer.step

                        def step(*args, **kwargs):
                            gradients.append(model.weight.grad.detach().clone())
                            return real_step(*args, **kwargs)

                        optimizer.step = step
                        prepared = []

                        def masked(_processor, row, _torch, **_kwargs):
                            index = row["id"]
                            prepared.append(index)
                            slices = 1 + index % 3
                            prompt = 2 + slices + index % 2
                            completion = 1 + index % 4
                            ids = torch.full((1, prompt + completion), index + 1, dtype=torch.long)
                            labels = torch.full_like(ids, -100)
                            labels[:, prompt:] = torch.arange(completion) % 2
                            modalities = torch.zeros_like(ids)
                            modalities[:, 1:1 + slices] = 1
                            return {
                                "input_ids": ids, "attention_mask": torch.ones_like(ids),
                                "mm_token_type_ids": modalities,
                                "pixel_values": torch.full((1, 3, 14, slices * 14), float(index + 1)),
                                "target_sizes": torch.ones((slices, 2), dtype=torch.int32),
                            }, labels

                        callbacks = []
                        with patch("alvarezgui.minicpm_training._imports", return_value=(
                            torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                            SimpleNamespace(from_pretrained=Mock(return_value=model)),
                            SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(
                                tokenizer=SimpleNamespace(pad_token_id=0)
                            ))),
                        )), patch("alvarezgui.minicpm_training._masked_batch", side_effect=masked), patch(
                            "alvarezgui.minicpm_training._checkpoint"
                        ), patch.object(torch.optim, "AdamW", return_value=optimizer):
                            metrics = train("sft", "unused-model", rows, directory, epochs=epochs,
                                            batch_size=batch_size, gradient_accumulation=accumulation,
                                            cache_preprocessed=cache, gradient_checkpointing=False,
                                            on_step=lambda step, values: callbacks.append((step, values)))
                        self.assertEqual(model.seen, list(range(count)) * epochs)
                        self.assertEqual(prepared, list(range(count)) * (1 if cache else epochs))
                        self.assertEqual(metrics["step_count"], count * epochs)
                        self.assertEqual(metrics["update_count"], math.ceil(count / 4) * epochs)
                        self.assertEqual(len(gradients), metrics["update_count"])
                        self.assertEqual([step for step, _ in callbacks], list(range(1, count * epochs + 1)))
                        self.assertEqual(sum("train/grad_norm" in values for _, values in callbacks), len(gradients))
                        self.assertEqual(metrics["batch_size"], batch_size)
                        self.assertEqual(metrics["effective_batch_size"], 4)
                        self.assertEqual(metrics["cache_preprocessed"], cache)
                        self.assertFalse(metrics["gradient_checkpointing"])
                        self.assertEqual(metrics["peak_gpu_memory_bytes"], 0)
                        self.assertGreater(metrics["preprocessing_seconds"], 0)
                        self.assertGreater(metrics["training_seconds"], 0)
                        self.assertGreater(metrics["samples_per_second"], 0)
                        losses = torch.tensor([values["train/loss"] for _, values in callbacks])
                        result = (model.weight.detach(), torch.stack(gradients), losses)
                        if baseline is None:
                            baseline = result
                        else:
                            for actual, expected in zip(result, baseline):
                                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    def test_accumulation_flushes_partial_groups_and_counts_updates_separately(self) -> None:
        try:
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate; runs on CPU without model downloads")

        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def forward(self, features, use_cache=False):
                score = self.weight * features
                return SimpleNamespace(logits=torch.stack((score, score * 0)).repeat(1, 2, 1))

        for algorithm, count, epochs in (("sft", 333, 1), ("sft", 5, 2), ("grpo", 5, 2), ("gigpo", 5, 2)):
            with self.subTest(algorithm=algorithm, count=count, epochs=epochs), tempfile.TemporaryDirectory() as directory:
                model = TinyModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                optimizer.step = Mock(wraps=optimizer.step)
                processor = Mock()
                callbacks = []
                rows = [{"id": index, "group_id": "group", "reward": {"total": 1},
                         "conversations": self._row(Path("unused"))["conversations"],
                         "old_logprob": -math.log(2), "advantage": 1.0}
                        for index in range(count)]

                def masked_batch(_processor, row, _torch, **_kwargs):
                    # Only the last sample has a gradient: dropping the tail leaves weight at zero.
                    return {"features": torch.tensor(float(row["id"] == count - 1))}, torch.tensor([[-100, 0]])

                def checkpoint(saved_model, _processor, accelerator, output):
                    self.assertIs(accelerator.unwrap_model(saved_model), model)
                    self.assertGreater(model.weight.item(), 0)
                    self.assertIsNone(model.weight.grad)

                with patch("alvarezgui.minicpm_training._imports", return_value=(
                    torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                    SimpleNamespace(from_pretrained=Mock(return_value=model)),
                    SimpleNamespace(from_pretrained=Mock(return_value=processor)),
                )), patch("alvarezgui.minicpm_training._masked_batch", side_effect=masked_batch), patch(
                    "alvarezgui.minicpm_training._checkpoint", side_effect=checkpoint
                ), patch.object(torch.optim, "AdamW", return_value=optimizer):
                    metrics = train(algorithm, "unused-model", rows, directory, epochs=epochs,
                                    gradient_accumulation=4, on_step=lambda step, values: callbacks.append((step, values)))
                expected_updates = math.ceil(count / 4) * epochs
                self.assertEqual(optimizer.step.call_count, expected_updates)
                self.assertEqual(metrics["update_count"], expected_updates)
                self.assertEqual(metrics["step_count"], count * epochs)
                self.assertEqual([step for step, _ in callbacks], list(range(1, count * epochs + 1)))
                self.assertEqual(sum("train/grad_norm" in values for _, values in callbacks), expected_updates)
                self.assertAlmostEqual(callbacks[count - 1][1]["train/grad_norm"], .5, places=6)
                if algorithm == "sft":
                    self.assertFalse(any("train/entropy" in values for _, values in callbacks))
                    self.assertNotIn("entropy_mean", metrics)
                else:
                    # Every policy step reports entropy; logits [w*x, 0] start uniform over 2 tokens (w=0).
                    self.assertTrue(all(values["train/entropy"] >= 0 for _, values in callbacks))
                    self.assertAlmostEqual(callbacks[0][1]["train/entropy"], math.log(2), places=6)
                    self.assertAlmostEqual(metrics["entropy_mean"], statistics.fmean(
                        values["train/entropy"] for _, values in callbacks), places=9)
                if epochs == 1:
                    self.assertAlmostEqual(model.weight.item(), .05, places=6)
                self.assertEqual(json.loads(Path(directory, "metrics.json").read_text()), metrics)
                self.assertEqual(json.loads(Path(directory, "policy_config.json").read_text()), {
                    "output_format": "action-v1", "max_tokens": 128, "finetuning_method": "full",
                    "train_merger_mlp": False, "visual_views": "full", "seed": 20261117,
                })

    def test_policy_loss_weights_and_clip_higher(self) -> None:
        try:
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate; runs on CPU without model downloads")

        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def forward(self, features, use_cache=False):
                score = self.weight * features
                return SimpleNamespace(logits=torch.stack((score, score * 0)).repeat(1, 2, 1))

        def run(rows, **options):
            model = TinyModel()
            callbacks = []
            with tempfile.TemporaryDirectory() as directory, patch("alvarezgui.minicpm_training._imports", return_value=(
                torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                SimpleNamespace(from_pretrained=Mock(return_value=model)),
                SimpleNamespace(from_pretrained=Mock(return_value=Mock())),
            )), patch("alvarezgui.minicpm_training._masked_batch", side_effect=lambda _p, row, _t, **_k: (
                {"features": torch.tensor(float(row["feature"]))}, torch.tensor([[-100, 0]]))
            ), patch("alvarezgui.minicpm_training._checkpoint"), patch.object(
                torch.optim, "AdamW", side_effect=lambda params, lr: torch.optim.SGD(params, lr=lr)
            ):
                train("dapo", "unused-model", rows, directory, learning_rate=1.0, gradient_accumulation=len(rows),
                      on_step=lambda step, values: callbacks.append(values), **options)
            return model.weight.item(), callbacks

        def row(feature, advantage, weight=None, old=-math.log(2)):
            return {"id": feature, "group_id": "g", "reward": {"total": 1}, "feature": feature,
                    "conversations": self._row(Path("unused"))["conversations"], "old_logprob": old,
                    "advantage": advantage, **({"loss_weight": weight} if weight is not None else {})}

        # One accumulated update; d logprob/d w = feature / 2 at w = 0, so the step is
        # lr * mean(normalized_weight * advantage * feature / 2). Weights 1 and 3 normalize to 0.5 and 1.5.
        weight, callbacks = run([row(1.0, 1.0, 1.0), row(-1.0, 1.0, 3.0)])
        self.assertAlmostEqual(weight, (0.5 * 0.5 - 1.5 * 0.5) / 2, places=6)
        self.assertEqual([values["train/loss_weight"] for values in callbacks], [0.5, 1.5])
        unweighted, _ = run([row(1.0, 1.0), row(-1.0, 1.0)])
        self.assertAlmostEqual(unweighted, 0.0, places=6)  # Missing weights are 1: the two rows cancel.
        # ratio = exp(logprob - old) = 1.25 for A > 0: clipped (no gradient) at eps 0.2, not at eps_high 0.28.
        stale = -math.log(2) - math.log(1.25)
        clipped, callbacks = run([row(1.0, 1.0, old=stale)])
        self.assertEqual((clipped, callbacks[0]["train/clipped"]), (0.0, 1.0))
        higher, callbacks = run([row(1.0, 1.0, old=stale)], clip_epsilon_high=0.28)
        self.assertGreater(higher, 0.0)
        self.assertEqual(callbacks[0]["train/clipped"], 0.0)
        with self.assertRaises(DatasetValidationError):
            run([row(1.0, 1.0, 0.0)])
        with self.assertRaises(ValueError):
            run([row(1.0, 1.0)], clip_epsilon_high=1.5)

    def test_policy_microbatches_and_cache_match_batch_one(self) -> None:
        """batch_size x gradient_accumulation windows give batch one's gradients, updates and per-row metrics."""
        try:
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate; runs on CPU without model downloads")
        test = self

        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.3))
                self.seen = []

            def forward(self, input_ids, attention_mask, mm_token_type_ids, pixel_values, target_sizes, use_cache=False):
                markers = pixel_values[0, 0, 0, ::14]
                features, offset = [], 0
                for ids, modalities in zip(input_ids, mm_token_type_ids):
                    slices = int(modalities.sum())
                    test.assertTrue(markers[offset:offset + slices].eq(int(ids[0])).all())
                    self.seen.append(int(ids[0]) - 1)
                    features.append(markers[offset:offset + slices].mean() / 7)
                    offset += slices
                position = torch.arange(input_ids.shape[1])[None, :] / 10
                score = self.weight * (torch.stack(features)[:, None] + position)
                pixel_values.zero_()  # The cache must survive models that mutate their inputs.
                return SimpleNamespace(logits=torch.stack((score, -score), dim=-1))

        def masked(_processor, row, _torch, **_kwargs):
            index = row["id"]
            prepared.append(index)
            slices, prompt, completion = 1 + index % 3, 3 + index % 2, 1 + index % 3
            ids = torch.full((1, prompt + completion), index + 1, dtype=torch.long)
            labels = torch.full_like(ids, -100)
            labels[:, prompt:] = torch.arange(completion) % 2
            modalities = torch.zeros_like(ids)
            modalities[:, 1:1 + slices] = 1
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "mm_token_type_ids": modalities,
                    "pixel_values": torch.full((1, 3, 14, slices * 14), float(index + 1)),
                    "target_sizes": torch.ones((slices, 2), dtype=torch.int32)}, labels

        count = 7  # Not a multiple of the window: the short last window must weigh like batch one's.
        baseline = None
        for batch_size, accumulation in ((1, 4), (2, 2), (4, 1)):
            for cache in (False, True):
                with self.subTest(batch_size=batch_size, cache=cache), tempfile.TemporaryDirectory() as directory:
                    rows = [{"id": index, "group_id": f"g{index % 2}", "reward": {"total": index % 3},
                             "advantage": (index % 3) - 1.0, "loss_weight": 1.0 + index % 2,
                             "conversations": self._row(Path("unused"))["conversations"]} for index in range(count)]
                    model, prepared, callbacks, gradients = TinyModel(), [], [], []
                    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
                    real_step = optimizer.step

                    def step(*args, **kwargs):
                        gradients.append(model.weight.grad.detach().clone())
                        return real_step(*args, **kwargs)

                    optimizer.step = step
                    with patch("alvarezgui.minicpm_training._imports", return_value=(
                        torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                        SimpleNamespace(from_pretrained=Mock(return_value=model)),
                        SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(
                            tokenizer=SimpleNamespace(pad_token_id=0)))),
                    )), patch("alvarezgui.minicpm_training._masked_batch", side_effect=masked), patch(
                        "alvarezgui.minicpm_training._checkpoint"
                    ), patch.object(torch.optim, "AdamW", return_value=optimizer):
                        metrics = train("dr_grpo", "unused-model", rows, directory, compute_old_logprob=True,
                                        batch_size=batch_size, gradient_accumulation=accumulation,
                                        cache_preprocessed=cache, gradient_checkpointing=False,
                                        on_step=lambda step, values: callbacks.append((step, values)))
                    # old_logprob pass, then the update pass; the cache prepares each row once for both.
                    self.assertEqual(prepared, list(range(count)) * (1 if cache else 2))
                    self.assertEqual(model.seen, list(range(count)) * 2)
                    self.assertEqual(metrics["update_count"], math.ceil(count / 4))
                    self.assertEqual([step for step, _ in callbacks], list(range(1, count + 1)))
                    per_row = torch.tensor([[values[key] for key in ("train/loss", "train/ratio", "train/advantage",
                                                                    "train/loss_weight", "train/entropy")]
                                            for _, values in callbacks])
                    result = (model.weight.detach(), torch.stack(gradients), per_row,
                              torch.tensor([row["old_logprob"] for row in rows]))
                    if baseline is None:
                        baseline = result
                    else:
                        for actual, expected in zip(result, baseline):
                            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_policy_training_uses_the_base_checkpoint_visual_views(self) -> None:
        try:
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate; runs on CPU without model downloads")

        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def forward(self, features, use_cache=False):
                score = self.weight * features
                return SimpleNamespace(logits=torch.stack((score, score * 0)).repeat(1, 2, 1))

        for views in ("full+board-crop", "full", None):
            with self.subTest(views=views), tempfile.TemporaryDirectory() as directory:
                base, output = Path(directory, "base"), Path(directory, "out")
                base.mkdir()
                output.mkdir()
                if views:
                    (base / "policy_config.json").write_text(json.dumps({"visual_views": views}))
                rows = [{"id": index, "group_id": "g", "reward": {"total": index}, "advantage": index - .5,
                         "conversations": self._row(Path("unused"))["conversations"]} for index in range(2)]
                calls = []

                def masked_batch(_processor, row, _torch, **kwargs):
                    calls.append(kwargs["multiview"])
                    return {"features": torch.tensor(1.)}, torch.tensor([[-100, 0]])

                with patch("alvarezgui.minicpm_training._imports", return_value=(
                    torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                    SimpleNamespace(from_pretrained=Mock(return_value=TinyModel())),
                    SimpleNamespace(from_pretrained=Mock()),
                )), patch("alvarezgui.minicpm_training._masked_batch", side_effect=masked_batch), patch(
                    "alvarezgui.minicpm_training._checkpoint"
                ):
                    train("grpo", str(base), rows, output, compute_old_logprob=True, multiview=False)
                multiview = views == "full+board-crop"
                self.assertEqual(calls, [multiview] * 4)  # old logprobs, then the update, both like sampling
                self.assertEqual(json.loads((output / "policy_config.json").read_text())["visual_views"],
                                 views or "full")

    def test_policy_logprobs_use_the_rollout_sampling_temperature(self) -> None:
        try:
            import torch
            from accelerate import Accelerator
        except ImportError:
            self.skipTest("requires torch and accelerate; runs on CPU without model downloads")

        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(model_type="minicpmv4_7")

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, features, use_cache=False):
                score = self.weight * features  # Logits (2, 0) at the token being scored.
                return SimpleNamespace(logits=torch.stack((score, score * 0)).repeat(1, 2, 1))

        def old_logprob(temperature):
            with tempfile.TemporaryDirectory() as directory:
                base, output = Path(directory, "base"), Path(directory, "out")
                base.mkdir()
                output.mkdir()
                rows = [{"id": index, "group_id": "g", "reward": {"total": index}, "advantage": index - .5,
                         "conversations": self._row(Path("unused"))["conversations"]} for index in range(2)]
                with patch("alvarezgui.minicpm_training._imports", return_value=(
                    torch, lambda **kwargs: Accelerator(cpu=True, **kwargs),
                    SimpleNamespace(from_pretrained=Mock(return_value=TinyModel())),
                    SimpleNamespace(from_pretrained=Mock()),
                )), patch("alvarezgui.minicpm_training._masked_batch", return_value=(
                    {"features": torch.tensor(2.)}, torch.tensor([[-100, 0]])
                )), patch("alvarezgui.minicpm_training._checkpoint"):
                    train("grpo", str(base), rows, output, compute_old_logprob=True, learning_rate=1e-9,
                          sampling_temperature=temperature)
                return rows[0]["old_logprob"]

        logprob = lambda t: float(torch.log_softmax(torch.tensor([2.0, 0.0]) / t, dim=-1)[0])
        for temperature, expected in ((None, logprob(1)), (0, logprob(1)), (1.0, logprob(1)), (0.5, logprob(0.5))):
            with self.subTest(temperature=temperature):
                self.assertAlmostEqual(old_logprob(temperature), expected, places=2)  # bf16 autocast
        with self.assertRaisesRegex(ValueError, "sampling_temperature"):
            train("grpo", "unused", [self._row(Path("unused"))], "out", sampling_temperature=-1)


if __name__ == "__main__":
    unittest.main()
