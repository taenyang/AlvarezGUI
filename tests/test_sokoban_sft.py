import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from alvarezgui.minicpm_service import service_instruction
from alvarezgui.minicpm_training import load_dataset
from alvarezgui.node_agent.android_tasks import SOKOBAN_INSTRUCTION
from alvarezgui.policy_output import OUTPUT_FORMATS, make_plan, parse_policy_output, policy_instruction
from alvarezgui.sokoban import ACTIONS, LEVEL_COUNTS, load_levels, parse_board, render_board, solve, transition
from alvarezgui.sokoban_sft import (
    augment_targets, collect, collect_augment_level, collect_level, evaluate_states, export_paired,
    observed_board, optimal_directions, registration_dataset,
)


class GameRuntime:
    """Plays the real single-box rules and serves boards/screenshots like the Android runtime."""

    def __init__(self, root, levels, broken=False):
        self.root, self.levels, self.broken, self.frames = root, levels, broken, 0

    def initialize_task(self, sandbox_id, task_type, app_image, instruction):
        self.walls, self.player, self.box, self.goal = parse_board(self.levels[app_image.split(":")[1]])

    def execute(self, sandbox_id, action):
        direction = next(key for key, value in ACTIONS.items() if value == action)
        if self.box != self.goal and not self.broken:
            self.player, self.box = transition(self.walls, self.player, self.box, direction)
        solved = self.box == self.goal
        return {"done": solved, "reward": {"total": int(solved), "success": solved}}

    def observe(self, sandbox_id):
        self.frames += 1
        image = self.root / f"frame-{self.frames}.png"
        board = render_board(self.walls, self.player, self.box, self.goal)
        image.write_bytes("|".join(board).encode())
        return {"image_uri": image.as_uri(), "screen_width": 720, "screen_height": 1280,
                "ui_text": ["Sokoban board: " + "|".join(board).replace(" ", "_")]
                + (["Sokoban complete"] if self.box == self.goal else [])}



class SokobanSFTTest(unittest.TestCase):
    board = ["########", "# .    #", "# $    #", "#   @  #", "#      #", "########"]

    def test_collection_aligns_pre_action_image_and_rejects_wrong_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            walls, player, box, goal = parse_board(self.board)
            observations = []
            boards = [self.board]
            for direction in solve(self.board):
                player, box = transition(walls, player, box, direction)
                boards.append(render_board(walls, player, box, goal))
            for index, board in enumerate(boards):
                image = root / f"source-{index}.png"
                image.write_bytes(f"frame-{index}".encode())
                observations.append({"image_uri": image.as_uri(), "screen_width": 720, "screen_height": 1280,
                                     "ui_text": ["Sokoban board: " + "|".join(board).replace(" ", "_")] +
                                     (["Sokoban complete"] if index == len(boards) - 1 else [])})
            runtime = Mock()
            runtime.observe.side_effect = observations
            runtime.execute.side_effect = [{"done": last, "reward": {"total": int(last), "success": last}}
                                           for last in (False, False, True)]
            output = root / "good"
            (output / "episodes").mkdir(parents=True)
            level = {"id": "train-000", "split": "train", "board": self.board}
            rows = collect_level(runtime, "sandbox", level, output)
            self.assertEqual([row["direction"] for row in rows], ["left", "left", "up"])
            self.assertEqual([Path(row["image"]).read_bytes() for row in rows],
                             [b"frame-0", b"frame-1", b"frame-2"])
            self.assertEqual(rows[0]["conversations"][0]["content"], service_instruction(SOKOBAN_INSTRUCTION))
            self.assertEqual(len(load_dataset(output / "episodes/train-000.jsonl")), 3)
            self.assertEqual(rows[-1]["reward"]["total"], 1)
            self.assertEqual(json.loads(rows[-1]["conversations"][-1]["content"]),
                             {"type": "tap", "x": 360, "y": 836})
            runtime.observe.side_effect = [observations[0], observations[0]]
            runtime.execute.side_effect = [{"done": False, "reward": {"total": 0, "success": False}}]
            bad = root / "bad"
            (bad / "episodes").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "expected move"):
                collect_level(runtime, "sandbox", level, bad)
            self.assertFalse((bad / "episodes/train-000.jsonl").exists())

    def test_augmented_states_are_reached_on_device_and_labeled_by_solver(self):
        import random
        level = {"id": "train-000", "split": "train", "board": self.board}
        targets = augment_targets(level, set(), random.Random(0))
        kinds = Counter(kind for kind, _ in targets)
        self.assertEqual(kinds, Counter({"player_on_goal": 3, "near_deadlock": 3, "off_path": 2}))
        walls, _, _, goal = parse_board(self.board)
        for kind, board in targets:
            _, player, box, _ = parse_board(board)
            solve(board)
            self.assertEqual(player == goal, kind == "player_on_goal")
        # The evaluation failure: box next to the left wall, one push from a deadlock.
        near = ["########", "# .    #", "# $@   #", "#      #", "#      #", "########"]
        with patch.dict("alvarezgui.sokoban_sft.AUGMENT_QUOTAS", {kind: 10_000 for kind in kinds}):
            everything = augment_targets(level, set(), random.Random(0))
        self.assertIn(("near_deadlock", near), everything)
        self.assertNotIn("+", "".join("".join(b) for k, b in everything if k != "player_on_goal"))
        excluded = {hashlib.sha256("\n".join(board).encode()).hexdigest() for _, board in targets}
        self.assertFalse(excluded & {hashlib.sha256("\n".join(b).encode()).hexdigest()
                                     for _, b in augment_targets(level, excluded, random.Random(0))})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "aug"
            rows = collect_augment_level(GameRuntime(root, {"train-000": self.board}), "s", level, targets, output)
            self.assertEqual([row["board"] for row in rows], [board for _, board in targets])
            for row in rows:
                self.assertIn(row["direction"], optimal_directions(row["board"]))
                self.assertEqual(Path(row["image"]).read_bytes(), "|".join(row["board"]).encode())
                self.assertEqual(row["output_format"], "action-v1")
            with self.assertRaisesRegex(ValueError, "differs from target"):
                collect_augment_level(GameRuntime(root, {"train-000": self.board}, broken=True),
                                      "s", level, targets, root / "broken")

    def test_workers_share_network_allocator_and_failed_collection_is_not_published(self):
        from threading import Barrier

        barrier = Barrier(2)

        def fail(*args):
            barrier.wait(timeout=5)
            raise ValueError("bad board")

        with tempfile.TemporaryDirectory() as directory, patch(
            "alvarezgui.node_agent.factory.runtime_from_settings"
        ) as factory, patch("alvarezgui.sokoban_sft.collect_level", side_effect=fail):
            output = Path(directory) / "dataset"
            settings = SimpleNamespace(sandbox_capacity=4, runtime_backend="firecracker")
            levels = Path(__file__).resolve().parents[1] / "fixtures/sokoban/levels.json"
            with self.assertRaisesRegex(ValueError, "bad board"):
                collect(settings, levels, output, 2, 7, ["train-000", "train-001"])
            factory.assert_called_once_with(settings)
            runtime = factory.return_value
            self.assertEqual(runtime.create.call_count, 2)
            self.assertEqual({call.args for call in runtime.create.call_args_list},
                             {call.args for call in runtime.destroy.call_args_list})
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], "failed")
            self.assertFalse((output / "train.jsonl").exists())

    def test_evaluation_uses_plain_prompt_and_scores_strict_json(self):
        from alvarezgui.model import InvalidActionOutputError, ModelUnavailableError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "screen.png"
            image.write_bytes(b"screenshot")
            action = {"type": "tap", "x": 176, "y": 980}
            row = {"id": "validation-000:0", "episode_id": "validation-000", "group_id": "validation-000",
                   "step_id": 0, "image": str(image), "split": "validation", "board": self.board,
                   "conversations": [{"role": "user", "content": service_instruction(SOKOBAN_INSTRUCTION)},
                                     {"role": "assistant", "content": json.dumps(action)}],
                   "reward": {"total": 0}, "done": False}
            dataset = root / "validation.jsonl"
            dataset.write_text((json.dumps(row) + "\n") * 3)
            with patch("alvarezgui.model_http.HttpModelService") as service:
                service.return_value.infer.side_effect = [
                    SimpleNamespace(action=action, raw_output=json.dumps(action)),
                    SimpleNamespace(action=action, raw_output="Answer: " + json.dumps(action)),
                    InvalidActionOutputError("invalid duration"),
                ] * 2
                with patch("alvarezgui.sokoban_sft.optimal_directions", wraps=optimal_directions) as solve_directions, \
                        patch("alvarezgui.sokoban_sft.base64.b64encode", wraps=base64.b64encode) as encode:
                    report = evaluate_states(dataset, "http://model", ["base", "candidate"], [0.0], root / "results.jsonl", 7)
                self.assertEqual(solve_directions.call_count, 3)
                self.assertEqual(encode.call_count, 3)
                self.assertEqual(service.return_value.infer.call_count, 6)
                self.assertEqual([result["valid"] for result in report["results"]], [1, 1])
                self.assertEqual([result["correct"] for result in report["results"]], [1, 1])
                request = service.return_value.infer.call_args_list[0].args[0]
                self.assertEqual(request.instruction, SOKOBAN_INSTRUCTION)
                self.assertTrue(request.image_uri.startswith("data:image/png;base64,"))
                service.return_value.infer.side_effect = ModelUnavailableError("offline")
                with self.assertRaises(ModelUnavailableError):
                    evaluate_states(dataset, "http://model", ["base"], [0.0], root / "failed.jsonl", 7)
                self.assertFalse((root / "failed.summary.json").exists())

    def _collection(self, root):
        root.mkdir()
        manifest = Path(__file__).resolve().parents[1] / "fixtures/sokoban/levels.json"
        (root / "levels.json").write_bytes(manifest.read_bytes())
        report = {"status": "complete", "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  "splits": {}}
        rows = {split: [] for split in LEVEL_COUNTS}
        for level in load_levels(manifest):
            walls, player, box, goal = parse_board(level["board"])
            for step, direction in enumerate(solve(level["board"])):
                board = render_board(walls, player, box, goal)
                image = root / f"{level['id']}-{step}.png"
                image.write_bytes(f"real-frame-{level['id']}-{step}".encode())
                player, box = transition(walls, player, box, direction)
                solved = box == goal
                rows[level["split"]].append({
                    "id": f"{level['id']}:{step}", "episode_id": level["id"], "group_id": level["id"],
                    "level_id": level["id"], "step_id": step, "image": str(image), "split": level["split"],
                    "board": board, "direction": direction, "optimal_directions": optimal_directions(board),
                    "state_hash": hashlib.sha256("\n".join(board).encode()).hexdigest(),
                    "screen_hash": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "conversations": [
                        {"role": "user", "content": policy_instruction(SOKOBAN_INSTRUCTION)},
                        {"role": "assistant", "content": json.dumps(ACTIONS[direction])},
                    ], "done": solved, "reward": {"total": int(solved), "success": solved},
                })
        for split, selected in rows.items():
            path = root / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in selected))
            report["splits"][split] = {
                "samples": len(selected), "levels": LEVEL_COUNTS[split],
                "directions": dict(Counter(row["direction"] for row in selected)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        (root / "report.json").write_text(json.dumps(report))
        return rows

    def test_paired_export_preserves_full_data_and_registers_balanced_train_only_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            original = self._collection(source)
            output = root / "paired"
            report = export_paired(source, output)
            self.assertEqual(report["teacher_source"], "solver-verified")
            source_before = (source / "train.jsonl").read_bytes()
            for output_format in OUTPUT_FORMATS:
                for split, original_rows in original.items():
                    rows = load_dataset(output / output_format / f"{split}.jsonl")
                    self.assertEqual([row["id"] for row in rows], [row["id"] for row in original_rows])
                    self.assertEqual([row["image"] for row in rows], [row["image"] for row in original_rows])
                    self.assertEqual([row["direction"] for row in rows], [row["direction"] for row in original_rows])
                    for row in rows:
                        self.assertEqual(row["teacher_source"], "solver-verified")
                        self.assertEqual(row["instruction"], SOKOBAN_INSTRUCTION)
                        self.assertEqual(row["conversations"][0]["content"], policy_instruction(SOKOBAN_INSTRUCTION, output_format))
                        self.assertNotIn("<image>", row["conversations"][0]["content"])
                        self.assertEqual(parse_policy_output(row["conversations"][-1]["content"], output_format), ACTIONS[row["direction"]])
                path, small = registration_dataset(output / output_format, "train-small.jsonl")
                self.assertEqual(path.name, "train-small.jsonl")
                self.assertEqual(len(small), 32)
                self.assertEqual(Counter(row["direction"] for row in small), Counter({direction: 8 for direction in ACTIONS}))
                self.assertTrue(all(row["split"] == "train" for row in small))
                self.assertEqual([row["id"] for row in small], report["small_train_ids"])
                _, full = registration_dataset(output / output_format)
                self.assertEqual(len(full), len(original["train"]))
            self.assertEqual((source / "train.jsonl").read_bytes(), source_before)
            with self.assertRaises(FileExistsError):
                export_paired(source, output)
            with self.assertRaisesRegex(ValueError, "registration requires"):
                registration_dataset(output / "action-v1", "test.jsonl")
            changed = output / "sokoban-plan-v1/train-small.jsonl"
            changed.write_text(changed.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                registration_dataset(output / "sokoban-plan-v1", "train-small.jsonl")

    def test_export_rejects_changed_hashes_mixed_splits_and_wrong_labels_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            original = self._collection(source)
            report_path = source / "report.json"
            original_report = report_path.read_text()
            train = source / "train.jsonl"
            original_train = train.read_text()
            output = root / "rejected"
            image = Path(original["train"][0]["image"])
            image_bytes = image.read_bytes()
            image.write_bytes(b"changed screenshot")
            with self.assertRaisesRegex(ValueError, "screen hash changed"):
                export_paired(source, output)
            self.assertFalse(output.exists())
            image.write_bytes(image_bytes)
            train.write_text(original_train + "\n")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                export_paired(source, output)
            for field, value, error in (
                ("split", "test", "mixed splits"),
                ("direction", "invalid", "direction counts disagree"),
                ("state_hash", "changed", "state hash changed"),
                ("conversations", [{"role": "user", "content": policy_instruction(SOKOBAN_INSTRUCTION)},
                                   {"role": "assistant", "content": '{"type":"back"}'}], "incorrect teacher action"),
            ):
                rows = json.loads(json.dumps(original["train"]))
                rows[0][field] = value
                train.write_text("".join(json.dumps(row) + "\n" for row in rows))
                report = json.loads(original_report)
                report["splits"]["train"]["sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
                report_path.write_text(json.dumps(report))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                    export_paired(source, output)
                self.assertFalse(output.exists())
            train.write_text(original_train)
            report_path.write_text(original_report)
            _, rows = registration_dataset(source)
            self.assertEqual(len(rows), len(original["train"]))

    def test_evaluation_reads_model_protocol_without_leaking_board_and_measures_wrong_state(self):
        from alvarezgui.model import InvalidActionOutputError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "screen.png"
            image.write_bytes(b"screenshot")
            action = ACTIONS["left"]
            row = {"id": "validation-000:0", "episode_id": "validation-000", "group_id": "validation-000",
                   "step_id": 0, "image": str(image), "split": "validation", "board": self.board,
                   "conversations": [{"role": "user", "content": policy_instruction(SOKOBAN_INSTRUCTION)},
                                     {"role": "assistant", "content": json.dumps(action)}],
                   "reward": {"total": 0}, "done": False}
            dataset = root / "validation.jsonl"
            dataset.write_text((json.dumps(row) + "\n") * 3)
            plan = make_plan(self.board, "left")
            wrong = json.loads(json.dumps(plan))
            wrong["expected_next_state"]["player"] = [4, 4]
            wrong["board"] = ["########", "# .    #", "# $    #", "#  @   #", "#      #", "########"]
            invalid = InvalidActionOutputError("bad plan")
            invalid.raw_output = "not json"
            invalid.output_format = "sokoban-plan-v1"
            with patch("alvarezgui.model_http.HttpModelService") as service:
                service.return_value.infer.side_effect = [
                    SimpleNamespace(action=action, raw_output=json.dumps(plan), output_format="sokoban-plan-v1", latency_ms=12),
                    SimpleNamespace(action=action, raw_output=json.dumps(wrong), output_format="sokoban-plan-v1", latency_ms=14),
                    invalid,
                ]
                report = evaluate_states(dataset, "http://model", ["plan"], [0.0], root / "results.jsonl", 7)
                result = report["results"][0]
                self.assertEqual(result["valid"], 2)
                self.assertEqual(result["correct_rate"], 2 / 3)
                self.assertEqual(result["board_exact_rate"], 1 / 3)
                self.assertEqual(result["next_state_correct_rate"], 1 / 3)
                self.assertEqual(result["output_length_samples"], 3)
                self.assertGreater(result["output_chars_mean"], 0)
                for call in service.return_value.infer.call_args_list:
                    request = call.args[0]
                    self.assertEqual(request.instruction, SOKOBAN_INSTRUCTION)
                    self.assertIsNone(request.max_tokens)
                records = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
                self.assertTrue(records[1]["valid"])
                self.assertFalse(records[1]["next_state_correct"])
                self.assertEqual(records[1]["raw_output"], json.dumps(wrong))
                row["split"] = "train"
                dataset.write_text(json.dumps(row) + "\n")
                with self.assertRaisesRegex(ValueError, "allow-train"):
                    evaluate_states(dataset, "http://model", ["plan"], [0.0], root / "denied.jsonl", 7)
                service.return_value.infer.side_effect = [invalid]
                diagnostic = evaluate_states(dataset, "http://model", ["plan"], [0.0], root / "diag.jsonl", 7, allow_train=True)
                self.assertTrue(diagnostic["diagnostic_only"])
                self.assertEqual(diagnostic["results"][0]["next_state_correct_rate"], 0)

    def test_state_reading_fails_closed_and_equivalent_optimal_moves_count(self):
        for observation in ({}, {"ui_text": ["Sokoban board: bad"]}):
            with self.assertRaises(ValueError):
                observed_board(observation)
        self.assertEqual(optimal_directions(self.board), ["left"])
        board = ["########", "#  .   #", "#  $   #", "#      #", "# @    #", "########"]
        self.assertEqual(set(optimal_directions(board)), {"up", "right"})


if __name__ == "__main__":
    unittest.main()
