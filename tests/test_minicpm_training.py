import json
import tempfile
import unittest
from pathlib import Path

from alvarezgui.minicpm_training import DatasetValidationError, group_advantages, load_dataset, train


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


if __name__ == "__main__":
    unittest.main()
