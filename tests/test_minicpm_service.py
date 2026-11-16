import tempfile
import unittest
from pathlib import Path

from alvarezgui.consumer import _model_image_uri
from alvarezgui.minicpm_service import MiniCPMService, _parse_action
from alvarezgui.model import InvalidActionOutputError


class MiniCPMServiceTest(unittest.TestCase):
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
        with self.assertRaises(InvalidActionOutputError):
            _parse_action('{"command":"shell","value":"id"}')
        with self.assertRaises(InvalidActionOutputError):
            _parse_action('{"type":"back"} trailing output')

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
