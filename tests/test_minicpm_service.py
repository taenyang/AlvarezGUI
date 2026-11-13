import tempfile
import unittest
from pathlib import Path

from alvarezgui.consumer import _model_image_uri
from alvarezgui.minicpm_service import _parse_action
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


if __name__ == "__main__":
    unittest.main()
