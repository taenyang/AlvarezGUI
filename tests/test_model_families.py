import unittest
from types import SimpleNamespace

from alvarezgui.model_families import FAMILIES, MINICPM_V_4_7, QWEN3_VL, model_family, prepare_model


class ModelFamilyTest(unittest.TestCase):
    def test_lookup_by_model_type_and_clear_error_for_unknown(self) -> None:
        self.assertIs(model_family(SimpleNamespace(model_type="minicpmv4_7")), MINICPM_V_4_7)
        self.assertIs(model_family(SimpleNamespace(model_type="qwen3_vl")), QWEN3_VL)
        with self.assertRaisesRegex(ValueError, "unsupported model architecture 'llava'.*model_families"):
            model_family(SimpleNamespace(model_type="llava"))
        for family in FAMILIES.values():
            self.assertIn("input_ids", family.text_keys)
            self.assertFalse(set(family.text_keys) & set(family.image_concat))

    def test_prepare_model_leaves_models_without_a_hook_unchanged(self) -> None:
        for model_type in ("minicpmv4_7", "qwen3_vl"):
            model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
            self.assertIs(prepare_model(model), model)
        self.assertIsNone(QWEN3_VL.prepare_model)  # Removed: torch 2.14+ runs its Conv3d in ~1 ms.


if __name__ == "__main__":
    unittest.main()
