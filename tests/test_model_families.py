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

    def test_qwen_patch_embedding_runs_as_identical_linear_map(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")

        class PatchEmbed(torch.nn.Module):
            # Same shape logic as transformers' Qwen3VLVisionPatchEmbed.
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Conv3d(3, 8, kernel_size=[2, 4, 4], stride=[2, 4, 4], bias=True)

            def forward(self, hidden_states):
                hidden_states = hidden_states.view(-1, 3, 2, 4, 4)
                return self.proj(hidden_states.to(self.proj.weight.dtype)).view(-1, 8)

        model = torch.nn.Module()
        model.config = SimpleNamespace(model_type="qwen3_vl")
        model.model = torch.nn.Module()
        model.model.visual = torch.nn.Module()
        model.model.visual.patch_embed = PatchEmbed()
        torch.manual_seed(0)
        patches = torch.randn(5, 3 * 2 * 4 * 4)
        expected = model.model.visual.patch_embed(patches)
        weight = model.model.visual.patch_embed.proj.weight
        self.assertIs(prepare_model(model), model)
        actual = model.model.visual.patch_embed(patches)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertIs(model.model.visual.patch_embed.proj.weight, weight)  # Checkpoints are unchanged.
        actual.sum().backward()
        self.assertIsNotNone(weight.grad)

        model.model.visual.patch_embed.proj = torch.nn.Conv3d(3, 8, kernel_size=[2, 4, 4], stride=[1, 4, 4])
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            prepare_model(model)
        minicpm = SimpleNamespace(config=SimpleNamespace(model_type="minicpmv4_7"))
        self.assertIs(prepare_model(minicpm), minicpm)


if __name__ == "__main__":
    unittest.main()
