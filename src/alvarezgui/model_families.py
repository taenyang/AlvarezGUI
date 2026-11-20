"""Architecture facts the trainer and model service need, keyed by ``config.json`` ``model_type``.

Supporting another Hugging Face image-text-to-text model means adding one entry here, after
checking on the real checkpoint: module paths (``named_modules``), the processor's output keys
and shapes for one and two images, and that the prompt-only tokens prefix the full conversation.
``scripts/probe_model_family.py`` checks these on the real checkpoint and runs a short SFT + reload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ModelFamily:
    name: str
    model_type: str
    # Image tensors returned by the processor -> dimension along which rows are concatenated
    # when several samples form one SFT microbatch.
    image_concat: dict[str, int]
    # Per-token tensors, each (1, sequence_length) for one sample; padded when batching.
    text_keys: tuple[str, ...] = ("input_ids", "attention_mask", "mm_token_type_ids")
    # LoRA targets every Linear under ``{language_layers}.{i}.<attention>.`` for each text layer.
    language_layers: str = "model.language_model.layers"
    attention_modules: tuple[str, ...] = ("self_attn",)
    # Vision-language projection trained in full by ``train_merger_mlp``.
    merger_module: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    generate_kwargs: dict[str, Any] = field(default_factory=dict)
    # Applied to the loaded base model before any adapter wraps it, in training and serving alike,
    # so both always run the same numerics. Must be numerically equivalent to the original module.
    prepare_model: Callable[[Any], None] | None = None


def _patch_embed_as_linear(model: Any) -> None:
    """Qwen3-VL's patch Conv3d has kernel == stride, i.e. one linear map per flattened patch.

    In the H800 image (torch 2.9.0+cu129) this Conv3d alone took 19.8 s for one 720x1280 screenshot
    and the matmul 5 ms; whole Qwen inference fell from ~22 s to 0.45 s. The cause (cuDNN algorithm
    choice vs. version) was not isolated. On the 4090 (torch 2.11) a whole inference averaged 0.8 s
    with the Conv3d. Results differ only by bf16 rounding and are slightly closer to an fp64
    reference. The same weight tensor is used, so checkpoints are unchanged.
    """
    import torch

    embed = model.get_submodule("model.visual.patch_embed")
    proj = embed.proj
    if proj.kernel_size != proj.stride or any(proj.padding) or proj.groups != 1:
        raise ValueError("Qwen3-VL patch embedding is not a non-overlapping Conv3d")

    def forward(hidden_states: Any) -> Any:
        weight = proj.weight
        patches = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
        return torch.nn.functional.linear(patches, weight.reshape(weight.shape[0], -1), proj.bias)

    embed.forward = forward


MINICPM_V_4_7 = ModelFamily(
    name="MiniCPM-V 4.7",
    model_type="minicpmv4_7",
    # NaViT packs all image slices into one strip; forward consumes pixel_values[:1].
    image_concat={"pixel_values": -1, "target_sizes": 0},
    # Qwen3.5 text alternates full and gated linear attention.
    attention_modules=("self_attn", "linear_attn"),
    merger_module="model.merger.mlp.0",
    chat_template_kwargs={"downsample_mode": "16x", "max_slice_nums": 36, "enable_thinking": False},
    generate_kwargs={"downsample_mode": "16x"},
)
QWEN3_VL = ModelFamily(
    name="Qwen3-VL",
    model_type="qwen3_vl",
    # Patches of all images are flattened along dim 0; grid_thw has one row per image.
    image_concat={"pixel_values": 0, "image_grid_thw": 0},
    merger_module="model.visual.merger",
    prepare_model=_patch_embed_as_linear,
)
FAMILIES = {family.model_type: family for family in (MINICPM_V_4_7, QWEN3_VL)}


def prepare_model(model: Any) -> Any:
    family = model_family(model.config)
    if family.prepare_model is not None:
        family.prepare_model(model)
    return model


def model_family(config: Any) -> ModelFamily:
    model_type = getattr(config, "model_type", None)
    family = FAMILIES.get(model_type)
    if family is None:
        raise ValueError(f"unsupported model architecture {model_type!r}; supported: {sorted(FAMILIES)} "
                         "(add it to alvarezgui.model_families)")
    return family

