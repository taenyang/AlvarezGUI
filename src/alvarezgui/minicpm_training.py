from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import random
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Iterable

from alvarezgui.minicpm_service import _read_image, token_entropies
from alvarezgui.model_families import MINICPM_V_4_7, ModelFamily, model_family, prepare_model
from alvarezgui.policy_output import PLAN_INSTRUCTION_MARKER, parse_policy_output, policy_instruction
from alvarezgui.sokoban import parse_board

POLICY_ALGORITHMS = {"grpo", "dapo", "dr_grpo", "gigpo"}
POLICY_MAX_TOKENS = {"action-v1": 128, "sokoban-plan-v1": 512}
PERCEPTION_CLASSES = 48
DIRECTION_CLASSES = 4
DIRECTION_HIDDEN = 256
DIRECTIONS = ("up", "down", "left", "right")
PERCEPTION_CROP_BOX = (180, 270, 540, 650)
PERCEPTION_CROP_SIZE = (720, 760)
PERCEPTION_MODEL_SIZE = (720, 1280)
PERCEPTION_HEAD_VIEWS = {"player": "full", "box": "full", "goal": "full", "wall": "board-crop", "direction": "board-crop"}


class DatasetValidationError(ValueError):
    pass


def _validate_protocol(row: dict[str, Any], output_format: str) -> None:
    row_format = row.get("output_format", "action-v1")
    if not isinstance(row_format, str) or row_format not in POLICY_MAX_TOKENS or row_format != output_format:
        raise DatasetValidationError(f"row output_format {row_format!r} does not match {output_format!r}")
    turns = row.get("conversations", [])
    if (not isinstance(turns, list) or len(turns) < 2 or not all(isinstance(turn, dict) for turn in turns)
            or turns[-2].get("role") != "user" or turns[-1].get("role") != "assistant"):
        raise DatasetValidationError("final turns must be user then assistant")
    prompt = turns[-2].get("content")
    if not isinstance(prompt, str) or not prompt.strip():
        raise DatasetValidationError("final user prompt must be non-empty text")
    instruction = row.get("instruction")
    if "instruction" in row:
        if not isinstance(instruction, str) or prompt != policy_instruction(instruction, output_format):
            raise DatasetValidationError("final user prompt does not match instruction/output_format")
    elif output_format != "action-v1":
        raise DatasetValidationError("structured rows require instruction to verify the final user prompt")
    elif PLAN_INSTRUCTION_MARKER in prompt:
        raise DatasetValidationError("action-v1 row contains a structured output prompt")
    if row.get("invalid_output") is True and isinstance(turns[-1].get("content"), str):
        return  # A rejected rollout output, trained only with its (negative) policy advantage.
    try:
        parse_policy_output(turns[-1].get("content"), output_format)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"assistant completion is not valid {output_format}: {exc}") from exc


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise DatasetValidationError(f"line {line_number}: row must be an object")
            rows.append(_validate_row(row, line_number))
    if not rows:
        raise DatasetValidationError("dataset must contain at least one row")
    return rows


def _validate_row(row: dict[str, Any], line_number: int) -> dict[str, Any]:
    required = ("id", "episode_id", "group_id", "step_id", "image", "conversations", "reward", "done")
    missing = [name for name in required if name not in row]
    if missing:
        raise DatasetValidationError(f"line {line_number}: missing {', '.join(missing)}")
    if not all(isinstance(row[name], str) and row[name] for name in ("id", "episode_id", "group_id", "image")):
        raise DatasetValidationError(f"line {line_number}: identifiers and image must be non-empty strings")
    if not Path(row["image"]).is_file():
        raise DatasetValidationError(f"line {line_number}: image does not exist on this host")
    if not isinstance(row["step_id"], int) or row["step_id"] < 0:
        raise DatasetValidationError(f"line {line_number}: step_id must be a non-negative integer")
    if not isinstance(row["done"], bool):
        raise DatasetValidationError(f"line {line_number}: done must be boolean")
    try:
        _validate_protocol(row, row.get("output_format", "action-v1"))
    except DatasetValidationError as exc:
        raise DatasetValidationError(f"line {line_number}: {exc}") from exc
    reward = row["reward"]
    if not isinstance(reward, dict):
        raise DatasetValidationError(f"line {line_number}: reward must be an object")
    try:
        float(reward["total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetValidationError(f"line {line_number}: reward.total must be numeric") from exc
    return row


def group_advantages(rows: Iterable[dict[str, Any]]) -> list[float]:
    rows = list(rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(float(row["reward"]["total"]))
    means = {group: sum(values) / len(values) for group, values in grouped.items()}
    scales = {
        group: math.sqrt(sum((value - means[group]) ** 2 for value in values) / len(values)) or 1.0
        for group, values in grouped.items()
    }
    return [
        (float(row["reward"]["total"]) - means[str(row["group_id"])]) / scales[str(row["group_id"])]
        for row in rows
    ]


def _imports() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from accelerate import Accelerator
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("real training requires torch, accelerate and transformers") from exc
    return torch, Accelerator, AutoModelForImageTextToText, AutoProcessor


def _adapter_base_model(path: str | Path) -> str | None:
    """Detect a local PEFT checkpoint without loading weights or initializing CUDA."""
    config_path = Path(path) / "adapter_config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if config.get("peft_type") != "LORA" or not isinstance(base_model, str) or not base_model:
        raise ValueError("adapter checkpoint must be LoRA with base_model_name_or_path")
    return base_model


def _language_attention_targets(model: Any, torch: Any) -> list[str]:
    # Full paths are essential: vision towers also have q_proj/k_proj/v_proj names.
    family = model_family(model.config)
    layers = getattr(getattr(model.config, "text_config", None), "num_hidden_layers", None)
    if not isinstance(layers, int) or layers < 1:
        raise ValueError("language layer configuration is incomplete")
    linear_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    targets = []
    for index in range(layers):
        prefixes = tuple(f"{family.language_layers}.{index}.{attention}." for attention in family.attention_modules)
        layer_targets = [name for name in linear_names if name.startswith(prefixes)]
        if not layer_targets:
            raise ValueError(f"no {family.name} language attention linear modules under {prefixes}")
        targets.extend(layer_targets)
    return targets


def _top_layer_modules(config: Any, count: int) -> list[str]:
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("train_top_layers must be a non-negative integer")
    if count == 0:
        return []
    family = model_family(config)
    total = config.text_config.num_hidden_layers
    if count >= total:
        raise ValueError("train_top_layers must leave at least one language layer for LoRA")
    return [f"{family.language_layers}.{index}" for index in range(total - count, total)]


def _lora_model(
    model: Any,
    torch: Any,
    adapter_path: str | None = None,
    *,
    train_merger_mlp: bool = False,
    train_top_layers: int = 0,
) -> Any:
    try:
        from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA training requires peft") from exc
    top_modules = _top_layer_modules(model.config, train_top_layers)
    targets = [name for name in _language_attention_targets(model, torch)
               if not any(name.startswith(f"{module}.") for module in top_modules)]
    family = model_family(model.config)
    merger_module = family.merger_module
    if train_merger_mlp and not any(name == merger_module for name, _ in model.named_modules()):
        raise ValueError(f"{family.name} model has no supported merger module: {merger_module}")
    modules_to_save = top_modules + ([merger_module] if train_merger_mlp else [])
    if adapter_path is None:
        config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=targets, modules_to_save=modules_to_save or None,
        )
    else:
        config = PeftConfig.from_pretrained(adapter_path)
        # Do not silently resume an adapter with a different trainable recipe.
        saved_targets = config.target_modules
        saved_modules = set(config.modules_to_save or [])
        expected_modules = set(modules_to_save or [])
        if isinstance(saved_targets, str) or saved_modules != expected_modules:
            raise ValueError("adapter does not match the supported LoRA recipe")
        expected_targets = set(targets)
        covered_targets = {
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and any(name == target or name.endswith(f".{target}") for target in saved_targets)
        }
        if (covered_targets != expected_targets
                or config.r != 16 or config.lora_alpha != 32 or config.lora_dropout != 0
                or config.bias != "none" or config.use_dora or config.use_rslora
                or config.rank_pattern or config.alpha_pattern or config.layers_to_transform
                or config.layers_pattern or config.exclude_modules or config.target_parameters):
            raise ValueError("adapter does not match the supported LoRA recipe")
    model.requires_grad_(False)
    if adapter_path is None:
        model = get_peft_model(model, config)
    else:
        # Cast before loading so BF16 base modules do not round saved FP32 updates.
        for module in modules_to_save:
            model.get_submodule(module).float()
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True, autocast_adapter_dtype=True)
    trainable = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            is_adapter = ".lora_A." in name or ".lora_B." in name
            is_saved_module = any(f".{module}.modules_to_save." in name for module in modules_to_save)
            if not is_adapter and not is_saved_module:
                raise ValueError(f"unexpected trainable parameter: {name}")
            parameter.data = parameter.data.float()
            trainable.append(parameter)
    if not trainable or any(
        not any(f".{module}.modules_to_save." in name and parameter.requires_grad
                for name, parameter in model.named_parameters())
        for module in modules_to_save
    ):
        raise ValueError("LoRA did not create the requested trainable parameters")
    return model


def _board_crop_image(image: Any) -> Any:
    from PIL import Image

    return (
        image.crop(PERCEPTION_CROP_BOX)
        .resize(PERCEPTION_CROP_SIZE, Image.Resampling.LANCZOS)
        .resize(PERCEPTION_MODEL_SIZE, Image.Resampling.LANCZOS)
        .convert("RGB")
    )


def _messages(row: dict[str, Any], image: Any) -> list[dict[str, Any]]:
    images = image if isinstance(image, (list, tuple)) else (image,)
    messages = []
    for turn in row["conversations"]:
        content = turn["content"]
        if turn["role"] == "user":
            content = [{"type": "image", "image": value} for value in images]
            content.append({"type": "text", "text": turn["content"]})
        messages.append({"role": turn["role"], "content": content})
    return messages


def _prepare(
    processor: Any,
    row: dict[str, Any],
    *,
    family: ModelFamily,
    prompt_only: bool = False,
    image: Any | None = None,
    multiview: bool = False,
) -> tuple[Any, int]:
    image = image if image is not None else _read_image(Path(row["image"]).as_uri(), 30)
    if multiview:
        image = (image, _board_crop_image(image))
    messages = _messages(row, image)
    if prompt_only:
        messages = messages[:-1]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=prompt_only,
        return_dict=True,
        return_tensors="pt",
        **family.chat_template_kwargs,
    )
    input_length = int(batch.input_ids.shape[-1])
    return batch, input_length


def _masked_batch(
    processor: Any,
    row: dict[str, Any],
    torch: Any,
    *,
    family: ModelFamily,
    multiview: bool = False,
) -> tuple[Any, Any]:
    batch, _ = _prepare(processor, row, family=family, multiview=multiview)
    prompt, prompt_length = _prepare(processor, row, family=family, prompt_only=True, multiview=multiview)
    if not torch.equal(batch.input_ids[:, :prompt_length], prompt.input_ids):
        # Loss masking assumes the chat template renders the prompt identically with and without the answer.
        raise ValueError(f"{family.name} chat template: the prompt is not a prefix of the full conversation")
    labels = batch.input_ids.clone()
    labels[:, :prompt_length] = -100
    if getattr(processor, "tokenizer", None) is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    if "tgt_sizes" in batch and "target_sizes" not in batch:
        batch["target_sizes"] = batch.pop("tgt_sizes")
    return batch, labels


def _collate_sft(
    samples: list[tuple[Any, Any]], processor: Any, torch: Any, *, family: ModelFamily,
) -> tuple[dict[str, Any], Any]:
    """Collate a family's verified processor schema, preserving row/image order."""
    image_keys, text_keys = family.image_concat, family.text_keys
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("SFT batching requires a tokenizer pad_token_id")
    for batch, labels in samples:
        if set(batch) != {*text_keys, *image_keys}:
            raise ValueError(f"unsupported {family.name} SFT processor keys: {sorted(batch)}")
        length = batch["input_ids"].shape[-1]
        if any(batch[key].shape != (1, length) for key in text_keys) or labels.shape != (1, length):
            raise ValueError("SFT collation expects individually prepared token sequences")
    length = max(labels.shape[-1] for _, labels in samples)

    def pad(value: Any, fill: int) -> Any:
        return torch.nn.functional.pad(value, (0, length - value.shape[-1]), value=fill)

    batch = {key: torch.cat([pad(sample[key], pad_token_id if key == "input_ids" else 0)
                             for sample, _ in samples], dim=0) for key in text_keys}
    for key, dim in image_keys.items():
        batch[key] = torch.cat([sample[key] for sample, _ in samples], dim=dim)
    labels = torch.cat([pad(labels, -100) for _, labels in samples], dim=0)
    labels[batch["attention_mask"] == 0] = -100
    return batch, labels


def _cpu_copy(value: Any, torch: Any) -> Any:
    """Own the cached CPU tensors; processor/model mutations cannot change them."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item, torch) for item in value)
    return copy.deepcopy(value)


def _move(batch: Any, device: Any) -> Any:
    return batch.to(device) if hasattr(batch, "to") else {key: value.to(device) for key, value in batch.items()}


def _perception_feature_views(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    torch: Any,
    device: Any,
    *,
    image: Any | None = None,
    multiview: bool = False,
    required_views: set[str] | None = None,
) -> dict[str, Any]:
    # Perception heads read MiniCPM vision-encoder internals; train_perception rejects other families.
    if not multiview:
        if image is None:
            batch, _ = _prepare(processor, row, family=MINICPM_V_4_7, prompt_only=True)
        else:
            batch, _ = _prepare(processor, row, family=MINICPM_V_4_7, prompt_only=True, image=image)
        return {"full": _visual_features(model, _move(batch, device), torch, spatial=True).float()}
    full_image = image if image is not None else _read_image(Path(row["image"]).as_uri(), 30)
    images = {"full": full_image, "board-crop": _board_crop_image(full_image)}
    return {
        view: _visual_features(
            model, _move(_prepare(processor, row, family=MINICPM_V_4_7, prompt_only=True, image=view_image)[0], device),
            torch, spatial=True
        ).float()
        for view, view_image in images.items()
        if required_views is None or view in required_views
    }


def _perception_features(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    torch: Any,
    device: Any,
    *,
    image: Any | None = None,
    multiview: bool = False,
) -> Any:
    views = _perception_feature_views(model, processor, row, torch, device, image=image, multiview=multiview)
    return torch.cat(list(views.values()), dim=-1) if multiview else views["full"]


def _perception_head_features(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    torch: Any,
    device: Any,
    head_names: Iterable[str],
    *,
    image: Any | None = None,
    multiview: bool = False,
    task_views: bool = False,
) -> dict[str, Any]:
    if task_views:
        head_names = tuple(head_names)
        views = _perception_feature_views(
            model, processor, row, torch, device, image=image, multiview=True,
            required_views={PERCEPTION_HEAD_VIEWS[name] for name in head_names},
        )
        return {name: views[PERCEPTION_HEAD_VIEWS[name]] for name in head_names}
    feature = _perception_features(model, processor, row, torch, device, image=image, multiview=multiview)
    return {name: feature for name in head_names}


def _structured_targets(row: dict[str, Any]) -> tuple[int, int, int, list[float]]:
    try:
        walls, player, box, goal = parse_board(row["board"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetValidationError("perception rows require a valid six-by-eight board") from exc
    wall_target = [float((row_index, column) in walls)
                   for row_index in range(6) for column in range(8)]
    return (player[0] * 8 + player[1], box[0] * 8 + box[1],
            goal[0] * 8 + goal[1], wall_target)


def _perception_targets(row: dict[str, Any]) -> tuple[int, int]:
    player, box, _, _ = _structured_targets(row)
    return player, box


def _direction_target(row: dict[str, Any]) -> int:
    direction = row.get("direction")
    if direction not in DIRECTIONS:
        raise DatasetValidationError("direction perception rows require a valid direction label")
    return DIRECTIONS.index(direction)


def _direction_targets(row: dict[str, Any]) -> tuple[int, ...]:
    directions = row.get("optimal_directions")
    if directions is None:
        return (_direction_target(row),)
    if (not isinstance(directions, list) or not directions
            or any(direction not in DIRECTIONS for direction in directions)):
        raise DatasetValidationError("direction perception rows require valid optimal direction labels")
    teacher = _direction_target(row)
    targets = tuple(dict.fromkeys(DIRECTIONS.index(direction) for direction in directions))
    if teacher not in targets:
        raise DatasetValidationError("teacher direction must be one of the optimal directions")
    return targets


def _visual_kwargs(method: Callable[..., Any], batch: Any) -> dict[str, Any]:
    values = {
        "pixel_values": batch.get("pixel_values"),
        "image_bound": batch.get("image_bound"),
        "tgt_sizes": batch.get("tgt_sizes", batch.get("target_sizes")),
        "target_sizes": batch.get("target_sizes", batch.get("tgt_sizes")),
        "image_grid_thw": batch.get("image_grid_thw"),
    }
    parameters = inspect.signature(method).parameters
    if any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()):
        return {key: value for key, value in values.items() if value is not None}
    return {key: value for key, value in values.items() if key in parameters and value is not None}


def _pool_visual_tensor(value: Any, torch: Any, batch_size: int) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        raise RuntimeError("visual encoder must return tensors")
    if value.ndim == 1:
        return value.unsqueeze(0)
    if value.ndim == 2:
        if value.shape[0] == batch_size:
            return value
        return value.mean(dim=0, keepdim=True)
    if value.shape[0] != batch_size:
        value = value.reshape(1, -1, value.shape[-1])
    else:
        value = value.reshape(batch_size, -1, value.shape[-1])
    return value.mean(dim=1)


def _pool_visual(value: Any, torch: Any, batch_size: int) -> Any:
    if hasattr(value, "last_hidden_state"):
        value = value.last_hidden_state
    elif hasattr(value, "pooler_output"):
        value = value.pooler_output
    if isinstance(value, list):
        if batch_size == 1:
            pooled = [_pool_visual(item, torch, 1) for item in value]
            return torch.cat(pooled, dim=0).mean(dim=0, keepdim=True)
        if all(isinstance(item, torch.Tensor) for item in value):
            return torch.cat([_pool_visual_tensor(item, torch, 1) for item in value], dim=0)
        return _pool_visual(value[0], torch, batch_size)
    if isinstance(value, tuple):
        return _pool_visual(value[0], torch, batch_size)
    return _pool_visual_tensor(value, torch, batch_size)


def _pool_spatial_tensor(value: Any, torch: Any, tokens: int) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
        raise RuntimeError("spatial visual encoder output must be a 2D or 3D tensor")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    pooled = torch.nn.functional.adaptive_avg_pool1d(value.transpose(1, 2), tokens)
    return pooled.transpose(1, 2).reshape(value.shape[0], -1)


def _pool_spatial(value: Any, torch: Any, batch_size: int, tokens: int = 16) -> Any:
    if hasattr(value, "pooler_output") and isinstance(value.pooler_output, list):
        value = value.pooler_output
    elif hasattr(value, "last_hidden_state"):
        value = value.last_hidden_state
    elif hasattr(value, "pooler_output"):
        value = value.pooler_output
    if isinstance(value, list):
        if batch_size != 1 or not value:
            raise RuntimeError("spatial visual pooling requires one image with non-empty slices")
        return torch.cat([_pool_spatial_tensor(item, torch, tokens) for item in value], dim=-1)
    if isinstance(value, tuple):
        return _pool_spatial(value[0], torch, batch_size, tokens)
    return _pool_spatial_tensor(value, torch, tokens)


def _visual_features(model: Any, batch: Any, torch: Any, *, spatial: bool = False) -> Any:
    owners = [model, getattr(model, "module", None), getattr(model, "model", None)]
    for owner in owners:
        if owner is None:
            continue
        for name in ("get_vision_embedding", "get_image_features"):
            method = getattr(owner, name, None)
            if callable(method):
                features = method(**_visual_kwargs(method, batch))
                batch_size = int(batch["input_ids"].shape[0]) if "input_ids" in batch else 1
                return (_pool_spatial(features, torch, batch_size)
                        if spatial else _pool_visual(features, torch, batch_size))
    for owner in owners:
        if owner is None:
            continue
        for name in ("vision_tower", "vision_model"):
            encoder = getattr(owner, name, None)
            pixel_values = batch.get("pixel_values")
            if callable(encoder) and pixel_values is not None:
                features = encoder(pixel_values, output_hidden_states=False)
                batch_size = int(batch["input_ids"].shape[0]) if "input_ids" in batch else 1
                return (_pool_spatial(features, torch, batch_size)
                        if spatial else _pool_visual(features, torch, batch_size))
    raise RuntimeError("MiniCPM model exposes no supported visual feature encoder")


def _perception_checkpoint(heads: Any, accelerator: Any, output_dir: Path, config: dict[str, Any]) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch = __import__("torch")
        torch.save(accelerator.unwrap_model(heads).state_dict(), output_dir / "perception_head.pt")
        (output_dir / "perception_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if config.get("direction") and config.get("visual_views") == "task-routed":
            (output_dir / "policy_config.json").write_text(json.dumps({
                "output_format": "action-v1",
                "max_tokens": 1,
                "inference_mode": "perception_direction",
                "base_model": config["base_model"],
            }, indent=2), encoding="utf-8")


def _direction_head(torch: Any, feature_dim: int, mlp: bool) -> Any:
    if not mlp:
        return torch.nn.Linear(feature_dim, DIRECTION_CLASSES)
    return torch.nn.Sequential(
        torch.nn.Linear(feature_dim, DIRECTION_HIDDEN),
        torch.nn.GELU(),
        torch.nn.Linear(DIRECTION_HIDDEN, DIRECTION_CLASSES),
    )


def load_perception_head(checkpoint: str | Path, torch: Any) -> Any:
    config = json.loads((Path(checkpoint) / "perception_config.json").read_text(encoding="utf-8"))
    if config.get("classes") != PERCEPTION_CLASSES or type(config.get("feature_dim")) is not int:
        raise ValueError("invalid perception checkpoint configuration")
    feature_dims = config.get("feature_dims", {})
    if not isinstance(feature_dims, dict) or any(type(value) is not int for value in feature_dims.values()):
        raise ValueError("invalid perception feature dimensions")

    def head_dim(name: str) -> int:
        return feature_dims.get(name, config["feature_dim"])

    heads = torch.nn.ModuleDict({
        "player": torch.nn.Linear(head_dim("player"), PERCEPTION_CLASSES),
        "box": torch.nn.Linear(head_dim("box"), PERCEPTION_CLASSES),
    })
    if config.get("structured_state", False):
        heads["goal"] = torch.nn.Linear(head_dim("goal"), PERCEPTION_CLASSES)
        heads["wall"] = torch.nn.Linear(head_dim("wall"), PERCEPTION_CLASSES)
    if config.get("direction", False):
        heads["direction"] = _direction_head(torch, head_dim("direction"), config.get("direction_mlp", False))
    try:
        state = torch.load(Path(checkpoint) / "perception_head.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(Path(checkpoint) / "perception_head.pt", map_location="cpu")
    heads.load_state_dict(state)
    heads.eval()
    return heads


def _checkpoint(model: Any, processor: Any, accelerator: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        processor.save_pretrained(output_dir)


def _save_epoch_model(model: Any, processor: Any, accelerator: Any, target: Path,
                      policy_config: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Publish an intermediate model; the rename keeps a partial save from looking complete."""
    pending = target.with_name(f".{target.name}.incomplete")
    if accelerator.is_main_process and pending.exists():
        shutil.rmtree(pending)
    _checkpoint(model, processor, accelerator, pending)
    if accelerator.is_main_process:
        (pending / "policy_config.json").write_text(json.dumps(policy_config, indent=2), encoding="utf-8")
        (pending / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        pending.rename(target)
    accelerator.wait_for_everyone()


def _validate_grpo_rows(rows: list[dict[str, Any]], group_size: int, require_old_logprob: bool = True) -> None:
    group_counts = defaultdict(int)
    for row in rows:
        group_counts[str(row["group_id"])] += 1
        if "old_logprob" not in row:
            if not require_old_logprob:
                continue
            raise DatasetValidationError("policy rows require old_logprob from rollout collection")
        try:
            float(row["old_logprob"])
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError("policy old_logprob must be numeric") from exc
    # Precomputed episode-level advantages already encode the group baseline.
    if not all("advantage" in row for row in rows) and any(count < group_size for count in group_counts.values()):
        raise DatasetValidationError("each policy group must contain group_size rollouts")


def train_perception(
    base_model: str,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    learning_rate: float = 1e-3,
    epochs: int = 10,
    gradient_accumulation: int = 1,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
    seed: int = 20261117,
    direction: bool = False,
    direction_mlp: bool = False,
    structured_state: bool = False,
    multiview: bool = False,
    task_views: bool = False,
) -> dict[str, Any]:
    """Train frozen-vision heads for board state and, optionally, direction."""
    if not rows or epochs < 1 or gradient_accumulation < 1 or learning_rate <= 0:
        raise ValueError("perception training requires rows, positive learning rate, epochs and gradient_accumulation")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if direction_mlp and not direction:
        raise ValueError("direction_mlp requires direction=True")
    if task_views:
        multiview = True
    structured_targets = [_structured_targets(row) for row in rows] if structured_state else None
    targets = [target[:2] for target in structured_targets] if structured_targets is not None else [
        _perception_targets(row) for row in rows
    ]
    direction_targets = [_direction_targets(row) for row in rows] if direction else None
    direction_labels = [_direction_target(row) for row in rows] if direction else None
    for row in rows:
        _validate_protocol(row, row.get("output_format", "action-v1"))
    if _adapter_base_model(base_model) is not None:
        raise ValueError("perception training requires a base model, not an adapter checkpoint")

    torch, Accelerator, AutoModelForImageTextToText, AutoProcessor = _imports()
    from accelerate.utils import set_seed

    set_seed(seed)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation, mixed_precision="bf16")
    processor = AutoProcessor.from_pretrained(base_model)
    model = AutoModelForImageTextToText.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    if model_family(model.config) is not MINICPM_V_4_7:
        raise ValueError("perception diagnostics support MiniCPM-V 4.7 only")
    model.requires_grad_(False)
    model.eval()
    model.to(accelerator.device)
    head_names = ["player", "box"]
    if structured_state:
        head_names.extend(("goal", "wall"))
    if direction:
        head_names.append("direction")
    with torch.no_grad():
        cached_features = [
            {
                name: feature.detach().cpu()
                for name, feature in _perception_head_features(
                    model, processor, row, torch, accelerator.device, head_names,
                    multiview=multiview, task_views=task_views,
                ).items()
            }
            for row in rows
        ]
    feature_dims = {name: int(feature.shape[-1]) for name, feature in cached_features[0].items()}
    feature_dim = feature_dims["player"]
    heads = torch.nn.ModuleDict({
        "player": torch.nn.Linear(feature_dims["player"], PERCEPTION_CLASSES),
        "box": torch.nn.Linear(feature_dims["box"], PERCEPTION_CLASSES),
    })
    if structured_state:
        heads["goal"] = torch.nn.Linear(feature_dims["goal"], PERCEPTION_CLASSES)
        heads["wall"] = torch.nn.Linear(feature_dims["wall"], PERCEPTION_CLASSES)
    if direction:
        heads["direction"] = _direction_head(torch, feature_dims["direction"], direction_mlp)
    heads = heads.to(accelerator.device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(range(len(rows)), batch_size=1, shuffle=False)
    del model, processor
    heads, optimizer, dataloader = accelerator.prepare(heads, optimizer, dataloader)

    started = time.monotonic()
    losses: list[float] = []
    updates = 0
    remainder = len(dataloader) % gradient_accumulation
    for _ in range(epochs):
        for batch_index, indices in enumerate(dataloader):
            index = int(indices[0])
            player_target, box_target = (torch.tensor(value, device=accelerator.device) for value in targets[index])
            goal_target = (
                torch.tensor(structured_targets[index][2], device=accelerator.device)
                if structured_targets is not None else None
            )
            wall_target = (
                torch.tensor(structured_targets[index][3], dtype=torch.float32, device=accelerator.device)
                if structured_targets is not None else None
            )
            direction_target = None
            if direction_targets is not None:
                direction_target = torch.zeros(len(DIRECTIONS), device=accelerator.device)
                direction_target[list(direction_targets[index])] = 1.0
            with accelerator.accumulate(heads):
                features = {
                    name: value.to(accelerator.device)
                    for name, value in cached_features[index].items()
                }
                logits_player = heads["player"](features["player"])
                logits_box = heads["box"](features["box"])
                loss = torch.nn.functional.cross_entropy(logits_player, player_target.unsqueeze(0))
                loss = loss + torch.nn.functional.cross_entropy(logits_box, box_target.unsqueeze(0))
                if goal_target is not None and wall_target is not None:
                    loss = loss + torch.nn.functional.cross_entropy(
                        heads["goal"](features["goal"]), goal_target.unsqueeze(0)
                    )
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                        heads["wall"](features["wall"]), wall_target.unsqueeze(0)
                    )
                if direction_target is not None:
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                        heads["direction"](features["direction"]), direction_target.unsqueeze(0)
                    )
                loss_scale = gradient_accumulation / remainder if remainder and batch_index >= len(dataloader) - remainder else 1
                accelerator.backward(loss * loss_scale)
                grad_norm = (
                    float(accelerator.clip_grad_norm_(heads.parameters(), float("inf")))
                    if accelerator.sync_gradients
                    else None
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    updates += 1
            losses.append(float(loss.detach().cpu()))
            if on_step is not None:
                values = {"train/loss": losses[-1], "train/learning_rate": learning_rate}
                if grad_norm is not None and math.isfinite(grad_norm):
                    values["train/grad_norm"] = grad_norm
                on_step(len(losses), values)

    heads.eval()
    predictions = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            features = {
                name: value.to(accelerator.device)
                for name, value in cached_features[index].items()
            }
            player_prediction = int(heads["player"](features["player"]).argmax(dim=-1).item())
            box_prediction = int(heads["box"](features["box"]).argmax(dim=-1).item())
            player_target, box_target = targets[index]
            state_prediction = None
            if structured_targets is not None:
                goal_prediction = int(heads["goal"](features["goal"]).argmax(dim=-1).item())
                wall_prediction = (heads["wall"](features["wall"]).squeeze(0) >= 0).to(torch.int64).tolist()
                goal_target = structured_targets[index][2]
                wall_target = [int(value) for value in structured_targets[index][3]]
                state_prediction = {
                    "goal_target": goal_target, "goal_prediction": goal_prediction,
                    "goal_correct": goal_prediction == goal_target,
                    "wall_target": wall_target, "wall_prediction": wall_prediction,
                    "wall_correct_cells": sum(predicted == target for predicted, target in zip(wall_prediction, wall_target)),
                    "wall_exact_correct": wall_prediction == wall_target,
                }
            prediction = {
                "id": row["id"], "player_target": player_target, "player_prediction": player_prediction,
                "box_target": box_target, "box_prediction": box_prediction,
                "player_correct": player_prediction == player_target,
                "box_correct": box_prediction == box_target,
            }
            if state_prediction is not None:
                prediction.update(state_prediction)
            if direction_targets is not None:
                direction_prediction = int(heads["direction"](features["direction"]).argmax(dim=-1).item())
                prediction.update({
                    "direction_target": direction_labels[index],
                    "direction_optimal_targets": list(direction_targets[index]),
                    "direction_prediction": direction_prediction,
                    "direction_label": DIRECTIONS[direction_labels[index]],
                    "direction_prediction_label": DIRECTIONS[direction_prediction],
                    "direction_correct": direction_prediction in direction_targets[index],
                })
            predictions.append(prediction)
    player_accuracy = sum(item["player_correct"] for item in predictions) / len(predictions)
    box_accuracy = sum(item["box_correct"] for item in predictions) / len(predictions)
    model_metadata = {
        "implementation": (
            "frozen-vision-structured-state+direction-mlp" if structured_state and direction_mlp
            else "frozen-vision-structured-state+direction" if structured_state and direction
            else "frozen-vision-structured-state" if structured_state
            else "frozen-vision-linear-head+direction-mlp" if direction_mlp
            else "frozen-vision-linear-head+direction" if direction
            else "frozen-vision-linear-head"
        ),
        "base_model": base_model,
        "feature_dim": feature_dim,
        "feature_pooling": "16-token adaptive average per visual slice",
        "visual_views": "task-routed" if task_views else "full+board-crop" if multiview else "full",
        "feature_dims": feature_dims if task_views else {},
        "head_views": {name: PERCEPTION_HEAD_VIEWS[name] for name in head_names} if task_views else {},
        "classes": PERCEPTION_CLASSES,
    }
    metrics = {
        **model_metadata,
        "algorithm": "perception",
        "loss": sum(losses) / len(losses),
        "sample_count": len(rows),
        "step_count": len(losses),
        "update_count": updates,
        "player_accuracy": player_accuracy,
        "box_accuracy": box_accuracy,
        "both_correct": sum(item["player_correct"] and item["box_correct"] for item in predictions),
        "both_accuracy": sum(item["player_correct"] and item["box_correct"] for item in predictions) / len(predictions),
        "elapsed_seconds": time.monotonic() - started,
        "seed": seed,
    }
    if structured_targets is not None:
        metrics["goal_accuracy"] = sum(item["goal_correct"] for item in predictions) / len(predictions)
        metrics["goal_correct"] = sum(item["goal_correct"] for item in predictions)
        metrics["wall_exact_accuracy"] = sum(item["wall_exact_correct"] for item in predictions) / len(predictions)
        metrics["wall_exact_correct"] = sum(item["wall_exact_correct"] for item in predictions)
        metrics["wall_cell_accuracy"] = sum(item["wall_correct_cells"] for item in predictions) / (48 * len(predictions))
        metrics["structured_state"] = True
    if direction_targets is not None:
        confusion = {target: {prediction: 0 for prediction in DIRECTIONS} for target in DIRECTIONS}
        for item in predictions:
            confusion[item["direction_label"]][item["direction_prediction_label"]] += 1
        metrics["direction_accuracy"] = sum(item["direction_correct"] for item in predictions) / len(predictions)
        metrics["direction_correct"] = sum(item["direction_correct"] for item in predictions)
        metrics["direction_classes"] = list(DIRECTIONS)
        metrics["direction_confusion"] = confusion
        metrics["direction_mlp"] = direction_mlp
        metrics["direction_target_mode"] = "optimal-multihot"
    config = {
        **model_metadata,
        "structured_state": structured_state,
        "direction": direction,
        "direction_mlp": direction_mlp,
        "direction_classes": list(DIRECTIONS) if direction else [],
        "direction_target_mode": "optimal-multihot" if direction else None,
        "seed": seed,
    }
    output_dir = Path(output_dir)
    _perception_checkpoint(heads, accelerator, output_dir, config)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "perception_predictions.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in predictions), encoding="utf-8"
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _sft_dataset_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())
    # Include image contents: a reused filename must not hide changed training data.
    for image in sorted({row["image"] for row in rows}):
        with Path(image).open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def _sft_checkpoints(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.glob("epoch-*")
         if path.is_dir() and path.name.removeprefix("epoch-").isdigit()),
        key=lambda path: int(path.name.removeprefix("epoch-")),
    )


def _save_sft_epoch(accelerator: Any, directory: Path, state: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / f"epoch-{state['completed_epochs']:08d}"
    pending = directory / f".{completed.name}.incomplete"
    # ponytail: one writer per checkpoint_dir; add locking if concurrent jobs share it.
    if pending.exists():
        shutil.rmtree(pending)
    accelerator.save_state(str(pending))
    (pending / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
    pending.rename(completed)  # Same-filesystem rename publishes only a finished state.
    for obsolete in _sft_checkpoints(directory)[:-2]:
        shutil.rmtree(obsolete)


def train(
    algorithm: str,
    base_model: str,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    learning_rate: float = 1e-5,
    epochs: int = 1,
    gradient_accumulation: int = 1,
    clip_epsilon: float = 0.2,
    clip_epsilon_high: float | None = None,
    group_size: int = 2,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
    compute_old_logprob: bool = False,
    finetuning_method: str = "full",
    output_format: str = "action-v1",
    seed: int = 20261117,
    multiview: bool = False,
    train_merger_mlp: bool = False,
    train_top_layers: int = 0,
    top_layer_learning_rate: float = 1e-6,
    checkpoint_dir: str | Path | None = None,
    batch_size: int = 1,
    cache_preprocessed: bool = False,
    gradient_checkpointing: bool = True,
    shuffle: bool = False,
    save_every: int | None = None,
    sampling_temperature: float | None = None,
) -> dict[str, Any]:
    """Train and save a checkpoint.

    ``sampling_temperature`` is the temperature online rollouts were sampled at; policy log-probabilities
    (old and current) are computed from logits divided by it, so the ratio is over the sampling
    distribution. None or 0 (greedy sampling) leaves logits unscaled. ``train/entropy`` stays at T=1.

    ``save_every`` also publishes the model after every N-th epoch (before the last) as
    ``<output_dir parent>/epoch-<completed epochs>``, with the same policy_config as the final one.

    ``checkpoint_dir`` enables single-process SFT epoch state saves and automatic resume.
    ``epochs`` is the total target, including completed epochs; it may be increased on resume.
    Keep the base model/processor immutable and use one writer per checkpoint directory.
    ``elapsed_seconds`` measures training/export in this invocation, excluding model/state loading;
    it does not include time spent in earlier invocations of a resumed run.

    ``train_top_layers`` saves/trains complete final language layers via PEFT at
    ``top_layer_learning_rate``; remaining attention layers retain LoRA at ``learning_rate``.
    This requires SFT + LoRA and at least one remaining language layer for LoRA.

    ``batch_size`` is the SFT microbatch size; effective batch size is this times
    ``gradient_accumulation``. Losses are token means per sample, then sample means
    per accumulation window. ``cache_preprocessed`` keeps deterministic processor
    outputs on CPU. ``training_seconds`` includes transfers/forward/backward/updates,
    excluding preprocessing, callbacks and checkpoint saves in this invocation.
    ``shuffle`` reorders SFT rows every epoch with a permutation seeded by ``seed`` and
    the epoch number, so a resumed run replays the same order.

    ``compute_old_logprob`` fills missing policy ``old_logprob`` values with the loaded base policy
    before any update, which matches on-policy rollouts sampled from that same checkpoint.
    Policy rows may carry ``loss_weight`` (set by ``training_agent.step_advantages``): the algorithm's
    per-step normalization (GRPO 1/|o_i|, DAPO and GiGPO 1/sum|o_i|, Dr. GRPO 1/max_steps). Weights are
    rescaled to mean 1 over the rows, so relative weighting is exact while the update size per optimizer
    step stays comparable across algorithms; the constant is absorbed by the learning rate. Rows without
    it weigh 1. ``clip_epsilon_high`` (DAPO clip-higher) bounds the ratio at 1 + clip_epsilon_high;
    None keeps the symmetric clip.
    Policy algorithms ignore ``multiview`` and use the base checkpoint's ``visual_views``.
    """
    if algorithm != "sft" and algorithm not in POLICY_ALGORITHMS:
        raise ValueError(f"algorithm must be sft or one of {sorted(POLICY_ALGORITHMS)}")
    if clip_epsilon_high is not None and (isinstance(clip_epsilon_high, bool) or not isinstance(clip_epsilon_high, (int, float))
                                          or not 0 < clip_epsilon_high <= 1):
        raise ValueError("clip_epsilon_high must be in (0, 1]")
    if checkpoint_dir is not None and algorithm != "sft":
        raise ValueError("checkpoint_dir is supported only for SFT")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not all(isinstance(value, bool) for value in (cache_preprocessed, gradient_checkpointing, shuffle)):
        raise ValueError("cache_preprocessed, gradient_checkpointing and shuffle must be booleans")
    if algorithm != "sft" and shuffle:
        raise ValueError("shuffle is supported only for SFT")
    if finetuning_method not in ("full", "lora"):
        raise ValueError("finetuning_method must be full or lora")
    if train_merger_mlp and finetuning_method != "lora":
        raise ValueError("train_merger_mlp requires finetuning_method='lora'")
    if not isinstance(train_top_layers, int) or isinstance(train_top_layers, bool) or train_top_layers < 0:
        raise ValueError("train_top_layers must be a non-negative integer")
    if train_top_layers:
        if algorithm != "sft" or finetuning_method != "lora":
            raise ValueError("train_top_layers requires SFT with finetuning_method='lora'")
        if (not isinstance(top_layer_learning_rate, (int, float)) or isinstance(top_layer_learning_rate, bool)
                or not math.isfinite(top_layer_learning_rate) or top_layer_learning_rate <= 0):
            raise ValueError("top_layer_learning_rate must be finite and positive")
    if output_format not in POLICY_MAX_TOKENS:
        raise ValueError("output_format must be action-v1 or sokoban-plan-v1")
    if algorithm in POLICY_ALGORITHMS and output_format != "action-v1":
        raise ValueError("structured output training is supported only for SFT")
    adapter_base = _adapter_base_model(base_model)
    if adapter_base is not None and finetuning_method != "lora":
        raise ValueError("an existing adapter requires finetuning_method='lora'; full training cannot merge adapters")
    if algorithm in POLICY_ALGORITHMS:
        # Train the policy that was sampled: the model service picks views from the checkpoint's policy_config.
        policy_path = Path(base_model) / "policy_config.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.is_file() else {}
        multiview = policy.get("visual_views", "full") == "full+board-crop"
    if not rows or epochs < 1 or gradient_accumulation < 1:
        raise ValueError("training requires rows, positive epochs and gradient_accumulation")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if save_every is not None and (not isinstance(save_every, int) or isinstance(save_every, bool) or save_every < 1):
        raise ValueError("save_every must be a positive integer")
    if sampling_temperature is not None and (isinstance(sampling_temperature, bool)
                                             or not isinstance(sampling_temperature, (int, float))
                                             or not math.isfinite(sampling_temperature) or sampling_temperature < 0):
        raise ValueError("sampling_temperature must be finite and non-negative")
    logit_scale = float(sampling_temperature) if sampling_temperature else 1.0
    for row in rows:
        _validate_protocol(row, output_format)
        if row.get("invalid_output") is True and algorithm not in POLICY_ALGORITHMS:
            raise DatasetValidationError("invalid_output rows are only trainable by policy algorithms")
    if algorithm in POLICY_ALGORITHMS:
        if group_size < 2:
            raise ValueError(f"{algorithm.upper()} requires group_size >= 2")
        _validate_grpo_rows(rows, group_size, require_old_logprob=not compute_old_logprob)
    top_layer_options = ({"train_top_layers": train_top_layers, "top_layer_learning_rate": top_layer_learning_rate}
                         if train_top_layers else {})
    policy_config = {
        "output_format": output_format,
        "max_tokens": POLICY_MAX_TOKENS[output_format],
        "finetuning_method": finetuning_method,
        "train_merger_mlp": train_merger_mlp,
        **top_layer_options,
        "visual_views": "full+board-crop" if multiview else "full",
        "seed": seed,
    }
    resume_path = None
    completed_epochs = 0
    losses: list[float] = []
    updates = 0
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        recipe = {
            "version": 1, "base_model": str(base_model), "adapter_base": adapter_base,
            "learning_rate": learning_rate, "gradient_accumulation": gradient_accumulation,
            "finetuning_method": finetuning_method, "output_format": output_format,
            "seed": seed, "multiview": multiview, "train_merger_mlp": train_merger_mlp,
            "mixed_precision": "bf16", "dataset_sha256": _sft_dataset_digest(rows),
            "batch_size": batch_size, "cache_preprocessed": cache_preprocessed,
            "gradient_checkpointing": gradient_checkpointing, "shuffle": shuffle,
            **top_layer_options,
        }
        checkpoints = _sft_checkpoints(checkpoint_dir)
        if checkpoints:
            resume_path = checkpoints[-1]
            state = json.loads((resume_path / "training_state.json").read_text(encoding="utf-8"))
            saved_recipe = {"batch_size": 1, "cache_preprocessed": False, "gradient_checkpointing": True,
                            "shuffle": False, **state["recipe"]}
            if saved_recipe != recipe:
                raise ValueError("SFT checkpoint recipe or dataset does not match this run")
            completed_epochs = state["completed_epochs"]
            losses, updates = state["losses"], state["updates"]
            if (completed_epochs != int(resume_path.name.removeprefix("epoch-"))
                    or not 1 <= completed_epochs <= epochs
                    or len(losses) != completed_epochs * len(rows)
                    or not 0 <= updates <= completed_epochs * math.ceil(len(rows) / (batch_size * gradient_accumulation))):
                raise ValueError("SFT checkpoint progress is invalid or exceeds requested epochs")
    torch, Accelerator, AutoModelForImageTextToText, AutoProcessor = _imports()
    from accelerate.utils import set_seed

    set_seed(seed)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation, mixed_precision="bf16")
    if checkpoint_dir is not None and accelerator.num_processes != 1:
        # ponytail: rank-local progress needs per-rank metadata before enabling distributed resume.
        raise ValueError("SFT checkpoint_dir currently requires a single Accelerate process")
    if batch_size > 1 and accelerator.num_processes != 1:
        # ponytail: distributed tail weighting needs rank-local sample counts before enabling microbatches.
        raise ValueError("batch_size > 1 currently requires a single Accelerate process")
    top_modules = []
    if train_top_layers:
        from transformers import AutoConfig

        top_modules = _top_layer_modules(AutoConfig.from_pretrained(adapter_base or base_model), train_top_layers)
    processor = AutoProcessor.from_pretrained(base_model)
    model = prepare_model(
        AutoModelForImageTextToText.from_pretrained(adapter_base or base_model, torch_dtype=torch.bfloat16))
    family = model_family(model.config)
    if finetuning_method == "lora":
        model = _lora_model(
            model, torch, base_model if adapter_base is not None else None,
            train_merger_mlp=train_merger_mlp, train_top_layers=train_top_layers,
        )
    model.train()
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if finetuning_method == "lora":
            model.enable_input_require_grads()
    elif not gradient_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    parameters = (p for p in model.parameters() if p.requires_grad)
    if top_modules:
        top_parameters, other_parameters = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                group = (top_parameters if any(f".{module}.modules_to_save." in name for module in top_modules)
                         else other_parameters)
                group.append(parameter)
        parameters = [{"params": other_parameters}, {"params": top_parameters, "lr": top_layer_learning_rate}]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(range(len(rows)), batch_size=batch_size, shuffle=False)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if resume_path is not None:
        accelerator.load_state(str(resume_path))
        print(json.dumps({"event": "sft_resume", "checkpoint": str(resume_path),
                          "completed_epochs": completed_epochs, "step": len(losses),
                          "total": epochs * len(rows), "updates": updates}), flush=True)

    preprocessing_seconds = 0.0
    cached_batches: dict[int, tuple[Any, Any]] = {}

    def masked_batch(row: dict[str, Any]) -> tuple[Any, Any]:
        nonlocal preprocessing_seconds
        before = time.monotonic()
        result = _masked_batch(processor, row, torch, family=family, multiview=multiview)
        preprocessing_seconds += time.monotonic() - before
        return result

    def sample_batch(index: int) -> tuple[Any, Any]:
        if not cache_preprocessed:
            return masked_batch(rows[index])
        if index not in cached_batches:
            cached_batches[index] = _cpu_copy(masked_batch(rows[index]), torch)
        # BatchFeature.to() and model code may mutate their inputs, even on CPU.
        return copy.deepcopy(cached_batches[index])

    started = time.monotonic()
    initial_steps = len(losses)
    training_seconds = 0.0
    on_cuda = getattr(accelerator.device, "type", None) == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    advantages: list[float] = []
    loss_weights: list[float] = []
    entropies: list[float] = []
    if algorithm in POLICY_ALGORITHMS:
        # Multi-step rollouts carry a per-step advantage; single-step rows are grouped here.
        if all("advantage" in row for row in rows):
            advantages = [float(row["advantage"]) for row in rows]
        else:
            advantages = group_advantages(rows)
        raw_weights = [float(row.get("loss_weight", 1.0)) for row in rows]
        if not all(math.isfinite(weight) and weight > 0 for weight in raw_weights):
            raise DatasetValidationError("policy loss_weight must be finite and positive")
        loss_weights = [weight * len(raw_weights) / sum(raw_weights) for weight in raw_weights]
    upper_clip = 1 + (clip_epsilon if clip_epsilon_high is None else clip_epsilon_high)
    if algorithm in POLICY_ALGORITHMS and compute_old_logprob:
        model.eval()
        with torch.no_grad():
            for index, row in enumerate(rows):
                if "old_logprob" in row:
                    continue
                batch, labels = sample_batch(index)
                batch = _move(batch, accelerator.device)
                with accelerator.autocast():
                    logits = model(**batch, use_cache=False).logits
                row["old_logprob"] = float(_sample_logprobs(torch, logits / logit_scale, labels.to(accelerator.device))[0])
        model.train()
    remainder = len(dataloader) % gradient_accumulation
    for epoch in range(completed_epochs, epochs):
        order = list(range(len(rows)))
        if shuffle:
            random.Random(f"{seed}:{epoch}").shuffle(order)
        for batch_index, indices in enumerate(dataloader):
            indices = [order[int(index)] for index in indices]
            samples = [sample_batch(index) for index in indices]
            batch, labels = samples[0] if len(samples) == 1 else _collate_sft(samples, processor, torch, family=family)
            before_training = time.monotonic()
            batch = _move(batch, accelerator.device)
            labels = labels.to(accelerator.device)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch, use_cache=False)
                    if algorithm == "sft":
                        if len(indices) == 1:
                            # Preserve the original batch-one reduction/numerical path.
                            shift_logits = outputs.logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            loss = torch.nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1), ignore_index=-100,
                            )
                            sample_losses = loss.unsqueeze(0)
                        else:
                            sample_losses = _sft_sample_losses(torch, outputs.logits, labels)
                            loss = sample_losses.mean()
                    else:
                        current_logprob = _sample_logprobs(torch, outputs.logits / logit_scale, labels)
                        with torch.no_grad():  # Monitoring only; the objective has no entropy term.
                            completion = labels[..., 1:] != -100
                            sample_entropies = [
                                float(token_entropies(torch, outputs.logits[sample, :-1, :][completion[sample]]).mean())
                                for sample in range(len(indices))]
                        per_sample = lambda values: torch.tensor(values, device=accelerator.device, dtype=torch.float32)
                        old_logprob = per_sample([float(rows[i]["old_logprob"]) for i in indices])
                        ratio = torch.exp(current_logprob - old_logprob)
                        advantage = per_sample([advantages[i] for i in indices])
                        sample_losses = -per_sample([loss_weights[i] for i in indices]) * torch.minimum(
                            ratio * advantage, ratio.clamp(1 - clip_epsilon, upper_clip) * advantage)
                        loss = sample_losses.mean()
                # Accelerate divides by the configured microbatch count. Weight by actual
                # samples in this window, including a short microbatch and/or final window.
                if batch_size == 1:
                    loss_scale = gradient_accumulation / remainder if remainder and batch_index >= len(dataloader) - remainder else 1
                else:
                    window_start = (batch_index // gradient_accumulation) * gradient_accumulation * batch_size
                    window_samples = min(batch_size * gradient_accumulation, len(rows) - window_start)
                    loss_scale = gradient_accumulation * len(indices) / window_samples
                accelerator.backward(loss * loss_scale)
                grad_norm = (
                    float(accelerator.clip_grad_norm_(model.parameters(), float("inf")))
                    if accelerator.sync_gradients
                    else None
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    updates += 1
            batch_losses = sample_losses.detach().cpu().tolist()
            training_seconds += time.monotonic() - before_training
            for sample_index, sample_loss in enumerate(batch_losses):
                losses.append(sample_loss)
                if on_step is not None:
                    step_metrics = {"train/loss": sample_loss, "train/learning_rate": learning_rate}
                    if train_top_layers:
                        step_metrics["train/top_layer_learning_rate"] = top_layer_learning_rate
                    if sample_index == len(batch_losses) - 1 and grad_norm is not None and math.isfinite(grad_norm):
                        step_metrics["train/grad_norm"] = grad_norm
                    if algorithm in POLICY_ALGORITHMS:
                        sample = indices[sample_index]
                        ratio_value = float(ratio[sample_index].detach().cpu())
                        entropy = sample_entropies[sample_index]
                        step_metrics["train/ratio"] = ratio_value
                        step_metrics["train/advantage"] = advantages[sample]
                        step_metrics["train/clipped"] = float(not 1 - clip_epsilon <= ratio_value <= upper_clip)
                        step_metrics["train/loss_weight"] = loss_weights[sample]
                        if math.isfinite(entropy):
                            step_metrics["train/entropy"] = entropy
                            entropies.append(entropy)
                    on_step(len(losses), step_metrics)
                if checkpoint_dir is not None and len(losses) % 50 == 0:
                    print(json.dumps({"event": "sft_progress", "step": len(losses), "total": epochs * len(rows),
                                      "loss": sample_loss, "updates": updates}), flush=True)
        # Before the resume state: a pod lost mid-save redoes this epoch and saves the model again.
        if save_every and (epoch + 1) % save_every == 0 and epoch + 1 < epochs:
            _save_epoch_model(model, processor, accelerator, Path(output_dir).parent / f"epoch-{epoch + 1}",
                              policy_config, {
                                  "algorithm": algorithm, "epoch": epoch + 1, "loss": sum(losses) / len(losses),
                                  "sample_count": len(rows), "step_count": len(losses), "update_count": updates,
                                  "base_model": base_model, **policy_config,
                              })
        if checkpoint_dir is not None:
            _save_sft_epoch(accelerator, checkpoint_dir, {
                "recipe": recipe, "completed_epochs": epoch + 1, "losses": losses, "updates": updates,
            })
            print(json.dumps({"event": "sft_checkpoint", "completed_epochs": epoch + 1,
                              "step": len(losses), "total": epochs * len(rows), "updates": updates}), flush=True)
    _checkpoint(model, processor, accelerator, Path(output_dir))
    metrics = {
        "algorithm": algorithm,
        "loss": sum(losses) / len(losses),
        "sample_count": len(rows),
        "update_count": updates,
        "step_count": len(losses),
        "reward_mean": sum(float(row["reward"]["total"]) for row in rows) / len(rows),
        "action_format_rate": 1.0,
        "invalid_action_rate": 0.0,
        "elapsed_seconds": time.monotonic() - started,
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds": training_seconds,
        "samples_per_second": (len(losses) - initial_steps) / training_seconds if training_seconds else 0.0,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(accelerator.device)) if on_cuda else 0,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "cache_preprocessed": cache_preprocessed,
        "gradient_checkpointing": gradient_checkpointing,
        "shuffle": shuffle,
        "base_model": base_model,
        "finetuning_method": finetuning_method,
        "train_merger_mlp": train_merger_mlp,
        **top_layer_options,
        "output_format": output_format,
        "visual_views": "full+board-crop" if multiview else "full",
        "seed": seed,
        "group_size": group_size if algorithm in POLICY_ALGORITHMS else 0,
        **({"entropy_mean": sum(entropies) / len(entropies)} if entropies else {}),
        "implementation": (
            "sft-cross-entropy"
            if algorithm == "sft"
            else f"{algorithm}-clipped-policy"
        ),
    }
    if checkpoint_dir is not None:
        metrics["completed_epochs"] = epochs
    if accelerator.is_main_process:
        Path(output_dir, "policy_config.json").write_text(json.dumps(policy_config, indent=2), encoding="utf-8")
        Path(output_dir, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _sft_sample_losses(torch: Any, logits: Any, labels: Any) -> Any:
    """Average assistant-token loss independently for each sample, ignoring padding."""
    shifted_labels = labels[..., 1:].contiguous()
    counts = (shifted_labels != -100).sum(dim=-1)
    if (counts == 0).any():
        raise DatasetValidationError("each SFT sample must contain an unmasked assistant token")
    token_losses = torch.nn.functional.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
        shifted_labels.view(-1), ignore_index=-100, reduction="none",
    ).view_as(shifted_labels)
    return token_losses.sum(dim=-1) / counts


def _sample_logprobs(torch: Any, logits: Any, labels: Any) -> Any:
    """Per-sample sum of completion-token log-probabilities, shape (batch,); masked/padded tokens add 0.

    Only completion positions go through the fp32 log_softmax: over the whole ~1300-token prompt that
    tensor is ~1.3 GB per row (248k vocabulary) and autograd keeps it, which exhausted 80 GB at batch 4.
    """
    mask = labels[..., 1:] != -100
    selected = logits[..., :-1, :][mask].float()
    token_log_probs = torch.log_softmax(selected, dim=-1).gather(-1, labels[..., 1:][mask].unsqueeze(-1)).squeeze(-1)
    rows = mask.nonzero()[:, 0]
    return torch.zeros(mask.shape[0], device=logits.device, dtype=token_log_probs.dtype).index_add(0, rows, token_log_probs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("sft", *sorted(POLICY_ALGORITHMS), "perception"), required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", help="SFT only: save/resume epoch state; keep the last two epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="SFT microbatch size")
    parser.add_argument("--cache-preprocessed", action="store_true", help="SFT: cache processor outputs on CPU")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--shuffle", action="store_true", help="SFT: reshuffle rows every epoch (seeded)")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--finetuning-method", choices=("full", "lora"), default="full")
    parser.add_argument("--output-format", choices=tuple(POLICY_MAX_TOKENS), default="action-v1")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--direction", action="store_true", help="also train a 4-class direction head")
    parser.add_argument("--direction-mlp", action="store_true", help="use a small MLP for the direction head")
    parser.add_argument("--structured-state", action="store_true", help="also train goal and wall-layout heads")
    parser.add_argument("--multiview", action="store_true", help="combine the full screenshot with a board crop")
    parser.add_argument("--train-merger-mlp", action="store_true", help="fully train the vision-language merger alongside language LoRA")
    parser.add_argument("--train-top-layers", type=int, default=0, help="SFT + LoRA: fully train the final N language layers")
    parser.add_argument("--top-layer-learning-rate", type=float, default=1e-6)
    parser.add_argument("--task-views", action="store_true", help="route state heads to full or board-crop views")
    args = parser.parse_args()
    if args.checkpoint_dir is not None and args.algorithm != "sft":
        parser.error("--checkpoint-dir is supported only for SFT")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.algorithm != "sft" and args.shuffle:
        parser.error("--shuffle is supported only for SFT")
    if args.algorithm == "perception" and (args.batch_size > 1 or args.cache_preprocessed):
        parser.error("--batch-size > 1 and --cache-preprocessed are not supported for perception")
    if args.train_top_layers < 0:
        parser.error("--train-top-layers must be non-negative")
    if args.train_top_layers and (args.algorithm != "sft" or args.finetuning_method != "lora"):
        parser.error("--train-top-layers requires SFT with --finetuning-method lora")
    rows = load_dataset(args.dataset)
    if args.algorithm == "perception":
        metrics = train_perception(
            args.base_model,
            rows,
            args.output_dir,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            gradient_accumulation=args.gradient_accumulation,
            seed=args.seed,
            direction=args.direction or args.direction_mlp,
            direction_mlp=args.direction_mlp,
            structured_state=args.structured_state,
            multiview=args.multiview,
            task_views=args.task_views,
        )
    else:
        metrics = train(
            args.algorithm,
            args.base_model,
            rows,
            args.output_dir,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            gradient_accumulation=args.gradient_accumulation,
            group_size=args.group_size,
            finetuning_method=args.finetuning_method,
            output_format=args.output_format,
            seed=args.seed,
            multiview=args.multiview,
            train_merger_mlp=args.train_merger_mlp,
            train_top_layers=args.train_top_layers,
            top_layer_learning_rate=args.top_layer_learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            cache_preprocessed=args.cache_preprocessed,
            gradient_checkpointing=args.gradient_checkpointing,
            shuffle=args.shuffle,
        )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
