from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from alvarezgui.action import validate_action
from alvarezgui.minicpm_service import _read_image


class DatasetValidationError(ValueError):
    pass


def _json_action(value: str) -> dict[str, Any]:
    try:
        action = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError("assistant completion must be JSON") from exc
    try:
        return validate_action(action)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError("assistant completion is not a safe action") from exc


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
    conversations = row["conversations"]
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise DatasetValidationError(f"line {line_number}: conversations must contain user and assistant")
    if not all(isinstance(turn, dict) for turn in conversations):
        raise DatasetValidationError(f"line {line_number}: conversation turns must be objects")
    if conversations[-2].get("role") != "user" or conversations[-1].get("role") != "assistant":
        raise DatasetValidationError(f"line {line_number}: final turns must be user then assistant")
    completion = conversations[-1].get("content")
    if not isinstance(completion, str) or not completion.strip():
        raise DatasetValidationError(f"line {line_number}: assistant completion must be non-empty")
    _json_action(completion)
    reward = row["reward"]
    if not isinstance(reward, dict):
        raise DatasetValidationError(f"line {line_number}: reward must be an object")
    try:
        float(reward["total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetValidationError(f"line {line_number}: reward.total must be numeric") from exc
    if not isinstance(row.get("conversations"), list):
        raise DatasetValidationError(f"line {line_number}: conversations must be a list")
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


def _messages(row: dict[str, Any], image: Any) -> list[dict[str, Any]]:
    messages = []
    for turn in row["conversations"]:
        content = turn["content"]
        if turn["role"] == "user":
            content = [{"type": "image", "image": image}, {"type": "text", "text": content}]
        messages.append({"role": turn["role"], "content": content})
    return messages


def _prepare(processor: Any, row: dict[str, Any], *, prompt_only: bool = False) -> tuple[Any, int]:
    image = _read_image(Path(row["image"]).as_uri(), 30)
    messages = _messages(row, image)
    if prompt_only:
        messages = messages[:-1]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=prompt_only,
        return_dict=True,
        return_tensors="pt",
        downsample_mode="16x",
        max_slice_nums=36,
        enable_thinking=False,
    )
    input_length = int(batch.input_ids.shape[-1])
    return batch, input_length


def _masked_batch(processor: Any, row: dict[str, Any], torch: Any) -> tuple[Any, Any]:
    batch, _ = _prepare(processor, row)
    _, prompt_length = _prepare(processor, row, prompt_only=True)
    labels = batch.input_ids.clone()
    labels[:, :prompt_length] = -100
    if getattr(processor, "tokenizer", None) is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    if "tgt_sizes" in batch and "target_sizes" not in batch:
        batch["target_sizes"] = batch.pop("tgt_sizes")
    return batch, labels


def _move(batch: Any, device: Any) -> Any:
    return batch.to(device) if hasattr(batch, "to") else {key: value.to(device) for key, value in batch.items()}


def _checkpoint(model: Any, processor: Any, accelerator: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        processor.save_pretrained(output_dir)


def _validate_grpo_rows(rows: list[dict[str, Any]], group_size: int, require_old_logprob: bool = True) -> None:
    group_counts = defaultdict(int)
    for row in rows:
        group_counts[str(row["group_id"])] += 1
        if "old_logprob" not in row:
            if not require_old_logprob:
                continue
            raise DatasetValidationError("GRPO rows require old_logprob from rollout collection")
        try:
            float(row["old_logprob"])
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError("GRPO old_logprob must be numeric") from exc
    # Precomputed episode-level advantages already encode the group baseline.
    if not all("advantage" in row for row in rows) and any(count < group_size for count in group_counts.values()):
        raise DatasetValidationError("each GRPO group must contain group_size rollouts")


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
    group_size: int = 2,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
    compute_old_logprob: bool = False,
) -> dict[str, Any]:
    """Train and save a checkpoint.

    ``compute_old_logprob`` fills missing GRPO ``old_logprob`` values with the loaded base policy
    before any update, which matches on-policy rollouts sampled from that same checkpoint.
    """
    if algorithm not in {"sft", "grpo"}:
        raise ValueError("algorithm must be sft or grpo")
    if algorithm == "grpo":
        if group_size < 2:
            raise ValueError("GRPO requires group_size >= 2")
        _validate_grpo_rows(rows, group_size, require_old_logprob=not compute_old_logprob)
    torch, Accelerator, AutoModelForImageTextToText, AutoProcessor = _imports()
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation, mixed_precision="bf16")
    processor = AutoProcessor.from_pretrained(base_model)
    model = AutoModelForImageTextToText.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)
    started = time.monotonic()
    losses: list[float] = []
    updates = 0
    advantages: list[float] = []
    if algorithm == "grpo":
        # Multi-step rollouts carry an episode-level advantage; single-step rows are grouped here.
        if all("advantage" in row for row in rows):
            advantages = [float(row["advantage"]) for row in rows]
        else:
            advantages = group_advantages(rows)
    if algorithm == "grpo" and compute_old_logprob:
        model.eval()
        with torch.no_grad():
            for row in rows:
                if "old_logprob" in row:
                    continue
                batch, labels = _masked_batch(processor, row, torch)
                batch = _move(batch, accelerator.device)
                with accelerator.autocast():
                    logits = model(**batch, use_cache=False).logits
                row["old_logprob"] = float(_sequence_logprob(torch, logits, labels.to(accelerator.device)))
        model.train()
    for _ in range(epochs):
        for index, row in enumerate(rows):
            batch, labels = _masked_batch(processor, row, torch)
            batch = _move(batch, accelerator.device)
            labels = labels.to(accelerator.device)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch, use_cache=False)
                    if algorithm == "sft":
                        shift_logits = outputs.logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )
                    else:
                        current_logprob = _sequence_logprob(torch, outputs.logits, labels)
                        old_logprob = torch.as_tensor(float(row["old_logprob"]), device=accelerator.device)
                        ratio = torch.exp(current_logprob - old_logprob)
                        advantage = torch.as_tensor(advantages[index], device=accelerator.device)
                        loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
                accelerator.backward(loss)
                grad_norm = (
                    float(accelerator.clip_grad_norm_(model.parameters(), float("inf")))
                    if accelerator.sync_gradients
                    else None
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            losses.append(float(loss.detach().cpu()))
            if on_step is not None:
                step_metrics = {"train/loss": losses[-1], "train/learning_rate": learning_rate}
                if grad_norm is not None and math.isfinite(grad_norm):
                    step_metrics["train/grad_norm"] = grad_norm
                if algorithm == "grpo":
                    ratio_value = float(ratio.detach().cpu())
                    step_metrics["train/ratio"] = ratio_value
                    step_metrics["train/advantage"] = advantages[index]
                    step_metrics["train/clipped"] = float(abs(ratio_value - 1) > clip_epsilon)
                on_step(updates, step_metrics)
    _checkpoint(model, processor, accelerator, Path(output_dir))
    metrics = {
        "algorithm": algorithm,
        "loss": sum(losses) / len(losses),
        "sample_count": len(rows),
        "update_count": updates,
        "reward_mean": sum(float(row["reward"]["total"]) for row in rows) / len(rows),
        "action_format_rate": 1.0,
        "invalid_action_rate": 0.0,
        "elapsed_seconds": time.monotonic() - started,
        "base_model": base_model,
        "group_size": group_size if algorithm == "grpo" else 0,
    }
    if accelerator.is_main_process:
        Path(output_dir, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _sequence_logprob(torch: Any, logits: Any, labels: Any) -> Any:
    """Sum of completion-token log-probabilities; prompt tokens are masked with -100."""
    mask = labels[..., 1:] != -100
    log_probs = torch.log_softmax(logits[..., :-1, :].float(), dim=-1)
    token_log_probs = log_probs.gather(-1, labels[..., 1:].clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return token_log_probs.masked_select(mask).sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("sft", "grpo"), required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
    args = parser.parse_args()
    rows = load_dataset(args.dataset)
    metrics = train(
        args.algorithm,
        args.base_model,
        rows,
        args.output_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        gradient_accumulation=args.gradient_accumulation,
        group_size=args.group_size,
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
