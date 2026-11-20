"""Real-checkpoint check for a new ``alvarezgui.model_families`` entry; run on a GPU host with the
project source on PYTHONPATH, e.g. on H800:
``python scripts/probe_model_family.py --model <dir> --dataset <action-v1 train.jsonl> --tmp <dir> --output report.json``

1. LoRA (+ merger) targets and trainable parameter counts on the real module tree.
2. SFT microbatch of 2 rows equals the mean of the two single-row losses (collation is lossless).
3. A few SFT steps on real rows lower their loss; the checkpoint reloads in the model service.
4. Base and trained model generate an action for a real screenshot.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from alvarezgui.minicpm_service import MiniCPMService
from alvarezgui.minicpm_training import (
    _collate_sft, _language_attention_targets, _lora_model, _masked_batch, _move, _sft_sample_losses,
    load_dataset, train,
)
from alvarezgui.model import InferenceRequest
from alvarezgui.model_families import model_family
from alvarezgui.node_agent.android_tasks import SOKOBAN_INSTRUCTION


def sample_losses(model, batch, labels):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**_move(batch, model.device), use_cache=False).logits
    return _sft_sample_losses(torch, logits.float(), labels.to(model.device))


def main(args):
    report = {"model": args.model}
    rows = load_dataset(args.dataset)[: args.rows]
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    family = model_family(model.config)
    report["family"] = family.name
    targets = _language_attention_targets(model, torch)
    report["lora_targets"] = len(targets)
    report["target_examples"] = targets[:4]
    lora = _lora_model(model, torch, train_merger_mlp=True)
    report["trainable_parameters"] = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    lora.eval()
    for multiview in (False, True):
        samples = [_masked_batch(processor, row, torch, family=family, multiview=multiview) for row in rows[:2]]
        report[f"processor_keys_multiview={multiview}"] = {
            key: list(value.shape) for key, value in samples[0][0].items()}
        single = torch.cat([sample_losses(lora, *sample) for sample in samples])
        batched = sample_losses(lora, *_collate_sft(samples, processor, torch, family=family))
        report[f"batch_parity_multiview={multiview}"] = {
            "single": single.tolist(), "batched": batched.tolist(),
            "max_abs_diff": float((single - batched).abs().max())}
    del lora, model
    torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory(dir=args.tmp) as directory:
        output = Path(directory, "trained")
        started, losses = time.monotonic(), []
        metrics = train("sft", args.model, rows, output, learning_rate=1e-4, epochs=args.epochs,
                        finetuning_method="lora", train_merger_mlp=True, multiview=True,
                        batch_size=2, cache_preprocessed=True, gradient_checkpointing=True,
                        on_step=lambda step, values: losses.append(values["train/loss"]))
        report["train"] = {key: metrics[key] for key in ("loss", "update_count", "step_count", "peak_gpu_memory_bytes")}
        report["train"]["seconds"] = time.monotonic() - started
        report["train"]["epoch_mean_losses"] = [sum(losses[i:i + len(rows)]) / len(rows)
                                                for i in range(0, len(losses), len(rows))]
        service = MiniCPMService(lambda version: args.model if version == "base" else str(output), max_loaded=1)
        image = Path(rows[0]["image"]).as_uri()
        for version in ("base", "trained"):
            request = InferenceRequest(request_id=version, model_version=version, instruction=SOKOBAN_INSTRUCTION,
                                       image_uri=image, temperature=0, timeout_ms=120000)
            try:
                response = service.infer(request)
                report[f"infer_{version}"] = {"action": response.action, "raw_output": response.raw_output,
                                              "latency_ms": response.latency_ms}
            except Exception as exc:  # The base model may answer in prose; record it as evidence.
                report[f"infer_{version}"] = {"error": str(exc)[:400], "raw_output": getattr(exc, "raw_output", None)}
        report["teacher_action"] = json.loads(rows[0]["conversations"][-1]["content"])
        service.unload_all()
    print(json.dumps(report, indent=2))
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tmp", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
