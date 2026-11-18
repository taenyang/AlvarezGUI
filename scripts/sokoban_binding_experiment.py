"""Bounded, reproducible direction-head ablations and strict service evaluation.

Run with the project environment and PYTHONPATH=src. Inputs are existing paired
Sokoban datasets; labels are used only for training and offline scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from alvarezgui.minicpm_service import MiniCPMService, load_pretrained
from alvarezgui.minicpm_training import (
    DIRECTIONS, _direction_targets, _perception_head_features, load_dataset,
    load_perception_head,
)
from alvarezgui.model import InferenceRequest, InvalidActionOutputError
from alvarezgui.node_agent.android_tasks import SOKOBAN_INSTRUCTION
from alvarezgui.policy_output import parse_policy_output
from alvarezgui.sokoban import ACTIONS


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def evaluate(checkpoint, dataset, output):
    rows = load_dataset(dataset)
    service = MiniCPMService.load_checkpoint(checkpoint, "candidate")
    records = []
    for row in rows:
        start = time.monotonic()
        raw, direction, error = "", None, None
        try:
            response = service.infer(InferenceRequest(
                request_id=row["id"], model_version="candidate", instruction=SOKOBAN_INSTRUCTION,
                image_uri=Path(row["image"]).as_uri(), temperature=0, timeout_ms=120000,
            ))
        except InvalidActionOutputError as exc:
            raw = getattr(exc, "raw_output", None) or raw
            error = str(exc)
        else:
            raw = response.raw_output
            try:
                action = parse_policy_output(raw, response.output_format)
            except (ValueError, TypeError) as exc:
                error = str(exc)
            else:
                direction = next((name for name, tap in ACTIONS.items() if action == tap), None)
        records.append({
            "id": row["id"], "raw_output": raw, "direction": direction,
            "valid": error is None, "allowed": direction is not None, "error": error,
            "correct": direction is not None and DIRECTIONS.index(direction) in _direction_targets(row),
            "latency_ms": (time.monotonic() - start) * 1000,
        })
    result = {
        "checkpoint": checkpoint, "dataset": str(dataset), "temperature": 0,
        "total": len(rows), "correct": sum(r["correct"] for r in records),
        "valid": sum(r["valid"] for r in records), "allowed": sum(r["allowed"] for r in records),
        "latency_ms_mean": sum(r["latency_ms"] for r in records) / len(records),
        "records": records,
    }
    save(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    service.unload_all()


def sweep(args):
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    specs = [
        {"name": "bce-lr1e-3-ordered", "lr": 1e-3, "shuffle": False, "loss": "bce"},
        {"name": "bce-lr1e-4-ordered", "lr": 1e-4, "shuffle": False, "loss": "bce"},
        {"name": "bce-lr1e-4-shuffle", "lr": 1e-4, "shuffle": True, "loss": "bce"},
        {"name": "ce-lr1e-4-shuffle", "lr": 1e-4, "shuffle": True, "loss": "ce"},
        {"name": "set-ce-lr1e-4-shuffle", "lr": 1e-4, "shuffle": True, "loss": "set-ce"},
    ]
    plan = {"runs": specs, "epochs": [30, 100], "seed": args.seed, "batch_size": 4,
            "selection": "maximum validation correct; ties prefer earlier listed run/epoch",
            "note": "Direction-only AdamW; shared initialization. No test-set selection."}
    save(root / "plan.json", plan)
    print(json.dumps(plan), flush=True)
    rows = {split: load_dataset(args.dataset_dir / f"{split}.jsonl") for split in ("train", "validation")}
    config = json.loads((args.template / "perception_config.json").read_text())
    model, processor, _ = load_pretrained(config["base_model"])
    features = {}
    start = time.monotonic()
    with torch.inference_mode():
        for split, samples in rows.items():
            values = []
            for index, row in enumerate(samples):
                value = _perception_head_features(
                    model, processor, row, torch, model.device, ("direction",),
                    multiview=True, task_views=True,
                )["direction"]
                values.append(value.float().cpu())
                if (index + 1) % 50 == 0:
                    print(f"features {split}: {index + 1}/{len(samples)}", flush=True)
            features[split] = torch.cat(values)
    torch.save(features, root / "features.pt")
    save(root / "features.json", {
        "seconds": time.monotonic() - start, "base_model": config["base_model"],
        "dataset_sha256": {s: hashlib.sha256((args.dataset_dir / f"{s}.jsonl").read_bytes()).hexdigest() for s in rows},
        "ids": {s: [r["id"] for r in samples] for s, samples in rows.items()},
        "shape": {s: list(x.shape) for s, x in features.items()},
    })
    del model, processor
    torch.cuda.empty_cache()
    features = {s: x.to("cuda") for s, x in features.items()}
    masks = {}
    for split, samples in rows.items():
        masks[split] = torch.zeros(len(samples), 4, device="cuda")
        for i, row in enumerate(samples):
            masks[split][i, list(_direction_targets(row))] = 1
    labels = torch.tensor([DIRECTIONS.index(r["direction"]) for r in rows["train"]], device="cuda")

    def score(head, split):
        with torch.no_grad():
            pred = head(features[split]).argmax(-1)
            correct = masks[split].gather(1, pred[:, None]).flatten().bool()
        return {"total": len(pred), "correct": int(correct.sum()), "accuracy": float(correct.float().mean()),
                "records": [{"id": row["id"], "direction": DIRECTIONS[p], "correct": bool(c)}
                            for row, p, c in zip(rows[split], pred.tolist(), correct.tolist())]}

    template = load_perception_head(args.template, torch)
    original = {s: score(template["direction"].to("cuda"), s) for s in rows}
    save(root / "original.json", original)
    template.cpu()
    torch.manual_seed(args.seed)
    initial = torch.nn.Linear(features["train"].shape[1], 4).state_dict()
    results, best = [], None
    for spec in specs:
        print("Run: " + json.dumps(spec) + "; compare matched training budgets", flush=True)
        head = torch.nn.Linear(features["train"].shape[1], 4).cuda()
        head.load_state_dict(initial)
        optimizer = torch.optim.AdamW(head.parameters(), lr=spec["lr"])
        generator = torch.Generator().manual_seed(args.seed)
        for epoch in range(1, 101):
            order = torch.randperm(len(labels), generator=generator) if spec["shuffle"] else torch.arange(len(labels))
            for indices in order.split(4):
                logits = head(features["train"][indices])
                if spec["loss"] == "bce":
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks["train"][indices])
                elif spec["loss"] == "ce":
                    loss = torch.nn.functional.cross_entropy(logits, labels[indices])
                else:
                    allowed = masks["train"][indices].bool()
                    loss = (torch.logsumexp(logits, -1) - torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), -1)).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss in {spec['name']}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if epoch not in (30, 100):
                continue
            scores = {s: score(head, s) for s in rows}
            name = f"{spec['name']}-e{epoch}"
            checkpoint = root / name
            checkpoint.mkdir()
            template["direction"].load_state_dict(head.state_dict())
            torch.save(template.state_dict(), checkpoint / "perception_head.pt")
            save(checkpoint / "perception_config.json", dict(config, direction_target_mode=spec["loss"], seed=args.seed))
            save(checkpoint / "policy_config.json", json.loads((args.template / "policy_config.json").read_text()))
            save(checkpoint / "scores.json", scores)
            result = dict(spec, epoch=epoch, checkpoint=str(checkpoint),
                          **{s: {k: v for k, v in v.items() if k != "records"} for s, v in scores.items()})
            results.append(result)
            save(root / "results.json", results)
            print(json.dumps(result), flush=True)
            if best is None or result["validation"]["correct"] > best["validation"]["correct"]:
                best = result
    # Keep the prior candidate if none of the bounded comparisons improves it.
    if original["validation"]["correct"] >= best["validation"]["correct"]:
        best = {"checkpoint": str(args.template), "name": "original-e30",
                **{s: {k: v for k, v in v.items() if k != "records"} for s, v in original.items()}}
    save(root / "selected.json", best)
    print("Selected on validation: " + json.dumps(best), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    train = sub.add_parser("sweep")
    train.add_argument("--dataset-dir", type=Path, required=True)
    train.add_argument("--template", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, default=20261117)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--dataset", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "sweep":
        sweep(args)
    else:
        evaluate(args.checkpoint, args.dataset, args.output)
