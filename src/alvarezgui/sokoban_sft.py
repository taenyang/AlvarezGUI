"""Collect checked Android demonstrations and evaluate held-out Sokoban states."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any
from urllib.parse import unquote, urlsplit
import uuid

from alvarezgui.policy_output import OUTPUT_FORMATS, make_plan, parse_policy_output, policy_instruction
from alvarezgui.minicpm_training import load_dataset
from alvarezgui.node_agent.android_tasks import SOKOBAN_INSTRUCTION, sokoban_action_allowed
from alvarezgui.sokoban import ACTIONS, LEVEL_COUNTS, load_levels, parse_board, render_board, solve, transition


def observed_board(observation: dict[str, Any]) -> list[str]:
    prefix = "Sokoban board: "
    boards = {text[len(prefix):] for text in observation.get("ui_text", []) if text.startswith(prefix)}
    if len(boards) != 1:
        raise ValueError("observation must expose exactly one Sokoban board")
    board = boards.pop().replace("_", " ").split("|")
    parse_board(board)
    return board


def optimal_directions(board: list[str]) -> list[str]:
    """Accept every shortest-path first move, including ties in the teacher."""
    distance = len(solve(board))
    if not distance:
        return []
    walls, player, box, goal = parse_board(board)
    result = []
    for direction in ACTIONS:
        next_player, next_box = transition(walls, player, box, direction)
        if (next_player, next_box) == (player, box):
            continue
        try:
            remaining = solve(render_board(walls, next_player, next_box, goal))
        except ValueError:
            continue
        if len(remaining) == distance - 1:
            result.append(direction)
    return result


def collect_level(runtime: Any, sandbox_id: str, level: dict, output: Path) -> list[dict]:
    level_id = level["id"]
    image_dir = output / "images" / level_id
    image_dir.mkdir(parents=True)
    runtime.initialize_task(sandbox_id, "sokoban", f"android-sokoban-v1:{level_id}", SOKOBAN_INSTRUCTION)
    walls, player, box, goal = parse_board(level["board"])
    rows = []
    observation = runtime.observe(sandbox_id)
    for step_id, direction in enumerate(solve(level["board"])):
        board = render_board(walls, player, box, goal)
        (image_dir / "latest-observation.json").write_text(json.dumps(observation, indent=2))
        if observed_board(observation) != board:
            raise ValueError(f"{level_id} step {step_id}: Android board differs from teacher")
        if (observation["screen_width"], observation["screen_height"]) != (720, 1280):
            raise ValueError("Sokoban fixture collection requires the calibrated 720x1280 device layout")
        source = urlsplit(observation["image_uri"])
        if source.scheme != "file" or source.netloc:
            raise ValueError("collector requires local runtime screenshots")
        image = image_dir / f"{step_id:03d}.png"
        shutil.copyfile(unquote(source.path), image)
        action = dict(ACTIONS[direction])
        result = runtime.execute(sandbox_id, action)
        player, box = transition(walls, player, box, direction)
        after = runtime.observe(sandbox_id)
        (image_dir / "latest-observation.json").write_text(json.dumps(after, indent=2))
        if observed_board(after) != render_board(walls, player, box, goal):
            raise ValueError(f"{level_id} step {step_id}: Android did not execute the expected move")
        solved = box == goal
        if (result["done"] != solved or result["reward"]["success"] != solved
                or result["reward"].get("invalid_action")):
            raise ValueError(f"{level_id} step {step_id}: runtime reward disagrees with board")
        if solved and "Sokoban complete" not in after.get("ui_text", []):
            raise ValueError(f"{level_id}: missing completion text")
        rows.append({
            "id": f"{output.name}:{level_id}:{step_id}",
            "episode_id": f"{output.name}:{level_id}", "group_id": level_id, "step_id": step_id,
            "image": str(image.resolve()),
            "conversations": [
                {"role": "user", "content": policy_instruction(SOKOBAN_INSTRUCTION)},
                {"role": "assistant", "content": json.dumps(action, separators=(",", ":"))},
            ],
            "reward": result["reward"], "done": result["done"],
            "level_id": level_id, "split": level["split"], "board": board,
            "direction": direction, "optimal_directions": optimal_directions(board),
            "state_hash": hashlib.sha256("\n".join(board).encode()).hexdigest(),
            "screen_hash": hashlib.sha256(image.read_bytes()).hexdigest(),
        })
        observation = after
    final_source = urlsplit(observation["image_uri"])
    shutil.copyfile(unquote(final_source.path), image_dir / "complete.png")
    (output / "episodes" / f"{level_id}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps({"collected": level_id, "samples": len(rows)}), flush=True)
    return rows


def collect(settings: Any, levels_file: Path, output: Path, workers: int, seed: int,
            level_ids: list[str] | None = None) -> dict:
    from alvarezgui.node_agent.factory import runtime_from_settings

    levels = load_levels(levels_file)
    if level_ids:
        unknown = set(level_ids) - {level["id"] for level in levels}
        if unknown:
            raise ValueError(f"unknown levels: {sorted(unknown)}")
        levels = [level for level in levels if level["id"] in level_ids]
    if not 1 <= workers <= settings.sandbox_capacity:
        raise ValueError("workers must be between one and the configured sandbox capacity")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    shutil.copyfile(levels_file, output / "levels.json")
    report = {"status": "collecting", "runtime_backend": settings.runtime_backend,
              "seed": seed, "level_ids": [level["id"] for level in levels],
              "manifest_sha256": hashlib.sha256(levels_file.read_bytes()).hexdigest()}
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    def batch_collect(batch: list[dict]) -> list[dict]:
        sandbox_id = f"sft-{uuid.uuid4().hex}"
        runtime.create(sandbox_id)
        try:
            return [row for level in batch for row in collect_level(runtime, sandbox_id, level, output)]
        finally:
            runtime.destroy(sandbox_id)

    try:
        # Share Firecracker's locked network allocator across collection workers.
        runtime = runtime_from_settings(settings)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batches = [levels[index::workers] for index in range(min(workers, len(levels)))]
            rows = [row for result in pool.map(batch_collect, batches) for row in result]
        unique = {row["state_hash"]: row for row in rows}
        if len(unique) != len(rows):
            raise ValueError("duplicate semantic states detected; refusing to publish dataset")
        report["splits"] = {}
        for split in LEVEL_COUNTS:
            selected = sorted((row for row in rows if row["split"] == split), key=lambda row: row["id"])
            if not selected:
                continue
            if split == "train":
                random.Random(seed).shuffle(selected)
            path = output / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
            load_dataset(path)
            report["splits"][split] = {
                "samples": len(selected), "levels": len({row["level_id"] for row in selected}),
                "directions": dict(Counter(row["direction"] for row in selected)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


AUGMENT_QUOTAS = {"player_on_goal": 3, "near_deadlock": 3, "off_path": 2}


def augment_targets(level: dict, exclude: set[str], rng: random.Random) -> list[tuple[str, list[str]]]:
    """Solvable train states the teacher trajectory never shows: player on goal, one move from a deadlock, off path."""
    walls, player, box, goal = parse_board(level["board"])
    seen, pending = {(player, box)}, [(player, box)]
    while pending:
        state = pending.pop()
        if state[1] == goal:
            continue
        for direction in ACTIONS:
            following = transition(walls, *state, direction)
            if following not in seen:
                seen.add(following)
                pending.append(following)

    def solvable(state: tuple) -> bool:
        try:
            solve(render_board(walls, *state, goal))
            return True
        except ValueError:
            return False

    kinds: dict[str, list[list[str]]] = {kind: [] for kind in AUGMENT_QUOTAS}
    for state in sorted(seen):
        board = render_board(walls, *state, goal)
        if state[1] == goal or hashlib.sha256("\n".join(board).encode()).hexdigest() in exclude or not solvable(state):
            continue
        if state[0] == goal:
            kinds["player_on_goal"].append(board)
        elif any(transition(walls, *state, d)[1] != state[1] and not solvable(transition(walls, *state, d))
                 for d in ACTIONS):
            kinds["near_deadlock"].append(board)
        else:
            kinds["off_path"].append(board)
    return [(kind, board) for kind, boards in kinds.items()
            for board in rng.sample(boards, min(AUGMENT_QUOTAS[kind], len(boards)))]


def _state_path(walls: frozenset, start: tuple, target: tuple, goal: tuple) -> list[str] | None:
    parents, pending = {start: None}, [start]
    for state in pending:
        if state == target:
            path = []
            while parents[state] is not None:
                state, direction = parents[state]
                path.append(direction)
            return path[::-1]
        if state[1] == goal:
            continue
        for direction in ACTIONS:
            following = transition(walls, *state, direction)
            if following not in parents:
                parents[following] = (state, direction)
                pending.append(following)
    return None


def collect_augment_level(runtime: Any, sandbox_id: str, level: dict, targets: list[tuple[str, list[str]]],
                          output: Path) -> list[dict]:
    """Walk the real Android game to each target, screenshot it, then execute and check the teacher move."""
    level_id = level["id"]
    image_dir = output / "images" / level_id
    image_dir.mkdir(parents=True)
    walls, start_player, start_box, goal = parse_board(level["board"])
    current, rows = None, []
    for index, (kind, board) in enumerate(targets):
        _, player, box, _ = parse_board(board)
        path = _state_path(walls, current, (player, box), goal) if current else None
        if path is None:
            runtime.initialize_task(sandbox_id, "sokoban", f"android-sokoban-v1:{level_id}", SOKOBAN_INSTRUCTION)
            path = _state_path(walls, (start_player, start_box), (player, box), goal)
        for direction in path:
            runtime.execute(sandbox_id, dict(ACTIONS[direction]))
        observation = runtime.observe(sandbox_id)
        if observed_board(observation) != board:
            raise ValueError(f"{level_id} target {index}: Android board differs from target")
        if (observation["screen_width"], observation["screen_height"]) != (720, 1280):
            raise ValueError("Sokoban fixture collection requires the calibrated 720x1280 device layout")
        source = urlsplit(observation["image_uri"])
        if source.scheme != "file" or source.netloc:
            raise ValueError("collector requires local runtime screenshots")
        image = image_dir / f"{index:03d}.png"
        shutil.copyfile(unquote(source.path), image)
        direction = solve(board)[0]
        action = dict(ACTIONS[direction])
        result = runtime.execute(sandbox_id, action)
        player, box = transition(walls, player, box, direction)
        after = runtime.observe(sandbox_id)
        if observed_board(after) != render_board(walls, player, box, goal):
            raise ValueError(f"{level_id} target {index}: Android did not execute the expected move")
        solved = box == goal
        if (result["done"] != solved or result["reward"]["success"] != solved
                or result["reward"].get("invalid_action")):
            raise ValueError(f"{level_id} target {index}: runtime reward disagrees with board")
        if solved and "Sokoban complete" not in after.get("ui_text", []):
            raise ValueError(f"{level_id}: missing completion text")
        current = None if solved else (player, box)
        rows.append(_paired_row({
            "id": f"{output.name}:{level_id}:{index}", "episode_id": f"{output.name}:{level_id}",
            "group_id": level_id, "step_id": index, "image": str(image.resolve()),
            "reward": result["reward"], "done": result["done"], "level_id": level_id, "split": "train",
            "board": board, "direction": direction, "optimal_directions": optimal_directions(board),
            "augment_kind": kind, "state_hash": hashlib.sha256("\n".join(board).encode()).hexdigest(),
            "screen_hash": hashlib.sha256(image.read_bytes()).hexdigest(),
        }, "action-v1"))
    print(json.dumps({"collected": level_id, "samples": len(rows)}), flush=True)
    return rows


def collect_augment(settings: Any, levels_file: Path, base_train: Path, output: Path, workers: int,
                    seed: int) -> dict:
    """Append real-Android off-trajectory TRAIN states to a verified action-v1 TRAIN file."""
    from alvarezgui.node_agent.factory import runtime_from_settings

    base_rows = load_dataset(base_train)
    if any(row.get("split") != "train" or row.get("output_format") != "action-v1" for row in base_rows):
        raise ValueError("base dataset must be the action-v1 TRAIN split")
    exclude = {row["state_hash"] for row in base_rows}
    levels = [level for level in load_levels(levels_file) if level["split"] == "train"]
    plans = [(level, augment_targets(level, exclude, random.Random(f"{seed}:{level['id']}"))) for level in levels]
    if not 1 <= workers <= settings.sandbox_capacity:
        raise ValueError("workers must be between one and the configured sandbox capacity")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "collecting", "version": "sokoban-sft-v3-augment", "teacher_source": "solver-verified",
              "runtime_backend": settings.runtime_backend, "seed": seed, "quotas": AUGMENT_QUOTAS,
              "base_train": str(base_train), "base_sha256": _sha256(base_train),
              "manifest_sha256": _sha256(levels_file)}
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    def batch_collect(batch: list[tuple[dict, list]]) -> list[dict]:
        sandbox_id = f"sft-{uuid.uuid4().hex}"
        runtime.create(sandbox_id)
        try:
            return [row for level, targets in batch
                    for row in collect_augment_level(runtime, sandbox_id, level, targets, output)]
        finally:
            runtime.destroy(sandbox_id)

    try:
        runtime = runtime_from_settings(settings)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batches = [plans[index::workers] for index in range(min(workers, len(plans)))]
            added = [row for result in pool.map(batch_collect, batches) for row in result]
        for row in added:
            _validate_teacher_row(row, "train", provenance=True)
        rows = base_rows + added
        if len({row["state_hash"] for row in rows}) != len(rows):
            raise ValueError("duplicate semantic states detected; refusing to publish dataset")
        random.Random(seed).shuffle(rows)
        for name, selected in (("augment.jsonl", added), ("train.jsonl", rows)):
            (output / name).write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
        load_dataset(output / "train.jsonl")
        report.update(status="complete", added=len(added), samples=len(rows),
                      kinds=dict(Counter(row["augment_kind"] for row in added)),
                      directions=dict(Counter(row["direction"] for row in rows)),
                      sha256=_sha256(output / "train.jsonl"))
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_teacher_row(row: dict, split: str, *, provenance: bool = False) -> list[str]:
    if row.get("split") != split:
        raise ValueError("mixed splits or incorrect row split")
    board = row["board"]
    directions = optimal_directions(board)
    output_format = row.get("output_format", "action-v1")
    completion = row["conversations"][-1]["content"]
    action = parse_policy_output(completion, output_format)
    direction = next((key for key, value in ACTIONS.items() if action == value), None)
    if direction not in directions or row.get("direction", direction) != direction:
        raise ValueError(f"{row['id']}: incorrect teacher action or direction")
    if output_format == "sokoban-plan-v1" and json.loads(completion) != make_plan(board, direction):
        raise ValueError(f"{row['id']}: incorrect teacher plan")
    if "optimal_directions" in row and row["optimal_directions"] != directions:
        raise ValueError(f"{row['id']}: incorrect teacher optimal directions")
    if provenance:
        if row.get("direction") != direction:
            raise ValueError("missing teacher direction")
        if row.get("state_hash") != hashlib.sha256("\n".join(board).encode()).hexdigest():
            raise ValueError(f"{row['id']}: state hash changed")
        image = Path(row["image"])
        if not image.is_absolute() or not image.is_file() or row.get("screen_hash") != _sha256(image):
            raise ValueError(f"{row['id']}: image missing or screen hash changed")
        if row["conversations"] != [
            {"role": "user", "content": policy_instruction(SOKOBAN_INSTRUCTION, output_format)},
            {"role": "assistant", "content": completion},
        ]:
            raise ValueError("teacher prompt differs from serving protocol")
        if row.get("instruction", SOKOBAN_INSTRUCTION) != SOKOBAN_INSTRUCTION:
            raise ValueError("teacher instruction differs from Sokoban instruction")
    return directions


def _load_collection(source: Path) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Validate original screenshots, labels, trajectories, split membership and hashes."""
    report_path = source / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or any(
        report.get("splits", {}).get(split, {}).get("levels") != count
        for split, count in LEVEL_COUNTS.items()
    ):
        raise ValueError("requires a complete 48/8/8 collection")
    manifest_path = source / "levels.json"
    if _sha256(manifest_path) != report.get("manifest_sha256"):
        raise ValueError("level manifest hash changed")
    levels = {level["id"]: level for level in load_levels(manifest_path)}
    files = {"report.json": _sha256(report_path), "levels.json": _sha256(manifest_path)}
    datasets = {}
    ids, states, images = set(), set(), set()
    for split, count in LEVEL_COUNTS.items():
        path = source / f"{split}.jsonl"
        expected = report["splits"][split]
        files[path.name] = _sha256(path)
        if files[path.name] != expected.get("sha256"):
            raise ValueError(f"{split} dataset hash changed after collection")
        rows = load_dataset(path)
        if len(rows) != expected.get("samples") or len({row.get("level_id") for row in rows}) != count:
            raise ValueError("collection sample or level counts disagree")
        if dict(Counter(row.get("direction") for row in rows)) != expected.get("directions"):
            raise ValueError("collection direction counts disagree")
        episodes: dict[str, list[dict]] = {}
        for row in rows:
            _validate_teacher_row(row, split, provenance=True)
            if row.get("output_format", "action-v1") != "action-v1":
                raise ValueError("paired export requires the original action-v1 collection")
            level_id = row.get("level_id")
            if level_id not in levels or levels[level_id]["split"] != split or row["group_id"] != level_id:
                raise ValueError("level or group belongs to another split")
            if row["id"] in ids or row["state_hash"] in states or row["image"] in images:
                raise ValueError("duplicate ID, state, or image across collection")
            ids.add(row["id"])
            states.add(row["state_hash"])
            images.add(row["image"])
            episodes.setdefault(level_id, []).append(row)
        for level_id, episode in episodes.items():
            episode.sort(key=lambda row: row["step_id"])
            board = levels[level_id]["board"]
            if len({row["episode_id"] for row in episode}) != 1:
                raise ValueError("level spans multiple episode IDs")
            for step, row in enumerate(episode):
                if row["step_id"] != step or row["board"] != board:
                    raise ValueError("teacher trajectory is incomplete or inconsistent")
                walls, player, box, goal = parse_board(board)
                player, box = transition(walls, player, box, row["direction"])
                solved = box == goal
                if (row["done"] != solved or row["reward"].get("success") != solved
                        or row["reward"].get("invalid_action") or solved != (step == len(episode) - 1)):
                    raise ValueError("teacher trajectory completion or reward disagrees")
                board = render_board(walls, player, box, goal)
        datasets[split] = rows
    return datasets, files


def _paired_row(row: dict, output_format: str) -> dict:
    label = ACTIONS[row["direction"]] if output_format == "action-v1" else make_plan(row["board"], row["direction"])
    return {
        **row, "instruction": SOKOBAN_INSTRUCTION, "output_format": output_format,
        "teacher_source": "solver-verified",
        "conversations": [
            {"role": "user", "content": policy_instruction(SOKOBAN_INSTRUCTION, output_format)},
            {"role": "assistant", "content": json.dumps(label, separators=(",", ":"))},
        ],
    }


def _small_train_subset(rows: list[dict]) -> tuple[list[dict], Counter]:
    selected, counts = [], Counter()
    for row in rows:
        direction = row["direction"]
        if counts[direction] < 8:
            selected.append(row)
            counts[direction] += 1
    return selected, counts


def export_paired(source: Path, output: Path) -> dict:
    """Export paired protocols and a deterministic 8-per-direction TRAIN-only subset."""
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"paired export requires a new output directory: {output}")
    datasets, source_files = _load_collection(source)
    selected, counts = _small_train_subset(datasets["train"])
    if counts != Counter({direction: 8 for direction in ACTIONS}):
        raise ValueError("training split needs at least eight states per direction")
    # Preserve source order in both full and diagnostic datasets.
    datasets["train-small"] = selected
    reports = {}
    contents = {}
    for output_format in OUTPUT_FORMATS:
        report = {"status": "complete", "version": "sokoban-sft-v2", "output_format": output_format,
                  "teacher_source": "solver-verified", "source_dir": str(source),
                  "source_files": source_files, "splits": {}}
        for split, source_rows in datasets.items():
            rows = [_paired_row(row, output_format) for row in source_rows]
            for row in rows:
                parse_policy_output(row["conversations"][-1]["content"], output_format)
            text = "".join(json.dumps(row) + "\n" for row in rows)
            contents[(output_format, split)] = text
            report["splits"][split] = {
                "samples": len(rows), "levels": len({row["level_id"] for row in rows}),
                "directions": dict(Counter(row["direction"] for row in rows)),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "ids_sha256": hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest(),
                "diagnostic_only": split == "train-small",
            }
        reports[output_format] = report
    if any(_sha256(source / name) != digest for name, digest in source_files.items()):
        raise ValueError("source changed during export")
    output.mkdir(parents=True, exist_ok=False)
    for output_format, report in reports.items():
        target = output / output_format
        target.mkdir()
        for split in datasets:
            (target / f"{split}.jsonl").write_text(contents[(output_format, split)], encoding="utf-8")
        (target / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    result = {"status": "complete", "teacher_source": "solver-verified", "source_dir": str(source),
              "source_files": source_files, "formats": reports,
              "small_train_ids": [row["id"] for row in selected]}
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def registration_dataset(directory: Path, train_file: str = "train.jsonl") -> tuple[Path, list[dict]]:
    """Resolve only a fully validated full TRAIN or paired diagnostic TRAIN file."""
    directory = directory.resolve()
    if train_file not in {"train.jsonl", "train-small.jsonl"}:
        raise ValueError("registration requires train.jsonl or train-small.jsonl")
    report = json.loads((directory / "report.json").read_text())
    if report.get("version") == "sokoban-sft-v3-augment":
        if train_file != "train.jsonl" or report.get("status") != "complete" \
                or _sha256(directory / train_file) != report.get("sha256"):
            raise ValueError("augmented TRAIN file is incomplete or changed")
        rows = load_dataset(directory / train_file)
        for row in rows:
            _validate_teacher_row(row, "train", provenance=True)
        return directory / train_file, rows
    if report.get("version") != "sokoban-sft-v2":
        datasets, _ = _load_collection(directory)
        if train_file != "train.jsonl":
            raise ValueError("small training datasets require a paired export")
        return directory / train_file, datasets["train"]
    source = Path(report["source_dir"])
    datasets, source_files = _load_collection(source)
    if report.get("source_files") != source_files or report.get("status") != "complete":
        raise ValueError("paired source provenance changed")
    output_format = report.get("output_format")
    if output_format not in OUTPUT_FORMATS or report.get("teacher_source") != "solver-verified":
        raise ValueError("invalid paired protocol or teacher source")
    datasets["train-small"], counts = _small_train_subset(datasets["train"])
    if counts != Counter({direction: 8 for direction in ACTIONS}):
        raise ValueError("small training dataset is not balanced")
    if set(report.get("splits", {})) != set(datasets):
        raise ValueError("paired split manifest is incomplete")
    validated = {}
    for split, source_rows in datasets.items():
        path = directory / f"{split}.jsonl"
        if _sha256(path) != report["splits"][split].get("sha256"):
            raise ValueError(f"{split} paired dataset hash changed")
        rows = load_dataset(path)
        if rows != [_paired_row(row, output_format) for row in source_rows]:
            raise ValueError("paired rows disagree with verified source")
        if len(rows) != report["splits"][split].get("samples"):
            raise ValueError("paired sample count disagrees")
        validated[split] = rows
    return directory / train_file, validated[Path(train_file).stem]


def evaluate_states(dataset: Path, model_url: str, models: list[str], temperatures: list[float],
                    output: Path, seed: int, allow_train: bool = False) -> dict:
    from alvarezgui.model import InferenceRequest, InvalidActionOutputError
    from alvarezgui.model_http import HttpModelService

    rows = load_dataset(dataset)
    splits = {row.get("split") for row in rows}
    allowed_splits = {"validation", "test", "train"} if allow_train else {"validation", "test"}
    if len(splits) != 1 or not splits <= allowed_splits:
        raise ValueError("state evaluation requires one held-out split; use --allow-train for diagnostics")
    split = next(iter(splits))
    prepared = []
    for row in rows:
        directions = _validate_teacher_row(row, split)
        if "screen_hash" in row and _sha256(Path(row["image"])) != row["screen_hash"]:
            raise ValueError("evaluation image hash changed")
        if "state_hash" in row and hashlib.sha256("\n".join(row["board"]).encode()).hexdigest() != row["state_hash"]:
            raise ValueError("evaluation state hash changed")
        image_uri = "data:image/png;base64," + base64.b64encode(Path(row["image"]).read_bytes()).decode()
        prepared.append((directions, image_uri))
    service = HttpModelService(model_url, timeout=120)
    summaries = []
    with output.open("x", encoding="utf-8") as stream:
        for model in models:
            for temperature in temperatures:
                counts = Counter(total=len(rows), valid=0, allowed=0, correct=0, board_exact=0,
                                 next_state_correct=0, effect_correct=0)
                latencies, lengths, byte_lengths = [], [], []
                formats = set()
                for index, row in enumerate(rows):
                    record = {"id": row["id"], "model": model, "temperature": temperature,
                              "valid": False, "allowed": False, "correct": False,
                              "board_exact": False, "next_state_correct": False, "effect_correct": False}
                    directions, image_uri = prepared[index]
                    started = time.monotonic()
                    try:
                        response = service.infer(InferenceRequest(
                            request_id=f"sokoban-state-{uuid.uuid4().hex}", model_version=model,
                            instruction=SOKOBAN_INSTRUCTION, image_uri=image_uri, temperature=temperature,
                            seed=seed + index, max_tokens=None, timeout_ms=120000,
                        ))
                    except InvalidActionOutputError as exc:
                        raw = getattr(exc, "raw_output", None)
                        output_format = getattr(exc, "output_format", None)
                        if output_format is not None:
                            if output_format not in OUTPUT_FORMATS:
                                raise ValueError(f"server returned unsupported output format: {output_format}") from exc
                            formats.add(output_format)
                        record.update(error=str(exc), latency_ms=round((time.monotonic() - started) * 1000),
                                      output_format=output_format, raw_output=raw,
                                      output_chars=len(raw) if raw is not None else None,
                                      output_bytes=len(raw.encode()) if raw is not None else None)
                        if raw is not None:
                            lengths.append(record["output_chars"])
                            byte_lengths.append(record["output_bytes"])
                    else:
                        output_format = getattr(response, "output_format", "action-v1")
                        if output_format not in OUTPUT_FORMATS:
                            raise ValueError(f"server returned unsupported output format: {output_format}")
                        formats.add(output_format)
                        action = response.action
                        raw = response.raw_output
                        record.update(output_format=output_format, action=action, raw_output=raw,
                                      latency_ms=getattr(response, "latency_ms", round((time.monotonic() - started) * 1000)),
                                      output_chars=len(raw), output_bytes=len(raw.encode()))
                        lengths.append(record["output_chars"])
                        byte_lengths.append(record["output_bytes"])
                        try:
                            valid = parse_policy_output(raw, output_format) == action
                        except (ValueError, TypeError):
                            valid = False
                        direction = next((name for name, target in ACTIONS.items() if action == target), None)
                        allowed = valid and sokoban_action_allowed(action)
                        correct = allowed and direction in directions
                        record.update(valid=valid, allowed=allowed, correct=correct)
                    if record.get("output_format") == "sokoban-plan-v1" and record.get("raw_output") is not None:
                        try:
                            plan = json.loads(record["raw_output"])
                        except ValueError:
                            plan = None
                        if isinstance(plan, dict):
                            record["board_exact"] = plan.get("board") == row["board"]
                            decision = plan.get("decision")
                            direction = decision.get("direction") if isinstance(decision, dict) else None
                            if direction in tuple(ACTIONS):
                                walls, player, box, _ = parse_board(row["board"])
                                next_player, next_box = transition(walls, player, box, direction)
                                expected_state = {"player": list(next_player), "box": list(next_box)}
                                effect = "push" if next_box != box else "move" if next_player != player else "blocked"
                                record.update(next_state_correct=plan.get("expected_next_state") == expected_state,
                                              effect_correct=decision.get("effect") == effect)
                    latencies.append(record["latency_ms"])
                    counts.update({key: int(record[key]) for key in
                                   ("valid", "allowed", "correct", "board_exact", "next_state_correct", "effect_correct")})
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                structured = "sokoban-plan-v1" in formats
                summaries.append({
                    "model": model, "temperature": temperature, "output_formats": sorted(formats), **counts,
                    **{f"{key}_rate": counts[key] / len(rows) for key in ("valid", "allowed", "correct")},
                    **{f"{key}_rate": counts[key] / len(rows) if structured else None
                       for key in ("board_exact", "next_state_correct", "effect_correct")},
                    "latency_ms_mean": sum(latencies) / len(latencies),
                    "output_chars_mean": sum(lengths) / len(lengths) if lengths else None,
                    "output_bytes_mean": sum(byte_lengths) / len(byte_lengths) if byte_lengths else None,
                    "output_length_samples": len(lengths),
                })
                print(json.dumps(summaries[-1]), flush=True)
    report = {"dataset": str(dataset.resolve()), "dataset_sha256": _sha256(dataset),
              "seed": seed, "split": split, "diagnostic_only": split == "train", "results": summaries}
    output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("collect")
    capture.add_argument("--env-file", action="append", required=True)
    capture.add_argument("--levels", type=Path, default=Path("fixtures/sokoban/levels.json"))
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--workers", type=int, default=1)
    capture.add_argument("--seed", type=int, default=20261117)
    capture.add_argument("--level-id", action="append")
    register = commands.add_parser("register")
    register.add_argument("--env-file", action="append", required=True)
    register.add_argument("--dataset-dir", type=Path, required=True)
    register.add_argument("--train-file", choices=("train.jsonl", "train-small.jsonl"), default="train.jsonl")
    register.add_argument("--version", required=True)
    register.add_argument("--control-plane-url", required=True)
    rollout = commands.add_parser("evaluate-rollouts")
    rollout.add_argument("--env-file", action="append", required=True)
    rollout.add_argument("--control-plane-url", required=True)
    rollout.add_argument("--levels", type=Path, default=Path("fixtures/sokoban/levels.json"))
    rollout.add_argument("--split", choices=("validation", "test"), required=True)
    rollout.add_argument("--model", action="append", required=True)
    rollout.add_argument("--name", required=True)
    rollout.add_argument("--temperature", type=float, default=0.0)
    rollout.add_argument("--seed", type=int, default=20261117)
    rollout.add_argument("--max-steps", type=int, default=20)
    evaluate = commands.add_parser("evaluate-states")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--model-url", required=True)
    evaluate.add_argument("--model", action="append", required=True)
    evaluate.add_argument("--temperature", type=float, action="append")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=20261117)
    evaluate.add_argument("--allow-train", action="store_true", help="explicitly enable training-state diagnostics")
    augment = commands.add_parser("collect-augment")
    augment.add_argument("--env-file", action="append", required=True)
    augment.add_argument("--levels", type=Path, default=Path("fixtures/sokoban/levels.json"))
    augment.add_argument("--base-train", type=Path, required=True)
    augment.add_argument("--output", type=Path, required=True)
    augment.add_argument("--workers", type=int, default=1)
    augment.add_argument("--seed", type=int, default=20261119)
    export = commands.add_parser("export-paired")
    export.add_argument("--source-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export-paired":
        result = export_paired(args.source_dir, args.output_dir)
    elif args.command == "evaluate-states":
        result = evaluate_states(args.dataset, args.model_url, args.model, args.temperature or [0.0],
                                 args.output, args.seed, allow_train=args.allow_train)
    else:
        from alvarezgui.config import Settings

        settings = Settings(_env_file=tuple(args.env_file))
        if args.command == "collect":
            result = collect(settings, args.levels, args.output, args.workers, args.seed, args.level_id)
        elif args.command == "collect-augment":
            result = collect_augment(settings, args.levels, args.base_train, args.output, args.workers, args.seed)
        elif args.command == "register":
            from alvarezgui.training_agent import ControlPlane

            path, rows = registration_dataset(args.dataset_dir, args.train_file)
            cp = ControlPlane(args.control_plane_url, settings.console_password.get_secret_value())
            result = cp.call("POST", "/internal/v1/datasets", {
                "dataset_version": args.version, "source": path.as_uri(),
                "schema_version": "minicpm-sft-v1", "sample_count": len(rows),
            })
        else:
            from alvarezgui.training_agent import ControlPlane

            levels = load_levels(args.levels)
            body = {
                "name": args.name, "model_versions": args.model,
                "tasks": [{"task_type": "sokoban", "app_image": f"android-sokoban-v1:{level['id']}"}
                          for level in levels if level["split"] == args.split],
                "trials_per_task": 1, "max_steps": args.max_steps, "temperature": args.temperature,
                "seed": args.seed, "runtime_backend": settings.runtime_backend,
            }
            key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            cp = ControlPlane(args.control_plane_url, settings.console_password.get_secret_value())
            result = cp.call("POST", "/api/v1/evaluations", body, {"Idempotency-Key": f"sokoban:{key}"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
