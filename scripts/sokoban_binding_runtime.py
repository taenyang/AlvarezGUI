"""Drive real Android screenshots and taps; board text is used only to audit outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from alvarezgui.config import Settings
from alvarezgui.minicpm_service import MiniCPMService
from alvarezgui.model import InferenceRequest
from alvarezgui.node_agent.android_tasks import SOKOBAN_INSTRUCTION
from alvarezgui.node_agent.factory import runtime_from_settings
from alvarezgui.sokoban import ACTIONS, parse_board, render_board, transition
from alvarezgui.sokoban_sft import observed_board, optimal_directions


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    settings = Settings(_env_file=str(args.env_file))
    runtime = runtime_from_settings(settings)
    if not str(runtime.config.state_dir.resolve()).startswith("/home/"):
        raise ValueError("runtime state must be under /home")
    service = MiniCPMService.load_checkpoint(args.checkpoint, "candidate")
    sandbox = "binding-" + str(time.time_ns())
    report = {"checkpoint": args.checkpoint, "max_steps": 24, "temperature": 0,
              "repeat_stop": "stop after third visit to the same board", "episodes": []}
    created = False
    try:
        runtime.create(sandbox)
        created = True
        phases = (("test", 24),) if args.rollouts_only else (("validation", 1), ("test", 24))
        for split, limit in phases:
            for index in range(8):
                level = f"{split}-{index:03d}"
                runtime.initialize_task(sandbox, "sokoban", f"android-sokoban-v1:{level}", SOKOBAN_INSTRUCTION)
                observation = runtime.observe(sandbox)
                episode = {"level": level, "kind": "single-step" if limit == 1 else "rollout",
                           "success": False, "steps": [], "stop": "step_limit"}
                report["episodes"].append(episode)
                visits = Counter()
                for step in range(limit):
                    board = observed_board(observation)
                    key = tuple(board)
                    visits[key] += 1
                    if visits[key] >= 3:
                        episode["stop"] = "repeated_board"
                        break
                    # No UI text, board, labels, or solver outputs enter inference.
                    response = service.infer(InferenceRequest(
                        request_id=f"{level}:{step}", model_version="candidate",
                        instruction=SOKOBAN_INSTRUCTION, image_uri=observation["image_uri"],
                        temperature=0, timeout_ms=120000,
                    ))
                    direction = next(name for name, action in ACTIONS.items() if action == response.action)
                    result = runtime.execute(sandbox, response.action)
                    after = runtime.observe(sandbox)
                    after_board = observed_board(after)
                    walls, player, box, goal = parse_board(board)
                    next_player, next_box = transition(walls, player, box, direction)
                    effect = "push" if next_box != box else "move" if next_player != player else "blocked"
                    expected = render_board(walls, next_player, next_box, goal)
                    try:
                        optimal = optimal_directions(board)
                    except ValueError:
                        optimal = []  # A model-created dead end is a policy failure, not an infrastructure error.
                    record = {
                        "step": step, "before": observation, "after": after,
                        "board": board, "after_board": after_board, "direction": direction,
                        "action": response.action, "raw_output": response.raw_output,
                        "latency_ms": response.latency_ms, "result": result, "effect": effect,
                        "transition_matches": expected == after_board,
                        "optimal": direction in optimal,
                    }
                    episode["steps"].append(record)
                    if expected != after_board:
                        raise RuntimeError(f"{level}:{step}: Android transition differs from tap semantics")
                    _, _, after_box, after_goal = parse_board(after_board)
                    if bool(result["reward"].get("success")) != (after_box == after_goal):
                        raise RuntimeError(f"{level}:{step}: runtime reward disagrees with observed board")
                    observation = after
                    if result["done"]:
                        episode["success"] = bool(result["reward"].get("success"))
                        episode["stop"] = "done"
                        break
                (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
                print(json.dumps({k: v for k, v in episode.items() if k != "steps"}), flush=True)
    except Exception as exc:
        report["evaluation_error"] = repr(exc)
        raise
    finally:
        if created:
            try:
                runtime.destroy(sandbox)
                report["sandbox_destroyed"] = True
            except Exception as exc:
                report["cleanup_error"] = repr(exc)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
        service.unload_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts-only", action="store_true")
    run(parser.parse_args())
