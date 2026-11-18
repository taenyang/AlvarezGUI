"""Shared policy prompts and strict output schemas; inference never repairs predictions."""

from __future__ import annotations

import json
from typing import Any

from alvarezgui.action import MODEL_SCREEN_HEIGHT, MODEL_SCREEN_WIDTH, validate_action
from alvarezgui.sokoban import ACTIONS, parse_board, transition


OUTPUT_FORMATS = ("action-v1", "sokoban-plan-v1")
PLAN_INSTRUCTION_MARKER = "Output format: sokoban-plan-v1."
POLICY_SUFFIXES = {
    "action-v1": (
        "The image is resized to a 720x1280 model canvas; output coordinates use that canvas, "
        "and the runtime maps them to the device screen.\n"
        "Return exactly one JSON object and nothing else. Use exactly one of these schemas: "
        '{"type":"tap","x":integer,"y":integer}, '
        '{"type":"swipe","x1":integer,"y1":integer,"x2":integer,"y2":integer,"duration_ms":integer}, '
        '{"type":"text","text":"..."}, {"type":"back"}, '
        '{"type":"wait","duration_ms":integer}, or {"type":"terminate","reason":"..."}. '
        "Never output a key named action, an action list, parentheses, Markdown, or explanation."
    ),
    "sokoban-plan-v1": (
        f"{PLAN_INSTRUCTION_MARKER}\n"
        "The image is resized to a 720x1280 model canvas; output coordinates use that canvas, "
        "and the runtime maps them to the device screen.\n"
        "Read the board from the image. Return exactly one strict JSON object and nothing else: "
        '{"board":[six strings of eight characters],'
        '"decision":{"direction":"up|down|left|right","effect":"move|push"},'
        '"expected_next_state":{"player":[row,col],"box":[row,col]},'
        '"action":{"type":"tap","x":integer,"y":integer}}. '
        "Use exactly these keys. Board symbols are # wall, space floor, @ player, $ box, "
        ". goal, + player on goal, and * box on goal. Board coordinates are 0-based: "
        "row 0..5, col 0..7. Predict the player and box positions after your chosen move. "
        "Tap coordinates by direction: up=(360,836), down=(360,980), left=(176,980), "
        "right=(544,980). Never output Markdown or explanation."
    ),
}


def policy_instruction(instruction: str, output_format: str = "action-v1") -> str:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"unsupported output format: {output_format!r}")
    return f"{instruction}\n{POLICY_SUFFIXES[output_format]}"


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly {', '.join(sorted(expected))}")


def parse_policy_output(raw_output: str, output_format: str = "action-v1") -> dict[str, Any]:
    """Validate a complete JSON prediction and return its executable action.

    A plan's board and future state are predictions, not teacher inputs. Only their
    schema is checked here; their accuracy belongs to offline evaluation.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"unsupported output format: {output_format!r}")
    if not isinstance(raw_output, str):
        raise ValueError("policy output must be a JSON string")
    value = json.loads(raw_output, object_pairs_hook=_object, parse_constant=_reject_constant)
    action_value = value
    if output_format == "sokoban-plan-v1":
        _keys(value, {"board", "decision", "expected_next_state", "action"}, "plan")
        parse_board(value["board"])
        decision = value["decision"]
        _keys(decision, {"direction", "effect"}, "decision")
        if decision["direction"] not in tuple(ACTIONS) or decision["effect"] not in ("move", "push"):
            raise ValueError("invalid plan direction or effect")
        state = value["expected_next_state"]
        _keys(state, {"player", "box"}, "expected_next_state")
        for position in state.values():
            if (not isinstance(position, list) or len(position) != 2
                    or any(type(coordinate) is not int for coordinate in position)
                    or not (0 <= position[0] < 6 and 0 <= position[1] < 8)):
                raise ValueError("state coordinates must be [row,col] integers within the board")
        action_value = value["action"]
        _keys(action_value, {"type", "x", "y"}, "plan action")
        if action_value != ACTIONS[decision["direction"]]:
            raise ValueError("plan action does not match direction")
    if not isinstance(action_value, dict) or not isinstance(action_value.get("type"), str):
        raise ValueError("action must be an object with a string type")
    action = validate_action(action_value)
    for key in ("x", "y", "x1", "y1", "x2", "y2", "duration_ms"):
        if key in action_value and type(action_value[key]) is not int:
            raise ValueError(f"{key} must be an integer")
        if key in action and key != "duration_ms":
            bound = MODEL_SCREEN_WIDTH if key.startswith("x") else MODEL_SCREEN_HEIGHT
            if not 0 <= action[key] < bound:
                raise ValueError(f"{key} is outside the model screen")
    return action


def make_plan(board: list[str], direction: str) -> dict[str, Any]:
    """Build an offline transition label; never called to repair model output."""
    walls, player, box, _ = parse_board(board)
    next_player, next_box = transition(walls, player, box, direction)
    if (next_player, next_box) == (player, box):
        raise ValueError("plan direction must produce a move or push")
    return {
        "board": list(board),
        "decision": {"direction": direction, "effect": "push" if next_box != box else "move"},
        "expected_next_state": {"player": list(next_player), "box": list(next_box)},
        "action": dict(ACTIONS[direction]),
    }
