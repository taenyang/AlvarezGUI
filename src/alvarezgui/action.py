from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


MODEL_SCREEN_WIDTH = 720
MODEL_SCREEN_HEIGHT = 1280


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str


class TapAction(Action):
    type: Literal["tap"] = "tap"
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class SwipeAction(Action):
    type: Literal["swipe"] = "swipe"
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)
    duration_ms: int = Field(default=300, ge=1, le=60_000)


class TextAction(Action):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=4_096)


class BackAction(Action):
    type: Literal["back"] = "back"


class WaitAction(Action):
    type: Literal["wait"] = "wait"
    duration_ms: int = Field(default=1_000, ge=1, le=120_000)


class TerminateAction(Action):
    type: Literal["terminate"] = "terminate"
    reason: str = Field(default="model_requested", min_length=1, max_length=500)


_ACTION_TYPES = {
    "tap": TapAction,
    "swipe": SwipeAction,
    "text": TextAction,
    "back": BackAction,
    "wait": WaitAction,
    "terminate": TerminateAction,
}


def validate_action(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("action must be an object")
    action_type = value.get("type")
    model = _ACTION_TYPES.get(action_type)
    if model is None:
        raise ValueError(f"unsupported action type: {action_type!r}")
    try:
        return model.model_validate(value).model_dump(mode="json")
    except ValidationError as exc:
        raise ValueError(f"invalid {action_type} action: {exc.errors()[0]['msg']}") from exc


def scale_action_to_screen(
    action: dict[str, Any], width: int, height: int
) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        raise ValueError("screen dimensions must be positive")
    scaled = dict(action)
    x_scale = width / MODEL_SCREEN_WIDTH
    y_scale = height / MODEL_SCREEN_HEIGHT
    coordinates = {
        key: action[key]
        for key in ("x", "y", "x1", "y1", "x2", "y2")
        if key in action
    }
    for key, value in coordinates.items():
        limit = MODEL_SCREEN_WIDTH if key.startswith("x") else MODEL_SCREEN_HEIGHT
        if value >= limit:
            raise ValueError(f"{key} is outside the model screen")
    for key, value in coordinates.items():
        scale = x_scale if key.startswith("x") else y_scale
        limit = width if key.startswith("x") else height
        scaled[key] = min(limit - 1, round(value * scale))
    return scaled


def is_terminal(action: dict[str, Any]) -> bool:
    return action.get("type") == "terminate"
