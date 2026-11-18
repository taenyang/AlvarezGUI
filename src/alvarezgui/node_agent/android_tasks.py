from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alvarezgui.sokoban import LEVEL_COUNTS


class UnsupportedAndroidTask(ValueError):
    """The local Android lab has no safe implementation for this task."""


@dataclass(frozen=True, slots=True)
class AndroidTask:
    """A task whose success can be checked from Android runtime state."""

    task_kind: str
    target_package: str
    target_text: str | None = None
    # Start from the launcher's app list instead of the home screen.
    open_app_drawer: bool = False
    sokoban_level: str | None = None

    def succeeded(
        self, foreground_package: str | None, ui_text: set[str] | None = None
    ) -> bool:
        return (
            foreground_package == self.target_package
            and (self.target_text is None or self.target_text in (ui_text or set()))
        )

    def reward(
        self,
        foreground_package: str | None,
        *,
        terminated: bool,
        ui_text: set[str] | None = None,
    ) -> dict[str, Any]:
        success = self.succeeded(foreground_package, ui_text)
        return {
            "total": 1.0 if success else 0.0,
            "progress": 1.0 if success else 0.0,
            "success": success,
            "invalid_action": False,
            "safety_penalty": 0.0,
            "reward_version": "android-task-v1",
            "task_kind": self.task_kind,
            "foreground_package": foreground_package,
            "target_text": self.target_text,
            "termination_requested": terminated,
        }


def resolve_android_task(task_type: str, app_image: str) -> AndroidTask:
    if task_type == "android_launch_smoke":
        if app_image not in {"builtin:settings", "android-settings-v1"}:
            raise UnsupportedAndroidTask(
                f"unsupported smoke app_image={app_image!r}; use builtin:settings"
            )
        return AndroidTask(
            task_kind="builtin_app_launch_smoke",
            target_package="com.android.settings",
        )
    if task_type == "android_launch_smoke_drawer" and app_image == "android-settings-v1":
        return AndroidTask(
            task_kind="builtin_app_launch_from_drawer",
            target_package="com.android.settings",
            open_app_drawer=True,
        )
    if task_type == "shopping" and app_image in {"android-shop-v1", "builtin:test-shop"}:
        return AndroidTask(
            task_kind="shopping_fixture",
            target_package="org.chromium.webview_shell",
            target_text="Order complete",
        )
    if task_type == "sokoban":
        base, separator, level = app_image.partition(":")
        known_levels = {
            f"{split}-{index:03d}" for split, count in LEVEL_COUNTS.items() for index in range(count)
        }
        if base == "android-sokoban-v1" and (not separator or level in known_levels):
            return AndroidTask(
                task_kind="sokoban_fixture",
                target_package="org.chromium.webview_shell",
                target_text="Sokoban complete",
                sokoban_level=level or None,
            )
    raise UnsupportedAndroidTask(
        f"no offline Android task oracle for task_type={task_type!r}, "
        f"app_image={app_image!r}"
    )


SHOPPING_INSTRUCTION = (
    "Buy the Blue Widget test item. Tap Buy test item and finish only after Order complete is visible.\n"
    "The model canvas is 720x1280; tap/swipe coordinates use that canvas, "
    "not normalized 0..1000 or cropped-WebView coordinates."
)

SOKOBAN_INSTRUCTION = (
    "Push the box onto the goal to solve Sokoban. The player does not need to stand on the goal. "
    "Board legend: # is a wall, @ is the player, $ is the box, . is the goal, "
    "* is the box on the goal, and + is the player on the goal. "
    "Inspect the current board after every action and choose the next direction from the current positions. "
    "Coordinates use the entire 720x1280 image, with (0,0) at its top-left including Android chrome. "
    "Use only the four direction buttons above the Android navigation bar. "
    "Tap these full-screen button centers exactly: Up (360,836), Left (176,980), "
    "Down (360,980), Right (544,980). "
    "Never swipe, type, press Back, tap the board, tap the status bar, or tap Android navigation. "
    "Finish only after Sokoban complete is visible."
)

# Tasks the console offers; each entry must resolve through resolve_android_task.
TASK_CATALOG: tuple[dict[str, str], ...] = (
    {
        "task_type": "shopping",
        "app_image": "android-shop-v1",
        "label": "测试商城下单",
        "instruction": SHOPPING_INSTRUCTION,
    },
    {
        "task_type": "sokoban",
        "app_image": "android-sokoban-v1",
        "label": "推箱子小游戏",
        "instruction": SOKOBAN_INSTRUCTION,
    },
    {
        "task_type": "android_launch_smoke_drawer",
        "app_image": "android-settings-v1",
        "label": "从应用列表打开设置",
        "instruction": "Open the Android Settings app. The app list is already open: tap the Settings icon.\n"
        "Screenshot is 720x1280; tap/swipe coordinates are absolute screenshot pixels.",
    },
)


def sokoban_action_allowed(action: dict[str, Any]) -> bool:
    """Keep model actions inside the fixture controls and away from Android chrome."""
    action_type = action["type"]
    if action_type in {"wait", "terminate"}:
        return True
    if action_type != "tap":
        return False
    x, y = action["x"], action["y"]
    return (
        (276 <= x <= 443 and 773 <= y <= 900)
        or (92 <= x <= 259 and 917 <= y <= 1044)
        or (276 <= x <= 443 and 917 <= y <= 1044)
        or (460 <= x <= 627 and 917 <= y <= 1044)
    )


def default_instruction(task_type: str, app_image: str) -> str:
    if task_type == "sokoban":
        resolve_android_task(task_type, app_image)
        return SOKOBAN_INSTRUCTION
    for entry in TASK_CATALOG:
        if entry["task_type"] == task_type and entry["app_image"] == app_image:
            return entry["instruction"]
    resolve_android_task(task_type, app_image)
    raise UnsupportedAndroidTask(f"no default instruction for {task_type!r}/{app_image!r}")
