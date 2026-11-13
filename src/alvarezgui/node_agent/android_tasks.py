from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class UnsupportedAndroidTask(ValueError):
    """The local Android lab has no safe implementation for this task."""


@dataclass(frozen=True, slots=True)
class AndroidTask:
    """A task whose success can be checked from Android runtime state."""

    task_kind: str
    target_package: str
    target_text: str | None = None

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
    if task_type == "shopping" and app_image in {"android-shop-v1", "builtin:test-shop"}:
        return AndroidTask(
            task_kind="shopping_fixture",
            target_package="org.chromium.webview_shell",
            target_text="Order complete",
        )
    raise UnsupportedAndroidTask(
        f"no offline Android task oracle for task_type={task_type!r}, "
        f"app_image={app_image!r}"
    )


SHOPPING_INSTRUCTION = (
    "Buy the Blue Widget test item. Tap Buy test item and finish only after Order complete is visible.\n"
    "Screenshot is 720x1280; tap/swipe coordinates are absolute screenshot pixels, "
    "not normalized 0..1000 or cropped-WebView coordinates."
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
        "task_type": "android_launch_smoke",
        "app_image": "android-settings-v1",
        "label": "打开系统设置",
        "instruction": "Open the Android Settings app.\n"
        "Screenshot is 720x1280; tap/swipe coordinates are absolute screenshot pixels.",
    },
)


def default_instruction(task_type: str, app_image: str) -> str:
    for entry in TASK_CATALOG:
        if entry["task_type"] == task_type and entry["app_image"] == app_image:
            return entry["instruction"]
    resolve_android_task(task_type, app_image)
    raise UnsupportedAndroidTask(f"no default instruction for {task_type!r}/{app_image!r}")
