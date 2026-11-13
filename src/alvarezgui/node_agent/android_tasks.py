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
