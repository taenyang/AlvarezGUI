import unittest

from alvarezgui.node_agent.android_tasks import (
    UnsupportedAndroidTask,
    resolve_android_task,
    sokoban_action_allowed,
)


class AndroidTaskTest(unittest.TestCase):
    def test_only_explicit_settings_smoke_is_available(self) -> None:
        task = resolve_android_task("android_launch_smoke", "builtin:settings")
        self.assertTrue(task.succeeded("com.android.settings"))
        self.assertFalse(task.succeeded("com.android.launcher3"))

    def test_drawer_variant_opens_app_list_with_same_oracle(self) -> None:
        task = resolve_android_task("android_launch_smoke_drawer", "android-settings-v1")
        self.assertTrue(task.open_app_drawer)
        self.assertTrue(task.succeeded("com.android.settings"))
        self.assertFalse(resolve_android_task("android_launch_smoke", "android-settings-v1").open_app_drawer)

    def test_shopping_fixture_requires_order_complete(self) -> None:
        task = resolve_android_task("shopping", "android-shop-v1")
        self.assertFalse(task.succeeded("org.chromium.webview_shell", {"Buy test item"}))
        self.assertTrue(task.succeeded("org.chromium.webview_shell", {"Order complete"}))

    def test_sokoban_fixture_requires_completion_text(self) -> None:
        task = resolve_android_task("sokoban", "android-sokoban-v1")
        self.assertFalse(task.succeeded("org.chromium.webview_shell", {"Push the box onto the goal."}))
        self.assertTrue(task.succeeded("org.chromium.webview_shell", {"Sokoban complete"}))

    def test_sokoban_actions_stay_inside_direction_controls(self) -> None:
        self.assertTrue(sokoban_action_allowed({"type": "tap", "x": 360, "y": 1016}))
        self.assertTrue(sokoban_action_allowed({"type": "tap", "x": 176, "y": 1160}))
        self.assertFalse(sokoban_action_allowed({"type": "tap", "x": 360, "y": 817}))
        self.assertFalse(sokoban_action_allowed({"type": "swipe", "x1": 0, "y1": 0, "x2": 0, "y2": 1}))

    def test_unknown_shopping_fixture_is_rejected(self) -> None:
        with self.assertRaises(UnsupportedAndroidTask):
            resolve_android_task("shopping", "missing-shop")


if __name__ == "__main__":
    unittest.main()
