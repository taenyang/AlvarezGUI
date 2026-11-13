import unittest

from alvarezgui.node_agent.android_tasks import UnsupportedAndroidTask, resolve_android_task


class AndroidTaskTest(unittest.TestCase):
    def test_only_explicit_settings_smoke_is_available(self) -> None:
        task = resolve_android_task("android_launch_smoke", "builtin:settings")
        self.assertTrue(task.succeeded("com.android.settings"))
        self.assertFalse(task.succeeded("com.android.launcher3"))

    def test_shopping_fixture_requires_order_complete(self) -> None:
        task = resolve_android_task("shopping", "android-shop-v1")
        self.assertFalse(task.succeeded("org.chromium.webview_shell", {"Buy test item"}))
        self.assertTrue(task.succeeded("org.chromium.webview_shell", {"Order complete"}))

    def test_unknown_shopping_fixture_is_rejected(self) -> None:
        with self.assertRaises(UnsupportedAndroidTask):
            resolve_android_task("shopping", "missing-shop")


if __name__ == "__main__":
    unittest.main()
