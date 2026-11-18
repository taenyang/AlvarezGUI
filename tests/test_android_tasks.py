import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
from urllib.parse import parse_qsl, urlsplit

from alvarezgui.node_agent.docker import DockerAndroidRuntime, DockerConfig
from alvarezgui.sokoban import DIRECTIONS, load_levels, parse_board, render_board, solve, transition

from alvarezgui.node_agent.android import AndroidRuntime
from alvarezgui.node_agent.runtime import RuntimeOperationError, Sandbox, SandboxState
from alvarezgui.action import scale_action_to_screen
from alvarezgui.node_agent.android import _png_dimensions
from alvarezgui.node_agent.android_tasks import (
    UnsupportedAndroidTask,
    default_instruction,
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
        self.assertTrue(sokoban_action_allowed({"type": "tap", "x": 360, "y": 836}))
        self.assertTrue(sokoban_action_allowed({"type": "tap", "x": 176, "y": 980}))
        self.assertFalse(sokoban_action_allowed({"type": "tap", "x": 360, "y": 700}))
        self.assertFalse(sokoban_action_allowed({"type": "tap", "x": 176, "y": 1100}))
        self.assertFalse(sokoban_action_allowed({"type": "swipe", "x1": 0, "y1": 0, "x2": 0, "y2": 1}))

    def test_sokoban_instruction_explains_symbols_and_rechecks_board(self) -> None:
        instruction = default_instruction("sokoban", "android-sokoban-v1")
        self.assertIn("@ is the player", instruction)
        self.assertIn("Inspect the current board after every action", instruction)
        self.assertNotIn("Left, Left, Up", instruction)
        self.assertIn("entire 720x1280 image", instruction)

    def test_png_dimensions_are_read_from_the_screenshot_header(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + (1080).to_bytes(4, "big") + (1920).to_bytes(4, "big")
        self.assertEqual(_png_dimensions(png), (1080, 1920))

    def test_model_coordinates_scale_to_the_actual_screen(self) -> None:
        self.assertEqual(
            scale_action_to_screen(
                {"type": "tap", "x": 176, "y": 1160}, 1080, 1920
            ),
            {"type": "tap", "x": 264, "y": 1740},
        )
        self.assertEqual(
            scale_action_to_screen(
                {"type": "swipe", "x1": 10, "y1": 20, "x2": 30, "y2": 40, "duration_ms": 5},
                360,
                640,
            ),
            {"type": "swipe", "x1": 5, "y1": 10, "x2": 15, "y2": 20, "duration_ms": 5},
        )

    def test_model_coordinates_outside_the_canvas_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            scale_action_to_screen({"type": "tap", "x": 720, "y": 100}, 720, 1280)

    def test_runtime_checks_model_canvas_before_scaling_without_guessing_offsets(self) -> None:
        runtime = AndroidRuntime(SimpleNamespace(network_enabled=False, api_timeout=0))
        runtime._sandboxes["test"] = Sandbox("test", SandboxState.READY, datetime.now(timezone.utc))
        runtime._tasks["test"] = resolve_android_task("sokoban", "android-sokoban-v1")
        runtime._task_state = Mock(return_value=("org.chromium.webview_shell", set()))
        runtime._adb = Mock()
        for width, height, x, y in ((720, 1280, "176", "980"), (1080, 1920, "264", "1470")):
            with self.subTest(width=width):
                runtime._screen_sizes["test"] = (width, height)
                result = runtime.execute("test", {"type": "tap", "x": 176, "y": 980})
                self.assertFalse(result["reward"]["invalid_action"])
                runtime._adb.assert_called_with("test", "shell", "input", "tap", x, y)
                runtime._adb.reset_mock()
                result = runtime.execute("test", {"type": "tap", "x": 360, "y": 642})
                self.assertTrue(result["reward"]["invalid_action"])
                runtime._adb.assert_not_called()

    def test_adb_error_preserves_operation_and_diagnostic_without_text_input(self) -> None:
        runtime = AndroidRuntime(SimpleNamespace(adb_binary="adb", api_timeout=10))
        runtime._adb_target = Mock(return_value="127.0.0.1:5555")
        failed = SimpleNamespace(returncode=1, stdout=b"", stderr=b"ERROR: could not get idle state")
        with patch("alvarezgui.node_agent.android.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(RuntimeOperationError, "shell uiautomator dump.*exit 1.*idle state"):
                runtime._adb("test", "shell", "uiautomator", "dump", "/sdcard/window.xml")
            failed.stderr = b"failed input: private-text"
            with self.assertRaises(RuntimeOperationError) as caught:
                runtime._adb("test", "shell", "input", "text", "private-text")
            self.assertNotIn("private-text", str(caught.exception))
            self.assertIn("text input omitted", str(caught.exception))

    def test_unknown_shopping_fixture_is_rejected(self) -> None:
        with self.assertRaises(UnsupportedAndroidTask):
            resolve_android_task("shopping", "missing-shop")

    def test_sokoban_level_ids_preserve_the_default_task_contract(self) -> None:
        default = resolve_android_task("sokoban", "android-sokoban-v1")
        self.assertIsNone(default.sokoban_level)
        for level_id in ("train-000", "train-047", "validation-007", "test-007"):
            with self.subTest(level=level_id):
                app_image = f"android-sokoban-v1:{level_id}"
                task = resolve_android_task("sokoban", app_image)
                self.assertEqual(task.sokoban_level, level_id)
                self.assertEqual(task.task_kind, default.task_kind)
                self.assertEqual(task.target_package, default.target_package)
                self.assertEqual(task.target_text, default.target_text)
                self.assertEqual(
                    default_instruction("sokoban", app_image),
                    default_instruction("sokoban", "android-sokoban-v1"),
                )
                self.assertFalse(task.succeeded(task.target_package, {"Push the box onto the goal."}))
                self.assertFalse(task.succeeded("com.android.launcher3", {"Sokoban complete"}))
                self.assertTrue(task.succeeded(task.target_package, {"Sokoban complete"}))

    def test_sokoban_invalid_level_ids_are_rejected_by_task_and_instruction(self) -> None:
        app_images = [f"android-sokoban-v1:{level}" for level in (
            "", "train-048", "validation-008", "test-008", "train-0010",
            "train-00", "train--001", "train-000:extra", "validation-007 ",
            "TEST-007", "unknown-000", "../train-000",
        )] + ["android-sokoban-v2:train-000", "builtin:sokoban"]
        for app_image in app_images:
            for resolver in (resolve_android_task, default_instruction):
                with self.subTest(app_image=app_image, resolver=resolver.__name__):
                    with self.assertRaisesRegex(
                        UnsupportedAndroidTask,
                        re.escape(f"no offline Android task oracle for task_type='sokoban', app_image={app_image!r}"),
                    ):
                        resolver("sokoban", app_image)

    def sokoban_runtime(self, fixture_url: str) -> DockerAndroidRuntime:
        runtime = DockerAndroidRuntime(DockerConfig(
            binary="docker", image="android:test", state_dir=Path("/home/unused-test-state"),
            sokoban_fixture_url=fixture_url, api_timeout=1,
        ))
        runtime._sandboxes["test"] = Sandbox("test", SandboxState.READY, datetime.now(timezone.utc))
        runtime._prepare_fixture_network = Mock()
        return runtime

    def test_initialize_sokoban_selects_level_and_waits_for_loaded_status(self) -> None:
        queries = ("", "?theme=light&level=train-001&tag=a%2Bb&level=test-000&tag=two#board")
        for level_id in (None, "train-000", "validation-007", "test-007"):
            for query in queries:
                with self.subTest(level=level_id, query=query):
                    runtime = self.sokoban_runtime(f"http://192.168.100.1:18090/sokoban/{query}")
                    runtime._adb = Mock(side_effect=[
                        b"", b"",
                        b"", b'<hierarchy><node text="Sokoban" /></hierarchy>',
                        b"", '<hierarchy><node text="Sokoban" /><node text="Loading…" /></hierarchy>'.encode(),
                        b"", b'<hierarchy><node text="Push the box onto the goal." /></hierarchy>',
                        b"mResumedActivity: ActivityRecord{ org.chromium.webview_shell/.WebViewBrowserActivity }",
                    ])
                    app_image = "android-sokoban-v1" + (f":{level_id}" if level_id else "")
                    with patch("alvarezgui.node_agent.android.time.monotonic", side_effect=[0, .1, .2, .3]), patch(
                        "alvarezgui.node_agent.android.time.sleep"
                    ) as sleep:
                        result = runtime.initialize_task("test", "sokoban", app_image, "solve")
                    self.assertEqual(result, {
                        "task_kind": "sokoban_fixture",
                        "target_package": "org.chromium.webview_shell",
                        "foreground_package": "org.chromium.webview_shell",
                    })
                    self.assertEqual(runtime._tasks["test"].sokoban_level, level_id)
                    runtime._prepare_fixture_network.assert_called_once_with("test")
                    start = runtime._adb.call_args_list[1].args
                    self.assertEqual(start[:-1], (
                        "test", "shell", "am", "start", "-n",
                        "org.chromium.webview_shell/.WebViewBrowserActivity", "-d",
                    ))
                    url = urlsplit(start[-1])
                    self.assertEqual((url.scheme, url.netloc, url.path), (
                        "http", "host.docker.internal:18090", "/sokoban/",
                    ))
                    expected = [("theme", "light"), ("tag", "a+b"), ("tag", "two")] if query else []
                    if level_id is not None:
                        expected.append(("level", level_id))
                    self.assertEqual(parse_qsl(url.query), expected)
                    self.assertEqual(url.fragment, "board" if query else "")
                    self.assertEqual(runtime._adb.call_args_list[2:8], [
                        call("test", "shell", "uiautomator", "dump", "/sdcard/window.xml"),
                        call("test", "shell", "cat", "/sdcard/window.xml"),
                    ] * 3)
                    self.assertEqual(sleep.call_args_list, [call(.1), call(.1)])

    def test_initialize_sokoban_loading_or_error_status_does_not_register_task(self) -> None:
        for status in ("Loading…", "Unknown Sokoban level", "Could not load Sokoban levels"):
            with self.subTest(status=status):
                runtime = self.sokoban_runtime("http://192.168.100.1:18090/sokoban/")
                runtime._tasks["test"] = resolve_android_task("sokoban", "android-sokoban-v1")
                runtime._adb = Mock(side_effect=[
                    b"", b"", b"",
                    f'<hierarchy><node text="Sokoban" /><node text="{status}" /></hierarchy>'.encode(),
                ])
                with patch("alvarezgui.node_agent.android.time.monotonic", side_effect=[0, .1, 1.1]), patch(
                    "alvarezgui.node_agent.android.time.sleep"
                ):
                    with self.assertRaisesRegex(
                        RuntimeOperationError, r"Android fixture did not load: Push the box onto the goal\."
                    ):
                        runtime.initialize_task("test", "sokoban", "android-sokoban-v1:train-000", "solve")
                self.assertNotIn("test", runtime._tasks)
                self.assertEqual(runtime._adb.call_count, 4)

    def test_ui_text_collects_visible_text_and_board_accessibility_description(self) -> None:
        board = ["########", "# .    #", "# $    #", "#   @  #", "#      #", "########"]
        description = "Sokoban board: " + "|".join(board)
        runtime = AndroidRuntime(SimpleNamespace())
        runtime._adb = Mock(side_effect=[b"", (
            '<hierarchy><node text="Sokoban">'
            f'<node text="Board" content-desc="{description}" />'
            '<node text="Push the box onto the goal." content-desc="Push the box onto the goal." />'
            '<node text="" content-desc="Up" /><node /><node text="" content-desc="" />'
            '</node></hierarchy>'
        ).encode()])
        self.assertEqual(runtime._ui_text("test"), {
            "Sokoban", "Board", description, "Push the box onto the goal.", "Up",
        })
        runtime._adb.assert_has_calls([
            call("test", "shell", "uiautomator", "dump", "/sdcard/window.xml"),
            call("test", "shell", "cat", "/sdcard/window.xml"),
        ])

    @unittest.skipUnless(shutil.which("node"), "Node is required to execute the fixture JavaScript")
    def test_fixture_javascript_matches_python_solver_for_all_levels(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "sokoban"
        source = re.search(r"<script>(.*?)</script>", (fixture / "index.html").read_text(), re.S).group(1)
        levels = load_levels(fixture / "levels.json")
        cases = []
        default = {"id": None, "board": [
            "########", "# .    #", "# $    #", "#   @  #", "#      #", "########",
        ]}
        for level in [default, *levels]:
            walls, player, box, goal = parse_board(level["board"])
            path = solve(level["board"])
            checks = []
            for direction in [*path, None]:
                probes = []
                for name, delta in DIRECTIONS.items():
                    next_player, next_box = (player, box) if box == goal else transition(walls, player, box, name)
                    probes.append([delta, "\n".join(render_board(walls, next_player, next_box, goal)), next_box == goal])
                checks.append({
                    "board": "\n".join(render_board(walls, player, box, goal)),
                    "probes": probes, "move": DIRECTIONS[direction] if direction else None,
                })
                if direction:
                    player, box = transition(walls, player, box, direction)
            self.assertEqual(box, goal, level["id"])
            cases.append({"id": level["id"], "checks": checks})
        # Execute the shipped script, including async query selection, with only DOM/fetch stubbed.
        javascript = r'''
const fs = require("node:fs"), vm = require("node:vm"), assert = require("node:assert/strict");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
(async () => {
  for (const entry of input.cases) {
    const elements = Object.fromEntries(["board", "status"].map(id => [id, {
      textContent: "", className: "", attributes: {},
      setAttribute(name, value) { this.attributes[name] = value; },
    }]));
    const context = vm.createContext({
      assert, checks: entry.checks, URLSearchParams,
      location: {search: entry.id ? "?level=" + entry.id : ""},
      document: {getElementById: id => elements[id], addEventListener() {}},
      fetch: async path => {
        assert.notEqual(entry.id, null, "default board must not fetch a catalog");
        assert.equal(path, "levels.json");
        return {ok: true, json: async () => ({levels: input.levels})};
      },
    });
    try {
      await vm.runInContext(input.source, context, {timeout: 1000});
      vm.runInContext(`
        assert.equal(status.textContent, "Push the box onto the goal.");
        for (const check of checks) {
          assert.equal(board.textContent, check.board);
          assert.equal(board.attributes["aria-label"], "Sokoban board: " + check.board.replace(/ /g, "_").split("\\n").join("|"));
          const saved = [player, box, complete, status.textContent, status.className];
          for (const [delta, expected, solved] of check.probes) {
            move(...delta);
            assert.equal(board.textContent, expected);
            assert.equal(complete, solved);
            assert.equal(status.textContent, solved ? "Sokoban complete" : "Push the box onto the goal.");
            assert.equal(status.className, solved ? "complete" : "");
            [player, box, complete, status.textContent, status.className] = saved;
            render();
          }
          if (check.move) move(...check.move);
        }
        assert.equal(complete, true);
        assert.equal(status.textContent, "Sokoban complete");
      `, context, {timeout: 1000});
    } catch (error) {
      throw new Error("level " + (entry.id || "default") + ": " + error.stack);
    }
  }
  process.stdout.write(JSON.stringify(input.cases.map(entry => entry.id)));
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
        result = subprocess.run(
            [shutil.which("node"), "-e", javascript],
            input=json.dumps({"source": source, "levels": levels, "cases": cases}),
            text=True, capture_output=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [None, *(level["id"] for level in levels)])


if __name__ == "__main__":
    unittest.main()
