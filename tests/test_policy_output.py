import copy
import json
import unittest
from unittest.mock import patch

from alvarezgui.policy_output import OUTPUT_FORMATS, make_plan, parse_policy_output, policy_instruction
from alvarezgui.sokoban import ACTIONS


class PolicyOutputTest(unittest.TestCase):
    board = ["########", "# .    #", "# $    #", "#   @  #", "#      #", "########"]

    def test_action_instruction_is_the_exact_legacy_prompt(self):
        expected = (
            "task\n"
            "The image is resized to a 720x1280 model canvas; output coordinates use that canvas, "
            "and the runtime maps them to the device screen.\n"
            "Return exactly one JSON object and nothing else. Use exactly one of these schemas: "
            '{"type":"tap","x":integer,"y":integer}, '
            '{"type":"swipe","x1":integer,"y1":integer,"x2":integer,"y2":integer,"duration_ms":integer}, '
            '{"type":"text","text":"..."}, {"type":"back"}, '
            '{"type":"wait","duration_ms":integer}, or {"type":"terminate","reason":"..."}. '
            "Never output a key named action, an action list, parentheses, Markdown, or explanation."
        )
        self.assertEqual(policy_instruction("task"), expected)
        plan_prompt = policy_instruction("task", "sokoban-plan-v1")
        self.assertIn("0-based", plan_prompt)
        self.assertIn("six strings of eight characters", plan_prompt)
        self.assertNotIn(self.board[1], plan_prompt)
        self.assertEqual(OUTPUT_FORMATS, ("action-v1", "sokoban-plan-v1"))
        with self.assertRaises(ValueError):
            policy_instruction("task", "other")

    def test_whole_json_and_strict_action_validation(self):
        raw = json.dumps(ACTIONS["left"])
        self.assertEqual(parse_policy_output(" \n" + raw + "\n"), ACTIONS["left"])
        for invalid in (
            "Answer: " + raw, "```json\n" + raw + "\n```", raw + " trailing", raw + raw,
            '[{"type":"back"}]', '{"type":"back","type":"back"}',
            '{"type":"tap","x":NaN,"y":10}', '{"type":"tap","x":true,"y":10}',
            '{"type":"tap","x":"10","y":10}', '{"type":"tap","x":10.0,"y":10}',
            '{"type":"tap","x":720,"y":10}', '{"type":"tap","x":10,"y":1280}',
            '{"type":"tap","x":-1,"y":10}', '{"type":[],"x":10,"y":10}',
            '{"type":"back","extra":1}', '{"type":"wait","duration_ms":0}',
        ):
            with self.subTest(raw=invalid), self.assertRaises(ValueError):
                parse_policy_output(invalid)
        with self.assertRaises(ValueError):
            parse_policy_output(raw, "unknown")

    def test_plan_labels_moves_pushes_and_solved_boards(self):
        move = make_plan(self.board, "left")
        self.assertEqual(move["decision"], {"direction": "left", "effect": "move"})
        self.assertEqual(move["expected_next_state"], {"player": [3, 3], "box": [2, 2]})
        push_board = ["########", "# .    #", "# $    #", "# @    #", "#      #", "########"]
        push = make_plan(push_board, "up")
        self.assertEqual(push["decision"]["effect"], "push")
        self.assertEqual(push["expected_next_state"], {"player": [2, 2], "box": [1, 2]})
        solved = ["########", "# *    #", "# @    #", "#      #", "#      #", "########"]
        self.assertEqual(parse_policy_output(json.dumps(make_plan(solved, "left")), "sokoban-plan-v1"), ACTIONS["left"])
        with self.assertRaisesRegex(ValueError, "move or push"):
            make_plan(solved, "up")

    def test_inference_does_not_correct_or_verify_predicted_transition(self):
        plan = make_plan(self.board, "left")
        plan["expected_next_state"] = {"player": [4, 6], "box": [1, 1]}
        plan["decision"]["effect"] = "push"
        raw = json.dumps(plan)
        with patch("alvarezgui.policy_output.transition", side_effect=AssertionError("no oracle")), patch(
            "alvarezgui.sokoban.solve", side_effect=AssertionError("no solver")
        ):
            self.assertEqual(parse_policy_output(raw, "sokoban-plan-v1"), ACTIONS["left"])
        self.assertEqual(json.loads(raw), plan)

    def test_plan_schema_board_and_action_direction_are_checked(self):
        original = make_plan(self.board, "left")
        bad_plans = []
        for key in original:
            missing = copy.deepcopy(original)
            del missing[key]
            bad_plans.append(missing)
        for key, value in (
            ("board", self.board[:-1]), ("board", ["########"] * 6),
            ("board", ["########", "# .    #", "# $    #", "#   @  #", "####### ", "########"]),
            ("decision", {"direction": "left", "effect": "blocked"}),
            ("decision", {"direction": [], "effect": "move"}),
            ("expected_next_state", {"player": [6, 3], "box": [2, 2]}),
            ("expected_next_state", {"player": [3, 8], "box": [2, 2]}),
            ("expected_next_state", {"player": [True, 3], "box": [2, 2]}),
            ("action", ACTIONS["right"]), ("action", {"type": "back"}),
        ):
            bad_plans.append({**original, key: value})
        bad_plans.append({**original, "explanation": "move"})
        for plan in bad_plans:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                parse_policy_output(json.dumps(plan), "sokoban-plan-v1")
        with self.assertRaises(ValueError):
            parse_policy_output(json.dumps(original))
        with self.assertRaises(ValueError):
            parse_policy_output(json.dumps(ACTIONS["left"]), "sokoban-plan-v1")


if __name__ == "__main__":
    unittest.main()
