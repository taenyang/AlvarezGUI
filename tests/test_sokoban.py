from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from alvarezgui.sokoban import (
    ACTIONS,
    DIRECTIONS,
    LEVEL_COUNTS,
    _generate_levels,
    load_levels,
    parse_board,
    render_board,
    solve,
    transition,
)


MANIFEST_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sokoban" / "levels.json"
BOARD = ["########", "#   .  #", "# $    #", "# @    #", "#      #", "########"]


class SokobanTest(unittest.TestCase):
    def test_action_contract(self):
        self.assertEqual(list(DIRECTIONS), ["up", "down", "left", "right"])
        self.assertEqual(list(DIRECTIONS.values()), [(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.assertEqual(ACTIONS, {
            "up": {"type": "tap", "x": 360, "y": 836},
            "down": {"type": "tap", "x": 360, "y": 980},
            "left": {"type": "tap", "x": 176, "y": 980},
            "right": {"type": "tap", "x": 544, "y": 980},
        })
        self.assertEqual(LEVEL_COUNTS, {"train": 48, "validation": 8, "test": 8})

    def test_parse_and_render_roundtrip_with_goal_overlaps(self):
        walls, player, box, goal = parse_board(BOARD)
        self.assertIsInstance(walls, frozenset)
        self.assertEqual((player, box, goal), ((3, 2), (2, 2), (1, 4)))
        for state in ((walls, player, box, goal), (walls, goal, box, goal), (walls, player, goal, goal)):
            with self.subTest(state=state[1:]):
                self.assertEqual(parse_board(render_board(*state)), state)
        self.assertEqual(render_board(walls, player, box, goal), BOARD)
        self.assertIn("+", "".join(render_board(walls, goal, box, goal)))
        self.assertIn("*", "".join(render_board(walls, player, goal, goal)))

    def test_invalid_boards(self):
        cases = [None, [], BOARD[:-1], [123] + BOARD[1:], ["#######"] + BOARD[1:]]
        for old, new in (("# @    #", "# @ @  #"), ("# $    #", "#      #"),
                         ("#   .  #", "#      #"), ("# $    #", "# $$   #"),
                         ("#   .  #", "#  ..  #"), ("#      #", "#  ?   #")):
            cases.append([row.replace(old, new) for row in BOARD])
        cases.append([" #######"] + BOARD[1:])
        cases.append(["########", "# .    #", "########", "# @$   #", "#      #", "########"])
        for board in cases:
            with self.subTest(board=board), self.assertRaises(ValueError):
                parse_board(board)

    def test_render_rejects_invalid_states(self):
        walls, player, box, goal = parse_board(BOARD)
        for bad_player in (box, (0, 0), (-1, 2), (6, 2)):
            with self.subTest(player=bad_player), self.assertRaises(ValueError):
                render_board(walls, bad_player, box, goal)

    def test_walk_push_and_collisions(self):
        walls, player, box, _ = parse_board(BOARD)
        self.assertEqual(transition(walls, player, box, "left"), ((3, 1), box))
        self.assertEqual(transition(walls, player, box, "up"), ((2, 2), (1, 2)))
        self.assertEqual(transition(walls, (2, 2), (1, 2), "up"), ((2, 2), (1, 2)))
        self.assertEqual(transition(walls, (1, 1), box, "up"), ((1, 1), box))
        self.assertEqual(transition(walls | {(2, 3)}, (2, 1), box, "right"), ((2, 1), box))
        with self.assertRaises(ValueError):
            transition(walls, player, box, "diagonal")

    def test_solver_shortest_path_and_deterministic_ties(self):
        # Two right pushes and one up push need two positioning moves: lower bound 5.
        path = solve(BOARD)
        self.assertEqual(path, ["up", "left", "up", "right", "right"])
        walls, player, box, goal = parse_board(BOARD)
        for direction in path:
            player, box = transition(walls, player, box, direction)
        self.assertEqual(box, goal)
        self.assertEqual(solve(BOARD), path)

    def test_solver_solved_and_deadlocked(self):
        walls, player, _, goal = parse_board(BOARD)
        self.assertEqual(solve(render_board(walls, player, goal, goal)), [])
        with self.assertRaisesRegex(ValueError, "unsolvable"):
            solve(render_board(walls, player, (1, 1), goal))

    def test_manifest_paths_splits_geometry_and_action_distribution(self):
        levels = load_levels(MANIFEST_PATH)
        self.assertEqual(Counter(level["split"] for level in levels), LEVEL_COUNTS)
        self.assertEqual(len({level["id"] for level in levels}), 64)
        self.assertEqual(len({tuple(level["board"]) for level in levels}), 64)
        geometries = set()
        goals = set()
        actions = {split: Counter() for split in LEVEL_COUNTS}
        for level in levels:
            with self.subTest(level=level["id"]):
                self.assertEqual(set(level), {"id", "split", "board"})
                walls, player, box, goal = parse_board(level["board"])
                geometry = (walls, goal)
                self.assertNotIn(geometry, geometries)
                geometries.add(geometry)
                goals.add(goal)
                self.assertTrue(any(0 < r < 5 and 0 < c < 7 for r, c in walls))
                self.assertNotEqual(box, goal)
                path = solve(level["board"])
                self.assertTrue(4 <= len(path) <= 10)
                for direction in path:
                    previous = (player, box)
                    player, box = transition(walls, player, box, direction)
                    self.assertNotEqual((player, box), previous)
                    self.assertEqual(parse_board(render_board(walls, player, box, goal)), (walls, player, box, goal))
                self.assertEqual(box, goal)
                actions[level["split"]].update(path)
        self.assertGreaterEqual(len(goals), 8)
        self.assertTrue(200 <= actions["train"].total() <= 500)
        for split, counts in actions.items():
            with self.subTest(split=split):
                self.assertEqual(set(counts), set(DIRECTIONS))
                self.assertTrue(all(count >= counts.total() * 0.15 for count in counts.values()))

    def test_manifest_generation_is_reproducible(self):
        self.assertEqual(_generate_levels(), load_levels(MANIFEST_PATH))

    def test_loader_rejects_malformed_manifests_and_leakage(self):
        original = json.loads(MANIFEST_PATH.read_text())
        cases = [[], {}, {**original, "version": "unknown"}, {**original, "levels": {}},
                 {**original, "levels": original["levels"][:-1]}]
        changes = [
            (0, "id", "train-999"), (0, "id", []), (0, "id", "train-001"),
            (0, "split", "test"), (0, "split", "unknown"), (0, "split", []),
            (0, "solution", ["up"]), (0, "board", BOARD[:-1]),
            (48, "board", original["levels"][0]["board"]),
        ]
        walls, player, _, goal = parse_board(BOARD)
        changes.extend([
            (0, "board", render_board(walls, player, goal, goal)),
            (0, "board", render_board(walls, player, (1, 1), goal)),
        ])
        for index, key, value in changes:
            manifest = deepcopy(original)
            manifest["levels"][index][key] = value
            cases.append(manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "levels.json"
            for index, manifest in enumerate(cases):
                path.write_text(json.dumps(manifest))
                with self.subTest(case=index), self.assertRaises(ValueError):
                    load_levels(path)


if __name__ == "__main__":
    unittest.main()
