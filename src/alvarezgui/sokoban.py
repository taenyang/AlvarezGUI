"""Single-box Sokoban teacher. Run as a module to reproduce the level manifest."""

from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path
import random


LEVEL_COUNTS = {"train": 48, "validation": 8, "test": 8}
DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTIONS = {
    "up": {"type": "tap", "x": 360, "y": 836},
    "down": {"type": "tap", "x": 360, "y": 980},
    "left": {"type": "tap", "x": 176, "y": 980},
    "right": {"type": "tap", "x": 544, "y": 980},
}
Position = tuple[int, int]


def parse_board(board: list[str]) -> tuple[frozenset[Position], Position, Position, Position]:
    """Validate a 6x8 enclosed board, including floor connectivity ignoring the box."""
    if not isinstance(board, list) or len(board) != 6 or any(
        not isinstance(row, str) or len(row) != 8 for row in board
    ):
        raise ValueError("board must contain six rows of eight characters")
    cells = {(r, c): cell for r, row in enumerate(board) for c, cell in enumerate(row)}
    if any(cell not in "# @$.*+" for cell in cells.values()):
        raise ValueError("unknown board symbol")
    if any(cell != "#" for (r, c), cell in cells.items() if r in (0, 5) or c in (0, 7)):
        raise ValueError("board must be enclosed by walls")
    players = [pos for pos, cell in cells.items() if cell in "@+"]
    boxes = [pos for pos, cell in cells.items() if cell in "$*"]
    goals = [pos for pos, cell in cells.items() if cell in ".*+"]
    if len(players) != 1 or len(boxes) != 1 or len(goals) != 1:
        raise ValueError("board must have exactly one player, box, and goal")
    walls = frozenset(pos for pos, cell in cells.items() if cell == "#")
    reachable = {players[0]}
    pending = [players[0]]
    while pending:
        r, c = pending.pop()
        for dr, dc in DIRECTIONS.values():
            neighbor = (r + dr, c + dc)
            if neighbor not in walls and neighbor not in reachable:
                reachable.add(neighbor)
                pending.append(neighbor)
    if reachable != cells.keys() - walls:
        raise ValueError("all floor positions must be reachable ignoring the box")
    return walls, players[0], boxes[0], goals[0]


def transition(
    walls: frozenset[Position], player: Position, box: Position, direction: str
) -> tuple[Position, Position]:
    """Move once on a validated board; a blocked walk or push is a no-op."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction: {direction!r}")
    dr, dc = DIRECTIONS[direction]
    next_player = (player[0] + dr, player[1] + dc)
    if next_player in walls:
        return player, box
    if next_player == box:
        next_box = (box[0] + dr, box[1] + dc)
        if next_box in walls:
            return player, box
        return next_player, next_box
    return next_player, box


def solve(board: list[str]) -> list[str]:
    """Return the shortest sequence of individual moves, with deterministic ties."""
    walls, player, box, goal = parse_board(board)
    start = (player, box)
    parents = {start: None}
    pending = deque([start])
    while pending:
        state = pending.popleft()
        if state[1] == goal:
            path = []
            while parents[state] is not None:
                state, direction = parents[state]
                path.append(direction)
            return path[::-1]
        for direction in DIRECTIONS:
            next_state = transition(walls, *state, direction)
            if next_state not in parents:
                parents[next_state] = (state, direction)
                pending.append(next_state)
    raise ValueError("board is unsolvable")


def render_board(
    walls: frozenset[Position], player: Position, box: Position, goal: Position
) -> list[str]:
    """Render a validated state, preserving goal overlap with '*' or '+'."""
    if player == box or any(pos in walls for pos in (player, box, goal)):
        raise ValueError("pieces cannot overlap each other or walls")
    if any(not (0 <= r < 6 and 0 <= c < 8) for r, c in (*walls, player, box, goal)):
        raise ValueError("position outside the board")
    board = [["#" if (r, c) in walls else " " for c in range(8)] for r in range(6)]
    board[goal[0]][goal[1]] = "."
    board[box[0]][box[1]] = "*" if box == goal else "$"
    board[player[0]][player[1]] = "+" if player == goal else "@"
    rows = ["".join(row) for row in board]
    parse_board(rows)
    return rows


def load_levels(path: Path) -> list[dict]:
    """Load the complete v1 manifest, rejecting solution leakage and shared geometry."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {"version", "levels"}:
        raise ValueError("manifest must contain version and levels")
    if manifest["version"] != "sokoban-sft-v1" or not isinstance(manifest["levels"], list):
        raise ValueError("invalid Sokoban manifest version or levels")
    levels = manifest["levels"]
    expected_ids = {f"{split}-{index:03d}" for split, count in LEVEL_COUNTS.items() for index in range(count)}
    ids = set()
    geometries = set()
    counts = Counter()
    for level in levels:
        if not isinstance(level, dict) or set(level) != {"id", "split", "board"}:
            raise ValueError("level must contain only id, split, and board")
        level_id, split = level["id"], level["split"]
        if not isinstance(split, str) or split not in LEVEL_COUNTS:
            raise ValueError("unknown level split")
        if not isinstance(level_id, str) or level_id not in expected_ids or not level_id.startswith(f"{split}-"):
            raise ValueError("invalid level id or split mismatch")
        if level_id in ids:
            raise ValueError("duplicate level id")
        walls, _, _, goal = parse_board(level["board"])
        geometry = (walls, goal)
        if geometry in geometries:
            raise ValueError("duplicate static geometry")
        if not 4 <= len(solve(level["board"])) <= 10:
            raise ValueError("level solution must require 4 to 10 moves")
        ids.add(level_id)
        geometries.add(geometry)
        counts[split] += 1
    if counts != LEVEL_COUNTS:
        raise ValueError("incorrect split counts")
    return levels


def _generate_levels() -> list[dict]:
    """Reproduce v1 with a fixed seed and row-major sampling, without stored answers."""
    rng = random.Random(20261117)
    boundary = {(r, c) for r in range(6) for c in range(8) if r in (0, 5) or c in (0, 7)}
    interior = [(r, c) for r in range(1, 5) for c in range(1, 7)]
    levels = []
    geometries = set()
    for split, count in LEVEL_COUNTS.items():
        for index in range(count):
            for _ in range(10000):
                walls = frozenset(boundary | set(rng.sample(interior, rng.randint(1, 5))))
                player, box, goal = rng.sample([pos for pos in interior if pos not in walls], 3)
                if (walls, goal) in geometries:
                    continue
                try:
                    board = render_board(walls, player, box, goal)
                    length = len(solve(board))
                except ValueError:
                    continue
                if 4 <= length <= 10:
                    break
            else:
                raise RuntimeError("could not generate a suitable level")
            geometries.add((walls, goal))
            levels.append({"id": f"{split}-{index:03d}", "split": split, "board": board})
    return levels


if __name__ == "__main__":
    print(json.dumps({"version": "sokoban-sft-v1", "levels": _generate_levels()}, indent=2))
