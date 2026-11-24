"""Resident H800 training worker for one online RL run (docs/h800-resident-pods.md, T1).

One process per run waits for each round's ``spec.json`` under the run directory and trains it with the
same ``run_pod`` a per-round pod runs, so a round's result does not depend on which of the two ran it.
The process (CUDA context, imports, fla/Triton kernel caches) stays warm across rounds; weights are still
loaded from CFS every round, which keeps every round exactly what a fresh pod would compute.

ponytail: weights reload per round; keep the trained model in GPU memory if that load shows up in round timings.

Usage inside the pod: ``python -m alvarezgui.train_worker <run_dir>``.
"""

from __future__ import annotations

import gc
import json
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Container-local, never on CFS: a hung CFS mount must not keep the worker looking alive.
HEARTBEAT = Path("/tmp/alive")
# A healthy round always grows run.log or metrics.jsonl within this time; the longest silent stretch is
# model load + preprocessing + the old_logprob pass (about 4 minutes for 500 rows).
STALL_SECONDS = 900.0


class Heartbeat:
    """Touches ``path`` while the worker shows progress; a livenessProbe on its mtime restarts a stalled worker.

    Progress is an explicit ``progress()`` call (idle polling) or growth of any ``watch`` file (a running
    round). Statting a watched CFS file that hangs blocks ``beat``, so a hung mount stops the heartbeat too.
    """

    def __init__(self, path: Path = HEARTBEAT, stall: float = STALL_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.path, self.stall, self.clock = path, stall, clock
        self.last = clock()
        self.watch: list[Path] = []
        self.sizes: tuple[int, ...] = ()

    def progress(self) -> None:
        self.last = self.clock()

    def beat(self) -> bool:
        sizes = tuple(path.stat().st_size if path.exists() else -1 for path in self.watch)
        if sizes != self.sizes:
            self.sizes, self.last = sizes, self.clock()
        if self.clock() - self.last >= self.stall:
            return False
        self.path.touch()
        return True

    def run(self, interval: float = 10.0) -> None:
        while True:
            self.beat()
            time.sleep(interval)


def _round_number(path: Path) -> int | None:
    suffix = path.name.removeprefix("round-")
    return int(suffix) if path.name.startswith("round-") and suffix.isdigit() else None


def pending_round(run_dir: Path) -> Path | None:
    """The earliest round with a readable spec that has neither completed nor failed.

    Rounds without ``spec.json`` are skipped: the launcher never writes one for a round whose groups all
    had zero reward variance. A spec that does not parse yet is mid-write, so wait rather than skip it.
    """
    rounds = sorted((path for path in run_dir.glob("round-*") if _round_number(path) is not None),
                    key=_round_number)
    for work in rounds:
        if (work / "COMPLETE").exists() or (work / "FAILED").exists() or not (work / "spec.json").exists():
            continue
        try:
            json.loads((work / "spec.json").read_text())
        except json.JSONDecodeError:
            return None
        return work
    return None


def train_round(work: Path, heartbeat: Heartbeat) -> None:
    """Run one round like its own pod would: its TMPDIR, stdout and stderr (C level included) in its run.log."""
    from alvarezgui.h800_launcher import run_pod

    (work / "tmp").mkdir(exist_ok=True)
    saved_tmp = tempfile.tempdir, os.environ.get("TMPDIR")
    tempfile.tempdir = os.environ["TMPDIR"] = str(work / "tmp")
    heartbeat.watch = [work / "run.log", work / "metrics.jsonl"]
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    try:
        with (work / "run.log").open("ab") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            print(f"\npod {socket.gethostname()} started: {datetime.now(timezone.utc).isoformat()} "
                  f"(resident worker)", flush=True)
            try:
                run_pod(work)
            except Exception:
                traceback.print_exc()
                if not (work / "FAILED").exists() and not (work / "COMPLETE").exists():
                    # run_pod marks only its own training failures; never retry a round in a hot loop.
                    (work / "FAILED").write_text(traceback.format_exc())
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        tempfile.tempdir = saved_tmp[0]
        if saved_tmp[1] is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = saved_tmp[1]
        heartbeat.watch = []
        heartbeat.progress()
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        # ponytail: logged only; a leak fails the next round with a CUDA OOM instead of silently slowing it.
        print(f"{work.name}: {torch.cuda.memory_allocated() / 2**30:.2f} GiB still allocated after the round",
              flush=True)


def serve(run_dir: Path, heartbeat: Heartbeat, poll: float = 5.0) -> None:
    """Train rounds as their specs appear; after the last one, idle until the launcher deletes the pod."""
    while True:
        heartbeat.progress()
        work = pending_round(run_dir)
        if work is None:
            time.sleep(poll)
            continue
        print(f"{work.name}: training", flush=True)
        train_round(work, heartbeat)
        print(f"{work.name}: {'completed' if (work / 'COMPLETE').exists() else 'failed'}", flush=True)


def main() -> None:
    run_dir = Path(sys.argv[1])
    heartbeat = Heartbeat()
    heartbeat.beat()
    threading.Thread(target=heartbeat.run, daemon=True).start()
    print(f"resident training worker for {run_dir} on {socket.gethostname()}", flush=True)
    serve(run_dir, heartbeat)


if __name__ == "__main__":
    main()
