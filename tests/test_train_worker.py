import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alvarezgui import train_worker
from alvarezgui.train_worker import Heartbeat, pending_round, serve, train_round


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Stop(Exception):
    pass


class TrainWorkerTest(unittest.TestCase):
    def test_pending_round_order_and_skips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertIsNone(pending_round(run))
            for number in (0, 1, 2, 10):
                (run / f"round-{number}").mkdir()
            (run / "round-x").mkdir()
            (run / "round-0/spec.json").write_text("{}")
            (run / "round-0/COMPLETE").write_text("{}")
            (run / "round-1/spec.json").write_text("{}")
            (run / "round-1/FAILED").write_text("boom")
            # round-2 has no spec (zero-variance round); round-10 sorts after round-2 numerically.
            (run / "round-10/spec.json").write_text("{}")
            self.assertEqual(pending_round(run), run / "round-10")
            (run / "round-2/spec.json").write_text('{"run": ')  # Mid-write: wait, do not skip ahead.
            self.assertIsNone(pending_round(run))
            (run / "round-2/spec.json").write_text("{}")
            self.assertEqual(pending_round(run), run / "round-2")

    def test_heartbeat_stops_when_progress_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clock, beat = Clock(), Path(directory, "alive")
            log = Path(directory, "run.log")
            heartbeat = Heartbeat(beat, stall=100, clock=clock)
            self.assertTrue(heartbeat.beat())
            heartbeat.watch = [log]
            clock.now = 60
            log.write_text("step 1\n")
            self.assertTrue(heartbeat.beat())  # The log grew: progress.
            clock.now = 150
            self.assertTrue(heartbeat.beat())  # 90 s since growth: still within the stall window.
            beat.unlink()
            clock.now = 170
            self.assertFalse(heartbeat.beat())  # 110 s without growth: stalled.
            self.assertFalse(beat.exists())
            heartbeat.progress()
            self.assertTrue(heartbeat.beat())
            self.assertTrue(beat.exists())

    def test_rounds_run_in_order_with_their_own_log_and_tmpdir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for number in (0, 1):
                (run / f"round-{number}").mkdir()
                (run / f"round-{number}/spec.json").write_text(json.dumps({"n": number}))
            trained = []

            def run_pod(work: Path) -> None:
                trained.append((work.name, os.environ["TMPDIR"]))
                print(f"training {work.name}")
                os.write(2, b"native stderr\n")  # e.g. CUDA / C++ warnings
                if work.name == "round-1":
                    raise RuntimeError("worker-side failure")  # Not marked by run_pod itself.
                (work / "COMPLETE").write_text("{}")

            sleeps = []

            def sleep(_seconds: float) -> None:
                sleeps.append(_seconds)
                raise Stop

            heartbeat = Heartbeat(run / "alive")
            with patch("alvarezgui.h800_launcher.run_pod", side_effect=run_pod), \
                    patch.object(train_worker.time, "sleep", side_effect=sleep), self.assertRaises(Stop):
                serve(run, heartbeat)
            self.assertEqual(trained, [("round-0", str(run / "round-0/tmp")), ("round-1", str(run / "round-1/tmp"))])
            for number in (0, 1):
                log = (run / f"round-{number}/run.log").read_text()
                self.assertIn("(resident worker)", log)
                self.assertIn(f"training round-{number}", log)
                self.assertIn("native stderr", log)
            self.assertIn("worker-side failure", (run / "round-1/FAILED").read_text())
            self.assertEqual(sleeps, [5.0])  # Idle once both rounds were done: no retry of round-1.
            self.assertEqual(heartbeat.watch, [])
            self.assertNotEqual(os.environ.get("TMPDIR"), str(run / "round-1/tmp"))  # Restored after each round.

    def test_round_marked_failed_by_run_pod_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory, "round-0")
            work.mkdir()

            def run_pod(work: Path) -> None:
                (work / "FAILED").write_text("trainer traceback")
                raise ValueError("trainer")

            with patch("alvarezgui.h800_launcher.run_pod", side_effect=run_pod):
                train_round(work, Heartbeat(Path(directory, "alive")))
            self.assertEqual((work / "FAILED").read_text(), "trainer traceback")


if __name__ == "__main__":
    unittest.main()
