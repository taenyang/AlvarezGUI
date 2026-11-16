"""Run against a local Docker daemon: PYTHONPATH=src python scripts/docker_smoke.py --help."""

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
from uuid import uuid4

from alvarezgui.node_agent.docker import DockerAndroidRuntime, DockerConfig
from alvarezgui.node_agent.runtime import SandboxState


def main():
    parser = argparse.ArgumentParser(description="Real Docker Android lifecycle and Settings task smoke")
    parser.add_argument("--image", required=True)
    parser.add_argument("--state-dir", type=Path, default=Path("var"))
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--api-timeout", type=float, default=30)
    parser.add_argument("--privileged", action="store_true")
    args = parser.parse_args()
    runtime = DockerAndroidRuntime(DockerConfig(
        binary="docker", image=args.image, state_dir=args.state_dir,
        privileged=args.privileged, api_timeout=args.api_timeout,
        guest_ready_timeout=args.startup_timeout,
    ))
    sandbox_id = f"docker-smoke-{uuid4().hex}"
    report_dir = args.state_dir / "runs" / sandbox_id
    report_dir.mkdir(parents=True)
    report = {"sandbox_id": sandbox_id, "image": args.image, "checks": [], "passed": False}
    created = False

    def record(stage, **details):
        result = {"stage": stage, **details}
        report["checks"].append(result)
        print(json.dumps(result), flush=True)

    def screenshot(stage):
        observation = runtime.observe(sandbox_id)
        path = Path(url2pathname(urlsplit(observation["image_uri"]).path))
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR", "Invalid PNG"
        width, height = struct.unpack(">II", data[16:24])
        assert width > 0 and height > 0 and data[-8:-4] == b"IEND", "Incomplete screenshot"
        assert hashlib.sha256(data).hexdigest() == observation["screen_hash"]
        record(stage, width=width, height=height, **observation)
        return width, height

    try:
        print(json.dumps({"stage": "creating", "sandbox_id": sandbox_id}), flush=True)
        sandbox = runtime.create(sandbox_id)
        created = True
        assert sandbox.state is SandboxState.READY
        assert runtime.health(sandbox_id).healthy
        record("ready")
        runtime.initialize_task(sandbox_id, "android_launch_smoke", "builtin:settings", "Open Settings")
        width, height = screenshot("home")
        runtime._adb(
            sandbox_id, "shell", "am", "start", "-n", "com.android.settings/.Settings",
        )
        deadline = time.monotonic() + args.api_timeout
        while True:
            if runtime._foreground_package(sandbox_id) == "com.android.settings":
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Settings activity did not become foreground")
            time.sleep(1)
        result = runtime.execute(sandbox_id, {
            "type": "tap", "x": width // 2, "y": height // 2,
        })
        assert result["done"] and result["reward"]["success"], result
        record("settings_task", **result)
        screenshot("settings")
        assert runtime.pause(sandbox_id).state is SandboxState.PAUSED
        assert runtime.health(sandbox_id).healthy
        record("paused")
        assert runtime.resume(sandbox_id).state is SandboxState.RUNNING
        assert runtime.health(sandbox_id).healthy
        record("resumed")
        runtime.execute(sandbox_id, {"type": "back"})
        screenshot("after_back")
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if created:
                assert runtime.destroy(sandbox_id).state is SandboxState.STOPPED
                assert not runtime.health(sandbox_id).healthy
                record("destroyed")
            assert not runtime._docker(
                "ps", "-aq", "--filter", f"label=alvarezgui.sandbox_id={sandbox_id}",
            ).stdout.strip(), "Smoke container was not removed"
            report["passed"] = "error" not in report
        except Exception as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            path = report_dir / "report.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Report: {path}", flush=True)


if __name__ == "__main__":
    main()
