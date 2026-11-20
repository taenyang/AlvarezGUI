"""Build the Firecracker template that sandboxes are cloned from, then list older templates.

Safe while the worker runs on the same state dir. Older templates are never deleted here: remove
one by hand once no sandbox started from it is still running.
"""
import argparse
import json
from pathlib import Path

from alvarezgui.config import Settings
from alvarezgui.node_agent.factory import runtime_from_settings
from alvarezgui.node_agent.firecracker import FirecrackerAndroidRuntime


def main(args):
    # As the service sees it: the repository's .env (secrets), overridden by --env-file.
    env_files = (Path(__file__).resolve().parents[1] / ".env", args.env_file)
    runtime = runtime_from_settings(Settings(_env_file=env_files))
    if not isinstance(runtime, FirecrackerAndroidRuntime):
        raise SystemExit("RUNTIME_BACKEND is not firecracker")
    template = runtime.build_template(args.settle_seconds)
    print(json.dumps(json.loads((template / "manifest.json").read_text()), indent=2))
    others = sorted(str(path) for path in template.parent.iterdir() if path != template)
    print("other templates (delete by hand when unused):", *others or ["none"], sep="\n  ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=float, default=90.0)
    main(parser.parse_args())
