"""Compare two ``/internal/v1/model/infer`` endpoints on real screenshots.

Both sides speak the same protocol, so this checks the whole sglang adapter path against HF serving.
Run from the repo root with ``set -a; . var/platform.env; set +a`` and ``PYTHONPATH=src .venv/bin/python``:

  scripts/sglang_parity.py cases --eval eval_dd67d274ff574c80ad2800273c04c903 --out DIR
  scripts/sglang_parity.py serve NAME '{"VERSION": "MODEL_DIR"}' [--sglang-arg FLAG]   # prints URL#TOKEN
  scripts/sglang_parity.py compare DIR --a URL[#TOKEN]=VERSION --b URL[#TOKEN]=VERSION --label greedy-base
  scripts/sglang_parity.py stop NAME

``compare`` runs temperature 0 once per case and temperature 1.0 ``--samples`` times, then writes
``DIR/<label>.json`` and checks the doc's thresholds (greedy match >= 0.95, mean TV <= 0.1).
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alvarezgui.model import InferenceRequest, InvalidActionOutputError
from alvarezgui.model_http import HttpModelService

KUBECTL = ["kubectl", "--kubeconfig", "/home/github/h800.yaml"]
TEMPLATE = "ainf-run-data-cleaning-kbench-20261118-t141"
PARITY_ROOT = Path("/ainf-cfs/yangtianze/alvarez-runtime/sglang-parity")
SOURCE = Path(__file__).resolve().parents[1] / "src/alvarezgui"


def cases(eval_id: str, out: Path, states: int) -> None:
    """``states`` distinct screens from the first episode of each level, saved as PNG with the instruction."""
    from alvarezgui.artifacts import get
    from alvarezgui.config import Settings
    from alvarezgui.db import connect

    settings = Settings()
    with connect(settings) as connection:
        rows = connection.execute("""
            SELECT t.app_image, t.instruction, ep.episode_id, s.step_id, s.observation
            FROM tasks t JOIN episodes ep USING (task_id) JOIN steps s USING (episode_id)
            WHERE t.eval_id = %s ORDER BY t.app_image, t.created_at, ep.episode_id, s.step_id""", (eval_id,)).fetchall()
    chosen: dict[str, list[dict]] = {}
    episode: dict[str, str] = {}
    for row in rows:
        level = row["app_image"]
        episode.setdefault(level, row["episode_id"])
        picked = chosen.setdefault(level, [])
        hashes = {item["observation"]["screen_hash"] for item in picked}
        if row["episode_id"] == episode[level] and len(picked) < states and row["observation"]["screen_hash"] not in hashes:
            picked.append(row)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for level, picked in sorted(chosen.items()):
        for row in picked:
            name = f"{level.rsplit(':', 1)[-1]}-{row['step_id']}.png"
            (out / name).write_bytes(get(settings, row["observation"]["image_uri"]))
            index.append({"case": name, "instruction": row["instruction"], "level": level,
                          "episode_id": row["episode_id"], "step_id": row["step_id"]})
    (out / "cases.json").write_text(json.dumps(index, indent=2))
    print(f"{len(index)} cases from {len(chosen)} levels in {out}")


def _endpoint(spec: str) -> tuple[HttpModelService, str]:
    url, version = spec.rsplit("=", 1)
    url, _, token = url.partition("#")
    return HttpModelService(url, timeout=300, token=token or None), version


def _ask(service: HttpModelService, version: str, case: dict, image: str, temperature: float,
         seed: int | None) -> dict:
    request = InferenceRequest(request_id=f"{case['case']}-{seed}", model_version=version,
                               instruction=case["instruction"], image_uri=image, temperature=temperature,
                               seed=seed, timeout_ms=300_000)
    try:
        response = service.infer(request)
        return {"raw": response.raw_output, "action": json.dumps(response.action, sort_keys=True)}
    except InvalidActionOutputError as exc:
        return {"raw": exc.raw_output, "action": "invalid"}


def total_variation(a: list[str], b: list[str]) -> float:
    left, right = Counter(a), Counter(b)
    return 0.5 * sum(abs(left[key] / len(a) - right[key] / len(b)) for key in left.keys() | right.keys())


def compare(directory: Path, a: str, b: str, label: str, samples: int, workers: tuple[int, int],
            b_seed_offset: int = 0) -> bool:
    index = json.loads((directory / "cases.json").read_text())
    images = {case["case"]: "data:image/png;base64," + base64.b64encode((directory / case["case"]).read_bytes()).decode()
              for case in index}
    jobs = [(case, temperature, seed) for case in index
            for temperature, seed in [(0.0, None), *((1.0, seed) for seed in range(samples))]]
    started = time.monotonic()

    def run(spec: str, pool_size: int, offset: int) -> list[dict]:
        service, version = _endpoint(spec)
        with ThreadPoolExecutor(pool_size) as pool:
            return list(pool.map(lambda job: _ask(service, version, job[0], images[job[0]["case"]], job[1],
                                                  None if job[2] is None else job[2] + offset), jobs))

    with ThreadPoolExecutor(2) as sides:
        left, right = (future.result() for future in [sides.submit(run, a, workers[0], 0),
                                                      sides.submit(run, b, workers[1], b_seed_offset)])
    per_case, matches, distances = [], 0, []
    for start in range(0, len(jobs), samples + 1):
        case = jobs[start][0]
        greedy_a, greedy_b = left[start], right[start]
        sampled_a = [item["action"] for item in left[start + 1:start + samples + 1]]
        sampled_b = [item["action"] for item in right[start + 1:start + samples + 1]]
        distance = total_variation(sampled_a, sampled_b) if samples else None
        matches += greedy_a["raw"] == greedy_b["raw"]
        if distance is not None:
            distances.append(distance)
        per_case.append({"case": case["case"], "greedy_a": greedy_a["raw"], "greedy_b": greedy_b["raw"],
                         "same_action": greedy_a["action"] == greedy_b["action"], "tv": distance,
                         "samples_a": dict(Counter(sampled_a)), "samples_b": dict(Counter(sampled_b))})
    report = {"a": a.split("#")[0] + "=" + a.rsplit("=", 1)[1], "b": b.split("#")[0] + "=" + b.rsplit("=", 1)[1],
              "cases": len(per_case), "samples": samples, "seconds": round(time.monotonic() - started),
              "greedy_match": matches / len(per_case),
              "greedy_same_action": sum(item["same_action"] for item in per_case) / len(per_case),
              "mean_tv": sum(distances) / len(distances) if distances else None, "per_case": per_case}
    report["pass"] = report["greedy_match"] >= 0.95 and (report["mean_tv"] is None or report["mean_tv"] <= 0.1)
    (directory / f"{label}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({key: value for key, value in report.items() if key != "per_case"}, indent=2))
    return report["pass"]


def _kube(*args: str, stdin: str | None = None) -> str:
    return subprocess.run([*KUBECTL, *args], input=stdin, capture_output=True, text=True, check=True).stdout


def serve(name: str, models: dict, sglang_args: list[str]) -> None:
    """One adapter pod built like an evaluation service; ``models`` is spec.json's ``models``
    (``{version: path}`` or ``{version: {path, backend, merged_copy?}}``)."""
    from alvarezgui.h800_launcher import CFS_ROOT, SGLANG_POD_SCRIPT, deployment_manifest

    work = PARITY_ROOT / name
    if work.exists():
        raise SystemExit(f"{work} exists; stop {name} first")
    shutil.copytree(SOURCE, work / "src/alvarezgui", ignore=shutil.ignore_patterns("__pycache__"))
    token = work / "token"
    token.touch(mode=0o600)
    token.write_text(base64.urlsafe_b64encode(__import__("os").urandom(24)).decode())
    (work / "spec.json").write_text(json.dumps({"models": models, "sglang_memory": 0.8, "sglang_args": sglang_args},
                                               indent=2))
    (work / "pod.sh").write_text(SGLANG_POD_SCRIPT.format(work=work, root=CFS_ROOT, src=work / "src"))
    deployment = f"alvarez-parity-{name}"
    template = json.loads(_kube("get", "deployment", TEMPLATE, "-o", "json"))
    _kube("apply", "-f", "-", stdin=json.dumps(
        deployment_manifest(template, deployment, f"parity-{name}", work / "pod.sh")))
    print(f"created {deployment}; waiting for {work / 'ready.json'}", file=sys.stderr)
    deadline = time.monotonic() + 1800  # First image pull is about 6 minutes.
    while not (work / "ready.json").exists():
        if (work / "FAILED").exists() or time.monotonic() > deadline:
            raise SystemExit("sglang adapter failed: " + ((work / "FAILED").read_text() if (work / "FAILED").exists()
                                                           else "not ready after 30 minutes"))
        time.sleep(10)
    pods = json.loads(_kube("get", "pods", "-l", f"gpu-be-label={deployment}", "-o", "json"))["items"]
    address = next(pod["status"]["podIP"] for pod in pods if pod["status"].get("phase") == "Running")
    print(f"http://{address}:{json.loads((work / 'ready.json').read_text())['port']}#{token.read_text()}")


def stop(name: str) -> None:
    _kube("delete", "deployment", f"alvarez-parity-{name}", "--ignore-not-found", "--wait=false")
    (PARITY_ROOT / name / "token").unlink(missing_ok=True)
    print(f"deleted alvarez-parity-{name}; logs stay in {PARITY_ROOT / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("cases")
    p.add_argument("--eval", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--states", type=int, default=2)
    p = commands.add_parser("serve")
    p.add_argument("name")
    p.add_argument("models", type=json.loads, help='spec.json models, e.g. \'{"v": "/ainf-cfs/.../model"}\'')
    p.add_argument("--sglang-arg", action="append", default=[], help="extra sglang.launch_server flag; may repeat")
    p = commands.add_parser("compare")
    p.add_argument("directory", type=Path)
    p.add_argument("--a", required=True, help="reference, e.g. http://127.0.0.1:18081=sokoban-sd1-merged")
    p.add_argument("--b", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--workers", type=int, nargs=2, default=(2, 16), metavar=("A", "B"))
    p.add_argument("--b-seed-offset", type=int, default=0, help="e.g. 1000 for a same-endpoint noise baseline")
    p = commands.add_parser("stop")
    p.add_argument("name")
    args = parser.parse_args()
    if args.command == "cases":
        cases(args.eval, args.out, args.states)
    elif args.command == "serve":
        serve(args.name, args.models, args.sglang_arg)
    elif args.command == "compare":
        sys.exit(0 if compare(args.directory, args.a, args.b, args.label, args.samples, tuple(args.workers),
                         args.b_seed_offset) else 1)
    else:
        stop(args.name)


if __name__ == "__main__":
    main()
