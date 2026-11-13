"""Single-node AlvarezGUI platform: control plane API + console, scheduler, consumer and node heartbeat."""

from __future__ import annotations

import functools
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable
from datetime import timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import uvicorn

from alvarezgui.app import create_app
from alvarezgui.config import Settings
from alvarezgui.console_store import ConsoleStore, NodeHeartbeat
from alvarezgui.consumer import Consumer
from alvarezgui.db import connect
from alvarezgui.model_http import HttpModelService
from alvarezgui.node_agent.factory import runtime_from_settings
from alvarezgui.node_agent.runtime import Runtime
from alvarezgui.queue import client
from alvarezgui.scheduler import Scheduler

log = logging.getLogger("alvarezgui.server")
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "shopping"
HEARTBEAT_SECONDS = 10
# ponytail: one static guest IP per node, so one sandbox at a time; raise with a TAP/IP allocator.
SANDBOX_CAPACITY = 1


def node_resources(settings: Settings) -> dict[str, object]:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        meminfo[name] = int(value.split()[0]) * 1024
    disk = shutil.disk_usage(settings.data_dir)
    return {
        "cpu_count": os.cpu_count(),
        "load_1m": os.getloadavg()[0],
        "memory_total_bytes": meminfo.get("MemTotal"),
        "memory_available_bytes": meminfo.get("MemAvailable"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "kvm_available": os.path.exists("/dev/kvm"),
        "runtime_backend": settings.runtime_backend,
    }


def heartbeat(settings: Settings) -> None:
    with connect(settings) as connection:
        ConsoleStore(connection).heartbeat_node(
            settings.node_id,
            NodeHeartbeat(
                role="microvm",
                address=settings.node_address,
                capacity={"sandboxes": SANDBOX_CAPACITY},
                resources=node_resources(settings),
                heartbeat_interval_seconds=HEARTBEAT_SECONDS,
            ),
        )


def forever(name: str, step: Callable[[], object], interval: float) -> threading.Thread:
    def loop() -> None:
        while True:
            try:
                step()
            except Exception:
                log.exception("%s loop failed; retrying", name)
                time.sleep(5)
            time.sleep(interval)

    thread = threading.Thread(target=loop, name=name, daemon=True)
    thread.start()
    return thread


def serve_fixture(port: int) -> ThreadingHTTPServer:
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(FIXTURE_DIR))
    # Bound on all interfaces because the guest-facing TAP address only exists while a sandbox runs.
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, name="fixture", daemon=True).start()
    return server


def destroy_active_sandboxes(settings: Settings, runtime: Runtime) -> None:
    with connect(settings) as connection:
        rows = ConsoleStore(connection).list_sandboxes()
    for row in rows:
        try:
            runtime.destroy(row["sandbox_id"])
        except Exception:
            log.exception("failed to destroy %s on shutdown", row["sandbox_id"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings()
    if settings.console_password is None:
        raise SystemExit("CONSOLE_PASSWORD must be set before exposing the console")
    runtime = runtime_from_settings(settings)
    model_service = (
        HttpModelService(settings.model_service_url, settings.model_request_timeout_seconds)
        if settings.model_backend == "http"
        else None
    )
    redis_client = client(settings)
    consumer = Consumer(
        settings,
        runtime,
        redis_client,
        node_id=settings.node_id,
        model_service=model_service,
        # Inference may include loading a new checkpoint; the lease must outlive the slowest request.
        lease_ttl=timedelta(seconds=settings.model_request_timeout_seconds + 120),
    )
    scheduler = Scheduler(settings, redis_client, max_active=SANDBOX_CAPACITY)
    forever("heartbeat", lambda: heartbeat(settings), HEARTBEAT_SECONDS)
    forever("scheduler", scheduler.run_once, 1.0)
    forever("consumer", consumer.run_once, 0.5)
    fixture = serve_fixture(settings.fixture_port)
    app = create_app(settings, runtime, model_service, redis_client)
    try:
        uvicorn.run(app, host=settings.console_host, port=settings.console_port, log_level="info")
    finally:
        fixture.shutdown()
        destroy_active_sandboxes(settings, runtime)


if __name__ == "__main__":
    main()
