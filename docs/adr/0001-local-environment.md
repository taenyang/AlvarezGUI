# ADR 0001: Local and migratable environment

- Status: Accepted
- Date: 2026-11-12

## Decision

The first control-plane environment uses Python 3.12 and `uv`. `pyproject.toml` is the source of declared dependencies and `uv.lock` is committed to reproduce their resolution. The repository-local `.venv` is disposable and is not committed.

PostgreSQL, Redis, and MinIO run through the root `compose.yaml` with pinned image digests from the available domestic registry, configurable bind addresses (localhost by default), health checks, and bind mounts under `ALVAREZ_DATA_DIR`. The default is `./var`; runtime nodes must use an explicit `/home/...` data directory. The expected layout is:

```text
$ALVAREZ_DATA_DIR/
├── postgres/
├── redis/
├── minio/
└── microvm/
    ├── images/
    ├── snapshots/
    └── runs/
```

Redis is used for streams, heartbeats, and short-lived scheduling state. PostgreSQL remains the source of truth. MinIO stores large artifacts such as screenshots, UI trees, trajectories, and checkpoints.

The Node Agent depends on the small runtime interface in `src/alvarezgui/node_agent/runtime.py`. Firecracker Android microVMs remain the default runtime; the in-memory runtime stub exists only under `tests/` for control-plane tests. A control-plane process can schedule both backend streams without running a local worker. Each runtime worker consumes only its configured backend: Firecracker requires a host with accessible `/dev/kvm`, while Docker requires a local Linux Docker daemon and has weaker isolation. Android containers do not run inside Compose; the Node Agent creates them dynamically.

## Consequences

- Moving runtime data means stopping services and moving or remounting one data directory.
- Secrets stay in an untracked `.env` file or an external secret manager.
- Model and training dependencies remain outside this control-plane environment.
- The production microVM adapter is Firecracker. `runtime_backend=firecracker` uses the repository's Firecracker plus ADB adapter and validates the binary, kernel image, and Android rootfs before boot. `runtime_backend=docker` requires `DOCKER_ANDROID_IMAGE` and validates Docker Linux-container mode, ADB, and the image contract before boot. Docker has pause/resume but no Firecracker-equivalent snapshot/restore, so snapshot workflows must use Firecracker. Historical `fake` backend values remain readable only for terminal provenance; new work cannot use them.

For Docker Desktop on Windows, the minimum untracked `.env` settings are:

```dotenv
ALVAREZ_RUNTIME_BACKEND=docker
DOCKER_ANDROID_IMAGE=redroid/redroid:16.0.0_64only-latest
DOCKER_PRIVILEGED=true
DOCKER_CONTAINER_PREFIX=alvarez-android
DOCKER_STARTUP_TIMEOUT_SECONDS=180
DOCKER_BINDERFS_PATH=/dev/binderfs
```

Prerequisites and image contract:

- Install Python 3.12, `uv`, Docker Desktop in Linux-container mode, and Android platform-tools (`adb`/`adb.exe` on `PATH`, or set `ADB_BINARY`). The Docker daemon must be local: remote Docker contexts are unsupported because ADB connects to the published port on `127.0.0.1`.
- The local prepared image was imported from `redroid/redroid@sha256:7b1e389bd15f37af3bcd06138f5b2ffa7cfba4332bd5ef54c53e99c2f160a15b` and is locally tagged `redroid/redroid:16.0.0_64only-latest` (image ID `sha256:281aa23c50705da1eb751bc58a5c22a3ff2c4ed1b0884f58fe5b24c797283970`). `docker load` did not retain a usable RepoDigest on this host, so use the imported tag until the digest reference has been registered by a verified pull. Import it into the local Docker daemon before starting tasks; sandbox creation uses `--pull=never`. A normal Linux image is not sufficient.
- The image must boot Android automatically, expose reachable TCP ADB on container port 5555 (`GUEST_ADB_PORT` can override it), report `sys.boot_completed=1`, and support `adb shell input`, `adb exec-out screencap -p`, `dumpsys`, and `uiautomator`. The runtime passes redroid memfd and guest-GPU boot properties. If the host exposes binder devices under `DOCKER_BINDERFS_PATH`, it makes them writable and bind-mounts that directory at `/dev/binderfs`; otherwise a privileged redroid image may mount its own binderfs instance. Shopping/Sokoban also require `org.chromium.webview_shell/.WebViewBrowserActivity` and guest access to `http://host.docker.internal:18090/`. `DOCKER_FIXTURE_HOST` overrides that hostname; configure the fixture URLs too if using a different fixture port.
- On standard Windows Docker Desktop, use an image with software emulation unless its hardware acceleration requirements have been independently verified. Redroid images requiring Binder/ashmem kernel modules and emulator images requiring `/dev/kvm` are **not** automatically compatible with Docker Desktop. Enabling privileged mode does not supply missing kernel features.
- Containers receive 2 CPUs and 4096 MiB each; set `SANDBOX_CAPACITY=1` initially and allow enough Docker Desktop memory. Pause/resume freezes the container during model inference; no snapshot/restore is available. Screenshots are captured temporarily under `ALVAREZ_DATA_DIR/runs` and uploaded by the Consumer to the shared MinIO bucket.

Keep the existing PostgreSQL/MinIO/console secrets in the untracked `.env` (do not retain an obsolete `RUNTIME_BACKEND=fake` value). Start the three infrastructure services with `docker compose up -d`, apply migrations through `009_docker_runtime.sql` in order for a new installation (only pending migrations on an existing installation), then run `uv sync` and `uv run python -m alvarezgui.server`. Configure the existing HTTP model service separately for real policy inference; selecting Docker does not install a model. For a multi-node deployment, run the control plane with `ALVAREZ_SERVER_MODE=control_plane` and `ALVAREZ_SCHEDULER_BACKENDS=firecracker,docker`; it does not register itself as a runtime node. Run each worker with `ALVAREZ_SERVER_MODE=worker`, one `ALVAREZ_RUNTIME_BACKEND`, and a unique `NODE_ID`. Workers share PostgreSQL and Redis, and each consumes only its backend stream. The console submits the selected backend and displays remote sandboxes as `REMOTE` until a node-local health API is added.

The configured image is trusted operator input. `DOCKER_PRIVILEGED=true` is available for images that require it and expands the container trust boundary. Docker isolation is weaker than Firecracker; this backend is for trusted local development, not hostile multi-tenant workloads. Windows may omit Linux-only memory/load telemetry. Mocked lifecycle and control-plane tests do not replace a real Docker Desktop create → boot → screenshot/input → pause/resume → destroy smoke test.

The Docker experiment target is the Ubuntu VM accessed with `/home/e2b-ubuntu24/id_ed25519`, on `ubuntu@127.0.0.1:2224` (host keys in `/home/e2b-ubuntu24/known_hosts`). Run the runtime **inside that VM**, using its local Docker daemon. The VM definition is `/home/e2b-ubuntu24/compose.yaml`; without host `/dev/kvm`, the existing `e2b-ubuntu24-vm-tcg` instance uses software virtualization and needs longer timeouts. The experiment workspace inside the VM is `/home/ubuntu/alvarezgui-docker-lab`.

For Ubuntu's generic kernel, install `linux-modules-extra-$(uname -r)` and load `binder_linux` before starting Redroid. Kernel upgrades need the matching extra-modules package; having the package for an older kernel is insufficient. Verify with `modinfo binder_linux`, `sudo modprobe binder_linux devices=binder,hwbinder,vndbinder`, and `grep binder /proc/filesystems`. Binderfs-enabled kernels may not expose `/dev/binder` on the VM itself: privileged Redroid mounts binderfs and gets its own container devices. Persist module loading in `/etc/modules-load.d/alvarez-android.conf` (`binder_linux` on one line), and the device names in `/etc/modprobe.d/alvarez-android.conf` (`options binder_linux devices=binder,hwbinder,vndbinder`).

The Android 16 image also needs `/dev/ashmem` in this VM despite `sys.use_memfd=1`. Its existing module source is `/opt/redroid-android16/modules/linux68/ashmem`. Rebuild against `/lib/modules/$(uname -r)/build` after a kernel upgrade, install the resulting `ashmem_linux.ko` under that kernel's `extra/` directory, run `depmod`, and add `ashmem_linux` to the module-load file. The 6.8.0-149 build workspace is `/home/ubuntu/alvarezgui-docker-lab/ashmem-6.8.0-149`.

`docker/android` prepares the pinned Redroid image for this VM: the PRNG seeder reads the kernel CSPRNG through a separate, service-owned device node, SurfaceFlinger shader-cache warm-up is disabled because its software-gralloc output buffer triggers an abort, and `ro.hw_timeout_multiplier=10` allows Android's Watchdog to tolerate the TCG VM's software-emulation latency. The timeout multiplier is for this software-emulated lab only; hardware-accelerated hosts should omit it. This image does not supply Binder or ashmem; those remain VM kernel prerequisites.

Run the standalone real-runtime check from the project directory in the VM (Python 3.12 with Pydantic 2, Docker access, and ADB are required):

```bash
docker build --pull=false -t local/alvarez-android:16-vm docker/android
PYTHONPATH=src .venv/bin/python scripts/docker_smoke.py \
  --image local/alvarez-android:16-vm \
  --privileged --startup-timeout 900 --api-timeout 90
```

The 2026-11-16 TCG run did not reach boot completion within 900 seconds; the container remained running without OOM and was cleaned up automatically. A longer timeout is a diagnostic option, not evidence of compatibility. Android logs may be absent from `docker logs`; for live diagnosis, use `docker exec <container> /system/bin/logcat -b all -v threadtime`. The runtime captures a bounded logcat excerpt before cleaning up a running container after a startup failure.

This creates a fresh Android sandbox, checks ADB boot completion and PNG screenshots, opens the Settings activity, verifies a tap task reward, pauses/resumes, sends Back, and destroys the container. The TCG lab smoke opens Settings with an explicit Android intent because the software-emulated launcher can expose no input channel while it is still initializing; hardware-accelerated hosts may additionally verify drawer navigation with `uiautomator`. Each attempt writes `var/runs/docker-smoke-*/report.json`; a passing result requires all checks and no remaining container with that sandbox label. PostgreSQL, Redis, MinIO, and a model service are not needed for this check.

Model versions can be removed through the authenticated console API only when no training run, evaluation, task, derived model, or artifact references the version. A checkpoint is physically removed only when its `file://` URI is a strict child of `ALVAREZ_CHECKPOINT_ROOT`; remote, unmanaged, shared, and symlinked paths fail closed. The delete flow retires the row before filesystem cleanup, keeps that retired row when cleanup fails so the operation can be retried, and checks the HTTP model service before deleting a loaded version. GPU-host checkpoints therefore remain until a shared mount or controlled storage adapter is configured.

## Local verification baseline

The local baseline is Python 3.12.8, uv 0.13.7, Docker 26.1.4, and Compose v2.26.1. The host exposes `/dev/kvm` and has sufficient disk for the repository data directory. The Compose file validates successfully, and the Python environment rebuilds from only `pyproject.toml` and `uv.lock`. On this host, the PostgreSQL Alpine image failed under the default Docker seccomp profile, so PostgreSQL uses the verified Debian 16.15 image digest. The three services reached healthy status and retained PostgreSQL, Redis, and MinIO verification data across a container restart.

## Remote host preflight

The GPU host is checked with read-only SSH commands before any runtime installation or configuration change. The 2026-11-12 preflight found Linux 6.8.0-54-generic, no `/dev/kvm`, an NVIDIA GeForce RTX 4090 with driver 595.71.05 and 23028 MiB memory, no `nvcc` on `PATH`, 934 GiB free on `/home`, and no active GPU compute process. The preflight does not copy models or modify the host. Because KVM is unavailable remotely, Android microVM execution stays on the local KVM-capable host until a separate remote virtualization host is provided.

## Multi-node runtime topology

As of 2026-11-16, the local host `10.41.51.215` runs the control plane and the schedulable Firecracker worker `node-10.41.51.215` (capacity 4). Loading `kvm_amd` created `/dev/kvm`; KVM API version 12 and `KVM_CREATE_VM` were verified. The module is persisted through `/etc/modules-load.d/alvarez-kvm.conf` and was reloaded successfully through `systemd-modules-load`. The previously proposed remote microVM host has been removed from the deployment plan and was never registered. `10.157.155.124` remains the Docker worker; its existing GPU training registration is separate.

Keep node workspaces, runtime assets, images, state, screenshots, and fixtures under `/home`. Device interfaces such as `/dev/kvm` remain kernel interfaces. Check Docker's data root before deploying the remote Android runtime to ensure its image and container storage also follows the `/home` constraint.

Firecracker sandboxes do not copy the rootfs. Each one gets a device-mapper snapshot (`/dev/mapper/alvarez-*`) of the shared image, read through a read-only loop device, with its writes in a sparse `MICROVM_STATE_DIR/rootfs/<sandbox>.cow` file. The host needs the `dm_snapshot` module (persisted in `/etc/modules-load.d/alvarez-dm-snapshot.conf`) plus `losetup` and `dmsetup`, run as root. Never modify the image file in place while sandboxes exist: install a new file and repoint `images/android-rootfs.ext4`, then remove the old file after its sandboxes are destroyed.

With networking enabled (the production setting), every Firecracker sandbox runs in its own network namespace `alv-<hash>`. Inside it, the guest address (`192.168.250.0/30`: TAP gateway `.1`, guest `.2`, fixed in code because templates record it), the TAP name `alvtap0`, and the disk path `MICROVM_STATE_DIR/rootfs/drive` are the same for every sandbox, which is what makes one snapshot restorable as any sandbox. Firecracker starts through `ip netns exec`, which gives it a private mount namespace where the sandbox's dm disk is bind-mounted onto that placeholder file. A veth pair connects the namespace to a unique `/30` from the `FIRECRACKER_NETWORK_HOST_IP/PREFIX_LEN` pool, whose last `/30` is reserved for template builds; NAT maps a per-sandbox random host-side ADB port (fresh each time, because host ADB keeps a crashed guest's `address:port` offline) to guest 5555 and port 8080 to the screen bridge. The guest can open connections only to the fixture port(s) of `SHOPPING_FIXTURE_URL`/`SOKOBAN_FIXTURE_URL`, reached at the TAP gateway; the namespace's FORWARD chain drops everything else from the guest. The `FIRECRACKER_NETWORK_HOST_IP/PREFIX_LEN` pool must not overlap `192.168.250.0/30`. The host needs `iptables` with the conntrack match.

`create()` clones the template `MICROVM_STATE_DIR/templates/<id>/` (`snapshot.bin`, `memory.bin`, a read-only `rootfs.ext4` holding the disk at snapshot time, `manifest.json`) when `FIRECRACKER_TEMPLATE_ENABLED=true` (default) and the template for the current inputs exists; it then sets the guest wall clock, which stood still in the snapshot. `<id>` hashes the Firecracker and kernel contents, the rootfs image path, size, and mtime, and the VM configuration, so any change makes the existing template unused until a new one is built. Build it with `.venv/bin/python scripts/build_firecracker_template.py --env-file var/platform.env` (about 2.5 minutes, safe while the worker runs; the worker picks it up on its next create). Without a usable template, or if loading one fails (that template is then skipped until the process restarts), sandboxes boot as before. Clones map `memory.bin` copy-on-write and read `rootfs.ext4` through their dm snapshot, so neither file may be changed or deleted while any sandbox started from it is running. At start, the worker reads both files into the page cache in the background (about 47 seconds on `/home`): with a cold cache, the first clone took 36 seconds instead of under one, and its first task timed out. Old templates (6.3 GB each) are not deleted automatically: the build script lists them for manual removal once their sandboxes are gone. To go back to booting every sandbox, set `FIRECRACKER_TEMPLATE_ENABLED=false` and restart.

The local platform runs in the default `ALVAREZ_SERVER_MODE=all` with `RUNTIME_BACKEND=firecracker`, `NODE_ID=node-10.41.51.215`, and `NODE_ADDRESS=10.41.51.215`. The existing service configuration is `/home/github/AlvarezGUI/var/platform.env`. Set `ALVAREZ_BIND_HOST` to an address reachable by remote workers for PostgreSQL, Redis, and MinIO; localhost remains the Compose default. A remote worker uses `ALVAREZ_SERVER_MODE=worker`, `ALVAREZ_RUNTIME_BACKEND=docker`, a unique `NODE_ID`, `NODE_ADDRESS`, `POSTGRES_HOST`, and `REDIS_HOST`. Apply migration `009_docker_runtime.sql` before workers start. The local platform has `ALVAREZ_SCHEDULER_BACKENDS=firecracker,docker` enabled.

The control plane dispatches remote pause, resume, snapshot/restore, and destroy requests through `alvarez:lifecycle:<backend>:<node_id>` Redis streams, routing to the lease owner even when multiple workers use the same backend. Workers revalidate the lease and backend before executing. `LIFECYCLE_TIMEOUT_SECONDS` defaults to 60 seconds for the HTTP wait; `LIFECYCLE_COMMAND_TTL_SECONDS` defaults to 300 seconds for execution eligibility. Inactive streams expire after twice the command TTL. Workers reclaim pending messages after one second, serialize reads/execution with a node-specific PostgreSQL transaction advisory lock, and do not overtake pending work. Results are cached for 24 hours before replies and ACK; a redelivery replays the saved result, including destroy after lease removal. Replies expire after 60 seconds. Successful remote responses and 504 details include `command_id`; query the authenticated `GET /internal/v1/sandboxes/{sandbox_id}/lifecycle/{command_id}` before submitting another command. Status is PENDING, STARTED, EXPIRED, or COMPLETED (inspect `result.ok` for success).

Recovery is bounded: lease and command expiry still apply; interrupted snapshot/restore without a saved result return a conflict instead of repeating an uncertain side effect. Runtime sandbox state remains process-local: a fresh worker can replay saved results but cannot reconstruct a lost VM/container handle, and such unsaved operations can fail with not-found/conflict. An operation and its result journal are not one atomic transaction, so exactly-once execution is not promised. Redis currently uses AOF with `appendfsync=everysec`; a Redis/host crash can lose the last second of journal writes. Destroy cancels an active task and episode and clears the sandbox reference and lease. Local leases still execute directly.

Workers and the control plane must access the same MinIO instance and bucket, with appropriate credentials and the bucket created before tasks start. Both deployed nodes now use `http://10.41.51.215:19000`, bucket `alvarez`; the worker configuration is `/home/alvarez-runtime/app/.env`. New screenshots are uploaded directly to MinIO as `s3://<bucket>/screenshots/<sandbox>/<digest>-<uuid>.png`; after successful upload, the worker removes only the local PNG under its managed runs directories. Failed uploads retain the local file for diagnosis. The console and training screenshot API read both MinIO objects and legacy local artifacts; the old HTTP upload endpoint remains compatible and also writes MinIO.

Run `python -m alvarezgui.artifact_maintenance migrate` in the node environment for a dry-run, then add `--apply` to migrate terminal episodes' step and final-observation references. Uploads are downloaded and SHA-256 verified before exact JSON-path compare-and-set updates; originals are retained. Only managed runs roots are read unless explicitly extended with `--root /home/...`; missing, unmanaged, conflicting and failed references are counted. On 2026-11-16, all 59 historical file references were migrated without missing files, conflicts or errors.

`python -m alvarezgui.artifact_maintenance gc` previews collection; `--apply` deletes only unreferenced objects under the configured bucket's `screenshots/` prefix older than `ARTIFACT_RETENTION_DAYS` (default 30, overridable with `--grace-days`). References across persisted tables, including registry and dataset sources, are protected regardless of age. GC skips active tasks/training or remaining leases, rechecks references under table locks, and rechecks object metadata before deletion. Keep keys immutable. References inside external dataset manifests or other databases are outside this collector's scope; register an S3 directory prefix in the dataset/artifact registry to protect those objects before enabling external reuse. The current deployment has no registered datasets. No bucket-wide age expiration rule is installed.

The control-plane host runs `alvarez-artifact-gc.timer` daily at 03:30 with up to 15 minutes jitter, using the units in `scripts/systemd/` and the platform environment. Check `systemctl list-timers alvarez-artifact-gc.timer` and `journalctl -u alvarez-artifact-gc.service`. The first run inspected 197 recent objects and deleted none. Deployment backups are `/home/github/AlvarezGUI/var/deployments/20261116-212047` and `/home/alvarez-runtime/backups/20261116-212047/app.tar.gz`; the local directory also contains pre-migration episode/step references. To roll back the remote release, stop its worker, restore its archived app/config, and restart it. Disable the GC timer before rolling back maintenance code. The local archive captures the pre-restart on-disk source/config, not the old process's previously loaded modules; historical screenshot originals remain available for a controlled reference rollback.

Consumer worker pools bound in-flight futures to their configured capacity and stop reading or claiming task messages when full. Android runtime supplies the shared ADB deadline and unsupported snapshot defaults used by Docker; Firecracker keeps its snapshot implementation.
