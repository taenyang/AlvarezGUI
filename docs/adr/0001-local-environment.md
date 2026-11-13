# ADR 0001: Local and migratable environment

- Status: Accepted
- Date: 2026-11-12

## Decision

The first control-plane environment uses Python 3.12 and `uv`. `pyproject.toml` is the source of declared dependencies and `uv.lock` is committed to reproduce their resolution. The repository-local `.venv` is disposable and is not committed.

PostgreSQL, Redis, and MinIO run through the root `compose.yaml` with pinned image digests from the available domestic registry, localhost-only ports, health checks, and bind mounts under `ALVAREZ_DATA_DIR`. The default is `./var`; operators can point it at a separate disk such as `/mnt/alvarezgui-data` without changing the repository. The expected layout is:

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

The Node Agent depends on the small runtime interface in `src/alvarezgui/node_agent/runtime.py`. The initial implementation is an in-memory fake runtime so lifecycle semantics can be checked before selecting Firecracker, Cloud Hypervisor, or another runtime. microVMs do not run inside Compose. The current placement is local Node Agent plus Android microVM on a host with accessible `/dev/kvm`; the remote GPU host is reserved for model inference and training because it has no `/dev/kvm`.

## Consequences

- Moving runtime data means stopping services and moving or remounting one data directory.
- Secrets stay in an untracked `.env` file or an external secret manager.
- Model and training dependencies remain outside this control-plane environment.
- The first real microVM adapter is Firecracker. `runtime_backend=firecracker` uses the repository's Firecracker plus ADB adapter and validates the binary, kernel image, and Android rootfs before boot; the default remains `fake` until those deployment assets are staged.

## Local verification baseline

The local baseline is Python 3.12.8, uv 0.13.7, Docker 26.1.4, and Compose v2.26.1. The host exposes `/dev/kvm` and has sufficient disk for the repository data directory. The Compose file validates successfully, and the Python environment rebuilds from only `pyproject.toml` and `uv.lock`. On this host, the PostgreSQL Alpine image failed under the default Docker seccomp profile, so PostgreSQL uses the verified Debian 16.15 image digest. The three services reached healthy status and retained PostgreSQL, Redis, and MinIO verification data across a container restart.

## Remote host preflight

The GPU host is checked with read-only SSH commands before any runtime installation or configuration change. The 2026-11-12 preflight found Linux 6.8.0-54-generic, no `/dev/kvm`, an NVIDIA GeForce RTX 4090 with driver 595.71.05 and 23028 MiB memory, no `nvcc` on `PATH`, 934 GiB free on `/home`, and no active GPU compute process. The preflight does not copy models or modify the host. Because KVM is unavailable remotely, Android microVM execution stays on the local KVM-capable host until a separate remote virtualization host is provided.
