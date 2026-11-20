from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_SANDBOXES = 32  # Per runtime node, whatever its hardware.


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        hide_input_in_errors=True,
    )

    data_dir: Path = Field(default=Path("./var"), validation_alias="ALVAREZ_DATA_DIR")
    checkpoint_cleanup_root: Path | None = Field(
        default=None,
        validation_alias="ALVAREZ_CHECKPOINT_ROOT",
    )
    postgres_db: str = "alvarez"
    postgres_user: str = "alvarez"
    postgres_password: SecretStr = Field(min_length=1)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = Field(default=15432, ge=1, le=65535)
    redis_host: str = "127.0.0.1"
    redis_port: int = Field(default=16379, ge=1, le=65535)
    minio_root_user: str = Field(default="alvarez", min_length=3)
    minio_root_password: SecretStr = Field(min_length=8)
    minio_host: str = "127.0.0.1"
    minio_port: int = Field(default=19000, ge=1, le=65535)
    minio_bucket: str = "alvarez"
    runtime_backend: str = Field(
        default="firecracker",
        pattern="^(firecracker|docker)$",
        validation_alias=AliasChoices("ALVAREZ_RUNTIME_BACKEND", "RUNTIME_BACKEND"),
    )
    server_mode: Literal["all", "control_plane", "worker"] = Field(
        default="all",
        validation_alias="ALVAREZ_SERVER_MODE",
    )
    scheduler_backends: str = Field(
        default="",
        validation_alias="ALVAREZ_SCHEDULER_BACKENDS",
    )
    docker_binary: str = Field(
        default="docker",
        validation_alias=AliasChoices("ALVAREZ_DOCKER_BINARY", "DOCKER_BINARY"),
    )
    docker_image: str = Field(
        default="",
        validation_alias=AliasChoices("ALVAREZ_DOCKER_IMAGE", "DOCKER_ANDROID_IMAGE"),
    )
    docker_container_prefix: str = Field(
        default="alvarez-android",
        validation_alias=AliasChoices("ALVAREZ_DOCKER_CONTAINER_PREFIX", "DOCKER_CONTAINER_PREFIX"),
    )
    docker_fixture_host: str = Field(
        default="host.docker.internal",
        validation_alias=AliasChoices("ALVAREZ_DOCKER_FIXTURE_HOST", "DOCKER_FIXTURE_HOST"),
    )
    docker_startup_timeout_seconds: float = Field(
        default=180.0,
        gt=0,
        le=900,
        validation_alias=AliasChoices("ALVAREZ_DOCKER_STARTUP_TIMEOUT_SECONDS", "DOCKER_STARTUP_TIMEOUT_SECONDS"),
    )
    docker_privileged: bool = Field(
        default=False,
        validation_alias=AliasChoices("ALVAREZ_DOCKER_PRIVILEGED", "DOCKER_PRIVILEGED"),
    )
    docker_binderfs_path: Path = Field(
        default=Path("/dev/binderfs"),
        validation_alias=AliasChoices("ALVAREZ_DOCKER_BINDERFS_PATH", "DOCKER_BINDERFS_PATH"),
    )
    firecracker_binary: Path = Path("/usr/local/bin/firecracker")
    microvm_kernel_image: Path = Path("./var/microvm/images/vmlinux")
    microvm_rootfs_image: Path = Path("./var/microvm/images/android-rootfs.ext4")
    microvm_state_dir: Path = Path("./var/microvm")
    firecracker_boot_args: str = (
        "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda "
        "rootfstype=ext4 rootwait rw"
    )
    firecracker_network_enabled: bool = True
    firecracker_network_iface_id: str = "eth0"
    firecracker_network_host_ip: str = "192.168.100.1"
    firecracker_network_prefix_len: int = Field(default=24, ge=1, le=30)
    firecracker_network_sandbox_prefix_len: int = Field(default=30, ge=30, le=30)
    firecracker_template_enabled: bool = True
    firecracker_ip_binary: str = "ip"
    guest_adb_port: int = Field(default=5555, ge=1, le=65535)
    guest_screen_bridge_port: int = Field(default=8080, ge=1, le=65535)
    shopping_fixture_url: str = "http://192.168.100.1:18090/shopping/"
    sokoban_fixture_url: str = "http://192.168.100.1:18090/sokoban/"
    adb_binary: str = "adb"
    adb_serial_prefix: str = "alvarez-"
    model_backend: str = Field(default="fake_policy", pattern="^(fake_policy|http)$")
    model_service_url: str = "http://127.0.0.1:18081"
    model_request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    lifecycle_timeout_seconds: int = Field(default=60, ge=1, le=300)
    lifecycle_command_ttl_seconds: int = Field(default=300, ge=1, le=3600)
    artifact_retention_days: int = Field(default=30, ge=1, le=3650)
    console_password: SecretStr | None = None
    console_host: str = "0.0.0.0"
    console_port: int = Field(default=8600, ge=1, le=65535)
    node_id: str = "node-local"
    node_address: str = ""
    sandbox_capacity: int = Field(default=4, ge=1, le=MAX_SANDBOXES)  # Default; the console may change it.
    fixture_port: int = Field(default=18090, ge=1, le=65535)

    @property
    def scheduler_backend_list(self) -> tuple[str, ...]:
        configured = tuple(
            backend.strip()
            for backend in self.scheduler_backends.split(",")
            if backend.strip()
        )
        backends = configured or (self.runtime_backend,)
        if any(backend not in {"firecracker", "docker"} for backend in backends):
            raise ValueError("ALVAREZ_SCHEDULER_BACKENDS must contain firecracker and/or docker")
        return tuple(dict.fromkeys(backends))

    @property
    def minio_endpoint(self) -> str:
        return f"http://{self.minio_host}:{self.minio_port}"
