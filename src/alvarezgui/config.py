from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        hide_input_in_errors=True,
    )

    data_dir: Path = Field(default=Path("./var"), validation_alias="ALVAREZ_DATA_DIR")
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
    runtime_backend: str = Field(default="fake", pattern="^(fake|firecracker)$")
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
    firecracker_network_guest_ip: str = "192.168.100.2"
    firecracker_network_prefix_len: int = Field(default=24, ge=1, le=32)
    firecracker_ip_binary: str = "ip"
    guest_adb_port: int = Field(default=5555, ge=1, le=65535)
    guest_screen_bridge_port: int = Field(default=8080, ge=1, le=65535)
    shopping_fixture_url: str = "http://192.168.100.1:18090/"
    adb_binary: str = "adb"
    adb_serial_prefix: str = "alvarez-"
    model_backend: str = Field(default="fake_policy", pattern="^(fake_policy|http)$")
    model_service_url: str = "http://127.0.0.1:18081"
    model_request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    console_password: SecretStr | None = None
    console_host: str = "0.0.0.0"
    console_port: int = Field(default=8600, ge=1, le=65535)
    node_id: str = "node-local"
    node_address: str = ""
    fixture_port: int = Field(default=18090, ge=1, le=65535)

    @property
    def minio_endpoint(self) -> str:
        return f"http://{self.minio_host}:{self.minio_port}"
