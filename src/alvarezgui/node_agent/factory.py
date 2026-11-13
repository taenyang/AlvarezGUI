from __future__ import annotations

from alvarezgui.config import Settings
from alvarezgui.node_agent.runtime import Runtime


def runtime_from_settings(settings: Settings) -> Runtime:
    if settings.runtime_backend == "fake":
        from alvarezgui.node_agent.runtime import FakeRuntime

        return FakeRuntime()
    from alvarezgui.node_agent.firecracker import FirecrackerAndroidRuntime, FirecrackerConfig

    return FirecrackerAndroidRuntime(
        FirecrackerConfig(
            binary=settings.firecracker_binary,
            kernel_image=settings.microvm_kernel_image,
            rootfs_image=settings.microvm_rootfs_image,
            state_dir=settings.microvm_state_dir,
            boot_args=settings.firecracker_boot_args,
            network_enabled=settings.firecracker_network_enabled,
            network_iface_id=settings.firecracker_network_iface_id,
            network_host_ip=settings.firecracker_network_host_ip,
            network_guest_ip=settings.firecracker_network_guest_ip,
            network_prefix_len=settings.firecracker_network_prefix_len,
            ip_binary=settings.firecracker_ip_binary,
            guest_adb_port=settings.guest_adb_port,
            guest_screen_bridge_port=settings.guest_screen_bridge_port,
            shopping_fixture_url=settings.shopping_fixture_url,
            adb_binary=settings.adb_binary,
            adb_serial_prefix=settings.adb_serial_prefix,
        )
    )
