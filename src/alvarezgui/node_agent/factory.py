from __future__ import annotations

from alvarezgui.config import Settings
from alvarezgui.node_agent.runtime import Runtime


def runtime_from_settings(settings: Settings) -> Runtime:
    if settings.runtime_backend == "docker":
        from alvarezgui.node_agent.docker import DockerAndroidRuntime, DockerConfig

        return DockerAndroidRuntime(
            DockerConfig(
                binary=settings.docker_binary,
                image=settings.docker_image,
                state_dir=settings.data_dir,
                adb_binary=settings.adb_binary,
                guest_adb_port=settings.guest_adb_port,
                api_timeout=settings.model_request_timeout_seconds,
                guest_ready_timeout=settings.docker_startup_timeout_seconds,
                container_prefix=settings.docker_container_prefix,
                fixture_host=settings.docker_fixture_host,
                shopping_fixture_url=settings.shopping_fixture_url,
                sokoban_fixture_url=settings.sokoban_fixture_url,
                privileged=settings.docker_privileged,
                binderfs_path=settings.docker_binderfs_path,
            )
        )

    if settings.runtime_backend != "firecracker":
        raise ValueError(f"unsupported runtime backend: {settings.runtime_backend}")
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
            network_prefix_len=settings.firecracker_network_prefix_len,
            network_sandbox_prefix_len=settings.firecracker_network_sandbox_prefix_len,
            template_enabled=settings.firecracker_template_enabled,
            ip_binary=settings.firecracker_ip_binary,
            guest_adb_port=settings.guest_adb_port,
            guest_screen_bridge_port=settings.guest_screen_bridge_port,
            shopping_fixture_url=settings.shopping_fixture_url,
            sokoban_fixture_url=settings.sokoban_fixture_url,
            adb_binary=settings.adb_binary,
            adb_serial_prefix=settings.adb_serial_prefix,
        )
    )
