# Firecracker 模板快照克隆启动方案

- 状态：已实现，2026-11-20 22:15 部署（线上模板 `var/microvm/templates/eb6ea93f1d26febb`）。FIRECRACKER_TEMPLATE_ENABLED 默认打开；第 2 节的 FORWARD 加固已做。原型记录和实现、部署结果见 [WORKLOG.md](../WORKLOG.md) 的"模板快照克隆启动原型"和"模板快照克隆启动实现"。与本文的差异：冷页缓存实测要预读 `memory.bin` 和 `rootfs.ext4` 两个文件；命名空间名按 COW 路径哈希；guest 内部网段 `192.168.250.0/30` 写死在代码里，没有做成 `FIRECRACKER_GUEST_CIDR` 配置（11-21 精简时去掉，改它就要重做模板，没有需要改的场景）。
- 原型脚本：`var/microvm-template-poc/poc-scripts/`（`poc.py` 主流程，`clock.py` 时钟，`crash.py` 崩溃后 ADB）。原型模板已在部署后删除。
- 预计工作量：3–4 小时（实现 1.5–2 小时，单测 45 分钟，真实验证 1–1.5 小时，文档与部署 30 分钟）。部署要先确认没有运行中的任务。

## 目标

`create()` 不再每次冷启动 Android（单个 19–27 秒，2 个并发 88–129 秒），而是从一份模板快照恢复。原型中 4 个克隆并发 0.6–0.8 秒全部就绪，shopping 任务都 reward=1。模板缺失、过期或加载失败时回退到冷启动。

## 原型结论（决定设计的事实）

| 事实 | 影响 |
|---|---|
| 快照里固定了 guest IP（内核参数 `ip=`）和 TAP 名 | 每个 sandbox 一个网络命名空间，命名空间内地址和 TAP 名都一样，用 NAT 映射成宿主机上唯一的地址 |
| 快照里固定了磁盘路径，Firecracker 1.14.4 没有 `drive_overrides`（上游仍在 Unreleased） | 在 Firecracker 进程自己的挂载命名空间里，把该 sandbox 的 dm 盘 bind mount 到快照记录的路径 |
| `ip netns exec` 会给命令一个新的挂载命名空间，并直接 exec | bind mount 只对该 Firecracker 可见（宿主机看不到），`Popen.pid` 就是 Firecracker |
| 宿主机 5.10 内核，克隆后 guest 墙上时间停在模板时刻（实测落后 191–349 秒） | 就绪后执行 `su 0 date @<now>`，实测校正后偏差 0 秒 |
| 上个 sandbox 崩溃没 `adb disconnect` 时，host ADB 把同一 `地址:端口` 卡在 offline，`adb disconnect` 删不掉，新克隆要等 34–80 秒 | 每个 sandbox 换一个新的宿主机侧 ADB 端口，DNAT 到 guest 的 5555；实测 0.58 秒就绪 |
| 命名空间内 `iptables -m multiport` 报错（模块未加载），单端口规则正常 | NAT 一条规则一个端口 |
| 模板内存文件按 `MAP_PRIVATE` 映射，没改过的页各克隆共享 | 每个克隆就绪时 PSS 约 180 MB，做完任务约 730 MB（冷启动为 2 GB）；模板文件在所有克隆结束前不能删、不能改 |

## 设计

### 1. 模板

目录 `MICROVM_STATE_DIR/templates/<template_id>/`：

| 文件 | 内容 |
|---|---|
| `snapshot.bin` | VM 状态（约 30 KB） |
| `memory.bin` | guest 内存（2 GB，稀疏度低） |
| `rootfs.ext4` | 与内存同一时刻的磁盘，只读（`chmod 444`），稀疏，实占约 4.3 GB |
| `manifest.json` | 计算 id 用的全部输入，和制作时间、冷启动/快照耗时 |

`template_id` 取下列内容 JSON 的 SHA-256 前 16 位，任何一项变了，模板就失效：
- Firecracker 二进制和内核文件的 SHA-256（各几十 MB，进程启动时算一次）；
- rootfs 镜像解析后的路径、大小、`mtime_ns`（10 GB 不做内容哈希）；
- `vcpu_count`、`mem_size_mib`、`boot_args`、guest 内部网段、guest ADB/screen bridge 端口。

制作流程（新方法 `FirecrackerAndroidRuntime.build_template(settle_seconds=90)`，入口脚本 `scripts/build_firecracker_template.py --env-file var/platform.env`，写法参照 `scripts/sokoban_binding_runtime.py`）：
1. 用普通冷启动路径（同样的命名空间网络）启动 `template-builder`。
2. 就绪后等 `settle_seconds`。之前的启动日志里 `apexd` 在后台跑了约 70 秒，不等的话每个克隆都要重跑一遍。`# ponytail: 固定等待，改成检测 load 稳定再快照如果 90 秒不够或太长`。
3. pause，`adb disconnect`，`/snapshot/create` Full 写到 `templates/<id>.building/`，停进程。
4. `dd if=<builder 的 dm 设备> of=<building>/rootfs.ext4 bs=4M conv=sparse`，再 `truncate -s <设备大小>`（`conv=sparse` 不写结尾的空洞），`chmod 444`。原型 8.6 秒。
5. destroy builder，写 `manifest.json`，`rename(<id>.building, <id>)`。总计约 2.5 分钟。

关键坑：快照里记录的是磁盘路径，所以磁盘路径不能放在模板目录里（第 5 步的 rename 会让它失效）。所有 sandbox 统一用一个固定的占位文件 `MICROVM_STATE_DIR/rootfs/drive`（空文件）作为 Firecracker 看到的磁盘路径，见第 3 节。内存文件路径在 load 时传入，rename 没影响。

worker 每次 `create()` 用缓存的二进制哈希和 rootfs 的 `stat` 重新算 id，检查目录是否存在，所以 worker 运行中制作好的模板，下一个 sandbox 就会用上，不需要重启。

旧模板不自动删除（每份 6.3 GB，只在 rootfs/内核/Firecracker/VM 配置变化时才重做）。制作脚本最后列出其他模板目录，由人确认没有 sandbox 在用后手动删除。`# ponytail: 手动 GC，模板变化频繁时再按 COW 引用自动清理`。

### 2. 网络：每个 sandbox 一个命名空间

所有 sandbox（克隆和冷启动）走同一条路径，替换现在的"宿主机 TAP + /32 路由 + 每 sandbox 不同的 `ip=`"：

```
宿主机                       命名空间 alv-<hash8>                  guest
alv-<hash8> (veth)  ──────  veth0                alvtap0 (TAP) ── eth0
endpoint.host_ip/30         endpoint.guest_ip/30  内部网关 IP/30     内部 guest IP/30
```

- 地址池不变：`FIRECRACKER_NETWORK_HOST_IP/PREFIX_LEN`（生产 `192.168.100.0/24`）切成 /30。宿主机仍连 `endpoint.guest_ip`，所以 `_adb_target`、screen bridge、`_network_endpoint` 的含义不变。宿主机 veth 名沿用 `_tap_name()`。
- 地址池保留最后一个 /30 给 `template-builder`。制作脚本是另一个进程，地址分配器在进程内，不保留的话可能和 worker 的 sandbox 撞地址。
- guest 内部网段是新配置 `FIRECRACKER_GUEST_CIDR`，默认 `192.168.250.0/30`（网关 .1，guest .2），写进模板 id。`_check_assets` 拒绝与地址池重叠的值，否则命名空间里 veth0 和 TAP 的路由会冲突。
- `_boot_args()` 变成常量：`ip=<内部 guest>::<内部网关>:255.255.255.252::eth0:off`。
- 冷启动的 `PUT /network-interfaces` 用固定的 `host_dev_name=alvtap0`。

命名空间内的设置（原型 `NamespacedRuntime._setup_network`）：

```
ip netns add alv-<hash8>
ip link add alv-<hash8> type veth peer name alv-<hash8>n
ip link set alv-<hash8>n netns alv-<hash8>
ip addr add <endpoint.host_ip>/30 dev alv-<hash8>; ip link set alv-<hash8> up
# 以下在命名空间内（ip netns exec alv-<hash8> ...）
ip link set lo up
ip link set alv-<hash8>n name veth0
ip addr add <endpoint.guest_ip>/30 dev veth0; ip link set veth0 up
ip tuntap add dev alvtap0 mode tap
ip addr add <内部网关>/30 dev alvtap0; ip link set alvtap0 up
sysctl -qw net.ipv4.ip_forward=1
iptables -t nat -A PREROUTING  -i veth0 -p tcp --dport <adb_port> -j DNAT --to-destination <内部 guest>:5555
iptables -t nat -A PREROUTING  -i veth0 -p tcp --dport 8080 -j DNAT --to-destination <内部 guest>:8080
iptables -t nat -A POSTROUTING -o alvtap0 -j SNAT --to-source <内部网关>
iptables -t nat -A PREROUTING  -i alvtap0 -d <内部网关> -p tcp --dport <fixture 端口> -j DNAT --to-destination <endpoint.host_ip>:<fixture 端口>
iptables -t nat -A POSTROUTING -o veth0 -j SNAT --to-source <endpoint.guest_ip>
```

- fixture 端口从 `shopping_fixture_url`/`sokoban_fixture_url` 解析（现在都是 18090）；fixture HTTP 服务已绑定 `0.0.0.0`（`server.py` 的 `serve_fixture`），宿主机 veth 地址可达。
- `_fixture_url()` 改写为 `http://<内部网关>:<端口>/...`。
- 清理：`ip netns del alv-<hash8>`（连同命名空间里的 veth 端和 TAP），再 `ip link del alv-<hash8>`（通常已随对端消失）。
- ADB 端口：`_setup_network` 时从 20000–59999 随机选一个当前 `adb devices` 里没有的 `<endpoint.guest_ip>:<port>`，存在 runtime 里；`_adb_target`、`_adb_connect`、`_adb_disconnect` 都用它（`_adb_disconnect` 现在拼的是 `guest_adb_port`，要改）。正常 destroy 会先断开 ADB，不留条目；崩溃留下的 offline 条目只能等 `alvarez-adb-host-runtime` 容器重启才清掉。
- 可选加固（会改变现有行为，实现前要确认）：现在 guest 能访问宿主机 TAP 地址上的任何服务。命名空间里可以只放行 fixture：`-A FORWARD -i alvtap0 -m state --state ESTABLISHED,RELATED -j ACCEPT`、`-A FORWARD -i alvtap0 -p tcp -d <endpoint.host_ip> --dport <fixture 端口> -j ACCEPT`、`-A FORWARD -i alvtap0 -j DROP`。

### 3. 磁盘

- `_prepare_rootfs(sandbox_id, origin)` 多一个参数：克隆用模板的 `rootfs.ext4`，冷启动用 `config.rootfs_image`。dm-snapshot 和 COW 文件的做法不变；不同 sandbox 可以用不同的 origin（新旧模板并存时）。
- Firecracker 一律看到占位文件 `MICROVM_STATE_DIR/rootfs/drive`。`_start_process` 改成：

  ```
  ip netns exec alv-<hash8> sh -c 'mount --bind "$1" "$2" && exec "$3" --api-sock "$4"' _ \
      /dev/mapper/alvarez-<hash> <state>/rootfs/drive <firecracker> <socket>
  ```

  冷启动的 `PUT /drives/rootfs` 也传占位路径。这样任何 sandbox 自己的 snapshot 也都记录占位路径，restore 时 `_start_process` 照样 bind mount，克隆和冷启动的 restore 是同一条路径。
- 和底座镜像一样，模板的 `rootfs.ext4` 和 `memory.bin` 在所有用它的 sandbox 结束前不能修改或删除。

### 4. `create()` 流程与回退

```
template = self._usable_template()     # 关闭、缺失、id 不符或本进程内加载失败过 → None
prepare_rootfs(sandbox_id, template.rootfs if template else config.rootfs_image)
setup_network(sandbox_id)              # 命名空间、veth、TAP、NAT、ADB 端口
start_process(sandbox_id)              # 命名空间内 bind mount 后 exec Firecracker
if template:
    PUT /snapshot/load {snapshot_path, mem_backend: {File, memory.bin}, resume_vm: true}
else:
    entropy, machine-config, boot-source, drives(占位路径), network-interfaces(alvtap0), InstanceStart
wait_for_guest_ready(sandbox_id)
if template:
    set_guest_clock(sandbox_id)        # adb shell su 0 date @<int(time.time())>
```

- 克隆失败（load 报错或未就绪）：记日志，把这个 `template_id` 标成本进程内不可用，清理后对同一个 sandbox 冷启动一次。之后的 create 不再试这个模板，直到重新制作出新 id。这样坏模板不会让每个任务都先失败一次。
- 时钟只精确到秒。`# ponytail: 秒级校正，需要毫秒时改用 guest 内 chrony/PHC`。restore 后同样校正（快照到恢复之间 guest 时间也停住了）。
- 新配置 `FIRECRACKER_TEMPLATE_ENABLED`（建议默认 `true`：有模板就用，没有就冷启动）。设成 `false` 可以不改代码退回冷启动。
- 模板只在 `network_enabled=True` 时使用（生产配置）。`network_enabled=False` 的 ADB serial 路径保持原样。

### 5. snapshot / restore

现有逻辑不变（snapshot 前 `adb disconnect`，restore 不配置设备、直接 load）。差别只在 `_start_process` 带 bind mount，以及 restore 后校正时钟。克隆出来的 sandbox 再 snapshot，Full 快照会把它当前全部内存写出（包括来自模板、没改过的页），这一点要实测。

### 6. 崩溃残留

现在的 `_remove_orphan_disks` 按 COW 文件的 flock 判断"属主进程已死"。扩展成 `_remove_orphans`：对每个无人持锁的 COW，除了删 dm 设备和 COW，还按 sandbox id 算出命名空间名，`ip netns pids` 里如果还有 Firecracker 就 kill，然后 `ip netns del`、`ip link del`。名字都由 sandbox id 哈希得到，不需要额外记录。

### 7. 代码改动清单

| 位置 | 改动 |
|---|---|
| `node_agent/firecracker.py` | `FirecrackerConfig` 加 `template_enabled`、`guest_cidr`；`create` 按第 4 节；`_setup_network`/`_cleanup_network` 换成命名空间；`_boot_args`、`_fixture_url`、`_adb_target`、`_adb_connect`、`_adb_disconnect` 用新地址和端口；`_start_process` 带 bind mount；`_prepare_rootfs` 加 origin；`_remove_orphans`；新增 `build_template`、`_usable_template`、`_template_id`、`_set_guest_clock`；`_allocate_network_endpoint` 保留最后一个 /30；`_check_assets` 校验网段不重叠 |
| `config.py`、`node_agent/factory.py` | `FIRECRACKER_TEMPLATE_ENABLED`、`FIRECRACKER_GUEST_CIDR` 两个配置透传 |
| `scripts/build_firecracker_template.py` | 新建，制作模板并列出旧模板 |
| `tests/test_firecracker.py` | 见第 8 节 |
| `docs/adr/0001-local-environment.md` | 命名空间网络、模板制作、模板不可修改、手动 GC |
| `WORKLOG.md`、`IMPLEMENTATION_PLAN.md` | 记录结果 |

不改：consumer、调度、控制面 API、guest 镜像。

## 8. 测试

单测（`tests/test_firecracker.py`）：
- 要改写：`test_create_sends_entropy_before_start_and_waits_for_ready`（磁盘占位路径、`alvtap0`）、`test_network_create_configures_tap_before_firecracker_start`（改为命名空间命令）、`test_network_setup_failure_cleans_tap_and_does_not_start_vm`（失败时删命名空间）、`test_network_allocates_independent_endpoints_per_sandbox` 和 `test_network_readiness_uses_guest_adb_and_bridge`（ADB 目标带新端口）。
- 新增：rootfs `mtime` 或 VM 配置变化时 `template_id` 变化；有模板时 create 只发 `/snapshot/load` 并校正时钟；克隆失败后冷启动一次、后续 create 跳过该模板；孤儿清理删除命名空间；`_fixture_url` 指向内部网关；`_check_assets` 拒绝重叠网段；地址池最后一个 /30 不分配给普通 sandbox。

真实验证（先用独立 state dir 和 `192.168.101.0/24`，不碰线上），全部通过才部署：

| 项目 | 通过标准 |
|---|---|
| 制作模板 | 脚本成功，`manifest.json` 正确，耗时约 2–3 分钟 |
| 4 个克隆并发 | 全部在 2 秒内就绪；settings、shopping、sokoban 各跑一次，reward 判定正常；A 写的文件 B 看不到；时钟偏差 ≤ 1 秒 |
| 克隆后 snapshot→restore | restore 在 2 秒内就绪，文件保留，时钟已校正 |
| 模板不在页缓存 | 对模板两个文件 `posix_fadvise(DONTNEED)` 后第一个克隆的就绪时间；超过 5 秒就在 worker 启动时后台顺序读一遍 `memory.bin`（机械盘约 15 秒） |
| 16 个克隆并发 | 全部就绪，记录总耗时、宿主机内存减少量、COW 大小 |
| 崩溃 | SIGKILL 持有 2 个克隆的进程，新 runtime 清掉命名空间/dm/COW，新克隆 2 秒内就绪 |
| 回退 | 模板目录不存在时冷启动正常（新网络路径）；模板文件损坏（用副本测）时冷启动一次且后续跳过 |
| 清理 | destroy 后无命名空间、veth、dm、COW 残留，ADB 无本次新增条目 |

部署后：在线上跑一次 `sokoban-validation`（8 题）Firecracker 评测，确认端到端正常，记录 sandbox 分配耗时。

## 9. 部署与回滚

1. 确认没有排队/运行中的 task、训练和评测。
2. 把改动的源码文件备份到 `var/deployments/<时间>/`。
3. 停 `alvarez-platform`（停止时会 destroy 本节点的 sandbox），用新代码制作模板，再启动服务。先停再做模板，可以避开旧代码的 TAP 地址和制作用的 /30 冲突。
4. 跑上面的线上评测。

回滚：只想退回冷启动，设 `FIRECRACKER_TEMPLATE_ENABLED=false` 并重启（仍是新的命名空间网络）；网络本身有问题，就恢复第 2 步的备份文件并重启。

## 10. 风险与未决问题

- 所有克隆的 Android 状态相同（ANDROID_ID、用户态随机数种子、WebView 状态）。模板在执行任何任务前制作，任务每次 `force-stop` 后重新打开页面，预计没影响，但没有逐项验证。
- bind mount 方案依赖 `ip netns exec` 为命令新建挂载命名空间（本机 iproute2-ss170501 已验证）。升级到带 `drive_overrides` 的 Firecracker 后可以去掉。
- 模板内存靠页缓存共享。宿主机内存紧张时页被回收，克隆缺页会去读机械盘。
- 崩溃留下的 ADB offline 条目会在共享 ADB 容器里累积，只有重启容器才清掉。
- 模板 6.3 GB，rootfs、内核、Firecracker 或 VM 配置变化都要重做，旧模板手动删除。
- 克隆的内存占用降到约 730 MB，资源总览的 sandbox 上限估算（按每个 2 GB 内存）会偏保守。是否调整不在本次范围。
- 第 2 节的 FORWARD 加固已确认并实现（2026-11-20）；实现用 `-m conntrack --ctstate`（与 `-m state` 等价）。
