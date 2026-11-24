# H800 在线 RL 改为按 run 常驻 pod 方案

- 状态：T1–T3 已实现并部署（2026-11-24 18:48，归档 `var/deployments/20261124-184838-resident-pods/`）；T4 端到端 run 已跑通（`run_658aa22c…`，见 WORKLOG）；人为卡死、crashloop 的探活验证未做。只对部署后新开始的在线 run 生效（`online.json` 的 `resident: true`），之前已 staging 的 run 继续用每轮新建的 pod。
- 与方案的差异：
  - 训练 worker 每轮仍从 CFS 加载权重（调用原 `run_pod`），没有把模型留在显存里。实测 transformers 从 CFS 加载这个 1B 模型不到 1 秒，留在显存里省不了时间，不做。省下的是 pod 调度、拉镜像、import、CUDA 初始化和 fla Triton 编译。
  - 训练 worker 的心跳以本轮 `run.log` / `metrics.jsonl` 增长为进展，15 分钟无进展停止心跳（livenessProbe 60 秒内没更新就重启）。
  - 合并副本由 run 的 `serve/merged-copies.json` 记录，服务切换离开后删除，run 结束时全部删除。
- 范围：H800 上的在线 RL（GRPO、DAPO、Dr. GRPO、GiGPO）的训练 pod 和 rollout pod。SFT、评测服务、4090 训练、控制面和 consumer 都不变。
- 用户决定（2026-11-24）：按 run 常驻（run 开始建、结束删，不跨 run 共享）；训练和 rollout 都常驻；要实时探活重建。
- 取代 `docs/h800-sglang-rollout.md` "不做的事"里"每轮新建 pod、从 CFS 加载"这一条，以及 H800 即用即释放的约定（对在线 RL run 而言）。
- 预计工作量：1.5–2 天。

## 目标

run_a58c3b10 第 9 轮（443 行）约 23 分钟，其中约 11.8 分钟是每轮的固定开销：

| 固定开销 | 时间 | 常驻后 |
|---|---|---|
| sglang rollout 服务启动到就绪 | 3–5 分钟（第 11 轮因端口冲突重试到 21 分钟） | 热更新权重约 20–37 秒 |
| 训练 pod 启动、import、从 CFS 加载模型 | 2–3 分钟 | 只在第 0 轮 |
| fla Triton 编译 / autotune | 新 pod 第一次遇到新形状时 | 只在 pod 生命周期内第一次 |
| checkpoint 拷到 4090 并核对 SHA-256 | 1.3–2.2 分钟 | 后台进行，不阻塞下一轮 |

目标：固定开销降到每轮约 2 分钟（热更新约 0.5 分钟、合并 LoRA、launcher 轮询）。

## 前置验证（2026-11-24，均通过）

| 验证 | 结果 |
|---|---|
| sglang `/update_weights_from_disk`（MiniCPM-V 4.7，打补丁后的 0.5.23，535 驱动加 compat） | `success: true`。换一次权重 19.5–37 秒（同节点全新启动 65–133 秒）。24 张截图、action-v1 prompt、temperature 0 逐条请求：热更新后与全新启动的文本一致 95.8%（23/24），文本一致样本的逐 token logprob 最大差 0.064；全新启动自身两遍 100%、0.061。第 9 轮对第 11 轮 87.5%、0.176，说明权重确实换了。热更新后的数值差异与 sglang 自身噪声同量级。结果在 `var/runs/sglang-hotreload-20261124/` |
| k8s `livenessProbe`（exec） | 模板 deployment 原本没有 probe，`restartPolicy: Always`，API 接受新增 probe。心跳停止约 41 秒后 kubelet 在**同一 pod、同一节点**原地重启容器（restartCount 0→1），事件 `failed liveness probe, will be restarted` |

## 设计

### 1. 生命周期

- run 被 launcher 领取并 staging 完成后，创建两个 Deployment，沿用 `alvarez-run=<run_id>` 标签：
  - `alvarez-train-<id>`：常驻训练 worker。
  - `alvarez-roll-<id>`：常驻 rollout 服务。
- run 成功、失败、被取消时，`delete_deployment(run_id)` 一次删掉两者（现有逻辑，按标签删）。
- 不跨 run 共享，不做 pod 池。

### 2. 训练 worker（`h800_launcher train-worker <run_dir>`）

- 进程启动时加载 `processor` 和基础模型（`prepare_model` 后的 bf16 模型），常驻显存。
- 循环：在 `<run_dir>/round-N/` 里等 `spec.json`，有 `COMPLETE` / `FAILED` 的跳过，找到最早一个待做的轮次就训练。
- 每轮：
  - 权重来源：`spec.json` 的 `base_model` 必须等于 worker 内存里当前权重对应的路径（第 0 轮是 base，之后是上一轮输出）。不相等时（例如容器刚重启、或 launcher 换了权重），从磁盘重新加载。这保证常驻与"每轮从 CFS 加载"在数值上等价。
  - optimizer 每轮新建，与现在一致（现在每轮也是新 AdamW）。
  - 训练、保存、LoRA 合并（`merge_for_rollout`）、写 `COMPLETE`，与现在的 `run_pod()` 相同。训练函数 `minicpm_training.train()` 增加一个可选参数接收已加载的 `(model, processor)`，不传时行为不变。
- 心跳：后台线程每 10 秒写 `/tmp/alive`（容器本地，不经 CFS，避免 CFS 卡顿被误判）。训练中 GPU 计算不阻塞这个线程。
- 显存：一轮结束后释放 optimizer 和激活；LoRA 轮次的合并副本写盘后不留在显存。

### 3. rollout 服务（常驻 sglang）

- 第 0 轮用 base 权重启动 sglang 和适配层（现有 `sglang-adapter`）。
- 之后每轮，launcher 在训练 `COMPLETE` 后不删 pod，而是写 `serve/next.json`（`{"version", "path"}`，path 是完整权重或合并副本）。适配层看到后：
  1. 停止接受新请求（返回 503，consumer 按现有的模型不可用逻辑重试）。
  2. 调 sglang `/update_weights_from_disk`（`abort_all_requests: false`；此时上一轮 rollout 已全部结束，没有在途请求）。
  3. 把对外的 `model_version` 改成新版本（适配层按版本拒绝请求，必须同步更新），写 `serve/current.json`，恢复服务。
- 如果权重不能给 sglang（`sglang_servable` 为假，或合并失败退回 HF）：这一轮删掉常驻 sglang pod，按现在的方式起一个 HF `serve-pod`，下一轮可以再用 sglang 启动。不在一个 pod 里混用两种后端。
- 合并副本：上一轮的合并副本在热更新成功后删除（现有 `merged_copy` 记录）。
- 心跳：livenessProbe 用 `exec` 在 pod 内请求适配层的本地 `/health`（`127.0.0.1:<port>`，端口从 `serve/ready.json` 读）。热更新期间 `/health` 继续返回 200，只是 `infer` 返回 503。

### 4. 探活与重建

两层，覆盖不同故障：

| 故障 | 发现 | 处理 |
|---|---|---|
| 进程卡死、崩溃、死锁（CUDA hang、sglang 子进程挂掉） | k8s livenessProbe：`periodSeconds: 10`、`failureThreshold: 3`；`startupProbe` 给首次加载留足时间（训练 10 分钟、sglang 15 分钟） | kubelet 原地重启容器，pod 不重新调度。训练 worker 重启后从 CFS 上当前轮的状态继续（本轮没有 `COMPLETE` 就重做本轮）；rollout 服务按 `serve/current.json` 的权重重新启动 sglang |
| pod 被抢占、deployment 被删、节点故障 | launcher 每 15 秒 tick 检查 deployment 是否存在、rollout endpoint 是否健康 | 重建 deployment（现有逻辑）。consumer 在 endpoint 撤下期间拿不到服务的 rollout 按现有重试处理 |
| 原地重启反复失败（crashloop） | launcher 读 pod 的 `restartCount` | 同一轮重启超过 5 次，判 run 失败并写明原因，不无限重试 |

心跳和健康检查都在容器本地完成，不经过 CFS。CFS 出现 NFS BADSESSION 卡死时，训练会因为读写 CFS 卡住，这种情况 probe 能发现并重启；但如果整台节点 CFS 都卡死，重启也恢复不了，靠 crashloop 上限让 run 失败（见 `cfs-nfs-badsession` 运维记录）。

### 5. checkpoint 拷到 4090

- `copy_to_serving` 改为后台线程：训练 `COMPLETE` 后 launcher 立即推进下一轮（热更新、请求 rollout），拷贝和 SHA-256 核对在后台完成后再 `register`。
- 下一轮 rollout 用的是 CFS 上的权重，不依赖 4090 上的副本，所以不阻塞。
- run 结束（`succeed`）前等待所有后台拷贝完成；任何一次核对失败都让 run 失败。

### 6. 资源占用

- 每个在线 run 常驻 2 张 H800（训练 1、rollout 1），整个 run 期间都占着，包括空闲阶段（训练时 rollout pod 空闲，rollout 时训练 pod 空闲）。
- `max_runs` 的含义不变（并发 run 数），实际 GPU 数是它的 2 倍。控制台资源页需要显示这一点。

## 验证（上线前必须全部通过）

| 检查 | 标准 |
|---|---|
| 数值等价 | 同一 run 配置（2 轮，每轮 4 题，dr_grpo 全量、LoRA 各一次）在常驻和现在的方式下各跑一次，固定 seed：第 0 轮训练 loss 一致到 bf16 噪声；第 1 轮 old_logprob 与 rollout 采样一致 |
| 热更新 | 每轮热更新后，适配层 `/health` 报的版本与 `current.json` 一致；用新版本请求成功，用旧版本请求 404 |
| 探活 | 人为让训练 worker 卡住（`kill -STOP`）和让 sglang 子进程退出，各一次：容器原地重启，run 继续并成功 |
| 重建 | 人为删除 rollout deployment：launcher 重建，run 继续并成功 |
| crashloop | 人为让 worker 启动即失败：重启 5 次后 run 失败，原因写明 |
| 时间 | 第 9 轮规模（约 440 行）的每轮总时长比现在少 6 分钟以上 |

## 任务拆分

| 编号 | 内容 | 依赖 |
|---|---|---|
| T1 | `train()` 接收已加载模型；`train-worker` 子命令（循环、权重来源校验、心跳）；单测：两轮 worker 与两次独立 `run_pod` 的权重和梯度一致 | 无（只改 `minicpm_training.py` 和新增入口） |
| T2 | 适配层热更新（`next.json`、503 窗口、版本切换、`current.json`）；单测用假 sglang 服务 | 评测 sglang 改动落地 |
| T3 | launcher：常驻两个 Deployment、轮次推进改为写 `spec.json` / `next.json`、probes、crashloop 上限、后台拷贝；单测覆盖 manifest 与状态机 | T1、T2 |
| T4 | 部署（只重启 `alvarez-h800-launcher`），按上表验证，结果写 WORKLOG | T3 |

## 风险

- 常驻占卡：run 期间空闲阶段的 GPU 不能给别的任务用；best-effort 资源被抢占时两个 pod 都要重建。
- 训练 worker 的显存泄漏会跨轮累积；每轮结束记录 `torch.cuda.memory_allocated`，超过首轮 1.5 倍时主动退出让容器重启。
- sglang 热更新依赖 0.5.23 的实现（同 rollout 方案的补丁风险）；升级镜像要重测本方案的前置验证。
- 热更新后的贪心输出与全新启动有约 4% 的样本不同（与 sglang 自身噪声同量级），rollout 是 temperature 1.0 采样，不影响；不用于评测。
