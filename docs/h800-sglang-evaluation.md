# H800 评测服务改用 sglang 方案

- 状态：已实现并部署（2026-11-24），T1–T4 完成，验证结果见 WORKLOG 11-24。
- 2026-11-24 更新：一个评测已改为每个模型一个 pod（迁移 016，见 WORKLOG），本文第 3 节"一个 pod，一个对外端口"的多 sglang 共卡设计只用于更早开始的评测。原因：多个 sglang 按启动时空闲显存分池，后启动的服务 KV 和 mamba 池都更小，模型数和模型大小又不固定。
- 前置：`docs/h800-sglang-rollout.md` 已实现并部署；所有 H800 pod 已统一用 `sglang_v0.5.23:base`（WORKLOG 11-24）。
- 范围：只改 H800 评测服务（`alvarez-eval-*` Deployment，`accelerator: h800` 的评测）。4090 评测、rollout 服务、训练 pod、控制面和 consumer 都不变。
- 预计工作量：1–1.5 天（实现半天，单测 2 小时，一致性验证半天，部署与一次真实评测 2 小时）。部署要先确认没有运行中的 H800 评测。

## 目标

现在 H800 评测走 `serve-pod`：HF transformers 逐条生成，一把 `generation_lock` 串行处理请求，同一 pod 预加载评测的全部模型（`serve_pod()`，`h800_launcher.py`）。rollout 实测这条路径约 0.53 请求/秒，sglang 适配层 16 并发 13.0 请求/秒。评测上限已调到 2304 个 trial，一次 1152 trial、每 trial 最多 5–50 步的评测，HF 服务是主要瓶颈。

目标：完整 MiniCPM-V 4.7 权重的模型在评测里走 sglang，评测总时长由 Android 沙箱步进和并发度决定，而不是模型服务。

## 现状（决定设计的事实）

| 事实 | 影响 |
|---|---|
| 一个评测对应一个 Deployment、一个 endpoint、一个 token；控制面 `evaluations.model_endpoint` 只有一列，consumer 用它请求所有模型版本 | 对外仍须是一个 endpoint，按 `model_version` 分发到各模型 |
| 评测最多 6 个模型版本（`EvaluationRequest.model_versions` `max_length=6`） | 一个 pod 里最多 6 个模型 |
| `SglangService` / `sglang_adapter.run()` 只支持一个模型（`[(version, path)] = spec["models"].items()`） | 适配层要改成多模型 |
| sglang 只能加载完整权重；`sglang_servable()` 拒绝 adapter、perception、`inference_mode`、非 `minicpmv4_7`（含 Qwen3-VL） | 每个模型单独判断后端；LoRA 要先合并 |
| rollout 用 `--mem-fraction-static 0.5`，1B 模型启动后显存约 40 GB；H800 80 GB | 多个 sglang 进程共卡要按模型数分显存 |
| rollout 一致性（WORKLOG 11-24）：temperature 0 与 4090 HF 逐字一致 90.6–92.7%，不一致都是 top-2 差 0–0.25 nats 的平局；sglang 与自己比 94.8%（并发批处理下贪心不确定）；temperature 1.0 平均 TV 0.142，在采样噪声内 | 评测要和历史 4090/HF 结果对比，贪心不确定性是评测特有的风险，见第 4 节 |
| sglang 0.5.23 有 `enable_deterministic_inference`，GDN 线性注意力后端（`gdn_backend.py`）会遵守 | 可作为评测的可选项，先测吞吐代价 |
| 合并后 LoRA 的三方比对（HF+adapter、HF+合并、sglang+合并）在 rollout 方案里没做 | 评测会对 LoRA checkpoint 打分，必须补做 |

## 设计

### 1. 每个模型选后端

`start_serving()` 对每个模型版本：

- 完整 MiniCPM-V 4.7 权重且 `sglang_servable()` 为真：走 sglang。
- LoRA adapter（`adapter_config.json`），其 base 是 MiniCPM-V 4.7：先合并成完整权重再走 sglang（第 2 节）；合并失败则这一个模型退回 HF。
- 其余（perception、`inference_mode`、Qwen3-VL、未知架构）：走 HF。

一个评测里只要有一个模型走 HF，pod 里就同时有 HF 和 sglang 两类服务。`spec.json` 每个模型记录 `{"path", "backend", "merged_copy"?}`，launcher、适配层和清理都只读这一份。

### 2. LoRA 合并

- 位置：评测工作目录下 `<serving>/<eval_id>/merged/<version>/`，不写到 checkpoint 旁边，避免和训练 run 的 `<checkpoint>-merged` 冲突，也不会误删注册模型。
- 时机：在 pod 里、sglang 启动前，用 `minicpm_service.merge_adapter()`（rollout 已用的同一函数）。1B 模型在 GPU 上约 1 分钟。放在 pod 里而不是 launcher 本机，是因为本机没有 torch。
- 清理：`stop_serving()` 删除整个 `merged/` 目录（只删 `spec.json` 里记录的 `merged_copy`）。
- BF16 合并可能改变输出（`merge_adapter` 文档、tiny4 案例），所以第 4 节的 LoRA 比对是上线门槛。

### 3. pod 与适配层

一个 pod，一个对外端口，一个 token：

- 每个 sglang 模型一个 `sglang.launch_server` 子进程，内部随机端口；`--mem-fraction-static` 取 `0.8 / sglang 模型数`，再给 HF 模型留出显存（每个 1B HF 模型约 4 GB）。6 个模型时每个约 0.13，即约 10 GB，1B 模型足够；启动前若计算值低于 0.1 直接失败并写 `FAILED`，不静默降级。
- 适配层（`sglang-adapter` 子命令扩展）持有 `{version: SglangService | MiniCPMService}`，`/internal/v1/model/infer` 按 `request.model_version` 分发，未知版本返回现有的 404。HF 模型仍用 `MiniCPMService`，保持现有串行锁。
- `/health` 汇总各模型；任何一个 sglang 子进程退出，写 `FAILED`，整个 pod 失败，launcher 现有的 `serving_failed` 路径负责通知控制面。
- 各 sglang 子进程串行启动（同时启动会争 CUDA graph 捕获的显存峰值），6 个模型约 5 分钟。`ready.json` 在全部就绪后才写。
- `pod.sh` 用现有的 `SGLANG_POD_SCRIPT`（已含 `OMP_NUM_THREADS=8`、CUDA compat）；`serve-pod` 只在所有模型都是 HF 时继续使用。

### 4. 验证（上线前必须全部通过）

评测分数要和历史结果对比，门槛比 rollout 严：

| 检查 | 标准 |
|---|---|
| 贪心一致性 | 复用 `scripts/sglang_parity.py` 和 `var/runs/sglang-parity-20261123/` 的 96 张截图。sglang 与 4090 HF 逐字一致率不低于 rollout 的 90.6%，不一致的都是 top-2 差 ≤ 0.25 nats 的平局 |
| 贪心可复现 | sglang 同一输入跑两次：默认模式记录一致率；再开 `enable_deterministic_inference` 测一次一致率和吞吐。若默认模式低于 99% 而确定性模式吞吐 ≥ 5 请求/秒，评测服务默认开启 |
| 合并后 LoRA | 用 `run_0a48e64d…` 某轮 adapter：HF+adapter、HF+合并、sglang+合并 三方 temperature 0 比对，标准同第一行 |
| 多模型分发 | 单测：两个假 sglang 服务 + 一个假 HF 服务，请求按版本路由，未知版本 404，任一子进程退出写 `FAILED` |
| 端到端分数 | 同一评测配置（2 个模型 × 48 关 × 1 trial，temperature 0）分别在 4090 和新 H800 服务上跑，每模型成功率差 ≤ 2 关；记录总时长 |

### 5. 不做的事

- 不做跨评测共享 sglang 服务；每个评测一个 pod。
- 不用 sglang 自带的 LoRA 加载（`--enable-lora`）；先用合并，比对不过再评估。
- 不改 4090 评测服务和控制面的 endpoint 模型（仍是每个评测一个 endpoint）。
- 不改 Qwen3-VL 的服务路径（sglang 0.5.23 对它的支持没探测过）。

## 任务拆分

| 编号 | 内容 | 依赖 |
|---|---|---|
| T1 | 适配层多模型：`run()` 读多模型 `spec.json`，串行启动 sglang 子进程，按版本分发，HF 模型内嵌 `MiniCPMService`；显存分配与下限检查；单测用假 sglang 服务 | 无 |
| T2 | launcher：`start_serving()` 按模型选后端、写新 `spec.json`、选 `pod.sh`；pod 内合并 LoRA；`stop_serving()` 清理 `merged/`；单测覆盖 spec、pod.sh、清理 | T1 |
| T3 | 补做合并后 LoRA 三方比对；贪心可复现与确定性模式吞吐测量；结果放 `var/runs/sglang-eval-parity-<日期>/` | 无 |
| T4 | 部署（只重启 `alvarez-h800-launcher`），按第 4 节最后一行跑 4090 对 H800 端到端评测，结果写 WORKLOG | T2、T3 |

## 风险

- 多个 sglang 进程共用一张卡，CUDA graph 捕获和 KV/mamba 缓存按比例缩小后，单模型吞吐低于 rollout 实测值；6 个模型时总吞吐要实测。
- 贪心解码在并发批处理下不确定，同一评测重跑分数会有小幅波动；如果确定性模式代价太高，评测报告要注明。
- best-effort GPU 可能和其它任务共节点；评测 pod 和 rollout pod 落到同一节点时互相影响。
- sglang 0.5.23 补丁依赖其内部实现，升级镜像前要重新探测（同 rollout 方案）。
