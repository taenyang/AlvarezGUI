# H800 rollout 推理改用 sglang 方案

- 状态：已实现并部署（2026-11-24）。结果见 WORKLOG 11-24 条目；T4 的合并后 LoRA 三方比对未做。探测文件在 `var/runs/sglang-probe-20261123/`。
- 范围：只改 H800 在线 RL 的 rollout 推理（`alvarez-roll-*` Deployment）。训练 pod、4090 模型服务、H800 评测服务（`alvarez-eval-*`）都不变；H800 评测之后可以复用同一套，作为单独一步。
- 预计工作量：1–1.5 天（实现半天，单测 2 小时，一致性验证半天，部署与真实 run 2 小时）。部署要先确认没有运行中的 H800 run。

## 目标

现在 rollout 服务是 `serve-pod`：HF transformers 逐条生成，一把 `generation_lock` 串行处理请求，线性注意力层走纯 PyTorch 参考实现（没有 `causal_conv1d`、`flash-linear-attention`）。实测 H800 上单服务约 0.53 请求/秒、单步约 1.5–1.7 秒，GPU 利用率 25%，比 4090（约 1.1 请求/秒）还慢。改用 sglang 的连续批处理，把 rollout 推理吞吐提到 10 请求/秒左右。

## 探测结论（决定设计的事实）

| 事实 | 影响 |
|---|---|
| `iregistry.baidu-int.com/sglang-public/sglang_v0.5.21:cu130` 只支持 MiniCPM-V 到 4.6，transformers 5.12.1 也没有 `minicpmv4_7` | 不能用 |
| `iregistry.baidu-int.com/sglang-public/sglang_v0.5.23:base`：sglang 0.5.23、torch 2.14.1+cu130、transformers 5.19.0、flashinfer 0.7.2；`_SUPPORT_VERSION` 有 `(4, 7)`，canvas M-RoPE 在 `multimodal/processors/minicpmv4_7.py` | 用这个镜像，Python 是 `/opt/sglang/bin/python3`；首次拉镜像约 6 分钟 |
| 0.5.23 启动预热报 `'MiniCPMV4_7' object has no attribute 'shared_experts_config'`：`MiniCPMV` 包装类不是 `nn.Module`，loader 把属性写到包装类，runner 从内层模块读 | 所有 MiniCPM-V 版本都受影响。用 `sitecustomize.py`（放进 `PYTHONPATH`，spawn 出的子进程也会加载）把这次写入同步到内层模块。不能用 `runpy` 包一层启动脚本：sglang 用 spawn 起子进程，会报 bootstrapping 错误 |
| 打补丁后 `--disable-cuda-graph` 不需要；权重加载约 1 秒，启动到就绪约 50 秒 | 默认参数加 `--mem-fraction-static 0.5`、`--trust-remote-code` |
| temperature 0 下 4 张真实截图（全图 + 棋盘裁剪）与 4090 HF 服务输出逐字一致 | 只说明图像预处理和位置编码大体对了；temperature 1.0 的分布还要另外验证 |
| 单服务吞吐：并发 1 为 2.4 请求/秒（单请求 160 ms），8 为 10.4，16 为 10.9，32 降到 7.9（p50 3.7 秒） | 每个 run 一个 sglang 服务，按 16 并发设计；32 并发时更慢 |
| `max_running_requests` 受 mamba state cache 限制为 186 | 远高于需要，不用调 |

## 设计

### 1. rollout pod

`rollout_service_ready()` 生成的 `serve/pod.sh` 改为两个进程：

- sglang：`PYTHONPATH=<run>/src/alvarezgui/sglang_patch /opt/sglang/bin/python3 -m sglang.launch_server --model-path <本轮权重> --host 127.0.0.1 --port <内部端口> --trust-remote-code --mem-fraction-static 0.5`。
- 适配层：同一个 pod 里，用 sglang 镜像自带的 Python 运行 `alvarezgui.h800_launcher sglang-adapter <serve 目录>`，在随机端口上提供现有协议（`/health`、`/internal/v1/model/infer`，Bearer token），写 `ready.json`。launcher、控制面、consumer 都不用改。

Deployment 的镜像在 manifest 里覆盖为 sglang 镜像，其余字段沿用模板。镜像名放在 launcher 的一个常量里，不做配置项。

补丁文件随源码快照一起提交：`src/alvarezgui/sglang_patch/sitecustomize.py`（内容同探测版本），只在 sglang 版本是 0.5.23 时生效，版本不符时什么都不做并打一行日志。

### 2. 适配层（`/internal/v1/model/infer`）

输入输出与 `MiniCPMService.infer()` 完全一致：

| 步骤 | 做法 |
|---|---|
| 图像 | 复用 `_read_image`（缩放到 720×1280）；`policy_config.json` 的 `visual_views` 为 `full+board-crop` 时，加一张 `_board_crop_image`，顺序为全图、裁剪 |
| prompt | `policy_instruction(instruction, output_format)`，与训练行的 user 内容逐字相同 |
| 生成参数 | `temperature`、`seed`、`max_tokens`（默认取 `policy_config`）、`chat_template_kwargs` 取自 `model_families` 的 `minicpmv4_7` 条目（`enable_thinking: False` 等）；temperature 0 走贪心 |
| 输出校验 | action-v1 用 `_parse_action`，结构化格式用 `parse_policy_output`；失败返回 422，`detail`、`raw_output`、`output_format` 与现有服务一致，consumer 按 invalid output 处理 |
| entropy | 请求 `logprobs` 和 `top_logprobs`，按每个生成 token 的 top-k 分布算熵的近似值；字段名不变。近似值和现在的精确值不可比，在 metrics 名或 WORKLOG 里注明 |
| 延迟 | `latency_ms` 从收到请求算起，包含 sglang 内部排队 |

### 3. 每轮权重

- 第 0 轮：base 模型目录，直接给 sglang。
- 之后每轮：训练产物是 LoRA adapter。服务前在训练 pod 结束后（或 rollout pod 启动前）用 `merge_adapter()` 合并成完整权重，写到 `<checkpoint>/merged-for-rollout/`，sglang 加载这个目录。合并在 H800 pod 里做（CPU 也可以，3 GB 模型约 1 分钟）。
- `merge_adapter` 的文档已经写明 BF16 合并可能改变输出（tiny4 出现过），所以要做第 4 节的一致性验证。不一致时的备选：sglang 自带 LoRA 加载（`--enable-lora --lora-paths`），需要另外验证它对 MiniCPM-V 4.7 是否可用。
- 训练端不变：old_logprob 仍由训练 pod 用 HF 前向重算（`compute_old_logprob=True`），所以采样框架的 logprob 偏差不会进入 ratio。

### 4. 验证（上线前必须全部通过）

| 检查 | 标准 |
|---|---|
| temperature 0 一致性 | 用 base 评测 `eval_dd67d274…` 的截图，取 48 题各 2 个状态共 96 步，sglang 与 4090 HF 输出逐字相同的比例 ≥ 95%；不同的逐条看是否只是相近动作 |
| temperature 1.0 分布 | 同样 96 步，每步各采样 16 次，比较两边动作分布（总变差距离均值 ≤ 0.1） |
| 合并后的 LoRA | 用 `run_0a48e64d…` 某轮 adapter：HF+adapter、HF+合并权重、sglang+合并权重三者在 temperature 0 下比对，标准同第一行 |
| 协议 | 现有 consumer 测试的 invalid output、422、超时、token 校验路径在适配层上跑通 |
| 端到端 | 用新镜像跑一个 2 轮的小 run（每轮 4 题），rollout 无运行错误，每轮 rollout 时间对比现在下降 |

### 5. 不做的事

- 不把 rollout 和训练合并到同一个常驻 pod（NeMo RL 那套），每轮仍是新建 pod、从 CFS 加载。
- 不改 4090 模型服务和 H800 评测服务。
- 不在适配层做跨 run 共享服务；每个 run 一个 sglang 进程。

## 任务拆分

| 编号 | 内容 | 依赖 |
|---|---|---|
| T1 | `sglang_patch/sitecustomize.py` 与版本检查；单测：版本不符时不打补丁 | 无 |
| T2 | 适配层 `sglang-adapter` 子命令：请求转换、输出校验、422、entropy 近似；单测用假的 sglang HTTP 服务 | 无 |
| T3 | launcher：rollout manifest 覆盖镜像，`pod.sh` 启动两个进程；LoRA 轮次先合并；单测覆盖 manifest 与 pod.sh 内容 | T1、T2 |
| T4 | 一致性验证脚本，产出放 `var/runs/sglang-parity-<日期>/` | T2 |
| T5 | 部署与 2 轮端到端 run，结果写 WORKLOG | T3、T4 |

## 风险

- sglang 0.5.23 的补丁依赖它的内部实现，升级镜像前要重新探测。
- top-k 近似的 entropy 与现在的精确值不可比，切换前后的 `rollout/entropy` 曲线不能直接对比。
- best-effort GPU 资源可能和其它任务共用；两个 run 的 rollout 服务被调到同一节点时，吞吐会互相影响。
