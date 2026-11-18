# 推箱子 SFT

本流程复用现有 MiniCPM 训练器、数据集注册 API 和控制台。教师只用于离线示范标签；自主评测时模型只收到当前截图和通用任务指令。

## 数据与任务

`fixtures/sokoban/levels.json` 固定包含 48 个训练、8 个验证和 8 个测试关卡，6×8 棋盘、单箱子、单目标。关卡边界封闭，最短解为 4～10 次方向键操作；墙与目标组成的静态布局在所有关卡间唯一。seed 控制训练样本顺序，不改变关卡布局。

- `android-sokoban-v1`：兼容原有固定关卡。
- `android-sokoban-v1:train-000` … `train-047`：训练关卡。
- `android-sokoban-v1:validation-000` … `validation-007`：验证关卡。
- `android-sokoban-v1:test-000` … `test-007`：最终测试关卡。

`src/alvarezgui/sokoban.py` 提供确定性的 BFS 求解器。重新生成清单可用 `.venv/bin/python -m alvarezgui.sokoban`；输出只含棋盘，没有预存动作序列。

完整采集预期产生 **333 条训练、57 条验证、55 条测试样本**。训练动作分布为上 71、下 61、左 98、右 103。这里是路径上的状态数，不能当作 333 个独立关卡。

## 真实 Android 采集

在具备 Android runtime 的节点执行。当前 GPU 节点同时运行 Docker Android，因此图片可以直接保存到 GPU 可读的 `/home/alvarez-datasets/`。采集前应清空同节点、同 backend 的活动任务，并在采集期间停止提交该节点的调度任务；采集器直接使用 runtime，不占控制面调度 lease。尤其 Firecracker 的地址分配器只在进程内共享，不能与另一个 runtime 进程同时创建 VM。fixture HTTP 服务需要已经运行，并已部署新的 `index.html`、`levels.json` 和 runtime 模块。

先运行一个关卡：

```bash
/home/alvarez-runtime/.venv/bin/python -m alvarezgui.sokoban_sft collect \
  --env-file /home/alvarez-runtime/app/.env \
  --levels /home/alvarez-runtime/app/fixtures/sokoban/levels.json \
  --output /home/alvarez-datasets/sokoban-smoke-new \
  --level-id train-000
```

再采集全量：

```bash
/home/alvarez-runtime/.venv/bin/python -m alvarezgui.sokoban_sft collect \
  --env-file /home/alvarez-runtime/app/.env \
  --levels /home/alvarez-runtime/app/fixtures/sokoban/levels.json \
  --output /home/alvarez-datasets/sokoban-sft-v1 \
  --workers 4
```

输出目录必须不存在，避免覆盖已有数据。每个 worker 复用一个 sandbox，逐关重新初始化。每步执行如下核验：

1. 获取真实截图及无障碍棋盘，确认与教师当前状态一致。
2. 保存动作前截图，执行对应方向键 tap。
3. 再次观察并核对整个棋盘、reward、done；最后必须看到 `Sokoban complete`。
4. 整个关卡成功后才写入该 episode 的 JSONL。全部关卡成功、语义状态去重通过后，才输出分割文件和 `status=complete` 报告。

无障碍棋盘用 `_` 表示空地，避免 Android 压缩空格；模型不读取这段状态数据。采集要求实际设备为经过校准的 720×1280 布局。失败保留 `latest-observation.json` 和已采集图片以便诊断，不注册半成品；修复后使用新目录重新采集。

输出：

```text
report.json                 # 采集状态、backend、seed、清单/JSONL hash、方向分布
levels.json                 # 使用的固定清单
train.jsonl                 # 按固定 seed 打乱
validation.jsonl
test.jsonl
episodes/<level>.jsonl      # 逐关核验完成的轨迹
images/<level>/000.png      # 动作前截图
images/<level>/complete.png # 完成状态证据，不作为训练标签
```

JSONL 遵循现有训练 schema：`id/episode_id/group_id/step_id/image/conversations/reward/done`。`image` 是本节点绝对路径；迁移数据时必须同步图片并改写路径。user turn 使用 `service_instruction(SOKOBAN_INSTRUCTION)`，assistant turn 仅为动作 JSON。`board`、`direction`、`optimal_directions` 是离线核验元数据，训练器不会将它们拼入 prompt。

终点 tap 的 `done=true`、reward=1，其他正确移动通常 reward=0；SFT 都会学习。runtime 在目标达成时自动结束，所以无需额外编造 terminate 样本。

## 注册与训练

只注册训练 split，验证和测试数据不能混入训练：

```bash
/home/alvarez-runtime/.venv/bin/python -m alvarezgui.sokoban_sft register \
  --env-file /home/alvarez-runtime/app/.env \
  --dataset-dir /home/alvarez-datasets/sokoban-sft-v1 \
  --version sokoban-sft-v1 \
  --control-plane-url http://10.41.51.215:8600
```

注册要求完整 48/8/8 采集、JSONL hash 不变、训练图片存在、动作合法。重复版本注册由现有 API 返回冲突。控制台刷新后，在新建训练实验中选择 SFT 和该数据集即可使用；学习率、epochs、梯度累积均已有对应字段。控制台任务单选仍是原固定关卡，多关卡留出评测使用下述 CLI/API。

本轮链路验证配置：算法 **SFT**，base **minicpm-v-4.7-1b**，学习率 **1e-7**，epochs **1**，gradient accumulation **4**。这是 BF16 全参数微调；该配置尚未证明能提升策略效果。训练会卸载同卡推理模型，释放显存后更新并保存、注册 checkpoint。

通过 API 可以明确配置八个验证关卡；从本机仓库执行以下 Python（密钥仅从已有配置读取）：

```python
from alvarezgui.config import Settings
from alvarezgui.training_agent import ControlPlane

settings = Settings(_env_file=(".env", "var/platform.env"))
cp = ControlPlane("http://127.0.0.1:8600", settings.console_password.get_secret_value())
run = cp.call("POST", "/api/v1/training/runs", {
    "algorithm": "sft",
    "base_model": "minicpm-v-4.7-1b",
    "dataset_version": "sokoban-sft-v1",
    "environment_version": "android-docker-v1",
    "reward_version": "android-task-v1",
    "runtime_backend": "docker",
    "config": {
        "model_name": "sokoban-sft-v1",
        "learning_rate": 1e-7, "epochs": 1, "gradient_accumulation": 4,
        "seed": 20261117, "max_steps": 20, "eval_trials": 0,
        "tasks": [{"task_type": "sokoban", "app_image": f"android-sokoban-v1:validation-{i:03d}"}
                  for i in range(8)],
    },
}, {"Idempotency-Key": "sokoban-sft-v1:first-training"})
print(run["run_id"])
```

使用 `.venv/bin/python` 执行。示例与已完成的首轮请求一致，固定 Idempotency-Key 可避免重复提交；要开展新实验应使用新 model_name 和 key。本轮 `eval_trials=0`，训练后单独顺序运行状态评测与自主评测，避免与模型加载争用。新实验若需要自动评测，可设置 `eval_trials=1`；自动评测沿用该 run 的 runtime backend。

## 留出评测

状态评测对基础模型和训练后的版本使用同一批真实留出截图：

```bash
/home/alvarez-runtime/.venv/bin/python -m alvarezgui.sokoban_sft evaluate-states \
  --dataset /home/alvarez-datasets/sokoban-sft-v1/validation.jsonl \
  --model-url http://127.0.0.1:28081 \
  --model minicpm-v-4.7-1b --model sokoban-sft-v1-final \
  --temperature 0 --temperature 1 \
  --output /home/alvarez-datasets/sokoban-sft-v1/validation-comparison.jsonl
```

模型服务地址应使用执行节点上实际监听的地址。命令记录原始输出，另写 `.summary.json`：

- `valid_rate`：严格单 JSON 且通过动作 schema 的比例，拒绝带解释前缀的输出。
- `allowed_rate`：合法输出中符合 Sokoban runtime 允许范围的动作，占所有样本的比例。
- `correct_rate`：使用指定按钮中心、且属于任一最短解首步的比例，占所有样本的比例；等价最短路径不会被误判。

错误输出计入分母；服务不可用等基础设施异常会使评测失败，不伪装成模型错误率。状态正确率不等于自主通关率。

最终测试需让模型从未训练的关卡初态自主交互：

```bash
/home/alvarez-runtime/.venv/bin/python -m alvarezgui.sokoban_sft evaluate-rollouts \
  --env-file /home/alvarez-runtime/app/.env \
  --control-plane-url http://10.41.51.215:8600 \
  --levels /home/alvarez-runtime/app/fixtures/sokoban/levels.json \
  --split test --model minicpm-v-4.7-1b --model sokoban-sft-v1-final \
  --temperature 0 --max-steps 20 --name sokoban-sft-v1-test-greedy
```

再以 `--temperature 1` 和另一个 `--name` 检查计划中的 GRPO 采样温度。相同请求重复提交会复用幂等结果。控制台可查看返回的 eval_id、成功率、非法输出及完整轨迹。关卡数量很少，结果只能说明这套单箱子小棋盘上的初步表现，不能外推到复杂推箱子或通用 GUI。

## 首轮实测（2026-11-17）

数据集 `sokoban-sft-v1` 已完成真实 Docker Android 采集和注册，共 333/57/55 条 train/validation/test 样本。445 张动作前图片均通过解码、720×1280 尺寸和 SHA-256 校验；64 关教师轨迹全部核验成功。这是示范采集结果，不是模型自主通关结果。

训练 `run_c76f9931087e4622908b7766a353243f` 状态为 `SUCCEEDED`，处理 333 条样本，实际执行 84 次 optimizer 更新，平均 loss 为 0.768415，耗时 475.45 秒。产出模型 `sokoban-sft-v1-final` 已注册并成功加载；checkpoint 位于 GPU 节点 `/home/alvarez-checkpoints/run_c76f9931087e4622908b7766a353243f/final`。

57 个 validation 状态使用相同截图、seed 和通用指令进行比较：

| 温度 | 模型 | 严格合法动作 | 最短路径正确动作 |
|---|---|---|---|
| 0 | 基础模型 | 57/57（100%） | 15/57（26.3%） |
| 0 | SFT | 57/57（100%） | 15/57（26.3%） |
| 1 | 基础模型 | 53/57（93.0%） | 13/57（22.8%） |
| 1 | SFT | 50/57（87.7%） | 16/57（28.1%） |

这些合法动作也均满足 runtime 允许范围。温度 0 下，两模型仅 3/57 个输出不同，SFT 在 54/57 个状态选择左键。首轮尚未证明策略提升，采样时合法率有所下降；不能用训练标签的 `action_format_rate=1.0` 代替上述推理指标。

完整权重比较发现 1,554,476,272 个参数元素中只有 1,897,762 个发生可见变化（0.1221%，涉及 431/780 个 tensor）。当前 BF16 权重与 `1e-7` 学习率组合可能限制可表示的更新幅度，但这不是对效果问题唯一根因的证明。本轮未增加学习率、训练轮数或引入 LoRA。

原始指标、模型输出和逐参数比较保存在本机 `var/runs/sokoban-sft-v1/`。自主 test 评测使用八个独立关卡、每关一次、20 步上限、seed=20261117，实际运行于 Docker Android；每步模型只接收当前截图和通用指令。

| 温度 | 模型 | 自主通关 | 达到步数上限 | 非法输出终止 |
|---|---|---|---|---|
| 0 | 基础模型（含一次基础设施补测） | 0/8 | 8 | 0 |
| 0 | SFT | 0/8 | 8 | 0 |
| 1 | 基础模型 | 0/8 | 6 | 2 |
| 1 | SFT | 0/8 | 3 | 5 |

温度 0 原评测 `eval_70ed13d56f61458a80cc53eb29da06c8` 中，基础模型的 test-004 在模型动作前因 `adb root` 连接关闭失败，故原始结果是 7 次步数上限和 1 次基础设施错误。原记录保留，独立补测 `eval_0b0034144c884558a36b607eb7b55a96` 使用相同模型、关卡、seed、温度和步数上限，正常执行 20 步后失败。表格采用该关补测结果，不将初始化错误当作策略失败，也不计作第九个关卡。

温度 1 评测 `eval_7c772b5242cb48388feb28f236229eb5` 没有基础设施错误。非法输出包括 `duration_ms=0` 的 swipe，以及 JSON 缺逗号或多余右花括号，均按原有规则拒绝，没有修补输出或放松校验。表中非法输出数是因此终止的 episode 数；与前面的固定留出截图合法率使用不同分母，不可直接比较。

两种温度下均未观察到自主通关，首轮 SFT 没有解决偏左和输出合法性问题。数据采集、训练、checkpoint 注册加载及评测流程已验证可用，但该 checkpoint 不能作为策略改善的证据。完整 33 次尝试（含 1 次初始化失败和 1 次补测）的 API 结果及轨迹已归档；评测结束后远端无运行中的 Docker 容器，相关服务保持 active。

## 回归验证

```bash
PYTHONPATH=src:tests .venv/bin/python -m unittest \
  tests.test_sokoban tests.test_sokoban_sft tests.test_android_tasks \
  tests.test_minicpm_service tests.test_minicpm_training tests.test_console.ConsolePureTest \
  tests.test_docker tests.test_firecracker
```

JS/Python 规则一致性检查需要 Node.js；真实梯度累积回归需要训练环境中的 torch 和 accelerate。依赖缺失时的 skip 应单独记录，不能当作真实训练验证。
