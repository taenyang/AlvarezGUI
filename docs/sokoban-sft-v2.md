# 推箱子 SFT v2：动作与结构化监督对照

本轮在 v1 的真实截图上比较两种监督。教师标签仍来自经真实 Android 执行核验的求解器，不冒称为大模型教师轨迹。学生输入只有截图和通用指令；正确棋盘只出现在训练目标和离线评测中。

## 已实现

- `action-v1` 保持旧提示词和动作输出；`sokoban-plan-v1` 输出 `board`、`decision`、`expected_next_state`、`action`。棋盘为 6×8，状态位置是从 0 开始的 `[row, col]`，动作坐标仍属于完整 720×1280 截图。
- 共享提示词和校验位于 `policy_output.py`。结构化输出必须是完整严格 JSON，不能从错误解释中截取动作；运行时不调用求解器、不修正预测，只执行经原动作校验的 `action`。预测状态的语义正确性由离线评测衡量。
- checkpoint 的 `policy_config.json` 决定协议及默认 token 预算（动作 128、结构化 512）。显式请求 `max_tokens` 仍优先。旧 checkpoint 缺少配置时按动作协议读取。
- 新增 `finetuning_method=lora`。MiniCPM 实际语言侧 114 个注意力线性层接入 rank=16、alpha=32、dropout=0 的 LoRA，共 5,514,240 个 FP32 可训练参数；基础权重为 BF16 并冻结，视觉层不加入 LoRA。
- GPU 环境新增 `peft==0.21.3`，安装时锁定现有 Torch/Transformers/Accelerate 等版本，`pip check` 通过。训练与推理环境使用该依赖，控制面不依赖 PEFT。
- adapter 不合并回 BF16。保存 processor、adapter 配置/权重及协议元数据；加载时需要原基础模型目录，支持相同 LoRA 配方的继续训练。
- 新模式本轮只支持 SFT；拒绝 LoRA/结构化 GRPO/GIGPO，也拒绝将 adapter 当作完整权重直接训练。原全参数模式和历史产物保留。
- 控制台增加微调方式和输出协议选项，轨迹可以展开结构化预测；数据库沿用现有训练 config、dataset、model registry 和 raw_output 字段，不增加表。

## 数据准备

在 GPU 节点执行：

```bash
/home/alvarez-runtime/.venv/bin/python -m alvarezgui.sokoban_sft export-paired \
  --source-dir /home/alvarez-datasets/sokoban-sft-v1 \
  --output-dir /home/alvarez-datasets/sokoban-sft-v2
```

输出目录必须不存在。导出会校验原清单、分割文件、图片及棋盘 hash、完整状态转移和奖励；两套协议保存相同的样本 ID、图片、动作和排列顺序。每套包含：

- `train.jsonl`：333 条，48 个训练关卡。
- `validation.jsonl`：57 条，8 个验证关卡。
- `test.jsonl`：55 条，8 个历史测试关卡。
- `train-small.jsonl`：只从 train 选出的 32 条状态，四方向各 8 条，用于学习能力诊断，不作为泛化成绩。
- `report.json`：来源、文件 hash、样本计数及 `teacher_source=solver-verified`。

注册小样本：

```bash
/home/alvarez-runtime/.venv/bin/python -m alvarezgui.sokoban_sft register \
  --env-file /home/alvarez-runtime/app/.env \
  --dataset-dir /home/alvarez-datasets/sokoban-sft-v2/action-v1 \
  --train-file train-small.jsonl --version sokoban-sft-v2-action-small \
  --control-plane-url http://10.41.51.215:8600
```

结构化版本换成 `sokoban-plan-v1` 目录和独立版本名。完整训练集省略 `--train-file`。外层 JSONL 沿用 `minicpm-sft-v1` 行结构，新增行级 `output_format` 和原始 `instruction` 明确 assistant 内容协议；注册会重新核验对应标签及来源。

## 训练与判定顺序

新建 SFT 时选择匹配的数据集，config 增加 `finetuning_method="lora"` 和对应 `output_format`。首个诊断配置为 lr=5e-5、epochs=10、gradient_accumulation=4、seed=20261117、eval_trials=0，基础模型为 minicpm-v-4.7-1b。协议不匹配在加载 GPU 权重前拒绝。

1. 先在 32 条训练状态上诊断。目标为严格合法输出 100%、正确动作至少 95%；结构化模型另外检查棋盘及动作后果。该目标是实验门槛，不是已取得结果。
2. 若没有学会，先检查监督 mask、梯度、精度和训练强度。保留失败结果，不能仅凭 loss 宣称策略进步。
3. 通过后才扩大到完整 333 条，同样本、同随机种子和训练更新预算做 A/B，使用 validation 选配置。
4. 自主评测与教师隔离，记录成功、步数上限、非法输出、基础设施错误。历史 test 只作回归；新的泛化结论需要另外封存未用于调参的关卡。
5. 恢复轨迹是后续阶段：仅从训练关卡的学生访问状态采集，教师示范与求解器核验结合。不可恢复死局不编造成功标签。

诊断训练状态时必须显式允许：

```bash
/home/alvarez-runtime/.venv/bin/python -m alvarezgui.sokoban_sft evaluate-states \
  --dataset /home/alvarez-datasets/sokoban-sft-v2/action-v1/train-small.jsonl \
  --allow-train --model-url http://127.0.0.1:28081 \
  --model MODEL_VERSION --temperature 0 --output NEW_OUTPUT.jsonl
```

评测按响应中的模型协议解析，数据文件协议不决定模型协议。新增棋盘完全正确率、下一状态正确率、move/push 正确率及延迟、字符/字节长度；字符和字节数不等于 tokenizer token 数。被拒绝的输出保留原文（最多 16000 字符）并计入分母，基础设施异常使评测失败。

## 验证与部署记录

本机定向回归 105 项：102 通过，3 项本地没有 Torch 等依赖而跳过。远端实际 CPU 环境的 9 项训练测试全部通过，包含真实 PEFT 更新/继续训练/保存重载、冻结基础权重、FP32 adapter、重复 seed、监督 mask 与梯度累积尾批。实际 MiniCPM 结构的 meta-device 包装也通过。Node.js 语法及 diff 检查通过。

远端部署备份：`/home/alvarez-runtime/backups/20261117-sokoban-sft-v2/`，包含 runtime/serving 源码归档和安装前依赖清单。本机归档：`var/deployments/20261117-sokoban-sft-v2/`，是重启前工作区快照，不是 v1 源码快照。所有实验产物在 `/home`；记录保存在本机 `var/runs/sokoban-sft-v2/`。

## 实际实验结果

首轮动作协议训练 `run_65de02163a8b44d7bd86bb9b312227a5` 已成功，模型为 `sokoban-v2-action-small-e10-final`：320 个样本步、80 次实际 optimizer 更新、469.10 秒、平均 loss=0.102219。checkpoint 包含 228 个 FP32 tensor，共 5,514,240 个参数，全部有限且非零；真实推理重载成功。

在相同的 32 个训练状态上、temperature=0：严格合法 32/32，正确动作 10/32（31.25%），平均请求耗时 1270.44 ms（包含首次模型加载）。预测为 27 次上、5 次下，没有左或右。各 epoch 的平均 loss 从 0.231006 降至 0.074208，后半程基本停滞；低平均 token loss 未转化为状态决策能力，未通过学习门槛。

真实 tokenizer 的四方向样本检查确认：训练完整序列的 prompt 前缀与推理 prompt 完全相同，动作标签监督 19 tokens（含结束符及换行），结构化示例为 71～73 tokens；没有把教师答案放进 user 输入。模型配置与服务均使用 16x 视觉下采样。

结构化组 `run_1d3482e3758e49779e829d9d51d017ff` 已成功训练：320 个样本步、80 次更新、499.60 秒、平均 loss=0.299136。结构化 adapter 同样包含 228 个有限 FP32 tensor，共 5,514,240 个参数，协议配置为 `sokoban-plan-v1`、默认 512 tokens。

同一批 32 状态、temperature=0 的严格评测结果：合法 13/32（40.625%），动作正确 6/32（18.75%），棋盘完全正确 13/32（40.625%），下一状态正确 13/32（40.625%），effect 正确 19/32（59.375%），平均耗时 3508.53 ms。13 个合法输出的棋盘和下一状态均正确；18 个非法输出中，17 个因棋盘不是六行八列，1 个因棋盘未被墙包围。结构化监督改善了合法样本中的状态预测，但没有达到可执行策略门槛，且严格格式生成成为主要瓶颈。

因此本轮两个 small32 实验都没有通过 100% 合法率、约 95% 动作正确率的门槛。基础模型在相同 action-v1 数据上的基线为合法 32/32、动作正确 9/32（28.125%）；动作 LoRA 只有 10/32（31.25%），且输出方向从基础模型的 29 次左键偏向变为训练后的 27 次上键偏向，尚未证明视觉状态绑定得到改善。基线原始输出和 summary 归档在 `var/runs/sokoban-sft-v2/`。

后续有界诊断保持数据、seed、LoRA 配方和评测不变，仅将动作组学习率从 5e-5 降到 1e-5，实验 `run_cdf63599348c4b6bb07c833f1d7296db` 已成功。32 状态评测合法 32/32、动作正确 15/32（46.875%），比 5e-5 的 10/32（31.25%）提高 5 条，但仍未达到约 95% 的过拟合门槛。训练 loss=0.131440，320 样本步/80 更新，476.67 秒。在额外的四状态记忆诊断中，固定选择每个方向 1 条、训练 20 epochs、lr=1e-5 的语言侧 LoRA checkpoint 只在 4 条中正确 2 条：按顺序为 up、down、left、right，模型输出 up、down、up、down。它仍不能记住左/右两张固定截图，进一步支持视觉路径适配不足的判断。该临时 checkpoint 已注册为 `sokoban-v2-action-tiny4-e20-final`，仅用于诊断，不用于泛化或产品默认模型。额外的视觉 merger LoRA（语言注意力加 `model.merger.mlp.0.linear_1/2`，共增加 237,568 个 adapter 参数）在相同四状态上仍为 2/4，未超过语言侧 LoRA。

真实截图中棋盘区域约 280×320 像素，周围包含浏览器栏、说明文字、控制按钮和 Android 导航栏。固定裁剪并放大的四状态输入诊断已完成：裁剪区域 `(x=180..540, y=270..650)` 后放大到 720×760，训练和评测输入完全一致；语言侧 LoRA 仍为 2/4，和原始截图相同。说明当前瓶颈不是整张截图中的浏览器栏、导航栏或棋盘像素尺寸单独造成的。当前不再继续盲目调整裁剪或学习率。额外的坐标 token 加权诊断（坐标位置权重 5、四状态、20 epochs、lr=1e-5）训练 loss 降至 0.010201，但重载后的动作仍为 2/4；因此低 token loss 和坐标加权都没有解决左/右视觉区分。该加权逻辑尚未合入正式训练器。尚未扩大到全量训练或开展 v2 自主评测。

方向词隔离诊断也未通过：将 assistant 目标压缩为单个 `up/down/left/right`，四状态、20 epochs、lr=1e-5 的训练已完成，loss=0.629424；修正临时推理脚本的服务初始化问题后，重载 checkpoint 的输出依次为 `up`、`down`、`right`、`up`，正确 2/4。失败原因为临时脚本直接构造 `MiniCPMService` 后调用 `_generate`，没有先经 `_model` 设置 `self.torch`，不是训练失败。进一步将四张截图交叉输入，四个来源的输出分别稳定为 `up`、`down`、`right`、`up`，与请求行无关，说明模型确实对截图变化产生不同输出，但仍不能正确区分左、右状态。该结果表明瓶颈在视觉状态到方向决策的准确绑定，不能归因于 JSON 或长序列格式；暂不进入 full333、GRPO 或自主评测。诊断结果归档于 `var/runs/sokoban-sft-v2/direction-tiny4-diagnostic.json` 和 `direction-tiny4-permutation.json`；初始脚本错误记录在 `direction-tiny4-initial-failure.json`，临时 checkpoint 保留在 GPU 节点。

后续基线和感知诊断结果进一步收敛了问题：基础模型在相同四张截图的方向词基线为 1/4，语言侧 LoRA 为 2/4；玩家坐标和箱子坐标的 tiny4 训练均为 4/4，但在相同训练预算的 small32 上分别只有 9/32 和 11/32。将视觉塔第 25、26 层 attention 以及 `vision_tower.vit_merger.linear_1/2` 加入 LoRA 后，玩家坐标 tiny4 仍为 4/4，small32 为 10/32，未超过语言侧 LoRA 的 9/32。说明当前问题不是输出 JSON，而是少量样本记忆无法转化为状态定位泛化；该临时视觉 LoRA 配方未合入正式训练器。随后将语言侧玩家坐标 small32 的每样本训练次数从 10 提高到 15，结果为 12/32，训练 loss=0.542418、120 次更新，仍未达到记忆门槛。因此简单增加训练预算也不是当前瓶颈。以上结果归档于 `var/runs/sokoban-sft-v2/direction-tiny4-base-diagnostic.json`、`player-coordinate-tiny4-diagnostic.json`、`box-coordinate-tiny4-diagnostic.json`、`player-coordinate-small32-diagnostic.json`、`box-coordinate-small32-diagnostic.json`、`player-coordinate-small32-vision-diagnostic.json` 和 `player-coordinate-small32-e15-diagnostic.json`。继续暂缓 full333、GRPO 和自主评测。

## 感知分类头诊断

训练器增加了临时 `--algorithm perception` 路径。它忽略 assistant/action 文本，只读取真实截图经过 MiniCPM 视觉编码器的特征，在冻结特征上训练两个 48 类线性头，分别预测玩家和箱子的 6×8 棋盘位置。命令示例：

```bash
/home/alvarez-runtime/.venv/bin/python -m alvarezgui.minicpm_training \
  --algorithm perception --base-model /home/alvarez-models/MiniCPM-V-4.7-1B \
  --dataset /home/alvarez-datasets/sokoban-sft-v2/action-v1/train-small.jsonl \
  --output-dir /home/alvarez-checkpoints/sokoban-perception-small32 \
  --learning-rate 1e-3 --epochs 10 --gradient-accumulation 4
```

输出 `metrics.json`、`perception_predictions.jsonl` 和可独立重载的 `perception_head.pt`。`player_accuracy` 与 `box_accuracy` 是训练完成后对输入样本的重新评估结果；只有两者都达到至少 31/32，才继续加入动作辅助损失。方向诊断在相同命令后增加 `--direction`，会额外训练 4 类方向头；`--direction-mlp` 仅用于线性头失败后的容量对照。

2026-11-18 在 GPU 节点真实运行 small32、冻结视觉特征、lr=1e-3、10 epochs、accumulation=4 的全局平均对照：视觉特征维度为 1152，80 次更新，玩家位置 4/32、箱子位置 5/32、两者同时正确 0/32。结果归档于 `var/runs/sokoban-sft-v2/perception-small32-e10-r2/`。

同日将真实 `pooler_output` 的 7 个视觉切片分别做 16-token 自适应平均池化后重跑相同配置：特征维度为 114688，80 次更新，玩家位置 31/32（96.875%）、箱子位置 32/32（100%）、两者同时正确 31/32（96.875%）。唯一玩家错误为目标类别 29、预测类别 30；箱子无错误。结果归档于 `var/runs/sokoban-sft-v2/perception-small32-spatial-e10-r1/`，通过感知分类头 31/32 门槛，说明前一轮低分主要由全局平均丢失空间信息造成。

为区分视觉特征和语言动作解码，新增可选 4 类方向头，标签只取数据行的监督方向，输入仍只有截图。相同 small32 配置下，冻结空间特征的线性方向头为 25/32（78.125%）：up/down 各 8/8，left 5/8，right 4/8；错误集中在左右方向。随后使用 256 维隐藏层的方向 MLP，结果降为 17/32（53.125%），未通过 31/32 门槛。结果分别归档于 `var/runs/sokoban-sft-v2/perception-small32-spatial-direction-e10-r1/` 和 `var/runs/sokoban-sft-v2/perception-small32-spatial-direction-mlp-e10-r1/`。

随后使用同一 crop 数据、seed、lr=1e-3、10 epochs、accum4 重新训练线性方向头。裁剪放大输入的方向准确率为 32/32（100%），up/down/left/right 均为 8/8；原始整图方向头为 25/32（78.125%）。这表明方向错误也显著受棋盘在输入中的占比和空间分辨率影响，当前 small32 上没有证据表明线性方向头本身缺乏区分能力。结果归档于 `var/runs/sokoban-sft-v2/perception-small32-crop-direction-e10-r1/`。

在此基础上实现 `--multiview`：训练器从同一原始截图生成完整视图和棋盘裁剪放大视图，分别提取冻结视觉特征后沿特征维拼接，checkpoint 记录 `visual_views=full+board-crop`。small32 联合训练 structured-state 和线性 direction head 的结果为：player 25/32、box 26/32、goal 25/32、wall cell accuracy 99.9349%、wall exact 31/32、direction 31/32；拼接后特征维度为 196608。方向和墙体保持高准确率，但位置和目标头下降，说明直接拼接两种高维视图在当前 32 条样本和 80 次更新下不适合作为统一状态头输入。结果归档于 `var/runs/sokoban-sft-v2/perception-small32-multiview-structured-direction-e10-r1/`。

随后增加 `--structured-state` 多任务诊断，在相同空间特征上保留 player/box 头，并新增 goal 48 类头和 48 格墙体二值头；输入仍只有真实截图。small32、lr=1e-3、10 epochs、accum4 结果为：player 31/32、box 32/32、goal 31/32；墙体逐格准确率 98.4375%，但严格整张墙体布局仅 15/32，未达到 31/32。goal 唯一错误为目标类别 38 预测为 37；墙体错误集中为少数格点误判。结果归档于 `var/runs/sokoban-sft-v2/perception-small32-spatial-structured-e10-r1/`。

为区分整张截图中的棋盘占比问题与视觉特征能力，使用相同 32 条样本、seed、训练配置和结构化头，将真实截图裁剪为 `(180, 270, 540, 650)` 的棋盘区域后放大到 `720×760`，重新训练并评估。crop 结果为：player 31/32、box 32/32、goal 27/32、wall cell accuracy 99.9349%、wall exact 31/32。与原始整图的 wall exact 15/32 相比，裁剪放大提升了 16 条，说明墙体整图恢复失败的主要原因是输入布局中棋盘占比和空间分辨率，而不是墙体标签或截图损坏；player/box 保持门槛水平也支持这一点。goal 降至 27/32，表明固定裁剪会改变目标识别分布，不能直接作为所有状态头的默认输入。结果归档于 `var/runs/sokoban-sft-v2/perception-small32-crop-structured-e10-r1/`。

随后实现 `--task-views` 任务级视图路由：player/box/goal 使用完整截图特征，wall/direction 使用同一原始截图生成的棋盘 crop 特征。crop 先按 `(180,270,540,650)` 裁剪并放大到 `720×760`，再统一送入模型的 `720×1280` 输入尺寸，保证与既有 crop 对照的视觉切片数一致。small32、lr=1e-3、10 epochs、accum4 结果为：player 31/32、box 32/32、goal 31/32、wall cell accuracy 100%、wall exact 32/32、direction 32/32、player/box 同时正确 31/32；各 head 的 checkpoint 输入视图和特征维度均被记录。结果归档于 `var/runs/sokoban-sft-v2/perception-small32-task-views-structured-direction-e10-r2/`。

任务级视图路由通过当前 small32 感知诊断门槛。直接拼接模式仍保留为诊断对照，但不作为统一状态输入。

使用最终任务级 checkpoint 重载 MiniCPM、processor 和各 perception heads，重新处理 32 张真实截图并将 direction 映射为 Sokoban 的四个 tap action。重载后 direction 为 32/32，action binding 也为 32/32；四方向抽样均正确：up `(360,836)`、down `(360,980)`、left `(176,980)`、right `(544,980)`。这验证了感知输出到 Android 控件动作的离线绑定，但尚未证明真实 runtime 中的自主连续通关。

双视图 action-v1 策略 SFT 已完成并成功重载：同一真实截图同时输入完整视图和棋盘 crop，语言侧 LoRA，lr=1e-5、10 epochs、accum4、seed=20261117，32 条样本、320 个样本步、80 次更新，loss=0.137431。对同一 32 条真实截图生成时，严格合法输出 32/32、协议校验 32/32、最优动作正确 15/32（46.875%），无解析错误；结果与单视图 lr=1e-5 对照的 15/32 相同，双视图未证明语言动作绑定改善。原始逐样本输出和汇总位于 `var/runs/sokoban-sft-v2/action-small-multiview-e10-r1-evaluation.jsonl` 与对应 `.summary.json`。策略 SFT 未达到约 31/32 门槛，继续不扩大到 full333、不启动 GRPO 或自主评测。

2026-11-18 完成单步视觉动作绑定改进：direction head 使用完整 train 333 条样本、board-crop 任务视图、冻结视觉特征和允许所有等价最优方向的 multi-hot 监督；训练开始时缓存冻结视觉特征，避免每个 epoch 重复编码。10 epochs 的 validation 为 31/57（54.39%），30 epochs 的 checkpoint `sokoban-v2-direction-full-multihot-e30` 达到 36/57（63.16%），比同一 validation 上的 10 epochs 多正确 5 条（提高 8.77 个百分点）。此前 15/32（46.875%）属于训练诊断集，不能与此相减计算提升。使用进程内真实 `MiniCPMService.infer` 复核仍为 36/57，平均延迟 273.74 ms；固定方向到 Android tap 的映射为 up `(360,836)`、down `(360,980)`、left `(176,980)`、right `(544,980)`。训练指标和服务逐样本结果归档在 `var/runs/sokoban-sft-v2/direction-full-multihot-e30-metrics.json`、`direction-full-multihot-e30-validation.json` 和 `direction-full-multihot-e30-service-validation.json`。该结果是单步 validation，不等价于自主连续通关；e30 checkpoint 尚未注册为默认模型。

## 2026-11-18 单步绑定受控复核

复用同一冻结视觉特征，限定五组配置，在 30、100 epochs 两个预算点比较。每组使用相同方向头初始化、seed=20261117、AdamW、batch=4；最后一个不足 4 条的 batch 按实际样本数平均。这里只更新方向头，其他独立头沿用模板权重、不参与方向推理。与之前 Accelerate trainer 的运行不构成逐位复现；本轮的五组之间才是固定初始化的对照。`set-ce` 的损失为全部方向 logsumexp 减去最优方向集合 logsumexp；单标签 CE 训练只取教师方向，但所有组评测均接受任一最优方向。

| 方向监督 / 学习率 / 顺序 | train，30 epochs | validation，30 epochs | train，100 epochs | validation，100 epochs |
| --- | ---: | ---: | ---: | ---: |
| multi-hot BCE / 1e-3 / 固定 | 222/333 | 36/57 | 238/333 | 33/57 |
| multi-hot BCE / 1e-4 / 固定 | 225/333 | 27/57 | 270/333 | 35/57 |
| multi-hot BCE / 1e-4 / 打乱 | 220/333 | 27/57 | 310/333 | 36/57 |
| 单标签 CE / 1e-4 / 打乱 | 230/333 | 28/57 | 314/333 | 35/57 |
| 最优集合 set-CE / 1e-4 / 打乱 | 216/333 | 25/57 | 319/333 | 36/57 |

训练准确率最高 95.80%，但 validation 未超过原 e30 的 36/57；按预先记录的并列保留旧候选规则，继续选择 `sokoban-v2-direction-full-multihot-e30`。未追加 MLP：本轮没有证据说明必须扩大头容量才能达到约 95% 的训练拟合率。降低学习率本身也没有产生稳定的验证收益。只有一个随机种子、8 个验证关卡，不将本轮小样本选优描述为稳定泛化改善。

复核还修复了服务忽略语言 checkpoint `visual_views=full+board-crop` 的问题，推理现在按训练顺序传完整截图与同一裁剪函数产生的棋盘视图。修复后的 small32 双视图语言策略在完全相同的 57 条 validation 上为 21/57（36.84%），严格合法及四方向可执行均 57/57，加载完成后平均请求 1113.50 ms；方向头为 36/57（63.16%）。语言策略只用 32 条训练样本，方向头用 333 条，此差值不是控制训练数据量后的架构因果效应。选定方向头在 55 条历史 test 上为 28/55（50.91%），合法及可执行均 55/55，平均请求 281.10 ms。历史 test 只用于此轮选定后的回归，不用于继续挑选配置，也不作为从未使用过的封存测试。

原 e30 的 21 条 validation 错误中，16 次合法走路偏离最短路径、各增加 2 步，5 次被墙阻挡（其中一次尝试把箱子推入墙）；没有错误实际移动箱子。17 次跨轴、4 次同轴反向；所有错误都只有一个最优方向。57 条图片 hash、棋盘 hash、最优标签与教师轨迹核验通过，抽查 5 张截图与标签一致。旧服务报告的 confusion 按每个最优方向重复累计，总数为 62，不应当成 57 样本互斥混淆矩阵。

复现入口为 `scripts/sokoban_binding_experiment.py` 的 `sweep` 和 `evaluate` 子命令，真实 Android 入口为 `scripts/sokoban_binding_runtime.py`。GPU 训练产物位于 `/home/alvarez-checkpoints/sokoban-binding-followup-r1/`，包括预先落盘的 `plan.json`、缓存、十个 checkpoint、逐样本分数和 `selected.json`；本地指标及错误分析位于 `var/runs/sokoban-binding-followup/`。远端训练与服务测试 32/32 通过（含真实 Torch 和 Pillow），本地定向测试共 39 项：27 通过、12 项因依赖缺失跳过。服务源码已同步到实验 runtime；未重启对外模型服务或修改默认模型。

真实 Android 的 8 个 validation 初始状态已完成单步模型测试：最优动作 6/8，新截图与对应历史截图的预测 8/8 一致，动作后的棋盘与所选方向的转移语义 8/8 一致（7 次移动、1 次被墙阻挡）。这里的 8/8 是点击效果核验，不是决策正确率。前后截图已查看并归档。首轮 test 连续运行在模型产生死局后，被评测脚本的求解器异常中断；该异常来自离线评分，未影响模型输入。失败轨迹保存在 `android-r1-report.json`，修复评分后重跑完整历史 test；不把这次中断伪装成 GPU/Android 故障或模型成功。

修复死局评分后，8 个历史 test 关卡全部完成真实 Android rollout：0/8 通关，53 次模型点击全部与 Android 实际棋盘转移一致（53/53）；模型选择的动作中 22/53 属于当前棋盘的最优方向。8 个关卡均因重复棋盘停止，未触及 24 步上限；`test-000` 有 2 次错误推箱，`test-007` 有 1 次错误推箱，其余主要是绕路或循环，`test-001` 另有 2 次被墙阻挡。沙箱最终销毁成功。该结果是候选方向头的真实连续行为证据，说明单步动作绑定已贯通，但当前策略不能可靠维持连续棋盘状态规划，不能注册为默认模型。完整报告位于 `var/runs/sokoban-binding-followup/android-r2-report.json`。

## 2026-11-18 全量 merger MLP 与 H800 训练旁路

训练器新增 `--train-merger-mlp`，在冻结视觉编码器、保留语言 attention LoRA 的条件下，全量训练 `model.merger.mlp.0`（pre_norm、linear_1、linear_2）。MiniCPM 的 `model.merger.mlp` 是 ModuleList，PEFT `modules_to_save` 必须指向实际子模块 `.0`。merger 共 25,967,104 个参数，加上 5,514,240 个语言 LoRA 参数，合计 31,481,344 个可训练参数；保存后已验证可重载。PEFT 可能压缩 target_modules 名称，恢复校验按实际匹配到的语言 attention 线性层集合判断，避免误拒绝合法 checkpoint。

相同 small32、双视图、seed=20261117、lr=1e-5、10 epochs、accumulation=4 下，merger + 语言 LoRA 为 17/32（53.125%），严格合法与可执行均 32/32；原双视图语言 LoRA 为 15/32。该结果是训练诊断集上的两条增益，不代表 validation 泛化提升。逐样本证据在 `var/runs/sokoban-binding-followup/action-small-merger-e10-evaluation.json`。按预定门槛启动的 4090 full333 同配置训练已完成：loss=0.05795616，3330 个样本步、840 次 optimizer update，训练/导出 6079.04 秒（101.32 分钟）。每轮 333 条在 accumulation=4 下有末尾不足组，所以每轮 84 次更新。指标归档为 `var/runs/sokoban-binding-followup/full333-merger-metrics.json`。同一 57 条 validation 的服务重载结果为 **39/57（68.42%）**，合法与四方向可执行均 57/57，平均推理 1107.71 ms；比完整 train333 方向头的 36/57 多正确 3 条（+5.26 个百分点），是本轮候选的点估计提升，尚不能证明稳定泛化收益。逐样本证据为 `full333-merger-validation.json`。达到门槛后的历史 test 为 **43/55（78.18%）**，合法与可执行均 55/55，平均推理 1127.82 ms；原方向头在同一历史 test 为 28/55。本轮多正确 15 条（+27.27 个百分点），证据为 `full333-merger-test.json`。历史 test 已被反复用于回归，不作为封存泛化结论。真实 Android 评测也已完成：8 个 validation 初始状态单步最优 6/8，8/8 点击状态转移一致（6 次移动、2 次受阻）；这不是 8 关通关测试。8 个历史 test 关卡连续 rollout **3/8 通关（37.5%）**，原方向头为 0/8；42 次点击中 27 次最优（64.29%），42/42 状态转移一致，22 次移动、12 次推箱、8 次受阻。成功关卡 test-000/001/002 分别用 7/5/4 步，全部动作最优；其余 5 关均因重复棋盘停止。沙箱正常销毁，无基础设施异常。报告为 `android-full333-merger-r1-report.json`，已查看并归档成功终局及错误推箱前后截图。

失败关卡首次错误：test-003 第 1 步向下撞墙、test-004 第 2 步向左把箱子推到不可恢复状态、test-005 第 1 步向右推被墙阻挡的箱子、test-006 第 4 步向右推被墙阻挡的箱子、test-007 第 2 步向上绕路后上下循环（步号从 0 开始）。其中 4/5 首错棋盘与历史 test 数据中的棋盘一致，而且离线预测也做出相同错误；test-005 的首错棋盘不在该教师轨迹数据中。故不能把所有连续失败仅归因于偏离教师轨迹；当前状态的动作选择仍有明确错误，但这些证据不能区分模型是否看错状态或未学会规则。逐项分析为 `android-full333-merger-r1-errors.json`。下一轮可以在相同数据配方下对照解冻顶部 2 层；若补充恢复/死锁训练样本，应从训练关卡另行生成，保持历史 test 仅作回归。当前 3/8 通关尚不足以替换默认模型。训练 metrics 中 action_format_rate=1.0 是训练器写入的固定值，不能作为模型生成合法率；合法率和动作正确率以服务重载评测为准。

新增 H800 训练旁路由 `kk /home/github/h800.yaml` 管理 deployment `ainf-run-data-cleaning-kbench-20261118-t141`，有训练需求时从 0 扩到 1。用户指定优先使用 H800；此环境的持久化目录是 `/ainf-cfs/yangtianze/`，不适用普通节点的 `/home` 限制。Pod 可被抢占，不能把容器临时层当作检查点存储。H800 与 4090 不依赖直连，模型和数据经本机中转。

已复制 3GB 基础模型、573 张 v1 图片及 v2 数据，629 个文件的 SHA256 与本机 staging 全部相同。v2 action-v1 与 sokoban-plan-v1 的 train/train-small/validation/test 图片路径改写为 CFS 后，分别通过 333/32/57/55 条加载校验，样本顺序、标签和图片内容不变。模型位于 `/ainf-cfs/yangtianze/alvarez-models/MiniCPM-V-4.7-1B`，数据位于 `alvarez-data`，实验源码位于 `alvarez-runtime/src`。校验清单归档在 `var/runs/sokoban-binding-followup/h800/transfer-manifest.json`。

H800 预装 Python 为 `/opt/nemo_rl_venv/bin/python`，Torch 2.9.0+cu129。实际训练探针发现预装 Transformers 5.15.0 不认识 `minicpmv4_7`，已将 5.18.0 安装到 CFS 的 `alvarez-runtime/deps` 并通过 PYTHONPATH 加载，保留容器预装环境。4 条真实样本已完成反向传播、保存、服务重载和截图推理，输出合法 tap。首轮训练/导出 23.36 秒，包含首次运行成本，不能当作稳定吞吐率；H800 默认约 92 个 CPU 线程，后续限制 OMP/MKL 为 8。两边均未安装 causal_conv1d / flash-linear-attention，使用 PyTorch 回退实现。

新增可选 `--checkpoint-dir`，仅用于单 Accelerate 进程 SFT。每个完整 epoch 使用 Accelerate 保存模型、optimizer 和 RNG，同时记录配方、图片内容/样本顺序摘要、累计 loss/update/step；同目录原子 rename 发布完成状态，只保留最近两份。Pod 恢复后自动读取最新完成 epoch，`--epochs 10` 始终表示总训练 10 轮，最多重做被抢占时未完成的一轮。基础模型与 processor 必须保持不可变，同一目录只允许一个写入进程；多卡恢复尚未支持。恢复后的 `elapsed_seconds` 仅计算本次调用的训练/导出时间，不含历史调用和模型/状态加载时间。

真实 Torch/Accelerate CPU 测试已验证连续训练与中断恢复的参数、AdamW moments、Python/NumPy/Torch RNG 和累计指标相同，并验证未完成保存不被识别、只保留两份完成状态。H800 完整训练测试 24/24 通过；真实 MiniCPM 四样本先训练一轮、从保存状态继续到第二轮，累计 8 样本步/2 更新，恢复探针通过。日志为 `h800/training-tests.log` 和 `h800/resume-smoke.log`。

H800 full333 相同配方已启动，产物根目录为 `/ainf-cfs/yangtianze/alvarez-checkpoints/sokoban-v2-action-full-merger-h800-e10/`，下含 `resume/`、`final/`、`run.log` 及评测 JSON。Deployment 启动命令指向持久化 `alvarez-runtime/scripts/run-full.sh`，Pod 被抢占重建后自动执行恢复；训练结束自动评测 validation，严格超过 36/57 才继续历史 test。启动前配置与 patch 归档在 `h800/deployment-before.json`、`h800/deployment-training-patch.json`。GPU、Torch、PEFT、Accelerate 环境与 4090 不同，H800 结果单独记录，不把两次运行的差值解释为只改变 merger 的效果。原计划完成归档后将副本缩回 0；用户随后授权在 H800 基线完成后继续顶部两层对照，因此整条实验链结束后再缩回 0。

H800 基线现已训练及重载评测完成：3330 样本步、840 更新、10 epochs，loss=0.05650979，训练/导出耗时 4903.91 秒（81.73 分钟）。validation **43/57（75.44%）**，合法/可执行 57/57，平均推理 857.60 ms；历史 test **43/55（78.18%）**，合法/可执行 55/55，平均推理 847.06 ms。validation 比 4090 候选多对 4 条，历史 test 持平，不能把环境不同的结果归因于训练模块变化。`COMPLETE` 已生成，完整逐样本结果与 metrics 归档在 `var/runs/sokoban-binding-followup/h800/`。该 checkpoint 已经本机中转到 4090 完成真实 Android 对照，权重哈希校验一致，仅副本 adapter 的基础模型路径改为 4090 本地路径。单步最优 6/8、状态转移一致 8/8；连续 rollout **2/8 通关（25%）**，成功 test-001/002 分别 5/4 步且全程最优，39 次点击中 21 次最优（53.85%），状态转移一致 39/39。其余 6 关均因重复棋盘停止：test-000/004 进入无解状态后重复受阻，test-003/005/006 在可解状态持续受阻，test-007 上下振荡。沙箱清理成功，Docker 标签残留为空，无评测或清理异常。完整报告、日志、配置改动证据与 63 张截图归档在 `var/runs/sokoban-binding-followup/android-full333-merger-h800-r1/`。相比原 4090 候选的 3/8，本次多失败 test-000；validation 提升未转化为本组连续通关提升，保持原候选和默认模型不变。

## 顶部两层全量微调

2026-11-18 用户明确要求等待 H800 同配置基线完成后尝试解冻语言模型顶部方案。第一轮固定顶部 2 层：MiniCPM 配置共 24 个语言层，第 22 层为 linear attention、第 23 层为 full attention。两层完整 attention、MLP 和 norm 参数参与训练，通过 PEFT `modules_to_save` 保存；这两层不再额外叠加 LoRA，其余 22 层保持 attention LoRA r=16/alpha=32，merger.mlp.0 继续完整训练，视觉编码器、embedding、末端 norm 与 lm_head 冻结。

顶部参数学习率 1e-6，其余可训练参数保持 1e-5。使用原始预训练模型初始化，固定 train333、双视图、seed=20261117、10 epochs、accumulation=4、batch=1、固定顺序和 H800 环境；不从已训练 10 轮的基线接着训练，避免训练预算不等。新配方通过实验 CLI 显式开启，默认训练行为不变；不扩展平台 API 或控制台。

新源码与运行脚本使用独立 CFS 目录 `alvarez-runtime-top2-20261118`，不覆盖正在运行的基线源码。启动前要求 H800 基线完成标记及 metrics/validation 完整；随后先用 4 条真实样本验证顶部模块训练、optimizer/RNG 恢复、checkpoint 重载推理，再启动完整训练。评测仍先看相同 validation；超过同机 H800 基线及当前 39/57、且合法可执行仍 57/57，才进入历史 test。历史 test 不低于现有候选且格式通过时，再做真实 Android 对照。并列保留原候选，不自动替换默认模型。计划和启动材料位于 `var/runs/sokoban-binding-followup/h800-top2/`，训练器已实现可选 `--train-top-layers 2 --top-layer-learning-rate 1e-6`，使用两组 AdamW 参数；启用时在 metrics、policy_config 和恢复配方中记录顶部层配置，默认关闭时保持已有 checkpoint 配方兼容。新源码已单独暂存，H800 环境下禁用 CUDA 的 CPU 回归 27/27 通过；真实 MiniCPM meta 架构检查确认顶部两层 39,908,000 参数、merger 25,967,104 参数、其余语言 LoRA 5,087,744 参数，共 70,962,848 个可训练 FP32 参数。原基线为 31,481,344。检查点保存范围是两个完整语言层与 merger，不包含顶部 LoRA。最终增加的精确断言发现 adapter 继续训练时，完整层 FP32 checkpoint 被 BF16 目标模块舍入；已在 trainable adapter 加载前将 modules_to_save 转为 FP32，避免丢失细微更新。修复后完整 CPU 回归仍为 27/27，包含保存参数精确一致和 Accelerate 恢复等价性。推理仍沿用基础模型 BF16 加载。基线完成后已验证顶部 runtime 八个文件 SHA256 与最终 CPU 验证版本一致，归档切换前 Deployment 并应用 `deployment-top2-patch.json`。新 Pod 已 Ready，真实模型 GPU 探针已通过：4 条样本训练 2 轮共 8 步/2 更新，恢复前后 loss、步数、更新数和轮数相同；保存模块恰为顶部两层及 merger，服务重载输出合法 tap（544,980）。报告为 `h800-top2/smoke-report.json`；短探针不用于准确率结论。已确认正式 full333 训练进程启动，最终验证门槛现为严格超过 H800 基线 43/57。运行至第 7 轮途中发生 Pod 更换；新 Pod 的 `sft_resume` 明确确认从 `epoch-00000006` 恢复，累计 1998 步/504 更新，无恢复错误。此前旧 Pod 日志到 2250 步，未完成的第 7 轮重新执行；最终 metrics 的 elapsed_seconds 仅包含恢复后的本次调用，不能直接作为整次实验耗时。

顶部两层实验已完成：10 epochs、3330 样本步、840 更新，平均 loss=0.05796100；恢复后训练/导出 2043.18 秒。服务重载 validation **43/57（75.44%）**，合法与可执行均 57/57，平均推理 891.65 ms，与 H800 merger 基线持平。逐条配对发现 4 条由错变对、4 条由对变错，共 10 条方向变化；不是参数未生效，但没有净正确率收益。按预先规定的严格提升门槛，不运行本候选的历史 test 和 Android rollout，保留原候选，不替换默认模型。完整 metrics、逐样本 validation、comparison、配对变化与 run.log 已归档在 `var/runs/sokoban-binding-followup/h800-top2/`。顶部两层对照期间未实施 batch/梯度检查点/预处理缓存加速。该实验链结束后曾将 H800 Deployment 副本缩回 0，CFS 模型、检查点与评测结果保留；随后用户授权实施下述加速。

## H800 训练与 4090 推理加速（2026-11-18）

训练器新增 `--batch-size`、`--cache-preprocessed`、`--no-gradient-checkpointing`，默认仍为 micro-batch 1、无缓存、开启梯度检查点。H800 对照固定有效 batch 4、语言 attention LoRA + 全量 merger、相同双视图/学习率/seed，比较 1×4、2×2、4×1；批大小增加不改变样本权重，每条样本先对有效 assistant token 平均，再按累积窗口实际样本数平均，尾批不漏样本。缓存保存 CPU tensor 副本，取用时复制，防止模型或设备搬运污染；它缓存完整预处理输入，333 条当前双视图输入约需 10 GiB 以上主机内存，并非零成本。三个开关写入恢复配方，旧配方缺失字段时按原默认解释；改变配方需要新的恢复目录。多样本组批目前限单 Accelerate 进程。

最终训练源码在 H800 运行完整 CPU 回归 **31/31 通过，无跳过**，覆盖变长 mask、图像顺序、每样本 loss/梯度、尾批、缓存隔离及模型/optimizer/RNG 恢复。真实 processor 对照覆盖固定尺寸和不同尺寸图片（14/10/8 个视觉切片）：统一右侧 padding 后，input_ids、attention_mask、mm_token_type_ids、pixel_values、target_sizes 全部逐元素一致；真实 BF16 GPU 批量与单例前向逐样本 loss 最大差为 0。初始对照失败来自验证脚本图像读取尺寸与 processor 默认左侧 padding 不一致，修正对照输入后通过，未改变训练样本内容。

测速计划与证据保存在 `var/runs/h800-acceleration-20261118/`；独立 CFS 运行目录为 `/ainf-cfs/yangtianze/alvarez-acceleration-20261118/`。每配置从相同预训练模型开始，32 条真实样本训练 2 轮，共 64 样本步/16 更新；完整第一轮预热，第二轮墙钟吞吐包含缓存复制、CPU→GPU、前后向及更新。短测仅衡量速度和实现正确性，不用于动作准确率结论。

五组测速全部完成，每组均为 64 样本步、16 次更新、有效 batch 4。下表稳态吞吐只计预热后的第二轮；训练/导出耗时包含两轮预处理和保存、不包含初次模型加载，显存是 PyTorch 峰值 allocated：

| micro-batch × 累积 | CPU 缓存 | 梯度检查点 | 稳态样本/秒 | 训练/导出秒 | 峰值显存 GiB |
|---|---|---|---:|---:|---:|
| 1 × 4 | 关 | 开 | 0.716 | 96.69 | 9.28 |
| 1 × 4 | 开 | 开 | 0.746 | 89.46 | 9.26 |
| 2 × 2 | 开 | 开 | 1.115 | 64.67 | 15.01 |
| 4 × 1 | 开 | 开 | 1.317 | 54.40 | 26.40 |
| 4 × 1 | 开 | 关 | **3.007** | **26.05** | **46.88** |

最快配置稳态吞吐为基线 **4.20 倍**；包含模型加载的完整短测墙钟从 101.56 秒降至 29.82 秒，缩短 70.64%。峰值 reserved 54.43 GiB，当前约 79 GiB 可见显存有余量。这是 train 前 32 条/2 epochs 的一次短测结果，不是完整 333 条/10 epochs 的实测耗时，也不证明训练后动作准确率不变；BF16 不同 batch 的训练轨迹可能有数值差异。

H800 专用入口 `scripts/h800_sft.sh` 已同步到 CFS 的 `alvarez-acceleration-20261118/scripts/h800_sft.sh`，默认采用 batch 4、累积 1、CPU 缓存、关闭梯度检查点、OMP/MKL 8；普通训练器调用者默认不变。后续新训练在 Pod 内使用：

```bash
bash /ainf-cfs/yangtianze/alvarez-acceleration-20261118/scripts/h800_sft.sh \
  --output-dir /ainf-cfs/yangtianze/alvarez-checkpoints/sokoban-next/final \
  --checkpoint-dir /ainf-cfs/yangtianze/alvarez-checkpoints/sokoban-next/resume
```

输出和恢复目录应选择新实验路径，不能复用原 batch 1 恢复状态。脚本调用现有训练 CLI，保留 epoch 级原子检查点与自动恢复；更大图像/更长答案需要重新确认显存。本次没有启动新的 full333 准确率实验。最快配置的 checkpoint 已验证 114/114 个 LoRA B 矩阵非零、merger 的 6 个权重/偏置 tensor 均与基础模型不同；通过实际 `MiniCPMService.infer` 重载，4 张真实截图全部生成合法 tap。该探针只确认参数更新、保存和服务加载成功，不作为动作准确率结论，详见 `reload-verification.log`。完成归档后已将 Deployment 副本缩回 0，并确认加速测试 Pod 删除；CFS 源码、启动脚本、五组 checkpoint 和报告保留。Deployment 暂留 `sleep infinity` 命令，后续任务需显式启动上述入口；不会重新扩容就自动重跑旧实验。

4090 推理侧采用仅内存的 LoRA 合并，只有 `ALVAREZ_MERGE_ADAPTER_PATHS` 中显式列出的绝对 checkpoint 路径启用 `merge_and_unload(safe_merge=True)`；基础模型与未验证的 adapter 保持原加载方式，原权重不回写。BF16 合并可能改变动作，safe_merge 的有限值检查不能替代输出验证。完整双视图、官方通用指令、57 条 validation 配对中，原 4090 full-merger-e10 候选合并前后 **57/57 原始输出完全一致**，合法/可执行 57/57，最优动作均为 38/57，延迟中位数 **1196.38 → 945.77 ms，降低 20.95%**。这是本次环境内的配对结果，与历史 39/57 差 1 条，原因尚未证实，不声称复现历史分数或提高正确率。

排查发现旧 `/home/alvarez-serving` 未部署双视图修复，初次单图结果 32/57 及 24.4% 提速不能作为正确推理路径的启用依据，相关临时启用已回滚后重测。最终采用已有 `/home/alvarez-runtime/app/src` 源码，仅添加本次合并开关，并通过模型服务专属环境文件设置 PYTHONPATH 与候选白名单；训练源码、训练代理环境不变。当前在线 tiny4 曾在合并对照出现 3 条最优动作回退，因此不在白名单内；部署后真实 HTTP 57/57 输出与原未合并 tiny4 完全一致、0 拒绝。候选未注册或替换当前模型，不能宣称当前 tiny4 或默认基础模型已提速。4090 部署源码服务测试 **13/13 通过**；本地因缺少 Pillow，同组为 12 通过、1 跳过。
