"""Import the 2026-11-13 checkpoints and shopping rollouts so the console has real history to show.

Idempotent: fixed ids plus ON CONFLICT DO NOTHING; rerunning changes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from psycopg.types.json import Json

from alvarezgui.config import Settings
from alvarezgui.db import connect

GPU_MODELS = [
    ("minicpm-v-4.7-1b", "file:///home/alvarez-models/MiniCPM-V-4.7-1B", {"note": "官方基础权重"}),
    ("sft-smoke-20261113", "file:///home/alvarez-checkpoints/sft-smoke-20261113",
     {"algorithm": "sft", "loss": 4.040100574493408, "sample_count": 1, "note": "1 条 reward 0 样本的 smoke"}),
    ("grpo-raw-smoke-20261113", "file:///home/alvarez-checkpoints/grpo-raw-smoke-20261113",
     {"algorithm": "grpo", "loss": 0.0, "sample_count": 2, "note": "reward 全 0，权重与 base 相同"}),
    ("grpo-shopping-stable-20261113", "file:///home/alvarez-checkpoints/grpo-shopping-wide-stable-20261113",
     {"algorithm": "grpo", "loss": 0.2670937329530716, "sample_count": 6, "reward_mean": 0.6666666666666666,
      "learning_rate": 1e-7}),
]
HISTORY_RUN = "run_history_grpo_shopping_20261113"
HISTORY_LOG = (
    "2026-11-13 手动执行的 GRPO 实验，导入自 var/runs/shopping-real-group-20261113.json。\n"
    "6 次同状态真实采样，reward [0,1,1,1,1,0]；base model 计算 exact-completion old logprob。\n"
    "lr 1e-5 版本 reload 后输出非法格式已弃用；lr 1e-7 产出 grpo-shopping-stable-20261113。\n"
)


def import_history(settings: Settings, runs_dir: Path) -> dict[str, int]:
    group = json.loads((runs_dir / "shopping-real-group-20261113.json").read_text())
    old_logprobs = [
        json.loads(line)["old_logprob"]
        for line in (runs_dir / "shopping-real-group-20261113" / "grpo-rollouts-old-logprob.jsonl").read_text().splitlines()
        if line.strip()
    ]
    counts = {"models": 0, "rollouts": 0}
    with connect(settings) as connection, connection.transaction():
        for version, uri, metrics in GPU_MODELS:
            counts["models"] += connection.execute(
                """
                INSERT INTO model_versions (version, base_model, checkpoint_uri, status, metrics)
                VALUES (%s, 'MiniCPM-V-4.7-1B', %s, 'READY', %s) ON CONFLICT (version) DO NOTHING
                """,
                (version, uri, Json(metrics)),
            ).rowcount
        rewards = [row["reward"]["total"] for row in group["rows"]]
        connection.execute(
            """
            INSERT INTO training_runs (run_id, algorithm, base_model, dataset_version, environment_version,
                                       reward_version, runtime_backend, config, status, checkpoint_uri, metrics, log_tail,
                                       idempotency_key, request_fingerprint, created_at, updated_at)
            VALUES (%s, 'grpo', 'minicpm-v-4.7-1b', NULL, 'android-firecracker-v1', 'android-task-v1', 'firecracker', %s,
                    'SUCCEEDED', %s, %s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s))
            ON CONFLICT (run_id) DO NOTHING
            """,
            (
                HISTORY_RUN,
                Json({"rounds": 1, "group_size": 6, "temperature": 1.0, "max_steps": 1, "learning_rate": 1e-7,
                      "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}], "imported": True}),
                "file:///home/alvarez-checkpoints/grpo-shopping-wide-stable-20261113",
                Json({"loss": 0.2670937329530716, "sample_count": 6, "update_count": 6,
                      "rollout/reward_mean": sum(rewards) / len(rewards),
                      "model_version": "grpo-shopping-stable-20261113"}),
                HISTORY_LOG,
                f"import:{HISTORY_RUN}",
                "0" * 64,
                group["started_unix"],
                group["started_unix"],
            ),
        )
        for name, value in (
            ("rollout/reward_mean", sum(rewards) / len(rewards)),
            ("rollout/success_rate", sum(r > 0 for r in rewards) / len(rewards)),
            ("rollout/zero_variance_groups", 0.0),
        ):
            connection.execute(
                "INSERT INTO training_metrics (run_id, name, step, value) VALUES (%s, %s, 0, %s) ON CONFLICT DO NOTHING",
                (HISTORY_RUN, name, value),
            )
        for index, (row, old_logprob) in enumerate(zip(group["rows"], old_logprobs)):
            counts["rollouts"] += _import_rollout(connection, group, row, index, old_logprob)
    return counts


def _import_rollout(connection: Any, group: dict[str, Any], row: dict[str, Any], index: int, old_logprob: float) -> int:
    task_id = f"task_history_shopping_{index}"
    episode_id = f"episode_history_shopping_{index}"
    reward = row["reward"]
    status = "SUCCEEDED" if reward["success"] else "FAILED"
    inserted = connection.execute(
        """
        INSERT INTO tasks (task_id, task_type, instruction, app_image, policy_version, max_steps, priority,
                           status, current_step, idempotency_key, request_fingerprint, temperature, seed,
                           runtime_backend, run_id, round, group_id, started_at, created_at, updated_at)
        VALUES (%s, 'shopping', %s, 'android-shop-v1', %s, 1, 5, %s, 1, %s, %s, %s, %s, 'firecracker', %s, 0, %s,
                to_timestamp(%s), to_timestamp(%s), to_timestamp(%s))
        ON CONFLICT (task_id) DO NOTHING
        """,
        (
            task_id, group["instruction"], group["model_version"], status, f"import:{task_id}", "0" * 64,
            group["sampling"]["temperature"], index, HISTORY_RUN, group["group_id"],
            group["started_unix"], group["started_unix"], group["started_unix"],
        ),
    ).rowcount
    if not inserted:
        return 0
    connection.execute(
        """
        INSERT INTO episodes (episode_id, task_id, environment_version, seed, status, actual_runtime_backend, result)
        VALUES (%s, %s, 'android-firecracker-v1', %s, %s, 'firecracker', %s)
        """,
        (episode_id, task_id, index, status, Json({"final_reward": reward, "final_observation": row["after"]})),
    )
    response = row["model_response"]
    connection.execute(
        """
        INSERT INTO steps (episode_id, step_id, observation, action, reward, done, model_output)
        VALUES (%s, 0, %s, %s, %s, true, %s)
        """,
        (
            episode_id,
            Json(row["initial_observation"]),
            Json(row["action"]),
            Json(reward),
            Json({"raw_output": response["raw_output"], "model_version": response["model_version"],
                  "latency_ms": response["latency_ms"], "old_logprob": old_logprob}),
        ),
    )
    return 1


SOKOBAN_DATASET = ("sokoban-sft-v2-action-full", "file:///home/alvarez-datasets/sokoban-sft-v2/action-v1/train.jsonl", 333)
SOKOBAN_MODEL = "sokoban-v2-action-full-merger-e10"
SOKOBAN_CHECKPOINT = "file:///home/alvarez-checkpoints/sokoban-v2-action-full-merger-e10"
SOKOBAN_RUN = "run_history_sokoban_full_merger_e10"
# Exactly the TrainingConfig knobs of the 4090 CLI run, so the console can reuse it as a recipe.
SOKOBAN_CONFIG = {
    "model_name": None, "finetuning_method": "lora", "train_merger_mlp": True, "multiview": True,
    "output_format": "action-v1", "learning_rate": 1e-5, "epochs": 10, "gradient_accumulation": 4,
    "batch_size": 1, "cache_preprocessed": False, "gradient_checkpointing": True,
    "eval_trials": 0, "max_steps": 24, "seed": 20261117,
    "tasks": [{"task_type": "sokoban", "app_image": "android-sokoban-v1"}],
}
SOKOBAN_LOG = """\
2026-11-18 在 4090 上用训练 CLI 手动执行的 SFT，2026-11-19 导入 console；当前 Sokoban 候选。
命令：python -m alvarezgui.minicpm_training --algorithm sft --finetuning-method lora --train-merger-mlp --multiview \\
  --base-model /home/alvarez-models/MiniCPM-V-4.7-1B --dataset /home/alvarez-datasets/sokoban-sft-v2/action-v1/train.jsonl \\
  --output-dir /home/alvarez-checkpoints/sokoban-v2-action-full-merger-e10 \\
  --learning-rate 1e-5 --epochs 10 --gradient-accumulation 4 --seed 20261117
实现：语言 attention LoRA r16/alpha32 + 全量 model.merger.mlp.0，共 31,481,344 个可训练 FP32 参数；视觉编码器冻结；
  输入为完整截图 + 棋盘裁剪双视图；只监督 assistant 的 action-v1 tap JSON；micro-batch 1、开启梯度检查点、BF16 混合精度。
数据：train 333 条 solver-verified 真实截图，SHA-256 2f3d22481836648a783f0b9e82f7d8090cf8ce02dee4b1801e445f54604fff81；
  标签由 BFS 求解器给出并在真实 Android 中逐步核验，求解器轨迹不进入模型输入。
训练：3330 样本步 / 840 次更新（每轮 84，含尾批），loss=0.05795616，6079.04 秒。
checkpoint：adapter_model.safetensors SHA-256 604654b0f02f81dbd13cd403a7d7ffe397187618206a618e228c1d0c4ab65c83。
评测（temperature 0）：validation 39/57，历史 test 43/55，合法/可执行全部通过；真实 Docker Android 8 个 test 关卡连续通关 3/8，
  42 次点击中 27 次最优，转移一致 42/42。2026-11-18 合并推理复测 validation 为 38/57，与 39/57 的差异原因未确认。
限制：该次运行未启用 --checkpoint-dir，没有训练器生成的配方摘要；训练源码早于提交 bc10a05，未单独归档其哈希。
证据：var/runs/sokoban-binding-followup/full333-merger-*.json、android-full333-merger-r1-report.json；说明见 docs/sokoban-sft-v2.md。
"""


def import_sokoban_candidate(settings: Settings, runs_dir: Path) -> dict[str, int]:
    evidence = runs_dir / "sokoban-binding-followup"
    trained = json.loads((evidence / "full333-merger-metrics.json").read_text())
    validation = json.loads((evidence / "full333-merger-validation.json").read_text())
    test = json.loads((evidence / "full333-merger-test.json").read_text())
    metrics = {
        **{key: trained[key] for key in ("loss", "sample_count", "update_count", "step_count", "elapsed_seconds")},
        "validation_correct": validation["correct"], "validation_total": validation["total"],
        "test_correct": test["correct"], "test_total": test["total"],
        "android_solved": 3, "android_levels": 8,
    }
    dataset_version, dataset_uri, sample_count = SOKOBAN_DATASET
    counts = {}
    with connect(settings) as connection, connection.transaction():
        counts["datasets"] = connection.execute(
            """
            INSERT INTO datasets (dataset_version, source, schema_version, sample_count)
            VALUES (%s, %s, 'minicpm-sft-v1', %s) ON CONFLICT (dataset_version) DO NOTHING
            """,
            (dataset_version, dataset_uri, sample_count),
        ).rowcount
        counts["models"] = connection.execute(
            """
            INSERT INTO model_versions (version, base_model, checkpoint_uri, status, metrics)
            VALUES (%s, 'minicpm-v-4.7-1b', %s, 'READY', %s) ON CONFLICT (version) DO NOTHING
            """,
            (SOKOBAN_MODEL, SOKOBAN_CHECKPOINT, Json(metrics)),
        ).rowcount
        counts["runs"] = connection.execute(
            """
            INSERT INTO training_runs (run_id, algorithm, base_model, dataset_version, environment_version,
                                       reward_version, runtime_backend, config, status, checkpoint_uri, metrics, log_tail,
                                       idempotency_key, request_fingerprint, created_at, updated_at)
            VALUES (%s, 'sft', 'minicpm-v-4.7-1b', %s, 'android-docker-v1', 'android-task-v1', 'docker', %s,
                    'SUCCEEDED', %s, %s, %s, %s, %s,
                    '2026-11-18 15:50:00+08', '2026-11-18 17:31:00+08')
            ON CONFLICT (run_id) DO NOTHING
            """,
            (SOKOBAN_RUN, dataset_version, Json(SOKOBAN_CONFIG), SOKOBAN_CHECKPOINT,
             Json({**metrics, "model_version": SOKOBAN_MODEL}), SOKOBAN_LOG, f"import:{SOKOBAN_RUN}", "0" * 64),
        ).rowcount
    return counts


def main() -> None:
    settings = Settings()
    print(json.dumps(import_history(settings, settings.data_dir / "runs")))


if __name__ == "__main__":
    main()
