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
                                       reward_version, config, status, checkpoint_uri, metrics, log_tail,
                                       idempotency_key, request_fingerprint, created_at, updated_at)
            VALUES (%s, 'grpo', 'minicpm-v-4.7-1b', NULL, 'android-firecracker-v1', 'android-task-v1', %s,
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
                           run_id, round, group_id, started_at, created_at, updated_at)
        VALUES (%s, 'shopping', %s, 'android-shop-v1', %s, 1, 5, %s, 1, %s, %s, %s, %s, %s, 0, %s,
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
        INSERT INTO episodes (episode_id, task_id, environment_version, seed, status, result)
        VALUES (%s, %s, 'android-firecracker-v1', %s, %s, %s)
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


def main() -> None:
    settings = Settings()
    print(json.dumps(import_history(settings, settings.data_dir / "runs")))


if __name__ == "__main__":
    main()
