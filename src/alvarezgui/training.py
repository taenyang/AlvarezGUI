from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from alvarezgui.action import validate_action


class TrainingError(Exception):
    pass


class InvalidTrainingInputError(TrainingError):
    pass


@dataclass(frozen=True, slots=True)
class Advantage:
    group_id: str
    reward: float
    mean: float
    standard_deviation: float
    value: float


class GroupAdvantageEstimator:
    """Compute per-group normalized rewards for policy optimization rounds."""

    def estimate(self, samples: Iterable[tuple[str, float]]) -> list[Advantage]:
        grouped: dict[str, list[float]] = defaultdict(list)
        rows = list(samples)
        for group_id, reward in rows:
            if not group_id:
                raise InvalidTrainingInputError("group_id must not be blank")
            grouped[group_id].append(float(reward))
        result: list[Advantage] = []
        for group_id, rewards in grouped.items():
            mean = sum(rewards) / len(rewards)
            variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
            standard_deviation = variance**0.5
            scale = standard_deviation or 1.0
            result.extend(
                Advantage(group_id, reward, mean, standard_deviation, (reward - mean) / scale)
                for reward in rewards
            )
        return result


class RewardEvaluator:
    def evaluate(self, reward: dict[str, Any]) -> float:
        try:
            total = float(reward["total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidTrainingInputError("reward.total must be numeric") from exc
        return total


class RolloutCollector:
    """Small boundary for a future Android rollout worker."""

    def collect(self, episodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(episodes)


class PolicyUpdater:
    """Training boundary; GPU-specific implementation is injected later."""

    def update(self, rollout: list[dict[str, Any]], advantages: list[Advantage]) -> dict[str, Any]:
        return {"sample_count": len(rollout), "advantage_count": len(advantages)}


class CheckpointEvaluator:
    def evaluate(self, metrics: dict[str, float]) -> bool:
        return metrics.get("action_format_rate", 0.0) >= 1.0 and metrics.get("success_rate", 0.0) > 0.0


@dataclass(frozen=True, slots=True)
class TrainingRoundResult:
    checkpoint_uri: str | None
    metrics: dict[str, float]
    accepted: bool


class TrainingRound:
    """Run the CPU-side, deterministic part of one SFT or policy-algorithm round."""

    def __init__(self) -> None:
        self.collector = RolloutCollector()
        self.rewards = RewardEvaluator()
        self.advantages = GroupAdvantageEstimator()
        self.updater = PolicyUpdater()
        self.checkpoints = CheckpointEvaluator()

    def run(self, algorithm: str, episodes: Iterable[dict[str, Any]]) -> TrainingRoundResult:
        if algorithm not in {"sft", "grpo", "dapo", "dr_grpo", "gigpo"}:
            raise InvalidTrainingInputError(f"unsupported algorithm: {algorithm}")
        rollout = self.collector.collect(episodes)
        if not rollout:
            raise InvalidTrainingInputError("at least one episode is required")
        rewards: list[tuple[str, float]] = []
        action_count = 0
        for episode in rollout:
            for step in episode.get("steps", []):
                validate_action(step.get("action"))
                action_count += 1
            rewards.append((str(episode.get("group_id", episode.get("episode_id", ""))), self.rewards.evaluate(episode.get("reward", {}))))
        advantages = self.advantages.estimate(rewards) if algorithm != "sft" else []
        self.updater.update(rollout, advantages)
        metrics = {
            "action_format_rate": 1.0 if action_count else 0.0,
            "success_rate": sum(reward > 0 for _, reward in rewards) / len(rewards),
            "sample_count": float(len(rollout)),
        }
        return TrainingRoundResult(
            checkpoint_uri=None,
            metrics=metrics,
            accepted=self.checkpoints.evaluate(metrics),
        )
