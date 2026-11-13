from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alvarezgui.config import Settings
from alvarezgui.db import connect
from alvarezgui.registry import RegistryStore
from alvarezgui.minicpm_training import load_dataset, train
from alvarezgui.training import TrainingRound


RolloutProvider = Callable[[dict[str, Any]], list[dict[str, Any]]]


class TrainingWorker:
    """Claim persisted runs and execute an injected SFT/GRPO rollout backend."""

    def __init__(
        self,
        settings: Settings | None = None,
        rollout_provider: RolloutProvider | None = None,
        training_round: TrainingRound | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.rollout_provider = rollout_provider
        self.training_round = training_round or TrainingRound()

    def run_once(self) -> dict[str, Any] | None:
        with connect(self.settings) as connection:
            run = RegistryStore(connection).claim_next_training_run()
        if run is None:
            return None
        try:
            if self.rollout_provider is None:
                config = run.get("config") or {}
                base_model = config.get("base_model_path") or run.get("base_model_checkpoint_uri")
                dataset_path = config.get("dataset_path") or run.get("dataset_source")
                output_dir = config.get("output_dir")
                if not base_model or not dataset_path or not output_dir:
                    raise RuntimeError(
                        "real training requires config.base_model_path, config.dataset_path and config.output_dir"
                    )
                metrics = train(
                    run["algorithm"],
                    str(base_model),
                    load_dataset(str(dataset_path)),
                    str(output_dir),
                    learning_rate=float(config.get("learning_rate", 1e-5)),
                    epochs=int(config.get("epochs", 1)),
                    gradient_accumulation=int(config.get("gradient_accumulation", 1)),
                    group_size=int(config.get("group_size", 2)),
                )
                checkpoint_dir = Path(str(output_dir))
                if not (checkpoint_dir / "config.json").is_file():
                    raise RuntimeError("real training did not produce a reloadable checkpoint")
                succeeded = True
                checkpoint_uri = checkpoint_dir.as_uri()
            else:
                result = self.training_round.run(run["algorithm"], self.rollout_provider(run))
                succeeded = result.accepted
                metrics = dict(result.metrics)
                if not succeeded:
                    metrics["checkpoint_rejected"] = 1.0
                checkpoint_uri = result.checkpoint_uri if succeeded else None
        except Exception as exc:
            succeeded = False
            checkpoint_uri = None
            metrics = {"error": str(exc)[:500]}
        with connect(self.settings) as connection:
            RegistryStore(connection).finish_training_run(
                run["run_id"],
                succeeded=succeeded,
                metrics=metrics,
                checkpoint_uri=checkpoint_uri,
            )
        return {"run_id": run["run_id"], "status": "SUCCEEDED" if succeeded else "FAILED", "metrics": metrics}

    def run_forever(self, interval: float = 1.0) -> None:
        while True:
            self.run_once()
            time.sleep(interval)
