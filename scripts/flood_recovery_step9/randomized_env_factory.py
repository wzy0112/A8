
from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym

from randomized_flood_recovery_env import (
    RandomizedFloodRecoveryEnv,
)


def make_randomized_env(
    *,
    rank: int,
    base_seed: int,
    sampling_mode: str,
    decision_period_s: int,
    output_root: Path,
    enable_episode_logging: bool,
    severity_weights: tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    ),
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        worker_seed = base_seed + rank
        return RandomizedFloodRecoveryEnv(
            sampling_mode=sampling_mode,
            scenario_seed=100_000 + worker_seed,
            severity_weights=severity_weights,
            decision_period_s=decision_period_s,
            sumo_seed=worker_seed,
            gui=False,
            sumo_backend="libsumo",
            output_dir=(
                output_root / f"worker_{rank:02d}"
            ),
            enable_episode_logging=enable_episode_logging,
            policy_version=(
                f"ppo_step9_curriculum_{sampling_mode}_v1"
            ),
        )

    return _init
