from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)

from randomized_flood_recovery_env import (
    RandomizedFloodRecoveryEnv,
)

from scripted_baseline import (
    get_scripted_action,
)


def make_env(
    *,
    mode: str,
    seed: int,
    output_dir: Path,
):
    def _init():
        return RandomizedFloodRecoveryEnv(
            sampling_mode=mode,
            scenario_seed=seed + 100_000,
            sumo_seed=seed,
            output_dir=output_dir,
            enable_episode_logging=True,
            policy_version="ppo_step9_evaluation_v2",
        )

    return _init


def run_episode(
    model,
    env,
    policy: str,
) -> dict:

    obs = env.reset()

    done = np.array(
        [False]
    )

    info = {}

    decision_epoch = 0

    while not bool(done[0]):

        decision_epoch += 1

        if policy == "ppo":

            if model is None:
                raise RuntimeError(
                    "PPO policy requires a loaded model."
                )

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

        elif policy == "noop":

            action = np.array(
                [[0, 0, 0]],
                dtype=np.int64,
            )

        elif policy == "scripted":

            scripted_action = (
                get_scripted_action(
                    decision_epoch=(
                        decision_epoch
                    ),
                    scenario_severity=(
                        info.get(
                            "sampled_severity"
                        )
                    ),
                )
            )

            action = np.asarray(
                [scripted_action],
                dtype=np.int64,
            )

        else:

            raise ValueError(
                "Unknown evaluation policy: "
                f"{policy}"
            )

        obs, _, done, infos = env.step(
            action
        )

        info = infos[0]

    return {
        "policy": policy,
        "scenario_id": info.get(
            "scenario_id"
        ),
        "sampling_mode": info.get(
            "sampling_mode"
        ),
        "severity": info.get(
            "sampled_severity"
        ),
        "site_id": info.get(
            "sampled_site_id"
        ),
        "mechanism": info.get(
            "sampled_mechanism"
        ),
        "demand_phase": info.get(
            "sampled_demand_phase"
        ),
        "onset_t": info.get(
            "sampled_onset_t"
        ),
        "peak_start_t": info.get(
            "sampled_peak_start_t"
        ),
        "peak_end_t": info.get(
            "sampled_peak_end_t"
        ),
        "recovery_end_t": info.get(
            "sampled_recovery_end_t"
        ),
        "peak_speed_mps": info.get(
            "sampled_peak_speed_mps"
        ),
        "closed_lane_indices": info.get(
            "sampled_closed_lane_indices"
        ),
        "max_sim_time_t": info.get(
            "sampled_max_sim_time_t"
        ),
        "actual_sumo_seed": info.get(
            "sampled_sumo_seed"
        ),
        "recovered": info.get(
            "recovered"
        ),
        "recovery_time_s": info.get(
            "recovery_time_s"
        ),
        "co2_total_kg": info.get(
            "co2_total_kg"
        ),
        "hard_braking_events_total": (
            info.get(
                "hard_braking_events_total"
            )
        ),
        "queue_exposure_veh_h": (
            info.get(
                "queue_exposure_veh_h"
            )
        ),
        "total_time_loss_veh_h": (
            info.get(
                "total_time_loss_veh_h"
            )
        ),
        "summary_path": info.get(
            "summary_path"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=None,
        help=(
            "Optional explicit VecNormalize file. "
            "Defaults to <run-dir>/vec_normalize_final.pkl."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"D:\SUMO_A9_Project\validation_results"
            r"\step9_generalization"
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        args.run_dir / "ppo_curriculum_final"
    )
    normalization_path = (
        args.normalization
        if args.normalization is not None
        else args.run_dir / "vec_normalize_final.pkl"
    )

    if not normalization_path.exists():
        raise FileNotFoundError(
            f"VecNormalize file not found: {normalization_path}"
        )

    results = []

    for episode in range(args.episodes):
        episode_seed = args.seed + episode

        for policy in ("noop", "scripted", "ppo"):
            raw_env = DummyVecEnv([
                make_env(
                    mode="randomized",
                    seed=episode_seed,
                    output_dir=(
                        args.output_dir
                        / f"episode_{episode:03d}"
                        / policy
                    ),
                )
            ])
            env = VecNormalize.load(
                str(normalization_path),
                raw_env,
            )
            env.training = False
            env.norm_reward = False

            model = None
            if policy == "ppo":
                model = PPO.load(
                    model_path,
                    env=env,
                )

            try:
                result = run_episode(
                    model=model,
                    env=env,
                    policy=policy,
                )
                result["evaluation_episode"] = episode
                result["evaluation_seed"] = episode_seed
                results.append(result)
            finally:
                env.close()

    if not results:
        raise RuntimeError(
            "No evaluation episodes were completed."
        )

    csv_path = (
        args.output_dir
        / "generalization_episodes.csv"
    )
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0]),
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = (
        args.output_dir
        / "generalization_episodes.json"
    )
    json_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("Evaluation complete.")
    print("Results:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
