from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from randomized_flood_recovery_env import RandomizedFloodRecoveryEnv


POLICIES = ("noop", "ppo")
SEVERITY_WEIGHTS = {
    "light": (1.0, 0.0, 0.0),
    "moderate": (0.0, 1.0, 0.0),
    "severe": (0.0, 0.0, 1.0),
}
DEFAULT_BASE_SEEDS = {
    "light": 11000,
    "moderate": 10000,
    "severe": 12000,
}


def make_env(
    *,
    severity: str,
    evaluation_seed: int,
    output_dir: Path,
    decision_period_s: int,
):
    weights = SEVERITY_WEIGHTS[severity]

    def _init():
        return RandomizedFloodRecoveryEnv(
            sampling_mode="mixed_fixed",
            scenario_seed=evaluation_seed + 100_000,
            severity_weights=weights,
            decision_period_s=decision_period_s,
            sumo_seed=evaluation_seed,
            gui=False,
            sumo_backend="libsumo",
            output_dir=output_dir,
            enable_episode_logging=True,
            policy_version=(
                f"part10_step3_zero_shot_{severity}_v1"
            ),
        )

    return _init


def noop_action(env) -> np.ndarray:
    # HOLD is tactic index 0 in the current ActionCodec. Location/intensity are
    # ignored by HOLD, so zeros are sufficient and remain inside MultiDiscrete.
    return np.zeros(len(env.action_space.nvec), dtype=np.int64)


def _unwrap_terminal_info(info: dict[str, Any]) -> dict[str, Any]:
    terminal_info = info.get("terminal_info")
    if isinstance(terminal_info, dict):
        merged = dict(info)
        merged.update(terminal_info)
        return merged
    return info


def run_episode(
    *,
    env,
    policy: str,
    model: PPO | None,
    severity: str,
    evaluation_seed: int,
) -> dict[str, Any]:
    obs = env.reset()
    done = np.array([False])
    reward_total = 0.0
    last_info: dict[str, Any] = {}
    n_steps = 0

    while not bool(done[0]):
        if policy == "noop":
            action = noop_action(env)
            action = np.asarray([action], dtype=np.int64)
        elif policy == "ppo":
            if model is None:
                raise RuntimeError("PPO model was not loaded.")
            action, _ = model.predict(obs, deterministic=True)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, rewards, done, infos = env.step(action)
        reward_total += float(rewards[0])
        last_info = _unwrap_terminal_info(dict(infos[0]))
        n_steps += 1

    recovered = bool(last_info.get("recovered", False))
    recovery_time = last_info.get("recovery_time_s")

    result = {
        "severity": severity,
        "policy": policy,
        "evaluation_seed": int(evaluation_seed),
        "actual_sumo_seed": last_info.get("sampled_sumo_seed"),
        "scenario_id": last_info.get("scenario_id"),
        "sampled_severity": last_info.get("sampled_severity"),
        "sampled_site_id": last_info.get("sampled_site_id"),
        "sampled_mechanism": last_info.get("sampled_mechanism"),
        "sampled_demand_phase": last_info.get("sampled_demand_phase"),
        "sampled_onset_t": last_info.get("sampled_onset_t"),
        "sampled_peak_start_t": last_info.get("sampled_peak_start_t"),
        "sampled_peak_end_t": last_info.get("sampled_peak_end_t"),
        "sampled_recovery_end_t": last_info.get("sampled_recovery_end_t"),
        "final_time_s": last_info.get("time_s"),
        "recovered": recovered,
        "recovery_time_s": recovery_time,
        "reward_total": reward_total,
        "co2_total_kg": last_info.get("co2_total_kg"),
        "hard_braking_events_total": last_info.get(
            "hard_braking_events_total"
        ),
        "queue_exposure_veh_h": last_info.get("queue_exposure_veh_h"),
        "total_time_loss_veh_h": last_info.get("total_time_loss_veh_h"),
        "decision_steps": n_steps,
        "episode_summary_path": last_info.get("episode_summary_path"),
        "epoch_log_path": last_info.get("epoch_log_path"),
        "second_trajectory_path": last_info.get("second_trajectory_path"),
        "state_log_path": last_info.get("state_log_path"),
    }
    return result


def assert_pair_is_comparable(
    pair_results: list[dict[str, Any]],
    *,
    expected_severity: str,
    evaluation_seed: int,
) -> None:
    if len(pair_results) != len(POLICIES):
        raise RuntimeError(
            f"Seed {evaluation_seed}: expected {len(POLICIES)} policies, "
            f"got {len(pair_results)}."
        )

    policies = {row["policy"] for row in pair_results}
    if policies != set(POLICIES):
        raise RuntimeError(
            f"Seed {evaluation_seed}: policy mismatch: {policies}"
        )

    severities = {row["sampled_severity"] for row in pair_results}
    if severities != {expected_severity}:
        raise RuntimeError(
            f"Seed {evaluation_seed}: severity mismatch: {severities}; "
            f"expected {expected_severity!r}."
        )

    scenario_ids = {row["scenario_id"] for row in pair_results}
    if len(scenario_ids) != 1:
        raise RuntimeError(
            f"Seed {evaluation_seed}: scenario mismatch: {scenario_ids}"
        )

    actual_sumo_seeds = {
        int(row["actual_sumo_seed"])
        for row in pair_results
        if row["actual_sumo_seed"] is not None
    }
    if len(actual_sumo_seeds) != 1:
        raise RuntimeError(
            f"Seed {evaluation_seed}: actual SUMO seed mismatch: "
            f"{actual_sumo_seeds}"
        )

    onset_times = {row["sampled_onset_t"] for row in pair_results}
    if len(onset_times) != 1:
        raise RuntimeError(
            f"Seed {evaluation_seed}: onset mismatch: {onset_times}"
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def mean_or_none(values: list[Any]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for severity in SEVERITY_WEIGHTS:
        for policy in POLICIES:
            subset = [
                row for row in rows
                if row["severity"] == severity and row["policy"] == policy
            ]
            if not subset:
                continue
            recovered_rows = [row for row in subset if row["recovered"]]
            summary.append({
                "severity": severity,
                "policy": policy,
                "episodes": len(subset),
                "recovered_episodes": len(recovered_rows),
                "recovery_rate": len(recovered_rows) / len(subset),
                "mean_recovery_time_s_recovered_only": mean_or_none(
                    [row["recovery_time_s"] for row in recovered_rows]
                ),
                "mean_total_time_loss_veh_h": mean_or_none(
                    [row["total_time_loss_veh_h"] for row in subset]
                ),
                "mean_queue_exposure_veh_h": mean_or_none(
                    [row["queue_exposure_veh_h"] for row in subset]
                ),
                "mean_co2_total_kg": mean_or_none(
                    [row["co2_total_kg"] for row in subset]
                ),
                "mean_hard_braking_events_total": mean_or_none(
                    [row["hard_braking_events_total"] for row in subset]
                ),
                "mean_reward_total": mean_or_none(
                    [row["reward_total"] for row in subset]
                ),
            })
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Part 10.3: frozen Moderate-trained PPO zero-shot evaluation "
            "on validated Light / Moderate / Severe fixed templates."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing final PPO and VecNormalize.",
    )
    parser.add_argument(
        "--severity",
        choices=("light", "moderate", "severe", "all"),
        default="all",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional base evaluation seed. For --severity all, normally omit "
            "this so Light/Moderate/Severe use 11000/10000/12000."
        ),
    )
    parser.add_argument("--decision-period", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive.")

    model_path = args.run_dir / "ppo_curriculum_final.zip"
    norm_path = args.run_dir / "vec_normalize_final.pkl"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not norm_path.exists():
        raise FileNotFoundError(norm_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    severities = (
        tuple(SEVERITY_WEIGHTS.keys())
        if args.severity == "all"
        else (args.severity,)
    )

    # Load policy without binding it to an evaluation environment. This keeps
    # training seed/model metadata separate from evaluation SUMO/scenario RNGs.
    model = PPO.load(str(model_path))

    all_results: list[dict[str, Any]] = []

    for severity_index, severity in enumerate(severities):
        if args.seed is None:
            base_seed = DEFAULT_BASE_SEEDS[severity]
        elif args.severity == "all":
            base_seed = args.seed + severity_index * 1000
        else:
            base_seed = args.seed

        print(
            f"\n=== Zero-shot severity: {severity.upper()} | "
            f"base evaluation seed={base_seed} ==="
        )

        for pair_index in range(args.n_seeds):
            evaluation_seed = base_seed + pair_index
            pair_results: list[dict[str, Any]] = []

            print(
                f"\n--- Pair {pair_index + 1}/{args.n_seeds} | "
                f"severity={severity} | evaluation_seed={evaluation_seed} ---"
            )

            for policy in POLICIES:
                print(f"Running {policy} ...")
                policy_dir = (
                    args.output_dir
                    / severity
                    / f"seed_{evaluation_seed}"
                    / policy
                )

                raw_env = DummyVecEnv([
                    make_env(
                        severity=severity,
                        evaluation_seed=evaluation_seed,
                        output_dir=policy_dir,
                        decision_period_s=args.decision_period,
                    )
                ])

                env = VecNormalize.load(str(norm_path), raw_env)
                env.training = False
                env.norm_reward = False

                try:
                    result = run_episode(
                        env=env,
                        policy=policy,
                        model=model if policy == "ppo" else None,
                        severity=severity,
                        evaluation_seed=evaluation_seed,
                    )
                finally:
                    env.close()

                pair_results.append(result)
                all_results.append(result)

                print(
                    "  "
                    f"actual_sumo_seed={result['actual_sumo_seed']} "
                    f"scenario={result['scenario_id']} "
                    f"recovered={result['recovered']} "
                    f"recovery_time_s={result['recovery_time_s']} "
                    f"reward_total={result['reward_total']}"
                )

            assert_pair_is_comparable(
                pair_results,
                expected_severity=severity,
                evaluation_seed=evaluation_seed,
            )
            print("Pair comparability: PASS")

            # Incremental checkpoint so completed pairs survive interruption.
            write_csv(
                args.output_dir / "zero_shot_episode_results.csv",
                all_results,
            )
            write_json(
                args.output_dir / "zero_shot_episode_results.json",
                all_results,
            )

    summary = build_summary(all_results)
    write_csv(args.output_dir / "zero_shot_policy_summary.csv", summary)

    manifest = {
        "experiment": "part10_step3_zero_shot_severity",
        "training_policy": "frozen Moderate-5k PPO",
        "model_path": str(model_path),
        "vecnormalize_path": str(norm_path),
        "ppo_deterministic": True,
        "vecnormalize_training": False,
        "vecnormalize_norm_reward": False,
        "sampling_mode": "mixed_fixed",
        "severity_weights": {
            key: list(value) for key, value in SEVERITY_WEIGHTS.items()
        },
        "severities_run": list(severities),
        "n_paired_seeds_per_severity": args.n_seeds,
        "policies": list(POLICIES),
        "decision_period_s": args.decision_period,
        "default_base_seeds": DEFAULT_BASE_SEEDS,
        "note": (
            "evaluation_seed is the reproducibility/pairing seed. "
            "RandomizedFloodRecoveryEnv deterministically draws an actual "
            "SUMO seed from its independent SUMO RNG; paired NOOP/PPO must "
            "therefore share actual_sumo_seed, but it need not equal "
            "evaluation_seed."
        ),
    }
    write_json(args.output_dir / "evaluation_manifest.json", manifest)

    print("\nEvaluation complete.")
    print(
        "Episode results: "
        f"{args.output_dir / 'zero_shot_episode_results.csv'}"
    )
    print(
        "Policy summary: "
        f"{args.output_dir / 'zero_shot_policy_summary.csv'}"
    )
    print(
        "Manifest: "
        f"{args.output_dir / 'evaluation_manifest.json'}"
    )


if __name__ == "__main__":
    main()
