from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from flood_recovery_env import FloodRecoveryEnv
from recovery_metrics import RecoveryConfig, extract_recovery_from_trajectory

EXPECTED_ACTION_NVEC = [5, 6, 4]
EXPECTED_STATE_DIM = 80
NOOP_ACTION = np.array([0, 0, 0], dtype=np.int64)


def read_trajectory(path: Path):e
    if not path.exists():
        raise FileNotFoundError(f"Trajectory not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Trajectory is empty: {path}")
    return rows


def assert_close(actual, expected, name, atol=1e-9):
    if not math.isclose(float(actual), float(expected),
                        rel_tol=0.0, abs_tol=atol):
        raise AssertionError(
            f"{name} mismatch: actual={actual}, expected={expected}"
        )


def run_validation(output_dir, backend, seed,
                   decision_period_s, recovery_threshold,
                   stability_window_s):
    output_dir.mkdir(parents=True, exist_ok=True)

    env = FloodRecoveryEnv(
        scenario_name="moderate",
        decision_period_s=decision_period_s,
        recovery_threshold=recovery_threshold,
        stability_window_s=stability_window_s,
        sumo_seed=seed,
        gui=False,
        sumo_backend=backend,
        output_dir=output_dir,
        enable_episode_logging=True,
        policy_version="part10_step1_fixed_moderate_noop_v1",
    )

    total_reward = 0.0
    epochs = 0
    terminated = False
    truncated = False
    final_info = {}

    try:
        obs, info = env.reset()

        print("Scenario:", info["scenario_id"])
        print("Action space:", env.action_space.nvec.tolist())
        print("Observation shape:", obs.shape)
        print("Initial PR:", info["pr_t"])
        print("Initial time:", info["time_s"])

        assert env.action_space.nvec.tolist() == EXPECTED_ACTION_NVEC
        assert info["action_space_nvec"] == EXPECTED_ACTION_NVEC
        assert obs.shape == (EXPECTED_STATE_DIM,)
        assert env.observation_space.shape == (EXPECTED_STATE_DIM,)
        assert info["state_dim"] == EXPECTED_STATE_DIM
        assert np.all(np.isfinite(obs))
        assert env.action_space.contains(NOOP_ACTION)
        print("[PASS] Gym action/state schema")

        assert env.scenario.severity == "moderate"
        onset_t = float(env.scenario.onset_t)
        peak_end_t = float(env.scenario.peak_end_t)
        max_time = float(env.scenario.max_sim_time_t)
        print("Severity:", env.scenario.severity)
        print("Onset:", onset_t)
        print("Peak end:", peak_end_t)
        print("Max simulation time:", max_time)
        print("[PASS] Fixed moderate scenario loaded")

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(NOOP_ACTION)
            epochs += 1
            total_reward += float(reward)
            final_info = dict(info)

            assert obs.shape == (EXPECTED_STATE_DIM,)
            assert np.all(np.isfinite(obs))
            assert math.isfinite(float(reward))
            assert math.isfinite(float(info["pr_t"]))

            if info.get("management_changed"):
                raise AssertionError(
                    "NOOP/HOLD unexpectedly changed management state."
                )
            if info.get("active_measures"):
                raise AssertionError(
                    "NOOP/HOLD unexpectedly left active measures."
                )

            print(
                f"epoch={epochs:02d} "
                f"time={float(info['time_s']):.0f} "
                f"PR={float(info['pr_t']):.6f} "
                f"reward={float(reward):.6f} "
                f"recovered={bool(info['recovered'])}"
            )

        assert terminated != truncated
        print("[PASS] NOOP episode completed")
        print("terminated:", terminated, "truncated:", truncated)
        print("episode reward:", total_reward)

        trajectory_value = final_info.get("second_trajectory_path")
        if not trajectory_value:
            raise AssertionError(
                "No second_trajectory_path returned at episode end."
            )
        trajectory_path = Path(trajectory_value)
        rows = read_trajectory(trajectory_path)

        required = {"time_s", "pr_t", "recovered"}
        missing = required - set(rows[0])
        if missing:
            raise AssertionError(
                f"Trajectory missing columns: {sorted(missing)}"
            )

        pr_values = [float(row["pr_t"]) for row in rows]
        assert all(math.isfinite(x) for x in pr_values)
        print("[PASS] PR trajectory generated:", trajectory_path)
        print("Trajectory samples:", len(rows))

        config = RecoveryConfig(
            threshold=recovery_threshold,
            stability_window_s=stability_window_s,
            search_start_t=peak_end_t,
        )
        offline = extract_recovery_from_trajectory(
            rows,
            time_key="time_s",
            pr_key="pr_t",
            config=config,
            flood_onset_t=onset_t,
        )

        online_recovered = bool(final_info["recovered"])
        assert online_recovered == bool(offline.recovered)

        if online_recovered:
            assert final_info["recovery_time_s"] is not None
            assert offline.recovery_time_s is not None
            assert_close(
                final_info["recovery_time_s"],
                offline.recovery_time_s,
                "recovery_time_s",
            )
            assert terminated and not truncated
        else:
            assert truncated and not terminated
            assert float(final_info["time_s"]) >= max_time

        print("[PASS] Online/offline recovery logic matches")
        print(
            "Recovery:",
            final_info.get("recovery_time_s"),
            offline.recovery_time_s,
        )

        expected_reward = (
            float(final_info["reward_pr_component"])
            - float(final_info["action_switch_cost"])
            - float(final_info["queue_growth_penalty"])
            + float(final_info["terminal_reward"])
        )
        assert_close(
            final_info["reward_total_step"],
            expected_reward,
            "final step reward",
        )
        assert_close(
            final_info["action_switch_cost"],
            0.0,
            "NOOP action_switch_cost",
        )
        print("[PASS] Reward component consistency")

        print("\n" + "=" * 60)
        print("PART 10.1 VALIDATION PASSED")
        print("=" * 60)
        print("Output:", output_dir)

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"D:\SUMO_A9_Project\validation_results\part10_step1"
        ),
    )
    parser.add_argument(
        "--backend", choices=("libsumo", "traci"), default="libsumo"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision-period", type=int, default=300)
    parser.add_argument("--recovery-threshold", type=float, default=0.90)
    parser.add_argument("--stability-window", type=int, default=600)
    args = parser.parse_args()

    run_validation(
        args.output_dir,
        args.backend,
        args.seed,
        args.decision_period,
        args.recovery_threshold,
        args.stability_window,
    )


if __name__ == "__main__":
    main()
