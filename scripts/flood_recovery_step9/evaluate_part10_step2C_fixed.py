from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from flood_recovery_env import FloodRecoveryEnv
from scripted_baseline import SCRIPTED_POLICY_NAME, get_scripted_action


POLICIES = ("noop", "scripted", "ppo")


def resolve_model_path(run_dir: Path) -> Path:
    candidates = (
        run_dir / "ppo_curriculum_final.zip",
        run_dir / "ppo_curriculum_final",
        run_dir / "stage_1_fixed_moderate" / "ppo_stage_model.zip",
        run_dir / "stage_1_fixed_moderate" / "ppo_stage_model",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find PPO model. Checked:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def resolve_norm_path(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit
    else:
        candidates = (
            run_dir / "vec_normalize_final.pkl",
            run_dir / "stage_1_fixed_moderate" / "vec_normalize.pkl",
        )
        path = next((p for p in candidates if p.exists()), candidates[0])

    if not path.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {path}")
    return path


def make_fixed_moderate_env(
    *,
    seed: int,
    output_dir: Path,
    policy: str,
    decision_period_s: int,
):
    def _init():
        return FloodRecoveryEnv(
            scenario_name="moderate",
            decision_period_s=decision_period_s,
            sumo_seed=seed,
            gui=False,
            sumo_backend="libsumo",
            output_dir=output_dir,
            enable_episode_logging=True,
            policy_version=(
                f"part10_step2C_fixed_moderate_{policy}_v1"
            ),
        )
    return _init


def load_summary(info: dict) -> tuple[dict, str]:
    summary_ref = info.get("summary_path")
    if not summary_ref:
        return {}, ""

    summary_path = Path(summary_ref)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Episode summary reported by environment does not exist: "
            f"{summary_path}"
        )

    return (
        json.loads(summary_path.read_text(encoding="utf-8")),
        str(summary_path),
    )


def compact_tactic_chain(summary: dict) -> list[dict]:
    """
    Keep only commands that actually changed persistent management state.
    HOLD and repeated no-change commands remain available in the full
    episode summary, but are omitted from this compact comparison field.
    """
    compact = []
    for row in summary.get("M", []):
        if not bool(row.get("management_changed", False)):
            continue
        compact.append({
            "t": row.get("t"),
            "tactic": row.get("tactic"),
            "location": row.get("location"),
            "intensity": row.get("intensity_name"),
            "operation": row.get("operation"),
        })
    return compact


def run_episode(
    *,
    model: PPO | None,
    env: VecNormalize,
    policy: str,
) -> dict:
    obs = env.reset()
    done = np.array([False])
    info: dict = {}
    decision_epoch = 0

    while not bool(done[0]):
        decision_epoch += 1

        if policy == "ppo":
            if model is None:
                raise RuntimeError("PPO policy requires a loaded model.")
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
            scripted_action = get_scripted_action(
                decision_epoch=decision_epoch,
                scenario_severity="moderate",
            )
            action = np.asarray(
                [scripted_action],
                dtype=np.int64,
            )

        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, _, done, infos = env.step(action)
        info = infos[0]

    summary, summary_path = load_summary(info)
    outcomes = summary.get("outcomes", {})
    scenario = summary.get("D", {})

    compact_chain = compact_tactic_chain(summary)

    return {
        "policy": policy,
        "policy_label": (
            SCRIPTED_POLICY_NAME
            if policy == "scripted"
            else policy
        ),
        "scenario_id": info.get(
            "scenario_id",
            scenario.get("scenario_id"),
        ),
        "severity": scenario.get("severity", "moderate"),
        "logged_sumo_seed": summary.get("sumo_seed"),
        "decision_epochs": decision_epoch,
        "recovered": bool(
            outcomes.get(
                "recovered",
                info.get("recovered", False),
            )
        ),
        "truncated": bool(
            outcomes.get("truncated", False)
        ),
        "recovery_time_s": outcomes.get(
            "recovery_time_s",
            info.get("recovery_time_s"),
        ),
        "final_time_s": outcomes.get("final_time_s"),
        "reward_total": outcomes.get("reward_total"),
        "co2_total_kg": outcomes.get(
            "co2_total_kg",
            info.get("co2_total_kg"),
        ),
        "hard_braking_events_total": outcomes.get(
            "hard_braking_events_total",
            info.get("hard_braking_events_total"),
        ),
        "queue_exposure_veh_h": outcomes.get(
            "queue_exposure_veh_h",
            info.get("queue_exposure_veh_h"),
        ),
        "total_time_loss_veh_h": outcomes.get(
            "total_time_loss_veh_h",
            info.get("total_time_loss_veh_h"),
        ),
        "tactic_chain_compact_json": json.dumps(
            compact_chain,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "final_management_state_json": json.dumps(
            summary.get("final_management_state", []),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "summary_path": summary_path,
    }


def assert_pair_is_comparable(
    seed_results: list[dict],
    expected_seed: int,
) -> None:
    if len(seed_results) != len(POLICIES):
        raise RuntimeError(
            f"Seed {expected_seed}: expected {len(POLICIES)} policies, "
            f"got {len(seed_results)}."
        )

    scenario_ids = {
        row["scenario_id"]
        for row in seed_results
    }
    if len(scenario_ids) != 1:
        raise RuntimeError(
            f"Seed {expected_seed}: policies did not use the same "
            f"scenario_id: {scenario_ids}"
        )

    evaluation_seeds = {
        int(row["evaluation_seed"])
        for row in seed_results
    }
    if evaluation_seeds != {expected_seed}:
        raise RuntimeError(
            f"Seed {expected_seed}: evaluation seed mismatch: "
            f"{evaluation_seeds}"
        )

    logged_seeds = {
        int(row["logged_sumo_seed"])
        for row in seed_results
        if row["logged_sumo_seed"] is not None
    }
    if logged_seeds != {expected_seed}:
        raise RuntimeError(
            f"Seed {expected_seed}: actual logged SUMO seed mismatch: "
            f"{logged_seeds}"
        )


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("No evaluation rows were produced.")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: list[float]) -> float | None:
    clean = [
        float(v)
        for v in values
        if v is not None and np.isfinite(float(v))
    ]
    return float(np.mean(clean)) if clean else None


def build_policy_summary(rows: list[dict]) -> list[dict]:
    output = []
    for policy in POLICIES:
        subset = [r for r in rows if r["policy"] == policy]
        recovered = [r for r in subset if r["recovered"]]

        output.append({
            "policy": policy,
            "n_episodes": len(subset),
            "recovered_episodes": len(recovered),
            "recovery_rate": (
                len(recovered) / len(subset)
                if subset else None
            ),
            # Recovery time is summarized only over recovered episodes.
            "mean_recovery_time_s_recovered_only": safe_mean([
                r["recovery_time_s"] for r in recovered
            ]),
            "mean_total_time_loss_veh_h": safe_mean([
                r["total_time_loss_veh_h"] for r in subset
            ]),
            "mean_queue_exposure_veh_h": safe_mean([
                r["queue_exposure_veh_h"] for r in subset
            ]),
            "mean_co2_total_kg": safe_mean([
                r["co2_total_kg"] for r in subset
            ]),
            "mean_hard_braking_events_total": safe_mean([
                r["hard_braking_events_total"] for r in subset
            ]),
            "mean_reward_total": safe_mean([
                r["reward_total"] for r in subset
            ]),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Part 10.2-C: paired fixed-moderate evaluation of "
            "NOOP vs scripted vs PPO."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Training run directory containing "
            "ppo_curriculum_final.zip and vec_normalize_final.pkl."
        ),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=None,
        help="Optional explicit VecNormalize .pkl path.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
        help="First paired SUMO seed.",
    )
    parser.add_argument(
        "--decision-period",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"D:\SUMO_A9_Project\validation_results"
            r"\part10_step2C_fixed_evaluation"
        ),
    )
    args = parser.parse_args()

    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be >= 1.")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = resolve_model_path(args.run_dir)
    norm_path = resolve_norm_path(
        args.run_dir,
        args.normalization,
    )

    print("Part 10.2-C fixed-moderate paired evaluation")
    print("Model:", model_path)
    print("VecNormalize:", norm_path)
    print("Policies:", ", ".join(POLICIES))
    print(
        "Seeds:",
        ", ".join(
            str(args.seed + i)
            for i in range(args.n_seeds)
        ),
    )

    all_results: list[dict] = []

    for pair_index in range(args.n_seeds):
        episode_seed = args.seed + pair_index
        seed_results: list[dict] = []

        print(
            f"\n=== Pair {pair_index + 1}/{args.n_seeds} "
            f"| SUMO seed={episode_seed} ==="
        )

        for policy in POLICIES:
            print(f"Running {policy} ...", flush=True)

            policy_dir = (
                args.output_dir
                / f"seed_{episode_seed}"
                / policy
            )

            raw_env = DummyVecEnv([
                make_fixed_moderate_env(
                    seed=episode_seed,
                    output_dir=policy_dir,
                    policy=policy,
                    decision_period_s=args.decision_period,
                )
            ])

            env = VecNormalize.load(
                str(norm_path),
                raw_env,
            )
            # Critical for evaluation: freeze observation statistics
            # and do not normalize rewards.
            env.training = False
            env.norm_reward = False

            model = None
            if policy == "ppo":
                model = PPO.load(
                    str(model_path),
                )

            try:
                result = run_episode(
                    model=model,
                    env=env,
                    policy=policy,
                )
            finally:
                env.close()

            result["pair_index"] = pair_index
            result["evaluation_seed"] = episode_seed

            seed_results.append(result)
            all_results.append(result)

            print(
                f"  recovered={result['recovered']} "
                f"recovery_time_s={result['recovery_time_s']} "
                f"reward_total={result['reward_total']}"
            )

        assert_pair_is_comparable(
            seed_results,
            episode_seed,
        )

    episode_csv = (
        args.output_dir
        / "fixed_moderate_paired_episodes.csv"
    )
    episode_json = (
        args.output_dir
        / "fixed_moderate_paired_episodes.json"
    )
    summary_csv = (
        args.output_dir
        / "fixed_moderate_policy_summary.csv"
    )
    manifest_json = (
        args.output_dir
        / "evaluation_manifest.json"
    )

    write_rows(episode_csv, all_results)

    episode_json.write_text(
        json.dumps(
            all_results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    policy_summary = build_policy_summary(all_results)
    write_rows(summary_csv, policy_summary)

    manifest = {
        "evaluation": "part10_step2C_fixed_moderate_paired",
        "scenario": "moderate",
        "policies": list(POLICIES),
        "scripted_policy_name": SCRIPTED_POLICY_NAME,
        "n_paired_seeds": args.n_seeds,
        "seeds": [
            args.seed + i
            for i in range(args.n_seeds)
        ],
        "decision_period_s": args.decision_period,
        "ppo_deterministic": True,
        "vecnormalize_training": False,
        "vecnormalize_norm_reward": False,
        "model_path": str(model_path),
        "normalization_path": str(norm_path),
        "episode_results": str(episode_csv),
        "policy_summary": str(summary_csv),
    }
    manifest_json.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nEvaluation complete.")
    print("Episode results:", episode_csv)
    print("Policy summary:", summary_csv)
    print("Manifest:", manifest_json)


if __name__ == "__main__":
    main()
