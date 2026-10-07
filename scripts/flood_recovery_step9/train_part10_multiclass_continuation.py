from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from randomized_env_factory import make_randomized_env
from training_callback import RecoveryTrainingCallback


CURRICULUM = (
    ("fixed_moderate", 20_000),
    ("mixed_fixed", 40_000),
    ("randomized", 100_000),
)


def build_raw_vec_env(
    *,
    stage_name: str,
    n_envs: int,
    base_seed: int,
    decision_period_s: int,
    stage_dir: Path,
    detailed_logs: bool,
):
    """
    Build the raw vector environment + VecMonitor only.

    VecNormalize is intentionally created/loaded separately so normalization
    statistics can continue across curriculum stages.
    """
    factories = [
        make_randomized_env(
            rank=rank,
            base_seed=base_seed,
            sampling_mode=stage_name,
            decision_period_s=decision_period_s,
            output_root=stage_dir / "raw_episodes",
            enable_episode_logging=detailed_logs,
        )
        for rank in range(n_envs)
    ]

    if n_envs == 1:
        env = DummyVecEnv(factories)
    else:
        env = SubprocVecEnv(
            factories,
            start_method="spawn",
        )

    return VecMonitor(
        env,
        filename=str(stage_dir / "monitor.csv"),
    )


def build_normalized_env(
    *,
    raw_env,
    previous_norm_path: Path | None,
):
    """
    Stage 1: create fresh VecNormalize statistics.
    Later stages: load the preceding stage's statistics and continue updating.
    """
    if previous_norm_path is None:
        env = VecNormalize(
            raw_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
    else:
        env = VecNormalize.load(
            str(previous_norm_path),
            raw_env,
        )
        env.training = True
        env.norm_reward = True

    return env


def create_model(env, tensorboard_dir: Path, seed: int):
    """
    Keep the PPO hyperparameters that already ran successfully in Step 7.
    This makes scenario randomization the main experimental change in Step 8.
    """
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=16,
        batch_size=16,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": {
                "pi": [128, 128],
                "vf": [128, 128],
            }
        },
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
        seed=seed,
        device="auto",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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
            r"\step9_curriculum"
        ),
    )
    parser.add_argument(
        "--stage1-steps",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--stage2-steps",
        type=int,
        default=40_000,
    )
    parser.add_argument(
        "--stage3-steps",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--no-detailed-logs",
        action="store_true",
        help=(
            "Disable full per-episode logging. "
            "Part 8 requires logging by default."
        ),
    )
    parser.add_argument(
        "--resume-model",
        type=Path,
        required=True,
        help="Existing PPO .zip checkpoint to continue from.",
    )
    parser.add_argument(
        "--resume-vecnormalize",
        type=Path,
        required=True,
        help="Existing VecNormalize .pkl to continue from.",
    )

    args = parser.parse_args()

    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1.")

    curriculum = (
        ("fixed_moderate", args.stage1_steps),
        ("mixed_fixed", args.stage2_steps),
        ("randomized", args.stage3_steps),
    )

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.resume_model.exists():
        raise FileNotFoundError(args.resume_model)
    if not args.resume_vecnormalize.exists():
        raise FileNotFoundError(args.resume_vecnormalize)

    # Continue from the completed 5k checkpoint. Do not create a fresh PPO.
    model = None
    previous_model_path: Path | None = args.resume_model
    previous_norm_path: Path | None = args.resume_vecnormalize
    stage_records = []

    for stage_index, (stage_name, stage_steps) in enumerate(
        curriculum,
        start=1,
    ):
        if stage_steps <= 0:
            continue

        stage_dir = (
            run_dir
            / f"stage_{stage_index}_{stage_name}"
        )
        stage_dir.mkdir(parents=True, exist_ok=True)

        raw_env = build_raw_vec_env(
            stage_name=stage_name,
            n_envs=args.n_envs,
            base_seed=args.seed + stage_index * 1000,
            decision_period_s=args.decision_period,
            stage_dir=stage_dir,
            detailed_logs=not args.no_detailed_logs,


        )

        env = build_normalized_env(
            raw_env=raw_env,
            previous_norm_path=previous_norm_path,
        )

        if model is None:
            model = PPO.load(
                str(args.resume_model),
                env=env,
                device="auto",
            )
            # Keep TensorBoard output in the continuation run directory.
            model.tensorboard_log = str(run_dir / "tensorboard")
        else:
            model.set_env(env)

        callback = RecoveryTrainingCallback(
            output_dir=stage_dir / "checkpoints",
            checkpoint_every_steps=25_000,
        )

        try:
            model.learn(
                total_timesteps=stage_steps,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=True,
                tb_log_name=stage_name,
            )

            model_path = stage_dir / "ppo_stage_model"
            norm_path = stage_dir / "vec_normalize.pkl"

            model.save(model_path)
            env.save(str(norm_path))

            previous_model_path = model_path
            previous_norm_path = norm_path

            stage_records.append({
                "stage": stage_index,
                "sampling_mode": stage_name,
                "timesteps": stage_steps,
                "model_path": str(
                    model_path.with_suffix(".zip")
                ),
                "normalization_path": str(norm_path),
            })
        finally:
            env.close()

    if (
        model is None
        or previous_model_path is None
        or previous_norm_path is None
    ):
        raise RuntimeError(
            "No curriculum stage was trained."
        )

    final_model_path = run_dir / "ppo_curriculum_final"
    final_norm_path = run_dir / "vec_normalize_final.pkl"

    model.save(final_model_path)
    shutil.copy2(
        previous_norm_path,
        final_norm_path,
    )

    (run_dir / "curriculum_manifest.json").write_text(
        json.dumps(stage_records, indent=2),
        encoding="utf-8",
    )

    print("Curriculum training complete.")
    print(
        "Final model:",
        final_model_path.with_suffix(".zip"),
    )
    print(
        "Final VecNormalize:",
        final_norm_path,
    )


if __name__ == "__main__":
    main()
