from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GroupShuffleSplit


def _group_split(df: pd.DataFrame, seed: int):
    groups = df["episode_id"].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed,
    )
    return next(
        splitter.split(
            df,
            groups=groups,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="recovery_surrogate_samples.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset)

    input_features = sorted(
        col for col in df.columns
        if (
            col.startswith("state::")
            or col.startswith("future_flood::")
            or col.startswith("M_future::")
        )
    )
    pr_targets = sorted(
        col for col in df.columns
        if col.startswith("target_pr_h")
    )

    if not input_features:
        raise ValueError("No surrogate input features found.")
    if not pr_targets:
        raise ValueError("No target_pr_h* targets found.")

    x_all = df[input_features].astype(float)
    finite_x = np.isfinite(x_all.to_numpy()).all(axis=1)

    # PR model: only rows where all requested horizons are available.
    pr_mask = (
        finite_x
        & np.isfinite(
            df[pr_targets].astype(float).to_numpy()
        ).all(axis=1)
    )
    pr_df = df.loc[pr_mask].reset_index(drop=True)
    x_pr = pr_df[input_features].astype(float)
    y_pr = pr_df[pr_targets].astype(float)

    if len(pr_df) < 10:
        raise ValueError(
            "Too few complete PR-horizon rows. "
            "Generate more completed episodes."
        )

    train_idx, test_idx = _group_split(
        pr_df,
        args.seed,
    )

    pr_model = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    pr_model.fit(
        x_pr.iloc[train_idx],
        y_pr.iloc[train_idx],
    )

    pr_pred = pr_model.predict(
        x_pr.iloc[test_idx]
    )
    pr_truth = y_pr.iloc[test_idx].to_numpy()

    metrics = {
        "pr_mae_overall": float(
            mean_absolute_error(
                pr_truth,
                pr_pred,
            )
        ),
        "pr_rmse_overall": float(
            mean_squared_error(
                pr_truth,
                pr_pred,
            ) ** 0.5
        ),
        "pr_train_rows": int(len(train_idx)),
        "pr_test_rows": int(len(test_idx)),
        "pr_train_episodes": int(
            pr_df.iloc[train_idx]["episode_id"].nunique()
        ),
        "pr_test_episodes": int(
            pr_df.iloc[test_idx]["episode_id"].nunique()
        ),
    }

    # Remaining-recovery-time model: recovered cases only.
    rt_col = "target_recovery_time_remaining_s"
    rt_mask = (
        finite_x
        & np.isfinite(
            pd.to_numeric(
                df[rt_col],
                errors="coerce",
            ).to_numpy()
        )
    )
    rt_df = df.loc[rt_mask].reset_index(drop=True)

    recovery_time_model = None
    if len(rt_df) >= 10 and rt_df["episode_id"].nunique() >= 3:
        rt_train, rt_test = _group_split(
            rt_df,
            args.seed,
        )
        recovery_time_model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            random_state=args.seed + 1,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        recovery_time_model.fit(
            rt_df[input_features].iloc[rt_train],
            rt_df[rt_col].astype(float).iloc[rt_train],
        )
        rt_pred = recovery_time_model.predict(
            rt_df[input_features].iloc[rt_test]
        )
        rt_truth = (
            rt_df[rt_col]
            .astype(float)
            .iloc[rt_test]
            .to_numpy()
        )
        metrics["recovery_time_mae_s"] = float(
            mean_absolute_error(rt_truth, rt_pred)
        )
        metrics["recovery_time_rmse_s"] = float(
            mean_squared_error(
                rt_truth,
                rt_pred,
            ) ** 0.5
        )

    # Eventual recovery classifier.
    recovered_col = "target_eventual_recovered"
    cls_df = df.loc[finite_x].reset_index(drop=True)
    recovery_classifier = None
    if (
        cls_df[recovered_col].nunique() >= 2
        and cls_df["episode_id"].nunique() >= 3
    ):
        cls_train, cls_test = _group_split(
            cls_df,
            args.seed,
        )
        recovery_classifier = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed + 2,
            n_jobs=-1,
            min_samples_leaf=2,
            class_weight="balanced",
        )
        recovery_classifier.fit(
            cls_df[input_features].iloc[cls_train],
            cls_df[recovered_col].astype(int).iloc[cls_train],
        )
        cls_pred = recovery_classifier.predict(
            cls_df[input_features].iloc[cls_test]
        )
        metrics["eventual_recovery_accuracy"] = float(
            accuracy_score(
                cls_df[recovered_col]
                .astype(int)
                .iloc[cls_test],
                cls_pred,
            )
        )

    artifact = {
        "model_type": "T3.6_recovery_surrogate_v1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_features": input_features,
        "pr_target_features": pr_targets,
        "pr_model": pr_model,
        "recovery_time_model": recovery_time_model,
        "recovery_classifier": recovery_classifier,
    }

    joblib.dump(
        artifact,
        args.output_dir / "recovery_surrogate.joblib",
    )
    (args.output_dir / "recovery_surrogate_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Recovery surrogate trained.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
