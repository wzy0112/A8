from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit


CONTEXT_NUMERIC_FEATURES = (
    "time_s",
)

# D/X fields are added when they are present in the decision-level dataset.
# The current builder mainly carries D through state/future-flood features, so
# missing explicit context columns are intentionally tolerated.
OPTIONAL_CONTEXT_FEATURES = (
    "severity",
    "site_id",
    "mechanism",
    "demand_phase",
    "day_type",
    "time_of_day_bucket",
)


def _group_split(df: pd.DataFrame, seed: int):
    groups = df["episode_id"].astype(str)
    if groups.nunique() < 2:
        raise ValueError("Need at least two episodes for grouped train/test split.")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed,
    )
    return next(splitter.split(df, groups=groups))


def _encode_context(
    df: pd.DataFrame,
    *,
    fitted_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create explicit D/X context columns when available."""
    pieces: list[pd.DataFrame] = []

    for col in CONTEXT_NUMERIC_FEATURES:
        if col in df.columns:
            pieces.append(
                pd.DataFrame(
                    {f"context::{col}": pd.to_numeric(df[col], errors="coerce")},
                    index=df.index,
                )
            )

    categorical = [c for c in OPTIONAL_CONTEXT_FEATURES if c in df.columns]
    if categorical:
        cat = df[categorical].fillna("unknown").astype(str)
        encoded = pd.get_dummies(
            cat,
            prefix=[f"context::{c}" for c in categorical],
            prefix_sep="::",
            dtype=float,
        )
        pieces.append(encoded)

    context = (
        pd.concat(pieces, axis=1)
        if pieces
        else pd.DataFrame(index=df.index)
    )

    if fitted_columns is not None:
        context = context.reindex(columns=fitted_columns, fill_value=0.0)

    return context.astype(float)


def _build_inputs(df: pd.DataFrame):
    base_features = sorted(
        col
        for col in df.columns
        if (
            col.startswith("state::")
            or col.startswith("future_flood::")
            or col.startswith("M_future::")
        )
    )
    if not base_features:
        raise ValueError("No surrogate input features found.")

    base = df[base_features].apply(pd.to_numeric, errors="coerce")
    context = _encode_context(df)
    x = pd.concat([base, context], axis=1)

    # Remove constant columns: they contain no learnable information and make
    # feature-importance output easier to interpret.
    variable = [
        col for col in x.columns
        if x[col].nunique(dropna=False) > 1
    ]
    x = x[variable]
    return x, list(x.columns)


def _regression_metrics(truth, pred, prefix: str) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        f"{prefix}_mae": float(mean_absolute_error(truth, pred)),
        f"{prefix}_rmse": float(mean_squared_error(truth, pred) ** 0.5),
        f"{prefix}_r2": float(r2_score(truth, pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="recovery_surrogate_samples.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.dataset)

    if "episode_id" not in df.columns:
        raise ValueError("Dataset must contain episode_id.")

    pr_targets = sorted(
        (c for c in df.columns if c.startswith("target_pr_h")),
        key=lambda c: int(c.replace("target_pr_h", "")),
    )
    if not pr_targets:
        raise ValueError("No target_pr_h* targets found.")

    x_all, input_features = _build_inputs(df)
    finite_x = np.isfinite(x_all.to_numpy(dtype=float)).all(axis=1)

    metrics: dict[str, object] = {
        "model_type": "T3.6_recovery_surrogate_final",
        "dataset_rows_total": int(len(df)),
        "dataset_episodes_total": int(df["episode_id"].nunique()),
        "input_feature_count": int(len(input_features)),
        "forecast_targets": pr_targets,
    }

    # ----- PR trajectory model -----
    pr_values = df[pr_targets].apply(pd.to_numeric, errors="coerce")
    pr_mask = finite_x & np.isfinite(pr_values.to_numpy(dtype=float)).all(axis=1)
    pr_df = df.loc[pr_mask].reset_index(drop=True)
    x_pr = x_all.loc[pr_mask].reset_index(drop=True)
    y_pr = pr_values.loc[pr_mask].reset_index(drop=True)

    if len(pr_df) < 10 or pr_df["episode_id"].nunique() < 3:
        raise ValueError("Too few complete PR-horizon samples/episodes.")

    train_idx, test_idx = _group_split(pr_df, args.seed)

    pr_model = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    pr_model.fit(x_pr.iloc[train_idx], y_pr.iloc[train_idx])

    pr_pred = pr_model.predict(x_pr.iloc[test_idx])
    pr_truth = y_pr.iloc[test_idx].to_numpy(dtype=float)

    metrics.update(_regression_metrics(pr_truth, pr_pred, "pr_overall"))
    metrics.update(
        {
            "pr_train_rows": int(len(train_idx)),
            "pr_test_rows": int(len(test_idx)),
            "pr_train_episodes": int(pr_df.iloc[train_idx]["episode_id"].nunique()),
            "pr_test_episodes": int(pr_df.iloc[test_idx]["episode_id"].nunique()),
        }
    )

    for j, target in enumerate(pr_targets):
        horizon = target.replace("target_pr_h", "")
        metrics.update(
            _regression_metrics(
                pr_truth[:, j],
                pr_pred[:, j],
                f"pr_h{horizon}",
            )
        )

    pred_df = pr_df.iloc[test_idx][
        ["episode_id", "decision_epoch", "time_s"]
    ].reset_index(drop=True)

    for j, target in enumerate(pr_targets):
        horizon = target.replace("target_pr_h", "")
        pred_df[f"truth_pr_h{horizon}"] = pr_truth[:, j]
        pred_df[f"pred_pr_h{horizon}"] = pr_pred[:, j]

    # ----- Remaining recovery-time model -----
    rt_col = "target_recovery_time_remaining_s"
    recovery_time_model = None
    if rt_col in df.columns:
        rt_numeric = pd.to_numeric(df[rt_col], errors="coerce")
        rt_mask = finite_x & np.isfinite(rt_numeric.to_numpy(dtype=float))
        rt_df = df.loc[rt_mask].reset_index(drop=True)
        x_rt = x_all.loc[rt_mask].reset_index(drop=True)
        y_rt = rt_numeric.loc[rt_mask].reset_index(drop=True)

        if len(rt_df) >= 10 and rt_df["episode_id"].nunique() >= 3:
            rt_train, rt_test = _group_split(rt_df, args.seed)
            recovery_time_model = ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                random_state=args.seed + 1,
                n_jobs=-1,
                min_samples_leaf=2,
            )
            recovery_time_model.fit(x_rt.iloc[rt_train], y_rt.iloc[rt_train])
            rt_pred = recovery_time_model.predict(x_rt.iloc[rt_test])
            rt_truth = y_rt.iloc[rt_test].to_numpy(dtype=float)

            metrics.update(
                _regression_metrics(
                    rt_truth,
                    rt_pred,
                    "recovery_time_s",
                )
            )
            metrics["recovery_time_train_rows"] = int(len(rt_train))
            metrics["recovery_time_test_rows"] = int(len(rt_test))
            metrics["recovery_time_train_episodes"] = int(
                rt_df.iloc[rt_train]["episode_id"].nunique()
            )
            metrics["recovery_time_test_episodes"] = int(
                rt_df.iloc[rt_test]["episode_id"].nunique()
            )

            rt_out = rt_df.iloc[rt_test][
                ["episode_id", "decision_epoch", "time_s"]
            ].reset_index(drop=True)
            rt_out["truth_recovery_time_remaining_s"] = rt_truth
            rt_out["pred_recovery_time_remaining_s"] = rt_pred
            rt_out.to_csv(
                args.output_dir / "recovery_time_test_predictions.csv",
                index=False,
            )

    # ----- Eventual-recovery classifier -----
    recovered_col = "target_eventual_recovered"
    recovery_classifier = None
    if recovered_col in df.columns:
        cls_mask = finite_x & pd.to_numeric(
            df[recovered_col], errors="coerce"
        ).notna().to_numpy()
        cls_df = df.loc[cls_mask].reset_index(drop=True)
        x_cls = x_all.loc[cls_mask].reset_index(drop=True)
        y_cls = (
            pd.to_numeric(
                df.loc[cls_mask, recovered_col],
                errors="coerce",
            )
            .astype(int)
            .reset_index(drop=True)
        )

        if y_cls.nunique() >= 2 and cls_df["episode_id"].nunique() >= 3:
            cls_train, cls_test = _group_split(cls_df, args.seed)
            recovery_classifier = ExtraTreesClassifier(
                n_estimators=args.n_estimators,
                random_state=args.seed + 2,
                n_jobs=-1,
                min_samples_leaf=2,
                class_weight="balanced",
            )
            recovery_classifier.fit(
                x_cls.iloc[cls_train],
                y_cls.iloc[cls_train],
            )
            cls_pred = recovery_classifier.predict(x_cls.iloc[cls_test])
            cls_truth = y_cls.iloc[cls_test].to_numpy(dtype=int)

            metrics["eventual_recovery_accuracy"] = float(
                accuracy_score(cls_truth, cls_pred)
            )
            metrics["eventual_recovery_balanced_accuracy"] = float(
                balanced_accuracy_score(cls_truth, cls_pred)
            )
            metrics["recovery_classifier_train_rows"] = int(len(cls_train))
            metrics["recovery_classifier_test_rows"] = int(len(cls_test))
            metrics["recovery_classifier_train_episodes"] = int(
                cls_df.iloc[cls_train]["episode_id"].nunique()
            )
            metrics["recovery_classifier_test_episodes"] = int(
                cls_df.iloc[cls_test]["episode_id"].nunique()
            )

    # ----- Feature importance -----
    importance_df = pd.DataFrame(
        {
            "feature": input_features,
            "importance": pr_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(
        args.output_dir / "pr_feature_importance.csv",
        index=False,
    )

    pred_df.to_csv(
        args.output_dir / "pr_test_predictions.csv",
        index=False,
    )

    artifact = {
        "model_type": "T3.6_recovery_surrogate_final",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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

    print("T3.6 recovery surrogate trained.")
    print(json.dumps(metrics, indent=2))
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
