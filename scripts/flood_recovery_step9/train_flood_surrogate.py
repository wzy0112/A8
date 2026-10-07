from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="flood_forecast_samples.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset)

    input_features = sorted(
        col for col in df.columns
        if col.startswith("state::")
    )
    target_features = sorted(
        col for col in df.columns
        if col.startswith("target_flood::")
    )

    if not input_features:
        raise ValueError("No state:: input features found.")
    if not target_features:
        raise ValueError("No target_flood:: targets found.")

    x = df[input_features].astype(float)
    y = df[target_features].astype(float)
    groups = df["episode_id"].astype(str)

    finite = (
        np.isfinite(x.to_numpy()).all(axis=1)
        & np.isfinite(y.to_numpy()).all(axis=1)
    )
    x = x.loc[finite].reset_index(drop=True)
    y = y.loc[finite].reset_index(drop=True)
    groups = groups.loc[finite].reset_index(drop=True)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=args.seed,
    )
    train_idx, test_idx = next(
        splitter.split(x, y, groups=groups)
    )

    model = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(
        x.iloc[train_idx],
        y.iloc[train_idx],
    )

    pred = model.predict(x.iloc[test_idx])
    truth = y.iloc[test_idx].to_numpy()

    metrics = {
        "mae_overall": float(
            mean_absolute_error(truth, pred)
        ),
        "rmse_overall": float(
            mean_squared_error(
                truth,
                pred,
            ) ** 0.5
        ),
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_train_episodes": int(
            groups.iloc[train_idx].nunique()
        ),
        "n_test_episodes": int(
            groups.iloc[test_idx].nunique()
        ),
    }

    artifact = {
        "model_type": "ExtraTreesRegressor",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_features": input_features,
        "target_features": target_features,
        "model": model,
    }

    joblib.dump(
        artifact,
        args.output_dir / "flood_surrogate.joblib",
    )
    (args.output_dir / "flood_surrogate_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Flood surrogate trained.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
