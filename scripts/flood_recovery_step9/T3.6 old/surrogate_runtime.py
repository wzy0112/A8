from __future__ import annotations

from pathlib import Path
from typing import Mapping

import joblib
import numpy as np


def _row_from_mapping(
    feature_names: list[str],
    values: Mapping[str, float],
) -> np.ndarray:
    missing = [
        name
        for name in feature_names
        if name not in values
    ]
    if missing:
        raise KeyError(
            "Missing surrogate features: "
            + ", ".join(missing[:20])
        )

    row = np.asarray(
        [
            float(values[name])
            for name in feature_names
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(row)):
        raise ValueError(
            "Surrogate input contains NaN/Inf."
        )

    return row.reshape(1, -1)


class FloodSurrogate:
    """
    T3.5 runtime wrapper.

    Current use:
        current 68-D state -> predicted future flood trajectory vector.

    It is intentionally not wired into FloodRecoveryEnv yet.
    """

    def __init__(self, model_path: Path):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.input_features = list(
            artifact["input_features"]
        )
        self.target_features = list(
            artifact["target_features"]
        )

    def predict(
        self,
        state_features: Mapping[str, float],
    ) -> dict[str, float]:
        values = {
            f"state::{key}": float(value)
            for key, value in state_features.items()
        }
        x = _row_from_mapping(
            self.input_features,
            values,
        )
        pred = self.model.predict(x)[0]

        # Tree regressors can slightly leave [0, 1] for one-hot targets.
        pred = np.clip(pred, 0.0, 1.0)

        return {
            name.replace(
                "target_flood::",
                "future_flood::",
                1,
            ): float(value)
            for name, value in zip(
                self.target_features,
                pred,
                strict=True,
            )
        }


class RecoverySurrogate:
    """
    T3.6 runtime wrapper.

    Inputs:
        current state
        + predicted future flood trajectory
        + encoded candidate future tactic chain

    Outputs:
        future PR trajectory at fixed horizons
        + optional remaining recovery time / recovery probability.
    """

    def __init__(self, model_path: Path):
        artifact = joblib.load(model_path)
        self.input_features = list(
            artifact["input_features"]
        )
        self.pr_targets = list(
            artifact["pr_target_features"]
        )
        self.pr_model = artifact["pr_model"]
        self.recovery_time_model = artifact.get(
            "recovery_time_model"
        )
        self.recovery_classifier = artifact.get(
            "recovery_classifier"
        )

    def predict(
        self,
        *,
        state_features: Mapping[str, float],
        future_flood_features: Mapping[str, float],
        tactic_chain_features: Mapping[str, float],
    ) -> dict:
        values = {
            f"state::{key}": float(value)
            for key, value in state_features.items()
        }
        values.update(
            {
                key: float(value)
                for key, value in future_flood_features.items()
            }
        )
        values.update(
            {
                key: float(value)
                for key, value in tactic_chain_features.items()
            }
        )

        x = _row_from_mapping(
            self.input_features,
            values,
        )

        pr_pred = self.pr_model.predict(x)[0]
        result = {
            "predicted_pr": {
                target: float(value)
                for target, value in zip(
                    self.pr_targets,
                    pr_pred,
                    strict=True,
                )
            }
        }

        if self.recovery_time_model is not None:
            result[
                "predicted_recovery_time_remaining_s"
            ] = float(
                max(
                    0.0,
                    self.recovery_time_model.predict(x)[0],
                )
            )

        if self.recovery_classifier is not None:
            probabilities = (
                self.recovery_classifier.predict_proba(x)[0]
            )
            classes = list(
                self.recovery_classifier.classes_
            )
            if 1 in classes:
                result[
                    "predicted_recovery_probability"
                ] = float(
                    probabilities[
                        classes.index(1)
                    ]
                )

        return result
