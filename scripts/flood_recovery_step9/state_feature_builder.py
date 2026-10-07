from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from action_space import TacticID
from state_segments import (
    STATE_SEGMENTS,
    get_state_segment_ids,
    segment_overlap_flags,
)

# PPO 的 normalization 已经由 SB3 负责。
# shared builder 返回真实物理值
SEGMENT_FEATURES = (
    "mean_speed_mps",
    "flow_veh_h",
    "density_veh_km_lane",
    "vc_ratio",
    "queue_length_veh",
)

MANAGEMENT_TACTICS = (
    TacticID.VSL,
    TacticID.LANE_CONTROL,
    TacticID.TRUCK_BAN,
    TacticID.REROUTE,
)

FLOOD_STAGE_FEATURES = (
    "normal",
    "onset",
    "peak",
    "recovery",
)

SEVERITY_FEATURES = (
    "light",
    "moderate",
    "severe",
)

TIME_OF_DAY_BUCKETS = (
    "pre_peak",
    "peak",
    "post_peak",
)

DEMAND_PHASES = (
    "non_peak",
    "peak",
)

DAY_TYPES = (
    "weekday",
    "weekend",
    "unknown",
)


class StateFeatureBuilder:
    """
    Shared Part-3 flat-vector builder for both RL and later T3.6 regression.

    The builder deliberately returns raw, physically interpretable values.
    PPO normalization is handled separately by VecNormalize, while T3.6 can
    reuse the identical feature ordering.

    With 6 attached state segments and 6 management locations:
        6 * 5 segment metrics             = 30
        3 global metrics                  =  3
        4 flood-stage one-hot             =  4
        3 severity one-hot                =  3
        1 time-since-onset                =  1
        6 spatial flood flags             =  6
        4 tactics * 6 locations           = 24
        1 PR(t)                            =  1
        3 time-of-day one-hot             =  3
        2 demand-phase one-hot             =  2
        3 day-type one-hot                 =  3
                                             --
                                             80 floats

    MQ/SL is intentionally excluded.
    """

    def __init__(
        self,
        *,
        location_ids: Sequence[str],
        segment_ids: Sequence[str] | None = None,
    ):
        self.segment_ids = list(
            segment_ids
            if segment_ids is not None
            else get_state_segment_ids()
        )

        self.location_ids = list(location_ids)

        if not self.segment_ids:
            raise ValueError(
                "At least one state segment is required."
            )
        if not self.location_ids:
            raise ValueError(
                "At least one management location is required."
            )

        unknown_segments = [
            segment_id
            for segment_id in self.segment_ids
            if segment_id not in STATE_SEGMENTS
        ]
        if unknown_segments:
            raise ValueError(
                "Unknown state segments: "
                + ", ".join(unknown_segments)
            )

        self._feature_names = self._build_feature_names()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def state_dim(self) -> int:
        return len(self._feature_names)

    def _build_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []

        for segment_id in self.segment_ids:
            for metric_name in SEGMENT_FEATURES:
                names.append(
                    f"segment.{segment_id}.{metric_name}"
                )

        names.extend([
            "global.running_vehicle_count",
            "global.mean_speed_mps",
            "global.mean_time_loss_s",
        ])

        for stage in FLOOD_STAGE_FEATURES:
            names.append(
                f"flood.stage.{stage}"
            )

        for severity in SEVERITY_FEATURES:
            names.append(
                f"flood.severity.{severity}"
            )

        names.append(
            "flood.time_since_onset_s"
        )

        for segment_id in self.segment_ids:
            names.append(
                f"flood.affected.{segment_id}"
            )

        for tactic in MANAGEMENT_TACTICS:
            tactic_name = tactic.name.lower()
            for location_id in self.location_ids:
                names.append(
                    "management."
                    f"{tactic_name}."
                    f"{location_id}.intensity"
                )

        names.append("performance.pr_t")

        for bucket in TIME_OF_DAY_BUCKETS:
            names.append(
                f"context.time_of_day.{bucket}"
            )

        for phase in DEMAND_PHASES:
            names.append(
                f"context.demand_phase.{phase}"
            )

        for day_type in DAY_TYPES:
            names.append(
                f"context.day_type.{day_type}"
            )

        return tuple(names)

    def management_slot_vector(
        self,
        *,
        active_measures: Mapping[
            tuple[int, str],
            object
        ],
    ) -> list[float]:
        """
        Build one raw intensity value (0/1/2/3) per tactic-location slot.
        """
        vector: list[float] = []

        for tactic in MANAGEMENT_TACTICS:
            for location_id in self.location_ids:
                action = active_measures.get(
                    (
                        int(tactic),
                        location_id,
                    )
                )

                if action is None:
                    vector.append(0.0)
                else:
                    vector.append(
                        float(
                            getattr(
                                action,
                                "intensity_index",
                            )
                        )
                    )

        return vector

    def build(
        self,
        *,
        current_time_s: float,
        global_metrics: Mapping[str, object],
        segment_metrics: Mapping[
            str,
            Mapping[str, float],
        ],
        scenario,
        flood_stage: str,
        flood_edge_ids: Sequence[str],
        management_vector: Sequence[float],
        pr_t: float,
        day_type: str | None = None,
    ) -> np.ndarray:
        values: list[float] = []

        for segment_id in self.segment_ids:
            try:
                metrics = segment_metrics[
                    segment_id
                ]
            except KeyError as exc:
                raise KeyError(
                    f"Missing segment metrics for {segment_id}"
                ) from exc

            for metric_name in SEGMENT_FEATURES:
                value = float(
                    metrics.get(
                        metric_name,
                        0.0,
                    )
                )

                if not math.isfinite(value):
                    value = 0.0

                values.append(value)

        values.extend([
            float(
                global_metrics[
                    "running_vehicle_count"
                ]
            ),
            float(
                global_metrics[
                    "mean_speed_mps"
                ]
            ),
            float(
                global_metrics[
                    "mean_time_loss_s"
                ]
            ),
        ])

        stage_key = str(flood_stage).lower()

        # The requested state schema has four stage dimensions.
        # POST_FLOOD therefore maps to NORMAL because the physical flood
        # restrictions are no longer active.
        if stage_key == "post_flood":
            stage_key = "normal"

        for stage in FLOOD_STAGE_FEATURES:
            values.append(
                float(stage_key == stage)
            )

        severity_key = str(
            scenario.severity
        ).lower()

        for severity in SEVERITY_FEATURES:
            values.append(
                float(
                    severity_key == severity
                )
            )

        time_since_onset = max(
            0.0,
            float(current_time_s)
            - float(scenario.onset_t),
        )
        values.append(time_since_onset)

        flood_active = stage_key in {
            "onset",
            "peak",
            "recovery",
        }

        all_flags = segment_overlap_flags(
            flood_edge_ids=flood_edge_ids,
            flood_active=flood_active,
        )

        for segment_id in self.segment_ids:
            values.append(
                float(
                    all_flags.get(
                        segment_id,
                        0.0,
                    )
                )
            )

        expected_management_dim = (
            len(MANAGEMENT_TACTICS)
            * len(self.location_ids)
        )

        if (
            len(management_vector)
            != expected_management_dim
        ):
            raise ValueError(
                "Management vector has length "
                f"{len(management_vector)}, expected "
                f"{expected_management_dim}."
            )

        values.extend(
            float(value)
            for value in management_vector
        )

        pr_value = float(pr_t)
        if not math.isfinite(pr_value):
            pr_value = 0.0
        values.append(max(0.0, pr_value))

        current_time = float(current_time_s)
        if current_time < 1800.0:
            time_bucket = "pre_peak"
        elif current_time < 3600.0:
            time_bucket = "peak"
        else:
            time_bucket = "post_peak"

        for bucket in TIME_OF_DAY_BUCKETS:
            values.append(
                float(
                    time_bucket == bucket
                )
            )

        demand_phase = str(
            getattr(
                scenario,
                "demand_phase",
                "non_peak",
            )
        ).lower()

        for phase in DEMAND_PHASES:
            values.append(
                float(
                    demand_phase == phase
                )
            )

        resolved_day_type = (
            day_type
            if day_type is not None
            else getattr(
                scenario,
                "day_type",
                None,
            )
        )

        if resolved_day_type not in {
            "weekday",
            "weekend",
        }:
            resolved_day_type = "unknown"

        for category in DAY_TYPES:
            values.append(
                float(
                    resolved_day_type == category
                )
            )

        state = np.asarray(
            values,
            dtype=np.float32,
        )

        if state.shape != (self.state_dim,):
            raise RuntimeError(
                f"State shape {state.shape} does not match "
                f"schema dimension {self.state_dim}."
            )

        if not np.all(np.isfinite(state)):
            raise RuntimeError(
                "State vector contains NaN or Inf."
            )

        return state

    def schema_dict(self) -> dict:
        return {
            "version": "part3_state_v1",
            "dimension": self.state_dim,
            "feature_names": list(
                self.feature_names
            ),
            "segment_ids": list(
                self.segment_ids
            ),
            "location_ids": list(
                self.location_ids
            ),
            "notes": {
                "mq_sl": "removed",
                "stage_post_flood": (
                    "encoded as flood.stage.normal"
                ),
                "day_type": (
                    "unknown until day-type demand "
                    "randomization is implemented"
                ),
            },
        }

    def save_schema(
        self,
        path: Path,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            json.dumps(
                self.schema_dict(),
                indent=2,
            ),
            encoding="utf-8",
        )
