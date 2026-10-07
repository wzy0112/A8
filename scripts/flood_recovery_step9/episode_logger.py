
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flood_scenario import FloodScenario


EPOCH_FIELDS = [
    "episode_id",
    "time_s",
    "decision_epoch",
    "tactic_id",
    "tactic_name",
    "location_index",
    "location_id",
    "intensity_index",
    "intensity_name",
    "operation",
    "management_changed",
    "active_measure_count",
    "active_measures_after_json",
    "applied",
    "invalid_reason",
    "changed_lanes",
    "changed_edges",
    "rerouted_vehicles",
    "flood_stage",
    "running_vehicle_count",
    "mean_speed_mps",
    "baseline_mean_speed_mps",
    "pr_t",
    "mean_waiting_time_s",
    "mean_time_loss_s",
    "halting_vehicle_count",
    "delta_pr",
    "pr_progress_reward",
    "action_switch_cost",
    "queue_growth_veh",
    "queue_growth_penalty",
    "terminal_reward",
    "reward",
    "co2_epoch_kg",
    "hard_braking_events_epoch",
    "queue_exposure_epoch_veh_h",
    "time_loss_epoch_veh_h",
    "co2_cumulative_kg",
    "hard_braking_events_cumulative",
    "queue_exposure_cumulative_veh_h",
    "time_loss_cumulative_veh_h",
    "recovered",
]

SECOND_FIELDS = [
    "episode_id",
    "time_s",
    "decision_epoch",
    "tactic_name",
    "location_id",
    "intensity_name",
    "flood_stage",
    "running_vehicle_count",
    "mean_speed_mps",
    "baseline_mean_speed_mps",
    "pr_t",
    "co2_step_mg",
    "hard_braking_events_step",
    "queue_exposure_step_veh_s",
    "time_loss_step_s",
    "recovered",
]


class EpisodeLogger:
    def __init__(
        self,
        *,
        output_dir: Path,
        scenario: FloodScenario,
        policy_version: str,
        sumo_seed: int,
        decision_period_s: int,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scenario = scenario
        self.policy_version = policy_version
        self.sumo_seed = int(sumo_seed)
        self.decision_period_s = int(decision_period_s)

        self.episode_id = str(uuid4())
        self.started_at_utc = datetime.now(
            timezone.utc
        ).isoformat()

        self.second_rows: list[dict] = []
        self.epoch_rows: list[dict] = []
        self.state_rows: list[dict] = []
        self.total_reward = 0.0
        self.initial_state: dict | None = None

    def set_initial_state(self, initial_state: dict) -> None:
        self.initial_state = dict(initial_state)

    def log_second(self, row: dict) -> None:
        self.second_rows.append({
            "episode_id": self.episode_id,
            **row,
        })

    def log_epoch(self, row: dict) -> None:
        stored = {
            "episode_id": self.episode_id,
            **row,
        }
        self.epoch_rows.append(stored)
        self.total_reward += float(row["reward"])

    def log_state(
            self,
            *,
            time_s: float,
            decision_epoch: int,
            feature_names,
            state_vector,
    ) -> None:

        row = {
            "episode_id": self.episode_id,
            "time_s": float(time_s),
            "decision_epoch": int(
                decision_epoch
            ),
        }

        for name, value in zip(
                feature_names,
                state_vector,
                strict=True,
        ):
            row[name] = float(value)

        self.state_rows.append(row)

    def _write_csv(
            self,
            path: Path,
            rows: list[dict],
            fieldnames: list[str],
    ) -> None:

        if not rows:
            raise ValueError(
                f"Cannot write empty log: {path.name}"
            )

        with path.open(
                "w",
                newline="",
                encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

    def finalize(
        self,
        *,
        recovered: bool,
        truncated: bool,
        recovery_status: dict,
        final_time_s: float,
        impact_totals: dict,
        context: dict | None = None,
    ) -> dict[str, Path]:
        second_path = (
            self.output_dir
            / f"episode_{self.episode_id}_second_trajectory.csv"
        )
        epoch_path = (
            self.output_dir
            / f"episode_{self.episode_id}_epoch_log.csv"
        )
        summary_path = (
            self.output_dir
            / f"episode_{self.episode_id}_summary.json"
        )
        state_path = (
                self.output_dir
                / f"episode_{self.episode_id}_state_log.csv"
        )
        self._write_csv(second_path, self.second_rows, SECOND_FIELDS)
        self._write_csv(epoch_path, self.epoch_rows, EPOCH_FIELDS)
        if self.state_rows:
            state_fields = list(
                self.state_rows[0]
            )

            self._write_csv(
                state_path,
                self.state_rows,
                state_fields,
            )
        tactic_chain = [
            {
                "t": row["time_s"],
                "tactic_id": row["tactic_id"],
                "tactic": row["tactic_name"],
                "location": row["location_id"],
                "intensity": row["intensity_index"],
                "intensity_name": row["intensity_name"],
                "operation": row.get("operation"),
                "management_changed": row.get(
                    "management_changed"
                ),
                "active_measures_after": (
                    json.loads(
                        row["active_measures_after_json"]
                    )
                    if row.get(
                        "active_measures_after_json"
                    )
                    else []
                ),
                "applied": row.get(
                    "applied",
                    row.get(
                        "action_applied",
                        True,
                    ),
                ),
            }
            for row in self.epoch_rows
        ]

        initial_time_s = float(
            (self.initial_state or {}).get(
                "time_s",
                self.scenario.onset_t,
            )
        )

        if initial_time_s < 1800:
            time_of_day_bucket = "pre_peak"
        elif initial_time_s < 3600:
            time_of_day_bucket = "peak"
        else:
            time_of_day_bucket = "post_peak"

        summary = {
            "episode_id": self.episode_id,
            "policy_version": self.policy_version,
            "sumo_seed": self.sumo_seed,
            "decision_period_s": self.decision_period_s,
            "started_at_utc": self.started_at_utc,
            "D": {
                **asdict(self.scenario),
                "mechanism": self.scenario.mechanism.value,
            },
            "S0": self.initial_state or {},
            "X": context or {
                "time_of_day_bucket": time_of_day_bucket,
                "demand_phase": getattr(
                    self.scenario,
                    "demand_phase",
                    "unknown",
                ),
                "day_type": "fixed",
            },
            "M": tactic_chain,
            "outcomes": {
                "recovered": bool(recovered),
                "truncated": bool(truncated),
                "final_time_s": float(final_time_s),
                "reward_total": float(self.total_reward),
                **recovery_status,
                **impact_totals,
            },
            "state_log_ref": str(state_path),
            "trajectory_ref": str(second_path),
            "epoch_log_ref": str(epoch_path),
            "final_management_state": (
                tactic_chain[-1][
                    "active_measures_after"
                ]
                if tactic_chain
                else []
            ),
        }

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        return {
            "second_trajectory_path": second_path,
            "epoch_log_path": epoch_path,
            "summary_path": summary_path,
            "state_log_path": state_path
        }
