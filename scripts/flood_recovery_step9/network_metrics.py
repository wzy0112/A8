from __future__ import annotations

import csv
from pathlib import Path


STEP_METRIC_FIELDS = [
    "time_s",
    "flood_stage",
    "running_vehicle_count",
    "loaded_vehicle_count",
    "departed_vehicle_count",
    "arrived_vehicle_count",
    "mean_speed_mps",
    "mean_waiting_time_s",
    "mean_time_loss_s",
    "halting_vehicle_count",
]


def collect_network_step_metrics(
    traci_api,
    flood_stage: str,
) -> dict[str, float | str]:
    vehicle_ids = list(traci_api.vehicle.getIDList())

    if vehicle_ids:
        speeds = [
            max(0.0, traci_api.vehicle.getSpeed(vehicle_id))
            for vehicle_id in vehicle_ids
        ]
        waiting_times = [
            traci_api.vehicle.getWaitingTime(vehicle_id)
            for vehicle_id in vehicle_ids
        ]
        time_losses = [
            traci_api.vehicle.getTimeLoss(vehicle_id)
            for vehicle_id in vehicle_ids
        ]

        mean_speed = sum(speeds) / len(speeds)
        mean_waiting = sum(waiting_times) / len(waiting_times)
        mean_time_loss = sum(time_losses) / len(time_losses)
        halting_count = sum(speed < 0.1 for speed in speeds)
    else:
        mean_speed = 0.0
        mean_waiting = 0.0
        mean_time_loss = 0.0
        halting_count = 0

    simulation = traci_api.simulation

    return {
        "time_s": float(simulation.getTime()),
        "flood_stage": flood_stage,
        "running_vehicle_count": len(vehicle_ids),
        "loaded_vehicle_count": int(
            simulation.getLoadedNumber()
        ),
        "departed_vehicle_count": int(
            simulation.getDepartedNumber()
        ),
        "arrived_vehicle_count": int(
            simulation.getArrivedNumber()
        ),
        "mean_speed_mps": mean_speed,
        "mean_waiting_time_s": mean_waiting,
        "mean_time_loss_s": mean_time_loss,
        "halting_vehicle_count": halting_count,
    }


class StepMetricsWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._file = None
        self._writer = None

    def __enter__(self):
        self._file = self.output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=STEP_METRIC_FIELDS,
        )
        self._writer.writeheader()
        return self

    def write(self, metrics: dict) -> None:
        self._writer.writerow(metrics)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file is not None:
            self._file.close()
