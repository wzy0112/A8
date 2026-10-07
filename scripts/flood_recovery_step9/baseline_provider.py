from __future__ import annotations

import csv
from pathlib import Path


class BaselineTrajectory:
    """Time-matched baseline network mean speed provider."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.mean_speed_by_time: dict[int, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Baseline CSV not found: {self.csv_path}")

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {"time_s", "mean_speed_mps"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Baseline CSV missing columns: {sorted(missing)}")

            for line_number, row in enumerate(reader, start=2):
                try:
                    time_s = int(round(float(row["time_s"])))
                    speed = float(row["mean_speed_mps"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid baseline value at line {line_number}") from exc
                self.mean_speed_by_time[time_s] = speed

        if not self.mean_speed_by_time:
            raise ValueError(f"Baseline CSV contains no rows: {self.csv_path}")

    def get_mean_speed(self, time_s: float) -> float:
        key = int(round(time_s))
        try:
            return self.mean_speed_by_time[key]
        except KeyError as exc:
            raise KeyError(f"No baseline speed at t={key} s") from exc
