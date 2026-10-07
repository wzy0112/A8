
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RecoveryConfig:
    """
    Shared definition used by:
    - offline CSV post-processing;
    - Gym reward/termination;
    - T3.6 label extraction.
    """
    threshold: float = 0.90
    stability_window_s: int = 600
    search_start_t: float | None = None

    def validate(self) -> None:
        if not 0.0 < self.threshold:
            raise ValueError("threshold must be positive.")
        if self.stability_window_s < 0:
            raise ValueError("stability_window_s cannot be negative.")


@dataclass
class RecoveryStatus:
    recovered: bool = False
    candidate_start_t: float | None = None
    threshold_entry_time_s: float | None = None
    confirmation_time_s: float | None = None
    recovery_time_s: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def compute_performance_ratio(
    current_performance: float,
    baseline_performance: float,
    *,
    zero_baseline_value: float = math.nan,
) -> float:
    """
    Compute PR(t) = current / baseline.

    Returns zero_baseline_value when the baseline is zero or invalid.
    """
    if not math.isfinite(current_performance):
        return math.nan

    if not math.isfinite(baseline_performance):
        return math.nan

    if baseline_performance <= 1e-9:
        return zero_baseline_value

    return current_performance / baseline_performance


class RecoveryTracker:
    """
    Online recovery detector.

    Call update() once per metric sample. The same class can be used inside
    a Gym environment during simulation and during offline trajectory replay.
    """

    def __init__(
        self,
        config: RecoveryConfig,
        flood_onset_t: float,
    ):
        config.validate()

        self.config = config
        self.flood_onset_t = float(flood_onset_t)
        self.status = RecoveryStatus()

    def reset(self) -> None:
        self.status = RecoveryStatus()

    def update(
        self,
        current_time_s: float,
        pr_t: float,
    ) -> RecoveryStatus:
        current_time_s = float(current_time_s)

        if self.status.recovered:
            return self.status

        if (
            self.config.search_start_t is not None
            and current_time_s < self.config.search_start_t
        ):
            self.status.candidate_start_t = None
            return self.status

        if not math.isfinite(pr_t):
            self.status.candidate_start_t = None
            return self.status

        if pr_t >= self.config.threshold:
            if self.status.candidate_start_t is None:
                self.status.candidate_start_t = current_time_s

            stable_duration = (
                current_time_s - self.status.candidate_start_t
            )

            if stable_duration >= self.config.stability_window_s:
                self.status.recovered = True
                self.status.threshold_entry_time_s = (
                    self.status.candidate_start_t
                )
                self.status.confirmation_time_s = current_time_s
                self.status.recovery_time_s = (
                    self.status.threshold_entry_time_s
                    - self.flood_onset_t
                )
        else:
            self.status.candidate_start_t = None

        return self.status


def extract_recovery_from_trajectory(
    trajectory: Iterable[Mapping[str, float]],
    *,
    time_key: str = "time_s",
    pr_key: str = "pr_t",
    config: RecoveryConfig,
    flood_onset_t: float,
) -> RecoveryStatus:
    """
    Offline label extraction using the same RecoveryTracker as online RL.
    """
    tracker = RecoveryTracker(
        config=config,
        flood_onset_t=flood_onset_t,
    )

    for row in trajectory:
        tracker.update(
            current_time_s=float(row[time_key]),
            pr_t=float(row[pr_key]),
        )

    return tracker.status


def rolling_mean(
    values: Sequence[float],
    window_size: int,
) -> list[float]:
    """
    Optional causal rolling mean.

    This is provided for future smoothing experiments. It is not enabled by
    default, so Step 3 preserves the Step 1/2 recovery definition.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    output: list[float] = []
    running_sum = 0.0
    queue: list[float] = []

    for value in values:
        queue.append(value)
        running_sum += value

        if len(queue) > window_size:
            running_sum -= queue.pop(0)

        output.append(running_sum / len(queue))

    return output
