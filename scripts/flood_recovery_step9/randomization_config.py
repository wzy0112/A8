
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioRandomizationConfig:
    """
    Randomization around real-world-grounded flood templates.

    Severity, flood site and flood mechanism remain coupled exactly as in
    the validated fixed scenarios.

    Episode variability comes from:
    - onset timing / demand phase;
    - SUMO stochastic seed;
    - +/- 5% peak-duration variation;
    - +/- 5% flood-intensity variation.
    """

    # Demand phases encoded through flood onset time.

    non_peak_onset_min_s: int = 600
    non_peak_onset_max_s: int = 800

    peak_onset_min_s: int = 1800
    peak_onset_max_s: int = 2000

    # Equal probability unless changed later.
    non_peak_probability: float = 0.50

    # Only peak duration varies by +/- 5%.
    # Onset and recovery durations remain fixed.
    onset_duration_scale_min: float = 1.00 #0.95
    onset_duration_scale_max: float = 1.00 #1.05

    peak_duration_scale_min: float = 0.95
    peak_duration_scale_max: float = 1.05

    recovery_duration_scale_min: float = 1.00 #0.95
    recovery_duration_scale_max: float = 1.00 #1.05

    # +/- 5% flood-intensity uncertainty.
    speed_scale_min: float = 0.95
    speed_scale_max: float = 1.05

    minimum_stage_duration_s: int = 60
    # 把 maximum_recovery_end_t 从 11500 放宽到 11900，是因为你的 severe template 原本 recovery_end 就已经是 11140；当 onset 变晚且 duration 又允许 +5% 时，比较容易碰到 11500 截断。固定 max_sim_time_t=12000 的情况下留 100 秒缓冲更自然。原始 severe 时序确实已经到 11140 s
    # Worst-case randomized severe recovery_end_t is about 11970 s.
    # Keep the simulation horizon at 12000 s.
    maximum_recovery_end_t: int = 12000

    def validate(self) -> None:
        if not (
                0 <= self.non_peak_onset_min_s
                <= self.non_peak_onset_max_s
                < 1800
        ):
            raise ValueError(
                "Non-peak onset interval must lie below 1800 s."
            )

        if not (
                1800
                <= self.peak_onset_min_s
                <= self.peak_onset_max_s
                < 3600
        ):
            raise ValueError(
                "Peak onset interval must lie in 1800–3600 s."
            )

        if not 0.0 <= self.non_peak_probability <= 1.0:
            raise ValueError(
                "non_peak_probability must lie in [0, 1]."
            )

        scale_pairs = [
            (
                self.onset_duration_scale_min,
                self.onset_duration_scale_max,
            ),
            (
                self.peak_duration_scale_min,
                self.peak_duration_scale_max,
            ),
            (
                self.recovery_duration_scale_min,
                self.recovery_duration_scale_max,
            ),
            (
                self.speed_scale_min,
                self.speed_scale_max,
            ),
        ]

        for lower, upper in scale_pairs:
            if lower <= 0 or lower > upper:
                raise ValueError(
                    f"Invalid scale interval: {lower}, {upper}"
                )

        if self.minimum_stage_duration_s <= 0:
            raise ValueError(
                "minimum_stage_duration_s must be positive."
            )
