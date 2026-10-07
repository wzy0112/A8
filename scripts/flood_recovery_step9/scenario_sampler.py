from __future__ import annotations

from dataclasses import replace
import random

from flood_scenario import FloodScenario, get_fixed_scenario
from randomization_config import ScenarioRandomizationConfig


SEVERITIES = ("light", "moderate", "severe")


class FloodScenarioSampler:
    """
    Samples either validated fixed scenarios or continuous perturbations.

    Modes:
        fixed_moderate
        mixed_fixed
        randomized
    """

    def __init__(
        self,
        *,
        mode: str,
        config: ScenarioRandomizationConfig | None = None,
        severity_weights: tuple[float, float, float] = (
            1.0,
            1.0,
            1.0,
        ),
    ):
        valid_modes = {
            "fixed_moderate",
            "mixed_fixed",
            "randomized",
        }
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown sampling mode {mode!r}. "
                f"Valid: {sorted(valid_modes)}"
            )

        if len(severity_weights) != 3:
            raise ValueError(
                "severity_weights must contain three values."
            )
        if sum(severity_weights) <= 0:
            raise ValueError(
                "severity_weights must have a positive sum."
            )

        self.mode = mode
        self.config = (
            config or ScenarioRandomizationConfig()
        )
        self.config.validate()
        self.severity_weights = tuple(
            float(value)
            for value in severity_weights
        )

    def sample(
        self,
        rng: random.Random,
        *,
        episode_index: int,
    ) -> FloodScenario:
        if self.mode == "fixed_moderate":
            return get_fixed_scenario("moderate")

        severity = rng.choices(
            population=SEVERITIES,
            weights=self.severity_weights,
            k=1,
        )[0]
        base = get_fixed_scenario(severity)

        if self.mode == "mixed_fixed":
            return base

        return self._perturb(
            base=base,
            rng=rng,
            episode_index=episode_index,
        )

    def _scaled_duration(
        self,
        *,
        base_duration: int,
        lower: float,
        upper: float,
        rng: random.Random,
    ) -> int:
        value = int(
            round(base_duration * rng.uniform(lower, upper))
        )
        return max(
            self.config.minimum_stage_duration_s,
            value,
        )

    def _sample_onset(
            self,
            rng: random.Random,
    ) -> tuple[int, str]:
        if rng.random() < self.config.non_peak_probability:
            onset_t = rng.randint(
                self.config.non_peak_onset_min_s,
                self.config.non_peak_onset_max_s,
            )
            demand_phase = "non_peak"
        else:
            onset_t = rng.randint(
                self.config.peak_onset_min_s,
                self.config.peak_onset_max_s,
            )
            demand_phase = "peak"

        return onset_t, demand_phase

    def _perturb(
        self,
        *,
        base: FloodScenario,
        rng: random.Random,
        episode_index: int,
    ) -> FloodScenario:
        onset_t, demand_phase = self._sample_onset(rng)

        onset_duration = self._scaled_duration(
            base_duration=base.onset_duration_s,
            lower=self.config.onset_duration_scale_min,
            upper=self.config.onset_duration_scale_max,
            rng=rng,
        )
        peak_duration = self._scaled_duration(
            base_duration=base.peak_duration_s,
            lower=self.config.peak_duration_scale_min,
            upper=self.config.peak_duration_scale_max,
            rng=rng,
        )
        recovery_duration = self._scaled_duration(
            base_duration=base.recovery_duration_s,
            lower=self.config.recovery_duration_scale_min,
            upper=self.config.recovery_duration_scale_max,
            rng=rng,
        )

        peak_start_t = onset_t + onset_duration
        peak_end_t = peak_start_t + peak_duration
        recovery_end_t = min(
            peak_end_t + recovery_duration,
            self.config.maximum_recovery_end_t,
        )

        # Clipping may shorten recovery; keep at least one stage interval.
        recovery_end_t = max(
            peak_end_t
            + self.config.minimum_stage_duration_s,
            recovery_end_t,
        )
        recovery_end_t = min(
            recovery_end_t,
            base.max_sim_time_t - 1,
        )

        peak_speed_mps = base.peak_speed_mps
        if peak_speed_mps is not None:
            peak_speed_mps = max(
                1.0,
                peak_speed_mps
                * rng.uniform(
                    self.config.speed_scale_min,
                    self.config.speed_scale_max,
                ),
            )

        # Stronger per-sample identifier.
        # Generated after the physical scenario parameters so it does not
        # change the random values of the current episode.
        sample_token = rng.getrandbits(32)

        scenario = replace(
            base,
            scenario_id=(
                f"{base.severity}_randomized_"
                f"e{episode_index:06d}_"
                f"r{sample_token:08x}"
            ),
            onset_t=onset_t,
            demand_phase=demand_phase,
            peak_start_t=peak_start_t,
            peak_end_t=peak_end_t,
            recovery_end_t=recovery_end_t,
            peak_speed_mps=peak_speed_mps,
        )
        scenario.validate()
        return scenario
