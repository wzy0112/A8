from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal
# flood scenarios除了severity site mechnism 还有time意外还需要demand phase


Severity = Literal["light", "moderate", "severe"]

DemandPhase = Literal["non_peak", "peak"]

class FloodMechanism(str, Enum):
    SPEED_REDUCTION = "speed_reduction"
    LANE_CLOSURE = "lane_closure"
    FULL_CLOSURE = "full_closure"


@dataclass(frozen=True)
class FloodScenario:
    scenario_id: str
    severity: Severity
    site_id: str
    mechanism: FloodMechanism

    onset_t: int
    peak_start_t: int
    peak_end_t: int
    recovery_end_t: int

    demand_phase: DemandPhase = "non_peak"

    max_sim_time_t: int = 12000

    peak_speed_mps: float | None = None
    closed_lane_indices: tuple[int, ...] = ()
    restricted_vclasses: tuple[str, ...] = (
        "passenger",
        "truck",
        "bus",
    )

    def validate(self) -> None:
        times = (
            self.onset_t,
            self.peak_start_t,
            self.peak_end_t,
            self.recovery_end_t,
            self.max_sim_time_t,
        )
        if list(times) != sorted(times):
            raise ValueError(
                f"Invalid scenario timeline for {self.scenario_id}: {times}"
            )

        if self.mechanism in {
            FloodMechanism.SPEED_REDUCTION,
            FloodMechanism.LANE_CLOSURE,
            FloodMechanism.FULL_CLOSURE,
        } and self.peak_speed_mps is None:
            raise ValueError(
                f"{self.scenario_id} requires peak_speed_mps."
            )

        if (
            self.mechanism == FloodMechanism.LANE_CLOSURE
            and not self.closed_lane_indices
        ):
            raise ValueError(
                f"{self.scenario_id} requires closed_lane_indices."
            )

    @property
    def onset_duration_s(self) -> int:
        return self.peak_start_t - self.onset_t

    @property
    def peak_duration_s(self) -> int:
        return self.peak_end_t - self.peak_start_t

    @property
    def recovery_duration_s(self) -> int:
        return self.recovery_end_t - self.peak_end_t


FIXED_SCENARIOS: dict[str, FloodScenario] = {
    "light": FloodScenario(
        scenario_id="light_b471_speed_reduction_v1",
        severity="light",
        site_id="b471_groebenried_dachauer_moos",
        mechanism=FloodMechanism.SPEED_REDUCTION,
        onset_t=1200,
        peak_start_t=2130,
        peak_end_t=2730,
        recovery_end_t=3050,
        peak_speed_mps=8.33,
    ),
    "moderate": FloodScenario(
        scenario_id="moderate_a8_lane_closure_v1",
        severity="moderate",
        site_id="a8_sulzemoos_dachau",
        mechanism=FloodMechanism.LANE_CLOSURE,
        onset_t=1200,
        peak_start_t=5530,
        peak_end_t=6130,
        recovery_end_t=10200,
        peak_speed_mps=8.33,
        closed_lane_indices=(0, 1),
    ),
    "severe": FloodScenario(
        scenario_id="severe_a8_b471_full_closure_v1",
        severity="severe",
        site_id="a8_b471_combined",
        mechanism=FloodMechanism.FULL_CLOSURE,
        onset_t=1200,
        peak_start_t=4750,
        peak_end_t=5350,
        recovery_end_t=11140,
        peak_speed_mps=5.56,
    ),
}


for _scenario in FIXED_SCENARIOS.values():
    _scenario.validate()


def get_fixed_scenario(name: str) -> FloodScenario:
    key = name.lower().strip()
    try:
        return FIXED_SCENARIOS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(FIXED_SCENARIOS))
        raise ValueError(
            f"Unknown scenario {name!r}. Valid values: {valid}"
        ) from exc
