from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from flood_scenario import FloodMechanism, FloodScenario
from flood_sites import resolve_site_edges


class FloodStage(str, Enum):
    NORMAL = "normal"
    ONSET = "onset"
    PEAK = "peak"
    RECOVERY = "recovery"
    POST_FLOOD = "post_flood"


@dataclass
class LaneBaseState:
    max_speed_mps: float
    allowed: tuple[str, ...]
    disallowed: tuple[str, ...]


@dataclass
class NetworkBaseState:
    lane_states: dict[str, LaneBaseState] = field(default_factory=dict)


class FloodEngine:
    """
    Unified fixed-scenario flood engine.

    The engine:
    - resolves the scenario's edge set once;
    - captures original lane speed and permissions;
    - applies one of the three existing staged flood mechanisms;
    - restores the exact captured base state after flooding.
    """

    def __init__(self, traci_api, net, scenario: FloodScenario):
        self.traci = traci_api
        self.net = net
        self.scenario = scenario
        self.scenario.validate()

        self.edge_ids = resolve_site_edges(
            net=self.net,
            site_id=self.scenario.site_id,
        )
        if not self.edge_ids:
            raise RuntimeError(
                f"No flood edges resolved for {self.scenario.site_id}"
            )

        self.base_state = NetworkBaseState()
        self._capture_base_state()

    def _capture_base_state(self) -> None:
        for edge_id in self.edge_ids:
            lane_count = self.traci.edge.getLaneNumber(edge_id)

            for lane_index in range(lane_count):
                lane_id = f"{edge_id}_{lane_index}"

                self.base_state.lane_states[lane_id] = LaneBaseState(
                    max_speed_mps=float(
                        self.traci.lane.getMaxSpeed(lane_id)
                    ),
                    allowed=tuple(
                        self.traci.lane.getAllowed(lane_id)
                    ),
                    disallowed=tuple(
                        self.traci.lane.getDisallowed(lane_id)
                    ),
                )

    def get_stage(self, current_time: float) -> FloodStage:
        s = self.scenario

        if current_time < s.onset_t:
            return FloodStage.NORMAL
        if current_time < s.peak_start_t:
            return FloodStage.ONSET
        if current_time < s.peak_end_t:
            return FloodStage.PEAK
        if current_time < s.recovery_end_t:
            return FloodStage.RECOVERY
        return FloodStage.POST_FLOOD

    def apply(self, current_time: float) -> FloodStage:
        stage = self.get_stage(current_time)

        if self.scenario.mechanism == FloodMechanism.SPEED_REDUCTION:
            self._apply_speed_reduction(current_time, stage)

        elif self.scenario.mechanism == FloodMechanism.LANE_CLOSURE:
            self._apply_lane_closure(current_time, stage)

        elif self.scenario.mechanism == FloodMechanism.FULL_CLOSURE:
            self._apply_full_closure(current_time, stage)

        else:
            raise ValueError(
                f"Unsupported mechanism: {self.scenario.mechanism}"
            )

        return stage

    def restore_base_state(self) -> None:
        for lane_id, state in self.base_state.lane_states.items():
            self.traci.lane.setMaxSpeed(
                lane_id,
                state.max_speed_mps,
            )
            self.traci.lane.setAllowed(
                lane_id,
                list(state.allowed),
            )
            self.traci.lane.setDisallowed(
                lane_id,
                list(state.disallowed),
            )

    def _set_all_lane_speeds(
        self,
        target_speed_mps: float,
    ) -> None:
        for lane_id, state in self.base_state.lane_states.items():
            speed = min(
                state.max_speed_mps,
                target_speed_mps,
            )
            self.traci.lane.setMaxSpeed(lane_id, speed)

    def _restore_all_lane_speeds(self) -> None:
        for lane_id, state in self.base_state.lane_states.items():
            self.traci.lane.setMaxSpeed(
                lane_id,
                state.max_speed_mps,
            )

    def _close_selected_lanes(
        self,
        closed_lane_indices: tuple[int, ...],
    ) -> None:
        restricted = list(self.scenario.restricted_vclasses)

        for edge_id in self.edge_ids:
            lane_count = self.traci.edge.getLaneNumber(edge_id)

            for lane_index in closed_lane_indices:
                if lane_index >= lane_count:
                    continue

                lane_id = f"{edge_id}_{lane_index}"
                self.traci.lane.setDisallowed(
                    lane_id,
                    restricted,
                )

    def _restore_selected_lanes(
        self,
        closed_lane_indices: tuple[int, ...],
    ) -> None:
        for edge_id in self.edge_ids:
            lane_count = self.traci.edge.getLaneNumber(edge_id)

            for lane_index in closed_lane_indices:
                if lane_index >= lane_count:
                    continue

                lane_id = f"{edge_id}_{lane_index}"
                state = self.base_state.lane_states[lane_id]

                self.traci.lane.setAllowed(
                    lane_id,
                    list(state.allowed),
                )
                self.traci.lane.setDisallowed(
                    lane_id,
                    list(state.disallowed),
                )

    def _close_all_lanes(self) -> None:
        restricted = list(self.scenario.restricted_vclasses)

        for lane_id in self.base_state.lane_states:
            self.traci.lane.setDisallowed(
                lane_id,
                restricted,
            )

    def _restore_all_permissions(self) -> None:
        for lane_id, state in self.base_state.lane_states.items():
            self.traci.lane.setAllowed(
                lane_id,
                list(state.allowed),
            )
            self.traci.lane.setDisallowed(
                lane_id,
                list(state.disallowed),
            )

    def _interpolated_speed(
        self,
        current_time: float,
        start_t: float,
        end_t: float,
        start_speed: float,
        end_speed: float,
    ) -> float:
        if end_t <= start_t:
            return end_speed

        ratio = (current_time - start_t) / (end_t - start_t)
        ratio = max(0.0, min(1.0, ratio))

        return start_speed + ratio * (end_speed - start_speed)

    def _representative_base_speed(self) -> float:
        speeds = [
            state.max_speed_mps
            for state in self.base_state.lane_states.values()
        ]
        return max(speeds)

    def _apply_speed_reduction(
        self,
        current_time: float,
        stage: FloodStage,
    ) -> None:
        base_speed = self._representative_base_speed()
        peak_speed = float(self.scenario.peak_speed_mps)

        if stage == FloodStage.NORMAL:
            self.restore_base_state()

        elif stage == FloodStage.ONSET:
            speed = self._interpolated_speed(
                current_time,
                self.scenario.onset_t,
                self.scenario.peak_start_t,
                base_speed,
                peak_speed,
            )
            self._set_all_lane_speeds(speed)

        elif stage == FloodStage.PEAK:
            self._set_all_lane_speeds(peak_speed)

        elif stage == FloodStage.RECOVERY:
            speed = self._interpolated_speed(
                current_time,
                self.scenario.peak_end_t,
                self.scenario.recovery_end_t,
                peak_speed,
                base_speed,
            )
            self._set_all_lane_speeds(speed)

        else:
            self.restore_base_state()

    def _apply_lane_closure(
        self,
        current_time: float,
        stage: FloodStage,
    ) -> None:
        base_speed = self._representative_base_speed()
        peak_speed = float(self.scenario.peak_speed_mps)
        closed = self.scenario.closed_lane_indices

        if stage == FloodStage.NORMAL:
            self.restore_base_state()

        elif stage == FloodStage.ONSET:
            self._close_selected_lanes(closed)
            speed = self._interpolated_speed(
                current_time,
                self.scenario.onset_t,
                self.scenario.peak_start_t,
                base_speed,
                peak_speed,
            )
            self._set_open_lane_speeds(speed, closed)

        elif stage == FloodStage.PEAK:
            self._close_selected_lanes(closed)
            self._set_open_lane_speeds(peak_speed, closed)

        elif stage == FloodStage.RECOVERY:
            # Preserve the original scripted behavior:
            # closed lanes remain closed until recovery_end_t while the
            # open-lane speed gradually returns to normal.
            self._close_selected_lanes(closed)
            speed = self._interpolated_speed(
                current_time,
                self.scenario.peak_end_t,
                self.scenario.recovery_end_t,
                peak_speed,
                base_speed,
            )
            self._set_open_lane_speeds(speed, closed)

        else:
            self.restore_base_state()

    def _set_open_lane_speeds(
        self,
        target_speed_mps: float,
        closed_lane_indices: tuple[int, ...],
    ) -> None:
        closed_set = set(closed_lane_indices)

        for edge_id in self.edge_ids:
            lane_count = self.traci.edge.getLaneNumber(edge_id)

            for lane_index in range(lane_count):
                if lane_index in closed_set:
                    continue

                lane_id = f"{edge_id}_{lane_index}"
                base_speed = (
                    self.base_state.lane_states[lane_id].max_speed_mps
                )
                self.traci.lane.setMaxSpeed(
                    lane_id,
                    min(base_speed, target_speed_mps),
                )

    def _apply_full_closure(
        self,
        current_time: float,
        stage: FloodStage,
    ) -> None:
        base_speed = self._representative_base_speed()
        approach_speed = float(self.scenario.peak_speed_mps)

        if stage == FloodStage.NORMAL:
            self.restore_base_state()

        elif stage == FloodStage.ONSET:
            self._restore_all_permissions()
            speed = self._interpolated_speed(
                current_time,
                self.scenario.onset_t,
                self.scenario.peak_start_t,
                base_speed,
                approach_speed,
            )
            self._set_all_lane_speeds(speed)

        elif stage == FloodStage.PEAK:
            self._set_all_lane_speeds(approach_speed)
            self._close_all_lanes()

        elif stage == FloodStage.RECOVERY:
            self._restore_all_permissions()
            speed = self._interpolated_speed(
                current_time,
                self.scenario.peak_end_t,
                self.scenario.recovery_end_t,
                approach_speed,
                base_speed,
            )
            self._set_all_lane_speeds(speed)

        else:
            self.restore_base_state()
