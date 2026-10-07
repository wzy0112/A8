
from __future__ import annotations

import math

from dataclasses import dataclass

from action_space import DecodedAction, TacticID
from tactic_sites import get_location_ids, resolve_tactic_site_edges


@dataclass(frozen=True)
class LaneState:
    max_speed_mps: float
    allowed: tuple[str, ...]
    disallowed: tuple[str, ...]

@dataclass
class MeasureUpdateResult:
    action: DecodedAction
    operation: str
    changed: bool
    existed_before: bool

@dataclass
class TacticApplicationResult:
    action: DecodedAction
    changed_lanes: int = 0
    rerouted_vehicles: int = 0
    changed_edges: int = 0
    invalid_reason: str | None = None

    @property
    def applied(self) -> bool:
        return self.invalid_reason is None

    def as_dict(self) -> dict:
        return {
            **self.action.as_dict(),
            "changed_lanes": self.changed_lanes,
            "changed_edges": self.changed_edges,
            "rerouted_vehicles": self.rerouted_vehicles,
            "applied": self.applied,
            "invalid_reason": self.invalid_reason,
        }


class TacticManager:
    """
       Maintains a persistent set of traffic-management measures.

       Decision semantics:
           HOLD:
               keep all currently active measures unchanged.

           intensity == 0:
               remove the selected tactic at the selected location.

           intensity > 0:
               activate a new measure or modify the intensity of an
               already-active measure.

       Simulation ordering:
           1. Remove the physical effects of the previous simulation-second
              management layer, while preserving active_measures.
           2. FloodEngine reconstructs the physical flood state.
           3. TacticManager layers every persistent active management measure
              on top of the physical flood state.
           4. SUMO advances one simulation step.

       This allows one new command per decision epoch (e.g. every 300 s), while
       measures persist across later epochs. Consecutive decisions therefore
       build a spatio-temporal tactic chain and approximate simultaneous tactics.
       """

    VSL_SPEEDS_MPS = {
        1: 27.78,  # 100 km/h
        2: 22.22,  # 80 km/h
        3: 16.67,  # 60 km/h
    }

    TRUCK_BAN_CLASSES = {
        1: ("truck",),
        2: ("truck", "trailer"),
        3: ("truck", "trailer", "delivery"),
    }

    # Management lane-closure strength.
    # The value indicates how many lanes per edge the RL management layer
    # attempts to additionally restrict.
    LANE_CONTROL_COUNT = {
        1: 1,
        2: 2,
        3: None,  # all lanes
    }

    # Reroute intensity now means routing-bias strength, not rerouting frequency.
    REROUTE_STRENGTH = {
        1: 1.25,
        2: 1.50,
        3: 2.00,
    }

    def __init__(self, traci_api, net):
        self.traci = traci_api
        self.net = net

        self.location_ids = get_location_ids()
        self.site_edges = {
            location_id: resolve_tactic_site_edges(
                net,
                location_id,
            )
            for location_id in self.location_ids
        }

        all_edges = list(
            dict.fromkeys(
                edge_id
                for edge_ids in self.site_edges.values()
                for edge_id in edge_ids
            )
        )

        self.base_lane_state: dict[str, LaneState] = {}
        self.edge_lanes: dict[str, list[str]] = {}
        self._capture_base_state(all_edges)

        self.active_measures: dict[
            tuple[int, str],
            DecodedAction,
        ] = {}



    def _capture_base_state(self, edge_ids: list[str]) -> None:
        for edge_id in edge_ids:
            lane_ids: list[str] = []
            lane_count = self.traci.edge.getLaneNumber(edge_id)

            for lane_index in range(lane_count):
                lane_id = f"{edge_id}_{lane_index}"
                lane_ids.append(lane_id)
                self.base_lane_state[lane_id] = LaneState(
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

            self.edge_lanes[edge_id] = lane_ids

    def set_action(
            self,
            action: DecodedAction,
    ) -> MeasureUpdateResult:

        tactic = TacticID(action.tactic_id)

        # HOLD: do not change persistent management state.
        if tactic == TacticID.HOLD:
            return MeasureUpdateResult(
                action=action,
                operation="hold",
                changed=False,
                existed_before=False,
            )

        key = (
            action.tactic_id,
            action.location_id,
        )

        previous = self.active_measures.get(key)

        # intensity 0 = targeted remove
        if action.intensity_index == 0:
            existed = previous is not None

            if existed:
                self.active_measures.pop(
                    key,
                    None,
                )

            return MeasureUpdateResult(
                action=action,
                operation="remove",
                changed=existed,
                existed_before=existed,
            )

        # intensity > 0 = activate or modify
        self.active_measures[key] = action

        return MeasureUpdateResult(
            action=action,
            operation=(
                "activate"
                if previous is None
                else "modify"
            ),
            changed=(
                    previous is None
                    or previous.intensity_index
                    != action.intensity_index
            ),
            existed_before=(
                    previous is not None
            ),
        )

    def restore_management_base(self) -> None:
        """
        Remove physical effects left by the previous management layer.

        Persistent active_measures are intentionally preserved.

        FloodRecoveryEnv must call this immediately before FloodEngine.apply()
        on every simulation second. FloodEngine then reconstructs the current
        physical flood state before persistent management measures are layered
        again.
        """
        for lane_id, base in self.base_lane_state.items():
            self.traci.lane.setMaxSpeed(
                lane_id,
                base.max_speed_mps,
            )
            self.traci.lane.setAllowed(
                lane_id,
                list(base.allowed),
            )
            self.traci.lane.setDisallowed(
                lane_id,
                list(base.disallowed),
            )


    def clear(self) -> None:
        """
        Fully clear the management layer.

        Intended for episode reset/close, not for ordinary decision epochs.
        """
        self.restore_management_base()
        self.active_measures.clear()


    def apply_active(
            self,
            current_time_s: float,
    ) -> list[TacticApplicationResult]:
        """
        Apply every persistent management measure currently active.

        HOLD is never stored in active_measures, and intensity 0 actions remove
        measures in set_action(), so only positive-intensity measures should
        appear here.
        """
        results: list[TacticApplicationResult] = []

        for key in sorted(self.active_measures):
            action = self.active_measures[key]
            tactic = TacticID(action.tactic_id)

            if action.intensity_index <= 0:
                results.append(
                    TacticApplicationResult(
                        action=action,
                        invalid_reason=(
                            "Persistent active measure has non-positive "
                            "intensity."
                        ),
                    )
                )
                continue

            edge_ids = self.site_edges[action.location_id]

            if tactic == TacticID.VSL:
                result = self._apply_vsl(
                    action,
                    edge_ids,
                )

            elif tactic == TacticID.LANE_CONTROL:
                result = self._apply_lane_control(
                    action,
                    edge_ids,
                )

            elif tactic == TacticID.TRUCK_BAN:
                result = self._apply_truck_ban(
                    action,
                    edge_ids,
                )

            elif tactic == TacticID.REROUTE:
                result = self._apply_reroute(
                    action,
                    edge_ids,
                    current_time_s,
                )

            else:
                result = TacticApplicationResult(
                    action=action,
                    invalid_reason=(
                        f"Unsupported tactic id "
                        f"{action.tactic_id}"
                    ),
                )

            results.append(result)

        return results

    def _apply_vsl(
        self,
        action: DecodedAction,
        edge_ids: list[str],
    ) -> TacticApplicationResult:

        if action.intensity_index not in self.VSL_SPEEDS_MPS:
            return TacticApplicationResult(
                action=action,
                invalid_reason=(
                    f"Invalid VSL intensity "
                    f"{action.intensity_index}"
                ),
            )

        target_speed = self.VSL_SPEEDS_MPS[
            action.intensity_index
        ]
        changed = 0

        for edge_id in edge_ids:
            for lane_id in self.edge_lanes[edge_id]:
                current_speed = float(
                    self.traci.lane.getMaxSpeed(lane_id)
                )
                new_speed = min(current_speed, target_speed)
                self.traci.lane.setMaxSpeed(lane_id, new_speed)
                changed += 1

        return TacticApplicationResult(
            action=action,
            changed_lanes=changed,
        )

    def _apply_lane_control(
            self,
            action: DecodedAction,
            edge_ids: list[str],
    ) -> TacticApplicationResult:
        """
        Apply an additional RL management lane restriction.

        This function NEVER restores base permissions. It only adds restrictions
        to the permissions already produced by FloodEngine.

        Therefore:
            - RL can close additional lanes;
            - lowering the persistent intensity across later epochs represents
              phased reopening of RL-controlled lanes;
            - intensity 0 removes the management measure;
            - RL cannot reopen a lane that FloodEngine physically keeps closed.
        """
        if action.intensity_index not in self.LANE_CONTROL_COUNT:
            return TacticApplicationResult(
                action=action,
                invalid_reason=(
                    f"Invalid lane-control intensity "
                    f"{action.intensity_index}"
                ),
            )

        requested_count = self.LANE_CONTROL_COUNT[
            action.intensity_index
        ]

        changed = 0

        for edge_id in edge_ids:
            lane_ids = self.edge_lanes[edge_id]

            if requested_count is None:
                target_lanes = lane_ids
            else:
                target_lanes = lane_ids[
                               :min(requested_count, len(lane_ids))
                               ]

            for lane_id in target_lanes:
                current_disallowed = set(
                    self.traci.lane.getDisallowed(
                        lane_id
                    )
                )

                current_disallowed.add("all")

                self.traci.lane.setDisallowed(
                    lane_id,
                    sorted(current_disallowed),
                )

                changed += 1

        return TacticApplicationResult(
            action=action,
            changed_lanes=changed,
        )


    def _apply_truck_ban(
        self,
        action: DecodedAction,
        edge_ids: list[str],
    ) -> TacticApplicationResult:

        if action.intensity_index not in self.TRUCK_BAN_CLASSES:
            return TacticApplicationResult(
                action=action,
                invalid_reason=(
                    f"Invalid truck-ban intensity "
                    f"{action.intensity_index}"
                ),
            )

        banned = set(
            self.TRUCK_BAN_CLASSES[action.intensity_index]
        )
        changed = 0

        for edge_id in edge_ids:
            for lane_id in self.edge_lanes[edge_id]:
                current_disallowed = set(
                    self.traci.lane.getDisallowed(lane_id)
                )
                current_disallowed.update(banned)
                self.traci.lane.setDisallowed(
                    lane_id,
                    sorted(current_disallowed),
                )
                changed += 1

        return TacticApplicationResult(
            action=action,
            changed_lanes=changed,
        )

    def _apply_reroute(
            self,
            action: DecodedAction,
            edge_ids: list[str],
            current_time_s: float,
    ) -> TacticApplicationResult:
        """
        Apply a routing-cost bias to the selected edges.

        Intensity controls bias strength rather than rerouting frequency.
        The existing SUMO rerouting mechanism can then bias routes away from
        higher-effort edges.
        """
        if action.intensity_index not in self.REROUTE_STRENGTH:
            return TacticApplicationResult(
                action=action,
                invalid_reason=(
                    f"Invalid reroute intensity "
                    f"{action.intensity_index}"
                ),
            )

        if not hasattr(self.traci.edge, "setEffort"):
            return TacticApplicationResult(
                action=action,
                invalid_reason=(
                    "This SUMO/TraCI build does not expose "
                    "edge.setEffort()."
                ),
            )

        multiplier = self.REROUTE_STRENGTH[
            action.intensity_index
        ]

        changed_edges = 0

        for edge_id in edge_ids:
            try:
                travel_time = float(
                    self.traci.edge.getTraveltime(
                        edge_id
                    )
                )

                if not math.isfinite(travel_time):
                    continue

                travel_time = max(
                    1.0,
                    travel_time,
                )

                effort = (
                        travel_time
                        * multiplier
                )

                self.traci.edge.setEffort(
                    edge_id,
                    effort,
                    begin=current_time_s,
                    end=current_time_s + 1.0,
                )

                changed_edges += 1

            except Exception:
                continue

        return TacticApplicationResult(
            action=action,
            changed_edges=changed_edges,
        )

    def active_measures_snapshot(
            self,
    ) -> list[dict]:
        """
        Return the current persistent management state in a deterministic order.
        """
        snapshot = [
            action.as_dict()
            for action in self.active_measures.values()
        ]

        return sorted(
            snapshot,
            key=lambda item: (
                item["location_id"],
                item["tactic_id"],
            ),
        )
