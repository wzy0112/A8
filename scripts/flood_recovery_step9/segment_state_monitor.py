from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping

from state_segments import (
    STATE_SEGMENTS,
    get_state_segment_ids,
    validate_state_segments,
)

#之前step8只有global network metrics i谢娜再有
# segment 5指标 mean_speed_mps
# flow_veh_h
# density_veh_km_lane
# vc_ratio
# queue_length_veh
@dataclass(frozen=True)
class SegmentStepSample:
    speed_sum_mps: float
    speed_sample_count: int
    vehicle_count: int
    queue_vehicle_count: int
    entries: int


@dataclass(frozen=True)
class SegmentAggregate:
    """
    Decision-epoch aggregate for one observation segment.

    MQ/SL is intentionally not included.
    """
    mean_speed_mps: float
    flow_veh_h: float
    density_veh_km_lane: float
    vc_ratio: float
    queue_length_veh: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_speed_mps": self.mean_speed_mps,
            "flow_veh_h": self.flow_veh_h,
            "density_veh_km_lane": self.density_veh_km_lane,
            "vc_ratio": self.vc_ratio,
            "queue_length_veh": self.queue_length_veh,
        }


class SegmentStateMonitor:
    """
    Rolling per-segment traffic-state aggregator.

    collect_step() is called once after each SUMO simulationStep().
    snapshot() aggregates the most recent window_s seconds, so a 300-second
    decision period naturally produces a 300-second state aggregation window.

    Flow:
        unique segment entries in the rolling window, annualized to veh/h.

    Density:
        mean vehicles present divided by segment lane-km.

    v/c:
        rolling flow divided by segment capacity.

    Queue length:
        mean number of vehicles with speed <= queue_speed_threshold_mps.

    Capacity:
        Supply calibrated values through capacity_veh_h_by_segment whenever
        available. If omitted, a transparent proxy is used:
            2 * average_lanes_per_directed_edge * capacity_per_lane_veh_h
        because the supplied observation regions contain both directions.
    """

    def __init__(
        self,
        *,
        traci_api,
        net,
        window_s: int = 300,
        queue_speed_threshold_mps: float = 0.1,
        capacity_per_lane_veh_h: float = 1800.0,
        capacity_veh_h_by_segment: Mapping[str, float] | None = None,
    ):
        if window_s <= 0:
            raise ValueError("window_s must be positive.")
        if queue_speed_threshold_mps < 0:
            raise ValueError(
                "queue_speed_threshold_mps cannot be negative."
            )
        if capacity_per_lane_veh_h <= 0:
            raise ValueError(
                "capacity_per_lane_veh_h must be positive."
            )

        self.traci = traci_api
        self.net = net
        self.window_s = int(window_s)
        self.queue_speed_threshold_mps = float(
            queue_speed_threshold_mps
        )
        self.capacity_per_lane_veh_h = float(
            capacity_per_lane_veh_h
        )

        validate_state_segments(self.net)

        self.segment_ids = get_state_segment_ids()

        self._history: dict[str, deque[SegmentStepSample]] = {
            segment_id: deque(maxlen=self.window_s)
            for segment_id in self.segment_ids
        }

        self._previous_vehicle_ids: dict[str, set[str]] = {
            segment_id: set()
            for segment_id in self.segment_ids
        }

        self._lane_km: dict[str, float] = {}
        self._capacity_veh_h: dict[str, float] = {}

        supplied_capacity = dict(
            capacity_veh_h_by_segment or {}
        )

        for segment_id in self.segment_ids:
            segment = STATE_SEGMENTS[segment_id]

            total_lane_m = 0.0

            for edge_id in segment.edge_ids:
                edge = self.net.getEdge(edge_id)
                edge_length_m = float(edge.getLength())
                lane_count = int(edge.getLaneNumber())
                total_lane_m += edge_length_m * lane_count

            lane_km = total_lane_m / 1000.0
            self._lane_km[segment_id] = max(
                lane_km,
                1e-9,
            )

            if segment_id in supplied_capacity:
                capacity = float(
                    supplied_capacity[segment_id]
                )
                if capacity <= 0:
                    raise ValueError(
                        f"Capacity for {segment_id} must be positive."
                    )
            else:
                directed_length_km = max(
                    segment.total_length_m / 1000.0,
                    1e-9,
                )
                average_lanes = (
                    lane_km / directed_length_km
                )
                capacity = (
                    2.0
                    * average_lanes
                    * self.capacity_per_lane_veh_h
                )

            self._capacity_veh_h[segment_id] = max(
                capacity,
                1e-9,
            )

    def reset(self) -> None:
        for history in self._history.values():
            history.clear()

        for segment_id in self.segment_ids:
            self._previous_vehicle_ids[
                segment_id
            ].clear()

    def collect_step(self) -> None:
        """
        Collect one post-simulationStep observation for every state segment.
        """
        for segment_id in self.segment_ids:
            segment = STATE_SEGMENTS[segment_id]

            vehicle_ids: set[str] = set()

            for edge_id in segment.edge_ids:
                try:
                    vehicle_ids.update(
                        self.traci.edge.getLastStepVehicleIDs(
                            edge_id
                        )
                    )
                except Exception:
                    continue

            previous = self._previous_vehicle_ids[
                segment_id
            ]
            entries = len(vehicle_ids - previous)

            speed_sum = 0.0
            speed_count = 0
            queue_count = 0

            for vehicle_id in vehicle_ids:
                try:
                    speed = max(
                        0.0,
                        float(
                            self.traci.vehicle.getSpeed(
                                vehicle_id
                            )
                        ),
                    )
                except Exception:
                    continue

                if math.isfinite(speed):
                    speed_sum += speed
                    speed_count += 1

                    if (
                        speed
                        <= self.queue_speed_threshold_mps
                    ):
                        queue_count += 1

            self._history[segment_id].append(
                SegmentStepSample(
                    speed_sum_mps=speed_sum,
                    speed_sample_count=speed_count,
                    vehicle_count=len(vehicle_ids),
                    queue_vehicle_count=queue_count,
                    entries=entries,
                )
            )

            self._previous_vehicle_ids[
                segment_id
            ] = vehicle_ids

    def snapshot(
        self,
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate the rolling window into one flat-state source dictionary.
        """
        output: dict[str, dict[str, float]] = {}

        for segment_id in self.segment_ids:
            history = self._history[segment_id]

            if not history:
                aggregate = SegmentAggregate(
                    mean_speed_mps=0.0,
                    flow_veh_h=0.0,
                    density_veh_km_lane=0.0,
                    vc_ratio=0.0,
                    queue_length_veh=0.0,
                )
                output[segment_id] = (
                    aggregate.as_dict()
                )
                continue

            speed_sum = sum(
                sample.speed_sum_mps
                for sample in history
            )
            speed_count = sum(
                sample.speed_sample_count
                for sample in history
            )

            mean_speed = (
                speed_sum / speed_count
                if speed_count > 0
                else 0.0
            )

            window_seconds = float(len(history))
            entry_count = sum(
                sample.entries
                for sample in history
            )
            flow_veh_h = (
                entry_count
                * 3600.0
                / window_seconds
            )

            mean_vehicle_count = (
                sum(
                    sample.vehicle_count
                    for sample in history
                )
                / window_seconds
            )

            density = (
                mean_vehicle_count
                / self._lane_km[segment_id]
            )

            capacity = self._capacity_veh_h[
                segment_id
            ]
            vc_ratio = (
                flow_veh_h / capacity
                if capacity > 0
                else 0.0
            )

            mean_queue = (
                sum(
                    sample.queue_vehicle_count
                    for sample in history
                )
                / window_seconds
            )

            aggregate = SegmentAggregate(
                mean_speed_mps=float(mean_speed),
                flow_veh_h=float(flow_veh_h),
                density_veh_km_lane=float(density),
                vc_ratio=float(vc_ratio),
                queue_length_veh=float(mean_queue),
            )

            output[segment_id] = (
                aggregate.as_dict()
            )

        return output

    def capacity_veh_h(
        self,
    ) -> dict[str, float]:
        """Expose the capacity values used for v/c diagnostics."""
        return dict(self._capacity_veh_h)

    def lane_km(self) -> dict[str, float]:
        """Expose segment lane-km for diagnostics."""
        return dict(self._lane_km)
