
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpochImpact:
    co2_mg: float = 0.0
    hard_braking_events: int = 0
    queue_exposure_veh_s: float = 0.0
    time_loss_delta_s: float = 0.0

    @property
    def co2_kg(self) -> float:
        return self.co2_mg / 1_000_000.0

    @property
    def time_loss_delta_veh_h(self) -> float:
        return self.time_loss_delta_s / 3600.0


class ImpactMonitor:
    """
    Collects emission and safety indicators every SUMO simulation step.

    CO2:
        SUMO vehicle.getCO2Emission() is treated as mg/s and multiplied by
        the simulation step length.

    Hard braking:
        Counts a new event when a vehicle crosses from acceleration above
        the threshold to acceleration <= threshold. This prevents one long
        braking episode from being counted once per second.

    Queue exposure:
        Accumulates vehicle-seconds spent below queue_speed_threshold_mps.

    Time loss:
        Accumulates positive per-vehicle increments of SUMO's cumulative
        vehicle time-loss value.
    """

    def __init__(
        self,
        *,
        hard_braking_threshold_mps2: float = -3.0,
        queue_speed_threshold_mps: float = 0.1,
        step_length_s: float = 1.0,
    ):
        self.hard_braking_threshold_mps2 = float(
            hard_braking_threshold_mps2
        )
        self.queue_speed_threshold_mps = float(
            queue_speed_threshold_mps
        )
        self.step_length_s = float(step_length_s)

        self.total_co2_mg = 0.0
        self.total_hard_braking_events = 0
        self.total_queue_exposure_veh_s = 0.0
        self.total_time_loss_delta_s = 0.0

        self._previous_hard_braking: dict[str, bool] = {}
        self._previous_time_loss_s: dict[str, float] = {}

    def reset(self) -> None:
        self.total_co2_mg = 0.0
        self.total_hard_braking_events = 0
        self.total_queue_exposure_veh_s = 0.0
        self.total_time_loss_delta_s = 0.0
        self._previous_hard_braking.clear()
        self._previous_time_loss_s.clear()

    def collect_step(self, traci_api) -> EpochImpact:
        impact = EpochImpact()
        vehicle_ids = list(traci_api.vehicle.getIDList())
        current_ids = set(vehicle_ids)

        for vehicle_id in vehicle_ids:
            co2_rate_mg_s = max(
                0.0,
                float(
                    traci_api.vehicle.getCO2Emission(vehicle_id)
                ),
            )
            co2_mg = co2_rate_mg_s * self.step_length_s
            impact.co2_mg += co2_mg

            speed = max(
                0.0,
                float(traci_api.vehicle.getSpeed(vehicle_id)),
            )
            if speed <= self.queue_speed_threshold_mps:
                impact.queue_exposure_veh_s += self.step_length_s

            acceleration = float(
                traci_api.vehicle.getAcceleration(vehicle_id)
            )
            braking_now = (
                acceleration
                <= self.hard_braking_threshold_mps2
            )
            braking_before = self._previous_hard_braking.get(
                vehicle_id,
                False,
            )
            if braking_now and not braking_before:
                impact.hard_braking_events += 1
            self._previous_hard_braking[vehicle_id] = braking_now

            current_time_loss = max(
                0.0,
                float(traci_api.vehicle.getTimeLoss(vehicle_id)),
            )
            previous_time_loss = self._previous_time_loss_s.get(
                vehicle_id,
                current_time_loss,
            )
            impact.time_loss_delta_s += max(
                0.0,
                current_time_loss - previous_time_loss,
            )
            self._previous_time_loss_s[
                vehicle_id
            ] = current_time_loss

        # Remove vehicles that have left the network.
        for vehicle_id in list(self._previous_hard_braking):
            if vehicle_id not in current_ids:
                self._previous_hard_braking.pop(
                    vehicle_id,
                    None,
                )
                self._previous_time_loss_s.pop(
                    vehicle_id,
                    None,
                )

        self.total_co2_mg += impact.co2_mg
        self.total_hard_braking_events += (
            impact.hard_braking_events
        )
        self.total_queue_exposure_veh_s += (
            impact.queue_exposure_veh_s
        )
        self.total_time_loss_delta_s += (
            impact.time_loss_delta_s
        )

        return impact

    def totals(self) -> dict[str, float | int]:
        return {
            "co2_total_kg": self.total_co2_mg / 1_000_000.0,
            "hard_braking_events_total": (
                self.total_hard_braking_events
            ),
            "queue_exposure_veh_h": (
                self.total_queue_exposure_veh_s / 3600.0
            ),
            "total_time_loss_veh_h": (
                self.total_time_loss_delta_s / 3600.0
            ),
        }
