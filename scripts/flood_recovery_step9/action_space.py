
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

# 定义并解码 MultiDiscrete 动作
# MultiDiscrete management action:
# [tactic, S1-S6 management location, intensity]
class TacticID(IntEnum):
    HOLD = 0
    VSL = 1
    LANE_CONTROL = 2
    TRUCK_BAN = 3
    REROUTE = 4


TACTIC_NAMES = {
    TacticID.HOLD: "hold",
    TacticID.VSL: "vsl",
    TacticID.LANE_CONTROL: "lane_control",
    TacticID.TRUCK_BAN: "truck_ban",
    TacticID.REROUTE: "reroute",
}

# OFF plus three discrete tactic-strength levels. Their physical meaning is tactic-specific.
INTENSITY_NAMES = {
    0: "off",
    1: "low",
    2: "medium",
    3: "high",
}


@dataclass(frozen=True)
class DecodedAction:
    tactic_id: int
    tactic_name: str
    location_index: int
    location_id: str
    intensity_index: int
    intensity_name: str
    operation: str

    def as_dict(self) -> dict:
        return {
            "tactic_id": self.tactic_id,
            "tactic_name": self.tactic_name,
            "location_index": self.location_index,
            "location_id": self.location_id,
            "intensity_index": self.intensity_index,
            "intensity_name": self.intensity_name,
            "operation": self.operation,
        }


class ActionCodec:
    """
    MultiDiscrete action:
        [tactic, location, intensity]

    The location and intensity components are ignored for HOLD.
For all other tactics, intensity 0 means OFF/remove.
    """

    def __init__(self, location_ids: list[str]):
        if not location_ids:
            raise ValueError("At least one candidate location is required.")

        self.location_ids = list(location_ids)
        self.n_tactics = len(TacticID)
        self.n_locations = len(self.location_ids)
        self.n_intensities = len(INTENSITY_NAMES)

        self.nvec = np.array(
            [
                self.n_tactics,
                self.n_locations,
                self.n_intensities,
            ],
            dtype=np.int64,
        )

    def decode(self, action) -> DecodedAction:
        raw = np.asarray(action, dtype=np.int64)

        if raw.shape != (3,):
            raise ValueError(
                f"Action must have shape (3,), received {raw.shape}."
            )

        tactic_index, location_index, intensity_index = (
            int(raw[0]),
            int(raw[1]),
            int(raw[2]),
        )

        if not 0 <= tactic_index < self.n_tactics:
            raise ValueError(f"Invalid tactic index: {tactic_index}")
        if not 0 <= location_index < self.n_locations:
            raise ValueError(f"Invalid location index: {location_index}")
        if not 0 <= intensity_index < self.n_intensities:
            raise ValueError(
                f"Invalid intensity index: {intensity_index}"
            )

        tactic = TacticID(tactic_index)

        if tactic == TacticID.HOLD:
            operation = "hold"
        elif intensity_index == 0:
            operation = "remove"
        else:
            operation = "apply_or_modify"

        return DecodedAction(
            tactic_id=tactic_index,
            tactic_name=TACTIC_NAMES[tactic],
            location_index=location_index,
            location_id=self.location_ids[location_index],
            intensity_index=intensity_index,
            intensity_name=INTENSITY_NAMES[intensity_index],
            operation=operation,
        )

    def hold(self) -> np.ndarray:
        return np.array([0, 0, 0], dtype=np.int64)
