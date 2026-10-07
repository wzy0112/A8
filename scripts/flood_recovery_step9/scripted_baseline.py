from __future__ import annotations

import numpy as np

from action_space import TacticID
from tactic_sites import get_location_ids


# Fixed/scripted response baseline requested for Part 7.
#
# Decision epochs are 1-based:
#   epoch 1: VSL,          S1_b471, medium
#   epoch 2: REROUTE,      S2_b471, low
#   epoch 3: LANE_CONTROL, S3_a8,   low
#   epoch 4+: HOLD
#
# This baseline was defined from the moderate-scenario response strategy.
# For fair generalization comparison it is applied unchanged to every
# evaluation scenario, rather than adapting to the sampled flood severity.


SCRIPTED_POLICY_NAME = "moderate_s1s6_fixed_v1"


def _location_index(location_id: str) -> int:
    location_ids = get_location_ids()

    try:
        return location_ids.index(location_id)
    except ValueError as exc:
        raise ValueError(
            f"Scripted baseline location {location_id!r} is not present "
            f"in current tactic locations: {location_ids}"
        ) from exc


def _action(
    tactic: TacticID,
    location_id: str,
    intensity_index: int,
) -> np.ndarray:
    return np.array(
        [
            int(tactic),
            _location_index(location_id),
            int(intensity_index),
        ],
        dtype=np.int64,
    )


def get_scripted_action(
    *,
    decision_epoch: int,
    scenario_severity: str | None = None,
) -> np.ndarray:
    """
    Return the fixed/scripted baseline action for one decision epoch.

    Parameters
    ----------
    decision_epoch:
        1-based decision epoch number.

    scenario_severity:
        Accepted for logging/API compatibility. The current scripted
        baseline is intentionally fixed and does not adapt its action
        sequence to severity.

    Returns
    -------
    np.ndarray
        MultiDiscrete action in the current format:
        [tactic_index, location_index, intensity_index].

    Notes
    -----
    Current action encoding:
      HOLD=0, VSL=1, LANE_CONTROL=2, TRUCK_BAN=3, REROUTE=4
      intensity: 0=off, 1=low, 2=medium, 3=high
    """

    if decision_epoch < 1:
        raise ValueError(
            "decision_epoch must be 1-based and >= 1."
        )

    if decision_epoch == 1:
        return _action(
            TacticID.VSL,
            "S1_b471",
            2,  # medium
        )

    if decision_epoch == 2:
        return _action(
            TacticID.REROUTE,
            "S2_b471",
            1,  # low
        )

    if decision_epoch == 3:
        return _action(
            TacticID.LANE_CONTROL,
            "S3_a8",
            1,  # low
        )

    # HOLD ignores location/intensity in ActionCodec.
    return np.array(
        [int(TacticID.HOLD), 0, 0],
        dtype=np.int64,
    )
