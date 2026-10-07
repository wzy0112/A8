from __future__ import annotations

# Fixed forecast horizons used by the first surrogate baseline.
# They are multiples of the current 300-s decision epoch.
FORECAST_HORIZONS_S: tuple[int, ...] = (
    300,
    600,
    900,
    1200,
    1800,
)

# Current management library (HOLD is not a persistent measure).
TACTICS: tuple[str, ...] = (
    "vsl",
    "lane_control",
    "truck_ban",
    "reroute",
)

# Must match the current ActionCodec / tactic_sites ordering.
# LOCATIONS: tuple[str, ...] = (
#     "b471_corridor",
#     "a8_corridor",
#     "a8_b471_combined",
# )

LOCATIONS: tuple[str, ...] = (
    "S1_b471",
    "S2_b471",
    "S3_a8",
    "S4_St2054",
    "S5_a8",
    "S6_St2051",
)

FLOOD_STAGES: tuple[str, ...] = (
    "normal",
    "onset",
    "peak",
    "recovery",
)

# Timing bucket for future tactic-chain encoding, relative to the current
# decision epoch. The returned bucket is 0 when the slot is absent.
TACTIC_TIMING_BUCKET_EDGES_S: tuple[int, ...] = (
    300,
    600,
    900,
    1200,
    1800,
)

STATE_SCHEMA_VERSION = "part3_state_v1"
SURROGATE_DATASET_VERSION = "surrogate_dataset_v1"
