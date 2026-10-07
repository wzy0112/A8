from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# 固定 State Space 的空间划分，并保证 RL 和未来 T3.6 永远使用同一组 6 segments。

@dataclass(frozen=True)
class StateSegment:
    """
    Fixed observation segment used by the Part-3 flat state vector.

    edge_ids:
        Exact directed SUMO edges from observation_regions_edges(1).csv.

    total_length_m:
        Sum of the directed edge lengths in the attachment. This is used only
        as a geometry reference; SegmentStateMonitor derives lane-km directly
        from the loaded SUMO network.
    """

    segment_id: str
    description: str
    edge_ids: tuple[str, ...]
    total_length_m: float


STATE_SEGMENTS: dict[str, StateSegment] = {
    "S1_b471": StateSegment(
        segment_id="S1_b471",
        description="B471 observation region S1.",
        edge_ids=(
            "372647238",
            "372647235",
            "372647241",
            "101081591",
            "101081608",
            "372647240",
            "563798645",
            "372647239",
            "563798644",
            "372647237",
            "372648087",
            "511296353",
            "-511296353",
            "-519402586",
            "-372647237",
            "-563798644",
            "-372647239",
            "-519366831",
            "-372647240",
            "-101081608",
            "-101081591",
            "-372647241",
            "-372647235",
            "-372647238",
        ),
        total_length_m=5598.51,
    ),
    "S2_b471": StateSegment(
        segment_id="S2_b471",
        description="B471 observation region S2.",
        edge_ids=(
            "32234550",
            "32473364",
            "4267816",
            "4268401",
            "493361413",
            "31672145",
            "519348430#1",
            "519348425#2",
            "493360265",
            "493360261",
            "493360263",
            "493360262",
            "493360266#0",
            "-493360266#0",
            "-493360262",
            "-493360263",
            "-493360261",
            "-493360265",
            "-519348425#3",
            "-519348425#1",
            "-519348430#0",
            "-493361413",
            "4267817",
            "520069748",
            "-519348437",
        ),
        total_length_m=6832.64,
    ),
    "S3_a8": StateSegment(
        segment_id="S3_a8",
        description="A8 observation region S3.",
        edge_ids=(
            "327464676",
            "251505089",
            "251505092",
            "830912806",
            "327462610",
            "251505085",
            "327462614",
            "327464675",
            "251505081",
        ),
        total_length_m=8623.63,
    ),
    "S4_St2054": StateSegment(
        segment_id="S4_St2054",
        description="St2054 / junction observation region S4.",
        edge_ids=(
            "-1057789612#0",
            "-25664390#10",
            "-1057789612#2",
            "-25664390#9",
            "-1057789612#3",
            "-25664390#8",
            "-552416002#0",
            "-25664390#7",
            "-552416002#2",
            "552416002#2",
            "-25664390#6",
            "-552416002#3",
            "-25664390#5",
            "25664390#5",
            "-552416002#4",
            "-25664390#4",
            "-552416002#6",
            "25664390#2",
            "-25664390#2",
            "552416002#7",
            "-552416002#7",
            "25664390#1",
            "-25664390#1",
            "552416002#8",
            "-552416002#8",
            "-25664390#0",
            "25664304#3",
            "-552416002#10",
            "552416002#10",
            "25664304#1",
            "-25664304#1",
            "552416002#11",
            "-552416002#11",
            "-25664304#0",
            "-552416002#12",
            "552416002#12",
            "4270949",
            "-552416002#14",
            "38517402#0",
            "-24543563#0",
            "38517402#1",
            "24543563#1",
            "-24543563#1",
            "38517402#2",
            "-24543564",
            "38517402#3",
            "199481929#1",
            "38517402#5",
            "199481929#3",
            "38517402#7",
            "-199481928",
            "-38517402#8",
            "38517402#8",
            "-256262777#0",
            "-1284230708",
            "1284230708",
            "256262777#1",
            "-256262777#1",
            "224243811#2",
            "-224243811#2",
            "256262777#2",
            "-256262777#2",
            "224243811#1",
            "-224243811#1",
            "256262777#3",
            "-256262777#3",
            "84207160#3",
            "-224243811#0",
            "-256262777#4",
            "256262777#4",
            "-84207160#2",
            "84207160#2",
            "256262777#5",
            "-256262777#5",
            "-84207160#1",
            "84207160#1",
            "256262777#6",
            "-256262777#6",
            "-84207160#0",
            "-256262777#7",
            "256262777#7",
            "-1377730276#1",
            "256262777#8",
            "-256262777#8",
            "-1377730276#0",
            "1377730276#0",
            "97262150#0",
            "-97262150#0",
            "-1377730277",
            "-97262150#1",
            "97262150#1",
            "-25016277#2",
            "-97262150#2",
            "97262150#2",
            "-25016277#1",
            "97262152#0",
            "-97262152#0",
            "-25016277#0",
            "-97262152#1",
            "97262152#1",
            "25016278#4",
            "-97262152#2",
            "97262152#2",
            "25016278#1",
            "97262152#3",
            "-27803378",
            "25016278#2",
            "-97262147",
            "854407862#1",
            "-854407862#1",
            "854407862#2",
            "-854407862#2",
            "-854407862#4",
            "854407862#4",
            "854407862#5",
            "-854407862#5",
            "854407862#6",
            "-854407862#6",
            "854407862#7",
            "-854407862#7",
            "-854407862#8",
            "854407862#8",
            "-1393378776#0",
            "1393378776#0",
            "1393378776#1",
            "-1393378776#1",
            "-1393378776#2",
            "1393378776#2",
            "-1393378776#3",
            "1393378776#3",
            "1393378776#4",
            "1393378776#6",
            "-1393378776#6",
            "-1393378776#7",
            "-494708203#0",
            "-494708203#1",
            "-494708203#3",
            "-494708203#4",
            "494708203#4",
        ),
        total_length_m=10641.85,
    ),
    "S5_a8": StateSegment(
        segment_id="S5_a8",
        description="A8 observation region S5.",
        edge_ids=(
            "-182556971",
            "200614327",
            "200614326",
            "-327473093",
            "184627122#2",
            "184627123#0",
            "-327473095",
            "184627122#1",
            "-184627122#1",
            "184627123#1",
            "10901655",
            "184627122#0",
            "-184627122#0",
            "280421287",
            "280421286",
            "50893709",
            "-50893709",
            "830912805#1",
            "27803377",
            "-27803377",
            "-38517406#2",
            "-38517406#1",
            "-38517406#0",
            "-27803376#2",
            "-27803376#1",
            "-27803376#0",
            "494708203#9",
            "494708203#8",
            "494708203#7",
        ),
        total_length_m=5778.38,
    ),
    "S6_St2051": StateSegment(
        segment_id="S6_St2051",
        description="St2051 / junction observation region S6.",
        edge_ids=(
            "-102858508#0",
            "-441828030#3",
            "-441828030#2",
            "-441828030#1",
            "-441828030#0",
            "-441828031",
            "-48781531#1",
            "-48781531#0",
            "-4270947#0",
            "-1381967172#0",
            "-4270947#1",
            "-123943388#0",
            "-4270947#2",
            "-48781532#0",
            "-4270947#3",
            "-1382013241",
            "-497862008#0",
            "1391055897",
            "-497862008#1",
            "52977564#8",
            "52977564#7",
            "52977564#6",
            "52977564#5",
            "52977564#4",
            "52977564#3",
            "52977564#2",
            "52977564#1",
        ),
        total_length_m=4982.08,
    ),
}


def get_state_segment_ids() -> list[str]:
    """Return the stable feature-order list of observation segment IDs."""
    return list(STATE_SEGMENTS)


def get_state_segment(segment_id: str) -> StateSegment:
    try:
        return STATE_SEGMENTS[segment_id]
    except KeyError as exc:
        valid = ", ".join(STATE_SEGMENTS)
        raise ValueError(
            f"Unknown state segment {segment_id!r}. Valid: {valid}"
        ) from exc


def get_state_segment_edges(segment_id: str) -> list[str]:
    """Return a copy of the exact edge list for one observation segment."""
    return list(get_state_segment(segment_id).edge_ids)


def validate_state_segments(net) -> None:
    """
    Fail early if an attached observation edge does not exist in the SUMO net.
    """
    missing: list[tuple[str, str]] = []

    for segment_id, segment in STATE_SEGMENTS.items():
        for edge_id in segment.edge_ids:
            try:
                net.getEdge(edge_id)
            except Exception:
                missing.append((segment_id, edge_id))

    if missing:
        preview = ", ".join(
            f"{segment_id}:{edge_id}"
            for segment_id, edge_id in missing[:20]
        )
        suffix = "" if len(missing) <= 20 else f" ... (+{len(missing)-20} more)"
        raise RuntimeError(
            "State-segment edges missing from network: "
            + preview
            + suffix
        )


def segment_overlap_flags(
    *,
    flood_edge_ids: Iterable[str],
    flood_active: bool,
) -> dict[str, float]:
    """
    Return one spatial flood flag per segment.

    A segment is flagged only while the physical flood is active and at least
    one of its observation edges overlaps the FloodEngine edge set.
    """
    if not flood_active:
        return {
            segment_id: 0.0
            for segment_id in get_state_segment_ids()
        }

    flood_edges = set(flood_edge_ids)

    return {
        segment_id: float(
            bool(
                flood_edges.intersection(
                    STATE_SEGMENTS[segment_id].edge_ids
                )
            )
        )
        for segment_id in get_state_segment_ids()
    }
