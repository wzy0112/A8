
from __future__ import annotations

from dataclasses import dataclass

# from flood_sites import EdgePathSpec, find_edge_path

from state_segments import (
    get_state_segment,
    get_state_segment_ids,
)

@dataclass(frozen=True)
class TacticSite:
    location_id: str
    description: str
    edge_ids: tuple[str, ...]


# imports
# ↓
# TacticSite
# ↓
# _build_tactic_sites()
# ↓
# TACTIC_SITES
# ↓
# get_location_ids()
# ↓
# resolve_tactic_site_edges()

# Small fixed candidate set for Step 6.
# These use boundary edges already validated by the fixed flood scenarios.
# state_segments.py
#         ↓
# S1–S6 exact edge lists
#         ↓
# tactic_sites.py
#         ↓
# Action Space / TacticManager
def _build_tactic_sites() -> dict[str, TacticSite]:
    sites: dict[str, TacticSite] = {}

    for segment_id in get_state_segment_ids():
        segment = get_state_segment(segment_id)

        sites[segment_id] = TacticSite(
            location_id=segment.segment_id,
            description=segment.description,
            edge_ids=segment.edge_ids,
        )

    return sites


TACTIC_SITES: dict[str, TacticSite] = _build_tactic_sites()


def get_location_ids() -> list[str]:
    return list(TACTIC_SITES)


def resolve_tactic_site_edges(
    net,
    location_id: str,
) -> list[str]:
    try:
        site = TACTIC_SITES[location_id]
    except KeyError as exc:
        valid = ", ".join(TACTIC_SITES)
        raise ValueError(
            f"Unknown tactic location {location_id!r}. "
            f"Valid: {valid}"
        ) from exc

    missing_edges = [
        edge_id
        for edge_id in site.edge_ids
        if not net.hasEdge(edge_id)
    ]

    if missing_edges:
        raise ValueError(
            f"Tactic location {location_id!r} contains "
            f"edges not present in the loaded SUMO network: "
            f"{missing_edges}"
        )

    return list(site.edge_ids)
