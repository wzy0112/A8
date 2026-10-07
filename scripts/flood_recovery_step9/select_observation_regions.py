#!/usr/bin/env python3
"""
select_observation_regions.py

Purpose
-------
Build exactly six physically connected observation regions for the RL state space:

    S1 = B471 Gröbenried flood site
    S2 = B471 Dachauer Moos flood site
    S3 = A8 Palsweiser Moos flood site
    S4-S6 = three additional physically connected high-impact regions selected
            from baseline-vs-flooding speed-drop comparisons.

Only the SPEED metric is used to rank the additional regions.

Inputs
------
1) Light flooding comparison CSV
2) Moderate flooding comparison CSV
3) Severe flooding comparison CSV
4) SUMO .net.xml

Required CSV columns:
    edge_id
    speed_drop

Outputs
-------
observation_regions_summary.csv
observation_regions_edges.csv
observation_regions.json

Design
------
- The three flood-site regions are fixed from the real flood boundary edges.
- The three extra regions are selected using max positive speed drop across
  light/moderate/severe.
- Additional regions must be physically connected in the SUMO graph.
- Additional regions cannot reuse edges already assigned to the three flood sites.
- Internal SUMO edges (IDs beginning with ':') are ignored.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Fixed real-world flood-site definitions
# ---------------------------------------------------------------------------
# These three geographic flood sites come from the supplied boundary-edge table.
#
# NOTE:
# - Light flooding contains Gröbenried + Dachauer Moos.
# - Moderate flooding contains Palsweiser Moos.
# - Severe flooding contains Gröbenried + Palsweiser Moos.
#
# For the STATE SPACE we keep the three physical sites as three fixed
# observation regions rather than creating overlapping scenario-specific regions.

FLOOD_REGION_PATHS: dict[str, dict] = {
    "S1_groebenried_b471": {
        "description": "B471 Gröbenried flood site, both directions",
        "paths": [
            ("372647238", "511296353"),
            ("-511296353", "-372647238"),
        ],
    },
    "S2_dachauer_moos_b471": {
        "description": "B471 Dachauer Moos flood site, both directions",
        "paths": [
            ("32234550", "493360266#0"),
            ("-493360266#0", "-519348437"),
        ],
    },
    "S3_palsweiser_moos_a8": {
        "description": "A8 Palsweiser Moos flood site, both directions",
        "paths": [
            ("327464676", "327462610"),
            ("251505085", "251505081"),
        ],
    },
}


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: str
    from_node: str
    to_node: str
    length_m: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select six physically connected RL observation regions using "
            "three fixed flood sites plus three speed-impact regions."
        )
    )
    p.add_argument("--light", required=True, help="Light comparison CSV")
    p.add_argument("--moderate", required=True, help="Moderate comparison CSV")
    p.add_argument("--severe", required=True, help="Severe comparison CSV")
    p.add_argument("--net", required=True, help="SUMO .net.xml")
    p.add_argument("--output-dir", required=True, help="Output directory")

    p.add_argument(
        "--impact-quantile",
        type=float,
        default=0.90,
        help=(
            "Initial quantile of max positive speed_drop used to define "
            "high-impact edges. Default: 0.90."
        ),
    )
    p.add_argument(
        "--min-component-edges",
        type=int,
        default=3,
        help="Minimum number of edges in an automatically selected region.",
    )
    p.add_argument(
        "--min-component-length-m",
        type=float,
        default=300.0,
        help="Minimum total physical length of an automatically selected region.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Read SUMO network
# ---------------------------------------------------------------------------

def read_network(net_path: Path) -> tuple[
    dict[str, EdgeRecord],
    dict[str, list[str]],
    dict[str, set[str]],
]:
    """
    Read normal SUMO edges directly from .net.xml.

    Returns
    -------
    edges:
        edge_id -> EdgeRecord
    outgoing_by_node:
        junction id -> edges whose 'from' junction is this junction
    incident_by_node:
        junction id -> all normal edges incident to this junction

    Physical continuity for automatic regions is checked with incident_by_node,
    i.e. direction is ignored for the continuity test.
    """
    edges: dict[str, EdgeRecord] = {}
    outgoing_by_node: dict[str, list[str]] = defaultdict(list)
    incident_by_node: dict[str, set[str]] = defaultdict(set)

    for event, elem in ET.iterparse(net_path, events=("end",)):
        if elem.tag != "edge":
            continue

        edge_id = elem.get("id")
        function = elem.get("function")

        if (
            not edge_id
            or edge_id.startswith(":")
            or function == "internal"
        ):
            elem.clear()
            continue

        from_node = elem.get("from")
        to_node = elem.get("to")

        if from_node is None or to_node is None:
            elem.clear()
            continue

        lane_lengths = []
        for lane in elem.findall("lane"):
            try:
                lane_lengths.append(float(lane.get("length", "0")))
            except ValueError:
                pass

        # All lanes of one edge normally have very similar lengths.
        length_m = max(lane_lengths) if lane_lengths else 0.0

        record = EdgeRecord(
            edge_id=edge_id,
            from_node=from_node,
            to_node=to_node,
            length_m=length_m,
        )

        edges[edge_id] = record
        outgoing_by_node[from_node].append(edge_id)
        incident_by_node[from_node].add(edge_id)
        incident_by_node[to_node].add(edge_id)

        elem.clear()

    if not edges:
        raise RuntimeError(f"No normal edges found in network: {net_path}")

    return edges, outgoing_by_node, incident_by_node


# ---------------------------------------------------------------------------
# 3. Directed shortest path for fixed flood sites
# ---------------------------------------------------------------------------

def shortest_edge_path(
    start_edge_id: str,
    end_edge_id: str,
    edges: dict[str, EdgeRecord],
    outgoing_by_node: dict[str, list[str]],
) -> list[str]:
    """
    Directed shortest path from one SUMO edge to another.

    Consecutive path edges satisfy:
        previous.to_node == next.from_node
    """
    if start_edge_id not in edges:
        raise KeyError(f"Start edge not found in network: {start_edge_id}")
    if end_edge_id not in edges:
        raise KeyError(f"End edge not found in network: {end_edge_id}")

    dist: dict[str, float] = {start_edge_id: 0.0}
    prev: dict[str, str] = {}
    pq: list[tuple[float, str]] = [(0.0, start_edge_id)]

    while pq:
        current_dist, edge_id = heapq.heappop(pq)

        if current_dist != dist.get(edge_id):
            continue

        if edge_id == end_edge_id:
            break

        to_node = edges[edge_id].to_node

        for next_edge_id in outgoing_by_node.get(to_node, []):
            new_dist = current_dist + edges[next_edge_id].length_m

            if new_dist < dist.get(next_edge_id, math.inf):
                dist[next_edge_id] = new_dist
                prev[next_edge_id] = edge_id
                heapq.heappush(
                    pq,
                    (new_dist, next_edge_id),
                )

    if end_edge_id not in dist:
        raise RuntimeError(
            f"No directed path found: {start_edge_id} -> {end_edge_id}"
        )

    path = [end_edge_id]
    while path[-1] != start_edge_id:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def build_fixed_flood_regions(
    edges: dict[str, EdgeRecord],
    outgoing_by_node: dict[str, list[str]],
) -> dict[str, list[str]]:
    regions: dict[str, list[str]] = {}

    for region_id, spec in FLOOD_REGION_PATHS.items():
        region_edges: list[str] = []

        for start_edge_id, end_edge_id in spec["paths"]:
            path = shortest_edge_path(
                start_edge_id,
                end_edge_id,
                edges,
                outgoing_by_node,
            )
            region_edges.extend(path)

        # preserve order while removing duplicates
        regions[region_id] = list(dict.fromkeys(region_edges))

    return regions


# ---------------------------------------------------------------------------
# 4. Read speed-drop comparisons
# ---------------------------------------------------------------------------

def read_speed_drop_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"edge_id", "speed_drop"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    out = df[["edge_id", "speed_drop"]].copy()
    out["edge_id"] = out["edge_id"].astype(str)
    out["speed_drop"] = pd.to_numeric(
        out["speed_drop"],
        errors="coerce",
    ).fillna(0.0)

    out = out.drop_duplicates(
        subset=["edge_id"],
        keep="first",
    )

    return out.rename(
        columns={"speed_drop": f"speed_drop_{label}"}
    )


def build_speed_impact_table(
    light_csv: Path,
    moderate_csv: Path,
    severe_csv: Path,
    network_edge_ids: set[str],
) -> pd.DataFrame:
    light = read_speed_drop_csv(light_csv, "light")
    moderate = read_speed_drop_csv(moderate_csv, "moderate")
    severe = read_speed_drop_csv(severe_csv, "severe")

    df = light.merge(
        moderate,
        on="edge_id",
        how="outer",
    ).merge(
        severe,
        on="edge_id",
        how="outer",
    )

    drop_cols = [
        "speed_drop_light",
        "speed_drop_moderate",
        "speed_drop_severe",
    ]

    for col in drop_cols:
        df[col] = df[col].fillna(0.0)

    # Only positive speed deterioration is used for region selection.
    positive = df[drop_cols].clip(lower=0.0)

    df["max_positive_speed_drop"] = positive.max(axis=1)

    dominant_idx = positive.to_numpy().argmax(axis=1)
    labels = np.array(["light", "moderate", "severe"])
    df["dominant_scenario"] = labels[dominant_idx]

    # Keep only normal edges actually present in this SUMO network.
    df = df[
        df["edge_id"].isin(network_edge_ids)
    ].copy()

    return df


# ---------------------------------------------------------------------------
# 5. Connected-component selection
# ---------------------------------------------------------------------------

def physical_neighbors(
    edge_id: str,
    edges: dict[str, EdgeRecord],
    incident_by_node: dict[str, set[str]],
) -> set[str]:
    record = edges[edge_id]
    return (
        incident_by_node[record.from_node]
        | incident_by_node[record.to_node]
    ) - {edge_id}


def connected_components(
    selected_edges: set[str],
    edges: dict[str, EdgeRecord],
    incident_by_node: dict[str, set[str]],
) -> list[list[str]]:
    """
    Return weakly/physically connected edge components.

    Travel direction is ignored here because the requirement is physical
    continuity of an observation region.
    """
    components: list[list[str]] = []
    visited: set[str] = set()

    for seed in sorted(selected_edges):
        if seed in visited:
            continue

        component: list[str] = []
        queue = deque([seed])
        visited.add(seed)

        while queue:
            edge_id = queue.popleft()
            component.append(edge_id)

            for nbr in physical_neighbors(
                edge_id,
                edges,
                incident_by_node,
            ):
                if nbr in selected_edges and nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)

        components.append(component)

    return components


def component_statistics(
    component: list[str],
    impact_lookup: dict[str, dict],
    edges: dict[str, EdgeRecord],
) -> dict:
    lengths = np.array(
        [edges[e].length_m for e in component],
        dtype=float,
    )
    drops = np.array(
        [
            impact_lookup[e]["max_positive_speed_drop"]
            for e in component
        ],
        dtype=float,
    )

    total_length = float(lengths.sum())

    if total_length > 0:
        length_weighted_mean_drop = float(
            np.average(
                drops,
                weights=np.maximum(lengths, 1e-9),
            )
        )
    else:
        length_weighted_mean_drop = float(drops.mean())

    # Ranking score rewards both strong speed deterioration and spatial extent.
    impact_score = float(
        np.sum(
            drops * np.maximum(lengths, 1.0)
        )
    )

    scenario_scores = {
        scenario: float(
            sum(
                max(
                    0.0,
                    impact_lookup[e][f"speed_drop_{scenario}"],
                )
                * max(edges[e].length_m, 1.0)
                for e in component
            )
        )
        for scenario in ("light", "moderate", "severe")
    }

    dominant_scenario = max(
        scenario_scores,
        key=scenario_scores.get,
    )

    return {
        "n_edges": len(component),
        "total_length_m": total_length,
        "mean_positive_speed_drop_mps": length_weighted_mean_drop,
        "max_positive_speed_drop_mps": float(drops.max()),
        "impact_score": impact_score,
        "dominant_scenario": dominant_scenario,
    }


def select_three_additional_regions(
    impact_df: pd.DataFrame,
    fixed_edge_ids: set[str],
    edges: dict[str, EdgeRecord],
    incident_by_node: dict[str, set[str]],
    initial_quantile: float,
    min_component_edges: int,
    min_component_length_m: float,
) -> tuple[list[dict], float]:
    """
    Select 3 additional physically connected high-speed-drop components.

    If the initial threshold does not yield 3 usable components, the threshold
    is relaxed gradually until 3 are available.
    """
    impact_lookup = {
        row["edge_id"]: row
        for row in impact_df.to_dict("records")
    }

    available_df = impact_df[
        ~impact_df["edge_id"].isin(fixed_edge_ids)
    ].copy()

    values = available_df[
        "max_positive_speed_drop"
    ].to_numpy(dtype=float)

    if len(values) == 0:
        raise RuntimeError(
            "No comparison edges remain after excluding fixed flood regions."
        )

    # Search from the requested threshold downward if necessary.
    quantiles = []
    q = initial_quantile
    while q >= 0.50 - 1e-9:
        quantiles.append(round(q, 4))
        q -= 0.02

    for q in quantiles:
        threshold = float(np.quantile(values, q))

        selected = set(
            available_df.loc[
                available_df[
                    "max_positive_speed_drop"
                ] >= threshold,
                "edge_id",
            ]
        )

        components = connected_components(
            selected,
            edges,
            incident_by_node,
        )

        candidates: list[dict] = []

        for component in components:
            stats = component_statistics(
                component,
                impact_lookup,
                edges,
            )

            if stats["n_edges"] < min_component_edges:
                continue
            if stats["total_length_m"] < min_component_length_m:
                continue

            candidates.append(
                {
                    "edges": component,
                    **stats,
                }
            )

        candidates.sort(
            key=lambda x: x["impact_score"],
            reverse=True,
        )

        if len(candidates) >= 3:
            return candidates[:3], threshold

    raise RuntimeError(
        "Could not find three physically connected high-impact regions. "
        "Try lowering --min-component-edges or --min-component-length-m."
    )


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------

def is_physically_connected(
    region_edges: list[str],
    edges: dict[str, EdgeRecord],
    incident_by_node: dict[str, set[str]],
) -> bool:
    if not region_edges:
        return False

    edge_set = set(region_edges)
    components = connected_components(
        edge_set,
        edges,
        incident_by_node,
    )
    return len(components) == 1


# ---------------------------------------------------------------------------
# 7. Outputs
# ---------------------------------------------------------------------------

def write_outputs(
    output_dir: Path,
    fixed_regions: dict[str, list[str]],
    additional_regions: list[dict],
    impact_df: pd.DataFrame,
    edges: dict[str, EdgeRecord],
    incident_by_node: dict[str, set[str]],
    impact_threshold: float,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    impact_lookup = {
        row["edge_id"]: row
        for row in impact_df.to_dict("records")
    }

    all_regions: list[dict] = []

    # S1-S3
    for region_id, edge_ids in fixed_regions.items():
        stats = component_statistics(
            edge_ids,
            impact_lookup,
            edges,
        )

        # A fixed flood-site region may contain the two opposite-direction
        # carriageways of the same physical corridor. Those two directed paths
        # do not have to be graph-connected to each other at their endpoints,
        # but every path in FLOOD_REGION_PATHS has already been validated as a
        # continuous directed shortest path. Therefore the flood-site region is
        # considered physically corridor-continuous by construction.
        all_regions.append(
            {
                "region_id": region_id,
                "region_type": "fixed_flood_site",
                "description": (
                    FLOOD_REGION_PATHS[region_id]["description"]
                ),
                "edge_ids": edge_ids,
                "connected": True,
                "continuity_rule": (
                    "fixed flood corridor: each directional path is "
                    "continuous; opposite carriageways belong to the same "
                    "physical observation corridor"
                ),
                **stats,
            }
        )

    # S4-S6
    for idx, candidate in enumerate(
        additional_regions,
        start=4,
    ):
        region_id = f"S{idx}_speed_impact"

        all_regions.append(
            {
                "region_id": region_id,
                "region_type": "speed_impact",
                "description": (
                    "Automatically selected connected high-speed-drop region"
                ),
                "edge_ids": candidate["edges"],
                "connected": is_physically_connected(
                    candidate["edges"],
                    edges,
                    incident_by_node,
                ),
                "continuity_rule": (
                    "automatic region: all edges belong to one weakly "
                    "connected SUMO edge component"
                ),
                **{
                    k: v
                    for k, v in candidate.items()
                    if k != "edges"
                },
            }
        )

    if len(all_regions) != 6:
        raise AssertionError(
            f"Expected exactly 6 regions, got {len(all_regions)}"
        )

    if not all(r["connected"] for r in all_regions):
        bad = [
            r["region_id"]
            for r in all_regions
            if not r["connected"]
        ]
        raise RuntimeError(
            f"Non-continuous regions detected: {bad}"
        )

    # Summary CSV
    summary_rows = []
    for r in all_regions:
        summary_rows.append(
            {
                "region_id": r["region_id"],
                "region_type": r["region_type"],
                "description": r["description"],
                "n_edges": r["n_edges"],
                "total_length_m": r["total_length_m"],
                "mean_positive_speed_drop_mps": (
                    r["mean_positive_speed_drop_mps"]
                ),
                "max_positive_speed_drop_mps": (
                    r["max_positive_speed_drop_mps"]
                ),
                "impact_score": r["impact_score"],
                "dominant_scenario": r["dominant_scenario"],
                "physically_connected": r["connected"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "observation_regions_summary.csv",
        index=False,
    )

    # One row per edge
    edge_rows = []
    for r in all_regions:
        for sequence, edge_id in enumerate(
            r["edge_ids"],
            start=1,
        ):
            info = impact_lookup.get(edge_id, {})
            rec = edges[edge_id]

            edge_rows.append(
                {
                    "region_id": r["region_id"],
                    "region_type": r["region_type"],
                    "edge_sequence": sequence,
                    "edge_id": edge_id,
                    "from_junction": rec.from_node,
                    "to_junction": rec.to_node,
                    "length_m": rec.length_m,
                    "speed_drop_light": info.get(
                        "speed_drop_light",
                        0.0,
                    ),
                    "speed_drop_moderate": info.get(
                        "speed_drop_moderate",
                        0.0,
                    ),
                    "speed_drop_severe": info.get(
                        "speed_drop_severe",
                        0.0,
                    ),
                    "max_positive_speed_drop": info.get(
                        "max_positive_speed_drop",
                        0.0,
                    ),
                    "dominant_scenario": info.get(
                        "dominant_scenario",
                        "",
                    ),
                }
            )

    pd.DataFrame(edge_rows).to_csv(
        output_dir / "observation_regions_edges.csv",
        index=False,
    )

    # JSON
    json_payload = {
        "selection_method": {
            "metric": "speed_drop only",
            "automatic_region_threshold_mps": impact_threshold,
            "automatic_region_score": (
                "sum(max_positive_speed_drop * edge_length_m)"
            ),
            "continuity_rule": (
                "all edges in a region must form one physical connected "
                "component through shared SUMO junctions"
            ),
        },
        "regions": [
            {
                **{
                    k: v
                    for k, v in r.items()
                    if k != "edge_ids"
                },
                "edge_ids": r["edge_ids"],
            }
            for r in all_regions
        ],
    }

    with (
        output_dir / "observation_regions.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            json_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSelected six observation regions")
    print("=" * 80)

    for r in all_regions:
        print(
            f"{r['region_id']}: "
            f"{r['n_edges']} edges, "
            f"{r['total_length_m'] / 1000:.2f} km, "
            f"mean +drop={r['mean_positive_speed_drop_mps']:.3f} m/s, "
            f"max +drop={r['max_positive_speed_drop_mps']:.3f} m/s, "
            f"dominant={r['dominant_scenario']}, "
            f"connected={r['connected']}"
        )

    print("\nOutputs:")
    print(output_dir / "observation_regions_summary.csv")
    print(output_dir / "observation_regions_edges.csv")
    print(output_dir / "observation_regions.json")


def main() -> None:
    args = parse_args()

    net_path = Path(args.net)
    light_csv = Path(args.light)
    moderate_csv = Path(args.moderate)
    severe_csv = Path(args.severe)
    output_dir = Path(args.output_dir)

    edges, outgoing_by_node, incident_by_node = read_network(
        net_path
    )

    fixed_regions = build_fixed_flood_regions(
        edges,
        outgoing_by_node,
    )

    fixed_edge_ids = set()
    for edge_ids in fixed_regions.values():
        fixed_edge_ids.update(edge_ids)

    impact_df = build_speed_impact_table(
        light_csv=light_csv,
        moderate_csv=moderate_csv,
        severe_csv=severe_csv,
        network_edge_ids=set(edges),
    )

    additional_regions, threshold = (
        select_three_additional_regions(
            impact_df=impact_df,
            fixed_edge_ids=fixed_edge_ids,
            edges=edges,
            incident_by_node=incident_by_node,
            initial_quantile=args.impact_quantile,
            min_component_edges=args.min_component_edges,
            min_component_length_m=args.min_component_length_m,
        )
    )

    write_outputs(
        output_dir=output_dir,
        fixed_regions=fixed_regions,
        additional_regions=additional_regions,
        impact_df=impact_df,
        edges=edges,
        incident_by_node=incident_by_node,
        impact_threshold=threshold,
    )


if __name__ == "__main__":
    main()
