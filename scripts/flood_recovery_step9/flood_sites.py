from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgePathSpec:
    start_edge_id: str
    end_edge_id: str


@dataclass(frozen=True)
class FloodSite:
    site_id: str
    description: str
    paths: tuple[EdgePathSpec, ...]


FLOOD_SITES: dict[str, FloodSite] = {
    "b471_groebenried_dachauer_moos": FloodSite(
        site_id="b471_groebenried_dachauer_moos",
        description=(
            "B471 Dachau-Sued/Groebenried and B471 Dachauer Moos, "
            "both travel directions."
        ),
        paths=(
            EdgePathSpec("372647238", "511296353"),
            EdgePathSpec("-511296353", "-372647238"),
            EdgePathSpec("32234550", "493360266#0"),
            EdgePathSpec("-493360266#0", "-519348437"),
        ),
    ),
    "a8_sulzemoos_dachau": FloodSite(
        site_id="a8_sulzemoos_dachau",
        description=(
            "A8 Sulzemoos/Fuchsberg to Dachau/Fuerstenfeldbruck, "
            "both travel directions."
        ),
        paths=(
            EdgePathSpec("327464676", "327462610"),
            EdgePathSpec("251505085", "251505081"),
        ),
    ),
    "a8_b471_combined": FloodSite(
        site_id="a8_b471_combined",
        description=(
            "Combined severe site: B471 Groebenried plus A8 "
            "Palsweiser Moos/Sulzemoos-Dachau."
        ),
        paths=(
            EdgePathSpec("372647238", "511296353"),
            EdgePathSpec("-511296353", "-372647238"),
            EdgePathSpec("327464676", "327462610"),
            EdgePathSpec("251505085", "251505081"),
        ),
    ),
}


def get_flood_site(site_id: str) -> FloodSite:
    try:
        return FLOOD_SITES[site_id]
    except KeyError as exc:
        valid = ", ".join(sorted(FLOOD_SITES))
        raise ValueError(
            f"Unknown flood site {site_id!r}. Valid values: {valid}"
        ) from exc


def find_edge_path(net, start_edge_id: str, end_edge_id: str) -> list[str]:
    try:
        path_edges, _cost = net.getShortestPath(
            net.getEdge(start_edge_id),
            net.getEdge(end_edge_id),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Path search failed: {start_edge_id} -> {end_edge_id}"
        ) from exc

    if path_edges is None:
        raise RuntimeError(
            f"No path found: {start_edge_id} -> {end_edge_id}"
        )

    return [edge.getID() for edge in path_edges]


def resolve_site_edges(net, site_id: str) -> list[str]:
    site = get_flood_site(site_id)
    edges: list[str] = []

    for path_spec in site.paths:
        edges.extend(
            find_edge_path(
                net,
                path_spec.start_edge_id,
                path_spec.end_edge_id,
            )
        )

    return list(dict.fromkeys(edges))
