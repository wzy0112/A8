import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


SUMO_HOME = r"D:\Eclipse\SUMO"


def load_sumo_tools():
    tools_path = os.path.join(SUMO_HOME, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)

    import sumolib
    return sumolib


def parse_edge_output(edge_output_file):
    """
    Parse SUMO edgeData output.
    """
    if not os.path.exists(edge_output_file):
        print("[ERROR] Edge output file not found:")
        print(edge_output_file)
        return pd.DataFrame()

    tree = ET.parse(edge_output_file)
    root = tree.getroot()

    records = []

    for interval in root.findall("interval"):
        begin = float(interval.get("begin", -1))
        end = float(interval.get("end", -1))

        for edge in interval.findall("edge"):
            records.append({
                "begin": begin,
                "end": end,
                "edge_id": edge.get("id", ""),
                "sampledSeconds": float(edge.get("sampledSeconds", 0)),
                "density": float(edge.get("density", 0)),
                "laneDensity": float(edge.get("laneDensity", 0)),
                "speed": float(edge.get("speed", -1)),
                "occupancy": float(edge.get("occupancy", 0)),
                "traveltime": float(edge.get("traveltime", -1)),
                "entered": float(edge.get("entered", 0)),
                "left": float(edge.get("left", 0)),
            })

    return pd.DataFrame(records)


def build_edge_geometry(net):
    """
    Extract geometry for each edge as a list of line segments.
    """
    edge_geom = {}

    for edge in net.getEdges():
        if edge.isSpecial():
            continue

        shape = edge.getShape()
        if shape is None or len(shape) < 2:
            continue

        segments = []
        for i in range(len(shape) - 1):
            segments.append([shape[i], shape[i + 1]])

        edge_geom[edge.getID()] = segments

    return edge_geom


def aggregate_edge_metric(df, metric):
    """
    Aggregate a metric over time for each edge.
    """
    if df.empty:
        return pd.DataFrame()

    work_df = df.copy()

    if metric in ["speed", "traveltime"]:
        work_df.loc[work_df[metric] < 0, metric] = np.nan

    grouped = work_df.groupby("edge_id", as_index=False)[metric].mean()
    grouped = grouped.rename(columns={metric: "value"})
    return grouped


def compute_flow_proxy(df):
    """
    Compute a simple flow proxy from entered vehicles per interval.
    Since freq=60 s, entered * 60 = veh/h.
    """
    if df.empty:
        return pd.DataFrame()

    work_df = df.copy()
    interval_seconds = work_df["end"] - work_df["begin"]
    interval_seconds = interval_seconds.replace(0, np.nan)

    work_df["flow_proxy"] = work_df["entered"] / interval_seconds * 3600.0
    grouped = work_df.groupby("edge_id", as_index=False)["flow_proxy"].mean()
    grouped = grouped.rename(columns={"flow_proxy": "value"})
    return grouped


def plot_network_heatmap(net, edge_geom, metric_df, title, colorbar_label, output_file):
    """
    Plot a network heatmap by coloring each edge.
    """
    value_map = dict(zip(metric_df["edge_id"], metric_df["value"]))

    segments = []
    values = []

    for edge_id, segs in edge_geom.items():
        if edge_id not in value_map:
            continue

        val = value_map[edge_id]
        if pd.isna(val):
            continue

        for seg in segs:
            segments.append(seg)
            values.append(val)

    if not segments:
        print(f"[WARNING] No segments available for plot: {title}")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    lc = LineCollection(segments, array=np.array(values), linewidths=2)
    ax.add_collection(lc)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)

    cbar = plt.colorbar(lc, ax=ax, shrink=0.8)
    cbar.set_label(colorbar_label)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Saved heatmap: {output_file}")


def visualize_network_heatmaps():
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    edge_output_file = os.path.join(project_root, "heat_edges", "a8_edge_output.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    output_speed = os.path.join(results_dir, "network_speed_heatmap.png")
    output_flow = os.path.join(results_dir, "network_flow_heatmap.png")
    output_density = os.path.join(results_dir, "network_density_heatmap.png")

    if not os.path.exists(net_file):
        print("[ERROR] Network file not found:")
        print(net_file)
        return

    if not os.path.exists(edge_output_file):
        print("[ERROR] Edge output file not found:")
        print(edge_output_file)
        return

    print("=====================================")
    print("Loading network-wide edge data")
    print(f"Network file    : {net_file}")
    print(f"Edge output file: {edge_output_file}")
    print("=====================================")

    sumolib = load_sumo_tools()
    net = sumolib.net.readNet(net_file)
    edge_geom = build_edge_geometry(net)

    df = parse_edge_output(edge_output_file)

    if df.empty:
        print("[ERROR] No edge data found.")
        return

    speed_df = aggregate_edge_metric(df, "speed")
    density_df = aggregate_edge_metric(df, "density")
    flow_df = compute_flow_proxy(df)

    plot_network_heatmap(
        net=net,
        edge_geom=edge_geom,
        metric_df=speed_df,
        title="Network Mean Speed Heatmap",
        colorbar_label="Speed (m/s)",
        output_file=output_speed
    )

    plot_network_heatmap(
        net=net,
        edge_geom=edge_geom,
        metric_df=flow_df,
        title="Network Mean Flow Heatmap",
        colorbar_label="Flow Proxy (veh/h)",
        output_file=output_flow
    )

    plot_network_heatmap(
        net=net,
        edge_geom=edge_geom,
        metric_df=density_df,
        title="Network Mean Density Heatmap",
        colorbar_label="Density (veh/km)",
        output_file=output_density
    )

    print("=====================================")
    print("[SUCCESS] Network heatmap visualization completed.")
    print(f"Results folder: {results_dir}")
    print("=====================================")


if __name__ == "__main__":
    visualize_network_heatmaps()