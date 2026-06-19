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


def aggregate_metrics(df, net):
    if df.empty:
        return pd.DataFrame()

    work_df = df.copy()

    interval_seconds = work_df["end"] - work_df["begin"]
    interval_seconds = interval_seconds.replace(0, np.nan)

    # flow (veh/h)
    work_df["flow"] = work_df["entered"] / interval_seconds * 3600.0

    # invalid values
    work_df.loc[work_df["speed"] < 0, "speed"] = np.nan
    work_df.loc[work_df["traveltime"] < 0, "traveltime"] = np.nan

    # capacity + v/c
    capacity_list = []

    for edge_id in work_df["edge_id"]:
        try:
            edge = net.getEdge(edge_id)
            num_lanes = edge.getLaneNumber()
            edge_type = edge.getType()
        except:
            capacity_list.append(np.nan)
            continue

        if edge_type and ("motorway" in edge_type or "highway.motorway" in edge_type):
            cap_per_lane = 2000
        else:
            cap_per_lane = 800

        capacity_list.append(num_lanes * cap_per_lane)

    work_df["capacity"] = capacity_list
    work_df["vc"] = work_df["flow"] / work_df["capacity"]

    agg = (
        work_df.groupby("edge_id", as_index=False)
        .agg({
            "speed": "mean",
            "density": "mean",
            "flow": "mean",
            "vc": "mean"
        })
    )

    return agg


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

def compute_vc_ratio(df, net):
    """
    Compute v/c ratio = flow / capacity
    """
    if df.empty:
        return pd.DataFrame()

    work_df = df.copy()

    # ===== 1. 计算 flow（veh/h） =====
    interval_seconds = work_df["end"] - work_df["begin"]
    interval_seconds = interval_seconds.replace(0, np.nan)

    work_df["flow"] = work_df["entered"] / interval_seconds * 3600.0

    # ===== 2. 获取 edge 属性 =====
    capacity_list = []

    for edge_id in work_df["edge_id"]:
        try:
            edge = net.getEdge(edge_id)
            num_lanes = edge.getLaneNumber()
            edge_type = edge.getType()
        except:
            capacity_list.append(np.nan)
            continue

        # ===== 3. capacity rule =====
        if edge_type and ("motorway" in edge_type or "highway.motorway" in edge_type):
            cap_per_lane = 2000
        else:
            cap_per_lane = 800

        capacity = num_lanes * cap_per_lane
        capacity_list.append(capacity)

    work_df["capacity"] = capacity_list

    # ===== 4. v/c =====
    work_df["vc"] = work_df["flow"] / work_df["capacity"]

    grouped = work_df.groupby("edge_id", as_index=False)["vc"].mean()
    grouped = grouped.rename(columns={"vc": "value"})

    return grouped

def plot_network_heatmap(net, edge_geom, metric_df, title, colorbar_label, output_file, cmap_type="density"):
    import matplotlib.colors as mcolors

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

    # 🔥 normalization（核心）
    vmin = np.percentile(values, 5)
    vmax = np.percentile(values, 95)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 🔥 colormap 选择
    if cmap_type == "density":
        cmap = "RdYlGn_r"
    elif cmap_type == "flow":
        cmap = "RdYlGn_r"
    elif cmap_type == "speed":
        cmap = "RdYlGn"
    elif cmap_type == "v/c":
        cmap = "RdYlGn_r"
    else:
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(12, 10))

    lc = LineCollection(
        segments,
        array=np.array(values),
        linewidths=2,
        cmap=cmap,
        norm=norm
    )

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
    edge_output_file = os.path.join(project_root, "heat_edges", "a8_flood_edge_output_od.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    output_speed = os.path.join(results_dir, "flood_network_speed_heatmap_od.png")
    output_flow = os.path.join(results_dir, "flood_network_flow_heatmap_od.png")
    output_density = os.path.join(results_dir, "flood_network_density_heatmap_od.png")
    output_vc = os.path.join(results_dir, "flood_network_vc_heatmap_od.png")



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
    vc_df = compute_vc_ratio(df, net)

    plot_network_heatmap(
        net=net,
        edge_geom=edge_geom,
        metric_df=speed_df,
        title="Network Mean Speed Heatmap",
        colorbar_label="Speed (m/s)",
        output_file=output_speed,
        cmap_type="speed"
    )

    plot_network_heatmap(
        net=net,
        edge_geom=edge_geom,
        metric_df=flow_df,
        title="Network Mean Flow Heatmap",
        colorbar_label="Flow Proxy (veh/h)",
        output_file=output_flow,
        cmap_type="flow"
    )

    plot_network_heatmap(
        net=net,
        edge_geom=edge_geom,
        metric_df=density_df,
        title="Network Mean Density Heatmap",
        colorbar_label="Density (veh/km)",
        output_file=output_density,
        cmap_type="density"
    )

    plot_network_heatmap(
        net=net,
        edge_geom=edge_geom,
        metric_df=vc_df,
        title="Network Mean v/c Ratio Heatmap",
        colorbar_label="v/c Ratio",
        output_file=output_vc,
        cmap_type="v/c"
    )

    print("=====================================")
    print("[SUCCESS] Network heatmap visualization completed.")
    print(f"Results folder: {results_dir}")
    print("=====================================")


if __name__ == "__main__":
    visualize_network_heatmaps()