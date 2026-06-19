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


def compute_flow_proxy(df):
    work_df = df.copy()

    interval_seconds = work_df["end"] - work_df["begin"]
    interval_seconds = interval_seconds.replace(0, np.nan)

    work_df["flow_proxy"] = work_df["entered"] / interval_seconds * 3600.0
    return work_df


def aggregate_metrics(df):
    if df.empty:
        return pd.DataFrame()

    work_df = compute_flow_proxy(df)

    work_df.loc[work_df["speed"] < 0, "speed"] = np.nan
    work_df.loc[work_df["traveltime"] < 0, "traveltime"] = np.nan

    agg = (
        work_df.groupby("edge_id", as_index=False)
        .agg({
            "speed": "mean",
            "density": "mean",
            "flow_proxy": "mean"
        })
    )

    return agg


def prepare_difference_dataframe(base_df, flood_df):
    if base_df.empty or flood_df.empty:
        return pd.DataFrame()

    merged = base_df.merge(
        flood_df,
        on="edge_id",
        how="inner",
        suffixes=("_base", "_flood")
    )

    merged["speed_drop"] = merged["speed_base"] - merged["speed_flood"]
    merged["flow_change"] = merged["flow_proxy_flood"] - merged["flow_proxy_base"]
    merged["density_increase"] = merged["density_flood"] - merged["density_base"]

    return merged


def plot_difference_map(edge_geom, diff_df, value_col, title, colorbar_label, output_file):
    value_map = dict(zip(diff_df["edge_id"], diff_df[value_col]))

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

    print(f"[SUCCESS] Saved map: {output_file}")


def main():
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    baseline_edge_file = os.path.join(project_root, "heat_edges", "a8_edge_output.xml")
    flood_edge_file = os.path.join(project_root, "heat_edges", "a8_flood_edge_output.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    output_speed = os.path.join(results_dir, "diff_speed_drop_map.png")
    output_flow = os.path.join(results_dir, "diff_flow_change_map.png")
    output_density = os.path.join(results_dir, "diff_density_increase_map.png")
    output_csv = os.path.join(results_dir, "baseline_vs_flood_diff.csv")

    if not os.path.exists(net_file):
        print("[ERROR] Network file not found:")
        print(net_file)
        return

    if not os.path.exists(baseline_edge_file):
        print("[ERROR] Baseline edge output file not found:")
        print(baseline_edge_file)
        return

    if not os.path.exists(flood_edge_file):
        print("[ERROR] Flood edge output file not found:")
        print(flood_edge_file)
        return

    print("=====================================")
    print("Loading baseline and flooding edge data")
    print(f"Network file        : {net_file}")
    print(f"Baseline edge file  : {baseline_edge_file}")
    print(f"Flood edge file     : {flood_edge_file}")
    print("=====================================")

    sumolib = load_sumo_tools()
    net = sumolib.net.readNet(net_file)
    edge_geom = build_edge_geometry(net)

    baseline_raw = parse_edge_output(baseline_edge_file)
    flood_raw = parse_edge_output(flood_edge_file)

    if baseline_raw.empty:
        print("[ERROR] Baseline edge data is empty.")
        return

    if flood_raw.empty:
        print("[ERROR] Flood edge data is empty.")
        return

    baseline_agg = aggregate_metrics(baseline_raw)
    flood_agg = aggregate_metrics(flood_raw)

    diff_df = prepare_difference_dataframe(baseline_agg, flood_agg)

    if diff_df.empty:
        print("[ERROR] Difference DataFrame is empty.")
        return

    diff_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved difference table: {output_csv}")

    plot_difference_map(
        edge_geom=edge_geom,
        diff_df=diff_df,
        value_col="speed_drop",
        title="Speed Drop Map (Baseline - Flooding)",
        colorbar_label="Speed Drop (m/s)",
        output_file=output_speed
    )

    plot_difference_map(
        edge_geom=edge_geom,
        diff_df=diff_df,
        value_col="flow_change",
        title="Flow Change Map (Flooding - Baseline)",
        colorbar_label="Flow Change (veh/h)",
        output_file=output_flow
    )

    plot_difference_map(
        edge_geom=edge_geom,
        diff_df=diff_df,
        value_col="density_increase",
        title="Density Increase Map (Flooding - Baseline)",
        colorbar_label="Density Increase (veh/km)",
        output_file=output_density
    )

    print("=====================================")
    print("[SUCCESS] Baseline vs Flood difference maps completed.")
    print(f"Results folder: {results_dir}")
    print("=====================================")


if __name__ == "__main__":
    main()