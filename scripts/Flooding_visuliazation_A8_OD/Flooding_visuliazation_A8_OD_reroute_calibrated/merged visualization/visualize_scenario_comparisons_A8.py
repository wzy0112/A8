import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# 相比正常情况，轻度/中度/重度封闭分别造成了多大变化？
##speed flow density vc
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

def parse_tripinfo(tripinfo_file):
    if not os.path.exists(tripinfo_file):
        print("[WARNING] tripinfo file not found:", tripinfo_file)
        return pd.DataFrame()

    tree = ET.parse(tripinfo_file)
    root = tree.getroot()

    records = []
    for trip in root.findall("tripinfo"):
        records.append({
            "id": trip.get("id"),
            "depart": float(trip.get("depart", 0)),
            "arrival": float(trip.get("arrival", 0)),
            "duration": float(trip.get("duration", 0)),
            "routeLength": float(trip.get("routeLength", 0)),
            "waitingTime": float(trip.get("waitingTime", 0)),
            "timeLoss": float(trip.get("timeLoss", 0)),
        })

    return pd.DataFrame(records)


def parse_summary(summary_file):
    if not os.path.exists(summary_file):
        print("[WARNING] summary file not found:", summary_file)
        return pd.DataFrame()

    tree = ET.parse(summary_file)
    root = tree.getroot()

    records = []
    for step in root.findall("step"):
        records.append({
            "time": float(step.get("time", 0)),
            "loaded": float(step.get("loaded", 0)),
            "inserted": float(step.get("inserted", 0)),
            "running": float(step.get("running", 0)),
            "waiting": float(step.get("waiting", 0)),
            "ended": float(step.get("ended", 0)),
        })

    return pd.DataFrame(records)


def load_step_metrics(step_metrics_file):
    if not os.path.exists(step_metrics_file):
        print("[WARNING] step metrics file not found:", step_metrics_file)
        return pd.DataFrame()

    return pd.read_csv(step_metrics_file)

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


def prepare_difference_dataframe(base_df, scenario_df):
    if base_df.empty or scenario_df.empty:
        return pd.DataFrame()

    merged = base_df.merge(
        scenario_df,
        on="edge_id",
        how="inner",
        suffixes=("_base", "_scenario")
    )

    # 1. speed：速度下降越大越红
    merged["speed_drop"] = merged["speed_base"] - merged["speed_scenario"]

    # 2. density：密度增加越大越红
    merged["density_increase"] = merged["density_scenario"] - merged["density_base"]

    # 3. flow：变化幅度越大越红
    merged["flow_change_abs"] = (merged["flow_scenario"] - merged["flow_base"]).abs()

    # 4. v/c：负荷增加越大越红
    merged["vc_increase"] = merged["vc_scenario"] - merged["vc_base"]

    return merged


def plot_difference_map(edge_geom, diff_df, value_col, title, colorbar_label, output_file):
    import matplotlib.colors as mcolors

    value_map = dict(zip(diff_df["edge_id"], diff_df[value_col]))

    segments = []
    values = []

    for edge_id, segs in edge_geom.items():
        val = value_map.get(edge_id, 0)

        if pd.isna(val):
            continue

        for seg in segs:
            segments.append(seg)
            values.append(val)

    if not segments:
        print(f"[WARNING] No segments available for plot: {title}")
        return

    # 差值图统一规则：小 = 绿，中 = 黄，大 = 红
    cmap = "RdYlGn_r"

    # 保持你原来 pairwise 代码的 density 特殊色标逻辑
    if value_col == "density_increase":
        positive_values = [v for v in values if v > 0]

        if positive_values:
            vmin = 0.0
            vmax = np.percentile(positive_values, 50)
        else:
            vmin = 0.0
            vmax = 1.0

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    else:
        vmin = 0.0
        vmax = max(values) if values else 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

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

    print(f"[SUCCESS] Saved map: {output_file}")


def plot_mean_speed_time(step_df, scenario_name, output_file):
    if step_df.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(step_df["time_s"], step_df["mean_speed_mps"])
    plt.xlabel("Time (s)")
    plt.ylabel("Mean speed (m/s)")
    plt.title(f"{scenario_name} - Mean Speed vs Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_vehicles_in_network(step_df, scenario_name, output_file):
    if step_df.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(step_df["time_s"], step_df["running_vehicle_count"])
    plt.xlabel("Time (s)")
    plt.ylabel("Running vehicles")
    plt.title(f"{scenario_name} - Vehicles in Network vs Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_recovery_curve(summary_df, scenario_name, output_file):
    if summary_df.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(summary_df["time"], summary_df["running"])
    plt.xlabel("Time (s)")
    plt.ylabel("Vehicles remaining in network")
    plt.title(f"{scenario_name} - Recovery Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_travel_time_cdf(trip_df, scenario_name, output_file):
    if trip_df.empty:
        return

    values = np.sort(trip_df["duration"].dropna().values)
    y = np.arange(1, len(values) + 1) / len(values)

    plt.figure(figsize=(8, 6))
    plt.plot(values, y)
    plt.xlabel("Travel time (s)")
    plt.ylabel("Cumulative probability")
    plt.title(f"{scenario_name} - Travel Time CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_timeloss_boxplot(trip_df, scenario_name, output_file):
    if trip_df.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.boxplot(trip_df["timeLoss"].dropna().values, labels=[scenario_name])
    plt.ylabel("Time loss (s)")
    plt.title(f"{scenario_name} - TimeLoss Boxplot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def visualize_one_comparison(
    base_key,
    base_label,
    base_agg,
    scenario_key,
    scenario_label,
    scenario_edge_file,
    net,
    edge_geom,
    results_dir
):
    print("=====================================")
    print(f"[COMPARISON] {base_label} vs {scenario_label}")
    print(f"Scenario edge file: {scenario_edge_file}")
    print("=====================================")

    if not os.path.exists(scenario_edge_file):
        print("[ERROR] Scenario edge output file not found:")
        print(scenario_edge_file)
        return

    scenario_raw = parse_edge_output(scenario_edge_file)

    if scenario_raw.empty:
        print(f"[ERROR] Scenario edge data is empty: {scenario_label}")
        return

    scenario_agg = aggregate_metrics(scenario_raw, net)

    diff_df = prepare_difference_dataframe(base_agg, scenario_agg)

    if diff_df.empty:
        print(f"[ERROR] Difference DataFrame is empty: {base_label} vs {scenario_label}")
        return

    output_prefix = f"{base_key}_vs_{scenario_key}"

    output_csv = os.path.join(results_dir, f"{output_prefix}_diff_od_reroute.csv")
    output_speed = os.path.join(results_dir, f"{output_prefix}_speed_drop_map_od_reroute.png")
    output_flow = os.path.join(results_dir, f"{output_prefix}_flow_change_map_od_reroute.png")
    output_density = os.path.join(results_dir, f"{output_prefix}_density_increase_map_od_reroute.png")
    output_vc = os.path.join(results_dir, f"{output_prefix}_vc_increase_map_od_reroute.png")

    diff_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved difference table: {output_csv}")

    plot_difference_map(
        edge_geom=edge_geom,
        diff_df=diff_df,
        value_col="speed_drop",
        title=f"Speed Drop Map ({base_label} - {scenario_label})",
        colorbar_label="Speed Drop (m/s)",
        output_file=output_speed
    )

    plot_difference_map(
        edge_geom=edge_geom,
        diff_df=diff_df,
        value_col="flow_change_abs",
        title=f"Flow Change Magnitude Map | abs({scenario_label} - {base_label})",
        colorbar_label="|Flow Change| (veh/h)",
        output_file=output_flow
    )

    plot_difference_map(
        edge_geom=edge_geom,
        diff_df=diff_df,
        value_col="density_increase",
        title=f"Density Increase Map ({scenario_label} - {base_label})",
        colorbar_label="Density Increase (veh/km)",
        output_file=output_density
    )

    plot_difference_map(
        edge_geom=edge_geom,
        diff_df=diff_df,
        value_col="vc_increase",
        title=f"v/c Increase Map | {scenario_label} - {base_label}",
        colorbar_label="v/c Increase",
        output_file=output_vc
    )


def visualize_scenario_comparisons():
    # =====================================
    # 1. Project paths
    # =====================================
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    # =====================================
    # 2. Baseline and three comparison scenarios
    #    Only edit file names here if your output names change.
    # =====================================
    baseline = {
        "key": "baseline",
        "label": "Baseline",
        "edge_output_file": os.path.join(
            results_dir,
            "a8_edge_output_od_reroute_calibrated.xml"
        )
    }

    comparisons = [
        {
            "key": "speedlimit_light_flooding",
            "label": "Speed limit / Light flooding",
            "edge_output_file": os.path.join(
                results_dir,
                "a8_flood_edge_output_od_reroute.xml"
            )
        },
        {
            "key": "laneclose_moderate_flooding",
            "label": "Lane close / Moderate flooding",
            "edge_output_file": os.path.join(
                results_dir,
                "a8_lane_close_edge_output_od_reroute.xml"
            )
        },
        {
            "key": "edgeclose_severe_flooding",
            "label": "Edge close / Severe flooding",
            "edge_output_file": os.path.join(
                results_dir,
                "a8_full_close_edge_output_od_reroute.xml"
            )
        },
    ]

    # =====================================
    # 3. Check network and baseline file
    # =====================================
    if not os.path.exists(net_file):
        print("[ERROR] Network file not found:")
        print(net_file)
        return

    if not os.path.exists(baseline["edge_output_file"]):
        print("[ERROR] Baseline edge output file not found:")
        print(baseline["edge_output_file"])
        return

    print("=====================================")
    print("Visualizing scenario comparison maps")
    print(f"Network file  : {net_file}")
    print(f"Results folder: {results_dir}")
    print(f"Baseline file : {baseline['edge_output_file']}")
    print("=====================================")

    # =====================================
    # 4. Load SUMO network only once
    # =====================================
    sumolib = load_sumo_tools()
    net = sumolib.net.readNet(net_file)
    edge_geom = build_edge_geometry(net)

    # =====================================
    # 5. Load and aggregate baseline only once
    # =====================================
    baseline_raw = parse_edge_output(baseline["edge_output_file"])

    if baseline_raw.empty:
        print("[ERROR] Baseline edge data is empty.")
        return

    baseline_agg = aggregate_metrics(baseline_raw, net)

    # =====================================
    # 6. Run three comparison pairs in one script
    # =====================================
    for scenario in comparisons:
        visualize_one_comparison(
            base_key=baseline["key"],
            base_label=baseline["label"],
            base_agg=baseline_agg,
            scenario_key=scenario["key"],
            scenario_label=scenario["label"],
            scenario_edge_file=scenario["edge_output_file"],
            net=net,
            edge_geom=edge_geom,
            results_dir=results_dir
        )

    print("=====================================")
    print("[SUCCESS] All scenario comparison maps completed.")
    print(f"Results folder: {results_dir}")
    print("=====================================")


if __name__ == "__main__":
    visualize_scenario_comparisons()
