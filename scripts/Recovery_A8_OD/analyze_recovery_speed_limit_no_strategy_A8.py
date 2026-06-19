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
                "time_min": begin / 60.0,
                "edge_id": edge.get("id", ""),
                "sampledSeconds": float(edge.get("sampledSeconds", 0)),
                "density": float(edge.get("density", 0)),
                "speed": float(edge.get("speed", -1)),
                "entered": float(edge.get("entered", 0)),
                "left": float(edge.get("left", 0)),
                "traveltime": float(edge.get("traveltime", -1)),
            })

    return pd.DataFrame(records)


def prepare_metrics(df):
    work_df = df.copy()

    interval_seconds = work_df["end"] - work_df["begin"]
    interval_seconds = interval_seconds.replace(0, np.nan)

    work_df["flow"] = work_df["entered"] / interval_seconds * 3600.0

    work_df.loc[work_df["speed"] < 0, "speed"] = np.nan
    work_df.loc[work_df["traveltime"] < 0, "traveltime"] = np.nan

    return work_df


def compute_baseline_reference(baseline_df, reference_start=0, reference_end=1200):
    """
    Compute edge-level baseline reference before flooding.
    """
    ref_df = baseline_df[
        (baseline_df["begin"] >= reference_start) &
        (baseline_df["end"] <= reference_end)
    ].copy()

    baseline_ref = (
        ref_df.groupby("edge_id", as_index=False)
        .agg({
            "speed": "mean",
            "density": "mean",
            "flow": "mean"
        })
        .rename(columns={
            "speed": "speed_base",
            "density": "density_base",
            "flow": "flow_base"
        })
    )

    return baseline_ref

def compute_weighted_baseline_speed(baseline_df, reference_start=600, reference_end=1200):
    ref_df = baseline_df[
        (baseline_df["begin"] >= reference_start) &
        (baseline_df["end"] <= reference_end)
    ].dropna(subset=["speed"]).copy()

    if ref_df.empty:
        return np.nan

    weights = ref_df["sampledSeconds"]

    if weights.sum() <= 0:
        return ref_df["speed"].mean()

    return (ref_df["speed"] * weights).sum() / weights.sum()

def compute_time_series_recovery(flood_df, baseline_ref, baseline_network_speed):
    merged = flood_df.merge(baseline_ref, on="edge_id", how="inner")

    # Edge-level ratios
    merged["speed_ratio"] = merged["speed"] / merged["speed_base"]
    merged["density_ratio"] = merged["density"] / merged["density_base"].replace(0, np.nan)
    merged["flow_ratio"] = merged["flow"] / merged["flow_base"].replace(0, np.nan)

    def weighted_mean_speed(group):
        valid = group.dropna(subset=["speed"]).copy()
        if valid.empty:
            return np.nan

        weights = valid["sampledSeconds"]

        if weights.sum() <= 0:
            return valid["speed"].mean()

        return (valid["speed"] * weights).sum() / weights.sum()

    # Network-level time series
    network_ts = (
        merged.groupby(["begin", "end", "time_min"])
        .apply(lambda g: pd.Series({
            "network_speed": weighted_mean_speed(g),
            "speed_ratio": weighted_mean_speed(g) / baseline_network_speed,
            "flow_ratio": g["flow_ratio"].mean(),
            "density_ratio": g["density_ratio"].mean(),
            "speed": g["speed"].mean(),
            "density": g["density"].mean(),
            "flow": g["flow"].mean()
        }))
        .reset_index()
    )

    return merged, network_ts


def find_network_recovery_time(network_ts, recovery_start=2400, threshold=0.9, stable_minutes=5):
    """
    Recovery time = first time after recovery_start when speed ratio >= threshold
    for stable_minutes consecutive minutes.
    """
    after = network_ts[network_ts["begin"] >= recovery_start].copy()

    if after.empty:
        return None

    stable_steps = stable_minutes

    values = after["speed_ratio"].values
    times = after["begin"].values

    for i in range(0, len(values) - stable_steps + 1):
        window = values[i:i + stable_steps]

        if np.all(window >= threshold):
            recovery_time_abs = times[i]
            recovery_duration = recovery_time_abs - recovery_start
            return recovery_time_abs, recovery_duration

    return None


def compute_edge_recovery_times(merged_df, recovery_start=2400, threshold=0.9, stable_minutes=5):
    """
    Compute recovery time for each edge based on speed recovery ratio.
    """
    results = []

    for edge_id, group in merged_df.groupby("edge_id"):
        g = group[group["begin"] >= recovery_start].sort_values("begin").copy()

        if g.empty:
            results.append({
                "edge_id": edge_id,
                "recovery_time_abs": np.nan,
                "recovery_duration": np.nan,
                "recovered": False
            })
            continue

        values = g["speed_ratio"].values
        times = g["begin"].values

        recovered = False

        for i in range(0, len(values) - stable_minutes + 1):
            window = values[i:i + stable_minutes]

            if np.all(window >= threshold):
                recovery_time_abs = times[i]
                recovery_duration = recovery_time_abs - recovery_start

                results.append({
                    "edge_id": edge_id,
                    "recovery_time_abs": recovery_time_abs,
                    "recovery_duration": recovery_duration,
                    "recovered": True
                })

                recovered = True
                break

        if not recovered:
            results.append({
                "edge_id": edge_id,
                "recovery_time_abs": np.nan,
                "recovery_duration": np.nan,
                "recovered": False
            })

    return pd.DataFrame(results)


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


def plot_recovery_curves(network_ts, output_file, recovery_start=2400):
    plt.figure(figsize=(12, 6))

    plt.plot(network_ts["time_min"], network_ts["speed_ratio"], label="Speed recovery ratio")
    plt.plot(network_ts["time_min"], network_ts["flow_ratio"], label="Flow recovery ratio")
    plt.plot(network_ts["time_min"], network_ts["density_ratio"], label="Density ratio")

    plt.axvline(recovery_start / 60.0, linestyle="--", label="Recovery starts")
    plt.axhline(0.9, linestyle="--", label="90% speed recovery threshold")

    plt.xlabel("Time (min)")
    plt.ylabel("Ratio to baseline")
    plt.title("Recovery Curves: Light Flooding Speed Limit, No Strategy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SUCCESS] Saved recovery curve: {output_file}")


def plot_recovery_time_map(edge_geom, recovery_df, output_file):
    import matplotlib.colors as mcolors

    value_map = dict(zip(recovery_df["edge_id"], recovery_df["recovery_duration"]))

    segments = []
    values = []

    for edge_id, segs in edge_geom.items():
        val = value_map.get(edge_id, 0)

        if pd.isna(val):
            continue

        for seg in segs:
            segments.append(seg)
            values.append(val / 60.0)

    if not segments:
        print("[WARNING] No recovered edge segments available for recovery map.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    vmin = 0
    vmax = np.percentile(values, 95)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    lc = LineCollection(
        segments,
        array=np.array(values),
        linewidths=2.5,
        cmap="RdYlGn_r",
        norm=norm
    )

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Recovery Time Map: Light Flooding Speed Limit, No Strategy")

    cbar = plt.colorbar(lc, ax=ax, shrink=0.8)
    cbar.set_label("Recovery duration after peak end (min)")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Saved recovery time map: {output_file}")


def main():
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")

    baseline_edge_file = os.path.join(
        project_root,
        "heat_edges",
        "a8_edge_output_od_reroute.xml"
    )

    flood_edge_file = os.path.join(
        project_root,
        "heat_edges",
        "a8_flood_edge_output_od_reroute.xml"
    )

    output_dir = os.path.join(
        project_root,
        "heat_edges",
        "recovery_speed_limit_no_strategy"
    )
    os.makedirs(output_dir, exist_ok=True)

    recovery_curve_png = os.path.join(output_dir, "recovery_curves_speed_limit_no_strategy.png")
    recovery_map_png = os.path.join(output_dir, "recovery_time_map_speed_limit_no_strategy.png")
    network_ts_csv = os.path.join(output_dir, "network_recovery_timeseries.csv")
    edge_recovery_csv = os.path.join(output_dir, "edge_recovery_summary.csv")
    summary_txt = os.path.join(output_dir, "recovery_summary.txt")

    # Flooding timeline
    baseline_reference_start = 0
    baseline_reference_end = 1200

    recovery_start = 2400
    threshold = 0.9
    stable_minutes = 5

    print("=====================================")
    print("Recovery analysis: light flooding speed limit, no strategy")
    print(f"Baseline file : {baseline_edge_file}")
    print(f"Flood file    : {flood_edge_file}")
    print(f"Output folder : {output_dir}")
    print("=====================================")

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

    sumolib = load_sumo_tools()
    net = sumolib.net.readNet(net_file)
    edge_geom = build_edge_geometry(net)

    baseline_raw = parse_edge_output(baseline_edge_file)
    flood_raw = parse_edge_output(flood_edge_file)

    baseline_df = prepare_metrics(baseline_raw)
    flood_df = prepare_metrics(flood_raw)

    baseline_ref = compute_baseline_reference(
        baseline_df,
        reference_start=baseline_reference_start,
        reference_end=baseline_reference_end
    )

    baseline_network_speed = compute_weighted_baseline_speed(
        baseline_df,
        reference_start=baseline_reference_start,
        reference_end=baseline_reference_end
    )

    print(f"[INFO] Weighted baseline network speed: {baseline_network_speed:.2f} m/s")

    merged_df, network_ts = compute_time_series_recovery(
        flood_df,
        baseline_ref,
        baseline_network_speed
    )

    network_ts.to_csv(network_ts_csv, index=False, encoding="utf-8")

    network_recovery = find_network_recovery_time(
        network_ts,
        recovery_start=recovery_start,
        threshold=threshold,
        stable_minutes=stable_minutes
    )

    edge_recovery_df = compute_edge_recovery_times(
        merged_df,
        recovery_start=recovery_start,
        threshold=threshold,
        stable_minutes=stable_minutes
    )

    edge_recovery_df.to_csv(edge_recovery_csv, index=False, encoding="utf-8")

    plot_recovery_curves(
        network_ts=network_ts,
        output_file=recovery_curve_png,
        recovery_start=recovery_start
    )

    plot_recovery_time_map(
        edge_geom=edge_geom,
        recovery_df=edge_recovery_df,
        output_file=recovery_map_png
    )

    recovered_count = int(edge_recovery_df["recovered"].sum())
    total_count = len(edge_recovery_df)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Recovery analysis: light flooding speed limit, no strategy\n")
        f.write("=========================================================\n")
        f.write(f"Recovery threshold: speed ratio >= {threshold}\n")
        f.write(f"Stability duration: {stable_minutes} minutes\n")
        f.write(f"Recovery start time: {recovery_start} s\n")
        f.write(f"Recovered edges: {recovered_count} / {total_count}\n")

        if network_recovery is None:
            f.write("Network recovery time: Not recovered within simulation period\n")
        else:
            abs_time, duration = network_recovery
            f.write(f"Network recovery absolute time: {abs_time:.0f} s\n")
            f.write(f"Network recovery duration: {duration / 60.0:.2f} min\n")

    print("=====================================")
    print("[SUCCESS] Recovery analysis completed.")
    print(f"Recovered edges: {recovered_count} / {total_count}")

    if network_recovery is None:
        print("Network recovery: Not recovered within simulation period")
    else:
        abs_time, duration = network_recovery
        print(f"Network recovery absolute time: {abs_time:.0f} s")
        print(f"Network recovery duration: {duration / 60.0:.2f} min")

    print(f"Recovery curve : {recovery_curve_png}")
    print(f"Recovery map   : {recovery_map_png}")
    print(f"Summary file   : {summary_txt}")
    print("=====================================")


if __name__ == "__main__":
    main()