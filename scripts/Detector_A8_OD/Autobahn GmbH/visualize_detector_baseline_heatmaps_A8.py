import os
import xml.etree.ElementTree as ET

import pandas as pd
import matplotlib.pyplot as plt


def parse_e1_output(xml_file):
    """
    Parse SUMO E1 detector output XML into a DataFrame.
    """
    if not os.path.exists(xml_file):
        print("[ERROR] E1 output file not found:")
        print(xml_file)
        return pd.DataFrame()

    tree = ET.parse(xml_file)
    root = tree.getroot()

    records = []

    for interval in root.findall("interval"):
        record = {
            "begin": float(interval.get("begin", -1)),
            "end": float(interval.get("end", -1)),
            "sensor_id": interval.get("id", ""),
            "nVehContrib": float(interval.get("nVehContrib", -1)),
            "flow": float(interval.get("flow", -1)),
            "occupancy": float(interval.get("occupancy", -1)),
            "speed": float(interval.get("speed", -1)),
            "harmonicMeanSpeed": float(interval.get("harmonicMeanSpeed", -1)),
            "length": float(interval.get("length", -1)),
            "nVehEntered": float(interval.get("nVehEntered", -1)),
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


def load_mapping_csv(mapping_csv):
    """
    Load detector mapping CSV for lane/edge reference and sensor ordering.
    """
    if not os.path.exists(mapping_csv):
        print("[ERROR] Mapping CSV not found:")
        print(mapping_csv)
        return pd.DataFrame()

    df = pd.read_csv(mapping_csv)
    return df


def prepare_baseline_dataframe(e1_df, mapping_df):
    """
    Merge detector output with mapping and clean invalid values.
    """
    if e1_df.empty:
        return pd.DataFrame()

    if not mapping_df.empty:
        df = e1_df.merge(mapping_df, on="sensor_id", how="left")
    else:
        df = e1_df.copy()

    # Replace SUMO "no data" values
    for col in ["speed", "harmonicMeanSpeed", "length"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = pd.NA

    # Flow and occupancy should not be negative, but clean just in case
    for col in ["flow", "occupancy", "nVehContrib", "nVehEntered"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = pd.NA

    # Create a density proxy from occupancy
    df["density_proxy"] = df["occupancy"]

    return df


def build_sensor_order(df):
    """
    Build a stable sensor order for heatmaps.
    Prefer edge/lane ordering if mapping exists, otherwise use sensor_id.
    """
    if "edge_id" in df.columns and "lane_id" in df.columns:
        sensor_order = (
            df[["sensor_id", "edge_id", "lane_id", "lane_pos"]]
            .drop_duplicates()
            .sort_values(by=["edge_id", "lane_id", "lane_pos"], na_position="last")
        )["sensor_id"].tolist()
    else:
        sensor_order = sorted(df["sensor_id"].dropna().unique().tolist())

    return sensor_order


def make_heatmap(df, value_col, sensor_order, output_file, title, colorbar_label):
    """
    Create and save a heatmap.
    Rows: sensors
    Columns: time intervals
    """
    if df.empty:
        print(f"[WARNING] No data available for {value_col} heatmap.")
        return

    plot_df = df.copy()

    # Time label in minutes
    plot_df["time_min"] = (plot_df["begin"] / 60.0).round(1)

    pivot = plot_df.pivot_table(
        index="sensor_id",
        columns="time_min",
        values=value_col,
        aggfunc="mean"
    )

    # Reindex sensor order
    pivot = pivot.reindex(sensor_order)

    if pivot.empty:
        print(f"[WARNING] Pivot table is empty for {value_col}.")
        return

    plt.figure(figsize=(16, 8))
    plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label=colorbar_label)

    plt.title(title)
    plt.xlabel("Time (min)")
    plt.ylabel("Sensor ID")

    # X ticks
    x_positions = range(len(pivot.columns))
    x_labels = [str(c) for c in pivot.columns]
    step_x = max(1, len(x_labels) // 15)
    plt.xticks(
        ticks=list(x_positions)[::step_x],
        labels=x_labels[::step_x],
        rotation=45
    )

    # Y ticks
    y_positions = range(len(pivot.index))
    y_labels = pivot.index.tolist()
    step_y = max(1, len(y_labels) // 20)
    plt.yticks(
        ticks=list(y_positions)[::step_y],
        labels=y_labels[::step_y]
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SUCCESS] Saved heatmap: {output_file}")


def visualize_baseline_heatmaps():
    # =====================================
    # 1. Project paths
    # =====================================
    project_root = r"D:\SUMO_A9_Project"

    e1_output_file = os.path.join(project_root, "detectors", "a8_e1_output.xml")
    mapping_csv = os.path.join(project_root, "detectors", "a8_detector_mapping.csv")

    results_dir = os.path.join(project_root, "detectors")
    os.makedirs(results_dir, exist_ok=True)

    output_flow = os.path.join(results_dir, "flow_heatmap.png")
    output_speed = os.path.join(results_dir, "speed_heatmap.png")
    output_density = os.path.join(results_dir, "density_proxy_heatmap.png")
    output_csv = os.path.join(results_dir, "baseline_detector_data.csv")

    # =====================================
    # 2. Load data
    # =====================================
    print("=====================================")
    print("Loading baseline detector output")
    print(f"E1 output file : {e1_output_file}")
    print(f"Mapping CSV    : {mapping_csv}")
    print("=====================================")

    e1_df = parse_e1_output(e1_output_file)
    mapping_df = load_mapping_csv(mapping_csv)

    if e1_df.empty:
        print("[ERROR] No detector data found in E1 output.")
        return

    df = prepare_baseline_dataframe(e1_df, mapping_df)

    if df.empty:
        print("[ERROR] Processed detector DataFrame is empty.")
        return

    # Save merged dataset
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved merged detector data: {output_csv}")

    # =====================================
    # 3. Sensor ordering
    # =====================================
    sensor_order = build_sensor_order(df)

    # =====================================
    # 4. Heatmaps
    # =====================================
    make_heatmap(
        df=df,
        value_col="flow",
        sensor_order=sensor_order,
        output_file=output_flow,
        title="Baseline Flow Heatmap",
        colorbar_label="Flow (veh/h)"
    )

    make_heatmap(
        df=df,
        value_col="speed",
        sensor_order=sensor_order,
        output_file=output_speed,
        title="Baseline Speed Heatmap",
        colorbar_label="Speed (m/s)"
    )

    make_heatmap(
        df=df,
        value_col="density_proxy",
        sensor_order=sensor_order,
        output_file=output_density,
        title="Baseline Density Proxy Heatmap (from Occupancy)",
        colorbar_label="Occupancy / Density Proxy"
    )

    print("=====================================")
    print("[SUCCESS] Baseline heatmap visualization completed.")
    print(f"Results folder: {results_dir}")
    print("=====================================")


if __name__ == "__main__":
    visualize_baseline_heatmaps()