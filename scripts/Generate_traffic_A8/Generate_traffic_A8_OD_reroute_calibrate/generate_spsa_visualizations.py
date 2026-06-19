import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Input files
# =========================
import os
import matplotlib.image as mpimg

WORK_DIR = r"D:\SUMO_A9_Project\detectors\7 30个加上all edge 结果加上dir手动修正 1800 迭代15次输出结果\3"

COMPARISON_FILE = os.path.join(WORK_DIR, "spsa_detector_comparison.csv")
INITIAL_OD_FILE = os.path.join(WORK_DIR, "initial_od_matrix_spsa.csv")
FINAL_OD_FILE = os.path.join(WORK_DIR, "calibrated_od_matrix_spsa.csv")


# Output folder
OUTPUT_DIR = r"D:\SUMO_A9_Project\detectors\730个加上all edge 结果加上dir手动修正 1800 迭代15次输出结果\3\spsa_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# According to your request:
# nonpeak_observed_vs_simulated.png -> interval 1800_3600
# peak_observed_vs_simulated.png    -> interval 0_1800
# If your project convention is the opposite, simply swap these two values.
PEAK_INTERVAL = "1800_3600"
NONPEAK_INTERVAL = "0_1800"

# Optional: set to "BASt" or "BAYSIS_SVZ" if you only want one source.
# Keep None to plot all detectors in spsa_detector_comparison.csv.
SOURCE_FILTER = None

ZONES = [
    "WEST_NORTH", "WEST_SOUTH", "NORTH", "SOUTH",
    "EAST", "EAST_NORTH", "EAST_SOUTH"
]


def read_comparison(path):
    df = pd.read_csv(path)

    required = {"interval", "sensor_id", "direction", "obs_flow", "sim_flow"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in comparison CSV: {missing}")

    if SOURCE_FILTER is not None and "source" in df.columns:
        df = df[df["source"] == SOURCE_FILTER].copy()

    # Build readable detector label.
    if "source" in df.columns:
        df["detector_dir"] = (
            df["source"].astype(str) + "_" +
            df["sensor_id"].astype(str) + "_R" +
            df["direction"].astype(str)
        )
    else:
        df["detector_dir"] = (
            df["sensor_id"].astype(str) + "_R" + df["direction"].astype(str)
        )

    # In case there are duplicated detector rows, aggregate them.
    # Classify detector source / road type
    def classify_road_type(row):
        text = (
                str(row.get("source", "")) + "_" +
                str(row.get("sensor_id", "")) + "_" +
                str(row.get("detector_dir", ""))
        ).lower()

        if "bast" in text:
            return "BASt / Motorway"
        elif "baysis" in text or "svz" in text:
            return "BAYSIS / Federal road"
        else:
            return "Unknown"

    df["road_type"] = df.apply(classify_road_type, axis=1)

    df = (
        df.groupby(["interval", "detector_dir", "road_type"], as_index=False)
        .agg(obs_flow=("obs_flow", "mean"), sim_flow=("sim_flow", "mean"))
    )

    return df


def plot_interval_observed_vs_simulated(df, interval_name, output_png, title):
    data = df[df["interval"] == interval_name].copy()
    if data.empty:
        print(f"[WARNING] No rows found for interval {interval_name}; skipped {output_png}")
        return

    data = data.sort_values("detector_dir")
    x = np.arange(len(data))

    plt.figure(figsize=(18, 5))
    plt.plot(x, data["obs_flow"], marker="o", linewidth=2, label="Observed flow")
    plt.plot(x, data["sim_flow"], marker="s", linewidth=2, label="Simulated flow")

    plt.xticks(x, data["detector_dir"], rotation=90)
    plt.xlabel("Detector direction")
    plt.ylabel("Flow / 30 min")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved: {output_png}")

def plot_peak_nonpeak_together(df, output_png):
    nonpeak = df[df["interval"] == NONPEAK_INTERVAL].copy()
    peak = df[df["interval"] == PEAK_INTERVAL].copy()

    if nonpeak.empty or peak.empty:
        print("[WARNING] Peak or non-peak data is empty.")
        return

    nonpeak = nonpeak.sort_values("detector_dir")
    peak = peak.sort_values("detector_dir")

    labels = nonpeak["detector_dir"].tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(22, 6))

    plt.plot(x, nonpeak["obs_flow"], marker="o", linewidth=2,
             label=f"Non-peak observed ({NONPEAK_INTERVAL})")
    plt.plot(x, nonpeak["sim_flow"], marker="s", linestyle="--", linewidth=2,
             label=f"Non-peak simulated ({NONPEAK_INTERVAL})")

    plt.plot(x, peak["obs_flow"], marker="o", linewidth=2,
             label=f"Peak observed ({PEAK_INTERVAL})")
    plt.plot(x, peak["sim_flow"], marker="s", linestyle="--", linewidth=2,
             label=f"Peak simulated ({PEAK_INTERVAL})")

    plt.xticks(x, labels, rotation=90)
    plt.xlabel("Detector direction")
    plt.ylabel("Flow / 30 min")
    plt.title("Observed vs Simulated Detector Flow: Non-peak and Peak")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved: {output_png}")

def plot_observed_vs_simulated_scatter(df, output_png):
    if df.empty:
        print(f"[WARNING] Empty dataframe; skipped {output_png}")
        return

    max_val = max(df["obs_flow"].max(), df["sim_flow"].max())
    min_val = min(df["obs_flow"].min(), df["sim_flow"].min(), 0)

    plt.figure(figsize=(7, 7))
    plt.scatter(df["obs_flow"], df["sim_flow"], alpha=0.7)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect fit")

    plt.xlabel("Observed flow")
    plt.ylabel("Simulated flow")
    plt.title("Observed vs Simulated Detector Flow")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved: {output_png}")

def plot_scatter_by_road_type(df, output_png):
    max_val = max(df["obs_flow"].max(), df["sim_flow"].max())
    min_val = 0

    plt.figure(figsize=(8, 8))

    styles = {
        "BASt / Motorway": {"color": "royalblue", "marker": "o"},
        "BAYSIS / Federal road": {"color": "orange", "marker": "s"},
        "Unknown": {"color": "gray", "marker": "x"},
    }

    for road_type, style in styles.items():
        data = df[df["road_type"] == road_type]
        if data.empty:
            continue

        plt.scatter(
            data["obs_flow"],
            data["sim_flow"],
            color=style["color"],
            marker=style["marker"],
            alpha=0.75,
            s=70,
            label=road_type
        )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "--",
        color="black",
        linewidth=2,
        label="Perfect fit"
    )

    plt.xlabel("Observed flow")
    plt.ylabel("Simulated flow")
    plt.title("Observed vs Simulated Flow by Road Type")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved: {output_png}")

def plot_scatter_by_road_type_and_period(df, output_png):
    max_val = max(df["obs_flow"].max(), df["sim_flow"].max())
    min_val = 0

    plt.figure(figsize=(8, 8))

    # BASt non-peak
    subset = df[
        (df["road_type"] == "BASt / Motorway") &
        (df["interval"] == NONPEAK_INTERVAL)
        ]

    plt.scatter(
        subset["obs_flow"],
        subset["sim_flow"],
        color="tab:blue",
        marker="o",
        s=80,
        alpha=0.75,
        label="Non-peak BASt"
    )

    # BAYSIS non-peak
    subset = df[
        (df["road_type"] == "BAYSIS / Federal road") &
        (df["interval"] == NONPEAK_INTERVAL)
        ]

    plt.scatter(
        subset["obs_flow"],
        subset["sim_flow"],
        color="tab:orange",
        marker="s",
        s=80,
        alpha=0.75,
        label="Non-peak BAYSIS"
    )

    # BASt peak
    subset = df[
        (df["road_type"] == "BASt / Motorway") &
        (df["interval"] == PEAK_INTERVAL)
        ]

    plt.scatter(
        subset["obs_flow"],
        subset["sim_flow"],
        color="tab:green",
        marker="o",
        s=80,
        alpha=0.75,
        label="Peak BASt"
    )

    # BAYSIS peak
    subset = df[
        (df["road_type"] == "BAYSIS / Federal road") &
        (df["interval"] == PEAK_INTERVAL)
        ]

    plt.scatter(
        subset["obs_flow"],
        subset["sim_flow"],
        color="tab:red",
        marker="s",
        s=80,
        alpha=0.75,
        label="Peak BAYSIS"
    )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "--",
        color="black",
        linewidth=2,
        label="Perfect fit"
    )

    plt.xlabel("Observed flow")
    plt.ylabel("Simulated flow")
    plt.title("Observed vs Simulated Flow by Road Type and Period")
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved: {output_png}")

def plot_peak_nonpeak_scatter(df, output_png):

    nonpeak = df[df["interval"] == NONPEAK_INTERVAL]
    peak = df[df["interval"] == PEAK_INTERVAL]

    max_val = max(df["obs_flow"].max(), df["sim_flow"].max())
    min_val = 0

    plt.figure(figsize=(8, 8))

    # Non-peak
    plt.scatter(
        nonpeak["obs_flow"],
        nonpeak["sim_flow"],
        marker='o',
        color='royalblue',
        alpha=0.7,
        s=70,
        label=f"Non-peak ({NONPEAK_INTERVAL})"
    )

    # Peak
    plt.scatter(
        peak["obs_flow"],
        peak["sim_flow"],
        marker='^',
        color='crimson',
        alpha=0.7,
        s=70,
        label=f"Peak ({PEAK_INTERVAL})"
    )

    # y=x
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        '--',
        color='black',
        linewidth=2,
        label='Perfect fit'
    )

    plt.xlabel("Observed flow")
    plt.ylabel("Simulated flow")
    plt.title("Observed vs Simulated Flow by Period")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved: {output_png}")

def combine_four_scatter_plots(output_png):
    image_files = [
        ("Overall", os.path.join(OUTPUT_DIR, "spsa_observed_vs_simulated.png")),
        ("Peak vs Non-peak", os.path.join(OUTPUT_DIR, "spsa_observed_vs_simulated_by_period.png")),
        ("Road Type", os.path.join(OUTPUT_DIR, "spsa_observed_vs_simulated_by_road_type.png")),
        ("Road Type and Period", os.path.join(OUTPUT_DIR, "spsa_observed_vs_simulated_by_road_type_and_period.png")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    for ax, (title, img_path) in zip(axes.flat, image_files):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved: {output_png}")

def read_od(path):
    df = pd.read_csv(path)
    required = {"interval", "origin", "destination", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in OD CSV: {missing}")
    return df


def od_to_matrix(od_df, interval_name):
    data = od_df[od_df["interval"] == interval_name].copy()
    matrix = pd.DataFrame(0, index=ZONES, columns=ZONES, dtype=float)

    for _, row in data.iterrows():
        origin = row["origin"]
        destination = row["destination"]
        demand = row["demand"]
        if origin in matrix.index and destination in matrix.columns:
            matrix.loc[origin, destination] = demand

    return matrix

# 保存四个OD heatmap initial final  nonpeak  peak 分别生成四个heatmap
# def plot_od_heatmap(od_df, interval_name, output_png, title, vmin=None, vmax=None):
#     matrix = od_to_matrix(od_df, interval_name)
#
#     plt.figure(figsize=(8, 6))
#
#     # RdYlGn_r = green for low values, yellow for middle, red for high values.
#     im = plt.imshow(matrix.values, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
#     cbar = plt.colorbar(im)
#     cbar.set_label("OD demand")
#
#     plt.xticks(range(len(ZONES)), ZONES, rotation=45, ha="right")
#     plt.yticks(range(len(ZONES)), ZONES)
#     plt.xlabel("Destination")
#     plt.ylabel("Origin")
#     plt.title(title)
#
#     for i in range(len(ZONES)):
#         for j in range(len(ZONES)):
#             value = int(matrix.iloc[i, j])
#             plt.text(j, i, str(value), ha="center", va="center", fontsize=8)
#
#     plt.tight_layout()
#     plt.savefig(output_png, dpi=300)
#     plt.close()
#     print(f"Saved: {output_png}")

# 4 个heatmap合在一起
def plot_all_od_heatmaps(initial_od, final_od, output_png):
    zones = ["WEST_NORTH", "WEST_SOUTH", "NORTH", "SOUTH",
             "EAST", "EAST_NORTH", "EAST_SOUTH"]

    datasets = [
        (initial_od, "0_1800", "Initial OD: 0–1800 s"),
        (initial_od, "1800_3600", "Initial OD: 1800–3600 s"),
        (final_od, "0_1800", "Final OD: 0–1800 s"),
        (final_od, "1800_3600", "Final OD: 1800–3600 s"),
    ]

    global_min = min(initial_od["demand"].min(), final_od["demand"].min())
    global_max = max(initial_od["demand"].max(), final_od["demand"].max())

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)

    im = None

    for ax, (df, interval_name, title) in zip(axes.flat, datasets):
        temp = df[df["interval"] == interval_name]

        matrix = pd.DataFrame(0, index=zones, columns=zones)

        for _, row in temp.iterrows():
            matrix.loc[row["origin"], row["destination"]] = row["demand"]

        im = ax.imshow(
            matrix.values,
            cmap="RdYlGn_r",
            vmin=global_min,
            vmax=global_max
        )

        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("Destination", fontsize=11, labelpad=10)
        ax.set_ylabel("Origin", fontsize=11, labelpad=10)

        ax.set_xticks(range(len(zones)))
        ax.set_xticklabels(zones, rotation=45, ha="right", fontsize=9)

        ax.set_yticks(range(len(zones)))
        ax.set_yticklabels(zones, fontsize=9)

        for i in range(len(zones)):
            for j in range(len(zones)):
                value = int(matrix.values[i, j])
                ax.text(
                    j, i, str(value),
                    ha="center",
                    va="center",
                    fontsize=8
                )

    fig.suptitle("Initial and Final OD Matrices", fontsize=18)

    cbar = fig.colorbar(
        im,
        ax=axes,
        location='right',
        shrink=0.8,
        pad=0.03
    )

    cbar.set_label("OD Demand", fontsize=12)

    plt.savefig(output_png, dpi=200)
    plt.close()

def main():
    comparison = read_comparison(COMPARISON_FILE)

    plot_interval_observed_vs_simulated(
        comparison,
        NONPEAK_INTERVAL,
        os.path.join(OUTPUT_DIR, "nonpeak_observed_vs_simulated.png"),
        f"Non-peak Observed vs Simulated Detector Flow ({NONPEAK_INTERVAL})"
    )

    plot_interval_observed_vs_simulated(
        comparison,
        PEAK_INTERVAL,
        os.path.join(OUTPUT_DIR, "peak_observed_vs_simulated.png"),
        f"Peak Observed vs Simulated Detector Flow ({PEAK_INTERVAL})"
    )

    plot_peak_nonpeak_together(
        comparison,
        os.path.join(OUTPUT_DIR, "peak_nonpeak_observed_vs_simulated.png")
    )

    plot_observed_vs_simulated_scatter(
        comparison,
        os.path.join(OUTPUT_DIR, "spsa_observed_vs_simulated.png")
    )

    plot_peak_nonpeak_scatter(
        comparison,
        os.path.join(
            OUTPUT_DIR,
            "spsa_observed_vs_simulated_by_period.png"
        )
    )

    plot_observed_vs_simulated_scatter(
        comparison,
        os.path.join(OUTPUT_DIR, "spsa_observed_vs_simulated.png")
    )

    plot_scatter_by_road_type(
        comparison,
        os.path.join(OUTPUT_DIR, "spsa_observed_vs_simulated_by_road_type.png")
    )

    plot_scatter_by_road_type_and_period(
        comparison,
        os.path.join(OUTPUT_DIR, "spsa_observed_vs_simulated_by_road_type_and_period.png")
    )

    combine_four_scatter_plots(
        os.path.join(OUTPUT_DIR, "spsa_observed_vs_simulated_combined.png")
    )

    initial_od = read_od(INITIAL_OD_FILE)
    final_od = read_od(FINAL_OD_FILE)

    # Use one shared color scale for all OD heatmaps, so initial and final are comparable.
    global_max = max(initial_od["demand"].max(), final_od["demand"].max())
    vmin, vmax = 0, global_max

# 保存四个OD heatmap initial final  nonpeak  peak 分别生成四个heatmap
    # plot_od_heatmap(
    #     initial_od,
    #     "0_1800",
    #     os.path.join(OUTPUT_DIR, "initial_od_heatmap_0_1800.png"),
    #     "Initial OD Matrix: 0-1800 s",
    #     vmin=vmin,
    #     vmax=vmax
    # )
    #
    # plot_od_heatmap(
    #     initial_od,
    #     "1800_3600",
    #     os.path.join(OUTPUT_DIR, "initial_od_heatmap_1800_3600.png"),
    #     "Initial OD Matrix: 1800-3600 s",
    #     vmin=vmin,
    #     vmax=vmax
    # )
    #
    # plot_od_heatmap(
    #     final_od,
    #     "0_1800",
    #     os.path.join(OUTPUT_DIR, "final_od_heatmap_0_1800.png"),
    #     "Final OD Matrix: 0-1800 s",
    #     vmin=vmin,
    #     vmax=vmax
    # )
    #
    # plot_od_heatmap(
    #     final_od,
    #     "1800_3600",
    #     os.path.join(OUTPUT_DIR, "final_od_heatmap_1800_3600.png"),
    #     "Final OD Matrix: 1800-3600 s",
    #     vmin=vmin,
    #     vmax=vmax
    # )
#4 个heatmap合在一起
    plot_all_od_heatmaps(
        initial_od,
        final_od,
        os.path.join(OUTPUT_DIR, "od_heatmaps_initial_final.png")
    )

if __name__ == "__main__":
    main()
