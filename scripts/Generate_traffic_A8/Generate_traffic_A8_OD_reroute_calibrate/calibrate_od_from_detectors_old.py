import os
import re
import copy
import random
import subprocess
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

#输入
#D:\SUMO_A9_Project\sumo\a8_corridor.net.xml
#D:\SUMO_A9_Project\detectors\a8_detectors.add.xml
#D:\SUMO_A9_Project\detectors\data from BASt\2023
#D:\SUMO_A9_Project\detectors\data from BAYSIS SVZ\Buchdruck_BY_2023_BLSKG_240409.xlsx

#输出
#D:\SUMO_A9_Project\routes\a8_od_spsa.trips.xml  每次iter的trips
#D:\SUMO_A9_Project\routes\a8_od_spsa.sumocfg 每次iter的config
#D:\SUMO_A9_Project\detectors\a8_e1_output.xml每次iter的detector output
# D:\SUMO_A9_Project\routes\calibrated_od_matrix_spsa.csv 最终优化后的两阶段 OD matrix
#D:\SUMO_A9_Project\detectors\spsa_detector_comparison.csv 最佳迭代下：BASt vs SUMO detector 对比

# 不做 detector–OD 一一对应
# 直接最小化 detector 实际值和模拟值差距
# 用 SPSA 修改已有 OD matrix
# 输出两个阶段的优化 OD

# reroute=300s=5min

# 15 个 BASt 单独 detector 文件
# → 读取非高峰和高峰小时
# → 各自除以 2
# → 与 SUMO 两个 1800s detector interval 比较
# → SPSA 优化两个阶段 OD
# → 输出 calibrated_od_matrix_spsa.csv

# =========================================================
# 0. Paths
# =========================================================
PROJECT_ROOT = r"D:\SUMO_A9_Project"
SUMO_EXE = r"D:\Eclipse\SUMO\bin\sumo.exe"

NET_FILE = os.path.join(PROJECT_ROOT, "sumo", "a8_corridor.net.xml")

TRIPS_FILE = os.path.join(PROJECT_ROOT, "routes", "a8_od_spsa.trips.xml")
CFG_FILE = os.path.join(PROJECT_ROOT, "routes", "a8_od_spsa.sumocfg")

DETECTOR_ADD_FILE = os.path.join(PROJECT_ROOT, "detectors", "a8_detectors.add.xml")
SUMO_E1_FILE = os.path.join(PROJECT_ROOT, "detectors", "a8_e1_output.xml")

BAST_A_FILE = os.path.join(PROJECT_ROOT, "detectors", "data from BASt", "2023_A_S.txt")
BAST_B_FILE = os.path.join(PROJECT_ROOT, "detectors", "data from BASt", "2023_B_S.txt")

OUTPUT_OD_CSV = os.path.join(PROJECT_ROOT, "routes", "calibrated_od_matrix_spsa.csv")
OUTPUT_COMPARISON_CSV = os.path.join(PROJECT_ROOT, "detectors", "spsa_detector_comparison.csv")
OUTPUT_HISTORY_CSV = os.path.join(PROJECT_ROOT, "detectors", "spsa_iteration_history.csv")
OUTPUT_HISTORY_PNG = os.path.join(PROJECT_ROOT, "detectors", "spsa_error_iteration.png")
OUTPUT_SCATTER_PNG = os.path.join(PROJECT_ROOT, "detectors", "spsa_observed_vs_simulated.png")
OUTPUT_OD_HEATMAP_PNG = os.path.join(PROJECT_ROOT, "routes", "final_od_heatmap.png")
OUTPUT_OBS_PEAK_NONPEAK_PNG = os.path.join(PROJECT_ROOT, "detectors", "observed_nonpeak_vs_peak.png")
OUTPUT_SIM_PEAK_NONPEAK_PNG = os.path.join(PROJECT_ROOT, "detectors", "simulated_nonpeak_vs_peak.png")

# =========================================================
# 1. Calibration settings
# =========================================================
SEED = 42
SIM_BEGIN = 0
SIM_END = 3600

TARGET_HOUR = 8      # BASt Stunde=08, i.e. 07:00-08:00
FAHRTZW = "w"        # w = normal working day

MAX_ITER = 60

LOWER_SCALE = 0.2
UPPER_SCALE = 8.0

# SPSA parameters
A = 10.0
a = 0.12 #0.25
c = 0.12 #0.18
alpha = 0.602
gamma = 0.101
WARMUP_HOUR = 10   # 非高峰，例如 09:00–10:00
PEAK_HOUR = 8      # 高峰，例如 07:00–08:00

# =========================================================
# 2. Initial OD matrix
# =========================================================
BASE_OD = {
    "0_1800": [
        ("WEST_NORTH", "WEST_SOUTH", 15),
        ("WEST_NORTH", "NORTH", 10),
        ("WEST_NORTH", "SOUTH", 10),
        ("WEST_NORTH", "EAST", 130),
        ("WEST_NORTH", "EAST_NORTH", 110),
        ("WEST_NORTH", "EAST_SOUTH", 320),

        ("WEST_SOUTH", "WEST_NORTH", 15),
        ("WEST_SOUTH", "NORTH", 10),
        ("WEST_SOUTH", "SOUTH", 10),
        ("WEST_SOUTH", "EAST", 70),
        ("WEST_SOUTH", "EAST_NORTH", 10),
        ("WEST_SOUTH", "EAST_SOUTH", 120),

        ("NORTH", "WEST_NORTH", 20),
        ("NORTH", "WEST_SOUTH", 10),
        ("NORTH", "SOUTH", 10),
        ("NORTH", "EAST", 15),
        ("NORTH", "EAST_NORTH", 60),
        ("NORTH", "EAST_SOUTH", 15),

        ("SOUTH", "WEST_NORTH", 20),
        ("SOUTH", "WEST_SOUTH", 15),
        ("SOUTH", "NORTH", 10),
        ("SOUTH", "EAST", 20),
        ("SOUTH", "EAST_NORTH", 10),
        ("SOUTH", "EAST_SOUTH", 90),

        ("EAST", "WEST_NORTH", 180),
        ("EAST", "WEST_SOUTH", 100),
        ("EAST", "NORTH", 90),
        ("EAST", "SOUTH", 100),
        ("EAST", "EAST_NORTH", 20),
        ("EAST", "EAST_SOUTH", 20),

        ("EAST_NORTH", "WEST_NORTH", 140),
        ("EAST_NORTH", "WEST_SOUTH", 15),
        ("EAST_NORTH", "NORTH", 15),
        ("EAST_NORTH", "SOUTH", 10),
        ("EAST_NORTH", "EAST", 15),
        ("EAST_NORTH", "EAST_SOUTH", 15),

        ("EAST_SOUTH", "WEST_NORTH", 420),
        ("EAST_SOUTH", "WEST_SOUTH", 180),
        ("EAST_SOUTH", "NORTH", 110),
        ("EAST_SOUTH", "SOUTH", 130),
        ("EAST_SOUTH", "EAST", 20),
        ("EAST_SOUTH", "EAST_NORTH", 20),
    ],

    "1800_3600": [
        ("WEST_NORTH", "WEST_SOUTH", 20),
        ("WEST_NORTH", "NORTH", 15),
        ("WEST_NORTH", "SOUTH", 15),
        ("WEST_NORTH", "EAST", 180),
        ("WEST_NORTH", "EAST_NORTH", 150),
        ("WEST_NORTH", "EAST_SOUTH", 420),

        ("WEST_SOUTH", "WEST_NORTH", 20),
        ("WEST_SOUTH", "NORTH", 15),
        ("WEST_SOUTH", "SOUTH", 15),
        ("WEST_SOUTH", "EAST", 90),
        ("WEST_SOUTH", "EAST_NORTH", 15),
        ("WEST_SOUTH", "EAST_SOUTH", 160),

        ("NORTH", "WEST_NORTH", 25),
        ("NORTH", "WEST_SOUTH", 15),
        ("NORTH", "SOUTH", 15),
        ("NORTH", "EAST", 20),
        ("NORTH", "EAST_NORTH", 80),
        ("NORTH", "EAST_SOUTH", 20),

        ("SOUTH", "WEST_NORTH", 25),
        ("SOUTH", "WEST_SOUTH", 20),
        ("SOUTH", "NORTH", 15),
        ("SOUTH", "EAST", 25),
        ("SOUTH", "EAST_NORTH", 15),
        ("SOUTH", "EAST_SOUTH", 120),

        ("EAST", "WEST_NORTH", 240),
        ("EAST", "WEST_SOUTH", 120),
        ("EAST", "NORTH", 120),
        ("EAST", "SOUTH", 130),
        ("EAST", "EAST_NORTH", 25),
        ("EAST", "EAST_SOUTH", 25),

        ("EAST_NORTH", "WEST_NORTH", 180),
        ("EAST_NORTH", "WEST_SOUTH", 20),
        ("EAST_NORTH", "NORTH", 20),
        ("EAST_NORTH", "SOUTH", 15),
        ("EAST_NORTH", "EAST", 20),
        ("EAST_NORTH", "EAST_SOUTH", 20),

        ("EAST_SOUTH", "WEST_NORTH", 560),
        ("EAST_SOUTH", "WEST_SOUTH", 240),
        ("EAST_SOUTH", "NORTH", 150),
        ("EAST_SOUTH", "SOUTH", 180),
        ("EAST_SOUTH", "EAST", 25),
        ("EAST_SOUTH", "EAST_NORTH", 25),
    ],
}


# =========================================================
# 3. Zone edge candidates
# =========================================================
EDGE_ZONES = {
    "WEST_NORTH": {
        "in": [("1346182146", 8), ("239790189", 1), ("90776168#0", 1)],
        "out": [("280422002", 8), ("-239790189", 1), ("-90776168#0", 1)],
    },
    "WEST_SOUTH": {
        "in": [("-34916816#4", 1), ("555770913", 1), ("-32235541", 1)],
        "out": [("34916816#4", 1), ("-555770913", 1), ("32235541", 1)],
    },
    "NORTH": {
        "in": [("381493208#0", 1), ("-130897589", 1), ("545778341#1", 1)],
        "out": [("-381493208#0", 1), ("130897589", 1), ("-545778341#1", 1)],
    },
    "SOUTH": {
        "in": [("37483196#0", 1), ("-4055538#4", 1), ("-393288753#1", 1), ("4274287", 5)],
        "out": [("-37483196#0", 1), ("4055538#4", 1), ("393288753#0", 1), ("895154059", 5)],
    },
    "EAST": {
        "in": [("144558389", 5), ("-519348421", 1)],
        "out": [("325030864", 5), ("217404013", 1)],
    },
    "EAST_NORTH": {
        "in": [("289220241", 1), ("-75124773#2", 1)],
        "out": [("428922462", 1), ("75124773#2", 1)],
    },
    "EAST_SOUTH": {
        "in": [("-30819906", 1), ("21458715#2", 1), ("310832991", 1), ("22931826#1", 1), ("3707399", 1)],
        "out": [("30819906", 1), ("152407669#3", 1), ("126760930#3", 1), ("146537576", 1), ("276604355", 1)],
    },
}


# =========================================================
# 4. OD / SUMO generation
# =========================================================
def weighted_choice(rng, candidates):
    edges = [x[0] for x in candidates]
    weights = [x[1] for x in candidates]
    return rng.choices(edges, weights=weights, k=1)[0]


def apply_theta_to_od(theta):
    od = copy.deepcopy(BASE_OD)
    idx = 0

    for interval_name in od:
        updated = []

        for origin, destination, demand in od[interval_name]:
            scale = float(theta[idx])
            new_demand = max(0, int(round(demand * scale)))
            updated.append((origin, destination, new_demand))
            idx += 1

        od[interval_name] = updated

    return od


def generate_trips_xml(od, trips_file):
    rng = random.Random(SEED)

    root = ET.Element("routes")

    ET.SubElement(root, "vType", {
        "id": "car",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "length": "5.0",
        "maxSpeed": "33.33",
    })

    all_trips = []
    trip_id = 0

    for interval_name, od_list in od.items():
        if interval_name == "0_1800":
            start, end = 0, 1800
        else:
            start, end = 1800, 3600

        for origin, destination, demand in od_list:
            departs = sorted(rng.uniform(start, end) for _ in range(demand))

            for depart in departs:
                from_edge = weighted_choice(rng, EDGE_ZONES[origin]["in"])
                to_edge = weighted_choice(rng, EDGE_ZONES[destination]["out"])

                all_trips.append({
                    "id": f"trip_{trip_id}",
                    "depart": depart,
                    "from": from_edge,
                    "to": to_edge,
                })
                trip_id += 1

    all_trips.sort(key=lambda x: x["depart"])

    for trip in all_trips:
        ET.SubElement(root, "trip", {
            "id": trip["id"],
            "depart": f"{trip['depart']:.2f}",
            "from": trip["from"],
            "to": trip["to"],
            "type": "car",
            "departLane": "best",
            "departSpeed": "max",
        })

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(trips_file, encoding="utf-8", xml_declaration=True)


def generate_sumo_config(cfg_file, trips_file):
    root = ET.Element("configuration")

    input_elem = ET.SubElement(root, "input")
    ET.SubElement(input_elem, "net-file", {"value": NET_FILE})
    ET.SubElement(input_elem, "route-files", {"value": trips_file})
    ET.SubElement(input_elem, "additional-files", {"value": DETECTOR_ADD_FILE})

    time_elem = ET.SubElement(root, "time")
    ET.SubElement(time_elem, "begin", {"value": str(SIM_BEGIN)})
    ET.SubElement(time_elem, "end", {"value": str(SIM_END)})

    processing_elem = ET.SubElement(root, "processing")
    ET.SubElement(processing_elem, "ignore-route-errors", {"value": "true"})
    ET.SubElement(processing_elem, "time-to-teleport", {"value": "-1"})
    ET.SubElement(processing_elem, "ignore-junction-blocker",{"value": "15"})

    routing_elem = ET.SubElement(root, "routing")
    ET.SubElement(routing_elem, "device.rerouting.probability", {"value": "0.82"})
    ET.SubElement(routing_elem, "device.rerouting.period", {"value": "300"})
    ET.SubElement(routing_elem, "weights.random-factor", {"value": "1.2"})
    ET.SubElement(routing_elem, "device.rerouting.threshold.factor", {"value": "1.05"})
    ET.SubElement(routing_elem, "device.rerouting.threshold.constant", {"value": "60"})

    random_elem = ET.SubElement(root, "random_number")
    ET.SubElement(random_elem, "seed", {"value": str(SEED)})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(cfg_file, encoding="utf-8", xml_declaration=True)


def run_sumo():
    if os.path.exists(SUMO_E1_FILE):
        os.remove(SUMO_E1_FILE)

    cmd = [
        SUMO_EXE,
        "-c", CFG_FILE,
        "--no-step-log", "true",
        "--duration-log.disable", "true",
    ]

    subprocess.run(cmd, check=True)


# =========================================================
# 5. Detector reading
# =========================================================
def read_sumo_e1_direction_level(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    rows = []
    pattern = re.compile(r"^(.+)_dir([12])_lane\d+$")

    for interval in root.findall("interval"):
        det_id = interval.get("id")
        flow = float(interval.get("flow", 0.0))
        begin = float(interval.get("begin", 0.0))

        match = pattern.match(det_id)
        if match is None:
            continue

        sensor_id = match.group(1)
        direction = int(match.group(2))

        if begin < 1800:
            sim_interval = "0_1800"
        else:
            sim_interval = "1800_3600"

        rows.append({
            "interval": sim_interval,
            "sensor_id": str(sensor_id),
            "direction": direction,
            "sim_flow": flow,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No direction-level detector IDs found.")

    return (
        df.groupby(["interval", "sensor_id", "direction"], as_index=False)
        .agg(sim_flow=("sim_flow", "sum"))
    )


def read_bast_file(txt_file, road_type, target_hour, fahrtzw):
    usecols = ["Zst", "Fahrtzw", "Stunde", "KFZ_R1", "KFZ_R2"]

    results = []

    for chunk in pd.read_csv(
        txt_file,
        sep=";",
        usecols=usecols,
        dtype=str,
        encoding="latin1",
        chunksize=500000,
        low_memory=False
    ):
        chunk["Stunde"] = pd.to_numeric(chunk["Stunde"], errors="coerce")

        chunk = chunk[
            (chunk["Fahrtzw"] == fahrtzw) &
            (chunk["Stunde"] == target_hour)
        ].copy()

        if chunk.empty:
            continue

        chunk["sensor_id"] = chunk["Zst"].astype(str)
        chunk["KFZ_R1"] = pd.to_numeric(chunk["KFZ_R1"], errors="coerce")
        chunk["KFZ_R2"] = pd.to_numeric(chunk["KFZ_R2"], errors="coerce")

        results.append(chunk[["sensor_id", "KFZ_R1", "KFZ_R2"]])

    if not results:
        raise RuntimeError(f"No BASt records found in {txt_file} for hour={target_hour}, Fahrtzw={fahrtzw}")

    df = pd.concat(results, ignore_index=True)

    r1 = df.groupby("sensor_id", as_index=False)["KFZ_R1"].mean()
    r1["direction"] = 1
    r1 = r1.rename(columns={"KFZ_R1": "obs_flow"})

    r2 = df.groupby("sensor_id", as_index=False)["KFZ_R2"].mean()
    r2["direction"] = 2
    r2 = r2.rename(columns={"KFZ_R2": "obs_flow"})

    out = pd.concat([
        r1[["sensor_id", "direction", "obs_flow"]],
        r2[["sensor_id", "direction", "obs_flow"]],
    ], ignore_index=True)

    out["road_type"] = road_type
    return out


BAST_DETECTOR_DIR = os.path.join(PROJECT_ROOT, "detectors", "data from BASt", "2023")


def read_one_bast_detector_file(file_path, target_hour, fahrtzw):
    df = pd.read_csv(
        file_path,
        sep=";",
        dtype=str,
        encoding="latin1"
    )

    # 如果 Excel 显示在一列里，代码仍然可以正常按 ; 拆开
    df["sensor_id"] = df["Zst"].astype(str)
    df["Stunde"] = pd.to_numeric(df["Stunde"], errors="coerce")
    df["KFZ_R1"] = pd.to_numeric(df["KFZ_R1"], errors="coerce")
    df["KFZ_R2"] = pd.to_numeric(df["KFZ_R2"], errors="coerce")

    df = df[
        (df["Fahrtzw"] == fahrtzw) &
        (df["Stunde"] == target_hour)
    ].copy()

    r1 = df.groupby("sensor_id", as_index=False)["KFZ_R1"].mean()
    r1["direction"] = 1
    r1 = r1.rename(columns={"KFZ_R1": "obs_flow"})

    r2 = df.groupby("sensor_id", as_index=False)["KFZ_R2"].mean()
    r2["direction"] = 2
    r2 = r2.rename(columns={"KFZ_R2": "obs_flow"})

    return pd.concat([
        r1[["sensor_id", "direction", "obs_flow"]],
        r2[["sensor_id", "direction", "obs_flow"]],
    ], ignore_index=True)


def read_bast_folder(target_hour, fahrtzw):
    all_rows = []

    for name in os.listdir(BAST_DETECTOR_DIR):
        if not name.lower().endswith(".csv"):
            continue

        if not name.lower().startswith("zst"):
            continue

        file_path = os.path.join(BAST_DETECTOR_DIR, name)

        try:
            one = read_one_bast_detector_file(file_path, target_hour, fahrtzw)
            all_rows.append(one)
            print(f"[OK] Read BASt detector file: {name}")
        except Exception as e:
            print(f"[WARNING] Failed to read {name}: {e}")

    if not all_rows:
        raise RuntimeError("No BASt detector CSV files were successfully read.")

    return pd.concat(all_rows, ignore_index=True)


def read_bast_observed():
    warmup = read_bast_folder(WARMUP_HOUR, FAHRTZW)
    warmup["interval"] = "0_1800"
    warmup["obs_flow"] = warmup["obs_flow"] / 2.0

    peak = read_bast_folder(PEAK_HOUR, FAHRTZW)
    peak["interval"] = "1800_3600"
    peak["obs_flow"] = peak["obs_flow"] / 2.0

    observed = pd.concat([warmup, peak], ignore_index=True)

    observed = observed.dropna(subset=["obs_flow"])
    observed = observed[observed["obs_flow"] >= 0]

    return observed[["interval", "sensor_id", "direction", "obs_flow"]]

# =========================================================
# 6. Loss function
# =========================================================
def compute_loss(observed, simulated):
    merged = observed.merge(
        simulated,
        on=["interval", "sensor_id", "direction"],
        how="inner"
    )

    if merged.empty:
        raise RuntimeError("No common BASt/SUMO detector-direction IDs.")

    merged["error"] = merged["sim_flow"] - merged["obs_flow"]

    rmse = np.sqrt(np.mean(merged["error"] ** 2))

    merged["rel_error"] = merged["error"] / merged["obs_flow"].clip(lower=1)
    nrmse = np.sqrt(np.mean(merged["rel_error"] ** 2))

    return nrmse, rmse, merged


def evaluate(theta, observed):
    theta = np.clip(theta, LOWER_SCALE, UPPER_SCALE)

    od = apply_theta_to_od(theta)

    generate_trips_xml(od, TRIPS_FILE)
    generate_sumo_config(CFG_FILE, TRIPS_FILE)
    run_sumo()

    simulated = read_sumo_e1_direction_level(SUMO_E1_FILE)

    nrmse, rmse, merged = compute_loss(observed, simulated)

    return nrmse, rmse, merged


# =========================================================
# 7. SPSA calibration
# =========================================================
def spsa_calibration():
    np.random.seed(SEED)

    observed = read_bast_observed()

    n_params = sum(len(v) for v in BASE_OD.values())
    theta = np.ones(n_params)

    base_loss, base_rmse, base_merged = evaluate(theta, observed)

    print("====================================")
    print(f"[BASELINE] loss={base_loss:.4f}, RMSE={base_rmse:.2f}")
    print("====================================")

    best_loss = base_loss
    best_rmse = base_rmse
    best_theta = theta.copy()
    best_merged = base_merged.copy()

    history = []

    history.append({
        "iteration": 0,
        "loss": base_loss,
        "rmse": base_rmse,
        "best_loss": best_loss,
        "best_rmse": best_rmse
    })

    print("====================================")
    print("SPSA OD calibration started")
    print(f"Number of OD parameters: {n_params}")
    print(f"Observed detector-direction records: {len(observed)}")
    print("====================================")

    for k in range(1, MAX_ITER + 1):
        ak = a / ((k + A) ** alpha)
        ck = c / (k ** gamma)

        delta = np.random.choice([-1, 1], size=n_params)

        theta_plus = np.clip(theta + ck * delta, LOWER_SCALE, UPPER_SCALE)
        theta_minus = np.clip(theta - ck * delta, LOWER_SCALE, UPPER_SCALE)

        loss_plus, rmse_plus, _ = evaluate(theta_plus, observed)
        loss_minus, rmse_minus, _ = evaluate(theta_minus, observed)

        ghat = (loss_plus - loss_minus) / (2.0 * ck * delta)

        theta = theta - ak * ghat
        theta = np.clip(theta, LOWER_SCALE, UPPER_SCALE)

        current_loss, current_rmse, current_merged = evaluate(theta, observed)

        print(
            f"[ITER {k:02d}] "
            f"loss={current_loss:.4f}, "
            f"RMSE={current_rmse:.2f}, "
            f"scale_min={theta.min():.2f}, "
            f"scale_max={theta.max():.2f}"
        )

        if current_loss < best_loss:
            best_loss = current_loss
            best_rmse = current_rmse
            best_theta = theta.copy()
            best_merged = current_merged.copy()

        history.append({
            "iteration": k,
            "loss": current_loss,
            "rmse": current_rmse,
            "best_loss": best_loss,
            "best_rmse": best_rmse
        })

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_HISTORY_CSV, index=False, encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["iteration"], history_df["rmse"], marker="o", label="Current RMSE")
    plt.plot(history_df["iteration"], history_df["best_rmse"], marker="s", label="Best RMSE")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("SPSA OD Calibration Error over Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_HISTORY_PNG, dpi=300)
    plt.close()

    final_od = apply_theta_to_od(best_theta)

    save_calibrated_od(final_od, OUTPUT_OD_CSV)

    plot_final_od_heatmap(final_od, OUTPUT_OD_HEATMAP_PNG)

    if best_merged is not None:
        best_merged.to_csv(OUTPUT_COMPARISON_CSV, index=False, encoding="utf-8")
        plt.figure(figsize=(6, 6))
        plt.scatter(best_merged["obs_flow"], best_merged["sim_flow"], alpha=0.7)

        max_val = max(best_merged["obs_flow"].max(), best_merged["sim_flow"].max())
        plt.plot([0, max_val], [0, max_val], linestyle="--", label="Perfect fit")

        plt.xlabel("Observed BASt flow")
        plt.ylabel("Simulated SUMO flow")
        plt.title("Observed vs Simulated Detector Flow")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_SCATTER_PNG, dpi=300)
        plt.close()

        plot_peak_nonpeak_detector_flow(
            best_merged,
            value_col="obs_flow",
            output_png=OUTPUT_OBS_PEAK_NONPEAK_PNG,
            title="Observed BASt Flow: Non-peak vs Peak"
        )

        plot_peak_nonpeak_detector_flow(
            best_merged,
            value_col="sim_flow",
            output_png=OUTPUT_SIM_PEAK_NONPEAK_PNG,
            title="Simulated SUMO Flow: Non-peak vs Peak"
        )



    print("====================================")
    print("[DONE] SPSA OD calibration finished")
    print(f"Best normalized loss: {best_loss:.4f}")
    print(f"Best RMSE: {best_rmse:.2f}")
    print(f"Saved calibrated OD: {OUTPUT_OD_CSV}")
    print(f"Saved comparison: {OUTPUT_COMPARISON_CSV}")
    print("====================================")


def save_calibrated_od(od, output_csv):
    rows = []

    for interval_name, od_list in od.items():
        for origin, destination, demand in od_list:
            rows.append({
                "interval": interval_name,
                "origin": origin,
                "destination": destination,
                "demand": demand,
            })

    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")

def plot_peak_nonpeak_detector_flow(df, value_col, output_png, title):
    temp = df.copy()
    temp["detector_dir"] = temp["sensor_id"].astype(str) + "_R" + temp["direction"].astype(str)

    pivot = temp.pivot_table(
        index="detector_dir",
        columns="interval",
        values=value_col,
        aggfunc="mean"
    )

    pivot = pivot.rename(columns={
        "0_1800": "Non-peak",
        "1800_3600": "Peak"
    })

    pivot = pivot.sort_index()

    plt.figure(figsize=(12, 5))
    x = np.arange(len(pivot.index))

    plt.plot(x, pivot["Non-peak"], marker="o", label="Non-peak")
    plt.plot(x, pivot["Peak"], marker="s", label="Peak")

    plt.xticks(x, pivot.index, rotation=90)
    plt.ylabel("Flow / 30 min")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

def plot_final_od_heatmap(od, output_png):
    zones = ["WEST_NORTH", "WEST_SOUTH", "NORTH", "SOUTH", "EAST", "EAST_NORTH", "EAST_SOUTH"]

    for interval_name, od_list in od.items():
        matrix = pd.DataFrame(0, index=zones, columns=zones)

        for origin, destination, demand in od_list:
            matrix.loc[origin, destination] = demand

        plt.figure(figsize=(8, 6))
        plt.imshow(matrix.values, aspect="auto")
        plt.colorbar(label="OD demand")
        plt.xticks(range(len(zones)), zones, rotation=45, ha="right")
        plt.yticks(range(len(zones)), zones)
        plt.title(f"Final Calibrated OD Matrix: {interval_name}")
        plt.xlabel("Destination")
        plt.ylabel("Origin")

        for i in range(len(zones)):
            for j in range(len(zones)):
                plt.text(j, i, str(matrix.values[i, j]), ha="center", va="center", fontsize=8)

        plt.tight_layout()
        out = output_png.replace(".png", f"_{interval_name}.png")
        plt.savefig(out, dpi=300)
        plt.close()

if __name__ == "__main__":
    spsa_calibration()