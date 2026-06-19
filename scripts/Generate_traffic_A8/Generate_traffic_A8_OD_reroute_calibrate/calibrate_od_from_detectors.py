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

DETECTOR_ADD_FILE = os.path.join(PROJECT_ROOT, "detectors", "a8_detectors_merged.add.xml")

SUMO_E1_FILE = os.path.join(PROJECT_ROOT, "detectors", "a8_e1_output.xml")
SUMO_E1_SECONDARY_FILE = os.path.join(PROJECT_ROOT, "detectors", "a8_e1_output_secondary.xml")

BAYSIS_SVZ_FILE = os.path.join(
    PROJECT_ROOT,
    "detectors",
    "data from BAYSIS SVZ",
    "a8_detector_mapping_secondary_DTV.csv"
)

SECONDARY_WEIGHT = 0.5

BAST_A_FILE = os.path.join(PROJECT_ROOT, "detectors", "data from BASt", "2023_A_S.txt")
BAST_B_FILE = os.path.join(PROJECT_ROOT, "detectors", "data from BASt", "2023_B_S.txt")

OUTPUT_OD_CSV = os.path.join(PROJECT_ROOT, "routes", "calibrated_od_matrix_spsa.csv")
OUTPUT_COMPARISON_CSV = os.path.join(PROJECT_ROOT, "detectors", "spsa_detector_comparison.csv")
OUTPUT_HISTORY_CSV = os.path.join(PROJECT_ROOT, "detectors", "spsa_iteration_history.csv")
OUTPUT_HISTORY_PNG = os.path.join(PROJECT_ROOT, "detectors", "spsa_error_iteration.png")
OUTPUT_SCATTER_PNG = os.path.join(PROJECT_ROOT, "detectors", "spsa_observed_vs_simulated.png")
OUTPUT_OD_HEATMAP_PNG = os.path.join(PROJECT_ROOT, "routes", "final_od_heatmap.png")
OUTPUT_BAST_NONPEAK_SCATTER_PNG = os.path.join(PROJECT_ROOT, "detectors", "bast_nonpeak_observed_vs_simulated.png")
OUTPUT_BAST_PEAK_SCATTER_PNG = os.path.join(PROJECT_ROOT, "detectors", "bast_peak_observed_vs_simulated.png")

OUTPUT_SVZ_NONPEAK_SCATTER_PNG = os.path.join(PROJECT_ROOT, "detectors", "svz_nonpeak_observed_vs_simulated.png")
OUTPUT_SVZ_PEAK_SCATTER_PNG = os.path.join(PROJECT_ROOT, "detectors", "svz_peak_observed_vs_simulated.png")

OUTPUT_BAST_SCATTER_PNG = os.path.join(PROJECT_ROOT, "detectors", "bast_observed_vs_simulated.png")
OUTPUT_SVZ_SCATTER_PNG = os.path.join(PROJECT_ROOT, "detectors", "svz_observed_vs_simulated.png")
# =========================================================
# 1. Calibration settings
# =========================================================
SEED = 42
SIM_BEGIN = 0
SIM_END = 3600

TARGET_HOUR = 8      # BASt Stunde=08, i.e. 07:00-08:00
FAHRTZW = "w"        # w = normal working day

MAX_ITER = 60

LOWER_SCALE = 0.5 #0.2
UPPER_SCALE = 4 #8

# SPSA parameters
A = 20.0 #10
a = 0.05 #0.12 #0.25
c = 0.06 #0.12 #0.18
alpha = 0.602
gamma = 0.101
WARMUP_HOUR = 10   # 非高峰，例如 09:00–10:00
PEAK_HOUR = 8      # 高峰，例如 07:00–08:00

# =========================================================
# 2. Initial OD matrix
# =========================================================
BASE_OD = {
    "0_1800": [
        ("WEST_NORTH", "WEST_SOUTH", 22),
        ("WEST_NORTH", "NORTH", 15),
        ("WEST_NORTH", "SOUTH", 15),
        ("WEST_NORTH", "EAST", 195),
        ("WEST_NORTH", "EAST_NORTH", 165),
        ("WEST_NORTH", "EAST_SOUTH", 480),

        ("WEST_SOUTH", "WEST_NORTH", 22),
        ("WEST_SOUTH", "NORTH", 15),
        ("WEST_SOUTH", "SOUTH", 15),
        ("WEST_SOUTH", "EAST", 105),
        ("WEST_SOUTH", "EAST_NORTH", 15),
        ("WEST_SOUTH", "EAST_SOUTH", 180),

        ("NORTH", "WEST_NORTH", 30),
        ("NORTH", "WEST_SOUTH", 15),
        ("NORTH", "SOUTH", 15),
        ("NORTH", "EAST", 22),
        ("NORTH", "EAST_NORTH", 90),
        ("NORTH", "EAST_SOUTH", 22),

        ("SOUTH", "WEST_NORTH", 30),
        ("SOUTH", "WEST_SOUTH", 22),
        ("SOUTH", "NORTH", 15),
        ("SOUTH", "EAST", 30),
        ("SOUTH", "EAST_NORTH", 15),
        ("SOUTH", "EAST_SOUTH", 135),

        ("EAST", "WEST_NORTH", 270),
        ("EAST", "WEST_SOUTH", 150),
        ("EAST", "NORTH", 135),
        ("EAST", "SOUTH", 150),
        ("EAST", "EAST_NORTH", 30),
        ("EAST", "EAST_SOUTH", 30),

        ("EAST_NORTH", "WEST_NORTH", 210),
        ("EAST_NORTH", "WEST_SOUTH", 22),
        ("EAST_NORTH", "NORTH", 22),
        ("EAST_NORTH", "SOUTH", 15),
        ("EAST_NORTH", "EAST", 22),
        ("EAST_NORTH", "EAST_SOUTH", 22),

        ("EAST_SOUTH", "WEST_NORTH", 630),
        ("EAST_SOUTH", "WEST_SOUTH", 270),
        ("EAST_SOUTH", "NORTH", 165),
        ("EAST_SOUTH", "SOUTH", 195),
        ("EAST_SOUTH", "EAST", 30),
        ("EAST_SOUTH", "EAST_NORTH", 30),
    ],

    "1800_3600": [
        ("WEST_NORTH", "WEST_SOUTH", 30),
        ("WEST_NORTH", "NORTH", 22),
        ("WEST_NORTH", "SOUTH", 22),
        ("WEST_NORTH", "EAST", 270),
        ("WEST_NORTH", "EAST_NORTH", 225),
        ("WEST_NORTH", "EAST_SOUTH", 630),

        ("WEST_SOUTH", "WEST_NORTH", 30),
        ("WEST_SOUTH", "NORTH", 22),
        ("WEST_SOUTH", "SOUTH", 22),
        ("WEST_SOUTH", "EAST", 135),
        ("WEST_SOUTH", "EAST_NORTH", 22),
        ("WEST_SOUTH", "EAST_SOUTH", 240),

        ("NORTH", "WEST_NORTH", 38),
        ("NORTH", "WEST_SOUTH", 22),
        ("NORTH", "SOUTH", 22),
        ("NORTH", "EAST", 30),
        ("NORTH", "EAST_NORTH", 120),
        ("NORTH", "EAST_SOUTH", 30),

        ("SOUTH", "WEST_NORTH", 38),
        ("SOUTH", "WEST_SOUTH", 30),
        ("SOUTH", "NORTH", 22),
        ("SOUTH", "EAST", 38),
        ("SOUTH", "EAST_NORTH", 22),
        ("SOUTH", "EAST_SOUTH", 180),

        ("EAST", "WEST_NORTH", 360),
        ("EAST", "WEST_SOUTH", 180),
        ("EAST", "NORTH", 180),
        ("EAST", "SOUTH", 195),
        ("EAST", "EAST_NORTH", 38),
        ("EAST", "EAST_SOUTH", 38),

        ("EAST_NORTH", "WEST_NORTH", 270),
        ("EAST_NORTH", "WEST_SOUTH", 30),
        ("EAST_NORTH", "NORTH", 30),
        ("EAST_NORTH", "SOUTH", 22),
        ("EAST_NORTH", "EAST", 30),
        ("EAST_NORTH", "EAST_SOUTH", 30),

        ("EAST_SOUTH", "WEST_NORTH", 840),
        ("EAST_SOUTH", "WEST_SOUTH", 360),
        ("EAST_SOUTH", "NORTH", 225),
        ("EAST_SOUTH", "SOUTH", 270),
        ("EAST_SOUTH", "EAST", 38),
        ("EAST_SOUTH", "EAST_NORTH", 38),
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
    ET.SubElement(routing_elem, "device.rerouting.probability", {"value": "0.2"}) #0.82
    ET.SubElement(routing_elem, "device.rerouting.period", {"value": "300"})
    ET.SubElement(routing_elem, "weights.random-factor", {"value": "1.0"}) #1.2
    ET.SubElement(routing_elem, "device.rerouting.threshold.factor", {"value": "1.05"})
    ET.SubElement(routing_elem, "device.rerouting.threshold.constant", {"value": "60"})

    random_elem = ET.SubElement(root, "random_number")
    ET.SubElement(random_elem, "seed", {"value": str(SEED)})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(cfg_file, encoding="utf-8", xml_declaration=True)


def run_sumo():
    for f in [SUMO_E1_FILE, SUMO_E1_SECONDARY_FILE]:
        if os.path.exists(f):
            os.remove(f)

    cmd = [
        SUMO_EXE,
        "-c", CFG_FILE,
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--no-warnings", "true",
    ]

    subprocess.run(cmd, check=True)


# =========================================================
# 5. Detector reading
# =========================================================
def read_sumo_e1_direction_level(xml_files):
    rows = []
    pattern = re.compile(r"^(.+)_dir([12])_lane\d+$")

    for xml_file in xml_files:
        if not os.path.exists(xml_file):
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()

        for interval in root.findall("interval"):
            det_id = interval.get("id")
            flow = float(interval.get("flow", 0.0))
            begin = float(interval.get("begin", 0.0))

            match = pattern.match(det_id)
            if match is None:
                continue

            sensor_id = match.group(1)
            direction = int(match.group(2))

            sim_interval = "0_1800" if begin < 1800 else "1800_3600"

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

def read_secondary_observed_from_svz():
    df = pd.read_csv(BAYSIS_SVZ_FILE, dtype=str)

    rows = []

    for _, r in df.iterrows():
        if pd.isna(r["group_id"]) or pd.isna(r["direction_id"]) or pd.isna(r["DTV"]):
            continue

        # group_id example: 77339100_dir1
        # detector output id example: 77339100_dir1_lane0
        m = re.match(r"^(.+)_dir([12])$", str(r["group_id"]))
        if m is None:
            print(f"[WARNING] Cannot parse group_id: {r['group_id']}")
            continue

        sensor_id = m.group(1)
        direction = int(m.group(2))
        dtv = float(r["DTV"])

        rows.append({
            "interval": "0_1800",
            "sensor_id": sensor_id,
            "direction": direction,
            "obs_flow": dtv * 0.05 / 2.0,
            "weight": SECONDARY_WEIGHT,
            "source": "BAYSIS_SVZ"
        })

        rows.append({
            "interval": "1800_3600",
            "sensor_id": sensor_id,
            "direction": direction,
            "obs_flow": dtv * 0.10 / 2.0,
            "weight": SECONDARY_WEIGHT,
            "source": "BAYSIS_SVZ"
        })

    out = pd.DataFrame(rows)

    print("Secondary SVZ observed records:", len(out))
    print(out.head())

    return out
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
        raise RuntimeError("No common observed/SUMO detector-direction IDs.")

    merged["error"] = merged["sim_flow"] - merged["obs_flow"]
    merged["rel_error"] = merged["error"] / merged["obs_flow"].clip(lower=1)

    if "weight" not in merged.columns:
        merged["weight"] = 1.0

    rmse = np.sqrt(
        (merged["weight"] * merged["error"] ** 2).sum()
        / merged["weight"].sum()
    )

    nrmse = np.sqrt(
        (merged["weight"] * merged["rel_error"] ** 2).sum()
        / merged["weight"].sum()
    )

    return rmse, rmse, merged


def evaluate(theta, observed):
    theta = np.clip(theta, LOWER_SCALE, UPPER_SCALE)

    od = apply_theta_to_od(theta)

    generate_trips_xml(od, TRIPS_FILE)
    generate_sumo_config(CFG_FILE, TRIPS_FILE)
    run_sumo()

    simulated = read_sumo_e1_direction_level([
        SUMO_E1_FILE,
        SUMO_E1_SECONDARY_FILE
    ])

    nrmse, rmse, merged = compute_loss(observed, simulated)

    return nrmse, rmse, merged


# =========================================================
# 7. SPSA calibration
# =========================================================
def spsa_calibration():
    np.random.seed(SEED)

    bast_observed = read_bast_observed()
    bast_observed["weight"] = 1.0
    bast_observed["source"] = "BASt"

    secondary_observed = read_secondary_observed_from_svz()

    observed = pd.concat(
        [bast_observed, secondary_observed],
        ignore_index=True
    )

    n_params = sum(len(v) for v in BASE_OD.values())
    theta = np.ones(n_params)

    base_loss, base_rmse, base_merged = evaluate(theta, observed)

    simulated_check = read_sumo_e1_direction_level([
        SUMO_E1_FILE,
        SUMO_E1_SECONDARY_FILE
    ])

    check = observed.merge(
        base_merged[["interval", "sensor_id", "direction", "sim_flow"]],
        on=["interval", "sensor_id", "direction"],
        how="left",
        indicator=True
    )

    unmatched = check[check["_merge"] == "left_only"]

    unmatched.to_csv(
        os.path.join(PROJECT_ROOT, "detectors", "unmatched_observed_detectors.csv"),
        index=False,
        encoding="utf-8"
    )

    print("Observed records:", len(observed))
    print("Matched records:", len(check[check['_merge'] == 'both']))
    print("Unmatched records:", len(unmatched))
    print(unmatched[["source", "interval", "sensor_id", "direction", "obs_flow"]].head(30))

    print("====================================")
    print(f"[BASELINE] loss={base_loss:.4f}, RMSE={base_rmse:.2f}")
    print("====================================")

    best_loss = base_loss
    best_rmse = base_rmse
    best_theta = theta.copy()
    best_merged = base_merged.copy()
    best_iteration = 0

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

        print(
            f"[ITER {k}] "
            f"loss+={loss_plus:.2f}, "
            f"loss-={loss_minus:.2f}, "
            f"diff={loss_plus - loss_minus:.2f}"
        )

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
            best_iteration = k
            print(f"Best iteration: {best_iteration}")

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

        bast_df = best_merged[best_merged["source"] == "BASt"].copy()
        svz_df = best_merged[best_merged["source"] == "BAYSIS_SVZ"].copy()

        plot_observed_vs_simulated_scatter(
            bast_df,
            OUTPUT_BAST_SCATTER_PNG,
            "BASt Detectors: Observed vs Simulated"
        )

        plot_observed_vs_simulated_scatter(
            svz_df,
            OUTPUT_SVZ_SCATTER_PNG,
            "BAYSIS SVZ Detectors: Observed vs Simulated"
        )

        plot_observed_vs_simulated_scatter(
            bast_df[bast_df["interval"] == "0_1800"],
            OUTPUT_BAST_NONPEAK_SCATTER_PNG,
            "BASt Non-peak: Observed vs Simulated"
        )

        plot_observed_vs_simulated_scatter(
            bast_df[bast_df["interval"] == "1800_3600"],
            OUTPUT_BAST_PEAK_SCATTER_PNG,
            "BASt Peak: Observed vs Simulated"
        )

        plot_observed_vs_simulated_scatter(
            svz_df[svz_df["interval"] == "0_1800"],
            OUTPUT_SVZ_NONPEAK_SCATTER_PNG,
            "BAYSIS SVZ Non-peak: Observed vs Simulated"
        )

        plot_observed_vs_simulated_scatter(
            svz_df[svz_df["interval"] == "1800_3600"],
            OUTPUT_SVZ_PEAK_SCATTER_PNG,
            "BAYSIS SVZ Peak: Observed vs Simulated"
        )


    print("====================================")
    print("[DONE] SPSA OD calibration finished")
    print(f"Best RMSE loss: {best_loss:.2f}")
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
        plt.imshow(matrix.values, aspect="auto", cmap="RdYlGn_r")
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


def plot_observed_vs_simulated_scatter(df, output_png, title):
    if df.empty:
        print(f"[WARNING] Empty dataframe, skip plot: {output_png}")
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(df["obs_flow"], df["sim_flow"], alpha=0.7)

    max_val = max(df["obs_flow"].max(), df["sim_flow"].max())
    plt.plot([0, max_val], [0, max_val], linestyle="--", label="Perfect fit")

    plt.xlabel("Observed flow / 30 min")
    plt.ylabel("Simulated flow / 30 min")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


if __name__ == "__main__":
    spsa_calibration()