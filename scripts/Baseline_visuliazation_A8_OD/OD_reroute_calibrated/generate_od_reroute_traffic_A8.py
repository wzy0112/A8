#根据最好OD生成config
import os
import random
import xml.etree.ElementTree as ET
import pandas as pd

# =========================================================
# 0. Paths
# =========================================================
PROJECT_ROOT = r"D:\SUMO_A9_Project"

NET_FILE = os.path.join(PROJECT_ROOT, "sumo", "a8_corridor.net.xml")

OD_CSV = os.path.join(PROJECT_ROOT, "routes", "calibrated_od_matrix_spsa.csv")

TRIPS_FILE = os.path.join(PROJECT_ROOT, "routes", "a8_od_spsa.trips.xml")
CFG_FILE = os.path.join(PROJECT_ROOT, "routes", "a8_od_spsa.sumocfg")

DETECTOR_ADD_FILE = os.path.join(
    PROJECT_ROOT,
    "detectors",
    "a8_detectors_merged.add.xml"
)

# =========================================================
# 1. Same simulation parameters as SPSA code
# =========================================================
SEED = 42
SIM_BEGIN = 0
SIM_END = 3600

# =========================================================
# 2. Same zone edge candidates as SPSA code
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
# 3. Read best OD matrix from calibrated_od_matrix_spsa.csv
# =========================================================
def read_best_od_from_csv(od_csv):
    df = pd.read_csv(od_csv)

    required_cols = {"interval", "origin", "destination", "demand"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"OD CSV missing columns: {missing}")

    od = {}

    for interval_name, group in df.groupby("interval", sort=False):
        od[interval_name] = []

        for _, row in group.iterrows():
            od[interval_name].append((
                str(row["origin"]),
                str(row["destination"]),
                int(round(row["demand"]))
            ))

    return od

# =========================================================
# 4. Same random edge choice as SPSA code
# =========================================================
def weighted_choice(rng, candidates):
    edges = [x[0] for x in candidates]
    weights = [x[1] for x in candidates]
    return rng.choices(edges, weights=weights, k=1)[0]

# =========================================================
# 5. Generate trips.xml from best OD
# =========================================================
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
        elif interval_name == "1800_3600":
            start, end = 1800, 3600
        else:
            raise ValueError(f"Unknown interval: {interval_name}")

        for origin, destination, demand in od_list:
            if origin not in EDGE_ZONES:
                raise ValueError(f"Unknown origin zone: {origin}")
            if destination not in EDGE_ZONES:
                raise ValueError(f"Unknown destination zone: {destination}")

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

    print(f"[OK] Saved trips file: {trips_file}")
    print(f"[OK] Total trips: {len(all_trips)}")

# =========================================================
# 6. Generate sumocfg with same parameters as SPSA code
# =========================================================
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
    ET.SubElement(processing_elem, "ignore-junction-blocker", {"value": "15"})

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

    print(f"[OK] Saved SUMO config: {cfg_file}")

# =========================================================
# 7. Main
# =========================================================
if __name__ == "__main__":
    best_od = read_best_od_from_csv(OD_CSV)

    generate_trips_xml(best_od, TRIPS_FILE)

    generate_sumo_config(CFG_FILE, TRIPS_FILE)

    print("====================================")
    print("[DONE] Best OD trips and SUMO config generated")
    print(f"OD CSV: {OD_CSV}")
    print(f"Trips: {TRIPS_FILE}")
    print(f"Config: {CFG_FILE}")
    print("====================================")