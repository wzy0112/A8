# 定义 zone 和候选边
# 定义 OD 需求
# 生成 trips.xml
# 调用 duarouter 生成 rou.xml
import os
import random
import subprocess
import xml.etree.ElementTree as ET


# =========================================================
# 1. Helper: add vehicle type to generated routes
# =========================================================
def add_vehicle_type_to_routes(route_file):
    """
    Insert a realistic vehicle type and assign it to all vehicles.
    """
    if not os.path.exists(route_file):
        print("[ERROR] Route file not found for vType insertion:")
        print(route_file)
        return

    tree = ET.parse(route_file)
    root = tree.getroot()

    existing_vtype = root.find("./vType[@id='car']")
    if existing_vtype is None:
        vtype = ET.Element("vType", {
            "id": "car",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.5",
            "length": "5.0",
            "maxSpeed": "33.33"   # ~120 km/h
        })
        root.insert(0, vtype)

    vehicle_count = 0
    for vehicle in root.findall("vehicle"):
        vehicle.set("type", "car")
        vehicle_count += 1

    tree.write(route_file, encoding="utf-8", xml_declaration=True)

    print("[SUCCESS] Vehicle type added.")
    print(f"Vehicles updated: {vehicle_count}")


# =========================================================
# 2. Weighted random selection
# =========================================================
def weighted_choice(rng, candidates):
    """
    candidates: list of tuples -> (edge_id, weight)
    """
    edges = [item[0] for item in candidates]
    weights = [item[1] for item in candidates]
    return rng.choices(edges, weights=weights, k=1)[0]


# =========================================================
# 3. Validation
# =========================================================
def validate_zones(zones):
    for zone_name, zone_data in zones.items():
        if "in" not in zone_data or "out" not in zone_data:
            raise ValueError(f"Zone '{zone_name}' must contain 'in' and 'out' lists.")
        if not zone_data["in"]:
            raise ValueError(f"Zone '{zone_name}' has no inbound candidate edges.")
        if not zone_data["out"]:
            raise ValueError(f"Zone '{zone_name}' has no outbound candidate edges.")


def validate_od_matrix(od_matrix, zones):
    for interval_name, od_list in od_matrix.items():
        for item in od_list:
            origin = item["origin"]
            destination = item["destination"]
            demand = item["demand"]
            t_start = item["start"]
            t_end = item["end"]

            if origin not in zones:
                raise ValueError(f"Unknown origin zone '{origin}' in interval '{interval_name}'")
            if destination not in zones:
                raise ValueError(f"Unknown destination zone '{destination}' in interval '{interval_name}'")
            if demand < 0:
                raise ValueError(f"Negative demand in interval '{interval_name}'")
            if t_end <= t_start:
                raise ValueError(f"Invalid time interval in '{interval_name}': {t_start}-{t_end}")


# =========================================================
# 4. Departure times
# =========================================================
def generate_departure_times(start_time, end_time, demand, rng):
    """
    Uniform random departures in the interval.
    Later you can upgrade this to Poisson arrivals.
    """
    if demand <= 0:
        return []

    departures = [rng.uniform(start_time, end_time) for _ in range(demand)]
    departures.sort()
    return departures


# =========================================================
# 5. Generate trips.xml from zone-based OD
# =========================================================
def generate_trips_xml(trips_file, zones, od_matrix, seed=42):
    rng = random.Random(seed)

    validate_zones(zones)
    validate_od_matrix(od_matrix, zones)

    root = ET.Element("routes")
    trip_id_counter = 0
    total_trips = 0

    for interval_name, od_list in od_matrix.items():
        print(f"[INFO] Generating trips for interval: {interval_name}")

        for item in od_list:
            origin_zone = item["origin"]
            destination_zone = item["destination"]
            demand = item["demand"]
            start_time = item["start"]
            end_time = item["end"]

            if origin_zone == destination_zone:
                # First version: skip intra-zone trips
                continue

            departures = generate_departure_times(start_time, end_time, demand, rng)

            for depart in departures:
                # IMPORTANT:
                # origin uses "in" edges (vehicles ENTER the study area here)
                # destination uses "out" edges (vehicles LEAVE the study area here)
                from_edge = weighted_choice(rng, zones[origin_zone]["in"])
                to_edge = weighted_choice(rng, zones[destination_zone]["out"])

                retry = 0
                while from_edge == to_edge and retry < 10:
                    to_edge = weighted_choice(rng, zones[destination_zone]["out"])
                    retry += 1

                ET.SubElement(root, "trip", {
                    "id": f"trip_{trip_id_counter}",
                    "depart": f"{depart:.2f}",
                    "from": from_edge,
                    "to": to_edge,
                    "type": "car",
                    "departLane": "best",
                    "departSpeed": "max"
                })

                trip_id_counter += 1
                total_trips += 1

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(trips_file, encoding="utf-8", xml_declaration=True)

    print("[SUCCESS] Trips file generated.")
    print(f"Trips file: {trips_file}")
    print(f"Total trips: {total_trips}")


# =========================================================
# 6. Run duarouter
# =========================================================
def run_duarouter(sumo_home, net_file, trips_file, route_file):
    duarouter_path = os.path.join(sumo_home, "bin", "duarouter.exe")

    if not os.path.exists(duarouter_path):
        raise FileNotFoundError(f"duarouter.exe not found: {duarouter_path}")

    cmd = [
        duarouter_path,
        "-n", net_file,
        "-r", trips_file,
        "-o", route_file,
        "--ignore-errors",
        "--remove-loops",
        "--repair",
        "--seed", "42"
    ]

    print("=====================================")
    print("Running duarouter")
    print(f"Network file : {net_file}")
    print(f"Trips file   : {trips_file}")
    print(f"Route file   : {route_file}")
    print("=====================================")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] duarouter failed.")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("duarouter execution failed")

    print("[SUCCESS] duarouter completed.")
    if result.stdout.strip():
        print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)


# =========================================================
# 7. Main generation function
# =========================================================
def generate_od_traffic():
    # =====================================
    # Project paths
    # =====================================
    project_root = r"D:\SUMO_A9_Project"
    sumo_home = r"D:\Eclipse\SUMO"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    route_dir = os.path.join(project_root, "routes")
    trips_file = os.path.join(route_dir, "a8_od_7zones.trips.xml")
    route_file = os.path.join(route_dir, "a8_od_7zones.rou.xml")

    os.makedirs(route_dir, exist_ok=True)

    if not os.path.exists(net_file):
        print("[ERROR] Network file not found:")
        print(net_file)
        return

    # =====================================
    # Edge weight rule
    # A8 > A99 > others
    # =====================================
    W_A8 = 8
    W_A99 = 5
    W_OTHER = 1

    # =====================================
    # 7 zones
    # Each zone has:
    #   "in"  = candidate edges used to generate vehicles INTO the study area
    #   "out" = candidate edges used for vehicles LEAVING the study area
    # =====================================
    zones = {
        "WEST_NORTH": {
            "in": [
                ("1346182146", W_A8),
                ("239790189", W_OTHER),
                ("90776168#0", W_OTHER),
            ],
            "out": [
                ("280422002", W_A8),
                ("-239790189", W_OTHER),
                ("-90776168#0", W_OTHER),
            ]
        },

        "WEST_SOUTH": {
            "in": [
                ("-34916816#4", W_OTHER),
                ("555770913", W_OTHER),
                ("-32235541", W_OTHER),
            ],
            "out": [
                ("34916816#4", W_OTHER),
                ("-555770913", W_OTHER),
                ("32235541", W_OTHER),
            ]
        },

        "NORTH": {
            "in": [
                ("381493208#0", W_OTHER),
                ("-130897589", W_OTHER),
                ("545778341#1", W_OTHER),
            ],
            "out": [
                ("-381493208#0", W_OTHER),
                ("130897589", W_OTHER),
                ("-545778341#1", W_OTHER),
            ]
        },

        "SOUTH": {
            "in": [
                ("37483196#0", W_OTHER),
                ("-4055538#4", W_OTHER),
                ("-393288753#1", W_OTHER),
                ("4274287", W_A99),
            ],
            "out": [
                ("-37483196#0", W_OTHER),
                ("4055538#4", W_OTHER),
                ("393288753#0", W_OTHER),   # verify this ID
                ("895154059", W_A99),
            ]
        },

        "EAST": {
            "in": [
                ("144558389", W_A99),
                ("-519348421", W_OTHER),
            ],
            "out": [
                ("325030864", W_A99),
                ("217404013", W_OTHER),
            ]
        },

        "EAST_NORTH": {
            "in": [
                ("289220241", W_OTHER),
                ("-75124773#2", W_OTHER),
            ],
            "out": [
                ("428922462", W_OTHER),
                ("75124773#2", W_OTHER),
            ]
        },

        "EAST_SOUTH": {
            "in": [
                ("-30819906", W_OTHER),
                ("21458715#2", W_OTHER),
                ("310832991", W_OTHER),
                ("22931826#1", W_OTHER),
                ("3707399", W_OTHER),
            ],
            "out": [
                ("30819906", W_OTHER),
                ("152407669#3", W_OTHER),
                ("126760930#3", W_OTHER),
                ("146537576", W_OTHER),
                ("276604355", W_OTHER),
            ]
        }
    }

    # =====================================
    # OD demand
    #
    # Design principles from your description:
    # 1) WEST_NORTH <-> EAST_SOUTH interaction is strongest
    # 2) A8-related movement higher than A99, higher than others
    # 3) east-south area and the A8/A99 corridor are busy
    #
    # Here I give a reasonable first baseline for 1 hour,
    # split into two 30-min intervals.
    # You can tune these numbers later.
    # =====================================
    od_matrix = {
        "0_1800": [
            # strongest cross-corridor interactions
            {"origin": "WEST_NORTH", "destination": "EAST_SOUTH", "demand": 420, "start": 0, "end": 1800},
            {"origin": "EAST_SOUTH", "destination": "WEST_NORTH", "demand": 320, "start": 0, "end": 1800},

            # A8 / A99 related major flows
            {"origin": "WEST_NORTH", "destination": "EAST", "demand": 180, "start": 0, "end": 1800},
            {"origin": "WEST_NORTH", "destination": "EAST_NORTH", "demand": 140, "start": 0, "end": 1800},
            {"origin": "WEST_NORTH", "destination": "SOUTH", "demand": 120, "start": 0, "end": 1800},

            {"origin": "WEST_SOUTH", "destination": "EAST_SOUTH", "demand": 180, "start": 0, "end": 1800},
            {"origin": "WEST_SOUTH", "destination": "EAST", "demand": 100, "start": 0, "end": 1800},

            {"origin": "NORTH", "destination": "EAST_SOUTH", "demand": 110, "start": 0, "end": 1800},
            {"origin": "NORTH", "destination": "EAST", "demand": 90, "start": 0, "end": 1800},

            {"origin": "SOUTH", "destination": "EAST_SOUTH", "demand": 130, "start": 0, "end": 1800},
            {"origin": "SOUTH", "destination": "EAST", "demand": 100, "start": 0, "end": 1800},

            # reverse and local balancing flows
            {"origin": "EAST", "destination": "WEST_NORTH", "demand": 130, "start": 0, "end": 1800},
            {"origin": "EAST", "destination": "WEST_SOUTH", "demand": 70, "start": 0, "end": 1800},
            {"origin": "EAST_NORTH", "destination": "WEST_NORTH", "demand": 110, "start": 0, "end": 1800},
            {"origin": "EAST_NORTH", "destination": "NORTH", "demand": 60, "start": 0, "end": 1800},
            {"origin": "EAST_SOUTH", "destination": "SOUTH", "demand": 90, "start": 0, "end": 1800},
            {"origin": "EAST_SOUTH", "destination": "WEST_SOUTH", "demand": 120, "start": 0, "end": 1800},
        ],

        "1800_3600": [
            # higher demand in second half-hour
            {"origin": "WEST_NORTH", "destination": "EAST_SOUTH", "demand": 560, "start": 1800, "end": 3600},
            {"origin": "EAST_SOUTH", "destination": "WEST_NORTH", "demand": 420, "start": 1800, "end": 3600},

            {"origin": "WEST_NORTH", "destination": "EAST", "demand": 240, "start": 1800, "end": 3600},
            {"origin": "WEST_NORTH", "destination": "EAST_NORTH", "demand": 180, "start": 1800, "end": 3600},
            {"origin": "WEST_NORTH", "destination": "SOUTH", "demand": 150, "start": 1800, "end": 3600},

            {"origin": "WEST_SOUTH", "destination": "EAST_SOUTH", "demand": 240, "start": 1800, "end": 3600},
            {"origin": "WEST_SOUTH", "destination": "EAST", "demand": 120, "start": 1800, "end": 3600},

            {"origin": "NORTH", "destination": "EAST_SOUTH", "demand": 150, "start": 1800, "end": 3600},
            {"origin": "NORTH", "destination": "EAST", "demand": 120, "start": 1800, "end": 3600},

            {"origin": "SOUTH", "destination": "EAST_SOUTH", "demand": 180, "start": 1800, "end": 3600},
            {"origin": "SOUTH", "destination": "EAST", "demand": 130, "start": 1800, "end": 3600},

            {"origin": "EAST", "destination": "WEST_NORTH", "demand": 180, "start": 1800, "end": 3600},
            {"origin": "EAST", "destination": "WEST_SOUTH", "demand": 90, "start": 1800, "end": 3600},
            {"origin": "EAST_NORTH", "destination": "WEST_NORTH", "demand": 150, "start": 1800, "end": 3600},
            {"origin": "EAST_NORTH", "destination": "NORTH", "demand": 80, "start": 1800, "end": 3600},
            {"origin": "EAST_SOUTH", "destination": "SOUTH", "demand": 120, "start": 1800, "end": 3600},
            {"origin": "EAST_SOUTH", "destination": "WEST_SOUTH", "demand": 160, "start": 1800, "end": 3600},
        ]
    }

    # =====================================
    # Generate trips.xml
    # =====================================
    generate_trips_xml(
        trips_file=trips_file,
        zones=zones,
        od_matrix=od_matrix,
        seed=42
    )

    # =====================================
    # Convert trips -> routes
    # =====================================
    run_duarouter(
        sumo_home=sumo_home,
        net_file=net_file,
        trips_file=trips_file,
        route_file=route_file
    )

    # =====================================
    # Add vehicle type
    # =====================================
    add_vehicle_type_to_routes(route_file)

    print("=====================================")
    print("[DONE] OD-based traffic generation completed.")
    print(f"Trips file : {trips_file}")
    print(f"Route file : {route_file}")
    print("=====================================")


if __name__ == "__main__":
    generate_od_traffic()