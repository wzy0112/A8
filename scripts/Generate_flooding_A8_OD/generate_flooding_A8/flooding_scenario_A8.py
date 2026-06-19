import os
import sys
import subprocess
import xml.etree.ElementTree as ET

SUMO_HOME = r"D:\Eclipse\SUMO"
if SUMO_HOME not in os.environ:
    os.environ["SUMO_HOME"] = SUMO_HOME

tools_path = os.path.join(SUMO_HOME, "tools")
if tools_path not in sys.path:
    sys.path.append(tools_path)

import traci
import sumolib


def create_edge_data_additional_file(additional_file, edge_output_file, freq=60):
    root = ET.Element("additional")

    ET.SubElement(
        root,
        "edgeData",
        {
            "id": "edge_metrics",
            "file": edge_output_file,
            "freq": str(freq),
            "excludeEmpty": "true",
        }
    )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(additional_file, encoding="utf-8", xml_declaration=True)


def get_neighboring_edges(net, x, y, radius):
    try:
        return net.getNeighboringEdges(x, y, radius, includeJunctions=False)
    except TypeError:
        return net.getNeighboringEdges(x, y, radius)


def select_flood_edges_by_center(net, center_lon, center_lat, search_radius_m=250):
    """
    Automatically select candidate flood edges around a geographic center.
    This avoids manual edge ID selection.
    """
    x, y = net.convertLonLat2XY(center_lon, center_lat)
    neighbors = get_neighboring_edges(net, x, y, search_radius_m)

    selected_edges = []

    for edge, dist in neighbors:
        if edge.isSpecial():
            continue

        edge_id = edge.getID()

        # Skip internal edges
        if edge_id.startswith(":"):
            continue

        # Prefer longer, drivable, motorway-like edges
        if edge.getLength() < 50:
            continue

        lanes = edge.getLanes()
        if not lanes:
            continue

        try:
            max_speed = max(l.getSpeed() for l in lanes)
        except Exception:
            max_speed = 0

        if max_speed < 15:
            continue

        selected_edges.append((edge_id, dist, edge.getLength(), max_speed))

    # sort by distance first
    selected_edges.sort(key=lambda item: item[1])

    edge_ids = []
    seen = set()

    for edge_id, dist, length, max_speed in selected_edges:
        if edge_id not in seen:
            seen.add(edge_id)
            edge_ids.append(edge_id)

    return edge_ids


def set_edge_speed_safely(edge_ids, speed_value):
    for edge_id in edge_ids:
        try:
            traci.edge.setMaxSpeed(edge_id, speed_value)
        except traci.exceptions.TraCIException:
            pass


def run_flooding_simulation():
    # =====================================
    # 1. Project paths
    # =====================================
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    route_file = os.path.join(project_root, "routes", "a8_random.rou.xml")
    detector_additional_file = os.path.join(project_root, "detectors", "a8_detectors.add.xml")
    label_additional_file = os.path.join(project_root, "labels", "a8_labels.add.xml")
    gui_settings_file = os.path.join(project_root, "labels", "a8_gui_settings.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    edge_data_additional_file = os.path.join(results_dir, "a8_flood_edge_data.add.xml")
    edge_output_file = os.path.join(results_dir, "a8_flood_edge_output.xml")

    sumo_gui_path = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")

    # =====================================
    # 2. Basic checks
    # =====================================
    required_files = [
        net_file,
        route_file,
        detector_additional_file,
        label_additional_file,
        gui_settings_file,
        sumo_gui_path,
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print("[ERROR] Required file not found:")
            print(file_path)
            return

    # =====================================
    # 3. Create edgeData output file
    # =====================================
    create_edge_data_additional_file(
        additional_file=edge_data_additional_file,
        edge_output_file=edge_output_file,
        freq=60
    )

    additional_files = ",".join([
        detector_additional_file,
        label_additional_file,
        edge_data_additional_file
    ])

    # =====================================
    # 4. Flood scenario settings
    # =====================================
    gui_delay_ms = 10

    # ---- Flooding timeline ----
    flood_onset_start = 1200
    flood_peak_start = 1500
    flood_peak_end = 2400
    flood_recovery_end = 3000

    # ---- Speed settings ----
    normal_speed = 33.33      # 120 km/h
    peak_flood_speed = 8.33   # 30 km/h

    # ---- Flood zone center ----
    # Replace with your chosen Sulzemoos-area coordinates if needed
    flood_center_lon = 11.43
    flood_center_lat = 48.26
    flood_radius_m = 500

    # =====================================
    # 5. Launch SUMO with TraCI
    # =====================================
    print("=====================================")
    print("Starting flooding simulation")
    print(f"Network file         : {net_file}")
    print(f"Route file           : {route_file}")
    print(f"Additional files     : {additional_files}")
    print(f"GUI settings         : {gui_settings_file}")
    print(f"Edge output file     : {edge_output_file}")
    print(f"Flood center (lon)   : {flood_center_lon}")
    print(f"Flood center (lat)   : {flood_center_lat}")
    print(f"Flood radius (m)     : {flood_radius_m}")
    print("=====================================")

    sumo_cmd = [
        sumo_gui_path,
        "-n", net_file,
        "-r", route_file,
        "-a", additional_files,
        "--gui-settings-file", gui_settings_file,
        "--start",
        "--delay", str(gui_delay_ms),
        "--no-warnings",
        "--duration-log.statistics"
    ]

    traci.start(sumo_cmd)

    # =====================================
    # 6. Read network and auto-select flood edges
    # =====================================
    net = sumolib.net.readNet(net_file)
    flood_edges = select_flood_edges_by_center(
        net=net,
        center_lon=flood_center_lon,
        center_lat=flood_center_lat,
        search_radius_m=flood_radius_m
    )

    print("[INFO] Flood edges selected:")
    for edge_id in flood_edges:
        try:
            print("   ", edge_id, "lanes =", traci.edge.getLaneNumber(edge_id))
        except Exception:
            print("   ", edge_id)

    if not flood_edges:
        print("[WARNING] No flood edges were selected.")
        print("[WARNING] Simulation will continue without speed intervention.")

    # =====================================
    # 7. Simulation loop
    # =====================================
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # Phase A: normal
        if step < flood_onset_start:
            pass

        # Phase B: onset (gradual speed reduction)
        elif flood_onset_start <= step < flood_peak_start:
            ratio = (step - flood_onset_start) / (flood_peak_start - flood_onset_start)
            current_speed = normal_speed - ratio * (normal_speed - peak_flood_speed)
            set_edge_speed_safely(flood_edges, current_speed)

        # Phase C: peak flood
        elif flood_peak_start <= step < flood_peak_end:
            set_edge_speed_safely(flood_edges, peak_flood_speed)

        # Phase D: recovery
        elif flood_peak_end <= step < flood_recovery_end:
            ratio = (step - flood_peak_end) / (flood_recovery_end - flood_peak_end)
            current_speed = peak_flood_speed + ratio * (normal_speed - peak_flood_speed)
            set_edge_speed_safely(flood_edges, current_speed)

        # Phase E: restored normal
        else:
            set_edge_speed_safely(flood_edges, normal_speed)

        step += 1

    traci.close()

    print("=====================================")
    print("[SUCCESS] Flooding simulation completed.")
    print(f"Edge output file: {edge_output_file}")
    print("=====================================")


if __name__ == "__main__":
    run_flooding_simulation()