import os
import sys
import xml.etree.ElementTree as ET

SUMO_HOME = r"D:\Eclipse\SUMO"
if SUMO_HOME not in os.environ:
    os.environ["SUMO_HOME"] = SUMO_HOME

tools_path = os.path.join(SUMO_HOME, "tools")
if tools_path not in sys.path:
        sys.path.append(tools_path)

import traci
import sumolib

def find_edge_path(net, start_edge_id, end_edge_id):
    try:
        path_edges, cost = net.getShortestPath(
            net.getEdge(start_edge_id),
            net.getEdge(end_edge_id)
        )
        if path_edges is None:
            print("[WARNING] No path found:", start_edge_id, "->", end_edge_id)
            return []
        return [edge.getID() for edge in path_edges]
    except Exception as e:
        print("[ERROR] Path search failed:", start_edge_id, "->", end_edge_id, e)
        return []

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


def close_full_edge(edge_ids):
    """
    Fully close an edge by disallowing road vehicles on all lanes.
    """
    for edge_id in edge_ids:
        try:
            lane_num = traci.edge.getLaneNumber(edge_id)
            for i in range(lane_num):
                lane_id = f"{edge_id}_{i}"
                traci.lane.setDisallowed(lane_id, ["passenger", "truck", "bus"])
        except traci.exceptions.TraCIException:
            pass


def reopen_full_edge(edge_ids):
    for edge_id in edge_ids:
        try:
            lane_num = traci.edge.getLaneNumber(edge_id)
            for i in range(lane_num):
                lane_id = f"{edge_id}_{i}"
                traci.lane.setDisallowed(lane_id, [])
        except traci.exceptions.TraCIException:
            pass


def set_edge_speed_safely(edge_ids, speed_value):
    for edge_id in edge_ids:
        try:
            traci.edge.setMaxSpeed(edge_id, speed_value)
        except traci.exceptions.TraCIException:
            pass


def run_full_closure_simulation():
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    cfg_file = os.path.join(project_root, "routes", "a8_dynamic_rerouting.sumocfg")
    detector_additional_file = os.path.join(project_root, "detectors", "a8_detectors.add.xml")
    label_additional_file = os.path.join(project_root, "labels", "a8_labels.add.xml")
    gui_settings_file = os.path.join(project_root, "labels", "a8_gui_settings.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    edge_data_additional_file = os.path.join(results_dir, "a8_full_close_edge_data_od_reroute.add.xml")
    edge_output_file = os.path.join(results_dir, "a8_full_close_edge_output_od_reroute.xml")

    sumo_gui_path = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")

    required_files = [
        net_file,
        cfg_file,
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

    gui_delay_ms = 10

    flood_onset_start = 1200
    flood_peak_start = 1500
    flood_peak_end = 2400
    flood_recovery_end = 3000

    normal_speed = 33.33
    approach_speed = 5.56

    print("=====================================")
    print("Starting full-segment-closure flooding simulation")
    print("Closed edges will be generated from boundary edges after SUMO starts.")
    print(f"Edge output file: {edge_output_file}")
    print("=====================================")

    sumo_cmd = [
        sumo_gui_path,
        "-c", cfg_file,
        "-a", additional_files,
        "--gui-settings-file", gui_settings_file,
        "--start",
        "--delay", str(gui_delay_ms),
        "--duration-log.statistics"
    ]

    traci.start(sumo_cmd)

    net = sumolib.net.readNet(net_file)

    # Severe flooding / full edge closure segments
    # B471 Gröbenried
    b471_forward = find_edge_path(net, "372647238", "511296353")
    b471_backward = find_edge_path(net, "-511296353", "-372647238")

    # A8 Palsweiser Moos / Sulzemoos-Fuchsberg--Dachau/Fürstenfeldbruck
    a8_forward = find_edge_path(net, "327464676", "327462610")
    a8_backward = find_edge_path(net, "251505085", "251505081")

    closed_edges = list(dict.fromkeys(
        b471_forward + b471_backward + a8_forward + a8_backward
    ))

    print("[INFO] Full-closure edges selected:")
    for edge_id in closed_edges:
        print("   ", edge_id, "| lanes =", traci.edge.getLaneNumber(edge_id))

    step = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # ---- Deadlock removal settings 解决最后43个车辆锁死问题----
        # DEADLOCK_WAIT = 600  # seconds
        # DEADLOCK_SPEED = 0.1  # m/s
        MAX_REMAINING = 50  # only trigger near the end

        # ---- Deadlock removal logic ----
        remaining = traci.simulation.getMinExpectedNumber()

        if remaining <= MAX_REMAINING:
            remove_list = []
            for vid in traci.vehicle.getIDList():
                remove_list.append(vid)
            # for vid in traci.vehicle.getIDList():
            #     speed = traci.vehicle.getSpeed(vid)
            #     wait = traci.vehicle.getWaitingTime(vid)
            #
            #     if speed < DEADLOCK_SPEED and wait > DEADLOCK_WAIT:
            #         remove_list.append(vid)

            for vid in remove_list:
                try:
                    print(
                        "[REMOVE DEADLOCK]",
                        vid,
                        "road=", traci.vehicle.getRoadID(vid),
                        "wait=", traci.vehicle.getWaitingTime(vid),
                        "remaining=", remaining
                    )
                    traci.vehicle.remove(vid)
                except Exception:
                    pass

        if step == flood_onset_start:
            print("[INFO] Full closure onset started.")
        if step == flood_peak_start:
            print("[INFO] Full closure peak started.")
        if step == flood_peak_end:
            print("[INFO] Recovery started.")
        if step == flood_recovery_end:
            print("[INFO] Recovery completed.")

        if step < flood_onset_start:
            pass

        elif flood_onset_start <= step < flood_peak_start:
            ratio = (step - flood_onset_start) / (flood_peak_start - flood_onset_start)
            current_speed = normal_speed - ratio * (normal_speed - approach_speed)
            set_edge_speed_safely(closed_edges, current_speed)

        elif flood_peak_start <= step < flood_peak_end:
            close_full_edge(closed_edges)

        elif flood_peak_end <= step < flood_recovery_end:
            reopen_full_edge(closed_edges)
            ratio = (step - flood_peak_end) / (flood_recovery_end - flood_peak_end)
            current_speed = approach_speed + ratio * (normal_speed - approach_speed)
            set_edge_speed_safely(closed_edges, current_speed)

        else:
            reopen_full_edge(closed_edges)
            set_edge_speed_safely(closed_edges, normal_speed)

        step += 1

    traci.close()

    print("=====================================")
    print("[SUCCESS] Full-segment-closure simulation completed.")
    print(f"Edge output file: {edge_output_file}")
    print("=====================================")


if __name__ == "__main__":
    run_full_closure_simulation()