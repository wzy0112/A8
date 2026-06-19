# 1. 读取 baseline edgeData
# 2. 读取 flooding edgeData
# 3. 按时间聚合 speed / density / flow
# 4. 计算每分钟 recovery ratio
# 5. 找到 recovery time
# 6. 输出 recovery curve 图
# 7. 输出 recovery_summary.csv
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

# B471 70-90 km/n 19-22 m/s
# peak_flood_speed = 8.33  # 30 km/h

# find continuous segment
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

def set_edge_speed_safely(edge_ids, speed_value):
    for edge_id in edge_ids:
        try:
            traci.edge.setMaxSpeed(edge_id, speed_value)
        except traci.exceptions.TraCIException:
            pass

def restore_edge_speeds(original_edge_speeds):
    for edge_id, speed_value in original_edge_speeds.items():
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
    cfg_file = os.path.join(project_root, "routes", "a8_dynamic_rerouting.sumocfg")
    detector_additional_file = os.path.join(project_root, "detectors", "a8_detectors.add.xml")
    label_additional_file = os.path.join(project_root, "labels", "a8_labels.add.xml")
    gui_settings_file = os.path.join(project_root, "labels", "a8_gui_settings.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    edge_data_additional_file = os.path.join(results_dir, "a8_flood_edge_data_od_reroute_nottest.add.xml")
    edge_output_file = os.path.join(results_dir, "a8_flood_edge_output_od_reroute_nottest.xml")

    sumo_gui_path = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")

    # =====================================
    # 2. Basic checks
    # =====================================
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
    # B471 70-90 km/n 19-22 m/s
    peak_flood_speed = 8.33   # 30 km/h

    # =====================================
    # 5. Launch SUMO with TraCI
    # =====================================
    print("=====================================")
    print("Starting flooding simulation")
    print(f"Network file         : {net_file}")
    print(f"Config file          : {cfg_file}")
    print(f"Additional files     : {additional_files}")
    print(f"GUI settings         : {gui_settings_file}")
    print(f"Edge output file     : {edge_output_file}")
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

    # =====================================
    # 6. Read network and auto-select flood edges
    # =====================================
    net = sumolib.net.readNet(net_file)

    # =====================================
    # B471 flooding segments
    # =====================================
    # Segment 1: B471 Dachau-Süd / Gröbenried
    segment_1_forward = find_edge_path(net, "372647238", "511296353")
    segment_1_backward = find_edge_path(net, "-511296353", "-372647238")
    # Segment 2: B471 Dachauer Moos
    segment_2_forward = find_edge_path(net, "32234550", "493360266#0")
    segment_2_backward = find_edge_path(net, "-493360266#0", "-519348437")

    flood_edges = (
            segment_1_forward
            + segment_1_backward
            + segment_2_forward
            + segment_2_backward
    )

    # remove duplicates while keeping order
    flood_edges = list(dict.fromkeys(flood_edges))

    original_edge_speeds = {}

    for edge_id in flood_edges:
        try:
            edge = net.getEdge(edge_id)
            original_edge_speeds[edge_id] = max(
                lane.getSpeed() for lane in edge.getLanes()
            )
        except Exception as e:
            print("[WARNING] Cannot read original speed for edge:", edge_id, e)

    print("[INFO] Flood edges selected:")
    for edge_id in flood_edges:
        try:
            print(
                "   ",
                edge_id,
                "| lanes =",
                traci.edge.getLaneNumber(edge_id),
                "| originalMaxSpeed =",
                original_edge_speeds.get(edge_id, "unknown")
            )
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

        # ---- Deadlock removal settings 解决最后57个车辆锁死问题----
        # DEADLOCK_WAIT = 600  # seconds
        # DEADLOCK_SPEED = 0.1  # m/s
        MAX_REMAINING = 60  # only trigger near the end

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

        # Phase A: normal
        if step < flood_onset_start:
            pass

        # Phase B: onset (gradual speed reduction)
        elif flood_onset_start <= step < flood_peak_start:
            ratio = (step - flood_onset_start) / (flood_peak_start - flood_onset_start)

            for edge_id in flood_edges:
                if edge_id in original_edge_speeds:
                    normal_speed_edge = original_edge_speeds[edge_id]
                    current_speed = normal_speed_edge - ratio * (normal_speed_edge - peak_flood_speed)
                    set_edge_speed_safely([edge_id], current_speed)

        # Phase C: peak flood
        elif flood_peak_start <= step < flood_peak_end:
            set_edge_speed_safely(flood_edges, peak_flood_speed)
        # Phase D: recovery
        elif flood_peak_end <= step < flood_recovery_end:
            ratio = (step - flood_peak_end) / (flood_recovery_end - flood_peak_end)

            for edge_id in flood_edges:
                if edge_id in original_edge_speeds:
                    normal_speed_edge = original_edge_speeds[edge_id]
                    current_speed = peak_flood_speed + ratio * (normal_speed_edge - peak_flood_speed)
                    set_edge_speed_safely([edge_id], current_speed)
        # Phase E: restored normal
        else:
            restore_edge_speeds(original_edge_speeds)

        step += 1

    traci.close()

    print("=====================================")
    print("[SUCCESS] Flooding simulation completed.")
    print(f"Edge output file: {edge_output_file}")
    print("=====================================")


if __name__ == "__main__":
    run_flooding_simulation()