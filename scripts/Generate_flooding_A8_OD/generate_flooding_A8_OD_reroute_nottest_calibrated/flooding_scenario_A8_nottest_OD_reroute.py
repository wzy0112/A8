# 1. 读取 baseline edgeData
# 2. 读取 flooding edgeData
# 3. 按时间聚合 speed / density / flow
# 4. 计算每分钟 recovery ratio
# 5. 找到 recovery time
# 6. 输出 recovery curve 图
# 7. 输出 recovery_summary.csv

#根据最好OD生成的config得到edge data
#等所有车跑完结束 错
#固定仿真到 7200 s 结束，并记录剩余车辆数
#只删除等待超过 600 秒且速度接近 0 的车。错不删除了

import os
import sys
import csv
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
    cfg_file = os.path.join(project_root, "routes", "a8_od_spsa.sumocfg")
    detector_additional_file = os.path.join(project_root, "detectors", "a8_detectors_merged.add.xml")
    label_additional_file = os.path.join(project_root, "labels", "a8_labels.add.xml")
    gui_settings_file = os.path.join(project_root, "labels", "a8_gui_settings.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    edge_data_additional_file = os.path.join(results_dir, "a8_flood_edge_data_od_reroute_nottest.add.xml")
    edge_output_file = os.path.join(results_dir, "a8_flood_edge_output_od_reroute_nottest.xml")
    tripinfo_output_file = os.path.join(results_dir, "a8_flood_tripinfo_od_reroute_calibrated.xml")
    summary_output_file = os.path.join(results_dir, "a8_flood_summary_od_reroute_calibrated.xml")
    statistic_output_file = os.path.join(results_dir, "a8_flood_statistics_od_reroute_calibrated.xml")
    step_metrics_csv = os.path.join(results_dir, "a8_flood_step_metrics_od_reroute_calibrated.csv")
    final_metrics_csv = os.path.join(results_dir, "a8_flood_final_metrics_od_reroute_calibrated.csv")
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
        "--duration-log.statistics",
        "--tripinfo-output", tripinfo_output_file,
        "--summary-output", summary_output_file,
        "--statistic-output", statistic_output_file,

        "--time-to-teleport", "300",
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

    with open(step_metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_s",
                "running_vehicle_count",
                "loaded_vehicle_count",
                "departed_vehicle_count",
                "arrived_vehicle_count",
                "mean_speed_mps",
                "mean_waiting_time_s",
                "mean_time_loss_s",
                "halting_vehicle_count"
            ]
        )
        writer.writeheader()

        MAX_SIM_TIME = 7200

        while traci.simulation.getTime() < MAX_SIM_TIME:
            traci.simulationStep()

            current_time = traci.simulation.getTime()

            # ===============================
            # Flooding speed-limit intervention
            # ===============================

            # Phase A: normal
            if step < flood_onset_start:
                pass

            # Phase B: onset
            elif flood_onset_start <= step < flood_peak_start:
                ratio = (step - flood_onset_start) / (flood_peak_start - flood_onset_start)

                for edge_id in flood_edges:
                    if edge_id in original_edge_speeds:
                        normal_speed_edge = original_edge_speeds[edge_id]
                        current_speed = normal_speed_edge - ratio * (
                                normal_speed_edge - peak_flood_speed
                        )
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
                        current_speed = peak_flood_speed + ratio * (
                                normal_speed_edge - peak_flood_speed
                        )
                        set_edge_speed_safely([edge_id], current_speed)

            # Phase E: restored normal
            else:
                restore_edge_speeds(original_edge_speeds)

            # ===============================
            # Step-level traffic metrics
            # ===============================
            vehicle_ids = traci.vehicle.getIDList()
            running_count = len(vehicle_ids)

            if running_count > 0:
                speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
                waiting_times = [traci.vehicle.getWaitingTime(v) for v in vehicle_ids]
                time_losses = [traci.vehicle.getTimeLoss(v) for v in vehicle_ids]

                mean_speed = sum(speeds) / running_count
                mean_waiting_time = sum(waiting_times) / running_count
                mean_time_loss = sum(time_losses) / running_count
                halting_count = sum(1 for v in vehicle_ids if traci.vehicle.getSpeed(v) < 0.1)
            else:
                mean_speed = 0
                mean_waiting_time = 0
                mean_time_loss = 0
                halting_count = 0

            writer.writerow({
                "time_s": current_time,
                "running_vehicle_count": running_count,
                "loaded_vehicle_count": traci.simulation.getLoadedNumber(),
                "departed_vehicle_count": traci.simulation.getDepartedNumber(),
                "arrived_vehicle_count": traci.simulation.getArrivedNumber(),
                "mean_speed_mps": mean_speed,
                "mean_waiting_time_s": mean_waiting_time,
                "mean_time_loss_s": mean_time_loss,
                "halting_vehicle_count": halting_count
            })

            step += 1
    remaining_vehicles = traci.simulation.getMinExpectedNumber()
    final_time = traci.simulation.getTime()

    traci.close()

    with open(final_metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "final_simulation_time_s",
                "remaining_vehicles",
                "edge_output_file",
                "tripinfo_output_file",
                "summary_output_file",
                "statistic_output_file",
                "step_metrics_csv"
            ]
        )
        writer.writeheader()
        writer.writerow({
            "final_simulation_time_s": final_time,
            "remaining_vehicles": remaining_vehicles,
            "edge_output_file": edge_output_file,
            "tripinfo_output_file": tripinfo_output_file,
            "summary_output_file": summary_output_file,
            "statistic_output_file": statistic_output_file,
            "step_metrics_csv": step_metrics_csv
        })

    print("=====================================")
    print("[SUCCESS] Flooding simulation completed.")
    print(f"Final simulation time: {final_time} s")
    print(f"Edge output file: {edge_output_file}")
    print(f"Tripinfo XML: {tripinfo_output_file}")
    print(f"Summary XML: {summary_output_file}")
    print(f"Statistics XML: {statistic_output_file}")
    print(f"Step metrics CSV: {step_metrics_csv}")
    print(f"Final metrics CSV: {final_metrics_csv}")
    print(f"Vehicles remaining at 7200 s: {remaining_vehicles}")
    print("=====================================")


if __name__ == "__main__":
    run_flooding_simulation()