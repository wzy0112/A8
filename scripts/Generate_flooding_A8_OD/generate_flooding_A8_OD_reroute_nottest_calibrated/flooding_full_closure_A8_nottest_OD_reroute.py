import os
import sys
import csv
import xml.etree.ElementTree as ET

#根据最好OD生成的config得到edge data
#等所有车跑完结束 错
#固定仿真到 7200 s 结束，并记录剩余车辆数
#只删除等待超过 600 秒且速度接近 0 的车。错不删除了

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
    cfg_file = os.path.join(project_root, "routes", "a8_od_spsa.sumocfg")
    detector_additional_file = os.path.join(project_root, "detectors", "a8_detectors_merged.add.xml")
    label_additional_file = os.path.join(project_root, "labels", "a8_labels.add.xml")
    gui_settings_file = os.path.join(project_root, "labels", "a8_gui_settings.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    edge_data_additional_file = os.path.join(results_dir, "a8_full_close_edge_data_od_reroute.add.xml")
    edge_output_file = os.path.join(results_dir, "a8_full_close_edge_output_od_reroute.xml")
    tripinfo_output_file = os.path.join(results_dir, "a8_full_close_tripinfo_od_reroute_calibrated.xml")
    summary_output_file = os.path.join(results_dir, "a8_full_close_summary_od_reroute_calibrated.xml")
    statistic_output_file = os.path.join(results_dir, "a8_full_close_statistics_od_reroute_calibrated.xml")
    step_metrics_csv = os.path.join(results_dir, "a8_full_close_step_metrics_od_reroute_calibrated.csv")
    final_metrics_csv = os.path.join(results_dir, "a8_full_close_final_metrics_od_reroute_calibrated.csv")

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
        "--duration-log.statistics",
        "--tripinfo-output", tripinfo_output_file,
        "--summary-output", summary_output_file,
        "--statistic-output", statistic_output_file,
        #"--time-to-teleport", "300",
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
    print("[SUCCESS] Full-segment-closure simulation completed.")
    print(f"Final simulation time: {final_time} s")
    print(f"Edge output file: {edge_output_file}")
    print(f"Tripinfo XML: {tripinfo_output_file}")
    print(f"Summary XML: {summary_output_file}")
    print(f"Statistics XML: {statistic_output_file}")
    print(f"Step metrics CSV: {step_metrics_csv}")
    print(f"Final metrics CSV: {final_metrics_csv}")
    print(f"Config file: {cfg_file}")
    print(f"Vehicles remaining at 7200 s: {remaining_vehicles}")
    print("=====================================")


if __name__ == "__main__":
    run_full_closure_simulation()