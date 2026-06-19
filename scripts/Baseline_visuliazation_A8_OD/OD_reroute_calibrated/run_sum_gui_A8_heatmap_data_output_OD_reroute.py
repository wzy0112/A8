import os
import subprocess
import xml.etree.ElementTree as ET
import traci
import csv

# 之前：
# trips → duarouter → rou.xml → SUMO跑
# 现在：
# trips → sumocfg → SUMO运行中动态算路径

#根据最好OD生成的config得到edge data
#等所有车跑完结束 错
## 固定仿真到 7200 s 结束，并记录剩余车辆数
#只删除等待超过 600 秒且速度接近 0 的车。
#7200结束
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


def run_sumo_gui():
    # =====================================
    # 1. Project paths
    # =====================================
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    cfg_file = os.path.join(project_root, "routes", "a8_od_spsa.sumocfg")
    detector_additional_file = os.path.join(project_root, "detectors", "a8_detectors_merged.add.xml")
    label_additional_file = os.path.join(project_root, "labels", "a8_labels.add.xml")
    # additional_files = ",".join([detector_additional_file,label_additional_file])
    gui_settings_file = os.path.join(project_root, "labels", "a8_gui_settings.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    edge_data_additional_file = os.path.join(results_dir, "a8_edge_data_od_reroute_calibrated.add.xml")
    edge_output_file = os.path.join(results_dir, "a8_edge_output_od_reroute_calibrated.xml")
    tripinfo_output_file = os.path.join(results_dir, "a8_tripinfo_od_reroute_calibrated.xml")
    summary_output_file = os.path.join(results_dir, "a8_summary_od_reroute_calibrated.xml")
    statistic_output_file = os.path.join(results_dir, "a8_statistics_od_reroute_calibrated.xml")
    step_metrics_csv = os.path.join(results_dir, "a8_step_metrics_od_reroute_calibrated.csv")
    final_metrics_csv = os.path.join(results_dir, "a8_final_metrics_od_reroute_calibrated.csv")
    # Update this path if your SUMO installation is different
    SUMO_HOME = r"D:\Eclipse\SUMO"
    sumo_gui_path = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")

    # =====================================
    # 2. GUI simulation settings
    # =====================================
    gui_delay_ms = 100

    # =====================================
    # 3. Check required files
    # =====================================
    if not os.path.exists(net_file):
        print("[ERROR] Network file not found:")
        print(net_file)
        return

    if not os.path.exists(cfg_file):
        print("[ERROR] Dynamic Rerouting SUMO file not found:")
        print(cfg_file)
        return

    if not os.path.exists(detector_additional_file):
        print("[ERROR] Detector additional file not found:")
        print(detector_additional_file)
        return

    if not os.path.exists(label_additional_file):
        print("[ERROR] Label additional file not found:")
        print(label_additional_file)
        return

    if not os.path.exists(gui_settings_file):
        print("[ERROR] Gui setting file not found:")
        print(gui_settings_file)
        return

    if not os.path.exists(sumo_gui_path):
        print("[ERROR] SUMO GUI executable not found:")
        print(sumo_gui_path)
        print("Please update the path in the script.")
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

    print("=====================================")
    print("Launching SUMO GUI")
    print(f"Network file: {net_file}")
    print(f"Config file: {cfg_file}")
    print(f"Detector file  : {detector_additional_file}")
    print(f"Label file  : {label_additional_file}")
    print(f"Edge data file    : {edge_data_additional_file}")
    print(f"Edge output file  : {edge_output_file}")
    print(f"GUI delay: {gui_delay_ms} ms")
    print("=====================================")

    # =====================================
    # 4. SUMO command
    # =====================================
    cmd = [
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
    # =====================================
    # 5. Run SUMO
    # =====================================
    traci.start(cmd)

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
    print("[DONE] Simulation stopped at fixed horizon 7200 s.")
    print(f"Final simulation time: {final_time} s")
    print(f"Step metrics CSV: {step_metrics_csv}")
    print(f"Final metrics CSV: {final_metrics_csv}")
    print(f"Tripinfo XML: {tripinfo_output_file}")
    print(f"Summary XML: {summary_output_file}")
    print(f"Statistics XML: {statistic_output_file}")
    print(f"Vehicles remaining at 7200 s: {remaining_vehicles}")
    print("=====================================")


if __name__ == "__main__":
    run_sumo_gui()