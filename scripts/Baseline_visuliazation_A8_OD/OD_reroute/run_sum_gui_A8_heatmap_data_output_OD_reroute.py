import os
import subprocess
import xml.etree.ElementTree as ET

# 之前：
# trips → duarouter → rou.xml → SUMO跑
# 现在：
# trips → sumocfg → SUMO运行中动态算路径

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
    cfg_file = os.path.join(project_root, "routes", "a8_dynamic_rerouting.sumocfg")
    detector_additional_file = os.path.join(project_root, "detectors", "a8_detectors.add.xml")
    label_additional_file = os.path.join(project_root, "labels", "a8_labels.add.xml")
    # additional_files = ",".join([detector_additional_file,label_additional_file])
    gui_settings_file = os.path.join(project_root, "labels", "a8_gui_settings.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    edge_data_additional_file = os.path.join(results_dir, "a8_edge_data_od_reroute.add.xml")
    edge_output_file = os.path.join(results_dir, "a8_edge_output_od_reroute.xml")

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
        "--duration-log.statistics"
    ]
    # =====================================
    # 5. Run SUMO
    # =====================================
    subprocess.run(cmd)


if __name__ == "__main__":
    run_sumo_gui()