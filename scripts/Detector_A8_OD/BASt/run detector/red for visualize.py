import os
import subprocess
import xml.etree.ElementTree as ET

# 之前：
# trips → duarouter → rou.xml → SUMO跑
# 现在：
# trips → sumocfg → SUMO运行中动态算路径

# freq=1h


def create_edge_data_additional_file(additional_file, edge_output_file, freq=1800):
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


def create_detector_poi_file(detector_file, detector_poi_file):
    """
    Convert detector definitions into red POIs for GUI display.

    只把 detector 作为 POI 显示为红色；不再生成/加载之前的
    waterbody 或 city POI。
    """
    detector_tree = ET.parse(detector_file)
    detector_root = detector_tree.getroot()

    poi_root = ET.Element("additional")
    detector_tags = {
        "inductionLoop",
        "e1Detector",
        "laneAreaDetector",
        "e2Detector",
        "instantInductionLoop",
        "entryExitDetector",
    }

    poi_count = 0
    for detector in detector_root.iter():
        tag = detector.tag.split("}")[-1]  # support namespaced XML too
        if tag not in detector_tags:
            continue

        detector_id = detector.get("id", f"detector_{poi_count}")
        lane = detector.get("lane")
        pos = detector.get("pos")

        # laneAreaDetector/e2Detector sometimes use startPos/endPos instead of pos.
        if pos is None:
            pos = detector.get("startPos") or detector.get("endPos")

        if lane is None or pos is None:
            print(f"[WARNING] Skip detector without lane/pos: {detector_id}")
            continue

        ET.SubElement(
            poi_root,
            "poi",
            {
                "id": f"poi_{detector_id}",
                "lane": lane,
                "pos": pos,
                "color": "205,51,51",
                "type": "detector",
                "layer": "10",
            }
        )
        poi_count += 1

    tree = ET.ElementTree(poi_root)
    ET.indent(tree, space="    ")
    tree.write(detector_poi_file, encoding="utf-8", xml_declaration=True)
    print(f"Created detector POI file: {detector_poi_file}")
    print(f"Detector POIs written: {poi_count}")


def run_sumo_gui():
    # =====================================
    # 1. Project paths
    # =====================================
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    cfg_file = os.path.join(project_root, "routes", "a8_dynamic_rerouting.sumocfg")
    detector_additional_file = os.path.join(project_root, "detectors", "a8_detectors.add.xml")
    detector_poi_file = os.path.join(project_root, "detectors", "a8_detector_pois.add.xml")
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

    if not os.path.exists(gui_settings_file):
        print("[ERROR] Gui setting file not found:")
        print(gui_settings_file)
        return

    if not os.path.exists(sumo_gui_path):
        print("[ERROR] SUMO GUI executable not found:")
        print(sumo_gui_path)
        print("Please update the path in the script.")
        return

    create_detector_poi_file(
        detector_file=detector_additional_file,
        detector_poi_file=detector_poi_file
    )

    create_edge_data_additional_file(
        additional_file=edge_data_additional_file,
        edge_output_file=edge_output_file,
        freq=1800
    )

    # 不再加载 labels/a8_labels.add.xml，因此之前的 waterbody/city POI 会被删除。
    # detector 现在作为红色 POI 加载，而不是继续额外加载为 label POI。
    additional_files = ",".join([
        detector_poi_file,
        edge_data_additional_file
    ])

    print("=====================================")
    print("Launching SUMO GUI")
    print(f"Network file: {net_file}")
    print(f"Config file: {cfg_file}")
    print(f"Detector source file : {detector_additional_file}")
    print(f"Detector POI file    : {detector_poi_file}")
    print(f"Edge data file       : {edge_data_additional_file}")
    print(f"Edge output file     : {edge_output_file}")
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
