import os
import subprocess
import xml.etree.ElementTree as ET

# 之前：
# trips → duarouter → rou.xml → SUMO跑
# 现在：
# trips → sumocfg → SUMO运行中动态算路径

# freq=0.5h


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


def create_detector_poi_file(detector_groups, detector_poi_file):
    """
    detector_groups = [
        (detector_file_1, "red"),
        (detector_file_2, "blue")
    ]

    将两类 detector 转成同一个 POI XML：
    第一类 red，第二类 blue。
    """
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

    for detector_file, color in detector_groups:
        detector_tree = ET.parse(detector_file)
        detector_root = detector_tree.getroot()

        for detector in detector_root.iter():
            tag = detector.tag.split("}")[-1]
            if tag not in detector_tags:
                continue

            detector_id = detector.get("id", f"detector_{poi_count}")
            lane = detector.get("lane")
            pos = detector.get("pos")

            if pos is None:
                pos = detector.get("startPos") or detector.get("endPos")

            if lane is None or pos is None:
                print(f"[WARNING] Skip detector without lane/pos: {detector_id}")
                continue

            color_map = {
                "red": "205,51,51",  # 暗红色
                "blue": "51,51,205",  # 暗蓝色
            }

            poi_color = color_map.get(color, color)

            ET.SubElement(
                poi_root,
                "poi",
                {
                    "id": f"poi_{color}_{detector_id}",
                    "lane": lane,
                    "pos": pos,
                    "color": poi_color,
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

def merge_detector_additional_files(detector_files, merged_detector_file):
    """
    将多个 detector add.xml 合并成一个 additional XML。
    """
    merged_root = ET.Element("additional")

    count = 0

    for detector_file in detector_files:
        tree = ET.parse(detector_file)
        root = tree.getroot()

        for child in list(root):
            merged_root.append(child)
            count += 1

    merged_tree = ET.ElementTree(merged_root)
    ET.indent(merged_tree, space="    ")
    merged_tree.write(
        merged_detector_file,
        encoding="utf-8",
        xml_declaration=True
    )

    print(f"Created merged detector file: {merged_detector_file}")
    print(f"Detector elements merged: {count}")

def run_sumo_gui():
    # =====================================
    # 1. Project paths
    # =====================================
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    cfg_file = os.path.join(project_root, "routes", "a8_dynamic_rerouting.sumocfg")
    detector_additional_file_red = os.path.join(project_root, "detectors", "a8_detectors.add.xml")
    detector_additional_file_blue = os.path.join(project_root, "detectors", "a8_detectors_secondary.add.xml")

    gui_settings_file = os.path.join(project_root, "labels", "a8_gui_settings.xml")

    results_dir = os.path.join(project_root, "heat_edges")
    os.makedirs(results_dir, exist_ok=True)

    edge_data_additional_file = os.path.join(results_dir, "a8_edge_data_od_reroute.add.xml")
    edge_output_file = os.path.join(results_dir, "a8_edge_output_od_reroute.xml")
    primary_e1_output = os.path.join(project_root, "detectors", "a8_e1_output.xml")
    secondary_e1_output = os.path.join(project_root, "detectors", "a8_e1_output_secondary.xml")

    for f in [primary_e1_output, secondary_e1_output, edge_output_file]:
        if os.path.exists(f):
            os.remove(f)

    merged_detector_additional_file = os.path.join(project_root, "detectors", "a8_detectors_merged.add.xml")
    detector_poi_file = os.path.join(project_root, "detectors", "a8_detector_pois_red_blue.add.xml")

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

    for detector_file in [
        detector_additional_file_red,
        detector_additional_file_blue
    ]:
        if not os.path.exists(detector_file):
            print("[ERROR] Detector additional file not found:")
            print(detector_file)
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

    merge_detector_additional_files(
        detector_files=[
            detector_additional_file_red,
            detector_additional_file_blue
        ],
        merged_detector_file=merged_detector_additional_file
    )

    create_detector_poi_file(
        detector_groups=[
            (detector_additional_file_red, "red"),
            (detector_additional_file_blue, "blue")
        ],
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
        merged_detector_additional_file,
        detector_poi_file,
        edge_data_additional_file
    ])

    print("=====================================")
    print("Launching SUMO GUI")
    print(f"Network file: {net_file}")
    print(f"Config file: {cfg_file}")
    print(f"Red detector file    : {detector_additional_file_red}")
    print(f"Blue detector file   : {detector_additional_file_blue}")
    print(f"Merged detector file : {merged_detector_additional_file}")
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
