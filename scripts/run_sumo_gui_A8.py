import os
import subprocess


def run_sumo_gui():
    # =====================================
    # 1. Project paths
    # =====================================
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    route_file = os.path.join(project_root, "routes", "a8_random.rou.xml")
    detector_additional_file = os.path.join(project_root, "detectors", "a8_detectors.add.xml")
    label_additional_file = os.path.join(project_root, "labels", "a8_labels.add.xml")
    additional_files = ",".join([detector_additional_file,label_additional_file])
    gui_settings_file = os.path.join(project_root, "labels", "a8_gui_settings.xml")

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

    if not os.path.exists(route_file):
        print("[ERROR] Route file not found:")
        print(route_file)
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

    print("=====================================")
    print("Launching SUMO GUI")
    print(f"Network file: {net_file}")
    print(f"Route file  : {route_file}")
    print(f"Detector file  : {detector_additional_file}")
    print(f"Label file  : {label_additional_file}")
    print(f"GUI delay: {gui_delay_ms} ms")
    print("=====================================")

    # =====================================
    # 4. SUMO command
    # =====================================
    cmd = [
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

    # =====================================
    # 5. Run SUMO
    # =====================================
    subprocess.run(cmd)


if __name__ == "__main__":
    run_sumo_gui()