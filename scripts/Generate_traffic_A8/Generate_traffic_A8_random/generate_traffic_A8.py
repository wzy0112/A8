import os
import sys
import subprocess
import xml.etree.ElementTree as ET

def add_vehicle_type_to_routes(route_file):
    """
    Insert a realistic vehicle type and assign it to all vehicles.
    """

    if not os.path.exists(route_file):
        print("[ERROR] Route file not found for vType insertion:")
        print(route_file)
        return

    tree = ET.parse(route_file)
    root = tree.getroot()

    # 1. Create vehicle type (A8 motorway realistic)
    vtype = ET.Element("vType", {
        "id": "car",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "length": "5.0",
        "maxSpeed": "33.33"   # ≈120 km/h
    })

    # Insert at top
    root.insert(0, vtype)

    # 2. Assign type to all vehicles
    vehicle_count = 0
    for vehicle in root.findall("vehicle"):
        vehicle.set("type", "car")
        vehicle_count += 1

    # 3. Save
    tree.write(route_file, encoding="utf-8", xml_declaration=True)

    print("[SUCCESS] Vehicle type added.")
    print(f"Vehicles updated: {vehicle_count}")

def generate_traffic():
    # =====================================
    # 1. Project paths
    # =====================================
    project_root = r"D:\SUMO_A9_Project"

    net_file = os.path.join(project_root, "sumo", "a8_corridor.net.xml")
    route_dir = os.path.join(project_root, "routes")
    trips_file = os.path.join(route_dir, "a8_random.trips.xml")
    route_file = os.path.join(route_dir, "a8_random.rou.xml")

    os.makedirs(route_dir, exist_ok=True)

    # =====================================
    # 2. Check network file
    # =====================================
    if not os.path.exists(net_file):
        print("[ERROR] Network file not found:")
        print(net_file)
        return

    # =====================================
    # 3. Find SUMO_HOME and randomTrips.py
    # =====================================
    sumo_home =  r"D:\Eclipse\SUMO"

    random_trips_script = os.path.join(sumo_home, "tools", "randomTrips.py")

    if not os.path.exists(random_trips_script):
        print("[ERROR] randomTrips.py not found:")
        print(random_trips_script)
        return

    # =====================================
    # 4. Traffic generation settings
    # =====================================
    begin_time = 0
    end_time = 3600          # 1 hour
    period = 2.5             # one vehicle every 2.5 seconds on average
    seed = 42
    fringe_factor = 10

    # Vehicle distribution:
    # passenger: regular cars
    # truck: freight vehicles
    # bus: optional small share
    additional_args = [
        "--validate",
        "--remove-loops",
        "--fringe-factor", str(fringe_factor),
        "--allow-fringe",
        "--seed", str(seed)
    ]

    # =====================================
    # 5. Build command
    # =====================================
    cmd = [
        sys.executable,
        random_trips_script,
        "-n", net_file,
        "-b", str(begin_time),
        "-e", str(end_time),
        "-p", str(period),
        "-o", trips_file,
        "-r", route_file
    ] + additional_args

    print("=====================================")
    print("Generating traffic for A8 corridor")
    print(f"Network file : {net_file}")
    print(f"Trips file   : {trips_file}")
    print(f"Route file   : {route_file}")
    print(f"Begin time   : {begin_time}")
    print(f"End time     : {end_time}")
    print(f"Period       : {period}")
    print("=====================================")

    # =====================================
    # 6. Run generation
    # =====================================
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("[SUCCESS] Traffic generation completed.")
        print(f"Route file created: {route_file}")
        add_vehicle_type_to_routes(route_file)
    else:
        print("[ERROR] Traffic generation failed.")
        print(result.stdout)
        print(result.stderr)




if __name__ == "__main__":
    generate_traffic()