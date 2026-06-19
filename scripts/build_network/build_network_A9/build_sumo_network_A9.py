import os
import subprocess
import sys

# =====================================
# 1. Project path settings
# =====================================
PROJECT_ROOT = r"D:\SUMO_A9_Project"
NETWORK_DIR = os.path.join(PROJECT_ROOT, "network")
SUMO_DIR = os.path.join(PROJECT_ROOT, "sumo")

OSM_FILE = os.path.join(NETWORK_DIR, "a9_neufahrn_allershausen.osm")
NET_FILE = os.path.join(SUMO_DIR, "a9_neufahrn_allershausen.net.xml")

# =====================================
# 2. SUMO Installation path settings
# =====================================
SUMO_HOME = r"D:\Eclipse\SUMO"
NETCONVERT_EXE = os.path.join(SUMO_HOME, "bin", "netconvert.exe")

# =====================================
# 3. Check files and directories
# =====================================
os.makedirs(SUMO_DIR, exist_ok=True)

if not os.path.exists(OSM_FILE):
    print(f"Cannot find the OSM file: {OSM_FILE}")
    sys.exit(1)

if not os.path.exists(NETCONVERT_EXE):
    print(f"Cannot find netconvert.exe: {NETCONVERT_EXE}")
    sys.exit(1)

# =====================================
# 4. The `netconvert` command
# =====================================
cmd = [
    NETCONVERT_EXE,
    "--osm-files", OSM_FILE,
    "--output-file", NET_FILE,
    "--geometry.remove",
    "--ramps.guess",
    "--junctions.join",
    "--tls.guess-signals",
    "--tls.discard-simple"
]

# 如果你后面想保留更干净的高速网络，可改用下面这组参数：
# cmd = [
#     NETCONVERT_EXE,
#     "--osm-files", OSM_FILE,
#     "--output-file", NET_FILE,
#     "--keep-edges.by-type", "motorway,motorway_link,trunk,trunk_link",
#     "--geometry.remove",
#     "--ramps.guess",
#     "--junctions.join"
# ]

# =====================================
# 5. Perform conversion
# =====================================
print("Start generating SUMO net.xml ...")
print("Execute command:")
print(" ".join(f'"{x}"' if " " in x else x for x in cmd))

try:
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )

    print("\n net.xml generated successfully")
    print(f"Output file: {NET_FILE}")

    if result.stdout.strip():
        print("\n--- netconvert stdout ---")
        print(result.stdout)

    if result.stderr.strip():
        print("\n--- netconvert stderr ---")
        print(result.stderr)

except subprocess.CalledProcessError as e:
    print("\n net.xml generated failed")
    print("Return code:", e.returncode)

    if e.stdout:
        print("\n--- stdout ---")
        print(e.stdout)

    if e.stderr:
        print("\n--- stderr ---")
        print(e.stderr)

    sys.exit(1)