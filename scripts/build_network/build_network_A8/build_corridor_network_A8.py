import os
import sys
import subprocess
import xml.etree.ElementTree as ET

# =====================================
# 1. Project paths
# =====================================
PROJECT_ROOT = r"D:\SUMO_A9_Project"
SUMO_DIR = os.path.join(PROJECT_ROOT, "sumo")

FULL_NET_FILE = os.path.join(SUMO_DIR, "a8_dachau_odelzhausen.net.xml")
KEEP_EDGES_FILE = os.path.join(SUMO_DIR, "keep_edges_A8.txt")
CORRIDOR_NET_FILE = os.path.join(SUMO_DIR, "a8_corridor.net.xml")

# =====================================
# 2. SUMO installation path
# =====================================
SUMO_HOME = r"D:\Eclipse\SUMO"
NETCONVERT_EXE = os.path.join(SUMO_HOME, "bin", "netconvert.exe")

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = SUMO_HOME

SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
if SUMO_TOOLS not in sys.path:
    sys.path.append(SUMO_TOOLS)

try:
    import sumolib
except ImportError:
    print("sumolib could not be imported. Check SUMO_HOME and SUMO tools path.")
    sys.exit(1)

# =====================================
# 3. Checks
# =====================================
if not os.path.exists(FULL_NET_FILE):
    print(f"Full net file not found: {FULL_NET_FILE}")
    sys.exit(1)

if not os.path.exists(NETCONVERT_EXE):
    print(f"netconvert not found: {NETCONVERT_EXE}")
    sys.exit(1)

# =====================================
# 4. Corridor geographic bounds
#    Format: lon/lat
# =====================================
# Wider corridor around A9 northbound from AK Neufahrn to AS Allershausen
MIN_LON = 11.19
MIN_LAT = 48.16
MAX_LON = 11.54
MAX_LAT = 48.35

# =====================================
# 5. Allowed road types
# =====================================
ALLOWED_TYPE_KEYWORDS = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
]

# =====================================
# 6. Optional edge ID keywords to force keep
#    Useful for A9 / A92 relations if present in IDs/names/types
# =====================================
FORCE_KEEP_NAME_KEYWORDS = [
    "A8",
    "A 8",
    "Dachau",
    "Odelzhausen"
]

# =====================================
# 7. Load network
# =====================================
print("Loading network...")
net = sumolib.net.readNet(FULL_NET_FILE)

# =====================================
# 8. Parse XML for edge metadata
# =====================================
print("Parsing edge metadata...")
tree = ET.parse(FULL_NET_FILE)
root = tree.getroot()

edge_meta = {}
for edge in root.findall("edge"):
    edge_id = edge.get("id")
    if edge_id is None:
        continue

    edge_type = edge.get("type", "")
    edge_name = edge.get("name", "")

    edge_meta[edge_id] = {
        "type": edge_type,
        "name": edge_name
    }

# =====================================
# 9. Helper functions
# =====================================
def is_allowed_type(edge_type: str) -> bool:
    edge_type_lower = edge_type.lower()
    return any(keyword in edge_type_lower for keyword in ALLOWED_TYPE_KEYWORDS)

def is_forced_keep(edge_id: str, edge_name: str, edge_type: str) -> bool:
    text = f"{edge_id} {edge_name} {edge_type}".lower()
    return any(keyword.lower() in text for keyword in FORCE_KEEP_NAME_KEYWORDS)

def point_in_bbox(lon: float, lat: float) -> bool:
    return MIN_LON <= lon <= MAX_LON and MIN_LAT <= lat <= MAX_LAT

# =====================================
# 10. Select edges
# =====================================
print("Selecting corridor edges...")
keep_edges = set()
total_edges = 0
candidate_edges = 0

for edge in net.getEdges():
    total_edges += 1

    edge_id = edge.getID()

    if edge_id.startswith(":"):
        continue

    meta = edge_meta.get(edge_id, {})
    edge_type = meta.get("type", "")
    edge_name = meta.get("name", "")

    if not is_allowed_type(edge_type) and not is_forced_keep(edge_id, edge_name, edge_type):
        continue

    candidate_edges += 1

    shape_xy = edge.getShape()

    keep_this_edge = False
    for x, y in shape_xy:
        lon, lat = net.convertXY2LonLat(x, y)
        if point_in_bbox(lon, lat):
            keep_this_edge = True
            break

    if keep_this_edge or is_forced_keep(edge_id, edge_name, edge_type):
        keep_edges.add(edge_id)

# =====================================
# 11. Save keep-edge list
# =====================================
with open(KEEP_EDGES_FILE, "w", encoding="utf-8") as f:
    for edge_id in sorted(keep_edges):
        f.write(edge_id + "\n")

print(f"Total edges in full net: {total_edges}")
print(f"Candidate edges after type filter: {candidate_edges}")
print(f"Edges selected for corridor net: {len(keep_edges)}")
print(f"Keep-edge file saved to: {KEEP_EDGES_FILE}")

if len(keep_edges) == 0:
    print("No edges selected. Adjust corridor bounds or allowed types.")
    sys.exit(1)

# =====================================
# 12. Build corridor network
# =====================================
cmd = [
    NETCONVERT_EXE,
    "--sumo-net-file", FULL_NET_FILE,
    "--keep-edges.input-file", KEEP_EDGES_FILE,
    "--remove-edges.isolated",
    "--output-file", CORRIDOR_NET_FILE,
    "--verbose"
]

print("\nBuilding corridor network...")
print("Command:")
print(" ".join(f'"{x}"' if " " in x else x for x in cmd))

try:
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )

    print("\nCorridor network successfully generated.")
    print(f"Saved to: {CORRIDOR_NET_FILE}")

    if result.stdout.strip():
        print("\n--- stdout ---")
        print(result.stdout)

    if result.stderr.strip():
        print("\n--- stderr ---")
        print(result.stderr)

except subprocess.CalledProcessError as e:
    print("\nFailed to generate corridor network.")
    print("Return code:", e.returncode)

    if e.stdout:
        print("\n--- stdout ---")
        print(e.stdout)

    if e.stderr:
        print("\n--- stderr ---")
        print(e.stderr)

    sys.exit(1)