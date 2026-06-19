import os
import sys

SUMO_HOME = r"D:\Eclipse\SUMO"
os.environ["SUMO_HOME"] = SUMO_HOME

sys.path.append(os.path.join(SUMO_HOME, "tools"))

import sumolib


def find_edges_by_area(net, center_lon, center_lat, radius=800):
    x, y = net.convertLonLat2XY(center_lon, center_lat)
    edges = net.getNeighboringEdges(x, y, radius)

    result = []

    for edge, dist in edges:
        if edge.isSpecial():
            continue

        edge_id = edge.getID()

        if edge_id.startswith(":"):
            continue

        lanes = edge.getLanes()
        if not lanes:
            continue

        length = edge.getLength()
        speed = max(l.getSpeed() for l in lanes)

        result.append((edge_id, dist, length, speed))

    result.sort(key=lambda x: x[1])

    return result


def main():
    net_file = r"D:\SUMO_A9_Project\sumo\a8_corridor.net.xml"
    net = sumolib.net.readNet(net_file)

    print("=====================================")
    print("Searching B471 Dachau-Süd area")
    print("=====================================")

    # 👉 你要找的区域（Dachau Süd /  Dachauer Moos / Gröbenried）
    # 可以在 Google Maps 复制坐标
    center_points = [
        (48.2625, 11.3825),  # Dachau-Süd（B471 主干，靠近 A8 连接区）
        (48.2585, 11.3700),  # Dachauer Moos 中段（低洼区域，典型 flooding 区）
        (48.2555, 11.3580),  # Gröbenried 附近（B471 西段）
    ]

    for lat, lon in center_points:
        print(f"\n--- Center: {lat}, {lon} ---")

        edges = find_edges_by_area(net, lon, lat, radius=2000)

        for edge_id, dist, length, speed in edges[:20]:
            print(
                f"{edge_id} | dist={dist:.1f}m | length={length:.1f}m | speed={speed:.2f}"
            )


if __name__ == "__main__":
    main()