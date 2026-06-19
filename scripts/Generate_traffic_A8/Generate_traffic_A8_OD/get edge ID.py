import xml.etree.ElementTree as ET

net_file = r"D:\SUMO_A9_Project\sumo\a8_corridor.net.xml"

tree = ET.parse(net_file)
root = tree.getroot()

edges = []

for edge in root.findall("edge"):
    edge_id = edge.get("id")
    if edge_id and not edge_id.startswith(":"):  # 排除 internal edge
        edges.append(edge_id)

print(f"Total edges: {len(edges)}")

# 打印前50个看看
for e in edges[:50]:
    print(e)