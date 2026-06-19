import os
import sys
import xml.etree.ElementTree as ET


def add_poi(root, poi_id, lon, lat, color="255,0,0", poi_type="label", layer="10"):
    ET.SubElement(
        root,
        "poi",
        {
            "id": poi_id,
            "type": poi_type,
            "color": color,
            "layer": str(layer),
            "lon": str(lon),
            "lat": str(lat),
        }
    )


def add_poly(root, poly_id, coords, color="0,180,255", layer="20", fill=False):
    """
    coords: [(lon, lat), (lon, lat), ...]
    These are geographic coordinates, so geo must be true.
    """
    shape_str = " ".join(f"{lon},{lat}" for lon, lat in coords)

    attrib = {
        "id": poly_id,
        "color": color,
        "layer": str(layer),
        "shape": shape_str,
        "fill": "1" if fill else "0",
        "geo": "true",
    }

    ET.SubElement(root, "poly", attrib)

def build_labels():
    project_root = r"D:\SUMO_A9_Project"
    sumo_dir = os.path.join(project_root, "sumo")

    label_dir = os.path.join(project_root, "labels")
    os.makedirs(label_dir, exist_ok=True)

    net_file = os.path.join(sumo_dir, "a8_corridor.net.xml")
    output_file = os.path.join(label_dir, "a8_labels.add.xml")

    if not os.path.exists(net_file):
        print(f"[ERROR] net file not found: {net_file}")
        return

    # =====================================
    # 1. Place name tag
    # =====================================
    place_labels = {
        "Dachau":      (11.4340, 48.2600),
        "Karlsfeld":   (11.4660, 48.2300),
        "Olching":     (11.3330, 48.2080),
        "Gröbenzell":  (11.3650, 48.1960),
        "Sulzemoos":   (11.2630, 48.2910),
        "Odelzhausen": (11.1980, 48.3150),
        "Einsbach":    (11.2680, 48.2640),
    }

    # =====================================
    # 2. Water body name label
    # =====================================
    water_labels = {
        "Amper":     (11.4150, 48.2520),
        "Ampersee":  (11.3020, 48.2510),
        "Würm":      (11.4080, 48.2010),
    }

    # =====================================
    # 3. High-speed numbering labels
    # =====================================
    motorway_labels = {
        "A8":  (11.2580, 48.2620),
        "A99": (11.4010, 48.2210),
    }

    # =====================================
    # 3.5 Road name labels
    # =====================================
    road_labels = {
        "Sigmertshauser Straße": (11.2850, 48.2850),
        "Ottmarshart": (11.3000, 48.2750),
        "Dachauer Straße": (11.4300, 48.2550),
        "Karlsfelder Straße": (11.4550, 48.2380),
        "Brucker Straße": (11.3400, 48.2100),
        "Münchner Straße": (11.4100, 48.2250),
        "Augsburger Straße": (11.3150, 48.2350),
    }

    # =====================================
    # 4. River/canal outline (polyline)
    #    This is just a rough visualisation; we can fine-tune it further later on.
    # =====================================
    amper_line = [
        (11.300, 48.252),
        (11.330, 48.251),
        (11.360, 48.250),
        (11.390, 48.249),
        (11.420, 48.248),
        (11.450, 48.247),
    ]

    wuerm_line = [
        (11.395, 48.180),
        (11.400, 48.190),
        (11.405, 48.200),
        (11.410, 48.210),
        (11.415, 48.220),
    ]

    kanal_line = [
        (11.360, 48.232),
        (11.375, 48.234),
        (11.390, 48.236),
        (11.405, 48.238),
    ]

    # =====================================
    # 5. Lake/water surface outline (surface)
    #    Here is a rough sketch of a polygon for now; we can refine it later.
    # =====================================
    ampersee_poly = [
        (11.292, 48.248),
        (11.298, 48.255),
        (11.307, 48.257),
        (11.314, 48.252),
        (11.309, 48.245),
        (11.299, 48.244),
    ]

    root = ET.Element("additional")

    # Place names
    for name, (lon, lat) in place_labels.items():
        add_poi(
            root,
            poi_id=f"place_{name}",
            lon=lon,
            lat=lat,
            color="0,0,255",
            poi_type="place",
            layer="20"
        )

    # Name of water body
    for name, (lon, lat) in water_labels.items():
        add_poi(
            root,
            poi_id=f"water_{name}",
            lon=lon,
            lat=lat,
            color="0,180,255",
            poi_type="water",
            layer="18"
        )

    # Highway number
    for name, (lon, lat) in motorway_labels.items():
        add_poi(
            root,
            poi_id=f"motorway_{name}",
            lon=lon,
            lat=lat,
            color="255,128,0",
            poi_type="motorway",
            layer="25"
        )

    # Road names
    # for name, (lon, lat) in road_labels.items():
    #     add_poi(
    #         root,
    #         poi_id=f"road_{name}",
    #         lon=lon,
    #         lat=lat,
    #         color="255,0,255",
    #         poi_type="road",
    #         layer="22"
    #     )

    # River/canal polygon
    #add_poly(root, "river_amper", amper_line, color="0,180,255", layer="8", fill=False)
    #add_poly(root, "river_wuerm", wuerm_line, color="0,180,255", layer="8", fill=False)
    #add_poly(root, "canal_main", kanal_line, color="0,180,255", layer="8", fill=False)

    # Lakes/water surfaces
    #add_poly(root, "lake_ampersee", ampersee_poly, color="0,0,255", layer="15", fill=True)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

    print(f"[SUCCESS] Label file written to: {output_file}")


if __name__ == "__main__":
    build_labels()