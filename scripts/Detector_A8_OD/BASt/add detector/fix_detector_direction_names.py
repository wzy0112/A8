# 1. 先运行add_detector_all_lanes 生成原始 a8_detectors.add.xml 和 a8_detector_mapping.csv
# 2. 手动/检查得到 a8_detector_mapping_R1R2_check.csv
# 3. 运行 fix_detector_direction_names.py
# 4. 用 a8_detectors_corrected.add.xml 跑 SUMO
# 5. 用 a8_detector_mapping_corrected.csv 做汇总

import csv
import os
import re
import xml.etree.ElementTree as ET


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_detector_id(detector_id):
    """
    Expected format:
    9014_dir1_lane0
    9014_dir2_lane1
    """
    match = re.match(r"^(?P<station>.+?)_dir[12]_lane(?P<lane>\d+)$", detector_id)

    if not match:
        raise ValueError(f"Cannot parse detector_id: {detector_id}")

    return match.group("station"), match.group("lane")


def fix_detector_direction_names(
    old_mapping_csv,
    r1r2_check_csv,
    old_add_xml,
    new_mapping_csv,
    new_add_xml,
):
    """
    Use direction_id from a8_detector_mapping_R1R2_check.csv as the source of truth.

    It rewrites:
    - detector_id
    - group_id
    - direction_id
    in the mapping CSV.

    It also rewrites detector ids in the SUMO additional XML.
    """
    corrected_rows = []
    rename_dict = {}

    with open(r1r2_check_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        required_cols = {"detector_id", "group_id", "direction_id", "matched"}
        missing_cols = required_cols - set(fieldnames)

        if missing_cols:
            raise ValueError(f"Missing columns in R1R2 check CSV: {missing_cols}")

        for row in reader:
            old_detector_id = row["detector_id"].strip()
            matched = is_true(row.get("matched", ""))

            if not matched:
                corrected_rows.append(row)
                continue

            station_id, lane_index = parse_detector_id(old_detector_id)

            correct_direction = row["direction_id"].strip().lower()

            if correct_direction not in {"dir1", "dir2"}:
                raise ValueError(
                    f"direction_id must be dir1 or dir2, got {correct_direction} "
                    f"for detector {old_detector_id}"
                )

            new_detector_id = f"{station_id}_{correct_direction}_lane{lane_index}"
            new_group_id = f"{station_id}_{correct_direction}"

            row["detector_id"] = new_detector_id
            row["group_id"] = new_group_id
            row["direction_id"] = correct_direction

            corrected_rows.append(row)
            rename_dict[old_detector_id] = new_detector_id

    # Check duplicated final detector ids
    final_ids = [
        row["detector_id"]
        for row in corrected_rows
        if is_true(row.get("matched", ""))
    ]

    duplicated_ids = sorted({
        detector_id for detector_id in final_ids
        if final_ids.count(detector_id) > 1
    })

    if duplicated_ids:
        raise ValueError(f"Duplicated detector ids after correction: {duplicated_ids}")

    # Write corrected mapping CSV
    with open(new_mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corrected_rows)

    # Rewrite additional XML detector ids
    tree = ET.parse(old_add_xml)
    root = tree.getroot()

    missing_in_check = []

    for detector in root.findall("inductionLoop"):
        old_id = detector.get("id")

        if old_id in rename_dict:
            detector.set("id", rename_dict[old_id])
        else:
            missing_in_check.append(old_id)

    if missing_in_check:
        print("[WARNING] These XML detector ids were not found in R1R2 check CSV:")
        for detector_id in missing_in_check:
            print("  ", detector_id)

    ET.indent(tree, space="    ")
    tree.write(new_add_xml, encoding="utf-8", xml_declaration=True)

    print("=====================================")
    print("[SUCCESS] Direction names corrected.")
    print(f"Corrected mapping CSV : {new_mapping_csv}")
    print(f"Corrected add XML     : {new_add_xml}")
    print(f"Renamed detectors     : {len(rename_dict)}")
    print("=====================================")


if __name__ == "__main__":
    detector_dir = r"D:\SUMO_A9_Project\detectors"

    old_mapping_csv = os.path.join(detector_dir, "a8_detector_mapping.csv")
    r1r2_check_csv = os.path.join(detector_dir, "a8_detector_mapping_R1R2_check.csv")
    old_add_xml = os.path.join(detector_dir, "a8_detectors.add.xml")

    new_mapping_csv = os.path.join(detector_dir, "a8_detector_mapping_corrected.csv")
    new_add_xml = os.path.join(detector_dir, "a8_detectors_corrected.add.xml")

    fix_detector_direction_names(
        old_mapping_csv=old_mapping_csv,
        r1r2_check_csv=r1r2_check_csv,
        old_add_xml=old_add_xml,
        new_mapping_csv=new_mapping_csv,
        new_add_xml=new_add_xml,
    )
