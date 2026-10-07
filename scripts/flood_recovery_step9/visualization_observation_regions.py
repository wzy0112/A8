from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. INPUT / OUTPUT
# ============================================================

# SUMO network
NET_PATH = Path(
    r"D:\SUMO_A9_Project\sumo\a8_corridor.net.xml"
)

# 前面生成的六个 observation regions 对应的 edge 表
REGIONS_PATH = Path(
    r"D:\SUMO_A9_Project\scripts\flood_recovery_step9\observation_regions_edges.csv"
)

# 输出图片
OUT_PATH = Path(
    r"D:\SUMO_A9_Project\scripts\flood_recovery_step9\observation_regions_6_map.png"
)


# ============================================================
# 2. READ OBSERVATION REGION TABLE
# ============================================================

regions = pd.read_csv(
    REGIONS_PATH,
    dtype={"edge_id": str}
)

print("Loaded observation regions:")
print(regions["region_id"].value_counts())


# ============================================================
# 3. READ SUMO NETWORK GEOMETRY
# ============================================================

edge_shapes = {}

for event, elem in ET.iterparse(
    NET_PATH,
    events=("end",)
):

    if elem.tag != "edge":
        continue

    edge_id = elem.get("id", "")
    function = elem.get("function")

    # Ignore SUMO internal edges
    if (
        not edge_id
        or edge_id.startswith(":")
        or function == "internal"
    ):
        elem.clear()
        continue

    # Use the first lane geometry as the edge geometry
    lane = elem.find("lane")

    if lane is None:
        elem.clear()
        continue

    shape_text = lane.get("shape", "")

    if shape_text:

        points = []

        for token in shape_text.split():

            x_str, y_str = token.split(",")

            points.append(
                (
                    float(x_str),
                    float(y_str)
                )
            )

        if len(points) >= 2:

            edge_shapes[edge_id] = np.asarray(
                points,
                dtype=float
            )

    elem.clear()


print(
    f"Loaded {len(edge_shapes)} SUMO edge geometries."
)


# ============================================================
# 4. GET REGION ORDER
# ============================================================

region_order = list(
    dict.fromkeys(
        regions["region_id"].tolist()
    )
)

print("\nRegion order:")

for region in region_order:
    print(region)


# ============================================================
# 5. FIX ONE COLOR FOR EACH REGION
# ============================================================

cmap = plt.get_cmap("tab10")

region_colors = {

    region_id: cmap(i)

    for i, region_id
    in enumerate(region_order)
}

# ============================================================
# Legend display names
# ============================================================

region_legend_names = {
    "S1": "S1 B471 Observation Area",
    "S2": "S2 B471 Observation Area",
    "S3": "S3 A8 Observation Area",
    "S4": "S4 St2054 Observation Area",
    "S5": "S5 A8 Observation Area",
    "S6": "S6 St2051 Observation Area",
}

# ============================================================
# 6. CREATE MAP
# ============================================================

fig, ax = plt.subplots(
    figsize=(15, 11)
)


# ------------------------------------------------------------
# 6.1 Draw complete SUMO network
# ------------------------------------------------------------

for pts in edge_shapes.values():

    ax.plot(
        pts[:, 0],
        pts[:, 1],

        linewidth=0.45,

        color="0.55",

        alpha=0.22,

        zorder=1
    )


# ------------------------------------------------------------
# 6.2 Draw six observation regions
# ------------------------------------------------------------

for region_id in region_order:

    subset = regions[
        regions["region_id"] == region_id
    ]

    color = region_colors[region_id]

    # Extract S1 / S2 / S3 / S4 / S5 / S6
    short_id = region_id.split("_")[0]

    # Legend name
    legend_name = region_legend_names.get(
        short_id,
        region_id
    )

    all_xy = []

    first = True


    for edge_id in subset["edge_id"]:

        edge_id = str(edge_id)

        pts = edge_shapes.get(edge_id)


        if pts is None:

            print(
                f"WARNING: edge {edge_id} "
                f"not found in SUMO network."
            )

            continue


        ax.plot(
            pts[:, 0],
            pts[:, 1],

            linewidth=3.2,

            color=color,

            label=legend_name if first else None,

            zorder=3
        )


        all_xy.append(pts)

        first = False


    # --------------------------------------------------------
    # Add S1 / S2 / ... label
    # --------------------------------------------------------

    if all_xy:

        stacked = np.vstack(all_xy)

        cx = stacked[:, 0].mean()

        cy = stacked[:, 1].mean()


        # Example:
        #
        # S1_groebenried_b471
        #
        # becomes
        #
        # S1



        ax.text(
            cx,
            cy,

            short_id,

            fontsize=12,

            fontweight="bold",

            ha="center",

            va="center",

            bbox=dict(
                boxstyle="round,pad=0.25",

                facecolor="white",

                alpha=0.9,

                edgecolor=color,

                linewidth=1.4
            ),

            zorder=4
        )


# ============================================================
# 7. FIGURE STYLE
# ============================================================

ax.set_title(
    "Six Observation Regions on the SUMO Network",
    fontsize=16
)

ax.set_aspect(
    "equal",
    adjustable="box"
)

ax.axis("off")


ax.legend(
    loc="upper right",
    frameon=True
)


fig.tight_layout()


# ============================================================
# 8. SAVE
# ============================================================

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


fig.savefig(
    OUT_PATH,
    dpi=220,
    bbox_inches="tight"
)


print(
    f"\nSaved visualization to:\n{OUT_PATH}"
)


# ============================================================
# 9. SHOW
# ============================================================

plt.show()